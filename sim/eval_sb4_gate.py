#!/usr/bin/env python
"""R30: session-balanced 4번째 멤버 게이트 판정 (codex R29 락 게이트).

tri_cond 파이프라인(m1 0.6 + m2 0.15 + m3 0.25 조건부 margin<0.5)을 홀드아웃
5810행에서 프롭 수준으로 에뮬레이션하고, sbwt를 4번째 저마진 멤버로 얹는 그리드 평가.
게이트: ΔF1 ≥ +0.0010 AND changed 0.4~1.2% AND low-NN 슬라이스 Δ ≥ -0.0010.
주의: FULL 멤버는 홀드아웃을 학습에 봤으므로 절대값 팽창 — 델타만 유효(R28 규약).

usage: python3 sim/eval_sb4_gate.py <sbwt_model_dir>
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np

R = "/root/Action_Decision"
sys.path.insert(0, R)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from common.io_utils import load_train, CLASSES, NUM_CLASSES
from common.cv import make_splits
from common.metrics import macro_f1
from common import ad_lib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.amp import autocast

PKG = f"{R}/packages/submit_tri_cond_retr_p384/model"
SBWT = sys.argv[1] if len(sys.argv) > 1 else f"{R}/work/raw_largev6sbwt"
meta = json.load(open(f"{PKG}/run_meta.json"))
pp = json.load(open(f"{PKG}/postproc.json"))
assert pp["classes"] == CLASSES
bias = np.array(pp["bias"])
W = meta["weights"]; MTH = meta["conditional"]["margin_th"]

samples, y, ids = load_train(); y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups)
hold_idx = np.asarray(sp["holdout_idx"]); y_hold = y[hold_idx]
hs = [samples[int(j)] for j in hold_idx]

def infer(model_dir, version, max_len=320):
    tok = AutoTokenizer.from_pretrained(model_dir); tok.truncation_side = "left"
    texts = [ad_lib.serialize(s, version) for s in hs]
    enc = tok(texts, truncation=True, max_length=max_len, padding=False)["input_ids"]
    imp = os.path.join(model_dir, "id_map.npy")
    if os.path.exists(imp):
        idm = np.load(imp)
        enc = [idm[np.asarray(e, dtype=np.int64)].tolist() for e in enc]
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, torch_dtype=torch.float16).cuda().eval()
    order = sorted(range(len(enc)), key=lambda k: len(enc[k]))
    probs = np.zeros((len(enc), NUM_CLASSES), np.float32)
    with torch.no_grad():
        for b in range(0, len(order), 192):
            ks = order[b:b + 192]
            e = tok.pad({"input_ids": [enc[k] for k in ks]}, return_tensors="pt").to("cuda")
            with autocast("cuda", dtype=torch.float16):
                lg = model(**e).logits.float()
            p = torch.softmax(lg, 1).cpu().numpy()
            for m_, k in enumerate(ks):
                probs[k] = p[m_]
    del model; torch.cuda.empty_cache()
    return probs

CACHE = f"{R}/work/sb4_gate_probs.npz"
if os.path.exists(CACHE):
    z = np.load(CACHE)
    P1, P2, P3, PS = z["p1"], z["p2"], z["p3"], z["ps"]
else:
    t0 = time.time()
    P1 = infer(f"{PKG}/m1", meta["ensemble"][0]["version"])
    P2 = infer(f"{PKG}/m2", meta["ensemble"][1]["version"])
    P3 = infer(f"{PKG}/m3", meta["ensemble"][2]["version"])
    PS = infer(SBWT, "v6")
    np.savez_compressed(CACHE, p1=P1, p2=P2, p3=P3, ps=PS)
    print(f"[infer] 4 models x {len(hs)} rows in {(time.time()-t0)/60:.1f}min", flush=True)

# tri_cond 에뮬레이션: 1차 p12 → 저마진만 m3 혼합
p12 = (W[0] * P1 + W[1] * P2) / (W[0] + W[1])
srt = np.sort(p12, 1)
margin = srt[:, -1] - srt[:, -2]
low = margin < MTH
P_tri = p12.copy()
P_tri[low] = W[0] * P1[low] + W[1] * P2[low] + W[2] * P3[low]
base_pred = (np.log(P_tri + 1e-12) + bias).argmax(1)
base_f1 = macro_f1(y_hold, base_pred)[0]
agree_sb = (PS.argmax(1) == P_tri.argmax(1)).mean()
print(f"[base] tri_cond 에뮬 holdout F1={base_f1:.5f} (팽창값, 델타만 유효) low-margin={low.mean()*100:.1f}% sb-agree={agree_sb:.4f}")

# low-NN 슬라이스 (holdout→dev, 그룹split이라 cross-session 자동)
emb = np.load(f"{R}/work/retrieval_pack/train_emb.npy", mmap_mode="r")
dev_idx = np.asarray(sp["dev_idx"])
dev_emb = np.asarray(emb[dev_idx], dtype=np.float32)
nn_sim = np.zeros(len(hold_idx), np.float32)
for b in range(0, len(hold_idx), 1024):
    q = np.asarray(emb[hold_idx[b:b + 1024]], dtype=np.float32)
    nn_sim[b:b + 1024] = (q @ dev_emb.T).max(1)
q1 = nn_sim < np.quantile(nn_sim, 0.25)
base_q1 = macro_f1(y_hold[q1], base_pred[q1])[0]

print(f"\n{'th2':>5} {'w_sb':>5} {'coverage%':>9} {'changed%':>8} {'ΔF1':>8} {'ΔnnQ1':>8} {'게이트':>6}")
best = None
for th2 in (0.5, 0.35, 0.25):
    sel = margin < th2
    for w_sb in (0.15, 0.25, 0.35):
        Pn = P_tri.copy()
        Pn[sel] = (1 - w_sb) * P_tri[sel] + w_sb * PS[sel]
        pred = (np.log(Pn + 1e-12) + bias).argmax(1)
        ch = (pred != base_pred).mean() * 100
        d = macro_f1(y_hold, pred)[0] - base_f1
        dq1 = macro_f1(y_hold[q1], pred[q1])[0] - base_q1
        ok = d >= 0.0010 and 0.4 <= ch <= 1.2 and dq1 >= -0.0010
        print(f"{th2:5.2f} {w_sb:5.2f} {sel.mean()*100:8.1f}% {ch:7.2f}% {d:+.4f} {dq1:+.4f} {'PASS' if ok else 'fail':>6}")
        if ok and (best is None or d > best[0]):
            best = (d, th2, w_sb, ch, dq1)

print("\n판정:", f"PASS 최적 th2={best[1]} w_sb={best[2]} ΔF1={best[0]:+.4f} changed={best[3]:.2f}% ΔnnQ1={best[4]:+.4f}"
      if best else "전 그리드 게이트 미달 — 오늘 예비 1발 보존 (codex R30 Q4)")
