#!/usr/bin/env python
"""session-balanced FT 검증용 stress 평가 (R26/R27).

fold0-val을 히든 유사 슬라이스로 쪼개 모델(들)을 비교:
  - overall / sim-only macro-F1
  - NN-sim 사분위 (Q1=low-NN=cross-session near-dup 없음=히든 유사)
  - 세션길이 버킷 (1-2 / 3-8 / 9+)  ※히든≈세션당 1관측
  - hist=0 슬라이스
  - (2개 이상일 때) 모델 간 argmax 일치율 — 앙상블 다양성 체크

usage: python3 sim/eval_stress.py <model_dir> [<model_dir2> ...]
출력: 표 + work/stress_probs_<tag>.npz (fold0-val probs, 재사용)
"""
from __future__ import annotations
import os, sys, time
import numpy as np

R = "/root/Action_Decision"
sys.path.insert(0, R)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from common.io_utils import load_train, NUM_CLASSES
from common.cv import make_splits
from common.metrics import macro_f1
from common import ad_lib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.amp import autocast

VERSION = os.environ.get("AD_VERSION", "v6")
MAX_LEN = int(os.environ.get("AD_MAXLEN", "320"))
SIM_CACHE = f"{R}/work/stress_nnsim_f0.npy"

samples, y, ids = load_train()
y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups)
tr0, va0 = sp["folds"][0]
tr0, va0 = np.asarray(tr0), np.asarray(va0)
GEN = np.array([s["gen"] for s in samples])
from collections import Counter
_slen = Counter(groups.tolist())
SLEN = np.array([_slen[g] for g in groups])
HLEN = np.array([len(s.get("history") or []) for s in samples])

# --- NN sim (fold0-val -> fold0-train, 세션은 그룹CV라 자동 배타) ---
if os.path.exists(SIM_CACHE):
    nn_sim = np.load(SIM_CACHE)
else:
    emb = np.load(f"{R}/work/retrieval_pack/train_emb.npy", mmap_mode="r")
    tr_emb = np.asarray(emb[tr0], dtype=np.float32)
    nn_sim = np.zeros(len(va0), np.float32)
    t0 = time.time()
    for b in range(0, len(va0), 1024):
        q = np.asarray(emb[va0[b:b + 1024]], dtype=np.float32)
        nn_sim[b:b + 1024] = (q @ tr_emb.T).max(1)
    np.save(SIM_CACHE, nn_sim)
    print(f"[nn_sim] computed in {time.time()-t0:.0f}s", flush=True)
qs = np.quantile(nn_sim, [0.25, 0.5, 0.75])
print(f"[nn_sim] fold0-val quartiles: {qs.round(4)} (Q1<{qs[0]:.3f}=히든유사)", flush=True)

texts = [ad_lib.serialize(samples[int(j)], VERSION) for j in va0]
yv = y[va0]

def slices():
    out = {"overall": np.ones(len(va0), bool), "sim": GEN[va0] == "sim"}
    out["nnQ1(low)"] = nn_sim < qs[0]
    out["nnQ2"] = (nn_sim >= qs[0]) & (nn_sim < qs[1])
    out["nnQ34(dup)"] = nn_sim >= qs[1]
    sl = SLEN[va0]
    out["sess1-2"] = sl <= 2
    out["sess3-8"] = (sl >= 3) & (sl <= 8)
    out["sess9+"] = sl >= 9
    out["hist0"] = HLEN[va0] == 0
    return out

SL = slices()

def infer(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir); tok.truncation_side = "left"
    enc_all = tok(texts, truncation=True, max_length=MAX_LEN, padding=False)["input_ids"]
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, torch_dtype=torch.float16).cuda().eval()
    order = sorted(range(len(texts)), key=lambda k: len(enc_all[k]))
    probs = np.zeros((len(texts), NUM_CLASSES), np.float32)
    with torch.no_grad():
        for b in range(0, len(order), 192):
            ks = order[b:b + 192]
            e = tok.pad({"input_ids": [enc_all[k] for k in ks]}, return_tensors="pt").to("cuda")
            with autocast("cuda", dtype=torch.float16):
                lg = model(**e).logits.float()
            p = torch.softmax(lg, 1).cpu().numpy()
            for m, k in enumerate(ks):
                probs[k] = p[m]
    del model; torch.cuda.empty_cache()
    return probs

results = {}
for md in sys.argv[1:]:
    tag = os.path.basename(md.rstrip("/"))
    t0 = time.time()
    probs = infer(md)
    np.savez_compressed(f"{R}/work/stress_probs_{tag}.npz", probs=probs, va0=va0)
    results[tag] = probs
    pred = probs.argmax(1)
    row = {name: macro_f1(yv[m], pred[m])[0] for name, m in SL.items() if m.sum() > 50}
    results[tag + "__row"] = row
    print(f"\n=== {tag} ({time.time()-t0:.0f}s) ===", flush=True)
    for name, v in row.items():
        print(f"  {name:<12} {v:.4f} (n={SL[name].sum()})", flush=True)

tags = [t for t in results if not t.endswith("__row")]
if len(tags) >= 2:
    base = results[tags[0]].argmax(1)
    print("\n=== vs " + tags[0] + " ===", flush=True)
    for t in tags[1:]:
        pred = results[t].argmax(1)
        agree = (pred == base).mean()
        d = {n: results[t + '__row'][n] - results[tags[0] + '__row'][n]
             for n in results[t + '__row']}
        print(f"  {t}: agree={agree:.4f} " +
              " ".join(f"{n}={v:+.4f}" for n, v in d.items()), flush=True)
