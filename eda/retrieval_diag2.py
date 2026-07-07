"""Retrieval 진단 v2 — 버그수정+순환성 검사 (R22 X1 게이트, 엄격판).

수정: ①모델 베이스라인=teacher hold(진짜, OOF아님) ②임베딩 이방성 체크(랜덤쌍 vs NN)
③핵심 순환성 검사: 모델이 틀린 홀드아웃 행에서 KNN이 진짜라벨 맞추나(=직교신호 실재)
④rank기반 게이트(절대 cosine 무의미, 이방성). 임베딩 저장(재사용).
"""
from __future__ import annotations
import os, sys, glob, time
import numpy as np

R = "/root/Action_Decision"
sys.path.insert(0, R)
from common.io_utils import load_train, CLASSES
from common.cv import make_splits
from common.metrics import macro_f1
from common import ad_lib
import torch
from transformers import AutoTokenizer

MDIR = f"{R}/action_decision_maximum/experiments/member_largefullv6"
samples, y, ids = load_train(); y = np.array(y)
sp = make_splits(ids, y, np.array([s["session"] for s in samples])); folds = sp["folds"]
hidx = sp["holdout_idx"]
EXPL = {CLASSES.index(c) for c in ["read_file","grep_search","list_directory","glob_pattern"]}

EMB_CACHE = f"{R}/work/emb_v6_70k.npy"
if os.path.exists(EMB_CACHE):
    emb = np.load(EMB_CACHE); print(f"[emb] 캐시 로드 {emb.shape}", flush=True)
else:
    tok = AutoTokenizer.from_pretrained(MDIR, local_files_only=True); tok.truncation_side = "left"
    model = ad_lib._load_model_maybe_quant(MDIR).to("cuda").half().eval()
    id_map = np.load(f"{MDIR}/id_map.npy") if os.path.exists(f"{MDIR}/id_map.npy") else None
    texts = [ad_lib.serialize(s, "v6") for s in samples]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    emb = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(order), 128):
            idx = order[b:b+128]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=320,
                      pad_to_multiple_of=8, return_tensors="pt")
            if id_map is not None:
                enc["input_ids"] = torch.from_numpy(id_map[enc["input_ids"].numpy()]).to(enc["input_ids"].dtype)
            enc = {k: v.to("cuda") for k, v in enc.items()}
            out = model.base_model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            for j, i in enumerate(idx): emb[i] = mean[j].float().cpu().numpy()
    np.save(EMB_CACHE, emb); print(f"[emb] 계산+저장 {(time.time()-t0)/60:.1f}m", flush=True)

# 이방성 체크: 원본 & 중심화(mean 제거) 후 코사인
embn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
mu = emb.mean(0); embc = emb - mu; embc /= (np.linalg.norm(embc, axis=1, keepdims=True) + 1e-9)
rng = np.random.default_rng(0)
ri, rj = rng.integers(0, len(y), 5000), rng.integers(0, len(y), 5000)
print(f"[이방성] 랜덤쌍 cos: 원본 {np.mean(np.sum(embn[ri]*embn[rj],1)):.3f} / 중심화 {np.mean(np.sum(embc[ri]*embc[rj],1)):.3f}")
print("  → 원본 랜덤쌍이 0.9+면 이방성(절대sim 무의미), 중심화로 완화. 중심화 임베딩으로 진단 진행.")

# 진짜 모델 홀드아웃 베이스라인 = teacher hold 평균
holds = []
for p in sorted(glob.glob(f"{R}/action_decision_maximum/experiments/teacher_largev6[AB]_a*.npz")):
    holds.append(np.load(p, allow_pickle=True)["hold"])
hold_p = np.mean(holds, 0)                      # [5810,14] 진짜 모델 확률
yh = y[hidx]
mdl_pred = hold_p.argmax(1); mdl_acc = (mdl_pred == yh).mean()
mdl_f1 = macro_f1(yh, mdl_pred, 14)[0]
srt = np.sort(hold_p, 1); mgn = srt[:, -1] - srt[:, -2]
print(f"\n진짜 모델 홀드아웃: acc {mdl_acc:.3f}, macro-F1 {mdl_f1:.4f}")

# 홀드아웃 → dev NN (중심화 임베딩, rank기반)
dev_mask = np.ones(len(y), bool); dev_mask[hidx] = False
dev_idx = np.where(dev_mask)[0]; E_dev = embc[dev_idx]; y_dev = y[dev_idx]
K = 16
knn_prior = np.zeros((len(hidx), 14)); nn_top1_sim = np.zeros(len(hidx))
for bi in range(0, len(hidx), 256):
    S = embc[hidx[bi:bi+256]] @ E_dev.T
    part = np.argpartition(-S, K, axis=1)[:, :K]
    for r in range(part.shape[0]):
        nn = part[r]; ss = S[r, nn]; o = np.argsort(-ss); nn, ss = nn[o], ss[o]
        nn_top1_sim[bi+r] = ss[0]
        w = np.clip(ss, 0, None) ** 4
        pr = np.bincount(y_dev[nn], weights=w, minlength=14); knn_prior[bi+r] = pr / pr.sum()
knn_pred = knn_prior.argmax(1)

# ★순환성 검사: 모델이 틀린 행에서 KNN이 진짜라벨 맞추나
wrong = mdl_pred != yh
print(f"\n=== 순환성 검사 (핵심) ===")
print(f"모델 오답 {wrong.sum()}행 중 KNN이 진짜라벨 맞춤: {(knn_pred[wrong]==yh[wrong]).mean():.3f}")
print(f"모델 정답 {(~wrong).sum()}행 중 KNN 일치: {(knn_pred[~wrong]==yh[~wrong]).mean():.3f}")
print(f"  → 오답행 KNN정확도가 0.3+ 유의미면 직교신호 실재. ~0.07(랜덤)이면 순환/무용")
lowm = mgn < 0.3
print(f"저마진(<0.3) {lowm.sum()}행: 모델 {(mdl_pred[lowm]==yh[lowm]).mean():.3f} vs KNN {(knn_pred[lowm]==yh[lowm]).mean():.3f}")

# 블렌드: 저마진 게이트 + purity, 진짜 모델 로짓에 prior 보정
lp = np.log(hold_p + 1e-9); prior_cls = hold_p.mean(0)
top8_purity = np.zeros(len(hidx))
for bi in range(0, len(hidx), 256):
    S = embc[hidx[bi:bi+256]] @ E_dev.T
    part = np.argpartition(-S, 8, axis=1)[:, :8]
    for r in range(part.shape[0]):
        lbls = y_dev[part[r]]; top8_purity[bi+r] = np.bincount(lbls, minlength=14).max()/8
print(f"\n모델 macro-F1 {mdl_f1:.4f}")
for lam, mth, pth in [(0.3,0.3,0.6),(0.5,0.3,0.7),(0.5,0.5,0.6),(0.8,0.2,0.75)]:
    gate = (mgn < mth) & (top8_purity >= pth)
    z = lp.copy(); z[gate] += lam*(np.log(knn_prior[gate]+1e-9)-np.log(prior_cls+1e-9))
    f1 = macro_f1(yh, z.argmax(1), 14)[0]
    print(f"  λ={lam} margin<{mth} purity≥{pth}: 게이트 {gate.mean()*100:4.1f}% → {f1:.5f} (Δ{f1-mdl_f1:+.5f})")
print("\n→ 오답행 KNN정확도(순환성)와 블렌드 Δ가 판정. 둘 다 유의미 양수면 GO.")
