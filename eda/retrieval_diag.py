"""Retrieval/near-dup prior 진단 (R22 X1) — 게이트: 데이터 기하에 검색가능 신호가 있나.

절차:
1. large-v6 FULL 멤버(frozen)로 train 70k를 mean-pool 임베딩.
2. leave-group-out(splits fold): 홀드아웃 각 행 → dev행에서 top-k 코사인 NN.
3. 측정: (a) max-sim 분포 (b) 고유사도 이웃 라벨 purity (c) retrieval-prior가
   모델 로짓(v6 teacher OOF) 대비 macro-F1 개선, 특히 저마진·탐색클러스터.
4. 판정: 고유사도 tail 두껍고 purity≥0.75면 GO(배포판 제작), 아니면 축 폐쇄.
주의: 홀드아웃은 train분포라 낙관적 — 히든테스트 전이는 LB canary로만 확정(codex R22).
usage: python3 eda/retrieval_diag.py
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
if not os.path.isdir(MDIR):
    import zipfile
    os.makedirs(MDIR, exist_ok=True)
    with zipfile.ZipFile(f"{MDIR}.zip") as z:
        z.extractall(MDIR)

samples, y, ids = load_train(); y = np.array(y)
sp = make_splits(ids, y, np.array([s["session"] for s in samples])); folds = sp["folds"]
hidx = sp["holdout_idx"]
au = np.char.startswith(np.array([str(i) for i in ids]), "sess_au")

# ---- 임베딩 (large-v6 frozen, mean-pool) ----
dev = "cuda"
tok = AutoTokenizer.from_pretrained(MDIR, local_files_only=True); tok.truncation_side = "left"
model = ad_lib._load_model_maybe_quant(MDIR).to(dev).half().eval()
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
        enc = {k: v.to(dev) for k, v in enc.items()}
        out = model.base_model(**enc).last_hidden_state          # [B,T,H]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        for j, i in enumerate(idx):
            emb[i] = mean[j].float().cpu().numpy()
        if b % 6400 == 0:
            print(f"  embed {b}/{len(order)} @{(time.time()-t0)/60:.1f}m", flush=True)
emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
print(f"[emb] done {(time.time()-t0)/60:.1f}m, dim={emb.shape[1]}", flush=True)

# ---- v6 teacher OOF (모델 로짓/마진) ----
oof = np.zeros((len(y), 14), np.float32); cs = set()
for p in sorted(glob.glob(f"{R}/action_decision_maximum/experiments/teacher_largev6[AB]_a*.npz")):
    z = np.load(p, allow_pickle=True)
    for f in range(int(z["fold_lo"]), int(z["fold_hi"])):
        if f in cs: continue
        oof[folds[f][1]] = z["oof"][folds[f][1]]; cs.add(f)

# ---- 홀드아웃 → dev NN (dev = 홀드아웃 제외 전체) ----
dev_mask = np.ones(len(y), bool); dev_mask[hidx] = False
dev_idx = np.where(dev_mask)[0]
E_dev = emb[dev_idx]; y_dev = y[dev_idx]
EXPL = {CLASSES.index(c) for c in ["read_file","grep_search","list_directory","glob_pattern"]}

K = 32
sims_top1 = np.zeros(len(hidx)); knn_prior = np.zeros((len(hidx), 14)); purity = np.zeros(len(hidx))
for bi in range(0, len(hidx), 256):
    q = emb[hidx[bi:bi+256]]                      # [b,H]
    S = q @ E_dev.T                                # [b, Ndev] 코사인(정규화됨)
    part = np.argpartition(-S, K, axis=1)[:, :K]
    for r in range(q.shape[0]):
        nn = part[r]; ss = S[r, nn]
        o = np.argsort(-ss); nn = nn[o]; ss = ss[o]
        sims_top1[bi+r] = ss[0]
        w = np.clip(ss, 0, None) ** 4              # 유사도 강조 가중
        pr = np.bincount(y_dev[nn], weights=w, minlength=14)
        knn_prior[bi+r] = pr / pr.sum()
        purity[bi+r] = (y_dev[nn[:8]] == np.bincount(y_dev[nn[:8]], minlength=14).argmax()).mean()

yh = y[hidx]; lp = np.log(oof[hidx] + 1e-9)
margin = np.sort(oof[hidx], axis=1)
mgn = margin[:, -1] - margin[:, -2]
print("\n=== 홀드아웃 → train NN 진단 ===")
print(f"max-sim 분포: p50={np.median(sims_top1):.3f} p90={np.percentile(sims_top1,90):.3f} p99={np.percentile(sims_top1,99):.3f} max={sims_top1.max():.3f}")
for th in (0.85, 0.9, 0.95, 0.98):
    m = sims_top1 >= th
    if m.sum() == 0: continue
    knn_acc = (knn_prior[m].argmax(1) == yh[m]).mean()
    mdl_acc = (oof[hidx][m].argmax(1) == yh[m]).mean()
    print(f"  sim≥{th}: 커버 {m.mean()*100:5.1f}% | KNN정확 {knn_acc:.3f} vs 모델 {mdl_acc:.3f} | purity평균 {purity[m].mean():.3f} | 탐색비율 {np.mean([yy in EXPL for yy in yh[m]]):.2f}")

# ---- prior 블렌딩 macro-F1 (게이트: 저마진·고유사도만 보정) ----
prior_cls = oof[hidx].mean(0)  # 근사 class prior
base_f1 = macro_f1(yh, oof[hidx].argmax(1), 14)[0]
print(f"\n모델 단독 holdout macro-F1: {base_f1:.5f}")
for lam, sth, pth in [(0.25,0.95,0.75),(0.4,0.9,0.75),(0.6,0.98,0.8)]:
    gate = (sims_top1 >= sth) & (purity >= pth)
    z = lp.copy()
    z[gate] += lam * (np.log(knn_prior[gate] + 1e-9) - np.log(prior_cls + 1e-9))
    f1 = macro_f1(yh, z.argmax(1), 14)[0]
    print(f"  λ={lam} sim≥{sth} purity≥{pth}: 게이트 {gate.mean()*100:.1f}% → macro-F1 {f1:.5f} (Δ{f1-base_f1:+.5f})")
print("\n→ Δ 양수+커버 유의미면 GO(배포판). tail 얇거나 Δ~0이면 축 폐쇄. 히든전이는 LB로만 확정.")
