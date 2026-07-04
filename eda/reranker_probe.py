"""top-2 gated reranker 프로브 — M6 스택 OOF에서 접전샘플만 flip 시 macro-F1 이득 측정.

codex: rank-2 정답 57% → "헷갈리는 2클래스 뒤집기"가 새 모델보다 큰 레버.
방식: stacker OOF proba로 top1/top2 margin 작은 샘플만 대상. 각 샘플에
      [멤버logit, margin, entropy, 구조피처, ngram top2지지]로 LightGBM 이진(=top2가 정답?) 학습(OOF).
      flip 게이트: p(top2정답) > τ 이면 top1↔top2 교체. τ는 macro-F1 최대화로 coordinate search.
출력: OOF net 이득 판정.
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.io_utils import load_train, CLASSES, NUM_CLASSES
from common.cv import make_splits
from common.metrics import macro_f1
from common import ad_lib
import lightgbm as lgb

EXPD = "action_decision_maximum/experiments"
samples, y, ids = load_train(); y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups); folds = sp["folds"]
cov = np.concatenate([f[1] for f in folds])

FAM = {"klue": ["teacher_klue_s1234.npz"],
       "largev6": ["teacher_largev6A_a*.npz", "teacher_largev6B_a*.npz"],
       "ngram": ["teacher_ngram.npz"]}
def load(k):
    o = np.zeros((len(samples), NUM_CLASSES), np.float32); cs = set()
    for pat in FAM[k]:
        for p in ([f"{EXPD}/{pat}"] if "*" not in pat else sorted(glob.glob(f"{EXPD}/{pat}"))):
            if not os.path.exists(p): continue
            z = np.load(p, allow_pickle=True)
            for f in range(int(z["fold_lo"]), int(z["fold_hi"])):
                if f in cs: continue
                o[folds[f][1]] = z["oof"][folds[f][1]]; cs.add(f)
    return o
M = {k: load(k) for k in FAM}
F = np.array([ad_lib.stack_features(s) for s in samples], dtype=np.float32)

# 1) 베이스 스태커 OOF (klue+largev6+ngram)
P = np.concatenate([M[k] for k in FAM] + [F], axis=1)
params = dict(objective="multiclass", num_class=14, learning_rate=0.05, num_leaves=63,
             min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
             verbose=-1, num_threads=8, seed=1234)
stack_oof = np.zeros((len(samples), NUM_CLASSES), np.float32)
for tr, va in folds:
    stack_oof[va] = lgb.train(params, lgb.Dataset(P[tr], label=y[tr]), num_boost_round=400).predict(P[va])
base_f1 = macro_f1(y[cov], stack_oof[cov].argmax(1))[0]
print(f"[base stack] pooled-OOF={base_f1:.4f}", flush=True)

# 2) top-2 reranker 피처/타깃 (cov만)
sc = stack_oof[cov]; yc = y[cov]
order = np.argsort(-sc, axis=1)
t1, t2 = order[:, 0], order[:, 1]
p1 = np.take_along_axis(sc, t1[:, None], 1)[:, 0]
p2 = np.take_along_axis(sc, t2[:, None], 1)[:, 0]
margin = p1 - p2
ent = -(sc * np.log(sc + 1e-9)).sum(1)
# reranker 타깃: top2가 실제 정답인가 (top1이 틀렸고 top2가 맞음)
tgt = (t2 == yc).astype(int)
# ngram이 top2를 top1보다 지지하는가
ng = M["ngram"][cov]
ng_t1 = np.take_along_axis(ng, t1[:, None], 1)[:, 0]
ng_t2 = np.take_along_axis(ng, t2[:, None], 1)[:, 0]
Rx = np.column_stack([p1, p2, margin, ent, ng_t2 - ng_t1, t1, t2, F[cov, 0], F[cov, 2], F[cov, 3]])

rparams = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
               feature_fraction=0.8, verbose=-1, num_threads=8, seed=1234)
# reranker OOF (cov 내부 재분할: 같은 세션fold 재사용)
rr = np.zeros(len(cov), np.float32)
covmap = {int(i): k for k, i in enumerate(cov)}
for tr, va in folds:
    trc = [covmap[int(i)] for i in tr if int(i) in covmap]
    vac = [covmap[int(i)] for i in va if int(i) in covmap]
    if not vac: continue
    m = lgb.train(rparams, lgb.Dataset(Rx[trc], label=tgt[trc]), num_boost_round=300)
    rr[vac] = m.predict(Rx[vac])

# 3) 게이트 τ, margin 상한 스윕
best = (base_f1, None)
pred = sc.argmax(1).copy()
for mg in [0.1, 0.2, 0.3, 0.5]:
    for tau in [0.4, 0.5, 0.6, 0.7]:
        flip = (margin < mg) & (rr > tau)
        p = pred.copy(); p[flip] = t2[flip]
        f1 = macro_f1(yc, p)[0]
        if f1 > best[0]:
            best = (f1, (mg, tau, int(flip.sum())))
print(f"[reranker] base={base_f1:.4f} best={best[0]:.4f} (+{best[0]-base_f1:.4f}) cfg={best[1]}", flush=True)
print(f"판정: {'reranker 채택' if best[0] > base_f1 + 0.003 else '이득 미미 — 보류'}")
