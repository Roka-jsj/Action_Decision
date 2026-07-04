"""4멤버 스택 평가 — base_s1234 + base_s777e5 + klue + large(분할 npz 병합).

large = probe2(fold0) + largeA_a*(1-2) + largeB_a*(3-4) 병합:
  oof: fold별 배타 행 복사 / hold: fold수 가중평균.
멤버 조합별 mean(+bias) vs LightGBM stack(+bias) 비교 → 최종 배포 조합 결정.
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.io_utils import load_train, CLASSES, NUM_CLASSES
from common.cv import make_splits
from common.metrics import macro_f1
from common.postproc import fit_bias
from common import ad_lib
import lightgbm as lgb

EXPD = "action_decision_maximum/experiments"
samples, y, ids = load_train(); y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups); folds = sp["folds"]; hold = sp["holdout_idx"]
cov = np.concatenate([f[1] for f in folds])


def merge_large():
    """분할 학습된 large npz들을 5-fold 하나로 병합. 커버 fold 반환."""
    paths = ([f"{EXPD}/teacher_large_probe2.npz"]
             + sorted(glob.glob(f"{EXPD}/teacher_largeA_a*.npz"))
             + sorted(glob.glob(f"{EXPD}/teacher_largeB_a*.npz")))
    oof = np.zeros((len(samples), NUM_CLASSES), np.float32)
    hold_sum = np.zeros((len(hold), NUM_CLASSES), np.float32)
    covered, w = set(), 0
    for p in paths:
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        lo, hi = int(z["fold_lo"]), int(z["fold_hi"])
        n = 0
        for fi in range(lo, hi):
            if fi in covered:
                continue
            va = folds[fi][1]
            oof[va] = z["oof"][va]
            covered.add(fi); n += 1
        if n:
            hold_sum += z["hold"] * n; w += n
        print(f"  [large merge] {os.path.basename(p)}: folds[{lo},{hi}) scores={np.round(z['scores'],4)}")
    return oof, hold_sum / max(w, 1), covered


T = {}
for k in ["base_s1234", "base_s777e5", "klue_s1234"]:
    d = np.load(f"{EXPD}/teacher_{k}.npz", allow_pickle=True)
    T[k.replace("_s1234", "").replace("_s777e5", "_e5")] = (d["oof"], d["hold"])
    if int(d["fold_hi"]) < 5:
        print(f"!! {k}: fold_hi={int(d['fold_hi'])} — 미완")
lo_oof, lo_hold, lcov = merge_large()
print(f"large covered folds: {sorted(lcov)}")
if len(lcov) == 5:
    T["large"] = (lo_oof, lo_hold)

ACT2I = {c: i for i, c in enumerate(CLASSES)}
F = np.array([ad_lib.stack_features(s) for s in samples], dtype=np.float32)
Fh = F[hold]

params = dict(objective="multiclass", num_class=14, learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              verbose=-1, num_threads=8)


def eval_combo(keys, rounds=400):
    P = np.concatenate([T[k][0] for k in keys], axis=1)
    Ph = np.concatenate([T[k][1] for k in keys], axis=1)
    X = np.concatenate([P, F], axis=1); Xh = np.concatenate([Ph, Fh], axis=1)
    # mean 기준선
    mo = sum(T[k][0] for k in keys) / len(keys)
    mh = sum(T[k][1] for k in keys) / len(keys)
    po = macro_f1(y[cov], mo[cov].argmax(1))[0]
    b, _ = fit_bias(np.log(mo[cov] + 1e-9), y[cov])
    hb = macro_f1(y[hold], (np.log(mh + 1e-9) + b).argmax(1))[0]
    # stack: 메타 CV → bias는 메타 OOF로 적합 → holdout은 dev 전체 재학습 모델로
    om = np.zeros((len(samples), NUM_CLASSES), np.float32)
    for tr, va in folds:
        m = lgb.train(params, lgb.Dataset(X[tr], label=y[tr]), num_boost_round=rounds)
        om[va] = m.predict(X[va])
    pm = macro_f1(y[cov], om[cov].argmax(1))[0]
    b2, _ = fit_bias(np.log(om[cov] + 1e-9), y[cov])
    mfull = lgb.train(params, lgb.Dataset(X[cov], label=y[cov]), num_boost_round=rounds)
    hm = mfull.predict(Xh)
    hb2 = macro_f1(y[hold], (np.log(hm + 1e-9) + b2).argmax(1))[0]
    print(f"[{'+'.join(keys)}]")
    print(f"   mean : pooled-OOF={po:.4f}  holdout+bias={hb:.4f}  (LB≈{hb-0.018:.4f})")
    print(f"   stack: pooled-OOF={pm:.4f}  holdout+bias={hb2:.4f}  (LB≈{hb2-0.018:.4f})")
    return hb2


combos = [["base", "base_e5", "klue"]]
if "large" in T:
    combos = [["base_e5", "klue", "large"],
              ["base", "base_e5", "klue", "large"],
              ["klue", "large"],
              ["base_e5", "large"]]
best, bk = -1, None
for c in combos:
    v = eval_combo(c)
    if v > best:
        best, bk = v, c
print(f"\n=== BEST: {'+'.join(bk)} holdout+bias={best:.4f} LB예측≈{best-0.018:.4f} ===")
