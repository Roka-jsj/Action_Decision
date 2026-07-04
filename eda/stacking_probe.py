"""스태킹 프로브 — 교사 확률 + 구조피처 → LightGBM 메타 vs 단순평균(+bias).

동일 세션-fold로 메타 CV(표준 스태킹). holdout으로 최종 비교.
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

samples, y, ids = load_train(); y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups); folds = sp["folds"]; hold = sp["holdout_idx"]
cov = np.concatenate([f[1] for f in folds])

KEYS = ["base_s1234", "base_s777e5"]           # 최적 페어(+mdeberta 오면 추가)
extra = sorted(glob.glob("action_decision_maximum/experiments/teacher_mdeberta*.npz"))
T = {}
for k in KEYS:
    d = np.load(f"action_decision_maximum/experiments/teacher_{k}.npz", allow_pickle=True)
    T[k] = (d["oof"], d["hold"])
for f in extra:
    d = np.load(f, allow_pickle=True)
    if int(d["fold_hi"]) - int(d["fold_lo"]) >= 3:   # 3fold 이상 커버 시만
        T["mdeberta"] = (d["oof"], d["hold"])
        KEYS.append("mdeberta")
print("teachers:", KEYS)

# ---- 구조 피처 ----
ACT2I = {c: i for i, c in enumerate(CLASSES)}
ST2I = {s: i for i, s in enumerate(["na","success","error","test_fail","test_pass","zero","nonzero_exit"])}
def feats(s):
    nm, args, rs, st = ad_lib.last_action(s)
    m = ad_lib.meta_fields(s)
    seq = ad_lib.action_sequence(s)
    f = [ACT2I.get(nm, -1), ST2I.get(st, 0), len(seq), m["turn_index"],
         {"passed":0,"failed":1,"none":2}.get(m["last_ci_status"], 2), int(m["git_dirty"]),
         m["n_open_files"], {"sim":0,"au":1}[s["gen"]],
         len(s.get("current_prompt") or ""), int(ad_lib.has_hangul(s.get("current_prompt") or ""))]
    cnt = np.zeros(14);
    for a in seq:
        if a in ACT2I: cnt[ACT2I[a]] += 1
    return f + cnt.tolist()

F = np.array([feats(s) for s in samples], dtype=np.float32)
P = np.concatenate([T[k][0] for k in KEYS], axis=1)             # (N, 14*k)
X = np.concatenate([P, F], axis=1)
Ph = np.concatenate([T[k][1] for k in KEYS], axis=1)
Fh = F[hold]
Xh = np.concatenate([Ph, Fh], axis=1)

# ---- 기준선: 단순평균(+bias) ----
mean_oof = sum(T[k][0] for k in KEYS) / len(KEYS)
mean_hold = sum(T[k][1] for k in KEYS) / len(KEYS)
po = macro_f1(y[cov], mean_oof[cov].argmax(1))[0]
bias, fit = fit_bias(np.log(mean_oof[cov] + 1e-9), y[cov])
hb = macro_f1(y[hold], (np.log(mean_hold + 1e-9) + bias).argmax(1))[0]
print(f"[mean]    pooled-OOF={po:.4f}  holdout+bias={hb:.4f}")

# ---- 스태킹: 세션-fold 그대로 메타 CV ----
params = dict(objective="multiclass", num_class=14, learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              verbose=-1, num_threads=8)
oof_meta = np.zeros((len(samples), NUM_CLASSES), np.float32)
for fi, (tr, va) in enumerate(folds):
    ds = lgb.Dataset(X[tr], label=y[tr])
    m = lgb.train(params, ds, num_boost_round=400)
    oof_meta[va] = m.predict(X[va])
pm = macro_f1(y[cov], oof_meta[cov].argmax(1))[0]
bias2, fit2 = fit_bias(np.log(oof_meta[cov] + 1e-9), y[cov])
# holdout: dev 전체로 재학습
mfull = lgb.train(params, lgb.Dataset(X[cov], label=y[cov]), num_boost_round=400)
hm = mfull.predict(Xh)
hb2 = macro_f1(y[hold], (np.log(hm + 1e-9) + bias2).argmax(1))[0]
hn2 = macro_f1(y[hold], hm.argmax(1))[0]
print(f"[stack]   pooled-OOF={pm:.4f}  holdout={hn2:.4f}  +bias={hb2:.4f}")
print(f"\n판정: {'STACK 채택' if hb2 > hb + 0.002 else 'MEAN 유지(스태킹 이득 미미)'} | LB예측: mean≈{hb-0.018:.4f} stack≈{hb2-0.018:.4f}")
