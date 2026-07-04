"""배포용 스태커 적합 — 지정 멤버 조합의 LightGBM 메타 + bias 산출.

usage: python eda/fit_stacker.py <outdir> <key1> <key2> [...]
keys: base | base_e5 | klue | large(분할 npz 자동병합)

절차(스태킹 표준):
  1) 세션-fold 메타 CV → meta-OOF로 성능 측정 + bias 적합
  2) dev(cov) 전체 재학습 booster = 배포본 (meta.lgb, 텍스트 저장)
  3) holdout으로 최종 검증 (LB예측 = holdout+bias - 0.018)
출력: <outdir>/meta.lgb, postproc.json, stack_meta.json
※ 배포 run_meta.json의 ensemble 멤버 순서 = 여기 키 순서와 반드시 동일해야 함.
"""
from __future__ import annotations
import os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.io_utils import load_train, NUM_CLASSES
from common.cv import make_splits
from common.metrics import macro_f1
from common.postproc import fit_bias, save as save_bias
from common import ad_lib
import lightgbm as lgb

EXPD = "action_decision_maximum/experiments"
outdir = sys.argv[1]
keys = sys.argv[2:]
assert keys, "usage: fit_stacker.py <outdir> <key...>"
os.makedirs(outdir, exist_ok=True)

samples, y, ids = load_train(); y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups); folds = sp["folds"]; hold = sp["holdout_idx"]
# FOLDS env: 멤버 OOF가 부분 커버일 때(예: "0123") 해당 fold 행만 사용
FOLD_IDS = [int(c) for c in os.environ.get("FOLDS", "01234")]
cov = np.concatenate([folds[i][1] for i in FOLD_IDS])

# 멤버 = npz 패턴 목록(fold 분할·재기동 병합). oof=fold별 배타 행, hold=fold수 가중평균.
FAMILY = {
    "base": ["teacher_base_s1234.npz"],
    "base_e5": ["teacher_base_s777e5.npz"],
    "klue": ["teacher_klue_s1234.npz"],
    "klue6": ["teacher_klue6ep_a*.npz"],
    "large": ["teacher_large_probe2.npz", "teacher_largeA_a*.npz", "teacher_largeB_a*.npz"],
    "large6": ["teacher_large6ep_a*.npz", "teacher_lg6b_a*.npz", "teacher_lg6c_a*.npz"],
    "large_v5": ["teacher_largev5p_a*.npz"],
    "basev6": ["teacher_basev6_a*.npz"],
    "largev6": ["teacher_largev6A_a*.npz", "teacher_largev6B_a*.npz"],   # 순수 6ep
    "large8v6": ["teacher_large8v6_a*.npz"],                             # 8ep 프로브(f0)
    "kluev6": ["teacher_kluev6_g*.npz"],
    "basev6e5": ["teacher_basev6e5_g*.npz"],
}


def load_member(k, require=5):
    paths = []
    for pat in FAMILY[k]:
        paths += sorted(glob.glob(f"{EXPD}/{pat}")) if "*" in pat else [f"{EXPD}/{pat}"]
    oof = np.zeros((len(samples), NUM_CLASSES), np.float32)
    hs = np.zeros((len(hold), NUM_CLASSES), np.float32)
    covered, w = set(), 0
    for p in paths:
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        n = 0
        for fi in range(int(z["fold_lo"]), int(z["fold_hi"])):
            if fi in covered:
                continue
            oof[folds[fi][1]] = z["oof"][folds[fi][1]]
            covered.add(fi); n += 1
        if n:
            hs += z["hold"] * n; w += n
    need = set(FOLD_IDS) if require == len(FOLD_IDS) else set(range(require))
    assert need <= covered, f"{k} covered={sorted(covered)} ⊉ 필요 {sorted(need)}"
    return oof, hs / max(w, 1)


T = {k: load_member(k, require=len(FOLD_IDS)) for k in keys}
F = np.array([ad_lib.stack_features(s) for s in samples], dtype=np.float32)
X = np.concatenate([T[k][0] for k in keys] + [F], axis=1)
Xh = np.concatenate([T[k][1] for k in keys] + [F[hold]], axis=1)

params = dict(objective="multiclass", num_class=14, learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              verbose=-1, num_threads=8, seed=1234)
ROUNDS = 400

# 1) 메타 CV → 측정 + bias (FOLDS 부분커버 시: 커버 fold 행만 train/val)
om = np.zeros((len(samples), NUM_CLASSES), np.float32)
for fi in FOLD_IDS:
    va = folds[fi][1]
    tr = np.setdiff1d(cov, va)
    m = lgb.train(params, lgb.Dataset(X[tr], label=y[tr]), num_boost_round=ROUNDS)
    om[va] = m.predict(X[va])
pooled = macro_f1(y[cov], om[cov].argmax(1))[0]
bias, fitted = fit_bias(np.log(om[cov] + 1e-9), y[cov])

# 2) 배포 booster (dev 전체)
mfull = lgb.train(params, lgb.Dataset(X[cov], label=y[cov]), num_boost_round=ROUNDS)
mfull.save_model(os.path.join(outdir, "meta.lgb"))
save_bias(os.path.join(outdir, "postproc.json"), bias, meta={"fitted_on": "stack_meta_oof"})

# 3) holdout 검증
hp = mfull.predict(Xh)
h_raw = macro_f1(y[hold], hp.argmax(1))[0]
h_bias = macro_f1(y[hold], (np.log(hp + 1e-9) + np.array(bias)).argmax(1))[0]

# mean 기준선(참고)
mo = sum(T[k][0] for k in keys) / len(keys)
mh = sum(T[k][1] for k in keys) / len(keys)
bm, _ = fit_bias(np.log(mo[cov] + 1e-9), y[cov])
hb_mean = macro_f1(y[hold], (np.log(mh + 1e-9) + np.array(bm)).argmax(1))[0]

meta = {"members": keys, "n_features": int(X.shape[1]), "rounds": ROUNDS,
        "pooled_oof": round(float(pooled), 4), "holdout": round(float(h_raw), 4),
        "holdout_bias": round(float(h_bias), 4), "holdout_mean_bias": round(float(hb_mean), 4),
        "lb_pred": round(float(h_bias) - 0.018, 4)}
json.dump(meta, open(os.path.join(outdir, "stack_meta.json"), "w"), indent=1)
print(f"[stacker {'+'.join(keys)}] pooled-OOF={pooled:.4f} holdout={h_raw:.4f} "
      f"+bias={h_bias:.4f} (mean+bias={hb_mean:.4f}) LB예측≈{h_bias-0.018:.4f}")
print(f"saved → {outdir}/meta.lgb postproc.json stack_meta.json")
