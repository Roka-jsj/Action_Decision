"""bias의 sim-only 적합 프로브 (R14 하네스 축, 5.14 발굴 후 미실행 무료 레버).

가설: 테스트=전부 sim이면 au(7%) 포함 OOF로 적합한 bias는 준최적.
방법: teacher largev6 OOF(6ep, 5fold)로 bias_all vs bias_sim 적합
      → teacher hold 확률(5810행)에서 전체/​sim-subset 채점 2×2 비교.
주의: 절대치는 6ep-teacher 프록시(배포는 8ep FULL). delta만 신뢰.
"""
from __future__ import annotations
import glob, os, sys
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
from common.io_utils import load_train
from common.cv import make_splits
from common.postproc import fit_bias, to_logprobs
from common.metrics import macro_f1

samples, y, ids = load_train(); y = np.array(y)
ids = np.array([str(i) for i in ids])
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups)
folds = sp["folds"]; hold_idx = sp["holdout_idx"]
cov = np.concatenate([f[1] for f in folds])

oof = np.zeros((len(samples), 14), np.float32); cs = set()
holds = []
for p in sorted(glob.glob(os.path.join(R, "action_decision_maximum/experiments/teacher_largev6[AB]_a*.npz"))):
    z = np.load(p, allow_pickle=True)
    for f in range(int(z["fold_lo"]), int(z["fold_hi"])):
        if f in cs: continue
        oof[folds[f][1]] = z["oof"][folds[f][1]]; cs.add(f)
        holds.append(np.asarray(z["hold"], np.float64))
hold_p = np.mean(holds, axis=0)
print(f"[probe] folds={sorted(cs)} hold파일={len(holds)}", flush=True)

au = np.char.startswith(ids, "sess_au")
print(f"[probe] au 비율: train {au.mean():.4f} / cov {au[cov].mean():.4f} / holdout {au[hold_idx].mean():.4f}", flush=True)

lp_oof = np.log(oof + 1e-9)
b_all, s_all = fit_bias(lp_oof[cov], y[cov])
print(f"[fit] bias_all  OOF={s_all:.4f}", flush=True)
cov_sim = cov[~au[cov]]
b_sim, s_sim = fit_bias(lp_oof[cov_sim], y[cov_sim])
print(f"[fit] bias_sim  OOF(sim)={s_sim:.4f}", flush=True)

lp_h = to_logprobs(np.log(hold_p + 1e-9))
yh = y[hold_idx]; sim_h = ~au[hold_idx]
def sc(b, m):
    return macro_f1(yh[m], (lp_h[m] + b).argmax(1), 14)[0]
allm = np.ones(len(yh), bool)
print(f"\n=== holdout 채점 (teacher-6ep 프록시) ===")
print(f"bias_all : 전체 {sc(b_all,allm):.5f} | sim-only {sc(b_all,sim_h):.5f}")
print(f"bias_sim : 전체 {sc(b_sim,allm):.5f} | sim-only {sc(b_sim,sim_h):.5f}")
print(f"delta(sim-subset, sim-bias 이득): {sc(b_sim,sim_h)-sc(b_all,sim_h):+.5f}")
print(f"bias 차이 L1: {np.abs(np.array(b_all)-np.array(b_sim)).sum():.3f}")
