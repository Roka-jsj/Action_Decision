#!/usr/bin/env python
"""R29 transductive 후보 검증: EM(Saerens) label-shift 보정 홀드아웃 시뮬레이션.

zip 스크립트는 serve-time에 히든 30k 전체를 본다 → 모델 probs만으로 히든 prior를
EM 추정해 재가중할 수 있다(제출 프로빙 불필요). 이 스크립트는 그 이득/해악을
홀드아웃 prior-shift 리샘플로 측정한다.

방법: p_i = 5-fold teacher 홀드아웃 평균확률. π_ref = OOF 평균확률(weighted-CE라
train빈도 아닌 모델 내재 prior 사용). EM: w_i(c)∝p_i(c)·π_c/π_ref_c, π←mean w.
shrink s: π_use = (1-s)·π_ref + s·π_EM (s=0 → no-op, do-no-harm 손잡이).

비교: raw argmax / +bias(OOF coord-ascent, 배포 동일) / EM / EM+bias.
시나리오: identity / Dirichlet(α·π, α=100/30/10) / minority×2 / flatten half.
"""
from __future__ import annotations
import sys, glob
import numpy as np

sys.path.insert(0, "/root/Action_Decision")
from common.io_utils import load_train, CLASSES, NUM_CLASSES
from common.cv import make_splits
from common.metrics import macro_f1
from common.postproc import fit_bias

samples, y, ids = load_train()
y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups)
hold_idx = np.asarray(sp["holdout_idx"])
y_hold = y[hold_idx]

# 5-fold teacher 앙상블 (배포 largeonly와 같은 계열)
P_hold, P_oof, cov = None, np.zeros((len(y), NUM_CLASSES), np.float64), np.zeros(len(y), bool)
files = sorted(glob.glob("/root/Action_Decision/action_decision_maximum/experiments/teacher_largev6[AB]_a*.npz"))
assert len(files) == 5, files
for f in files:
    z = np.load(f, allow_pickle=True)
    h = z["hold"].astype(np.float64)
    P_hold = h if P_hold is None else P_hold + h
    o = z["oof"].astype(np.float64)
    m = o.sum(1) > 0
    P_oof[m] = o[m]; cov |= m
P_hold /= len(files)
P_hold /= P_hold.sum(1, keepdims=True)
oof_idx = np.where(cov)[0]
P_oof_c = P_oof[oof_idx]; P_oof_c /= P_oof_c.sum(1, keepdims=True)
y_oof = y[oof_idx]

pi_ref = P_oof_c.mean(0)                      # 모델 내재 prior (OOF 평균)
pi_train = np.bincount(y, minlength=NUM_CLASSES) / len(y)
bias, bias_f1 = fit_bias(np.log(P_oof_c + 1e-12), y_oof)
print(f"[setup] hold={len(y_hold)} oof={len(y_oof)} bias_oof_f1={bias_f1:.4f}")
print(f"[setup] pi_ref(모델) vs pi_train L1={np.abs(pi_ref-pi_train).sum():.4f}")

def em_prior(P, pi0, iters=100, tol=1e-8):
    pi = pi0.copy()
    for _ in range(iters):
        w = P * (pi / pi0)
        w /= w.sum(1, keepdims=True)
        newpi = w.mean(0)
        if np.abs(newpi - pi).sum() < tol:
            pi = newpi; break
        pi = newpi
    return pi

def correct(P, pi_use, pi0):
    w = P * (pi_use / pi0)
    return w / w.sum(1, keepdims=True)

def eval_methods(idx_sample):
    P = P_hold[idx_sample]; yt = y_hold[idx_sample]
    out = {}
    out["raw"] = macro_f1(yt, P.argmax(1))[0]
    lb = np.log(P + 1e-12) + bias
    out["bias"] = macro_f1(yt, lb.argmax(1))[0]
    pi_em = em_prior(P, pi_ref)
    for s in (0.5, 1.0):
        pi_use = (1 - s) * pi_ref + s * pi_em
        Pc = correct(P, pi_use, pi_ref)
        out[f"em{s}"] = macro_f1(yt, Pc.argmax(1))[0]
        out[f"em{s}+bias"] = macro_f1(yt, (np.log(Pc + 1e-12) + bias).argmax(1))[0]
    out["_pi_em_l1_true"] = np.abs(pi_em - np.bincount(yt, minlength=NUM_CLASSES)/len(yt)).sum()
    return out

def resample_to_prior(pi_target, n, rng):
    per = np.maximum((pi_target * n).astype(int), 1)
    idxs = []
    for c in range(NUM_CLASSES):
        pool = np.where(y_hold == c)[0]
        if len(pool) == 0: continue
        idxs.append(rng.choice(pool, per[c], replace=True))
    return np.concatenate(idxs)

pi_hold = np.bincount(y_hold, minlength=NUM_CLASSES) / len(y_hold)
scen = [("identity", None)]
rng0 = np.random.RandomState(7)
for alpha, k in ((100, 3), (30, 3), (10, 3)):
    for s_i in range(k):
        scen.append((f"dir{alpha}#{s_i}", rng0.dirichlet(alpha * pi_hold)))
mino = pi_hold.copy(); mino[pi_hold < 0.05] *= 2.0; mino /= mino.sum()
scen.append(("minor2x", mino))
scen.append(("flat50", 0.5 * pi_hold + 0.5 / NUM_CLASSES))

METHODS = ["raw", "bias", "em0.5", "em0.5+bias", "em1.0", "em1.0+bias"]
print(f"\n{'scenario':<12} {'L1shift':>7} " + " ".join(f"{m:>10}" for m in METHODS) + "  pi_err")
agg = {}
for name, pi_t in scen:
    if pi_t is None:
        res = [eval_methods(np.arange(len(y_hold)))]
        l1 = 0.0
    else:
        rng = np.random.RandomState(hash(name) % 2**31)
        res = [eval_methods(resample_to_prior(pi_t, 6000, rng)) for _ in range(5)]
        l1 = np.abs(pi_t - pi_hold).sum()
    mean = {m: np.mean([r[m] for r in res]) for m in METHODS}
    pi_err = np.mean([r["_pi_em_l1_true"] for r in res])
    agg[name] = (l1, mean)
    print(f"{name:<12} {l1:7.3f} " + " ".join(f"{mean[m]:10.4f}" for m in METHODS) + f"  {pi_err:.3f}")

print("\n=== Δ vs bias(배포 기준) ===")
for name, (l1, mean) in agg.items():
    print(f"{name:<12} em0.5+bias {mean['em0.5+bias']-mean['bias']:+.4f}   em1.0+bias {mean['em1.0+bias']-mean['bias']:+.4f}")
