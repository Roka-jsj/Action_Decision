"""pairwise post-hoc 경계교정 (codex R11/R12) — list_directory 정밀도 0.384 병목 타격.

규칙: pred=A이고 runner-up=B이며 margin<th일 때 B로 스위치. (A,B)쌍·th를 OOF crossfit으로 학습.
- 5-fold 교차적합: fold별로 나머지 4fold에서 최적 (pair, th) 선택 → 해당 fold에 적용 → pooled 평가.
- 입력: tri_cond와 동일한 조건부 혼합 OOF + bias 적용 후의 log-prob (배포와 동일 지점에 삽입).
판정(codex): pooled Δ ≥ +0.0007 → 축 인정·배포 / +0.00035~0.0007 → 방향(1발 시험) / 미만 폐기.
"""
from __future__ import annotations
import os, sys, glob, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.io_utils import load_train, CLASSES
from common.cv import make_splits
from common.postproc import fit_bias
from sklearn.metrics import f1_score

samples, y, ids = load_train(); y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups); folds = sp["folds"]
cov = np.concatenate([f[1] for f in folds])
E = "action_decision_maximum/experiments/"

def asm(pats):
    o = np.zeros((len(samples), 14), np.float32); cs = set()
    for pat in pats:
        for p in sorted(glob.glob(E + pat)):
            z = np.load(p, allow_pickle=True)
            for f in range(int(z["fold_lo"]), int(z["fold_hi"])):
                if f in cs: continue
                o[folds[f][1]] = z["oof"][folds[f][1]]; cs.add(f)
    return o

v6 = asm(["teacher_largev6A_a*.npz", "teacher_largev6B_a*.npz"])
be = asm(["teacher_basev6e5_g0.npz"])
v4 = asm(["teacher_largev4mix.npz"])
# tri_cond 배포와 동일 혼합 (w 0.6/0.15/0.25, th 0.5)
W = (0.6, 0.15, 0.25)
p12 = (W[0] * v6 + W[1] * be) / (W[0] + W[1])
srt = np.sort(p12, axis=1)
sel = (srt[:, -1] - srt[:, -2]) < 0.5
mix = p12.copy()
mix[sel] = W[0] * v6[sel] + W[1] * be[sel] + W[2] * v4[sel]
b, _ = fit_bias(np.log(mix[cov] + 1e-9), y[cov])
S = np.log(mix + 1e-9) + np.array(b)     # 배포 동일: bias 적용 log-prob

ci = {c: i for i, c in enumerate(CLASSES)}
fold_of = np.full(len(samples), -1)
for fi, (_, va) in enumerate(folds):
    fold_of[va] = fi

top2 = np.argsort(S, axis=1)[:, -2:]      # [:,1]=top1, [:,0]=top2
pred0 = top2[:, 1]
run0 = top2[:, 0]
marg = np.take_along_axis(S, top2[:, 1:], 1)[:, 0] - np.take_along_axis(S, top2[:, :1], 1)[:, 0]

base_pool = f1_score(y[cov], pred0[cov], average="macro")
print(f"기준(pooled, tri_cond OOF+bias): {base_pool:.5f}")

# 후보 쌍: 탐색계열 + 혼동 상위쌍
EXPC = [ci[c] for c in ["list_directory", "read_file", "grep_search", "glob_pattern"]]
PAIRS = [(a, b) for a in EXPC for b in EXPC if a != b] + \
        [(ci["run_tests"], ci["lint_or_typecheck"]), (ci["lint_or_typecheck"], ci["run_tests"]),
         (ci["ask_user"], ci["web_search"]), (ci["web_search"], ci["ask_user"])]
THS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.45]

def apply_rules(pred, rules, idx):
    p = pred[idx].copy()
    for (a, bcls, th) in rules:
        m = (pred[idx] == a) & (run0[idx] == bcls) & (marg[idx] < th)
        p[m] = bcls
    return p

# 규칙 1개 crossfit: fold별로 train-folds에서 최고 (pair,th) 선택
deltas = []
chosen = []
for fi in range(5):
    tr = cov[np.isin(fold_of[cov], [f for f in range(5) if f != fi])]
    va = cov[fold_of[cov] == fi]
    base_tr = f1_score(y[tr], pred0[tr], average="macro")
    best = (0.0, None)
    for (a, bcls) in PAIRS:
        for th in THS:
            s = f1_score(y[tr], apply_rules(pred0, [(a, bcls, th)], tr), average="macro")
            if s - base_tr > best[0]:
                best = (s - base_tr, (a, bcls, th))
    if best[1] is None:
        deltas.append(0.0); chosen.append(None); continue
    a, bcls, th = best[1]
    dv = f1_score(y[va], apply_rules(pred0, [best[1]], va), average="macro") - \
         f1_score(y[va], pred0[va], average="macro")
    deltas.append(dv)
    chosen.append((CLASSES[a], CLASSES[bcls], th, round(best[0], 5)))
    print(f"fold{fi}: 선택 {CLASSES[a]}->{CLASSES[bcls]} th={th} (train Δ{best[0]:+.5f}) → val Δ{dv:+.5f}")

md = float(np.mean(deltas))
print(f"\ncrossfit 평균 val Δ = {md:+.5f}  (양수 {sum(d > 0 for d in deltas)}/5)")
print("판정:", "축 인정(+0.0007↑) → 배포 구현" if md >= 0.0007 else
      ("방향(+0.00035↑) → 1발 시험" if md >= 0.00035 else "폐기"))
