#!/usr/bin/env python
"""R33 mdeberta 게이트 판정 (codex 락 기준).

ρ_low = corr(mdeberta 잔차, large-v6 잔차) — fold0-val의 대형모델 저마진 27% 구간.
잔차 r_{i,c} = 1[y_i=c] − p_{i,c} (클래스 차원 포함 flatten 상관).
게이트: 통과 ρ<0.82 / 강통과 ρ<0.75 / ρ≥0.90 은행 영구금지.
fold0 성능: mdeb > large_fold0(0.7485) − 0.0015 (동일 하네스 기준).
usage: python3 eda/mdeb_gate.py
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, "/root/Action_Decision")
from common.io_utils import load_train, NUM_CLASSES
from common.cv import make_splits
from common.metrics import macro_f1

samples, y, ids = load_train(); y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups)
va0 = np.asarray(sp["folds"][0][1])

zl = np.load("/root/Action_Decision/work/teacher_largev6_f0ckpt.npz", allow_pickle=True)
P_l = zl["oof"][va0].astype(np.float64); P_l /= P_l.sum(1, keepdims=True)
zm = np.load("/root/Action_Decision/work/teacher_mdeb_f0.npz", allow_pickle=True)
P_m = zm["oof"][va0].astype(np.float64); P_m /= P_m.sum(1, keepdims=True)
yv = y[va0]

f1_l = macro_f1(yv, P_l.argmax(1))[0]
f1_m = macro_f1(yv, P_m.argmax(1))[0]
print(f"fold0: large {f1_l:.4f} vs mdeberta {f1_m:.4f} ({f1_m-f1_l:+.4f})  [게이트: > {f1_l-0.0015:.4f}]")

srt = np.sort(P_l, 1); margin = srt[:, -1] - srt[:, -2]
Y1 = np.eye(NUM_CLASSES)[yv]
for name, mask in (("low27%", margin < np.quantile(margin, 0.27)), ("전체", np.ones(len(yv), bool))):
    rl = (Y1[mask] - P_l[mask]).ravel()
    rm = (Y1[mask] - P_m[mask]).ravel()
    rho = float(np.corrcoef(rl, rm)[0, 1])
    print(f"ρ({name}) = {rho:.3f}", end="")
    if name == "low27%":
        v = "강통과" if rho < 0.75 else ("통과" if rho < 0.82 else ("은행금지" if rho >= 0.90 else "불통과"))
        print(f"  → 게이트 {v}")
    else:
        print()

# 5% blend 프로브 (fold0-val, 클린): P = 0.95·large + 0.05·mdeb
for w in (0.05, 0.10):
    Pb = (1 - w) * P_l + w * P_m
    f1b = macro_f1(yv, Pb.argmax(1))[0]
    print(f"blend {int(w*100)}%: {f1b:.4f} ({f1b-f1_l:+.4f})  [게이트 Δ>+0.00025]")
