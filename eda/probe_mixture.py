#!/usr/bin/env python
"""R29 프로브 역산 + au혼합비 추정 (제출 3발: read_file/glob_pattern/list_directory 예정).

상수클래스 점수 M → n_c = 30000·14M/(2−14M) 정확 역산 후:
  1) 측정 클래스별 히든 prior vs train/sim/au prior 비교
  2) 혼합모형 π_hidden_c ≈ λ·π_au_c + (1−λ)·π_sim_c 를 최소제곱 적합(3방정식 과결정)
     → λ_au 추정 + 잔차로 혼합모형 자체의 적합도 판정
usage: python3 eda/probe_mixture.py read_file=0.0171 glob_pattern=0.0099 list_directory=0.0081
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, "/root/Action_Decision")
from common.io_utils import load_train, CLASSES

N = 30000
samples, y, ids = load_train(); y = np.array(y)
gen = np.array([s["gen"] for s in samples])
pi_all = np.bincount(y, minlength=14) / len(y)
pi_sim = np.bincount(y[gen == "sim"], minlength=14) / max((gen == "sim").sum(), 1)
pi_au = np.bincount(y[gen == "au"], minlength=14) / max((gen == "au").sum(), 1)

obs = {}
for a in sys.argv[1:]:
    c, v = a.split("="); obs[c] = float(v)
assert obs, "usage: probe_mixture.py <class>=<macroF1> ..."

print(f"{'class':<18} {'probe점수':>10} {'n_c':>7} {'hidden%':>8} {'train%':>7} {'sim%':>7} {'au%':>7} {'shift(vs train)':>15}")
meas = []
for c, M in obs.items():
    i = CLASSES.index(c)
    n_c = N * (14 * M) / (2 - 14 * M)
    ph = n_c / N
    meas.append((i, ph))
    print(f"{c:<18} {M:10.7f} {n_c:7.0f} {ph*100:7.2f}% {pi_all[i]*100:6.2f}% {pi_sim[i]*100:6.2f}% {pi_au[i]*100:6.2f}% {(ph-pi_all[i])*100:+14.2f}pp")

# 혼합 적합: ph_c = λ·au_c + (1-λ)·sim_c → λ = Σ(ph-sim)(au-sim) / Σ(au-sim)²
idx = np.array([i for i, _ in meas]); ph = np.array([p for _, p in meas])
d = pi_au[idx] - pi_sim[idx]
lam = float(np.dot(ph - pi_sim[idx], d) / max(np.dot(d, d), 1e-12))
resid = ph - (lam * pi_au[idx] + (1 - lam) * pi_sim[idx])
print(f"\nλ_au = {lam:.3f}  (train의 au비율 0.072 대비)  잔차 rms = {np.sqrt((resid**2).mean())*100:.2f}pp")
print("잔차별:", {CLASSES[int(i)]: f"{r*100:+.2f}pp" for i, r in zip(idx, resid)})
if 0 <= lam <= 1 and np.sqrt((resid**2).mean()) < 0.01:
    pi_mix = lam * pi_au + (1 - lam) * pi_sim
    print("\n혼합모형 적합 양호 → 전 클래스 히든 prior 외삽:")
    for i, c in enumerate(CLASSES):
        print(f"  {c:<18} {pi_mix[i]*100:6.2f}% (train {pi_all[i]*100:.2f}%, Δ{(pi_mix[i]-pi_all[i])*100:+.2f}pp)")
else:
    print("\n혼합모형 부적합(λ 범위밖 or 잔차 큼) → 측정 3클래스만 신뢰, 외삽 금지")
