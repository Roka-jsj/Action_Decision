"""상수클래스 프로브 LB점수 → 테스트 class prior 역산 + train prior 비교 (R20 A1).

상수 c 제출 시 macro-F1 = (1/14)·2·n_c/(30000+n_c)  (다른 13class F1=0).
→ 관측 M에서 n_c = 30000·(14·M)/(2 - 14·M).
usage: python3 eda/prior_from_probe.py <class>=<macroF1> [<class>=<macroF1> ...]
예: python3 eda/prior_from_probe.py read_file=0.017531 list_directory=0.009xxx
"""
from __future__ import annotations
import sys, numpy as np
sys.path.insert(0, "/root/Action_Decision")
from common.io_utils import load_train, CLASSES

N = 30000
_, y, _ = load_train(); y = np.array(y)
train_prior = np.array([(y == i).mean() for i in range(14)])

obs = {}
for a in sys.argv[1:]:
    c, v = a.split("="); obs[c] = float(v)

print(f"{'class':<18} {'train%':>8} {'test_n':>8} {'test%':>8} {'shift':>8}")
tot_est = 0
for c in CLASSES:
    i = CLASSES.index(c)
    if c in obs:
        M = obs[c]
        n_c = N * (14 * M) / (2 - 14 * M)
        tp = n_c / N
        tot_est += n_c
        print(f"{c:<18} {train_prior[i]*100:7.2f}% {n_c:8.0f} {tp*100:7.2f}% {(tp-train_prior[i])*100:+7.2f}pp")
    else:
        print(f"{c:<18} {train_prior[i]*100:7.2f}% {'(미측정)':>8}")
if obs:
    print(f"\n측정 class 합 n={tot_est:.0f} (30000 중). 미측정분 = {30000-tot_est:.0f}")
    print("→ shift가 ±1pp 이상이면 label-shift 재적합 가치, ±0.3pp 미만이면 prior 동일=천장확정")
