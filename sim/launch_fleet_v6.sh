#!/bin/bash
# launch_fleet_v6.sh — 충전 후 원커맨드: large-v6 함대 + 8ep 프로브 + FULL 멤버(에폭 자동결정)
# 슬롯A: largev6A(f0-2, 6ep) → [fold0 6ep vs 8ep 비교] → FULL 멤버(승자 에폭, v6, 프루닝)
# 슬롯B: large8ep 프로브(f0, 8ep) → largev6B(f3-4, 6ep)
cd /home/vasebull/action_decision

chainA() {
  bash sim/babysit_teacher.sh largev6A xlm-roberta-large v6 320 6 2e-5 64 0 0 3
  # 8ep 프로브 npz 대기 (최대 2h)
  for i in $(seq 1 60); do
    [ -f action_decision_maximum/experiments/teacher_large8v6_a1.npz ] && break
    sleep 120
  done
  EP=$(python3 - <<'PY'
import glob
import numpy as np
def f0(pat):
    for p in sorted(glob.glob(f"action_decision_maximum/experiments/{pat}")):
        z = np.load(p, allow_pickle=True)
        if int(z["fold_lo"]) == 0:
            return float(z["scores"][0])
    return None
s6, s8 = f0("teacher_largev6A_a*.npz"), f0("teacher_large8v6_a*.npz")
print(8 if (s8 or 0) > (s6 or 0) + 0.002 else 6)
PY
)
  echo "### FULL 멤버 에폭 결정: ${EP}ep ###"
  bash sim/babysit_full.sh largefullv6 xlm-roberta-large $EP 2e-5 64 1 v6
}

chainB() {
  bash sim/babysit_teacher.sh large8v6 xlm-roberta-large v6 320 8 2e-5 64 0 0 1
  bash sim/babysit_teacher.sh largev6B xlm-roberta-large v6 320 6 2e-5 64 0 3 5
}

chainA > /tmp/fleet_chainA.log 2>&1 &
chainB > /tmp/fleet_chainB.log 2>&1 &
wait
echo "### FLEET v6 COMPLETE ###"
ls -la action_decision_maximum/experiments/teacher_largev6* action_decision_maximum/experiments/teacher_large8v6* action_decision_maximum/experiments/member_largefullv6.zip 2>/dev/null
