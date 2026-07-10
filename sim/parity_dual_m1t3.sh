#!/bin/bash
# m1t3 parity 이중대조 (codex R63 층화 게이트) — 같은 seed/N 홀드아웃을 th85(앵커)와 m1t3에 실행.
#  게이트A(m2/m3 불변)는 assemble_m1t3.sh 의 inode 동일성 + h8 직렬화 SHA 검증으로 구조 증명 완료.
#  게이트B: m1 h12 텍스트 변경행 비율 in [37%, 45%] (30k 실측 40.8% 의 95% 밴드 확장).
#  게이트C: F1(m1t3) - F1(th85) >= -0.005 (비대칭 — 큰 양수는 정상, 큰 음수만 적색).
#  절대치는 FULL 홀드아웃 암기 인플레라 판정에 쓰지 않음(M2 교리) — 상대차만.
set -euo pipefail
R=/root/Action_Decision; cd $R
N=${1:-5000}

echo "[dual] 게이트B: h12 직렬화 변경행 비율 (CPU)"
python3 - <<PY
import sys, numpy as np
sys.path.insert(0, "$R/packages/submit_m1t3/model")
import ad_lib as dep
sys.path.insert(0, "$R")
from common.io_utils import load_train
from common.cv import make_splits
samples, y, ids = load_train()
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, np.array(y), groups)
rng = np.random.RandomState(0)
sub = rng.choice(sp["holdout_idx"], size=min($N, len(sp["holdout_idx"])), replace=False)
n12 = sum(1 for i in sub if dep.serialize(samples[i], "v6", 12) != dep.serialize(samples[i], "v6", 8))
frac = n12 / len(sub)
print(f"[dual-B] h12 변경행 {n12}/{len(sub)} = {frac:.3f}")
assert 0.37 <= frac <= 0.45, f"게이트B FAIL: {frac:.3f} not in [0.37,0.45] — mht 배선 의심"
print("[dual-B] PASS")
PY

echo "[dual] 게이트C: 이중 parity (GPU, 각 ~수분)"
A=$(python3 sim/parity_check.py packages/submit_th85 $N 2>&1 | tee /dev/stderr | grep -oP 'macro-F1=\K[0-9.]+')
B=$(python3 sim/parity_check.py packages/submit_m1t3 $N 2>&1 | tee /dev/stderr | grep -oP 'macro-F1=\K[0-9.]+')
python3 - <<PY
a, b = float("$A"), float("$B")
d = b - a
print(f"[dual-C] th85={a:.5f}  m1t3={b:.5f}  Δ={d:+.5f}")
assert d >= -0.005, f"게이트C FAIL: Δ={d:+.5f} < -0.005 — 배선/조립 부검 필요"
print(f"[dual-C] PASS (주의: 절대치·Δ 모두 홀드아웃 암기 포함 — LB 예측치 아님)")
PY
echo "[dual] 전 게이트 PASS"
