#!/bin/bash
# docker_verify.sh <submit.zip> [GPU번호=0] [이미지=action-clf:eval] — 평가서버 재현 검증 (서버 호스트에서 실행).
# 재현: --network none / 12g RAM / 3 cpu / GPU 1대. 게이트: ①환산 서버시간 ≤540s ②피크 VRAM ≤14GB ③행수/클래스 ④holdout 채점.
# 시간 캘리브레이션: sim/calib.json {"ratio": 서버시간/A6000시간} — 앵커(largeonly=257s, tri_cond=427s)로 1회 산출.
set -e
ZIP=$(realpath "$1"); GPU=${2:-0}; IMG=${3:-action-clf:eval}
R=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d /tmp/adverify.XXXX)
trap 'rm -rf "$WORK"' EXIT

# 1) holdout 30k 테스트셋 생성 (호스트 python; 없으면 학습 컨테이너에서 생성해 경로 지정)
python3 "$R/sim/make_holdout_test.py" 30000 "$WORK/data"
unzip -q "$ZIP" -d "$WORK/pkg"
cp -r "$WORK/data" "$WORK/pkg/data"

# 2) VRAM 피크 폴링 (호스트에서)
PEAK=0
( while true; do
    V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | head -1)
    [ -n "$V" ] && [ "$V" -gt "$PEAK" ] 2>/dev/null && PEAK=$V && echo "$V" > "$WORK/peak_vram"
    sleep 2
  done ) & POLL=$!

# 3) 평가서버 재현 실행
T0=$(date +%s)
docker run --rm --gpus "\"device=$GPU\"" --network none \
  --memory 12g --memory-swap 12g --cpus 3 \
  -v "$WORK/pkg":/workspace -w /workspace "$IMG" \
  python script.py
T1=$(date +%s); kill $POLL 2>/dev/null || true
ELAPSED=$((T1 - T0)); PEAK=$(cat "$WORK/peak_vram" 2>/dev/null || echo 0)

# 4) 채점 + 게이트
python3 "$R/sim/score_holdout.py" "$WORK/pkg/output/submission.csv"
ROWS=$(tail -n +2 "$WORK/pkg/output/submission.csv" | wc -l)
echo "----------------------------------------"
echo "A6000 실측: ${ELAPSED}s | 피크 VRAM: ${PEAK}MB | 행수: $ROWS"
if [ -f "$R/sim/calib.json" ]; then
  RATIO=$(python3 -c "import json; print(json.load(open('$R/sim/calib.json'))['ratio'])")
  EST=$(python3 -c "print(round($ELAPSED * $RATIO))")
  echo "환산 서버시간: ~${EST}s (ratio=$RATIO)"
  [ "$EST" -le 540 ] && echo "시간 게이트: PASS" || echo "시간 게이트: FAIL (>540s)"
else
  echo "⚠ sim/calib.json 없음 — 앵커(largeonly 257s / tri_cond 427s)를 이 스크립트로 돌려"
  echo "  ratio = mean(서버시간/A6000시간) 계산 후 저장: {\"ratio\": X.XX}"
fi
[ "$PEAK" -le 14000 ] && echo "VRAM 게이트: PASS (≤14GB)" || echo "VRAM 게이트: FAIL (T4 16GB 위험)"