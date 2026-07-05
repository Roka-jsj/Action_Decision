#!/bin/bash
# docker_verify.sh <submit.zip> [GPU번호=0] [이미지=action-clf:eval]
# 평가서버 재현 검증 — 학습 컨테이너 안에서 실행하면 형제 검증 컨테이너를 자동 발사·회수한다.
#   전제(학습 컨테이너): docker CLI + /var/run/docker.sock 마운트 + AD_HOST_REPO=호스트측 저장소 절대경로
#   호스트에서 직접 실행할 땐 AD_HOST_REPO 불필요(자동으로 저장소 경로 사용).
# 재현: --network none / 12g RAM / 3 cpu / GPU 1대.
# 게이트: ①환산 서버시간 ≤540s(sim/calib.json ratio) ②피크 VRAM ≤14GB ③행수 ④holdout 채점.
set -e
ZIP=$(realpath "$1"); GPU=${2:-0}; IMG=${3:-action-clf:eval}
R=$(cd "$(dirname "$0")/.." && pwd)
HOST_R="${AD_HOST_REPO:-$R}"          # docker -v 는 호스트 경로 기준 (DooD 핵심)
NAME=$(basename "$ZIP" .zip)
WORK="$R/verify_work/$NAME"
rm -rf "$WORK"; mkdir -p "$WORK"

# 1) holdout 30k 테스트셋 (train 컨테이너/호스트 파이썬 — data/, splits/ 필요)
python3 "$R/sim/make_holdout_test.py" 30000 "$WORK/data"
unzip -q "$ZIP" -d "$WORK/pkg"
mv "$WORK/data" "$WORK/pkg/data"

# 2) VRAM 피크 폴링
PEAK_F="$WORK/peak_vram"; echo 0 > "$PEAK_F"
( while true; do
    V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | head -1)
    P=$(cat "$PEAK_F"); [ -n "$V" ] && [ "$V" -gt "$P" ] 2>/dev/null && echo "$V" > "$PEAK_F"
    sleep 2
  done ) & POLL=$!

# 3) 평가서버 재현 (형제 컨테이너)
T0=$(date +%s)
docker run --rm --gpus "\"device=$GPU\"" --network none \
  --memory 12g --memory-swap 12g --cpus 3 \
  -v "$HOST_R/verify_work/$NAME/pkg":/workspace -w /workspace "$IMG" \
  python script.py
T1=$(date +%s); kill $POLL 2>/dev/null || true
ELAPSED=$((T1 - T0)); PEAK=$(cat "$PEAK_F" 2>/dev/null || echo 0)

# 4) 채점 + 게이트
python3 "$R/sim/score_holdout.py" "$WORK/pkg/output/submission.csv"
ROWS=$(tail -n +2 "$WORK/pkg/output/submission.csv" | wc -l)
echo "----------------------------------------"
echo "[$NAME] A6000 실측: ${ELAPSED}s | 피크 VRAM: ${PEAK}MB | 행수: $ROWS"
if [ -f "$R/sim/calib.json" ]; then
  RATIO=$(python3 -c "import json; print(json.load(open('$R/sim/calib.json'))['ratio'])")
  EST=$(python3 -c "print(round($ELAPSED * $RATIO))")
  echo "환산 서버시간: ~${EST}s (ratio=$RATIO) → $([ "$EST" -le 540 ] && echo '시간 게이트 PASS' || echo '시간 게이트 FAIL(>540s)')"
else
  echo "⚠ sim/calib.json 없음 — 앵커 2개로 캘리브레이션 필요:"
  echo "  bash sim/docker_verify.sh packages/submit_largeonly.zip $GPU   # 서버실측 257s"
  echo "  bash sim/docker_verify.sh packages/submit_tri_cond.zip $GPU   # 서버실측 427s"
  echo "  ratio = mean(257/측정1, 427/측정2) → echo '{\"ratio\": X.XX}' > sim/calib.json"
fi
[ "$PEAK" -le 14000 ] && echo "VRAM 게이트: PASS (≤14GB)" || echo "VRAM 게이트: FAIL (T4 16GB 위험)"