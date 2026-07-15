#!/bin/bash
# #6 막차 원샷: 에폭 자동룰(발사시각+E×페이스+25분 ≤ 09:05 되는 최대 E∈{8..5}) + 시드트리
# 사용: bash work/launch_hash6.sh [pace_min=27] [seed=555] [reinit=0]
# 시드트리(사전등록): 기본 표준 s555 / 밤 전판독 최고 <0.7908이면 reinit=1 s7 문샷
set -euo pipefail
cd /root/Action_Decision
PACE="${1:-27}"; SEED="${2:-555}"; REINIT="${3:-0}"

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA 죽음'"
pgrep -f "AD_TAG=m1d64_6" >/dev/null 2>&1 && { echo "FATAL: #6 이미 실행중"; exit 1; }

NOW=$(date +%s); CUT=$(date -d "09:05" +%s); [ "$CUT" -gt "$NOW" ] || CUT=$(date -d "tomorrow 09:05" +%s)
AVAIL_MIN=$(( (CUT - NOW) / 60 - 25 ))
# codex 2R 최종재정(07-15): 잭팟성분은 8ep 표준레시피에서만 실증 — 8ep 미확보 시 발사취소(에폭강등 금지)
E=8
[ $(( E * PACE )) -le "$AVAIL_MIN" ] || { echo "FATAL: 8ep가 09:05 내 불가(가용 ${AVAIL_MIN}분, 필요 $((E*PACE))분) — 사전등록대로 발사취소"; exit 1; }

EXTRA=""
TAGSUF=""
if [ "$REINIT" = "1" ]; then EXTRA="AD_REINIT_N=2"; TAGSUF="r2"; fi
TAG="m1d64_6${TAGSUF}"

echo "[#6 에폭룰] pace=${PACE}분/ep, 가용 ${AVAIL_MIN}분 → E=${E}ep, seed=${SEED}, reinit=${REINIT}"
env AD_WORK=/root/Action_Decision/work AD_MODEL=xlm-roberta-large AD_VERSION=v6 AD_MAXLEN=320 \
  AD_LR=2e-5 AD_BATCH=64 AD_LLRD=1 AD_FGM=1 AD_PRUNE=1 AD_GEN_RESCUE=1 AD_MHT=12 AD_SWA_K=2 \
  AD_SOFTF1=1 AD_SOFTF1_W=0.5 CUDA_VISIBLE_DEVICES=0 AD_EPOCHS=$E AD_SEED=$SEED AD_HEADSEED=$SEED \
  $EXTRA AD_TAG=$TAG \
  nohup python3 action_decision_maximum/src/train_full_cli.py > work/${TAG}.log 2>&1 &
P6=$!
echo "[$(date '+%m-%d %H:%M:%S')] LAUNCH draw#6 GPU0 PID=$P6 s${SEED} ${E}ep reinit=${REINIT}" | tee -a work/queue_draws.log
sleep 90
ps -p $P6 >/dev/null || { echo "FATAL: #6 90초 내 사망"; exit 1; }
echo "[OK] #6 발사 확인 PID=$P6 (${E}ep, 완성예상 $(date -d "+$((E*PACE+20)) minutes" '+%H:%M'))"
