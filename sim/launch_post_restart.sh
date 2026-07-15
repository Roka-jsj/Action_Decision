#!/bin/bash
# docker restart 직후 원샷: CUDA 프로브 → GPU0 #5(s999 표준 8ep 고속) + GPU1 5ep(s314) 동시 발사
# 사용: bash work/launch_post_restart.sh
set -euo pipefail
cd /root/Action_Decision

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA 아직 죽어있음 — restart 미완?'; import sys; print('CUDA OK, dev:', torch.cuda.device_count())"

# 두 GPU 모두 비어있는지 확인 (학습 잔존 시 중단)
for p in $(pgrep -f "train_full_cli" 2>/dev/null || true); do echo "FATAL: 기존 학습 잔존 PID=$p — 발사 중단"; exit 1; done

COMMON="AD_WORK=/root/Action_Decision/work AD_MODEL=xlm-roberta-large AD_VERSION=v6 AD_MAXLEN=320 AD_LR=2e-5 AD_BATCH=64 AD_LLRD=1 AD_FGM=1 AD_PRUNE=1 AD_GEN_RESCUE=1 AD_MHT=12 AD_SWA_K=2 AD_SOFTF1=1 AD_SOFTF1_W=0.5"

# GPU0: #5 s999 표준 8ep, 비-gradckpt 고속레인 (빈 GPU0 32.6GB 적합, 26.3분/ep)
env $COMMON CUDA_VISIBLE_DEVICES=0 AD_EPOCHS=8 AD_SEED=999 AD_HEADSEED=999 AD_TAG=m1d64_5 \
  nohup python3 action_decision_maximum/src/train_full_cli.py > work/m1d64_5.log 2>&1 &
P5=$!
echo "[$(date '+%m-%d %H:%M:%S')] LAUNCH draw#5 GPU0 PID=$P5 s999 8ep 표준" | tee -a work/queue_draws.log

# GPU1: 5ep 단축티켓 s314, gradckpt (73분/ep × 5 ≈ 6.1h)
env $COMMON CUDA_VISIBLE_DEVICES=1 AD_EPOCHS=5 AD_SEED=314 AD_HEADSEED=314 AD_GRADCKPT=1 AD_TAG=m1d64_s314e5 \
  nohup python3 action_decision_maximum/src/train_full_cli.py > work/m1d64_s314e5.log 2>&1 &
P1E=$!
echo "[$(date '+%m-%d %H:%M:%S')] LAUNCH gpu1-5ep GPU1 PID=$P1E s314 5ep gradckpt" | tee -a work/queue_draws.log

sleep 90
for P in $P5 $P1E; do
  ps -p $P >/dev/null || { echo "FATAL: PID $P 90초 내 사망 — 로그 확인"; exit 1; }
done
grep -l "Traceback" work/m1d64_5.log work/m1d64_s314e5.log 2>/dev/null && { echo "FATAL: Traceback 발생"; exit 1; }
echo "[OK] 양 GPU 발사 확인 (#5=$P5, 5ep=$P1E)"
