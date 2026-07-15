#!/bin/bash
# R71b 새벽 체인(d) — 체인c 완료 대기(파일마커) → FGM 판독 → m1wsT3 FT → 양자화 → 게이트.
cd /root/Action_Decision
L=work/night_chain_gpu0b.log
while ! grep -q "GPU0 체인 완료" $L 2>/dev/null; do sleep 60; done
echo "$(date +%F_%H:%M:%S) 체인d 시작" >> $L
# FGM 포함규칙 (FGM런이 존재할 때만)
FGMFLAG=0
if [ -d work/member_m1h8full_fgm1_s777 ]; then
  env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/Action_Decision python3 sim/eval_fgm_paired.py > work/fgm_paired.json 2>/dev/null
  FGMFLAG=$(python3 -c "import json;print(1 if json.load(open('work/fgm_paired.json'))['include_fgm'] else 0)" 2>/dev/null || echo 0)
  echo "$(date +%F_%H:%M:%S) FGM판독 $(cat work/fgm_paired.json 2>/dev/null) → FT FGM=$FGMFLAG" >> $L
fi
# m1wsT3 FT (2ep lr5e-6, 게이트행 제외, rescue+mht12 학습·서빙은 h8 예정)
env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/Action_Decision AD_WORK=/root/Action_Decision/work \
 AD_MODEL=xlm-roberta-large AD_INIT_FROM=/root/Action_Decision/work/warmstart_m1 \
 AD_VERSION=v6 AD_MAXLEN=320 AD_EPOCHS=2 AD_LR=5e-6 AD_BATCH=64 AD_LLRD=1 AD_FGM=$FGMFLAG \
 AD_SEED=777 AD_PRUNE=0 AD_GEN_RESCUE=1 AD_MHT=12 \
 AD_EXCLUDE_ROWS=/root/Action_Decision/work/exclude_rows_5k.npy AD_TAG=m1wsT3 \
 python3 action_decision_maximum/src/train_full_cli.py > work/m1wsT3.log 2>&1 &
P=$!; echo "$(date +%F_%H:%M:%S) launch m1wsT3 pid=$P fgm=$FGMFLAG" >> $L
wait $P; echo "$(date +%F_%H:%M:%S) done m1wsT3 rc=$?" >> $L
grep -m1 "\[exclude\]" work/m1wsT3.log >> $L 2>&1   # codex 조건①: 5k 미유입 로그 증명
# 양자화(codex 조건②: 게이트는 양자화본으로)
env PYTHONPATH=/root/Action_Decision python3 sim/quantize_member.py work/member_m1wsT3.zip > work/quant_m1wsT3.log 2>&1
mkdir -p work/member_m1wsT3_q8dir && cd work/member_m1wsT3_q8dir && python3 -c "
import zipfile; zipfile.ZipFile('/root/Action_Decision/work/member_m1wsT3_q8.zip').extractall('.')" && cd /root/Action_Decision
echo "$(date +%F_%H:%M:%S) 양자화·추출 완료" >> $L
# 5k 게이트 (h8 서빙, 양자화본)
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/Action_Decision python3 sim/gate_m1wsT3.py > work/gate_m1wsT3.json 2> work/gate_m1wsT3.err
echo "$(date +%F_%H:%M:%S) 게이트 $(cat work/gate_m1wsT3.json 2>/dev/null) — 체인d 완료(조립은 06:30 검토 후)" >> $L
