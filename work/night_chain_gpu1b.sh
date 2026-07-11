#!/bin/bash
# R72 GPU1 새벽 체인 — mdeb_s2025 완주 대기(파일마커) → m1wsT3b(LR 1e-5 변형) FT → 양자화 → 게이트.
cd /root/Action_Decision
L=work/night_chain_gpu1.log
while [ ! -e work/DONE_mdebfull_s2025 ]; do sleep 60; done
echo "$(date +%F_%H:%M:%S) 체인1b 시작(mdeb_s2025 완주)" >> $L
FGMFLAG=0
[ -e work/fgm_paired.json ] && FGMFLAG=$(python3 -c "import json;print(1 if json.load(open('work/fgm_paired.json'))['include_fgm'] else 0)" 2>/dev/null || echo 0)
env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/root/Action_Decision AD_WORK=/root/Action_Decision/work \
 AD_MODEL=xlm-roberta-large AD_INIT_FROM=/root/Action_Decision/work/warmstart_m1 \
 AD_VERSION=v6 AD_MAXLEN=320 AD_EPOCHS=2 AD_LR=1e-5 AD_BATCH=64 AD_LLRD=1 AD_FGM=$FGMFLAG \
 AD_SEED=777 AD_PRUNE=1 AD_GEN_RESCUE=1 AD_MHT=12 \
 AD_EXCLUDE_ROWS=/root/Action_Decision/work/exclude_rows_5k.npy AD_TAG=m1wsT3b \
 python3 action_decision_maximum/src/train_full_cli.py > work/m1wsT3b.log 2>&1 &
P=$!; echo "$(date +%F_%H:%M:%S) launch m1wsT3b pid=$P fgm=$FGMFLAG" >> $L
wait $P; echo "$(date +%F_%H:%M:%S) done m1wsT3b rc=$?" >> $L
grep -m1 "\[exclude\]" work/m1wsT3b.log >> $L 2>&1
env PYTHONPATH=/root/Action_Decision python3 sim/quantize_member.py work/member_m1wsT3b.zip > work/quant_m1wsT3b.log 2>&1
mkdir -p work/member_m1wsT3b_q8dir && cd work/member_m1wsT3b_q8dir && python3 -c "
import zipfile; zipfile.ZipFile('/root/Action_Decision/work/member_m1wsT3b_q8.zip').extractall('.')" && cd /root/Action_Decision
sed 's|member_m1wsT3_q8dir|member_m1wsT3b_q8dir|; s|gate_m1wsT3_5k|gate_m1wsT3b_5k|' sim/gate_m1wsT3.py > /tmp/gate_b.py
env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/root/Action_Decision python3 /tmp/gate_b.py > work/gate_m1wsT3b.json 2> work/gate_m1wsT3b.err
echo "$(date +%F_%H:%M:%S) v2 게이트 $(cat work/gate_m1wsT3b.json 2>/dev/null) — 체인1b 완료" >> $L
