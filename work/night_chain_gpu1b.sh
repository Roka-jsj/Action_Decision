#!/bin/bash
# R72/R73 GPU1 새벽 체인(수정판) — mdeb_s2025 완주 대기 → m1wsT3b(lr1e-5 변형) FT → 양자화 → 게이트(+0.0035).
# R73 수리 3종: AD_PRUNE=0(이중프루닝 방지) · FGMFLAG 체인d 상속(+타임아웃 폴백) · 게이트 클론 sim/ 경로(ROOT 버그).
cd /root/Action_Decision
L=work/night_chain_gpu1.log
while [ ! -e work/DONE_mdebfull_s2025 ]; do sleep 60; done
echo "$(date +%F_%H:%M:%S) 체인1b(수정판) 시작(mdeb_s2025 완주)" >> $L
# FGM 플래그: 체인d의 v1 런치라인에서 상속(판독-작동 일치). +75분 타임아웃 → 0 폴백.
FGMFLAG=""; DL=$(( $(date +%s)+4500 ))
while [ -z "$FGMFLAG" ] && [ $(date +%s) -lt $DL ]; do
  FGMFLAG=$(grep -m1 'launch m1wsT3 ' work/night_chain_gpu0b.log 2>/dev/null | grep -oE 'fgm=[01]' | cut -d= -f2)
  [ -z "$FGMFLAG" ] && sleep 60
done
[ -z "$FGMFLAG" ] && { FGMFLAG=0; echo "$(date +%F_%H:%M:%S) FGM대기 타임아웃(+75분)→FGM=0 폴백" >> $L; }
env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/root/Action_Decision AD_WORK=/root/Action_Decision/work \
 AD_MODEL=xlm-roberta-large AD_INIT_FROM=/root/Action_Decision/work/warmstart_m1 \
 AD_VERSION=v6 AD_MAXLEN=320 AD_EPOCHS=2 AD_LR=1e-5 AD_BATCH=64 AD_LLRD=1 AD_FGM=$FGMFLAG \
 AD_SEED=777 AD_PRUNE=0 AD_GEN_RESCUE=1 AD_MHT=12 \
 AD_EXCLUDE_ROWS=/root/Action_Decision/work/exclude_rows_5k.npy AD_TAG=m1wsT3b \
 python3 action_decision_maximum/src/train_full_cli.py > work/m1wsT3b.log 2>&1 &
P=$!; echo "$(date +%F_%H:%M:%S) launch m1wsT3b pid=$P fgm=$FGMFLAG" >> $L
wait $P; echo "$(date +%F_%H:%M:%S) done m1wsT3b rc=$?" >> $L
grep -m1 "\[exclude\]" work/m1wsT3b.log >> $L 2>&1
env PYTHONPATH=/root/Action_Decision python3 sim/quantize_member.py work/member_m1wsT3b.zip > work/quant_m1wsT3b.log 2>&1
mkdir -p work/member_m1wsT3b_q8dir && cd work/member_m1wsT3b_q8dir && python3 -c "
import zipfile; zipfile.ZipFile('/root/Action_Decision/work/member_m1wsT3b_q8.zip').extractall('.')" && cd /root/Action_Decision
# v2 게이트 클론: sim/ 안에 생성(ROOT 계산 보존) + 임계 +0.0035 치환(v2 선택페널티 바)
sed 's|member_m1wsT3_q8dir|member_m1wsT3b_q8dir|; s|gate_m1wsT3_5k|gate_m1wsT3b_5k|; s|= 0.0025,|= 0.0035,|' sim/gate_m1wsT3.py > sim/gate_bx.py
env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/root/Action_Decision python3 sim/gate_bx.py > work/gate_m1wsT3b.json 2> work/gate_m1wsT3b.err
echo "$(date +%F_%H:%M:%S) v2 게이트 $(cat work/gate_m1wsT3b.json 2>/dev/null) — 체인1b 완료" >> $L
