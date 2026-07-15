#!/bin/bash
# infoxlm8(GPU1) 완주 대기(PID+마커, 좀비회피) → AWP+EMA+R-Drop(Variant B) m1 게이트후보 학습.
cd /root/Action_Decision
L=work/m1r74_chain.log
# infoxlm PID 확보
IP=$(pgrep -f "train_full_cli" | while read p; do t=$(tr '\0' '\n' </proc/$p/environ 2>/dev/null|grep '^AD_TAG='|cut -d= -f2); [ "$t" = "infoxlm8" ] && echo $p; done | head -1)
echo "$(date +%F_%H:%M:%S) infoxlm PID=$IP 완주 대기" >> $L
# 좀비 회피: DONE 마커 또는 프로세스 stat=Z/부재까지
while true; do
  [ -e work/DONE_infoxlm8 ] && break
  st=$(ps -o stat= -p "$IP" 2>/dev/null | tr -d ' ')
  { [ -z "$st" ] || [ "${st:0:1}" = "Z" ]; } && break
  sleep 60
done
echo "$(date +%F_%H:%M:%S) infoxlm 완주 → m1r74 Variant B 발진" >> $L
env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/root/Action_Decision AD_WORK=/root/Action_Decision/work \
 AD_MODEL=xlm-roberta-large AD_VERSION=v6 AD_MAXLEN=320 AD_EPOCHS=8 AD_LR=2e-5 AD_BATCH=64 \
 AD_LLRD=1 AD_PRUNE=1 AD_GEN_RESCUE=0 AD_MHT=8 AD_SEED=777 \
 AD_FGM=1 AD_AWP=1 AD_AWP_LR=1.0 AD_AWP_EPS=0.01 AD_AWP_START_EP=1 AD_EMA=1 AD_EMA_DECAY=0.999 \
 AD_RDROP=1 AD_RDROP_ALPHA=0.5 AD_GRADCKPT=1 \
 AD_EXCLUDE_ROWS=/root/Action_Decision/work/exclude_rows_5k.npy AD_TAG=m1r74gate \
 python3 action_decision_maximum/src/train_full_cli.py > work/m1r74gate.log 2>&1 &
P=$!; echo "$(date +%F_%H:%M:%S) launch m1r74gate pid=$P" >> $L
wait $P; echo "$(date +%F_%H:%M:%S) done m1r74gate rc=$?" >> $L
