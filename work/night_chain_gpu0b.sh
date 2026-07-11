#!/bin/bash
# R70c 재편성 체인(GPU0) — 3자 서명(codex#11 + 레드팀 조건부 + 운영자).
# s2025(PID 374303) 완주 대기 → run1: m1h8full_swa2_s777(+무료 raw쌍둥이=동일시드 재현노이즈 ε̂)
# → 5k 판정 → 분기 run2: CONFIRM→m1t3full_swa2_s777 / GRAY·DEAD→m1h8full_fgm1_s777(FGM 단일차분).
# 사전등록: CONFIRM = δ_swa ≥ -0.005 ∧ SWA_paired ≥ +0.010 / DEAD = δ_swa ≤ -0.012 / 그외 GRAY.
cd /root/Action_Decision
L=work/night_chain_gpu0b.log

# 0) s2025 완주 대기 (PID 374303 — 기록된 PID로만, 신교리)
while kill -0 374303 2>/dev/null; do sleep 30; done
[ -e work/DONE_m1h8full_s2025 ] || { echo "$(date +%F_%H:%M:%S) s2025 DONE 마커 없음 — 중단" >> $L; exit 1; }
echo "$(date +%F_%H:%M:%S) s2025 완주 확인" >> $L

# 1) run1: SWA2 + s777 (정확 레시피 재현 시도)
env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/Action_Decision AD_WORK=/root/Action_Decision/work \
 AD_MODEL=xlm-roberta-large AD_VERSION=v6 AD_MAXLEN=320 AD_EPOCHS=8 AD_LR=2e-5 AD_BATCH=64 \
 AD_LLRD=1 AD_FGM=0 AD_SEED=777 AD_PRUNE=1 AD_GEN_RESCUE=0 AD_MHT=8 AD_SWA_K=2 AD_TAG=m1h8full_swa2_s777 \
 python3 action_decision_maximum/src/train_full_cli.py > work/m1h8full_swa2_s777.log 2>&1 &
P=$!; echo "$(date +%F_%H:%M:%S) launch m1h8full_swa2_s777 pid=$P start=$(awk '{print $22}' /proc/$P/stat 2>/dev/null)" >> $L
wait $P; RC=$?
echo "$(date +%F_%H:%M:%S) done m1h8full_swa2_s777 rc=$RC" >> $L
# 레드팀 로그 assert: [swa] 스냅샷 2회 + 평균적용
grep -c "\[swa\] snapshot" work/m1h8full_swa2_s777.log >> $L 2>&1
grep -m1 "\[swa\].*평균" work/m1h8full_swa2_s777.log >> $L 2>&1

# 2) 5k 판정 (GPU0 해방 상태)
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/Action_Decision python3 sim/eval_swa_branch.py > work/swa_branch.json 2> work/swa_branch.err
BR=$(python3 -c "import json;print(json.load(open('work/swa_branch.json'))['branch'])" 2>/dev/null || echo ERR)
echo "$(date +%F_%H:%M:%S) 판정 branch=$BR" >> $L
cat work/swa_branch.json >> $L 2>/dev/null

# 3) 분기 run2
if [ "$BR" = "CONFIRM" ]; then
  TAG=m1t3full_swa2_s777; EXTRA="AD_GEN_RESCUE=1 AD_MHT=12 AD_SWA_K=2"
else
  TAG=m1h8full_fgm1_s777; EXTRA="AD_GEN_RESCUE=0 AD_MHT=8 AD_SWA_K=0 AD_FGM_OVERRIDE=1"
fi
if [ "$TAG" = "m1t3full_swa2_s777" ]; then
  env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/Action_Decision AD_WORK=/root/Action_Decision/work \
   AD_MODEL=xlm-roberta-large AD_VERSION=v6 AD_MAXLEN=320 AD_EPOCHS=8 AD_LR=2e-5 AD_BATCH=64 \
   AD_LLRD=1 AD_FGM=0 AD_SEED=777 AD_PRUNE=1 AD_GEN_RESCUE=1 AD_MHT=12 AD_SWA_K=2 AD_TAG=$TAG \
   python3 action_decision_maximum/src/train_full_cli.py > work/$TAG.log 2>&1 &
else
  env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/Action_Decision AD_WORK=/root/Action_Decision/work \
   AD_MODEL=xlm-roberta-large AD_VERSION=v6 AD_MAXLEN=320 AD_EPOCHS=8 AD_LR=2e-5 AD_BATCH=64 \
   AD_LLRD=1 AD_FGM=1 AD_SEED=777 AD_PRUNE=1 AD_GEN_RESCUE=0 AD_MHT=8 AD_TAG=$TAG \
   python3 action_decision_maximum/src/train_full_cli.py > work/$TAG.log 2>&1 &
fi
P=$!; echo "$(date +%F_%H:%M:%S) launch $TAG pid=$P start=$(awk '{print $22}' /proc/$P/stat 2>/dev/null)" >> $L
wait $P
echo "$(date +%F_%H:%M:%S) done $TAG rc=$? — GPU0 체인 완료" >> $L
