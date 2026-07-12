#!/bin/bash
# ============================================================================
# R74 — "배포 m1 을 이겨라": 신규 학습기법(FGM+AWP+EMA) m1 재학습. 동결 레시피.
# ----------------------------------------------------------------------------
# 발사 전제(핸드북 수칙): (1) GPU0/GPU1 중 하나가 비어야 함 — 진행중 학습(mdeb14/
#   infoxlm8) 절대 방해 금지. (2) 3자(운영자+codex+레드팀) 서명 후에만 발사.
#   이 스크립트는 준비물이며, 실행은 운영자가 직접 한다(자동 발사 금지).
# 사용법: 비는 GPU 인덱스를 인자로. 예) bash work/launch_m1_r74.sh 1
# ----------------------------------------------------------------------------
# 왜 이 콤보인가(실측 근거):
#   배포 m1 solo(5k) = 0.82584. 우리 plain 재현 = 0.80752(-0.01832).
#   FGM 단독 = 0.81773(fgm_paired +0.01021, 배포대비 -0.00811). AWP(가중치 교란,
#   +0.002~0.005)+EMA(수렴궤적, +0.001~0.003)로 잔여 -0.00811 을 닫는 것이 목표.
# 단계: (1) 게이트후보 65k(5k제외) 학습 → 양자화 → 사전등록 게이트
#       (2) 게이트 GO 이면 최종멤버 FULL 70k 동일레시피 학습(앙상블 투입용)
# ============================================================================
set -u
GPU="${1:?사용법: bash work/launch_m1_r74.sh <free_gpu_index>}"
cd /root/Action_Decision
L=work/launch_m1_r74.log
: > $L
echo "$(date +%F_%H:%M:%S) START gpu=$GPU" >> $L

# ── 동결 레시피(배포 m1h8full + FGM + AWP + EMA). R-Drop 은 변형 B(하단 주석). ──
COMMON="HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=/root/Action_Decision \
 AD_WORK=/root/Action_Decision/work AD_MODEL=xlm-roberta-large AD_VERSION=v6 AD_MAXLEN=320 \
 AD_EPOCHS=8 AD_LR=2e-5 AD_BATCH=64 AD_LLRD=1 AD_PRUNE=1 AD_GEN_RESCUE=0 AD_MHT=8 AD_SEED=777 \
 AD_FGM=1 AD_AWP=1 AD_AWP_LR=1.0 AD_AWP_EPS=0.01 AD_AWP_START_EP=1 AD_EMA=1 AD_EMA_DECAY=0.999"
# OOM 시(48GB A6000 에서 large b64 + EMA shadow ~2.2GB): 위 COMMON 에 AD_GRADCKPT=1 추가.

# ── (1) 게이트 후보: 65k(5k 제외) ── ~3-3.5h
env $COMMON AD_EXCLUDE_ROWS=/root/Action_Decision/work/exclude_rows_5k.npy AD_TAG=m1r74gate \
  python3 action_decision_maximum/src/train_full_cli.py > work/m1r74gate.log 2>&1
P=$?; echo "$(date +%F_%H:%M:%S) train m1r74gate rc=$P" >> $L
grep -m1 "\[exclude\]" work/m1r74gate.log >> $L 2>&1   # 5k 미유입 증명(codex 조건)
grep -m1 "\[ema\]" work/m1r74gate.log >> $L 2>&1
[ -e work/DONE_m1r74gate ] || { echo "게이트후보 미완주 — 중단" >> $L; exit 1; }

# 양자화(배포 int8 group-64) → gate dir
env PYTHONPATH=/root/Action_Decision python3 sim/quantize_member.py work/member_m1r74gate.zip \
  > work/quant_m1r74gate.log 2>&1
mkdir -p work/member_m1r74gate_q8dir
( cd work/member_m1r74gate_q8dir && \
  python3 -c "import zipfile; zipfile.ZipFile('/root/Action_Decision/work/member_m1r74gate_q8.zip').extractall('.')" )

# ── (2) 사전등록 게이트: solo≥0.82584 ∧ casc≥+0.0025 ∧ CI>0 ── ~5분
env CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=/root/Action_Decision \
  AD_GATE_MEMBER=/root/Action_Decision/work/member_m1r74gate_q8dir \
  python3 sim/gate_m1_r74.py > work/gate_m1_r74.json 2> work/gate_m1_r74.err
GV=$(python3 -c "import json;print(json.load(open('work/gate_m1_r74.json'))['verdict'])" 2>/dev/null || echo ERR)
echo "$(date +%F_%H:%M:%S) GATE=$GV $(cat work/gate_m1_r74.json 2>/dev/null)" >> $L

# ── (3) GO 이면 최종멤버 FULL 70k(동일레시피, 제외 없음) → 앙상블 투입 ── ~3-3.5h
if [ "$GV" = "GO" ]; then
  env $COMMON AD_TAG=m1r74full \
    python3 action_decision_maximum/src/train_full_cli.py > work/m1r74full.log 2>&1
  echo "$(date +%F_%H:%M:%S) train m1r74full rc=$? — 재조립 준비 완료" >> $L
else
  echo "$(date +%F_%H:%M:%S) 게이트 NO-GO — FULL 학습 보류, 부검 필요" >> $L
fi
echo "$(date +%F_%H:%M:%S) DONE" >> $L

# ============================================================================
# 변형 B (최대 상한, 시간 여유 시): 위 COMMON 에 다음을 추가해 R-Drop 스택.
#   AD_RDROP=1 AD_RDROP_ALPHA=0.5
# 비용: forward 4x/step(rdrop 2 + fgm 1 + awp 1) → ~5-6h. EV: +0.002~0.004 추가상한.
# ============================================================================
