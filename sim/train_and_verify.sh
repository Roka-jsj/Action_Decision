#!/bin/bash
# train_and_verify.sh — 학습→패키징→검증컨테이너 자동검증→결과, 전자동 원커맨드.
# 학습 컨테이너(mun-jtrain) 안에서 실행. 모든 경로는 /workspace 절대경로 고정.
#
# 사용: bash /workspace/sim/train_and_verify.sh <TAG> <SEED> [EPOCHS=8] [VERSION=v6] [TEST_CT=mun-jtest]
# 예:   nohup bash /workspace/sim/train_and_verify.sh largev6s3 777 > /workspace/work/auto_largev6s3.log 2>&1 &
#
# 흐름: ①FULL 학습(A6000 ~40분, 이미 member 있으면 스킵) ②bias 포함 단일멤버 패키징
#       ③/share에 자가완결 검증셋 조립 ④docker exec로 검증컨테이너 실행(오프라인 30k)
#       ⑤시간·VRAM 게이트 + holdout 점수 출력 → /workspace/work/verify_<TAG>.txt
set -e
TAG=$1; SEED=$2; EP=${3:-8}; VER=${4:-v6}; TC=${5:-mun-test}
[ -n "$TAG" ] && [ -n "$SEED" ] || { echo "usage: $0 <TAG> <SEED> [EPOCHS] [VERSION] [TEST_CT]"; exit 1; }
R=/workspace
E=$R/action_decision_maximum/experiments
mkdir -p $R/work $E

# ① 학습 (member 이미 있으면 스킵 — 재실행 안전)
if [ ! -f "$E/member_$TAG.zip" ]; then
  echo "[auto] 학습 시작: $TAG (seed=$SEED ep=$EP ver=$VER)"
  AD_WORK=$R/work AD_MODEL=xlm-roberta-large AD_VERSION=$VER AD_MAXLEN=320 \
  AD_EPOCHS=$EP AD_LR=2e-5 AD_BATCH=64 AD_LLRD=1 AD_SEED=$SEED AD_SWA_K=3 \
  AD_PRUNE=1 AD_TAG=$TAG \
  python $R/action_decision_maximum/src/train_full_cli.py 2>&1 | tee $R/work/train_$TAG.log
  cp $R/work/member_$TAG.zip $E/
  echo "[auto] 학습 완료 → $E/member_$TAG.zip"
else
  echo "[auto] member_$TAG.zip 존재 — 학습 스킵"
fi

# ② 패키징 (단일 멤버 + per-class bias)
if [ "$VER" = "v6" ]; then BIAS="$E/teacher_largev6[AB]_a*.npz"; else BIAS="$E/teacher_largev4mix.npz"; fi
python3 $R/sim/package_single.py --out submit_$TAG --member $E/member_$TAG.zip::$VER --bias "$BIAS"
python3 $R/sim/check_zip.py $R/packages/submit_$TAG.zip

# ③ 검증셋 조립 → ④ 검증 컨테이너 자동 실행 → ⑤ 결과
bash $R/sim/prep_verify.sh $R/packages/submit_$TAG.zip
echo "[auto] 검증 컨테이너($TC) 실행..."
docker exec $TC bash /share/verify/submit_$TAG/run.sh 2>&1 | tee $R/work/verify_$TAG.txt
echo "[auto] 완료 — 결과: $R/work/verify_$TAG.txt / 게이트 PASS면 packages/submit_$TAG.zip 제출 후보"