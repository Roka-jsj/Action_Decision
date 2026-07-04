#!/bin/bash
# babysit_full.sh <TAG> <MODEL> <EPOCHS> <LR> <BATCH> <PRUNE> — FULL-70k 멤버 1개 학습·회수.
# member_<TAG>.zip 로컬 존재 시 즉시 성공 종료. 세션 사망 시 처음부터 재기동(가중치 증분 불가), 최대 4회.
cd /home/vasebull/action_decision
TAG=$1; MODEL=$2; EPOCHS=$3; LR=$4; BATCH=$5; PRUNE=$6; VER=${7:-v4}; GPU=${8:-A100}
EXPD=action_decision_maximum/experiments
DEST=$EXPD/member_${TAG}.zip

for ATTEMPT in 1 2 3 4; do
  if [ -f "$DEST" ] && [ "$(stat -c%s "$DEST")" -gt 50000000 ]; then
    echo "### $TAG already secured ###"; exit 0
  fi
  S=$(echo "f${TAG}${ATTEMPT}" | tr -cd 'a-zA-Z0-9' | cut -c1-14)
  OK=0
  for t in $(seq 1 40); do
    R=$(timeout 150 colab new -s $S --gpu $GPU 2>&1 | grep -icE "session ready")
    [ "$R" -ge 1 ] && { OK=1; break; }
    echo "[$(date +%H:%M)] $S: slot busy, retry in 3min"
    sleep 180
  done
  [ "$OK" = "0" ] && { echo "### $TAG: 세션 확보 실패(2h) ###"; exit 1; }

  for f in open.zip ad_common.zip; do timeout 200 colab upload -s $S $f /content/$f 2>&1 | tail -1; done
  timeout 90 colab upload -s $S action_decision_maximum/src/train_full_cli.py /content/train_full_cli.py 2>&1 | tail -1
  printf 'import subprocess, os\nos.chdir("/content")\nenv=dict(os.environ, AD_MODEL="%s", AD_VERSION="%s", AD_MAXLEN="320", AD_EPOCHS="%s", AD_LR="%s", AD_BATCH="%s", AD_LLRD="1", AD_SEED="1234", AD_PRUNE="%s", AD_TAG="%s", TOKENIZERS_PARALLELISM="false")\np=subprocess.Popen(["python","/content/train_full_cli.py"], stdout=open("/content/train.log","w"), stderr=subprocess.STDOUT, env=env)\nprint("launched", p.pid)\n' "$MODEL" "$VER" "$EPOCHS" "$LR" "$BATCH" "$PRUNE" "$TAG" | timeout 120 colab exec -s $S 2>&1 | tail -1
  echo "### $S launched FULL $TAG ($MODEL ep=$EPOCHS attempt=$ATTEMPT) ###"

  MISS=0
  while true; do
    sleep 240
    ST=$(echo "
import os
print('DONE' if os.path.exists('/content/DONE_${TAG}') else 'RUNNING')
L=open('/content/train.log').read() if os.path.exists('/content/train.log') else ''
ls=[x for x in L.splitlines() if 'epoch' in x or 'prune' in x or 'zip' in x or 'Traceback' in x or 'Error' in x]
print(ls[-1] if ls else '')
" | timeout 75 colab exec -s $S 2>/dev/null | grep -aE "DONE|RUNNING|epoch|prune|zip|Traceback|Error")
    HEAD=$(echo "$ST" | head -1)
    echo "[$(date +%H:%M)] $S: $HEAD | $(echo "$ST" | tail -1)"
    if echo "$ST" | grep -qa "Traceback"; then
      echo "### $S CRASHED — 로그 회수 후 재기동 ###"
      timeout 90 colab download -s $S /content/train.log $EXPD/${TAG}_crash.log 2>&1 | tail -1
      timeout 40 colab stop -s $S 2>&1 | tail -1
      break
    fi
    if [ "$HEAD" = "DONE" ]; then
      echo "[$(date +%H:%M)] $S: downloading member zip..."
      timeout 1800 colab download -s $S /content/member_${TAG}.zip $DEST 2>&1 | tail -1
      if [ -f "$DEST" ] && [ "$(stat -c%s "$DEST")" -gt 50000000 ]; then
        timeout 40 colab stop -s $S 2>&1 | tail -1
        echo "### $TAG SECURED ($(stat -c%s "$DEST" | awk '{printf "%.0fMB", $1/1e6}')) ###"
        exit 0
      fi
      echo "### $TAG download 불완전 — 재시도 ###"
      continue
    fi
    if [ -z "$HEAD" ]; then
      MISS=$((MISS+1))
      if [ "$MISS" -ge 2 ]; then
        echo "### $S lost — 재기동 ###"
        timeout 40 colab stop -s $S 2>&1 | tail -1
        break
      fi
    else
      MISS=0
    fi
  done
done
echo "### $TAG: 4회 시도 실패 ###"; exit 1
