#!/bin/bash
# babysit_teacher.sh <NAME> <MODEL> <VERSION> <MAXLEN> <EPOCHS> <LR> <BATCH> <FGM> <LO> <HI> [GPU] [EXAU]
# 교사 fold 범위를 자동 재기동으로 확보 (npz 증분 회수, 첫 미확보 fold부터 재개).
cd /home/vasebull/action_decision
NAME=$1; MODEL=$2; VERSION=$3; MAXLEN=$4; EPOCHS=$5; LR=$6; BATCH=$7; FGM=$8; LO=$9; HI=${10}
GPU=${11:-A100}; EXAU=${12:-0}
EXPD=action_decision_maximum/experiments

next_fold() {
python3 - "$NAME" "$LO" "$HI" <<'PY'
import sys, glob
import numpy as np
name, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
done = set()
for p in glob.glob(f"action_decision_maximum/experiments/teacher_{name}_a*.npz"):
    try:
        z = np.load(p, allow_pickle=True)
        done.update(range(int(z["fold_lo"]), int(z["fold_hi"])))
    except Exception:
        pass
miss = [f for f in range(lo, hi) if f not in done]
print(miss[0] if miss else "ALL")
PY
}

ATTEMPT=0
while true; do
  NEXT=$(next_fold)
  if [ "$NEXT" = "ALL" ]; then echo "### $NAME folds [$LO,$HI) ALL SECURED ###"; break; fi
  ATTEMPT=$((ATTEMPT+1))
  if [ "$ATTEMPT" -gt 8 ]; then echo "### $NAME: 8회 시도 초과 — 중단 ###"; exit 1; fi
  S=$(echo "${NAME}${ATTEMPT}" | tr -cd 'a-zA-Z0-9' | cut -c1-14)
  TAG="${NAME}_a${ATTEMPT}"

  OK=0
  for t in $(seq 1 60); do
    R=$(timeout 150 colab new -s $S --gpu $GPU 2>&1 | grep -icE "session ready")
    [ "$R" -ge 1 ] && { OK=1; break; }
    echo "[$(date +%H:%M)] $S: slot busy, retry in 3min"
    sleep 180
  done
  [ "$OK" = "0" ] && { echo "### $NAME: 세션 확보 실패(3h) — 중단 ###"; exit 1; }

  for f in open.zip ad_common.zip; do timeout 200 colab upload -s $S $f /content/$f 2>&1 | tail -1; done
  timeout 90 colab upload -s $S action_decision_maximum/src/teacher_cli.py /content/teacher_cli.py 2>&1 | tail -1
  printf 'import subprocess, os\nos.chdir("/content")\nenv=dict(os.environ, AD_MODEL="%s", AD_SEED="1234", AD_VERSION="%s", AD_MAXLEN="%s", AD_EPOCHS="%s", AD_LR="%s", AD_BATCH="%s", AD_LLRD="1", AD_FGM="%s", AD_FOLD_LO="%s", AD_FOLD_HI="%s", AD_TAG="%s", AD_EXCLUDE_AU="%s", TOKENIZERS_PARALLELISM="false")\np=subprocess.Popen(["python","/content/teacher_cli.py"], stdout=open("/content/teach.log","w"), stderr=subprocess.STDOUT, env=env)\nprint("launched", p.pid)\n' \
    "$MODEL" "$VERSION" "$MAXLEN" "$EPOCHS" "$LR" "$BATCH" "$FGM" "$NEXT" "$HI" "$TAG" "$EXAU" | timeout 120 colab exec -s $S 2>&1 | tail -1
  echo "### $S launched $NAME folds [$NEXT,$HI) ($MODEL $VERSION@$MAXLEN ep=$EPOCHS fgm=$FGM gpu=$GPU exau=$EXAU) ###"

  MISS=0
  while true; do
    sleep 240
    ST=$(echo "
import os
print('DONE' if os.path.exists('/content/DONE_${TAG}') else 'RUNNING')
L=open('/content/teach.log').read() if os.path.exists('/content/teach.log') else ''
ls=[x for x in L.splitlines() if 'val=' in x or 'covered-OOF' in x or 'Traceback' in x or 'Error' in x]
print(ls[-1] if ls else '')
" | timeout 75 colab exec -s $S 2>/dev/null | grep -aE "DONE|RUNNING|val=|covered|Traceback|Error")
    HEAD=$(echo "$ST" | head -1)
    echo "[$(date +%H:%M)] $S: $HEAD | $(echo "$ST" | tail -1)"
    timeout 120 colab download -s $S /content/teacher_${TAG}.npz $EXPD/teacher_${TAG}.npz 2>/dev/null | tail -1
    if echo "$ST" | grep -qa "Traceback"; then
      echo "### $S CRASHED — 로그 회수 후 재기동 ###"
      timeout 90 colab download -s $S /content/teach.log $EXPD/${TAG}_crash.log 2>&1 | tail -1
      timeout 40 colab stop -s $S 2>&1 | tail -1
      break
    fi
    if [ "$HEAD" = "DONE" ]; then
      timeout 40 colab stop -s $S 2>&1 | tail -1
      echo "### $S DONE ###"
      break
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
