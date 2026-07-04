#!/bin/bash
# monitor_session.sh <SESSION> <TAG> — 이미 떠 있는 교사 세션 폴링·증분 회수·DONE 시 정지.
# 세션 손실 시 종료(재기동은 babysit_teacher로 별도 수행).
cd /home/vasebull/action_decision
S=$1; TAG=$2
EXPD=action_decision_maximum/experiments
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
    timeout 90 colab download -s $S /content/teach.log $EXPD/${TAG}_crash.log 2>&1 | tail -1
    timeout 40 colab stop -s $S 2>&1 | tail -1
    echo "### $S CRASHED — 로그 회수 완료 ###"; exit 1
  fi
  if [ "$HEAD" = "DONE" ]; then
    timeout 40 colab stop -s $S 2>&1 | tail -1
    echo "### $S DONE — $TAG SECURED ###"; exit 0
  fi
  if [ -z "$HEAD" ]; then
    MISS=$((MISS+1))
    [ "$MISS" -ge 2 ] && { echo "### $S lost ###"; timeout 40 colab stop -s $S 2>&1 | tail -1; exit 2; }
  else
    MISS=0
  fi
done
