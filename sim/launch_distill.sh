#!/bin/bash
# launch_distill.sh <session> [FULL(0|1)] — 교사 npz 업로드 + distill_cli 기동 + babysit + zip 회수
S=${1:-d1}; FULL=${2:-0}
cd /home/vasebull/action_decision
EXPD=action_decision_maximum/experiments
echo "=== distill session $S (FULL=$FULL) ==="
timeout 150 colab new -s $S --gpu A100 2>&1 | grep -iE "ready|error|toomany" | tail -1 || exit 1
for f in open.zip ad_common.zip; do timeout 200 colab upload -s $S $f /content/$f 2>&1 | tail -1; done
timeout 90 colab upload -s $S action_decision_maximum/src/distill_cli.py /content/distill_cli.py 2>&1 | tail -1
for npz in $EXPD/teacher_*.npz; do
  timeout 120 colab upload -s $S "$npz" /content/$(basename $npz) 2>&1 | tail -1
done
printf 'import subprocess, os\nos.chdir("/content")\nenv=dict(os.environ, AD_VERSION="v4", AD_MAXLEN="320", AD_EPOCHS="3", AD_LR="3e-5", AD_BATCH="96", AD_ALPHA="0.7", AD_LLRD="1", AD_FULL="%s", TOKENIZERS_PARALLELISM="false")\np=subprocess.Popen(["python","/content/distill_cli.py"], stdout=open("/content/distill.log","w"), stderr=subprocess.STDOUT, env=env)\nprint("distill launched", p.pid)\n' "$FULL" | timeout 120 colab exec -s $S 2>&1 | tail -1

# babysit
for i in $(seq 1 30); do
  sleep 300
  R=$(echo "
import os
print('DONE' if os.path.exists('/content/DONE_DISTILL') else 'RUNNING')
L=open('/content/distill.log').read() if os.path.exists('/content/distill.log') else ''
ls=[x for x in L.splitlines() if 'ENSEMBLE' in x or 'distill epoch' in x or 'postproc' in x or 'timing' in x or 'zip' in x or 'Traceback' in x]
print('\n'.join(ls[-4:]))
" | timeout 75 colab exec -s $S 2>/dev/null | grep -aE "DONE|RUNNING|ENSEMBLE|distill|postproc|timing|zip|Traceback")
  echo "[$(date +%H:%M)] $R" | head -5
  if echo "$R" | grep -q "^DONE"; then
    timeout 500 colab download -s $S /content/submit_maximum.zip ./submit_distill_full$FULL.zip 2>&1 | tail -1
    timeout 90 colab download -s $S /content/distill.log $EXPD/distill_full$FULL.log 2>&1 | tail -1
    timeout 40 colab stop -s $S 2>&1 | tail -1
    echo "### DISTILL SECURED: submit_distill_full$FULL.zip ###"
    ls -la submit_distill_full$FULL.zip
    break
  fi
  if echo "$R" | grep -qa "Traceback"; then
    timeout 90 colab download -s $S /content/distill.log $EXPD/distill_crash.log 2>&1 | tail -1
    echo "### DISTILL CRASHED — log saved ###"; break
  fi
done