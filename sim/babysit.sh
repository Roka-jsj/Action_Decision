#!/bin/bash
# babysit.sh — 교사 세션 자동 관리: t5 회수 + t6 할당·기동·회수
cd /home/vasebull/action_decision
EXPD=action_decision_maximum/experiments
T6_LAUNCHED=0
for i in $(seq 1 40); do
  # ---- t5 (mdeberta) ----
  T5=$(echo "
import os
print('DONE' if any(f.startswith('DONE_') for f in os.listdir('/content')) else 'RUNNING')
L=open('/content/teach.log').read() if os.path.exists('/content/teach.log') else ''
ls=[x for x in L.splitlines() if 'val=' in x or 'covered-OOF' in x or 'Error' in x or 'Traceback' in x]
print('\n'.join(ls[-3:]))
" | timeout 75 colab exec -s t5 2>/dev/null | grep -aE "DONE|RUNNING|val=|covered|Error|Traceback")
  echo "[$(date +%H:%M)] t5: $(echo "$T5" | head -1) | $(echo "$T5" | tail -1)"
  if echo "$T5" | grep -q "^DONE"; then
    timeout 120 colab download -s t5 /content/teacher_mdeberta_s1234.npz $EXPD/teacher_mdeberta_s1234.npz 2>&1 | tail -1
    timeout 40 colab stop -s t5 2>&1 | tail -1
    echo "### t5(mdeberta) npz secured ###"
    T5_DONE=1
  fi
  if echo "$T5" | grep -qa "Traceback"; then
    echo "### t5 CRASHED — 로그 확인 필요 ###"
    timeout 90 colab download -s t5 /content/teach.log $EXPD/teach_mdeberta_crash.log 2>&1 | tail -1
    T5_DONE=1
  fi
  # ---- t6 (base s777 5ep 재실행) ----
  if [ "$T6_LAUNCHED" = "0" ]; then
    R=$(timeout 150 colab new -s t6 --gpu A100 2>&1 | grep -icE "session ready")
    if [ "$R" -ge 1 ]; then
      for f in open.zip ad_common.zip; do timeout 200 colab upload -s t6 $f /content/$f 2>&1 | tail -1; done
      timeout 90 colab upload -s t6 action_decision_maximum/src/teacher_cli.py /content/teacher_cli.py 2>&1 | tail -1
      printf 'import subprocess, os\nos.chdir("/content")\nenv=dict(os.environ, AD_MODEL="xlm-roberta-base", AD_SEED="777", AD_VERSION="v4", AD_MAXLEN="320", AD_EPOCHS="5", AD_LR="3e-5", AD_BATCH="96", AD_LLRD="1", AD_TAG="base_s777e5", TOKENIZERS_PARALLELISM="false")\np=subprocess.Popen(["python","/content/teacher_cli.py"], stdout=open("/content/teach.log","w"), stderr=subprocess.STDOUT, env=env)\nprint("t6 launched", p.pid)\n' | timeout 120 colab exec -s t6 2>&1 | tail -1
      T6_LAUNCHED=1
      echo "### t6(base s777 5ep) launched ###"
    fi
  else
    T6=$(echo "
import os
print('DONE' if any(f.startswith('DONE_') for f in os.listdir('/content')) else 'RUNNING')
L=open('/content/teach.log').read() if os.path.exists('/content/teach.log') else ''
ls=[x for x in L.splitlines() if 'covered-OOF' in x or 'Traceback' in x]
print('\n'.join(ls[-1:]))
" | timeout 75 colab exec -s t6 2>/dev/null | grep -aE "DONE|RUNNING|covered|Traceback")
    echo "[$(date +%H:%M)] t6: $(echo "$T6" | head -1)"
    if echo "$T6" | grep -q "^DONE"; then
      timeout 120 colab download -s t6 /content/teacher_base_s777e5.npz $EXPD/teacher_base_s777e5.npz 2>&1 | tail -1
      timeout 40 colab stop -s t6 2>&1 | tail -1
      echo "### t6 npz secured ###"
      T6_DONE=1
    fi
  fi
  # 종료 조건
  if [ "${T5_DONE:-0}" = "1" ] && [ "${T6_DONE:-0}" = "1" ]; then
    echo "### ALL TEACHERS SECURED ###"; break
  fi
  sleep 300
done
ls -la $EXPD/*.npz