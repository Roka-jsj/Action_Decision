#!/bin/bash
# run_soft_labels.sh — T4 세션에서 2-large teacher 소프트라벨 생성·회수 (증류 대비).
cd /home/vasebull/action_decision
S=softlab
EXPD=action_decision_maximum/experiments
for t in $(seq 1 20); do
  R=$(timeout 150 colab new -s $S --gpu T4 2>&1 | grep -icE "session ready")
  [ "$R" -ge 1 ] && break
  echo "[$(date +%H:%M)] $S: slot busy, retry 3min"; sleep 180
done
for f in open.zip ad_common.zip; do timeout 200 colab upload -s $S $f /content/$f 2>&1 | tail -1; done
timeout 60 colab upload -s $S sim/gen_soft_labels.py /content/gen_soft_labels.py 2>&1 | tail -1

for M in largefullv6_q8:m_v6 largev4_8ep_q8:m_v4; do
  Z=${M%%:*}; D=${M##*:}
  rm -f /tmp/slpart_*
  split -b 25M $EXPD/member_$Z.zip /tmp/slpart_
  for p in /tmp/slpart_*; do
    for r in 1 2 3; do
      OUT=$(timeout 600 colab upload -s $S "$p" /content/$(basename $p) 2>&1 | tail -1)
      echo "$OUT" | grep -q "Uploaded" && break
      echo "chunk retry $r: $OUT"; sleep 10
    done
  done
  EXPECT=$(stat -c%s $EXPD/member_$Z.zip)
  echo "
import glob, os, zipfile
parts = sorted(glob.glob('/content/slpart_*'))
with open('/content/mm.zip','wb') as o:
    for p in parts: o.write(open(p,'rb').read()); os.remove(p)
sz = os.path.getsize('/content/mm.zip')
assert sz == $EXPECT, f'SIZE_BAD {sz}'
with zipfile.ZipFile('/content/mm.zip') as z: z.extractall('/content/$D')
os.remove('/content/mm.zip')
print('$D READY')
" | timeout 300 colab exec -s $S 2>&1 | grep -aE "READY|BAD|Error" | tail -1
  rm -f /tmp/slpart_*
done

echo '
import zipfile, subprocess, sys, os
os.chdir("/content")
for z in ["open.zip", "ad_common.zip"]:
    with zipfile.ZipFile(z) as f: f.extractall(".")
p = subprocess.Popen([sys.executable, "gen_soft_labels.py"], stdout=open("/content/soft.log","w"), stderr=subprocess.STDOUT)
print("launched", p.pid)
' | timeout 300 colab exec -s $S 2>&1 | tail -1

for i in $(seq 1 40); do
  sleep 90
  ST=$(echo "
import os
print('DONE' if os.path.exists('/content/DONE_soft') else 'RUNNING')
L = open('/content/soft.log').read() if os.path.exists('/content/soft.log') else ''
ls = [x for x in L.splitlines() if x.strip()]
print(ls[-1] if ls else '')
" | timeout 75 colab exec -s $S 2>/dev/null | grep -aE "DONE|RUNNING|done|soft|Error|Traceback")
  echo "[$(date +%H:%M)] soft: $(echo "$ST" | tail -1)"
  echo "$ST" | grep -qa "Traceback" && { timeout 90 colab download -s $S /content/soft.log $EXPD/soft_crash.log 2>&1|tail -1; break; }
  if echo "$ST" | head -1 | grep -qa "DONE"; then
    timeout 300 colab download -s $S /content/soft_labels.npz $EXPD/soft_labels_str2.npz 2>&1 | tail -1
    echo "### SOFT LABELS SECURED ###"
    break
  fi
done
timeout 40 colab stop -s $S 2>&1 | tail -1