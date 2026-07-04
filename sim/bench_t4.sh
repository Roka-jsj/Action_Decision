#!/bin/bash
# bench_t4.sh <package.zip> — 실제 T4에서 30k 추론 타이밍 벤치 (제출 전 필수 관문).
cd /home/vasebull/action_decision
PKG=$1
S=bencht4
for t in $(seq 1 20); do
  R=$(timeout 150 colab new -s $S --gpu T4 2>&1 | grep -icE "session ready")
  [ "$R" -ge 1 ] && break
  echo "[$(date +%H:%M)] $S: slot busy, retry in 3min"; sleep 180
done
echo "[$(date +%H:%M)] uploading package ($(stat -c%s "$PKG" | awk '{printf "%.0fMB", $1/1e6}'))..."
timeout 1800 colab upload -s $S "$PKG" /content/pkg.zip 2>&1 | tail -1
timeout 200 colab upload -s $S open.zip /content/open.zip 2>&1 | tail -1
timeout 60 colab upload -s $S sim/make_synth_test.py /content/sim/make_synth_test.py 2>&1 | tail -1

echo '
import zipfile, os, subprocess, sys
os.chdir("/content")
for z, d in [("open.zip", "."), ("pkg.zip", "pkg")]:
    with zipfile.ZipFile(z) as f: f.extractall(d)
subprocess.run([sys.executable, "sim/make_synth_test.py", "30000", "/content/pkg/data"], check=True)
code = "import time,subprocess,sys;t=time.time();r=subprocess.run([sys.executable,\"script.py\"]);print(\"ELAPSED\",round(time.time()-t,1),\"rc\",r.returncode,flush=True)"
p = subprocess.Popen([sys.executable, "-c", code], cwd="/content/pkg",
                     stdout=open("/content/bench.log","w"), stderr=subprocess.STDOUT)
print("bench launched", p.pid)
' | timeout 300 colab exec -s $S 2>&1 | tail -2

for i in $(seq 1 20); do
  sleep 120
  ST=$(echo "
import os
L = open('/content/bench.log').read() if os.path.exists('/content/bench.log') else ''
print('ELAPSED' if 'ELAPSED' in L else 'RUNNING')
ls = [x for x in L.splitlines() if x.strip()]
print(ls[-1] if ls else '')
" | timeout 75 colab exec -s $S 2>/dev/null | grep -aE "ELAPSED|RUNNING|rows|Error|Traceback")
  echo "[$(date +%H:%M)] bench: $(echo "$ST" | tail -1)"
  if echo "$ST" | grep -q "ELAPSED"; then
    echo "=== BENCH RESULT ==="
    echo "
print(open('/content/bench.log').read())
import csv
rows = list(csv.reader(open('/content/pkg/output/submission.csv')))
print('rows:', len(rows)-1)
" | timeout 75 colab exec -s $S 2>/dev/null | tail -5
    break
  fi
done
timeout 40 colab stop -s $S 2>&1 | tail -1
