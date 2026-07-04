#!/bin/bash
# bench_t4_hold.sh <package.zip> [session] — T4 30k 추론 타이밍 + holdout 5.8k 채점 (제출 전 최종 관문).
# 사전에 sim/make_holdout_test.py 로 /tmp/ad_hold/hold_data.zip 생성 필요.
cd /home/vasebull/action_decision
PKG=$1
S=${2:-bencht4h}
for t in $(seq 1 20); do
  R=$(timeout 150 colab new -s $S --gpu T4 2>&1 | grep -icE "session ready")
  [ "$R" -ge 1 ] && break
  echo "[$(date +%H:%M)] $S: slot busy, retry in 3min"; sleep 180
done
echo "[$(date +%H:%M)] uploading package ($(stat -c%s "$PKG" | awk '{printf "%.0fMB", $1/1e6}')) in 25MB chunks..."
rm -f /tmp/pkgpart_${S}_*
split -b 25M "$PKG" /tmp/pkgpart_${S}_
for p in /tmp/pkgpart_${S}_*; do
  for r in 1 2 3; do
    OUT=$(timeout 600 colab upload -s $S "$p" /content/$(basename $p) 2>&1 | tail -1)
    echo "$OUT" | grep -q "Uploaded" && break
    echo "[$(date +%H:%M)] chunk $(basename $p) retry $r: $OUT"; sleep 10
  done
done
EXPECT=$(stat -c%s "$PKG")
ASM=$(echo "
import glob, os
parts = sorted(glob.glob('/content/pkgpart_*'))
with open('/content/pkg.zip', 'wb') as o:
    for p in parts:
        o.write(open(p, 'rb').read()); os.remove(p)
sz = os.path.getsize('/content/pkg.zip')
print('ASSEMBLY_OK' if sz == $EXPECT else f'ASSEMBLY_BAD {sz} != $EXPECT (parts={len(parts)})')
" | timeout 300 colab exec -s $S 2>&1 | grep -aE "ASSEMBLY")
echo "[assembly] $ASM"
rm -f /tmp/pkgpart_${S}_*
if ! echo "$ASM" | grep -q "ASSEMBLY_OK"; then
  echo "### 조립 실패 — 중단 ###"; timeout 40 colab stop -s $S 2>&1 | tail -1; exit 1
fi
timeout 200 colab upload -s $S /tmp/ad_hold/hold_data.zip /content/hold_data.zip 2>&1 | tail -1

echo '
import zipfile, os, subprocess, sys
os.chdir("/content")
with zipfile.ZipFile("pkg.zip") as f: f.extractall("pkg")
with zipfile.ZipFile("hold_data.zip") as f: f.extractall("pkg")   # data/test.jsonl 30k(ho 5.8k 포함)
code = "import time,subprocess,sys;t=time.time();r=subprocess.run([sys.executable,\"script.py\"]);print(\"ELAPSED\",round(time.time()-t,1),\"rc\",r.returncode,flush=True)"
p = subprocess.Popen([sys.executable, "-c", code], cwd="/content/pkg",
                     stdout=open("/content/bench.log","w"), stderr=subprocess.STDOUT)
print("bench launched", p.pid)
' | timeout 300 colab exec -s $S 2>&1 | tail -2

for i in $(seq 1 25); do
  sleep 60
  ST=$(echo "
import os
L = open('/content/bench.log').read() if os.path.exists('/content/bench.log') else ''
print('ELAPSED' if 'ELAPSED' in L else 'RUNNING')
ls = [x for x in L.splitlines() if x.strip()]
print(ls[-1] if ls else '')
" | timeout 75 colab exec -s $S 2>/dev/null | grep -aE "ELAPSED|RUNNING|rows|Error|Traceback")
  echo "[$(date +%H:%M)] bench: $(echo "$ST" | tail -1)"
  if echo "$ST" | grep -qa "ELAPSED"; then
    echo "=== BENCH RESULT ==="
    echo "print(open('/content/bench.log').read())" | timeout 75 colab exec -s $S 2>/dev/null | tail -4
    timeout 120 colab download -s $S /content/pkg/output/submission.csv /tmp/ad_hold/submission_bench.csv 2>&1 | tail -1
    python3 sim/score_holdout.py /tmp/ad_hold/submission_bench.csv
    break
  fi
done
timeout 40 colab stop -s $S 2>&1 | tail -1