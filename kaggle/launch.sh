#!/bin/bash
# kaggle/launch.sh — kaggle.json 배치 후 원커맨드 발진.
# 1) USERNAME 치환 2) 데이터셋 생성 3) 커널 2개 push
set -e
cd /home/vasebull/action_decision/kaggle
export PATH=$PATH:~/.local/bin
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
U=$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")
echo "kaggle user: $U"
for f in ds/dataset-metadata.json k_kluev6/kernel-metadata.json k_basev6e5/kernel-metadata.json; do
  sed -i "s/USERNAME/$U/g" $f
done
kaggle datasets create -p ds --dir-mode zip 2>&1 | tail -2
echo "데이터셋 처리 대기 60s..."; sleep 60
kaggle kernels push -p k_kluev6 2>&1 | tail -1
kaggle kernels push -p k_basev6e5 2>&1 | tail -1
echo "=== 상태 확인: kaggle kernels status $U/ad-kluev6-teacher ==="
