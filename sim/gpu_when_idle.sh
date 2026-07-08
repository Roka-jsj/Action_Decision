#!/bin/bash
# GPU 유휴시에만 학습 발사 — 다른 컨테이너와 GPU 공유 정책 (사용자 지시 07-09).
# 유휴 판정: compute 프로세스 0개 AND 사용 메모리 < 2GB. 폴링 60s, 최대 대기 기본 6h.
# usage: bash sim/gpu_when_idle.sh [max_wait_sec] -- <command...>
#   예: bash sim/gpu_when_idle.sh 21600 -- env AD_MODEL=... python3 .../teacher_cli.py
set -e
MAX=${1:-21600}
shift
[ "$1" = "--" ] && shift

t0=$(date +%s)
while true; do
  nproc_gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ "${nproc_gpu:-0}" -eq 0 ] && [ "${mem:-99999}" -lt 2000 ]; then
    echo "[gpu_when_idle] GPU 유휴 확인(mem=${mem}MB) — 발사: $*"
    exec "$@"
  fi
  now=$(date +%s)
  if [ $((now - t0)) -ge "$MAX" ]; then
    echo "[gpu_when_idle] 최대 대기(${MAX}s) 초과 — 발사 포기. GPU 사용중(proc=${nproc_gpu}, mem=${mem}MB)"
    exit 1
  fi
  sleep 60
done
