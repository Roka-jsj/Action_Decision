---
description: 실험실 상태 한눈에 — GPU·진행중 학습·LB은행·조원최고·제출후보
---

아래를 실행해 현재 상태를 요약 보고한다 (실행 후 표로 정리):

```bash
cd /root/Action_Decision
export DOCKER_API_VERSION=1.43
echo "=== 시각 ==="; date
echo "=== GPU (내 GPU0) ==="; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "=== 진행중 학습 ==="; pgrep -fa "teacher_cli|train_full_cli" | grep -v grep | sed 's|python3 ||'
echo "=== 컨테이너 ==="; docker ps --format '{{.Names}}\t{{.Status}}' | grep mun
echo "=== 최근 work 로그 (진행 마커) ==="; ls -t work/*.log 2>/dev/null | head -5 | while read f; do echo "--- $f ---"; grep -vE "WARNING|pip |You're using" "$f" | tail -2; done
echo "=== LB 은행 (내 트랙 실측) ==="; grep "LB실측" experiments_master.csv | grep -oE "^[a-z_0-9]+,|0\.7[0-9]{4,}" | tail -20
echo "=== 제출후보 (packages) ==="; ls -t packages/*.zip 2>/dev/null | head -8 | xargs -I{} sh -c 'echo "$(du -h {} | cut -f1)  {}"'
```

요약 시 포함: ①GPU가 놀고 있으면 즉시 다음 학습 연쇄(유휴 금지) ②팀 최고점 대비 현 위치 ③제출 대기 후보 ④다음 수 1줄.
