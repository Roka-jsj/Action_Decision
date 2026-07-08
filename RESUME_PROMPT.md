# 재시작 후 새 Claude Code 세션에 붙여넣을 프롬프트

아래 블록을 새 세션 첫 입력으로 그대로 붙여넣으세요.

---

```
지금 브랜치 jeong에서 Dacon 236694(Action Decision) 작업을 이어간다. 컨테이너 재시작으로 이전 세션이 끊겼다. 먼저 PROJECT.md의 §5.23(재개 스냅샷 — 최우선), DEBATE.md의 R22~R26, 그리고 메모리를 읽고 전체 맥락을 복원해라.

핵심만: 팀최고=조원 tri_v4new 0.78449(14위, 컷 top12 밖). 목표=0.80추격이 아니라 top12 방어(0.80은 1~2%). 히든테스트가 train에 OOD라 train기반 실험이 다 반전됨(LB만 진실). 나=jeong 트랙(retrieval/구조/LB프로빙/session-balanced), 조원=mun-train GPU1(model/mdeberta/seed/tri, 중복금지). codex와 상시 비판토론(/codex 또는 work/rNN_brief 절차), 하네스 엔지니어링 항상 포함, GPU 놀리지 말 것.

재개 순서(§5.23): ①nvidia-smi로 GPU 복구 확인(안되면 host에서 docker restart 재시도) ②LB 클래스분포 프로빙(packages/probe_*.zip 제출→eda/prior_from_probe.py 역산→bias 히든prior 재적합) ③session-balanced fine-tune 구현·검증·LB canary ④retrieval→tri 보수 이식(조원 tri logits 필요). 매 수마다 codex 토론하고 DEBATE.md·experiments_master.csv 기록.

지금 GPU/컨테이너 상태부터 점검하고, 위 순서로 이어서 진행해라.
```

---

## 참고: 재시작 절차 (호스트 터미널)
```
docker restart mun-jtrain mun-jtest mun-train mun-test    # 데이터 보존(rm 아님), NVML cgroup 복구
```
- 재시작 후 VS Code에서 mun-jtrain에 재접속 → 폴더 /root/Action_Decision → Claude Code 새 세션 → 위 프롬프트.
- 파일 전부 보존됨(restart는 파일시스템 유지). 코드·문서는 origin/jeong에도 push됨(2중 안전).
- packages/(33개 zip)·action_decision_maximum/experiments/(19 member) 보존.

## 현재 in-flight (재개 시 확인)
- 실행중 GPU작업 없음(NVML 다운으로 전부 정지). 깨끗한 재개 지점.
- LB 프로빙 대기: probe zip 14개 준비됨. 사용자가 제출→점수 회신→역산.
- codex R26까지 완료(work/r26_codex.md). R27부터 이어감.
