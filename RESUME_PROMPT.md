# 재시작/컨테이너 전환 후 새 Claude Code 세션에 붙여넣을 프롬프트

---

```
jeong 브랜치 Dacon 236694(AI Agent Action Decision) 이어받기. 이 파일은 git으로 자동 이전되지만,
전체 인수인계 본문은 **HANDOFF.md**(repo 루트, .gitignore 예외 — git clone만으로는 안 옮겨짐,
work/·packages/와 함께 docker cp 등으로 별도 이전 필요)에 있다. 없으면 아래 요약으로 시작하고
DEBATE.md를 R53(GEN-rescue 발견)부터 읽을 것 — 그 이전(R1~R52)은 이미 종결된 축들의 기록이다.

## 3줄 요약 (2026-07-10 23시 기준)
1. 은행(최고 LB) = 0.79026 (packages/submit_th85.zip). 다음 후보 submit_thasym95.zip은 조립 완료,
   30k 시간 리플레이만 남음(docker exec mun-jtest .../run.sh — 권한상 운영자 실행 필요할 수 있음).
2. GPU: mdeb-T3 프로브 학습 중 → 완료 시 데드맨 워처가 m1-T3 FULL 8ep 자동 발진 → 완료 후
   양자화·parity·조립 절차를 거쳐야 submit_m1t3.zip이 생긴다(아직 존재하지 않음).
3. 모든 발사·전략변경은 3자(운영자+codex+독립 레드팀) 서명 후에만 — 이 규율 절대 유지.

## 확정 교리 (재도출 시간 낭비 방지)
- 입력축(훈련분포 복원, 예: GEN-rescue)은 액면전이(~0.8) / 결정규칙 미세조정(가중·th·bias)은
  저전이(0.04~0.26) — 자원은 입력축·재학습(T3)에 집중.
- 선택기 상한: 관측가능 신호로 만든 최적 선택기도 honest 델타 음수 — th축(margin_th 미세조정)
  완전 종결, 재론 금지.
- FULL 재학습은 fold 프로브 best-epoch을 정확히 그대로 씀(외삽 절대 금지 — C1 사고 -0.0082 전례).
- splits/splits.npz는 절대 재생성 금지(이 환경 sklearn으로 재현 안 됨 — 재생성 시 전 OOF 조용히 오염).

## 사용자 확정 목표
우승(0.80) — 단 3자 실측(레드팀 R57/R61)이 물리적 상한 ~0.796(P≈1~2%)을 수치로 입증.
현실 목표는 top-12 방어(컷 인플레 +0.001/일, D-0 예상 ~0.792-3, P≈45~60%) + 코어 확보 후
문샷(합성데이터 증강, "제한된 복권")을 병행. 이 긴장은 미해소 — 새 세션이 이어받을 핵심 결정.

## codex 호출(모델: GPT-5.6 Terra xhigh, 2026-07-10 밤부터)
V=/root/.vscode-server/extensions/openai.chatgpt-26.707.31428-linux-x64/bin/linux-x86_64/codex
CODEX_HOME=/root/.codex $V exec --sandbox read-only --skip-git-repo-check --ephemeral \
  -C /root/Action_Decision -m gpt-5.6-terra -c model_reasoning_effort='"xhigh"' \
  --color never - < brief.md > out.md
(구버전 26.623 바이너리는 gpt-5.6-terra 미지원 400 에러 — 반드시 26.707 사용)

세부 3종 인수인계(work/ 하위, HANDOFF.md와 함께 이전 필요): handoff_redteam.md(전 판정 이력·
방법론자산·확정교리), handoff_strategist.md(GEN-rescue 발견전문·죽은축·문샷설계),
handoff_engineer.md(코드자산 전체·ad_lib opt-in 스키마·splits 지뢰 상세·m1-T3 FULL 진행상황).
```
