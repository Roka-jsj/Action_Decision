# 재시작/컨테이너 전환 후 새 Claude Code 세션에 붙여넣을 프롬프트

---

```
jeong 브랜치 Dacon 236694(AI Agent Action Decision) 이어받기. 이 파일은 git으로 자동 이전되지만,
전체 인수인계 본문은 **HANDOFF.md**(repo 루트 — 단 .gitignore에 포함돼 git으로 안 옮겨진다.
work/·packages/와 함께 반드시 docker cp 등으로 별도 이전할 것)에 있다. 없으면 아래 요약으로 시작하고
DEBATE.md를 R53(GEN-rescue 발견)부터 읽을 것 — 그 이전(R1~R52)은 이미 종결된 축들의 기록이다.

## 4줄 요약 (2026-07-11 08:40 기준, 이전 직전 확정)
1. 은행(최고 LB) = 0.79026 (packages/submit_th85.zip). 컷 인플레 +0.001/일, D-0 예상 0.792~3.
2. **packages/submit_m1t3.zip은 조립 완료됐으나 제출 보류(HOLD)** — 홀드아웃 게이트C 적색(-0.0089),
   부검 결과 배선 무혐의·우리 FULL 런이 조원 m1보다 약함(파이프라인 격차 δ 가설, C1 참사 재해석).
3. 대조군 m1h8full(T3만 뺀 짝, 08:12 발진)이 완주돼 있으면 **python3 sim/eval_tau_delta.py** 실행(GPU 5분)
   → GO/축폐쇄/회색 자동 판정(codex R63b 사전고정: GO=τ_casc≥+0.003∧CI>0∧τ_solo≥0일 때만 제출).
   미완주/부재면 HANDOFF.md §5.6 명령으로 재발진(~2h). **mdeb-T3 FULL(프로브 +0.015)은 τ 양수 전 보류.**
4. 모든 발사·전략변경은 3자(운영자+codex+독립 레드팀) 서명 후에만 — 이 규율 절대 유지.

## 확정 교리 (재도출 시간 낭비 방지)
- 입력축(훈련분포 복원, 예: GEN-rescue)은 액면전이(~0.8) / 결정규칙 미세조정(가중·th·bias)은
  저전이(0.04~0.26) — 자원은 입력축·재학습(T3)에 집중. 단 T3-FULL 전이는 τ 판정 대기 중.
- 선택기 상한: 관측신호 기반 최적 선택기도 honest 델타 음수 — th축 완전 종결, 재론 금지.
- FULL 재학습은 fold 프로브 best-epoch의 업데이트 등가 환산(×0.8, fold ep10→FULL 8ep) — 외삽 금지.
- **타인(조원) 산출물의 레시피는 원자료 없이 "재현" 주장 금지**(wm20·C1·게이트C 3중 교훈).
- splits/splits.npz 절대 재생성 금지(재계산 비재현 — 전 OOF 조용히 오염).
- 홀드아웃은 FULL끼리 비교라도 암기 공유 계기일 뿐 — LB 예측치 아님. 부검엔 균일결손 분해
  (동일텍스트 슬라이스 vs 변경텍스트 슬라이스)를 쓸 것.

## 사용자 확정 목표
우승(0.80) — 단 3자 실측(레드팀 R57/R61)이 물리적 상한 ~0.796(P≈1~2%)을 수치로 입증.
현실 목표는 top-12 방어 + 코어 확보 후 문샷(합성데이터 증강, "제한된 복권") 병행.
새 컨테이너는 GPU 0,1 두 개 — codex R63c 배치안: GPU0=τ비의존 학습/준비, GPU1=검증·판정 대기열.

## codex 호출(모델: GPT-5.6 Terra xhigh)
V=/root/.vscode-server/extensions/openai.chatgpt-26.707.31428-linux-x64/bin/linux-x86_64/codex
CODEX_HOME=/root/.codex $V exec --sandbox read-only --skip-git-repo-check --ephemeral \
  -C /root/Action_Decision -m gpt-5.6-terra -c model_reasoning_effort='"xhigh"' \
  --color never - < brief.md > out.md
(구버전 26.623 바이너리는 gpt-5.6-terra 미지원 400 에러 — 반드시 26.707 사용. bwrap 제약으로
브리핑에 참고자료 인라인 필수. pkill은 브래킷 패턴([c]odex)·kill/재발사 호출 분리.)

세부 3종 인수인계(work/ 하위, HANDOFF.md와 함께 이전 필요): handoff_redteam.md(전 판정 이력·
방법론자산·확정교리), handoff_strategist.md(GEN-rescue 발견전문·죽은축·문샷설계),
handoff_engineer.md(코드자산 전체·ad_lib opt-in 스키마·splits 지뢰 상세·m1t3/대조군 상태).
```
