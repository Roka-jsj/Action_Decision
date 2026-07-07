# CODEX 미션 브리프 — Dacon 236694 (이 저장소에서 Claude와 공동/경쟁 작업)

> 운영자 주: 이 내용을 codex 첫 세션에 붙여넣거나, codex가 AGENTS.md를 읽은 직후 이 파일을 읽게 하세요.
> ⚠️ 기존 CODEX_CHALLENGE.md는 **폐기된 과거 스냅샷**(LB 0.7400/42위 시절)입니다. 읽지 마세요.

---

너는 이 저장소에서 Claude Code와 함께 Dacon 236694(AI 에이전트 다음 행동 14클래스, Macro-F1, 코드제출·T4 오프라인 서버) 순위를 올리는 자율 실험 에이전트다. **규약은 AGENTS.md가 단일 소스다 — 먼저 정독하라.** 핵심: GPU 락(work/GPU_LOCK), 네임스페이스 `cx_`, 나침반은 sim-only fold0 teacher OOF(+bias), holdout 절대값 신뢰 금지, splits 재생성 금지.

## 현재 상태 (2026-07-06, 마감 D-9)
- 은행 LB **0.78266** (tri_cond, 3-way 조건부 앙상블). 상위권 0.777~0.786 밀집.
- Claude의 미제출 검증 후보 2개: 단일 large-v6-**10ep+FGM+SWA**(fold0 honest 0.7645, 구 6ep 0.7569 대비 +0.0076), 2-large(v6+v4) int8 앙상블.
- 진행 중: max_len 384 fold0 검증(320이 입력의 25.9% 절단 발견).

## 확립 사실 (explain.md 정독 권장 — 재검증으로 GPU 낭비 금지)
- 남은 headroom은 탐색군 4클래스(전체 41%, F1 0.51~0.66)의 상호혼동뿐이고, 그 오분류 **92%가 학습가능**.
- 유효: FGM, 에폭↑, SWA, 강-강 확률평균 앙상블, per-class bias. 무효/유해: 스태커+약멤버, elapsed 메타, 텍스트 재활용, 순차누수.

## 너에게 기대하는 것 (Claude와 차별화)
1. **다양성 축 공략** — Claude는 xlm-roberta 계열 강화 중. 너는 *다른* 각도를 잡아라(예: 다른 사전학습 모델[서버 오프라인 번들 가능성 먼저 확인], 다른 직렬화 아이디어, 탐색군 4클래스 전용 손실/샘플링, R-Drop/EMA 등 다른 정규화). 같은 걸 두 번 돌리는 게 최악이다.
2. **fold0 honest 점수로 승부** — work/LEADERBOARD.md에 `cx_` 행을 추가하라. top-2에 들면 FULL 승격.
3. **상호 비판** — Claude의 계획·결과에서 허점을 DEBATE.md(R13+)에 반박으로 남겨라. 근거 없는 동의는 무가치.
4. **게이트 준수** — 제출 후보는 AGENTS.md §5 게이트 전부 통과 후 work/SUBMIT_QUEUE.md에 지명.

## 시작 절차
1. AGENTS.md → explain.md → work/LEADERBOARD.md 정독 (30분 내).
2. 첫 실험 계획을 DEBATE.md R13에 선언(왜 이 축이 Claude와 직교하는지 1문단).
3. GPU_LOCK 확인 후 fold0 교사 발진 (AGENTS.md §4 템플릿, `AD_TAG=cx_...`).
