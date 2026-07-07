# AGENTS.md — 이중 에이전트(Claude ⇄ codex) 공동 작업 규약

> Dacon 236694 예선. 마감 **2026-07-15**. 현재 은행 최고 **LB 0.78266**(tri_cond). 목표 0.80+.
> 이 저장소에서 **Claude Code와 codex가 동시에** 실험한다. 사람(운영자)은 Dacon 제출과 LB 회신만 담당.
> 이 파일이 규약의 단일 소스다. 위반하면 상대 에이전트의 실험이 깨진다.

## 0. 절대 규칙
1. **splits/splits.npz 재생성 금지** — 두 에이전트 모두 이 fold를 사용해야 점수가 비교 가능하다.
2. **공식 나침반 = 누수 없는 sim-only fold0 teacher OOF macro-F1(+bias)**. holdout 절대값은 누수 편향(FULL 멤버가 학습한 데이터)이므로 순위 판단 금지. 최종 심판은 LB뿐.
3. **data/test.jsonl은 5행 데모 스텁**(전부 train 세션). 실제 30k는 서버 전용. 이걸로 점수 내지 말 것.
4. 대용량 아티팩트(멤버 zip, npz)는 커밋 금지(.gitignore 준수). **코드·문서 변경은 즉시 커밋** — 상대가 봐야 한다.
5. 삭제는 자기 네임스페이스만. 상대 파일·공용 파일 삭제 금지.

## 1. GPU 사용 (A6000 1장 공유 — 잠금 필수)
- 학습 시작 전 **`work/GPU_LOCK`** 확인. 존재하면 대기(또는 소유자에게 메모).
- 시작 시: `echo "owner=<claude|codex> tag=<TAG> eta=<종료예상 ISO시각>" > work/GPU_LOCK`
- 종료(성공/실패 불문) 시 **반드시 삭제**. 고아 락(소유 프로세스 없음)은 제거 가능.
- **워처(대기 스크립트)는 GPU_LOCK 존재만 검사할 것 — 프로세스 grep 금지** (07-06 좀비 프로세스로 유휴 대기 사고. 락이 유일한 중재자).
- 추론·평가(짧은 GPU 사용 ≤10분)는 락 없이 가능하되 학습 중이면 자제(시간 측정 오염).

## 2. 네임스페이스
- 태그·파일명 접두사: Claude=`cc_`, codex=`cx_`. 예: `AD_TAG=cx_deberta_f0` → `work/teacher_cx_deberta_f0.npz`, `packages/submit_cx_*.zip`.
- (기존 무접두사 파일은 Claude의 초기 산출물 — 건드리지 말 것.)

## 3. 실험 원장 (공유 리더보드)
- **`work/LEADERBOARD.md`** 에 모든 결과를 기록. 행 추가만, 남의 행 수정 금지.
- 필수 열: recipe / max_len / ep / 특이사항 / **fold0 sim +bias** / 상태.
- FULL 승격(5h) 은 fold0에서 기존 top-2를 이겼을 때만.

## 4. 학습·평가 표준 절차
```bash
# fold0 교사 (honest 측정, ~4h w/ FGM)
PYTHONPATH=/workspace AD_WORK=/workspace/work HF_HUB_OFFLINE=1 \
AD_MODEL=... AD_VERSION=v6 AD_MAXLEN=... AD_EPOCHS=... AD_LR=2e-5 AD_BATCH=32 \
AD_FGM=1 AD_FOLD_LO=0 AD_FOLD_HI=1 AD_TAG=<cc_|cx_>... \
python3.11 action_decision_maximum/src/teacher_cli.py > work/train_<TAG>.log 2>&1

# sim-only fold0 측정 (npz 나온 뒤)
PYTHONPATH=/workspace python3.11 eda/honest_oof_eval.py   # 패턴 참고 — fold0 npz를 sim 마스크로 평가
```
- 파이썬은 **python3.11** (3.10엔 torch 없음). 서버도 3.11 — 파리티 유지.
- xlm-roberta-large는 HF 캐시에 있음. **새 모델 카드 쓰려면 반드시 서버 오프라인 번들 가능 여부부터 확인**.

## 5. 제출 후보 게이트 (전부 통과해야 후보 자격 — 예외 없음)
```bash
python3.11 sim/check_zip.py packages/<pkg>.zip                       # 구조/1GB
python3.11 sim/run_offline_sim.py --model packages/<pkg>/model --script packages/<pkg>/script.py --n 3000
# 타이밍: A6000 30k 추정초 × 3.21(sim/calib.json) ≤ 540s
# parity 스모크: train 4000행 macro가 OOF보다 확실히 높고 14클래스 전부 발화
```
- requirements.txt는 빈 파일 유지. sklearn pickle 금지. 멤버 직렬화 버전은 `::v4/::v6` 명시(침묵 실패 1순위 원인).

## 6. 제출 예산 (10회/일, 운영자가 집행)
- **게이트 통과 후보는 기대LB와 무관하게 전부 패키징·큐 등재**(07-07 운영자 지시 — 로컬≠서버 괴리 대비, LB 데이터 포인트 자체가 가치). 지명 시 `work/SUBMIT_QUEUE.md`에 한 줄 추가:
  `<zip경로> | 소유 | fold0점수 | 기대LB | 근거 1줄`
- LB 회신은 운영자가 같은 파일에 기록 → 두 에이전트 모두 재캘리브레이션.

## 7. 상호 피드백 (자동 토론)
- 라운드 로그는 **DEBATE.md에 append** (R13부터). 형식: 주장 → 근거(실측) → 반박 요청.
- 상대 계획에서 (a) 근거 없는 가정 (b) 낭비 GPU 블록 (c) 게이트 미통과 리스크를 지적할 것. 동의보다 반박이 가치 있다.

## 8. 확립된 사실 (재검증 낭비 금지 — 근거는 explain.md·DEBATE.md)
- 병목 = 탐색군 4클래스(read_file/grep_search/list_directory/glob_pattern) 상호혼동, 오분류의 **92% 학습가능**(모호성 8%뿐).
- FGM+10ep: fold0 honest +0.0076 (0.7569→0.7645). max_len 320은 **25.9%를 절단**(384면 2.5%) — 검증 중.
- 단순 log-prob 평균 앙상블 유효(+0.0058), LightGBM 스태커+약멤버는 유해(-0.014). v6 직렬화가 base엔 +0.018, large엔 중립.
- 순차누수 없음(hidden coverage 0). elapsed/PACE 메타 기각(-0.0082). au 7.2%는 OOD(test는 all-sim 추정).
