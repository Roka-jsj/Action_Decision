# PROJECT.md — Dacon 236694 (AI Agent Action Decision) 종합 기록

> 단일 진실 원천: 설계 근거·파라미터·모델 구조·실험 결과·운영 노하우.
> 갱신 규칙: 매 실험/제출 후 즉시 업데이트. 상세 실험 수치는 `experiments_master.csv`.

## 1. 대회 요약
- **문제**: 에이전트 세션 상태(메타+이력+현재발화) → 다음 action **14클래스** 분류. **지표 = Macro-F1**.
- **형식**: 코드 제출 zip(`model/`+`script.py`+`requirements.txt`). 서버가 30k 비공개 테스트 실행.
- **제약**: ≤1GB / 설치≤10분 / 추론≤10분 / **완전 오프라인** / 10회 제출/일.
- **서버**: Ubuntu 22.04.5, Python 3.11.15, CUDA 12.8, **T4 16GB(fp16만, bf16 불가)**, 3vCPU, 12GB RAM.
  사전설치: torch 2.7.1+cu128, **transformers 4.46.3**, sklearn 1.8.0, pandas 2.0.3, numpy 1.26.4.
- **일정**: 예선 마감 **2026-07-15 10:00**. 본선 = 예선환산 50% + 추론속도 10% + 전문가심사 40%.
- **LB 현황(07-02)**: #1=0.7798, **top-12 진출선≈0.729**. 우리 1차 제출 0.62851(87등).

## 2. 데이터 & EDA 핵심 (검증된 사실)
- train 70,000 / test 30,000(비공개, 배포본 5). id↔라벨 완전 매칭. 클래스 불균형 완만(8.8×, 최소 web_search 1,273).
- **id 2종**: `sess_sim_*`(64,975) / `sess_au_*`(5,025) — 분포 상이(au는 read_file 26%). visible test=전부 sim. 세션키 = `id.rsplit("-step_",1)[0]` (9,429 세션, 평균 7.4 step).
- **텍스트 지배 문제**: 구조-only 오라클 0.17~0.31 vs prompt-only 일반화 0.435. **직전 action+status가 최강 보조신호(+0.089)**.
- **하드클래스 = 탐색계열** read/grep/glob/list/apply/web (선형 F1 0.13~0.35) — macro-F1 병목. `respond_only` 0.9995, `write_file` 0.99 (거의 해결).
- **respond_only는 history에 절대 등장 안 함**(터미널 행동). history ≤12턴(≤6 action).
- prompt 평균 61자(p99 169); 직렬화 p99 ~770자. **GroupKFold vs random gap ≈ 0**(그래도 GroupKFold 유지).

## 3. 모델 구조 & 파라미터 (현재 확정)
### 직렬화 (common/ad_lib.py `serialize()`)
- **v1**: prompt만 / **v2**: `[CUR] {prompt} [LAST] {action}({ext}) [{status}] {result}` / **v3**: +메타토큰(`[TIER][LANG][TURN][CI][GIT]`)+최근 history 8턴 / **v4**: v3 + `[GEN][BUDGET][LOC][TOPLANG][NOPEN][OPENEXT]` + args 확장자
- `result_status()` 정규화: error/test_fail/test_pass/zero/nonzero_exit/success/na
- **토크나이저 `truncation_side="left"`** (필수! [CUR]가 뒤쪽 → 오래된 history부터 잘림)

### 학습 (action_decision_balance/src/train_cli.py)
- 모델: **xlm-roberta-base** (fp16-safe, 배포용). 분류헤드 init seed 고정(HEAD_SEED=1234, soup 정합).
- CV: 세션단위 **8% 프로즌 홀드아웃 + StratifiedGroupKFold(5)** (splits/splits.npz, 전 실험 공유)
- 손실: class-weighted CE (weight = N/(C·count), 평균1 정규화)
- HP: warmup 6%, weight decay 0.01, linear schedule, fp16 autocast(GradScaler)
- **사전토큰화 1회 → 배치 pad** (에폭당 재토큰화 제거, ~3× 속도)
- env 파라미터: `AD_MODEL, AD_VERSION, AD_MAXLEN, AD_EPOCHS, AD_BATCH, AD_LR, AD_FOLDS`

### 후처리 (common/postproc.py)
- OOF log-prob에 **per-class additive bias** 좌표상승(coarse 3단 grid) → `postproc.json`
- 합성 검증 +0.081, 실측 holdout +0.024. **의사결정은 pooled-OOF 기준**(홀드아웃은 soup/bias 선택에 쓰여 낙관적)

### 추론 (common/server_script.py + model/ad_lib.py)
- `TRANSFORMERS_OFFLINE=1`, `local_files_only=True, use_safetensors=True`
- 길이 정렬 배칭 + fp16(GPU)/fp32(CPU) + **id 조인**으로 submission 작성
- model/ 구성: `model.safetensors`(fp16) + 토크나이저 4종 + `config.json` + `ad_lib.py` + `postproc.json` + `run_meta.json` (**.pt 불필요** — 공식도 파일명 자유, 베이스라인은 .pkl)

## 4. 실험 결과 (요약; 상세=experiments_master.csv)
| 실험 | 설정 | pooled-OOF | LB | 비고 |
|---|---|---|---|---|
| 선형 v1 | TF-IDF+SVC, prompt만 | 0.4346 | - | 기준선 |
| 선형 v2 | +직전action | 0.5237(f0) | - | +0.089 최대 단일 신호 |
| **xlmr_v3_calib** | v3/160/3ep/b128 | **0.6183** | **0.62851** | CV↔LB gap +0.010 ✅ |
| esc1 (진행중) | **v4/256/4ep/LR3e-5** | f0 val **0.6843** | - | 캘리브 대비 +0.056 |

- ~~CV↔LB: pooled-OOF ≈ LB (+0.01)~~ **[폐기 07-04 감사]** 잔차 −0.006~+0.010로 불안정. **유일 예측 앵커 = holdout(스택)+bias − [0.010, 0.018], ±0.004 밴드** (5회 제출 전건 성립). FULL멤버 보너스(+0.003~0.0075)는 n=2라 예측에 미반영, 상방 여유로만 취급.
- 에폭 수렴: 3ep에서도 상승 중 → 4~5ep 유효. batch128이면 LR 3e-5로 보정.
- soup: 수렴 부족 시 greedy가 1개만 선택(효과 없음) → 충분한 epoch 후 재평가.

## 5. 에스컬레이션 사다리 (LB 0.729+ 목표)
1. ✅ 캘리브레이션 (pooled-OOF 0.6183 / LB 0.6285)
2. ✅ **v4+256+4ep+LR3e-5 (esc1)**: pooled-OOF **0.6770**(+0.059), holdout+bias 0.6892 → 제출#2(예상 LB ~0.687)
3. 🔄 v5 vs v4 @320+LLRD 스크리닝 (scr1) → 교사 직렬화 확정
4. **멀티아키텍처 교사**: xlm-r-base×2seed + mdeberta-v3-base + xlm-r-large(폴드분할) — teacher_cli.py로 확률 npz만 수집
5. **OOF 증류 → student 2종 비교**: ① base student(556MB, T4 ~95s) ② **vocab-pruned xlm-r-large student(~750MB, T4 ~290s)** — 임베딩 25만→~7만 토큰 프루닝으로 1GB 안에. holdout 우위 쪽 제출 (정확도50% > 속도10%)
6. **에러분석 루프**: 혼동쌍(read↔grep↔glob↔list) 신호 주입 — +0.02~0.04
7. **최종**: 확정 레시피 **전체 70k 재학습**(distill_cli AD_FULL=1) + bias 재적합 — +0.015~0.025

### ⏱ T4 실측 캘리브레이션 (제출#1 서버 로그)
- **서버 총 소요 59s** (30k, base, maxlen160) ↔ A100 25s → **T4/A100 ≈ 2.4×** (length-sort 배칭 효과)
- 환산: base@320 ≈ 95s / **large@320 ≈ 290s** → large 배포 가능 확정. 속도보너스도 최상위권.

### 💾 1GB 예산 활용
- base fp16=0.56GB(현재), large fp16=1.12GB(초과) → **vocab 프루닝**(임베딩 512→~150MB)으로 large ~0.75GB 진입 가능
- 2모델 앙상블(1.11GB)은 초과 → 증류로 대체

## 5.5 교사·앙상블 실측 학습 (07-02 오후)
- **T1**(base v4@320 s1234 4ep) pooled-OOF **0.6766** / **T2**(s2024) **0.6763** — npz 확보
- **seed 다양성만으론 무의미**: T1+T2 앙상블 0.6774(+0.001). **아키텍처·에폭·목적함수 다양성이 필수**
- **앙상블일수록 bias 효과 큼**: T1+T2 holdout 0.6794 → +bias **0.6922**(+0.013)
- **large 기각**(현 설정): fold0 3ep 0.6697 < base 0.6863 (수렴 느림, ROI 낮음 — 최후 옵션)
- **템플릿/검색 가설 반증**: 세션교차 커버리지 14%, 매칭정확도 0.54 → 검색 하이브리드 폐기
- **에러분석**: 오답 17.9%가 rank-2 (앙상블·후처리 회수 풀); read↔list 최대 혼동원=history없는 세션시작(1,339건); run_tests↔lint 23%, ask_user↔web_search 27%
- **운영 사고 2건**: t3 mdeberta protobuf 크래시(`protobuf==3.20.3`+`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`으로 해결) / t4 npz 미회수 소실(**교훈: 모든 잡에 기동 즉시 babysitter 장착** — sim/babysit.sh)

## 5.6 예선 최고점 전략 확정 (07-02 저녁) — 프루닝 앙상블 배포
- **결정**: 증류 대신 **vocab-pruned 모델 N개를 zip에 직접 실어 추론시 앙상블** (예선=정확도 100%, 속도 무관·10분만 통과)
- 근거: base 556MB 중 임베딩 384MB → 프루닝(250k→49k 토큰)시 **~280MB/모델** → 3모델 840MB<1GB. 앙상블 실측(holdout+bias 0.7053)을 증류 손실 없이 그대로 배포
- **인프라 구현·검증 완료**: `common/vocab_prune.py`(id리매핑 방식, spm 무수정) + `ad_lib.predict_ensemble_probs`(run_meta "ensemble" 키) + 오프라인 시뮬 PASS
- 교사 현황: base_s1234(0.6766)·base_s2024(0.6763)·base_s777e5(0.6840, **5ep**) npz 확보 | **에폭 +0.007/ep** | **다양성>개수**(상관 페어 추가시 하락) | mdeberta 3회 실패로 손절
- 진행: lg2=large 2차probe(LR2e-5, ep1 0.6139 순항) / tk=klue 교사 / 로컬 스태킹 프로브
- **다음**: 판정 후 FULL-70k 재학습 멤버들(base5ep+klue+large?) 프루닝→로컬 패키징→제출#3
- 운영: 세션 수명 44~90분 변동 → teacher_cli **fold별 증분 npz 저장** + babysitter **매폴링 다운로드** 패치 완료

## 5.7 대형교사 확보 작전 (07-02 밤) — 자동 재기동 babysitter
- **probe_large2 돌파**: large 4ep LR2e-5 LLRD batch64 → fold0 **0.7196** (base +0.033). config가 문제였음(LR 1e-5→2e-5)
- **klue 5-fold**: fold 0-3 = 0.681/0.662/0.682/0.686 — 아키텍처 다양성 멤버로 확정
- **lga 사고**: large f1-2 세션이 기동 6분만에 사망(최단 기록), 기존 babysitter는 손실 시 그냥 종료 → **sim/babysit_range.sh** 신설: 로컬 npz의 fold_lo/fold_hi 스캔 → 첫 미확보 fold부터 **자동 재기동**(최대 6회, 슬롯 3분 재시도, 크래시 로그 회수)
- 배치: largeA(folds 1-2)·largeB(folds 3-4, tk 슬롯 대기) 병렬 — npz명 teacher_largeA_a*/largeB_a*
- **npz 병합 규칙**: oof=fold별 배타 행 복사, hold=fold수 가중평균 (eda/stack_eval4.py merge_large)
- **eda/stack_eval4.py** 준비 완료: 조합별 mean vs LightGBM stack 평가. 1GB 감안 배포 조합 후보: large프루닝(~750MB)+klue(~220MB)≈970MB / base계열 3종≈780MB — 성능·용량 트레이드오프는 스택평가 결과로 결정

## 5.8 목표 재상향 (07-02 밤) — LB 인플레이션, 통과선 0.78 추정
- LB 실측(07-02 밤): #1 0.7852, **#12(통과선) 0.7395** — 이틀 전 0.729에서 급상승 중. 사용자 판단: **최종 통과선 ≈ 0.78**
- 우리 위치: LB 0.6709 (86위권). 현 파이프라인(large 5fold+4멤버 스택) 기대치 LB 0.72~0.74 → **부족, +0.04~0.06 추가 필요**
- 에스컬레이션(승인된 방향): ① large 확장 프로브(v5직렬화@384 / 6ep / FGM, fold0 각 ~15분) → ② 승자 조합으로 large 변종 3-4종 교사 함대 → 앙상블 스택 OOF 0.75+ → ③ **OOF 증류 → large-pruned student(750MB)+klue+스태커 = 1GB 내 배포** (증류 손실 -0.005~0.01 감수, 앙상블 상한 돌파용) → ④ 에러분석 루프 2차, au 서브셋 처리 프로브
- 무리수 후보(보류): Qwen3-0.6B vocab프루닝(~0.96GB) — fp16 오버플로 리스크(bf16 체크포인트), T4 추론 ~6.4분. 함대 증류로 0.75 미달 시에만 검토

## 5.9 제출 #3·#4 대성공 (07-02 22:04) — 스택 배포 검증, 38위
- **submit_stack_kl.zip (klue+large+스태커): LB 0.72191, 5분48초, 38위** | submit_stack2.zip (base_e5+klue): LB 0.71490, 3분09초
- **캘리브레이션 법칙 재확증(스택 체제)**: LB = holdout(stack)+bias − 0.018, M2 오차 0.0003 / M1은 예측+0.003 (FULL 멤버가 fold평균 가정보다 강함)
- 시사점: ① T4 타이밍 여유 큼(5:48/10:00) ② 1GB에 3멤버 불가(잔여 110MB) → **상한 돌파는 증류 확정** ③ LB 인플레 가속: #12 컷 0.7395→**0.74488**(당일 +0.005). 0.78 목표 유지
- 목표 환산: 컷 0.745 → holdout+bias 0.763 필요 / 0.78 → **0.798 필요** (현 4멤버 스택 상한 측정중, 함대+증류 필수)

## 5.11 야간 결과 (07-03 04:30 기준) — 6ep 함대 4/5 + 유닛 소진
- **large-6ep folds 0-3 확보**: 0.7399/0.7253/0.7425/0.7246 (4ep 대비 평균 **+0.029**) | fold4는 유닛 소진으로 미완
- **FULL-6ep 배포 멤버 확보** (member_largefull6.zip 661MB) — M3 주력
- 세션 대량 사망의 밤: klue6ep 2연속 조기사망(0 fold, 컷 처리) / lg6b·lg6c 각 1회 사망(증분 npz로 fold 손실 0) → babysit 자동재기동이 전부 회수
- **04:10 유닛 소진 확진**: "Backend rejected accelerator A100(no entitlement)" — 8ep 프로브·fold4 미실행. **충전 필요**
- 플랜B: FOLDS=0123 부분커버 스태커(fit_stacker에 FOLDS env 추가)로 M3 진행 — 커버 fold 행만 meta CV/bias
- 에러분석 2차(eda/error2_report.md): 탐색계열 혼동 클러스터(read/grep/list/glob 상호 ~9천건), 세션시작(hist=0) err 43.9%, au err 44.1%(sim 28.6%), **rank-2 정답 57%**(스택/증류 회복 여지 큼), 6ep는 전 클래스 고른 개선
- 충전 후 우선순위: ① lg6c fold4(3u)→5fold 스태커 재적합 ② 8ep 프로브(4.3u) ③ 추세 유지 시 8ep 함대+FULL-8ep ④ v6 직렬화 프로브(경로/심볼/글로브/디렉터리 플래그) ⑤ au제외 프로브 ⑥ 메가스택 증류
- **M3 완성 (05:00)**: packages/submit_stack_kl6.zip 0.886GB, 오프라인 시뮬 PASS, **제출만 하면 됨**. [klue,large6] 스택 holdout+bias **0.7505**(folds0-3 부분커버), **LB예측 0.7325±0.003** (T4 ~6분 예상, M2와 동일 구성). base_e5+large6은 0.7475로 차점
- **M3 LB 실측 (07-03 12:49): 0.74000, 5분33초** — 예측 대비 **+0.0075** (FULL 멤버가 fold평균 가정보다 강함, M1에서도 +0.003 동일 방향). 보정: 스택 배포 LB ∈ [holdout+bias−0.018, −0.010], FULL멤버 강세 편향. 이틀 누적 0.6709→0.7400(+0.069), 컷(0.745 어제밤 기준) 버블권
- v6 직렬화: LinearSVC 스크리닝 **+0.026**(glob+0.060 grep+0.028 read+0.023 apply_patch+0.075) → T4 무료폴백에서 base-v6 fold0 GPU 프로브 진행중 (기준선 v4 0.6862)

## 5.12 프로브 결론 & 함대 준비 완료 (07-03 오후)
- **v6 GPU 프로브 확정: base fold0 0.7040 (v4 0.6862 대비 +0.018)** — [SEQ]트레일+[PFLAG]+[NOHIST], 좌측절단 생존배치. v6 = 함대 표준
- **au-제외 기각**: 통제비교 sim-only 0.7197 vs 0.7179(+0.0018, 노이즈) & 하방리스크 비대칭 → au 유지
- **LB 인플레**: 26위(0.73999), #12 컷 0.76376 (하루 +0.019). 0.78 목표 재확인
- **sim/launch_fleet_v6.sh 준비완료** (충전 시 원커맨드): 슬롯A largev6A(f0-2)→FULL멤버(6/8ep 자동결정, v6, 프루닝) | 슬롯B 8ep프로브→largev6B(f3-4). babysit_full에 VER 파라미터 추가됨
- **Kaggle 트랙**: CLI(KGAT Bearer=KAGGLE_API_TOKEN env) + 데이터셋(자동해제→ /kaggle/input/datasets/ 하위 마운트, 재귀탐색 필수) + 커널 v4 정상이나 **계정 전화인증 없으면 GPU/인터넷 차단** → 인증 대기. 커널: tistmesp03/ad-kluev6-teacher, ad-basev6e5-teacher (klue-v6-6ep / base-v6-5ep, T4x2 fold분할)

## 5.10 야간 일시정지 & 재개 체크리스트 (07-02 심야 작성)
- **정지 계획**: lg6b(f1-2)→largefull6(FULL 6ep 멤버) / klue6ep→lg6c(f3-4) 완료(~00:40) 후 전면 정지. 이후 신규 작업 금지(사용자 PC 종료).
- **재개 시 순서 (07-03)**:
  1. `eda/fit_stacker.py artifacts/stack_kl6 klue6 large6` — 단, load_member에 klue6(teacher_klue6ep_a*)·large6(teacher_large6ep_a1 f0 + lg6b f1-2 + lg6c f3-4 병합) 키 추가 필요
  2. 조합 비교: [klue6,large6] vs [klue,large6] vs [base_e5,large6] → 최고 조합으로 M3
  3. `sim/package_ensemble.py --out submit_stack_kl6 --stacker <승자> --member <klue6 or kluefull.zip> --member member_largefull6.zip` → 오프라인 시뮬 → 제출 (LB예측 ~0.74)
  4. 유닛 충전 확인 → 8ep 프로브(fold0, ~3.7u) → 추세 지속 시 8ep 함대 재구축 + FULL-8ep
  5. 병행: 에러분석 2차, FGM 프로브, 메가스택(large6+large4+klue6+base_e5) 증류 검토
- **핵심 수치 암기**: LB법칙 = holdout(stack)+bias − 0.018 (오차 ±0.003) | 에폭 스케일링 large +0.010/ep (4→6ep 실증) | 약멤버 추가 무의미(4멤버 +0.0014) | 컷 0.745↑, 목표 0.78 = holdout+bias 0.798
- ⚠️ klue6ep FULL 멤버는 아직 없음 — M3에서 klue6 교사가 klue보다 좋으면 FULL-klue6ep도 학습 필요(~2u)

## 5.13 순차누출 가설 검증·기각 (07-04 오전) — 예선=순수 모델 싸움 확정
- 발견: train은 세션×스텝 구조, **step N 라벨 = step N+1 history 마지막 action** (22,587쌍 전수 100%, 교차검증 231,664건 충돌 0). 공개 test 5건은 train과 세션 공유(전부 조회 가능)
- 구현: ad_lib posmap 직독(3중 폴백: train지도 > test내부 관측 > 모델), 모의 히든테스트 커버리지 86.6%·정확도 100%
- **LB 실측 0.7399998302 = M3와 완전 동일 → 히든테스트 커버리지 0 확정**: 세션 비공유 + 내부 누출 없음(세션당 1스텝 × 30k 추정). 공개 5건은 train 발췌 placeholder였음
- 교훈: 폴백 설계 덕에 무손실 검증(제출 1회). 이후 패키지에서 posmap 제외(코드검증 시비 예방). **0.78+는 모델 품질로만 달성 가능** → large-v6 함대·8ep·증류가 유일 경로

## 5.14 전략 감사 결과 (07-04 오전, 3렌즈 독립감사) — 기대치 하향·게이트 신설
- **기대치 실측 보정**: 스택 이득 +0.01~0.02 → **+0.005~0.01** (스택 pooled-OOF가 최강멤버 fold평균보다 낮았음; LB 이득은 FULL재학습+bias 포함 +0.007) | 에러루프 v7 +0.02~0.04 → **+0.005~0.015** (v6가 이미 혼동클러스터 수확, 이중계상 제거) | 증류 손실 −0.005~0.01은 **실측 0건 가정** → 착수 전 저비용 프로브 필수
- **0.8 산술은 현 레버로 닫히지 않음** (현실 상한 ≈0.77~0.78). 컷 +0.016/일 추세 → 최종컷 0.78~0.80 개연. **7/8 하드게이트**: 함대+증류 상한이 예상컷 미달이면 Qwen 조기 착수
- **Qwen 함정 발견**: 서버 transformers 4.46.3은 qwen3 미지원(4.51+) → **Qwen2.5-0.5B로 대체** 또는 wheel 벤더링 선검증 필요. fp16 오버플로 검증은 Kaggle 무료로 선행
- **침묵실패 게이트 신설**: ① sim/parity_check.py(홀드아웃 300행 엔드투엔드, 순서교체·버전오지정 검출) ② fit_stacker fold-정체성 assert 수정 ③ 배포 ad_lib에서 posmap 블록 자동 스트립 ④ check_zip 앙상블 대응(TODO) ⑤ 멤버 train_meta.json 마커(다음 학습분부터)
- **무료 레버 발굴**: v6@320 절단 25.8% → @384는 2.4% (Kaggle 프로브 발진) | 스태커에 margin/entropy·해시 n-gram 피처 부재 | bias의 sim-only 적합 미검토(visible test 전부 sim) | FGM 구현완료·미실행 | 증류 T=1 나이브(temperature·스태커타깃 미구현)
- 판정 규칙 명문화: **holdout Δ<0.005 = 무승부** (pooled-OOF·sim-only로 타이브레이크)

## 5.15 재개 상태 스냅샷 (07-04 14:05, 노트북 2h 오프) ⚡최우선 읽기
**확보된 교사 npz (전부 splits.npz 공유, 디스크 안전):**
- `teacher_largev6[AB]_a*` = **large-v6-6ep 5fold 완비** [0.7391 0.7318 0.746 0.7386 0.733] 평균 **0.7377** ← 최강
- `teacher_large8v6_a1` = 8ep fold0 **0.7456** (8ep 채택, but 5fold 미완)
- `teacher_kluev6_g*` = klue-v6 (Kaggle, 재개시 회수 필요 — 커널 RUNNING이었음)
- `teacher_basev6e5_g0` = base-v6-5ep 5fold [0.707 0.699 0.707 0.701 0.705]
- `teacher_b384_g0` = v6@384 기각용 (0.7001 < @320)

**확보된 FULL 배포 멤버 (v4 직렬화!):** member_largefull6(v4 6ep large 661MB), member_kluefull(v4), member_basee5full(v4)
**미확보 (재개시 GPU 필요):** ⚠️ **member_largefullv6.zip (v6 8ep large) — M5 제출의 핵심, ~40분 A100** | member_basev6e5full (4회 실패)

**재개 즉시 실행 순서 (07-04 오후):**
1. Kaggle kluev6 회수: `kaggle kernels output tistmesp03/ad-kluev6-teacher` → teacher_kluev6_g*.npz
2. **member_largev6 FULL 학습** (최우선): `bash sim/babysit_full.sh largefullv6 xlm-roberta-large 8 2e-5 64 1 v6` (8ep, v6, 프루닝)
3. 병행 로컬(무료): `FOLDS=01234 python3 eda/fit_stacker.py artifacts/stack_b6lv6 basev6e5 largev6` + `... stack_klv6 klue largev6` (largev6 5fold 완비됐으니 full CV)
4. member 도착 → **M5 패키징** `python3 sim/package_ensemble.py --out submit_M5 --stacker artifacts/stack_klv6 --member member_kluefull.zip::v4 --member member_largefullv6.zip::v6 ...` ⚠️멤버별 ::버전 필수!
5. **parity_check + 오프라인시뮬** 후 제출 (LB예측 0.755~0.765)
- ⚠️ 함대는 **max_len=320 확정**(384 기각). 8ep 채택이나 FULL 멤버도 8ep로.
- ⚠️ 스태커/멤버 **직렬화 버전 반드시 명시**(largev6=v6, klue/base계열 기존=v4). parity_check.py로 검증 필수.

## 5.16 M5 돌파 — 캘리브레이션 법칙 반전 (07-04 저녁) ⚡전략전환
- **submit_M5 (klue-v4 + largev6-8ep 2멤버): LB 0.76470, 22위** — 예측 0.732~0.740을 **+0.025 대폭 상회**
- **캘리브레이션 법칙 반전**: holdout+bias 0.7499는 **6ep 교사 OOF** 기반 → 배포는 **8ep FULL-70k 멤버**(교사보다 강함). 즉 **OOF-holdout은 실제 배포성능의 하한**. 새 관계식: FULL-8ep 배포 시 **LB ≈ holdout+bias(OOF) + 0.010~0.015** (n=1, 신중)
- **함의**: ① 컷(0.772)까지 **+0.007**만 남음(어제 +0.03 아님) ② 3멤버 스택(holdout 0.7531)을 배포하면 같은 논리로 **~0.78+ 예상** ③ 유일 장벽 = **1GB 용량** (사용자 지적 정확: 채점 6분/10분 = 시간 여유 4분, 병목은 시간 아닌 용량)
- **3멤버 용량 실측**: klue 204 + basev6e5 236 + largev6 661 = **1.102GB** → **110MB만 감량하면 1GB 진입**
- **최우선 과제**: 3멤버를 1GB에 — ① large 공격적 프루닝(50k→30k토큰, ~40MB) + klue/base 추가프루닝 ② 또는 int8 양자화 ③ 또는 증류(3멤버→단일 large student 661MB + klue = 865MB)
- 남은 시간여유 4분 → TTA(다중 max_len 평균)로 정확도 전환 가능
- **codex(GPT-5.5 xhigh) 자동토론 도입**: VS Code 확장 번들 `bin/linux-x86_64/codex` + CODEX_HOME=/mnt/c/Users/vaseb/.codex → WSL에서 직접 호출. 계획 반전 검증용

## 5.17 codex(GPT-5.5) 자동토론 2R → 전략 확정 (07-04 저녁)
- **캘리브레이션 정밀보정**: M5 lift = +0.0148 (0.7647−0.7499), 새 식 **LB ≈ holdout+bias + 0.012~0.018** (이전 −0.010~0.018은 폐기; FULL-8ep 배포 기준). bias는 meta-OOF 적합·holdout 별도 → 오염 아님(검증)
- **large 배포상한 재추정**: local 0.755~0.765 → LB 0.775~0.783. **0.80은 캘리브레이션만으론 불가** — 새 신호(reranking/템플릿/직교모델) 필수
- **KD 강등**: 같은 70k OOF 증류 = 정규화(+0.002~0.008), T=1은 0/음수. 최후순위
- **Qwen 강등**: 배포 제외(느림/fp16위험/직교성낮음), teacher 후보만
- **확정 제출 로드맵** (codex 합의):
  - **M6 = 3멤버 FULL stack** (large-8ep + klue + base) — 단 **base 포함시 1.10GB 초과** → base를 n-gram(~20MB)로 대체가 실질 M6. 기대 LB 0.767~0.771
  - **M7 = large + klue + n-gram(TF-IDF LogReg)** — n-gram 직교신호면 **LB 0.772~0.778, 컷돌파 1순위**. 885MB로 1GB 적합
  - **M8 = top-2 gated reranker** (M7 위, margin작은 샘플만 flip, OOF net +0.003 이상일때만). reranker는 stacker 다음(threshold 과적합 방지)
- **vocab prune 50k→union**: 재프루닝만으론 무효(토큰화 같으면 출력동일), 재학습 필요 → 우선순위 낮음, 1-fold 검증 선행(8ep fold0 0.7456 대비 +0.004↑면 채택)
- **n-gram 배포**: sklearn pickle 금지 → HashingVectorizer(무상태) 또는 LogReg coef를 numpy로 ship + 순수 python transform

## 5.18 M6 플래토 + codex 4R 토론 수렴 (07-04 밤) — 전략 재정립
- **M6 = LB 0.76639 (23위)**, M5 대비 +0.0017뿐. n-gram holdout +0.0083 → LB +0.0017 (**전이율 20%**)
- **offset 추이**: v4멤버 LB−holdout −0.010~0.018 / v6-8ep FULL +0.0148(M5)→+0.0082(M6). holdout+bias는 신뢰할 LB 예측자 아님
- **me⇄codex 4라운드 반박 수렴** (핵심 진단):
  - 나: "20% 전이 = 과최적화 아님, large가 직교신호 흡수" → codex 수용·정밀화: **"6ep proxy 기준으로만 직교처럼 보인 신호가 8ep large에서 collinear"**
  - 함의: **신규 멤버/reranker/hist=0는 6ep OOF가 아니라 8ep-large 잔차 기준으로만 검증**. large 강화보다 후순위로 강등
  - 나: "large 변종은 스태커 없이 fold0 raw로 비교하면 깨끗" → codex 수용
- **확정 우선순위**:
  1. **large-only LB 앵커** (진단): submit_largeonly.zip(0.661GB, largev6-8ep+bias) — 앙상블이 실제 LB로 뭘 더하는지 분해. large단독≈M6면 멤버는 장식/흡수 / M6>large단독이면 멤버 유효 / M6<large단독이면 스태커가 해침
  2. **large 자체 강화** (fold0 raw 0.7456 기준): 10ep+SWA(ep8/9/10) +0.003~0.006 > 10ep raw > FGM +0.001~0.003 > focal/logit-adjust > max_len↑ > vocab-union
  3. hist=0 specialist: 테스트 hist=0 비율 14% → 손익분기 subset delta **≥+0.015** (미만이면 noise에 묻힘)
- **판정룰 개정**: 신규후보 holdout +0.003↑여도 **그 이득의 50%↑가 raw model 단계(스태커 이전)에서 보여야** 제출. ngram/stacker/bias 계열 holdout gain은 **×0.2 할인**

## 5.19 강모델 앙상블 + int8 양자화 (07-05 심야)
- **R7 실측 종결**: 탐색4class에서 large+ngram 확률평균 순증분 +0.0007 → **large가 텍스트 신호 전부 흡수**. reranker/specialist 영구 기각. codex R8 수용.
- **codex R8 천장 진단**: 텍스트 ceiling 0.790~0.800 / 메타 정보원 진짜면 0.800~0.815 / 0.82+는 새 신호 없으면 불가. 배분 70% 함대 / 30% 메타 프로브.
- **코드 검증으로 codex 오류 수정**: "직전 행동 미사용" 틀림(v6 [SEQ]). 진짜 미사용 = elapsed_session_sec(직렬화 전무), raw 수치(bin만), 전체 세션 행동카운트([SEQ]는 최근12 순서만). → eda/meta_probe.py (LGB, 스케일붕괴 원천차단).
- **1GB 킬러 발견**: v6-8ep+v4-8ep 함대 앙상블 = 661×2 = 1.32GB > 1GB. R7 합의에 용량계산 누락.
- **해결 = int8 weight-only 양자화** (sim/quantize_member.py + ad_lib._maybe_dequant):
  - 2D 가중치(≥1e6 원소) group-64 per-group scale int8, 나머지 fp16. 661MB→**353MB**.
  - 배포시 로드 직전 npz→safetensors(fp16) 1회 복원 → 기존 from_pretrained 경로 무변경.
  - 패리티(48표본 CPU): mean|Δp|=0.0005, max|Δp|=0.025, argmax 47/48(뒤집힌 1건=경계표본). 최종판정은 T4 holdout 5.8k 델타로.
- **앙상블 OOF 선판정(klue 희석 재발 방지)**: v6+v4mix 확률평균 — MEAN +0.0019, **가중 w_v6=0.65 +0.0034** (0.7468→0.7502). → ad_lib/package_ensemble에 weights 지원 추가.
- **v4 teacher fold0 구멍**: 6ep(f0-3)+4ep(f4) 혼합 teacher_largev4mix.npz (pooled 0.7254)로 해결.
- **v4-8ep FULL 확보** (attempt2, 8ep 55분 완주. attempt1은 ep7 pruning중 사망).
- **산출물**: packages/submit_str2q8.zip (0.707GB, m1=v6-8ep-q8 w0.65 + m2=v4-8ep-q8 w0.35, bias=가중OOF적합, check_zip PASS).
- **신규 게이트 도구**: make_holdout_test.py(ho::5.8k+필러 30k) + score_holdout.py + bench_t4_hold.sh (T4 타이밍+holdout 채점, 25MB 청크 업로드 — colab contents API ~30MB 한도).

## 5.20 재개 스냅샷 (07-05 11:00, PC-off 12:30) ⚡최우선 읽기
**LB: tri_cond = 0.78266 (신최고, 7분07초). lgb8 0.78226, str2q8 0.78189도 은행. 5위 0.78429 추격. D-10.**

### 오전 대반전 (R11~R12, DEBATE.md)
1. **holdout 게이트 누수편향 판명**: lgb8 holdout -0.0024 "기각"이 실제 LB **+0.0017 (0.78226 최고)**. FULL멤버가 holdout을 학습했으므로 large비중↑=암기 과대평가, base 섞이면 과소평가. → **교차-계열 비교는 LB만. holdout은 동일계열/파이프라인 검증용.**
2. **서버가 Colab T4보다 빠름**: 2-large가 서버 8분07초 통과(Colab 측정 719초는 과대). **서버 시간예산(채점시간 역산): large +230초, base +117초, 기본 ~4분17초.** full 3-way(large2+base)는 ~10분04초로 캡 초과 위험.
3. **조건부 앙상블 구현·검증**(ad_lib.predict_conditional_probs): full 멤버 혼합→저마진 행만 조건부 멤버 재추론. OOF: 3-way full 0.7505(w 0.6/0.15/0.25), cond th=0.5(35% 선택) 0.7501, 서버추정 ~7분40초.
4. batch256는 서버에서 오히려 느림(str2q8b 8분56초) → **batch128 고정.**

### 즉시 제출 후보 (조립완료)
- **packages/submit_tri_cond.zip (0.943GB)**: m1=large-v6-8ep-q8(w0.6) + m2=base-v6-e5(w0.15) 전체 + m3=large-v4-8ep-q8(w0.25) 저마진 35%만. cond-bias 동일규칙 적합. CPU 스모크 검증 (완료 확인 후 제출). 기대 LB ~0.783±0.001, 서버 ~7분40초.

### 클라우드에서 돌아가는 것 (PC 무관)
- **Kaggle 커널 `tistmesp03/ad-full2-s2-dist`** (~14시경 완료 예상): (a) largev6s2 = seed2024+SWA3 v6-8ep FULL (soup 재료), (b) largev6dist = teacher(0.65·v6+0.35·v4) soft0.3 T=2 증류+SWA3. 상태: `KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) kaggle kernels status tistmesp03/ad-full2-s2-dist`, 회수: `kaggle kernels output tistmesp03/ad-full2-s2-dist -p <dir>`.
- Colab: A100·T4 모두 유닛 소진(확보 실패 지속). 사용자 잔량 확인/충전 필요.

### 재개 시 실행 순서
1. (안 했으면) submit_tri_cond.zip 제출 → 채점시간·점수로 조건부 모드 서버 검증.
2. Kaggle output 회수 → member_largev6s2.zip / member_largev6dist.zip → experiments/.
3. soup: `python3 sim/soup_members.py --out .../member_soup12.zip .../member_largefullv6.zip .../member_largev6s2.zip` → 양자화 → package_single(bias=teacher_largev6[AB]) → **LB로 직접 판정**(holdout은 계열편향, 동일계열 soup는 참고만). dist도 동일.
4. 승자 조합으로 tri_cond의 m1 교체(soup가 v6단독을 이기면) → 재조립 → 제출.
5. 백로그: pairwise post-hoc(list P=0.384 저마진 스위치, crossfit), sim-only FULL, 3-way soup.
6. 제출 캘린더: ~7/13 실험, 7/14 저녁 final 동결, 7/15 오전 제출 회피. 하루 10회 슬롯 아끼지 말 것(저위험 후보 즉시 은행).

## 5.21 연구실 서버 전환 + R13 체제 (07-05 저녁) ⚡최우선 읽기
- **컴퓨트 전환**: Colab/Kaggle 은퇴 → A6000 48GB 무제한(mun-jtrain) + 오프라인 검증 컨테이너(mun-jtest, 12g/3cpu, network none). `sim/train_and_verify.sh <TAG> <SEED>` 원커맨드 전자동(학습→패키징→check_zip→오프라인 30k→게이트+holdout). calib ratio **2.62**(A6000→서버). SERVER_SETUP.md 참조.
- **기준선 이식 검증**: largeonly holdout **0.78113** — T4 실측과 완전 일치. 동일계열 delta 비교 유효.
- **검증 큐 규율**: 학습 공존 시 VRAM 게이트 오염(41.5GB 실측) — holdout은 공존 OK, **시간·VRAM 게이트는 GPU 유휴시에만**. 최종 배포후보만 유휴 재측정.
- **R13 합의(DEBATE.md)**: soup/s2 단독 LB 발사 금지(기준=tri_cond 0.78266, holdout ≥0.7833 아니면 m1 후보로만) / sim-only>dist 순위 / dist teacher는 tri_cond 3-way 재생성 / base+·v4+ weight probe는 m1 교체 후 보류 / 가중 soup 5점 비교(s1·0.25·0.5·0.75·s2).
- **신규 도구**: soup_members.py `--w` 가중 soup / train_full_cli `AD_SIMONLY=1`(au 제외) / sim/gen_soft_labels_3way.py(3-way teacher) / `.claude/settings.json`(docker 권한+DOCKER_API_VERSION — 컨테이너 재생성시 자동 복원).
- **m1 교체 제약**: FULL soup엔 OOF 없음 → cond-bias/th 재적합 불가, 기존 teacher-OOF bias 재사용이 원칙.
- **자동 체인 가동**(work/r13_chain.sh): s2(2024,SWA3) → 가중 soup 3종 검증 → s3(777) → sim-only. Kaggle 커널(ad-full2-s2-dist)은 회수 불필요(로컬 재학습 대체).

## 5.22 R15~R16 스냅샷 (07-06 아침) ⚡최우선 읽기
- **LB 은행 최고 tri_cond 0.78266** (m1=largev6-8ep-q8 w0.6 + m2=basev6e5 w0.15 전체 + m3=largev4-8ep-q8 w0.25 저마진th0.5 조건부). 5위 0.78429까지 +0.0016.
- **[GEN] 토큰 = id 파생 판명 → World C 확정** (v6n no-GEN 재학습 LB 0.7337 폭락 -0.047). 함의: 테스트 id에 sess_au 표식 존재, [GEN] 필수, au는 이미 정상처리. **no-GEN·soup·SWA·sim-only·dist 전부 사망.**
- **유일 남은 레버 = 듀얼 bias**(id prefix로 au행에 bias_au 라우팅, teacher-OOF 프록시 +0.00208). 
- **즉시 실행 (제출 대기 중)**:
  1. 🔴 `packages/submit_largeonly_dual.zip` 제출 → 듀얼 LB 전이 확인 (0.78051→~0.7825 예상, 확인용). 
  2. 성공 시: q8 재양자화(진행중 member_largefullv6_q8·largev4_8ep_q8) → `package_ensemble --dual 1 --margin_th 0.5 --weights 0.6,0.15,0.25 --member ...q8 --member basev6e5 --member ...v4q8 --bias "<v6teacher>,<basev6teacher>,<v4teacher>"` → tri_cond_dual (~0.7847=5위) → 오프라인검증 → 제출.
  3. λ/shrink 미세조정 1~2발. 실패 시 au감지기(AUC 0.974, th0.99) 라우팅 폴백.
- **신 하네스 규율**: anon 모드(id 익명화)로 World A 재현 가능 / 검증셋 id 원본보존(prefix 경로 실전동일) / calib 2.62 / holdout 절대치는 FULL 암기팽창 — 동일계열 delta·fold0 정직치·LB만 신뢰.
- **도구**: eda/au_detector.py, eda/dual_lambda_sweep.py, eda/simbias_probe.py, package_single/ensemble --dual, train_full_cli AD_SIMONLY, sim/gen_soft_labels_3way.py. 전부 jeong 브랜치.

## 5.23 재개 스냅샷 (07-08, GPU NVML다운→docker restart 후) ⚡최우선 읽기
**브랜치 = jeong (main 아님). 목표 재설정: 0.80(1~2%) 추격 접고 → top12 컷 방어.**

### 팀 현황 (같은 계정·쿼터공유, 조원=별도 Claude on mun-train/GPU1)
- **팀 최고 = 조원 tri_v4new 0.78449 (현 14위, 컷=top12 밖).** tri_cond의 m3(저마진25% 조건부)를 v4-s777 large로 교체(다양성법칙). 조원 dev/cx 브랜치에 상세.
- **내(jeong) 최고 = retr_v6 보수 0.78096** (largeonly 0.78051 +0.00045, 내 유일 LB양성). 내 tri_cond=0.78266.
- 0.79+ 20팀·0.80+ 존재(사용자 확인). +0.0155 격차 = 증분 불가.

### 핵심 진단 (이번 세션 최대 성과)
- **히든 테스트 = train에 OOD 확정**: 홀드아웃→train near-dup retrieval +0.0233 vs 히든→train +0.0005 (46배차). **모든 train기반 실험이 히든서 반전하는 근본원인.** LB만이 진실(holdout·fold-OOF 방향성 없음).
- FULL-8ep가 fold-6ep보다 +0.041(비정상 큼) — 히든이 train-OOF보다 쉬울 가능성(미해결).

### 사망 확정 (재시도 금지)
no-GEN(-0.047) / v9 rich(-0.011) / histdrop 증강(-0.011, 오염) / v8 [GEN]꼬리(-0.0023) / retr_mid(게이트18%, -0.0004) / soup(V자) / sim-only / dual-bias / klue·ngram·reranker(조원) / 메타v7 / SWA3(조원은 SWA2 사용).

### 살아있는 축 (R30 개정, 07-08 밤 — 상세 DEBATE.md R29~R30)
**⚠ R33: 신규 제안은 STRATEGY.md의 M1~M5 필터 통과 필수 (실패 5메커니즘·성공 4메커니즘 통일이론).**
목표 공식 전환: **현실 0.786~0.792 + 등수방어 최적화** (0.80은 대형 신규축 발견 시 상방). 조원과 자료공유 없음(쿼터만).
1. **confusion-pair specialist (새 1순위, +0.0015~0.004)**: E={read,grep,list,glob} 4-way 전용모델. 개입셋(base pred∈E & margin<0.1)=5.4%·base acc 0.399·라우팅순도 0.991 실측 — spec이 50%만 맞혀도 +0.003. fold0 클린 프로브 = `spec_ft_cli.py`(작성완료, foldckpt_largev6_f0ckpt_f0에서 4way 헤드 FT). GO문턱 spec≥base+0.05.
2. **sb-4th 멤버**: largev6sbwt FULL FT(id_map 리매핑 수정판). 게이트 판정 = `sim/eval_sb4_gate.py`(ΔF1≥+0.0010/changed 0.4~1.2%/ΔnnQ1≥-0.0010/≤560s). 통과시에만 제출.
3. 그라인딩(시드/증류/조건부 재조합) 24발 / specialist 16발 / sb 8발 / 이종백본 5발 / 예비 7발. 동결 7/13 22:00.

### 종결 확정 (R28~R30 실측, 재시도 금지 추가분)
- **prior/transductive 축 전체**: blind EM(em075 LB -0.00093) + 정답앵커 count-matching(오프라인 -0.0009) + 프로브 3발(히든 prior≈train, λ_au 0.130) 3중증거. **원리: macro-F1 최적 bias는 저정밀 클래스 의도적 과다예측 — 카운트매칭은 해악.**
- **retrieval→tri**: p384 LB 0.78261(-0.00005 평탄).
- **test-test 클러스터 일관성**: 이웃 라벨일치 69% 천장 → 최대 +0.0015 (eda/testtest_cluster_sim.py).
- session-balanced 단독배포(fold0 nnQ1 하락) — 멤버 후보로만.

### 즉시 실행 순서 (재개 시)
1. `nvidia-smi` GPU 확인 → work/sbwt_full.log(sb-FULL FT)·게이트 판정 상태 확인.
2. specialist fold0 프로브: `PYTHONPATH=$R python3 action_decision_maximum/src/spec_ft_cli.py` (~25분) → GO시 5fold+FULL 확장.
3. sb-4th: `python3 sim/eval_sb4_gate.py work/raw_largev6sbwt` → PASS시 패키징·T4검증·제출.
4. 매 결정 codex 토론(work/rNN_brief 절차, R31부터) — 판정 DEBATE.md 기록.

### 운영 필수
- 브랜치 jeong. 커밋 push: DNS 자주 끊김 → `.claude/settings.json` PreToolUse hook이 git/curl 전 8.8.8.8 자동추가. 수동시 `echo 'nameserver 8.8.8.8'>>/etc/resolv.conf`.
- docker: `export DOCKER_API_VERSION=1.43`. 검증=mun-jtest(오프라인 T4재현). `bash sim/train_and_verify.sh` 또는 수동 package_single+prep_verify+docker exec.
- codex: `CODEX_HOME=/root/.codex /root/.vscode-server/extensions/openai.chatgpt-*/bin/linux-x86_64/codex exec --sandbox read-only --skip-git-repo-check --ephemeral -C /root/Action_Decision -c model_reasoning_effort='"xhigh"' --color never - < brief.md`. bwrap로 파일 못읽음→자료 인라인 필수. `pgrep -f "[c]odex exec"`(브래킷). 커맨드 `/codex`·`/labstatus` 있음(.claude/commands).
- ⚠️ pkill은 브래킷패턴(`[t]eacher_cli`), kill/재발사 분리(자기매칭 자살 방지).
- 조원: **제출쿼터만 공유(하루10). 자료·대화·산출물 공유 일절 없음** — 조원 의존 계획 금지(codex 착각 교정 필수).

## 5.24 R27 session-balanced FT 결과 (07-08) ⚡재개 시 반영
- **구현 완료**: `teacher_cli.py`에 `AD_SESSION_BALANCED=weight|sample`, `AD_INIT_FROM`, `AD_SAVE_WEIGHTS` 추가. `train_full_cli.py`에도 `AD_SESSION_BALANCED`/`AD_INIT_FROM` 이식(게이트 통과시 FULL FT 가능). `sim/eval_stress.py` 신규: fold0-val을 overall/sim/NN사분위/session길이/hist0로 평가.
- **fold0 기준점**: `foldckpt_largev6_f0ckpt_f0` = 0.7485 / sim 0.7424 / nnQ1(low) 0.6155 / sess1-2 0.6314 / hist0 0.4852.
- **weight FT** (`1/session_len`, ckpt에서 LR5e-6 2ep): best epoch1 0.7528 / sim 0.7443. stress Δ = overall +0.0042, sim +0.0019, **nnQ1 -0.0007**, sess1-2 +0.0478, hist0 +0.0124, agree 0.9649.
- **sample FT** (epoch당 세션당 1step, LR5e-6 6ep): best epoch6 0.7508 / sim 0.7421. stress Δ = overall +0.0024, sim -0.0001, **nnQ1 -0.0048**, sess1-2 +0.0502, hist0 +0.0219, agree 0.9701.
- **판정**: R26 사전 게이트(`nnQ1(low)`와 `sess1-2` 동시 상승 + overall 폭락 없음 + agree<0.97) 기준 **불통과**. 짧은 세션/hist0에는 강한 양성이나, 히든 OOD proxy로 둔 low-NN이 하락. **FULL 학습/LB canary 보류**. 제출 슬롯 부족으로 **LB prior probe는 제출하지 않음**. weight-FT는 tri 결합 재료로만 보관하고, 이후 계획은 조원 자산 없이 우리 자체 구조/파라미터 로그 조합으로 갱신한다.
- **운영 수정(사용자 지시)**: 조원과는 제출만 공유할 뿐 자료 공유/대화가 없다. 이후 계획에서 조원 의존성 제거. 목표는 우리 보유 구조·파라미터·로그를 조합해 자체 최고 기록을 갱신하는 것.

## 5.25 R28 compact retrieval → 자체 tri_cond 후보 (07-08)
- **문제**: 기존 retrieval pack은 137MB라 `tri_cond`(약0.943GB)에 동봉하면 1GB 초과. 조원 자산 없이 우리 최고 `tri_cond 0.78266` 위에 retrieval을 얹으려면 pack 축소가 필요.
- **구현**: `sim/build_retrieval_pack_compact.py` 신규. `work/emb_v6_70k.npy`를 centered-normalized 후 Gaussian projection 384d로 줄이고 fp16 저장. `work/retrieval_pack_p384` 크기 54.7MB, train top1 agreement 0.725. `common/ad_lib.py`는 `proj.npy`가 있으면 query embedding도 같은 projection으로 변환. `package_single/ensemble`은 `proj.npy` 선택 복사.
- **시간 최적화**: 앙상블 retrieval이 m1 large를 중복 forward하지 않도록 `predict_ensemble_probs`/`predict_conditional_probs`에서 m1 확률과 mean-pool embedding을 동시에 회수.
- **후보 생성**: `packages/submit_tri_cond_retr_p384.zip` = `tri_cond`(w 0.6/0.15/0.25, th0.5) + compact retrieval conservative cfg(`lambda=0.3`, `margin_th=0.3`, `purity=0.7`, `sim_th=0.8943`). 용량 **0.993GB**, `check_zip` PASS.
- **오프라인 30k 검증**: 기준 `submit_tri_cond_rebuild` A6000 212s / VRAM 10868MB / holdout 0.80702. retrieval 후보 A6000 **216s** / VRAM 10964MB / retrieval gate 2.62% / holdout 0.80984. holdout +0.00282는 retrieval pack이 train 전체를 포함해 self-contamination이 있으므로 LB 예측에 쓰지 않는다. 실제 의미 있는 수치: changed 147/30000(0.49%), 시간 +4s, 용량 +50MB.
- **p256 fallback**: `packages/submit_tri_cond_retr_p256.zip`도 생성. 용량 0.976GB, A6000 211s, gate 2.72%, changed 172/30000(0.57%), holdout net +16(p384 net +20). 더 안전하지만 retrieval 보존율이 낮아 2순위.
- **LB 실측**: `submit_tri_cond_retr_p384` = **0.78261**(07-08 14:38, 6분54초). `tri_cond 0.78266` 대비 -0.00005 평탄. largeonly에서 보였던 retrieval 보수 이득(+0.00045)은 tri 위에서 소멸했다. **retrieval→tri 축 종료**.

## 5.26 R29 transductive EM label-shift 후보 (07-08) ⚡현재 1순위
- **동기**: probe는 제출하지 않는다. 대신 zip 추론은 hidden/test 30k 전체 확률을 한 번에 볼 수 있으므로, Saerens EM으로 serve-time prior를 자체 추정해 log-prob를 재가중한다. 구현 위치는 `scores` 산출 후, retrieval 뒤, bias 적용 직전.
- **구현**: `common/ad_lib.py`에 `labelshift_em` 후처리 추가. `sim/package_ensemble.py`/`sim/package_single.py`는 `--labelshift_em <shrink>`와 OOF 평균 `pi_ref`를 `run_meta.json`에 저장한다. 작은 batch는 `min_n=512` 아래에서 skip해 smoke test를 안전하게 둔다.
- **largeonly seed 검증**: `eda/labelshift_em_sim.py` 기준 largeonly는 `em0.5+bias`가 identity -0.0003으로 거의 중립, dir30/dir10 prior-shift에서는 +0.005~+0.020. 단독 제출감은 약해 tri_cond로 재검증.
- **tri_cond holdout-only 시뮬레이션**: 실제 `submit_tri_cond_rebuild` 모델로 holdout 5,810개 확률을 재추론. bias 0.8070 대비 `em0.5+bias` 0.8084(+0.0014), `em0.75+bias` 0.8091(+0.0020). dir100/30/10 및 flat/minority resample 대부분 양수. `em1.0`은 일부 scenario에서 붕괴해 금지.
- **30k 오프라인 검증**: `submit_tri_cond_em05.zip` 0.943GB check PASS, A6000 209s/VRAM10868, holdout 0.80768(+0.00066 vs rebuild). `submit_tri_cond_em075.zip` 0.943GB check PASS, A6000 210s/VRAM10868, holdout **0.80907(+0.00205)**.
- **변경량**: `em075`는 EM pre-bias changed 1.76%, 최종 라벨 changed 373/30000=1.24%. holdout 67/5810=1.15% 변경, fixes 31 / breaks 24 / net +7. class F1은 ask_user +0.0162, plan_task +0.0117, read_file +0.0068가 이득이고 grep_search -0.0057가 비용.
- **판정**: R29 1순위 자체 LB 후보는 **`packages/submit_tri_cond_em075.zip`**. retrieval과 달리 train-near-dup에 기대지 않고 hidden batch 확률만 쓰므로 probe 없이 제출 가능한 구조 축이다. 제출 전 추가 probe 금지, LB 한 발을 쓸지 여부만 결정.

## 6. 컴퓨트 운영 노하우 (colab CLI) ⚠️
- `colab` CLI(google-colab-cli)로 전 과정 자동화: `new/upload/exec/download/stop`. 세션=라이브 커널, exec 간 상태 유지.
- **세션 수명 ≈ 60분** (OAuth 토큰 만료 시 keep-alive 데몬이 갱신 못함 → VM 회수). **모든 작업을 55분 내 완결 + 즉시 다운로드** 설계.
- 학습은 `subprocess.Popen` 디태치 + `train.log` 기록 + **DONE 마커** → 폴링 후 자동 다운로드.
- **GPU**: A100 사용 (L4 대비 4.5×: base/160=3.2분/fold, base/256/4ep=6분/fold). Colab Pro 필수(무료는 사용량 잠금 겪음).
- VM 기본 transformers는 5.x → **학습 스크립트가 4.46.3 설치 후 임포트**(서버 호환 저장).
- 로컬 WSL: GPU 없음, Python 3.10(서버 3.11) → **sklearn pickle 배포 금지**. 오프라인 시뮬은 로컬 CPU로 (`sim/run_offline_sim.py`).

## 7. 파일 맵
```
common/         ad_lib.py(직렬화+추론 단일소스) io_utils cv metrics postproc soup server_script.py
splits/         splits.npz (프로즌 홀드아웃+5fold, 전 실험 공유 — 재생성 금지)
action_decision_balance/src/train_cli.py     # Track A CLI 학습(env 파라미터)
action_decision_maximum/src/train_cli.py     # Track B: 교사 앙상블→OOF 증류(분해 예정)
sim/            run_offline_sim.py(네트차단 검증) check_zip.py make_synth_test.py make_tiny_model.py
eda/            eda_report.md transition_matrix.csv
experiments_master.csv                        # 실험 로그(CV/LB)
ad_common.zip   # Colab 업로드용(common+splits) — common/ 수정 시 재빌드 필수!
submit_balance.zip  # 1차 제출본(LB 0.62851)
```

## 8. 체크리스트 (제출 전 필수)
- [ ] `python3 sim/check_zip.py <zip>` PASS (구조/1GB/토크나이저/safetensors)
- [ ] `python3 sim/run_offline_sim.py --model <dir> --n 0` PASS (네트차단 실행)
- [ ] 30k 추론시간 (A100 실측 ×4~5 = T4 추정 < 600s)
- [ ] pooled-OOF가 기존 best 초과할 때만 제출 (10회/일 절약)
- [ ] 마감(7/15 10:00) 전 best가 LB에 올라가 있는지 확인

## 9. 제출 이력
| # | 일시 | 파일 | pooled-OOF | LB | 등수 |
|---|---|---|---|---|---|
| 1 | 07-02 | submit_balance.zip (v3 calib) | 0.6183 | 0.62851 | 87 |
잔여 제출: 9회(오늘)
