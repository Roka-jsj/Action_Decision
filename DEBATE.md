# 무한 비판 토론 로그 (me ⇄ codex, 데이터 기반)

목표: 0.80 통과가 아니라 **우승(0.82+)**. 매 라운드 실측 근거, 양방향 반박, 결과마다 재점화.

## 확정 사실 (LB 실측)
- large-v6-8ep-FULL + bias = **0.78051 (7위)**. raw(bias없음)=0.77474 → bias +0.0058.
- 스태킹/klue/ngram/reranker/hist0 = 전부 저가치(강 large와 collinear, 희석).
- 캘리브레이션: 6ep OOF holdout → 8ep FULL 배포는 +0.008~0.015 상방(교사<배포).

## 클래스별 헤드룸 (largev6 OOF, macro-F1 0.738)
| class | sup | prec | rec | f1 |
|---|---|---|---|---|
| list_directory | 3952 | **0.384** | 0.690 | 0.493 | ← 과다예측(정밀도 최악) |
| read_file | 8522 | 0.589 | 0.518 | 0.552 |
| grep_search | 9099 | 0.666 | 0.544 | 0.598 |
| ask_user | 2467 | 0.685 | 0.570 | 0.622 |
| lint_or_typecheck | 2061 | 0.577 | 0.711 | 0.637 |
| glob_pattern | 4844 | 0.684 | 0.603 | 0.641 |
| (상위 8클래스: 0.68~0.999, 거의 해결) |
→ **0.78→0.82 = 탐색클러스터(read/grep/list/glob)+ask/lint 전부**. 나머지는 천장.

## 라운드 로그
(R1~R5: 스태킹 폐기·large단독·bias유지 확정. 상세 PROJECT.md 5.17~5.18)

## R6 근거 데이터 (내가 측정)
- **탐색클러스터 = 전체 41.1%** (28782/70000). macro-F1 지배 → 여기가 전부.
- 탐색샘플 중 현재발화에 경로/패턴/디렉 신호 있는 것 **39%뿐**. 61%는 상태만으론 탐색종류 결정 난해(Bayes error 하한 or 신호가 history/args에).
- 키(prompt+직전action+status) 충돌: 표본 작아(416) 비결정적, 순도 0.64. → Bayes error 정밀추정엔 더 강한 프로브 필요.
- 함의: list_directory recall 0.69/prec 0.38 → precision 교정 여지 큼. 61% 무신호 구간이 진짜 상한 결정.

## R6 codex 반박 + R7 쟁점
- codex: H2 과함(list P0.384=남발≠모름), 승부수=raw args/result 탐색 pairwise reranker + list precision. H4 hard gate 위험(top-k일때만 corrector). H5 후순위.
- 내 재반론: reranker 입력(path/glob/args-ext)이 **이미 v6 [PFLAG]+history에 있음** → large가 학습함 → ngram식 잉여 위험. 또 global bias(coordinate ascent)가 이미 per-class threshold → H1 "list 교정"은 이미 반영됐을 수 있음.
- oracle(내 측정): prompt 94% unique, exact-key 부적합. 중복그룹 oracle ~0.60, +result/+open_files로 개선 안됨.
- R7 질문: reranker가 large+bias 대비 신호를 더하는지 가르는 최소 결정실험은? near-dup embedding oracle 가치?

## R7 결론 (실측 종결)
- 견고한 검증(large vs large+ngram 확률평균, 스케일이슈 없음): **탐색4class에서 +0.0007** (최대). 전체 +0.0022.
- LogReg 스태킹 실험은 스케일붕괴로 무효(0.001). 확률평균이 신뢰가능.
- **결론: large가 텍스트 신호(프롬프트+원문args) 이미 흡수. reranker/specialist=잉여 확정(내 반론 승).** codex의 raw-args 승부수 기각.
- 냉정한 함의: 탐색클러스터=현 모델 천장. text 재활용으론 0.82 불가. 남은 확실경로=강모델 함대(+0.005~0.01)뿐. 0.79+ 돌파엔 **새 정보원 또는 더 강한 base**가 필요.

## R8 (codex 답 + 내 검증/반박)
- codex 수용: raw-args 기각 인정. 텍스트 후처리 계열 폐쇄 합의.
- codex 천장추정: 텍스트 ceiling 0.790~0.800 / 강앙상블 0.785~0.795 / **메타가 진짜면 0.800~0.815** / 0.82+는 새 신호 없으면 불가.
- codex 배분: top10 유지 70%(함대 앙상블+bias sweep) / 도박 30%(메타 프로브). 대체 base는 mDeBERTa-v3-base 1개만, full fleet 금지. 마지막 3일 새 실험 금지.
- **내 반박(코드로 검증)**: codex의 미사용 후보 리스트 중 "previous action type"은 **틀림** — v6 [SEQ]가 최근 12개 행동 순서를 이미 넣음. 반면:
  - `elapsed_session_sec` — **어떤 직렬화 버전에도 없음 (완전 미사용)** ✓
  - turn/budget/loc — **bin만 노출**, 원값·bin내부위치 미노출 ✓
  - 전체 세션 행동카운트/에러카운트 — [SEQ]는 최근12 순서만, 카운트 미노출 ✓
- **프로브 전략(내 수정)**: codex의 "+0.005 → 새 정보원" 판정은 얕은 corrector 배포용이 아니라 **v7 직렬화 투자 판단용**. 4class F1 +0.005는 전체 macro +0.0014에 불과(4/14 가중) — corrector로는 노이즈. 진짜 무기는 프로브 양성 시 v7(elapsed bin+pace+카운트 요약 토큰) large-8ep 재학습.
- 실험: eda/meta_probe.py — LGB(logit) vs LGB(logit+meta), 탐색4class, elapsed만/카운트만 ablation 포함. GBDT라 R7 스케일붕괴 이슈 원천 차단.

## R9 — 메타 프로브 GO: elapsed_sec가 탐색클러스터를 가른다 (07-05 심야)
- 1차 실행: np.empty 미예측(홀드아웃 행) 쓰레기값→유령클래스 오염 버그. 단 **fold별 델타는 유효**: +META [0.0107, 0.0073, 0.0168, 0.0141, 0.0043] — **5/5 양수, 평균 +0.0106** (codex GO 문턱 +0.005의 2배). elapsed 단독 ≈ META 전체.
- **분포 실측 (탐색4class, elapsed 5분위)**: 초반<258s: read 38%/grep 28%/list 22%/glob 12% → 후반>686s: read 28%/grep 41%/**list 6%**/glob 25%. **단조 이동**. 해석: 초반=구조탐색(list/read), 후반=내용검색(grep/glob). list_directory 정밀도 0.384 병목의 설명변수일 가능성.
- **v7 직렬화 구현**: v6 + `[PACE] e{0-6} p{0-5}` (elapsed 8분위 bin + elapsed/turn pace bin). ad_lib 단일소스, ad_common.zip 재빌드.
- **fold0 스크리닝 발사** (largev7p, v6과 동일설정 6ep): 기준 v6 fold0=0.7391, GO 문턱 ≥0.7431. 통과 시 v7 FULL-8ep 재학습 → 차기 배포 후보.
- 오염 수정판 프로브 재실행 중 (홀드아웃 행 제외).

### R9 정정 (오염수정 프로브, 07-05 02:00) — 서사 반전
- 깨끗한 실측: logit-only 0.5526 / **+META전체 +0.0068 (4/5 양수)** / **+elapsed단독 -0.0016 (1/5)** / +카운트단독 +0.0004.
- **1차 실행의 "elapsed 주역" 서사 = 오염 착시.** elapsed 무조건부 분포신호(list 22%→6%)는 진짜지만 large가 [TURN]bin·[SEQ]·hist길이 프록시로 이미 흡수 — 로짓 조건부 잔여 0.
- META 이득은 raw수치+카운트+상호작용 합산 +0.0068. 단일 지배블록 없음. 형식상 GO(문턱 +0.005)지만 **약한 GO**.
- 냉정 환산: 클러스터 +0.0068 → 전체 macro +0.0019(얕은 corrector) → 5.18 할인 시 LB +0.0004 = **corrector 배포 무가치**. 유일 경로 = v7류 심층통합(재학습), 기대 상한도 하향(+0.002 macro 수준?).
- codex R8의 "메타 진짜면 0.800~0.815" 추정 → **과대평가로 판명될 공산**. v7 fold0 스크리닝(진행중)이 최종 결정 실험: 기준 0.7391, 문턱 0.7431. 미달 시 메타 문 닫고 함대(시드/에폭 다양화)+속도최적화 올인.
- 교훈(방법론): np.empty + 부분 채움 = 유령클래스 오염. pooled 지표는 전 행 예측 보장 후에만.

### R9 종결 — v7 fold0 실측 (07-05 02:30): NO-GO, 메타 문 폐쇄
- **v7(=[PACE] elapsed+pace bin) fold0 6ep = 0.7309 vs v6 기준 0.7391 = -0.0082.** GO문턱(0.7431) 한참 미달.
- 깨끗한 프로브(elapsed 조건부 -0.0016)와 정합. **codex R8 "메타 진짜면 0.800~0.815" 실증 기각.**
- 판정 확정: 텍스트도(R7) 메타도(R9) large가 프록시로 흡수완료. **새 정보원은 없다.** 남은 것 = 모델/앙상블 공학.

### R10 데이터 — T4 10분 캡이 앙상블 설계를 지배 (07-05 심야 실측)
- largeonly(단일 large, LB 0.78051): T4 30k **348초**, holdout 5810행 **0.78113** (LB와 0.0006 차 — holdout=우수한 LB 예측자로 재캘리브레이션).
- str2q8(2-large q8, w0.65): holdout **0.78395 (+0.0028)** — 그러나 **719~734초 > 600초 캡, 배포 불가 확정** (batch256/pad8/in-memory dequant 다 해도 -2%: 병목=순수 FLOPs, T4 포화).
- int8 group-64 양자화 자체는 무손상 확인(동일 holdout 재현). 353MB/멤버.
- **함의: T4 캡 안에서 가능한 앙상블 = large 1 + base 1 (~450초) 또는 단일 large 강화(soup/증류)뿐.**
- large+basev6e5(w0.8/0.2) OOF: 0.7483 (+0.0015) → submit_lgb8.zip (0.589GB) 벤치 중.
- **lgb8 판정 (02:40): 속도 500.6s PASS / holdout 0.77873 = -0.0024 FAIL. 제출 금지.**
  - OOF 선판정 +0.0015 → 배포 -0.0024 **반전**. 약멤버 희석법칙(R8) 3번째 실증: klue(-0.014), ngram(전이 20%), base(-0.0024). **8ep FULL large에겐 5ep base도 순수 잡음.**
  - 2-large만 +0.0028이 유지된 이유 = 둘 다 8ep-강. **결론: 확률 앙상블은 강-강만 유효한데 강-강은 T4 캡 초과 → 확률 앙상블 전면 폐기.**
- **남은 유일 경로 = weight-space: model soup** (같은 레시피 8ep 시드들 가중치평균 → 단일모델 348s, 희석구조 없음). 시드2024 v6-8ep FULL 학습중, soup_members.py 준비완료. 실패 시 largeonly 0.78051 현상유지 + 증류 검토.

## R10 codex 응답 + 실행 결정 (07-05 03:00)
- codex 정량 랭킹(수용): ①증류 45-60%×(+0.0008~0.0015) ②2-way soup 40-55%×(+0.0005~0.0013), 조건부 3-way ③largeonly 유지+최종재검증(필수 은행) ④FGM 폐기(15-25%, 기대 음수 가능).
- soup 기대 하향(codex): 확률앙상블 +0.0028의 20~60% 회수 = +0.0006~0.0017. **판정선: holdout >0.7825 제출 / 0.7815~0.7825 보류 / <0.7815 폐기.**
- 증류 레시피(codex): hard 0.6~0.8 + soft 0.2~0.4, T=1.5~2.5, FULL-blend soft 사용 가능(배포=FULL이라 누수 개념 약함). 회수율 25~70%.
- **새 아이디어 즉시 채택: 체크포인트 평균(SWA-lite)** — 비용 0, +0.0003~0.0012, cross-seed soup과 결합. train_full_cli에 AD_SWA_K 구현, **s2(시드2024)를 SWA-3 내장으로 재발사** (ep1-2 재시작 비용 15분 < 기대이득). → s2 산출물 = seed다양성+SWA 동시 획득.
- codex 현실 상방 추정: largeonly 0.7805 → **0.782~0.784**. 0.790 추격은 단일 트릭으론 불가.
- 실행 순서 확정: s2-SWA 회수 → 2-way soup → T4 holdout 게이트 → (soup 양수면 3-way / 아니면 증류 1-2 variant) → 마지막 3일 동결·재검증.

## R11 종합 재점검 (07-05 오전) — codex가 내 주장 2개를 꺾음
- 배경: **9위로 하락** (0.777~0.782에 6팀 밀집, +0.001=두 계단). Colab 유닛 소진 → s2/dist를 Kaggle 커널(ad-full2-s2-dist, 서버측 ~5-6h)로 이전.
- **codex 유효 반박 ①: "확률 앙상블 전면 폐기"는 과잉 — 조건부 2-pass가 살아있다.**
  m1 전체 → top1-top2 마진<th 행만 m2 재추론·혼합. **OOF 실측: th=0.5 → 32% 선택, +0.0023 (full +0.0033의 70%), T4 추정 485s.** th=0.7이면 +0.0030/536s(위험). → **submit_cond2.zip 조립완료(th=0.5, cond-bias 동일규칙 적합), T4 벤치 중.** 예상 배포 holdout ~0.7831 → LB ~0.7825 (5-6위권).
- **codex 유효 반박 ②: 게이트 0.7825 과보수** — holdout-LB 갭 -0.0006이므로 0.7817이면 7-8위 추월. **저위험 후보 게이트 0.7817로 하향, bias/pairwise류만 0.7822+.** 은행 아끼지 말 것.
- **내 재반박(codex 오류): "test 세션겹침 누수 audit" — 이미 실측 기각.** posmap 제출(=이 가설의 직접 시험)이 히든 30k에서 점수 완전 동일(0.7399998302) = coverage 0. 재실험 불필요.
- codex 기타: 손실개선(R-Drop/LS/focal)은 bias와 중복, -0.001~+0.001로 하향(기각 유지). **pairwise post-hoc(list_directory 저마진 스위치) +0.0005~0.0015 crossfit 필요(백로그).** au제외 FULL(sim-only) +0.0008~0.0020(백로그, GPU 확보시). ONNX/jit 기각 합의. 서버 3vCPU 마진: 토크나이즈<126s면 안전, 후보 p95≤500s 통과 / 500~540s 고득점시만 / >540s 폐기.
- 상위팀 추정(codex): 0.784~0.787 = 직렬화+sim-only+soup/증류/조건부TTA 조합권. 0.790+ = 구조 활용 or 합성데이터 가능성(우리 posmap 기각으로 전자는 의문).
- 제출 캘린더: ~7/13 실험, 7/14 저녁 final 고정, 7/15 오전 제출 회피.

## R12 — LB 직접최적화 체제 확립 (07-05 오후)
- **공동 결론: private=public(동일 30k, 결정적) → LB는 노이즈가 아니라 최종 목적함수 그 자체.** 스윕은 "일반화 추정"이 아니라 "고정 함수 힐클라이밍". 단 다음 수 선택의 추론에는 해상도 규칙 필요.
- **codex 스윕 판정규칙(채택)**: Δ<+0.00025 무승부 / ≥+0.00035 방향 후보 / ≥+0.0007 축 인정. 가중치는 1축씩 0.05 coarse → 승자만 0.02 refine. 3축 grid 금지.
- **codex가 내 Wave1을 부숨(수용)**: 가중스윕 EV ±0.0002~5로 저가치(OOF 곡률 평평). 주축 = ①m1 교체(soup/dist) ②pairwise/class post-hoc(macro-F1 특성상 global weight보다 ROI↑) ③조건부 멤버 확장 ④weight refine은 마지막.
- th 0.65 기대 +0.00005~0.00018 (cond2 역산: 35% 조건부가 이미 full 이득의 79% 회수, 잔여 0.00027뿐) → best m1 확정 후 1발만.
- klue 재시험 허용: w=0.03~0.05 저가중 1발만(스태커 -0.014 전과로 global 기대 음수), <+0.00025면 영구 폐기.
- soup/dist 기대(중앙값): s2교체 -0.0005~+0.0007 / s1+s2 soup +0.0004~6 / dist -0.0003~+0.0008 / soup+dist 멤버화 +0.0006~14(시간위험, 조건부로). **cross-seed weight soup은 LB 1발 전 동일계열 holdout 붕괴점검 필수.**
- 도달확률(codex): 0.7843(4위) 35~45% / 0.7870(3위) 5~10%(구조적 신규익 필요) / 컷 방어실패 8~12%→0.7834+ 은행 시 3~5%.
- **90슬롯 배분표**: m1교체·soup·dist 18 / weight sweep 20 / th·조건부 10 / pairwise post-hoc 14 / 약멤버 probe 8 / sim-only 8 / 예비·막판 12.
- 오늘 실행: base+ (0.55/0.2/0.25) 1발 + v4+ (0.55/0.15/0.3) 1발 → 커널 도착 즉시 soup/dist m1로 슬롯 전환. pairwise crossfit(eda/pairwise_posthoc.py) 병행.

## R13 — 환경 전환(A6000 무제한) 후 슬레이트 재편 (07-05 저녁)
- **환경**: Colab/Kaggle 은퇴 → 연구실 A6000 48GB(mun-jtrain) + 오프라인 검증(mun-jtest, 12g/3cpu, 대콘 재현). train_and_verify.sh 전자동 루프, calib ratio 2.62. **기준선 재현 확인: largeonly holdout 0.78113 (T4 실측과 소수점 5자리 일치)** — 판정선 이식 유효.
- **codex 수정 5건 (전부 수용)**:
  1. **soup 단독 LB 발사 강등**: 기준점은 largeonly가 아니라 tri_cond 0.78266. soup 기대 +0.0004~6으론 단독으론 못 넘음 → s2단독/soup 변종 전부 **m1 후보군**으로만 만들고 tri_cond 재조립 후 LB 직행.
  2. **판정선 개정**: 절대 holdout-LB 매핑 깨짐(largeonly -0.0006 / lgb8 +0.0035 / str2q8 -0.0021, 계열의존 편향 ±0.002~4). 새 규칙: large계열 단독 제출선 holdout ≥**0.7833**(tri_cond 환산) / m1 후보 탈락선 0.7815 유지 / 동일계열 상대비교(delta)만 신뢰.
  3. **sim-only SWA3 > dist 순위 역전** (R11 기대 +0.0008~20 > dist -0.0003~+0.0008). AD_SIMONLY 구현 완료.
  4. **dist teacher 업그레이드**: soft_labels_str2(2-way, 상한 0.78189) 대신 **tri_cond 3-way 확률로 재생성** 후 증류 (hard0.7/soft0.3/T2/8ep+SWA3). 자기증류 성분 0.195는 무해.
  5. **가중 soup 5점 비교**: s1/0.25/0.5/0.75/s2를 동일 holdout에서. 50:50이 양끝보다 낮으면 s1(no-SWA) 이질성 → s2+s3 동질 soup 전환. s2단독<s1이면 seed 품질 문제 → s3·sim-only 우선. soup_members.py --w 구현 완료.
- **96슬롯 재배분(codex)**: m1교체·soup·dist·sim-only·에폭변형 32 / tri_cond 재조립·th·weight 24 / weight probe 12 / 약멤버·이종교사 6 / bias·재검증 8 / 최종·예비 14. base+/v4+ 2발은 m1 교체 전 보류.
- **내 명확화 2건 (codex 미언급)**: ① m1 교체시 cond-bias 재적합 불가(소스=6ep teacher OOF, FULL soup엔 OOF 없음) → 기존 teacher-OOF bias·th 재사용이 원칙, "재적합"은 OOF 가진 조합만. ② 검증 큐 규율: 학습 공존시 VRAM 게이트 오염(41.5GB 실측) — holdout은 공존 허용, **시간·VRAM 게이트는 GPU 유휴시에만 유효**.
- **합의 실행순서**: s2 검증 → 가중 s1/s2 soup 5점 → m1 치환 tri_cond 오프라인 → s3(777)+sim-only 학습 → greedy 3~5 seed soup → tri_cond-teacher dist → best 기준 th/weight → LB.

## R14 — 3축 사망·듀얼bias 발견·하네스 개방 (07-06 새벽)
- **밤새 실측 (A6000 파이프라인, holdout 5810 = 전체/sim 5407/au 403)**:
  - s1(no-SWA) 0.78113/0.79831/0.523 | s2(SWA3) 0.77600 | s3(SWA3) 0.77436 | sim-only(SWA3) 0.76275 | sim-only-raw 0.76851
  - **soup V자 붕괴**(75:25 0.77385 / 50:50 0.76304 / 25:75 0.76957) → cross-seed weight soup 사망 (R10 기각)
  - **SWA-lite 해악 확증**: 같은런 SWA 제거 +0.0058. 원인=6→8ep 미수렴 구간 평균. → 이중저장 패치(raw+SWA), **전면 raw 채택** (codex Q1 승인). seed 잭팟 없음(단일모델 천장 ≈0.781)
  - **sim-only 사망**: raw로도 sim-subset -0.0093 — au 5k행이 sim 성능에도 기여
  - **히든테스트=혼합 증거**: largeonly LB 0.78051 ≈ holdout-전체 0.78113(-0.0006), holdout-sim 0.79831과 -0.018 괴리 → all-sim 가정(5.14) 기각 방향
- **듀얼 bias 발견 (하네스 축 1호, 사용자 지시로 개방된 공간)**: bias를 sim/au 서브셋별 적합, 추론시 id prefix로 행별 선택(무표식 폴백=글로벌·무손실). 프록시: au-subset 0.622→0.688(+0.067), **holdout 전체 +0.00208**. codex GO — λ 셔링키지(bias' = global + λ·(subset−global)) 0.5/0.75/1.0 스윕 후 최적 배포. **GO문턱: largeonly-dual +0.0010↑ 제출 / +0.0005~10 tri_cond 확인 후 / 미만 보류**. 산술: tri_cond 0.78266+0.00208=0.78474 > 5위컷 0.78429
- codex Q3 수정: H1 LB프로빙(상수클래스 n_c 역산)은 **2~4발** — au-heavy 2발로 혼합비 검증 먼저, 문제클래스 1~2발은 조건부. (내 보완: 듀얼bias LB 1발 자체가 au-prefix 존재의 결정 실험을 겸함 — 점수 나오는 프로브)
- codex Q4: dist 교체선 = raw 기준 s1+0.0003(≥0.78143), tri_cond 재조립은 기존 대비 +0.0005↑일 때만 LB
- codex Q5 도달확률: 듀얼 성공시 5위 55~70% / 부분성공 30~45% / 실패 12~20%. 슬롯: 듀얼 8-12, dist 8-12, H1 2-4, th/weight 10-14, 약멤버 4-6, 예비 20+
- **구현 완료**: postproc 3-bias 저장 + ad_lib 행별 선택 + package_single --dual/--dual_shrink + **검증셋 id 원본보존**(prefix 의존 경로 실전동일 — 하네스 수리) + eda/dual_lambda_sweep.py(배포모델 확률 1회 추론 후 즉석 λ스윕)

## R15 — [GEN] 버그 발견·3세계 판별트리 (07-06 새벽)
- 발견: v4+ 직렬화 [GEN] 토큰 = `id.startswith("sess_au_")` 파생. 구 검증하네스(ho:: id변조)가 au행을 [GEN]sim으로 오서빙 → au 0.523(정상 0.945). 하네스 수리(id 원본보존) + anon 모드(id 익명화=히든 무표식 재현) 신설.
- **3세계 가설**: A=무표식혼합(au 버려짐→수리시 +0.005~15 최대레버) / B=all-sim / C=표식혼합(GEN 정상, 듀얼bias 발동).
- 판별도구: probe_read_file(상수제출 n_c역산) + largeonly_dual + au감지기(LGB 콘텐츠, 행AUC 0.974 th0.99 P0.998).
- v6n(no-GEN) 직렬화 구현 + FULL 학습(World A 무기 선행). dist 기각(sim -0.008). codex: 제출순서 largeonly_dual 먼저, A1(no-GEN재학습)>A2(감지기복원)>A4.

## R16 — v6n 폭락(-0.047)이 World C 확정 (07-06 아침) ⚡map 재편
- **submit_largev6nraw LB = 0.73371** (largeonly 0.78051 대비 -0.047). fold0 정직 0.7327 ≈ LB 0.7337 정합 → 학습실패 아닌 진짜 no-GEN 천장.
- **3세계 판별 종료 → World C 확정**: A라면 v6n이 au회복해 올라야(반대) / B라면 [GEN]상수라 -0.047 붕괴불가 / ∴ C = 테스트 id 표식有·[GEN] 정상서빙·필수신호. v6n==v6−[GEN] 불일치 0 재확인(codex 유보 해소).
- **핵심 해석**: [GEN]이 v6 FULL부스트(+0.041)의 열쇠(id파생+암기/분포정렬). no-GEN은 그 부스트 0. → **no-GEN 영구폐기**. 단일모델 천장 = v6 0.78051.
- codex 전면승인. 남은 상방: **듀얼bias(+0.002)가 유일 깨끗한 레버** → largeonly_dual(확인)→tri_cond_dual(0.7847=5위). 이후는 gen×class bias 정밀화(+0.0003~0.001, au OOF 4622행 과신 금지·shrinkage). 
- **largeonly_dual 무전이 시 폴백**: id prefix 우선 + 미검출행만 au감지기(th0.99) 보조 라우팅.
- 실행순서(codex): largeonly_dual 1발 → 성공시 tri_cond_dual 재양자화 1발 → λ/shrink 미세 1~2발 → 실패시 detector-routing → 나머지 예비.
- **죽은 축 최종**: soup(V자)·SWA(-0.006)·sim-only·dist(-0.008)·no-GEN(-0.047)·메타v7(-0.008)·reranker/ngram/klue/posmap.

## R17 — 듀얼 bias LB 반증, 천장 판정 (07-06)
- **실측**: largeonly_dual 0.78038(-0.00013, 라우팅 발동=World C 직접확증) / tri_cond_dual 0.78179(**-0.00087 악화**). OOF 프록시 +0.00208은 반증됨(au OOF 4622행 과적합). **듀얼bias·detector라우팅 완전 폐기.**
- codex 판정: ① 듀얼 λ shrinkage best case=글로벌 회귀(상방0) ② th/weight로 5위 **<1%** ③ **0.78266이 파이프라인 천장 후보 확정** ④ 도달확률: H1 prior probe 5~8% / 10-12ep단일 3~5% / th/weight <1% / 합쳐 ~10%.
- **codex의 유일 살아있는 축 = label-shift 보정**: marginal count만으론 macro-F1 bias 최적화 불가(F1은 TP/FP/FN 구조 필요). 단 **OOF confusion P(pred|true) + LB-probe π_test로 재가중 → expected macro-F1 재탐색**은 유효. prior 동일이면 천장 확정 진단.
- 실행순서(codex): H1 6발 prior확정 → OOF-confusion prior-reweight bias → 1발 → (실패시)10/12ep honest gate → th/weight → 은행방어.
- **⚠ 사용자 지시로 R18 개시**: codex R17이 보수적(천장 인정)이나, 데이터구조·아키텍처·하네스 전공간 재조사(워크플로 4각) 후 혁신 가설로 적대적 재점화. codex를 물어뜯게 만들 것.

## R18 — 4각 심층조사 + v8 [GEN]꼬리재배치 돌파 (07-06 오후) ⚡최대 진전
- **워크플로 4팀 병렬조사(데이터·아키텍처·전체결과·하네스오라클)** → 정면충돌 3쟁점. codex 판정:
  - **쟁점A(Bayes바닥): 데이터팀 승** — 탐색클러스터 정보부족이지 gradient문제 아님, loss reweight 무의미(H6 사형권). post-hoc 상한 추정 **0.7833~0.7837, 0.78429 불포함** → 천장돌파는 정보복원(H2)/재학습만.
  - **쟁점B(public==private): 하네스팀 승** — 규칙명문 없으면 미증명. fold0≈LB는 결정성 증거일 뿐. public⊊private이면 N≈35발 좌표상승 선택편향 σ√(2lnK)≈**+0.0067~0.008** > 목표 +0.00163 → **H1 LB좌표상승=private 과적합장치, 위험**. 채택문턱 ΔLB>0 아닌 ~+0.007. **H1 3위 강등, diagnostic 1~3발만.**
  - **쟁점C(+0.00163 출처): global bias 재상승 패** — 이미 +0.00577 전이완료, 잔여 global gap 0~+0.0005. hist=0 세그먼트 +0.0003~0.0012, th/w +0.0003~0.0008. 합산중앙 ~+0.001 = "+0.00163은 계획 아닌 베팅".
  - **codex 재랭킹: H2(1위, 유일 구조적 돌파) > H4 > H1(강등) > H3 > H5 > H6(무료진단 통과 전 금지).**
- **★v8 fold0 실측 = 0.7529 vs v6 0.7391 = +0.0138** (게이트 +0.003의 4.6배, 정직프로브). **[GEN] 좌측절단 소실(긴세션 26.8%가 320초과, [GEN] 10.8%서 삭제) 복원이 진짜 병목이었다.** no-GEN -0.047·FULL부스트 +0.041과 정합. **프로젝트 전체 최대 단일이득. 천장 0.78266 돌파 실사격 개시.**
- Bayes바닥 진단은 키 99% 유니크로 암기착시(무효) — 단 v8 +0.0138이 "회복가능 신호 존재"를 직접증명(절단버그≠Bayes바닥).
- **v8 함대 체인 발사**: folds1-4 교사(OOF/bias) → FULL-8ep(SWA없음) → largeonly-v8 패키징+검증 → LB. v6 바이트동일 보장(0/30). fold1 재현 확인 후 FULL 확정.
- **다음 R19 쟁점**: v8 FULL-LB 전이율(fold0 +0.0138 → LB 얼마?), v8을 tri_cond m1에 이식, v8 멀티시드/8ep, [GEN]류 추가세분화(과적합 경계).

## R19 — v8 LB 반전(0.77819), 오프라인 메트릭 사망 확정 (07-07)
- **submit_largev8 LB 0.77819**: v6 largeonly 0.78051 대비 **-0.0023**, 은행 0.78266 대비 -0.0045. fold OOF +0.0142(전폴드 견고)가 LB 역전. FULL부스트 v6 +0.0428 vs v8 **+0.0263**(재배치가 숨은 LB전이 -0.0165 깎음).
- **층화진단**: v8이득 전구간 균일, **hist=0(절단0%)이 최대 +0.0178** → 절단복원 가설 완전기각. 이득원=[CUR]인접 재배치.
- **codex 진단(수정승인)**: v8은 정보복원이 아니라 **[GEN]의 coarse-prior 사용 → GEN×prompt 상호작용 shortcut 전환**. train 생태계 전폴드서 먹히나 hidden서 조건부관계 다름 → 손해.
- **4연속 OOF→LB 반전 확정**(dual/lgb8/str2q8/v8). codex: OOF는 noisy가 아니라 **hidden서 깨지는 latent shortcut을 그대로 보상하는 구조**. 방향성 없음(false-neg·false-pos 둘 다). **신규모델·직렬화 OOF판정 영구금지.**
- **신뢰 3원천**: ①LB양성 구성요소 소재조합 ②LB프로브 prior→OOF confusion 재가중 bias보정 ③적대적split(GEN/template/source leave-out) 생존. **은행 0.78266=검증된 천장**(임시 아님). 방어가 기본.
- 히든-대표 오프라인 oracle은 원천불가(공개5건으론 고차 조건부분포 못봄). 가능한 건 v8류 사전차단 **failure filter**뿐.

## R22~R24 — 팀조율 + retrieval축 + 증강 (07-07, jeong 브랜치)
- **팀상황**: 조원(별도 Claude) tri_v4new **0.78449=14위(컷 top12 밖)**. tri_cond m3를 v4-s777로 교체(다양성법칙). 리더보드 +0.001/일 인플레. **0.80+ 팀 실존=미발견 +0.018 축.** 같은계정·쿼터공유(제출 다 함). 조원 커버=시드/base-m2/mdeberta. 내 직교축 분담.
- **R22 codex**: 내 축=**X1 retrieval/near-dup prior 1순위**(최대직교·EV). 역할=조원 tri위 조건부 correction layer(또다른 앙상블러 아님).
- **retrieval 진단 GO**: large-v6 frozen mean-pool(이방성→중심화). **순환성 통과: 모델오답 27% KNN회복(랜덤7%)=직교신호 실재.** holdout 블렌드 +0.0233.
- **R23 codex**: 히든전이 감산 기대 +0.006~15. 보수 canary(λ0.3 margin<0.3 purity≥0.7 sim_th0.89 게이트5.7% OOD guard) largeonly위. 배포=ad_lib return_emb(base_model+classifier 1forward, T4안전)+_retrieval_adjust(train_emb 144MB동봉).
- **retrieval canary LB 0.78096 = +0.00045**: 히든 전이 **양성(음수아님)** but 약함(전이율~6%). **holdout +0.023 vs 히든 +0.0005 = 우리 holdout이 히든 미대표 확증**(v8/v9/retrieval 동일패턴).
- **v9(rich) 완전사망**: LB 0.76903(-0.011), @384 fold0 0.7317(-0.0074, 예산가설 기각). rich경로신호는 large에 무용(v6서 이미 추출). codex R21 rich35% 가설 large엔 사망.
- **R24 codex — +0.018 정체 베팅: 구조 > backbone > 착시.** holdout≠히든 확증. 0.80팀은 "더 잘 외우기"아닌 **히든불변 구조**(test-time intra-session 일관성·action transition·history길이불변 표현) or input construction. metric 착시(public/private) 체크 권고.
- **★history-dropout 증강=내 GPU 1순위(codex)**: 히든 history분포 shift 직격 정규화(라벨오염 아님). drop전체20%/최근1-2 30%/랜덤prefix30%/원본20% + field dropout. hist=0오류43.9% 타격. gate=short-history slice·LB canary. **FULL-8ep 학습중.** 사용자 증강제안=유효(정규화형).
- 제출후보: submit_retr_v6_mid(retrieval 공격 λ0.45 게이트18%), submit_histdrop(학습후).

## R20 — "모든 전략" 캠페인 (07-07, 사용자 지시) 진행중
- 14개 상수클래스 프로브 zip 생성(prior 역산: n_c=30000·14M/(2-14M)). prior_from_probe.py + labelshift_bias.py(자기검증 통과, test=train시 0.7501 수렴) 준비완료.
- codex에 캠페인 잠금 요청: A(prior프로브+label-shift) B(LB양성 재조합 th/weight) C(적대적split 신규모델) D(str2q8 조건부/TTA). 순서·사전등록 문턱·선택편향 예산 확정 대기.
- **R20 codex 답(천장전제)**: A1 13발 prior프로브(정보용) → A2 label-shift(L1≥0.04 또는 class shift≥0.8pp AND reweighted OOF≥+0.0013일때만 1발) → B1 th0.6 1발 → B2 조건부. bank교체 문턱 0.7871(best-of-K=3 선택편향 +0.0044 보정). **단 이 캠페인은 천장전제라 아래 R21에서 폐기.**

## ⚡R21 — 천장 반증! 0.80+ 팀 실존 → 전면 재설계 (07-07)
- **사용자 확정 외부사실: LB 0.79+ 20팀, 0.80+ 존재.** "0.78266 천장"(R17~R20) 완전 반증. +0.018 격차 = post-hoc 불가, 구조적 누락.
- **데이터 재발견: 64% 한국어**(ko45k/en18k/mixed7k, 현재발화 71% 한글). BUT klue<xlm-r였으므로 실신호=한국어의미 아닌 **tool trajectory·path/result·혼합기술어**.
- **codex Q1 확률배분**: (b)버린 rich feature[full path/result/elapsed] **35%** / (d)구조·retrieval·약누출 28% / (c)이종앙상블 15% / (e)문제재정의 14% / (a)강한 base모델 **8%**(klue약세가 강력 제약). **1순위=모델교체 아닌 표현 재설계.**
- **내 Q4 진단(정합)**: 탐색클러스터(41% 병목)의 **36.1%가 경로/패턴을 history args에만** 보유 — 우리는 ext만 씀. 45.3%는 [CUR]에 있음(모델이 봄), 17.7%만 진짜 무신호.
- **codex LB정책 개정**: naive OOF 폐기. **portable/인과 feature면 fold 애매해도 LB canary 1발**. fold>+0.010이면 canary. v8류 배치/shortcut은 1회실패 후 폐기. 검증재설계=session/template/language/action-family split.
- **8일계획(codex)**: D2 rich직렬화(v9) → D3 ablation LB → D4 retrieval/구조 prior → D5 모델다양성(mdeberta rich) → D6 문제재정의(action-family head/transition prior) → D7 distill → D8 수렴. 산식 rich+0.006~12 / retrieval+0.004~10 / 앙상블+0.003~6.
- **실행 개시: v9 rich직렬화 구현·발사.** v9 = v6 순수상위집합(경로 role/depth/basename + result[:100] 유지 + [PACE] elapsed). v6 바이트동일(0/30). read_file(tsx|src|d1|Button.tsx)[success] ok;258 lines 식. portable 인과신호(v8 위치shortcut과 구분). 함대체인(fold0게이트 0.736 → folds1-4 → FULL → 패키징 → 검증) 발사.

## R27 — session-balanced FT 게이트 판정 (07-08)
- R26의 유일 GPU실험을 구현·실행: `AD_SESSION_BALANCED=weight`(1/session_len) / `sample`(epoch당 세션당 1step), baseline fold0 checkpoint에서 LR5e-6 FT. `sim/eval_stress.py`로 fold0-val stress 평가.
- baseline: overall 0.7485, sim 0.7424, nnQ1(low) 0.6155, sess1-2 0.6314, hist0 0.4852.
- **weight**: overall 0.7527(+0.0042), sim 0.7443(+0.0019), sess1-2 0.6792(+0.0478), hist0 0.4976(+0.0124), agree 0.9649. 단 **nnQ1 0.6149(-0.0007)**.
- **sample**: overall 0.7509(+0.0024), sim 0.7423(-0.0001), sess1-2 0.6816(+0.0502), hist0 0.5071(+0.0219), agree 0.9701. 단 **nnQ1 0.6108(-0.0048)**.
- 판정: session-balanced는 짧은 세션/hist0 편향 제거에는 성공했으나, 가장 히든유사 proxy로 둔 low-NN이 둘 다 악화. R26 게이트(`nnQ1`·`sess1-2` 동시 상승 + agree<0.97) **불통과**. FULL/LB canary는 보류. `weight` checkpoint는 tri/retrieval 결합 재료로 보관만.
- 다음 수: GPU 추가 train 변형 중단. 사용자가 제출 슬롯 부족으로 **LB prior probe 미제출**을 확정. 또한 조원과는 자료 공유/대화가 없으므로 조원 의존 축 제거. 우리 자체 최고 `tri_cond 0.78266` 위에 retrieval을 넣는 쪽으로 전환.

## R28 — compact retrieval로 자체 tri_cond 후보 생성 (07-08)
- 기존 retrieval pack 137MB는 tri_cond(0.943GB)에 못 들어감. `emb_v6_70k`를 384d Gaussian projection+fp16으로 줄여 `work/retrieval_pack_p384` 54.7MB 생성. train top1 agreement 0.725라 보수 prior 용도로는 사용 가능.
- `ad_lib` 수정: `proj.npy` 지원, ensemble/conditional 점수 위 retrieval adjust, m1 확률+embedding 동시 회수로 중복 forward 제거. `package_ensemble`/`package_single`도 compact pack 복사 지원.
- 후보 `submit_tri_cond_retr_p384.zip`: tri_cond(w0.6/0.15/0.25, th0.5) + retrieval conservative cfg. 용량 0.993GB, check_zip PASS, offline n=64 PASS.
- 30k 검증: tri_rebuild A6000 212s/VRAM10868/holdout0.80702 vs retr_p384 A6000 216s/VRAM10964/holdout0.80984. changed 147/30000=0.49%, retrieval gate 2.62%. holdout 이득은 self-retrieval 오염이라 LB예측 금지.
- p256 fallback도 생성: 0.976GB, A6000 211s, gate 2.72%, changed 172/30000=0.57%, holdout net +16(p384 +20). 용량은 더 안전하지만 보존율 낮아 2순위.
- 판정: 자체 기록 갱신용 LB 1발 후보는 **p384**. 근거는 largeonly retrieval 보수 실측 +0.00045뿐이라 기대는 `+0~+0.0005`, 하방은 gate 0.49%로 제한적. mid/aggressive는 이미 음수였으므로 금지. 업로드/용량 문제가 생기면 p256으로 대체.
- **LB 실측 (07-08 14:38): 0.78261 = tri_cond 0.78266 대비 -0.00005 평탄.** 러닝타임 6:54. retrieval prior의 largeonly 이득(+0.00045)이 tri 위에서 소멸 — codex R25 "+0.0000~+0.0004" 예측의 하단 적중(tri가 이미 같은 오류를 고침). **retrieval→tri 축 실측 종결.**
- 팀 운영 확정(사용자): **조원과는 제출쿼터만 공유, 자료·대화 공유 일절 없음.** 조원 산출물 의존 계획 전면 폐기.
- 목표 재상향(사용자 지시): top12 방어 수비 아닌 **0.80 돌파용 혁신 결과물** — 모든 정보(transductive test-time·LB이력 역공학·하네스 전체) 총동원.

## R29 — 프로브 3발 설계 + prior축 최종 판정 (07-08 저녁, codex xhigh)
- **입력 실측**: em075(blind EM s0.75, changed 1.24%) LB **0.78173 = -0.00093** — 홀드아웃 게이트(+0.0020)가 또 반전. EM이 추정한 히든 shift(pi_l1 0.064)는 대부분 캘리브레이션 노이즈. 내 시뮬(eda/labelshift_em_sim.py): em0.5+clip은 identity -0.0003 무해·진짜 shift(L1 0.5+)면 +0.008~+0.023 — 방법은 견고하나 shift 존재가 미검증.
- **Q1 프로브 3발 수정 GO**: read_file(+13.43pp)/glob_pattern(-6.22pp)/list_directory(-4.33pp) — au/sim 판별 최대 3종. edit_file은 질량 커도 판별력 1.72pp라 부적합, grep_search는 4번째 후보. λ_c 산식: λ_read=(p−12.26)/13.43, λ_glob=(8.00−p)/6.22, λ_list=(6.49−p)/4.33. 판정: max−min λ ≤0.12 혼합축 신뢰 / 0.12~0.20 measured만 / >0.20 λ폐기.
- **Q2 재적합 산식 락 — measured-only damped bias**: b_c(α)=α·log((π*_c+ε)/(π̂_c+ε)) 측정 클래스만, α∈{0.25,0.35,0.50}, changed-rate ≤0.8%(하드 1.0%). **raw logits→prior target 1개→bias 1개→1회 적용**(EM×bias 중첩 금지). λ 일관시에만 au-가중 OOF 재적합이 2순위. 제약-EM은 제출용 아님(γ≤0.3 분석용).
- **Q3 session-balanced FULL: 학습 GO / blind 제출 NO-GO / tri 4번째 멤버 조건부 GO**: 게이트 = OOF(tri+sb) ≥ tri_cond+0.0010 AND nnQ1 drop ≤0.0010 AND changed 0.4~1.2% AND ≤560s. m2/m3 교체보다 4번째 low-margin 멤버 우선. → **largev6sbwt FULL FT 발사됨**(member_largefullv6에서 weight변형 lr5e-6 2ep).
- **Q4 0.80 경로 판정: prior축으로 +0.017 불가.** EV 순: ①test-test near-dup/template 클러스터 일관성 +0.005~0.015(유일한 큰 후보, high-conf만) ②confusion-pair specialist/계층분류 +0.002~6 ③이종백본 +0.001~4 ④sb-4th +0.0005~2 ⑤grinding +0.0003~15/발. 은행=tri_cond 0.78266 기준.
- **Q5 예비 1발 조건부**: Σ|π*−π̂|≥1.5pp AND λ 일관 → measured-only damped bias 오늘 1발 / λ 불일치 → 4번째 프로브 grep_search / 둘 다 아니면 보존.
- 준비 완료: eda/probe_mixture.py(역산+λ적합), H0 사전등록(read 0.0166772/glob 0.0100298/list 0.0083147 = 히든이 train prior일 때 기대점수).

## R29 — transductive EM label-shift, probe 없이 hidden prior 추정 (07-08)
- Claude 잔여물 `eda/labelshift_em_sim.py` 실행: largeonly 5fold 확률에서 Saerens EM. `em0.5+bias`는 identity -0.0003로 거의 중립, dir30/dir10 shift에서는 +0.005~+0.020. 단 largeonly 단독은 약해 tri_cond에서 재검증.
- 구현: `ad_lib.predict()`에 `meta["labelshift_em"]` 처리 추가. 위치는 ensemble/conditional/retrieval score 산출 뒤, postproc bias 직전. `package_ensemble.py`/`package_single.py`는 `--labelshift_em`으로 OOF 평균 `pi_ref`를 run_meta에 저장.
- tri_cond holdout-only 재추론(실제 rebuild 패키지, 5,810행): bias 0.8070 → em0.5 0.8084(+0.0014), em0.75 0.8091(+0.0020). dir100/30/10 prior-shift resample 대부분 양수. `em1.0`은 일부 붕괴하므로 금지.
- 후보 생성/검증:
  - `submit_tri_cond_em05.zip`: 0.943GB, check PASS, A6000 209s, VRAM10868, EM pi_l1=0.064, changed(prebias)=1.27%, holdout 0.80768.
  - `submit_tri_cond_em075.zip`: 0.943GB, check PASS, A6000 210s, VRAM10868, EM pi_l1=0.064, changed(prebias)=1.76%, holdout **0.80907**.
- `em075` 최종 라벨 변화: 373/30000(1.24%), holdout 67/5810(1.15%), fixes31/breaks24/net+7. macro 이득은 ask_user(+0.016), plan_task(+0.012), read_file(+0.0068) 쪽. grep_search -0.0057 비용.
- 판정: **새 자체 LB 후보 1순위 = `packages/submit_tri_cond_em075.zip`**. probe 제출 없이 hidden batch 확률만 쓰는 구조 축이라 R28 retrieval보다 더 정당한 한 발. 단 LB 실측 전까지 과신 금지 — 제출 슬롯을 쓸지 사용자 결정 필요.

## R30 — 프로브 실측 → prior축 공식 종결 + 목표 전환 (07-08 밤, codex xhigh)
- **프로브 3발 LB 실측**: read 0.0174632→π*13.93%(+0.70pp) / glob 0.0090039→6.73%(-0.82pp) / list 0.0085129→6.34%(+0.15pp). λ_au=0.130(잔차 0.36pp, train 0.072의 1.8배 au기움), per-class λ 스프레드 0.169=약신호. **히든 prior≈train 확정.**
- **damped bias 기각(오프라인)**: 정답앵커로도 α0.25~0.5 전부 -0.0003~-0.0009. **원리: macro-F1 최적 bias는 저정밀 클래스를 의도적 과다예측(list π̂9.28% vs 진실6.49%) — 카운트매칭은 F1최적을 되돌림.** codex 동의, R29 Q2 산식 자진 철회. em075(-0.00093) 근본원인 동일. **prior축 3중증거(blind EM/정답앵커/프로브) 종결, 추가 제출 0발.**
- **λ_au=0.13 잔여 용도**: 공격축 아님. ①au-가중 bias 재적합(+0.0007±0.0016)은 최종 앙상블에 5~12% 이내 저가중 동봉만, non-au 손실 -0.0003 이내 조건 ②"au 1.8배 시나리오 생존" 방어 스트레스테스트.
- **test-test 클러스터 일관성 강등**: 홀드아웃 실측 커버25%(sim≥0.95)·이웃 라벨일치 69%(aleatoric 천장)·모델 이미 96.4% 자기일관 → 스무딩 최대 +0.0015. R29 Q4 1순위에서 탈락.
- **새 1순위 = confusion-pair specialist/계층분류 (기대 +0.0015~0.0040)**: E={read,grep,list,glob} 라우팅 게이트(P(E)≥0.70~0.80, precision 우선) → E내부 4-way specialist → pairwise veto(read/grep·list/glob·read/list, base margin<0.12~0.18 AND spec margin>0.08~0.12만 flip) → soft hierarchy 결합 → bias 재최적화. 반전회피 조건: OOF confusion에서만 개입·prior 무접촉·변경률 0.8~2.5%·non-E Δ≥-0.0003·OOF +0.0015 미만 제출금지.
- **공식 목표 전환(codex 판정)**: 측정축 전합산으로도 0.80(+0.01734) 빡빡 → **현실 0.786~0.792, 등수방어 최적화. 0.80은 대형 신규축 발견 시 상방으로만.**
- **60발 배분**: 24 그라인딩(시드/증류/조건부) / 16 specialist / 8 sb계열 / 5 이종백본(OOF diversity 조건) / 7 은행방어 예비. 동결: 7/13 22:00 top3 은행 고정, 7/14 16:00 신규축 금지, 마지막 12h 방어만.
- **오늘 예비 1발**: sb-4th 게이트(OOF+0.0010↑/nnQ1 -0.0010내/changed 0.4~1.2%/≤560s) 통과시에만 사용 — 정보가치(내일 10발 배분 결정)가 점수가치보다 큼. 미달시 보존.

## R31 — sb 앙상블 canary GO (07-08 밤, codex 속결)
- sb 게이트 PASS(4멤버 +0.0041/nnQ1 +0.0041/changed 0.93%) but 용량 불가(m2=base 260MB, 3-large >1GB) → 배포가능 전수 시뮬: **m1+sb조건부 w[0.55,0.45] = +0.00628/nnQ1 +0.00526/changed 2.00%** 최적. bias는 tri_cond LB검증본(자동적합 대비 holdout +0.0006).
- codex 판정 **GO**: changed 2.0%는 교정 아닌 멤버스왑이라 캡 비적용. 예상 LB 중앙 0.7843~0.7852 / 보수 0.7832~40 / 하방 0.78266 미만 가능(오염+다양성손실 동시). 결과별 내일 8발: ≥0.785→sb 4~5발 / 0.783~785→sb 2~3발 / <0.78266→sb 1발 진단만·specialist 주력.
- mun-jtest 30k 검증: **A6000 138s→환산 362s PASS**(tri_cond 427s보다 고속), VRAM 10888MB PASS, holdout 0.81261(시뮬 패리티 -0.0006), 분포 붕괴 없음(14클래스 전부).
- Q3 선반영: **sb+m2+m3(m1 제거) 시뮬 +0.00782** — sb가 m1 상위호환(m1에서 FT했으니 자연). submit_sb_tri.zip 899MB 조립·검증대기 → 내일 1착 후보.
- 오늘 소비: 프로브3 + tri_sb 1 = 예비 0. 내일 10발 재개.
- **R31 LB 실측 (21:04): submit_tri_sb = 0.78098 (-0.00168 vs tri_cond) — 사전등록 최악밴드.** 분해: sb 기여 = largeonly 대비 +0.00047(보수 retrieval급). sb의 holdout +0.00628은 오염(2ep 추가 암기) 확정, nnQ1 동반상승도 방어 못함. **교훈(3번째 실증): FULL모델 holdout 델타는 슬라이스 불문 신뢰 불가 — 클린 fold 평가만 유효.** 사전등록대로 sb축 감량(≤1발 진단), 주력=specialist fold0+tri_cond 방어. submit_sb_tri 기대값도 tri_cond±0.001로 강등.

## R32 준비 — specialist 사망 + 최종 주간 배분 (07-08 심야)
- **specialist fold0 클린 프로브 NO-GO**: 표적셋 spec 0.360 vs base 0.375(-0.015). 저마진행 = aleatoric(클러스터 라벨일치 69%·R6 61% 무신호와 정합). R30 1순위 축 폐쇄 — 제출 0발 소모.
- 측정 완료된 축 전황: prior 종결 / cluster +0.0015캡 / sb +0.0005(LB실측) / specialist NO-GO / retrieval→tri 평탄. **남은 것 = 그라인딩·이종백본·은행방어.**

## R32 — 최종 6일·60발 배분 확정 (07-08 심야, codex xhigh)
- 재배분: **그라인딩 42 / sb_tri 진단 1 / 이종백본 fold0 게이트 1(+통과시 6) / au·bias 4 / 예비 6.** specialist·sb확장·retrieval·prior 재공략 전면 금지. FULL-holdout 델타 기반 의사결정 금지(3연속 반전).
- 그라인딩 기대순위: ①m2 weight sweep(최우선 8~12발, ±0.03/±0.06 좌표탐색) ②m1/m3 micro-sweep ③m3 문턱 0.45/0.55 ④dist3·s2/s3 저비율(5~15%) correlation break ⑤lgb 2.5~7.5% ⑥au-bias 상방티켓 2~4발.
- sb_tri 1발 GO: EV +0.00015~25, 신기록권 0.7828~0.7832. 양수여도 sb 확장 금지(weight sweep feature로만).
- 이종백본(mdeberta류): klue와 다른 점 = 오차 상관구조. fold0 클린 프로브 통과시에만 3~8% 혼합 제출.
- 동결 규칙: 20발 연속 <0.7829 → 조기동결 / ≥0.7832 → 7/14까지 지속 / ≥0.7838 → 계정최고 추격권.
- 판정: "0.80 경로 없음. 0.78449 초과는 낮은 확률의 grind 문제."
- **7/09 물량 사전조립 완료(9개)**: submit_sb_tri(899MB) + tc_m2p03/m2m03/m1m03/m1p03/m2p06/m1m06/tc_mth45/tc_mth55 + tc_aubias(0.5·tri+0.5·au바이어스). 전부 rebuild 기반·check PASS.

## R33 — 성패 통일이론 수립 (07-09, 사용자 지시)
- **STRATEGY.md 신설**: 실패 5대 메커니즘(M1 분포이동 승자의저주 / M2 암기오염 / M3 목적함수 기하불일치 / M4 신호포화·알레아토릭 바닥 / M5 약멤버 희석·가중공간 비볼록) + 성공 4대 메커니즘(S1 결정규칙 최적화 / S2 강-강 조건부 분산감소 / S3 히든 실존신호 수호 / S4 측정>추정·사전등록). 전 실측 40여 건 총정리.
- **운영 규칙**: R33+ 모든 codex 브리핑에 필터표 인라인 — 신규 제안은 "어느 M에 왜 안 걸리는가" 명시해야 채택. 채택문턱 +0.0005(선택편향 보정 상향).
- **최종선정 규칙 신설(R18 쟁점B 반영)**: 은행1=tri_cond 0.78266(저탐색) + 은행2=그라인딩 최고 — public⊊private 세계와 public=private 세계 모두에서 후회 최소화.
- 실행: 그라인딩 10발 조립분부터(R32 캘린더) + 이종백본 mdeberta fold0 클린 프로브(잔차 상관 게이트) 발사.
- **R33 codex 적대검증**: 이론 승인 + 수정 2건 — M6(public 다중비교 승자의저주: 42발 max선택=+2.1σ 편향 → 은행2는 사전등록최고·LB는 거부권만, public 그라인딩 10~15발 캡) + S5(희소 고정밀 패치: retr_v6 = 0.49%게이트×행당+9.2%p — S4 안전성과 별개의 이득기전) 신설, M4 완화(알레아토릭 "비중 높음", 잔여 추정분산 +0.0007 실재). 이중은행 승인(p(pub≈priv)=0.55, 기대후회 A+B 최소). mdeberta 게이트 락(ρ_low<0.82/강0.75/≥0.90금지, fold0>기준-0.0015, 기본 5%혼합). "새 대형 아이디어 없음" 확인.
- **오늘 10발 사전등록**(제출 전 기대 고정): sb_tri +0~+0.0005 / m2±·m1±·m3± 곡면평평 ±0.0005(R12 정합) / m2p06 하방리스크(M5) / mth45·55 ±0.0002 / aubias +0.0007±0.0016 고분산. 이 기대와의 잔차가 판단 재료 — max-pick 금지(M6).
- **R33 보정(사용자 지적 — 발수 낭비)**: 고정 10발 일괄발사 철회 → **적응형 5+α**: 1차 = sb_tri·m2p03·m2m03·m1m03·aubias(서로 다른 4축 방향 확정) / 2차 = ≥+0.0005 방향만 refine(m1p03·m2p06·m1m06·mth55는 조건부, mth45 사실상 폐기) / 전축 평탄이면 "곡면 최적" 결론 확보 후 조기동결. 기대수확 동일, 3~5발 절약. M6 자기적용.

## R34 — S5 메뉴 + mdeberta 종결 + ρ 재조정 (07-09, GPU병행 토론)
- **mdeberta fold0 실측**: 0.6853(-0.063), ρ_low 0.934, 클린 5%혼합 +0.0004 but 3멤버 시뮬서 기존 m2(base-e5 +0.0018)에 열세(+0.0006) → **이종백본 축 제출 0발 종결**. Part3 가설(a) 폐쇄 — 0.79+ 팀의 비밀은 백본 다양성 아님.
- **★ρ 재조정(내 실측이 codex 예측 반박)**: 작동하는 쌍 v6↔v4-large의 ρ_low = 0.959 > mdeberta 0.934. codex는 mdeberta ρ 0.78~0.84 예측(오차 큼), ρ<0.82 게이트면 v4도 금지됐을 것. **앙상블 이득의 실체 = 탈상관이 아니라 강도 동급**(M5·S2 정합). ρ게이트는 참고지표로 강등.
- **S5 메뉴(codex)**: (b)soft transition penalty(fold별 재계산, eps 1e-4~1e-3·저마진 tau만 — 내 hard-zero 측정은 위반 0이었으나 soft는 미측정) (d1)명시적 금지문구 veto (d2)action precondition veto (d3)duplicate-failed-action veto. 승인기준: gate<1%·OOF exact ΔF1+·fold 반복성·사전등록·수정금지. measure_patch 프로토콜 락. 누적 기대 +0.001~3.
- **불가능전이 hard-zero: 위반 0 실측**(노출≥500) — 모델이 전이구조 완전 흡수(M4 재확증). soft 버전만 잔존.
- **codex 0.795 역설계**: 강화 packing/TAPT + 이종 강교사 + constrained decoder + S5 누적 = 0.790~0.796 (우리 실측과 무모순). 6일 내 재현가능분 = S5 sweep·transition/precondition decoder·max_len 대체 시뮬.
- **1차 5발 flat 확률 70~80%(codex 사전등록)**. flat이면 남은 ~50발 채우지 말 것: S5 5~10발·max_len m3대체 확인·decoder 3~5발·sanity 3~5발만, 나머지 미제출(M6).
- GPU 다음 수: v6@384 fold0(강도동급 다양성, m3슬롯 비교) — mdeberta와 달리 동급 강도 전제라 S2 자격.
- **S5 스윕 실측(R34 후속, 제출 0발)**: 4후보 전부 공간없음 — soft-transition 1~4행 / 금지문구 3행 / dup-failed 음수(재시도가 정답 우세) / precondition 3행. **텍스트 유도가능 규칙은 전부 모델에 흡수됨(M4 확장). S5는 "모델 비접근 정보"(retr_v6의 near-dup 조회)에서만 성립** — 그 공간은 히든 near-dup 빈약으로 이미 소진. S5 축도 종결.

## R35 — 잔여공간 최종점검 + 동결 프로토콜 락 (07-09 새벽)
- **Q1 신규 미시도 없음 확정**(codex): TAPT/continued-pretraining no-go 유지(5일 리스크·M1 노출·기대값 음수). 남은 것 = 1차 5발 + v6@384 m3슬롯 비교 + 동결.
- **Q2 384 게이트 락**: fold0 ≥ 0.7470(완화 없음 — 입력확장 변형). 통과시 A(m1+m2+m384) vs B(m1+m384+m3) 클린 폴드 시뮬, **A ≥ B+0.0004일 때만 v4 교체**. m2 제거안은 기본후보 아님(base-e5 이득 실측 확인됨).
- **Q3 동결 프로토콜**: D-5(내일)=사전등록 5발만, flat이면 탐색 중단 / D-4=은행 zip 재제출 sanity 1발(hash·행수·분포 확인) / D-3=최종후보 2개 재현성 / D-2=cold-run·시간마진, 신규실험 금지 / D-1=최종 2개 확정, 제출 0발 / D-0=상태확인만.
- **Q4 은행2 = largeonly+retr(0.78096)**: 사전등록 초과분 부재시. tri_cond 재제출은 동일노출이라 private 헤지 약함 — 구조가 다른 2위가 분산 매수에 우월.

## R36 — 조원 0.78522 돌파 레시피 입수 (07-09 00:02, 읽기전용 조사)
- **submit_cc_tri_mdeb LB 0.78522 = 계정 신기록**(+0.0007 vs 조원 tri_m2new, +0.0026 vs 내 tri_cond). 구성(mun-train 읽기전용 확인, 무수정): m1=xlm-r-large v6(w0.6) + **m2=mdeberta-v3-base(v6, w0.15, vocab prune 58366, 271MB)** + m3=xlm-r-large v4(w0.25), **cond_members=[1,2]** — m2·m3 둘 다 저마진 조건부.
- **핵심 레시피 차이: mdeberta는 12ep+FGM으로 수렴시켜야 강도동급** — 조원 fold0: 6ep+FGM 0.7126 / **12ep+FGM 0.7465**(large 0.7485 근접). 내 6ep noFGM 프로브 0.6853은 수렴 전 중단이었음(M4 판정 아님 — 측정 오류). FULL=12ep FGM prune 283MB.
- **이론 정합**: S2(강도동급 달성) + M5 방어(약할 수 있는 멤버를 저마진 조건부로만 개입 — cond_members 확장). 내 클린시뮬 mdeb 10% 전면혼합 +0.0006이 저마진 한정으로 +0.0008 LB 실현.
- **우리 트랙 이식(자체 학습, 산출물 미복사)**: train_full_cli에 FGM 이식 완료 → mdebfull(12ep FGM v6@320 prune) FULL 학습 gpu_when_idle 체인 발사(384 종료 후 자동). 조합 후보: ①m1+mdeb(cond)+m3 재현 ②384 게이트 통과시 m1+mdeb+m384(977MB, 조원 구성+우리 384 결합) ③+aubias postproc.
- **dev 브랜치 대조(사용자 지시) 완료**: 레시피 일치 확인 + 추가수확 — ①전량통과 697s FAIL→이중조건부 486s 구제(cond[1,2]의 이유) ②프록시→LB 전이율 40%(+0.0018→+0.00073, 시뮬 보정계수) ③w2=0.20 변형은 조원측 사후튜닝 보류 중 ④**LB 급변: 1위 0.79796·컷 0.78847·계정 0.78522=20위 컷밖** ⑤조원 신축: klue-large(ko 69% 근거)·koelectra·int4·증류. b128 OOM까지 양 트랙 동일 재현(자기 OOM, 외부침입 아님).
- **재계획**: 1차 5발(tri_cond 기반 sweep) 보류 — 은행이 tri_mdeb로 바뀌어 낡은 구조. 우리 몫 = mdebfull 재현 후 조원 미발사 변형(aubias-on-tri_mdeb 등)만. 쿼터는 조원 큐(klue 등)와 분할. 역할: 조원=ko백본·int4·증류 / 우리=mdeb재현+후처리 변형+동결설계.

## R37 — 은행교체 후 트랙 재편 확정 (07-09, codex)
- **1차 5발 전면 보류 승인**: tri_cond 기반은 신은행 대비 -0.00256이라 컷추격 효율 상실. aubias-on-tri_cond도 기본 보류(전이율 40%·분산 ±0.0016이면 1발 값 안 됨).
- **우리 변형 우선순위 락**(mdebfull 완성 후): ①aubias-on-tri_mdeb(고유, 기대 LB +0.00028±64 → 0.78550) ②th 0.45/0.40 ≤2발 ③w2=0.20 후순위(조원 중복 위험) ④동시 미세변형은 단독 양수 확인 후만.
- **동결·발사 기준 재설정**: 은행1=tri_mdeb 0.78522 / 은행2=자체 mdebfull 재현이 ≥0.7850일 때만 / **발사기준 기대LB ≥0.78545 / 폐기기준 proxy<+0.0005 or LB기대<+0.0002**. R32의 0.7829 동결선 폐기 → 0.7852 주변.
- **정직한 상한**: 우리 트랙(후처리·조합)만으론 +0.0003~0.0015 — 컷 격차 0.00325 단독 봉쇄 비현실. **컷 진입은 조원 klue/koelectra/증류/int4에서 +0.002+ 필요. 우리=보조트랙(은행 안정화·중복회피·+0.001 복구).** klue 실패 시나리오 = 계정 컷밖 고착 가능성 높음.
- **운영 지시(사용자, 07-09)**: 제출쿼터 조율은 팀이 알아서 함 — 우리는 점수 향상에만 집중. R37의 쿼터 양보 논리 해제(보류 판정의 가치 논리는 유지). w2=0.20 중복회피 제약도 완화 — 가치 기준으로만 발사안 제안.

## R38 — 자체 mdeb 재현 + aubias 발사 승인 (07-09 아침, codex)
- 재현 실측: mdebfull 12ep FGM 283MB. 기준본 jtri_mdeb holdout 0.80856(+0.00154), 환산 411s PASS. 변형(동일멤버 델타): **aubias +0.00286** / w20 +0.00015 / th45 -0.00013 / th40 -0.00022.
- **aubias 발사 승인**: bias-변경 전이율 중앙 60%(밴드 20~85, 상단 100). 기대 LB = 0.78522 + 0.00286×t → **중앙 0.78694**, 20%도 0.78579(발사기준 0.78545 도달에 필요 전이율 8%뿐).
- **사전등록 해석표**: ≥0.7860 상단=전이확인→aubias 새 기준+alpha 스윕(0.625/0.75, +0.0005↑만 발사) / 0.78545~0.78599 중단=확장보류·기준본 사후진단 / <0.78545 하단=확장중단(0.78522 미만이면 반전 or 재현 confound → 기준본 발사로 분리).
- w20·th45·th40 폐기 확정(w20은 100% 전이해도 기준 미달). 조원 w20 프록시 +0.0007과의 괴리 = fold-OOF 프록시 vs 동일멤버 holdout 델타의 계열 차이 — 내부 기록: "외부 프록시 양성, 내부 검증 미확인, no-submit".
- 발사 순서 락: aubias 단독 선발사 → 상단이면 기준본 생략.
- **R38 LB 실측 (10:09): jtri_mdeb_aubias = 0.78221 — 최하단 미달(-0.0030 vs 조원본).** 동일멤버 holdout 델타 +0.00286조차 반전: bias-only 변경의 반전 첫 실측. 후보 원인: ①재현본 자체가 조원본보다 약함(프록시 동급이었으나 LB 미검증 confound) ②aubias 반전 — λ_au=0.130 혼합외삽(3점 적합·스프레드 0.169 약신호)이 틀렸거나 au-가중 재적합이 M3-인접(가정분포 정렬)이었음. **사전등록 분기 발동: 기준본(jtri_mdeb, tri bias) 발사로 원인 분리.** 기준본 ≈0.78522면 aubias 단독 -0.0030 확정(축 폐쇄+λ재평가), <0.7845면 재현품질 문제(mdeb 시드/프룬 차이 조사).

## R39 — 3자 삼각측량: aubias 반전 해부 (07-09, me ⇄ codex ⇄ 독립 Claude 레드팀)
- **레드팀 결정 발견 ①앵커 오류(40%)**: 발사 zip 계보 = 우리 tri_cond_rebuild 멤버(LB 0.78266 계보), 조원 0.78522 계보 아님(조립 스크립트로 증명). 올바른 기대선 ≈ 0.78266 + mdeb스왑×전이40% ≈ **0.7833** — 사전등록 밴드가 시작부터 +0.0019 과대. ②**유령 델타(30%)**: 클린 OOF 재계산 = aubias **+0.00002**(holdout +0.00286의 1/143), au행조차 -0.00077. bias 큰 이동은 미측정 클래스(write +0.70, respond -0.80)에 몰림 — 소표본 coord-ascent 과적합. ③잔여 -0.0011 = 재현품질 후보(우리 mdeb 12ep FGM은 클린 fold0 부재).
- **codex(병행 독립)**: 주범 (d)동일멤버 델타 결함 38% — 방향 일치. 단 앵커 오류는 못 봄.
- **이론 교정 3건(레드팀 적발, 채택)**: ①**M2 확장 — "동일멤버 델타=암기상쇄"는 무근거 발명**(em075가 1호 반증이었는데 M3로만 분류, aubias가 2호). 클린 분포에서 잴 수 있는 델타는 클린 측정만 유효. ②**M3 재정의 — "히든 분포 가정을 결정규칙에 주입하는 모든 시도" 금지**(dual/em075/damped/aubias 4전4패; 성공한 bias는 가정 무개입·64k 대표본). ③**M6 확장 — 추정기 선택도 다중비교**(R37 +0.00028 등록 → R38 오염 델타로 6배 상향을 무해명 통과). + **사전등록 계승 규칙 신설**: 제안이 과거 락(R29 스프레드·R30 캡)을 위반하는지 필터표에 상설.
- **384 재개봉(레드팀 2순위)은 기각**: 레드팀이 못 본 클린 A/B 실측 존재(A -0.0023/B -0.0005, 07-09 새벽) — DEBATE 기록 누락이 원인(본 라운드로 정정). 축 폐쇄 유지.
- **기준본 진단 재앵커(채택)**: 중앙 **0.7833**, 밴드 [0.7827, 0.7842]. 해석: 0.7827~0.7842(60%)=계보격차 실증→aubias 순효과 -0.0005~-0.0020 확정·우리 계보 변형의 은행 기여 수학적 사망→슬롯 재배분 / ≥0.7845(15%)=변형 부활 / <0.7825(25%)=재현품질 문제→mdeb 전 계획 중지.
- **레드팀 3순위(채택)**: w20/th 델타도 같은 오염 도구 측정이었음 — 조원 fold-프록시 +0.0007이 옳고 우리 +0.00015가 오염이었을 가능성. 클린 재감사(단 mdeb 12ep 클린 OOF 부재 한계). + 조원 발사물 사전검증 인프라 제공(EV가 우리 변형의 수 배) + 동결 집행.
- **R39 기준본 LB 실측 (10:34): jtri_mdeb = 0.78283 — 재앵커 밴드 정중앙 적중(60% 시나리오).** 확정: ①계보 격차 -0.0024 실증(레드팀 앵커분석 승리 — codex·me 둘 다 놓쳤던 것) ②mdeb 스왑의 우리계보 효과 +0.00017(프록시 +0.00154의 11% 전이 — 조원 40%보다도 낮음: 계보 멤버 품질이 지배) ③aubias 순효과 -0.00062 확정, au-가중 계열 영구 폐쇄 ④우리계보 상한 0.78283+0.0015=0.7843 < 은행 0.78522 < 컷 0.78847 — **우리계보 변형의 은행 기여 수학적 사망, R37 은행2 규칙 자동 불성립.**
- **사전등록 분기 집행**: 우리 트랙 잔여 역할 = ①조원 발사물 사전검증 인프라(697s FAIL 적발 전례, EV 최대) ②동결 프로토콜(R35)·최종 이중은행 선정 집행 ③점수사냥 제출 중단(정보발 제외). 3자 토론 체제(codex+독립 레드팀)는 중대 판정마다 유지.

## R40 — 총공세 재편: int4 완성 + 멤버 농사 포트폴리오 (07-09 오전, 사용자 전권 지시)
- **사용자 지시**: 조원 방법·산출물 참고 전면 허용, 모든 방법 동원 최고점. GPU 상시 가동.
- **int4 구현 완료(우리 선점)**: sim/quantize_member_int4.py(group-64 symmetric nibble-pack) + ad_lib p4 복원경로. 실측: m1/v4 353→**171MB**, mdeb 249→**107MB** — 현행 tri 구조 449MB, **6멤버 캐스케이드 1GB 성립**. 복원오차 mean 0.005~0.009 → parity 게이트(GPU 복구 후) 필수.
- **codex R40 포트폴리오**: GPU0 1순위 = **v6-large 12ep noFGM fold0 수렴 프로브**(mdeb 12ep 도약의 xlm 버전 검증 — Green ≥+0.0040 vs fold0 6ep 0.7485 → m1 전면 업그레이드 / Yellow +0.0020~40 / Red <+0.0020 농사 중단. FGM은 noFGM Yellow+일 때만 분리 진단). 2순위 mdeb777, 3순위 v4 신시드. deberta-v3-en·koelectra·klue는 GPU0 배정 금지. int4 실패 시 "top3 교체 가능 재고"만 농사.
- 충돌 규율: 산출물 prefix 분리·cascade config 불변 파일·최종제출 단일소유 규칙.
- NVML 2차 사망 — CUDA 복구 감시자(체인: v6 12ep 프로브→mdeb777 FULL) 재무장. docker restart 대기.

## R41 준비 — 조원 R67~R69 정독 + int4 재판정 대기 (07-09 오후)
- **조원 신정보**: klue-large fold0 **0.7446 완주**(m1급 -0.004, 이종 강멤버). kf-deberta-base 발견(YNAT 87.51, 차기 프로브). koelectra 병행. **조원 codex의 int4 기각은 문헌 근거(auto-gptq 미지원·bnb 의존·PTQ 0.3~1pt)** — 우리 구현(group-64 weight-only·메모리 내 fp16 복원·의존성 0, 검증된 int8과 동일 경로)에는 부적용. 유효한 우려는 "저마진 argmax 뒤집힘"뿐 → parity 게이트로 실측 판정.
- **용량 프론티어 대조**: 조원 소형-ko fp16 노선은 1GB 재초과(m1+mdeb+koelectra+kfdeb≈1.1GB). **우리 int4 통과 시 전 멤버 805MB 수납** — 두 트랙 곱셈 가능.
- GPU0 = 사용자 타대회 학습 점유(오인 정정: 조원 아님). 대기열 재구성: 유휴 → **parity 2건(mdeb·m1, 사전등록 게이트: argmax≥99.3%·ΔF1≥-0.0005·저마진 flip≤3%·14클래스)** → v6 12ep fold0 → mdeb777.

## R41 — C0 캐스케이드 락 + 크리티컬 패스 전환 (07-09 오후, codex)
- **C0 락**: m1(v6) + klue + mdeb, 전부 int8, 868MB, **w 0.55/0.25/0.20, th 0.50, 구 global bias 유지** — 첫 제출에서 변수 최소화. 기대 LB **0.7862**(클린 +0.0025 보정 × 전이 0.40, 밴드 0.7857~69, 스트레스 +0.0001~5). 시간 사전등록 430s(420~455).
- v4 구제 대안 실측 폐기: V(klue+mdeb+v4) -0.0047 / V2(v4 m1) -0.0034 vs C — **v6=대체불가 m1, klue의 이득은 v6와의 탈상관(0.922)에서 옴.**
- int8 게이트 production 기준 재캘리브레이션(ΔF1 무손상 = pass, argmax/flip은 진단값).
- 순서 규율(M6): C0 제출 → v6 12ep Green(≥0.7525)시 **C1 = m1만 교체**(가중 동결) → 그 다음에야 가중 재최적화. koelectra 승격은 ΔOOF ≥+0.0010.
- **크리티컬 패스 발견·전환**: klue FULL 멤버 부재(조원 fold0는 확률만) → 체인 mdeb777 취소, **klue-large FULL(10ep FGM b64) 대기자 장전**(v6 12ep 종료 후 자동). v6 12ep 중간: **ep7 0.7500 — 6ep 기준 0.7485 이미 돌파**, Green 가시권.

## R42 — 레드팀 3연타: klue 조기피크 적발 → 조립 사양 전면 수정 (07-09 저녁)
- **레드팀 최우선 발견**: klue는 조기피크 곡선(조원 fold0 ep7 0.7446 → ep10 0.7330, **-0.0116**). klue_f0.npz 0.7446 = best-epoch 저장본(teacher_cli), 예약했던 FULL 10ep(최종에폭 저장)은 **~0.733짜리 배포 = v4(0.7399)보다 약해 스왑이득 증발/음수** — 제출권 소각 1순위 결함. **수정: b32/7ep 재장전 완료**(피크 에폭 고정이 FULL의 유일한 방어).
- **가중 수정**: R41 락(0.55/0.25/0.20)은 클린 -0.0013 실측 + M6 동시 2변경 → **은행 가중 유지 0.6/0.15/0.25(klue를 v4 슬롯 w0.25)**. 순수 스왑 클린 델타 +0.0050(가중변경 오염 제거).
- 레드팀 검증 통과 항목: 구 bias = v6-OOF 적합본으로 양 패키지 소수점 동일·C0 분포 실측 +0.0040(유지 옳음) / 파이프라인 완전 호환(양자화 스킴·md5 일치) / klue UNK 신화 기각(0.011%) — 대신 **절단율 59%(vs 26.5%)** 발견 → 전이율 하향 근거.
- **사전등록(레드팀 재작성)**: 정직 클린 스왑 +0.0035[+0.002~+0.005] × 전이 0.30[0.20~0.40] → **중앙 0.7863, 밴드 [0.7855, 0.7872]**. 해석표: ≥0.7865 고전이→C1+koelectra / 0.78545~0.7865 은행갱신 / 0.7845~0.78544 klue 순기여0·재고강등 / <0.7845 품질사고 분리진단.
- 구성 확정: **조원 m1(q8 370MB) + 조원 mdeb(fp16 283MB) + 우리 klue(int8 ~341MB) ≈ 994MB**, cond[1,2], th0.5, 구 bias. 시간 사전등록 430s 밴드 [410,470](klue 패딩 +10% 실측 반영).
- v6 12ep 진행: ep9 0.7500 플래토 — Green(0.7525) 미달 유력, C1은 Yellow 조건부.
- **codex R42 맞서명(3자 합의 완료)**: ①7ep b32 승인, 8ep 기각 — FULL 70k는 에폭당 업데이트 +25%라 FULL 8ep ≈ fold 10ep 붕괴구간 ②은행가중 유지 승인 ③사전등록 밴드 승인(0.78522+0.0035×0.30=0.78627≈중앙 0.7863) ④klue max_len 확장은 차라운드 분리(320/336/384 ladder, M1 게이트 별도).
- **v6 12ep fold0 = 0.7541 Green PASS**(ep11 피크, +0.0056 vs 6ep): xlm-large 8ep 미수렴 실증 — R40 1순위 프로브 적중. C1 재료 = v6 FULL **11ep**(조기피크 규율, ep12 -0.0032 하락 반영) 대기자 장전(klue 뒤 자동). klue FULL 7ep/b32 정상 점화 확인.
- **C0 파이프라인 (07-09 저녁)**: klue FULL 완주(599MB) → int8 319MB(오차 0.00016) → **parity 전 게이트 PASS**(argmax 99.95%·flip 0.35%·ΔF1 -0.00037) → C0 조립 0.921GB check PASS. 1차 30k 검증은 v6 FULL 학습과 경합 상태에서 실행한 **무효 측정**(VRAM 41.5GB=학습 합산, 272s 경합 부풀림 — R13 검증 큐 규율 자기위반 기록). 유휴 재검증 대기자 장전(v6 FULL 종료 후 자동).
- **LB 갱신(조원, 07-09)**: cc_tri_w2 **0.78554 신은행**(w2=0.20, +0.0003) [⚠️R47 정정: 'w2'는 "w_mdeb=0.20"이 아니라 **가중세트 v2** — 실구성 0.55/**0.30**/0.15 + **th0.4**(mdeb 0.15→0.30·v4 0.25→0.15·th 3노브 동시변경, 원자료 run_meta 확인). 우리 구상 "w2=0.20"(위 L304)과의 이름충돌 오기. +0.0003은 mdeb 단독 귀속 불가] [⚠️⚠️R48b 재정정(운영자 확인): **tri_w2는 제출된 적 없음 — 0.78554는 유령**. 실계보 0.78522(tri_mdeb)→0.78567(C0), 아래 "+0.00013 vs w2"는 실제 **+0.00045 vs tri_mdeb**] / cc_tri_koel 0.78406(koelectra 편입 음수 — base급 한계 실증). **시간캡 재조정(사용자 지적)**: 실채점 최장 8:56(536s) 정상 통과 — 실캡 600s, 내부 540s 게이트는 보수였음. C0(추정 432s) 검증 생략 제출 승인(사용자).
- **C0 발사(사전등록 갱신)**: 은행 0.78554 기준 — 기대 중앙 0.7863은 유지(앵커는 tri_mdeb 0.78522 + v4→klue 스왑이므로 불변). 해석: ≥0.7865 고전이→C1 즉시 / **0.78555~0.7865 은행갱신** / 0.7845~0.78554 klue 순기여 미달 / <0.7845 품질사고 진단.
- **C0 LB 실측 (17:44): 0.78567 — 계정 신기록**(+0.00013 vs w2 은행). klue 순기여 +0.00045, 전이 ~13%(밴드 하단이나 양수·은행갱신). 380s(여유 大). 사전등록 분기 = C1 진행(m1→11ep 단일교체). 3자 검증 체계의 첫 실전 성공 — 레드팀 조기피크 방어가 없었으면 음수였을 것.

## R43 — C1 사전등록 + 후속 레버 정리 (07-09 저녁)
- **C1 락(codex)**: C0에서 m1만 우리 v6-11ep로 교체. 밴드 [0.7853, 0.7880] 중앙 0.7864. <0.78567이면 m1 교체 폐기(우리 11ep ≤ 조원 8ep 판정). 상단이면 후속 레버 검토.
- **교정(me)**: codex "4멤버 v4 재편입"은 용량 불가(m1+klue+v4 3-large = int8 1185MB > 1GB — mdeb int8 변환해도 초과). 실제 후속 = kf-deberta(조원 fold0 대기, base급 271MB로 4멤버 성립 가능) or 가중 미세조정 or 동결. koelectra 제외(LB 음수), klue/mdeb 에폭연장 보류.

## R44 — 3자 총력전 회의: 0.80 불가 판정, 컷 돌파 로드맵 확정 (07-09 밤)
- **합의 판정**: 0.80 P<1%(전 레버 상단 동시적중도 0.792), 0.798 P≈2%. **작전 목표 = 컷 0.78847 돌파(P 35~45%) + 0.79 진입 시도.** 잔여 갭은 6일 내 재현 불가한 구조축(외부데이터/재생성 추정) — 추격 시도(신규 직렬화·증강)는 0승5패 계열이라 금지.
- **레드팀 신규 레버 2(codex·me 맹점)**: ①**클린 앙상블 OOF 재적합(S1 부활)** — 현 bias는 v6단독 적합본, klue·mdeb 앙상블 위 bias/w/th 재최적화 미실행. 멤버 5-fold OOF 농사(klue+mdeb+v6, ~26h+) 후 재적합. 기대 LB +0.0003~0.0015, 잔여 최대 EV. 부산물=증류 soft label 무료 보존 ②**klue@384 fold0** — 384 폐쇄는 v6(절단 26.5%) 증거뿐, klue 절단 59%는 딴 레짐(R42 codex 자기유보 회수).
- **교정(레드팀)**: v4 12ep 강등(용량상 자리 없음 — codex 오늘밤 계획 오류) / 증류 기각(이중계상: student+자기교사 = 한계이득 0. 단 OOF 부산물로 fold0 ≥0.7585 무료달성시만 재론) / int4 최종기각(손상이 저마진 집중 = 캐스케이드와 자기모순) / 프로브 11발 사용 금지(M3 종결로 정보가치 0).
- **codex 산수**: 메인라인 0.7887 / 좋은케이스 0.7926. kfdeb 게이트 락(fold0 ≥0.740·ρ<0.93·클린 Δ≥+0.0015·parity·≤480s).
- **D-6~D-0 로드맵(레드팀 안 채택)**: 오늘밤 C1 발사 + GPU 체인[kfdeb fold0 12ep FGM → klue folds1-4 OOF 농사(밤샘)] / D-5 kfdeb 판정·4멤버 C2·mdeb folds / D-4 OOF 완성→**클린 재적합 그리드→최적점 1발** / D-3 klue@384 or 재적합 2발째, 자정 신규학습 마감 / D-2 동결 / D-1 이중은행 확정. LB 예산 5~7발.
- **앵커 규율**: C1 판독 전 후속 조립을 C1에 앵커 금지(R39 재발 방지). kfdeb 착수 전 조원 중복 1회 확인.
- **kfdeb 이중 사망(조원 fold0 재활용, 우리 GPU 0시간)**: 단독 0.7299·ρ_low 0.954·4멤버 델타 -0.0006 — 게이트 전항 미달. 레드팀 중복경고 덕에 프로브 재학습도 안 함. 오늘밤 GPU = klue folds1-4 OOF 농사(7ep b32 FGM, 배포 레시피 정합, ~15.5h) 대기자 장전. 이후 mdeb·v6-12ep folds로 이어 D-4 클린 재적합.

## R45 — C1 완성·발사 대기 (07-09 19:30)
- **v6 FULL 11ep 완주**: 에폭 13.6분 케이던스 클린. 단 **발사→실시작 3h20m 스톨 발견**(13:31 발사, 16:51 첫 에폭 — 셋업 단계 GPU 경합 추정). 사용자 정책 변경: **GPU0 Claude 무조건 1순위** — 이후 발사는 gpu_when_idle 대기 없이 직행.
- **파이프라인**: 661MB → q8 353MB(복원오차 max 0.0045) → parity N=2000 flip 3.37% 경계선 FAIL → **N=5000 재측정 전 게이트 PASS**(99.62%/2.90%/+0.00059/14-14) → 조립 0.921GB check PASS. 판정 원칙: 저마진 flip 게이트는 260행 표본에서 1건 차이로 뒤집힘 — 경계선이면 N 증량 재측정이 규율(int4 33%같은 파국만 즉기각).
- **C0 잔존 클린검증 기각 기록**: klue 농사와 경합 상태로 실행돼 VRAM(35.7GB)·환산시간(681s) 모두 오염 — LB 실측(380s)과 모순. 검증 무의미 확정(제출 완료 후였음).
- **발사 대기**: submit_c1_11ep.zip. 사전등록 재확인 — 중앙 0.7864 밴드 [0.7853,0.7880], <0.78567 → m1 교체 폐기·C0 앵커 유지. 시간 ~380s(C0 동일 구조·동일 멤버 크기).
- 밤샘 klue folds1-4 농사 정상 점화(19:25~).
- **C1 LB 판독 (20:51): 0.77751 — 사전등록 파국 하한 미달(-0.0082 vs C0)** → 규칙대로 **m1 교체 폐기, C0 0.78567 앵커 복귀**. 381s(시간 무결 — 구조 아닌 멤버 품질 사고).
- **사후 가설(차기 라운드 검증 의제)**: fold0 프로브는 *12ep 스케줄의 ep11 best-checkpoint*(검증 오라클)였고, C1 재료는 *11ep 스케줄의 최종에폭*(무검증) — 스케줄이 다르면 궤적·피크가 이동하는데 이를 등치한 것은 측정 아닌 추정(M6 확장: **FULL 끝점은 fold 곡선에서 외삽 불가**, klue 7ep 성공은 우연/저가중 완충이었을 수 있음). 부차 가설: FULL 70k는 fold 56k보다 에폭당 업데이트 +25% — FULL 11ep ≈ fold ~13.75ep(ep12 이미 -0.0032 하락 구간 너머).
- **영향 범위**: 오늘 밤 농사(fold 학습·best-epoch 저장)는 무관 — D-4 클린 재적합 경로 그대로. 앞으로 FULL 재학습 멤버는 (a)기존 검증된 FULL 레시피 그대로(조원 8ep 계열) 또는 (b)fold-앙상블 배포로 우회. 신규 FULL 끝점 추정 금지.

## R46 — codex 사후분석 판정 + 야간 좌표 프로브 (07-09 밤)
- **codex 판정**: H2 주범 승인(FULL 11ep = fold 13.75ep 등가, 외삽 -0.0088 member → w0.6 + 게이트 손상 경로로 실측 -0.0082 설명), H1 결합(12ep 스케줄 91.7% 지점 ≠ 11ep 스케줄 100% 지점), H3 기각(381s 동일·parity 무결). 안전상한 역산: fold 피크 11 / 1.25 = **FULL 8.8ep** — 조원 8ep가 정확히 안전권.
- **재발방지 규칙(codex 문안 채택)**: FULL endpoint는 fold best-epoch에서 외삽 금지. 신규 FULL 멤버는 LB-positive 기존 FULL의 exact replay가 아니면 배포 금지. 새 학습 산물은 fold-ensemble 배포 또는 검증된 anchor 위 후처리만. **마진 게이트 원천은 검증된 C0 m1에 고정**(멤버 사고의 게이트 전파 차단).
- **C0 앵커 승인 + klue 멤버 단위 해석 반박**: C0 zip은 앙상블 단위 실측이라 exact artifact로 유효. 단 +0.00013은 C1 손실의 1.6%라 klue 개별 기여 증명엔 부족 — klue 논법을 새 FULL에 재사용 금지.
- **재적합 전이 신뢰도 하향(Q3)**: fold-OOF 최적화→FULL 배포본 적용의 전이 오차가 노리는 이득(+0.0003~0.001)보다 한 자릿수 클 수 있음이 실증됨. 신뢰 순위: ①fold최적화→fold-ensemble 배포 ②검증멤버 고정+bias/th/게이트 소폭 ③fold최적 가중→신규 FULL 적용(준금지) ④신규 FULL이 게이트 담당(금지).
- **로드맵 재배치(Q4)**: 농사 유지(재적합+fold-ensemble 재료) / klue@384는 진단 전용 / v6 신규 FULL 9-12ep 중단 / 탐색 제출은 fold-ensemble 1·C0 고정 후처리 1·klue@384 조건부 1 / 최종선택용 2발 보존.
- **야간 좌표 프로브 발사(쿼터 자정 소멸 활용, codex ②범주 정합)**: C0 단일변경 2종 조립 — (A) wk30: 가중 0.55/0.15/0.30(klue 상향, 게이트·멤버·th 불변), (B) th55: margin_th 0.55(커버리지 ~31→36%, 가중·멤버 불변). 사전등록: 앵커 C0 0.78567, 기대 중앙 +0.0002/[−0.0008,+0.0010], 채택문턱 +0.0005(R33), 음수여도 D-4 재적합 그리드의 실LB 기울기 정보로 회수. 시간 A≈380s·B≈410s.
- **야간 프로브 판독 (21:23/21:25)**: wk30 **0.78556**(-0.00011, klue w 0.25가 국소최적 — 상향 상한 확정) / th55 **0.78581**(+0.00015, 계정 최고점 갱신이나 채택문턱 미달 — 커버리지 방향 약양성). 둘 다 평평 구간 = C0 구조가 좌표공간의 평탄 정상부에 있음을 실증. 시간 379s/391s 정상. **은행 앵커는 C0 유지**(사전등록 규칙), th55는 D-1 max-public 후보로 별도 기록.
- **다음 좌표(같은 밤, 쿼터 소멸 전)**: wm20 = 0.55/0.20/0.25(mdeb 0.15→0.20 단일변경) — 근거는 노이즈 기울기가 아니라 **조원 LB 실측 w2=0.20 +0.0003**(형제 구조 tri에서 mdeb 가중 동일 방향 양성). 사전등록: 중앙 +0.0002 [−0.0006, +0.0010], 채택문턱 +0.0005, 시간 ≈380s. [⚠️R47: 근거 허구 판명 — 발사 취소, 아래 참조]

## R47 — 3자 사후검증: wm20 발사 취소 (07-09 밤, 사용자 지적 "다른 agent들과 만든 거 맞아?"가 트리거)
- **고백 선행**: 야간 프로브 3종(wk30/th55/wm20)은 3자 검증 없는 단독 설계였음 — wm20 발사 전 codex(R47)+독립 레드팀 병행 검증 실시.
- **레드팀 V1 [치명·원자료 4중 확인]**: 조원 cc_tri_w2의 run_meta 실물 = weights [0.55, **0.30**, 0.15] + **th 0.4** — 'w2'는 "가중세트 v2"(mdeb 0.15→0.30 + v4 0.25→0.15 + th 0.5→0.4 **3노브 동시변경**)이지 "w_mdeb=0.20"이 아님. 내 L361 기록이 과거 우리 구상(L304 "w2=0.20 변형")과 이름충돌한 **오기**였고, wm20의 발사 근거("조원 LB 실측 mdeb 0.20 +0.0003")는 허구.
- **레드팀 V2 [치명]**: wm20 좌표(0.55/0.20/0.25)는 **R41 락과 동일 — R42에서 클린 −0.0013 실측으로 이미 기각된 좌표**. 우리 오프라인 이력에도 "w20 +0.00015 폐기"(jtri_mdeb_variants) 존재. 기대 LB −0.0002~−0.0005로 사전등록 중앙(+0.0002)과 정면 모순.
- **codex R47 교차**: "V1 미확인이면 NO-GO" + wm20은 실제로는 m1 0.60→0.55 **감량 실험이기도 함**(wk30 −0.00011도 m1 0.55 — 감량 무해 증거 없음). 정지 규칙 채택: **|δ|≤0.0002 flat 2연속 → 그날 좌표탐색 종료**(오늘 이미 발동).
- **판정: wm20 발사 취소**(패키지는 무결 — 결함은 산물이 아니라 논리. 보관만). 오늘 밤 좌표탐색 종료. 은행 앵커 C0 0.78567 유지, th55 0.78581은 D-1 max-public 후보.
- **교훈**: ①기록 오기가 다음 결정의 "실측 근거"로 둔갑하는 경로 실증 — 타인 산출물 인용 시 원자료(run_meta) 확인 의무화 ②"쿼터 소멸이라 공짜"는 발사 논리를 면제하지 않음 ③3자 검증은 중대 결정만이 아니라 **모든 발사**에 적용(사용자 지적이 옳았음 — 이번 검증이 없었으면 음수 기대 1발 + 오기 박제).

## R48 — 소멸쿼터 최종 소집: 레드팀이 '0발'을 뒤집다 → c0_wd30 조건부 발사안 (07-09 밤, 사용자 트리거)
- **codex 1차(R48)**: [없음] — 기각 A~F 전부 유지, 최선 후보 VOI ≤3e-5 < D-4 오염 리스크. D-4 사전등록 초안 제출(발사조건: OOF Δ≥+0.0004·3/5폴드 비음수·최악폴드 ≥−0.0007).
- **레드팀 반전 3건(원자료+시뮬)**: ①**R42 "클린 −0.0013" = mdeb 6ep 프록시(0.685) 아티팩트 유력** — 조원 컨테이너의 동세대 mdeb-12ep fold0 OOF(0.7465)로 재시뮬 시 wD축 단조 양성: 0.20 +0.0003 / 0.25 +0.0011 / **0.30 +0.0021(플래토)** ②**하네스 캘리브레이션**: D12 하네스가 오늘 밤 LB 실측 2발 부호 2/2 적중(wk30·th55), D6는 1/2 실패 — 구조내 가중 섭동 계열에서 D12 신뢰 확립, 전이 0.25~0.40 ③조원 독립 스윕(wD→0.30 +0.0010~12, V-프록시 오염 플래그) 방향 일치. E(fold0 bias 재적합)는 정직 시뮬 +0.0007 클린 → LB +0.0002~3 문턱미달로 기각 사유만 교체. F 수치 사망(cond[mdeb만] −0.0012, [klue만] −0.0010).
- **codex R48b 맞서명**: flat-2 예외 서명(좌표 채굴이 아닌 신규 증거 기반 단발), 사전등록 원안 서명, 해석은 "0.55/0.30/0.15 **net perturbation**"으로 기록(klue 0.25→0.15 동반 하향 — wk30 실측상 중립~양성 추정).
- **c0_wd30 사전등록(3자 완서명)**: C0에서 가중만 0.6/0.15/0.25→**0.55/0.30/0.15**(th0.5·cond[1,2]·구bias·게이트 m1 불변). **중앙 +0.0004, 밴드 [−0.0003,+0.0010], 채택 ≥0.78617**. 분기: ≥문턱 은행갱신+D-4 그리드중심 wD0.30 / 0~문턱 max-public 기록 / <0 mdeb-up 축 폐쇄(그리드 1차원 절약). 결과 불문 오늘 추가 좌표 금지. 조립 0.921GB·check PASS·bias md5=C0 동일 확인.
- **발사 전제 잔여 1(운영자)**: ⓐ**tri_w2 0.78554 실존 모순** — 조원 자체 기록(R77 "미제출"·큐 LB공란·궤적 0.78522→0.78567에 0.78554 부재) vs 우리 L361 기록. 유령이면 우리 계보 기록 정정 필요(C0 순증은 +0.00045 vs tri_mdeb), 실존이면 조원 미인지 재제출 위험 ⓑ잔여 쿼터 ≥1 + 조원 tri_w2 발사계획 충돌 확인(레드팀 재현실패로 tri_w2 자체는 비추천 의견).
- **조율 발견(운영자 전달 요망)**: 조원이 우리 C0 klue를 "출처불명 zip"으로 보고 **klue FULL 재학습 중**(21:38~, GPU 4h 중복) — C0 레시피(klue/roberta-large 7ep b32 FGM lr2e-5 → int8) 전달 시 중복 제거.
- **klue 농사 fold1 완주**: 피크 ep6 0.7245(ep7 0.7242) — **조원 fold0 0.7446 대비 −0.020, 폴드 편차 초과 의심.** fold2 판독(~00:40)에서 지속 시 레시피 격차 조사(조원 fold0 로그 대조) — D-4 재적합의 klue OOF 균질성 문제로 격상 가능. 감시 항목.
- **wd30 LB 판독 (23:15): 0.78621 — 채택(+0.00054 ≥ 문턱 +0.0005). 신은행·계정 신기록.** 382s. 컷 0.78847까지 −0.00226.
- **하네스 지위 격상**: D12 하네스(조원 mdeb-12ep OOF 기반) 부호 예측 **3/3**(wk30 −·th55 +·wd30 +), wd30은 크기도 밴드 적중(예측 +0.0005~0.0008, 실측 +0.00054, 실현 전이율 0.26). **D-4 재적합의 공식 사전 시뮬레이터로 승격** — 그리드 중심 wD0.30, 전이율 prior 0.25~0.30.
- **오늘 결산**: 제출 5발(C1 −0.0082 수업료 / wk30·th55 기울기 / wd30 +0.00054 채택) — 순증 C0 대비 +0.00054, 주간 계보 0.78522→0.78567→0.78621. 레드팀 원자료 발굴(mdeb-12ep OOF·유령 0.78554·w2 오기)과 캘리브레이션이 오늘의 결정적 기여. 사용자 재촉 2회가 모두 옳았음(쿼터 소멸 인식·재소집 요구).
- **내일(D-5) 계획**: klue 농사 완료(~05:30, fold2 0.7245 재현 여부로 균질성 판정) → R49 아침 소집(D12 하네스로 wd30 인근 th축·0.50/0.30/0.20 사전 시뮬 → +EV면 1~2발, 아니면 mdeb f1-4 농사 대기) → 밤 mdeb 농사 → D-4 정식 재적합(그리드 중심 wd30).

## R49 — 전면 재설계 3자 회의 (07-10 자정~새벽, 사용자 지시)
- **정직 산수 수렴(codex·레드팀 독립 도달)**: 은행 wd30 0.78621 + D-4 재적합(+0.0006~0.0013, 중앙 +0.0009) = **0.7869~0.7875 착지, 컷 0.78847에 중앙 −0.0014 부족.** 컷 P ≈ 8~12%(레드팀)/N1·N4 사망 후 codex 분기 소멸. 1위와의 갭은 이 판에 없는 구조축 — 잔여 자원은 컷 추격 신축이 아니라 **private 견고성**(플래토 중앙, 폴드 안정성)에 배분.
- **D-4 상한 실측(레드팀, D12 half-fit/half-eval)**: 조인트 +0.00305 / w+th만 +0.00216(교차 half 동일좌표 (0.45/0.35/0.20) th0.6 = 유일 안정점) / 잔차 bias +0.00026(구bias 거의 소진, 불안정) / 경계확장 시 넓은 플래토(th0.55~0.75×mdeb0.35~0.40)이나 확장하면 선택노이즈. **bias 전이율 판정**: 5fold 64k 적합+폴드 부호일치 시 0.4~0.7, fold0 단독이면 0.26 취급(codex 0.55~0.75와 합치, 부호일치를 게이트로 강제).
- **N1(mdeb 시드 이중화) 사망**: 내 발굴(우리 mdeb s1234 12ep FGM FULL 완성본 — 조립비용 0)에도 불구하고 EV 자체가 죽음 — **캐스케이드 민감도 0.044 실측**(멤버 F1 +0.06이 캐스케이드 +0.0027로 감쇠), 시드평균 기대 LB +0.00002~3, 실페어 직접측정 **−0.00039**. codex 게이트(seedB f0 3.2h)도 회수.
- **N4(klue 레시피 복구) 사망**: 조원 f0 로그 대조 — 핵심 파라미터 전부 동일(mismatch 없음). 격차 −0.02의 정체 = **폴드 난이도**(자연실험 v8 f0→f1 −0.0053, v9 −0.0162) + 런 분산. f2 반등(ep5 0.7254) 정합. klue OOF는 D12 프록시 전용(§6 옵션 c).
- **N2·N3 용량 사망 확인**(klue 2폴드 1151MB, m1 이중화 1185MB). 조원 자산 재스캔: mdeb 2호 시드 없음, v4 12ep 없음, emb 2종(kNN 보류)뿐.
- **하네스 판정(codex)**: calib.json 전체 재캘리브 불요 — 동구조=실측 앵커 원칙, 단 멤버 수 변화 시 timing replay 필수. D-4 테스트 스위트: row hash·OOF 폴드 배타성 assert·known-probe 부호 재현(wk30/th55/wd30/C1)·그리드 후보 수 사전등록·LB 후 확장 금지. **D12 하네스 적용범위 명문화: 구조내 w/th/bias/cond 섭동 전용(멤버스왑·이종구조 금지).**
- **이중은행(codex 구조 채택·레드팀 내용 수정)**: robust = wd30 + D-4 bias λ0.5~0.7 + 보수 th / max-public = 풀 조인트 λ1.0 + th max. 두 슬롯의 차이는 bias λ와 th뿐(N1/N4 삭제).
- **즉발 후보 cw45**((0.45/0.35/0.20) th0.6, 구bias): 그리드 유일 안정점, 클린 +0.00216 → LB 기대 +0.0006. codex 맞서명 대기(R49b) — 동시 2변경 M6 긴장 vs 조인트 안정점 단일좌표 논리.
- **codex R49b 맞서명(3자 합의 완성)**: N1·N4 기각 동의(정정 명기 — 민감도 0.044·실페어 −0.00039·mismatch 부재가 결정타). **cw45 오늘 즉발 조건부 서명**: "가중+th 독립 2변경이 아니라 D12 조인트 그리드 교차 half 동일선택 단일좌표의 확인런" — 중앙 +0.00055~60, 밴드 [−0.0002,+0.0010], 채택 ≥0.78671, **단발·애매하면 추격 금지·결과를 D-4 그리드 확장 근거로 사용 금지**. 로드맵 서명: GPU(mdeb f1-4→v6 folds, N1/N4/klue재현 회수, 384 최후순위)·D-4(사전등록 그리드·확장금지·플래토 중앙 선호·5폴드 부호일치 게이트)·이중은행(bias λ 0.5/1.0 × th 보수/max만)·D12 범위(구조내 섭동 전용)·잔여자원은 private 견고성(컷 P 8~12% 수용).
- **cw45 조립 완료**: 0.921GB·check PASS·bias md5=구bias 동일·weights [0.45,0.35,0.20]·th0.6 확인. 시간 예상 ~410s(th0.6 커버리지 증가 +25~35s 모델).
