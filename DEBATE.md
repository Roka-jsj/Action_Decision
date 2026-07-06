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
