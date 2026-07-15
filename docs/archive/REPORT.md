# 상세 작업 보고서 — Dacon 236694 · AI 에이전트 행동 예측 (Action Decision)

> 2026 AI·SW중심대학 디지털 경진대회 AI부문 예선. 작성 시점 2026-07-05, **LB 0.78226 (6위권)**, 마감 D-10.
> 토론·의사결정 원본 로그: [DEBATE.md](DEBATE.md) · 운영 문서: [PROJECT.md](PROJECT.md) · 실험 대장: [experiments_master.csv](experiments_master.csv)

---

## 1. 문제와 제약

**과제**: AI 코딩 에이전트의 세션 상태(구조 메타 + 대화/행동 이력 + 현재 발화)를 입력으로 **다음 행동 14클래스** 예측. 지표 **Macro-F1** (희귀 클래스가 결정적).

**제출 형식이 곧 제약**: 코드 제출(zip = `model/` + `script.py` + `requirements.txt`) → 서버(T4 16GB, 3vCPU, 12GB RAM, **완전 오프라인**)가 히든 30k를 직접 추론. **모델 ≤1GB, 설치 ≤10분, 추론 ≤10분, 하루 10회 제출.** 서버 환경: Python 3.11 / torch 2.7.1 / transformers 4.46.3 / sklearn 1.8.0.

이 제약들이 프로젝트 후반의 모든 의사결정을 지배했다 — §5의 "시간캡 발견"과 §6의 양자화가 그 결과물이다.

## 2. 데이터 핵심 (EDA 결론)

- train 70,000행 (sim 세션 64,975 + au 세션 5,025), visible test는 전부 sim.
- **텍스트 지배 문제**: 구조-only 상한 macro-F1 0.17~0.31 / prompt-only 0.435. 프롬프트(평균 61자)가 주신호, **직전 행동이 최강 보조신호(+0.089)**.
- **탐색 클러스터(read_file/grep_search/list_directory/glob_pattern) = 전체의 41%, F1 0.49~0.64** — 이 4클래스가 macro-F1 헤드룸의 사실상 전부. 나머지 클래스는 0.68~0.999로 근천장.
- GroupKFold(세션) vs Random gap ≈ 0 → 세션 누수 위험 낮음. 그래도 세션단위 CV 유지.
- 프롬프트 94% unique → 암기식 매핑 불가, 일반화만이 답.

## 3. 접근 타임라인 (점수 궤적)

| 단계 | 접근 | LB |
|---|---|---|
| 베이스라인 | TF-IDF+LogReg (공식) | 0.629 |
| 직렬화 v3 + xlm-r-base | 구조 신호를 텍스트 마커로 주입 | 0.671 |
| 교사 앙상블 + LightGBM 스태킹 | base/klue/large 5-fold + 스태커 | 0.722 → 0.740 |
| **대전환: 스태킹 폐기** | 진단 제출로 "약멤버 희석" 발견 | — |
| **large 단독 (v6-8ep FULL + bias)** | xlm-r-large, 전량 70k 8에폭 | **0.78051** |
| 강-강 확률 앙상블 (2-large) | v6-8ep + v4-8ep 가중평균 | 0.78189 |
| **large + base 가중평균** | 다양성 멤버 재평가 | **0.78226** ← 현재 최고 |

## 4. 직렬화 진화 (v1→v7)

모델 입력은 세션 dict를 단일 문자열로 직렬화한 것. 좌측절단(max_len 320, truncation_side=left) 아래에서 **중요 신호를 문자열 끝(생존 위치)에 배치**하는 것이 설계 원칙.

- v1: prompt only → v2: +직전 행동/상태 → v3: +최근 이력+메타 토큰 → v4: +수치 bin·args 확장자·open_files
- **v6 (주력)**: v4 + `[SEQ]` 행동 트레일(최근 12) + `[PFLAG]` 프롬프트 패턴 플래그 + `[NOHIST]` — 좌측절단 생존형 압축 신호
- v7 (기각): v6 + `[PACE]` elapsed/페이스 bin — fold0 실측 **-0.0082**, 폐기 (§7-③)

## 5. 확립한 법칙 (전부 실측 근거)

1. **시간캡이 앙상블 설계를 지배한다.** 서버 실측: large 1개 ≈ +230초, base 1개 ≈ +117초 (채점시간 역산). 2-large 확률 앙상블은 8분07초로 턱걸이 통과, 3-large는 불가. → 시간 예산 내 조합 탐색이 곧 전략.
2. **용량은 해결된 제약**: int8 group-64 weight-only 양자화(sim/quantize_member.py + 로드시 fp16 복원)로 large 661→353MB, **품질 무손상** (argmax 일치 ~100%, LB 검증). 1GB 안에 large 2개+base까지 적재 가능.
3. **holdout 게이트는 누수 편향**: FULL 멤버는 holdout도 학습했으므로, large 비중이 높은 후보는 암기로 **과대평가**, base 섞인 후보는 **과소평가**된다. 실증: lgb8은 holdout에서 -0.0024로 "기각"이었으나 실제 LB +0.0017로 최고 기록. → **교차-계열 비교의 심판은 LB뿐**, holdout은 동일 계열 내 비교/파이프라인 검증용으로 강등.
4. **캘리브레이션**: 6ep 교사 OOF의 개선분은 배포(8ep FULL)에서 계열에 따라 40~110% 전이. 같은-계열(large+large)은 저전이, 다양성(base) 조합은 고전이.
5. **per-class bias(coordinate ascent)** 는 안정적으로 +0.006 기여. 모든 배포에 포함.

## 6. 배포 파이프라인 (현재)

```
zip = model/{m1(양자화 large-v6-8ep), m2(base), m3(양자화 large-v4-8ep)} + ad_lib.py + run_meta.json + postproc.json + script.py
서버: script.py → ad_lib.predict()
  ├─ 멤버별 직렬화(버전별 캐시) → 길이정렬 배칭 → fp16 추론
  ├─ qweights.npz 감지 시 in-memory int8→fp16 복원 (디스크 재작성 없음)
  ├─ (조건부 모드) full 멤버 혼합 → top1-top2 마진<th 행만 조건부 멤버 재추론·재혼합
  └─ log-prob + per-class bias → argmax → id 조인 → submission.csv
```

- **조건부 앙상블**(codex R11 제안 → 구현): full 3-way가 시간캡을 넘을 때, 확신 낮은 행(~35%)에만 세 번째 모델을 투입해 이득의 ~95%를 회수하면서 시간을 예산 안으로.
- requirements.txt는 빈 파일(설치 0초, 오프라인 리스크 0). 토크나이저 전량 번들, sklearn pickle 금지(로컬 3.10 vs 서버 3.11), LightGBM은 텍스트 포맷.

## 7. 기각된 가설들 (증거 포함 — 재탐색 방지)

| # | 가설 | 결정 실험 | 결과 |
|---|---|---|---|
| ① | 순차 누수(테스트 세션 겹침으로 라벨 복원) | posmap 내장 제출 (히든 30k 직접 시험) | 점수 완전 동일 = coverage 0. **기각** |
| ② | 텍스트 재활용(reranker/ngram/원문 args) | large vs large+ngram 확률평균, 탐색4class | +0.0007. large가 이미 흡수. **기각** |
| ③ | 미사용 메타(elapsed 등)가 새 정보원 | LGB 프로브(조건부) + v7 [PACE] fold0 재학습 | elapsed 조건부 -0.0016, v7 -0.0082. **이중 기각** |
| ④ | LightGBM 스태킹 | 진단 제출 (스태커 제거 비교) | 스태커가 -0.014. **기각** |
| ⑤ | full 2-large가 시간캡 초과 (Colab 측정 719초) | 실제 서버 제출 | 8분07초 통과 — **Colab T4 측정이 과대. 부분 기각** |
| ⑥ | 10ep FULL | Colab 세션수명(~40-55분) < 학습시간 | 3회 사망. 환경적 불가 |

②③이 함의하는 것: **현 입력에서 새 정보원은 없다.** 남은 개선축은 모델·앙상블·시간예산 공학뿐.

## 8. 인프라 (재현 도구)

- **학습**: Colab CLI 자동화 babysitter(`sim/babysit_*.sh`) — 세션 사망 자동 재기동·산출물 회수. 유닛 소진 후 **Kaggle 커널**(`kaggle/k_full2/`)로 이전(서버측 실행, 9h 세션).
- **train_full_cli.py**: FULL-70k 학습 + vocab 프루닝(250k→50k) + **SWA-lite**(마지막 K에폭 가중치 평균) + **증류 모드**(teacher soft label, KL+temperature).
- **검증 게이트**: `sim/check_zip.py`(구조/1GB) → CPU 오프라인 심(코드경로) → `sim/bench_t4_hold.sh`(T4 30k 타이밍 + holdout 5.8k 채점) → LB.
- **soup**: `sim/soup_members.py` — 같은 레시피·다른 시드 멤버들의 가중치 평균(단일모델 추론비용으로 앙상블 효과).
- **GPT-5.5(codex) 자동 토론**: 매 결과마다 계획을 상호 비판(R1~R11, [DEBATE.md](DEBATE.md)). codex 제안을 그대로 따르지 않고 실측으로 반박한 사례(R8 메타 낙관론 기각)와, 내 결론을 codex가 뒤집은 사례(R11 조건부 앙상블, 게이트 하향)가 모두 기록돼 있다.

## 9. 방법론 교훈

1. **진단 제출의 가치**: 앙상블에서 멤버를 하나씩 뺀 제출(largeonly)이 23위→7위 도약의 계기. 시스템의 어느 부품이 기여하는지 LB로 분해하라.
2. **게이트의 한계를 게이트하라**: holdout 게이트가 lgb8을 잘못 기각했다. 게이트 자체의 편향(누수·측정환경 차이)을 주기적으로 LB와 대조.
3. **측정 환경 ≠ 배포 환경**: Colab T4 719초 → 서버 8분07초. 환경차를 안전마진으로만 다루면 유효한 후보를 죽인다.
4. **오염된 실험의 부분 구제**: 메타 프로브 1차는 np.empty 미예측행 오염으로 pooled 지표가 무효였지만 fold별 지표는 유효했다 — 어디까지 오염됐는지 정확히 가려내면 실험을 전부 버리지 않아도 된다.
5. **약한 신호는 깊은 통합으로만**: 얕은 corrector(+0.0068 클러스터)는 전체 macro로 환산하면 노이즈. 재학습(v7)으로 검증했고, 그마저 음수 — 신호가 프록시로 이미 흡수된 경우다.

## 10. 현재 상태와 남은 계획 (D-10)

- **은행**: lgb8 = 0.78226 (6위권). 클라우드에서 s2(seed 다양화+SWA)·dist(2-large teacher 증류) 학습 중.
- **다음 후보**: 조건부 3-way(`submit_tri_cond.zip`, OOF 0.7505, 서버 ~7분40초 추정) → soup(s1+s2) → dist → (양수 조합 시) 3-way soup.
- **제출 전략**: 저위험 후보는 즉시 은행(하루 10회), 7/13까지 실험, 7/14 저녁 최종본 동결.
- 백로그: pairwise post-hoc 경계교정(list_directory P=0.384), sim-only FULL 재학습.
