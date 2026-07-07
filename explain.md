# explain.md — Dacon 236694 (AI Agent Action Decision) 프로젝트 종합 설명 + 엄격 평가

> 2026 AI·SW중심대학 디지털 경진대회 AI부문 예선. 작성 2026-07-05, 마감 D-10(7/15).
> 본 문서는 ① 프로젝트가 **무엇을·왜·어떻게** 하는지 설명하고, ② 대회 **채점·조건·데이터셋**을 저장소 파일에서 **직접 측정한 실측치**로 엄격 평가한다.
> 원문 로그: [README.md](README.md) · [PROJECT.md](PROJECT.md) · [REPORT.md](REPORT.md) · [DEBATE.md](DEBATE.md) · [SERVER_SETUP.md](SERVER_SETUP.md) · [experiments_master.csv](experiments_master.csv)
>
> **표기 규약**: `(D)` = 이번에 데이터/코드에서 **직접 검증**한 사실. `(C)` = 문서·실험로그의 **주장**(별도 미검증).

---

## Part I — 프로젝트가 무엇인가

### 1. 과제 한 줄 요약
AI 코딩 에이전트의 **세션 상태**(구조 메타 + 대화/행동 이력 + 현재 발화)를 입력받아 **다음 행동 14클래스**를 예측한다. 평가지표는 **Macro-F1**(희귀 클래스가 결정적).

### 2. 제출 형식이 곧 제약 (D)
"예측 CSV 제출"이 아니라 **코드 제출**이다. `zip = model/ + script.py + requirements.txt`를 올리면 Dacon **완전 오프라인 서버**가 히든 30k 테스트를 직접 추론한다.

| 제약 | 값 | 성격 |
|---|---|---|
| 하드웨어 | T4 16GB GPU / 3 vCPU / 12GB RAM (C) | 고정 |
| 네트워크 | 완전 오프라인 | 고정 |
| 모델 크기 | `model/` ≤ 1GB | 고정 |
| 설치 시간 | ≤ 10분 | 고정 |
| 추론 시간 | ≤ 10분 (30k) | **유일한 구속 제약** |
| 제출 횟수 | 10회/일 | 고정 |
| 사전설치 | Python 3.11 / torch 2.7.1 / transformers 4.46.3 / sklearn 1.8.0 (C) | 고정 |

이 제약들이 프로젝트 후반 의사결정 전체를 지배한다(§II-2 참조).

### 3. 데이터셋 실측 (D)
| 항목 | 값 |
|---|---|
| train | 70,000행 / 9,429 세션 / 세션당 평균 7.42스텝(1~18) |
| 로컬 `data/test.jsonl` | **단 5행(데모 스텁)** — 5개 id 전부 train 세션 |
| 실제 히든 테스트 | 30k, **Dacon 서버 전용**(로컬 부재, 오프라인 채점 불가) |
| 생성기 분포 | sim 64,975(92.8%) + au 5,025(7.2%). 스텁 test는 100% sim |
| 콜드스타트 | history 없는 step_01 = **9,000행(12.9%)** |

**스키마(각 행)**: `id`, `session_meta`, `history`, `current_prompt`.
- `session_meta` = `{user_tier, language_pref, workspace{language_mix, loc, git_dirty, open_files, last_ci_status}, budget_tokens_remaining, turn_index, elapsed_session_sec}`
- `history` = `{role}` 리스트(user↔assistant_action 교대). user엔 `content`, assistant_action엔 `name·args·result_summary`. **최근 12엔트리로 좌측 절단**(turn_index는 최대 18까지).
- `current_prompt` = 현재 사용자 발화(평균 61자, 90.4% 고유) — **주신호**.

**클래스 분포(D)**: edit_file 16.0% / grep_search 14.2% / read_file 13.2% / glob_pattern 7.6% / respond_only 7.4% / run_bash 7.2% / apply_patch 6.9% / run_tests 6.5% / list_directory 6.2% / ask_user 3.9% / plan_task 3.8% / lint_or_typecheck 3.3% / write_file 2.1% / web_search 1.8%. **불균형 8.8배**.

### 4. 코드·디렉터리 구조
```
common/            공용 라이브러리(DRY 단일 소스, 서버 model/에 ad_lib.py 복사)
  io_utils.py      jsonl 로드, 14클래스 매핑
  serialize.py     dict→문자열 직렬화 v1~v6 (⭐v6 최강)
  ad_lib.py        직렬화+추론+앙상블/스태커 (서버 script.py가 import, 583줄)
  cv.py            홀드아웃 8% + StratifiedGroupKFold(5), splits.npz 캐시
  metrics.py       pooled-OOF macro-F1 (Dacon 정의 일치)
  postproc.py      per-class bias coordinate-ascent 후처리
  vocab_prune.py   임베딩 프루닝 250k→~50k
  leak.py          (기각) 순차 지도 — 커버리지 0 검증 후 미사용
action_decision_maximum/src/
  teacher_cli.py       교사 5-fold OOF 확률 npz (증류/스택용)
  train_full_cli.py    FULL-70k 멤버 학습→프루닝→member_<TAG>.zip
sim/               오프라인 시뮬 + 패키징 + 자동화
  package_single.py / package_ensemble.py   제출 zip 조립
  parity_check.py      ⭐제출 전 필수: 버전오지정/순서교체 침묵실패 검출
  run_offline_sim.py   네트워크 차단 설치+추론 검증
  train_and_verify.sh  서버 원커맨드(학습→패키징→검증컨테이너 자동실행)
  babysit_*.sh         Colab 세션 자동 재기동·산출물 회수
kaggle/            Kaggle 커널 트랙(무료 GPU)
eda/               분석 스크립트(에러분석, 스크리닝, 스태킹 프로브)
splits/            fold 인덱스(전 팀원 공유, 커밋됨)
```
**설계 원칙**: 직렬화·추론이 학습/서버에서 100% 동일하도록 `ad_lib.serialize`/`ad_lib.predict` 단일 함수만 사용. 제출 `model/`은 `ad_lib.py`를 복사해 자체완결.

### 5. 방법론 요약
- **직렬화 진화(v1→v7)**: 좌측 절단(max_len 320, truncation_side=left) 하에서 중요 신호를 문자열 끝에 배치. v6 = v4 + `[SEQ]`(최근 12행동) + `[PFLAG]`(프롬프트 패턴) + `[NOHIST]`. v7(`[PACE]`)은 fold0 −0.0082로 기각.
- **모델**: xlm-roberta-large(v6, 8ep, FULL-70k) 단일이 최강. base/klue는 다양성 멤버.
- **후처리**: per-class bias(coordinate-ascent). 안정적 +0.006 기여, 모든 배포에 포함.
- **용량 공학**: vocab 프루닝(250k→49,822)으로 fp16 large를 709MB로 축소(D). int8 group-64 양자화(661→353MB)는 **인프라만 존재하며 현 배포엔 미사용**(D).
- **앙상블**: 강-강 확률 평균 / 조건부 3-way(확신 낮은 행에만 3번째 모델 투입).

### 6. 점수 궤적 (C)
| 단계 | 접근 | LB |
|---|---|---|
| 베이스라인 | TF-IDF+LogReg | 0.629 |
| 직렬화 v3 + xlm-r-base | 구조 신호 텍스트 주입 | 0.671 |
| 교사 앙상블 + LightGBM 스태킹 | | 0.722→0.740 |
| **대전환: 스태킹 폐기** | 약멤버 희석 발견 | — |
| large 단독(v6-8ep FULL + bias) | | **0.78051** |
| large+base 가중평균 | | **0.78226**(README 최고) |
| 조건부 3-way(tri_cond) | | 0.78266 기록 후 **열위로 은퇴**(git a58ccbf) |

### 7. 인프라
로컬에 GPU가 없어 **Colab(A100) + Kaggle(무료 T4)** 로 학습하다가, 현재는 **연구실 A6000×2 서버의 무마운트 컨테이너 파이프라인**(SERVER_SETUP.md)으로 이행: 학습 컨테이너(온라인·소켓)가 `train_and_verify.sh` 한 방으로 학습→패키징→검증컨테이너(오프라인·12g/3cpu) 자동 실행까지 흘린다. **codex(GPT-5.5) 자동 반박토론**(DEBATE.md, R1~R12)으로 전략을 상호 비판.

---

## Part II — 대회 채점·조건·데이터셋 엄격 평가

> **실측 검증 요약 (데이터·코드 직접 확인)**
> - **로컬 채점 불가**: `data/test.jsonl`은 5행 스텁이고 5개 id 모두 `train_labels`에 존재 → 실제 30k는 서버 전용. 모든 실측 신호는 일 10회 제출을 소비해야만 얻어짐.
> - **8.8× 불균형 + 등가 가중**: macro-F1은 14클래스를 각 1/14=7.14%로 가중 → 예제당 leverage가 edit_file 0.45 ~ web_search 3.92로 벌어짐.
> - **콜드스타트 12.9%**: 9,000행이 history 없는 step_01(현재 프롬프트+메타만). 단 이 수치는 **train에서만** 확인, test 분포는 추정.
> - **모호성 하한**: 서로 다른 라벨로 매핑되는 프롬프트 2,219개 → last_action 결합 시 **439 충돌키로 80% 감소** → 라벨노이즈는 구속 제약이 아님.

### 1. 채점(Scoring) — Macro-F1 정밀 평가

**집계 구조와 공식 일치성 (D).** `common/metrics.py`는 클래스별 TP/FP/FN → `p,r`(분모 0이면 0) → `f1`(p+r=0이면 0) → `f1s.mean()`을 반환한다. 14클래스가 전부 support>0인 실제 30k 테스트에서 이는 sklearn `f1_score(average='macro', zero_division=0)`와 **수치적으로 동일**함을 순수 파이썬 재구현으로 확인했다. 단 한 가지 규약 주의: metrics.py는 **고정 14라벨** 평균을 쓴다. 만약 서버가 "관측된 클래스만" 평균한다면 값이 달라지지만(예시로 0.1286 vs 0.6000까지 벌어짐), 30k엔 14클래스가 모두 등장하므로 **안전**하다. 또한 폴드 이어붙여 1회 계산하는 **pooled-OOF** 방식이라 mean-of-fold-F1과 다르며 그 차이는 희소 클래스에서 가장 크다.

**희소 클래스 leverage (D).** 최희소 4클래스(write_file 2.1%, web_search 1.8%, lint 3.3%, plan_task 3.8%)는 데이터의 ~11%지만 **점수의 28.6%**를 차지한다. 어느 클래스든 F1 +0.10 = macro **+0.0071**.

| 클래스 | 데이터 점유 | 점수 점유 | 예제당 leverage |
|---|---|---|---|
| edit_file(최다) | 16.0% | 7.14% | 0.45 |
| web_search(최소) | 1.8% | 7.14% | **3.92** |

**단, "희소가 marginal gain을 지배"는 절반만 참 (D).** 희소는 *예제당 leverage*에서 이기지만, 실제 *가용 headroom*은 탐색군(read_file/grep_search/list_directory/glob_pattern, 전체 41%, F1 0.49~0.64)에 있다(나머지 클래스는 이미 0.68~0.999). 탐색군을 현실적 ~0.85까지 올리면 **~+0.086**로, 이것이 남은 개선의 실질 대부분이다.

**모호성 하한(Bayes floor) (D).** 439개 충돌키에서 각 키의 다수 라벨만 취해도 불가피 오분류는 **≤439행**뿐 → 정확도 상한 **≥~99.4%**. 즉 라벨노이즈는 구속이 아니며, "prompt-only 0.435가 상한"류의 서술도 틀렸다(현 모델은 last_action·history·meta로 이미 0.782). 헤지된 현실 상한 **~0.82~0.87**.

**분산·셰이크업 (D-보정).** 고정 30k에서 나온 LB(0.78226)는 **재표집 노이즈 없는 정확값**이다. 변동은 나중에 만들어질 public/private 분할에서만 생기며, 희소 클래스 positive가 절반으로 줄어 split 간 macro-F1 격차는 **±0.006~0.008(1σ)** 규모. 상위권이 0.777~0.782로 밀집(C)해 있으면 private 재편은 web_search/write_file 몇 표에 좌우된다.

**합리적 채점 전략 (D).** `postproc.py`의 클래스별 가법 bias는 방향상 옳다(web_search +0.10=+0.0071 ≫ edit_file −0.01). 실제 배포 bias 벡터(D)는 web_search −4.0, run_bash −2.85처럼 강하게 억제하고 read_file/grep_search를 +0.5/+0.4로 올린다. 다만 ~550 support 위 1e-6 tol coordinate-ascent는 **OOF 과적합** 위험 → 0쪽 shrink + CV 권장. 정확도는 support 가중이라 macro의 나쁜 대리지표다(더 나은 모델은 둘 다 올리지만, 희소 recall을 head 정확도와 맞바꿀 때 괴리).

### 2. 조건(Conditions) — 배포 제약 감사

**구속 제약은 '추론 ≤10분' 하나뿐 (D+C).** 600s/30k = **20ms/sample**. 서버 실측(C):

| 후보 | 구성 | 서버 시간 | LB |
|---|---|---|---|
| **submit_largev6s3(현 스테이징)** | **단일 large**, v6, max_len 320, batch 128 (D) | 미측정(추정 여유 큼) | — |
| str2q8 | 2-large 확률평균 | 487s(8:07) | 0.78189 |
| tri_cond | 3-way 조건부 | 427s(7:07) | 0.78266(은퇴) |

**핵심 정정 (D).** REPORT.md §6은 배포 zip을 "large+base+large 3멤버 + 양자화"로 서술하지만, **실제 스테이징 패키지 `packages/submit_largev6s3`는 단일 large 모델**이다: `run_meta.json = {version v6, max_len 320, batch 128}`(앙상블 키 없음), `model/`에 `model.safetensors` **1개**(709,888,900 B = fp16+vocab-prune 49822/250002), `postproc.json`은 14원소 bias 1벡터. 즉 int8→353MB "필수 양자화"는 **현 빌드에 없는 계획**이고, 단일 모델은 이미 <1GB이며 타이밍 캡에 여유가 크다. Colab이 측정한 719~734s는 **과대측정**이었고 앙상블도 서버에선 캡 내 완주했다 — "2-large가 위험하다"는 옛 서사는 이 아티팩트에 부합하지 않는다.

**오프라인 랜딩마인은 대부분 이미 방어됨 (D).** `packages/submit_largev6s3/requirements.txt` = **0바이트**(의도적; transformers 사전설치, lightgbm은 앙상블 시에만 벤더링). Rust 토크나이저(`tokenizer.json` 17MB)·`sentencepiece.bpe.model`·`special_tokens_map.json` 전량 동봉. sklearn 피클 미사용(로컬 py3.10 vs 서버 py3.11 회피). max_len은 320토큰(346은 프롬프트 최대 char 수, 혼동 주의).

**실질 미해소 리스크 (D).** `work/verify_largev6s3.txt`의 내용은 문자 그대로 `bash: /share/verify/submit_largev6s3/run.sh: No such file or directory` — 즉 **패키징만 통과했고 end-to-end 타이밍/peak-VRAM/홀드아웃 채점은 이 아티팩트에서 미검증**. `sim/calib.json`의 `{"ratio": 3.21}`은 public/private 비율이 아니라 **A6000 wallclock→T4초 환산계수**(게이트 ≤540s용)다.

### 3. 데이터셋(Dataset) — 품질·구조 감사

**검증 설계는 견고 (D+C).** 세션 단위 GroupKFold가 정답 단위이며, 두 독립 점검(GroupKFold≈Random gap ~0; position-map 내장 제출의 hidden coverage 0)이 test가 train과 **세션을 공유하지 않음**을 확인했다. CV가 일반화 타깃(신규 세션·프롬프트)을 정확히 모사. 프롬프트 90.4% 고유·평균 61자 → 암기 불가, 일반화만이 답.

**분포 이동 리스크 (D+C).** au 5,025행(7.2%)은 구조적 OOD: history 5.04 vs sim 7.08, 프롬프트 51.9 vs 61.7자, read_file-heavy — 하필 headroom이 있는 탐색군 경계를 왜곡한다. "test=all-sim"은 **문서 주장(C)**이므로, au 포함/제외 효과는 반드시 **sim-only OOF**로 재측정해야 한다(전체 OOF에 au를 섞으면 효과가 희석돼 "무효"로 오판).

**콜드스타트 12.9% (D).** 9,000행이 history 없는 step_01. 지배 라벨은 list_directory/read_file/plan_task(탐색 개시). 이 하위집단은 현재 프롬프트+메타만으로 예측해야 하는 별도 레짐이며, **test 내 존재·비중은 추정(C)**이므로 empty-history 슬라이스 OOF로 견고성을 따로 봐야 한다.

**정보 상한 (D).** history 12엔트리(≈6턴) 캡 + turn_index 최대 18 → 긴 세션은 step_01의 최초 태스크 프레이밍을 잃는다. train/test 동일 적용이라 누수는 아니고 **고정 상한**.

**미개발 신호 (D).** last_action 단독은 다수결 정확도 0.233로 약하지만 +0.089(C) 기여 → `result_summary`("tests failed" 등)·`last_ci_status=failed`·`git_dirty` 같은 구조화 메타가 저활용 상태. **가장 강한 구체 레버**: metrics.py는 예측 positive가 0인 클래스에 F1=0을 준다 → 희소 클래스를 한 번도 안 뱉으면 하드 0. **클래스별 floor/threshold로 14클래스 전부 발화 보장**이 최우선 레버다.

### 4. 종합 리스크 레지스터 (심각도순)

| # | 리스크 | 근거 | 심각도 |
|---|---|---|---|
| R1 | **최종 zip end-to-end 미벤치마크 → 0점** | `verify_largev6s3.txt` = run.sh 부재, 타이밍/VRAM 게이트 미실행 (D) | 치명(재시도 없음) |
| R2 | **환경 파리티 실패** | 완전 오프라인·requirements 0바이트·py3.10→3.11·sklearn피클 금지; 불일치 시 macro-F1과 무관하게 전체 0 (D) | 치명 |
| R3 | **private≠public 미확인** | DEBATE.md가 codex와 공동가정, 미검증; 상위 0.777~0.782 밀집이면 재편 (C) | 높음 |
| R4 | **홀드아웃 게이트 누수** | FULL 학습분 포함 → large-heavy 후보 과대평가. 이미 파이프라인 검증용으로 격하됨 (C) | 중~높음 |
| R5 | **calibration 소표본** | (holdout,LB) 소수 쌍에서 cross-family 최대 ~0.0035 격차(lgb8 0.77873 vs 0.78226; str2q8 0.78395 vs 0.78189) → 고정 게이트 위험 (C) | 중 |
| R6 | **희소·콜드스타트 셰이크업** | web_search ~540 test샘플 추정, 수개 오류가 F1 급변; 콜드스타트 test 분포 미확인 (D) | 중 |
| R7 | **headroom 착시** | 텍스트·메타·reranker 모두 "large가 이미 흡수"(C); 실질 신규익 5~10%뿐일 수 있음 | 중(기회비용) |

### 5. 판정(Verdict)

**핵심 강점.** ① 세션 GroupKFold + 이중 누수 점검으로 일반화 타깃을 정확히 모사 — 방법론적으로 드물게 견고하다. ② 단일 large 배포는 타이밍 캡에 여유가 크고, 앙상블(tri_cond 427s)도 서버 실측으로 캡 내 완주가 확인됐다. ③ 오프라인 랜딩마인(토크나이저 동봉·offline 플래그·sklearn 피클 회피·requirements 0바이트)이 이미 대부분 방어돼 있다.

**치명 약점.** ① **최종 제출 zip의 end-to-end 타이밍/VRAM/홀드아웃 검증이 미완**(패키징 게이트만 통과) — 0점 직결. ② 모든 실측 신호가 서버 전용·일 10회 한도에 갇혀 있고 로컬 스텁은 5행이라 무의미. ③ private=public 가정이 미확인인 채 4자리 LB 델타를 신호로 쓰는 종반 전략.

**구체 권고.**
1. **마감 48h 전** 얼린 최종 zip을 오프라인 컨테이너에서 30k end-to-end(타이밍 + peak-VRAM + 홀드아웃 채점) 실제 완주시키고, 홀드아웃 수치 없는 로그는 PASS가 아니라 **FAIL**로 처리한다.
2. 대회 규정에서 **public/private 분할 여부를 확인**한다. 분할이 있으면 <0.003 LB 델타 추종을 멈추고 session-CV OOF로 재정박(홀드아웃 절대값은 누수로 배제).
3. **클래스별 floor/threshold**로 14클래스 전부 발화를 보장(하드 0 방지) — 본 감사가 짚은 최강 구체 레버. `fit_bias`는 0쪽 shrink 후 CV로 이관.
4. au·콜드스타트 견고성을 **sim-only·empty-history 슬라이스 OOF**로 재측정한 뒤에만 sim-only 배포를 결정한다.
5. 검증된 상위 후보를 **조기 뱅킹**하고, cross-family ~0.0035 격차를 감안해 고정 게이트 대신 (holdout, OOF, LB) 트리플의 관측 스프레드 밖 마진만 배포한다.

---

### 부록 — 이 문서의 실측 재현
```bash
python3 - <<'PY'   # 핵심 통계 재현 (numpy 불필요)
import json, collections, csv, re
tr=[json.loads(l) for l in open('data/train.jsonl') if l.strip()]
te=[json.loads(l) for l in open('data/test.jsonl') if l.strip()]
lab={r[0]:r[1] for r in list(csv.reader(open('data/train_labels.csv')))[1:]}
print("train", len(tr), "test-stub", len(te), "stub∈train_labels", all(r['id'] in lab for r in te))
c=collections.Counter(lab.values()); print("imbalance", max(c.values())/min(c.values()))
print("cold-start", sum(1 for r in tr if not r['history']))
p2l=collections.defaultdict(set)
for r in tr: p2l[r['current_prompt']].add(lab[r['id']])
print("ambiguous prompts", sum(1 for s in p2l.values() if len(s)>1))
PY
```
문서 내 `(D)` 항목은 위 스크립트 및 `packages/submit_largev6s3/{run_meta.json,postproc.json,requirements.txt}`, `work/{member_largev6s3/prune_meta.json,verify_largev6s3.txt}`, `sim/calib.json`, `common/metrics.py`에서 직접 확인했다.
