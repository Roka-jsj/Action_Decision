# 문샷(합성증강) 하네스 — 주입 설계 + fold0 게이트 사이클 (2026-07-11, CPU-전용 준비 완료)

승인 근거: codex R57b "제한된 복권" 조건부 + R61 "문샷 비GPU 준비 병행 승인" + 전략가 R65 §5 설계초안.
상태: **GPU 0초 사용. 발사는 3자(지휘+codex+레드팀) 서명 후만.**

---

## 0. 요약

| 항목 | 값 |
|---|---|
| 신규 파일 | `sim/synth_gen.py`(생성기) · `sim/synth_gates.py`(3게이트+dedup) · `work/synth_batch0.jsonl`(500행 자가시험) · `work/synth_batch0_pass.jsonl`(435행) · `work/synth_gates_b0.json`(게이트 리포트) |
| 자가시험 결과 | **전 게이트 PASS** (gate0 오염 0건 / gate1 null-보정 근접성 4클래스 전부 통과 / gate2 타겟팅 통과 / gate3 TVD 0.0057 / dedup reject 13.0%) |
| 캐리어 | **mdeb 전용** (m1 캐리어는 δ 세금 상속 — R63b 미해결) |
| 기준선 | mdebr_f0 **0.7640** (mdeberta-v3-base, 12ep, gen_rescue, mht12, FGM, fold0, work/mdebr_f0.log 201분) |
| 통과선 | fold0 **+0.0025** (R57b 사전등록) = **best val ≥ 0.7665** + 타겟클래스 집중 검증 |
| GPU 견적 | 프로브 1발 ≈ **3.4~3.6h** (A6000, 51.4k+1.3k행 12ep) |

---

## 1. 타겟 클래스 선정 — fold0-val 실측 (mdebr_f0 OOF)

mdebr_f0(캐리어 기준선)의 fold0-val 12,829행 per-class F1, 약한 순:

| 클래스 | F1 | P | R | n_val | 비고 |
|---|---|---|---|---|---|
| **list_directory** | **0.480** | **0.364** | 0.703 | 784 | 최약. 정밀도 붕괴(과예측) — 인용된 0.384 와 동일 현상(현행 실측 mdebr 0.364 / v6r 0.366) |
| **read_file** | 0.577 | 0.618 | 0.541 | 1,737 | |
| **grep_search** | 0.621 | 0.708 | 0.552 | 1,867 | |
| ask_user | 0.651 | 0.756 | 0.571 | 504 | plan_task 와 별도 혼동쌍 — 이번 제외 |
| lint_or_typecheck | 0.666 | 0.663 | 0.669 | 408 | run_tests 혼동쌍 — 이번 제외 |
| **glob_pattern** | 0.682 | 0.718 | 0.649 | 1,020 | |

혼동행렬 상위 5셀이 전부 **내비게이션 4중주 {read, grep, list, glob} 내부**:
read→list 428 / grep→read 370 / grep→list 356 / read→grep 253 / glob→list 173.
list_directory 의 문제는 재현율이 아니라 **정밀도**(read/grep 행이 list 로 흡수됨) — 즉 4중주 경계
전체를 동시 강화해야 하며(4클래스 동반 타겟), list 단독 증량은 과예측을 악화시킬 수 있음.
v6r_f0/mdeb12ep_f0 에서도 동일 서열(모델 불변 구조) — 히든 특이 신호 아님.

기본 타겟/배분: `list 0.35 / read 0.30 / grep 0.20 / glob 0.15` (약함×혼동질량 절충, CLI 로 조정 가능).

---

## 2. 생성기 `sim/synth_gen.py` — 실데이터 재조합만 (LLM 0, 신규정보 주입 0)

전략가 R65 첫 필터 준수: "이미 흡수된 신호를 더 많은 표본으로 강화"만. 템플릿 3종:

| 템플릿 | 조작 | 근거 |
|---|---|---|
| **T1 histswap** | 같은 (클래스, gen, 마지막액션명, 결과상태, turn_bin) 버킷 내 두 행: a 의 meta+프롬프트 × b 의 history | 라벨 결정 신호(프롬프트·[LAST]·상태) 보존, 문맥만 다양화. 앵커포라(again/다시/that file 등) 프롬프트는 공여자에서 제외(자기 히스토리 강결합) |
| **T2 tailcrop** | history 절단(최근 2-4턴 / 랜덤 prefix) + 필드 드롭아웃(rs 12%, args 10%) — **마지막 액션턴 면제, 빈-히스토리 모드 제거** | histdrop(R24, LB 검증)와 동일 섭동 패밀리의 클래스 타겟판. 두 수정은 batch0 실측 반영(§3 반복 이력) |
| **T3 metaswap** | none|na(무액션) 버킷 전용: a 의 프롬프트/히스토리 × b 의 session_meta | **실데이터 none-share 정확 재현** — list_directory 는 41.6%가 무히스토리 행(세션 첫 액션). 이 채널 없이는 클래스-조건 분포가 성립 안 됨 |

오염 차단(전부 assert 사망 방식):
- 소스는 **fold0-train(51,361행)만**. fold0-val(12,829)·holdout(5,810) 유래 소스 발견 시 즉사.
- splits 는 `refit_lib.load_splits()` 로만(sha256 고정, 재생성 금지 교리).
- synth id = `sess_{gen}_syn{tag}NNNNN-step_NN` — 신규 세션 네임스페이스(실세션 그룹 불교차,
  세션-스텝 누출 tb/416939 원천 차단), gen(au/sim)은 프롬프트 공여자 보존([GEN] 태그 정합).
- v6 직렬화 유효성(mht 8/12 양쪽) 검사 통과 행만 출력.
- 출력 행 = 표준 스키마 + `label` + `_synth`(template, src_ids, bucket, seed) — 주입 시 pop.

batch0 (500행, seed 20260711): T1 268 / T2 116 / T3 116, sim 458/au 42, 앵커포라 스킵 43, v6 무효 0.

### 반복 이력 (1회 허용분 사용)
1차 batch0: gate1 js_last FAIL (list 0.177 = null 대비 7배). 부검 → 두 원인:
(a) T2 의 빈-히스토리 모드·마지막 액션 필드드롭이 last-action 분포 왜곡,
(b) **synth 가 무히스토리 행을 하나도 못 만듦**(실데이터 list 는 41.6%가 무히스토리).
수정: T2 에서 빈 모드 제거+마지막 액션턴 면제, T3 metaswap 신설(none-share 정확 매칭)
→ 재생성 후 4클래스 전부 PASS (js_last: list 0.0341 vs null 0.0269).

---

## 3. 게이트 `sim/synth_gates.py` — batch0 실측 수치와 제안 컷

**gate1 근접성 (null 보정)** — 절대 JS 는 소표본 양의 편향(실측: n=75 에서 실-vs-실 null js_tok 0.092)
→ 같은 n 의 실데이터 부분표본 null(6회 평균) 대비 비율로 판정. `js ≤ max(0.03, cap×null)`.
컷: len(재조합 T1/T3만; T2 길이이동은 histdrop 선례의 설계된 이동이라 정보성 보고) **4.0×**, tok **1.5×**, last **2.0×**.

| 클래스 | js_len(재조합)/null | js_tok/null | js_last/null | 판정 |
|---|---|---|---|---|
| read_file | 0.0168/0.0143 | 0.0685/0.0629 | 0.0224/0.0342 | PASS |
| grep_search | 0.0124/0.0138 | 0.0803/0.0783 | 0.0374/0.0623 | PASS |
| list_directory | 0.0167/0.0068 | 0.0739/0.0698 | 0.0341/0.0269 | PASS |
| glob_pattern | 0.0301/0.0280 | 0.0994/0.0907 | 0.0425/0.0546 | PASS |

**gate2 타겟팅**: 타겟 비중 1.000 (컷 ≥0.98) / uplift = list **5.5%**, read 2.2%, glob 2.0%, grep 1.4%
(컷 ≥0.5%, 생산배치는 §5 에서 상향). PASS.

**gate3 사전확률 이동**: TVD **0.0057** (컷 0.015) / max|Δp| **0.00278** (컷 0.010, list). PASS.
참고: 히든 실측 사전확률(R30 프로브, list 6.34% vs train 6.18%)이 약간 높아 소폭 +이동은 방향 정합.
**이 게이트가 생산 배치 크기의 구속 조건** — TVD 0.015 컷 기준 주입 상한 ≈ **1,350행**.

**dedup** (3계측 — 교차공간 중심화cos 는 q4-CPU vs 저장 fp16 self-cos 0.84 실측으로 부적합 판명, 미사용):
| 계측 | 분포 (batch0) | 제안 컷 | reject |
|---|---|---|---|
| (a) 공여자쌍 저장공간 중심화cos (T1/T3: 쌍이 같으면 재조합=복제) | p50 0.747 / p90 0.910 / p99 0.963 / max 0.971 | **> 0.90 기각** | 51/384 |
| (b) 어휘 5-gram cos, synth vs 공여자∪공여자NN8 | T1 p99 0.960 / T2 p99 0.967 / T3 p99 0.977 | **T1 > 0.95, T2/T3 > 0.985 기각** (T2/T3 은 원본근접이 설계) | 14/500 |
| (c) q4 동일공간 스팟체크(48행, 정보성) | synth↔공여자 평균 0.981 vs 무관쌍 평균 0.891 | "가깝지만 미포화" 밴드 확인용 | — |
총 reject **65/500 (13.0%)** (컷 ≤20%) → PASS. `--emit-pass` 가 통과 435행을 `work/synth_batch0_pass.jsonl` 로 방출.

---

## 4. 주입 배선 — `AD_AUG=synth` (teacher_cli.py 패치, **서명 후 적용**)

histdrop 선례(train_full_cli.py:69-104)와 동형이되, **fold-train 전용 주입**(va/holdout 무오염)이 핵심 차이.
teacher_cli.py 에는 현재 AUG 배선이 없음 — 아래 블록을 **line 67(`cw = ...`) 다음, line 69(`SRC = ...`) 이전**에 삽입:

```python
# ── AD_AUG=synth (문샷 R57b/R65) — fold-train 전용 주입, va/holdout 무오염 ──
N_REAL = len(samples)
AUG = os.environ.get("AD_AUG", "")
if AUG == "synth":
    from common.io_utils import CLASS_TO_IDX as _C2I, session_id as _sid, generator as _g, step_num as _st
    _syn, _syny, _src = [], [], set()
    with open(os.environ["AD_SYNTH_PATH"], encoding="utf-8") as _f:
        for _ln in _f:
            _d = json.loads(_ln)
            _pv = _d.pop("_synth"); _lab = _d.pop("label")
            _d["session"] = _sid(_d["id"]); _d["gen"] = _g(_d["id"]); _d["step"] = _st(_d["id"])
            _d["label"] = _lab; _d["y"] = _C2I[_lab]
            _syn.append(_d); _syny.append(_C2I[_lab]); _src.update(_pv["src_ids"])
    _r = {v: k for k, v in enumerate(ids)}
    _srows = {_r[s_] for s_ in _src}          # 미지 소스 id → KeyError 즉사(의도)
    for _fi in range(FOLD_LO, FOLD_HI):
        assert not (_srows & set(np.asarray(folds[_fi][1]).tolist())), \
            f"synth 소스가 fold{_fi} va 에 존재 — 게이트 계기 오염"
    assert not (_srows & set(np.asarray(hold_idx).tolist())), "synth 소스가 holdout 에 존재"
    samples = list(samples) + _syn
    y = np.concatenate([y, np.asarray(_syny, dtype=y.dtype)])
    groups = np.concatenate([groups, np.asarray([_d["session"] for _d in _syn])])
    _sidx = np.arange(N_REAL, len(samples))
    folds = [(np.concatenate([np.asarray(_tr), _sidx]), _va) for _tr, _va in folds]
    print(f"[aug:synth] +{len(_syn)}행 (소스 {len(_src)}행, va/holdout 무교차 assert 통과)", flush=True)
```

그리고 line 150 을 1줄 수정 (OOF npz 를 70k 행으로 유지 — refit_lib.MemberOOF N_TRAIN assert 호환):
```python
oof = np.zeros((N_REAL, NUM_CLASSES), np.float32)   # (기존: len(samples))
```

설계 결정 근거:
- **cw(클래스 가중)는 주입 전 실데이터 dev 로 계산된 상태 유지** — 타겟 클래스는 행수로만 증량.
  (가중까지 재계산하면 count×weight 이중 부스트, 사전확률 이동 게이트 무력화)
- 검증/holdout 추론 인덱스(va, hold_idx)는 전부 실행렬 인덱스(< N_REAL) — epoch val= 판독이 자동으로 클린.
- 세션가중(ROW_W_ALL)은 synth 각 행이 독립 세션이므로 자연 처리(단, 프로브 레시피는 sess_bal=off).
- FULL 단계(게이트 통과 후): train_full_cli.py 의 기존 `AUG == "histdrop"` 블록 옆에 동형 elif 추가
  (FULL 은 fold 개념이 없으므로 assert 는 holdout 만; 소스 풀 재생성 여부는 §6-1 결정사항).

---

## 5. fold0 게이트 사이클 — 발사 절차 (3자 서명 후)

```bash
cd /root/Action_Decision
# (0) 생산 배치 생성 + 게이트 (CPU ~3분; dedup ~13% 감모 반영해 1500 생성 → pass ~1300)
PYTHONPATH=. python3 sim/synth_gen.py --n 1500 --tag p0 --seed 20260711 --out work/synth_p0.jsonl
PYTHONPATH=. python3 sim/synth_gates.py --synth work/synth_p0.jsonl \
    --report work/synth_gates_p0.json --emit-pass work/synth_p0_pass.jsonl
#     → overall_pass true 필수 (gate3 TVD ≤0.015 가 주입량 상한 ≈1,350행을 강제)

# (1) 3자 서명 → §4 패치를 teacher_cli.py 에 적용(diff 리뷰 포함)

# (2) fold0 프로브 — mdebr_f0 레시피 완전 동결 + synth 만 추가 (단일 차분 원칙)
AD_WORK=/root/Action_Decision/work AD_MODEL=microsoft/mdeberta-v3-base \
AD_VERSION=v6 AD_MAXLEN=320 AD_EPOCHS=12 AD_LR=2e-5 AD_BATCH=64 AD_LLRD=1 \
AD_FGM=1 AD_SEED=1234 AD_FOLD_LO=0 AD_FOLD_HI=1 AD_GEN_RESCUE=1 AD_MHT=12 \
AD_AUG=synth AD_SYNTH_PATH=/root/Action_Decision/work/synth_p0_pass.jsonl \
AD_TAG=mdebsyn_f0 nohup python3 action_decision_maximum/src/teacher_cli.py \
    > work/mdebsyn_f0.log 2>&1 &
```

- **GPU 시간**: mdebr_f0 실측 201분 × (1+1300/51361) ≈ **206분 + 토크나이즈 ≈ 3.5h** (A6000 1대).
- **판정**(사전등록): best val ≥ **0.7665** (= 0.7640 + 0.0025) 이면서
  (a) 타겟 4클래스 F1 paired delta 합이 전체 이득의 ≥50% (집중성 — "이득이 무관 클래스 집중이면
  인공물/선택노이즈 자동기각" 교리 4 적용),
  (b) 비타겟 클래스 평균 F1 악화 ≤0.002.
  per-class 판독: `work/teacher_mdebsyn_f0.npz` 의 oof[va0] vs `work/teacher_mdebr_f0.npz` (동일 va0).
- **δ 주의**(R63b): 기준선 0.7640 은 단일 런. m1h8full 대조군의 run-to-run 분산(δ) 판독이 mdeb 급에서
  ≥0.002 로 나오면 통과선 재산정 또는 paired-seed 재런 요구 — 3자 라운드에서 τ/δ 판독과 함께 결정.

---

## 6. 게이트 후 결정트리

1. **PASS (Δ ≥ +0.0025, 집중성 충족)** → 2단계:
   (1) 소스 풀 재산정 결정: FULL 용 synth 를 그대로 쓸지(감사 용이) vs 전체 train−holdout 로 재생성할지
   (분포 대표성) — 3자 결정. (2) mdeb-T3 FULL(환산 에폭 = 프로브 best-epoch 고정, 교리 5) + synth
   → 양자화(q8) → parity → 조립(mdeb 슬롯 교체) → 30k 리플레이 → 캐너리 → LB.
   기대 LB 전이: T3 계보 전이율 0.82 적용 시 +0.002 급.
2. **회색 (0 < Δ < +0.0025)** → 재롤 1회 한도: 배분 변경(list 집중 or ask_user/lint 쌍 추가) 뒤 재프로브
   — 단 GPU 유휴가 확보될 때만. 아니면 폐쇄.
3. **FAIL (Δ ≤ 0)** → 문샷 폐쇄, experiments_master 등재. 재론 시 신논거 필수(교리).
4. **인공물 판정** (Δ 양수지만 이득이 비타겟 집중) → 자동 기각, 폐쇄와 동일 처리.

## 7. 잔여 리스크 (정직 목록)

- T1 의 잔여 앵커포라(정규식이 못 잡는 은근한 히스토리 참조) — fold0 게이트가 최종 심판.
- gate1 은 유니그램/길이/구조 근접성만 봄 — 고차 상호작용(프롬프트×히스토리 결합분포) 왜곡은 미계측.
- 주입량 1,300행(tr0 의 2.5%)이 +0.0025 를 만들기에 충분한지는 사전 지식 없음 — 이것이 "복권"의 본체.
- q4 스팟체크의 p99 0.9997 은 T3(메타만 교체) 행 — 어휘 컷이 완전복제는 차단하나, T3 비중이 큰
  list_directory 는 실질 신규성이 낮음. 회색 판정 시 T3 비중 하향이 첫 재롤 후보.
