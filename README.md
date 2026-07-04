# Action Decision — Dacon 236694 (AI Agent 다음 행동 예측)

에이전트 세션 상태(구조 메타 + 대화 이력 + 현재 발화)로 **다음 action(14클래스)** 을 예측한다.
평가지표 = **Macro-F1**. 제출은 **코드 제출**(zip: `model/` + `script.py` + `requirements.txt`)이고,
Dacon 서버(T4 16GB, 오프라인, ≤1GB, 설치·추론 각 ≤10분)가 30k 히든 테스트로 채점한다.

> 상세 전략·실험 로그·의사결정 근거는 전부 [`PROJECT.md`](PROJECT.md) 에 있다. **작업 전 반드시 정독.**
> 실험 결과 원장은 [`experiments_master.csv`](experiments_master.csv).

---

## 0. 빠른 시작 (팀원용)

```bash
# 1) 클론
git clone https://github.com/Roka-jsj/Action_Decision.git
cd Action_Decision

# 2) 파이썬 환경 (로컬은 분석/패키징용, GPU 학습은 Colab/Kaggle)
python3 -m pip install -r requirements-dev.txt

# 3) 대회 데이터 배치 (⚠️ repo에 없음 — Dacon에서 직접 다운로드)
#    https://dacon.io/competitions/official/236694/data 에서 open.zip 받기
#    받은 open.zip을 repo 루트에 두고 압축 해제:
python3 -c "import zipfile; zipfile.ZipFile('open.zip').extractall('.')"
#    → data/{train.jsonl, train_labels.csv, test.jsonl, sample_submission.csv} 생성 확인
ls data/

# 4) fold 인덱스는 이미 splits/splits.npz 로 커밋돼 있음 (재생성 금지 — 전 팀원 동일 fold 사용)
#    혹시 없으면: python3 -c "from common.cv import make_splits; ..." (PROJECT.md 참고)
```

데이터 무결성 확인:
```bash
python3 -c "from common.io_utils import load_train; s,y,i=load_train(); print(len(s),'samples', len(set(y)),'classes')"
# → 70000 samples 14 classes
```

---

## 1. 디렉터리 구조

```
common/          공용 라이브러리 (DRY 단일 소스, 서버 추론에도 복사됨)
  io_utils.py    jsonl 로드, 14클래스 매핑, seed
  serialize.py   dict→문자열 직렬화 (v1~v6). ⭐ v6 = 현재 최강
  ad_lib.py      직렬화 + 추론 + 앙상블/스태커 (서버 script.py가 이걸 import)
  cv.py          홀드아웃 8% + StratifiedGroupKFold(5), splits.npz 캐시
  metrics.py     pooled-OOF macro-F1 (Dacon 공식정의와 일치)
  postproc.py    per-class bias 좌표상승 후처리
  vocab_prune.py 임베딩 프루닝 (id 리매핑, 250k→~50k 토큰, 모델 축소)
  leak.py        (기각) 순차 지도 — 히든테스트 커버리지 0으로 검증 후 미사용

action_decision_maximum/src/
  teacher_cli.py     교사 1구성 5-fold 학습 → 확률 npz만 저장 (증류/스택용)
  train_full_cli.py  FULL-70k 멤버 학습 → 프루닝 → member_<TAG>.zip (배포용)
  distill_cli.py     교사 OOF → student 증류

sim/             오프라인 시뮬 + 패키징 + 자동화
  package_ensemble.py  제출 zip 조립 (멤버+스태커+벤더링 lightgbm)
  parity_check.py      ⭐제출 전 필수 — 홀드아웃 재현으로 버전오지정/순서교체 검출
  run_offline_sim.py   네트워크 차단 설치+추론+메모리+스키마 검증
  check_zip.py         구조/1GB/오프라인 최종 점검
  babysit_*.sh         Colab 세션 자동 관리 (사망시 재기동, npz 증분 회수)
  bench_t4.sh          실제 T4 추론 타이밍 벤치

kaggle/          Kaggle 커널 트랙 (무료 GPU 30h/주)
  launch.sh            데이터셋 업로드 + 커널 push 원커맨드
  k_*/                 각 교사 커널 (script.py + kernel-metadata.json)

eda/             분석 스크립트 (에러분석, 스크리닝, 스태킹 프로브)
splits/          fold 인덱스 (전 팀원 공유, 커밋됨)
PROJECT.md       ⭐ 전략·실험·의사결정 단일 소스 (매 실험 후 갱신)
experiments_master.csv  실험 결과 원장
```

**설계 원칙**: `common/`은 개발 공유(DRY), **제출 `model/`에 `common/ad_lib.py`를 복사**해 자체완결.
직렬화/추론이 학습·서버에서 100% 동일하도록 단일 함수(`ad_lib.serialize`, `ad_lib.predict`)만 사용.

---

## 2. 학습 (GPU) — 두 가지 트랙

로컬에 GPU가 없으므로 **Colab(주력) + Kaggle(무료 보조)** 로 학습한다.
학습 산출물은 두 종류:
- **교사 npz** (`teacher_<TAG>.npz`): 5-fold OOF/holdout 확률만 (~1MB). 스태킹·증류용.
- **FULL 멤버 zip** (`member_<TAG>.zip`): 70k 전체 학습 + 프루닝된 배포용 가중치.

### A) Colab CLI (주력, A100)

```bash
# google-colab-cli 설치 + 로그인은 각자 본인 계정으로 (OAuth)
pip install google-colab-cli
colab login

# 교사 학습 (예: xlm-roberta-large, v6 직렬화, 6epoch, 5-fold)
#   babysit_teacher.sh <NAME> <MODEL> <VER> <MAXLEN> <EP> <LR> <BATCH> <FGM> <fold_lo> <fold_hi> [GPU]
bash sim/babysit_teacher.sh mylarge xlm-roberta-large v6 320 6 2e-5 64 0 0 5 A100

# FULL 배포 멤버 학습 (프루닝 포함)
#   babysit_full.sh <TAG> <MODEL> <EP> <LR> <BATCH> <PRUNE> <VER> [GPU]
bash sim/babysit_full.sh mylarge_full xlm-roberta-large 8 2e-5 64 1 v6 A100
```
babysitter가 세션 사망 시 자동 재기동하고 npz/zip을 로컬로 증분 회수한다.
환경변수 상세는 `action_decision_maximum/src/teacher_cli.py` 상단 docstring 참고.

### B) Kaggle 커널 (무료 T4×2, 30h/주)

```bash
# 1) 본인 Kaggle API 토큰 준비 (아래 3장 참고)
# 2) 데이터셋 업로드 + 교사 커널 발진
bash kaggle/launch.sh          # USERNAME 자동 치환 → 데이터셋 생성 → 커널 push

# 상태 확인 / 결과 회수
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
kaggle kernels status <username>/ad-kluev6-teacher
kaggle kernels output <username>/ad-kluev6-teacher -p ./out   # teacher_*.npz 회수
```
⚠️ Kaggle 커널 GPU/인터넷은 **전화번호 인증 계정**에서만 활성화된다 (Settings → Phone verification).

---

## 3. 인증 설정 (각자 본인 계정 — repo에 키 없음)

**절대 키를 커밋하지 말 것.** `.gitignore`가 `kaggle.json`/`access_token`/`.env`를 차단하지만,
근본적으로 인증 파일은 **홈 디렉터리**에 둔다.

```bash
# Kaggle (신형 KGAT 토큰 방식)
mkdir -p ~/.kaggle
echo "KGAT_xxxxxxxx" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
# CLI가 legacy kaggle.json도 요구하면:
printf '{"username":"<본인아이디>","key":"KGAT_xxxxxxxx"}' > ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json

# Colab: colab login (브라우저 OAuth) — 파일 저장 없음
```

---

## 4. 제출물 만들기

```bash
# 0) lightgbm 벤더링 (서버 오프라인 대비 — 최초 1회, vendor/는 gitignore됨)
pip download lightgbm==4.6.0 --no-deps -d vendor/
python3 -c "import zipfile,glob; zipfile.ZipFile(glob.glob('vendor/lightgbm-*.whl')[0]).extractall('vendor')"

# 1) 스태커 적합 (교사 npz들로 — 멤버 조합 지정)
python3 eda/fit_stacker.py artifacts/mystack klue largev6

# 2) 패키징 (⚠️ 멤버별 직렬화 버전을 ::v4 / ::v6 로 반드시 명시!)
python3 sim/package_ensemble.py --out submit_x --stacker artifacts/mystack \
  --member action_decision_maximum/experiments/member_kluefull.zip::v4 \
  --member action_decision_maximum/experiments/member_largev6full.zip::v6 \
  --version v4 --max_len 320 --batch 128

# 3) ⭐제출 전 필수 게이트 — 침묵 실패(버전 오지정/멤버 순서교체) 검출
python3 sim/parity_check.py packages/submit_x 300     # holdout macro-F1이 기대치 근처인지
python3 sim/run_offline_sim.py --model packages/submit_x/model --script packages/submit_x/script.py --n 400

# → 둘 다 PASS면 packages/submit_x.zip 을 Dacon에 업로드
```

**핵심 함정** (PROJECT.md §5.14 감사 결과):
- 멤버 zip엔 직렬화 버전 마커가 없다 → `::버전` 누락 시 크래시 없이 F1만 조용히 하락. **parity_check 필수.**
- 스태커 멤버 순서 = `--member` 순서 = 스태커 학습 시 키 순서. 어긋나면 무증상 하락.
- sklearn pickle 금지(서버 버전 상이). LightGBM은 텍스트 모델(`meta.lgb`). requirements.txt는 빈 파일(트랜스포머는 사전설치, lightgbm은 벤더링).

---

## 5. 현재 상태 (요약 — 최신은 PROJECT.md)

- **LB 최고 0.78051 (7위)** = **xlm-roberta-large v6 8ep FULL 단일모델 + per-class bias**. 컷(0.776) 돌파.
- **핵심 반전(07-04)**: 앙상블/스태킹이 강한 large를 약멤버(klue·n-gram)로 **희석해 −0.014 훼손**. large 단독이 최강 → **스태킹 폐기**. (large단독 0.78051 vs klue+large+ngram 스택 0.76639)
- 캘리브레이션: v4멤버는 `LB≈holdout−0.015`, v6-8ep FULL 배포는 `LB≈holdout+0.01~0.015`(교사OOF가 배포모델보다 약해 하한). **holdout gain은 LB로 부분전이만** → 신규 후처리는 ×0.2 할인.
- 다음(0.80 목표, +0.020): **large 자체 강화** — 10ep+SWA / FGM / focal·logit-adjust / 강한 large끼리만의 앙상블. 각 fold0 raw macro-F1(0.7456@8ep)로 검증(희석 없어 LB 전이).
- 전략 검증에 **codex(GPT-5.5) 자동 반박토론** 활용: `CODEX_CHALLENGE.md` 참고.

### 배포 스크립트 지도 (신규)
- `sim/package_single.py` — 단일모델 제출(현 최강 경로). `--member <zip>::v6 [--bias <teacher_npz>]`
- `sim/package_ensemble.py` — 다멤버+스태커(현재는 비권장, 희석 확인됨)
- `sim/parity_check.py` — 제출 전 필수: 버전오지정/순서교체 침묵실패 검출
- `action_decision_maximum/src/train_full_cli.py` — FULL-70k 멤버 학습(+프루닝). `sim/babysit_full.sh <TAG> <MODEL> <EP> <LR> <BATCH> <PRUNE> <VER> [GPU]`

---

## 6. 재현 체크리스트 (본선 코드검증 대비)

- seed 고정, 상대경로, UTF-8, 버전 명시 (`transformers==4.46.3`, `numpy==1.26.4` 등)
- `splits/splits.npz` 로 fold 고정 (재생성 금지)
- 학습코드는 감사용, 추론코드(`model/` + `script.py`)는 채점에 그대로 사용됨
- 외부 자원/모델 출처·라이선스 명시
