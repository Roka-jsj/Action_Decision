# 연구실 서버(A6000×2) — 컨테이너 전자동 파이프라인 (확정판 v3)

> **철칙 1**: 호스트에는 아무것도 만들지 않는다. 호스트 터미널은 §1의 docker 명령 실행용일 뿐.
> **철칙 2**: 모든 작업 경로는 컨테이너 안 **`/workspace` 절대경로**로 통일 (저장소가 /workspace에 직접 클론됨).
> **철칙 3**: 학습→검증은 **컨테이너끼리 자동으로** 흐른다 (사람 개입 없음). Claude Code/codex는 VS Code 확장으로만.

## 0. 구조와 자동화 흐름

```
[mun-jtrain GPU0 · 온라인 · 소켓]                [mun-jtest GPU0 · 오프라인 · 12g/3cpu]
  학습(train_full_cli)                              평가서버 재현
    → 패키징(package_*)                                ↑
    → 검증셋 조립(prep_verify) ──/share 볼륨──→  run.sh 자동 실행 (docker exec)
    → 결과 회수(work/verify_*.txt) ←──/share──── 시간·VRAM 게이트 + holdout 점수
```
- 상호작용 채널 = ①named volume `/share`(파일 전달) + ②학습 컨테이너의 docker 소켓(검증 실행 트리거). 이 두 옵션이 "컨테이너끼리 자동화"의 전부다.
- 조원 세트(mun-train/mun-test, GPU1)도 완전 동일 구조, 볼륨만 `mun-share`.

| 컨테이너 | GPU | 네트워크 | 자원 | 소켓 |
|---|---|---|---|---|
| `mun-jtrain` (나) | 0 | 허용 | 무제한 | ✅ |
| `mun-jtest` (나) | 0 | **none** | 12g/3cpu | ❌ |
| `mun-train` (조원) | 1 | 허용 | 무제한 | ✅ |
| `mun-test` (조원) | 1 | **none** | 12g/3cpu | ❌ |

⚠️ 무마운트의 대가: `docker rm` = 컨테이너 내부 전부 소멸. **코드는 git push, member zip은 Kaggle 데이터셋 업로드**가 백업 생명줄 (§5).

## 1. 컨테이너 생성 (호스트 터미널, 통째로 붙여넣기 가능)

### 내 세트 (GPU 0)
```bash
docker volume create mun-jshare
docker rm -f mun-jtrain mun-jtest 2>/dev/null
docker run -d --name mun-jtrain \
  --gpus '"device=0"' --ipc=host --shm-size=32g \
  -v mun-jshare:/share \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -w /workspace \
  action-clf:eval sleep infinity
docker run -d --name mun-jtest \
  --gpus '"device=0"' --network none \
  --memory 12g --memory-swap 12g --cpus 3 \
  -v mun-jshare:/share \
  action-clf:eval sleep infinity
```

### 조원 세트 (GPU 1)
```bash
docker volume create mun-share
docker rm -f mun-train mun-test 2>/dev/null
docker run -d --name mun-train \
  --gpus '"device=1"' --ipc=host --shm-size=32g \
  -v mun-share:/share \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -w /workspace \
  action-clf:eval sleep infinity
docker run -d --name mun-test \
  --gpus '"device=1"' --network none \
  --memory 12g --memory-swap 12g --cpus 3 \
  -v mun-share:/share \
  action-clf:eval sleep infinity
```
- 검증 컨테이너는 상주하되 **절대 들어가서 작업하지 않는다** — 학습 컨테이너가 자동으로 부린다.
- 같은 GPU를 train/test가 공유하므로 **시간 측정용 검증은 학습이 쉬는 틈에** (VRAM 48GB라 공존은 가능, 시간만 오염됨).

## 2. 학습 컨테이너 셋업 (1회, 컨테이너 안)

```bash
docker exec -it mun-jtrain bash    # 조원: mun-train
```
안에서 — **블록 1 (저장소·도구)**:
```bash
cd /workspace
git clone https://github.com/Roka-jsj/Action_Decision.git .
git config user.name "Roka-jsj" && git config user.email "vasebull@gmail.com"
apt-get update -qq && apt-get install -y -qq zip unzip docker.io
echo 'export DOCKER_API_VERSION=1.43' >> ~/.bashrc && export DOCKER_API_VERSION=1.43   # 신형 CLI ↔ 구형 호스트데몬 정합
docker ps        # 소켓 확인 (컨테이너 목록 보이면 OK; "client version too new" 뜨면 위 줄 재확인)
pip install -q kaggle
mkdir -p ~/.kaggle
```
**블록 2 (kaggle 인증)** — ⚠️ `<...>`는 자리표시자: kaggle.com→Settings→API→Create New Token으로 받은 **실제 username/key로 교체**해서 실행:
```bash
echo '{"username":"실제username","key":"실제key"}' > ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
python3 -c "import json;json.load(open('/root/.kaggle/kaggle.json'));print('json OK')"
```
(pip가 DNS 오류나면: `echo 'nameserver 8.8.8.8' >> /etc/resolv.conf` 후 재시도)
**블록 3 (데이터·멤버 반입 + 부트스트랩)**:
```bash
cd /workspace
kaggle datasets download tistmesp03/ad236694-train-bundle -p /workspace/bundle --unzip
bash /workspace/sim/server_bootstrap.sh /workspace/bundle    # "bootstrap OK: 70000 samples" 확인
```

### Claude Code / codex (VS Code 확장)
로컬 VS Code: **Remote-SSH** 접속 → Remote Explorer → **Attach to Running Container → mun-jtrain** → 폴더 `/workspace` 열기 → 확장(Claude Code, Codex) 설치·로그인. 이후 전략·실험 지휘는 이 Claude가 담당.

## 3. 전자동 실험 — 원커맨드 (학습→패키징→검증→결과)

```bash
nohup bash /workspace/sim/train_and_verify.sh largev6s3 777 \
  > /workspace/work/auto_largev6s3.log 2>&1 &
tail -f /workspace/work/auto_largev6s3.log    # 관전 (Ctrl+C로 관전만 중단, 작업은 계속)
```
`train_and_verify.sh <TAG> <SEED> [EPOCHS=8] [VERSION=v6] [TEST_CT=mun-jtest]` 가 자동 수행:
1. FULL 학습 (A6000 ~40분; member 있으면 스킵) → `/workspace/action_decision_maximum/experiments/member_<TAG>.zip`
2. bias 포함 패키징 + zip 구조검사
3. `/share`에 자가완결 검증셋 조립 (holdout 30k + 무의존 채점기)
4. **검증 컨테이너를 docker exec로 자동 실행** (오프라인 30k 추론)
5. 결과 → `/workspace/work/verify_<TAG>.txt`: A6000시간 / 피크VRAM / **환산 서버시간 게이트(≤540s)** / **VRAM 게이트(≤14GB)** / holdout macro-F1

앙상블(예: tri_cond의 m1 교체) 같은 조합 실험은 컨테이너 Claude Code가 `package_ensemble.py` + `prep_verify.sh` + `docker exec`를 같은 방식으로 조립해 실행한다.

## 4. 캘리브레이션 (최초 1회 — 시간 게이트의 기준)

A6000은 T4보다 3~4배 빨라 절대시간이 무의미. **서버 실측 앵커**로 환산비율 산출:
```bash
# mun-jtrain 안 (앵커1: largeonly, Dacon 서버실측 257초)
python3 /workspace/sim/package_single.py --out submit_largeonly \
  --member /workspace/action_decision_maximum/experiments/member_largefullv6.zip::v6 \
  --bias "/workspace/action_decision_maximum/experiments/teacher_largev6[AB]_a*.npz"
bash /workspace/sim/prep_verify.sh /workspace/packages/submit_largeonly.zip
docker exec mun-jtest bash /share/verify/submit_largeonly/run.sh    # → 측정 T1
# ratio = 257 / T1  (예: T1=80초면 3.21)
echo '{"ratio": 3.21}' > /workspace/sim/calib.json    # 이후 모든 검증에 자동 동봉
```
(tri_cond 앵커 427초로 교차확인 권장. 컨테이너 Claude에게 "캘리브레이션 해줘"로 위임 가능)

## 5. 산출물 백업·제출

- member zip 백업: `cd /workspace && cp action_decision_maximum/experiments/member_<TAG>.zip kaggle/ds/ && KAGGLE_API_TOKEN=... kaggle datasets version -p kaggle/ds -m "add <TAG>" --dir-mode zip`
- 제출: 게이트 PASS한 `packages/submit_*.zip`을 로컬로 (호스트에서 `docker cp mun-jtrain:/workspace/packages/submit_X.zip .` → scp/다운로드) → Dacon 웹 수동 제출 → 점수를 컨테이너 Claude에게 전달.
- Colab/Kaggle GPU 은퇴 — Kaggle은 파일 전송 채널로만.

## 6. 복구 (컨테이너 소실 시)

§1 생성 → §2 셋업 = 15분 원상복구. git push 안 한 코드·업로드 안 한 member만 영구 소실.
