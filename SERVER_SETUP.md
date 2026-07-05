# 연구실 서버(A6000×2) 운영 — Docker 컨테이너 전용 체제 (확정판 v2)

> 원칙: **호스트에 코드/파일을 만들지 않는다. 모든 작업은 컨테이너 안에서.** 호스트 터미널에서 하는 일은 아래 §1의 docker 명령 실행뿐.
> Claude Code / codex는 **VS Code 확장으로만** 사용 (CLI 설치 없음).

## 0. 구성 — 2인 × (학습+검증) = 컨테이너 4개

| 컨테이너 | 사용자 | GPU | 네트워크 | 자원 | 역할 |
|---|---|---|---|---|---|
| `mun-jtrain` | 나 | 0 | 허용 | 무제한 | 학습·패키징·Claude Code 작업 |
| `mun-jtest` | 나 | 0 | **없음** | 12g/3cpu | 평가서버 재현 검증 (상주) |
| `mun-train` | 조원 | 1 | 허용 | 무제한 | 〃 |
| `mun-test` | 조원 | 1 | **없음** | 12g/3cpu | 〃 |

**상호작용 채널**: 학습↔검증은 도커 **named volume** `/share` 하나로만 연결(내 것 `mun-jshare`, 조원 것 `mun-share`).
호스트 디렉터리 바인드 마운트가 아니라 도커가 내부 관리하는 볼륨이므로 "호스트에 파일 안 만든다" 원칙과 충돌하지 않음 — 이게 없으면 오프라인 검증 컨테이너에 제출물을 넣을 방법 자체가 없다.

⚠️ **마운트가 없다는 것의 대가**: `docker rm` 하면 컨테이너 안 모든 것(저장소·모델·로그인)이 소멸한다.
→ 코드는 **수시로 git push**, 대용량 산출물(member zip)은 **Kaggle 데이터셋 업로드**로 백업이 생명줄.

## 1. 컨테이너 생성 (호스트 터미널 — 여기서 하는 일은 이것뿐)

### 내 세트 (GPU 0)
```bash
docker volume create mun-jshare

docker run -d --name mun-jtrain \
  --gpus '"device=0"' --ipc=host --shm-size=32g \
  -v mun-jshare:/share -w /workspace \
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

docker run -d --name mun-train \
  --gpus '"device=1"' --ipc=host --shm-size=32g \
  -v mun-share:/share -w /workspace \
  action-clf:eval sleep infinity

docker run -d --name mun-test \
  --gpus '"device=1"' --network none \
  --memory 12g --memory-swap 12g --cpus 3 \
  -v mun-share:/share \
  action-clf:eval sleep infinity
```

- `-d + sleep infinity`: 원격(Chrome원격/SSH) 끊겨도 컨테이너 생존. 진입: `docker exec -it mun-jtrain bash`
- 검증 컨테이너는 **상주하되 손대지 않는다** — 실행은 §4처럼 호스트에서 `docker exec` 한 줄.
- GPU 공유 주의: 내 train/test가 같은 GPU0을 쓰므로, **시간 측정용 검증은 학습이 안 도는 틈에** 실행(동시 실행 시 시간 오염; VRAM은 48GB라 공존 가능).

## 2. 학습 컨테이너 초기 셋업 (컨테이너 안, 1회)

```bash
docker exec -it mun-jtrain bash   # (조원은 mun-train)
```
안에서:
```bash
# 저장소 (호스트가 아니라 컨테이너 내부에 클론)
cd /workspace && git clone https://github.com/Roka-jsj/Action_Decision.git . \
  || git clone https://github.com/Roka-jsj/Action_Decision.git && cd Action_Decision || true
git config user.name "Roka-jsj" && git config user.email "vasebull@gmail.com"

# kaggle 인증 — ⚠️ cat 히어독 금지(블록 통붙여넣기 시 뒷줄을 파일내용으로 삼킴). echo 한 줄로:
pip install -q kaggle
mkdir -p ~/.kaggle
echo '{"username":"<KAGGLE_USERNAME>","key":"<KAGGLE_KEY>"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 데이터·멤버 번들 + 부트스트랩
cd /workspace
kaggle datasets download tistmesp03/ad236694-train-bundle -p ./bundle --unzip
apt-get update -qq && apt-get install -y -qq zip unzip
bash sim/server_bootstrap.sh ./bundle   # 끝에 "bootstrap OK: 70000 samples" 확인
```

### Claude Code / codex — VS Code 확장으로 접속
1. 로컬(또는 연구실 PC) VS Code에 **Remote - SSH** + **Dev Containers** 확장 설치
2. Remote-SSH로 서버 접속 → 좌측 Remote Explorer → **Attach to Running Container** → `mun-jtrain` 선택
3. 컨테이너에 붙은 VS Code 창에서 확장 설치: **Claude Code**, **Codex** → 각 확장의 로그인 버튼으로 인증
4. 터미널·에디터·Claude 모두 컨테이너 내부 컨텍스트로 동작. 폴더는 `/workspace/Action_Decision` 열기
- git push는 컨테이너 안에서: 처음 push 때 GitHub 인증 필요 → VS Code의 GitHub Authentication 팝업 사용(브라우저 OAuth)

## 3. 학습 실행 (mun-jtrain 안)

```bash
cd /workspace && mkdir -p work    # 저장소는 /workspace에 직접 클론돼 있음
AD_WORK=/workspace/work AD_MODEL=xlm-roberta-large AD_VERSION=v6 \
AD_MAXLEN=320 AD_EPOCHS=8 AD_LR=2e-5 AD_BATCH=64 AD_LLRD=1 AD_SEED=777 \
AD_SWA_K=3 AD_PRUNE=1 AD_TAG=largev6s3 \
nohup python action_decision_maximum/src/train_full_cli.py > work/train_s3.log 2>&1 &
sleep 15 && tail work/train_s3.log   # "[full] largev6s3: ..." 확인 (안 나오면 실패 — 로그 정독)
# A6000 예상 ~4-5분/ep → 8ep ≈ 35-40분. VS Code 끊겨도 nohup으로 계속.
```
잠금해제된 축(DEBATE.md R12): seed 함대 soup(40분/런), 10ep+SWA 부활, sim-only FULL, 증류 스윕.

## 4. 검증 — 2단계 (평가서버 재현 + 시간·VRAM 게이트)

**① 준비 (mun-jtrain 안)** — 패키징 후:
```bash
bash sim/prep_verify.sh packages/submit_X.zip
```
→ `/share/verify/submit_X/`에 자가완결 검증셋 생성(패키지+holdout 30k+순수파이썬 채점기+run.sh — 오프라인 컨테이너에 의존성 불필요).

**② 실행 — 방법 A (수동 한 줄, 호스트 터미널)**:
```bash
docker exec -it mun-jtest bash /share/verify/submit_X/run.sh
```
**② 실행 — 방법 B (완전 자동: 검증 데몬, 호스트 tmux에 1회 켜두기)**:
```bash
while true; do docker exec mun-jtest sh -c 'for f in /share/verify/*/run.sh; do d=$(dirname $f); [ -f "$d/.done" ] || { bash "$f" > "$d/result.txt" 2>&1; touch "$d/.done"; }; done' 2>/dev/null; sleep 30; done
```
→ prep_verify로 새 검증셋이 생기면 30초 내 자동 실행, 결과는 `/share/verify/<이름>/result.txt`로 회신 —
**학습 컨테이너의 Claude Code가 이 파일을 읽어 게이트 판정까지 이어감** (학습→패키징→검증→판정 전자동 고리 완성).
출력: A6000 실측시간 / 피크 VRAM / 행수 / holdout macro-F1 / **환산 서버시간 게이트(≤540s)** / **VRAM 게이트(≤14GB)**.

**캘리브레이션 (최초 1회)** — A6000은 T4보다 3~4배 빨라 절대시간이 무의미. 서버 실측 앵커로 비율 산출:
```bash
# mun-jtrain 안: 앵커 2개 재조립
python3 sim/package_single.py --out submit_largeonly \
  --member action_decision_maximum/experiments/member_largefullv6.zip::v6 \
  --bias "action_decision_maximum/experiments/teacher_largev6[AB]_a*.npz"
bash sim/prep_verify.sh packages/submit_largeonly.zip
# 호스트: docker exec -it mun-jtest bash /share/verify/submit_largeonly/run.sh   → 측정치 T1 (서버실측 257s)
# tri_cond 앵커(서버실측 427s)도 동일 절차 → 측정치 T2
# mun-jtrain 안: echo '{"ratio": <mean(257/T1, 427/T2)>}' > sim/calib.json  (이후 prep_verify가 자동 동봉)
```
(컨테이너 VS Code의 Claude Code에게 "SERVER_SETUP.md 읽고 캘리브레이션 이어서 해줘"로 위임 가능)

## 5. 산출물 회수·제출 흐름

- 서버 학습 산출물(member zip) → mun-jtrain 안에서 **Kaggle 데이터셋 업로드**(`kaggle datasets version -p kaggle/ds ...`) → 어디서든 회수 가능
- 제출용 zip → 로컬 PC로 가져와 Dacon 웹에서 수동 제출(기존 흐름). 컨테이너→로컬 전송: Kaggle 데이터셋 경유가 표준, 급하면 호스트 터미널에서 `docker cp mun-jtrain:/workspace/Action_Decision/packages/submit_X.zip .` 후 scp.
- **Colab/Kaggle GPU 은퇴 확정 (07-05)** — Kaggle은 파일 전송 채널(데이터셋)로만 유지.

## 6. 복구 절차 (컨테이너 소실 시)

`docker rm` 또는 서버 재부팅으로 컨테이너가 사라져도: §1 생성 → §2 셋업(git clone + 번들 다운로드)이면 15분 내 원상복구.
단, **git push 안 한 코드와 Kaggle 업로드 안 한 멤버는 영구 소실** — 작업 단위마다 push/업로드 습관화.
