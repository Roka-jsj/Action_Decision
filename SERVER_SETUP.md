# 연구실 서버(A6000×2) 이행 계획 — Docker + Claude Code

> 배경: Colab 유닛 소진·세션수명 55분 문제, Kaggle 주30h 쿼터. A6000 48GB 1대 전용 확보로 전부 해소.
> 원칙: **학습 컨테이너(내 GPU 1대) + 검증 컨테이너(평가서버 재현)**, Claude Code는 학습 컨테이너 안에서 상주.

## ⚡ 컨테이너 역할 (혼동 주의 — 실전 Q&A로 확정)

| | 학습 컨테이너 `ad-train` | 검증 컨테이너 |
|---|---|---|
| 수명 | 상주 (직접 만들고 유지) | **일회용 — docker_verify.sh가 자동 생성→추론→삭제. 손으로 만들지 않는다** |
| 도커 소켓 | ✅ **필요** (검증 컨테이너를 발사하는 쪽) | ❌ 불필요 (평가서버 재현이라 `--network none` 격리가 정상) |
| 들어가서 작업? | ✅ Claude Code 상주, 모든 작업 여기서 | ❌ 들어갈 일 없음 |

- 기존 컨테이너에 마운트(소켓 등)는 **추가 불가** → 학습 컨테이너만 재생성(§2 명령 그대로). /workspace·claude-home은 호스트에 있어 잃는 것 없음. 컨테이너 안에 깔았던 node/kaggle만 재설치(2-3분).
- 자동 검증 체인: `python3 sim/package_ensemble.py --out submit_X ... && bash sim/docker_verify.sh packages/submit_X.zip 0`

## 0. 조원 명령어 검토 결과

- 학습 `--gpus all` → **`--gpus '"device=N"'`로 고정** (조원 GPU 침범 방지, N은 배정 번호). torchrun DDP 불필요 — 우리 train_full_cli는 단일GPU 설계이고 A6000 48GB면 large b64도 gradckpt 없이 여유.
- 학습을 `--rm -it` 포그라운드로 → 원격 끊김 = 학습 사망. **호스트 tmux 안에서 실행** + `--name` 부여.
- 검증 명령(오프라인/12g/3cpu/GPU1대)은 평가서버 재현으로 정확. 단 **2개 게이트 추가 필수**:
  1. **시간 캘리브레이션**: A6000은 T4 대비 3~4배 빠름 → 절대시간 무의미. 서버 실측 앵커(largeonly=257s, tri_cond=427s, str2q8=487s)를 검증 컨테이너에서 재측정해 비율 산출 → 신규 후보는 `환산 서버시간 ≤ 540s` 게이트.
  2. **VRAM 게이트**: 48GB에선 T4 16GB 초과를 못 잡음 → 실행 중 nvidia-smi 피크 기록, **≤ 14GB** 확인.

## 1. 서버 호스트 준비 (1회)

```bash
# 배정 GPU 번호 확인 (조원과 합의된 번호 사용; 이하 예시는 0)
nvidia-smi
# tmux 필수 (원격 끊김 생존)
tmux new -s ad    # 이후 모든 작업은 이 tmux 안에서. 재접속: tmux attach -t ad

mkdir -p ~/ad && cd ~/ad
git clone https://github.com/Roka-jsj/Action_Decision.git
cd Action_Decision
mkdir -p ~/ad/claude-home     # Claude Code 설정/로그인/메모리 영속화용
```

## 2. 학습 컨테이너 (Claude Code 상주) — 확정판

```bash
docker rm -f ad-train 2>/dev/null
docker run -d --name ad-train \
  --gpus '"device=0"' --ipc=host --shm-size=32g \
  -v ~/ad/Action_Decision:/workspace -w /workspace \
  -v ~/ad/claude-home:/root/.claude \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e AD_HOST_REPO="$HOME/ad/Action_Decision" \
  action-clf:eval sleep infinity
docker exec -it ad-train bash
```
- `-d + sleep infinity` → 원격 끊겨도 컨테이너 생존. 재진입: `docker exec -it ad-train bash`
- `docker.sock` 마운트 + `AD_HOST_REPO` → 이 안에서 검증 컨테이너 자동 발사 가능 (DooD; -v 경로는 호스트 기준이라 AD_HOST_REPO로 변환)

컨테이너 안에서 (1회 셋업, 순서대로):
```bash
# ① docker CLI (검증 자동화용) — docker ps 로 확인
apt-get update -qq && apt-get install -y -qq docker.io
docker ps

# ② node + Claude Code (nvm은 현재 셸에 source 필수 — 서브셸에서 깔면 npm not found)
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
export NVM_DIR="$HOME/.nvm" && . "$NVM_DIR/nvm.sh" && nvm install 22
npm install -g @anthropic-ai/claude-code
claude   # 로그인: URL을 Chrome원격 브라우저에서 열고 코드 입력 (claude-home 마운트로 영속)

# ③ kaggle 인증 + 번들 (번들은 /workspace 안에 받아야 컨테이너 재생성에도 생존)
pip install -q kaggle
mkdir -p ~/.kaggle && cat > ~/.kaggle/kaggle.json   # kaggle.json 내용 붙여넣기 → Enter → Ctrl+D
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download tistmesp03/ad236694-train-bundle -p /workspace/bundle --unzip
bash sim/server_bootstrap.sh /workspace/bundle
```

학습 실행 (예: v6-8ep seed 스윕 — babysitter 불필요, 세션수명 무한):
```bash
AD_WORK=/workspace/work AD_MODEL=xlm-roberta-large AD_VERSION=v6 AD_MAXLEN=320 \
AD_EPOCHS=8 AD_LR=2e-5 AD_BATCH=64 AD_LLRD=1 AD_SEED=777 AD_SWA_K=3 AD_PRUNE=1 \
AD_TAG=largev6s3 python action_decision_maximum/src/train_full_cli.py
# A6000 예상: ~4-5분/ep → 8ep ≈ 35-40분 (Colab A100 6.9분/ep 대비 비슷하거나 약간 느림)
```

## 3. 검증 (학습 컨테이너 안에서 실행 — 형제 검증 컨테이너 자동 발사)

```bash
bash sim/docker_verify.sh packages/submit_X.zip 0   # <zip> <GPU번호>
```
자동 수행: `--network none --memory 12g --cpus 3 --gpus device=N` 일회용 컨테이너 발사 →
holdout 30k 추론 → ①환산 서버시간 게이트(≤540s) ②피크 VRAM 게이트(≤14GB) ③행수 ④holdout 채점 → 컨테이너 삭제.

**캘리브레이션(최초 1회)**: 서버 실측 앵커 2개를 재조립해 측정, ratio 저장:
```bash
python3 sim/package_single.py --out submit_largeonly \
  --member action_decision_maximum/experiments/member_largefullv6.zip::v6 \
  --bias "action_decision_maximum/experiments/teacher_largev6[AB]_a*.npz"
bash sim/docker_verify.sh packages/submit_largeonly.zip 0     # 서버 실측 257s와 비교
# tri_cond 앵커(서버 427s)도 동일하게 → ratio = mean(257/측정1, 427/측정2)
echo '{"ratio": X.XX}' > sim/calib.json
```
(컨테이너 안 Claude Code에게 "캘리브레이션 이어서 해줘"라고 맡기면 됨)

## 4. 전략적 함의 (R12 배분표 갱신)

컴퓨트 병목 해소로 잠금해제되는 축 (DEBATE.md R12 참조):
- **seed 함대 soup**: s3, s4, ... 8ep 런이 40분/개 → 3-4개 soup 재료 하루에 확보
- **10ep+SWA**: 세션수명 제약 소멸 → 5.18에서 포기했던 축 부활 (+0.003~0.006 추정였음)
- **sim-only FULL** (au 5k 제외): R11 백로그, 1런 40분
- **증류 variant 스윕**: soft_w/T 조합
- 검증 컨테이너로 **제출 전 게이트 완전 내재화** → LB 슬롯은 순수 힐클라이밍에만

**Colab/Kaggle 은퇴 확정 (07-05)**: 학습·검증 전부 연구실 서버로 이행. Kaggle은 GPU 학습 용도로는 종료 — 단 `tistmesp03/ad236694-train-bundle` 데이터셋은 **파일 전송 채널**로 유지(멤버 zip·teacher npz 서버 반입용, GPU 쿼터와 무관). 진행 중이던 커널(s2/dist)은 회수되면 쓰고, 늦으면 서버에서 재학습(40분/런)이 더 빠르므로 폐기 가능.

## 5. 이 로컬(WSL)의 역할

- 저장소 정본(git push/pull로 서버와 동기화), Dacon 제출용 zip 보관.
- 서버에서 만든 member/package는 `git`이 아니라 **Kaggle 데이터셋 경유 또는 scp**로 로컬 회수 후 제출.
  (제출은 사용자가 Dacon 웹에서 수동 — 기존 흐름 유지)
