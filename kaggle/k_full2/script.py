# FULL v6-8ep 2종 (s2: seed2024+SWA3 / dist: +soft증류) — Kaggle 서버측 학습(PC 꺼도 지속)
# T4x2: GPU별 병렬 / 단일GPU(P100): 순차(s2 우선). 출력 = member_*.zip (커널 output에서 회수)
import os, shutil, subprocess, sys, glob, time

WORK = "/kaggle/working"
os.chdir(WORK)
def find_ds():
    hits = [p for p in glob.glob("/kaggle/input/**/", recursive=True) if "ad236694" in p]
    return sorted(hits, key=len)[0] if hits else None
DS = find_ds()
for _ in range(30):
    if DS:
        break
    time.sleep(10)
    DS = find_ds()
assert DS, "dataset not mounted"
print("DS =", DS, flush=True)

# Kaggle은 업로드 zip을 폴더로 자동 해제: ad_common/ open/ + 평문 파일
# 신버전(v2: train_full_cli 포함) 처리 지연 대비 재시도
def fetch_file(name, tries=30):
    for _ in range(tries):
        hits = glob.glob(os.path.join(DS, "**", name), recursive=True)
        if hits:
            shutil.copy(hits[0], WORK)
            return
        time.sleep(20)
    raise AssertionError(f"{name} not in dataset (v2 미반영?)")

shutil.copytree(os.path.join(DS, "ad_common", "common"), os.path.join(WORK, "common"), dirs_exist_ok=True)
shutil.copytree(os.path.join(DS, "ad_common", "splits"), os.path.join(WORK, "splits"), dirs_exist_ok=True)
shutil.copytree(os.path.join(DS, "open", "data"), os.path.join(WORK, "data"), dirs_exist_ok=True)
fetch_file("train_full_cli.py")
fetch_file("soft_labels_str2.npz")
# 신버전 common인지 검증(v7 [PACE] 존재 = 최신 ad_lib)
assert "[PACE]" in open(os.path.join(WORK, "common", "ad_lib.py"), encoding="utf-8").read(), "구버전 ad_common 마운트 — 데이터셋 버전 확인"

import torch
NGPU = torch.cuda.device_count()
print("GPUs:", NGPU, torch.cuda.get_device_name(0), flush=True)
if torch.cuda.get_device_capability(0)[0] < 7:
    print("Pascal GPU — torch 2.4.1+cu121 설치", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.4.1", "--index-url", "https://download.pytorch.org/whl/cu121"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                    "torchvision", "torchaudio"], check=False)

BASE = dict(AD_MODEL="xlm-roberta-large", AD_VERSION="v6", AD_MAXLEN="320",
            AD_EPOCHS="8", AD_LR="2e-5", AD_BATCH="64", AD_LLRD="1",
            AD_PRUNE="1", AD_SWA_K="3", AD_GRADCKPT="1", AD_WORK=WORK,
            TOKENIZERS_PARALLELISM="false")
JOBS = [
    ("largev6s2", dict(AD_SEED="2024", AD_TAG="largev6s2")),
    ("largev6dist", dict(AD_SEED="1234", AD_TAG="largev6dist",
                         AD_SOFT=os.path.join(WORK, "soft_labels_str2.npz"))),
]

def run(job, dev):
    tag, extra = job
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=dev, **BASE, **extra)
    return subprocess.Popen([sys.executable, "train_full_cli.py"], env=env,
                            stdout=open(f"log_{tag}.txt", "w"), stderr=subprocess.STDOUT)

if NGPU >= 2:
    procs = []
    for i, job in enumerate(JOBS):
        procs.append((job[0], run(job, str(i))))
        time.sleep(60)
    for tag, p in procs:
        rc = p.wait()
        print(tag, "rc=", rc, flush=True)
        print(open(f"log_{tag}.txt").read()[-2500:], flush=True)
else:
    for job in JOBS:   # 순차 — 9h 한도 내 s2 우선
        tag = job[0]
        p = run(job, "0")
        rc = p.wait()
        print(tag, "rc=", rc, flush=True)
        print(open(f"log_{tag}.txt").read()[-2500:], flush=True)

# output 정리: member zip + 로그만 남김 (20GB 한도)
for d in glob.glob(os.path.join(WORK, "raw_*")) + glob.glob(os.path.join(WORK, "member_*/")) \
       + [os.path.join(WORK, x) for x in ("common", "splits", "data")]:
    shutil.rmtree(d, ignore_errors=True)
for f in ["ad_common.zip", "open.zip", "soft_labels_str2.npz"]:
    p = os.path.join(WORK, f)
    if os.path.exists(p):
        os.remove(p)
print("FILES:", os.listdir(WORK), flush=True)
