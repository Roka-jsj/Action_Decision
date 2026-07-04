# klue-v6-6ep 5-fold 교사 — Kaggle T4x2 듀얼GPU 병렬 (folds 0-2 / 3-4)
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
print("input tree:", glob.glob("/kaggle/input/**/", recursive=True)[:10], flush=True)
assert DS, "dataset not mounted"
print("DS =", DS, flush=True)
shutil.copytree(os.path.join(DS, "ad_common", "common"), os.path.join(WORK, "common"), dirs_exist_ok=True)
shutil.copytree(os.path.join(DS, "ad_common", "splits"), os.path.join(WORK, "splits"), dirs_exist_ok=True)
shutil.copytree(os.path.join(DS, "open", "data"), os.path.join(WORK, "data"), dirs_exist_ok=True)
shutil.copy(os.path.join(DS, "teacher_cli.py"), WORK)

import torch
print("GPUs:", torch.cuda.device_count(), flush=True)
if torch.cuda.device_count() and torch.cuda.get_device_capability(0)[0] < 7:
    # P100(sm60): 최신 torch가 커널 미지원 → 호환 버전으로 교체
    print("Pascal GPU 감지 — torch 2.4.1+cu121 설치(~5분)", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.4.1", "--index-url", "https://download.pytorch.org/whl/cu121"],
                   check=False)
    # torch 다운그레이드와 버전 불일치 → transformers가 감지 못 하게 제거
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                    "torchvision", "torchaudio"], check=False)

NGPU = torch.cuda.device_count()
if NGPU >= 2:
    JOBS = [("0", 0, 3, "kluev6_g0"), ("1", 3, 5, "kluev6_g1")]
else:
    JOBS = [("0", 0, 5, "kluev6_g0")]   # 단일GPU: 5fold 순차
print("JOBS:", JOBS, flush=True)
procs = []
for dev, lo, hi, tag in JOBS:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=dev, AD_WORK=WORK,
               AD_MODEL="klue/roberta-base", AD_VERSION="v6", AD_MAXLEN="320",
               AD_EPOCHS="6", AD_LR="3e-5", AD_BATCH="64", AD_LLRD="1",
               AD_FOLD_LO=str(lo), AD_FOLD_HI=str(hi), AD_TAG=tag,
               TOKENIZERS_PARALLELISM="false")
    p = subprocess.Popen([sys.executable, "teacher_cli.py"], env=env,
                         stdout=open(f"log_{tag}.txt", "w"), stderr=subprocess.STDOUT)
    procs.append((tag, p))
    time.sleep(90)   # pip 설치 경합 방지

for tag, p in procs:
    rc = p.wait()
    print(tag, "rc=", rc, flush=True)
    print(open(f"log_{tag}.txt").read()[-3000:], flush=True)

for d in ("common", "splits", "data"):
    shutil.rmtree(d, ignore_errors=True)
print("FILES:", os.listdir(WORK), flush=True)
