#!/usr/bin/env python
"""FULL-70k 최종 멤버 학습 — 전체 데이터(검증 없음) → (옵션)vocab 프루닝 → member zip.

env: AD_MODEL, AD_VERSION(v4), AD_MAXLEN(320), AD_EPOCHS, AD_LR, AD_BATCH,
     AD_LLRD(1), AD_SEED, AD_TAG, AD_PRUNE(1: xlm-r 계열 프루닝 / 0: klue 등 소형 vocab).
출력: /content/member_<TAG>.zip (모델 디렉터리) + DONE_<TAG>
"""
import os, sys, subprocess, time, zipfile, json, shutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"
WORK = os.environ.get("AD_WORK", "/content" if os.path.isdir("/content") else os.getcwd())
os.chdir(WORK)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==4.46.3", "accelerate==1.9.0", "sentencepiece==0.1.99"], check=False)
for z in ["open.zip", "ad_common.zip"]:
    if os.path.exists(z):
        with zipfile.ZipFile(z) as f:
            f.extractall(".")
sys.path.insert(0, WORK)

import numpy as np, torch
from common.io_utils import load_train, CLASSES, NUM_CLASSES, set_seed
from common import ad_lib
from common.vocab_prune import prune_model_dir
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.amp import autocast, GradScaler

MODEL = os.environ.get("AD_MODEL", "xlm-roberta-base")
VERSION = os.environ.get("AD_VERSION", "v4")
MAX_LEN = int(os.environ.get("AD_MAXLEN", "320"))
EPOCHS = int(os.environ.get("AD_EPOCHS", "5"))
LR = float(os.environ.get("AD_LR", "3e-5"))
BATCH = int(os.environ.get("AD_BATCH", "96"))
LLRD = os.environ.get("AD_LLRD", "1") == "1"
SEED = int(os.environ.get("AD_SEED", "1234"))
PRUNE = os.environ.get("AD_PRUNE", "1") == "1"
TAG = os.environ.get("AD_TAG", "member")
device = "cuda"; assert torch.cuda.is_available()
print(f"[full] {TAG}: {MODEL} v={VERSION} len={MAX_LEN} ep={EPOCHS} lr={LR} b={BATCH} prune={PRUNE}", flush=True)

set_seed(SEED)
samples, y, ids = load_train()
y = np.array(y)
tok = AutoTokenizer.from_pretrained(MODEL); tok.truncation_side = "left"
texts = [ad_lib.serialize(s, VERSION) for s in samples]
enc_all = tok(texts, truncation=True, max_length=MAX_LEN, padding=False)
INPUT_IDS = enc_all["input_ids"]
cnt = np.bincount(y, minlength=NUM_CLASSES)
cw = len(y) / (NUM_CLASSES * np.maximum(cnt, 1)); cw /= cw.mean()

torch.manual_seed(1234)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=NUM_CLASSES,
    id2label={i: c for i, c in enumerate(CLASSES)},
    label2id={c: i for i, c in enumerate(CLASSES)}).to(device)

def make_opt(m):
    if not LLRD:
        return torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    base, decay = LR, 0.9
    nl = m.config.num_hidden_layers
    groups, seen = [], set()
    def add(ps, lr):
        ps = [p for p in ps if id(p) not in seen and p.requires_grad]
        for p in ps: seen.add(id(p))
        if ps: groups.append({"params": ps, "lr": lr})
    add([p for n, p in m.named_parameters() if "classifier" in n or "pooler" in n], base * 1.5)
    for i in range(nl - 1, -1, -1):
        add([p for n, p in m.named_parameters() if f"encoder.layer.{i}." in n], base * (decay ** (nl - 1 - i)))
    add([p for n, p in m.named_parameters() if "embeddings" in n], base * (decay ** nl))
    add([p for _, p in m.named_parameters()], base)
    return torch.optim.AdamW(groups, lr=base, weight_decay=0.01)

def pad_batch(idx_list):
    return tok.pad({"input_ids": [INPUT_IDS[j] for j in idx_list]}, return_tensors="pt")

class DS(torch.utils.data.Dataset):
    def __len__(s): return len(samples)
    def __getitem__(s, i): return i

def coll(b):
    return pad_batch(b), torch.tensor([y[j] for j in b])

opt = make_opt(model)
dl = torch.utils.data.DataLoader(DS(), batch_size=BATCH, shuffle=True, collate_fn=coll,
                                 num_workers=4, pin_memory=True, persistent_workers=True)
tot = len(dl) * EPOCHS
sch = get_linear_schedule_with_warmup(opt, int(tot * 0.06), tot)
scaler = GradScaler("cuda")
lossfn = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float, device=device))
t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    for enc, lb in dl:
        enc = {k: v.to(device) for k, v in enc.items()}; lb = lb.to(device); opt.zero_grad()
        with autocast("cuda", dtype=torch.float16):
            loss = lossfn(model(**enc).logits, lb)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
    print(f"  epoch {ep+1} done @{(time.time()-t0)/60:.1f}min", flush=True)

# 저장 → (옵션) 프루닝
raw_dir = os.path.join(WORK, f"raw_{TAG}")
model.half().save_pretrained(raw_dir, safe_serialization=True)
tok.save_pretrained(raw_dir)
out_dir = os.path.join(WORK, f"member_{TAG}")
if PRUNE:
    K, _ = prune_model_dir(raw_dir, out_dir, tok, texts, max_len=MAX_LEN)
    print(f"[prune] vocab -> {K}", flush=True)
else:
    shutil.copytree(raw_dir, out_dir, dirs_exist_ok=True)
mb = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(out_dir) for f in fs) / 1e6
print(f"[member] {out_dir} size={mb:.0f}MB", flush=True)

zp = os.path.join(WORK, f"member_{TAG}.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for r, _, fs in os.walk(out_dir):
        for f in fs:
            z.write(os.path.join(r, f), os.path.relpath(os.path.join(r, f), out_dir))
print(f"[zip] {zp} {os.path.getsize(zp)/1e6:.0f}MB", flush=True)
open(os.path.join(WORK, f"DONE_{TAG}"), "w").write(f"size={mb:.0f}MB time={(time.time()-t0)/60:.1f}min")
print("=== DONE ===", flush=True)
