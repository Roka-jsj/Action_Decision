#!/usr/bin/env python3
"""DAPT: train 직렬화 텍스트(v6, mht12)로 xlm-roberta-large MLM 계속학습.
라벨 무접촉(Class-M 암기반전 우회). 산출 ckpt는 train_full_cli AD_MODEL=<path>로 파인튜닝.
사용: CUDA_VISIBLE_DEVICES=N python3 sim/dapt_mlm.py [--epochs 2] [--lr 1e-5] [--out work/dapt_xlmr]
"""
import os, sys, json, time, argparse
sys.path.insert(0, "/root/Action_Decision/common")
os.chdir("/root/Action_Decision")
import ad_lib

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--max_len", type=int, default=320)
ap.add_argument("--out", default="work/dapt_xlmr")
ap.add_argument("--model", default="xlm-roberta-large")
a = ap.parse_args()

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
assert torch.cuda.is_available()

rows = [json.loads(l) for l in open("data/train.jsonl")]
texts = [ad_lib.serialize(r, "v6", 12) for r in rows]   # mht12 = 파인튜닝과 동일 직렬화
print(f"[dapt] texts {len(texts)}  ep={a.epochs} lr={a.lr} b={a.batch}", flush=True)

tok = AutoTokenizer.from_pretrained(a.model)
tok.truncation_side = "left"
model = AutoModelForMaskedLM.from_pretrained(a.model).cuda()
model.gradient_checkpointing_enable()

class DS(Dataset):
    def __len__(self): return len(texts)
    def __getitem__(self, i):
        e = tok(texts[i], truncation=True, max_length=a.max_len, return_special_tokens_mask=True)
        return e
coll = DataCollatorForLanguageModeling(tok, mlm_probability=0.15)
dl = DataLoader(DS(), batch_size=a.batch, shuffle=True, collate_fn=coll, num_workers=2, drop_last=True)

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda")
steps = len(dl) * a.epochs
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.06)
t0 = time.time()
model.train()
for ep in range(a.epochs):
    tot = n = 0
    for i, b in enumerate(dl):
        b = {k: v.cuda() for k, v in b.items()}
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(**b)
        scaler.scale(out.loss).backward()
        scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        tot += out.loss.item(); n += 1
        if i % 200 == 0:
            print(f"  ep{ep} step{i}/{len(dl)} loss={tot/max(n,1):.4f} @{(time.time()-t0)/60:.1f}min", flush=True)
    print(f"[dapt] epoch {ep} done loss={tot/n:.4f} @{(time.time()-t0)/60:.1f}min", flush=True)

os.makedirs(a.out, exist_ok=True)
# MLM헤드 제외 인코더만 저장해도 되지만, 전체 저장 후 분류학습이 무시하게 둠
model.save_pretrained(a.out, safe_serialization=True)
tok.save_pretrained(a.out)
open("work/DONE_dapt_mlm", "w").write(f"{time.time()}")
print(f"[dapt] saved -> {a.out}  total {(time.time()-t0)/60:.1f}min", flush=True)
