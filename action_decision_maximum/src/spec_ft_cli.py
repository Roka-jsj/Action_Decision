#!/usr/bin/env python
"""R30 confusion-pair specialist fold0 클린 프로브 (codex 새 1순위, 기대 +0.0015~0.004).

E={read_file,grep_search,list_directory,glob_pattern} 4-way 전용 모델:
fold0 baseline ckpt(fold0-val 미학습)에서 4-way 헤드 재초기화 FT → fold0-train∩E 학습
→ 판정: 배포표적셋(base pred∈E & margin<th)에서 spec 4-way acc vs base acc.
GO 문턱: spec ≥ base + 0.05 (base ≈ 0.40, OOF 시뮬상 +0.05p ≈ ΔF1 +0.0045).

env: AD_INIT_FROM(기본 work/foldckpt_largev6_f0ckpt_f0), AD_EPOCHS(3), AD_LR(1e-5),
     AD_BATCH(64), AD_SAVE_DIR(통과시 저장).
"""
from __future__ import annotations
import os, sys, glob, time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
R = "/root/Action_Decision"
sys.path.insert(0, R)

import numpy as np, torch
from common.io_utils import load_train, CLASSES, NUM_CLASSES, set_seed
from common.cv import make_splits
from common.metrics import macro_f1
from common.postproc import fit_bias
from common import ad_lib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.amp import autocast, GradScaler

INIT = os.environ.get("AD_INIT_FROM", f"{R}/work/foldckpt_largev6_f0ckpt_f0")
EPOCHS = int(os.environ.get("AD_EPOCHS", "3"))
LR = float(os.environ.get("AD_LR", "1e-5"))
BATCH = int(os.environ.get("AD_BATCH", "64"))
MAX_LEN = 320
SAVE_DIR = os.environ.get("AD_SAVE_DIR", "")
E_CLASSES = ["read_file", "grep_search", "list_directory", "glob_pattern"]
E_IDS = [CLASSES.index(c) for c in E_CLASSES]
device = "cuda"; assert torch.cuda.is_available()
set_seed(1234)

samples, y, ids = load_train(); y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups)
tr0, va0 = np.asarray(sp["folds"][0][0]), np.asarray(sp["folds"][0][1])

# base(fold0 baseline) 확률 + 배포형 bias → 표적셋 정의
bz = np.load(f"{R}/work/teacher_largev6_f0ckpt.npz", allow_pickle=True)
P_base = bz["oof"][va0].astype(np.float64)
P_base /= P_base.sum(1, keepdims=True)
P_oof, cov = np.zeros((len(y), NUM_CLASSES)), np.zeros(len(y), bool)
for f in sorted(glob.glob(f"{R}/action_decision_maximum/experiments/teacher_largev6[AB]_a*.npz")):
    z = np.load(f, allow_pickle=True)
    o = z["oof"].astype(np.float64); m = o.sum(1) > 0; P_oof[m] = o[m]; cov |= m
oi = np.where(cov)[0]; Po = P_oof[oi]; Po /= Po.sum(1, keepdims=True)
bias, _ = fit_bias(np.log(Po + 1e-12), y[oi])
Lb = np.log(P_base + 1e-12) + bias
Pb = np.exp(Lb); Pb /= Pb.sum(1, keepdims=True)
pred_b = Pb.argmax(1)
srt = np.sort(Pb, 1); margin = srt[:, -1] - srt[:, -2]
TARGETS = {f"margin<{t}": np.isin(pred_b, E_IDS) & (margin < t) for t in (0.10, 0.15)}
for k, m in TARGETS.items():
    ba = (pred_b[m] == y[va0][m]).mean()
    print(f"[target {k}] n={m.sum()} ({m.mean()*100:.1f}% of val) base_acc={ba:.3f} true∈E={np.isin(y[va0][m], E_IDS).mean():.3f}", flush=True)

# specialist 학습 데이터: fold0-train ∩ (y∈E), 라벨 0-3 재맵
remap = {c: i for i, c in enumerate(E_IDS)}
trE = tr0[np.isin(y[tr0], E_IDS)]
yE = np.array([remap[int(c)] for c in y[trE]])
print(f"[spec] train rows {len(trE)} (fold0-train의 {len(trE)/len(tr0)*100:.0f}%)", flush=True)

tok = AutoTokenizer.from_pretrained(INIT); tok.truncation_side = "left"
texts = [ad_lib.serialize(s, "v6") for s in samples]
enc_all = tok(texts, truncation=True, max_length=MAX_LEN, padding=False)["input_ids"]

model = AutoModelForSequenceClassification.from_pretrained(
    INIT, num_labels=4, torch_dtype=torch.float32, ignore_mismatched_sizes=True,
    id2label={i: c for i, c in enumerate(E_CLASSES)},
    label2id={c: i for i, c in enumerate(E_CLASSES)}).to(device)

def make_opt(m):
    base_lr, decay = LR, 0.9
    nl = m.config.num_hidden_layers
    gs, seen = [], set()
    def add(ps, lr):
        ps = [p for p in ps if id(p) not in seen and p.requires_grad]
        for p in ps: seen.add(id(p))
        if ps: gs.append({"params": ps, "lr": lr})
    add([p for n, p in m.named_parameters() if "classifier" in n or "pooler" in n], base_lr * 3)  # 새 헤드 가속
    for i in range(nl - 1, -1, -1):
        add([p for n, p in m.named_parameters() if f"encoder.layer.{i}." in n], base_lr * (decay ** (nl - 1 - i)))
    add([p for n, p in m.named_parameters() if "embeddings" in n], base_lr * (decay ** nl))
    add([p for _, p in m.named_parameters()], base_lr)
    return torch.optim.AdamW(gs, lr=base_lr, weight_decay=0.01)

def pad(idx_list):
    return tok.pad({"input_ids": [enc_all[j] for j in idx_list]}, return_tensors="pt")

def infer4(idx):
    model.eval(); out = np.zeros((len(idx), 4), np.float32)
    order = sorted(range(len(idx)), key=lambda k: len(enc_all[int(idx[k])]))
    with torch.no_grad():
        for b in range(0, len(order), 192):
            ks = order[b:b + 192]
            e = pad([int(idx[k]) for k in ks]).to(device)
            with autocast("cuda", dtype=torch.float16):
                lg = model(**e).logits.float()
            p = torch.softmax(lg, 1).cpu().numpy()
            for m_, k in enumerate(ks):
                out[k] = p[m_]
    return out

cnt = np.bincount(yE, minlength=4)
cw = len(yE) / (4 * np.maximum(cnt, 1)); cw /= cw.mean()
opt = make_opt(model)
steps = (len(trE) + BATCH - 1) // BATCH
sch = get_linear_schedule_with_warmup(opt, int(steps * EPOCHS * 0.06), steps * EPOCHS)
scaler = GradScaler("cuda")
lossfn = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float, device=device))
rng = np.random.RandomState(42)
t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    order = rng.permutation(len(trE))
    for b in range(0, len(order), BATCH):
        bb = order[b:b + BATCH]
        e = {k: v.to(device) for k, v in pad([int(trE[j]) for j in bb]).items()}
        lb = torch.tensor(yE[bb]).to(device)
        opt.zero_grad()
        with autocast("cuda", dtype=torch.float16):
            loss = lossfn(model(**e).logits, lb)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
    # 판정: 표적셋 spec acc vs base acc
    PS = infer4(va0)
    spec_pred_full = np.array(E_IDS)[PS.argmax(1)]
    lines = [f"epoch {ep+1} @{(time.time()-t0)/60:.1f}min"]
    for k, m in TARGETS.items():
        sa = (spec_pred_full[m] == y[va0][m]).mean()
        ba = (pred_b[m] == y[va0][m]).mean()
        lines.append(f"{k}: spec {sa:.3f} vs base {ba:.3f} ({sa-ba:+.3f})")
    vE = np.isin(y[va0], E_IDS)
    lines.append(f"val∩E 4way acc {(spec_pred_full[vE]==y[va0][vE]).mean():.3f}")
    print("  " + " | ".join(lines), flush=True)

if SAVE_DIR:
    model.half().save_pretrained(SAVE_DIR, safe_serialization=True)
    tok.save_pretrained(SAVE_DIR)
    print(f"[saved] {SAVE_DIR}", flush=True)
print("=== DONE ===", flush=True)
