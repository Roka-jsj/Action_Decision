#!/usr/bin/env python
"""교사 1개 구성(모델×시드) 5-fold 학습 → OOF/holdout 확률만 npz 저장(~8MB).

가중치는 저장 안 함(증류에는 확률만 필요) → 다운로드 초경량, 60분 세션에 최적.
env: AD_MODEL, AD_SEED, AD_VERSION, AD_MAXLEN, AD_EPOCHS, AD_LR, AD_BATCH,
     AD_FP16(1/0), AD_LLRD, AD_FGM, AD_FOLD_LO, AD_FOLD_HI(대형모델 분할용), AD_TAG.
출력: /content/teacher_<TAG>.npz + DONE_<TAG>
"""
import os, sys, subprocess, time, zipfile, json
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
from common.cv import make_splits
from common.metrics import macro_f1
from common import ad_lib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.amp import autocast, GradScaler

MODEL = os.environ.get("AD_MODEL", "xlm-roberta-base")
SEED = int(os.environ.get("AD_SEED", "1234"))
VERSION = os.environ.get("AD_VERSION", "v5")
MAX_LEN = int(os.environ.get("AD_MAXLEN", "320"))
EPOCHS = int(os.environ.get("AD_EPOCHS", "4"))
LR = float(os.environ.get("AD_LR", "3e-5"))
BATCH = int(os.environ.get("AD_BATCH", "128"))
FP16 = os.environ.get("AD_FP16", "1") == "1"
LLRD = os.environ.get("AD_LLRD", "1") == "1"
FGM_ON = os.environ.get("AD_FGM", "0") == "1"
FOLD_LO = int(os.environ.get("AD_FOLD_LO", "0"))
FOLD_HI = int(os.environ.get("AD_FOLD_HI", "5"))
EXCLUDE_AU = os.environ.get("AD_EXCLUDE_AU", "0") == "1"   # sim-only 학습 프로브
TAG = os.environ.get("AD_TAG", f"{MODEL.split('/')[-1]}_s{SEED}_f{FOLD_LO}{FOLD_HI}")
HEAD_SEED = 1234
device = "cuda"; assert torch.cuda.is_available()
print(f"[teacher] {TAG} model={MODEL} v={VERSION} len={MAX_LEN} ep={EPOCHS} lr={LR} "
      f"b={BATCH} fp16={FP16} llrd={LLRD} fgm={FGM_ON} folds=[{FOLD_LO},{FOLD_HI})", flush=True)

set_seed(SEED)
samples, y, ids = load_train()
y = np.array(y); groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups)
dev_idx, hold_idx = sp["dev_idx"], sp["holdout_idx"]
folds = sp["folds"]
cnt = np.bincount(y[dev_idx], minlength=NUM_CLASSES)
cw = len(dev_idx) / (NUM_CLASSES * np.maximum(cnt, 1)); cw /= cw.mean()

tok = AutoTokenizer.from_pretrained(MODEL); tok.truncation_side = "left"
texts = [ad_lib.serialize(s, VERSION) for s in samples]
t0 = time.time()
enc_all = tok(texts, truncation=True, max_length=MAX_LEN, padding=False)
INPUT_IDS = enc_all["input_ids"]
print(f"[tok] {len(texts)} in {time.time()-t0:.0f}s", flush=True)


def build():
    torch.manual_seed(HEAD_SEED)
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=NUM_CLASSES,
        id2label={i: c for i, c in enumerate(CLASSES)},
        label2id={c: i for i, c in enumerate(CLASSES)}).to(device)


def pad_batch(idx_list):
    return tok.pad({"input_ids": [INPUT_IDS[j] for j in idx_list]}, return_tensors="pt")


def infer_probs(model, idx):
    model.eval(); bs = 192
    order = sorted(range(len(idx)), key=lambda k: len(INPUT_IDS[int(idx[k])]))
    out = np.zeros((len(idx), NUM_CLASSES), np.float32)
    with torch.no_grad():
        for b in range(0, len(order), bs):
            ks = order[b:b + bs]; sub = [int(idx[k]) for k in ks]
            enc = pad_batch(sub).to(device)
            if FP16:
                with autocast("cuda", dtype=torch.float16):
                    lg = model(**enc).logits.float()
            else:
                lg = model(**enc).logits.float()
            p = torch.softmax(lg, 1).cpu().numpy()
            for m, k in enumerate(ks):
                out[k] = p[m]
    return out


class FGM:
    def __init__(self, model, eps=1.0):
        self.model, self.eps, self.backup = model, eps, {}
    def attack(self, emb_name="word_embeddings"):
        for n, p in self.model.named_parameters():
            if p.requires_grad and emb_name in n and p.grad is not None:
                self.backup[n] = p.data.clone()
                norm = torch.norm(p.grad)
                if norm and not torch.isnan(norm):
                    p.data.add_(self.eps * p.grad / norm)
    def restore(self):
        for n, p in self.model.named_parameters():
            if n in self.backup:
                p.data = self.backup[n]
        self.backup = {}


def make_opt(model):
    if not LLRD:
        return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    base, decay = LR, 0.9
    nl = model.config.num_hidden_layers
    groups, seen = [], set()
    def add(ps, lr):
        ps = [p for p in ps if id(p) not in seen and p.requires_grad]
        for p in ps: seen.add(id(p))
        if ps: groups.append({"params": ps, "lr": lr})
    add([p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n], base * 1.5)
    for i in range(nl - 1, -1, -1):
        add([p for n, p in model.named_parameters() if f"encoder.layer.{i}." in n], base * (decay ** (nl - 1 - i)))
    add([p for n, p in model.named_parameters() if "embeddings" in n], base * (decay ** nl))
    add([p for _, p in model.named_parameters()], base)
    return torch.optim.AdamW(groups, lr=base, weight_decay=0.01)


oof = np.zeros((len(samples), NUM_CLASSES), np.float32)
hold_sum = np.zeros((len(hold_idx), NUM_CLASSES), np.float32)
scores = []
t0 = time.time()
GEN = np.array([s["gen"] for s in samples])
for fi in range(FOLD_LO, FOLD_HI):
    tr, va = folds[fi]
    print(f"=== fold {fi} ===", flush=True)
    model = build(); opt = make_opt(model)
    tr = np.asarray(tr)
    if EXCLUDE_AU:
        n0 = len(tr); tr = tr[GEN[tr] == "sim"]
        print(f"    [exclude_au] train {n0} -> {len(tr)}", flush=True)

    class DS(torch.utils.data.Dataset):
        def __len__(s): return len(tr)
        def __getitem__(s, i): return int(tr[i])

    def coll(b):
        return pad_batch(b), torch.tensor([y[j] for j in b])

    dl = torch.utils.data.DataLoader(DS(), batch_size=BATCH, shuffle=True, collate_fn=coll,
                                     num_workers=4, pin_memory=True, persistent_workers=True)
    tot = len(dl) * EPOCHS
    sch = get_linear_schedule_with_warmup(opt, int(tot * 0.06), tot)
    scaler = GradScaler("cuda", enabled=FP16)
    lossfn = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float, device=device))
    fgm = FGM(model) if FGM_ON else None
    best, bva, bho = -1, None, None
    for ep in range(EPOCHS):
        model.train()
        for enc, lb in dl:
            enc = {k: v.to(device) for k, v in enc.items()}; lb = lb.to(device); opt.zero_grad()
            if FP16:
                with autocast("cuda", dtype=torch.float16):
                    loss = lossfn(model(**enc).logits, lb)
            else:
                loss = lossfn(model(**enc).logits, lb)
            scaler.scale(loss).backward()
            if fgm is not None:
                fgm.attack()
                if FP16:
                    with autocast("cuda", dtype=torch.float16):
                        aloss = lossfn(model(**enc).logits, lb)
                else:
                    aloss = lossfn(model(**enc).logits, lb)
                scaler.scale(aloss).backward()
                fgm.restore()
            scaler.step(opt); scaler.update(); sch.step()
        pv = infer_probs(model, va)
        mf1, _ = macro_f1(y[va], pv.argmax(1))
        sim_mask = GEN[np.asarray(va)] == "sim"
        smf1, _ = macro_f1(y[np.asarray(va)[sim_mask]], pv[sim_mask].argmax(1))
        print(f"    epoch {ep+1}: val={mf1:.4f} sim={smf1:.4f} @{(time.time()-t0)/60:.1f}min", flush=True)
        if mf1 > best:
            best = mf1; bva = pv; bho = infer_probs(model, hold_idx)
    oof[va] = bva; hold_sum += bho; scores.append(best)
    del model; torch.cuda.empty_cache()
    # 증분 저장: 세션이 죽어도 완료 fold까지 보존 (fold_hi=현재까지)
    np.savez_compressed(os.path.join(WORK, f"teacher_{TAG}.npz"),
                        oof=oof, hold=hold_sum / max(len(scores), 1),
                        scores=np.array(scores), fold_lo=FOLD_LO, fold_hi=fi + 1,
                        model=MODEL, version=VERSION, max_len=MAX_LEN)
    print(f"    [incremental npz saved: folds {FOLD_LO}..{fi}]", flush=True)

nf = FOLD_HI - FOLD_LO
np.savez_compressed(os.path.join(WORK, f"teacher_{TAG}.npz"),
                    oof=oof, hold=hold_sum / max(nf, 1),
                    scores=np.array(scores), fold_lo=FOLD_LO, fold_hi=FOLD_HI,
                    model=MODEL, version=VERSION, max_len=MAX_LEN)
cov = np.concatenate([folds[i][1] for i in range(FOLD_LO, FOLD_HI)])
pmf1, _ = macro_f1(y[cov], oof[cov].argmax(1))
print(f"[teacher {TAG}] fold_scores={[round(s,4) for s in scores]} covered-OOF={pmf1:.4f} "
      f"time={(time.time()-t0)/60:.1f}min", flush=True)
open(os.path.join(WORK, f"DONE_{TAG}"), "w").write(f"oof={pmf1:.4f} scores={scores}")
print("=== DONE ===", flush=True)
