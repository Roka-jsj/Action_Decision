#!/usr/bin/env python
"""Track A(균형) CLI 학습 — Colab VM. 사전토큰화로 고속화(에폭마다 재토큰화 제거).

입력(같은 폴더): open.zip, ad_common.zip
출력: submit_balance.zip + DONE.
env: AD_MODEL, AD_VERSION, AD_MAXLEN, AD_EPOCHS, AD_BATCH, AD_FOLDS, AD_OUT, AD_WORK.
"""
import os, sys, subprocess, time, zipfile, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
WORK = os.environ.get("AD_WORK", "/content" if os.path.isdir("/content") else os.getcwd())
os.chdir(WORK)
print(f"[train_cli] WORK={WORK}", flush=True)
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
from common.metrics import macro_f1, print_report
from common import ad_lib, postproc, soup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.amp import autocast, GradScaler

CFG = dict(MODEL_NAME=os.environ.get("AD_MODEL", "xlm-roberta-base"),
           VERSION=os.environ.get("AD_VERSION", "v3"),
           MAX_LEN=int(os.environ.get("AD_MAXLEN", "160")),
           LR=float(os.environ.get("AD_LR", "2e-5")), EPOCHS=int(os.environ.get("AD_EPOCHS", "2")),
           BATCH=int(os.environ.get("AD_BATCH", "64")), EVAL_BATCH=256,
           N_FOLDS=int(os.environ.get("AD_FOLDS", "5")),
           FGM=os.environ.get("AD_FGM", "0") == "1",       # 임베딩 적대학습(시간 ~1.8x)
           LLRD=os.environ.get("AD_LLRD", "0") == "1",     # 층별 LR 감쇠(비용 0)
           WARMUP=0.06, WD=0.01, SEED=42, HEAD_SEED=1234, USE_CLASS_WEIGHT=True)
OUTDIR = os.environ.get("AD_OUT", "submit_balance"); MODELDIR = OUTDIR + "/model"
device = "cuda"; assert torch.cuda.is_available(), "GPU 필요"
print(f"[cfg] {CFG} device={device}", flush=True)

set_seed(CFG["SEED"])
samples, y, ids = load_train()
y = np.array(y); groups = np.array([s["session"] for s in samples]); gen = np.array([s["gen"] for s in samples])
sp = make_splits(ids, y, groups)
dev_idx, hold_idx = sp["dev_idx"], sp["holdout_idx"]
folds = sp["folds"][:CFG["N_FOLDS"]]
cnt = np.bincount(y[dev_idx], minlength=NUM_CLASSES); cw = len(dev_idx) / (NUM_CLASSES * np.maximum(cnt, 1)); cw /= cw.mean()

tok = AutoTokenizer.from_pretrained(CFG["MODEL_NAME"]); tok.truncation_side = "left"
# --- 사전토큰화(1회): 패딩 없이 input_ids 만 저장 → 배치에서 tok.pad (에폭 반복 비용 제거) ---
t_tok = time.time()
texts = [ad_lib.serialize(s, CFG["VERSION"]) for s in samples]
enc_all = tok(texts, truncation=True, max_length=CFG["MAX_LEN"], padding=False)
INPUT_IDS = enc_all["input_ids"]
print(f"[tok] pre-tokenized {len(texts)} in {time.time()-t_tok:.0f}s | dev={len(dev_idx)} holdout={len(hold_idx)} folds={len(folds)}", flush=True)


def build():
    torch.manual_seed(CFG["HEAD_SEED"])
    return AutoModelForSequenceClassification.from_pretrained(
        CFG["MODEL_NAME"], num_labels=NUM_CLASSES,
        id2label={i: c for i, c in enumerate(CLASSES)},
        label2id={c: i for i, c in enumerate(CLASSES)}).to(device)


def pad_batch(idx_list):
    return tok.pad({"input_ids": [INPUT_IDS[j] for j in idx_list]}, return_tensors="pt")


def infer_logits(model, idx):
    model.eval(); bs = CFG["EVAL_BATCH"]
    order = sorted(range(len(idx)), key=lambda k: len(INPUT_IDS[int(idx[k])]))
    out = np.zeros((len(idx), NUM_CLASSES), np.float32)
    with torch.no_grad():
        for b in range(0, len(order), bs):
            ks = order[b:b + bs]; sub = [int(idx[k]) for k in ks]
            enc = pad_batch(sub).to(device)
            with autocast("cuda", dtype=torch.float16):
                lg = model(**enc).logits.float().cpu().numpy()
            for m, k in enumerate(ks):
                out[k] = lg[m]
    return out


class FGM:
    """임베딩 공간 적대 섭동(eps=1.0) — macro-F1 안정 가산 기법."""
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


def make_optimizer(model):
    if not CFG["LLRD"]:
        return torch.optim.AdamW(model.parameters(), lr=CFG["LR"], weight_decay=CFG["WD"])
    # LLRD: 헤드 1.5x, 인코더 층 위→아래 0.9^k 감쇠, 임베딩 최저
    base, decay = CFG["LR"], 0.9
    nl = model.config.num_hidden_layers
    groups = []
    seen = set()
    def add(params, lr):
        ps = [p for p in params if id(p) not in seen and p.requires_grad]
        for p in ps: seen.add(id(p))
        if ps: groups.append({"params": ps, "lr": lr})
    add([p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n], base * 1.5)
    for i in range(nl - 1, -1, -1):
        add([p for n, p in model.named_parameters() if f"encoder.layer.{i}." in n], base * (decay ** (nl - 1 - i)))
    add([p for n, p in model.named_parameters() if "embeddings" in n], base * (decay ** nl))
    add([p for _, p in model.named_parameters()], base)  # 잔여
    return torch.optim.AdamW(groups, lr=base, weight_decay=CFG["WD"])


def train_fold(tr, va):
    model = build()
    opt = make_optimizer(model)
    tr = np.asarray(tr)

    class DS(torch.utils.data.Dataset):
        def __len__(s): return len(tr)
        def __getitem__(s, i): return int(tr[i])

    def coll(b):
        enc = pad_batch(b); return enc, torch.tensor([y[j] for j in b])

    dl = torch.utils.data.DataLoader(DS(), batch_size=CFG["BATCH"], shuffle=True, collate_fn=coll,
                                     num_workers=4, pin_memory=True, persistent_workers=True)
    tot = len(dl) * CFG["EPOCHS"]; sch = get_linear_schedule_with_warmup(opt, int(tot * CFG["WARMUP"]), tot)
    scaler = GradScaler("cuda")
    wt = torch.tensor(cw, dtype=torch.float, device=device) if CFG["USE_CLASS_WEIGHT"] else None
    lossfn = torch.nn.CrossEntropyLoss(weight=wt)
    fgm = FGM(model) if CFG["FGM"] else None
    best, best_state = -1, None
    for ep in range(CFG["EPOCHS"]):
        model.train()
        for enc, lb in dl:
            enc = {k: v.to(device) for k, v in enc.items()}; lb = lb.to(device); opt.zero_grad()
            with autocast("cuda", dtype=torch.float16):
                loss = lossfn(model(**enc).logits, lb)
            scaler.scale(loss).backward()
            if fgm is not None:
                fgm.attack()
                with autocast("cuda", dtype=torch.float16):
                    adv_loss = lossfn(model(**enc).logits, lb)
                scaler.scale(adv_loss).backward()
                fgm.restore()
            scaler.step(opt); scaler.update(); sch.step()
        vlog = infer_logits(model, va); mf1, _ = macro_f1(y[va], vlog.argmax(1))
        print(f"    epoch {ep+1}: val macroF1={mf1:.4f}", flush=True)
        if mf1 > best:
            best = mf1; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_state, best


oof = np.zeros((len(samples), NUM_CLASSES), np.float32)
fold_states, fold_scores = [], []
t0 = time.time()
for fi, (tr, va) in enumerate(folds):
    print(f"=== fold {fi} (tr={len(tr)} va={len(va)}) ===", flush=True)
    st, sc = train_fold(tr, va); fold_states.append(st); fold_scores.append(sc)
    m = build(); m.load_state_dict(st); oof[va] = infer_logits(m, va); del m; torch.cuda.empty_cache()
    print(f"    [fold {fi} done @ {(time.time()-t0)/60:.1f} min]", flush=True)
print(f"[train] {(time.time()-t0)/60:.1f} min | fold scores {[round(s,4) for s in fold_scores]}", flush=True)

covered = np.concatenate([va for _, va in folds])
oof_pred = oof[covered].argmax(1); oof_mf1, _ = macro_f1(y[covered], oof_pred)
sim_cov = covered[gen[covered] == "sim"]
print(f"[OOF] pooled-OOF macroF1={oof_mf1:.4f} acc={(oof[covered].argmax(1)==y[covered]).mean():.4f} n={len(covered)}", flush=True)
print(f"[OOF] sim-only macroF1={macro_f1(y[sim_cov], oof[sim_cov].argmax(1))[0]:.4f}", flush=True)
print_report(y[covered], oof_pred, "OOF per-class")


def eval_state(state):
    m = build(); m.load_state_dict(state); s = macro_f1(y[hold_idx], infer_logits(m, hold_idx).argmax(1))[0]
    del m; torch.cuda.empty_cache(); return s


souped, ingr, soup_hold = soup.greedy_soup(fold_states, fold_scores, eval_state)
uni = soup.uniform_soup(fold_states); uni_hold = eval_state(uni)
print(f"[soup] greedy ingr={ingr} holdout={soup_hold:.4f} | uniform holdout={uni_hold:.4f}", flush=True)
best_state = souped if soup_hold >= uni_hold else uni

bias, fit = postproc.fit_bias(oof[covered], y[covered])
sm = build(); sm.load_state_dict(best_state); hlog = infer_logits(sm, hold_idx)
h_nob = macro_f1(y[hold_idx], hlog.argmax(1))[0]
h_b = macro_f1(y[hold_idx], (ad_lib._to_logprobs_np(hlog, np) + bias).argmax(1))[0]
USE_BIAS = bool(h_b >= h_nob - 1e-4)
print(f"[postproc] OOF bias-fit={fit:.4f} | holdout no-bias={h_nob:.4f} +bias={h_b:.4f} use_bias={USE_BIAS}", flush=True)

import shutil
os.makedirs(MODELDIR, exist_ok=True)
sm.half().save_pretrained(MODELDIR, safe_serialization=True)
tok.save_pretrained(MODELDIR)
shutil.copy("common/ad_lib.py", os.path.join(MODELDIR, "ad_lib.py"))
postproc.save(os.path.join(MODELDIR, "postproc.json"), bias if USE_BIAS else np.zeros(NUM_CLASSES),
              {"use_bias": USE_BIAS, "oof_macf1": float(oof_mf1), "holdout_macf1": float(max(h_b, h_nob))})
json.dump({"version": CFG["VERSION"], "max_len": CFG["MAX_LEN"], "batch_size": 128},
          open(os.path.join(MODELDIR, "run_meta.json"), "w"))
shutil.copy("common/server_script.py", OUTDIR + "/script.py")
open(OUTDIR + "/requirements.txt", "w").write("")
mb = sum(os.path.getsize(os.path.join(MODELDIR, f)) for f in os.listdir(MODELDIR)) / 1e6
print(f"[save] model/ size={mb:.1f}MB", flush=True)

bidx = np.random.RandomState(0).choice(dev_idx, size=min(30000, len(dev_idx)), replace=False)
bench = [samples[int(i)] for i in bidx]; t = time.time()
_ = ad_lib.predict(MODELDIR, bench, version=CFG["VERSION"], max_len=CFG["MAX_LEN"], batch_size=128,
                   postproc_path=MODELDIR + "/postproc.json")
dt = time.time() - t
print(f"[timing] 30k 추론 {dt:.1f}s (제한 600s) {'OK' if dt<600 else 'SLOW'}", flush=True)

import subprocess as sp2
run = os.path.join(WORK, "simrun"); shutil.rmtree(run, ignore_errors=True); os.makedirs(run + "/data")
for p in ["model", "script.py", "requirements.txt"]:
    sp2.run(["cp", "-r", f"{OUTDIR}/{p}", run + "/"], check=True)
for p in ["test.jsonl", "sample_submission.csv"]:
    sp2.run(["cp", f"data/{p}", run + "/data/"], check=True)
r = sp2.run([sys.executable, "script.py"], cwd=run, capture_output=True, text=True)
print("[smoke]", r.stdout.strip(), r.stderr.strip()[-300:], flush=True)

def zipdir(src, zf):
    for root, _, fs in os.walk(src):
        for fn in fs:
            fp = os.path.join(root, fn); zf.write(fp, os.path.relpath(fp, src))
zippath = os.path.join(WORK, OUTDIR + ".zip")
with zipfile.ZipFile(zippath, "w", zipfile.ZIP_DEFLATED) as z:
    zipdir(OUTDIR, z)
zmb = os.path.getsize(zippath) / 1e6
tops = sorted(set(n.split("/")[0] for n in zipfile.ZipFile(zippath).namelist()))
print(f"[zip] {zippath} {zmb:.0f}MB tops={tops} ok={set(tops)<={'model','script.py','requirements.txt'}}", flush=True)
open(os.path.join(WORK, "DONE"), "w").write(f"oof={oof_mf1:.4f} holdout={max(h_b,h_nob):.4f} zip={zmb:.0f}MB timing={dt:.0f}s")
print("=== DONE ===", flush=True)
