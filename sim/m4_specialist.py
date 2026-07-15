#!/usr/bin/env python3
"""M4 4-class 전문가(specialist) 학습기 — 신규파일(기존파일 무수정).

가설: M4 항법블록 {read_file=0, grep_search=1, list_directory=2, glob_pattern=3}의
4-way 구분신호가 HISTORY/문맥에 있을 수 있고, 4-way에 전 용량을 쓰는 전문가가 이를 추출한다.

설계(문서화):
- Base: xlm-roberta-large, num_labels=4 (label 0/1/2/3 항등 매핑).
- 학습셋: fold0-TRAIN(folds[0][0]) 중 true label∈{0,1,2,3} 21009행. 이를 세션그룹으로
  train(92%)/internal-dev(8%) 분할 — best-epoch은 internal-dev 4-way macro-F1로 선택
  (fold0-val 무접촉 → 게이트 무편향). fold0-val 성능은 로깅만(선택 미사용).
- 직렬화: ad_lib.serialize(s,"v6",8) (배포 m1과 동일 입력체계, 좌측절단 max_len=320,
  gen_rescue 미사용 — 최근 history 꼬리가 M4 구분에 핵심이고 fold0 OOF 체계와 정합).
- Loss: class-balanced focal (gamma=2, alpha=역빈도 mean정규화). macro-F1 대상 + 저margin
  하드샘플 집중 — M4 문제구조에 정합.
- LLRD(decay 0.9), fp16, lr 2e-5, batch 48, 5 epoch, 선형 워ミ업 6%. FGM off(m1t3 레시피 정합).
- 하드네거티브(5th 'other') 미채택: 블렌드 대상행의 99.3%가 true∈M4(진단 m4_diag) →
  5th 로짓은 용량비용만 발생, 블렌드 이득 ~0.

출력: work/m4_spec_f0val.npz (rows=va0, y_true(14way), probs4, best_epoch, 이력).
평가는 sim/m4_eval.py(CPU, leak-free)에서.
"""
import os, sys, time, hashlib
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.io_utils import load_train, CLASSES, NUM_CLASSES, set_seed  # noqa: E402
from common import ad_lib  # noqa: E402
from sim import refit_lib as L  # noqa: E402
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,  # noqa: E402
                          get_linear_schedule_with_warmup)
from torch.amp import autocast, GradScaler  # noqa: E402

# ---- 하이퍼파라미터 (env 오버라이드 가능) ----
MODEL = os.environ.get("AD_MODEL", "xlm-roberta-large")
VERSION = "v6"
MHT = 8
MAX_LEN = int(os.environ.get("AD_MAXLEN", "320"))
EPOCHS = int(os.environ.get("AD_EPOCHS", "5"))
LR = float(os.environ.get("AD_LR", "2e-5"))
BATCH = int(os.environ.get("AD_BATCH", "48"))
SEED = int(os.environ.get("AD_SEED", "1234"))
GAMMA = float(os.environ.get("AD_FOCAL_GAMMA", "2.0"))
IDEV_FRAC_BUCKET = 8       # 세션해시 %100 < 8 → internal-dev (~8%)
M4 = [0, 1, 2, 3]
OUT_NPZ = os.path.join(ROOT, "work", "m4_spec_f0val.npz")
HEAD_SEED = 1234
device = "cuda"
assert torch.cuda.is_available(), "GPU 필요 (CUDA_VISIBLE_DEVICES=1)"

print(f"[m4spec] PID={os.getpid()} model={MODEL} v={VERSION} mht={MHT} len={MAX_LEN} "
      f"ep={EPOCHS} lr={LR} b={BATCH} seed={SEED} focal_gamma={GAMMA} "
      f"cuda_dev={os.environ.get('CUDA_VISIBLE_DEVICES','?')}", flush=True)

set_seed(SEED)
samples, y, ids = load_train()
y = np.array(y)
folds, dev, hold = L.load_splits()
tr0, va0 = folds[0]
tr0 = np.asarray(tr0); va0 = np.asarray(va0)

# fold0-train 중 M4 행
m4_mask_tr = np.isin(y[tr0], M4)
m4_rows = tr0[m4_mask_tr]

# 세션그룹 split (deterministic md5 해시) — train/internal-dev
def _bucket(sess):
    return int(hashlib.md5(sess.encode()).hexdigest(), 16) % 100

sess_of = np.array([samples[int(r)]["session"] for r in m4_rows])
buck = np.array([_bucket(s) for s in sess_of])
idev_rows = m4_rows[buck < IDEV_FRAC_BUCKET]
train_rows = m4_rows[buck >= IDEV_FRAC_BUCKET]
# 세션 배타성 assert
assert set(sess_of[buck < IDEV_FRAC_BUCKET]).isdisjoint(set(sess_of[buck >= IDEV_FRAC_BUCKET])), \
    "세션 누수 — internal-dev/train 세션 교차"
print(f"[split] m4_train_rows total={len(m4_rows)} -> train={len(train_rows)} "
      f"idev={len(idev_rows)} (세션배타 OK)", flush=True)

# class-balanced alpha (train_rows M4 빈도 역수, mean 정규화)
cnt = np.bincount(y[train_rows], minlength=NUM_CLASSES)[M4].astype(np.float64)
alpha = len(train_rows) / (4.0 * np.maximum(cnt, 1))
alpha = alpha / alpha.mean()
print(f"[alpha] M4 counts(train)={cnt.tolist()} alpha={np.round(alpha,3).tolist()}", flush=True)
alpha_t = torch.tensor(alpha, dtype=torch.float, device=device)

# 4-way 라벨 (M4는 0..3 항등)
def y4_of(rows):
    return torch.tensor(y[rows].astype(np.int64))   # 이미 0..3

# 직렬화 + 토크나이즈 (train_rows ∪ idev_rows ∪ va0 만 — 전량 불필요)
tok = AutoTokenizer.from_pretrained(MODEL)
tok.truncation_side = "left"
need = np.unique(np.concatenate([train_rows, idev_rows, va0]))
t0 = time.time()
texts = {int(r): ad_lib.serialize(samples[int(r)], VERSION, MHT) for r in need}
enc = tok([texts[int(r)] for r in need], truncation=True, max_length=MAX_LEN, padding=False)
INPUT_IDS = {int(r): enc["input_ids"][k] for k, r in enumerate(need)}
print(f"[tok] {len(need)} rows in {time.time()-t0:.0f}s", flush=True)


def build():
    torch.manual_seed(HEAD_SEED)
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=4, torch_dtype=torch.float32,
        id2label={i: CLASSES[i] for i in M4},
        label2id={CLASSES[i]: i for i in M4}).to(device)


def pad_batch(rows):
    return tok.pad({"input_ids": [INPUT_IDS[int(r)] for r in rows]}, return_tensors="pt")


def make_opt(model):
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


def focal_loss(logits, target):
    logp = F.log_softmax(logits.float(), dim=1)
    logpt = logp.gather(1, target[:, None]).squeeze(1)
    pt = logpt.exp()
    at = alpha_t[target]
    return (-at * (1.0 - pt) ** GAMMA * logpt).mean()


@torch.no_grad()
def infer_probs4(model, rows):
    model.eval()
    order = sorted(range(len(rows)), key=lambda k: len(INPUT_IDS[int(rows[k])]))
    out = np.zeros((len(rows), 4), np.float32)
    bs = 192
    for b in range(0, len(order), bs):
        ks = order[b:b + bs]
        sub = [int(rows[k]) for k in ks]
        e = pad_batch(sub).to(device)
        with autocast("cuda", dtype=torch.float16):
            lg = model(**e).logits.float()
        p = torch.softmax(lg, 1).cpu().numpy()
        for m, k in enumerate(ks):
            out[k] = p[m]
    return out


def macro_f1_4(y_true4, y_pred4):
    return L.fast_macro_f1(y_true4, y_pred4, n_classes=4)


model = build()
opt = make_opt(model)
steps_per_ep = (len(train_rows) + BATCH - 1) // BATCH
tot = steps_per_ep * EPOCHS
sch = get_linear_schedule_with_warmup(opt, int(tot * 0.06), tot)
scaler = GradScaler("cuda", enabled=True)
rng = np.random.RandomState(SEED)

yv = y[va0]                       # 14-way true (평가용)
va_m4 = np.isin(yv, M4)           # true-M4 val 마스크
idev_y4 = y[idev_rows]            # 0..3
va_y4 = yv[va_m4]                 # 0..3 (true-M4 val)

t0 = time.time()
best_idev, best_ep, best_val_probs4 = -1.0, -1, None
idev_hist, val_hist = [], []
for ep in range(EPOCHS):
    model.train()
    perm = rng.permutation(len(train_rows))
    for b in range(0, len(perm), BATCH):
        bi = perm[b:b + BATCH]
        rows = train_rows[bi]
        e = pad_batch(rows).to(device)
        lb = y4_of(rows).to(device)
        opt.zero_grad()
        with autocast("cuda", dtype=torch.float16):
            loss = focal_loss(model(**e).logits, lb)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sch.step()
    # eval
    idev_p4 = infer_probs4(model, idev_rows)
    idev_f1 = macro_f1_4(idev_y4, idev_p4.argmax(1))
    val_all_p4 = infer_probs4(model, va0)
    val_f1 = macro_f1_4(va_y4, val_all_p4[va_m4].argmax(1))   # 로깅용(true-M4 val 4-way)
    idev_hist.append(round(float(idev_f1), 5)); val_hist.append(round(float(val_f1), 5))
    print(f"    epoch {ep+1}: idev4F1={idev_f1:.4f} (SELECT)  val4F1(true-M4)={val_f1:.4f} "
          f"@{(time.time()-t0)/60:.1f}min", flush=True)
    if idev_f1 > best_idev:
        best_idev = idev_f1; best_ep = ep + 1; best_val_probs4 = val_all_p4

print(f"[best] epoch={best_ep} idev4F1={best_idev:.4f} "
      f"val4F1(true-M4)@best={val_hist[best_ep-1]:.4f}", flush=True)

np.savez_compressed(OUT_NPZ, rows=va0, y_true=yv.astype(np.int64),
                    probs4=best_val_probs4.astype(np.float32),
                    best_epoch=best_ep, idev_f1_hist=np.array(idev_hist),
                    val_f1_hist=np.array(val_hist), model=MODEL,
                    train_rows=train_rows, idev_rows=idev_rows)
open(os.path.join(ROOT, "work", "DONE_m4spec"), "w").write(
    f"best_ep={best_ep} idev4F1={best_idev:.4f}\n")
print(f"[saved] {OUT_NPZ}  time={(time.time()-t0)/60:.1f}min", flush=True)
print("=== DONE ===", flush=True)
