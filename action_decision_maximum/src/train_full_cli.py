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
# 서버(컨테이너) 배치: common/이 WORK가 아니라 저장소 루트에 있으면 루트도 경로에 추가
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.isdir(os.path.join(_REPO, "common")):
    sys.path.insert(0, _REPO)

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
SOFT = os.environ.get("AD_SOFT", "")            # teacher 소프트라벨 npz(probs [N,14]) → 증류 모드
SOFT_W = float(os.environ.get("AD_SOFT_W", "0.3"))   # loss = (1-w)*CE(hard) + w*KL(soft, T)
SOFT_T = float(os.environ.get("AD_SOFT_T", "2.0"))
device = "cuda"; assert torch.cuda.is_available()
print(f"[full] {TAG}: {MODEL} v={VERSION} len={MAX_LEN} ep={EPOCHS} lr={LR} b={BATCH} prune={PRUNE}"
      + (f" distill(w={SOFT_W},T={SOFT_T})" if SOFT else ""), flush=True)

set_seed(SEED)
samples, y, ids = load_train()
y = np.array(y)
SIMONLY = os.environ.get("AD_SIMONLY", "0") == "1"
_keep = None
if SIMONLY:
    _keep = [i for i, _id in enumerate(ids) if not str(_id).startswith("sess_au")]
    samples = [samples[i] for i in _keep]
    y = y[_keep]
    ids = [ids[i] for i in _keep]
    print(f"[simonly] au 제외 학습: {len(ids)}행", flush=True)
# history-dropout 증강 (R24, 히든 history분포 shift 직격 정규화). AD_AUG=histdrop, AD_AUG_RATIO(0.5).
AUG = os.environ.get("AD_AUG", "")
if AUG == "histdrop":
    import random as _rnd
    _r = _rnd.Random(SEED)
    ratio = float(os.environ.get("AD_AUG_RATIO", "0.5"))
    n_aug = int(len(samples) * ratio)
    aug_s, aug_y = [], []
    picks = _r.sample(range(len(samples)), n_aug)
    for i in picks:
        s = samples[i]; h = s.get("history") or []
        if not h:
            continue
        u = _r.random()
        if u < 0.20:                       # 전체 history 제거
            nh = []
        elif u < 0.50:                     # 최근 1-2 step만 (turn 2-4개)
            nh = h[-_r.choice([2, 3, 4]):]
        else:                              # 랜덤 prefix 길이 유지 (뒤쪽 절단)
            k = _r.randint(1, max(1, len(h) - 1))
            nh = h[-k:]
        sa = dict(s); sa["history"] = nh
        # field dropout: result_summary 12%, args 10% (액션명 보존)
        if nh:
            nh2 = []
            for t in nh:
                if t.get("role") == "assistant_action":
                    t = dict(t)
                    if _r.random() < 0.12: t["result_summary"] = ""
                    if _r.random() < 0.10: t["args"] = {}
                nh2.append(t)
            sa["history"] = nh2
        aug_s.append(sa); aug_y.append(y[i])
    samples = list(samples) + aug_s
    y = np.concatenate([y, np.array(aug_y, dtype=y.dtype)])
    print(f"[aug:histdrop] +{len(aug_s)}행 증강 → 총 {len(samples)}행 (ratio={ratio})", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL); tok.truncation_side = "left"
texts = [ad_lib.serialize(s, VERSION) for s in samples]
enc_all = tok(texts, truncation=True, max_length=MAX_LEN, padding=False)
INPUT_IDS = enc_all["input_ids"]
cnt = np.bincount(y, minlength=NUM_CLASSES)
cw = len(y) / (NUM_CLASSES * np.maximum(cnt, 1)); cw /= cw.mean()

SOFT_P = None
if SOFT:
    _z = np.load(SOFT, allow_pickle=True)
    _probs = _z["probs"].astype(np.float32)
    _zids = list(_z["ids"])
    if SIMONLY:
        _zids = [_zids[i] for i in _keep]
        _probs = _probs[_keep]
    assert _zids == list(ids), "소프트라벨 id 순서 불일치"
    SOFT_P = torch.tensor(_probs)   # [N,14]

torch.manual_seed(1234)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=NUM_CLASSES,
    id2label={i: c for i, c in enumerate(CLASSES)},
    label2id={c: i for i, c in enumerate(CLASSES)}).to(device)
if os.environ.get("AD_GRADCKPT", "0") == "1":
    # 16GB GPU(T4/P100)에서 large b64 OOM 방지 — 레시피(배치/LR) 보존, ~30% 감속
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    print("[gradckpt] enabled", flush=True)

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
    return pad_batch(b), torch.tensor([y[j] for j in b]), torch.tensor(b)

opt = make_opt(model)
dl = torch.utils.data.DataLoader(DS(), batch_size=BATCH, shuffle=True, collate_fn=coll,
                                 num_workers=4, pin_memory=True, persistent_workers=True)
tot = len(dl) * EPOCHS
sch = get_linear_schedule_with_warmup(opt, int(tot * 0.06), tot)
scaler = GradScaler("cuda")
lossfn = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float, device=device))
SWA_K = int(os.environ.get("AD_SWA_K", "0"))   # 마지막 K에폭 가중치 평균(SWA-lite, codex R10)
swa_sum, swa_n = None, 0
t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    for enc, lb, bi in dl:
        enc = {k: v.to(device) for k, v in enc.items()}; lb = lb.to(device); opt.zero_grad()
        with autocast("cuda", dtype=torch.float16):
            logits = model(**enc).logits
            loss = lossfn(logits, lb)
            if SOFT_P is not None:
                tp = SOFT_P[bi].to(device)                                   # teacher probs
                logq = torch.log_softmax(logits.float() / SOFT_T, dim=1)
                tp_t = torch.softmax(torch.log(tp + 1e-9) / SOFT_T, dim=1)   # T-스케일 teacher
                kl = torch.nn.functional.kl_div(logq, tp_t, reduction="batchmean") * (SOFT_T ** 2)
                loss = (1 - SOFT_W) * loss + SOFT_W * kl
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
    print(f"  epoch {ep+1} done @{(time.time()-t0)/60:.1f}min", flush=True)
    if SWA_K and ep >= EPOCHS - SWA_K:
        sd = {k: v.detach().float().cpu() for k, v in model.state_dict().items()
              if v.dtype.is_floating_point}
        if swa_sum is None:
            swa_sum = sd
        else:
            for k in swa_sum: swa_sum[k] += sd[k]
        swa_n += 1
        print(f"  [swa] snapshot ep{ep+1} ({swa_n}/{SWA_K})", flush=True)
def save_member(tag):
    # 저장 → (옵션) 프루닝 → zip. 반환: zip 경로
    raw_dir = os.path.join(WORK, f"raw_{tag}")
    model.half().save_pretrained(raw_dir, safe_serialization=True)
    tok.save_pretrained(raw_dir)
    out_dir = os.path.join(WORK, f"member_{tag}")
    if PRUNE:
        K, _ = prune_model_dir(raw_dir, out_dir, tok, texts, max_len=MAX_LEN)
        print(f"[prune] vocab -> {K}", flush=True)
    else:
        shutil.copytree(raw_dir, out_dir, dirs_exist_ok=True)
    mb = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(out_dir) for f in fs) / 1e6
    print(f"[member] {out_dir} size={mb:.0f}MB", flush=True)
    zp = os.path.join(WORK, f"member_{tag}.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for r, _, fs in os.walk(out_dir):
            for f in fs:
                z.write(os.path.join(r, f), os.path.relpath(os.path.join(r, f), out_dir))
    print(f"[zip] {zp} {os.path.getsize(zp)/1e6:.0f}MB", flush=True)
    return mb

if swa_sum is not None and swa_n > 1:
    save_member(TAG + "raw")   # SWA 적용 전 raw-final 멤버도 저장 (SWA 해악 분리실험, R13)
    fin = model.state_dict()
    for k, v in swa_sum.items():
        fin[k] = (v / swa_n).to(fin[k].dtype)
    model.load_state_dict(fin)
    print(f"[swa] {swa_n}개 에폭 평균 적용", flush=True)

mb = save_member(TAG)
open(os.path.join(WORK, f"DONE_{TAG}"), "w").write(f"size={mb:.0f}MB time={(time.time()-t0)/60:.1f}min")
print("=== DONE ===", flush=True)
