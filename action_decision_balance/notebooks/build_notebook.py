"""Track A(균형) Colab 학습 노트북 생성기 → train_colab.ipynb.

자체완결: 사용자는 open.zip + ad_common.zip 업로드 후 '모두 실행'만 하면
5-fold 학습 → soup → 후처리 → T4 30k 타이밍 → submit_balance.zip 생성/다운로드.
"""
import json, os

CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

md("""# Dacon 236694 — Track A (균형) 학습 노트북
**실행 방법**: 런타임 → 런타임 유형 변경 → **T4 GPU** 선택 → **런타임 → 모두 실행**.
업로드 요청이 뜨면 `open.zip` 과 `ad_common.zip` **두 개를 함께** 선택하세요.
끝나면 `submit_balance.zip` 이 자동 다운로드되고, CV/추론시간 로그가 출력됩니다.
그 로그(또는 zip)를 Claude Code 에 전달하면 됩니다.""")

code(r"""# 1) 환경 (서버와 동일 버전 고정) + GPU 확인
!nvidia-smi -L
!pip -q install "transformers==4.46.3" "accelerate==1.9.0" "sentencepiece==0.1.99" 2>/dev/null
import torch; print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
assert torch.cuda.is_available(), "런타임을 T4 GPU로 바꾸세요"
""")

code(r"""# 2) 데이터/코드 업로드 (open.zip + ad_common.zip 함께 선택)
import os, zipfile
if not (os.path.exists("data/train.jsonl") and os.path.exists("common/ad_lib.py")):
    from google.colab import files
    up = files.upload()   # open.zip, ad_common.zip 선택
    for z in ["open.zip", "ad_common.zip"]:
        with zipfile.ZipFile(z) as f: f.extractall(".")
assert os.path.exists("data/train.jsonl"), "open.zip 안에 data/train.jsonl 필요"
assert os.path.exists("common/ad_lib.py") and os.path.exists("splits/splits.npz"), "ad_common.zip 필요"
print("데이터/코드 준비 완료")
""")

code(r"""# 3) 설정 (균형 트랙: 짧은 max_len → 빠른 추론)
CFG = dict(MODEL_NAME="xlm-roberta-base", VERSION="v3", MAX_LEN=192,
           LR=2e-5, EPOCHS=3, BATCH=32, EVAL_BATCH=128, WARMUP=0.06, WD=0.01,
           SEED=42, HEAD_SEED=1234, USE_CLASS_WEIGHT=True, TRACK="balance")
OUTDIR = "submit_balance"; MODELDIR = OUTDIR + "/model"
print(CFG)
""")

code(r"""# 4) 로드 + 직렬화 + CV 분할(프로즌) + 클래스가중치
import sys; sys.path.insert(0, ".")
import numpy as np, time, json, os
from common.io_utils import load_train, CLASSES, NUM_CLASSES, set_seed
from common.cv import make_splits
from common.metrics import macro_f1, print_report
from common import ad_lib, postproc, soup
set_seed(CFG["SEED"])
samples, y, ids = load_train()
y = np.array(y); groups = np.array([s["session"] for s in samples]); gen = np.array([s["gen"] for s in samples])
sp = make_splits(ids, y, groups)               # splits.npz 로드(로컬과 동일)
dev_idx, hold_idx, folds = sp["dev_idx"], sp["holdout_idx"], sp["folds"]
texts = [ad_lib.serialize(s, CFG["VERSION"]) for s in samples]
cnt = np.bincount(y[dev_idx], minlength=NUM_CLASSES)
cw = len(dev_idx) / (NUM_CLASSES * np.maximum(cnt, 1)); cw = cw / cw.mean()
print("dev", len(dev_idx), "holdout", len(hold_idx), "| classweight[min,max]=", round(cw.min(),2), round(cw.max(),2))
""")

code(r"""# 5) 토크나이저/모델/학습 함수
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.amp import autocast, GradScaler
device = "cuda"
tok = AutoTokenizer.from_pretrained(CFG["MODEL_NAME"])

class DS(torch.utils.data.Dataset):
    def __init__(self, idx): self.idx = idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = int(self.idx[i]); return texts[j], int(y[j])

def collate(b):
    enc = tok([x[0] for x in b], padding=True, truncation=True, max_length=CFG["MAX_LEN"], return_tensors="pt")
    return enc, torch.tensor([x[1] for x in b])

def build_model():
    torch.manual_seed(CFG["HEAD_SEED"])   # 헤드 init 고정 → soup 정합
    return AutoModelForSequenceClassification.from_pretrained(
        CFG["MODEL_NAME"], num_labels=NUM_CLASSES,
        id2label={i: c for i, c in enumerate(CLASSES)},
        label2id={c: i for i, c in enumerate(CLASSES)}).to(device)

def infer_logits(model, idx):
    model.eval(); bs = CFG["EVAL_BATCH"]
    order = sorted(range(len(idx)), key=lambda k: len(texts[int(idx[k])]))
    out = np.zeros((len(idx), NUM_CLASSES), np.float32)
    with torch.no_grad():
        for b in range(0, len(order), bs):
            ks = order[b:b+bs]; sub = [int(idx[k]) for k in ks]
            enc = tok([texts[j] for j in sub], padding=True, truncation=True,
                      max_length=CFG["MAX_LEN"], return_tensors="pt").to(device)
            with autocast("cuda", dtype=torch.float16):
                lg = model(**enc).logits.float().cpu().numpy()
            for m, k in enumerate(ks): out[k] = lg[m]
    return out

def train_fold(tr, va):
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["LR"], weight_decay=CFG["WD"])
    dl = torch.utils.data.DataLoader(DS(tr), batch_size=CFG["BATCH"], shuffle=True,
                                     collate_fn=collate, num_workers=2, drop_last=False)
    total = len(dl)*CFG["EPOCHS"]; sch = get_linear_schedule_with_warmup(opt, int(total*CFG["WARMUP"]), total)
    scaler = GradScaler("cuda")
    wt = torch.tensor(cw, dtype=torch.float, device=device) if CFG["USE_CLASS_WEIGHT"] else None
    lossfn = torch.nn.CrossEntropyLoss(weight=wt)
    best, best_state = -1, None
    for ep in range(CFG["EPOCHS"]):
        model.train()
        for enc, lb in dl:
            enc = {k: v.to(device) for k, v in enc.items()}; lb = lb.to(device)
            opt.zero_grad()
            with autocast("cuda", dtype=torch.float16):
                loss = lossfn(model(**enc).logits, lb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        vlog = infer_logits(model, va); mf1, _ = macro_f1(y[va], vlog.argmax(1))
        print(f"  epoch {ep+1}: val macroF1={mf1:.4f}")
        if mf1 > best:
            best = mf1; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_state, best
""")

code(r"""# 6) 5-fold 학습 (OOF/holdout logits + fold 가중치 수집)
oof = np.zeros((len(samples), NUM_CLASSES), np.float32)
fold_states, fold_scores, hold_logits = [], [], []
t0 = time.time()
for fi, (tr, va) in enumerate(folds):
    print(f"=== fold {fi} (tr={len(tr)} va={len(va)}) ===")
    st, sc = train_fold(tr, va); fold_states.append(st); fold_scores.append(sc)
    m = build_model(); m.load_state_dict(st)
    oof[va] = infer_logits(m, va); hold_logits.append(infer_logits(m, hold_idx))
    del m; torch.cuda.empty_cache()
print(f"학습 시간 {(time.time()-t0)/60:.1f} min | fold scores {[round(s,4) for s in fold_scores]}")
""")

code(r"""# 7) pooled-OOF 평가 (주지표) + sim-only + per-class
oof_pred = oof[dev_idx].argmax(1)
oof_mf1, _ = macro_f1(y[dev_idx], oof_pred)
sim_dev = dev_idx[gen[dev_idx] == "sim"]
print("pooled-OOF macroF1 =", round(oof_mf1, 4), "| acc =", round((oof_pred==y[dev_idx]).mean(),4))
print("sim-only OOF macroF1 =", round(macro_f1(y[sim_dev], oof[sim_dev].argmax(1))[0], 4))
print_report(y[dev_idx], oof_pred, "OOF per-class")
""")

code(r"""# 8) Greedy Model Soup (홀드아웃 macro-F1 기준)
def eval_state(state):
    m = build_model(); m.load_state_dict(state)
    s = macro_f1(y[hold_idx], infer_logits(m, hold_idx)[:, :].argmax(1))[0]
    del m; torch.cuda.empty_cache(); return s
souped, ingr, soup_hold = soup.greedy_soup(fold_states, fold_scores, eval_state)
uni = soup.uniform_soup(fold_states); uni_hold = eval_state(uni)
print(f"greedy soup: ingredients={ingr} holdout={soup_hold:.4f} | uniform holdout={uni_hold:.4f}")
best_state = souped if soup_hold >= uni_hold else uni
""")

code(r"""# 9) 후처리(per-class bias, pooled-OOF에서 적합) + soup에서 홀드아웃 검증
bias, fit = postproc.fit_bias(oof[dev_idx], y[dev_idx]); print("OOF bias-fit macroF1 =", round(fit,4))
sm = build_model(); sm.load_state_dict(best_state)
hlog = infer_logits(sm, hold_idx)
h_nob = macro_f1(y[hold_idx], hlog.argmax(1))[0]
h_b = macro_f1(y[hold_idx], (ad_lib._to_logprobs_np(hlog, np) + bias).argmax(1))[0]
USE_BIAS = bool(h_b >= h_nob - 1e-4)
print(f"holdout soup: no-bias={h_nob:.4f} | +bias={h_b:.4f} -> use_bias={USE_BIAS}")
""")

code(r"""# 10) model/ 아티팩트 저장 (fp16 safetensors + tokenizer + ad_lib + postproc + meta)
import shutil, os, json
os.makedirs(MODELDIR, exist_ok=True)
sm.half()                                   # fp16 → <1GB, 서버 추론과 동일 dtype
sm.save_pretrained(MODELDIR, safe_serialization=True)
tok.save_pretrained(MODELDIR)
shutil.copy("common/ad_lib.py", os.path.join(MODELDIR, "ad_lib.py"))
postproc.save(os.path.join(MODELDIR, "postproc.json"),
              bias if USE_BIAS else np.zeros(NUM_CLASSES),
              {"use_bias": USE_BIAS, "oof_macf1": float(oof_mf1),
               "holdout_macf1": float(max(h_b, h_nob))})
json.dump({"version": CFG["VERSION"], "max_len": CFG["MAX_LEN"], "batch_size": 128},
          open(os.path.join(MODELDIR, "run_meta.json"), "w"))
# 불필요 파일 없음 확인
print("model/ 파일:", sorted(os.listdir(MODELDIR)))
print("model/ 용량(MB):", round(sum(os.path.getsize(os.path.join(MODELDIR,f)) for f in os.listdir(MODELDIR))/1e6,1))
""")

import base64 as _b64
_CANON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "submission", "script.py")
_SCRIPT_B64 = _b64.b64encode(open(_CANON, "rb").read()).decode()
code('# 11) script.py(정본, base64 임베드) + requirements.txt 작성\n'
     'import base64, os\n'
     f'open(OUTDIR + "/script.py", "wb").write(base64.b64decode("{_SCRIPT_B64}"))\n'
     'open(OUTDIR + "/requirements.txt", "w").write("")   # 서버 사전설치로 충분 → 설치 0분\n'
     'print("script.py(정본), requirements.txt 작성 완료")')

code(r"""# 12) T4 30k 추론 타이밍 (실제 서버 경로: 디스크에서 fp16 재로드)
bench_idx = np.random.RandomState(0).choice(dev_idx, size=min(30000, len(dev_idx)), replace=False)
bench = [samples[int(i)] for i in bench_idx]
t = time.time()
_ = ad_lib.predict(MODELDIR, bench, version=CFG["VERSION"], max_len=CFG["MAX_LEN"],
                   batch_size=128, postproc_path=MODELDIR + "/postproc.json")
dt = time.time() - t
print(f"30k 추론 {dt:.1f}s (제한 600s) -> {'OK ✅' if dt < 600 else 'TOO SLOW ❌'}  | 여유 {600-dt:.0f}s")
""")

code(r"""# 13) 오프라인 script.py 스모크 테스트 (서버와 동일 구조) + zip 패키징 + 다운로드
import subprocess, zipfile, os
os.makedirs("/content/simrun/data", exist_ok=True)
for p in ["model", "script.py", "requirements.txt"]:
    subprocess.run(["cp", "-r", f"{OUTDIR}/{p}", "/content/simrun/"], check=True)
for p in ["test.jsonl", "sample_submission.csv"]:
    subprocess.run(["cp", f"data/{p}", "/content/simrun/data/"], check=True)
r = subprocess.run(["python", "script.py"], cwd="/content/simrun", capture_output=True, text=True)
print("STDOUT:", r.stdout.strip(), "\nSTDERR:", r.stderr.strip()[-500:])
print("--- submission head ---")
print(open("/content/simrun/output/submission.csv").read())

def zipdir(src, zf):
    for root, _, fs in os.walk(src):
        for fn in fs:
            fp = os.path.join(root, fn); zf.write(fp, os.path.relpath(fp, src))
with zipfile.ZipFile("submit_balance.zip", "w", zipfile.ZIP_DEFLATED) as z:
    zipdir(OUTDIR, z)
mb = os.path.getsize("submit_balance.zip")/1e6
print(f"\nsubmit_balance.zip = {mb:.0f} MB (제한 1000) -> {'OK ✅' if mb < 1000 else 'TOO BIG ❌'}")
tops = sorted(set(n.split('/')[0] for n in zipfile.ZipFile('submit_balance.zip').namelist()))
print("zip 최상위:", tops, "->", "OK ✅" if set(tops) <= {'model','script.py','requirements.txt'} else "구조오류 ❌")
from google.colab import files; files.download("submit_balance.zip")
""")

md("""## 완료 — Claude Code 에 보낼 것
아래 로그를 복사해 전달하세요:
- 셀 7 **pooled-OOF macroF1 / sim-only / per-class**
- 셀 8 **soup** 결과, 셀 9 **holdout no-bias vs +bias**
- 셀 12 **30k 추론 시간**, 셀 13 **zip 용량/최상위 구조 + submission head**
그리고 `submit_balance.zip` 파일. (LB 제출은 Dacon에 직접 업로드)""")

nb = {"cells": [{"cell_type": t,
                 "metadata": {},
                 "source": s.splitlines(keepends=True),
                 **({"outputs": [], "execution_count": None} if t == "code" else {})}
                for t, s in CELLS],
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "accelerator": "GPU", "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_colab.ipynb")
json.dump(nb, open(out, "w"), ensure_ascii=False, indent=1)
print("wrote", out, "cells:", len(CELLS))
