"""Track B(정확도 최대) Colab 노트북 생성기 → train_colab.ipynb.

멀티시드(옵션 멀티아키텍처) 5-fold 앙상블 → OOF soft-label 지식증류 → 단일 xlm-r student.
서버 추론은 student 1개(fp16, <1GB, 단일 속도)로 앙상블급 정확도.
"""
import os, json, base64

CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

md("""# Dacon 236694 — Track B (정확도 최대) 학습 노트북
멀티시드 5-fold 앙상블 → **OOF soft-label 지식증류 → 단일 student**. 서버엔 student 1개만 올라가
앙상블급 정확도 + 단일 모델 속도(추론속도 10% 보너스 유지).
**먼저 Track A를 돌려 결과를 확인한 뒤 실행 권장.** 실행 시간이 길어(멀티시드) **Kaggle(9h) 또는 Colab Pro** 권장.
실행: 런타임 → T4 GPU → 모두 실행 → (`open.zip`+`ad_common.zip` 업로드) → `submit_maximum.zip` 다운로드.""")

code(r"""# 1) 환경
!nvidia-smi -L
!pip -q install "transformers==4.46.3" "accelerate==1.9.0" "sentencepiece==0.1.99" 2>/dev/null
import torch; print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
assert torch.cuda.is_available(), "런타임을 T4 GPU로"
""")

code(r"""# 2) 업로드/추출 (open.zip + ad_common.zip)
import os, zipfile
if not (os.path.exists("data/train.jsonl") and os.path.exists("common/ad_lib.py")):
    from google.colab import files
    up = files.upload()
    for z in ["open.zip", "ad_common.zip"]:
        with zipfile.ZipFile(z) as f: f.extractall(".")
assert os.path.exists("data/train.jsonl") and os.path.exists("splits/splits.npz")
print("준비 완료")
""")

code(r"""# 3) 설정 (최대 트랙: v4, max_len 256, 멀티시드 앙상블 → 증류)
CFG = dict(
    VERSION="v4", MAX_LEN=256, EVAL_BATCH=96, BATCH=24,
    LR=2e-5, EPOCHS=3, WARMUP=0.06, WD=0.01, HEAD_SEED=1234, USE_CLASS_WEIGHT=True,
    # 교사 앙상블: (모델명, fp16학습여부). mdeberta는 T4 fp16 불안정 → fp32(False) 권장.
    TEACHERS=[("xlm-roberta-base", True, 1234), ("xlm-roberta-base", True, 2024)],
    # 더 강하게: ("microsoft/mdeberta-v3-base", False, 1234) 를 추가(시간 2배↑)
    STUDENT="xlm-roberta-base", DISTILL_ALPHA=0.7, DISTILL_EPOCHS=3, TRACK="maximum",
)
OUTDIR="submit_maximum"; MODELDIR=OUTDIR+"/model"; print(CFG)
""")

code(r"""# 4) 로드 + 직렬화 + CV + 클래스가중치
import sys; sys.path.insert(0, ".")
import numpy as np, time, json, os
from common.io_utils import load_train, CLASSES, NUM_CLASSES, set_seed
from common.cv import make_splits
from common.metrics import macro_f1, print_report
from common import ad_lib, postproc, soup
samples, y, ids = load_train()
y = np.array(y); groups = np.array([s["session"] for s in samples]); gen = np.array([s["gen"] for s in samples])
sp = make_splits(ids, y, groups); dev_idx, hold_idx, folds = sp["dev_idx"], sp["holdout_idx"], sp["folds"]
texts = [ad_lib.serialize(s, CFG["VERSION"]) for s in samples]
cnt = np.bincount(y[dev_idx], minlength=NUM_CLASSES); cw = len(dev_idx)/(NUM_CLASSES*np.maximum(cnt,1)); cw/=cw.mean()
print("dev", len(dev_idx), "holdout", len(hold_idx))
""")

code(r"""# 5) 학습/추론 함수 (모델명·시드·fp16 파라미터화)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.amp import autocast, GradScaler
device = "cuda"
_TOK = {}
def get_tok(name):
    if name not in _TOK: _TOK[name] = AutoTokenizer.from_pretrained(name)
    return _TOK[name]

def build(name):
    torch.manual_seed(CFG["HEAD_SEED"])
    return AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=NUM_CLASSES,
        id2label={i:c for i,c in enumerate(CLASSES)}, label2id={c:i for i,c in enumerate(CLASSES)}).to(device)

def infer_probs(model, tok, idx, fp16=True):
    model.eval(); bs=CFG["EVAL_BATCH"]
    order=sorted(range(len(idx)), key=lambda k: len(texts[int(idx[k])]))
    out=np.zeros((len(idx), NUM_CLASSES), np.float32)
    with torch.no_grad():
        for b in range(0,len(order),bs):
            ks=order[b:b+bs]; sub=[int(idx[k]) for k in ks]
            enc=tok([texts[j] for j in sub], padding=True, truncation=True, max_length=CFG["MAX_LEN"], return_tensors="pt").to(device)
            if fp16:
                with autocast("cuda", dtype=torch.float16): lg=model(**enc).logits.float()
            else:
                lg=model(**enc).logits.float()
            p=torch.softmax(lg,1).cpu().numpy()
            for m,k in enumerate(ks): out[k]=p[m]
    return out

def train_one(name, seed, tr, va, fp16=True):
    set_seed(seed)
    tok=get_tok(name); model=build(name)
    opt=torch.optim.AdamW(model.parameters(), lr=CFG["LR"], weight_decay=CFG["WD"])
    class DS(torch.utils.data.Dataset):
        def __len__(s): return len(tr)
        def __getitem__(s,i): j=int(tr[i]); return texts[j], int(y[j])
    def coll(b):
        enc=tok([x[0] for x in b], padding=True, truncation=True, max_length=CFG["MAX_LEN"], return_tensors="pt")
        return enc, torch.tensor([x[1] for x in b])
    dl=torch.utils.data.DataLoader(DS(), batch_size=CFG["BATCH"], shuffle=True, collate_fn=coll, num_workers=2)
    tot=len(dl)*CFG["EPOCHS"]; sch=get_linear_schedule_with_warmup(opt, int(tot*CFG["WARMUP"]), tot)
    scaler=GradScaler("cuda", enabled=fp16)
    wt=torch.tensor(cw, dtype=torch.float, device=device) if CFG["USE_CLASS_WEIGHT"] else None
    lossfn=torch.nn.CrossEntropyLoss(weight=wt)
    best=-1; best_probs_va=None; best_probs_hold=None
    for ep in range(CFG["EPOCHS"]):
        model.train()
        for enc,lb in dl:
            enc={k:v.to(device) for k,v in enc.items()}; lb=lb.to(device); opt.zero_grad()
            if fp16:
                with autocast("cuda", dtype=torch.float16): loss=lossfn(model(**enc).logits, lb)
            else:
                loss=lossfn(model(**enc).logits, lb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        pv=infer_probs(model, tok, va, fp16); mf1,_=macro_f1(y[va], pv.argmax(1))
        if mf1>best: best=mf1; best_probs_va=pv; best_probs_hold=infer_probs(model, tok, hold_idx, fp16)
    print(f"    [{name} seed{seed}] best val macroF1={best:.4f}")
    del model; torch.cuda.empty_cache()
    return best_probs_va, best_probs_hold, best
""")

code(r"""# 6) 교사 앙상블 학습 → OOF/holdout 확률 누적 (가중치는 버리고 확률만 보관)
oof_sum=np.zeros((len(samples), NUM_CLASSES), np.float32); oof_cnt=np.zeros(len(samples), np.float32)
hold_sum=np.zeros((len(hold_idx), NUM_CLASSES), np.float32); hold_n=0
t0=time.time()
for (name, fp16, seed) in CFG["TEACHERS"]:
    print(f"=== teacher {name} seed{seed} (fp16={fp16}) ===")
    for fi,(tr,va) in enumerate(folds):
        pv, ph, sc = train_one(name, seed, tr, va, fp16)
        oof_sum[va]+=pv; oof_cnt[va]+=1
        hold_sum+=ph; hold_n+=1
teacher_oof=np.zeros_like(oof_sum); m=oof_cnt>0
teacher_oof[m]=oof_sum[m]/oof_cnt[m,None]
teacher_hold=hold_sum/max(hold_n,1)
print(f"교사 학습 시간 {(time.time()-t0)/60:.1f} min")
""")

code(r"""# 7) 교사 앙상블 pooled-OOF 평가
oof_pred=teacher_oof[dev_idx].argmax(1); ens_mf1,_=macro_f1(y[dev_idx], oof_pred)
sim_dev=dev_idx[gen[dev_idx]=="sim"]
print("ENSEMBLE pooled-OOF macroF1 =", round(ens_mf1,4))
print("ENSEMBLE sim-only OOF =", round(macro_f1(y[sim_dev], teacher_oof[sim_dev].argmax(1))[0],4))
print("ENSEMBLE holdout =", round(macro_f1(y[hold_idx], teacher_hold.argmax(1))[0],4))
print_report(y[dev_idx], oof_pred, "ENSEMBLE OOF")
""")

code(r"""# 8) 지식증류 → 단일 student (soft OOF + hard label, 전체 dev 학습)
tok=get_tok(CFG["STUDENT"])
soft=torch.tensor(teacher_oof[dev_idx], dtype=torch.float)   # 교사 soft target (OOF)
hard=torch.tensor(y[dev_idx], dtype=torch.long)
class KDS(torch.utils.data.Dataset):
    def __len__(s): return len(dev_idx)
    def __getitem__(s,i): j=int(dev_idx[i]); return texts[j], soft[i], int(y[j])
def kcoll(b):
    enc=tok([x[0] for x in b], padding=True, truncation=True, max_length=CFG["MAX_LEN"], return_tensors="pt")
    return enc, torch.stack([x[1] for x in b]), torch.tensor([x[2] for x in b])
dl=torch.utils.data.DataLoader(KDS(), batch_size=CFG["BATCH"], shuffle=True, collate_fn=kcoll, num_workers=2)
student=build(CFG["STUDENT"])
opt=torch.optim.AdamW(student.parameters(), lr=CFG["LR"], weight_decay=CFG["WD"])
tot=len(dl)*CFG["DISTILL_EPOCHS"]; sch=get_linear_schedule_with_warmup(opt, int(tot*CFG["WARMUP"]), tot)
scaler=GradScaler("cuda"); wt=torch.tensor(cw,dtype=torch.float,device=device)
ce=torch.nn.CrossEntropyLoss(weight=wt); a=CFG["DISTILL_ALPHA"]
best=-1; best_state=None
for ep in range(CFG["DISTILL_EPOCHS"]):
    student.train()
    for enc, sp_t, hy in dl:
        enc={k:v.to(device) for k,v in enc.items()}; sp_t=sp_t.to(device); hy=hy.to(device); opt.zero_grad()
        with autocast("cuda", dtype=torch.float16):
            z=student(**enc).logits
            logp=torch.log_softmax(z,1)
            soft_ce=-(sp_t*logp).sum(1).mean()
            loss=a*soft_ce + (1-a)*ce(z,hy)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
    ph=infer_probs(student, tok, hold_idx); mf1,_=macro_f1(y[hold_idx], ph.argmax(1))
    print(f"  distill epoch {ep+1}: holdout macroF1={mf1:.4f}")
    if mf1>best: best=mf1; best_state={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
student.load_state_dict(best_state); print("student best holdout =", round(best,4))
""")

code(r"""# 9) 후처리(교사 OOF에서 bias 적합) + student 홀드아웃 검증
bias, fit = postproc.fit_bias(np.log(teacher_oof[dev_idx]+1e-9), y[dev_idx]); print("bias-fit(on teacher OOF) =", round(fit,4))
hlog = infer_probs(student, tok, hold_idx)
import numpy as np
h_nob=macro_f1(y[hold_idx], hlog.argmax(1))[0]
h_b=macro_f1(y[hold_idx], (np.log(hlog+1e-9)+bias).argmax(1))[0]
USE_BIAS=bool(h_b>=h_nob-1e-4)
print(f"student holdout: no-bias={h_nob:.4f} +bias={h_b:.4f} use_bias={USE_BIAS}")
""")

code(r"""# 10) student 저장 (fp16 safetensors + tokenizer + ad_lib + postproc + meta)
import shutil, os, json
os.makedirs(MODELDIR, exist_ok=True)
student.half().save_pretrained(MODELDIR, safe_serialization=True)
tok.save_pretrained(MODELDIR)
shutil.copy("common/ad_lib.py", os.path.join(MODELDIR,"ad_lib.py"))
postproc.save(os.path.join(MODELDIR,"postproc.json"), bias if USE_BIAS else np.zeros(NUM_CLASSES),
              {"use_bias":USE_BIAS, "ensemble_oof":float(ens_mf1), "student_holdout":float(max(h_b,h_nob))})
json.dump({"version":CFG["VERSION"], "max_len":CFG["MAX_LEN"], "batch_size":128},
          open(os.path.join(MODELDIR,"run_meta.json"),"w"))
print("model/:", sorted(os.listdir(MODELDIR)), "| MB:",
      round(sum(os.path.getsize(os.path.join(MODELDIR,f)) for f in os.listdir(MODELDIR))/1e6,1))
""")

# --- cell 11: 정본 script.py b64 임베드 ---
_CANON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "submission", "script.py")
_B64 = base64.b64encode(open(_CANON, "rb").read()).decode()
code('# 11) script.py(정본) + requirements.txt\n'
     'import base64, os\n'
     f'open(OUTDIR+"/script.py","wb").write(base64.b64decode("{_B64}"))\n'
     'open(OUTDIR+"/requirements.txt","w").write("")\n'
     'print("script.py, requirements.txt 작성 완료")')

code(r"""# 12) T4 30k 추론 타이밍 (student, 디스크 재로드)
bidx=np.random.RandomState(0).choice(dev_idx, size=min(30000,len(dev_idx)), replace=False)
bench=[samples[int(i)] for i in bidx]; t=time.time()
_=ad_lib.predict(MODELDIR, bench, version=CFG["VERSION"], max_len=CFG["MAX_LEN"], batch_size=128, postproc_path=MODELDIR+"/postproc.json")
dt=time.time()-t; print(f"30k 추론 {dt:.1f}s (제한 600s) -> {'OK ✅' if dt<600 else 'TOO SLOW ❌'} 여유 {600-dt:.0f}s")
""")

code(r"""# 13) 오프라인 스모크 + zip 패키징 + 다운로드
import subprocess, zipfile, os
os.makedirs("/content/simrunB/data", exist_ok=True)
for p in ["model","script.py","requirements.txt"]: subprocess.run(["cp","-r",f"{OUTDIR}/{p}","/content/simrunB/"], check=True)
for p in ["test.jsonl","sample_submission.csv"]: subprocess.run(["cp",f"data/{p}","/content/simrunB/data/"], check=True)
r=subprocess.run(["python","script.py"], cwd="/content/simrunB", capture_output=True, text=True)
print("STDOUT:", r.stdout.strip(), "\nSTDERR:", r.stderr.strip()[-500:])
print(open("/content/simrunB/output/submission.csv").read())
def zipdir(src,zf):
    for root,_,fs in os.walk(src):
        for fn in fs: zf.write(os.path.join(root,fn), os.path.relpath(os.path.join(root,fn),src))
with zipfile.ZipFile("submit_maximum.zip","w",zipfile.ZIP_DEFLATED) as z: zipdir(OUTDIR,z)
mb=os.path.getsize("submit_maximum.zip")/1e6
tops=sorted(set(n.split('/')[0] for n in zipfile.ZipFile('submit_maximum.zip').namelist()))
print(f"submit_maximum.zip={mb:.0f}MB -> {'OK ✅' if mb<1000 else 'TOO BIG ❌'} | 최상위 {tops} -> {'OK ✅' if set(tops)<={'model','script.py','requirements.txt'} else '오류 ❌'}")
from google.colab import files; files.download("submit_maximum.zip")
""")

md("""## 완료 — 보낼 것
- 셀 7 **ENSEMBLE pooled-OOF / sim-only / holdout / per-class**
- 셀 8 **student holdout**, 셀 9 **no-bias vs +bias**
- 셀 12 **30k 시간**, 셀 13 **zip 용량/구조**
그리고 `submit_maximum.zip`.""")

nb = {"cells": [{"cell_type": t, "metadata": {},
                 "source": s.splitlines(keepends=True),
                 **({"outputs": [], "execution_count": None} if t == "code" else {})}
                for t, s in CELLS],
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "accelerator": "GPU", "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_colab.ipynb")
json.dump(nb, open(out, "w"), ensure_ascii=False, indent=1)
print("wrote", out, "cells:", len(CELLS))
