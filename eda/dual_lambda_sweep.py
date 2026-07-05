"""듀얼 bias λ 스윕 — 배포 모델(largev6-8ep FULL) holdout 확률 1회 추론 후 numpy 스윕 (R14 Q2).

bias_au' = global + λ*(bias_au - global), bias_sim' 동일. λ ∈ {0, 0.25, 0.5, 0.75, 1.0}.
λ=0 = 글로벌 bias(현행). 보고: holdout 전체/sim/au macro-F1.
usage: python3 eda/dual_lambda_sweep.py <member.zip::ver>
"""
from __future__ import annotations
import os, sys, glob, json, tempfile, zipfile, shutil
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
from common.io_utils import load_train
from common.cv import make_splits
from common.postproc import fit_bias, to_logprobs
from common.metrics import macro_f1
from common import ad_lib

spec = sys.argv[1]
zp, _, ver = spec.partition("::")
ver = ver or "v6"

samples, y, ids = load_train(); y = np.array(y); ids = np.array([str(i) for i in ids])
sp = make_splits(ids, y, np.array([s["session"] for s in samples]))
folds = sp["folds"]; hidx = sp["holdout_idx"]
cov = np.concatenate([f[1] for f in folds])
au = np.char.startswith(ids, "sess_au")

# 1) teacher OOF로 bias 3종 적합 (배포 패키저와 동일 소스)
oof = np.zeros((len(y), 14), np.float32); cs = set()
for p in sorted(glob.glob(os.path.join(R, "action_decision_maximum/experiments/teacher_largev6[AB]_a*.npz"))):
    z = np.load(p, allow_pickle=True)
    for f in range(int(z["fold_lo"]), int(z["fold_hi"])):
        if f in cs: continue
        oof[folds[f][1]] = z["oof"][folds[f][1]]; cs.add(f)
lp = np.log(oof + 1e-9)
b_all, _ = fit_bias(lp[cov], y[cov])
b_sim, _ = fit_bias(lp[cov[~au[cov]]], y[cov[~au[cov]]])
b_au, _ = fit_bias(lp[cov[au[cov]]], y[cov[au[cov]]])
b_all, b_sim, b_au = map(np.asarray, (b_all, b_sim, b_au))
print("[sweep] bias 3종 적합 완료", flush=True)

# 2) 배포 모델 holdout 확률 1회 추론
d = tempfile.mkdtemp(prefix="dls_")
with zipfile.ZipFile(zp) as zf:
    zf.extractall(d)
hs = [samples[i] for i in hidx]
logits = ad_lib.predict_logits(d, hs, version=ver, max_len=320, batch_size=256, return_probs=True)
shutil.rmtree(d)
scores = np.log(np.asarray(logits, np.float64) + 1e-9)
yh = y[hidx]; auh = au[hidx]
print(f"[sweep] holdout 추론 완료 ({len(hs)}행, au {auh.sum()})", flush=True)

# 3) λ 스윕
def sc(pred, m=None):
    m = np.ones(len(yh), bool) if m is None else m
    return macro_f1(yh[m], pred[m], 14)[0]
for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
    bs = b_all + lam * (b_sim - b_all)
    ba = b_all + lam * (b_au - b_all)
    brow = np.where(auh[:, None], ba[None, :], bs[None, :])
    pred = (scores + brow).argmax(1)
    print(f"λ={lam:4.2f}: 전체 {sc(pred):.5f} | sim {sc(pred, ~auh):.5f} | au {sc(pred, auh):.5f}")
