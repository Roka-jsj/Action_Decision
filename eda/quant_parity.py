"""int8 양자화 패리티 — 원본 멤버 vs q8 멤버 확률 비교 (CPU, holdout 소표본).

게이트: max|Δp| < 0.02 & argmax 일치율 >= 99% → PASS (양자화 배포 안전).
usage: python eda/quant_parity.py <orig.zip> <q8.zip> <version> [n=48]
"""
from __future__ import annotations
import os, sys, zipfile, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.io_utils import load_train
from common.cv import make_splits
from common import ad_lib

orig, q8, ver = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 48

samples, y, ids = load_train(); y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups)
ho = sp["holdout_idx"]
rng = np.random.RandomState(0)
pick = rng.choice(ho, size=n, replace=False)
sub = [samples[i] for i in pick]

def probs(zp):
    d = tempfile.mkdtemp(prefix="qp_")
    with zipfile.ZipFile(zp) as z:
        z.extractall(d)
    return ad_lib.predict_logits(d, sub, version=ver, max_len=320, batch_size=16,
                                 device="cpu", return_probs=True)

p0 = probs(orig)
p1 = probs(q8)
dp = np.abs(p1 - p0)
agree = float((p0.argmax(1) == p1.argmax(1)).mean())
print(f"[quant-parity] n={n} ver={ver}")
print(f"  max|Δp|={dp.max():.5f}  mean|Δp|={dp.mean():.6f}  argmax일치={agree:.4f}")
print(f"  판정: {'PASS' if dp.max() < 0.02 and agree >= 0.99 else 'FAIL — 양자화 재검토'}")
