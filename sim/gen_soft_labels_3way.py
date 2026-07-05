"""3-way full-blend teacher 소프트라벨 생성 (서버 A6000, R13 Q3 합의).

기존 soft_labels_str2(2-way, 상한 0.78189) 대체: 현 최강 tri_cond의 full 3-way 구성
blend = 0.6*P(largev6-8ep) + 0.15*P(basev6e5) + 0.25*P(largev4-8ep)  (full 3-way OOF 0.7505)
bias/조건부는 결정층 산물이라 미적용 — raw 확률 블렌드만.

usage: python3 sim/gen_soft_labels_3way.py --out <out.npz> <zip::ver::w> [<zip::ver::w> ...]
예: python3 sim/gen_soft_labels_3way.py --out action_decision_maximum/experiments/soft_labels_tri.npz \
    action_decision_maximum/experiments/member_largefullv6.zip::v6::0.6 \
    action_decision_maximum/experiments/member_basev6e5full.zip::v6::0.15 \
    action_decision_maximum/experiments/member_largev4_8ep.zip::v4::0.25
"""
from __future__ import annotations
import argparse, os, sys, tempfile, zipfile, shutil
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
from common.io_utils import load_train
from common import ad_lib

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("members", nargs="+", help="zip::version::weight")
a = ap.parse_args()

samples, y, ids = load_train()
blend, wsum = None, 0.0
for spec in a.members:
    z, ver, w = spec.split("::")
    w = float(w)
    d = tempfile.mkdtemp(prefix="soft3_")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(d)
    p = ad_lib.predict_logits(d, samples, version=ver, max_len=320,
                              batch_size=256, return_probs=True)
    blend = w * p if blend is None else blend + w * p
    wsum += w
    shutil.rmtree(d)
    print(f"[soft3] {os.path.basename(z)} (v={ver} w={w}) done", flush=True)

blend = (blend / wsum).astype(np.float16)
np.savez_compressed(a.out, probs=blend, ids=np.array(ids))
acc = float((blend.argmax(1) == np.array(y)).mean())
print(f"[soft3] saved {a.out}  train-acc(참고)={acc:.4f}", flush=True)
