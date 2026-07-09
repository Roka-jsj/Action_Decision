#!/usr/bin/env python
"""int4 parity 게이트 (R41): fp16 멤버 vs q4 멤버의 holdout 확률 대조.

조원 R67의 int4 기각은 문헌(PTQ 0.3~1pt·툴링) 근거 — 우리 구현(group-64 weight-only,
메모리 내 fp16 복원, 서버 의존성 0)에 대한 실측 판정은 이것이 최초.
게이트(사전등록): argmax 일치 ≥99.3% AND holdout macro-F1 델타 ≥-0.0005 AND
                 저마진(margin<0.2) 행 flip률 ≤3% AND 14클래스 전부 발화.

usage: python3 sim/parity_int4.py <fp16_member.zip|dir> <q4_member.zip> <version> [N=3000]
"""
from __future__ import annotations
import os, sys, tempfile, zipfile, shutil
import numpy as np

R = "/root/Action_Decision"
sys.path.insert(0, R)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from common.io_utils import load_train, CLASSES, NUM_CLASSES
from common.cv import make_splits
from common.metrics import macro_f1
from common import ad_lib

fp16_src, q4_src, ver = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 3000

samples, y, ids = load_train(); y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups)
hold_idx = np.asarray(sp["holdout_idx"])[:N]
hs = [samples[int(j)] for j in hold_idx]
yt = y[hold_idx]

def as_dir(src):
    if os.path.isdir(src):
        return src, None
    tmp = tempfile.mkdtemp(prefix="parity_")
    with zipfile.ZipFile(src) as z:
        z.extractall(tmp)
    return tmp, tmp

out = {}
for tag, src in (("fp16", fp16_src), ("q4", q4_src)):
    d, tmp = as_dir(src)
    out[tag] = ad_lib.predict_logits(d, hs, version=ver, max_len=320, batch_size=128, return_probs=True)
    if tmp:
        shutil.rmtree(tmp)

P0, P1 = out["fp16"], out["q4"]
a0, a1 = P0.argmax(1), P1.argmax(1)
agree = (a0 == a1).mean()
srt = np.sort(P0, 1); margin = srt[:, -1] - srt[:, -2]
low = margin < 0.2
flip_low = (a0[low] != a1[low]).mean() if low.any() else 0.0
f0 = macro_f1(yt, a0)[0]; f1 = macro_f1(yt, a1)[0]
maxdiff = float(np.abs(P0 - P1).max())
classes_fired = len(set(a1.tolist()))
print(f"[parity] n={N} ver={ver}")
print(f"  argmax 일치     {agree*100:.2f}%  (게이트 ≥99.3%)")
print(f"  저마진 flip률    {flip_low*100:.2f}% (저마진 {low.mean()*100:.0f}%행, 게이트 ≤3%)")
print(f"  holdout F1      fp16 {f0:.5f} → q4 {f1:.5f} ({f1-f0:+.5f}, 게이트 ≥-0.0005)")
print(f"  prob maxdiff    {maxdiff:.4f} | q4 발화 클래스 {classes_fired}/14 (게이트 14)")
ok = agree >= 0.993 and (f1 - f0) >= -0.0005 and flip_low <= 0.03 and classes_fired == NUM_CLASSES
print(f"  판정: {'PASS ✅' if ok else 'FAIL ❌'}")
