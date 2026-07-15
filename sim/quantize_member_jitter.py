"""멤버 zip int8 양자화 — 확률적 반올림(SR) 지터판.

quantize_member.py와 동일 파이프라인이되 np.round(RTN) 대신
q = floor(x + u), u~U(0,1) 확률적 반올림(비편향)으로 저마진 예측 플립을 유도.
같은 fp16 원본에서 --qseed만 바꾸면 서로 다른 추첨 티켓이 나온다.

usage: python sim/quantize_member_jitter.py <member.zip> --out <q8_out> --qseed 101
"""
from __future__ import annotations
import os, sys, json, shutil, zipfile, tempfile, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--out", default="")
ap.add_argument("--group", type=int, default=64)
ap.add_argument("--qseed", type=int, required=True)
ap.add_argument("--amp", type=float, default=1.0)  # 지터폭(LSB 단위): 1.0=SR, 2.0=±1LSB 균등노이즈 후 반올림
a = ap.parse_args()
G = a.group
out = a.out or a.src.replace(".zip", f"_qj{a.qseed}")
rng = np.random.default_rng(a.qseed)

tmp = tempfile.mkdtemp(prefix="qj_")
with zipfile.ZipFile(a.src) as z:
    z.extractall(tmp)
st_path = os.path.join(tmp, "model.safetensors")
assert os.path.exists(st_path), "model.safetensors 없음"

from safetensors import safe_open

tensors = {}
with safe_open(st_path, framework="np") as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

npz = {}
names_q, names_f = [], []
tot_q = tot_f = 0
diff_n = diff_d = 0  # RTN 대비 플립된 int8 엔트리 수
for k, w in tensors.items():
    if w.ndim == 2 and w.size >= 1_000_000 and w.shape[1] % G == 0:
        w32 = w.astype(np.float32)
        o, i = w32.shape
        g = w32.reshape(o, i // G, G)
        s = np.abs(g).max(axis=2) / 127.0
        s = np.maximum(s, 1e-12)
        x = g / s[:, :, None]
        if a.amp == 1.0:
            q_sr = np.clip(np.floor(x + rng.random(x.shape, dtype=np.float32)), -127, 127).astype(np.int8)
        else:
            # 비편향 균등노이즈 z~U(-amp/2, amp/2) 후 RTN — amp=1.0의 SR과 달리 폭 가변
            z = (rng.random(x.shape, dtype=np.float32) - 0.5) * a.amp
            q_sr = np.clip(np.round(x + z), -127, 127).astype(np.int8)
        q_rtn = np.clip(np.round(x), -127, 127).astype(np.int8)
        diff_n += int((q_sr != q_rtn).sum()); diff_d += q_sr.size
        npz[f"q::{k}"] = q_sr.reshape(o, i)
        npz[f"s::{k}"] = s.astype(np.float16)
        names_q.append(k); tot_q += w.size
    else:
        npz[f"f::{k}"] = w.astype(np.float16)
        names_f.append(k); tot_f += w.size

qp = os.path.join(tmp, "qweights.npz")
np.savez_compressed(qp, **npz)
os.remove(st_path)

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for r, _, fs in os.walk(tmp):
        for fn in fs:
            p = os.path.join(r, fn)
            z.write(p, os.path.relpath(p, tmp))
shutil.rmtree(tmp)

mb = os.path.getsize(out) / 1e6
print(f"[qj seed={a.qseed}] {a.src} -> {out}  {mb:.0f}MB")
print(f"  int8 {len(names_q)}개 {tot_q/1e6:.0f}M | fp16 {len(names_f)}개 {tot_f/1e6:.1f}M")
print(f"  RTN 대비 지터 플립 엔트리: {diff_n/max(diff_d,1)*100:.1f}% ({diff_n}/{diff_d})")
