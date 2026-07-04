"""멤버 zip int8 양자화 — 두 large 멤버(1.32GB)를 1GB 제한 안으로.

model.safetensors → qweights.npz: 대형 2D 가중치(>=1e6 원소)는 per-row int8+fp16 scale,
나머지(LayerNorm/bias/포지션임베딩/분류기)는 fp16 원본 유지.
배포시 ad_lib._maybe_dequant가 npz → model.safetensors(fp16)를 복원 → 기존 로드경로 그대로.

usage: python sim/quantize_member.py <member.zip> [--out member_q8.zip]
검증: 복원 가중치 vs 원본 max/mean 상대오차 출력. (확률 패리티는 parity_check로 별도)
"""
from __future__ import annotations
import os, sys, json, shutil, zipfile, tempfile, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--out", default="")
ap.add_argument("--group", type=int, default=64)   # 양자화 그룹 크기(열 방향)
a = ap.parse_args()
G = a.group
out = a.out or a.src.replace(".zip", "_q8.zip")

tmp = tempfile.mkdtemp(prefix="q8_")
with zipfile.ZipFile(a.src) as z:
    z.extractall(tmp)
st_path = os.path.join(tmp, "model.safetensors")
assert os.path.exists(st_path), "model.safetensors 없음"

from safetensors import safe_open
from safetensors.numpy import save_file as np_save  # 검증용 미사용, torch 불필요 경로 유지

tensors = {}
with safe_open(st_path, framework="np") as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

npz = {}
names_q, names_f = [], []
tot_q = tot_f = 0
err_max = err_mean_n = err_mean_d = 0.0
for k, w in tensors.items():
    if w.ndim == 2 and w.size >= 1_000_000 and w.shape[1] % G == 0:
        w32 = w.astype(np.float32)
        o, i = w32.shape
        g = w32.reshape(o, i // G, G)                           # group-G 양자화
        s = np.abs(g).max(axis=2) / 127.0                       # (o, i/G)
        s = np.maximum(s, 1e-12)
        q = np.clip(np.round(g / s[:, :, None]), -127, 127).astype(np.int8)
        npz[f"q::{k}"] = q.reshape(o, i)
        npz[f"s::{k}"] = s.astype(np.float16)
        r = (q.astype(np.float32) * s[:, :, None].astype(np.float32)).reshape(o, i).astype(np.float16).astype(np.float32)
        e = np.abs(r - w32)
        err_max = max(err_max, float(e.max()))
        err_mean_n += float(e.sum()); err_mean_d += w.size
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
print(f"[q8] {a.src} -> {out}  {mb:.0f}MB")
print(f"  int8 {len(names_q)}개 텐서 {tot_q/1e6:.0f}M params | fp16 유지 {len(names_f)}개 {tot_f/1e6:.1f}M")
print(f"  복원오차 max={err_max:.5f} mean={err_mean_n/max(err_mean_d,1):.6f}")
