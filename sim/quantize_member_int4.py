"""멤버 int4 양자화 — 캐스케이드 폭 확장(R40): large 353MB→~190MB, 1GB에 5~6멤버.

int8(quantize_member.py)과 동일 구조에 4-bit 팩킹: 대형 2D 가중치를 group-G
symmetric int4(q∈[-7,7], scale=max|g|/7)로, 두 값을 nibble 팩(uint8 1바이트=2값).
복원은 ad_lib._dequant_state_dict의 "p4::" 태그 경로(신설).

⚠ 배포 전 parity 게이트 필수(조원 R66 codex): holdout macro-F1 델타·14클래스 발화·
   탐색4 혼동·offline load. 이 스크립트는 가중치 복원오차만 출력.

usage: python sim/quantize_member_int4.py <member.zip|member_dir> [--out out.zip] [--group 64]
"""
from __future__ import annotations
import os, sys, shutil, zipfile, tempfile, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--out", default="")
ap.add_argument("--group", type=int, default=64)
a = ap.parse_args()
G = a.group
out = a.out or (a.src.rstrip("/").replace(".zip", "") + "_q4.zip")

tmp = tempfile.mkdtemp(prefix="q4_")
if a.src.endswith(".zip"):
    with zipfile.ZipFile(a.src) as z:
        z.extractall(tmp)
else:
    shutil.copytree(a.src, tmp, dirs_exist_ok=True)
st_path = os.path.join(tmp, "model.safetensors")
assert os.path.exists(st_path), "model.safetensors 없음 (이미 양자화된 멤버면 원본 fp16 멤버를 입력하라)"

from safetensors import safe_open

tensors = {}
with safe_open(st_path, framework="np") as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

npz = {}
names_q, names_f = [], []
tot_q = tot_f = 0
err_max, err_sum, err_den = 0.0, 0.0, 0
for k, w in tensors.items():
    if w.ndim == 2 and w.size >= 1_000_000 and w.shape[1] % G == 0 and w.shape[1] % 2 == 0:
        w32 = w.astype(np.float32)
        o, i = w32.shape
        g = w32.reshape(o, i // G, G)
        s = np.abs(g).max(axis=2) / 7.0
        s = np.maximum(s, 1e-12)
        q = np.clip(np.round(g / s[:, :, None]), -7, 7).astype(np.int8).reshape(o, i)
        u = (q + 8).astype(np.uint8)                       # [1,15]
        packed = (u[:, 0::2] << 4) | u[:, 1::2]            # (o, i/2)
        npz[f"p4::{k}"] = packed
        npz[f"s4::{k}"] = s.astype(np.float16)
        r = (q.astype(np.float32).reshape(o, i // G, G) * s[:, :, None]).reshape(o, i)
        e = np.abs(r.astype(np.float16).astype(np.float32) - w32)
        err_max = max(err_max, float(e.max()))
        err_sum += float(e.sum()); err_den += w.size
        names_q.append(k); tot_q += w.size
    else:
        npz[f"f::{k}"] = w.astype(np.float16)
        names_f.append(k); tot_f += w.size

qp = os.path.join(tmp, "qweights.npz")
np.savez_compressed(qp, **npz)
os.remove(st_path)

if os.path.exists(out):
    os.remove(out)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for r, _, fs in os.walk(tmp):
        for fn in fs:
            p = os.path.join(r, fn)
            z.write(p, os.path.relpath(p, tmp))
shutil.rmtree(tmp)

mb = os.path.getsize(out) / 1e6
print(f"[q4] {a.src} -> {out}  {mb:.0f}MB")
print(f"  int4 {len(names_q)}개 텐서 {tot_q/1e6:.0f}M params | fp16 유지 {len(names_f)}개 {tot_f/1e6:.1f}M")
print(f"  복원오차 max={err_max:.5f} mean={err_sum/max(err_den,1):.6f}  (int8 대비 ~4배 큼 — parity 게이트 필수)")
