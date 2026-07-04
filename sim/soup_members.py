"""FULL 멤버 zip N개 → 가중치 평균(model soup) 단일 멤버 zip.

T4 10분 캡 대응: 2-large 앙상블(719s) 불가 → soup은 단일모델 추론비용(348s)으로
앙상블성 이득을 노림. 같은 사전학습 init + 같은 직렬화/하이퍼 + seed만 다른 런끼리만 유효.
전제: id_map.npy 동일(같은 직렬화 → 결정적 vocab pruning). 불일치 시 중단.

usage: python sim/soup_members.py --out member_soup.zip <m1.zip> <m2.zip> [...]
검증: quant/토크나이저는 m1 것 복사, safetensors는 fp32 누적평균→fp16. 이후 T4 벤치로 게이트.
"""
from __future__ import annotations
import os, sys, json, shutil, zipfile, tempfile, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("members", nargs="+")
a = ap.parse_args()
assert len(a.members) >= 2

from safetensors import safe_open
from safetensors.numpy import save_file

dirs = []
for m in a.members:
    d = tempfile.mkdtemp(prefix="soup_")
    with zipfile.ZipFile(m) as z:
        z.extractall(d)
    assert os.path.exists(os.path.join(d, "model.safetensors")), f"{m}: safetensors 없음(양자화 zip은 원본으로)"
    dirs.append(d)

# id_map 동일성 (다르면 soup 무효)
maps = [np.load(os.path.join(d, "id_map.npy")) for d in dirs if os.path.exists(os.path.join(d, "id_map.npy"))]
if maps:
    assert len(maps) == len(dirs), "일부만 pruned — 혼합 불가"
    for m2 in maps[1:]:
        assert maps[0].shape == m2.shape and (maps[0] == m2).all(), "id_map 불일치 — soup 중단"

acc = {}
n = len(dirs)
for i, d in enumerate(dirs):
    with safe_open(os.path.join(d, "model.safetensors"), framework="np") as f:
        for k in f.keys():
            t = f.get_tensor(k).astype(np.float32)
            acc[k] = t / n if i == 0 else acc[k] + t / n

out_t = {k: v.astype(np.float16) for k, v in acc.items()}
save_file(out_t, os.path.join(dirs[0], "model.safetensors"), metadata={"format": "pt"})

with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
    for r, _, fs in os.walk(dirs[0]):
        for fn in fs:
            p = os.path.join(r, fn)
            z.write(p, os.path.relpath(p, dirs[0]))
for d in dirs:
    shutil.rmtree(d)
print(f"[soup] {n}개 평균 -> {a.out}  {os.path.getsize(a.out)/1e6:.0f}MB")
