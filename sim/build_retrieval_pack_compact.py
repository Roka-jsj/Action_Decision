#!/usr/bin/env python3
"""Compact retrieval pack for fitting retrieval into tri_cond (R28).

Existing pack stores 70k x 1024 fp16 embeddings (~137MB). tri_cond has only
~57MB zip headroom, so we project centered-normalized embeddings to D dims and
store fp16 projected vectors plus the projection matrix.

usage: python3 sim/build_retrieval_pack_compact.py [out_dir] [dim=384]
"""
from __future__ import annotations
import os, sys, json
import numpy as np

R = "/root/Action_Decision"
sys.path.insert(0, R)
from common.io_utils import load_train

out = sys.argv[1] if len(sys.argv) > 1 else f"{R}/work/retrieval_pack_p384"
dim = int(sys.argv[2]) if len(sys.argv) > 2 else 384
os.makedirs(out, exist_ok=True)

emb = np.load(f"{R}/work/emb_v6_70k.npy", mmap_mode="r")
_, y, _ = load_train()
y = np.asarray(y, dtype=np.int16)
mu = np.asarray(emb, dtype=np.float32).mean(0)

rng = np.random.default_rng(20260708 + dim)
proj = rng.normal(0.0, 1.0 / np.sqrt(dim), size=(emb.shape[1], dim)).astype(np.float32)

dst = np.lib.format.open_memmap(os.path.join(out, "train_emb.npy"),
                                mode="w+", dtype=np.float16, shape=(emb.shape[0], dim))
bs = 4096
for b in range(0, emb.shape[0], bs):
    x = np.asarray(emb[b:b + bs], dtype=np.float32) - mu
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    z = x @ proj
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-9
    dst[b:b + bs] = z.astype(np.float16)
dst.flush()

np.save(os.path.join(out, "emb_mean.npy"), mu.astype(np.float32))
np.save(os.path.join(out, "proj.npy"), proj.astype(np.float16))
np.save(os.path.join(out, "train_labels.npy"), y)
meta = {"dim": dim, "source": "emb_v6_70k", "projection": "gaussian", "seed": 20260708 + dim}
json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)

mb = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 1e6
print(f"[compact-retrieval] {out} dim={dim} size={mb:.1f}MB")

# Lightweight preservation diagnostic on random train queries: compare top1 between
# full centered space and compact space. It is not a hidden proxy, just a sanity check.
idx = rng.choice(emb.shape[0], size=512, replace=False)
base = np.asarray(emb, dtype=np.float32)
base_c = base - mu
base_c /= np.linalg.norm(base_c, axis=1, keepdims=True) + 1e-9
comp = np.asarray(dst, dtype=np.float32)
agree = []
sim_full = []
sim_comp = []
for b in range(0, len(idx), 64):
    ii = idx[b:b + 64]
    qf = base_c[ii]
    qc = comp[ii]
    sf = qf @ base_c.T
    sc = qc @ comp.T
    for r, j in enumerate(ii):
        sf[r, j] = -9
        sc[r, j] = -9
    tf = sf.argmax(1)
    tc = sc.argmax(1)
    agree.extend((tf == tc).tolist())
    sim_full.extend(sf[np.arange(len(ii)), tf].tolist())
    sim_comp.extend(sc[np.arange(len(ii)), tc].tolist())
print(f"[diag] train top1 agreement={np.mean(agree):.3f} "
      f"full_med={np.median(sim_full):.4f} compact_med={np.median(sim_comp):.4f}")

