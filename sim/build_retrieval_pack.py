"""배포용 retrieval 번들 생성 (R22 X1). 캐시된 train 임베딩 → 중심화·정규화·fp16 + 라벨 + mean.

출력: <out>/train_emb.npy (fp16 [70000,H], 중심화·정규화), train_labels.npy (int16), emb_mean.npy (fp32 [H]).
usage: python3 sim/build_retrieval_pack.py <out_dir>  (기본 work/retrieval_pack)
전제: work/emb_v6_70k.npy 존재(retrieval_diag2가 생성).
"""
from __future__ import annotations
import os, sys
import numpy as np

R = "/root/Action_Decision"
sys.path.insert(0, R)
from common.io_utils import load_train

out = sys.argv[1] if len(sys.argv) > 1 else f"{R}/work/retrieval_pack"
os.makedirs(out, exist_ok=True)
emb = np.load(f"{R}/work/emb_v6_70k.npy").astype(np.float32)
_, y, _ = load_train(); y = np.array(y, dtype=np.int16)
mu = emb.mean(0)
embc = emb - mu
embc /= (np.linalg.norm(embc, axis=1, keepdims=True) + 1e-9)
np.save(f"{out}/train_emb.npy", embc.astype(np.float16))
np.save(f"{out}/train_labels.npy", y)
np.save(f"{out}/emb_mean.npy", mu.astype(np.float32))
mb = sum(os.path.getsize(f"{out}/{f}") for f in os.listdir(out)) / 1e6
print(f"[retrieval-pack] {out}  {embc.shape} fp16 + labels + mean = {mb:.0f}MB")
