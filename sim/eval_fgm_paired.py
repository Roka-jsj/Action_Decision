#!/usr/bin/env python3
"""FGM 차분런 판독 — FGM_paired = F1(fgm1_s777) - 0.80752(plain s777). R71b FGM 포함규칙 입력."""
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common import ad_lib
from common.io_utils import load_train
from sim import refit_lib as L
d = np.load(os.path.join(ROOT, "work/autopsy_m1t3_5k.npz"))
rows, yb = d["rows"], d["y"]
samples, _, _ = load_train()
sub = [samples[i] for i in rows]
tx8 = [ad_lib.serialize(s, "v6", 8) for s in sub]
p = ad_lib.predict_logits(os.path.join(ROOT, "work/member_m1h8full_fgm1_s777"), sub,
                          version="v6", max_len=320, batch_size=128, texts=tx8,
                          return_probs=True, gen_rescue=True)
f = float(L.fast_macro_f1(yb, p.argmax(1)))
out = {"f1_fgm": round(f,5), "fgm_paired": round(f-0.80752,5), "delta_vs_deployed": round(f-0.82584,5),
       "include_fgm": bool(f-0.80752 >= 0.002)}
print(json.dumps(out))
