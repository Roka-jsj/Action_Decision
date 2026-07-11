#!/usr/bin/env python3
"""R70c SWA 재현런 판정 — 사전등록 분기(codex#11 + 레드팀 조건):
CONFIRM = δ_swa ≥ -0.005 ∧ SWA_paired ≥ +0.010  →  run2 = T3+SWA 후보
DEAD    = δ_swa ≤ -0.012                        →  run2 = FGM 단일차분
GRAY    = 그 외(부분효과)                        →  run2 = FGM 단일차분(원인 분해)
ε̂(무료 노이즈 계기) = |F1(raw쌍둥이) - 0.80752| — ≥0.008이면 전 단일점 δ 판독 ±ε̂ 강등(기록).
기준값(5k solo, 실측): 배포m1 0.82584 / plain s777 0.80752.
출력: JSON(stdout) — night_chain_gpu0b.sh가 branch 필드로 분기.
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common import ad_lib                    # noqa: E402
from common.io_utils import load_train       # noqa: E402
from sim import refit_lib as L               # noqa: E402

DEPLOYED, PLAIN_S777 = 0.82584, 0.80752

d = np.load(os.path.join(ROOT, "work/autopsy_m1t3_5k.npz"))
rows, yb = d["rows"], d["y"]
samples, _, _ = load_train()
sub = [samples[i] for i in rows]
tx8 = [ad_lib.serialize(s, "v6", 8) for s in sub]

def f1_of(member):
    p = ad_lib.predict_logits(os.path.join(ROOT, "work", member), sub, version="v6",
                              max_len=320, batch_size=128, texts=tx8,
                              return_probs=True, gen_rescue=True)
    return float(L.fast_macro_f1(yb, p.argmax(1)))

f_swa = f1_of("member_m1h8full_swa2_s777")
d_swa = f_swa - DEPLOYED
paired = f_swa - PLAIN_S777
eps = None
if os.path.isdir(os.path.join(ROOT, "work", "member_m1h8full_swa2_s777raw")):
    eps = abs(f1_of("member_m1h8full_swa2_s777raw") - PLAIN_S777)

if d_swa >= -0.005 and paired >= 0.010:
    br = "CONFIRM"
elif d_swa <= -0.012:
    br = "DEAD"
else:
    br = "GRAY"
out = {"f1_swa": round(f_swa, 5), "delta_swa": round(d_swa, 5), "swa_paired": round(paired, 5),
       "eps_rerun_noise": round(eps, 5) if eps is not None else None,
       "eps_alert": bool(eps is not None and eps >= 0.008), "branch": br}
print(json.dumps(out))
