#!/usr/bin/env python3
"""m1wsT3 제출 게이트 — R71b 사전등록(codex#13 조건 반영: 양자화본·paired casc CI).

5k paired vs 배포 m1(p_old), 서빙 h8+rescue(τ 측정 기준과 동일·시간 th85 물리):
GO = casc ≥ +0.0025 ∧ paired-casc CI95 하한 > 0 ∧ solo ≥ 0 ∧ same-text solo ≥ -0.002
LB 사전등록: Δ = β×casc, 중앙 0.36×, 밴드 [0.25×, 0.50×]. 앵커 0.79026.
주의: 학습이 5k를 제외했으므로(65k) 이 게이트는 배포와 동일노출 조건 — 편향은 보수 방향.
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common import ad_lib                    # noqa: E402
from common.io_utils import load_train       # noqa: E402
from sim import refit_lib as L               # noqa: E402
from sim.refit_d4 import load_all            # noqa: E402

W, TH = (0.45, 0.40, 0.15), 0.85
GO_CASC, GO_SOLO, GO_SAMETEXT = 0.0025, 0.0, -0.002
MEMBER = os.path.join(ROOT, "work", "member_m1wsT3_q8dir")

d = np.load(os.path.join(ROOT, "work/autopsy_m1t3_5k.npz"))
rows, yb, p_old = d["rows"], d["y"], d["p_old"]
c = np.load(os.path.join(ROOT, "work/tau_delta_5k_partial.npz"))
p_m2, p_m3 = c["p_m2"], c["p_m3"]
samples, _, _ = load_train()
sub = [samples[i] for i in rows]
tx8 = [ad_lib.serialize(s, "v6", 8) for s in sub]
tx12 = [ad_lib.serialize(s, "v6", 12) for s in sub]
same = np.array([a == b for a, b in zip(tx8, tx12)])   # h12 학습이 바꾼 행(참고 슬라이스)

p_new = ad_lib.predict_logits(MEMBER, sub, version="v6", max_len=320, batch_size=128,
                              texts=tx8, return_probs=True, gen_rescue=True)

class _A: m1 = mdeb = klue = ""
_, _, _, _, _, old_bias = load_all(_A())
def casc(pm1):
    P, _ = L.cascade_probs([pm1, p_m2, p_m3], W, TH)
    return L.bias_argmax(P, old_bias)

f = L.fast_macro_f1
pred_new, pred_old = casc(p_new), casc(p_old)
d_casc = float(f(yb, pred_new) - f(yb, pred_old))
d_solo = float(f(yb, p_new.argmax(1)) - f(yb, p_old.argmax(1)))
s_new, s_old = p_new.argmax(1), p_old.argmax(1)
d_solo_same = float(f(yb[same], s_new[same]) - f(yb[same], s_old[same]))

rng = np.random.default_rng(20260711)
n = len(yb); boots = []
for _ in range(2000):
    i = rng.integers(0, n, n)
    boots.append(f(yb[i], pred_new[i]) - f(yb[i], pred_old[i]))
lo, hi = np.percentile(boots, [2.5, 97.5])

go = (d_casc >= GO_CASC) and (lo > 0) and (d_solo >= GO_SOLO) and (d_solo_same >= GO_SAMETEXT)
out = {"d_casc": round(d_casc, 5), "ci95": [round(float(lo), 5), round(float(hi), 5)],
       "d_solo": round(d_solo, 5), "d_solo_sametext": round(d_solo_same, 5),
       "lb_pred": {"low_025x": round(0.79026 + 0.25 * d_casc, 5),
                   "central_036x": round(0.79026 + 0.36 * d_casc, 5),
                   "high_050x": round(0.79026 + 0.50 * d_casc, 5)},
       "verdict": "GO" if go else "NO-GO"}
print(json.dumps(out))
np.savez(os.path.join(ROOT, "work/gate_m1wsT3_5k.npz"), rows=rows,
         p_new=p_new.astype(np.float32), summary=json.dumps(out))
sys.exit(0 if go else 1)
