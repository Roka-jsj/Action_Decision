#!/usr/bin/env python3
"""R74 사전등록 게이트 — 신규 학습기법 콤보(FGM+AWP+EMA 등)로 학습한 m1 이 배포 m1 을 이겼는가.

구조는 sim/gate_m1wsT3.py 와 동일(5k paired, 서빙 h8+rescue, th85). 차이: MEMBER 를 env
AD_GATE_MEMBER 로 파라미터화하고, 배포 m1 의 solo(0.82584)를 명시 보고한다.

사전등록(코드로 동결):
  GO = casc ≥ +0.0025               (5k-swap 캐스케이드 게이트)
     ∧ paired-casc CI95 하한 > 0    (부트스트랩 2000)
     ∧ solo ≥ 0  (= new solo ≥ 배포 m1 solo 0.82584 → "배포 m1 을 solo 로 이김")
     ∧ same-text solo ≥ -0.002      (h12↔h8 텍스트변경 슬라이스 안전판)
전제: 학습이 5k(exclude_rows_5k.npy)를 제외했으므로 이 5k 는 우리 모델에 held-out.
      배포 m1 은 이 5k 를 봤을 개연(70k full) → 편향은 보수(우리에게 불리) 방향.
실행: env AD_GATE_MEMBER=work/member_<tag>_q8dir CUDA_VISIBLE_DEVICES=<free_gpu> \
      PYTHONPATH=/root/Action_Decision python3 sim/gate_m1_r74.py
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
DEPLOYED_M1_SOLO = 0.82584                    # 배포 m1 solo(5k) — 넘어야 할 절대선
MEMBER = os.environ.get("AD_GATE_MEMBER", os.path.join(ROOT, "work", "member_m1r74gate_q8dir"))

d = np.load(os.path.join(ROOT, "work/autopsy_m1t3_5k.npz"))
rows, yb, p_old = d["rows"], d["y"], d["p_old"]
c = np.load(os.path.join(ROOT, "work/tau_delta_5k_partial.npz"))
p_m2, p_m3 = c["p_m2"], c["p_m3"]
samples, _, _ = load_train()
sub = [samples[i] for i in rows]
tx8 = [ad_lib.serialize(s, "v6", 8) for s in sub]
tx12 = [ad_lib.serialize(s, "v6", 12) for s in sub]
same = np.array([a == b for a, b in zip(tx8, tx12)])

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
solo_new = float(f(yb, p_new.argmax(1)))
solo_old = float(f(yb, p_old.argmax(1)))
d_solo = solo_new - solo_old
s_new, s_old = p_new.argmax(1), p_old.argmax(1)
d_solo_same = float(f(yb[same], s_new[same]) - f(yb[same], s_old[same]))

rng = np.random.default_rng(20260712)
n = len(yb); boots = []
for _ in range(2000):
    i = rng.integers(0, n, n)
    boots.append(f(yb[i], pred_new[i]) - f(yb[i], pred_old[i]))
lo, hi = np.percentile(boots, [2.5, 97.5])

go = (d_casc >= GO_CASC) and (lo > 0) and (d_solo >= GO_SOLO) and (d_solo_same >= GO_SAMETEXT)
out = {"member": MEMBER,
       "solo_new": round(solo_new, 5), "deployed_m1_solo": DEPLOYED_M1_SOLO,
       "beat_deployed_solo": bool(solo_new >= DEPLOYED_M1_SOLO),
       "d_solo": round(d_solo, 5), "d_solo_sametext": round(d_solo_same, 5),
       "d_casc": round(d_casc, 5), "ci95": [round(float(lo), 5), round(float(hi), 5)],
       "lb_pred": {"low_025x": round(0.79026 + 0.25 * d_casc, 5),
                   "central_036x": round(0.79026 + 0.36 * d_casc, 5),
                   "high_050x": round(0.79026 + 0.50 * d_casc, 5)},
       "verdict": "GO" if go else "NO-GO"}
print(json.dumps(out, ensure_ascii=False))
np.savez(os.path.join(ROOT, "work/gate_m1_r74_5k.npz"), rows=rows,
         p_new=p_new.astype(np.float32), summary=json.dumps(out, ensure_ascii=False))
sys.exit(0 if go else 1)
