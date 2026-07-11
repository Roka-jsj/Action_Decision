#!/usr/bin/env python3
"""게이트E — mdeb-T3 제출 최종 판정기 (R65 레드팀 사전등록, 완주 직후 실행).

5k 홀드아웃 m2-스왑 paired: 배포 캐스케이드에서 m2만 조원-mdeb(h8@384) → mdeb-T3(h12@320)로
교체했을 때의 Δ를 잰다. p_old(조원m1)/p_m3(klue)는 캐시 재사용 — δ_mdeb 프로브와 동일 계기.

GO = casc Δ ≥ +0.002 ∧ CI95 하한 > -0.001 ∧ same-text solo Δ ≥ -0.004
LB 판독 = 0.42 × casc Δ (m1t3 단일점 캘리브). 앵커 0.79026.
사전등록 밴드(레드팀 R65 §4): 중앙 +0.0008, [-0.0010, +0.0035].

실행: python3 sim/gateE_mdebt3.py   (GPU ~10분, DONE_mdebt3full 필요)
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common import ad_lib                              # noqa: E402
from common.io_utils import load_train                 # noqa: E402
from sim import refit_lib as L                         # noqa: E402
from sim.refit_d4 import load_all                      # noqa: E402

W, TH = (0.45, 0.40, 0.15), 0.85
GO_CASC, GO_CI_LO, GO_SAMETEXT = 0.002, -0.001, -0.004   # R65 사전등록 — 수정 금지
MEMBER = os.path.join(ROOT, "work", "member_mdebt3full")

assert os.path.exists(os.path.join(ROOT, "work", "DONE_mdebt3full")), "mdeb-T3 미완주"
d = np.load(os.path.join(ROOT, "work/autopsy_m1t3_5k.npz"))
rows, yb, p_old = d["rows"], d["y"], d["p_old"]
c = np.load(os.path.join(ROOT, "work/tau_delta_5k_partial.npz"))
assert np.array_equal(c["rows"], rows)
p_m2_cc, p_m3 = c["p_m2"], c["p_m3"]                    # 조원-mdeb@384(h8), klue@320(h8)

samples, _, _ = load_train()
sub = [samples[i] for i in rows]
tx8 = [ad_lib.serialize(s, "v6", 8) for s in sub]
tx12 = [ad_lib.serialize(s, "v6", 12) for s in sub]
same = np.array([a == b for a, b in zip(tx8, tx12)])
print(f"[gateE] h12 변경행 {100*(1-same.mean()):.1f}% (게이트B 밴드 [37,45])")

p_m2_new = ad_lib.predict_logits(MEMBER, sub, version="v6", max_len=320, batch_size=128,
                                 texts=tx12, return_probs=True, gen_rescue=True)

class _A: m1 = mdeb = klue = ""
_, _, _, _, _, old_bias = load_all(_A())

def casc(pm2):
    P, _ = L.cascade_probs([p_old, pm2, p_m3], W, TH)
    return L.bias_argmax(P, old_bias)

pred_cc, pred_new = casc(p_m2_cc), casc(p_m2_new)
f_cc, f_new = L.fast_macro_f1(yb, pred_cc), L.fast_macro_f1(yb, pred_new)
d_casc = f_new - f_cc

s_cc, s_new = p_m2_cc.argmax(1), p_m2_new.argmax(1)
f_s_cc, f_s_new = L.fast_macro_f1(yb, s_cc), L.fast_macro_f1(yb, s_new)
d_solo = f_s_new - f_s_cc
d_solo_same = L.fast_macro_f1(yb[same], s_new[same]) - L.fast_macro_f1(yb[same], s_cc[same])
d_solo_chg = (L.fast_macro_f1(yb[~same], s_new[~same]) - L.fast_macro_f1(yb[~same], s_cc[~same])
              if (~same).sum() else float("nan"))

rng = np.random.default_rng(20260711)
n = len(yb); boots = []
for _ in range(2000):
    idx = rng.integers(0, n, n)
    boots.append(L.fast_macro_f1(yb[idx], pred_new[idx]) - L.fast_macro_f1(yb[idx], pred_cc[idx]))
lo, hi = np.percentile(boots, [2.5, 97.5])

go = (d_casc >= GO_CASC) and (lo > GO_CI_LO) and (d_solo_same >= GO_SAMETEXT)
out = {"casc_cc": round(float(f_cc), 5), "casc_new": round(float(f_new), 5),
       "d_casc": round(float(d_casc), 5), "ci95": [round(float(lo), 5), round(float(hi), 5)],
       "d_solo": round(float(d_solo), 5), "d_solo_sametext": round(float(d_solo_same), 5),
       "d_solo_changed": round(float(d_solo_chg), 5) if d_solo_chg == d_solo_chg else None,
       "changed_ratio": round(float(1 - same.mean()), 4),
       "lb_reading_042": round(float(0.42 * d_casc), 5),
       "verdict": "GO" if go else "NO-GO"}
print("[gateE]", json.dumps(out, ensure_ascii=False))
np.savez(os.path.join(ROOT, "work/gateE_mdebt3_5k.npz"), rows=rows,
         p_m2_new=p_m2_new.astype(np.float32), summary=json.dumps(out))
print(f"[gateE] 판정: {out['verdict']} — GO조건: Δcasc≥{GO_CASC} ∧ CI하한>{GO_CI_LO} ∧ same-text≥{GO_SAMETEXT}")
sys.exit(0 if go else 1)
