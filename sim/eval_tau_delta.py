#!/usr/bin/env python3
"""R63b 판정 계약 — τ(동일 파이프라인 내 T3 효과)/δ(파이프라인 격차) 분리 후 m1t3 제출 GO/NO.

전제 산출물:
  work/autopsy_m1t3_5k.npz   — 게이트C 부검 산출(rows/y/p_old=조원m1 solo/p_new=m1t3 m1 solo, 5k 홀드아웃)
  work/member_m1h8full/      — 대조군 FULL(구입력 h8·8ep, R63b 서명, seed1234) — DONE_m1h8full 확인 후 실행
  packages/submit_th85/      — 앵커 패키지(m2/m3·postproc 원천)

서빙 정합(중요): 조원m1·대조군은 h8+rescue(배포등가 — rescue는 추론시 글로벌), m1t3는 h12+rescue.
  δ = ctl − 조원m1 (동일 서빙·동일 텍스트 → 순수 파이프라인/레시피 격차)
  τ = m1t3 − ctl   (같은 우리 파이프라인 → T3(rescue+h12 학습)의 배포등가 효과)

재개 조건(codex R63b 사전 고정, 변경 금지):
  GO       = τ_cascade ≥ +0.003 ∧ τ_cascade CI95 하한 > 0 ∧ τ_solo ≥ 0
             → submit_m1t3.zip 제출. 개정 사전등록: 중앙 = 0.31×τ_cascade, 밴드 ±0.005,
               파국하한 <0.79026 → th85 앵커 복귀.
  축 폐쇄   = τ_cascade ≤ 0 → m1t3 미제출·T3-FULL 축 폐쇄(mdeb-T3 FULL도 재설계 전 보류).
  회색      = 그 외 → 소형 paired ablation 추가 설계, 제출 금지.

실행: python3 sim/eval_tau_delta.py   (GPU 필요 ~5분; m2/m3/ctl 확률은 work/tau_delta_5k.npz 캐시)

주의(경미한 비대칭): p_old/p_new는 패키지 int8 양자화본, ctl은 fp16 원본 — 양자화 잡음은
parity 게이트에서 무해 실측(이탈 스캔 #4)이라 판정 임계값 여유 안에서 무시한다.
"""
from __future__ import annotations
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common import ad_lib                              # noqa: E402
from common.io_utils import load_train                 # noqa: E402
from sim import refit_lib as L                         # noqa: E402
from sim.refit_d4 import load_all                      # noqa: E402

AUT = os.path.join(ROOT, "work", "autopsy_m1t3_5k.npz")
CTL = os.path.join(ROOT, "work", "member_m1h8full")
CACHE = os.path.join(ROOT, "work", "tau_delta_5k.npz")
W, TH = (0.45, 0.40, 0.15), 0.85

assert os.path.exists(AUT), "autopsy npz 없음 — 게이트C 부검 산출물이 이전되지 않았다"
assert os.path.exists(os.path.join(ROOT, "work", "DONE_m1h8full")), "대조군 미완료(DONE_m1h8full 없음)"
d = np.load(AUT)
rows, yb = d["rows"], d["y"]
p_old, p_new = d["p_old"], d["p_new"]          # 조원m1(h8+rescue) / m1t3(h12+rescue) solo probs

if os.path.exists(CACHE):
    c = np.load(CACHE)
    assert np.array_equal(c["rows"], rows), "캐시 행 불일치"
    p_ctl, p_m2, p_m3 = c["p_ctl"], c["p_m2"], c["p_m3"]
else:
    samples, _, _ = load_train()
    sub_s = [samples[i] for i in rows]
    tx8 = [ad_lib.serialize(s, "v6", 8) for s in sub_s]
    p_ctl = ad_lib.predict_logits(CTL, sub_s, version="v6", max_len=320, batch_size=128,
                                  texts=tx8, return_probs=True, gen_rescue=True)
    p_m2 = ad_lib.predict_logits(os.path.join(ROOT, "packages/submit_th85/model/m2"), sub_s,
                                 version="v6", max_len=384, batch_size=128,
                                 texts=tx8, return_probs=True, gen_rescue=True)
    p_m3 = ad_lib.predict_logits(os.path.join(ROOT, "packages/submit_th85/model/m3"), sub_s,
                                 version="v6", max_len=320, batch_size=128,
                                 texts=tx8, return_probs=True, gen_rescue=True)
    np.savez(CACHE, rows=rows, p_ctl=p_ctl.astype(np.float32),
             p_m2=p_m2.astype(np.float32), p_m3=p_m3.astype(np.float32))

class _A:
    m1 = mdeb = klue = ""
_, _, _, _, _, old_bias = load_all(_A())

def casc(p_m1):
    P, _ = L.cascade_probs([p_m1, p_m2, p_m3], W, TH)
    return L.bias_argmax(P, old_bias)

pred = {k: casc(p) for k, p in (("old", p_old), ("ctl", p_ctl), ("new", p_new))}
solo = {k: p.argmax(1) for k, p in (("old", p_old), ("ctl", p_ctl), ("new", p_new))}
f_c = {k: L.fast_macro_f1(yb, v) for k, v in pred.items()}
f_s = {k: L.fast_macro_f1(yb, v) for k, v in solo.items()}
d_solo = f_s["ctl"] - f_s["old"]
t_solo = f_s["new"] - f_s["ctl"]
d_casc = f_c["ctl"] - f_c["old"]
t_casc = f_c["new"] - f_c["ctl"]

rng = np.random.default_rng(631)
n = len(yb)
bs = []
for _ in range(2000):
    idx = rng.integers(0, n, n)
    bs.append(L.fast_macro_f1(yb[idx], pred["new"][idx]) - L.fast_macro_f1(yb[idx], pred["ctl"][idx]))
lo, hi = np.percentile(bs, [2.5, 97.5])

print(f"[τδ] 5k 홀드아웃 (암기 공유 계기 — 절대치는 LB 예측 아님)")
print(f"  solo   : 조원m1 {f_s['old']:.5f} | ctl {f_s['ctl']:.5f} | m1t3 {f_s['new']:.5f}"
      f"  →  δ_solo={d_solo:+.5f}  τ_solo={t_solo:+.5f}")
print(f"  cascade: 조원m1 {f_c['old']:.5f} | ctl {f_c['ctl']:.5f} | m1t3 {f_c['new']:.5f}"
      f"  →  δ_casc={d_casc:+.5f}  τ_casc={t_casc:+.5f}  CI95[{lo:+.5f},{hi:+.5f}]")
if t_casc >= 0.003 and lo > 0 and t_solo >= 0:
    print(f"  판정: GO — submit_m1t3.zip 제출. 개정 사전등록: 앵커 0.79026, "
          f"중앙 {0.31*t_casc:+.5f}, 밴드 ±0.005, <앵커 시 th85 복귀")
elif t_casc <= 0:
    print("  판정: 축 폐쇄 — m1t3 미제출, T3-FULL 축 닫음(mdeb-T3 FULL 재설계 전 보류)")
else:
    print("  판정: 회색 — 제출 금지, 소형 paired ablation 설계(3자 토론)")
