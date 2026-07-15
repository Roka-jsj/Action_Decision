#!/usr/bin/env python3
"""M4 전문가 게이트 평가 — leak-free (CPU).

입력: work/m4_spec_f0val.npz (m4_specialist.py 산출, fold0-val 4-way probs).
캐스케이드: load_all()로 fold0 멤버 OOF 복원 → W=(0.45,0.40,0.15), TH=0.85, old_bias.

(a) 전문가 4-way F1 (true-M4 val) vs 캐스케이드 M4-restricted F1 (동일 행).
(b) REAL METRIC: eligible = (cascade argmax∈M4) & (margin<0.85) 행에서 캐스케이드 예측을
    전문가 4-way argmax로 교체(M4 내부 재정규 = 4-way head argmax) → 전체 14클래스 macro-F1.
    ΔmacroF1 = blended - baseline.
HARD GATE(사전등록): fold0-val ΔmacroF1 >= +0.006 → PASS.
"""
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sim import refit_lib as L
from sim.refit_d4 import load_all
from common.io_utils import CLASSES

M4 = [0, 1, 2, 3]
W, TH = (0.45, 0.40, 0.15), 0.85
GATE = 0.006


class _A:
    m1 = mdeb = klue = ""


def per_class_f1(y_true, y_pred, classes):
    out = {}
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        d = 2 * tp + fp + fn
        out[c] = (2 * tp / d) if d > 0 else 0.0
    return out


def main():
    d = np.load(os.path.join(ROOT, "work", "m4_spec_f0val.npz"), allow_pickle=True)
    rows = d["rows"]; yv = d["y_true"]; probs4 = d["probs4"]
    best_ep = int(d["best_epoch"])
    print(f"[spec] best_epoch={best_ep} idev_hist={d['idev_f1_hist'].tolist()} "
          f"val_hist(true-M4)={d['val_f1_hist'].tolist()}")

    ids, y, folds, fmap, members, old_bias = load_all(_A())
    va0 = folds[0][1]
    assert np.array_equal(np.asarray(rows), np.asarray(va0)), "npz rows != fold0-val"
    assert np.array_equal(yv, y[va0]), "y mismatch"

    P, sel = L.cascade_probs([members[0].oof[va0], members[1].oof[va0],
                              members[2].oof[va0]], W, TH)
    pred = L.bias_argmax(P, old_bias)
    srt = np.sort(P, axis=1)
    margin = srt[:, -1] - srt[:, -2]

    base_f1 = L.fast_macro_f1(yv, pred)
    print(f"\n[baseline] fold0-val 14-class macro-F1 = {base_f1:.5f}")

    spec_pred4 = probs4.argmax(1)   # 0..3
    m4_true = np.isin(yv, M4)

    # (a) 4-way F1 비교 (true-M4 val 행)
    spec_4f1 = L.fast_macro_f1(yv[m4_true], spec_pred4[m4_true], n_classes=4)
    # 캐스케이드: 14-way as-is per-class(참고) + M4로 사영한 4-way(사과대사과)
    casc_pcf = per_class_f1(yv[m4_true], pred[m4_true], M4)
    casc_asis = float(np.mean([casc_pcf[c] for c in M4]))
    casc_proj4 = P[:, M4].argmax(1)     # M4 사영 argmax
    casc_4f1 = L.fast_macro_f1(yv[m4_true], casc_proj4[m4_true], n_classes=4)
    print(f"\n(a) 4-way discrimination on true-M4 val rows (N={int(m4_true.sum())}):")
    print(f"    cascade 14-way as-is  mean-M4-F1 = {casc_asis:.4f}  "
          f"(per-class {[round(casc_pcf[c],3) for c in M4]})")
    print(f"    cascade M4-projected  4-way-F1   = {casc_4f1:.4f}")
    print(f"    SPECIALIST            4-way-F1   = {spec_4f1:.4f}  "
          f"(Δ vs proj {spec_4f1-casc_4f1:+.4f})")
    spec_pcf = per_class_f1(yv[m4_true], spec_pred4[m4_true], M4)
    print(f"    specialist per-class F1 = {[round(spec_pcf[c],3) for c in M4]}")

    # (b) REAL METRIC — 블렌드
    def blend_delta(elig, tag):
        bp = pred.copy()
        bp[elig] = spec_pred4[elig]          # 4-way head argmax = M4 내부 재정규 argmax
        f = L.fast_macro_f1(yv, bp)
        n = int(elig.sum())
        changed = int((bp[elig] != pred[elig]).sum())
        # 교정/파손 분해 (true 기준)
        was_wrong = pred[elig] != yv[elig]
        now_right = bp[elig] == yv[elig]
        fixed = int((was_wrong & now_right).sum())
        broke = int((~was_wrong & (bp[elig] != yv[elig])).sum())
        print(f"    [{tag}] elig={n} changed={changed} → 14F1={f:.5f} Δ={f-base_f1:+.5f} "
              f"| fixed={fixed} broke={broke} net={fixed-broke}")
        return f - base_f1

    argmax_m4 = np.isin(pred, M4)
    elig_final = argmax_m4 & (margin < TH)
    elig_internal = argmax_m4 & sel
    elig_nomargin = argmax_m4
    print(f"\n(b) REAL METRIC — blend specialist on eligible rows (HARD argmax replace):")
    d_primary = blend_delta(elig_final, "PRIMARY margin_final<0.85 & argmax∈M4")
    d_internal = blend_delta(elig_internal, "sens: internal-flag & argmax∈M4")
    d_nomargin = blend_delta(elig_nomargin, "sens: argmax∈M4 (no margin filter)")

    # 소프트 혼합 변형: eligible 행의 M4 질량을 전문가 4-way 분포로 재분배(비M4 열 유지),
    # bias 포함 argmax. 캐스케이드가 비M4로 확신하면 뒤집을 수 있는 정직한 대안.
    def soft_delta(elig, lam, tag):
        Q = P.copy()
        ei = np.where(elig)[0]
        m4mass = Q[ei][:, M4].sum(1, keepdims=True)
        newm4 = (1 - lam) * Q[ei][:, M4] + lam * probs4[ei] * m4mass
        Q[np.ix_(ei, M4)] = newm4
        Q[ei] /= Q[ei].sum(1, keepdims=True)
        bp = L.bias_argmax(Q, old_bias)
        f = L.fast_macro_f1(yv, bp)
        print(f"    [SOFT λ={lam} {tag}] Δ={f-base_f1:+.5f} (changed={int((bp[elig]!=pred[elig]).sum())})")
        return f - base_f1
    print(f"\n(b') SOFT-mixture variants (redistribute M4 mass via specialist, bias-aware):")
    d_soft10 = soft_delta(elig_final, 1.0, "margin<0.85")
    d_soft05 = soft_delta(elig_final, 0.5, "margin<0.85")

    verdict = "PASS" if d_primary >= GATE else "FAIL"
    print(f"\n===== HARD GATE (pre-registered ΔmacroF1 >= +{GATE}) =====")
    print(f"  fold0-val ΔmacroF1 (PRIMARY) = {d_primary:+.5f}  ->  {verdict}")

    # 오라클 상한(전문가가 eligible true-M4 전부 정답이면)
    bp_oracle = pred.copy()
    el = elig_final
    bp_oracle[el] = np.where(np.isin(yv[el], M4), yv[el], spec_pred4[el])
    f_oracle = L.fast_macro_f1(yv, bp_oracle)
    print(f"  [ref] oracle ceiling (eligible true-M4 all correct) Δ = {f_oracle-base_f1:+.5f}")

    out = {"baseline_14f1": round(base_f1, 5), "spec_4way_f1": round(spec_4f1, 5),
           "casc_proj4_f1": round(casc_4f1, 5), "casc_asis_m4f1": round(casc_asis, 5),
           "delta_primary_hard": round(float(d_primary), 5),
           "delta_internal_hard": round(float(d_internal), 5),
           "delta_nomargin_hard": round(float(d_nomargin), 5),
           "delta_soft_lam1.0": round(float(d_soft10), 5),
           "delta_soft_lam0.5": round(float(d_soft05), 5),
           "gate": GATE, "verdict": verdict,
           "oracle_delta": round(float(f_oracle - base_f1), 5),
           "best_epoch": best_ep,
           "best_delta_any": round(float(max(d_primary, d_internal, d_nomargin,
                                             d_soft10, d_soft05)), 5)}
    for p in ("m4_gate.json", "m4_eval.json"):
        with open(os.path.join(ROOT, "work", p), "w") as f:
            json.dump(out, f, indent=2)
    print("\n" + json.dumps(out))


if __name__ == "__main__":
    main()
