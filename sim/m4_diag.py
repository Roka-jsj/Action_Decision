#!/usr/bin/env python3
"""M4 사전분석(진단) — 학습 전 캐스케이드 베이스라인/블렌드 대상행 특성화.

배포 캐스케이드(W=(0.45,0.40,0.15), TH=0.85, old_bias)를 fold0-val 멤버 OOF로 복원하고
- 14클래스 macro-F1 (베이스라인)
- true-M4 val 행에서의 M4-restricted F1
- 블렌드 대상행(cascade argmax∈M4 & margin<0.85) 규모/true 분포
를 출력한다. GPU 불필요.
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sim import refit_lib as L
from sim.refit_d4 import load_all
from common.io_utils import CLASSES

M4 = [0, 1, 2, 3]
W, TH = (0.45, 0.40, 0.15), 0.85


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
    ids, y, folds, fmap, members, old_bias = load_all(_A())
    va0 = folds[0][1]
    yv = y[va0]
    pm1 = members[0].oof[va0]
    pm2 = members[1].oof[va0]
    pm3 = members[2].oof[va0]
    P, sel = L.cascade_probs([pm1, pm2, pm3], W, TH)   # sel = 내부 게이트(p_full margin<TH)
    pred = L.bias_argmax(P, old_bias)

    base_f1 = L.fast_macro_f1(yv, pred)
    print(f"[fold0-val] N={len(va0)}  baseline 14-class macro-F1 = {base_f1:.5f}")

    # 최종 캐스케이드 확률 margin
    srt = np.sort(P, axis=1)
    margin_final = srt[:, -1] - srt[:, -2]

    m4_true = np.isin(yv, M4)
    print(f"[fold0-val] true-M4 rows = {int(m4_true.sum())}")

    # (a) M4-restricted F1: true-M4 행 한정, 캐스케이드가 M4 4-way를 얼마나 맞추나
    pcf = per_class_f1(yv[m4_true], pred[m4_true], M4)
    print("[cascade] per-class F1 on true-M4 rows (14-way pred, restricted eval):")
    for c in M4:
        print(f"    {CLASSES[c]:16s} F1={pcf[c]:.4f}")
    print(f"    mean M4 F1 = {np.mean([pcf[c] for c in M4]):.4f}")

    # 블렌드 대상행: cascade argmax∈M4 & margin<0.85 (최종 확률 margin 기준)
    argmax_m4 = np.isin(pred, M4)
    elig_final = argmax_m4 & (margin_final < TH)
    elig_internal = argmax_m4 & sel   # 캐스케이드 내부 flag 기준
    print(f"\n[blend-eligible] argmax∈M4 & margin_final<{TH}: {int(elig_final.sum())} rows")
    print(f"[blend-eligible] argmax∈M4 & internal-flag(sel): {int(elig_internal.sum())} rows")
    print(f"[blend-eligible] argmax∈M4 (no margin filter): {int(argmax_m4.sum())} rows")

    for nm, elig in [("margin_final", elig_final), ("internal_sel", elig_internal)]:
        et = yv[elig]
        n_true_m4 = int(np.isin(et, M4).sum())
        print(f"  [{nm}] eligible={int(elig.sum())}  true∈M4={n_true_m4} "
              f"({100*n_true_m4/max(int(elig.sum()),1):.1f}%)  true∉M4={int(elig.sum())-n_true_m4}")
        # 이 중 캐스케이드가 M4-4way를 틀린 행(true∈M4지만 argmax != true)
        wrong_within = int(((et != pred[elig]) & np.isin(et, M4)).sum())
        print(f"         그중 true∈M4 & cascade가 4-way 오분류 = {wrong_within} (specialist 교정 잠재대상)")

    # 참고: 전체 M4-argmax 행에서 cascade의 within-M4 정확도
    tm = argmax_m4 & m4_true
    print(f"\n[ref] argmax∈M4 & true∈M4 = {int(tm.sum())}, "
          f"그중 정답 = {int((pred[tm]==yv[tm]).sum())} "
          f"(within-M4 acc={100*(pred[tm]==yv[tm]).sum()/max(int(tm.sum()),1):.1f}%)")


if __name__ == "__main__":
    main()
