#!/usr/bin/env python3
"""M4 상한 대리실험 (angle4) — 재학습 없이 fold0 leak-free 로짓만으로 M4 개선 상한 추정.

측정(전부 fold0-val, leak-free):
 1) 다아키텍처 M4 4-way within-accuracy/F1 → Bayes 수렴밴드 확증.
 2) 오라클 멤버선택기 상한 (행별 정답멤버 존재시 = 완벽 combiner/selector).
 3) 만장일치-오답 비율 (환원불가 코어).
 4) 재보정 상한 (M4 로짓블록 per-class 온도+bias를 eval에 직접 적합 = 부정행위 천장).
    → 기존 로짓기하로 얻을 수 있는 M4 4-way F1 최대치.
 5) 캐스케이드 레벨: eligible 행 M4-argmax를 오라클재보정으로 교체시 14F1 Δ.
"""
import os, sys, json, itertools
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


def within_f1(y_true14, probs14, mask_rows):
    """true-M4 행에서 M4 사영 4-way argmax의 within-accuracy & macro-F1."""
    proj = probs14[:, M4].argmax(1)
    acc = float((proj[mask_rows] == y_true14[mask_rows]).mean())
    f1 = L.fast_macro_f1(y_true14[mask_rows], proj[mask_rows], n_classes=4)
    return acc, f1, proj


def main():
    ids, y, folds, fmap, members, old_bias = load_all(_A())
    va0 = folds[0][1]
    yv = y[va0]
    m4t = np.isin(yv, M4)          # true-M4 val 마스크
    N4 = int(m4t.sum())
    print(f"[setup] fold0-val N={len(va0)}  true-M4 N={N4}")

    # ---- 추가 아키텍처 로드 (fold0 leak-free OOF) ----
    def loadoof(fn):
        return np.load(os.path.join(ROOT, "work", fn), allow_pickle=True)["oof"][va0]
    arch = {
        "m1_v6_12ep": members[0].oof[va0],
        "mdeb_12ep":  members[1].oof[va0],
        "klue":       members[2].oof[va0],
        "m1_rescue_8ep": loadoof("m1_f0ckpt_rescue.npz"),
        "infoxlm":    loadoof("teacher_infoxlm_f0.npz"),
        "kfdeb":      loadoof("kfdeb_f0.npz"),
    }
    # specialist (xlm-r focal 4-way)
    sd = np.load(os.path.join(ROOT, "work", "m4_spec_f0val.npz"), allow_pickle=True)
    assert np.array_equal(np.asarray(sd["rows"]), np.asarray(va0))
    spec_probs4 = sd["probs4"]     # (N,4)

    # ---- (1) 다아키텍처 within-M4 밴드 ----
    print("\n=== (1) 다아키텍처 M4 4-way within (true-M4 val, N=%d) ===" % N4)
    projs = {}
    for name, P in arch.items():
        acc, f1, proj = within_f1(yv, P, m4t)
        projs[name] = proj
        print(f"  {name:16s} within-acc={acc:.4f}  4way-F1={f1:.4f}")
    sp_acc = float((spec_probs4.argmax(1)[m4t] == yv[m4t]).mean())
    sp_f1 = L.fast_macro_f1(yv[m4t], spec_probs4.argmax(1)[m4t], n_classes=4)
    projs["specialist"] = spec_probs4.argmax(1)
    print(f"  {'specialist':16s} within-acc={sp_acc:.4f}  4way-F1={sp_f1:.4f}  (전용 xlm-r focal 4way)")
    # cascade projection
    P_casc, sel = L.cascade_probs([members[0].oof[va0], members[1].oof[va0],
                                   members[2].oof[va0]], W, TH)
    cacc, cf1, cproj = within_f1(yv, P_casc, m4t)
    print(f"  {'CASCADE(proj)':16s} within-acc={cacc:.4f}  4way-F1={cf1:.4f}  (배포 3멤버 혼합)")
    accs = [within_f1(yv, P, m4t)[0] for P in arch.values()] + [sp_acc, cacc]
    print(f"  --> within-acc 밴드 [{min(accs):.4f}, {max(accs):.4f}] (span {max(accs)-min(accs):.4f})")

    # ---- (2) 오라클 멤버선택기 상한 (완벽 combiner/selector) ----
    # 각 true-M4 행에서 아키텍처(주요 3+2+spec) 중 하나라도 맞으면 정답 처리
    core_members = ["m1_v6_12ep", "mdeb_12ep", "klue"]
    all_members = list(arch.keys()) + ["specialist"]
    def oracle_any(names):
        preds = np.stack([projs[n] for n in names], 0)  # (M, N)
        correct = (preds == yv[None, :]) & m4t[None, :]
        any_right = correct.any(0) & m4t
        # F1 상한: 정답가능 행은 정답, 나머지는 최빈오답 유지(보수적으로 최고멤버 예측)
        best_single = max(names, key=lambda n: (projs[n][m4t] == yv[m4t]).mean())
        oracle_pred = projs[best_single].copy()
        oracle_pred[any_right] = yv[any_right]
        acc = float((oracle_pred[m4t] == yv[m4t]).mean())
        f1 = L.fast_macro_f1(yv[m4t], oracle_pred[m4t], n_classes=4)
        return acc, f1, float(any_right[m4t].mean())
    a3, f3, cov3 = oracle_any(core_members)
    aA, fA, covA = oracle_any(all_members)
    print("\n=== (2) 오라클 멤버선택기 상한 (행별 정답멤버 존재→정답) ===")
    print(f"  배포3멤버(m1/mdeb/klue): within-acc상한={a3:.4f}  4way-F1상한={f3:.4f}  (coverage={cov3:.4f})")
    print(f"  7아키텍처 전부:          within-acc상한={aA:.4f}  4way-F1상한={fA:.4f}  (coverage={covA:.4f})")

    # ---- (3) 만장일치-오답 (환원불가 코어) ----
    core_preds = np.stack([projs[n] for n in core_members], 0)  # (3,N)
    unanimous = (core_preds == core_preds[0][None, :]).all(0)
    unan_wrong = unanimous & (core_preds[0] != yv) & m4t
    unan_right = unanimous & (core_preds[0] == yv) & m4t
    print("\n=== (3) 배포3멤버 만장일치 구조 (true-M4) ===")
    print(f"  만장일치 비율      = {unanimous[m4t].mean():.4f}")
    print(f"  만장일치&오답(환원불가) = {unan_wrong.sum()}/{N4} = {unan_wrong.sum()/N4:.4f}")
    print(f"  만장일치&정답       = {unan_right.sum()}/{N4} = {unan_right.sum()/N4:.4f}")
    disagree = (~unanimous) & m4t
    # 불일치 행에서 최소1개 정답 비율
    dis_any = ((core_preds == yv[None, :]).any(0) & disagree)
    print(f"  불일치 비율        = {disagree.sum()/N4:.4f}  그중 최소1멤버정답 = {dis_any.sum()}/{max(disagree.sum(),1)} = {dis_any.sum()/max(disagree.sum(),1):.4f}")

    # ---- (4) 재보정 상한 (M4 블록 per-class bias를 eval에 직접 적합) ----
    # 캐스케이드 P의 M4 블록 log확률에 per-class 가산bias b(4개)를 부여, true-M4 4-way F1 최대화.
    # eval 라벨로 직접 최적화 = 부정행위 천장. 좌표하강.
    logM4 = np.log(P_casc[m4t][:, M4] + 1e-9)   # (N4,4)
    ytr = yv[m4t]
    def f1_with_bias(b):
        pr = (logM4 + b).argmax(1)
        return L.fast_macro_f1(ytr, pr, n_classes=4)
    b = np.zeros(4)
    base4 = f1_with_bias(b)
    grid = np.linspace(-3, 3, 61)
    for _ in range(6):
        for c in range(4):
            best_v, best_f = b[c], f1_with_bias(b)
            for v in grid:
                b[c] = v
                f = f1_with_bias(b)
                if f > best_f:
                    best_f, best_v = f, v
            b[c] = best_v
    recal4 = f1_with_bias(b)
    print("\n=== (4) 재보정 상한 (M4 per-class bias, eval에 직접적합=천장) ===")
    print(f"  캐스케이드 M4 4way-F1: base={base4:.4f} -> 오라클재보정={recal4:.4f}  (Δ={recal4-base4:+.4f})")
    print(f"  최적 bias={np.round(b,2).tolist()}")

    # ---- (5) 캐스케이드 14F1 Δ: eligible 행에 오라클재보정 M4-argmax 적용 ----
    pred = L.bias_argmax(P_casc, old_bias)
    base14 = L.fast_macro_f1(yv, pred)
    srt = np.sort(P_casc, axis=1); margin = srt[:, -1] - srt[:, -2]
    elig = np.isin(pred, M4) & (margin < TH)
    # 재보정 argmax (전 val행에 동일 bias 적용, M4 블록만)
    recal_argmax_full = (np.log(P_casc[:, M4] + 1e-9) + b).argmax(1)
    bp = pred.copy()
    bp[elig] = recal_argmax_full[elig]
    recal14 = L.fast_macro_f1(yv, bp)
    print("\n=== (5) 캐스케이드 14F1: eligible 행 오라클재보정 교체 ===")
    print(f"  base14={base14:.5f} -> recal14={recal14:.5f}  Δ={recal14-base14:+.5f}")
    print(f"  (참고) 완벽M4분리 오라클천장 Δ=+0.10208 [m4_eval]")

    out = {
        "N_trueM4": N4, "base14": round(base14, 5),
        "within_band": [round(min(accs), 4), round(max(accs), 4)],
        "cascade_within_f1": round(cf1, 4), "specialist_within_f1": round(sp_f1, 4),
        "oracle_selector_3mem_f1": round(f3, 4), "oracle_selector_3mem_cov": round(cov3, 4),
        "oracle_selector_7arch_f1": round(fA, 4), "oracle_selector_7arch_cov": round(covA, 4),
        "unanimous_frac": round(float(unanimous[m4t].mean()), 4),
        "unanimous_wrong_frac": round(float(unan_wrong.sum()/N4), 4),
        "recal_4way_base": round(base4, 4), "recal_4way_ceiling": round(recal4, 4),
        "recal_cascade14_delta": round(float(recal14-base14), 5),
        "oracle_cascade14_delta": 0.10208,
    }
    with open(os.path.join(ROOT, "work", "m4_ceiling_proxy.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + json.dumps(out))


if __name__ == "__main__":
    main()
