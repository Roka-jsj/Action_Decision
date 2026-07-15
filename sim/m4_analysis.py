#!/usr/bin/env python3
"""M4 aleatoric read — 신호가 history에 있는가? (leak-free, CPU)

- 전문가 within-M4 정확도를 history 유무/턴수로 층화 → history가 판별력을 주는지.
- 4-way 혼동행렬(어느 쌍이 섞이나).
- 블렌드 대상 오류행 표본을 직렬화 텍스트째로 덤프(수기 aleatoric 판독용).
"""
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sim import refit_lib as L
from sim.refit_d4 import load_all
from common.io_utils import load_train, CLASSES
from common import ad_lib

M4 = [0, 1, 2, 3]
W, TH = (0.45, 0.40, 0.15), 0.85


class _A:
    m1 = mdeb = klue = ""


def main():
    d = np.load(os.path.join(ROOT, "work", "m4_spec_f0val.npz"), allow_pickle=True)
    va0 = d["rows"]; yv = d["y_true"]; probs4 = d["probs4"]
    spec = probs4.argmax(1)
    samples, y, ids = load_train()

    ids2, y2, folds, fmap, members, old_bias = load_all(_A())
    P, sel = L.cascade_probs([members[0].oof[va0], members[1].oof[va0],
                              members[2].oof[va0]], W, TH)
    pred = L.bias_argmax(P, old_bias)
    srt = np.sort(P, axis=1); margin = srt[:, -1] - srt[:, -2]

    m4t = np.isin(yv, M4)
    # history 특성
    n_hist = np.array([len(samples[int(r)].get("history") or []) for r in va0])
    has_hist = n_hist > 0

    # 전문가 within-M4 정확도 (true-M4 행)
    corr = (spec == yv)
    print("=== 전문가 within-M4 accuracy (true-M4 val 행) by history ===")
    for lbl, msk in [("no history (n_hist=0)", m4t & ~has_hist),
                     ("has history (>=1)", m4t & has_hist),
                     ("hist>=4", m4t & (n_hist >= 4)),
                     ("hist>=8", m4t & (n_hist >= 8))]:
        n = int(msk.sum())
        acc = float(corr[msk].mean()) if n else 0.0
        # 같은 슬라이스 캐스케이드 within-M4 acc (M4 사영)
        cp = P[:, M4].argmax(1)
        cacc = float((cp[msk] == yv[msk]).mean()) if n else 0.0
        print(f"  {lbl:24s} n={n:5d}  spec_acc={acc:.3f}  casc_proj_acc={cacc:.3f}  Δ={acc-cacc:+.3f}")

    # 4-way 혼동행렬 (전문가, true-M4)
    print("\n=== 전문가 4-way confusion (rows=true, cols=pred) true-M4 val ===")
    print("            " + " ".join(f"{CLASSES[c][:8]:>9s}" for c in M4))
    for rt in M4:
        row = [int(((yv == rt) & (spec == ct)).sum()) for ct in M4]
        print(f"  {CLASSES[rt][:10]:10s} " + " ".join(f"{v:9d}" for v in row))

    # 블렌드 대상 오류행 표본
    elig = np.isin(pred, M4) & (margin < TH)
    err = elig & m4t & (spec != yv)
    err_idx = np.where(err)[0]
    print(f"\n=== 블렌드 eligible & true-M4 & 전문가 오류: {len(err_idx)}행 ===")
    n_err_nohist = int((~has_hist[err_idx]).sum())
    print(f"  그중 history 없음(n_hist=0): {n_err_nohist} ({100*n_err_nohist/max(len(err_idx),1):.1f}%)")
    print(f"  전체 true-M4 val 중 history 없음: "
          f"{int((~has_hist & m4t).sum())}/{int(m4t.sum())} "
          f"({100*(~has_hist & m4t).sum()/int(m4t.sum()):.1f}%)")

    rng = np.random.RandomState(0)
    samp = rng.choice(err_idx, size=min(20, len(err_idx)), replace=False)
    dump = []
    for i in samp:
        r = int(va0[i])
        txt = ad_lib.serialize(samples[r], "v6", 8)
        rec = {"row": r, "true": CLASSES[yv[i]], "spec_pred": CLASSES[spec[i]],
               "casc_pred": CLASSES[pred[i]], "n_hist": int(n_hist[i]),
               "spec_probs": {CLASSES[c]: round(float(probs4[i][c]), 3) for c in M4},
               "text": txt[:1200]}
        dump.append(rec)
    with open(os.path.join(ROOT, "work", "m4_err_samples.json"), "w") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False)
    print(f"\n[dump] work/m4_err_samples.json ({len(dump)} 표본)")
    # 콘솔에 5개 요약
    for rec in dump[:8]:
        print(f"\n  row={rec['row']} true={rec['true']} spec={rec['spec_pred']} "
              f"casc={rec['casc_pred']} nhist={rec['n_hist']} probs={rec['spec_probs']}")
        print(f"    {rec['text'][:400]}")


if __name__ == "__main__":
    main()
