#!/usr/bin/env python3
"""mdeb@384 부호 프로브 (전략가 #9, R57 승격) — fold0-val, mdeb FULL 멤버.

mdeb 토크나이저 기준 320절단 행을 max_len 320 vs 384 두 패스(둘 다 gen_rescue 가드,
배포 정합) 재추론 — 절단행 슬라이스 acc·전체 ΔF1·캐스케이드 부호.
캐비앗: FULL 은 320절단형을 암기(fold0-val 이 훈련집합) → 편향이 384에 불리.
그럼에도 양수면 강한 GO (전략가 게이트). GPU 소배치(농사 병행).
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import refit_lib as L  # noqa: E402
from common import ad_lib, postproc  # noqa: E402
from common.io_utils import load_train  # noqa: E402

ROOT = L.ROOT
MDEB = os.path.join(ROOT, "work", "cc_members", "mdeb")


def main():
    samples, y_all, _ = load_train()
    y_all = np.asarray(y_all)
    folds, _, _ = L.load_splits()
    va0 = np.sort(folds[0][1])
    sub = [samples[i] for i in va0]
    yb = y_all[va0]
    texts = [ad_lib.serialize(s, "v6") for s in sub]
    is_sim = np.array([str(s["id"]).startswith("sess_sim") for s in sub])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MDEB, local_files_only=True)
    tok.truncation_side = "left"
    nsp = tok.num_special_tokens_to_add(False)
    lens = np.array([len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]])
    tr320 = lens + nsp > 320
    tr384 = lens + nsp > 384
    gd = np.zeros(len(sub), bool)
    gd[sorted(ad_lib._gen_rescue_ids(tok, texts, 320).keys())] = True
    print(f"[mdeb384] fold0-val {len(sub)}행 | mdeb 절단@320 {tr320.sum()}({tr320.mean()*100:.1f}%) "
          f"| 절단@384 {tr384.sum()}({tr384.mean()*100:.1f}%) | GEN삭제@320 {gd.sum()}")

    # 전체 @320 기준 패스 + 절단행만 @384 패스 (둘 다 gen_rescue — 배포 정합)
    t0 = time.time()
    P320 = ad_lib.predict_logits(MDEB, sub, version="v6", max_len=320, batch_size=16,
                                 texts=texts, return_probs=True, gen_rescue=True)
    t_full = time.time() - t0
    rows_tr = np.where(tr320)[0]
    t0 = time.time()
    P384_tr = ad_lib.predict_logits(MDEB, [sub[i] for i in rows_tr], version="v6",
                                    max_len=384, batch_size=16,
                                    texts=[texts[i] for i in rows_tr],
                                    return_probs=True, gen_rescue=True)
    t_384 = time.time() - t0
    P384 = P320.copy()
    P384[rows_tr] = P384_tr
    print(f"  시간: full@320 {t_full:.0f}s / 절단행@384 {t_384:.0f}s "
          f"(배율 실측 {1 + t_384/max(t_full,1e-9)*0:.3f} — 아래 서빙 추정 참조)")

    def acc(P, m):
        return (float((P.argmax(1)[m] == yb[m]).mean()), int(m.sum())) if m.sum() else (float("nan"), 0)

    print(f"  {'슬라이스':<26} {'n':>6} {'@320':>8} {'@384':>8} {'Δ':>9}")
    rows = [("절단@320 전체", tr320), ("절단@320 ∩ sim", tr320 & is_sim),
            ("절단@320 ∩ au", tr320 & ~is_sim), ("GEN삭제@320", gd),
            ("절단@320 ∩ ~절단@384", tr320 & ~tr384), ("절단@384(양쪽 절단)", tr384),
            ("비절단(불변 확인)", ~tr320), ("전체", np.ones(len(sub), bool))]
    lines = []
    for name, m in rows:
        a0, n = acc(P320, m)
        a1, _ = acc(P384, m)
        r = f"{name:<26} {n:>6} {a0:>8.4f} {a1:>8.4f} {a1-a0:>+9.4f}"
        print("  " + r)
        lines.append(r)
    f0 = L.fast_macro_f1(yb, P320.argmax(1))
    f1_ = L.fast_macro_f1(yb, P384.argmax(1))
    print(f"  mdeb 멤버 단독 macro-F1: {f0:.5f} → {f1_:.5f} (Δ{f1_-f0:+.5f})")

    # 캐스케이드 부호 (m1-rescue + mdeb + klue, th75+구bias)
    OLD = postproc.load(L.OLD_BIAS_PATH)
    M1r = np.load(os.path.join(ROOT, "work", "m1_f0ckpt_rescue.npz"), allow_pickle=True)["oof"][va0]
    K = np.load(os.path.join(ROOT, "work", "klue_f0.npz"), allow_pickle=True)["oof"][va0]
    Pa, selm = L.cascade_probs([M1r, P320, K], L.RESCUE_ANCHOR_W, L.RESCUE_ANCHOR_TH)
    Pb, _ = L.cascade_probs([M1r, P384, K], L.RESCUE_ANCHOR_W, L.RESCUE_ANCHOR_TH)
    fa = L.score(Pa, OLD, yb)
    fb = L.score(Pb, OLD, yb)
    print(f"  캐스케이드 th75(m1-rescue/mdeb/klue): {fa:.5f} → {fb:.5f} (Δ{fb-fa:+.5f}) "
          f"[게이트커버리지 {selm.mean():.3f}]")
    lines.append(f"멤버 ΔF1 {f1_-f0:+.5f} / 캐스케이드 Δ {fb-fa:+.5f}")

    # 서빙 시간: mdeb 는 cond 패스(게이트행만) — 전략가 실측 배율 1.049 적용
    verdict = "GO(강한 — FULL 암기편향에도 양수)" if (f1_ - f0) > 0 and (fb - fa) >= 0 else \
              ("중립/혼합 — 3자 판독" if (fb - fa) >= 0 or (f1_ - f0) > 0 else "축 폐쇄(음수)")
    print(f"  => 부호 판정: 멤버 {f1_-f0:+.5f} / 캐스케이드 {fb-fa:+.5f} → {verdict}")
    with open(os.path.join(ROOT, "work", "mdeb384_probe.md"), "w", encoding="utf-8") as f:
        f.write("# mdeb@384 부호 프로브 (fold0-val, FULL 멤버 — 암기편향 384 불리 캐비앗)\n\n")
        for ln in lines:
            f.write(f"- {ln}\n")
        f.write(f"- 판정: {verdict}\n")
    print("[report] work/mdeb384_probe.md")


if __name__ == "__main__":
    main()
