#!/usr/bin/env python3.11
"""정직한 sim-only OOF 평가 — 누수 없는 로컬 나침반.

- bundle/teacher_*.npz(5-fold OOF 확률)를 이어붙여 large-v6 / base-v6 / large-v4 OOF 재구성.
- dev(홀드아웃 제외) ∩ sim 에서 pooled macro-F1 측정.
- per-class floor/threshold 레버 = postproc.fit_bias 를 sim-only OOF 로 적합 → macro 이득 + hard-0 클래스 점검.
- au 포함/제외 효과, 강-강 앙상블(log-prob 평균)까지 한 파일로.

실행: PYTHONPATH=/workspace python3.11 eda/honest_oof_eval.py
"""
import sys, os, glob, re
sys.path.insert(0, "/workspace")
import numpy as np
from common.io_utils import load_train, CLASSES, NUM_CLASSES
from common.metrics import macro_f1, per_class_report
from common import postproc

BUNDLE = "/workspace/bundle"


def stitch(files):
    """여러 fold 조각 npz의 oof(확률)를 disjoint 행에서 합쳐 전체 OOF 복원."""
    oof = np.zeros((70000, NUM_CLASSES), np.float32)
    for f in files:
        z = np.load(f, allow_pickle=True)
        o = z["oof"]
        m = o.sum(1) > 0
        oof[m] = o[m]
    return oof


def load_members():
    lv6 = stitch([
        f"{BUNDLE}/teacher_largev6A_a2.npz", f"{BUNDLE}/teacher_largev6A_a3.npz",
        f"{BUNDLE}/teacher_largev6A_a4.npz", f"{BUNDLE}/teacher_largev6B_a1.npz",
        f"{BUNDLE}/teacher_largev6B_a2.npz",
    ])
    bv6 = stitch([f"{BUNDLE}/teacher_basev6e5_g0.npz"])
    lv4 = stitch([f"{BUNDLE}/teacher_largev4mix.npz"])
    return {"large-v6": lv6, "base-v6": bv6, "large-v4": lv4}


def to_logprob(p):
    p = np.clip(np.asarray(p, np.float64), 1e-9, None)
    return np.log(p) - np.log(p.sum(1, keepdims=True))


def evaluate(name, logp, y, mask):
    ym = y[mask]
    base = macro_f1(ym, logp[mask].argmax(1))[0]
    bias, fit = postproc.fit_bias(logp[mask], ym)
    tuned = macro_f1(ym, (logp[mask] + bias).argmax(1))[0]
    # hard-0 클래스: 튜닝 후 한 번도 예측되지 않는 클래스
    pred = (logp[mask] + bias).argmax(1)
    fired = set(np.unique(pred).tolist())
    hard0 = [CLASSES[c] for c in range(NUM_CLASSES) if c not in fired]
    print(f"\n[{name}]  n={mask.sum()}  base={base:.4f}  +bias={tuned:.4f}  (Δ={tuned-base:+.4f})")
    if hard0:
        print(f"    ⚠ hard-0 (never predicted): {hard0}")
    return base, tuned, bias


def main():
    samples, y, ids = load_train()
    y = np.array(y)
    gen = np.array(["au" if i.startswith("sess_au_") else "sim" for i in ids])
    mem = load_members()

    # dev = OOF가 채워진 행(홀드아웃 제외). 멤버별 커버리지 교집합.
    covered = np.ones(70000, bool)
    for p in mem.values():
        covered &= (p.sum(1) > 0)
    sim = gen == "sim"
    dev_all = covered
    dev_sim = covered & sim
    print(f"coverage: dev_all={dev_all.sum()}  dev_sim={dev_sim.sum()}  (au in dev={ (covered & (gen=='au')).sum() })")

    lp = {k: to_logprob(v) for k, v in mem.items()}

    print("\n================ 단일 멤버 (정직 OOF) ================")
    for name in ["large-v6", "base-v6", "large-v4"]:
        print(f"\n---- {name} : ALL-dev vs SIM-dev ----")
        evaluate(f"{name} ALL", lp[name], y, dev_all)
        b, t, bias = evaluate(f"{name} SIM", lp[name], y, dev_sim)
        if name == "large-v6":
            print("\n  per-class F1 (large-v6 SIM, +bias):")
            pred = (lp[name][dev_sim] + bias).argmax(1)
            for cname, sup, pr, rc, f1 in sorted(per_class_report(y[dev_sim], pred), key=lambda x: x[4]):
                print(f"    {cname:18s} sup={sup:5d} P={pr:.3f} R={rc:.3f} F1={f1:.4f}")

    print("\n================ 강-강 앙상블 (log-prob 평균, SIM-dev) ================")
    combos = [
        ("large-v6 + base-v6", ["large-v6", "base-v6"]),
        ("large-v6 + large-v4", ["large-v6", "large-v4"]),
        ("large-v6 + large-v4 + base-v6", ["large-v6", "large-v4", "base-v6"]),
    ]
    for label, keys in combos:
        avg = sum(lp[k] for k in keys) / len(keys)
        evaluate(label, avg, y, dev_sim)

    # 가중 평균 스윕 (large-v6 우세 조합)
    print("\n---- large-v6 ⊕ base-v6 가중 스윕 (SIM-dev) ----")
    for w in [0.6, 0.7, 0.8, 0.9]:
        avg = w * lp["large-v6"] + (1 - w) * lp["base-v6"]
        yy = y[dev_sim]
        base = macro_f1(yy, avg[dev_sim].argmax(1))[0]
        bias, _ = postproc.fit_bias(avg[dev_sim], yy)
        tuned = macro_f1(yy, (avg[dev_sim] + bias).argmax(1))[0]
        print(f"    w(large-v6)={w:.1f}  base={base:.4f}  +bias={tuned:.4f}")


if __name__ == "__main__":
    main()
