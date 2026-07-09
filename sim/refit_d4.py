#!/usr/bin/env python3
"""refit_d4 — D-4(7/11) 클린 앙상블 OOF 재적합 파이프라인 (D12 하네스 정식화·5폴드 확장판).

모드:
  inventory  멤버 npz 인벤토리·폴드 커버리지·usable folds (재료 도착 시 자동 확장 확인)
  probes     fold0 known-probe 회귀테스트 (wk30 −/th55 +/wd30 +/cw45 +0.002대)
  fold0      레드팀 D12 실측 재현 (half-fit/half-eval seed7: wth +0.0022, joint +0.0030)
  refit      정식 재적합: half-fit/half-eval + 폴드 LOFO 부호표 + λ·온도 레이어 + 게이트
  smoke25    다폴드 기계 검증(2.5폴드): klue f0+f14 × m1프록시(v6f0+v9f14) 2멤버 — 판정 무효
  all        위 전부 실행 + work/refit_report.md 기록

전부 CPU. 제출 패키지 생성 없음(재적합 실행은 내일 3자 검증 후).
사용 예:
  python3 sim/refit_d4.py --mode all
  python3 sim/refit_d4.py --mode refit --mdeb work/mdeb12ep_f0.npz,work/mdeb12ep_f14.npz
"""
from __future__ import annotations
import argparse
import datetime
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import refit_lib as L  # noqa: E402
from common import postproc  # noqa: E402

ROOT = L.ROOT

# 기본 재료 (존재하는 파일만 로드 — 내일 mdeb f1-4 / v6-12ep f1-4 도착 시 자동 확장)
DEFAULT_M1 = ["work/teacher_largev6_12ep_f0.npz",
              "work/teacher_largev6_12ep_f14.npz"]           # (내일 예정)
DEFAULT_MDEB = ["work/mdeb12ep_f0.npz",                      # 조원 mdeb-12ep fold0 (docker cp, 읽기전용)
                "work/mdeb12ep_f14.npz",                     # (내일 낮 예정 — 도착 시 이 이름으로 저장)
                "work/teacher_cc_mdeberta12_f14.npz"]        # (대안 파일명)
DEFAULT_KLUE = ["work/klue_f0.npz",                          # 조원 fold0
                "work/teacher_klue_f14.npz"]                 # 우리 folds1-4 (농사 증분)

# LB 실측 참조(부호 검증용 각주 — 시뮬 게이트에는 미사용)
# wk30/th55/wd30 은 vs C0(0.78567), cw45 는 vs wd30(0.78621) 델타 — 07-10 판독 0.78655
# (+0.00034, 문턱미달·max-public 신기록, 시뮬 +0.00256 대비 실현전이 0.13 → R50 회부)
LB_ACTUALS = {"wk30": -0.0001, "th55": +0.00015, "wd30": +0.00054, "cw45": +0.00034}

# smoke25 전용(사전등록 아님 — 기계 검증용, 판정 무효)
SMOKE_W_GRID = ((0.70, 0.30), (0.60, 0.40), (0.80, 0.20), (0.75, 0.25))


def _git_rev():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


def load_all(args):
    ids, y, groups, ids_hash = L.load_ids_labels()
    folds, dev, hold = L.load_splits()
    fmap = L.fold_of_rows(folds)

    def paths(cli, default):
        raw = cli.split(",") if cli else default
        out = []
        for p in raw:
            p = p if os.path.isabs(p) else os.path.join(ROOT, p)
            if os.path.exists(p):
                out.append(p)
        return out

    members = [
        L.MemberOOF("m1(xlm-r-large v6-12ep)", paths(args.m1, DEFAULT_M1), folds, y),
        L.MemberOOF("mdeb(mdeberta-12ep)", paths(args.mdeb, DEFAULT_MDEB), folds, y),
        L.MemberOOF("klue(roberta-large)", paths(args.klue, DEFAULT_KLUE), folds, y),
    ]
    old_bias = postproc.load(L.OLD_BIAS_PATH)
    import hashlib
    import json as _json
    bh = hashlib.sha256(_json.dumps(list(old_bias)).encode()).hexdigest()[:16]
    assert bh == L.OLD_BIAS_SHA16, f"구bias 해시 불일치: {bh}"
    return ids, y, folds, fmap, members, old_bias


def print_inventory(y, folds, members, old_bias):
    print("=" * 72)
    print("[inventory] 사전등록 grid_hash =", L.grid_hash(),
          "| ids", L.IDS_SHA16, "| splits", L.SPLITS_SHA256[:16], "| bias", L.OLD_BIAS_SHA16)
    for m in members:
        s = m.summary()
        print(f"  member {s['name']}: folds={s['folds']}" + (" [PROXY]" if s["proxy"] else ""))
        for fi in s["files"]:
            print(f"    - {fi['path']} folds={fi['folds']} rows={fi['rows']} "
                  f"scores={fi['scores']} oof_f1={fi['oof_f1']}")
    uf = L.usable_folds(members, folds)
    print(f"  usable folds (전멤버 커버): {uf} / 5  — 재료 추가 시 자동 확장")
    # 폴드별 클래스 0셀 감시
    for fi, (_, va) in enumerate(folds):
        cnt = np.bincount(y[va], minlength=L.NUM_CLASSES)
        if (cnt == 0).any():
            print(f"  [warn] fold{fi} 클래스 0셀: {np.where(cnt == 0)[0]}")
    return uf


def slice_rows(members, y, fmap, folds, use_folds):
    rows = np.sort(np.concatenate([folds[f][1] for f in use_folds]))
    for m in members:
        assert m.cov_mask[rows].all(), f"{m.name} 커버 누락"
    mems = [m.oof[rows] for m in members]
    return rows, mems, y[rows], fmap[rows]


# ---------------------------------------------------------------------------
# probes — known-probe 회귀 (fold0)
# ---------------------------------------------------------------------------
def run_probes(y, folds, fmap, members, old_bias, eps=L.EPS_DEPLOY):
    print("=" * 72)
    print(f"[probes] fold0 known-probe 회귀 (eps={eps:g}, 기준=C0 {L.C0_BASE_W} th{L.C0_BASE_TH})")
    uf = L.usable_folds(members, folds)
    assert 0 in uf, "fold0 재료 불완전"
    rows, mems, yb, _ = slice_rows(members, y, fmap, folds, [0])

    def F(w, th):
        P, _ = L.cascade_probs(mems, w, th)
        return L.score(P, old_bias, yb, eps)

    f_base = F(L.C0_BASE_W, L.C0_BASE_TH)
    f_wd30 = F(L.ANCHOR_W, L.ANCHOR_TH)
    f_cw45 = F((0.45, 0.35, 0.20), 0.6)
    P_tt30, _ = L.cascade_probs(mems, L.TT30_W, L.TT30_TH, stages=L.TT30_STAGES)
    f_tt30 = L.score(P_tt30, old_bias, yb, eps)
    checks = [
        ("wk30 (0.55/0.15/0.30 th0.5)", F((0.55, 0.15, 0.30), 0.5) - f_base,
         lambda d: d < 0, "음수"),
        ("th55 (0.6/0.15/0.25 th0.55)", F((0.60, 0.15, 0.25), 0.55) - f_base,
         lambda d: d > 0, "양수"),
        ("wd30 (0.55/0.30/0.15 th0.5)", f_wd30 - f_base,
         lambda d: d > 0, "양수"),
        ("cw45 (0.45/0.35/0.20 th0.6) vs wd30", f_cw45 - f_wd30,
         lambda d: 0.0010 <= d <= 0.0035, "+0.002대"),
        ("tt30 (2단 t1=0.3 wd0.40/0.35/0.25) vs cw45", f_tt30 - f_cw45,
         lambda d: 0.0005 <= d <= 0.0015, "+0.00091 인근"),
    ]
    all_ok = True
    lines = [f"C0 base F1={f_base:.5f}, wd30 anchor F1={f_wd30:.5f}"]
    for name, d, ok_fn, expect in checks:
        ok = ok_fn(d)
        all_ok &= ok
        key = name.split(" ")[0]
        lb = LB_ACTUALS.get(key)
        lbs = f" (LB 실측 {lb:+.5f})" if lb is not None else ""
        line = f"{'PASS' if ok else 'FAIL'}  {name}: Δ={d:+.5f} (기대 {expect}){lbs}"
        print("  " + line)
        lines.append(line)
    print(f"  probes: {'ALL PASS' if all_ok else 'FAILURE — 수식 차이 규명 필요'}")
    return all_ok, lines


# ---------------------------------------------------------------------------
# fold0 — 레드팀 D12 실측 재현 (redteam_sim2 프로토콜 포트, seed7)
# ---------------------------------------------------------------------------
def run_fold0_repro(y, folds, fmap, members, old_bias, eps=L.EPS_REDTEAM, seed=7):
    print("=" * 72)
    print(f"[fold0] 레드팀 D12 재현 (seed={seed}, eps={eps:g}, anchor=wd30+구bias)")
    rows, mems, yb, _ = slice_rows(members, y, fmap, folds, [0])
    P_anchor, _ = L.cascade_probs(mems, L.ANCHOR_W, L.ANCHOR_TH)
    f_anchor = L.score(P_anchor, old_bias, yb, eps)
    print(f"  anchor wd30+구bias fold0 clean F1 = {f_anchor:.5f}  (레드팀 실측 0.76680)")

    rs = np.random.RandomState(seed)
    perm = rs.permutation(len(rows))
    hA, hB = perm[:len(perm) // 2], perm[len(perm) // 2:]

    def honest(mode, fit_i, ev_i):
        if mode == "bias_only":
            P, _ = L.cascade_probs(mems, L.ANCHOR_W, L.ANCHOR_TH)
            b, _ = L.fit_bias_cd(P[fit_i], yb[fit_i], old_bias, passes=4, eps=eps)
            sel = (L.ANCHOR_W, L.ANCHOR_TH, b)
        elif mode == "wth_only":
            (w, th), _ = L.search_wth(mems, yb, fit_i, old_bias, eps=eps)
            sel = (w, th, old_bias)
        else:
            (w, th, b), _ = L.search_joint(mems, yb, fit_i, old_bias, bias_passes=2, eps=eps)
            sel = (w, th, b)
        w, th, b = sel
        P, _ = L.cascade_probs(mems, w, th)
        d = L.score(P[ev_i], b, yb[ev_i], eps) - L.score(P_anchor[ev_i], old_bias, yb[ev_i], eps)
        return d, (w, th)

    ref = {"bias_only": +0.00026, "wth_only": +0.00216, "joint": +0.00305}
    tol = {"bias_only": 0.0008, "wth_only": 0.0008, "joint": 0.0010}
    results, all_ok, lines = {}, True, [f"anchor F1={f_anchor:.5f} (ref 0.76680)"]
    for mode in ("bias_only", "wth_only", "joint"):
        d1, s1 = honest(mode, hA, hB)
        d2, s2 = honest(mode, hB, hA)
        avg = (d1 + d2) / 2
        ok = abs(avg - ref[mode]) <= tol[mode]
        all_ok &= ok
        results[mode] = (avg, s1, s2)
        line = (f"[{mode}] honest Δ {d1:+.5f}/{d2:+.5f} avg {avg:+.5f} "
                f"(ref {ref[mode]:+.5f}±{tol[mode]:.4f}) {'PASS' if ok else 'FAIL'} "
                f"| 선택 {s1} / {s2}")
        print("  " + line)
        lines.append(line)
    ok_anchor = abs(f_anchor - 0.76680) < 5e-4
    all_ok &= ok_anchor
    # 교차 half 동일좌표(cw45 유일 안정점) 확인
    st = {results["joint"][1], results["joint"][2]}
    print(f"  joint 교차half 선택좌표: {st} (레드팀: (0.45,0.35,0.2) th0.6 양쪽)")
    print(f"  fold0 재현: {'ALL PASS' if all_ok else 'FAIL — 수식 차이 규명 필요'}")
    return all_ok, lines, results


# ---------------------------------------------------------------------------
# refit — 정식 프로토콜 (half-fit + LOFO 부호표 + λ/온도 + 게이트)
# ---------------------------------------------------------------------------
def gate_judge(fold_deltas, pooled_delta, n_total=5):
    n = len(fold_deltas)
    nonneg = sum(1 for d in fold_deltas if d >= 0)
    worst = min(fold_deltas) if fold_deltas else float("nan")
    ok = (nonneg >= L.GATE_MIN_NONNEG and worst >= L.GATE_WORST_FOLD
          and pooled_delta >= L.GATE_JOINT_DELTA)
    tag = "PASS" if ok else "FAIL"
    if n < n_total:
        tag += f" (INCOMPLETE {n}/{n_total}폴드 — 판정은 5폴드 완성 후 확정)"
    return (f"게이트[비음수 {nonneg}/{n} (기준≥{L.GATE_MIN_NONNEG}/5) | "
            f"최악 {worst:+.5f} (기준≥{L.GATE_WORST_FOLD:+.4f}) | "
            f"조인트 {pooled_delta:+.5f} (기준≥{L.GATE_JOINT_DELTA:+.4f})] → {tag}"), ok


def run_refit(y, folds, fmap, members, old_bias, eps=L.EPS_DEPLOY, seed=42,
              w_grid=L.W_GRID, th_grid=L.TH_GRID, anchor_w=L.ANCHOR_W, anchor_th=L.ANCHOR_TH,
              cond=L.COND_MEMBERS, label="refit", with_temp=True):
    print("=" * 72)
    uf = L.usable_folds(members, folds)
    print(f"[{label}] 정식 재적합 프로토콜 (eps={eps:g}, usable folds={uf}, grid_hash={L.grid_hash()})")
    lines = [f"usable folds: {uf} (전멤버 커버 폴드 — 재료 도착 시 자동 확장)"]
    rows, mems, yb, fb = slice_rows(members, y, fmap, folds, uf)
    P_anchor, selm = L.cascade_probs(mems, anchor_w, anchor_th, cond)
    f_anchor = L.score(P_anchor, old_bias, yb, eps)
    print(f"  anchor {anchor_w} th{anchor_th}+구bias: F1={f_anchor:.5f} "
          f"(rows={len(rows)}, 게이트커버리지={selm.mean():.3f})")
    lines.append(f"anchor F1={f_anchor:.5f} rows={len(rows)} coverage={selm.mean():.3f}")

    def anchor_f(ev_i):
        return L.score(P_anchor[ev_i], old_bias, yb[ev_i], eps)

    # --- (A) half-fit/half-eval ---
    rs = np.random.RandomState(seed)
    perm = rs.permutation(len(rows))
    halves = [(perm[:len(perm) // 2], perm[len(perm) // 2:]),
              (perm[len(perm) // 2:], perm[:len(perm) // 2])]
    print(f"  --- (A) half-fit/half-eval (seed={seed}) ---")
    half_out = {}
    for mode in ("wth_only", "joint"):
        ds, sels, lam_ds = [], [], {lam: [] for lam in L.LAMBDA_GRID}
        temp_ds, temp_sel = [], []
        for fit_i, ev_i in halves:
            if mode == "wth_only":
                (w, th), _ = L.search_wth(mems, yb, fit_i, old_bias, w_grid, th_grid, eps, cond)
                b_fit = old_bias
            else:
                (w, th, b_fit), _ = L.search_joint(mems, yb, fit_i, old_bias, w_grid, th_grid, 2, eps, cond)
            P, _ = L.cascade_probs(mems, w, th, cond)
            ds.append(L.score(P[ev_i], b_fit if mode == "joint" else old_bias, yb[ev_i], eps)
                      - anchor_f(ev_i))
            sels.append((w, th))
            if mode == "joint":
                for lam in L.LAMBDA_GRID:
                    bl = L.shrink_bias(old_bias, b_fit, lam)
                    lam_ds[lam].append(L.score(P[ev_i], bl, yb[ev_i], eps) - anchor_f(ev_i))
                if with_temp and len(mems) >= 2:
                    Ts, _ = L.fit_temps_greedy([m[fit_i] for m in mems], yb[fit_i],
                                               w, th, b_fit, cond, eps=eps)
                    mm = [L.apply_temperature(m, t) for m, t in zip(mems, Ts)]
                    Pt, _ = L.cascade_probs(mm, w, th, cond)
                    bt, _ = L.fit_bias_cd(Pt[fit_i], yb[fit_i], old_bias, passes=2, eps=eps)
                    temp_ds.append(L.score(Pt[ev_i], bt, yb[ev_i], eps) - anchor_f(ev_i))
                    temp_sel.append(tuple(Ts))
        line = f"[{mode}] honest Δ {ds[0]:+.5f}/{ds[1]:+.5f} avg {np.mean(ds):+.5f} | 선택 {sels[0]} / {sels[1]}"
        print("    " + line)
        lines.append("(A) " + line)
        half_out[mode] = (float(np.mean(ds)), sels)
        if mode == "joint":
            for lam in L.LAMBDA_GRID:
                line = f"[joint λ={lam}] honest Δ avg {np.mean(lam_ds[lam]):+.5f} ({lam_ds[lam][0]:+.5f}/{lam_ds[lam][1]:+.5f})"
                print("    " + line)
                lines.append("(A) " + line)
            if temp_ds:
                line = f"[joint+temp] honest Δ avg {np.mean(temp_ds):+.5f} | T선택 {temp_sel}"
                print("    " + line)
                lines.append("(A) " + line)

    # --- (B) leave-one-fold-out 부호표 ---
    print(f"  --- (B) 폴드 LOFO 부호표 ({len(uf)}폴드) ---")
    lofo = {}
    if len(uf) < 2:
        msg = "LOFO 불가(usable fold 1개) — mdeb/v6 folds1-4 도착 시 자동 활성"
        print("    " + msg)
        lines.append("(B) " + msg)
    else:
        cand = {"wth_only": {}, **{f"joint λ={lam}": {} for lam in L.LAMBDA_GRID}}
        pooled_pred = {k: np.zeros(len(rows), dtype=np.int64) for k in cand}
        sel_log = {}
        for f in uf:
            fit_i = np.where(fb != f)[0]
            ev_i = np.where(fb == f)[0]
            (w1, th1), _ = L.search_wth(mems, yb, fit_i, old_bias, w_grid, th_grid, eps, cond)
            P1, _ = L.cascade_probs(mems, w1, th1, cond)
            cand["wth_only"][f] = L.score(P1[ev_i], old_bias, yb[ev_i], eps) - anchor_f(ev_i)
            pooled_pred["wth_only"][ev_i] = L.bias_argmax(P1[ev_i], old_bias, eps)
            (w2, th2, b2), _ = L.search_joint(mems, yb, fit_i, old_bias, w_grid, th_grid, 2, eps, cond)
            P2, _ = L.cascade_probs(mems, w2, th2, cond)
            for lam in L.LAMBDA_GRID:
                bl = L.shrink_bias(old_bias, b2, lam)
                cand[f"joint λ={lam}"][f] = L.score(P2[ev_i], bl, yb[ev_i], eps) - anchor_f(ev_i)
                pooled_pred[f"joint λ={lam}"][ev_i] = L.bias_argmax(P2[ev_i], bl, eps)
            sel_log[f] = {"wth": (w1, th1), "joint": (w2, th2)}
        hdr = "candidate        | " + " | ".join(f"f{f}" for f in uf) + " | pooledΔ | 게이트"
        print("    " + hdr)
        lines.append("(B) " + hdr)
        pooled_anchor = L.score(P_anchor, old_bias, yb, eps)
        for k, per in cand.items():
            fd = [per[f] for f in uf]
            pooled = L.fast_macro_f1(yb, pooled_pred[k]) - pooled_anchor
            verdict, ok = gate_judge(fd, pooled)
            row = (f"{k:16s} | " + " | ".join(f"{d:+.5f}" for d in fd)
                   + f" | {pooled:+.5f} | {verdict}")
            print("    " + row)
            lines.append("(B) " + row)
            lofo[k] = {"folds": dict(zip(uf, fd)), "pooled": pooled, "gate": ok}
        line = "선택좌표(폴드별): " + "; ".join(f"f{f}: wth={v['wth']} joint={v['joint']}"
                                            for f, v in sel_log.items())
        print("    " + line)
        lines.append("(B) " + line)

    # --- (C) 전체 usable 행 in-sample 최종 후보 (내일 3자 검증용 — 낙관치, 발사 판단 금지) ---
    print("  --- (C) in-sample 최종 후보 (낙관 상한 — 발사 판단은 (A)/(B)로) ---")
    all_i = np.arange(len(rows))
    (wj, thj, bj), fj = L.search_joint(mems, yb, all_i, old_bias, w_grid, th_grid, 2, eps, cond)
    line = f"joint in-sample: w={wj} th={thj} F1={fj:.5f} (Δ{fj - f_anchor:+.5f})"
    print("    " + line)
    lines.append("(C) " + line)
    for lam in L.LAMBDA_GRID:
        bl = L.shrink_bias(old_bias, bj, lam)
        Pj, _ = L.cascade_probs(mems, wj, thj, cond)
        line = (f"λ={lam}: bias={[round(v, 3) for v in bl]} "
                f"in-sample F1={L.score(Pj, bl, yb, eps):.5f}")
        print("    " + line)
        lines.append("(C) " + line)
    return {"anchor": f_anchor, "half": half_out, "lofo": lofo,
            "insample": {"w": wj, "th": thj, "bias": list(bj)}, "lines": lines}


# ---------------------------------------------------------------------------
# smoke25 — 다폴드 기계 검증 (프록시 멤버, 판정 무효)
# ---------------------------------------------------------------------------
def run_smoke25(y, folds, fmap, old_bias, args):
    print("=" * 72)
    print("[smoke25] 2.5폴드 다폴드 기계 검증 — 프록시 멤버(v9_f14→m1슬롯), 판정 무효(SMOKE)")
    m1p = L.MemberOOF("m1PROXY(v6f0+v9f14)",
                      [os.path.join(ROOT, "work/teacher_largev6_12ep_f0.npz"),
                       os.path.join(ROOT, "work/teacher_largev9_f14.npz")], folds, y, proxy=True)
    klue = L.MemberOOF("klue(f0+f14)",
                       [os.path.join(ROOT, "work/klue_f0.npz"),
                        os.path.join(ROOT, "work/teacher_klue_f14.npz")], folds, y)
    members = [m1p, klue]
    uf = L.usable_folds(members, folds)
    print(f"  smoke usable folds = {uf} (klue 농사 진행에 따라 자동 확장)")
    assert len(uf) >= 2, "smoke25: 다폴드 재료 부족"
    out = run_refit(y, folds, fmap, members, old_bias,
                    w_grid=SMOKE_W_GRID, th_grid=L.TH_GRID,
                    anchor_w=(0.70, 0.30), anchor_th=0.5, cond=(1,),
                    label="smoke25(판정무효)", with_temp=False)
    return uf, out


# ---------------------------------------------------------------------------
def write_report(path, sections):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# D-4 클린 앙상블 OOF 재적합 리포트 (refit_d4)\n\n")
        f.write(f"- 생성: {datetime.datetime.now().isoformat(timespec='seconds')} · git {_git_rev()}\n")
        f.write(f"- 해시: grid={L.grid_hash()} ids={L.IDS_SHA16} splits={L.SPLITS_SHA256[:16]} 구bias={L.OLD_BIAS_SHA16}\n")
        f.write(f"- 수식: ad_lib.predict_conditional_probs 미러 + pred=argmax(log(P+{L.EPS_DEPLOY:g})+bias)\n")
        f.write(f"- 게이트: 비음수 {L.GATE_MIN_NONNEG}+/5폴드, 최악 ≥{L.GATE_WORST_FOLD:+.4f}, 조인트 ≥{L.GATE_JOINT_DELTA:+.4f}\n\n")
        for title, lines in sections:
            f.write(f"## {title}\n\n")
            for ln in lines:
                f.write(f"- {ln}\n")
            f.write("\n")
    print(f"[report] {path} 기록")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["inventory", "probes", "fold0", "refit", "smoke25", "all"])
    ap.add_argument("--m1", default="")
    ap.add_argument("--mdeb", default="")
    ap.add_argument("--klue", default="")
    ap.add_argument("--eps", type=float, default=L.EPS_DEPLOY)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", default=os.path.join(ROOT, "work", "refit_report.md"))
    ap.add_argument("--no-temp", action="store_true")
    args = ap.parse_args()

    ids, y, folds, fmap, members, old_bias = load_all(args)
    sections = []
    failures = []

    if args.mode in ("inventory", "all", "refit", "probes", "fold0"):
        uf = print_inventory(y, folds, members, old_bias)
        inv_lines = []
        for m in members:
            s = m.summary()
            inv_lines.append(f"{s['name']}: folds={s['folds']} files=" +
                             "; ".join(f"{fi['path']}(f{fi['folds']}, oof_f1={fi['oof_f1']})"
                                       for fi in s["files"]))
        inv_lines.append(f"usable folds: {uf}/5")
        sections.append(("재료 인벤토리", inv_lines))

    if args.mode in ("probes", "all"):
        ok, lines = run_probes(y, folds, fmap, members, old_bias, eps=args.eps)
        if not ok:
            failures.append("probes")
        sections.append(("known-probe 회귀 (fold0)", lines))

    if args.mode in ("fold0", "all"):
        ok, lines, _ = run_fold0_repro(y, folds, fmap, members, old_bias)
        if not ok:
            failures.append("fold0-repro")
        sections.append(("레드팀 D12 재현 (fold0, seed7, eps=1e-12)", lines))

    if args.mode in ("refit", "all"):
        out = run_refit(y, folds, fmap, members, old_bias, eps=args.eps, seed=args.seed,
                        with_temp=not args.no_temp)
        sections.append(("정식 재적합 (half-fit + LOFO + λ/온도 + 게이트)", out["lines"]))

    if args.mode in ("smoke25", "all"):
        try:
            uf, out = run_smoke25(y, folds, fmap, old_bias, args)
            sections.append((f"smoke25 다폴드 기계검증 (프록시 — 판정 무효, folds={uf})", out["lines"]))
        except AssertionError as e:
            print(f"  smoke25 스킵: {e}")
            sections.append(("smoke25", [f"스킵: {e}"]))

    if args.mode in ("refit", "all"):
        sections.append(("내일(D-4) 실행 절차", [
            "mdeb folds1-4 npz 도착 → work/mdeb12ep_f14.npz 로 저장(docker cp, 원본 수정 금지)",
            "v6-12ep folds1-4 npz 도착 → work/teacher_largev6_12ep_f14.npz 로 저장",
            "python3 sim/test_refit_d4.py 전체 통과 확인 (회귀·수식동일성)",
            "python3 sim/refit_d4.py --mode all → usable folds 자동 확장·LOFO 부호표·게이트 판정",
            "게이트 PASS 좌표만 3자(codex·레드팀) 검증에 상정 — 그리드 확장 금지(R49 서명)",
        ]))
        write_report(args.report, sections)

    if failures:
        print(f"\n[refit_d4] FAIL: {failures}")
        sys.exit(1)
    print("\n[refit_d4] 완료 (전 검증 통과)")


if __name__ == "__main__":
    main()
