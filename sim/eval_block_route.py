"""sim/eval_block_route.py — ready-to-run class-conditional BLOCK-ROUTING evaluator.

Given a NEW member's fold0 OOF npz (e.g. work/teacher_infoxlm_f0.npz) + the deployed
cascade fold0 members, decide via LEAK-SAFE nested-CV (within fold0) whether
class-conditional routing on a target block (DIALOG / EXEC / web_search) beats the
deployed cascade. CPU-only, read-only (no GPU, no writes to work/).

The lever: on rows where the deployed-cascade argmax falls in the TARGET block
(e.g. DIALOG={ask_user,plan_task,web_search}), replace the cascade posterior with a
blend  beta*new + (1-beta)*cascade  (beta chosen on the train split), then argmax with
the FIXED deployed bias. Everything else is untouched. This isolates a diverse backbone
as a class-conditional specialist without disturbing the rest of the 14-way decision.

-----------------------------------------------------------------------------
PRE-REGISTERED GO GATE  (3-sig required to relax):
  GO  iff   (1) nested-CV mean  Delta macro-F1  >=  +0.002        [effect bar]
      AND   (2) new member beats deployed LARGE on >= 1 target-block class
                 (per-class F1 on fold0 va)                        [diversity is real]
      AND   (3) nested-CV Delta 95% CI lower bound  >  0           [selection-noise screen]
      AND   (4) BLOCK-SPECIFIC gain >= +0.001                      [not generic ensembling]
                 block-routing d  -  max(global-blend d, random-subset d)
  Condition (4) was ADDED after 2026-07-12 diagnosis: a strong member (mdebr) passed
  (1)-(3) with d=+0.0035 but its global-blend control gave +0.0045 (block-specific = -0.0009)
  => the "gain" was generic member-addition, which R76 already found ~exhausted, NOT a
  DIALOG/EXEC block lever. Without (4) the gate false-fires on any strong diverse member.
  Honest LB projection = Delta_nestedCV * TRANSFER.
  TRANSFER default 0.42 (fold0-OOF -> LB, measured on m1 axis: beta=0.36+-0.06, and
  holdout->LB ~0.42). fold0 OOF is IN-distribution; the hidden 30k is partly OOD, so the
  REALISTIC transfer is <= 0.42 (use 0.30-0.42 band; do NOT assume 1.0).
-----------------------------------------------------------------------------
Usage:
  python3 sim/eval_block_route.py work/teacher_infoxlm_f0.npz --block dialog
  python3 sim/eval_block_route.py work/teacher_infoxlm_f0.npz --block all6
  python3 sim/eval_block_route.py --self-test          # runs on klue as pseudo-new-member
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from sim import refit_lib as R
from common.io_utils import IDX_TO_CLASS
from sklearn.metrics import f1_score

# ---- deployed cascade fold0 config (bank th85; mirrors ceiling_f0.py) ----
LARGE_NPZ = os.path.join(ROOT, "work", "m1_f0ckpt_rescue.npz")
MDEB_NPZ = os.path.join(ROOT, "work", "mdeb12ep_f0.npz")
KLUE_NPZ = os.path.join(ROOT, "work", "klue_f0.npz")
W = (0.45, 0.40, 0.15)
TH = 0.85
COND = (1, 2)
BIAS = np.array(json.load(open(os.path.join(ROOT, "packages", "submit_th85", "model", "postproc.json")))["bias"])

BLOCKS = {"dialog": [10, 11, 12], "exec": [7, 8, 9], "web": [12], "all6": [7, 8, 9, 10, 11, 12]}
BETA_GRID = (0.3, 0.5, 0.7, 1.0)      # 1.0 == full replace
GO_BAR = 0.002                        # pre-registered effect bar
BLOCK_SPECIFIC_BAR = 0.001            # block-routing must beat generic controls by this
TRANSFER = 0.42                       # fold0-OOF -> LB (upper edge; OOD may be lower)


def _load_va(path, va):
    d = np.load(path, allow_pickle=True)
    o = np.asarray(d["oof"], np.float64)
    assert o.shape[1] == 14, f"{path}: bad oof shape {o.shape}"
    p = o[va]
    cover = (np.abs(p).sum(1) > 0).mean()
    assert cover > 0.999, f"{path}: only {cover:.3f} of fold0 va rows covered — not a fold0 OOF?"
    p = np.clip(p, 1e-9, None)
    return p / p.sum(1, keepdims=True)


def macro14(yt, pred):
    return f1_score(yt, pred, average="macro", labels=range(14), zero_division=0)


def evaluate(new_npz, block="dialog", K=5, seeds=(0, 1, 2), verbose=True):
    folds, dev, hold = R.load_splits()
    ids, y, groups, _ = R.load_ids_labels()
    va = np.sort(folds[0][1])
    yt = y[va]
    N = len(va)
    large = _load_va(LARGE_NPZ, va)
    mdeb = _load_va(MDEB_NPZ, va)
    klue = _load_va(KLUE_NPZ, va)
    new = _load_va(new_npz, va)
    tgt = BLOCKS[block]

    # deployed cascade posterior + baseline pred (fixed bias)
    Pc, _ = R.cascade_probs([large, mdeb, klue], W, TH, COND)
    base_pred = R.bias_argmax(Pc, BIAS)
    base_full = macro14(yt, base_pred)
    casc_arg = Pc.argmax(1)  # routing trigger = cascade argmax in target block

    # ---- descriptive: per-class F1 new vs large on target-block classes ----
    f_new = f1_score(yt, new.argmax(1), average=None, labels=range(14), zero_division=0)
    f_lrg = f1_score(yt, large.argmax(1), average=None, labels=range(14), zero_division=0)
    wins = [c for c in tgt if f_new[c] > f_lrg[c] + 1e-9]
    if verbose:
        print(f"  deployed cascade+bias  macro-F1(14) = {base_full:.5f}   (N={N}, block={block} classes {tgt})")
        print(f"  routing trigger rows (cascade argmax in block) = {int(np.isin(casc_arg, tgt).sum())}")
        print("  per-class F1  target block   NEW    LARGE    d(new-large)")
        for c in tgt:
            flag = "  <-- new wins" if c in wins else ""
            print(f"     {IDX_TO_CLASS[c]:16s}(c{c:2d})  {f_new[c]:.3f}  {f_lrg[c]:.3f}   {f_new[c]-f_lrg[c]:+.3f}{flag}")

    def route_pred(beta, rowmask):
        """replace posterior on rowmask with beta*new+(1-beta)*cascade, argmax+bias."""
        P = Pc.copy()
        P[rowmask] = beta * new[rowmask] + (1 - beta) * Pc[rowmask]
        return R.bias_argmax(P, BIAS)

    trigger = np.isin(casc_arg, tgt)  # rows the deployed cascade sends to the target block

    def nested_cv(trig_of):
        """leak-safe: pick beta on train, apply on test. trig_of(seed)->bool mask over N."""
        ds, picks = [], []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            perm = rng.permutation(N)
            fid = np.zeros(N, int)
            for i in range(K):
                fid[perm[i::K]] = i
            trig = trig_of(sd)
            for k in range(K):
                trn = np.where(fid != k)[0]
                tst = np.where(fid == k)[0]
                best_beta, best_f = None, macro14(yt[trn], base_pred[trn])
                for beta in BETA_GRID:
                    rmask = np.zeros(N, bool); rmask[trn[trig[trn]]] = True
                    f = macro14(yt[trn], route_pred(beta, rmask)[trn])
                    if f > best_f + 1e-12:
                        best_f, best_beta = f, beta
                if best_beta is None:
                    ds.append(0.0)
                else:
                    rmask = np.zeros(N, bool); rmask[tst[trig[tst]]] = True
                    ds.append(macro14(yt[tst], route_pred(best_beta, rmask)[tst])
                              - macro14(yt[tst], base_pred[tst]))
                picks.append(best_beta)
        return np.array(ds), picks

    # main arm: route on target-block trigger rows
    deltas, beta_picks = nested_cv(lambda sd: trigger)
    mu = deltas.mean()
    se = deltas.std(ddof=1) / np.sqrt(len(deltas))
    ci_lo, ci_hi = mu - 1.96 * se, mu + 1.96 * se

    # CONTROL arms (isolate whether the gain is BLOCK-SPECIFIC or generic ensembling):
    #  - global: blend new member on ALL rows (same beta search)
    #  - random: blend on a random equal-sized subset (matched count, not the block)
    n_trig = int(trigger.sum())
    def rand_trig(sd):
        r = np.random.default_rng(1000 + sd)
        m = np.zeros(N, bool); m[r.choice(N, n_trig, replace=False)] = True
        return m
    d_glob, _ = nested_cv(lambda sd: np.ones(N, bool))
    d_rand, _ = nested_cv(rand_trig)

    # ---- oracle ceiling (upper bound if replace were always right on trigger rows) ----
    r_all = np.zeros(N, bool); r_all[trigger] = True
    oracle_replace = macro14(yt, route_pred(1.0, r_all)) - base_full

    picks = [b for b in beta_picks if b is not None]
    from collections import Counter
    pick_str = dict(Counter([b if b is not None else "baseline" for b in beta_picks]))

    mu_glob = d_glob.mean(); mu_rand = d_rand.mean()
    # block-specific gain = block arm minus the better generic control
    block_specific = mu - max(mu_glob, mu_rand)
    go = (mu >= GO_BAR) and (len(wins) >= 1) and (ci_lo > 0) and (block_specific >= BLOCK_SPECIFIC_BAR)
    if verbose:
        print(f"\n  nested-CV ({len(seeds)}seed x {K}fold = {len(deltas)} est):  "
              f"mean d macro-F1 = {mu:+.5f}  CI95[{ci_lo:+.5f},{ci_hi:+.5f}]")
        print(f"    (within-fold0 CV: the 15 est are correlated -> this CI is OPTIMISTIC; treat as screen)")
        print(f"  beta picks: {pick_str}")
        print(f"  in-sample replace-on-trigger ceiling (NOT honest) = {oracle_replace:+.5f}")
        print(f"  CONTROL  block-routing d ={mu:+.5f}  |  global-blend d ={mu_glob:+.5f}  |  random-subset d ={mu_rand:+.5f}")
        print(f"  BLOCK-SPECIFIC gain (block - best generic control) = {block_specific:+.5f}"
              f"   {'(gain is generic ensembling, NOT block lever)' if block_specific < GO_BAR else '(block conditioning adds value)'}")
        print(f"  new-wins target classes: {[IDX_TO_CLASS[c] for c in wins]}")
        print(f"\n  GO GATE: bar(mu>=+{GO_BAR})={mu>=GO_BAR}  wins>=1={len(wins)>=1}  CIlo>0={ci_lo>0}"
              f"  block-specific(>=+{BLOCK_SPECIFIC_BAR})={block_specific>=BLOCK_SPECIFIC_BAR}"
              f"   ==>  {'GO' if go else 'NO-GO'}")
        print(f"  honest LB projection = {mu:+.5f} * transfer[0.30..0.42] = "
              f"[{mu*0.30:+.5f}, {mu*TRANSFER:+.5f}]  (hidden 30k partly OOD -> use lower edge)")
    return {"block": block, "base_macro": base_full, "delta_mu": mu, "ci": (ci_lo, ci_hi),
            "wins": wins, "oracle_replace": oracle_replace, "GO": go, "beta_picks": pick_str,
            "d_global": mu_glob, "d_random": mu_rand, "block_specific": block_specific}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new_npz", nargs="?", default=None, help="new member fold0 OOF npz")
    ap.add_argument("--block", default="dialog", choices=list(BLOCKS))
    ap.add_argument("--self-test", action="store_true", help="use klue fold0 as pseudo-new-member")
    args = ap.parse_args()
    if args.self_test or args.new_npz is None:
        new = KLUE_NPZ
        print(f"[SELF-TEST] pseudo-new-member = {os.path.relpath(new, ROOT)} (existing klue fold0)")
        for blk in ("dialog", "exec", "web", "all6"):
            print(f"\n===== block={blk} =====")
            evaluate(new, blk)
    else:
        evaluate(args.new_npz, args.block)


if __name__ == "__main__":
    main()
