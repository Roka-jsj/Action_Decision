# DIALOG/EXEC block: aleatoric-vs-recoverable verdict + routing harness

Date 2026-07-12. CPU-only, read-only (GPU0 mdeb14 / GPU1 infoxlm_f0 untouched, both 100%).
Scope: validate whether the DIALOG {ask_user=10, plan_task=11, web_search=12} and
EXEC {run_bash=7, run_tests=8, lint=9} blocks are a REAL (learnable) score lever, per
solver D's "diverse-backbone → class-conditional routing → +0.003-4" claim.

Artifacts: `sim/eval_block_route.py` (deliverable), scratchpad `aleatoric_test.py`,
`read_rows.py`. Deployed cascade fold0 = m1_f0ckpt_rescue + mdeb12ep + klue, W(0.45,0.40,0.15),
th0.85, cond(mdeb,klue), th85 bias. fold0 va N=12829, base macro-F1(14)=0.77017.

## TL;DR VERDICT
**The diverse-backbone / block-routing lever is NOT real at the magnitude needed.**
Solver D's headline — "web_search is least-aleatoric (arch-spread 0.135)" — is REFUTED:
the 0.135 spread is mostly small-sample noise (web_search support=239) and, crucially, the
DEPLOYED large is already the single BEST member on web_search (0.749); no diverse backbone
among 9 tested beats it. web_search twin-conflict is 87% (HIGHER than M4's 83%) → web_search
is **also aleatoric**. The genuinely more-recoverable slice is ask_user/plan_task (member
disagreement is real there) and EXEC/run_bash (context-recoverable), but the routing evaluator
shows that per-class F1 wins do NOT convert to macro gains: every existing diverse member is
NO-GO once you subtract the generic-ensembling control. **DIALOG is decision-critically close
to aleatoric like M4.**

---

## TASK 1 — Is DIALOG/EXEC aleatoric or recoverable? (decisive numbers)

### (a) Exact-prompt-twin conflicting-label rate  (M4 solver-A baseline replicated at 83%)
| block / class | twin-rows | conflict-ANY | read |
|---|---|---|---|
| M4 (read/grep/list/glob) | 2895 | **0.832** | reproduces solver-A 80.5% |
| EXEC (bash/tests/lint) | 2105 | **0.708** | lower → somewhat more determinable |
| DIALOG (ask/plan/web) | 1393 | **0.804** | ≈ M4 |
| run_bash (7) | 960 | 0.628 | least conflicting |
| run_tests (8) | 754 | 0.731 | |
| lint (9) | 391 | 0.862 | aleatoric |
| ask_user (10) | 586 | 0.805 | |
| plan_task (11) | 536 | 0.769 | |
| **web_search (12)** | 271 | **0.871** | **most conflicting of ALL — NOT least-aleatoric** |

The same current_prompt maps to a conflicting label 87% of the time for web_search — the
worst of any class. Caveat: twins are on current_prompt only (transformer also sees history);
addressed by the context tests below.

### (b) Prompt-only vs full-context in-block macro-F1 (where does the signal live?)
| block | prompt-only F1 | full-ctx F1 | chance |
|---|---|---|---|
| M4 | 0.245 | 0.439 | 0.25 |
| EXEC | 0.338 | **0.621** | 0.33 |
| DIALOG | 0.357 | 0.556 | 0.33 |

Prompt alone ≈ chance for every block (the current_prompt does not carry the label). Context
lifts EXEC most (0.62) → EXEC is the most context-recoverable block; DIALOG less so.

### (c) Per-class F1: TF-IDF (14-way) vs transformer (deployed large OOF)
| class | transformer | tfidf-prompt | tfidf-fullctx | gap (tr−full) |
|---|---|---|---|---|
| run_bash (7) | 0.834 | 0.471 | 0.631 | +0.204 |
| run_tests (8) | 0.795 | 0.388 | 0.495 | +0.300 |
| lint (9) | 0.655 | 0.141 | 0.240 | +0.415 |
| ask_user (10) | 0.613 | 0.465 | 0.433 | +0.180 |
| plan_task (11) | 0.656 | 0.469 | 0.496 | +0.160 |
| **web_search (12)** | 0.694 | 0.120 | **0.136** | **+0.559** |

web_search has the largest TF-IDF→transformer gap (0.559): shallow lexical models are near-zero
on it. The transformer already extracts whatever recoverable signal exists; there is no lexical
handle a "better backbone" would obviously exploit.

### (d) Architecture-spread: SIGNAL or NOISE? (the crux)
Per-member web_search F1: large_v6r **0.749**, mdebr 0.729, large12 0.681, v8 0.682, sbwt 0.677,
sbsp 0.675, v9 0.658, klue 0.621, mdeb 0.614. **spread 0.135.**

Bootstrap (200 resamples, is spread > sampling noise?):
| class | support | arch-spread | 1-member boot-std | spread / (4·std) |
|---|---|---|---|---|
| grep (1) | 1867 | 0.032 | 0.010 | 0.77 |
| run_bash (7) | 931 | 0.017 | 0.010 | 0.43 |
| lint (9) | 408 | 0.068 | 0.020 | 0.87 |
| ask_user (10) | 504 | 0.062 | 0.017 | 0.92 |
| plan_task (11) | 486 | 0.049 | 0.015 | 0.82 |
| **web_search (12)** | **239** | 0.135 | 0.022 | **1.55** |

web_search's spread is only ~1.5× its 4-sigma sampling band — i.e. a large chunk of the 0.135
is the n=239 small-sample noise, not architecture "learnability." For ask/plan/lint the spread is
WITHIN sampling noise (ratio <1). **Decisive: the "high web_search architecture spread = learnable"
inference is a small-sample artifact.** And the top of the range is the deployed large itself, so
"a diverse backbone could beat large on web_search" is contradicted by all 9 existing members.

Oracle-within-members & error-overlap (recoverable-disagreement test):
| class | best-recall | oracle-recall | oracle gain | err-Jaccard(top2) |
|---|---|---|---|---|
| grep (1, M4 ref) | 0.570 | 0.693 | +0.123 | 0.769 |
| run_bash (7) | 0.821 | 0.865 | +0.044 | 0.803 |
| lint (9) | 0.674 | 0.767 | +0.093 | 0.768 |
| **ask_user (10)** | 0.651 | 0.780 | **+0.129** | **0.621** |
| **plan_task (11)** | 0.737 | 0.848 | **+0.111** | 0.665 |
| web_search (12) | 0.845 | 0.891 | +0.046 | 0.702 |

ask_user & plan_task show the MOST recoverable disagreement (low Jaccard 0.62/0.67, big oracle
gain) — MORE than web_search (+0.046, Jaccard 0.70). This **inverts** solver D's ranking: within
DIALOG, web_search is the LEAST recoverable, ask/plan the most. But note M4's grep has the same
+0.123 oracle gain with high Jaccard 0.77 — and solver A proved M4 disagreement does NOT convert
to gain. The oracle gain is a ceiling on an ORACLE selector, not a realizable lever.

**Block verdict:** web_search = aleatoric mirage (small-sample spread, large already ceiling, 87%
twin-conflict, TF-IDF≈0). EXEC/run_bash = most context-recoverable but members already agree
(spread 0.017) and large is at 0.83 → negligible diverse-backbone headroom. lint = weak but
aleatoric (86% conflict). ask_user/plan_task = the only genuinely-disagreeing slice, but see Task 3.

---

## TASK 2 — Row reading: is the correct label DETERMINABLE from text? (21 rows)
Sampled misclassified rows (deployed cascade wrong) for web_search/ask_user/plan_task. Full text
in `read_rows.py` output. Judgment: the DIALOG triad is a **3-way surface-entangled ambiguity zone**.
The generator assigns the label from a private CONTEXT/intent prior that is decoupled from the
current_prompt surface form (exactly solver A's M4 mechanism).

Representative pattern (surface cue points to a DIFFERENT triad member than the true label):
- TRUE=web_search but prompt reads plan/ask: *"포맷별로 분기되게 고치려고요. 근데 그 전에 작업 단계 좀 정리해주면 좋겠어요"* (="lay out the steps" = plan_task surface); *"do you want me to make the workers deterministic too, or just the global useFetch?"* (= ask_user surface).
- TRUE=ask_user but prompt reads web/plan: *"어떻게 바꾸는 게 요즘 권장인지 잠깐 검색 좀"* (="search" = web_search surface); *"단계별로 어떻게 갈지 짜줘"* (= plan_task surface).
- TRUE=plan_task but prompt reads ask/web: *"베스트프랙티스가 뭔지 좀 찾아봐줘"* (= web_search surface); *"which direction fits our convention better?"* (= ask_user surface).

Count: of 21 misclassified rows, ~13-15 are **NOT determinable** as their true label (surface cues
point elsewhere in the triad); ~4-6 ARE determinable (clear "찾아봐/검색"→web, "which way you
want/너가 정해"→ask) and for several of those the strong deployed large already gets the class right
elsewhere — the misclassified POOL is dominated by genuinely ambiguous rows. **A better/diverse
model cannot systematically fix these: the "correct" label is not recoverable from the input.**
This matches the 80-87% twin-conflict and the TF-IDF≈0 result. A THIN recoverable slice exists
(the clean "search" prompts the model missed), but it is minority and low-mass.

---

## TASK 3 — Ready-to-run routing evaluator `sim/eval_block_route.py`

Given a new member fold0 OOF npz + the deployed cascade fold0 members, it computes:
1. per-class F1 of the new member vs deployed large on the target block (7-12);
2. leak-safe nested-CV (within fold0, 3seed×5fold) macro-F1 of class-conditional routing —
   on rows where cascade argmax ∈ target block, replace posterior with β·new+(1−β)·cascade
   (β picked on train, applied on test), argmax with FIXED deployed bias — vs the deployed baseline;
3. CONTROL arms: same routing on ALL rows (global-blend) and on a random equal-sized subset, to
   isolate whether the gain is block-specific or generic ensembling.

### Pre-registered GO gate (3-sig to relax)
```
GO iff (1) nested-CV mean ΔmacroF1 ≥ +0.002
   AND (2) new member beats deployed large on ≥1 target-block class (per-class F1)
   AND (3) nested-CV Δ 95% CI lower bound > 0
   AND (4) BLOCK-SPECIFIC gain ≥ +0.001   [= block-routing Δ − max(global-blend Δ, random Δ)]
Honest LB projection = Δ_nestedCV × transfer,  transfer ∈ [0.30, 0.42]  (use LOWER edge, hidden OOD)
```
Condition (4) was ADDED after diagnosis (see below): without it the gate false-fires on any strong
member via generic ensembling. It encodes "class-conditional routing must beat generic member-add."

### Tested on EXISTING members (ready to run; drop in `work/teacher_infoxlm_f0.npz` when it lands)
| pseudo-new-member | block | nested-CV Δ | CI95 | global-blend Δ | block-specific | GO? |
|---|---|---|---|---|---|---|
| klue (in-cascade) | dialog | +0.00000 | [0,0] | +0.00000 | +0.00000 | NO-GO |
| **mdebr** (strong, ~mdeb) | all6 | **+0.00354** | [+0.0021,+0.0050] | **+0.00446** | **−0.00092** | **NO-GO** |
| mdebr | dialog | +0.00127 | [+0.0000,+0.0025] | +0.00446 | −0.00319 | NO-GO |
| v9 | all6 | −0.00135 | [−0.0024,−0.0003] | — | — | NO-GO |
| v8 | all6 | −0.00128 | [−0.0022,−0.0004] | — | — | NO-GO |
| v9 | web | +0.00000 | [0,0] | — | −0.00005 | NO-GO |
| sbwt | all6 | +0.00000 | [0,0] | — | — | NO-GO |

**The mdebr diagnosis (why control arm is mandatory):** mdebr passes gate (1)-(3) with Δ=+0.0035
and wins 5/6 per-class F1 — but its GLOBAL blend gives +0.0045 (bigger), so block-specific = −0.0009.
The gain is generic "add a strong member," which R76 already found ~exhausted (stacking dev-OOF
−0.00015), NOT a DIALOG/EXEC block lever. **Every existing diverse member is NO-GO on the true
gate.** Per-class F1 wins ≠ macro routing gains (klue wins ask/plan +0.017/+0.053 → still NO-GO,
in-sample replace ceiling negative).

### Usage
```
python3 sim/eval_block_route.py work/teacher_infoxlm_f0.npz --block dialog
python3 sim/eval_block_route.py work/teacher_infoxlm_f0.npz --block all6
python3 sim/eval_block_route.py --self-test        # klue as pseudo-new-member
```

## TASK 3b — Honest transfer caveat (pre-registered)
fold0-OOF nested-CV Δ is IN-distribution and OPTIMISTIC: (i) the 15 nested-CV estimates share the
same 12829 rows (correlated → the CI understates true cross-fold uncertainty); (ii) the hidden 30k
is partly OOD vs fold0 (holdout→LB shrink ~0.42; measured m1-axis fold0→LB transfer β=0.36±0.06).
**Apply transfer ∈ [0.30, 0.42] and use the LOWER edge.** A fold0 Δ that only clears +0.002 projects
to ~+0.0006-0.0008 LB — below noise and far below the +0.00256 LB cut gap.

---

## TASK D — Honest estimate: if infoxlm/rembert beat large on web_search by +0.05
- web_search is 1 of 14 classes, support 1.9% of rows. Arithmetic macro ceiling of a +0.05
  web_search-F1 gain = 0.05/14 = **+0.0036 macro** IF perfectly swapped with zero collateral.
- Realizable is a small fraction of that: routing triggers on cascade-argmax∈block rows (imprecise,
  includes false-positives), recall gains cost neighbor precision, and empirically every tested
  member's block-specific routing gain is ≤ 0. Optimistic fold0 block-specific ΔmacroF1 ≈
  **+0.0003 to +0.0010**; × transfer[0.30-0.42] → **LB +0.0001 to +0.0004** (could be ~0 or slightly
  negative if it dents ask/plan). Whole-DIALOG-block +0.05 on all three classes → arithmetic ceiling
  +0.0107, realizable fold0 ~+0.001-0.002, **LB ~+0.0004-0.0007**.
- Cut needs +0.00256 LB (≈ casc +0.0055). **This lever delivers ~1/10 of the gap even under the
  favorable (and counterfactual — large is already the best web_search member) +0.05 assumption.**

## Decision-critical bottom line
DIALOG is **also aleatoric**, essentially like M4 for web_search and lint; ask_user/plan_task carry
a thin recoverable-disagreement slice that does not convert to routing macro-gain. The
diverse-backbone routing path to the cut is NOT supported by the data. Keep `eval_block_route.py`
armed so infoxlm_f0/rembert_f0 get a rigorous, generic-ensembling-proof GO/NO-GO the instant they
land — but the prior from 9 existing members and the aleatoric evidence is NO-GO. Do not gate the
endgame on this lever.
