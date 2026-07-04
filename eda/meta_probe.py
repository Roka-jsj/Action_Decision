"""결정적 실험 (codex R8) — 텍스트 밖 메타가 탐색클러스터를 가르는가.

large가 못 본(또는 bin으로만 본) 신호만 정조준:
  A. elapsed_session_sec — 어떤 직렬화 버전에도 없음 (완전 미사용)
  B. raw 수치: turn_index/budget/loc 원값 + 파생(pace=elapsed/turn, budget/turn)
  C. 전체 세션 행동 카운트 14종 + status 카운트 (v6 [SEQ]는 최근12 순서만)
LGB(logit) vs LGB(logit+meta) — GBDT라 스케일붕괴 이슈 없음. 탐색4class GroupKFold-5.
판정(codex 합의): +0.005↑ & fold 4/5 양수 → 새 정보원(→ v7 직렬화 투자). +0.003↓ → 문 닫음.
"""
from __future__ import annotations
import os, sys, re, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor"))
import lightgbm as lgb
from sklearn.metrics import f1_score
from common.io_utils import load_train, CLASSES
from common.cv import make_splits
from common import ad_lib

samples, y, ids = load_train(); y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups); folds = sp["folds"]
ci = {c: i for i, c in enumerate(CLASSES)}
EXP = {ci[c] for c in ["read_file", "grep_search", "list_directory", "glob_pattern"]}

oof = np.zeros((len(samples), 14), np.float32); cs = set()
for pat in ["teacher_largev6A_a*", "teacher_largev6B_a*"]:
    for p in sorted(glob.glob(f"action_decision_maximum/experiments/{pat}.npz")):
        z = np.load(p, allow_pickle=True)
        for f in range(int(z["fold_lo"]), int(z["fold_hi"])):
            if f in cs: continue
            oof[folds[f][1]] = z["oof"][folds[f][1]]; cs.add(f)
assert cs == {0, 1, 2, 3, 4}, f"fold coverage {cs}"
L = np.log(oof + 1e-9)

ST = ["na", "success", "error", "test_fail", "test_pass", "zero", "nonzero_exit"]

def meta_vec(s):
    m = ad_lib.meta_fields(s)
    hist = s.get("history") or []
    acts = [t for t in hist if t.get("role") == "assistant_action"]
    el, ti, bu, lo = float(m["elapsed"]), float(m["turn_index"]), float(m["budget"]), float(m["loc"])
    # A: elapsed 원값 + 파생 pace
    fA = [el, el / max(ti, 1.0), el / max(len(acts), 1)]
    # B: raw 수치 (bin 경계 내부 위치 정보)
    fB = [ti, bu, lo, bu / max(ti, 1.0), np.log1p(max(bu, 0)), np.log1p(max(lo, 0))]
    # C: 전체 세션 카운트 (SEQ 최근12 밖 정보)
    cnt = [0.0] * 14
    for t in acts:
        n = t.get("name", "")
        if n in ci: cnt[ci[n]] += 1
    nerr = sum(1 for t in acts if "error" in (t.get("result_summary") or "").lower())
    nuser = sum(1 for t in hist if t.get("role") == "user")
    fC = cnt + [len(hist), len(acts), nuser, nerr, len(acts) - len(set(t.get("name", "") for t in acts))]
    # 공통 컨텍스트 (seen이지만 상호작용용)
    fD = [float(m["n_open_files"]), float(m["git_dirty"]),
          {"passed": 0, "failed": 1}.get(m["last_ci_status"], 2),
          len(s.get("current_prompt") or "")]
    return fA + fB + fC + fD

NA = 3; NB = 6; NC = 19  # 블록 경계 (ablation용)

fold_of = np.full(len(samples), -1)
for fi, (_, va) in enumerate(folds):
    fold_of[va] = fi
# 홀드아웃(fold 밖) 행 제외 — np.empty 미예측 쓰레기값이 유령클래스로 macro 오염시키는 버그 수정
idx = np.array([i for i in range(len(samples)) if y[i] in EXP and fold_of[i] >= 0])
Ls = L[idx]
M = np.array([meta_vec(samples[i]) for i in idx], dtype=np.float64)
y4 = np.array([sorted(EXP).index(y[i]) for i in idx])
fs = fold_of[idx]

PAR = dict(objective="multiclass", num_class=4, learning_rate=0.06, num_leaves=63,
           min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=1.0, verbosity=-1, num_threads=8, seed=1234)

def run(X):
    pred = np.empty(len(y4), dtype=int)
    for fi in range(5):
        tr = np.where(fs != fi)[0]; va = np.where(fs == fi)[0]
        d = lgb.Dataset(X[tr], label=y4[tr])
        bst = lgb.train(PAR, d, num_boost_round=400)
        pred[va] = bst.predict(X[va]).argmax(1)
    return f1_score(y4, pred, average="macro"), pred

def foldgain(pa, pb):
    out = []
    for fi in range(5):
        va = np.where(fs == fi)[0]
        out.append(f1_score(y4[va], pa[va], average="macro") - f1_score(y4[va], pb[va], average="macro"))
    return out

s0, p0 = run(Ls)                                        # logit only
sM, pM = run(np.hstack([Ls, M]))                        # + 전체 메타
sA, pA = run(np.hstack([Ls, M[:, :NA]]))                # + elapsed만 (완전 미사용 필드 격리)
sC, pC = run(np.hstack([Ls, M[:, NA + NB:NA + NB + NC]]))  # + 세션카운트만

fgM = foldgain(pM, p0)
fgA = foldgain(pA, p0)
fgC = foldgain(pC, p0)
print(f"[탐색4class LGB macro-F1] logit-only={s0:.4f}  (n={len(y4)}, holdout 제외)")
print(f"  +META전체={sM:.4f} (Δ{sM - s0:+.4f})  fold별 {[round(x, 4) for x in fgM]} (양수 {sum(x > 0 for x in fgM)}/5)")
print(f"  +elapsed만={sA:.4f} (Δ{sA - s0:+.4f})  fold별 {[round(x, 4) for x in fgA]}   ← 완전 미사용 필드")
print(f"  +세션카운트만={sC:.4f} (Δ{sC - s0:+.4f})  fold별 {[round(x, 4) for x in fgC]} ← SEQ 최근12 밖 정보")
go = (sM - s0 >= 0.005) and (sum(x > 0 for x in fgM) >= 4)
weak = (sM - s0 >= 0.003)
print(f"\n판정: {'GO — 새 정보원 존재 → v7 직렬화 투자' if go else ('WEAK — 재현성 추가확인' if weak else 'DROP — 메타도 잉여, 문 닫음 → 함대 올인')}")
