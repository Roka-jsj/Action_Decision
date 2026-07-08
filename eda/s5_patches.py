#!/usr/bin/env python
"""R34 S5(희소 고정밀 패치) 후보 일괄 측정 — codex measure_patch 프로토콜.

후보: (b)soft transition penalty(fold-clean, eps·tau 그리드) (d1)명시적 금지문구 veto
      (d3)duplicate-failed-action veto (d2)precondition veto(약식).
승인기준: gate<1% · OOF exact ΔF1 양수 · hit>harm · (통과시) fold 반복성 확인 후 사전등록.
"""
from __future__ import annotations
import sys, re, glob
import numpy as np

sys.path.insert(0, "/root/Action_Decision")
from common.io_utils import load_train, CLASSES, NUM_CLASSES
from common.cv import make_splits
from common.metrics import macro_f1
from common.postproc import fit_bias

samples, y, ids = load_train(); y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups)
folds = sp["folds"]

P_oof, cov = np.zeros((len(y), NUM_CLASSES)), np.zeros(len(y), bool)
for f in sorted(glob.glob("/root/Action_Decision/action_decision_maximum/experiments/teacher_largev6[AB]_a*.npz")):
    z = np.load(f, allow_pickle=True)
    o = z["oof"].astype(np.float64); m = o.sum(1) > 0; P_oof[m] = o[m]; cov |= m
oi = np.where(cov)[0]
Po = P_oof[oi]; Po /= Po.sum(1, keepdims=True)
yo = y[oi]
bias, _ = fit_bias(np.log(Po + 1e-12), yo)
L = np.log(Po + 1e-12) + bias
pred = L.argmax(1)
order2 = np.argsort(-L, 1)[:, 1]
Pn = np.exp(L); Pn /= Pn.sum(1, keepdims=True)
srt = np.sort(Pn, 1); margin = srt[:, -1] - srt[:, -2]
base = macro_f1(yo, pred)[0]
print(f"[base] OOF F1={base:.4f} rows={len(yo)}")

# 행별 fold id (oi 기준)
fold_of = np.full(len(y), -1)
for fi, (tr, va) in enumerate(folds):
    fold_of[np.asarray(va)] = fi
fo = fold_of[oi]

# 부가 정보 추출
def last_turn(s):
    h = s.get("history") or []
    for t in reversed(h):
        if t.get("role") == "assistant_action":
            return t
    return None
LT = [last_turn(samples[int(j)]) for j in oi]
LA = np.array([(t.get("name") or "<none>") if t else "<none>" for t in LT])
LR = np.array([str(t.get("result_summary") or "") if t else "" for t in LT])
CP = np.array([str(samples[int(j)].get("current_prompt") or "") for j in oi])

def measure(name, gate):
    g = np.asarray(gate, bool)
    if g.sum() == 0:
        print(f"{name:<42} gate 0행 — 공간 없음"); return
    y1 = pred.copy(); y1[g] = order2[g]
    hit = ((pred != yo) & (y1 == yo) & g).sum()
    harm = ((pred == yo) & (y1 != yo) & g).sum()
    d = macro_f1(yo, y1)[0] - base
    signs = []
    for fi in range(5):
        m = fo == fi
        if (g & m).sum() == 0: continue
        y1f = pred[m].copy(); gf = g[m]; y1f[gf] = order2[m][gf]
        signs.append(int(np.sign(macro_f1(yo[m], y1f)[0] - macro_f1(yo[m], pred[m])[0])))
    print(f"{name:<42} gate {g.sum():4d}({g.mean()*100:.2f}%) hit {hit:3d} harm {harm:3d} ΔF1 {d:+.5f} fold부호 {signs}")

# (b) soft transition penalty — fold-clean 전이표
acts = ["<none>"] + CLASSES
la_idx = np.array([acts.index(a) if a in acts else 0 for a in LA])
LA_all = np.array([(last_turn(s) or {}).get("name") or "<none>" for s in samples])
la_all_idx = np.array([acts.index(a) if a in acts else 0 for a in LA_all])
Tf = np.zeros((5, len(acts), NUM_CLASSES))
for fi, (tr, va) in enumerate(folds):
    tr = np.asarray(tr)
    np.add.at(Tf[fi], (la_all_idx[tr], y[tr]), 1)
Pt = Tf / np.maximum(Tf.sum(2, keepdims=True), 1)
p_trans = Pt[fo, la_idx, pred]
print("\n=== (b) soft transition penalty ===")
for eps in (1e-4, 1e-3, 3e-3):
    for tau in (0.04, 0.08):
        measure(f"trans<{eps} & margin<{tau}", (p_trans < eps) & (margin < tau) & (la_idx > 0))

# (d1) 명시적 금지문구 + web_search 예측
print("\n=== (d1) 금지문구 veto ===")
prohib = re.compile(r"(검색\s*하지\s*마|검색\s*금지|웹\s*검색\s*없이|인터넷\s*없이|오프라인|don'?t\s+search|no\s+web|without\s+(search|internet)|offline\s+only)", re.I)
has_p = np.array([bool(prohib.search(c)) for c in CP])
wsi = CLASSES.index("web_search")
measure("금지문구 & pred=web_search", has_p & (pred == wsi))
measure("금지문구 & pred=ws & margin<0.3", has_p & (pred == wsi) & (margin < 0.3))
print(f"  (참고: 금지문구 행 자체 {has_p.sum()}개)")

# (d3) duplicate failed action
print("\n=== (d3) duplicate-failed-action veto ===")
fail = np.array([bool(re.search(r"(error|fail|nonzero|denied|not found|timeout)", r, re.I)) for r in LR])
same = LA == np.array(CLASSES)[pred]
for tau in (0.08, 0.15, 0.3):
    measure(f"직전동일액션 실패 & margin<{tau}", same & fail & (margin < tau))

# (d2) precondition 약식: 경로류 액션 예측인데 prompt+최근4턴에 경로형 토큰 전무
print("\n=== (d2) precondition veto(약식) ===")
pathre = re.compile(r"[\w\-./]+\.[a-zA-Z]{1,4}\b|/")
def ctx_has_path(j_local):
    s = samples[int(oi[j_local])]
    txts = [str(s.get("current_prompt") or "")]
    for t in (s.get("history") or [])[-4:]:
        txts.append(str(t.get("content") or "") + str(t.get("args") or "") + str(t.get("result_summary") or ""))
    return bool(pathre.search(" ".join(txts)))
target = np.isin(pred, [CLASSES.index(c) for c in ("edit_file", "apply_patch")])
cand = np.where(target & (margin < 0.15))[0]
nopath = np.zeros(len(yo), bool)
for j in cand:
    if not ctx_has_path(j):
        nopath[j] = True
measure("edit/apply 예측 & 저마진 & 경로부재", nopath)
