"""label-shift bias 재적합 (R20 A2) — 측정된 test prior로 OOF confusion 재가중 → macro-F1 최적 bias.

절차(codex R17/R19):
1. tri_cond와 동일 구성의 앙상블 OOF 확률 재현(가중·조건부 혼합).
2. bias 후보를 좌표상승하되, macro-F1을 **test prior로 재가중된 기대 confusion**에서 계산.
   구체: 각 예측 argmax(logp+bias)에 대해, 클래스별 TP/FP/FN을 OOF에서 세되
   각 행의 기여를 w(y_i) = π_test(y_i)/π_train(y_i)로 가중 → prior-shifted macro-F1.
3. 최적 bias를 postproc에 저장(글로벌 대체). test prior=train이면 기존 bias와 동일 수렴.
usage: python3 eda/labelshift_bias.py <test_prior.json>  (test_prior.json: {class: count})
      → artifacts/labelshift_bias.json 출력
"""
from __future__ import annotations
import sys, os, json, glob
import numpy as np

R = "/root/Action_Decision"
sys.path.insert(0, R)
from common.io_utils import load_train, CLASSES
from common.cv import make_splits
from common.metrics import macro_f1

samples, y, ids = load_train(); y = np.array(y)
sp = make_splits(ids, y, np.array([s["session"] for s in samples])); folds = sp["folds"]
cov = np.concatenate([f[1] for f in folds])

# tri_cond 앙상블 OOF 재현: 0.6·v6 + 0.15·base + 0.25·v4mix, 저마진th0.5 조건부
def load_oof(g):
    o = np.zeros((len(y), 14), np.float32); cs = set()
    for p in sorted(glob.glob(g)):
        z = np.load(p, allow_pickle=True)
        for f in range(int(z["fold_lo"]), int(z["fold_hi"])):
            if f in cs: continue
            o[folds[f][1]] = z["oof"][folds[f][1]]; cs.add(f)
    return o
E = f"{R}/action_decision_maximum/experiments"
p6 = load_oof(f"{E}/teacher_largev6[AB]_a*.npz")
pb = load_oof(f"{E}/teacher_basev6e5_g0.npz")
p4 = load_oof(f"{E}/teacher_largev4mix.npz")
W = [0.6, 0.15, 0.25]; TH = 0.5
p_full = (W[0] * p6 + W[1] * pb) / (W[0] + W[1])
srt = np.sort(p_full, axis=1); sel = (srt[:, -1] - srt[:, -2]) < TH
mean_oof = p_full.copy()
mean_oof[sel] = ((W[0] + W[1]) * p_full[sel] + W[2] * p4[sel]) / (W[0] + W[1] + W[2])
lp = np.log(mean_oof + 1e-9)

train_prior = np.array([(y == i).mean() for i in range(14)])
tp_in = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else None
if tp_in:
    cnt = np.array([tp_in.get(c, 0) for c in CLASSES], dtype=float)
    test_prior = cnt / cnt.sum()
else:
    test_prior = train_prior.copy()  # 프로브 전 = 동일가정(기존 bias 재현 확인용)
w_row = (test_prior / np.maximum(train_prior, 1e-9))[y[cov]]   # 행별 prior-shift 가중

def weighted_macro_f1(pred, yy, w):
    f1s = []
    for c in range(14):
        tp = w[(pred == c) & (yy == c)].sum()
        fp = w[(pred == c) & (yy != c)].sum()
        fn = w[(pred != c) & (yy == c)].sum()
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return np.mean(f1s)

yc = y[cov]; lpc = lp[cov]
bias = np.zeros(14)
best = weighted_macro_f1(lpc.argmax(1), yc, w_row)
base_unw = macro_f1(yc, lpc.argmax(1), 14)[0]
print(f"[labelshift] 시작 prior-weighted macroF1={best:.5f} (unweighted={base_unw:.4f})")
grids = [np.linspace(-2, 2, 21), np.linspace(-1, 1, 21), np.linspace(-0.4, 0.4, 17)]
rng = np.random.default_rng(42)
for grid in grids:
    improved = True; it = 0
    while improved and it < 8:
        improved = False; it += 1
        for c in rng.permutation(14):
            cur = bias[c]; bv, bs = cur, best
            for d in grid:
                bias[c] = cur + d
                s = weighted_macro_f1((lpc + bias).argmax(1), yc, w_row)
                if s > bs + 1e-6: bs, bv = s, bias[c]
            bias[c] = bv
            if bs > best + 1e-9: best = bs; improved = True
print(f"[labelshift] 최적 prior-weighted macroF1={best:.5f}")
print(f"[labelshift] unweighted(train prior)에서 이 bias: {macro_f1(yc,(lpc+bias).argmax(1),14)[0]:.4f}")
os.makedirs(f"{R}/artifacts", exist_ok=True)
json.dump({"bias": bias.tolist(), "classes": CLASSES, "test_prior": test_prior.tolist()},
          open(f"{R}/artifacts/labelshift_bias.json", "w"), indent=2)
print(f"→ artifacts/labelshift_bias.json 저장. test_prior=train이면 기존 bias와 수렴")
