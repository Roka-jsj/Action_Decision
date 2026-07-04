"""에러분석 2차 — large(4ep 5fold 병합) OOF 기준 혼동 구조 + 슬라이스 분석.

목적: ① 혼동쌍 상위 → v6 직렬화 타깃 신호 설계 ② au 서브셋 처리 방향
     ③ 6ep가 fold0에서 무엇을 개선했나 (에폭 스케일링의 정체)
출력: eda/error2_report.md
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.io_utils import load_train, CLASSES, NUM_CLASSES
from common.cv import make_splits
from common.metrics import macro_f1, per_class_report
from common import ad_lib

EXPD = "action_decision_maximum/experiments"
samples, y, ids = load_train(); y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups); folds = sp["folds"]; hold = sp["holdout_idx"]
cov = np.concatenate([f[1] for f in folds])

# large 4ep 병합 OOF
oof = np.zeros((len(samples), NUM_CLASSES), np.float32)
for p in ([f"{EXPD}/teacher_large_probe2.npz"] + sorted(glob.glob(f"{EXPD}/teacher_largeA_a*.npz"))
          + sorted(glob.glob(f"{EXPD}/teacher_largeB_a*.npz"))):
    z = np.load(p, allow_pickle=True)
    for fi in range(int(z["fold_lo"]), int(z["fold_hi"])):
        oof[folds[fi][1]] = z["oof"][folds[fi][1]]
pred = oof[cov].argmax(1); yt = y[cov]
mf1, f1s = macro_f1(yt, pred)

L = [f"# 에러분석 2차 (large-4ep 5fold OOF, pooled={mf1:.4f})\n"]

# 1) per-class
L.append("## 클래스별 F1 (오름차순)\n")
L.append("| class | support | P | R | F1 |\n|---|---|---|---|---|")
for name, sup, p, r, f1 in sorted(per_class_report(yt, pred), key=lambda x: x[4]):
    L.append(f"| {name} | {sup} | {p} | {r} | {f1} |")

# 2) 혼동쌍 상위
cm = np.zeros((14, 14), int)
for t, q in zip(yt, pred):
    cm[t, q] += 1
pairs = []
for i in range(14):
    for j in range(14):
        if i != j and cm[i, j] > 0:
            pairs.append((cm[i, j], CLASSES[i], CLASSES[j]))
pairs.sort(reverse=True)
L.append("\n## 혼동쌍 상위 15 (정답→오답, 건수)\n")
for c, a, b in pairs[:15]:
    L.append(f"- {a} → {b}: {c}")

# 3) 슬라이스: 빈 history / gen / last_action별 오류율
idx_cov = cov
hist_len = np.array([len(ad_lib.action_sequence(samples[i])) for i in idx_cov])
gen = np.array([samples[i]["gen"] for i in idx_cov])
err = (pred != yt)
L.append("\n## 슬라이스별 오류율\n")
for name, mask in [("history=0(세션시작)", hist_len == 0), ("history 1-2", (hist_len >= 1) & (hist_len <= 2)),
                   ("history 3+", hist_len >= 3), ("gen=sim", gen == "sim"), ("gen=au", gen == "au")]:
    if mask.sum():
        m_f1, _ = macro_f1(yt[mask], pred[mask])
        L.append(f"- {name}: n={mask.sum()}, err={err[mask].mean():.3f}, macroF1={m_f1:.4f}")

# 4) fold0: 4ep vs 6ep 클래스별 delta
z6 = np.load(f"{EXPD}/teacher_large6ep_a1.npz", allow_pickle=True)
va0 = folds[0][1]
p4 = oof[va0].argmax(1); p6 = z6["oof"][va0].argmax(1); y0 = y[va0]
_, f4 = macro_f1(y0, p4); _, f6 = macro_f1(y0, p6)
L.append("\n## fold0: 6ep − 4ep 클래스별 F1 변화 (내림차순)\n")
for d, c in sorted(zip(f6 - f4, CLASSES), reverse=True):
    L.append(f"- {c}: {'+' if d>=0 else ''}{d:.4f}")

# 5) rank2 회복 가능량
order = np.argsort(-oof[cov], axis=1)
r2 = (order[:, 1] == yt) & err
L.append(f"\n## rank-2 정답 비율(오류 중): {r2.sum()}/{err.sum()} = {r2.sum()/max(err.sum(),1):.3f}")

open("eda/error2_report.md", "w").write("\n".join(L))
print("\n".join(L[:40]))
print(f"\nsaved → eda/error2_report.md")
