"""에러분석 루프 — teacher npz(OOF 확률)로 혼동 구조 해부.

사용: python3 eda/error_analysis.py action_decision_maximum/experiments/teacher_base_s1234.npz
출력: 혼동행렬 상위쌍, 쌍별 오분류 샘플 특성(직전action/키워드), 개선 가설.
"""
from __future__ import annotations
import os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.io_utils import load_train, CLASSES
from common.cv import make_splits
from common.metrics import macro_f1, per_class_report
from common import ad_lib

npz_path = sys.argv[1]
d = np.load(npz_path, allow_pickle=True)
oof = d["oof"]
samples, y, ids = load_train()
y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups)
folds = sp["folds"]
lo, hi = int(d["fold_lo"]), int(d["fold_hi"])
cov = np.concatenate([folds[i][1] for i in range(lo, hi)])
pred = oof[cov].argmax(1)
yt = y[cov]
mf1, f1s = macro_f1(yt, pred)
print(f"=== {os.path.basename(npz_path)}  pooled-OOF={mf1:.4f}  n={len(cov)} ===")
print(f"{'class':20} {'F1':>7} {'sup':>6}")
order = np.argsort(f1s)
for c in order:
    print(f"{CLASSES[c]:20} {f1s[c]:7.4f} {int((yt==c).sum()):6d}")

# 혼동행렬 상위 오류쌍
cm = np.zeros((14, 14), int)
for t, p in zip(yt, pred):
    cm[t, p] += 1
pairs = []
for i in range(14):
    for j in range(14):
        if i != j and cm[i, j] > 0:
            pairs.append((cm[i, j], i, j))
pairs.sort(reverse=True)
print("\n=== 상위 오류쌍 (true -> pred) ===")
for n, i, j in pairs[:12]:
    print(f"  {CLASSES[i]:18} -> {CLASSES[j]:18} {n:5d}  ({100*n/max((yt==i).sum(),1):.1f}% of {CLASSES[i]})")

# 상위 3개 쌍의 샘플 해부: 직전 action 분포 + 프롬프트 키워드
def brief(s):
    nm, args, rs, st = ad_lib.last_action(s)
    return nm or "none", st

print("\n=== 상위 3쌍 해부 (직전 action 분포 & 예시) ===")
idx_map = {k: i for i, k in enumerate(cov)}
for n, i, j in pairs[:3]:
    sel = [k for k in cov if y[k] == i and oof[k].argmax() == j]
    la = collections.Counter(brief(samples[k]) for k in sel)
    print(f"\n[{CLASSES[i]} -> {CLASSES[j]}] n={len(sel)}")
    print("  직전(action,status) top5:", la.most_common(5))
    for k in sel[:4]:
        print(f"   ex) last={brief(samples[k])} | {(samples[k].get('current_prompt') or '')[:80]}")

# 마진 분석: 정답이 2등인 비율(후처리/온도로 회수 가능성)
top2 = np.argsort(-oof[cov], axis=1)[:, :2]
second_correct = float(((top2[:, 0] != yt) & (top2[:, 1] == yt)).mean())
print(f"\n오답 중 정답이 2등인 비율(전체 대비): {second_correct:.3f} -> 후처리/앙상블 회수 여지")
