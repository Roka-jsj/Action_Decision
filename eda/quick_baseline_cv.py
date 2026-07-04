"""피벗 실험: current_prompt → action 매핑이 '일반화'되는가?

TF-IDF(word+char) + LinearSVC 를 (a) StratifiedGroupKFold(session) 와
(b) 무작위 StratifiedKFold 로 각각 pooled-OOF macro-F1 측정 → 누수 gap 정량화.
+ 프로즌 홀드아웃 성능, + sim-only pooled-OOF.
sklearn 1.7.2(로컬)는 '추정용'일 뿐 배포 아티팩트 아님.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.io_utils import load_train
from common.cv import make_splits, fold_class_counts
from common.metrics import macro_f1, print_report

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedKFold

t0 = time.time()
samples, y, ids = load_train()
y = np.array(y)
groups = np.array([s["session"] for s in samples])
gen = np.array([s["gen"] for s in samples])
text = [(s.get("current_prompt") or "") for s in samples]
N = len(y)

def make_model():
    return Pipeline([
        ("feat", FeatureUnion([
            ("w", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                  min_df=2, max_features=80000, sublinear_tf=True)),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                  min_df=2, max_features=120000, sublinear_tf=True)),
        ])),
        ("clf", LinearSVC(C=1.0)),
    ])

def run_cv(folds, name):
    oof = np.full(N, -1, dtype=int)
    for i, (tr, va) in enumerate(folds):
        m = make_model()
        m.fit([text[j] for j in tr], y[tr])
        oof[va] = m.predict([text[j] for j in va])
    mask = oof >= 0
    mf1, _ = macro_f1(y[mask], oof[mask])
    acc = float(np.mean(oof[mask] == y[mask]))
    print(f"[{name}] pooled-OOF macro-F1={mf1:.4f} acc={acc:.4f} (n={mask.sum()})")
    return oof

# splits (holdout + GroupKFold on dev)
sp = make_splits(ids, y, groups, holdout_frac=0.08, n_splits=5, seed=42, force=True)
dev_idx, hold_idx, folds = sp["dev_idx"], sp["holdout_idx"], sp["folds"]
print(f"dev={len(dev_idx)} holdout={len(hold_idx)}")
tbl = fold_class_counts(y, folds)
print("fold class-count min per fold:", tbl.min(axis=1), "(0이면 문제)")

# (a) GroupKFold pooled-OOF
oof_g = run_cv(folds, "GroupKFold(session)")

# (b) 무작위 StratifiedKFold (누수 비교) — dev 인덱스 위에서
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rand_folds = [(dev_idx[tr], dev_idx[va]) for tr, va in skf.split(dev_idx, y[dev_idx])]
oof_r = run_cv(rand_folds, "RandomKFold (누수有)")

# sim-only pooled-OOF (visible test 포맷과 매칭)
mask_dev = np.zeros(N, bool); mask_dev[dev_idx] = True
sim_mask = mask_dev & (gen == "sim") & (oof_g >= 0)
mf1_sim, _ = macro_f1(y[sim_mask], oof_g[sim_mask])
print(f"[GroupKFold sim-only] macro-F1={mf1_sim:.4f} (n={sim_mask.sum()})")

# 홀드아웃: dev 전체로 학습 → holdout 예측
m = make_model(); m.fit([text[j] for j in dev_idx], y[dev_idx])
hp = m.predict([text[j] for j in hold_idx])
mf1_h, _ = macro_f1(y[hold_idx], hp)
print(f"[HOLDOUT] macro-F1={mf1_h:.4f} acc={np.mean(hp==y[hold_idx]):.4f} (n={len(hold_idx)})")

# per-class on GroupKFold OOF (dev)
print_report(y[dev_idx], oof_g[dev_idx], "GroupKFold OOF per-class")
print(f"\n총 소요 {time.time()-t0:.0f}s")
