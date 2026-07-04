"""v6 직렬화 싼 스크리닝 — LinearSVC fold0, v4 대비 (하드클래스 delta 포함)."""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.io_utils import load_train, CLASSES
from common.cv import make_splits
from common.metrics import macro_f1
from common import ad_lib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from scipy.sparse import hstack

samples, y, ids = load_train(); y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups)
tr, va = sp["folds"][0]

HARD = ["read_file", "grep_search", "list_directory", "glob_pattern"]


def run(ver):
    texts = [ad_lib.serialize(s, ver) for s in samples]
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=200000, sublinear_tf=True)
    Xw = vw.fit_transform([texts[i] for i in tr]); Xc = vc.fit_transform([texts[i] for i in tr])
    Vw = vw.transform([texts[i] for i in va]); Vc = vc.transform([texts[i] for i in va])
    clf = LinearSVC(C=0.5)
    clf.fit(hstack([Xw, Xc]).tocsr(), y[tr])
    pred = clf.predict(hstack([Vw, Vc]).tocsr())
    mf1, f1s = macro_f1(y[va], pred)
    return mf1, {c: f1s[i] for i, c in enumerate(CLASSES)}


m4, f4 = run("v4")
m6, f6 = run("v6")
print(f"v4 fold0 LinearSVC macro-F1 = {m4:.4f}")
print(f"v6 fold0 LinearSVC macro-F1 = {m6:.4f}  (delta {m6-m4:+.4f})")
print("\n하드클래스 delta:")
for c in HARD:
    print(f"  {c}: {f4[c]:.4f} → {f6[c]:.4f} ({f6[c]-f4[c]:+.4f})")
print("\n기타 delta 상위:")
for c in sorted(CLASSES, key=lambda c: f6[c]-f4[c], reverse=True)[:5]:
    print(f"  {c}: {f6[c]-f4[c]:+.4f}")
