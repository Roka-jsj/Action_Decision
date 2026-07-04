"""직렬화 v1~v4 스크리닝(HR-1: fold-0 단일) + 혼동클래스 분리성 정성 점검.

질문: 구조/history를 넣으면 애매한 탐색행동(read/grep/glob/list)이 일반화되게 갈리는가?
"""
from __future__ import annotations
import os, sys, time, collections, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.io_utils import load_train, CLASSES
from common.cv import make_splits
from common.metrics import macro_f1, per_class_report
from common.serialize import serialize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion

samples, y, ids = load_train()
y = np.array(y); groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups, force=False)
tr, va = sp["folds"][0]
HARD = ["read_file", "grep_search", "list_directory", "glob_pattern", "apply_patch", "web_search"]

def model():
    return Pipeline([("feat", FeatureUnion([
        ("w", TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=2, max_features=80000, sublinear_tf=True)),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2, max_features=120000, sublinear_tf=True)),
    ])), ("clf", LinearSVC(C=1.0))])

print("=== 직렬화 스크리닝 (fold-0, LinearSVC) ===")
for ver in ["v1","v2","v3","v4"]:
    t=time.time()
    Xtr=[serialize(samples[i], ver) for i in tr]
    Xva=[serialize(samples[i], ver) for i in va]
    m=model(); m.fit(Xtr, y[tr]); pred=m.predict(Xva)
    mf1,_=macro_f1(y[va], pred)
    rep={r[0]:r[4] for r in per_class_report(y[va], pred)}
    hardf1=np.mean([rep[h] for h in HARD])
    print(f"  {ver}: macro-F1={mf1:.4f}  hard6-F1={hardf1:.4f}  ({time.time()-t:.0f}s)  "
          f"| read={rep['read_file']:.2f} grep={rep['grep_search']:.2f} glob={rep['glob_pattern']:.2f} "
          f"list={rep['list_directory']:.2f} apply={rep['apply_patch']:.2f}")

# 분리성 정성: 혼동클래스 프롬프트 샘플
print("\n=== 혼동 탐색행동 프롬프트 샘플 (라벨별 5개) ===")
random.seed(1)
by=collections.defaultdict(list)
for s in samples:
    if s["label"] in ["read_file","grep_search","list_directory","glob_pattern"]:
        by[s["label"]].append(s)
for c in ["read_file","grep_search","list_directory","glob_pattern"]:
    print(f"\n[{c}]")
    for s in random.sample(by[c], 5):
        nm,_,_,st=__import__("common.parse",fromlist=["last_action"]).last_action(s)
        print(f"   last={nm}[{st}] | {(s.get('current_prompt') or '')[:90]}")
