"""n-gram 직교 멤버 검증 — TF-IDF(char+word)+LogReg 5-fold OOF.

codex 제안: 짧은 템플릿성 프롬프트엔 n-gram이 XLM-R와 직교 inductive bias.
크기 작아(~수십MB) 1GB 제약 무해. 배포는 HashingVectorizer+numpy weight로 sklearn-free 가능.
여기선 먼저 "스택에 이득 있나"만 5-fold OOF로 판정.
출력: teacher_ngram.npz (oof/hold), 스택 비교 로그.
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.io_utils import load_train, CLASSES, NUM_CLASSES
from common.cv import make_splits
from common.metrics import macro_f1
from common.postproc import fit_bias
from common import ad_lib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import lightgbm as lgb

EXPD = "action_decision_maximum/experiments"
samples, y, ids = load_train(); y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups); folds = sp["folds"]; hold = sp["holdout_idx"]
cov = np.concatenate([f[1] for f in folds])

# v6 직렬화(구조+이력+플래그 다 포함) 텍스트
texts = [ad_lib.serialize(s, "v6") for s in samples]
oof = np.zeros((len(samples), NUM_CLASSES), np.float32)
hold_sum = np.zeros((len(hold), NUM_CLASSES), np.float32)

def vec():
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=150000, sublinear_tf=True)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=150000, sublinear_tf=True)
    return vw, vc

for fi, (tr, va) in enumerate(folds):
    vw, vc = vec()
    Xtr = hstack([vw.fit_transform([texts[i] for i in tr]), vc.fit_transform([texts[i] for i in tr])]).tocsr()
    clf = LogisticRegression(C=3.0, max_iter=1000, class_weight="balanced", n_jobs=8)
    clf.fit(Xtr, y[tr])
    Xva = hstack([vw.transform([texts[i] for i in va]), vc.transform([texts[i] for i in va])]).tocsr()
    oof[va] = clf.predict_proba(Xva)
    Xho = hstack([vw.transform([texts[i] for i in hold]), vc.transform([texts[i] for i in hold])]).tocsr()
    hold_sum += clf.predict_proba(Xho)
    mf, _ = macro_f1(y[va], oof[va].argmax(1))
    print(f"  fold{fi}: ngram val macro-F1={mf:.4f}", flush=True)

hold_oof = hold_sum / len(folds)
pm, _ = macro_f1(y[cov], oof[cov].argmax(1))
print(f"[ngram] standalone pooled-OOF={pm:.4f}", flush=True)
np.savez_compressed(f"{EXPD}/teacher_ngram.npz", oof=oof, hold=hold_oof,
                    scores=np.array([pm]), fold_lo=0, fold_hi=5, model="tfidf-logreg-v6",
                    version="v6", max_len=0)

# 스택 비교: klue+large (기존 최고 2멤버) vs +ngram
def load(k):
    import glob
    pats = {"klue": ["teacher_klue_s1234.npz"],
            "large": ["teacher_large_probe2.npz", "teacher_largeA_a*.npz", "teacher_largeB_a*.npz"]}
    o = np.zeros((len(samples), NUM_CLASSES), np.float32); h = np.zeros((len(hold), NUM_CLASSES), np.float32)
    cset, w = set(), 0
    for pat in pats[k]:
        for p in ([f"{EXPD}/{pat}"] if "*" not in pat else sorted(glob.glob(f"{EXPD}/{pat}"))):
            if not os.path.exists(p): continue
            z = np.load(p, allow_pickle=True); n = 0
            for f in range(int(z["fold_lo"]), int(z["fold_hi"])):
                if f in cset: continue
                o[folds[f][1]] = z["oof"][folds[f][1]]; cset.add(f); n += 1
            if n: h += z["hold"] * n; w += n
    return o, h / max(w, 1)

import glob
To = {"klue": load("klue"), "large": load("large"), "ngram": (oof, hold_oof)}
F = np.array([ad_lib.stack_features(s) for s in samples], dtype=np.float32); Fh = F[hold]
params = dict(objective="multiclass", num_class=14, learning_rate=0.05, num_leaves=63,
             min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
             verbose=-1, num_threads=8, seed=1234)

def stack(keys):
    P = np.concatenate([To[k][0] for k in keys] + [F], axis=1)
    Ph = np.concatenate([To[k][1] for k in keys] + [Fh], axis=1)
    om = np.zeros((len(samples), NUM_CLASSES), np.float32)
    for tr, va in folds:
        m = lgb.train(params, lgb.Dataset(P[tr], label=y[tr]), num_boost_round=400)
        om[va] = m.predict(P[va])
    b, _ = fit_bias(np.log(om[cov] + 1e-9), y[cov])
    mf = lgb.train(params, lgb.Dataset(P[cov], label=y[cov]), num_boost_round=400)
    hp = mf.predict(Ph)
    hb = macro_f1(y[hold], (np.log(hp + 1e-9) + np.array(b)).argmax(1))[0]
    print(f"[{'+'.join(keys)}] holdout+bias={hb:.4f}")
    return hb

b2 = stack(["klue", "large"])
b3 = stack(["klue", "large", "ngram"])
print(f"\n판정: ngram 멤버 {'채택(+' + format(b3-b2, '.4f') + ')' if b3 > b2 + 0.002 else '기각(이득 미미)'}")
