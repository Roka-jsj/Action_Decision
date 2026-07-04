"""결정적 실험 (codex R7) — raw 원문 args가 v6-visible 대비 탐색클러스터 순증분을 내는가.

logit-only vs +visible(v6가 인코딩한 신호) vs +raw(원문 args+정규화 path/glob/regex/line).
GroupKFold-5(세션), char-wb 3-5 TFIDF + LogReg on [largev6 OOF logit + text].
탐색 4-class true label 행만. GO: raw-visible >= +0.004 & 4/5 fold positive.
결론: raw가 순증분 크면 specialist GO(codex), 아니면 잉여 확정(내 반론) → 함대 집중.
"""
from __future__ import annotations
import os, sys, re, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.io_utils import load_train, CLASSES
from common.cv import make_splits
from common import ad_lib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.sparse import hstack, csr_matrix

samples, y, ids = load_train(); y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups); folds = sp["folds"]
ci = {c: i for i, c in enumerate(CLASSES)}
EXP = {ci[c] for c in ["read_file", "grep_search", "list_directory", "glob_pattern"]}

# largev6 OOF (probs → log = pseudo-logit)
oof = np.zeros((len(samples), 14), np.float32); cs = set()
for pat in ["teacher_largev6A_a*", "teacher_largev6B_a*"]:
    for p in sorted(glob.glob(f"action_decision_maximum/experiments/{pat}.npz")):
        z = np.load(p, allow_pickle=True)
        for f in range(int(z["fold_lo"]), int(z["fold_hi"])):
            if f in cs: continue
            oof[folds[f][1]] = z["oof"][folds[f][1]]; cs.add(f)
L = np.log(oof + 1e-9)

def hist_actions(s):
    return [t for t in (s.get("history") or []) if t.get("role") == "assistant_action"]

def visible_text(s):
    # v6가 인코딩한 신호 재현: PFLAG + history action명+ext
    m = ad_lib.meta_fields(s)
    fl = ad_lib._prompt_flags(s.get("current_prompt") or "", m.get("open_files") or [])
    parts = [f"PFLAG={fl}"]
    for t in hist_actions(s)[-6:]:
        nm = t.get("name", ""); a = ad_lib.arg_path_or_pattern(nm, t.get("args") or {})
        ext = ad_lib.path_ext(a) or ad_lib.glob_ext(a)
        parts.append(f"{nm}:{ext}")
    return " ".join(parts)

def prompt_text(s):
    return (s.get("current_prompt") or "")[:200]

def rawargs_text(s):
    # 원문 args/result만 (current_prompt 제외 — 교란 제거)
    parts = []
    for t in hist_actions(s)[-4:]:
        args = t.get("args") or {}
        for v in args.values():
            if isinstance(v, str):
                parts.append(v)
                if re.search(r"[*?\[]", v): parts.append("GLOB=" + v)
                elif "/" in v or re.search(r"\.\w{1,5}$", v):
                    parts.append("PATH=" + v)
                    parts.append("BASENAME=" + v.rsplit("/", 1)[-1])
                    parts.append("PARENT=" + (v.rsplit("/", 1)[0].rsplit("/", 1)[-1] if "/" in v else ""))
                if re.search(r"\\[sSwWdb]|\.\*|\[\^", v): parts.append("REGEX=1")
        rs = t.get("result_summary") or ""
        mnums = re.findall(r"(\d+)\s*(?:L|lines|items|occurrences)", rs)
        if mnums: parts.append("CNT=" + "_".join(mnums[:2]))
    return " ".join(parts)

def raw_text(s):
    return visible_text(s) + " " + rawargs_text(s) + " " + prompt_text(s)

idx = np.array([i for i in range(len(samples)) if y[i] in EXP])
Ls = L[idx]; ys = y[idx]; gs = groups[idx]
vis = [visible_text(samples[i]) for i in idx]
raw = [raw_text(samples[i]) for i in idx]
# fold 매핑 (세션 기준 기존 folds 재사용)
fold_of = np.full(len(samples), -1)
for fi, (_, va) in enumerate(folds):
    fold_of[va] = fi
fs = fold_of[idx]

from sklearn.preprocessing import StandardScaler

def score(texts, use_text=True):
    pred = np.empty(len(ys), dtype=int)
    for fi in range(5):
        tr = np.where(fs != fi)[0]; va = np.where(fs == fi)[0]
        sc = StandardScaler().fit(Ls[tr])          # logit 컬럼 표준화(tfidf와 스케일 정합)
        Ltr, Lva = sc.transform(Ls[tr]), sc.transform(Ls[va])
        if use_text:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=60000)
            Xtr = hstack([csr_matrix(Ltr), vec.fit_transform([texts[j] for j in tr])]).tocsr()
            Xva = hstack([csr_matrix(Lva), vec.transform([texts[j] for j in va])]).tocsr()
        else:
            Xtr, Xva = csr_matrix(Ltr), csr_matrix(Lva)
        clf = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced", n_jobs=8)
        clf.fit(Xtr, ys[tr]); pred[va] = clf.predict(Xva)
    return f1_score(ys, pred, average="macro"), pred

prm = [prompt_text(samples[i]) for i in idx]
prm_args = [prompt_text(samples[i]) + " " + rawargs_text(samples[i]) for i in idx]
s_logit, _ = score(None, use_text=False)
s_prm, p_prm = score(prm)             # logit + current_prompt (large가 이미 본 것)
s_pa, p_pa = score(prm_args)          # logit + prompt + raw args (args 순증분 격리)
def foldgain(pa, pb):
    out = []
    for fi in range(5):
        va = np.where(fs == fi)[0]
        out.append(f1_score(ys[va], pa[va], average="macro") - f1_score(ys[va], pb[va], average="macro"))
    return out
fg_prm = foldgain(p_prm, p_prm)  # dummy
fg_args = foldgain(p_pa, p_prm)
print(f"[탐색4class macro-F1] logit-only={s_logit:.4f}  +prompt={s_prm:.4f}  +prompt+rawargs={s_pa:.4f}")
print(f"  prompt gain (large 로짓이 놓친 프롬프트 신호): {s_prm-s_logit:+.4f}")
print(f"  rawargs 순증분 (프롬프트 위에 원문args): {s_pa-s_prm:+.4f}")
print(f"  fold별 rawargs 순증분: {[round(x,4) for x in fg_args]}  (양수 {sum(x>0 for x in fg_args)}/5)")
go = (s_pa - s_prm >= 0.004) and (sum(x > 0 for x in fg_args) >= 4)
print(f"\n판정(rawargs specialist): {'GO(codex 옳음)' if go else 'DROP — args 잉여(내 반론)'}")
print(f"보조판정(prompt 신호 회복): {'프롬프트에 회복가능 신호 있음 → 클러스터 전용 텍스트분류기 검토' if s_prm-s_logit>=0.01 else '프롬프트도 large가 대부분 흡수'}")