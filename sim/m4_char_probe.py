#!/usr/bin/env python3
"""codex char-expert 프로브: v6가 버리는 raw신호(full paths·args·result·path/pattern 구조토큰)를
char-ngram이 회수해 M4 F1을 올리나? 같은 세션폴드 OOF로 base(m1)와 leak-free 비교.

판정선(codex): fold0 M4 평균 F1 lift ≥+0.02면 0.795 경로 생존, <+0.005면 원본기준도 환원불가.
"""
import json, csv, re, os, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = "/root/Action_Decision"
CLASSES = ['read_file','grep_search','list_directory','glob_pattern','edit_file','write_file',
           'apply_patch','run_bash','run_tests','lint_or_typecheck','ask_user','plan_task','web_search','respond_only']
CI = {c:i for i,c in enumerate(CLASSES)}
M4 = [CI[c] for c in ('read_file','grep_search','list_directory','glob_pattern')]
M4set = set(M4)

# --- load ---
rows = [json.loads(l) for l in open(f"{ROOT}/data/train.jsonl")]
lab = {}
with open(f"{ROOT}/data/train_labels.csv") as f:
    rd = csv.reader(f); next(rd)
    for r in rd: lab[r[0]] = r[1]
y = np.array([CI[lab[r['id']]] for r in rows])

sp = np.load(f"{ROOT}/splits/splits.npz", allow_pickle=True)
folds = [(sp[f"tr{i}"], sp[f"va{i}"]) for i in range(int(sp["n_splits"]))]

base_oof = np.load(f"{ROOT}/work/m1_f0ckpt_rescue.npz", allow_pickle=True)["oof"].astype(np.float64)  # (70000,14) logits/probs

# --- raw feature string (codex 스펙: v6가 버리는 것 전부) ---
_GLOB = re.compile(r"[*?]|\[[^\]]+\]")
_EXT = re.compile(r"\.([a-z0-9]{1,5})\b", re.I)
def path_struct_tokens(s):
    t = []
    if _GLOB.search(s): t.append("HASGLOB")
    if "**" in s: t.append("RECGLOB")
    if re.search(r"[\\^$.|+()]", s): t.append("HASREGEX")
    if "/" in s: t.append(f"DEPTH{min(s.count('/'),6)}")
    for e in set(_EXT.findall(s)): t.append(f"EXT_{e.lower()}")
    if re.search(r"['\"`]", s): t.append("QUOTED")
    return t

def raw_feat(r):
    parts = []
    parts.append("CUR " + (r.get('current_prompt') or ''))
    hist = r.get('history') or []
    # 최근 user 전문 + 최근 액션 full(name/args/result_summary)
    for h in hist[-8:]:
        if h.get('role') == 'user':
            parts.append("U " + (h.get('content') or ''))
        elif h.get('role') == 'assistant_action':
            nm = h.get('name',''); args = h.get('args') or {}
            argstr = " ".join(f"{k}={v}" for k,v in args.items())
            rs = h.get('result_summary') or ''
            parts.append(f"A {nm} {argstr} R {rs}")
            parts += path_struct_tokens(argstr)
    # open_files 전체경로
    of = (r.get('session_meta',{}).get('workspace',{}).get('open_files') or [])
    if of:
        parts.append("OPEN " + " ".join(of))
        for p in of: parts += path_struct_tokens(p)
    # current_prompt 구조토큰
    parts += path_struct_tokens(r.get('current_prompt') or '')
    return " ".join(parts)

texts = [raw_feat(r) for r in rows]

# --- 5-fold OOF: char 3-5gram TFIDF + LogReg ---
char_oof = np.full((len(rows), 14), -1e9)
covered = np.zeros(len(rows), bool)
for fi,(tr,va) in enumerate(folds):
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2, max_features=200000, sublinear_tf=True)
    Xtr = vec.fit_transform([texts[i] for i in tr])
    Xva = vec.transform([texts[i] for i in va])
    clf = LogisticRegression(max_iter=200, C=4.0, class_weight='balanced', n_jobs=-1)
    clf.fit(Xtr, y[tr])
    P = clf.predict_proba(Xva)  # (n, n_classes_present)
    full = np.zeros((len(va),14))
    for j,c in enumerate(clf.classes_): full[:,c] = P[:,j]
    char_oof[va] = np.log(full+1e-9)
    covered[va] = True
    print(f"fold{fi}: train {len(tr)} val {len(va)} done")

# --- 평가 (dev 커버행만) ---
def m4_f1(pred, yv, mask):
    f1s=[]
    for c in M4:
        tp=int(((pred==c)&(yv==c)&mask).sum()); fp=int(((pred==c)&(yv!=c)&mask).sum()); fn=int(((pred!=c)&(yv==c)&mask).sum())
        pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0
        f1s.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    return np.mean(f1s), f1s

def macro_f1(pred, yv, mask):
    f1s=[]
    for c in range(14):
        tp=int(((pred==c)&(yv==c)&mask).sum()); fp=int(((pred==c)&(yv!=c)&mask).sum()); fn=int(((pred!=c)&(yv==c)&mask).sum())
        pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0
        f1s.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    return np.mean(f1s)

mask = covered
base_pred = base_oof.argmax(1)
char_pred = char_oof.argmax(1)

# base softmax for blend
def softmax(z):
    z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
pb = softmax(base_oof); pc = softmax(char_oof)

print("\n=== M4 4클래스 평균 F1 (dev OOF, leak-free) ===")
bf,bfl = m4_f1(base_pred,y,mask); print(f"base(m1)      M4avg={bf:.4f}  per={[round(x,3) for x in bfl]}")
cf,cfl = m4_f1(char_pred,y,mask); print(f"char-expert   M4avg={cf:.4f}  per={[round(x,3) for x in cfl]}")

# blend: base가 M4후보(argmax∈M4)인 행만, M4 4클래스 내부에서 base+char 혼합
for w in (0.3,0.5,0.7):
    blend_pred = base_pred.copy()
    is_m4 = np.isin(base_pred, M4)
    mix = (1-w)*pb + w*pc
    # M4 내부 재argmax (M4 클래스 중에서만)
    m4cols = np.array(M4)
    m4_re = m4cols[mix[:,m4cols].argmax(1)]
    blend_pred[is_m4] = m4_re[is_m4]
    bff,_ = m4_f1(blend_pred,y,mask)
    print(f"blend w={w}   M4avg={bff:.4f}  (Δvs base {bff-bf:+.4f})   macroΔ≈{(bff-bf)*4/14:+.4f}")

print(f"\n[판정선] fold M4 lift ≥+0.02 → 0.795경로 생존 / <+0.005 → 환원불가 근접")
print(f"[전체 macro] base={macro_f1(base_pred,y,mask):.4f} char={macro_f1(char_pred,y,mask):.4f}")
