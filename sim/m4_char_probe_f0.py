#!/usr/bin/env python3
"""codex char-expert 프로브 fold0 단독 — 결정적 판정.
raw features(full paths·args·result·구조토큰) char-ngram이 M4 F1을 base(m1) 위로 올리나?
판정선: fold0 M4 평균 F1 lift ≥+0.02 → 0.795경로 / <+0.005 → 환원불가 근접.
"""
import json, csv, re, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT="/root/Action_Decision"
CLASSES=['read_file','grep_search','list_directory','glob_pattern','edit_file','write_file',
 'apply_patch','run_bash','run_tests','lint_or_typecheck','ask_user','plan_task','web_search','respond_only']
CI={c:i for i,c in enumerate(CLASSES)}; M4=[CI[c] for c in ('read_file','grep_search','list_directory','glob_pattern')]

rows=[json.loads(l) for l in open(f"{ROOT}/data/train.jsonl")]
lab={}
with open(f"{ROOT}/data/train_labels.csv") as f:
    rd=csv.reader(f); next(rd)
    for r in rd: lab[r[0]]=r[1]
y=np.array([CI[lab[r['id']]] for r in rows])
sp=np.load(f"{ROOT}/splits/splits.npz",allow_pickle=True)
tr,va=sp["tr0"],sp["va0"]
base_oof=np.load(f"{ROOT}/work/m1_f0ckpt_rescue.npz",allow_pickle=True)["oof"].astype(np.float64)

_GLOB=re.compile(r"[*?]|\[[^\]]+\]"); _EXT=re.compile(r"\.([a-z0-9]{1,5})\b",re.I)
def pst(s):
    t=[]
    if _GLOB.search(s):t.append("HASGLOB")
    if "**" in s:t.append("RECGLOB")
    if re.search(r"[\\^$.|+()]",s):t.append("HASREGEX")
    if "/" in s:t.append(f"DEPTH{min(s.count('/'),6)}")
    for e in set(_EXT.findall(s)):t.append(f"EXT_{e.lower()}")
    if re.search(r"['\"`]",s):t.append("QUOTED")
    return t
def rawfeat(r):
    p=["CUR "+(r.get('current_prompt') or '')]
    for h in (r.get('history') or [])[-8:]:
        if h.get('role')=='user': p.append("U "+(h.get('content') or ''))
        elif h.get('role')=='assistant_action':
            nm=h.get('name',''); args=h.get('args') or {}
            argstr=" ".join(f"{k}={v}" for k,v in args.items())
            p.append(f"A {nm} {argstr} R {h.get('result_summary') or ''}"); p+=pst(argstr)
    of=(r.get('session_meta',{}).get('workspace',{}).get('open_files') or [])
    if of:
        p.append("OPEN "+" ".join(of))
        for x in of: p+=pst(x)
    p+=pst(r.get('current_prompt') or '')
    return " ".join(p)
texts=[rawfeat(r) for r in rows]

vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=3,max_features=120000,sublinear_tf=True)
Xtr=vec.fit_transform([texts[i] for i in tr]); Xva=vec.transform([texts[i] for i in va])
clf=LogisticRegression(max_iter=1000,C=3.0,class_weight='balanced',solver='saga',n_jobs=-1,tol=1e-3)
clf.fit(Xtr,y[tr])
P=clf.predict_proba(Xva); full=np.zeros((len(va),14))
for j,c in enumerate(clf.classes_): full[:,c]=P[:,j]
char_log=np.log(full+1e-9)

yv=y[va]; bp=base_oof[va].argmax(1); cp=char_log.argmax(1)
def m4f1(pred):
    f=[]
    for c in M4:
        tp=int(((pred==c)&(yv==c)).sum());fp=int(((pred==c)&(yv!=c)).sum());fn=int(((pred!=c)&(yv==c)).sum())
        pr=tp/(tp+fp) if tp+fp else 0;rc=tp/(tp+fn) if tp+fn else 0
        f.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    return np.mean(f),f
def softmax(z):z=z-z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)
pb=softmax(base_oof[va]);pc=softmax(char_log)

bf,bfl=m4f1(bp); cf,cfl=m4f1(cp)
print(f"[fold0 val n={len(va)}]")
print(f"base(m1)    M4avg={bf:.4f} per={[round(x,3) for x in bfl]}  (list,read,grep,glob순 아님: read,grep,list,glob)")
print(f"char-expert M4avg={cf:.4f} per={[round(x,3) for x in cfl]}")
mm=np.array(M4)
for w in (0.3,0.5,0.7):
    bpn=bp.copy(); ism4=np.isin(bp,M4); mix=(1-w)*pb+w*pc
    reM=mm[mix[:,mm].argmax(1)]; bpn[ism4]=reM[ism4]
    bff,_=m4f1(bpn)
    print(f"blend w={w}  M4avg={bff:.4f}  Δbase={bff-bf:+.4f}  macroΔ≈{(bff-bf)*4/14:+.4f}  LB추정(β0.82)≈{(bff-bf)*4/14*0.82:+.5f}")
print(f"[판정선] M4 lift ≥+0.02 → 0.795경로 / <+0.005 → 환원불가")
