#!/usr/bin/env python3
"""char-expert fold0 결정타(빠름): raw char-ngram(LinearSVC)이 M4를 v6트랜스포머(0.59)보다 잘 뽑나?"""
import json, csv, re, numpy as np, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

ROOT="/root/Action_Decision"
CLASSES=['read_file','grep_search','list_directory','glob_pattern','edit_file','write_file','apply_patch','run_bash','run_tests','lint_or_typecheck','ask_user','plan_task','web_search','respond_only']
CI={c:i for i,c in enumerate(CLASSES)}; M4=[CI[c] for c in ('read_file','grep_search','list_directory','glob_pattern')]
t0=time.time()
rows=[json.loads(l) for l in open(f"{ROOT}/data/train.jsonl")]
lab={}
with open(f"{ROOT}/data/train_labels.csv") as f:
    rd=csv.reader(f); next(rd)
    for r in rd: lab[r[0]]=r[1]
y=np.array([CI[lab[r['id']]] for r in rows])
sp=np.load(f"{ROOT}/splits/splits.npz",allow_pickle=True); tr,va=sp["tr0"],sp["va0"]
base=np.load(f"{ROOT}/work/m1_f0ckpt_rescue.npz",allow_pickle=True)["oof"]

_GLOB=re.compile(r"[*?]|\[[^\]]+\]"); _EXT=re.compile(r"\.([a-z0-9]{1,5})\b",re.I)
def pst(s):
    t=[]
    if _GLOB.search(s):t.append("HASGLOB")
    if "**" in s:t.append("RECGLOB")
    if re.search(r"[\\^$.|+()]",s):t.append("HASREGEX")
    if "/" in s:t.append(f"DEPTH{min(s.count('/'),6)}")
    for e in set(_EXT.findall(s)):t.append("EXT_"+e.lower())
    if re.search(r"['\"`]",s):t.append("QUOTED")
    return t
def rf(r):
    p=["CUR "+(r.get('current_prompt') or '')]
    for h in (r.get('history') or [])[-8:]:
        if h.get('role')=='user': p.append("U "+(h.get('content') or ''))
        elif h.get('role')=='assistant_action':
            nm=h.get('name',''); a=h.get('args') or {}
            astr=" ".join(f"{k}={v}" for k,v in a.items())
            p.append(f"A {nm} {astr} R {h.get('result_summary') or ''}"); p+=pst(astr)
    of=(r.get('session_meta',{}).get('workspace',{}).get('open_files') or [])
    if of:
        p.append("OPEN "+" ".join(of))
        for x in of: p+=pst(x)
    p+=pst(r.get('current_prompt') or '')
    return " ".join(p)
texts=[rf(r) for r in rows]; print(f"feat build {time.time()-t0:.0f}s")

vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=3,max_features=80000,sublinear_tf=True)
Xtr=vec.fit_transform([texts[i] for i in tr]); Xva=vec.transform([texts[i] for i in va])
print(f"vectorize {time.time()-t0:.0f}s  vocab={len(vec.vocabulary_)}")
clf=LinearSVC(C=0.5,class_weight='balanced'); clf.fit(Xtr,y[tr])
print(f"fit {time.time()-t0:.0f}s")
dec=clf.decision_function(Xva)  # (n,14) scores
full=np.full((len(va),14),-1e9)
for j,c in enumerate(clf.classes_): full[:,c]=dec[:,j]

yv=y[va]; bp=base[va].argmax(1); cp=full.argmax(1)
def m4f1(pred):
    f=[]
    for c in M4:
        tp=int(((pred==c)&(yv==c)).sum());fp=int(((pred==c)&(yv!=c)).sum());fn=int(((pred!=c)&(yv==c)).sum())
        pr=tp/(tp+fp) if tp+fp else 0;rc=tp/(tp+fn) if tp+fn else 0
        f.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    return np.mean(f),f
bf,bfl=m4f1(bp); cf,cfl=m4f1(cp)
print(f"\n[fold0 val n={len(va)}]  (read,grep,list,glob)")
print(f"base(m1 v6)  M4avg={bf:.4f} per={[round(x,3) for x in bfl]}")
print(f"char-expert  M4avg={cf:.4f} per={[round(x,3) for x in cfl]}   Δalone={cf-bf:+.4f}")
# blend: base가 M4후보인 행만 M4 4클래스 내부에서 base+char 점수합(정규화 후)
def norm(z):
    z=z.copy(); mn=z[:,M4].min(1,keepdims=True); mx=z[:,M4].max(1,keepdims=True)
    return (z-mn)/(mx-mn+1e-9)
bn=norm(base[va].astype(float)); cn=norm(full)
mm=np.array(M4)
for w in (0.3,0.5,0.7):
    bpn=bp.copy(); ism4=np.isin(bp,M4); mix=(1-w)*bn+w*cn
    re_=mm[mix[:,mm].argmax(1)]; bpn[ism4]=re_[ism4]
    bff,bffl=m4f1(bpn)
    print(f"blend w={w}  M4avg={bff:.4f}  Δbase={bff-bf:+.4f}  macroΔ≈{(bff-bf)*4/14:+.4f}  LB(β0.82)≈{(bff-bf)*4/14*0.82:+.5f}")
print("[판정선] M4 lift ≥+0.02 → 0.795경로 생존 / <+0.005 → 환원불가 근접")
