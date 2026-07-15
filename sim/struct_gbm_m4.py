#!/usr/bin/env python3
"""풍부한 구조피처 GBM으로 M4 신호 실측: 트랜스포머(text)가 못뽑는 meta구조 신호가 있나?
전체상태(session_meta 분해·language_mix·open_files·last action·prompt키워드)로 HistGBM fold0 OOF.
판정: M4 F1이 트랜스포머 0.585 넘거나, 블렌드가 앙상블을 올리면 → 회수신호. 아니면 알레아토릭 확증."""
import json, csv, re, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
ROOT="/root/Action_Decision"
CLASSES=['read_file','grep_search','list_directory','glob_pattern','edit_file','write_file','apply_patch','run_bash','run_tests','lint_or_typecheck','ask_user','plan_task','web_search','respond_only']
CI={c:i for i,c in enumerate(CLASSES)}; M4=[CI[c] for c in ('read_file','grep_search','list_directory','glob_pattern')]
rows=[json.loads(l) for l in open(f'{ROOT}/data/train.jsonl')]
lab={}
with open(f'{ROOT}/data/train_labels.csv') as f:
    rd=csv.reader(f); next(rd)
    for r in rd: lab[r[0]]=r[1]
y=np.array([CI[lab[r['id']]] for r in rows])
sp=np.load(f'{ROOT}/splits/splits.npz',allow_pickle=True); tr,va=sp['tr0'],sp['va0']

TIER={'free':0,'pro':1,'enterprise':2}; LANG={'ko':0,'en':1,'mixed':2}; CIS={'passed':0,'failed':1,'none':2,'success':0}
EXTS=['py','js','ts','tsx','jsx','json','yaml','yml','md','txt','go','rs','java','css','html','sql','sh','toml','dockerfile','cfg']
KW=['open','read','show','find','search','grep','look','list','directory','folder','file','glob','pattern','all','test','run','edit','fix','change','write','create','commit','lint','check','plan','ask','web','build','너','파일','열어','읽어','찾아','목록','수정','실행','테스트']
_GLOB=re.compile(r'[*?\[\]]')
def feats(r):
    sm=r.get('session_meta',{}); ws=sm.get('workspace',{}); lm=ws.get('language_mix',{}) or {}
    p=(r.get('current_prompt') or '').lower()
    hist=r.get('history') or []
    # last action
    la=None
    for h in reversed(hist):
        if h.get('role')=='assistant_action': la=h; break
    la_name=la.get('name','') if la else ''
    la_args=la.get('args',{}) if la else {}
    la_argstr=' '.join(f'{k}={v}' for k,v in (la_args or {}).items())
    rs=(la.get('result_summary','') if la else '').lower()
    of=ws.get('open_files') or []
    f=[]
    f.append(TIER.get(sm.get('user_tier'),0))
    f.append(LANG.get(sm.get('language_pref'),1))
    f.append(sm.get('budget_tokens_remaining',0)/1e5)
    f.append(sm.get('turn_index',0))
    f.append(sm.get('elapsed_session_sec',0)/1000)
    f.append(ws.get('loc',0)/1e4)
    f.append(int(ws.get('git_dirty',False)))
    f.append(CIS.get(ws.get('last_ci_status'),2))
    f.append(len(of))
    # language_mix 분해 (top20 ext 비율)
    for e in EXTS: f.append(lm.get(e,0.0))
    # open_files 확장자 존재
    ofext=set()
    for x in of:
        m=re.search(r'\.([a-z0-9]+)$',x.lower())
        if m: ofext.add(m.group(1))
    for e in EXTS[:10]: f.append(int(e in ofext))
    # last action
    f.append(CI.get(la_name,-1))
    for tok in ['pass','fail','error','not found','permission','match','no match','0 ','ok']:
        f.append(int(tok in rs))
    f.append(int(bool(_GLOB.search(la_argstr))))
    f.append(int('/' in la_argstr))
    f.append(len(la_argstr))
    # history 통계
    seq=[h.get('name') for h in hist if h.get('role')=='assistant_action']
    f.append(len(seq))
    for c in CLASSES: f.append(seq.count(c))
    f.append(seq.count(la_name) if la_name else 0)
    # prompt
    f.append(len(p)); f.append(int(bool(re.search(r'[가-힣]',p))))
    f.append(int(bool(_GLOB.search(r.get('current_prompt') or ''))))
    f.append(int(bool(re.search(r'\.[a-z]{1,4}\b',p))))  # 확장자 언급
    for k in KW: f.append(int(k in p))
    return f
print("피처 추출...")
X=np.array([feats(r) for r in rows],dtype=np.float32)
print(f"X {X.shape}")
clf=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.08,max_depth=8,l2_regularization=1.0,random_state=0)
clf.fit(X[tr],y[tr])
proba=clf.predict_proba(X[va])
full=np.zeros((len(va),14))
for j,c in enumerate(clf.classes_): full[:,c]=proba[:,j]
gbm_pred=full.argmax(1); yv=y[va]
def f1blk(pred,blk):
    f=[]
    for c in blk:
        tp=int(((pred==c)&(yv==c)).sum());fp=int(((pred==c)&(yv!=c)).sum());fn=int(((pred!=c)&(yv==c)).sum())
        pr=tp/(tp+fp) if tp+fp else 0;rc=tp/(tp+fn) if tp+fn else 0
        f.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    return np.mean(f)
print(f"\n=== 구조GBM fold0-val ===")
print(f"  GBM M4avg={f1blk(gbm_pred,M4):.4f}  전체macro={f1blk(gbm_pred,list(range(14))):.4f}")
print(f"  (트랜스포머 m1 M4 0.585 / 배포캐스케이드 M4 0.606 대비)")
# 트랜스포머와 블렌드
m1=np.load(f'{ROOT}/work/m1_f0ckpt_rescue.npz',allow_pickle=True)['oof'][va].astype(float)
m1=np.clip(m1,1e-9,None); m1=m1/m1.sum(1,keepdims=True)
for w in (0.2,0.35,0.5):
    blend=((1-w)*m1+w*full).argmax(1)
    print(f"  blend w={w}: M4avg={f1blk(blend,M4):.4f} (m1단독 {f1blk(m1.argmax(1),M4):.4f})  전체macro={f1blk(blend,list(range(14))):.4f} (m1 {f1blk(m1.argmax(1),list(range(14))):.4f})")
print("[판정] GBM M4 >0.60 또는 blend가 m1 M4 넘으면 → 회수신호 / 아니면 알레아토릭+구조무익 확증")
