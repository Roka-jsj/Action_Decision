#!/usr/bin/env python3
"""goal-rescue 캐스케이드 게이트: m1을 affected행서 rescue한 뒤 배포 조건부캐스케이드 fold0 macro Δ."""
import os,sys,json,csv,numpy as np,time
sys.path.insert(0,'/root/Action_Decision/common'); import ad_lib
from transformers import AutoTokenizer
ROOT="/root/Action_Decision"; os.chdir(ROOT)
CLASSES=ad_lib.CLASSES; CI={c:i for i,c in enumerate(CLASSES)}
M4=[CI[c] for c in ('read_file','grep_search','list_directory','glob_pattern')]
CKPT=f"{ROOT}/work/foldckpt_largev6_f0ckpt_f0"; t0=time.time()
rows=[json.loads(l) for l in open('data/train.jsonl')]
lab={}
with open('data/train_labels.csv') as f:
    rd=csv.reader(f);next(rd)
    for r in rd: lab[r[0]]=r[1]
y=np.array([CI[lab[r['id']]] for r in rows]); sp=np.load('splits/splits.npz',allow_pickle=True); va=sp['va0']; yv=y[va]
BIAS=np.exp(np.array(json.load(open('packages/submit_th85/model/postproc.json'))['bias']))
tok=AutoTokenizer.from_pretrained(CKPT,local_files_only=True); tok.truncation_side="left"
KEEP=320-tok.num_special_tokens_to_add(False)
def first_user(r):
    for h in (r.get('history') or []):
        if h.get('role')=='user': return h.get('content') or ''
    return ''
def build(r):
    base=ad_lib.serialize(r,'v6',8); ids=tok(base,add_special_tokens=False)['input_ids']
    if len(ids)<=KEEP: return None
    fu=first_user(r)
    if not fu: return None
    if fu[:30] in tok.decode(ids[-KEEP:]): return None
    hpos=base.find(" [HIST]"); header=base[:hpos] if hpos>0 else ""
    pref=tok(header+f" [HIST] u: {fu[:150]}",add_special_tokens=False)['input_ids']
    budget=KEEP-len(pref)
    if budget<20: pref=pref[:KEEP-20]; budget=20
    return pref+ids[-budget:]
va_rows=[rows[i] for i in va]
aff=[i for i,r in enumerate(va_rows) if build(r) is not None]
print(f"affected {len(aff)} ({len(aff)/len(va)*100:.1f}%)  {time.time()-t0:.0f}s")
# m1 base OOF(fold0-val)
m1_oof=np.load('work/m1_f0ckpt_rescue.npz',allow_pickle=True)['oof'][va].astype(float)
m1_oof=np.clip(m1_oof,1e-9,None); m1_base=m1_oof/m1_oof.sum(1,keepdims=True)
# affected행만 rescue 재추론
sub=[va_rows[i] for i in aff]; resc_texts=[tok.decode(build(va_rows[i])) for i in aff]
Pr=ad_lib.predict_logits(CKPT,sub,version='v6',max_len=320,texts=resc_texts,return_probs=True,device='cuda')
Pr=np.clip(Pr,1e-9,None); Pr=Pr/Pr.sum(1,keepdims=True)
m1_resc=m1_base.copy(); m1_resc[aff]=Pr
# m2,m3
def load(f):
    d=np.load('work/'+f,allow_pickle=True); o=np.clip(d['oof'][va].astype(float),1e-9,None); return o/o.sum(1,keepdims=True)
mdeb=load('mdeb12ep_f0.npz'); klue=load('klue_f0.npz')
def casc(m1p):
    s=np.sort(m1p,1); hi=(s[:,-1]-s[:,-2])>=0.85
    full=(0.45*m1p+0.40*mdeb+0.15*klue)
    return (np.where(hi[:,None],m1p,full)*BIAS).argmax(1)
def mac(pred):
    f=[]
    for c in range(14):
        tp=int(((pred==c)&(yv==c)).sum());fp=int(((pred==c)&(yv!=c)).sum());fn=int(((pred!=c)&(yv==c)).sum())
        pr=tp/(tp+fp) if tp+fp else 0;rc=tp/(tp+fn) if tp+fn else 0
        f.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    return np.mean(f)
bm=mac(casc(m1_base)); rm=mac(casc(m1_resc))
print(f"\n=== goal-rescue 캐스케이드 fold0 게이트 (m1만 rescue) ===")
print(f"  baseline macro={bm:.4f}   goal-rescue macro={rm:.4f}   Δ={rm-bm:+.4f}")
print(f"  β0.82→LB{(rm-bm)*0.82:+.5f}  β0.36→LB{(rm-bm)*0.36:+.5f}   (컷갭 +0.0028)")
# m1-solo 전체 fold0도
def solomac(p):
    pred=(p*BIAS).argmax(1); return mac(pred)
print(f"  (m1-solo 전체 fold0: {mac(m1_base.argmax(1)):.4f}→{mac(m1_resc.argmax(1)):.4f} Δ{mac(m1_resc.argmax(1))-mac(m1_base.argmax(1)):+.4f})")
print(f"[게이트] casc Δ≥+0.004 → GEN-rescue급, m2/m3도 rescue시 더 큼. 조립·제출 후보")
