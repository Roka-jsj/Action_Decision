#!/usr/bin/env python3
"""확장 앙상블 탐욕탐색 (fold0-val, 올바른 배포 baseline 0.7702 대비).
다양한 학습완료 멤버를 조건부 캐스케이드에 추가해 macro 상승분 실측. 게이트 casc Δ≥+0.004."""
import numpy as np, json, csv, os
CLASSES=['read_file','grep_search','list_directory','glob_pattern','edit_file','write_file','apply_patch','run_bash','run_tests','lint_or_typecheck','ask_user','plan_task','web_search','respond_only']
CI={c:i for i,c in enumerate(CLASSES)}; M4=[CI[c] for c in ('read_file','grep_search','list_directory','glob_pattern')]
rows=[json.loads(l) for l in open('data/train.jsonl')]
lab={}
with open('data/train_labels.csv') as f:
    rd=csv.reader(f); next(rd)
    for r in rd: lab[r[0]]=r[1]
y=np.array([CI[lab[r['id']]] for r in rows])
sp=np.load('splits/splits.npz',allow_pickle=True); va=sp['va0']; yv=y[va]
BIAS=np.exp(np.array(json.load(open('packages/submit_th85/model/postproc.json'))['bias']))
def load(f):
    p='work/'+f
    if not os.path.exists(p): return None
    d=np.load(p,allow_pickle=True)
    if d['oof'].shape[0]!=70000: return None
    o=np.clip(d['oof'][va].astype(float),1e-9,None); return o/o.sum(1,keepdims=True)
M={n:load(f) for n,f in {
 'm1':'m1_f0ckpt_rescue.npz','mdeb12':'mdeb12ep_f0.npz','mdeb14':'teacher_mdeb14_f0.npz',
 'klue':'klue_f0.npz','sbwt':'teacher_sbwt_f0.npz','sbsp':'teacher_sbsp_f0.npz',
 'v12':'teacher_largev6_12ep_f0.npz','mdebsyn':'teacher_mdebsyn_f0.npz','mdebr':'teacher_mdebr_f0.npz',
 'v6r':'teacher_v6r_f0.npz'}.items()}
M={k:v for k,v in M.items() if v is not None}
print("로드 멤버:", list(M.keys()))
def sc(P):
    p=P.argmax(1); mac=[]
    for c in range(14):
        tp=int(((p==c)&(yv==c)).sum());fp=int(((p==c)&(yv!=c)).sum());fn=int(((p!=c)&(yv==c)).sum())
        pr=tp/(tp+fp) if tp+fp else 0;rc=tp/(tp+fn) if tp+fn else 0
        mac.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    return np.mean(mac)
def casc(members_w, th=0.85, m1=M['m1']):
    # m1 항상 + 저마진행만 나머지 가중혼합
    s=np.sort(m1,1); hi=(s[:,-1]-s[:,-2])>=th
    tot=sum(w for _,w in members_w)
    full=sum(w*p for p,w in members_w)/tot
    return np.where(hi[:,None],m1,full)*BIAS
# 배포 baseline
base=casc([(M['m1'],0.45),(M['mdeb12'],0.40),(M['klue'],0.15)])
bm=sc(base); print(f"\n배포 baseline macro={bm:.4f}")
# 탐욕: m1(0.45고정)+klue(0.15고정) 유지, 중간멤버 풀에서 조합
pool=['mdeb12','mdeb14','sbwt','sbsp','v12','mdebsyn','mdebr','v6r']
# 2~4개 중간멤버 균등, 총중간가중=0.40, m1=0.45, klue=0.15
from itertools import combinations
best=(bm,'baseline')
for k in (1,2,3,4):
    for combo in combinations(pool,k):
        wmid=0.40/k
        mw=[(M['m1'],0.45)]+[(M[c],wmid) for c in combo]+[(M['klue'],0.15)]
        for th in (0.80,0.85,0.90):
            m=sc(casc(mw,th))
            if m>best[0]: best=(m,f"{combo} th{th}")
print(f"best 앙상블 macro={best[0]:.4f}  Δ={best[0]-bm:+.4f}  ({best[1]})  LB추정(β0.36)≈{(best[0]-bm)*0.36:+.5f}")
# klue 가중도 탐색 + m1 가중
best2=(bm,'')
for combo in [('mdeb12','mdeb14'),('mdeb12','mdeb14','sbwt'),('mdeb12','mdeb14','v12'),('mdeb14','sbwt','v12'),('mdeb12','mdeb14','sbwt','v12')]:
    for w1 in (0.35,0.40,0.45):
        for wk in (0.12,0.15,0.20):
            wmid=(1-w1-wk)/len(combo)
            mw=[(M['m1'],w1)]+[(M[c],wmid) for c in combo]+[(M['klue'],wk)]
            for th in (0.82,0.85,0.88):
                m=sc(casc(mw,th))
                if m>best2[0]: best2=(m,f"m1={w1} {combo} klue={wk} th{th}")
print(f"best2 (가중탐색) macro={best2[0]:.4f} Δ={best2[0]-bm:+.4f} ({best2[1]}) LB추정≈{(best2[0]-bm)*0.36:+.5f}")
print("[게이트] casc Δ≥+0.004 → 제출가치 / 미만 → 신기루")
