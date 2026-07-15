#!/usr/bin/env python3
"""mdeb14 멤버 스왑 leak-free 게이트: 배포 조건부 캐스케이드(m1 항상+저마진행 m2,m3)로
baseline(deployed m2) vs mdeb14-swap fold0-val macro/M4 정확 비교."""
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
def sm(z):
    z=z.astype(float); z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
def load(f):
    d=np.load('work/'+f,allow_pickle=True); return sm(d['oof'])[va]
m1=load('m1_f0ckpt_rescue.npz'); mdeb=load('teacher_mdeb_f0.npz'); mdeb14=load('teacher_mdeb14_f0.npz'); klue=load('kfdeb_f0.npz')

def cascade(pm1,pm2,pm3,w=(0.45,0.40,0.15),th=0.85):
    # margin_th 조건부: m1 항상, 저마진(top1-top2<th)행만 m2,m3 추가
    s=np.sort(pm1,1); margin=s[:,-1]-s[:,-2]
    hi = margin>=th
    full = (w[0]*pm1+w[1]*pm2+w[2]*pm3)/sum(w)
    out=np.where(hi[:,None], pm1, full)
    return out
def scores(P):
    p=P.argmax(1); mac=[];m4=[]
    for c in range(14):
        tp=int(((p==c)&(yv==c)).sum());fp=int(((p==c)&(yv!=c)).sum());fn=int(((p!=c)&(yv==c)).sum())
        pr=tp/(tp+fp) if tp+fp else 0;rc=tp/(tp+fn) if tp+fn else 0
        v=2*pr*rc/(pr+rc) if pr+rc else 0; mac.append(v)
        if c in M4:m4.append(v)
    return np.mean(mac),np.mean(m4)

base=cascade(m1,mdeb,klue)
swap=cascade(m1,mdeb14,klue)
bm,b4=scores(base); sm_,s4=scores(swap)
print("=== 배포 조건부 캐스케이드 (fold0-val, leak-free) ===")
print(f"baseline (m1,mdeb,klue)   macro={bm:.4f}  M4={b4:.4f}")
print(f"mdeb14-swap              macro={sm_:.4f}  M4={s4:.4f}")
print(f"Δ macro = {sm_-bm:+.4f}   Δ M4 = {s4-b4:+.4f}   LB추정(β0.36)≈{(sm_-bm)*0.36:+.5f}")
# 4멤버(mdeb+mdeb14 둘다)도
def cascade4(pm1,pa,pb,pk,w=(0.40,0.22,0.23,0.15),th=0.85):
    s=np.sort(pm1,1); margin=s[:,-1]-s[:,-2]; hi=margin>=th
    full=(w[0]*pm1+w[1]*pa+w[2]*pb+w[3]*pk)/sum(w)
    return np.where(hi[:,None],pm1,full)
c4=cascade4(m1,mdeb,mdeb14,klue); c4m,c44=scores(c4)
print(f"4멤버(m1,mdeb,mdeb14,klue) macro={c4m:.4f} M4={c44:.4f} Δ={c4m-bm:+.4f} LB추정≈{(c4m-bm)*0.36:+.5f}")
# 가중 그리드 (mdeb14 스왑 최적 th/w)
print("\n=== mdeb14-swap 가중/th 미세탐색 ===")
best=(bm,None)
for th in (0.75,0.85,0.90):
    for w1 in (0.40,0.45,0.50):
        w=(w1,0.85-w1+0.15-0.15+ (0.40) , 0.15)  # placeholder
        pass
for th in (0.70,0.80,0.85,0.90):
    for w in [(0.45,0.40,0.15),(0.40,0.45,0.15),(0.50,0.35,0.15),(0.40,0.40,0.20)]:
        m,_=scores(cascade(m1,mdeb14,klue,w,th))
        if m>best[0]: best=(m,(th,w))
print(f"baseline={bm:.4f}  best mdeb14-swap={best[0]:.4f} @ {best[1]}  Δ={best[0]-bm:+.4f} LB추정(β0.36)≈{(best[0]-bm)*0.36:+.5f}")
print("[게이트] fold0 casc Δ≥+0.004 → 제출가치(β0.36시 LB+0.0014+). 컷갭 +0.0028")
