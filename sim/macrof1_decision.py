#!/usr/bin/env python3
"""리서치 레버#1: macro-F1 최적 결정규칙(GFM/per-class threshold)이 배포 bias-argmax를 넘나?
그리고 OOS 전이하나? 배포 캐스케이드 posterior에 결정규칙만 교체(모델 불변)."""
import json,csv,numpy as np
ROOT="/root/Action_Decision"
CLASSES=['read_file','grep_search','list_directory','glob_pattern','edit_file','write_file','apply_patch','run_bash','run_tests','lint_or_typecheck','ask_user','plan_task','web_search','respond_only']
CI={c:i for i,c in enumerate(CLASSES)}
rows=[json.loads(l) for l in open(f'{ROOT}/data/train.jsonl')]
lab={}
with open(f'{ROOT}/data/train_labels.csv') as f:
    rd=csv.reader(f); next(rd)
    for r in rd: lab[r[0]]=r[1]
y=np.array([CI[lab[r['id']]] for r in rows])
sp=np.load(f'{ROOT}/splits/splits.npz',allow_pickle=True); va=sp['va0']; yv=y[va]
BIAS=np.array(json.load(open(f'{ROOT}/packages/submit_th85/model/postproc.json'))['bias'])
def load(f):
    d=np.load(f'{ROOT}/work/'+f,allow_pickle=True); o=np.clip(d['oof'][va].astype(float),1e-9,None); return o/o.sum(1,keepdims=True)
m1=load('m1_f0ckpt_rescue.npz'); mdeb=load('mdeb12ep_f0.npz'); klue=load('klue_f0.npz')
s=np.sort(m1,1); hi=(s[:,-1]-s[:,-2])>=0.85
P=np.where(hi[:,None],m1,(0.45*m1+0.40*mdeb+0.15*klue))  # 캐스케이드 posterior (bias 전)
logP=np.log(P+1e-12)

def macro(pred,yy):
    f=[]
    for c in range(14):
        tp=int(((pred==c)&(yy==c)).sum());fp=int(((pred==c)&(yy!=c)).sum());fn=int(((pred!=c)&(yy==c)).sum())
        pr=tp/(tp+fp) if tp+fp else 0;rc=tp/(tp+fn) if tp+fn else 0
        f.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    return np.mean(f)

# 배포: bias-argmax
base_pred=(logP+BIAS).argmax(1)
print(f"배포 bias-argmax macro={macro(base_pred,yv):.4f}")
print(f"순수 argmax(bias없음) macro={macro(logP.argmax(1),yv):.4f}")

# 레버: 철저한 per-class additive offset 좌표상승 (다중시작), fit on idx, eval on idx
def opt_offset(lp,yy,iters=5,starts=(0,)):
    best_b=None; best_m=-1
    for st in starts:
        b=np.full(14,float(st))
        for _ in range(iters):
            for c in range(14):
                bb=b.copy(); bestc=(-1,b[c])
                for d in np.linspace(-4,4,81):
                    bb[c]=d; m=macro((lp+bb).argmax(1),yy)
                    if m>bestc[0]: bestc=(m,d)
                b[c]=bestc[1]
        m=macro((lp+b).argmax(1),yy)
        if m>best_m: best_m,best_b=m,b
    return best_b,best_m

# GFM-style: 기대 macro-F1 극대화 (per-class threshold rule, F1_c/2)
def gfm_rule(lp,yy_for_f1star=None,iters=8):
    # posterior probs
    p=np.exp(lp-lp.max(1,keepdims=True)); p=p/p.sum(1,keepdims=True)
    # 초기 F1* = bias-argmax 기준
    pred=(lp+BIAS).argmax(1)
    for _ in range(iters):
        # 현재 예측의 per-class F1*
        F=np.zeros(14)
        for c in range(14):
            tp=((pred==c)&(y[va]==c)).sum() if yy_for_f1star is not None else 0
        # threshold t_c = F1*_c/2 대신, 실측 좌표상승으로 대체(단일라벨엔 offset가 등가)
        break
    return pred

# in-sample 상한
b_in,m_in=opt_offset(logP,yv,starts=(0,-1,-2))
print(f"\n[in-sample] 철저 offset최적 macro={m_in:.4f}  Δ배포={m_in-macro(base_pred,yv):+.4f}")

# OOS 전이: A서 fit, B서 eval (핵심)
rng=np.random.RandomState(0); perm=rng.permutation(len(va)); A=perm[:len(va)//2]; B=perm[len(va)//2:]
bA,_=opt_offset(logP[A],yv[A],starts=(0,-1,-2))
base_B=macro((logP[B]+BIAS).argmax(1),yv[B])
opt_B=macro((logP[B]+bA).argmax(1),yv[B])
print(f"[OOS] A서 결정규칙 fit → B 평가: 배포bias {base_B:.4f} → A학습규칙 {opt_B:.4f}  Δ={opt_B-base_B:+.4f}")
print(f"      ← OOS Δ가 양수·안정이면 결정규칙 레버 생존. 음/0이면 in-sample 과적합(전이불가)")
# 5-fold 교차로 안정성
deltas=[]
for seed in range(5):
    rng=np.random.RandomState(seed); pm=rng.permutation(len(va)); a=pm[:len(va)//2]; bb=pm[len(va)//2:]
    ba,_=opt_offset(logP[a],yv[a],iters=3,starts=(0,-1))
    deltas.append(macro((logP[bb]+ba).argmax(1),yv[bb])-macro((logP[bb]+BIAS).argmax(1),yv[bb]))
print(f"[OOS×5seed] Δ평균={np.mean(deltas):+.4f} 범위[{min(deltas):+.4f},{max(deltas):+.4f}]")
print(f"[판정] OOS Δ 안정 양수(≥+0.002) → 결정규칙 레버, 조립·제출 / ~0·음 → 배포bias 이미최적, 死")
