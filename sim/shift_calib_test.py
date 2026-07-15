#!/usr/bin/env python3
"""shift-주입 보정 테스트: 홀드아웃에 측정된 히든 label-shift를 주입한 뒤,
히든분포-최적 bias가 배포 bias를 이기나? (damped-bias는 shift없는 홀드아웃서 검증→무효 의심)
이기면 → 투과적 보정이 히든서 전이 가능 → 프로브+제출 가치. 아니면 死."""
import numpy as np, json, csv
CLASSES=['read_file','grep_search','list_directory','glob_pattern','edit_file','write_file','apply_patch','run_bash','run_tests','lint_or_typecheck','ask_user','plan_task','web_search','respond_only']
CI={c:i for i,c in enumerate(CLASSES)}
rows=[json.loads(l) for l in open('data/train.jsonl')]
lab={}
with open('data/train_labels.csv') as f:
    rd=csv.reader(f); next(rd)
    for r in rd: lab[r[0]]=r[1]
y=np.array([CI[lab[r['id']]] for r in rows])
sp=np.load('splits/splits.npz',allow_pickle=True)
dev=sp['va0']  # fold0-val만 (m1/mdeb/klue 전부 leak-free)
BIAS=np.array(json.load(open('packages/submit_th85/model/postproc.json'))['bias'])
def load(f):
    d=np.load('work/'+f,allow_pickle=True); o=np.clip(d['oof'].astype(float),1e-9,None); return o/o.sum(1,keepdims=True)
m1=load('m1_f0ckpt_rescue.npz'); mdeb=load('mdeb12ep_f0.npz'); klue=load('klue_f0.npz')
# 배포 조건부 캐스케이드 probs (bias 전)
def cascade_probs(w=(0.45,0.40,0.15),th=0.85):
    s=np.sort(m1,1); hi=(s[:,-1]-s[:,-2])>=th
    full=(w[0]*m1+w[1]*mdeb+w[2]*klue)/sum(w)
    return np.where(hi[:,None],m1,full)
P=cascade_probs()[dev]; yv=y[dev]; logP=np.log(P+1e-12)
N=len(yv)

# 측정된 히든 shift (probe_3shot, 절대빈도)
train_freq=np.bincount(yv,minlength=14)/N
hidden_freq=train_freq.copy()
hidden_freq[CI['read_file']]=0.1393; hidden_freq[CI['glob_pattern']]=0.0673; hidden_freq[CI['list_directory']]=0.0634
# 나머지는 train비율 유지하며 잔여질량 재분배
known=[CI['read_file'],CI['glob_pattern'],CI['list_directory']]
rem_mass=1-hidden_freq[known].sum(); rest=[i for i in range(14) if i not in known]
tr_rest=train_freq[rest]/train_freq[rest].sum()
for i,c in zip(range(len(rest)),rest): hidden_freq[c]=rem_mass*tr_rest[i]
print("측정 shift (pp):", {CLASSES[i]:round((hidden_freq[i]-train_freq[i])*100,2) for i in known})

def macro_f1_weighted(pred, yv, wt):
    f=[]
    for c in range(14):
        tp=(wt*((pred==c)&(yv==c))).sum(); fp=(wt*((pred==c)&(yv!=c))).sum(); fn=(wt*((pred!=c)&(yv==c))).sum()
        pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0
        f.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    return np.mean(f)
# importance weight으로 히든분포 주입 (라벨별 w = π_hidden/π_train)
w_shift=(hidden_freq/np.maximum(train_freq,1e-9))[yv]

def opt_bias(logP, yv, wt, iters=3):
    b=np.zeros(14)
    for _ in range(iters):
        for c in range(14):
            best=(-1,0)
            for delta in np.linspace(-3,3,61):
                bb=b.copy(); bb[c]=delta
                pred=(logP+bb).argmax(1)
                m=macro_f1_weighted(pred,yv,wt)
                if m>best[0]: best=(m,delta)
            b[c]=best[1]
    return b

# 1) 배포 bias 성능 (히든분포 평가)
base_shift=macro_f1_weighted((logP+BIAS).argmax(1), yv, w_shift)
noB_shift=macro_f1_weighted(logP.argmax(1), yv, w_shift)
# 2) 히든분포-최적 bias
b_hidden=opt_bias(logP,yv,w_shift)
opt_shift=macro_f1_weighted((logP+b_hidden).argmax(1), yv, w_shift)
# 3) 대조: shift없는(train) 평가에서도
w_none=np.ones(N)
base_none=macro_f1_weighted((logP+BIAS).argmax(1), yv, w_none)
b_train=opt_bias(logP,yv,w_none)
opt_none=macro_f1_weighted((logP+b_train).argmax(1), yv, w_none)

print(f"\n=== shift주입(히든분포) 평가 ===")
print(f"  배포bias      macro={base_shift:.4f}")
print(f"  히든최적bias  macro={opt_shift:.4f}   Δ={opt_shift-base_shift:+.4f}  ← 이게 양수면 투과적보정 전이가능")
print(f"=== 대조: shift없음(train분포) 평가 ===")
print(f"  배포bias      macro={base_none:.4f}")
print(f"  train최적bias macro={opt_none:.4f}   Δ={opt_none-base_none:+.4f}  (damped-bias 재현: ~0/음수 예상)")
print(f"\n[판정] shift주입 Δ ≥+0.002 → 프로브+제출 가치 / 미만 → 이 축도 死")
