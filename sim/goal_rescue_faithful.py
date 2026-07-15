#!/usr/bin/env python3
"""GEN-rescue 충실판 goal-rescue: 신규태그 없이, 첫 user턴을 학습된 u:형식·원위치(헤더 직후)로
토큰 재구성 보존(좌측절단 생존). GEN-rescue와 동일 기제. affected fold0 A/B."""
import os,sys,json,csv,numpy as np,time
sys.path.insert(0,'/root/Action_Decision/common'); import ad_lib
from transformers import AutoTokenizer
ROOT="/root/Action_Decision"; os.chdir(ROOT)
CLASSES=ad_lib.CLASSES; CI={c:i for i,c in enumerate(CLASSES)}
M4=[CI[c] for c in ('read_file','grep_search','list_directory','glob_pattern')]
CKPT=f"{ROOT}/work/foldckpt_largev6_f0ckpt_f0"
t0=time.time()
rows=[json.loads(l) for l in open('data/train.jsonl')]
lab={}; 
with open('data/train_labels.csv') as f:
    rd=csv.reader(f);next(rd)
    for r in rd: lab[r[0]]=r[1]
y=np.array([CI[lab[r['id']]] for r in rows]); sp=np.load('splits/splits.npz',allow_pickle=True); va=sp['va0']
tok=AutoTokenizer.from_pretrained(CKPT,local_files_only=True); tok.truncation_side="left"
NSPEC=tok.num_special_tokens_to_add(False); KEEP=320-NSPEC
def first_user(r):
    for h in (r.get('history') or []):
        if h.get('role')=='user': return h.get('content') or ''
    return ''
# 헤더경계: 첫 " [HIST]" 까지가 헤더. 그 다음이 첫 u: 턴.
def build(r):
    base=ad_lib.serialize(r,'v6',8)
    ids=tok(base,add_special_tokens=False)['input_ids']
    if len(ids)<=KEEP: return None  # 비절단→무대상
    fu=first_user(r)
    if not fu: return None
    kept=tok.decode(ids[-KEEP:])
    if fu[:30] in kept: return None  # 이미 생존
    # 헤더 텍스트: [HIST] 포함 전까지 + 첫 user턴 u: 복원
    hpos=base.find(" [HIST]")
    header = base[:hpos] if hpos>0 else ""
    goal_turn = f" [HIST] u: {fu[:150]}"
    prefix = header + goal_turn                       # [header] [HIST] u:{첫목표}
    pref_ids = tok(prefix,add_special_tokens=False)['input_ids']
    # 나머지 예산을 base의 tail(최근)로 채움
    budget = KEEP - len(pref_ids)
    if budget<20: pref_ids=pref_ids[:KEEP-20]; budget=20
    tail_ids = ids[-budget:]
    return pref_ids+tail_ids
va_rows=[rows[i] for i in va]; va_y=y[va]
recon={}
aff=[]
for i,r in enumerate(va_rows):
    ids=build(r)
    if ids is not None: recon[i]=ids; aff.append(i)
aff=np.array(aff)
print(f"[fold0-val] 충실 goal-rescue 대상 {len(aff)} ({len(aff)/len(va)*100:.1f}%)  build {time.time()-t0:.0f}s")
# 서브셋: affected 전부
sub=[va_rows[i] for i in aff]; suby=va_y[aff]
base_texts=[ad_lib.serialize(r,'v6',8) for r in sub]
# rescue: 재구성 ids 직접 사용 → texts로 못넣으니, predict_logits를 ids로. 우회: decode해서 texts로.
resc_texts=[tok.decode(recon[i]) for i in aff]
print(f"추론(GPU) {len(sub)}×2... {time.time()-t0:.0f}s")
Pb=ad_lib.predict_logits(CKPT,sub,version='v6',max_len=320,texts=base_texts,return_probs=True,device='cuda')
Pr=ad_lib.predict_logits(CKPT,sub,version='v6',max_len=320,texts=resc_texts,return_probs=True,device='cuda')
def mm(P):
    pred=P.argmax(1); f=[];m4=[]
    for c in range(14):
        tp=int(((pred==c)&(suby==c)).sum());fp=int(((pred==c)&(suby!=c)).sum());fn=int(((pred!=c)&(suby==c)).sum())
        pr=tp/(tp+fp) if tp+fp else 0;rc=tp/(tp+fn) if tp+fn else 0
        v=2*pr*rc/(pr+rc) if pr+rc else 0;f.append(v)
        if c in M4:m4.append(v)
    return np.mean(f),np.mean(m4)
bm,b4=mm(Pb); rm,r4=mm(Pr)
print(f"\n=== 충실 goal-rescue (affected n={len(aff)}, m1-solo) ===")
print(f"  macro {bm:.4f}→{rm:.4f} (Δ{rm-bm:+.4f})   M4 {b4:.4f}→{r4:.4f} (Δ{r4-b4:+.4f})")
print(f"  argmax변화 {int((Pb.argmax(1)!=Pr.argmax(1)).sum())}행")
print(f"[판정] affected macro Δ≥+0.01 → GEN-rescue급 생존 / ≤0 → 死")
