#!/usr/bin/env python3
"""root-goal rescue fold0 A/B: 긴 세션서 좌측절단으로 소실되는 첫 user목표를 복원하면
M4/macro가 오르나? (codex#2, 22.3% 행 영향, GEN-rescue급 후보) — m1 fold0 ckpt 재추론."""
import os, sys, json, csv, numpy as np, time
sys.path.insert(0,'/root/Action_Decision/common')
import ad_lib
from transformers import AutoTokenizer

ROOT="/root/Action_Decision"; os.chdir(ROOT)
CLASSES=ad_lib.CLASSES; CI={c:i for i,c in enumerate(CLASSES)}
M4=[CI[c] for c in ('read_file','grep_search','list_directory','glob_pattern')]
CKPT=f"{ROOT}/work/foldckpt_largev6_f0ckpt_f0"
t0=time.time()
rows=[json.loads(l) for l in open('data/train.jsonl')]
lab={}
with open('data/train_labels.csv') as f:
    rd=csv.reader(f); next(rd)
    for r in rd: lab[r[0]]=r[1]
y=np.array([CI[lab[r['id']]] for r in rows])
sp=np.load('splits/splits.npz',allow_pickle=True); va=sp['va0']

tok=AutoTokenizer.from_pretrained(CKPT, local_files_only=True); tok.truncation_side="left"
KEEP=320-tok.num_special_tokens_to_add(False)

# affected 행 식별 + goal-rescue 텍스트 생성
def first_user(r):
    for h in (r.get('history') or []):
        if h.get('role')=='user': return h.get('content') or ''
    return ''
def is_affected(r):
    txt=ad_lib.serialize(r,'v6',8); ids=tok(txt,add_special_tokens=False)['input_ids']
    if len(ids)<=KEEP: return False
    fu=first_user(r)
    if not fu: return False
    kept=tok.decode(ids[-KEEP:])
    return fu[:30] not in kept
def goal_rescue_text(r):
    base=ad_lib.serialize(r,'v6',8); fu=first_user(r)
    if not fu: return base
    goal=f"[GOAL] {fu[:150]} "
    # [CUR] 앞에 삽입(좌측절단 생존)
    if " [CUR]" in base: return base.replace(" [CUR]", " "+goal+"[CUR]",1)
    return base+" "+goal

va_rows=[rows[i] for i in va]; va_y=y[va]
aff_mask=np.array([is_affected(r) for r in va_rows])
print(f"[fold0-val n={len(va)}] affected(목표소실) {aff_mask.sum()} ({aff_mask.mean()*100:.1f}%)  build {time.time()-t0:.0f}s")

# 속도: affected 전부 + unaffected 동수 서브샘플
aff_idx=np.where(aff_mask)[0]
un_idx=np.where(~aff_mask)[0]
rng=np.random.RandomState(0); un_sub=rng.choice(un_idx, min(len(aff_idx),len(un_idx)), replace=False)
test_idx=np.concatenate([aff_idx,un_sub]); test_idx.sort()
sub=[va_rows[i] for i in test_idx]; suby=va_y[test_idx]; subaff=aff_mask[test_idx]
print(f"테스트 서브셋 {len(sub)} (aff {subaff.sum()} + un {(~subaff).sum()})")

base_texts=[ad_lib.serialize(r,'v6',8) for r in sub]
resc_texts=[goal_rescue_text(r) for r in sub]
print(f"추론 시작(CPU, {len(sub)}행×2)... {time.time()-t0:.0f}s")
P_base=ad_lib.predict_logits(CKPT, sub, version='v6', max_len=320, texts=base_texts, return_probs=True, device='cuda')
print(f"  base done {time.time()-t0:.0f}s")
P_resc=ad_lib.predict_logits(CKPT, sub, version='v6', max_len=320, texts=resc_texts, return_probs=True, device='cuda')
print(f"  rescue done {time.time()-t0:.0f}s")

def m4macro(P, mask):
    pred=P.argmax(1); yy=suby[mask]; pp=pred[mask]; f=[];m4=[]
    for c in range(14):
        tp=int(((pp==c)&(yy==c)).sum());fp=int(((pp==c)&(yy!=c)).sum());fn=int(((pp!=c)&(yy==c)).sum())
        pr=tp/(tp+fp) if tp+fp else 0;rc=tp/(tp+fn) if tp+fn else 0
        v=2*pr*rc/(pr+rc) if pr+rc else 0; f.append(v)
        if c in M4:m4.append(v)
    return np.mean(f),np.mean(m4)
print("\n=== m1-solo fold0 (goal-rescue A/B) ===")
for name,mask in [('affected행',subaff),('전체',np.ones(len(sub),bool))]:
    bm,b4=m4macro(P_base,mask); rm,r4=m4macro(P_resc,mask)
    print(f"[{name} n={mask.sum()}] macro {bm:.4f}→{rm:.4f} (Δ{rm-bm:+.4f})  M4 {b4:.4f}→{r4:.4f} (Δ{r4-b4:+.4f})")
# argmax 변화 행수
flips=int((P_base.argmax(1)!=P_resc.argmax(1)).sum())
print(f"argmax 변화 {flips}행 / {len(sub)}")
print("[판정] affected macro Δ≥+0.01 (casc≈+0.003, β0.82→LB+0.0025) → GEN-rescue급 생존")
