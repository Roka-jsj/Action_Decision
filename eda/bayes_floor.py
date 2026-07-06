"""쟁점 A 진단: 탐색4클래스(read/grep/list/glob)가 Bayes 바닥인가 재학습 여지인가.

두 가지를 실측:
1. **Bayes 바닥 추정**: 사용 가능한 최강 feature key로 그룹화 → 그룹 내 최빈 클래스 채택시 오류율(=주어진 feature로 환원불가한 하한). key를 점점 강화(prompt → +직전action+status → +[SEQ]서명 → +open_ext)해 floor가 얼마나 내려가나 = "아직 안 쓴 분리신호"의 크기.
2. **모델 OOF confusion**과 비교: v6 largeonly OOF에서 탐색4클래스 오류율. 모델오류 >> Bayes바닥이면 재학습 여지(H6 GO), ≈이면 환원불가(H6 사망).
탐색4클래스만, sim/au 합산. train 70k 실측.
"""
from __future__ import annotations
import sys, os, glob
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
from common.io_utils import load_train, CLASSES
from common.cv import make_splits
from common import ad_lib

EXPLORE = ["read_file", "grep_search", "list_directory", "glob_pattern"]
eidx = {CLASSES.index(c) for c in EXPLORE}

samples, y, ids = load_train(); y = np.array(y); ids = np.array([str(i) for i in ids])
expl = np.array([yy in eidx for yy in y])
print(f"[floor] 탐색4클래스 {expl.sum()}행 ({expl.mean()*100:.1f}%)", flush=True)

def floor_err(keys, mask):
    """key로 그룹화, 그룹 최빈라벨 채택시 오류율 (탐색4클래스 행만 채점)."""
    from collections import defaultdict
    grp = defaultdict(list)
    for k, yy, m in zip(keys, y, mask):
        if m: grp[k].append(yy)
    tot = err = 0
    for k, ys in grp.items():
        ys = np.array(ys); tot += len(ys)
        err += len(ys) - np.bincount(ys).max()
    return err / tot, len(grp), tot

def key_prompt(s): return (s.get("current_prompt") or "").strip().lower()
def key_pa(s):
    nm, args, rs, st = ad_lib.last_action(s)
    return (key_prompt(s), nm, st)
def key_seq(s):
    nm, args, rs, st = ad_lib.last_action(s)
    seq = tuple(t.get("name","") for t in (s.get("history") or []) if t.get("role")=="assistant_action")[-6:]
    return (key_prompt(s), nm, st, seq)
def key_seq_ext(s):
    m = ad_lib.meta_fields(s)
    exts = tuple(sorted({ad_lib.path_ext(p) for p in m["open_files"] if ad_lib.path_ext(p)}))
    return key_seq(s) + (exts, m["n_open_files"])

for name, kf in [("prompt", key_prompt), ("prompt+직전act+status", key_pa),
                 ("+[SEQ]6", key_seq), ("+open_ext+nopen", key_seq_ext)]:
    keys = [kf(s) for s in samples]
    fe, ng, tot = floor_err(keys, expl)
    print(f"[floor] key={name:24s} Bayes바닥오류={fe:.4f} (그룹{ng}, 유니크율{ng/tot:.2f})", flush=True)

# 모델 OOF 탐색4클래스 오류율 (v6 teacher)
oof = np.zeros((len(y),14), np.float32); cs=set()
for p in sorted(glob.glob(os.path.join(R,"action_decision_maximum/experiments/teacher_largev6[AB]_a*.npz"))):
    z=np.load(p,allow_pickle=True)
    for f in range(int(z["fold_lo"]),int(z["fold_hi"])):
        if f in cs: continue
        oof[__import__("numpy").array(make_splits(ids,y,np.array([s['session'] for s in samples]))["folds"][f][1])]=z["oof"][make_splits(ids,y,np.array([s['session'] for s in samples]))["folds"][f][1]]; cs.add(f)
pred = oof.argmax(1)
model_err = (pred[expl] != y[expl]).mean()
print(f"[model] v6 OOF 탐색4클래스 오류율={model_err:.4f}")
print(f"→ 모델오류 {model_err:.3f} vs Bayes바닥(최강key) → 갭이 크면 H6(재학습) 여지, ≈면 환원불가")
