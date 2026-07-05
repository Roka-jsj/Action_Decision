"""콘텐츠 기반 au 감지기 프로브 (R15) — id 없이 au/sim 판별 가능한가.

배경: [GEN] 토큰이 id prefix 파생 → 히든 id가 무표식이면 au행이 [GEN] sim으로 서빙돼
au 서브스코어 0.52(정상 0.67+). 콘텐츠로 au를 감지해 [GEN]을 복원하면 회수 가능.
요구: sim 오탐(false-au)은 해당 행을 망가뜨리므로 고정밀 임계 필요.
GroupKFold(세션)로 AUC + 임계별 precision/recall. 세션 단위 판별력도 병기(au는 세션 통째).
"""
from __future__ import annotations
import sys, os
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
from common.io_utils import load_train
from common import ad_lib

samples, y, ids = load_train()
ids = np.array([str(i) for i in ids])
au = np.char.startswith(ids, "sess_au").astype(int)
groups = np.array([s["session"] for s in samples])

feats, names = [], None
for s in samples:
    m = ad_lib.meta_fields(s)
    hist = s.get("history") or []
    acts = [t for t in hist if t.get("role") == "assistant_action"]
    users = [t for t in hist if t.get("role") == "user"]
    prompt = s.get("current_prompt") or ""
    row = {
        "turn_index": m["turn_index"], "budget": m["budget"], "loc": m["loc"],
        "n_open": m["n_open_files"], "git": int(m["git_dirty"]),
        "elapsed": m.get("elapsed") or 0,
        "tier": hash(m["user_tier"]) % 97, "lang": hash(m["language_pref"]) % 97,
        "toplang": hash(m["top_lang"]) % 97, "ci": hash(m["last_ci_status"]) % 97,
        "n_hist": len(hist), "n_act": len(acts), "n_user": len(users),
        "plen": len(prompt), "p_hangul": int(ad_lib.has_hangul(prompt)),
        "avg_alen": np.mean([len(str(t.get("args", ""))) for t in acts]) if acts else 0,
        "avg_rlen": np.mean([len(str(t.get("result", ""))) for t in acts]) if acts else 0,
    }
    if names is None: names = list(row.keys())
    feats.append([row[k] for k in names])
X = np.array(feats, dtype=np.float32)
print(f"[au-det] X {X.shape}, au율 {au.mean():.4f}", flush=True)

from sklearn.model_selection import GroupKFold
import lightgbm as lgb
oofp = np.zeros(len(au))
for tr, va in GroupKFold(5).split(X, au, groups):
    mdl = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, verbose=-1)
    mdl.fit(X[tr], au[tr])
    oofp[va] = mdl.predict_proba(X[va])[:, 1]

from sklearn.metrics import roc_auc_score
print(f"행단위 AUC: {roc_auc_score(au, oofp):.5f}")
for th in (0.5, 0.8, 0.9, 0.95, 0.99):
    pred = oofp > th
    tp = (pred & (au == 1)).sum(); fp = (pred & (au == 0)).sum()
    prec = tp / max(tp + fp, 1); rec = tp / au.sum()
    print(f"  th={th}: precision {prec:.4f} recall {rec:.4f} (탐지 {pred.sum()}, sim오탐 {fp})")

# 세션 단위 (au는 세션 전체가 au — 세션 평균 확률로 판별)
import pandas as pd
df = pd.DataFrame({"g": groups, "p": oofp, "au": au})
ag = df.groupby("g").agg(p=("p", "mean"), au=("au", "max"))
print(f"세션단위 AUC: {roc_auc_score(ag['au'], ag['p']):.5f}")
for th in (0.5, 0.9, 0.95):
    pred = ag["p"] > th
    tp = int((pred & (ag["au"] == 1)).sum()); fp = int((pred & (ag["au"] == 0)).sum())
    print(f"  th={th}: precision {tp/max(tp+fp,1):.4f} recall {tp/int(ag['au'].sum()):.4f} (sim세션 오탐 {fp})")
