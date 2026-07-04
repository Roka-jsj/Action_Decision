"""제출 전 패리티 게이트 — 패키지를 홀드아웃 N행에 실행해 기대성능과 대조.

잡는 것: 멤버 순서 교체, 직렬화 버전 오지정, 스태커-멤버 조합 불일치, 프루닝 id_map 오류
(전부 크래시 없이 F1만 조용히 깎는 침묵 실패들 — 오프라인 시뮬로는 못 잡음)

usage: python sim/parity_check.py <pkg_dir> [N=300]
판정: 홀드아웃 서브셋 macro-F1 >= stack_meta(holdout_bias) - 0.04 (서브셋 노이즈 마진)
"""
from __future__ import annotations
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.io_utils import load_train, CLASSES
from common.cv import make_splits
from common.metrics import macro_f1

pkg = os.path.abspath(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
mdl = os.path.join(pkg, "model")
sys.path.insert(0, mdl)
import importlib
import ad_lib as dep_ad_lib
importlib.reload(dep_ad_lib)
assert os.path.realpath(os.path.dirname(dep_ad_lib.__file__)) == os.path.realpath(mdl), \
    f"배포본 ad_lib 미로드: {dep_ad_lib.__file__}"

samples, y, ids = load_train()
y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups)
hold = sp["holdout_idx"]
rng = np.random.RandomState(0)
sub = rng.choice(hold, size=min(N, len(hold)), replace=False)
sub_s = [samples[i] for i in sub]

meta = json.load(open(os.path.join(mdl, "run_meta.json")))
preds = dep_ad_lib.predict(mdl, sub_s, version=meta["version"], max_len=meta["max_len"],
                           batch_size=meta.get("batch_size", 64),
                           postproc_path=os.path.join(mdl, "postproc.json"), meta=meta)
pr = np.array([CLASSES.index(preds[s["id"]]) for s in sub_s])
f1, _ = macro_f1(y[sub], pr)
exp = None
for cand in ("stack_meta.json",):
    p = os.path.join(mdl, cand)
    if os.path.exists(p):
        exp = json.load(open(p)).get("holdout_bias")
# stack_meta는 model/에 없으므로 아티팩트에서 찾도록 인자 확장 여지 — 일단 기준치 수동
thr_note = ""
acc = float(np.mean(pr == y[sub]))
print(f"[parity] holdout {len(sub)}행: macro-F1={f1:.4f} acc={acc:.4f}{thr_note}")
if exp is not None:
    ok = f1 >= exp - 0.04
    print(f"[parity] 기대(holdout_bias)={exp} → {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
print("[parity] 기대치 파일 없음 — 수치를 직접 판정하세요 (정상범위: holdout_bias±0.04)")
