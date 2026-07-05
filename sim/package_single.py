"""단일 모델 제출 패키저 (앵커/ablation용) — 스태커 없이 멤버 1개 + bias.

usage: python sim/package_single.py --out submit_largeonly --member <zip|dir>[::ver]
       [--bias <teacher_npz for bias fit>] [--version v6 --max_len 320]
run_meta: ensemble/stacker 키 없음 → ad_lib.predict가 predict_logits 단일경로.
"""
from __future__ import annotations
import os, sys, json, shutil, zipfile, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--member", required=True)
ap.add_argument("--bias", default="")   # teacher npz(oof/hold)로 bias 적합
ap.add_argument("--dual", type=int, default=0)          # 1=듀얼 bias(sim/au 서브셋 적합, R14)
ap.add_argument("--dual_shrink", type=float, default=1.0)  # bias_au/sim ← λ*서브셋 + (1-λ)*글로벌
ap.add_argument("--version", default="v6")
ap.add_argument("--max_len", type=int, default=320)
ap.add_argument("--batch", type=int, default=128)
a = ap.parse_args()

pkg = os.path.join(ROOT, "packages", a.out); shutil.rmtree(pkg, ignore_errors=True)
mdl = os.path.join(pkg, "model"); os.makedirs(mdl)
src, _, mver = a.member.partition("::")
if src.endswith(".zip"):
    with zipfile.ZipFile(src) as z: z.extractall(mdl)
else:
    shutil.copytree(src, mdl, dirs_exist_ok=True)
for junk in ["training_args.bin", "optimizer.pt"]:
    p = os.path.join(mdl, junk)
    if os.path.exists(p): os.remove(p)

# ad_lib (posmap 스트립)
adl = open(os.path.join(ROOT, "common", "ad_lib.py"), encoding="utf-8").read()
s0 = adl.find("# ==================== POSMAP_BLOCK_START"); s1 = adl.find("# ==================== POSMAP_BLOCK_END ====================")
if s0 >= 0 and s1 > s0:
    adl = adl[:s0] + adl[s1 + len("# ==================== POSMAP_BLOCK_END ===================="):]
compile(adl, "ad_lib.py", "exec")
open(os.path.join(mdl, "ad_lib.py"), "w", encoding="utf-8").write(adl)

rm = {"version": mver or a.version, "max_len": a.max_len, "batch_size": a.batch}
# bias 적합 (largev6 OOF로)
if a.bias:
    from common.io_utils import load_train, CLASSES
    from common.cv import make_splits
    from common.postproc import fit_bias, save as save_bias
    import glob
    samples, y, ids = load_train(); y = np.array(y)
    groups = np.array([s["session"] for s in samples]); sp = make_splits(ids, y, groups)
    folds = sp["folds"]; cov = np.concatenate([f[1] for f in folds])
    oof = np.zeros((len(samples), 14), np.float32); cs = set()
    for p in sorted(glob.glob(a.bias)):
        z = np.load(p, allow_pickle=True)
        for f in range(int(z["fold_lo"]), int(z["fold_hi"])):
            if f in cs: continue
            oof[folds[f][1]] = z["oof"][folds[f][1]]; cs.add(f)
    b, _ = fit_bias(np.log(oof[cov] + 1e-9), y[cov])
    extra = None
    if a.dual:
        au = np.char.startswith(np.array([str(i) for i in ids]), "sess_au")
        cov_au, cov_sim = cov[au[cov]], cov[~au[cov]]
        b_au, _ = fit_bias(np.log(oof[cov_au] + 1e-9), y[cov_au])
        b_sim, _ = fit_bias(np.log(oof[cov_sim] + 1e-9), y[cov_sim])
        lam = a.dual_shrink
        extra = {"bias_sim": lam * np.array(b_sim) + (1 - lam) * np.array(b),
                 "bias_au": lam * np.array(b_au) + (1 - lam) * np.array(b)}
        print(f"[bias] dual fit (au {len(cov_au)}행, sim {len(cov_sim)}행, λ={lam})")
    save_bias(os.path.join(mdl, "postproc.json"), b, meta={"single": True}, extra_biases=extra)
    print(f"[bias] fit on {len(cs)} folds")

json.dump(rm, open(os.path.join(mdl, "run_meta.json"), "w"))
shutil.copy(os.path.join(ROOT, "common", "server_script.py"), os.path.join(pkg, "script.py"))
open(os.path.join(pkg, "requirements.txt"), "w").close()
zp = os.path.join(ROOT, "packages", f"{a.out}.zip")
if os.path.exists(zp): os.remove(zp)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for r, _, fs in os.walk(pkg):
        for f in fs: z.write(os.path.join(r, f), os.path.relpath(os.path.join(r, f), pkg))
gb = os.path.getsize(zp) / 1e9
print(f"[package] {zp} {gb:.3f}GB (single model, version={rm['version']})")
assert gb < 1.0
