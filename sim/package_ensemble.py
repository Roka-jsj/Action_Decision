"""제출 zip 조립기 — 멤버(디렉터리 or member_*.zip) + 스태커 + 벤더링 lightgbm.

usage:
  python sim/package_ensemble.py --out submit_stack2 --stacker artifacts/stack_e5klue \
      --member <path1> --member <path2> [--version v4 --max_len 320 --batch 128]

멤버 순서 = 스태커 학습 시 키 순서와 동일해야 함(확률 concat 순서).
구성: model/{m1..mN, ad_lib.py, meta.lgb, postproc.json, run_meta.json, lightgbm/}
      + script.py + requirements.txt(빈 파일 — 설치단계 무의존, 완전 오프라인 안전)
"""
from __future__ import annotations
import os, sys, json, shutil, zipfile, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--stacker", default="")          # 비우면 mean 앙상블(스태커 없음)
ap.add_argument("--bias", default="")             # mean 모드에서 per-class bias 적합용 teacher npz glob
ap.add_argument("--weights", default="")          # mean 모드 멤버 가중치 "0.65,0.35" (비우면 균등)
ap.add_argument("--margin_th", type=float, default=0)  # >0이면 조건부 2-pass(m1 마진<th만 m2 혼합)
ap.add_argument("--dual", type=int, default=0)          # 1=듀얼 bias(sim/au 서브셋 적합, R14/World C)
ap.add_argument("--dual_shrink", type=float, default=1.0)  # bias_au/sim ← λ*서브셋 + (1-λ)*글로벌
ap.add_argument("--member", action="append", required=True)
ap.add_argument("--version", default="v4")
ap.add_argument("--max_len", type=int, default=320)
ap.add_argument("--batch", type=int, default=128)
a = ap.parse_args()

pkg = os.path.join(ROOT, "packages", a.out)
shutil.rmtree(pkg, ignore_errors=True)
mdl = os.path.join(pkg, "model")
os.makedirs(mdl)

ens = []
for i, spec in enumerate(a.member, 1):
    # "경로" 또는 "경로::버전" (멤버별 직렬화 버전 오버라이드, 예: v6 멤버 + v4 멤버 혼합)
    src, _, mver = spec.partition("::")
    dst = os.path.join(mdl, f"m{i}")
    if src.endswith(".zip"):
        os.makedirs(dst)
        with zipfile.ZipFile(src) as z:
            z.extractall(dst)
    else:
        shutil.copytree(src, dst)
    for junk in ["training_args.bin", "optimizer.pt"]:
        p = os.path.join(dst, junk)
        if os.path.exists(p):
            os.remove(p)
    entry = {"dir": f"m{i}", **({"version": mver} if mver else {})}
    if os.path.exists(os.path.join(dst, "coef.npy")):   # n-gram 멤버(HashingVectorizer+numpy)
        entry["type"] = "ngram"
    ens.append(entry)

MEAN = not a.stacker
if not MEAN:
    for f in ["meta.lgb", "postproc.json", "stack_meta.json"]:
        shutil.copy(os.path.join(ROOT, a.stacker, f), os.path.join(mdl, f))
    sm = json.load(open(os.path.join(ROOT, a.stacker, "stack_meta.json")))
    assert len(sm["members"]) == len(ens), f"스태커 멤버수 {len(sm['members'])} != 전달 멤버수 {len(ens)}"

# ad_lib 배포본: 기각된 posmap 블록 스트립 (본선 코드검증 대비)
src = open(os.path.join(ROOT, "common", "ad_lib.py"), encoding="utf-8").read()
s0, s1 = src.find("# ==================== POSMAP_BLOCK_START"), src.find("# ==================== POSMAP_BLOCK_END ====================")
if s0 >= 0 and s1 > s0:
    src = src[:s0] + src[s1 + len("# ==================== POSMAP_BLOCK_END ===================="):]
open(os.path.join(mdl, "ad_lib.py"), "w", encoding="utf-8").write(src)
compile(src, "ad_lib.py", "exec")   # 스트립 후 구문 검증

need_lgb = not MEAN or any(e.get("type") == "ngram" for e in ens)
if need_lgb:
    shutil.copytree(os.path.join(ROOT, "vendor", "lightgbm"), os.path.join(mdl, "lightgbm"))
rm = {"version": a.version, "max_len": a.max_len, "batch_size": a.batch, "ensemble": ens}
W = [float(x) for x in a.weights.split(",")] if a.weights else None
if W:
    assert len(W) == len(ens), "weights 개수 != 멤버수"
    rm["weights"] = W
if a.margin_th > 0:
    # 마지막 멤버만 조건부(저마진 행 재추론), 앞 멤버들은 전체 추론
    rm["conditional"] = {"margin_th": a.margin_th, "cond_members": [len(ens) - 1]}
if not MEAN:
    rm["stacker"] = "meta.lgb"; rm["stack_members"] = sm["members"]
# mean 모드: 멤버 prob 평균(ad_lib predict가 stacker 없으면 mean_p 사용) + per-class bias
if MEAN and a.bias:
    sys.path.insert(0, ROOT)
    import numpy as np, glob
    from common.io_utils import load_train
    from common.cv import make_splits
    from common.postproc import fit_bias, save as save_bias
    samples, y, ids = load_train(); y = np.array(y)
    groups = np.array([s["session"] for s in samples]); sp = make_splits(ids, y, groups)
    folds = sp["folds"]; cov = np.concatenate([f[1] for f in folds])
    # 멤버 OOF 평균으로 bias 적합 (배포와 동일: prob 평균 → log → bias)
    oofs = []
    for g in a.bias.split(","):
        o = np.zeros((len(samples), 14), np.float32); cs = set()
        for p in sorted(glob.glob(g.strip())):
            z = np.load(p, allow_pickle=True)
            for fdi in range(int(z["fold_lo"]), int(z["fold_hi"])):
                if fdi in cs: continue
                o[folds[fdi][1]] = z["oof"][folds[fdi][1]]; cs.add(fdi)
        oofs.append(o)
    if a.margin_th > 0:
        # 배포와 동일한 조건부 혼합으로 bias 적합 (마지막 멤버 = 조건부)
        assert W and len(W) == len(oofs)
        wf = sum(W[:-1])
        p_full = sum(w * o for w, o in zip(W[:-1], oofs[:-1])) / wf
        srt = np.sort(p_full, axis=1)
        sel = (srt[:, -1] - srt[:, -2]) < a.margin_th
        mean_oof = p_full.copy()
        mean_oof[sel] = (wf * p_full[sel] + W[-1] * oofs[-1][sel]) / (wf + W[-1])
        print(f"[cond-bias] margin_th={a.margin_th} 선택률={sel[cov].mean()*100:.1f}%")
    elif W:
        assert len(W) == len(oofs), "weights 개수 != bias glob 개수"
        mean_oof = sum(w * o for w, o in zip(W, oofs)) / sum(W)
    else:
        mean_oof = sum(oofs) / len(oofs)
    lp_mean = np.log(mean_oof + 1e-9)
    b, _ = fit_bias(lp_mean[cov], y[cov])
    extra = None
    if a.dual:
        auk = np.char.startswith(np.array([str(i) for i in ids]), "sess_au")
        cov_au, cov_sim = cov[auk[cov]], cov[~auk[cov]]
        b_au, _ = fit_bias(lp_mean[cov_au], y[cov_au])
        b_sim, _ = fit_bias(lp_mean[cov_sim], y[cov_sim])
        lam = a.dual_shrink
        extra = {"bias_sim": lam * np.array(b_sim) + (1 - lam) * np.array(b),
                 "bias_au": lam * np.array(b_au) + (1 - lam) * np.array(b)}
        print(f"[dual-bias] au {len(cov_au)}행 / sim {len(cov_sim)}행 (λ={lam})")
    save_bias(os.path.join(mdl, "postproc.json"), b, meta={"mean_ensemble": True, "weights": W,
                                                           "margin_th": a.margin_th or None},
              extra_biases=extra)
    print(f"[mean-bias] fit on {len(oofs)} member OOF {'weighted ' + str(W) if W else 'avg'}")
json.dump(rm, open(os.path.join(mdl, "run_meta.json"), "w"))
shutil.copy(os.path.join(ROOT, "common", "server_script.py"), os.path.join(pkg, "script.py"))
open(os.path.join(pkg, "requirements.txt"), "w").close()

zp = os.path.join(ROOT, "packages", f"{a.out}.zip")
if os.path.exists(zp):
    os.remove(zp)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for r, _, fs in os.walk(pkg):
        for f in fs:
            p = os.path.join(r, f)
            z.write(p, os.path.relpath(p, pkg))
gb = os.path.getsize(zp) / 1e9
if MEAN:
    print(f"[package] {zp}  {gb:.3f}GB  MEAN ensemble  members={[e['dir'] for e in ens]}")
else:
    print(f"[package] {zp}  {gb:.3f}GB  members={sm['members']}  "
          f"stack holdout+bias={sm.get('holdout_bias')} LB예측={sm.get('lb_pred')}")
assert gb < 1.0, "1GB 초과!"
