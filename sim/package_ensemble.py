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
ap.add_argument("--stacker", required=True)
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

shutil.copytree(os.path.join(ROOT, "vendor", "lightgbm"), os.path.join(mdl, "lightgbm"))
rm = {"version": a.version, "max_len": a.max_len, "batch_size": a.batch,
      "ensemble": ens, "stacker": "meta.lgb", "stack_members": sm["members"]}
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
print(f"[package] {zp}  {gb:.3f}GB  members={sm['members']}  "
      f"stack holdout+bias={sm.get('holdout_bias')} LB예측={sm.get('lb_pred')}")
assert gb < 1.0, "1GB 초과!"
