#!/bin/bash
# R38: 자체 mdebfull 기반 tri_mdeb 변형 조립 (mdebfull 완료 후 실행)
# 산출: submit_jtri_mdeb(재현 기준본·오프라인 패리티용), _aubias(1순위), _th45, _th40, _w20
# 구조: m1(largefullv6 q8) + m2(자체 mdeb fp16 prune) + m3(v4 q8), cond_members=[1,2]
set -euo pipefail
R=/root/Action_Decision; cd $R
MDEB=$R/work/member_mdebfull.zip
[ -f "$MDEB" ] || { echo "member_mdebfull.zip 없음 — 학습 미완"; exit 1; }
G="action_decision_maximum/experiments/teacher_largev6[AB]_a*.npz"

build_base() { # $1=out $2=weights $3=mth
  python3 sim/package_ensemble.py --out "$1" \
    --member packages/submit_tri_cond_rebuild/model/m1::v6 \
    --member "$MDEB::v6" \
    --member packages/submit_tri_cond_rebuild/model/m3::v4 \
    --weights "$2" --margin_th "$3" --bias "$G,$G,$G" \
    --version v6 --max_len 320 --batch 128 > /dev/null
}

fix_and_zip() { # $1=out $2=mth $3=bias_src(tri|aubias)
  python3 - "$1" "$2" "$3" << 'PY'
import json, sys, os, subprocess
out, mth, bias_src = sys.argv[1], float(sys.argv[2]), sys.argv[3]
d = f"packages/{out}"
rm = json.load(open(f"{d}/model/run_meta.json"))
rm["conditional"] = {"margin_th": mth, "cond_members": [1, 2]}   # 이중조건부(조원 검증 구조)
json.dump(rm, open(f"{d}/model/run_meta.json", "w"), ensure_ascii=False)
pp = json.load(open("packages/submit_tri_cond_rebuild/model/postproc.json"))
if bias_src == "aubias":
    pp["bias"] = json.load(open("work/aubias_bias.json"))["bias"]
    pp["note"] = "0.5*tri + 0.5*au-reweight(lam0.13) — R37 1순위"
json.dump(pp, open(f"{d}/model/postproc.json", "w"), ensure_ascii=False)
zp = f"packages/{out}.zip"
if os.path.exists(zp): os.remove(zp)
subprocess.run(["zip", "-qr", os.path.abspath(zp), "."], cwd=d, check=True)
print(f"{out}: mth={mth} bias={bias_src} zip={os.path.getsize(zp)/1e9:.3f}GB")
PY
}

build_base submit_jtri_mdeb        "0.6,0.15,0.25" 0.5
fix_and_zip submit_jtri_mdeb        0.5  tri
build_base submit_jtri_mdeb_aubias "0.6,0.15,0.25" 0.5
fix_and_zip submit_jtri_mdeb_aubias 0.5  aubias
build_base submit_jtri_mdeb_th45   "0.6,0.15,0.25" 0.45
fix_and_zip submit_jtri_mdeb_th45   0.45 tri
build_base submit_jtri_mdeb_th40   "0.6,0.15,0.25" 0.40
fix_and_zip submit_jtri_mdeb_th40   0.40 tri
build_base submit_jtri_mdeb_w20    "0.55,0.20,0.25" 0.5
fix_and_zip submit_jtri_mdeb_w20    0.5  tri

for z in jtri_mdeb jtri_mdeb_aubias jtri_mdeb_th45 jtri_mdeb_th40 jtri_mdeb_w20; do
  python3 sim/check_zip.py packages/submit_$z.zip 2>&1 | tail -1 | sed "s/^/[$z] /"
done
echo "[다음] 기준본 30k 검증: bash sim/prep_verify.sh packages/submit_jtri_mdeb.zip && docker exec mun-jtest bash /share/verify/submit_jtri_mdeb/run.sh"
