#!/bin/bash
# C0 인근 단일변경 좌표 프로브 2종 (R33 그라인딩 계열 — 학습 불요, 조립만)
# A: klue 가중 상향 0.25→0.30 (m1 0.6→0.55) — C0에서 klue 순기여 양성(+0.00045)·ρ_low 0.922 최저 실증 방향
# B: margin_th 0.5→0.55 — 조건부 커버리지 확대(같은 근거: 양성 멤버의 관여 행 확대)
set -euo pipefail
R=/root/Action_Decision; cd $R
G="action_decision_maximum/experiments/teacher_largev6[AB]_a*.npz"

build() { # $1=name $2=weights $3=th
  python3 sim/package_ensemble.py --out "$1" \
    --member work/cc_members/m1::v6 \
    --member work/cc_members/mdeb::v6 \
    --member work/member_kluefull_q8.zip::v6 \
    --weights "$2" --margin_th "$3" --bias "$G,$G,$G" \
    --version v6 --max_len 320 --batch 128 > /dev/null
  NAME="$1" TH="$3" python3 - << 'PY'
import json, os, subprocess
d = f"packages/{os.environ['NAME']}"
rm = json.load(open(f"{d}/model/run_meta.json"))
rm["conditional"] = {"margin_th": float(os.environ['TH']), "cond_members": [1, 2]}
json.dump(rm, open(f"{d}/model/run_meta.json", "w"), ensure_ascii=False)
pp = json.load(open("packages/submit_tri_cond_rebuild/model/postproc.json"))
json.dump(pp, open(f"{d}/model/postproc.json", "w"), ensure_ascii=False)
zp = f"{d}.zip"
if os.path.exists(zp): os.remove(zp)
subprocess.run(["zip", "-qr", os.path.abspath(zp), "."], cwd=d, check=True)
print(f"{os.environ['NAME']}: zip={os.path.getsize(zp)/1e9:.3f}GB")
PY
  python3 sim/check_zip.py "packages/$1.zip" 2>&1 | tail -1
}

build submit_c0_wk30 "0.55,0.15,0.30" 0.5
build submit_c0_th55 "0.6,0.15,0.25" 0.55
