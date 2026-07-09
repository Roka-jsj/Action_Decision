#!/bin/bash
# R43 C1: C0에서 m1만 우리 v6-11ep FULL(q8)로 교체 — 단일변경
# 사전등록: 중앙 0.7864, 밴드 [0.7853, 0.7880]. <0.78567(C0)이면 m1 교체 폐기.
set -euo pipefail
R=/root/Action_Decision; cd $R
M1_Q8=$R/work/member_largev6_11ep_q8.zip
[ -f "$M1_Q8" ] || { echo "먼저: python3 sim/quantize_member.py work/member_largev6_11ep.zip --out $M1_Q8"; exit 1; }
G="action_decision_maximum/experiments/teacher_largev6[AB]_a*.npz"
python3 sim/package_ensemble.py --out submit_c1_11ep \
  --member "$M1_Q8::v6" \
  --member work/cc_members/mdeb::v6 \
  --member work/member_kluefull_q8.zip::v6 \
  --weights 0.6,0.15,0.25 --margin_th 0.5 --bias "$G,$G,$G" \
  --version v6 --max_len 320 --batch 128 > /dev/null
python3 - << 'PY'
import json, os, subprocess
d = "packages/submit_c1_11ep"
rm = json.load(open(f"{d}/model/run_meta.json"))
rm["conditional"] = {"margin_th": 0.5, "cond_members": [1, 2]}
json.dump(rm, open(f"{d}/model/run_meta.json", "w"), ensure_ascii=False)
pp = json.load(open("packages/submit_tri_cond_rebuild/model/postproc.json"))
json.dump(pp, open(f"{d}/model/postproc.json", "w"), ensure_ascii=False)
zp = "packages/submit_c1_11ep.zip"
if os.path.exists(zp): os.remove(zp)
subprocess.run(["zip", "-qr", os.path.abspath(zp), "."], cwd=d, check=True)
print(f"c1_11ep: zip={os.path.getsize(zp)/1e9:.3f}GB")
PY
python3 sim/check_zip.py packages/submit_c1_11ep.zip 2>&1 | tail -2
