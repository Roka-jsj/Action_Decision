#!/bin/bash
# R42 C0 조립: 조원 m1(q8) + 조원 mdeb(fp16) + 우리 klue(int8) — 3자 락 사양
# 가중 0.6/0.15/0.25(klue=v4슬롯), th0.5, cond[1,2], 구 bias(tri_cond_rebuild본=조원본과 동일)
# 사전등록: LB 중앙 0.7863 [0.7855, 0.7872], 시간 430s [410,470]
set -euo pipefail
R=/root/Action_Decision; cd $R
KLUE_Q8=$R/work/member_kluefull_q8.zip
[ -f "$KLUE_Q8" ] || { echo "klue q8 없음 — 먼저: python3 sim/quantize_member.py work/member_kluefull.zip --out $KLUE_Q8"; exit 1; }
G="action_decision_maximum/experiments/teacher_largev6[AB]_a*.npz"

python3 sim/package_ensemble.py --out submit_c0_klue \
  --member work/cc_members/m1::v6 \
  --member work/cc_members/mdeb::v6 \
  --member "$KLUE_Q8::v6" \
  --weights 0.6,0.15,0.25 --margin_th 0.5 --bias "$G,$G,$G" \
  --version v6 --max_len 320 --batch 128 > /dev/null

python3 - << 'PY'
import json, os, subprocess
d = "packages/submit_c0_klue"
rm = json.load(open(f"{d}/model/run_meta.json"))
rm["conditional"] = {"margin_th": 0.5, "cond_members": [1, 2]}
json.dump(rm, open(f"{d}/model/run_meta.json", "w"), ensure_ascii=False)
pp = json.load(open("packages/submit_tri_cond_rebuild/model/postproc.json"))
json.dump(pp, open(f"{d}/model/postproc.json", "w"), ensure_ascii=False)
zp = "packages/submit_c0_klue.zip"
if os.path.exists(zp): os.remove(zp)
subprocess.run(["zip", "-qr", os.path.abspath(zp), "."], cwd=d, check=True)
print(f"c0_klue: zip={os.path.getsize(zp)/1e9:.3f}GB (게이트 <1.0)")
PY
python3 sim/check_zip.py packages/submit_c0_klue.zip 2>&1 | tail -2
echo "[다음] 30k 검증: bash sim/prep_verify.sh packages/submit_c0_klue.zip && docker exec mun-jtest bash /share/verify/submit_c0_klue/run.sh"
