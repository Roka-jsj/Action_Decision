#!/bin/bash
# thasym95 조립 (R60 최종 좌표) — th85 클론 + margin_th 0.95 + member_th {1:0.95, 2:0.75}.
# mdeb@384 유지(S1 기각), gen_rescue 유지. 발사는 30k 리플레이(<=555s)·3자 서명 후.
set -euo pipefail
R=/root/Action_Decision; cd $R
SRC=packages/submit_th85
DST=packages/submit_thasym95

[ -d "$SRC" ] || { echo "원본 $SRC 없음"; exit 1; }
rm -rf "$DST" packages/submit_thasym95.zip
cp -al "$SRC" "$DST"
rm "$DST/model/ad_lib.py" "$DST/model/run_meta.json"

python3 - << 'PY'
import json, os
R = "/root/Action_Decision"
SRC = f"{R}/packages/submit_th85"
DST = f"{R}/packages/submit_thasym95"

src = open(f"{R}/common/ad_lib.py", encoding="utf-8").read()
s0 = src.find("# ==================== POSMAP_BLOCK_START")
s1 = src.find("# ==================== POSMAP_BLOCK_END ====================")
if s0 >= 0 and s1 > s0:
    src = src[:s0] + src[s1 + len("# ==================== POSMAP_BLOCK_END ===================="):]
compile(src, "ad_lib.py", "exec")
assert "member_th" in src, "신형 ad_lib 에 member_th 경로 없음"
open(f"{DST}/model/ad_lib.py", "w", encoding="utf-8").write(src)

rm = json.load(open(f"{SRC}/model/run_meta.json"))
assert rm.get("gen_rescue") is True and rm["conditional"]["margin_th"] == 0.85
assert rm["ensemble"][1].get("max_len") == 384, "th85 원본이 mdeb@384 가 아님"
rm["conditional"]["margin_th"] = 0.95
rm["conditional"]["member_th"] = {"1": 0.95, "2": 0.75}
json.dump(rm, open(f"{DST}/model/run_meta.json", "w"), ensure_ascii=False)

rm_src = json.load(open(f"{SRC}/model/run_meta.json"))
rm_dst = json.load(open(f"{DST}/model/run_meta.json"))
d = json.loads(json.dumps(rm_dst))
d["conditional"].pop("member_th")
d["conditional"]["margin_th"] = 0.85
assert d == rm_src, "run_meta 차분이 margin_th/member_th 외에도 존재"
for mdir in ("m1", "m2", "m3"):
    for f in os.listdir(f"{SRC}/model/{mdir}"):
        assert os.stat(f"{SRC}/model/{mdir}/{f}").st_ino == os.stat(f"{DST}/model/{mdir}/{f}").st_ino
assert os.stat(f"{SRC}/model/postproc.json").st_ino == os.stat(f"{DST}/model/postproc.json").st_ino
print("[thasym95] margin_th=0.95 + member_th{1:0.95,2:0.75} 패치 + 차분검증 OK (멤버·bias = th85 동일)")
PY

(cd "$DST" && zip -qr "$R/packages/submit_thasym95.zip" .)
ls -la packages/submit_thasym95.zip
python3 sim/check_zip.py packages/submit_thasym95.zip
