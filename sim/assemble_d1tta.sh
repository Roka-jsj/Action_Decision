#!/bin/bash
# D1 CompressView-TTA 조립 (R55) — genrescue 패키지 클론 + run_meta compress_tta 단일변경.
#
# 주의: 게이트 ⑤(시간 <=525s 추정)가 3멤버 스펙 구성에서 FAIL(534s 추정) —
# 2멤버(m1+mdeb, klue 제외) 옵션은 520s 로 통과. 조립 구성은 3자 서명으로 결정:
#   bash sim/assemble_d1tta.sh                 # 스펙 구성(3멤버 공통)
#   AD_TTA_MEMBERS="0,1" bash sim/assemble_d1tta.sh   # klue 제외 옵션
# 발사는 3자 몫. 원본 packages/submit_genrescue 무수정(하드링크 클론).
set -euo pipefail
R=/root/Action_Decision; cd $R
SRC=packages/submit_genrescue
DST=packages/submit_d1tta

[ -d "$SRC" ] || { echo "원본 $SRC 없음"; exit 1; }
rm -rf "$DST" packages/submit_d1tta.zip
cp -al "$SRC" "$DST"
rm "$DST/model/ad_lib.py" "$DST/model/run_meta.json"

python3 - << 'PY'
import json, os
R = "/root/Action_Decision"
SRC = f"{R}/packages/submit_genrescue"
DST = f"{R}/packages/submit_d1tta"

src = open(f"{R}/common/ad_lib.py", encoding="utf-8").read()
s0 = src.find("# ==================== POSMAP_BLOCK_START")
s1 = src.find("# ==================== POSMAP_BLOCK_END ====================")
if s0 >= 0 and s1 > s0:
    src = src[:s0] + src[s1 + len("# ==================== POSMAP_BLOCK_END ===================="):]
compile(src, "ad_lib.py", "exec")
assert "serialize_compress" in src and "compress_tta" in src, "신형 ad_lib 에 TTA 경로 없음"
open(f"{DST}/model/ad_lib.py", "w", encoding="utf-8").write(src)

rm = json.load(open(f"{SRC}/model/run_meta.json"))
assert rm.get("gen_rescue") is True, "원본이 genrescue 패키지가 아님"
tta = {"lambda": 0.5, "margin_th": 0.5}
mem = os.environ.get("AD_TTA_MEMBERS", "")
if mem:
    tta["members"] = [int(x) for x in mem.split(",")]
rm["compress_tta"] = tta
json.dump(rm, open(f"{DST}/model/run_meta.json", "w"), ensure_ascii=False)

rm_src = json.load(open(f"{SRC}/model/run_meta.json"))
rm_dst = json.load(open(f"{DST}/model/run_meta.json"))
d = {k: v for k, v in rm_dst.items() if k != "compress_tta"}
assert d == rm_src, "run_meta 차분이 compress_tta 외에도 존재"
for mdir in ("m1", "m2", "m3"):
    for f in os.listdir(f"{SRC}/model/{mdir}"):
        assert os.stat(f"{SRC}/model/{mdir}/{f}").st_ino == os.stat(f"{DST}/model/{mdir}/{f}").st_ino
assert os.stat(f"{SRC}/model/postproc.json").st_ino == os.stat(f"{DST}/model/postproc.json").st_ino
print(f"[d1tta] compress_tta={tta} 패치 + 차분검증 OK (멤버·bias = genrescue 하드링크 동일)")
PY

(cd "$DST" && zip -qr "$R/packages/submit_d1tta.zip" .)
ls -la packages/submit_d1tta.zip
python3 sim/check_zip.py packages/submit_d1tta.zip
