#!/bin/bash
# mdeb@384 조립 (R57 전략가 #9) — genrescue 클론 + m2(mdeb) max_len=384 단일변경.
# 전제: sim/probe_mdeb384.py 부호 양수(GO) + 3자 서명. 원본 무수정(하드링크 클론).
set -euo pipefail
R=/root/Action_Decision; cd $R
SRC=packages/submit_genrescue
DST=packages/submit_mdeb384

[ -d "$SRC" ] || { echo "원본 $SRC 없음"; exit 1; }
rm -rf "$DST" packages/submit_mdeb384.zip
cp -al "$SRC" "$DST"
rm "$DST/model/ad_lib.py" "$DST/model/run_meta.json"

python3 - << 'PY'
import json, os
R = "/root/Action_Decision"
SRC = f"{R}/packages/submit_genrescue"
DST = f"{R}/packages/submit_mdeb384"

src = open(f"{R}/common/ad_lib.py", encoding="utf-8").read()
s0 = src.find("# ==================== POSMAP_BLOCK_START")
s1 = src.find("# ==================== POSMAP_BLOCK_END ====================")
if s0 >= 0 and s1 > s0:
    src = src[:s0] + src[s1 + len("# ==================== POSMAP_BLOCK_END ===================="):]
compile(src, "ad_lib.py", "exec")
assert 'm.get("max_len", ml)' in src, "신형 ad_lib 에 멤버별 max_len 오버라이드 없음"
open(f"{DST}/model/ad_lib.py", "w", encoding="utf-8").write(src)

rm = json.load(open(f"{SRC}/model/run_meta.json"))
assert rm.get("gen_rescue") is True and rm["conditional"]["margin_th"] == 0.75
rm["ensemble"][1]["max_len"] = 384            # m2 = mdeb 만 384
json.dump(rm, open(f"{DST}/model/run_meta.json", "w"), ensure_ascii=False)

rm_src = json.load(open(f"{SRC}/model/run_meta.json"))
rm_dst = json.load(open(f"{DST}/model/run_meta.json"))
d = json.loads(json.dumps(rm_dst))
d["ensemble"][1].pop("max_len")
assert d == rm_src, "run_meta 차분이 m2.max_len 외에도 존재"
for mdir in ("m1", "m2", "m3"):
    for f in os.listdir(f"{SRC}/model/{mdir}"):
        assert os.stat(f"{SRC}/model/{mdir}/{f}").st_ino == os.stat(f"{DST}/model/{mdir}/{f}").st_ino
assert os.stat(f"{SRC}/model/postproc.json").st_ino == os.stat(f"{DST}/model/postproc.json").st_ino
print("[mdeb384] m2.max_len=384 패치 + 차분검증 OK (멤버·bias = genrescue 하드링크 동일)")
PY

(cd "$DST" && zip -qr "$R/packages/submit_mdeb384.zip" .)
ls -la packages/submit_mdeb384.zip
python3 sim/check_zip.py packages/submit_mdeb384.zip
