#!/bin/bash
# powmean 조립 (R64) — th85 클론 + conditional.combiner={"kind":"powmean","p":-1.0} 단일 추가
#   + ad_lib 신선본(멱평균 조합기 배선). 멤버 m1/m2/m3·bias(postproc)·gen_rescue·mdeb@384 전부 th85 동일.
# 조합기는 이미 계산된 확률의 elementwise 수학 → 시간중립(th85 558s 그대로).
# best p = -1.0 (배포 metric=argmax(log(P+eps)+bias) 기준 fold0 +0.00133 / 5k +0.00159, 둘 다 양수·최상 worst-case).
# 발사는 parity(test_powmean 전PASS) + 30k 리플레이(<=555s) + 3자 서명 후.
set -euo pipefail
R=/root/Action_Decision; cd $R
SRC=packages/submit_th85
DST=packages/submit_powmean
PVAL=-1.0

[ -d "$SRC" ] || { echo "원본 $SRC 없음"; exit 1; }

# 앵커(SRC) 사전 스냅샷 — 조립 후 불변 증명용
SRC_RM_SHA=$(sha256sum "$SRC/model/run_meta.json" | cut -d' ' -f1)
SRC_AL_SHA=$(sha256sum "$SRC/model/ad_lib.py" | cut -d' ' -f1)

rm -rf "$DST" packages/submit_powmean.zip
cp -al "$SRC" "$DST"
# 변경 대상만 hardlink 끊고 실사본으로 — SRC 오염 방지
rm -f "$DST/model/ad_lib.py" "$DST/model/run_meta.json"
rm -rf "$DST/model/__pycache__"

PVAL=$PVAL python3 - << 'PY'
import json, os
R = "/root/Action_Decision"
SRC = f"{R}/packages/submit_th85"
DST = f"{R}/packages/submit_powmean"
PVAL = float(os.environ["PVAL"])

# ad_lib 신선본 (POSMAP 블록 제거 관행 유지)
src = open(f"{R}/common/ad_lib.py", encoding="utf-8").read()
s0 = src.find("# ==================== POSMAP_BLOCK_START")
s1 = src.find("# ==================== POSMAP_BLOCK_END ====================")
if s0 >= 0 and s1 > s0:
    src = src[:s0] + src[s1 + len("# ==================== POSMAP_BLOCK_END ===================="):]
compile(src, "ad_lib.py", "exec")
assert "_parse_combiner" in src and "_pm_inv_renorm" in src, "신형 ad_lib 에 멱평균 조합기 배선 없음"
open(f"{DST}/model/ad_lib.py", "w", encoding="utf-8").write(src)

# run_meta: conditional.combiner 단일 추가 — 차분 단일성 증명(제거하면 SRC 와 동일해야 함)
rm = json.load(open(f"{SRC}/model/run_meta.json"))
assert rm.get("gen_rescue") is True and rm["conditional"]["margin_th"] == 0.85
assert rm["ensemble"][1].get("max_len") == 384, "th85 원본이 mdeb@384 가 아님"
assert "combiner" not in rm["conditional"]
rm["conditional"]["combiner"] = {"kind": "powmean", "p": PVAL}
json.dump(rm, open(f"{DST}/model/run_meta.json", "w"), ensure_ascii=False)

d = json.load(open(f"{DST}/model/run_meta.json"))
d["conditional"].pop("combiner")
assert d == json.load(open(f"{SRC}/model/run_meta.json")), "run_meta 차분이 combiner 외에도 존재"

# 멤버 m1/m2/m3·postproc 는 SRC 와 inode 동일(불변 공유) 증명
for mdir in ("m1", "m2", "m3"):
    for f in os.listdir(f"{SRC}/model/{mdir}"):
        assert os.stat(f"{SRC}/model/{mdir}/{f}").st_ino == os.stat(f"{DST}/model/{mdir}/{f}").st_ino, \
            f"{mdir}/{f} inode 불일치(멤버 변경됨)"
assert os.stat(f"{SRC}/model/postproc.json").st_ino == os.stat(f"{DST}/model/postproc.json").st_ino, "postproc 변경됨"
print(f"[powmean] combiner=powmean p={PVAL} 단일차분 + ad_lib 신선본 + m1/m2/m3/postproc inode공유 검증 OK")
PY

# 앵커 불변 증명
[ "$(sha256sum "$SRC/model/run_meta.json" | cut -d' ' -f1)" = "$SRC_RM_SHA" ] || { echo "!!! SRC run_meta 오염"; exit 1; }
[ "$(sha256sum "$SRC/model/ad_lib.py" | cut -d' ' -f1)" = "$SRC_AL_SHA" ] || { echo "!!! SRC ad_lib 오염"; exit 1; }
echo "[powmean] 앵커(th85) 불변 SHA 증명 OK"

(cd "$DST" && zip -qr "$R/packages/submit_powmean.zip" .)
ls -la packages/submit_powmean.zip
python3 sim/check_zip.py packages/submit_powmean.zip
