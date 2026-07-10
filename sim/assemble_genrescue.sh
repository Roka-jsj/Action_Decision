#!/bin/bash
# GEN-rescue 조립 (R53) — th75 패키지 클론 + run_meta "gen_rescue": true + 신형 ad_lib.
#
# GEN-rescue = 좌측절단으로 [GEN] 헤더가 삭제될 행(fold0-val 11.4%)만 헤더보존 절단.
# 비대상 행 input_ids 불변(byte-identity 회귀 증명), cond 추론 행수/시간 영향 0(가중·입력만).
# 게이트: sim/eval_genrescue.py (a')acc>=0.753 (b')byte-identity (c)F1>=0 (d')mdeb부호 (e)시간.
#
# 멤버·구bias·직렬화 = th75 와 동일해야 하므로 하드링크 클론(cp -al):
# 원본 packages/submit_th75 는 절대 무수정(교체 파일만 rm 후 새 inode).
# 발사는 3자 서명 후 — 이 스크립트는 조립·검증만.
set -euo pipefail
R=/root/Action_Decision; cd $R
SRC=packages/submit_th75
DST=packages/submit_genrescue

[ -d "$SRC" ] || { echo "원본 $SRC 없음"; exit 1; }
rm -rf "$DST" packages/submit_genrescue.zip
cp -al "$SRC" "$DST"
rm "$DST/model/ad_lib.py" "$DST/model/run_meta.json"

python3 - << 'PY'
import json, os
R = "/root/Action_Decision"
SRC = f"{R}/packages/submit_th75"
DST = f"{R}/packages/submit_genrescue"

# 1) ad_lib 배포본: POSMAP 스트립(package_ensemble.py 동일) + 구문 검증 + rescue 경로 확인
src = open(f"{R}/common/ad_lib.py", encoding="utf-8").read()
s0 = src.find("# ==================== POSMAP_BLOCK_START")
s1 = src.find("# ==================== POSMAP_BLOCK_END ====================")
if s0 >= 0 and s1 > s0:
    src = src[:s0] + src[s1 + len("# ==================== POSMAP_BLOCK_END ===================="):]
compile(src, "ad_lib.py", "exec")
assert "_gen_rescue_ids" in src and "gen_rescue" in src, "신형 ad_lib 에 gen_rescue 경로 없음"
open(f"{DST}/model/ad_lib.py", "w", encoding="utf-8").write(src)

# 2) run_meta: th75 원본 + gen_rescue 플래그만 추가
rm = json.load(open(f"{SRC}/model/run_meta.json"))
assert rm["weights"] == [0.45, 0.4, 0.15] and rm["conditional"]["margin_th"] == 0.75, \
    f"th75 좌표 아님: {rm.get('weights')} th{rm['conditional'].get('margin_th')}"
rm["gen_rescue"] = True
json.dump(rm, open(f"{DST}/model/run_meta.json", "w"), ensure_ascii=False)

# 3) 차분 검증: run_meta 는 gen_rescue 외 동일, bias·멤버 파일은 하드링크(바이트 동일)
rm_src = json.load(open(f"{SRC}/model/run_meta.json"))
rm_dst = json.load(open(f"{DST}/model/run_meta.json"))
d = {k: v for k, v in rm_dst.items() if k != "gen_rescue"}
assert d == rm_src, "run_meta 차분이 gen_rescue 외에도 존재"
assert os.stat(f"{SRC}/model/postproc.json").st_ino == os.stat(f"{DST}/model/postproc.json").st_ino
for mdir in ("m1", "m2", "m3"):
    for f in os.listdir(f"{SRC}/model/{mdir}"):
        assert os.stat(f"{SRC}/model/{mdir}/{f}").st_ino == os.stat(f"{DST}/model/{mdir}/{f}").st_ino, \
            f"멤버 파일 비링크: {mdir}/{f}"
print("[genrescue] run_meta gen_rescue=true 패치 + 차분검증 OK (멤버·bias = th75 하드링크 동일)")
PY

(cd "$DST" && zip -qr "$R/packages/submit_genrescue.zip" .)
ls -la packages/submit_genrescue.zip
python3 sim/check_zip.py packages/submit_genrescue.zip
