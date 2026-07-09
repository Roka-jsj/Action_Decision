#!/bin/bash
# tt30 조립 (R51 아침 큐 4번 후보) — cw45 패키지 클론 + 2단 조건부 가중(run_meta stages) 패치.
#
# tt30 = cw45(0.45/0.35/0.20, th0.6) + deep-margin 2단: margin<0.3 행만 0.40/0.35/0.25 재혼합.
# 레드팀 R51 fold0 클린 +0.00091. cond 추론 행수는 cw45 와 동일(가중만 다름 → 시간 영향 0).
#
# 멤버·구bias·직렬화 전부 cw45 와 동일해야 하므로 재조립 대신 하드링크 클론(cp -al):
# 원본 packages/submit_cw45 는 절대 무수정(교체 파일은 rm 후 새 inode 생성).
# 발사는 3자 서명 후 — 이 스크립트는 조립·검증만.
set -euo pipefail
R=/root/Action_Decision; cd $R
SRC=packages/submit_cw45
DST=packages/submit_tt30

[ -d "$SRC" ] || { echo "원본 $SRC 없음"; exit 1; }
rm -rf "$DST" packages/submit_tt30.zip
cp -al "$SRC" "$DST"                      # 하드링크 클론 (디스크 ~0, 원본 무수정)
rm "$DST/model/ad_lib.py" "$DST/model/run_meta.json"   # 교체 대상만 링크 해제

python3 - << 'PY'
import json, os
R = "/root/Action_Decision"
SRC = f"{R}/packages/submit_cw45"
DST = f"{R}/packages/submit_tt30"

# 1) ad_lib 배포본: package_ensemble.py 와 동일한 POSMAP 블록 스트립 + 구문 검증
src = open(f"{R}/common/ad_lib.py", encoding="utf-8").read()
s0 = src.find("# ==================== POSMAP_BLOCK_START")
s1 = src.find("# ==================== POSMAP_BLOCK_END ====================")
if s0 >= 0 and s1 > s0:
    src = src[:s0] + src[s1 + len("# ==================== POSMAP_BLOCK_END ===================="):]
compile(src, "ad_lib.py", "exec")
assert "stages" in src, "신형 ad_lib 에 stages 경로 없음"
open(f"{DST}/model/ad_lib.py", "w", encoding="utf-8").write(src)

# 2) run_meta: cw45 원본 + conditional.stages 만 추가 (사전등록 tt30 좌표)
rm = json.load(open(f"{SRC}/model/run_meta.json"))
assert rm["weights"] == [0.45, 0.35, 0.2] and rm["conditional"]["margin_th"] == 0.6, \
    f"cw45 좌표 아님: {rm.get('weights')} th{rm['conditional'].get('margin_th')}"
rm["conditional"]["stages"] = [
    {"th": 0.6, "weights": [0.45, 0.35, 0.20]},
    {"th": 0.3, "weights": [0.40, 0.35, 0.25]},
]
json.dump(rm, open(f"{DST}/model/run_meta.json", "w"), ensure_ascii=False)

# 3) 차분 검증: run_meta 는 stages 외 동일, postproc(구bias)·멤버 파일은 하드링크(=바이트 동일)
rm_src = json.load(open(f"{SRC}/model/run_meta.json"))
rm_dst = json.load(open(f"{DST}/model/run_meta.json"))
d = dict(rm_dst); d["conditional"] = {k: v for k, v in rm_dst["conditional"].items() if k != "stages"}
assert d == rm_src, "run_meta 차분이 stages 외에도 존재"
assert os.stat(f"{SRC}/model/postproc.json").st_ino == os.stat(f"{DST}/model/postproc.json").st_ino
for mdir in ("m1", "m2", "m3"):
    for f in os.listdir(f"{SRC}/model/{mdir}"):
        a, b = f"{SRC}/model/{mdir}/{f}", f"{DST}/model/{mdir}/{f}"
        assert os.stat(a).st_ino == os.stat(b).st_ino, f"멤버 파일 비링크: {b}"
print("[tt30] run_meta stages 패치 + 차분검증 OK (멤버·bias = cw45 하드링크 동일)")
PY

(cd "$DST" && zip -qr "$R/packages/submit_tt30.zip" .)
ls -la packages/submit_tt30.zip
python3 sim/check_zip.py packages/submit_tt30.zip
