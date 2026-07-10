#!/bin/bash
# m1t3 조립 (R63) — th85 클론 + m1만 m1-T3 FULL(8ep, rescue+mht12 학습)의 int8 양자화본으로 교체
#   + run_meta m1 엔트리 "mht":12 + ad_lib 신선본(멤버별 mht 배선, cd28165).
# codex R63 조건부GO 게이트: hardlink 앵커 오염 방지(교체 대상 unlink 후 실사본), SRC 불변 SHA 증명,
#   run_meta 차분 단일성, qweights 단독 로드(model.safetensors 부재), m2/m3/postproc inode 동일.
# 발사는 parity 이중대조 + 30k 리플레이 + 3자 서명 후.
set -euo pipefail
R=/root/Action_Decision; cd $R
SRC=packages/submit_th85
DST=packages/submit_m1t3
MQ8=work/member_m1t3full_q8.zip

[ -d "$SRC" ] || { echo "원본 $SRC 없음"; exit 1; }
[ -f "$R/work/DONE_m1t3full" ] || { echo "m1-T3 FULL 미완료(DONE_m1t3full 없음)"; exit 1; }
if [ ! -f "$MQ8" ]; then
    echo "[m1t3] int8 양자화: work/member_m1t3full.zip -> $MQ8"
    python3 sim/quantize_member.py work/member_m1t3full.zip --out "$MQ8"
fi

# 앵커(SRC) 사전 스냅샷 — 조립 후 불변 증명용
SRC_RM_SHA=$(sha256sum "$SRC/model/run_meta.json" | cut -d' ' -f1)
SRC_AL_SHA=$(sha256sum "$SRC/model/ad_lib.py" | cut -d' ' -f1)

rm -rf "$DST" packages/submit_m1t3.zip
cp -al "$SRC" "$DST"
# 교체 대상은 hardlink 끊고 실사본으로 — SRC 오염 방지 (codex R63 파국모드 1순위)
rm -f "$DST/model/ad_lib.py" "$DST/model/run_meta.json"
rm -rf "$DST/model/m1" "$DST/model/__pycache__"
mkdir "$DST/model/m1"
python3 - <<PY
import zipfile
with zipfile.ZipFile("$MQ8") as z:
    z.extractall("$DST/model/m1")
PY

python3 - << 'PY'
import json, os, hashlib
R = "/root/Action_Decision"
SRC = f"{R}/packages/submit_th85"
DST = f"{R}/packages/submit_m1t3"

# ad_lib 신선본 (POSMAP 블록 제거 관행 유지)
src = open(f"{R}/common/ad_lib.py", encoding="utf-8").read()
s0 = src.find("# ==================== POSMAP_BLOCK_START")
s1 = src.find("# ==================== POSMAP_BLOCK_END ====================")
if s0 >= 0 and s1 > s0:
    src = src[:s0] + src[s1 + len("# ==================== POSMAP_BLOCK_END ===================="):]
compile(src, "ad_lib.py", "exec")
assert "_member_mht" in src, "신형 ad_lib 에 멤버별 mht 배선 없음"
open(f"{DST}/model/ad_lib.py", "w", encoding="utf-8").write(src)

# run_meta: m1 엔트리에 mht:12 — 차분 단일성 증명(제거하면 SRC 와 동일해야 함)
rm = json.load(open(f"{SRC}/model/run_meta.json"))
assert rm.get("gen_rescue") is True and rm["conditional"]["margin_th"] == 0.85
assert rm["ensemble"][0]["dir"] == "m1" and "mht" not in rm["ensemble"][0]
assert rm["ensemble"][1].get("max_len") == 384, "th85 원본이 mdeb@384 가 아님"
rm["ensemble"][0]["mht"] = 12
json.dump(rm, open(f"{DST}/model/run_meta.json", "w"), ensure_ascii=False)
d = json.loads(open(f"{DST}/model/run_meta.json").read())
d["ensemble"][0].pop("mht")
assert d == json.load(open(f"{SRC}/model/run_meta.json")), "run_meta 차분이 m1.mht 외에도 존재"

# m1: qweights 단독 로드 보장 + 필수 파일
m1 = f"{DST}/model/m1"
fs = set(os.listdir(m1))
assert "qweights.npz" in fs, f"qweights.npz 없음: {fs}"
assert "model.safetensors" not in fs, "fp 원본 잔존 — loader 오선택 위험"
for need in ("config.json", "id_map.npy", "prune_meta.json", "tokenizer.json"):
    assert need in fs, f"{need} 없음: {fs}"
# m1 은 SRC 와 hardlink 비공유(실사본) 증명
for f in fs:
    ino = os.stat(f"{m1}/{f}").st_ino
    old = f"{SRC}/model/m1/{f}"
    assert not (os.path.exists(old) and os.stat(old).st_ino == ino), f"m1/{f} 가 SRC 와 hardlink 공유"

# m2/m3/postproc 는 SRC 와 inode 동일(불변 공유) 증명
for mdir in ("m2", "m3"):
    for f in os.listdir(f"{SRC}/model/{mdir}"):
        assert os.stat(f"{SRC}/model/{mdir}/{f}").st_ino == os.stat(f"{DST}/model/{mdir}/{f}").st_ino
assert os.stat(f"{SRC}/model/postproc.json").st_ino == os.stat(f"{DST}/model/postproc.json").st_ino
print("[m1t3] run_meta(m1.mht=12 단일차분)·qweights 단독·hardlink 격리·m2/m3 공유 전부 검증 OK")
PY

# 앵커 불변 증명
[ "$(sha256sum "$SRC/model/run_meta.json" | cut -d' ' -f1)" = "$SRC_RM_SHA" ] || { echo "!!! SRC run_meta 오염"; exit 1; }
[ "$(sha256sum "$SRC/model/ad_lib.py" | cut -d' ' -f1)" = "$SRC_AL_SHA" ] || { echo "!!! SRC ad_lib 오염"; exit 1; }
echo "[m1t3] 앵커(th85) 불변 SHA 증명 OK"

(cd "$DST" && zip -qr "$R/packages/submit_m1t3.zip" .)
ls -la packages/submit_m1t3.zip
python3 sim/check_zip.py packages/submit_m1t3.zip
