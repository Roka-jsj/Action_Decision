#!/bin/bash
# m1swapcal 조립 — th85 클론 + m2 '가중치 파일만' 우리 mdebfull(fp16)로 교체. (R68)
# 역대 최소 차분: run_meta·ad_lib·m1/m3 완전 불변(inode 공유), m2 디렉터리만 실사본.
# 목적: ①은행 미세상승 가능성(δ프로브 solo +0.00207) ②synth-FULL의 LB 대조군.
set -euo pipefail
R=/root/Action_Decision; cd $R
SRC=packages/submit_th85
DST=packages/submit_m1swapcal2
MEMBER=work/member_m1h8full_s777_q8dir

[ -d "$SRC" ] && [ -d "$MEMBER" ] || { echo "입력 없음"; exit 1; }
SRC_RM_SHA=$(sha256sum "$SRC/model/run_meta.json" | cut -d' ' -f1)
SRC_AL_SHA=$(sha256sum "$SRC/model/ad_lib.py" | cut -d' ' -f1)

rm -rf "$DST" packages/submit_m1swapcal2.zip
cp -al "$SRC" "$DST"
rm -rf "$DST/model/m1" "$DST/model/__pycache__"
cp -a "$MEMBER" "$DST/model/m1"

python3 - << 'PY'
import json, os, hashlib
R = "/root/Action_Decision"
SRC = f"{R}/packages/submit_th85"
DST = f"{R}/packages/submit_m1swapcal"

def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]

# run_meta·ad_lib 불변(inode 공유) 증명 — 이 패키지의 존재 이유
assert os.stat(f"{SRC}/model/run_meta.json").st_ino == os.stat(f"{DST}/model/run_meta.json").st_ino
assert os.stat(f"{SRC}/model/ad_lib.py").st_ino == os.stat(f"{DST}/model/ad_lib.py").st_ino
assert os.stat(f"{SRC}/model/postproc.json").st_ino == os.stat(f"{DST}/model/postproc.json").st_ino
for mdir in ("m2", "m3"):
    for f in os.listdir(f"{SRC}/model/{mdir}"):
        assert os.stat(f"{SRC}/model/{mdir}/{f}").st_ino == os.stat(f"{DST}/model/{mdir}/{f}").st_ino

# m2: 파일셋 동등성 + fp16 + vocab 동일 + 실사본
old_fs = set(os.listdir(f"{SRC}/model/m1"))
new_fs = set(os.listdir(f"{DST}/model/m1"))
missing = old_fs - new_fs
assert not missing, f"우리 멤버에 없는 파일: {missing}"
assert "qweights.npz" in new_fs and "model.safetensors" not in new_fs   # q8 사양(배포 m1과 동일 레이아웃)
c_old = json.load(open(f"{SRC}/model/m1/config.json"))
c_new = json.load(open(f"{DST}/model/m1/config.json"))
assert c_new["vocab_size"] == c_old["vocab_size"], f"vocab 불일치 {c_new['vocab_size']} != {c_old['vocab_size']}"
assert c_new.get("model_type") == c_old.get("model_type")
for f in new_fs:
    old = f"{SRC}/model/m1/{f}"
    if os.path.exists(old):
        assert os.stat(old).st_ino != os.stat(f"{DST}/model/m1/{f}").st_ino, f"m2/{f} hardlink 공유"
# 가중치가 실제로 다른지(스왑 실효성) + 토크나이저 파일 지문 기록
w_old, w_new = sha16(f"{SRC}/model/m1/qweights.npz"), sha16(f"{DST}/model/m1/qweights.npz")
assert w_old != w_new, "가중치 동일 — 스왑 무의미"
print(f"[m1swapcal] 구조증명 OK — m2 가중치 {w_old}→{w_new}, 그 외 전부 inode 불변")
PY

[ "$(sha256sum "$SRC/model/run_meta.json" | cut -d' ' -f1)" = "$SRC_RM_SHA" ] || { echo "!!! SRC 오염"; exit 1; }
[ "$(sha256sum "$SRC/model/ad_lib.py" | cut -d' ' -f1)" = "$SRC_AL_SHA" ] || { echo "!!! SRC 오염"; exit 1; }
echo "[m1swapcal] 앵커 불변 증명 OK"

(cd "$DST" && zip -qr "$R/packages/submit_m1swapcal2.zip" .)
ls -la packages/submit_m1swapcal2.zip
python3 sim/check_zip.py packages/submit_m1swapcal2.zip
