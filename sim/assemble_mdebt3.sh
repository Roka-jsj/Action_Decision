#!/bin/bash
# mdebt3 조립 — th85 클론 + m2만 mdeb-T3 FULL(10ep FGM+rescue+mht12@320)의 **fp16 원본**으로 교체.
# 실행 전제: DONE_mdebt3full + 게이트E GO + 3자 서명. (R65 레드팀 게이트A 사양 채택:
#   m2 fp16 유지 = th85 대칭·양자화 잡음 0·게이트E 계기와 배포본 동일물. 전략가 초안의 q8안은 기각.)
# 차분 사전등록: run_meta ensemble[1] 2키 — max_len 384 제거→320, mht:12 추가. (m1t3 단일차분 선례
#   이탈은 R65 레드팀 §3 게이트A + max_len=320 판정(훈련정합·계기정합·시간절약 3중 근거)으로 서명.)
set -euo pipefail
R=/root/Action_Decision; cd $R
SRC=packages/submit_th85
DST=packages/submit_mdebt3
MEMBER=work/member_mdebt3full

[ -d "$SRC" ] || { echo "원본 $SRC 없음"; exit 1; }
[ -f "$R/work/DONE_mdebt3full" ] || { echo "mdeb-T3 FULL 미완료"; exit 1; }
[ -d "$MEMBER" ] || { echo "멤버 디렉터리 $MEMBER 없음"; exit 1; }

# 앵커(SRC) 사전 스냅샷
SRC_RM_SHA=$(sha256sum "$SRC/model/run_meta.json" | cut -d' ' -f1)
SRC_AL_SHA=$(sha256sum "$SRC/model/ad_lib.py" | cut -d' ' -f1)

rm -rf "$DST" packages/submit_mdebt3.zip
cp -al "$SRC" "$DST"
# 교체 대상 hardlink 절단 후 실사본 (codex R63 파국모드 1순위 차단)
rm -f "$DST/model/ad_lib.py" "$DST/model/run_meta.json"
rm -rf "$DST/model/m2" "$DST/model/__pycache__"
cp -a "$MEMBER" "$DST/model/m2"

python3 - << 'PY'
import json, os, re
R = "/root/Action_Decision"
SRC = f"{R}/packages/submit_th85"
DST = f"{R}/packages/submit_mdebt3"

# ad_lib 신선본 (POSMAP 제거 관행)
src = open(f"{R}/common/ad_lib.py", encoding="utf-8").read()
s0 = src.find("# ==================== POSMAP_BLOCK_START")
s1 = src.find("# ==================== POSMAP_BLOCK_END ====================")
if s0 >= 0 and s1 > s0:
    src = src[:s0] + src[s1 + len("# ==================== POSMAP_BLOCK_END ===================="):]
compile(src, "ad_lib.py", "exec")
assert "_member_mht" in src, "신형 ad_lib에 멤버별 mht 배선 없음"
open(f"{DST}/model/ad_lib.py", "w", encoding="utf-8").write(src)

# run_meta: 정확히 2키 차분 (m2.max_len 384 제거→320 + m2.mht=12)
rm = json.load(open(f"{SRC}/model/run_meta.json"))
assert rm.get("gen_rescue") is True and rm["conditional"]["margin_th"] == 0.85
assert rm["ensemble"][1]["dir"] == "m2" and "mht" not in rm["ensemble"][1]
assert rm["ensemble"][1].get("max_len") == 384, "th85 원본이 mdeb@384가 아님"
rm["ensemble"][1]["mht"] = 12
rm["ensemble"][1]["max_len"] = 320
json.dump(rm, open(f"{DST}/model/run_meta.json", "w"), ensure_ascii=False)
d = json.loads(open(f"{DST}/model/run_meta.json").read())
d["ensemble"][1].pop("mht"); d["ensemble"][1]["max_len"] = 384
assert d == json.load(open(f"{SRC}/model/run_meta.json")), "run_meta 차분이 2키 외에도 존재"

# m2: fp16 원본 세트 (레드팀 게이트A — qweights 금지, th85 대칭)
m2 = f"{DST}/model/m2"
fs = set(os.listdir(m2))
assert "model.safetensors" in fs, f"fp16 원본 없음: {fs}"
assert "qweights.npz" not in fs, "양자화본 혼입 — fp16 사양 위반"
for need in ("config.json", "id_map.npy", "prune_meta.json", "tokenizer.json",
             "spm.model", "added_tokens.json", "special_tokens_map.json"):
    assert need in fs, f"{need} 없음: {fs}"

# vocab 정합: config.json == 학습로그 [prune] 값 (하드코딩 금지 — h12 프룬은 h8과 다를 수 있음)
log = open(f"{R}/work/mdebt3full.log", encoding="utf-8", errors="ignore").read()
m = re.search(r"\[prune\][^\d]*(\d{4,6})", log)
assert m, "학습로그에 [prune] 어휘수 없음"
vocab_log = int(m.group(1))
cfg = json.load(open(f"{m2}/config.json"))
assert cfg["vocab_size"] == vocab_log, f"vocab 불일치: config {cfg['vocab_size']} != log {vocab_log}"

# tokenizer 지문: 학습멤버 tokenizer.json == member_mdebfull의 것 (protobuf 변환 드리프트 방어, 레드팀 게이트B 신설항)
import hashlib
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
assert sha(f"{m2}/tokenizer.json") == sha(f"{R}/work/member_mdebfull/tokenizer.json"), \
    "tokenizer.json 지문 불일치 — protobuf 변환 드리프트 의심"

# m2 실사본(hardlink 비공유) + m1/m3/postproc inode 공유 증명
for f in fs:
    old = f"{SRC}/model/m2/{f}"
    if os.path.exists(old):
        assert os.stat(old).st_ino != os.stat(f"{m2}/{f}").st_ino, f"m2/{f} SRC와 hardlink 공유"
for mdir in ("m1", "m3"):
    for f in os.listdir(f"{SRC}/model/{mdir}"):
        assert os.stat(f"{SRC}/model/{mdir}/{f}").st_ino == os.stat(f"{DST}/model/{mdir}/{f}").st_ino
assert os.stat(f"{SRC}/model/postproc.json").st_ino == os.stat(f"{DST}/model/postproc.json").st_ino
print(f"[mdebt3] 게이트A 구조증명 전부 OK (vocab={vocab_log}, fp16, 2키 차분, 격리·공유 증명)")
PY

# 앵커 불변 증명
[ "$(sha256sum "$SRC/model/run_meta.json" | cut -d' ' -f1)" = "$SRC_RM_SHA" ] || { echo "!!! SRC run_meta 오염"; exit 1; }
[ "$(sha256sum "$SRC/model/ad_lib.py" | cut -d' ' -f1)" = "$SRC_AL_SHA" ] || { echo "!!! SRC ad_lib 오염"; exit 1; }
echo "[mdebt3] 앵커(th85) 불변 SHA 증명 OK"

(cd "$DST" && zip -qr "$R/packages/submit_mdebt3.zip" .)
ls -la packages/submit_mdebt3.zip
python3 sim/check_zip.py packages/submit_mdebt3.zip
