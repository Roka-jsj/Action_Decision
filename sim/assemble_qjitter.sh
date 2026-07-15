#!/bin/bash
# 양자화 지터 티켓 조립: 은행 기반, m1(필수) + m3(선택)을 SR 지터 q8로 교체.
# run_meta 불변(멤버 정체성 동일). 사용:
#   bash sim/assemble_qjitter.sh <out_tag> <qseed_m1> [qseed_m3]
set -euo pipefail
cd /root/Action_Decision
TAG="${1:?out tag}"; QS1="${2:?qseed m1}"; QS3="${3:-}"; AMP="${4:-1.0}"
# SRC_PKG: 스캐폴드 지정(기본 원은행). qj3 스캐폴드 사용 시 m3=SR505 유지됨(QS3 생략 권장)
SRC=${SRC_PKG:-packages/submit_sf_mht12}; DST=packages/submit_$TAG
[ "$DST" != "$SRC" ] || { echo "FATAL: DST==SRC(스캐폴드) 금지"; exit 1; }

# 잭팟 재중심: M1SRC 환경변수로 임의 멤버 fp16 zip을 지터 소스로 지정 가능 (기본=은행 m1sf)
M1SRC="${M1SRC:-work/member_m1_softf1.zip}"
[ -f "$M1SRC" ] || { echo "FATAL: M1SRC=$M1SRC 없음"; exit 1; }
SRCBASE=$(basename "$M1SRC" .zip)
Q1=work/${SRCBASE}_qj${QS1}_a${AMP}
[ -f "$Q1" ] || python3 sim/quantize_member_jitter.py "$M1SRC" --out "$Q1" --qseed "$QS1" --amp "$AMP"

rm -rf "$DST" "packages/submit_$TAG.zip"
cp -al "$SRC" "$DST"
rm -rf "$DST/model/m1" "$DST/model/__pycache__"; mkdir -p "$DST/model/m1"
(cd "$DST/model/m1" && unzip -q "/root/Action_Decision/$Q1")

if [ -n "$QS3" ]; then
  Q3=work/m3qj_${QS3}_a${AMP}
  [ -f "$Q3" ] || python3 sim/quantize_member_jitter.py work/member_m3_softf1.zip --out "$Q3" --qseed "$QS3" --amp "$AMP"
  rm -rf "$DST/model/m3"; mkdir -p "$DST/model/m3"
  (cd "$DST/model/m3" && unzip -q "/root/Action_Decision/$Q3")
fi

python3 - << PY
import json
m=json.load(open("$DST/model/run_meta.json"))
assert m['ensemble'][0].get('mht')==12, "m1 mht12 서빙 누락!"
print("run_meta 불변 OK:", m['ensemble'], m['weights'])
PY

# 은행과 바이트동일이면 슬롯낭비 — qweights 해시 상이 필수
H_BANK=$(md5sum packages/submit_sf_mht12/model/m1/qweights.npz | cut -d' ' -f1)
H_NEW=$(md5sum "$DST/model/m1/qweights.npz" | cut -d' ' -f1)
[ "$H_BANK" != "$H_NEW" ] || { echo "FATAL: m1 qweights 은행과 동일(지터 무효)"; exit 1; }
echo "m1 qweights byte-diff OK"

(cd "$DST" && zip -qr "$PWD/../submit_$TAG.zip" . -x '*__pycache__*')
python3 sim/check_zip.py "packages/submit_$TAG.zip" | tail -1
B=$(stat -c%s "packages/submit_$TAG.zip"); echo "zip bytes: $B"
[ "$B" -lt 1000000000 ] || { echo "FATAL: 1GB 초과"; exit 1; }

RUN=/tmp/ad_work/rt_$TAG
rm -rf "$RUN"; mkdir -p "$RUN/data"
(cd "$RUN" && unzip -q "/root/Action_Decision/packages/submit_$TAG.zip")
cp data/test.jsonl data/sample_submission.csv "$RUN/data/"
OUT=$(cd "$RUN" && CUDA_VISIBLE_DEVICES="" timeout 500 python3 script.py 2>&1 | grep -E "^wrote" || true)
echo "$OUT"
echo "$OUT" | grep -q "missing=0" || { echo "FATAL: 캐너리 missing!=0"; exit 1; }
python3 - << PY
import csv
CLASSES={'read_file','grep_search','list_directory','glob_pattern','edit_file','write_file','apply_patch','run_bash','run_tests','lint_or_typecheck','ask_user','plan_task','web_search','respond_only'}
rows=list(csv.DictReader(open("$RUN/output/submission.csv")))
assert len(rows)==5 and all(r['action'] in CLASSES for r in rows), "캐너리 action 클래스 위반"
print("캐너리 5행 전 유효클래스 OK:", [r['action'] for r in rows])
PY
echo "[OK] packages/submit_$TAG.zip 제출준비 완료 ($(stat -c%s packages/submit_$TAG.zip | awk '{printf "%.1fMB", $1/1e6}'))"
