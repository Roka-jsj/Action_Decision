#!/bin/bash
# m1 스왑 고속 조립: 신은행(submit_sf_mht12) 기반, m1만 교체. (점검판: 3결함 수리)
# 사용: bash sim/assemble_m1swap_fast.sh <member_zip> <out_tag>
set -euo pipefail
cd /root/Action_Decision
MZIP="${1:?member zip}"; TAG="${2:?out tag}"
# SRC_PKG로 스캐폴드 지정 가능(기본 원은행). 07-14 codex 합의: 신티켓은 qj3 스캐폴드(전역최대의 m2/m3 유지)
SRC=${SRC_PKG:-packages/submit_sf_mht12}; DST=packages/submit_$TAG
Q=work/$(basename "$MZIP" .zip)_q8

# [수리2] TAG 가드: 은행/원본 자기파괴 방지
[ "$DST" != "$SRC" ] || { echo "FATAL: DST==SRC(은행) 금지"; exit 1; }
[ -f "$MZIP" ] || { echo "FATAL: $MZIP 없음"; exit 1; }

# [수리1] q8 캐시 신선도: zip보다 오래되면 재양자화
if [ -f "$Q" ] && [ "$Q" -nt "$MZIP" ]; then
  echo "[cache] q8 최신 — 재사용"
else
  rm -f "$Q"; python3 sim/quantize_member.py "$MZIP" --out "$Q"
fi

rm -rf "$DST" "packages/submit_$TAG.zip"
cp -al "$SRC" "$DST"
rm -rf "$DST/model/m1" "$DST/model/__pycache__"; mkdir -p "$DST/model/m1"
(cd "$DST/model/m1" && unzip -q "/root/Action_Decision/$Q")

python3 - << PY
import json
m=json.load(open("$DST/model/run_meta.json"))
assert m['ensemble'][0].get('mht')==12, "m1 mht12 서빙 누락!"
print("mht12 서빙 OK:", m['ensemble'][0])
PY

(cd "$DST" && zip -qr "$PWD/../submit_$TAG.zip" . -x '*__pycache__*')
python3 sim/check_zip.py "packages/submit_$TAG.zip" | tail -1

# [수리3] 캐너리 강화: missing=0 필수 + action 클래스 유효성
RUN=/tmp/claude-0/-root-Action-Decision/0b8c94fb-8cfc-4609-aaa0-00660d405e6c/scratchpad/rt_$TAG
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
echo "[OK] packages/submit_$TAG.zip 제출준비 완료 ($(du -h packages/submit_$TAG.zip|cut -f1))"
