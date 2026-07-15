#!/bin/bash
# m3 슬롯 스왑 조립: 신은행(submit_sf_mht12) 기반, m3만 교체 (시드다양성 셀).
# 사용: bash sim/assemble_m3swap.sh <member_zip> <out_tag>
# 주의: 교체 멤버가 MHT=12 학습본이면 run_meta ensemble[2]에 mht:12를 심는다.
set -euo pipefail
cd /root/Action_Decision
MZIP="${1:?member zip}"; TAG="${2:?out tag}"
SRC=packages/submit_sf_mht12; DST=packages/submit_$TAG
Q=work/$(basename "$MZIP" .zip)_q8

[ "$DST" != "$SRC" ] || { echo "FATAL: DST==SRC(은행) 금지"; exit 1; }
[ -f "$MZIP" ] || { echo "FATAL: $MZIP 없음"; exit 1; }

if [ -f "$Q" ] && [ "$Q" -nt "$MZIP" ]; then
  echo "[cache] q8 최신 — 재사용"
else
  rm -f "$Q"; python3 sim/quantize_member.py "$MZIP" --out "$Q"
fi

rm -rf "$DST" "packages/submit_$TAG.zip"
cp -al "$SRC" "$DST"
rm -rf "$DST/model/m3" "$DST/model/__pycache__"; mkdir -p "$DST/model/m3"
(cd "$DST/model/m3" && unzip -q "/root/Action_Decision/$Q")

# run_meta: m3 슬롯에 mht=12 (교체 멤버는 MHT=12 학습본) + maxlen 320 명시
# 메타파일은 은행에서 chmod a-w 하드링크 → 새 파일로 대체 기록
python3 - << PY
import json, os
p = "$DST/model/run_meta.json"
m = json.load(open(p))
assert m['ensemble'][0].get('mht') == 12, "m1 mht12 서빙 누락!"
m['ensemble'][2] = {"dir": "m3", "version": "v6", "mht": 12}
os.remove(p)
with open(p, "w") as f: json.dump(m, f, indent=1)
print("m3 슬롯 갱신 OK:", m['ensemble'][2], "| weights:", m['weights'])
PY

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
echo "[OK] packages/submit_$TAG.zip 제출준비 완료 ($(du -h packages/submit_$TAG.zip|cut -f1))"
