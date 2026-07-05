#!/bin/bash
# prep_verify.sh <submit.zip> — 학습 컨테이너 안에서 실행. /share에 자가완결 검증셋을 조립한다.
# 이후 호스트 터미널에서:  docker exec -it <mun-jtest|mun-test> bash /share/verify/<이름>/run.sh
# 검증 컨테이너는 오프라인·의존성 없음 전제 → 채점기는 sklearn 없이 순수 파이썬으로 생성.
set -e
ZIP=$(realpath "$1")
R=$(cd "$(dirname "$0")/.." && pwd)
NAME=$(basename "$ZIP" .zip)
W=/share/verify/$NAME
rm -rf "$W"; mkdir -p "$W"

# 1) 패키지 + holdout 30k 테스트셋 (train 컨테이너에서 생성 — data/, splits/ 필요)
unzip -q "$ZIP" -d "$W/pkg"
python3 "$R/sim/make_holdout_test.py" 30000 "$W/pkg/data"
cp "$R/data/train_labels.csv" "$W/labels.csv"
[ -f "$R/sim/calib.json" ] && cp "$R/sim/calib.json" "$W/calib.json"

# 2) 순수 파이썬 채점기 (오프라인 검증 컨테이너용 — sklearn 불필요)
cat > "$W/score.py" <<'PY'
import csv, sys, os
CLASSES = ["read_file","grep_search","list_directory","glob_pattern","edit_file","write_file",
           "apply_patch","run_bash","run_tests","lint_or_typecheck","ask_user","plan_task",
           "web_search","respond_only"]
d = os.path.dirname(os.path.abspath(__file__))
lab = {r["id"]: r["action"] for r in csv.DictReader(open(os.path.join(d, "labels.csv"), encoding="utf-8"))}
tp = {c: 0 for c in CLASSES}; fp = dict(tp); fn = dict(tp); n = 0
for r in csv.DictReader(open(os.path.join(d, "pkg", "output", "submission.csv"), encoding="utf-8")):
    tag, _, oid = r["id"].partition("::")
    if tag != "ho" or oid not in lab: continue
    t, p = lab[oid], r["action"]; n += 1
    if t == p: tp[t] += 1
    else: fp[p] = fp.get(p, 0) + 1; fn[t] += 1
f1s = []
for c in CLASSES:
    pr = tp[c] / (tp[c] + fp.get(c, 0)) if tp[c] + fp.get(c, 0) else 0.0
    rc = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
    f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
print(f"[score] holdout n={n}  macro-F1={sum(f1s)/len(f1s):.5f}")
PY

# 3) 검증 실행기 (검증 컨테이너 안에서 도는 자가완결 스크립트)
cat > "$W/run.sh" <<'SH'
#!/bin/bash
set -e
W=$(cd "$(dirname "$0")" && pwd)
PEAK_F=$W/peak; echo 0 > "$PEAK_F"
( while true; do
    V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    P=$(cat "$PEAK_F"); [ -n "$V" ] && [ "$V" -gt "$P" ] 2>/dev/null && echo "$V" > "$PEAK_F"
    sleep 2
  done ) & POLL=$!
cd "$W/pkg"
T0=$(date +%s)
python script.py
T1=$(date +%s); kill $POLL 2>/dev/null || true
E=$((T1 - T0)); PEAK=$(cat "$PEAK_F")
ROWS=$(tail -n +2 output/submission.csv | wc -l)
python "$W/score.py"
echo "----------------------------------------"
echo "A6000 실측: ${E}s | 피크 VRAM: ${PEAK}MB | 행수: $ROWS (30000 기대)"
if [ -f "$W/calib.json" ]; then
  RATIO=$(python - <<P
import json; print(json.load(open("$W/calib.json"))["ratio"])
P
)
  EST=$(python - <<P
print(round($E * $RATIO))
P
)
  echo "환산 서버시간: ~${EST}s → $([ "$EST" -le 540 ] && echo '시간 게이트 PASS' || echo '시간 게이트 FAIL(>540s)')"
else
  echo "⚠ calib.json 없음 — 앵커(largeonly 257s / tri_cond 427s) 측정 후 sim/calib.json 생성 필요"
fi
[ "$PEAK" -le 14000 ] && echo "VRAM 게이트: PASS (≤14GB)" || echo "VRAM 게이트: FAIL (T4 16GB 위험)"
SH
chmod +x "$W/run.sh"
echo "[prep] 완료. 호스트 터미널에서 실행:"
echo "  docker exec -it mun-jtest bash /share/verify/$NAME/run.sh"