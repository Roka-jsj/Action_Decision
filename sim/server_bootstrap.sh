#!/bin/bash
# server_bootstrap.sh <bundle_dir> — Kaggle 데이터셋 번들을 저장소 배치로 변환 (학습 컨테이너 안에서 1회).
# 주의: Kaggle은 업로드된 zip을 자동 해제해 폴더로 저장 → member 폴더는 다시 zip으로 복원한다.
set -e
B=$1; [ -d "$B" ] || { echo "usage: $0 <bundle_dir>"; exit 1; }
R=$(cd "$(dirname "$0")/.." && pwd)
E=$R/action_decision_maximum/experiments
mkdir -p "$E" "$R/data"

# 데이터
if [ -d "$B/open/data" ]; then cp -r "$B/open/data/." "$R/data/"; fi
ls "$R/data/train.jsonl" >/dev/null

# splits/common은 git 저장소 것 사용(ad_common보다 최신). splits만 없으면 번들에서.
[ -f "$R/splits/splits.npz" ] || cp -r "$B/ad_common/splits/." "$R/splits/"

# teacher npz + soft labels (평문 파일)
find "$B" -maxdepth 1 -name "teacher_*.npz" -exec cp {} "$E/" \;
find "$B" -maxdepth 1 -name "soft_labels_*.npz" -exec cp {} "$E/" \;

# member 폴더 → zip 복원
for d in "$B"/member_*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  if [ ! -f "$E/$name.zip" ]; then
    (cd "$d" && zip -qr "$E/$name.zip" .)
    echo "restored $name.zip"
  fi
done
# 이미 zip으로 온 경우
find "$B" -maxdepth 1 -name "member_*.zip" -exec cp -n {} "$E/" \;

pip install -q "transformers==4.46.3" "scikit-learn" "sentencepiece" 2>/dev/null || true
python3 -c "import sys; sys.path.insert(0,'$R'); from common.io_utils import load_train; s,y,i=load_train(); print('bootstrap OK:', len(s), 'samples')"
ls -la "$E" | grep -E "member|teacher|soft" | head -20