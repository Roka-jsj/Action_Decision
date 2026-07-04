"""30k 합성 테스트 생성 — 오프라인 시뮬(타이밍/메모리)용.

실제 test.jsonl(5건)은 너무 작아 타이밍 측정 불가. train 샘플에서 N건을
뽑아 id만 test 형식으로 바꿔 test.jsonl 규모를 재현(라벨 없음, 구조 동일).
사용: python sim/make_synth_test.py <N> <out_dir>
"""
from __future__ import annotations
import sys, os, json, random

def main(n=30000, out="/tmp/ad_sim/data"):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(out, exist_ok=True)
    lines = [l for l in open(os.path.join(root, "data", "train.jsonl"), encoding="utf-8") if l.strip()]
    random.seed(0)
    pick = random.sample(lines, min(n, len(lines)))
    if n > len(lines):  # 필요시 복제
        pick = (pick * (n // len(lines) + 1))[:n]
    ids = []
    with open(os.path.join(out, "test.jsonl"), "w", encoding="utf-8") as f:
        for i, l in enumerate(pick):
            o = json.loads(l)
            o["id"] = f"synth-step_{i:06d}"   # 고유 id
            o.pop("label", None); o.pop("y", None)
            ids.append(o["id"])
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "sample_submission.csv"), "w", encoding="utf-8") as f:
        f.write("id,action\n")
        for i in ids:
            f.write(f"{i},read_file\n")
    print(f"wrote {len(ids)} synth test rows to {out}")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ad_sim/data"
    main(n, out)
