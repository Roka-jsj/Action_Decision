"""holdout 채점형 합성테스트 — T4 벤치가 타이밍+품질 게이트를 겸하게.

test.jsonl = 프로즌 홀드아웃 전량 + train 랜덤 필러로 총 N건. **id는 원본 그대로**
(sess_sim_/sess_au_ prefix 보존 — 듀얼 bias 등 id-의존 추론경로가 실전과 동일하게 동작, R14).
채점은 holdout 전용 labels(3번째 인자)에 id가 있는 행만 — 필러는 labels에 없어 자동 제외.
FULL 멤버는 70k 전체로 학습돼 절대값은 누수-팽창이지만, 원본 vs q8 / 단독 vs 앙상블
**델타 비교**엔 유효(같은 행, 같은 팽창).
사용: python sim/make_holdout_test.py <N> <out_dir> [holdout_labels_out.csv]
"""
from __future__ import annotations
import sys, os, json, random, csv
import numpy as np

def main(n=30000, out="/tmp/ad_hold/data", labels_out=""):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    from common.io_utils import load_train, CLASSES
    from common.cv import make_splits
    os.makedirs(out, exist_ok=True)
    lines = [l for l in open(os.path.join(root, "data", "train.jsonl"), encoding="utf-8") if l.strip()]
    samples, y, ids = load_train()
    groups = np.array([s["session"] for s in samples])
    sp = make_splits(ids, np.array(y), groups)
    ho = set(int(i) for i in sp["holdout_idx"])
    rng = random.Random(0)
    filler = rng.sample([i for i in range(len(lines)) if i not in ho], max(0, n - len(ho)))
    rows = [(i, "ho") for i in sorted(ho)] + [(i, "synth") for i in filler]
    rng.shuffle(rows)
    out_ids = []
    with open(os.path.join(out, "test.jsonl"), "w", encoding="utf-8") as f:
        for i, _tag in rows:
            o = json.loads(lines[i])
            out_ids.append(o["id"])
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "sample_submission.csv"), "w", encoding="utf-8") as f:
        f.write("id,action\n")
        for i in out_ids:
            f.write(f"{i},read_file\n")
    if labels_out:
        with open(labels_out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["id", "action"])
            for i in sorted(ho):
                w.writerow([ids[i], CLASSES[y[i]]])
    print(f"wrote {len(out_ids)} rows ({len(ho)} holdout + {len(filler)} filler) to {out}"
          + (f" / holdout labels -> {labels_out}" if labels_out else ""))

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30000,
         sys.argv[2] if len(sys.argv) > 2 else "/tmp/ad_hold/data",
         sys.argv[3] if len(sys.argv) > 3 else "")
