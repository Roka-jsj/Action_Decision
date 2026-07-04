"""holdout 채점형 합성테스트 — T4 벤치가 타이밍+품질 게이트를 겸하게.

test.jsonl = 프로즌 홀드아웃 전량(id에 ho:: 접두) + train 랜덤 필러(synth:: 접두)로 총 N건.
FULL 멤버는 70k 전체로 학습돼 절대값은 누수-팽창이지만, 원본 vs q8 / 단독 vs 앙상블
**델타 비교**엔 유효(같은 행, 같은 팽창).
사용: python sim/make_holdout_test.py <N> <out_dir>  → 이후 score_holdout.py로 채점.
"""
from __future__ import annotations
import sys, os, json, random
import numpy as np

def main(n=30000, out="/tmp/ad_hold/data"):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    from common.io_utils import load_train
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
        for i, tag in rows:
            o = json.loads(lines[i])
            o["id"] = f"{tag}::{o['id']}"
            out_ids.append(o["id"])
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "sample_submission.csv"), "w", encoding="utf-8") as f:
        f.write("id,action\n")
        for i in out_ids:
            f.write(f"{i},read_file\n")
    print(f"wrote {len(out_ids)} rows ({len(ho)} holdout + {len(filler)} filler) to {out}")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30000,
         sys.argv[2] if len(sys.argv) > 2 else "/tmp/ad_hold/data")
