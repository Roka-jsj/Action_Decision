"""서버 추론 스크립트 (오프라인). 최상단에서 오프라인 플래그 설정.

구조: 이 파일 옆의 model/ 에 가중치·토크나이저·ad_lib.py·postproc.json·run_meta.json.
data/test.jsonl 로드 → ad_lib.predict → data/sample_submission.csv 순서로 id 조인
→ output/submission.csv.
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import csv
import json

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model")
sys.path.insert(0, MODEL)
import ad_lib  # model/ad_lib.py (학습/추론 공용 단일 소스)


def load_jsonl(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    os.makedirs("output", exist_ok=True)
    samples = load_jsonl(os.path.join("data", "test.jsonl"))
    meta = json.load(open(os.path.join(MODEL, "run_meta.json")))
    ppath = os.path.join(MODEL, "postproc.json")
    preds = ad_lib.predict(
        MODEL, samples,
        version=meta["version"], max_len=meta["max_len"],
        batch_size=meta.get("batch_size", 128),
        postproc_path=ppath if os.path.exists(ppath) else None,
    )
    rows = list(csv.DictReader(open(os.path.join("data", "sample_submission.csv"), encoding="utf-8")))
    miss = 0
    for r in rows:
        if r["id"] in preds:
            r["action"] = preds[r["id"]]
        else:
            miss += 1
    with open(os.path.join("output", "submission.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "action"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows (missing={miss})")


if __name__ == "__main__":
    main()
