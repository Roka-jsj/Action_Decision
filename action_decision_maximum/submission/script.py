"""서버 추론 스크립트 (오프라인) — 트랙 B(정확도 최대) 단일 student.

트랙 A와 동일 구조. model/ 의 run_meta.json(version,max_len) 과 postproc.json 사용.
증류된 단일 student(xlm-roberta-base)라 추론 속도·용량은 단일 모델 수준.
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
import ad_lib


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
