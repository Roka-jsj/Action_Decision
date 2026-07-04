"""로컬 검증용 초소형 모델 생성 (실제 xlm-roberta 토크나이저 + 랜덤 소형 가중치).

목적: 서버 추론 경로 전체(오프라인 로드→직렬화→predict→id조인→스키마)를
로컬 CPU에서 검증. 정확도 아닌 '기계적 동작' 확인용.
사용: python sim/make_tiny_model.py <out_model_dir>
"""
from __future__ import annotations
import sys, os, json, shutil

def main(out="/tmp/ad_sim/model"):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    import torch
    from transformers import AutoTokenizer, XLMRobertaConfig, AutoModelForSequenceClassification
    from common.ad_lib import CLASSES, NUM_CLASSES

    os.makedirs(out, exist_ok=True)
    print("토크나이저 다운로드(로컬 인터넷 OK)...")
    tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
    cfg = XLMRobertaConfig(
        vocab_size=tok.vocab_size, hidden_size=128, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=256, max_position_embeddings=258,
        num_labels=NUM_CLASSES,
        id2label={i: c for i, c in enumerate(CLASSES)},
        label2id={c: i for i, c in enumerate(CLASSES)})
    model = AutoModelForSequenceClassification.from_config(cfg)
    model.half()
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    shutil.copy(os.path.join(root, "common", "ad_lib.py"), os.path.join(out, "ad_lib.py"))
    # postproc(zeros) + run_meta
    json.dump({"bias": [0.0] * NUM_CLASSES, "classes": CLASSES},
              open(os.path.join(out, "postproc.json"), "w"))
    json.dump({"version": "v3", "max_len": 192, "batch_size": 128},
              open(os.path.join(out, "run_meta.json"), "w"))
    files = sorted(os.listdir(out))
    mb = sum(os.path.getsize(os.path.join(out, f)) for f in files) / 1e6
    print(f"tiny model → {out}  files={files}  size={mb:.1f}MB")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ad_sim/model")
