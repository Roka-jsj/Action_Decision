"""증류용 teacher 소프트라벨 생성 (Colab GPU에서 실행).

2-large 앙상블(holdout +0.0028, T4캡으로 배포불가)을 soft label로 압축:
blend = 0.65*P(largev6-8ep) + 0.35*P(largev4-8ep), train 70k 전행.
출력: /content/soft_labels.npz (probs fp16 [70000,14], ids)
전제: /content/m_v6/, /content/m_v4/ 에 멤버 압축해제, ad_common.zip 해제, open.zip 해제.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
sys.path.insert(0, "/content")
from common.io_utils import load_train
from common import ad_lib

samples, y, ids = load_train()
p6 = ad_lib.predict_logits("/content/m_v6", samples, version="v6", max_len=320,
                           batch_size=256, return_probs=True)
print("v6 done", flush=True)
p4 = ad_lib.predict_logits("/content/m_v4", samples, version="v4", max_len=320,
                           batch_size=256, return_probs=True)
print("v4 done", flush=True)
blend = (0.65 * p6 + 0.35 * p4).astype(np.float16)
np.savez_compressed("/content/soft_labels.npz", probs=blend, ids=np.array(ids))
acc = float((blend.argmax(1) == np.array(y)).mean())
print(f"[soft] saved 70k blend, train-acc(참고)={acc:.4f}", flush=True)
open("/content/DONE_soft", "w").write("ok")
