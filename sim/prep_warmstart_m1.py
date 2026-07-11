#!/usr/bin/env python3
"""m1wsT3 준비 — 배포 m1(q8 qweights)을 fp16 학습가능 디렉터리로 복원. (R71b)

배포 q8이 곧 서빙되는 함수이므로 이 복원본 = 배포와 정확히 동일한 init.
ad_lib의 양자화 복원 경로(_load_model_maybe_quant)를 그대로 사용.
출력: work/warmstart_m1/ (model.safetensors fp16 + config/id_map/tokenizer 일체)
검증: 복원 모델의 5k 앞 8행 argmax == 배포 p_old의 argmax (배포등가 스모크).
"""
import os, sys, shutil
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common import ad_lib                    # noqa: E402
from common.io_utils import load_train       # noqa: E402

SRC = os.path.join(ROOT, "packages", "submit_th85", "model", "m1")   # 배포 m1 (q8)
DST = os.path.join(ROOT, "work", "warmstart_m1")

model = ad_lib._load_model_maybe_quant(SRC)          # q8 → fp16 복원
os.makedirs(DST, exist_ok=True)
model.save_pretrained(DST, safe_serialization=True)
for f in os.listdir(SRC):
    if f != "qweights.npz" and not os.path.exists(os.path.join(DST, f)):
        shutil.copy(os.path.join(SRC, f), os.path.join(DST, f))
assert os.path.exists(os.path.join(DST, "model.safetensors"))
assert os.path.exists(os.path.join(DST, "id_map.npy"))
print(f"[prep] 복원 저장: {DST} ({sorted(os.listdir(DST))})")

# 배포등가 스모크: 8행 argmax 대조
d = np.load(os.path.join(ROOT, "work/autopsy_m1t3_5k.npz"))
rows, p_old = d["rows"][:8], d["p_old"][:8]
samples, _, _ = load_train()
sub = [samples[i] for i in rows]
tx8 = [ad_lib.serialize(s, "v6", 8) for s in sub]
p = ad_lib.predict_logits(DST, sub, version="v6", max_len=320, batch_size=8,
                          texts=tx8, return_probs=True, gen_rescue=True)
agree = int((p.argmax(1) == p_old.argmax(1)).sum())
print(f"[prep] 배포등가 스모크 {agree}/8 (요구 8)")
assert agree == 8, "복원본이 배포와 불일치"
print("[prep] PASS — warm-start init 준비 완료")
