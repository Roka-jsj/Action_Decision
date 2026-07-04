"""make_holdout_test 산출 submission.csv 채점 — ho:: 행만 라벨 조인 macro-F1.

사용: python sim/score_holdout.py <submission.csv>
주의: FULL 멤버는 홀드아웃도 학습에 봤으므로 절대값은 팽창(~0.80+).
      용도는 원본 vs q8 / 단독 vs 앙상블 **델타** 및 파이프라인 정합성 확인.
"""
from __future__ import annotations
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import f1_score

sub = sys.argv[1]
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lab = {r["id"]: r["action"] for r in csv.DictReader(open(os.path.join(root, "data", "train_labels.csv"), encoding="utf-8"))}
yt, yp = [], []
for r in csv.DictReader(open(sub, encoding="utf-8")):
    tag, _, oid = r["id"].partition("::")
    if tag == "ho" and oid in lab:
        yt.append(lab[oid]); yp.append(r["action"])
print(f"[score-holdout] n={len(yt)}  macro-F1={f1_score(yt, yp, average='macro'):.5f}")
