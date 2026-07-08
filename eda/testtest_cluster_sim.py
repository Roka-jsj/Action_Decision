#!/usr/bin/env python
"""R30 후보: test-test 클러스터 일관성 (codex R29 Q4 1순위, +0.005~0.015 유일 대형후보).

아이디어: 히든 30k는 train과 OOD여도 **내부적으로는** 같은 생성기의 template
near-dup 클러스터를 가질 수 있다(0.80팀 가설). zip 스크립트는 serve-time에
30k 전체를 보므로 고신뢰 클러스터 안에서 확률을 섞어 일관성을 강제할 수 있다
— train 의존이 없어 OOD 반전을 원천 회피.

오프라인 설계: 홀드아웃 5810행을 유사-테스트로. cross-session만 클러스터(히든=
세션 비공유). 측정: ①클러스터 커버리지 ②라벨 순도 ③확률 스무딩의 macro-F1 이득
그리드(sim_th × λ × 게이트).
"""
from __future__ import annotations
import sys, glob
import numpy as np

sys.path.insert(0, "/root/Action_Decision")
from common.io_utils import load_train, NUM_CLASSES
from common.cv import make_splits
from common.metrics import macro_f1
from common.postproc import fit_bias

samples, y, ids = load_train(); y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups)
hold_idx = np.asarray(sp["holdout_idx"]); y_hold = y[hold_idx]
sess_hold = groups[hold_idx]

P_hold, P_oof, cov = None, np.zeros((len(y), NUM_CLASSES)), np.zeros(len(y), bool)
for f in sorted(glob.glob("/root/Action_Decision/action_decision_maximum/experiments/teacher_largev6[AB]_a*.npz")):
    z = np.load(f, allow_pickle=True)
    h = z["hold"].astype(np.float64); P_hold = h if P_hold is None else P_hold + h
    o = z["oof"].astype(np.float64); m = o.sum(1) > 0; P_oof[m] = o[m]; cov |= m
P_hold /= 5; P_hold /= P_hold.sum(1, keepdims=True)
oi = np.where(cov)[0]; Po = P_oof[oi]; Po /= Po.sum(1, keepdims=True)
bias, _ = fit_bias(np.log(Po + 1e-12), y[oi])

emb = np.load("/root/Action_Decision/work/retrieval_pack/train_emb.npy", mmap_mode="r")
E = np.asarray(emb[hold_idx], dtype=np.float32)          # 중심화·정규화 완료본
S = E @ E.T                                              # 5810² 코사인
np.fill_diagonal(S, -1)
same_sess = sess_hold[:, None] == sess_hold[None, :]
S[same_sess] = -1                                        # 히든=세션 비공유 → cross-session만

base_pred = (np.log(P_hold + 1e-12) + bias).argmax(1)
base_f1 = macro_f1(y_hold, base_pred)[0]
print(f"[base] holdout bias macro-F1 = {base_f1:.4f}")

# ① 구조 실측: threshold별 커버리지/이웃 라벨 일치율
print(f"\n{'sim_th':>7} {'row가짐%':>9} {'평균이웃수':>9} {'이웃라벨일치%':>12} {'pred일치%':>9}")
for th in (0.90, 0.93, 0.95, 0.97, 0.99):
    nb = S >= th
    has = nb.any(1)
    n_nb = nb.sum(1)[has].mean() if has.any() else 0
    # 라벨 일치: 이웃 중 자기 라벨과 같은 비율
    agree_lab, agree_pred = [], []
    rows = np.where(has)[0]
    for i in rows[:4000]:
        js = np.where(nb[i])[0]
        agree_lab.append((y_hold[js] == y_hold[i]).mean())
        agree_pred.append((base_pred[js] == base_pred[i]).mean())
    print(f"{th:7.2f} {has.mean()*100:8.1f}% {n_nb:9.1f} {np.mean(agree_lab)*100:11.1f}% {np.mean(agree_pred)*100:8.1f}%")

# ② 스무딩 이득 그리드: 이웃 확률평균 blend (고신뢰 게이트)
print(f"\n{'sim_th':>7} {'lam':>5} {'gate':>16} {'coverage%':>9} {'changed%':>8} {'ΔF1':>8}")
for th in (0.95, 0.97):
    nb = S >= th
    for lam in (0.3, 0.5):
        for gate_name in ("all", "conf_nb"):
            Pn = P_hold.copy()
            rows = np.where(nb.any(1))[0]
            touched = 0
            for i in rows:
                js = np.where(nb[i])[0]
                pbar = P_hold[js].mean(0)
                if gate_name == "conf_nb" and pbar.max() < 0.7:
                    continue   # 이웃 평균이 고신뢰일 때만
                Pn[i] = (1 - lam) * P_hold[i] + lam * pbar
                touched += 1
            pred = (np.log(Pn + 1e-12) + bias).argmax(1)
            f1 = macro_f1(y_hold, pred)[0]
            ch = (pred != base_pred).mean() * 100
            print(f"{th:7.2f} {lam:5.1f} {gate_name:>16} {touched/len(y_hold)*100:8.1f}% {ch:7.2f}% {f1-base_f1:+.4f}")
