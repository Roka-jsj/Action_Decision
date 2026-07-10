"""refit_lib — D-4 클린 앙상블 OOF 재적합 하네스 코어 (D12 하네스 정식화·5폴드 확장판).

레드팀 D12 하네스(fold0, LB 부호 3/3 적중)를 정식화한 라이브러리.
- 배포 캐스케이드 수식은 common/ad_lib.py predict_conditional_probs / predict 와 동일
  (단위테스트 sim/test_refit_d4.py 에서 monkeypatch 로 실제 ad_lib 함수와 대조 증명).
- 모든 그리드는 사전등록 상수(R49 서명: 확장 금지). 실행 시 grid_hash() 출력.
- GPU 불필요(전부 CPU/numpy).

npz 규약(teacher_cli.py 산출):
  oof (70000,14) float32 — load_train() 행 순서, 담당 fold의 va 행만 채움(나머지 0)
  fold_lo/fold_hi     — 이 파일이 커버하는 fold 반개구간 [lo, hi)
  scores              — fold별 best-epoch val macro-F1
"""
from __future__ import annotations
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.io_utils import CLASSES, NUM_CLASSES, CLASS_TO_IDX, session_id, load_labels  # noqa: E402
from common.metrics import macro_f1 as macro_f1_ref  # noqa: E402

N_TRAIN = 70000
N_FOLDS = 5

# ---------------------------------------------------------------------------
# 사전등록 상수 (R49/R49b 3자 서명 — 실행 중 확장 금지)
# ---------------------------------------------------------------------------
# 배포(은행 wd30) 좌표: m1(xlm-r-large v6) 전행 + [mdeb, klue] margin<th 조건부
ANCHOR_W = (0.55, 0.30, 0.15)     # 은행 wd30
ANCHOR_TH = 0.5
COND_MEMBERS = (1, 2)
C0_BASE_W = (0.60, 0.15, 0.25)    # C0 기준 좌표(프로브 델타 기준점)
C0_BASE_TH = 0.5

# w/th 그리드 — 레드팀 D12(redteam_sim2.py) 사전등록본 그대로
W_GRID = (
    (0.55, 0.30, 0.15), (0.50, 0.30, 0.20), (0.50, 0.35, 0.15),
    (0.55, 0.35, 0.10), (0.60, 0.30, 0.10), (0.50, 0.25, 0.25),
    (0.45, 0.35, 0.20), (0.55, 0.25, 0.20), (0.60, 0.25, 0.15),
    (0.45, 0.30, 0.25),
)
TH_GRID = (0.45, 0.5, 0.55, 0.6)
LAMBDA_GRID = (0.5, 0.7, 1.0)          # bias shrink: b = old + λ(fit-old)
TEMP_GRID = (0.8, 0.9, 1.0, 1.1, 1.25)  # per-member temperature(자유도 최소)

# tt30 (R51 아침 큐 4번 후보) — cw45 앵커 + deep-margin 2단 가중 (레드팀 fold0 +0.00091).
# 주의: grid_hash 페이로드 밖의 별도 후보 상수(기존 해시 불변). 발사는 3자 서명 후.
TT30_W = (0.45, 0.35, 0.20)   # top-level weights = cw45 (게이트·1단)
TT30_TH = 0.6                 # margin_th = cw45 (cond 추론 행 동일 → 시간 영향 0)
TT30_STAGES = ({"th": 0.6, "weights": (0.45, 0.35, 0.20)},
               {"th": 0.3, "weights": (0.40, 0.35, 0.25)})

# --- R54 T5-light: rescue 기준 재적합 (genrescue 배포 LB 0.78985 후 — 구입력 그리드 낡음) ---
# 주판정 = fold0-rescue OOF(m1 8ep f0ckpt 재추론), veto = folds1-4 구입력 + 이중 L-프록시(v8/v9).
# 그리드 = 레드팀 T5(redteam_t1.py) 사전등록본 그대로. grid_hash 페이로드 밖(별도 해시).
RESCUE_ANCHOR_W = (0.45, 0.40, 0.15)   # 배포 genrescue = th75 좌표 (신규 앵커)
RESCUE_ANCHOR_TH = 0.75
W_GRID_R54 = ((0.45, 0.40, 0.15), (0.45, 0.35, 0.20), (0.55, 0.30, 0.15),
              (0.50, 0.35, 0.15), (0.40, 0.40, 0.20))
TH_GRID_R54 = (0.5, 0.6, 0.75)
GATE_R54_PRIMARY = +0.0010    # 주판정: fold0-rescue ΔF1 최소선 (codex R54b 문안)
GATE_R54_OUTLIER = -0.0020    # veto: folds1-4 음수 outlier 제외선 (엔지니어 기본값, 3자 조정 가능)


def rescue_grid_hash() -> str:
    payload = json.dumps({
        "W_GRID_R54": W_GRID_R54, "TH_GRID_R54": TH_GRID_R54,
        "RESCUE_ANCHOR_W": RESCUE_ANCHOR_W, "RESCUE_ANCHOR_TH": RESCUE_ANCHOR_TH,
        "GATE_R54_PRIMARY": GATE_R54_PRIMARY, "GATE_R54_OUTLIER": GATE_R54_OUTLIER,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# 배포 수식 상수: ad_lib.predict() L847 scores = log(probs + 1e-9)
EPS_DEPLOY = 1e-9
# 레드팀 D12 재현용: redteam_sim*.py 는 log(P + 1e-12)
EPS_REDTEAM = 1e-12

# 무결성 앵커
SPLITS_SHA256 = "026ff8810b7da4c057d09bf4982d7bc54b96058e197112ca43e710d8194f90a3"
IDS_SHA16 = "251209af8b0d35f9"   # sha256("\n".join(train.jsonl ids))[:16]
OLD_BIAS_SHA16 = "b1164f379e2198ae"  # 구bias(postproc.json) json 해시
OLD_BIAS_PATH = os.path.join(ROOT, "packages", "submit_tri_cond_rebuild", "model", "postproc.json")

# 게이트 (사양 §4)
GATE_MIN_NONNEG = 3          # 5폴드 중 비음수 폴드 수
GATE_WORST_FOLD = -0.0007
GATE_JOINT_DELTA = +0.0004


def grid_hash() -> str:
    """사전등록 그리드의 canonical 해시 — 리포트에 박아 확장/변조 감지."""
    payload = json.dumps({
        "W_GRID": W_GRID, "TH_GRID": TH_GRID, "LAMBDA_GRID": LAMBDA_GRID,
        "TEMP_GRID": TEMP_GRID, "ANCHOR_W": ANCHOR_W, "ANCHOR_TH": ANCHOR_TH,
        "C0_BASE_W": C0_BASE_W, "C0_BASE_TH": C0_BASE_TH,
        "COND_MEMBERS": COND_MEMBERS,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 라벨 / 분할 로딩 (+ 사양 §1 assert)
# ---------------------------------------------------------------------------
def load_ids_labels():
    """train.jsonl 행 순서 기준 (ids, y, groups). 클래스 14개 전부 등장 assert."""
    ids = []
    with open(os.path.join(ROOT, "data", "train.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(json.loads(line)["id"])
    assert len(ids) == N_TRAIN, f"train.jsonl 행수 {len(ids)} != {N_TRAIN}"
    ids_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]
    assert ids_hash == IDS_SHA16, f"train.jsonl id 해시 불일치: {ids_hash} != {IDS_SHA16}"
    lab = load_labels()
    y = np.array([CLASS_TO_IDX[lab[i]] for i in ids], dtype=np.int64)
    present = set(np.unique(y).tolist())
    assert present == set(range(NUM_CLASSES)), f"클래스 누락: {sorted(set(range(NUM_CLASSES)) - present)}"
    groups = np.array([session_id(i) for i in ids])
    return ids, y, groups, ids_hash


def load_splits():
    """splits/splits.npz 캐시 로드 + 무결성 assert.

    주의: 현행 sklearn(1.8.0)의 StratifiedGroupKFold 는 캐시 생성 당시와 폴드 배정이
    다르다(실측: dev/holdout 은 재현, fold 는 불일치). 캐시 파일이 유일한 진실 —
    절대 재생성(make_splits force=True) 금지. sha256 으로 고정.
    """
    path = os.path.join(ROOT, "splits", "splits.npz")
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert h == SPLITS_SHA256, f"splits.npz 해시 불일치 — 재생성 금지 위반 의심: {h}"
    d = np.load(path, allow_pickle=True)
    n_splits = int(d["n_splits"])
    assert n_splits == N_FOLDS
    folds = [(np.asarray(d[f"tr{i}"]), np.asarray(d[f"va{i}"])) for i in range(n_splits)]
    dev, hold = np.asarray(d["dev_idx"]), np.asarray(d["holdout_idx"])
    # ② 각 train 행은 정확히 1개 폴드의 va 에만 등장
    va_all = np.concatenate([va for _, va in folds])
    assert len(va_all) == len(set(va_all.tolist())), "폴드 va 중복 — 배타성 위반"
    assert set(va_all.tolist()) == set(dev.tolist()), "va 합집합 != dev_idx"
    assert set(dev.tolist()).isdisjoint(set(hold.tolist())), "dev/holdout 교차"
    assert len(dev) + len(hold) == N_TRAIN
    return folds, dev, hold


def fold_of_rows(folds):
    """행 -> 폴드번호 (-1 = holdout)."""
    fmap = np.full(N_TRAIN, -1, dtype=np.int64)
    for fi, (_, va) in enumerate(folds):
        fmap[va] = fi
    return fmap


# ---------------------------------------------------------------------------
# 멤버 OOF 로더 (다중 npz 조인, 폴드 자동 확장)
# ---------------------------------------------------------------------------
class MemberOOF:
    """멤버 1슬롯 = npz 파일 목록. 폴드가 채워지면(fold_hi 증가/파일 추가) 자동 확장.

    assert(사양 §1): 파일별 커버 행 == splits 의 [fold_lo,fold_hi) va 합집합(행순서 정합),
    파일 간 배타(각 행 정확히 1회), 확률행 합=1, 멤버 F1 vs npz scores 정합(오배열 감지).
    """

    def __init__(self, name, paths, folds, y, proxy=False):
        self.name = name
        self.paths = [p for p in paths if os.path.exists(p)]
        assert self.paths, f"[{name}] npz 파일 없음: {paths}"
        self.proxy = proxy
        self.oof = np.zeros((N_TRAIN, NUM_CLASSES), dtype=np.float32)
        cov = np.zeros(N_TRAIN, dtype=np.int32)
        self.file_info = []
        for p in self.paths:
            d = np.load(p, allow_pickle=True)
            o = np.asarray(d["oof"], dtype=np.float32)
            assert o.shape == (N_TRAIN, NUM_CLASSES), f"[{name}] {p} oof shape {o.shape}"
            lo, hi = int(d["fold_lo"]), int(d["fold_hi"])
            rows = np.where(np.abs(o).sum(1) > 0)[0]
            expected = np.sort(np.concatenate([folds[i][1] for i in range(lo, hi)]))
            assert np.array_equal(rows, expected), (
                f"[{name}] {p} 커버 행({len(rows)}) != folds[{lo},{hi}) va({len(expected)}) — 행순서/폴드 정합 실패")
            s = o[rows].sum(1)
            assert np.abs(s - 1.0).max() < 5e-3, f"[{name}] {p} 확률행 합 이상 max|Δ|={np.abs(s-1).max()}"
            assert (cov[rows] == 0).all(), f"[{name}] {p} 폴드 중복 커버 — 배타성 위반"
            cov[rows] = 1
            self.oof[rows] = o[rows]
            # 오배열(행 셔플) 감지: 커버 행 argmax F1 vs npz 기록 scores 평균
            f1_here = fast_macro_f1(y[rows], o[rows].argmax(1))
            scores = np.asarray(d["scores"], dtype=float).ravel()
            drift = abs(f1_here - float(scores.mean()))
            assert drift < 0.03, (
                f"[{name}] {p} OOF F1 {f1_here:.4f} vs npz scores 평균 {scores.mean():.4f} — 행 오배열 의심")
            self.file_info.append({
                "path": os.path.relpath(p, ROOT), "folds": list(range(lo, hi)),
                "rows": int(len(rows)), "scores": [round(float(x), 5) for x in scores],
                "oof_f1": round(float(f1_here), 5),
                "model": str(d["model"]) if "model" in d.files else "?",
            })
        self.cov_mask = cov.astype(bool)
        self.folds_covered = sorted(
            fi for fi in range(N_FOLDS)
            if self.cov_mask[folds[fi][1]].all()
        )

    def summary(self):
        return {"name": self.name, "proxy": self.proxy,
                "folds": self.folds_covered, "files": self.file_info}


def usable_folds(members, folds):
    """모든 멤버가 완전 커버한 폴드 목록 — 재료가 채워지면 자동 확장되는 지점."""
    out = []
    for fi in range(N_FOLDS):
        if all(fi in m.folds_covered for m in members):
            out.append(fi)
    return out


# ---------------------------------------------------------------------------
# 캐스케이드 시뮬 — ad_lib.predict_conditional_probs L586-600 미러
# ---------------------------------------------------------------------------
def cascade_probs(mems, w, th, cond_members=COND_MEMBERS, stages=None):
    """mems: [(n,14) 확률배열,...] (동일 행 슬라이스). ad_lib 과 동일 연산 순서.

    full 멤버 가중혼합 -> top1-top2 마진 < th 행만 cond 멤버 추가 혼합.
    stages(R51 tt30, 선택적): [{"th":0.6,"weights":[...]},{"th":0.3,"weights":[...]}] —
    ad_lib 의 다단 조건부 가중과 동일 수식(th 내림차순 덮어쓰기, cond 추론 행은
    margin_th 게이트 1회 그대로 → 시간 영향 0).
    반환 (P, sel_mask).
    """
    cond_idx = set(int(i) for i in cond_members)
    full_idx = [i for i in range(len(mems)) if i not in cond_idx]
    assert full_idx, "full 멤버 없음"
    if stages:
        stages = sorted(({"th": float(s["th"]), "weights": [float(x) for x in s["weights"]]}
                         for s in stages), key=lambda s: -s["th"])
        for s in stages:
            assert len(s["weights"]) == len(mems), "stage weights 길이 != 멤버수"
            assert s["th"] <= float(th) + 1e-9, "stage th > margin_th"
    wf = sum(w[i] for i in full_idx)
    p_full = sum(w[i] * mems[i] for i in full_idx) / wf
    srt = np.sort(p_full, axis=1)
    sel = (srt[:, -1] - srt[:, -2]) < th
    out = p_full.copy()
    if sel.any():
        if stages:
            sel_i = np.where(sel)[0]
            marg_sel = (srt[:, -1] - srt[:, -2])[sel_i]
            cond_p = {mi: mems[mi][sel_i] for mi in sorted(cond_idx)}
            for st in stages:                  # 넓은 th → 좁은 th (깊은 단이 덮어씀)
                rr = np.where(marg_sel < st["th"])[0]
                if not len(rr):
                    continue
                sw = st["weights"]
                wf_s = sum(sw[i] for i in full_idx)
                acc = wf_s * p_full[sel_i[rr]]
                wt = wf_s
                for mi in sorted(cond_idx):
                    acc = acc + sw[mi] * cond_p[mi][rr]
                    wt += sw[mi]
                out[sel_i[rr]] = acc / wt
        else:
            acc = wf * p_full[sel]
            wt = wf
            for mi in sorted(cond_idx):
                acc = acc + w[mi] * mems[mi][sel]
                wt += w[mi]
            out[sel] = acc / wt
    return out, sel


def bias_argmax(P, bias, eps=EPS_DEPLOY):
    """ad_lib.predict() L847+L902 미러: pred = argmax(log(P+eps) + bias)."""
    return (np.log(P + eps) + np.asarray(bias)).argmax(1)


# ---------------------------------------------------------------------------
# macro-F1 — common.metrics.macro_f1 과 동치인 고속판(단위테스트로 증명)
# ---------------------------------------------------------------------------
def fast_macro_f1(y_true, y_pred, n_classes=NUM_CLASSES):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes)
    cm = cm.reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(f1.mean())


def score(P, bias, y_true, eps=EPS_DEPLOY):
    return fast_macro_f1(y_true, bias_argmax(P, bias, eps))


# ---------------------------------------------------------------------------
# 재적합 레이어 (사양 §3)
# ---------------------------------------------------------------------------
def fit_bias_cd(P, yt, init, step=0.05, rng=(-3.0, 3.0), passes=4, eps=EPS_DEPLOY):
    """per-class bias 좌표하강 — 레드팀 redteam_sim2.fit_bias 포트(재현성 유지)."""
    b = np.asarray(init, dtype=np.float64).copy()
    logp = np.log(P + eps)
    best = fast_macro_f1(yt, (logp + b).argmax(1))
    for _ in range(passes):
        improved = False
        for c in range(NUM_CLASSES):
            for d in (+step, -step):
                nb = b.copy()
                nb[c] = np.clip(nb[c] + d, *rng)
                f = fast_macro_f1(yt, (logp + nb).argmax(1))
                if f > best + 1e-9:
                    best = f
                    b = nb
                    improved = True
        if not improved:
            break
    return b, best


def shrink_bias(old, fitted, lam):
    """shrink: old + λ(fitted - old)."""
    old = np.asarray(old, dtype=np.float64)
    return old + lam * (np.asarray(fitted, dtype=np.float64) - old)


def apply_temperature(P, T):
    """확률 온도스케일 p^(1/T) 재정규화 (T=1 항등)."""
    if T == 1.0:
        return P
    q = np.power(np.clip(P, 1e-12, None), 1.0 / T)
    return q / q.sum(1, keepdims=True)


def fit_temps_greedy(mems_fit, y_fit, w, th, bias, cond_members=COND_MEMBERS,
                     temp_grid=TEMP_GRID, eps=EPS_DEPLOY):
    """per-member 온도 greedy 1패스(자유도 최소: 멤버당 TEMP_GRID 5지선다 1회)."""
    Ts = [1.0] * len(mems_fit)

    def eval_T(Ts_):
        mm = [apply_temperature(m, t) for m, t in zip(mems_fit, Ts_)]
        P, _ = cascade_probs(mm, w, th, cond_members)
        return score(P, bias, y_fit, eps)

    best = eval_T(Ts)
    for mi in range(len(mems_fit)):
        cur = Ts[mi]
        for T in temp_grid:
            if T == cur:
                continue
            trial = list(Ts)
            trial[mi] = T
            f = eval_T(trial)
            if f > best + 1e-9:
                best = f
                Ts = trial
    return Ts, best


# ---------------------------------------------------------------------------
# 그리드 탐색 (fit 인덱스에서 선택 -> eval 인덱스에서 정직 평가)
# ---------------------------------------------------------------------------
def search_wth(mems, y_rows, fit_i, bias, w_grid=W_GRID, th_grid=TH_GRID, eps=EPS_DEPLOY,
               cond_members=COND_MEMBERS):
    """(w,th) 그리드에서 fit셋 최고 좌표. mems 는 공통 행으로 슬라이스된 멤버 확률."""
    best_f, best = -1.0, None
    for w in w_grid:
        for th in th_grid:
            P, _ = cascade_probs(mems, w, th, cond_members)
            f = score(P[fit_i], bias, y_rows[fit_i], eps)
            if f > best_f:
                best_f, best = f, (w, th)
    return best, best_f


def search_joint(mems, y_rows, fit_i, old_bias, w_grid=W_GRID, th_grid=TH_GRID,
                 bias_passes=2, eps=EPS_DEPLOY, cond_members=COND_MEMBERS):
    """(w,th)×bias 조인트: 각 좌표에서 bias 좌표하강 후 fit셋 최고 선택."""
    best_f, best = -1.0, None
    for w in w_grid:
        for th in th_grid:
            P, _ = cascade_probs(mems, w, th, cond_members)
            b, f = fit_bias_cd(P[fit_i], y_rows[fit_i], old_bias, passes=bias_passes, eps=eps)
            if f > best_f:
                best_f, best = f, (w, th, b)
    return best, best_f
