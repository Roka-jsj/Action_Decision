#!/usr/bin/env python3
"""test_refit_d4 — D-4 재적합 하네스 회귀·수식동일성 테스트 (pytest 호환, 단독 실행 가능).

핵심 증명:
  1) cascade_probs == ad_lib.predict_conditional_probs (배포 코드 monkeypatch 실측 대조)
  2) bias 적용식 == ad_lib.predict() L847+L902 (log(P+1e-9)+bias argmax)
  3) fast_macro_f1 == common.metrics.macro_f1
  4) 로더 assert(행순서·폴드 배타성·클래스 14종·행수 정합) — 실제 npz
  5) known-probe 부호/크기 재현 (wk30 −/th55 +/wd30 +/cw45 +0.002대)
  6) 레드팀 D12 실측 재현 (wth +0.00216 / joint +0.00305, seed7)  [slow ~4분]

실행: python3 sim/test_refit_d4.py            # 전체 (slow 포함)
      python3 -m pytest sim/test_refit_d4.py -x -q
      AD_SKIP_SLOW=1 python3 sim/test_refit_d4.py   # slow 제외
"""
from __future__ import annotations
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import refit_lib as L  # noqa: E402
from sim import refit_d4 as D  # noqa: E402
from common import ad_lib  # noqa: E402
from common.metrics import macro_f1 as macro_f1_ref  # noqa: E402
from common import postproc  # noqa: E402

SKIP_SLOW = os.environ.get("AD_SKIP_SLOW", "0") == "1"

_CACHE = {}


def _materials():
    if "mat" not in _CACHE:
        ids, y, groups, ids_hash = L.load_ids_labels()
        folds, dev, hold = L.load_splits()
        fmap = L.fold_of_rows(folds)

        class A:  # refit_d4.load_all 기본 재료
            m1 = mdeb = klue = ""
        _, y2, folds2, fmap2, members, old_bias = D.load_all(A())
        assert np.array_equal(y, y2)
        _CACHE["mat"] = (y, folds, fmap, members, old_bias)
    return _CACHE["mat"]


# ---------------------------------------------------------------------------
def test_fast_macro_f1_equals_common_metrics():
    rng = np.random.default_rng(0)
    for n in (50, 1000, 20000):
        yt = rng.integers(0, 14, n)
        yp = rng.integers(0, 14, n)
        ref, _ = macro_f1_ref(yt, yp)
        assert abs(L.fast_macro_f1(yt, yp) - ref) < 1e-12
    # 클래스 미등장 케이스
    yt = np.zeros(10, dtype=int)
    yp = np.zeros(10, dtype=int)
    ref, _ = macro_f1_ref(yt, yp)
    assert abs(L.fast_macro_f1(yt, yp) - ref) < 1e-12


def test_cascade_equals_ad_lib_conditional():
    """배포 함수 ad_lib.predict_conditional_probs 를 monkeypatch(가짜 predict_logits)로
    구동해 cascade_probs 와 배열 단위 동일성 증명."""
    rng = np.random.default_rng(42)
    n, C, M = 800, 14, 3
    probs = []
    for _ in range(M):
        z = rng.normal(size=(n, C)).astype(np.float32) * 2
        e = np.exp(z - z.max(1, keepdims=True))
        probs.append((e / e.sum(1, keepdims=True)).astype(np.float32))
    samples = [{"id": f"r{i}"} for i in range(n)]
    id2row = {f"r{i}": i for i in range(n)}

    def fake_predict_logits(model_dir, subset, *a, **kw):
        mi = int(os.path.basename(model_dir).replace("m", ""))
        rows = [id2row[s["id"]] for s in subset]
        assert kw.get("return_probs")
        return probs[mi][rows]

    orig_pl, orig_ser = ad_lib.predict_logits, ad_lib.serialize
    ad_lib.predict_logits = fake_predict_logits
    ad_lib.serialize = lambda s, ver: s["id"]
    try:
        cases = [
            ((0.55, 0.30, 0.15), 0.5, (1, 2)),   # 은행 wd30
            ((0.60, 0.15, 0.25), 0.55, (1, 2)),  # th55
            ((0.45, 0.35, 0.20), 0.6, (1, 2)),   # cw45
            ((0.55, 0.30, 0.15), 0.0, (1, 2)),   # 게이트 0 (전행 m1 단독)
            ((0.55, 0.30, 0.15), 1.1, (1, 2)),   # 게이트 전개방
            ((0.70, 0.30), 0.5, (1,)),           # 2멤버 (smoke25 구조)
        ]
        for w, th, cond in cases:
            m = M if len(w) == 3 else 2
            meta = {"ensemble": [{"dir": f"m{i}"} for i in range(m)],
                    "weights": list(w), "version": "v6", "max_len": 320, "batch_size": 128,
                    "conditional": {"margin_th": th, "cond_members": list(cond)}}
            ref = ad_lib.predict_conditional_probs("/fake", samples, meta)
            mine, _ = L.cascade_probs(probs[:m], w, th, cond)
            assert np.allclose(ref, mine, atol=1e-7, rtol=0), f"cascade 불일치 w={w} th={th}"
    finally:
        ad_lib.predict_logits, ad_lib.serialize = orig_pl, orig_ser


def test_bias_argmax_equals_deploy_formula():
    """ad_lib.predict() L847(scores=log(probs+1e-9)) + L902(pred=(scores+bias).argmax(1))."""
    rng = np.random.default_rng(1)
    P = rng.dirichlet(np.ones(14), size=5000).astype(np.float32)
    bias = postproc.load(L.OLD_BIAS_PATH)
    scores = np.log(P + 1e-9)              # ad_lib L847 그대로
    ref = (scores + bias).argmax(1)        # ad_lib L902 그대로
    assert np.array_equal(L.bias_argmax(P, bias), ref)


def test_splits_and_ids_integrity():
    ids, y, groups, ids_hash = L.load_ids_labels()
    assert ids_hash == L.IDS_SHA16
    folds, dev, hold = L.load_splits()   # sha256 + 배타성 + dev/holdout 무결성 assert 내장
    assert len(folds) == 5
    # 폴드별 클래스 0셀 없음 (14클래스 전부 등장)
    for fi, (_, va) in enumerate(folds):
        cnt = np.bincount(y[va], minlength=14)
        assert (cnt > 0).all(), f"fold{fi} 클래스 0셀"


def test_loader_real_members():
    y, folds, fmap, members, old_bias = _materials()
    uf = L.usable_folds(members, folds)
    assert 0 in uf, "fold0 재료 불완전"
    for m in members:
        assert 0 in m.folds_covered
    # klue 는 조원 f0 + 우리 f14 증분 (folds1+ 자동 확장 확인)
    klue = members[2]
    assert 1 in klue.folds_covered, "klue fold1 누락 — 다폴드 조인 실패"
    # 행수 정합: 커버 행수 == 커버 폴드 va 행수 합
    for m in members:
        n_cov = int(m.cov_mask.sum())
        n_exp = sum(len(folds[f][1]) for f in m.folds_covered)
        assert n_cov == n_exp, f"{m.name} 행수 부정합 {n_cov} != {n_exp}"


def test_multi_fold_member_join():
    """다폴드 파일(v9 f1-4) + 단폴드 파일(v6 f0) 조인 — 배타성/커버리지 기계 검증."""
    y, folds, fmap, members, old_bias = _materials()
    m = L.MemberOOF("m1proxy", [os.path.join(L.ROOT, "work/teacher_largev6_12ep_f0.npz"),
                                os.path.join(L.ROOT, "work/teacher_largev9_f14.npz")],
                    folds, y, proxy=True)
    assert m.folds_covered == [0, 1, 2, 3, 4]
    # 같은 파일 2회 = 같은 폴드 이중 커버 → 배타성 assert 에 걸려야 함
    caught = False
    try:
        L.MemberOOF("dup", [os.path.join(L.ROOT, "work/klue_f0.npz")] * 2, folds, y)
    except AssertionError as e:
        caught = "배타성" in str(e)
    assert caught, "중복 커버가 배타성 assert 를 통과함"


def test_shrink_and_temperature():
    rng = np.random.default_rng(2)
    old = rng.normal(size=14)
    fit = rng.normal(size=14)
    assert np.allclose(L.shrink_bias(old, fit, 1.0), fit)
    assert np.allclose(L.shrink_bias(old, fit, 0.0), old)
    assert np.allclose(L.shrink_bias(old, fit, 0.5), (old + fit) / 2)
    P = rng.dirichlet(np.ones(14), size=100)
    assert np.allclose(L.apply_temperature(P, 1.0), P)
    Q = L.apply_temperature(P, 1.25)
    assert np.allclose(Q.sum(1), 1.0) and not np.allclose(Q, P)


def test_grid_hash_stable():
    assert L.grid_hash() == "bc1f671e7f9620c1", (
        "사전등록 그리드 변경 감지 — R49 서명 위반 여부 확인 필요")


def test_known_probes():
    y, folds, fmap, members, old_bias = _materials()
    ok, _ = D.run_probes(y, folds, fmap, members, old_bias)
    assert ok, "known-probe 회귀 실패 — 수식 차이 규명 필요"


def test_fold0_redteam_reproduction():
    if SKIP_SLOW:
        print("skip (AD_SKIP_SLOW=1)")
        return
    y, folds, fmap, members, old_bias = _materials()
    ok, lines, res = D.run_fold0_repro(y, folds, fmap, members, old_bias)
    assert ok, f"레드팀 D12 재현 실패: {lines}"
    # 교차 half 동일좌표 (0.45,0.35,0.20) 확인 — cw45 안정점
    for _, (w, th) in [("A", res["joint"][1]), ("B", res["joint"][2])]:
        assert w == (0.45, 0.35, 0.20), f"joint 선택 가중 상이: {w}"


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for name, fn in fns:
        print(f"--- {name} ---")
        try:
            fn()
            print(f"    PASS")
        except Exception as e:
            print(f"    FAIL: {e}")
            failed.append(name)
    print("=" * 60)
    print("FAILED:", failed) if failed else print("ALL TESTS PASS")
    sys.exit(1 if failed else 0)
