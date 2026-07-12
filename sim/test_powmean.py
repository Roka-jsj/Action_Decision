#!/usr/bin/env python3
"""test_powmean — R64 멱평균(power-mean/geometric) 조합기 회귀·수식동일성 테스트.

증명:
  1) refit_lib.cascade_probs(combiner=p) == ad_lib.predict_conditional_probs(
     conditional.combiner={"kind":"powmean","p":p}) — 배열 단위 (atol=1e-6),
     simple·stages·member_th 세 경로 전부, p ∈ {0(기하), -0.5, -1.0, 0.5, 1.0}.
  2) 산술(기본, combiner 부재) == refit_lib 산술 미러 (동일성 회귀).
  3) DEFAULT-OFF BYTE-IDENTITY: combiner 키 없으면 결과가 산술 경로와 완전 동일
     (np.array_equal) — p=1.0(멱평균 항등 근사)가 아니라 코드 경로 자체가 불변임을 증명.
  4) p=1.0(멱평균) ≈ 산술 (재정규·eps 오차만; atol=1e-6).
  5) combiner+compress_tta 동시 사용 가드(ValueError), kind 미지원 가드.

실행: python3 sim/test_powmean.py   /   python3 -m pytest sim/test_powmean.py -q
"""
from __future__ import annotations
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import refit_lib as L  # noqa: E402
from common import ad_lib  # noqa: E402


def _mk_probs(n=1000, seed=64):
    rng = np.random.default_rng(seed)
    probs = []
    for _ in range(3):
        z = rng.normal(size=(n, 14)).astype(np.float32) * rng.uniform(0.5, 5.0, size=(n, 1)).astype(np.float32)
        e = np.exp(z - z.max(1, keepdims=True))
        probs.append((e / e.sum(1, keepdims=True)).astype(np.float32))
    samples = [{"id": f"r{i}"} for i in range(n)]
    id2row = {f"r{i}": i for i in range(n)}
    return probs, samples, id2row


def _patch(probs, id2row):
    def fake_predict_logits(model_dir, subset, *a, **kw):
        mi = int(os.path.basename(model_dir).replace("m", ""))
        return probs[mi][[id2row[s["id"]] for s in subset]]
    orig = ad_lib.predict_logits, ad_lib.serialize
    ad_lib.predict_logits = fake_predict_logits
    ad_lib.serialize = lambda s, ver, mht=8: s["id"]
    return orig


def _base(cond):
    return {"ensemble": [{"dir": f"m{i}"} for i in range(3)], "weights": [0.45, 0.40, 0.15],
            "version": "v6", "max_len": 320, "batch_size": 128, "conditional": cond}


_STAGES = [{"th": 0.6, "weights": [0.45, 0.35, 0.20]}, {"th": 0.3, "weights": [0.30, 0.40, 0.30]}]
_P_GRID = [0.0, -0.5, -1.0, 0.5, 1.0]


def test_powmean_mirror_equals_ad_lib():
    """멱평균 3경로 × p그리드: ad_lib 실경로 == refit_lib 미러 (atol=1e-6). 산술도 포함."""
    probs, samples, id2row = _mk_probs()
    orig = _patch(probs, id2row)
    try:
        def chk(cond_ad, refkw, combiner, tag):
            out = ad_lib.predict_conditional_probs("/fake", samples, cond_ad)
            mine, _ = L.cascade_probs(probs, (0.45, 0.40, 0.15), refkw["th"],
                                      refkw["cond"], stages=refkw.get("stages"),
                                      member_th=refkw.get("member_th"), combiner=combiner)
            e = float(np.abs(out - mine).max())
            assert e < 1e-6, f"{tag}: ad_lib!=refit_lib max|Δ|={e:.2e}"
            # 확률 유효성: 행합 1
            assert np.allclose(out.sum(1), 1.0, atol=1e-5), f"{tag}: 행합≠1"
            return e

        cfgs = [
            ("simple", dict(th=0.85, cond=(1, 2)),
             lambda cb: _base({"margin_th": 0.85, "cond_members": [1, 2], **cb})),
            ("stages", dict(th=0.6, cond=(1, 2), stages=_STAGES),
             lambda cb: _base({"margin_th": 0.6, "cond_members": [1, 2],
                               "stages": [dict(s) for s in _STAGES], **cb})),
            ("member_th", dict(th=0.95, cond=(1, 2), member_th={1: 0.95, 2: 0.75}),
             lambda cb: _base({"margin_th": 0.95, "cond_members": [1, 2],
                               "member_th": {"1": 0.95, "2": 0.75}, **cb})),
        ]
        worst = 0.0
        # 산술(기본): combiner 인자 없음
        for tag, refkw, mk in cfgs:
            worst = max(worst, chk(mk({}), refkw, None, f"arith/{tag}"))
        # 멱평균: p 그리드
        for p in _P_GRID:
            for tag, refkw, mk in cfgs:
                cb = {"combiner": {"kind": "powmean", "p": p}}
                worst = max(worst, chk(mk(cb), refkw, p, f"pm(p={p})/{tag}"))
        print(f"  ad_lib==refit_lib 최대오차 {worst:.2e} (arith+powmean 전경로, atol=1e-6) PASS")
    finally:
        ad_lib.predict_logits, ad_lib.serialize = orig


def test_powmean_default_off_byte_identity():
    """combiner 키 부재 = 산술 경로 완전 불변(byte-identity). 산술 참조 스냅샷과 array_equal."""
    probs, samples, id2row = _mk_probs(seed=99)
    orig = _patch(probs, id2row)
    try:
        cfgs = {
            "simple": _base({"margin_th": 0.85, "cond_members": [1, 2]}),
            "stages": _base({"margin_th": 0.6, "cond_members": [1, 2],
                             "stages": [dict(s) for s in _STAGES]}),
            "member_th": _base({"margin_th": 0.95, "cond_members": [1, 2],
                                "member_th": {"1": 0.95, "2": 0.75}}),
        }
        # 산술 참조 = combiner 부재 실행. refit_lib 산술과도 정확 일치.
        for tag, meta in cfgs.items():
            out = ad_lib.predict_conditional_probs("/fake", samples, meta)
            mine, _ = L.cascade_probs(probs, (0.45, 0.40, 0.15),
                                      meta["conditional"]["margin_th"], (1, 2),
                                      stages=meta["conditional"].get("stages"),
                                      member_th={int(k): v for k, v in
                                                 meta["conditional"].get("member_th", {}).items()} or None)
            assert np.array_equal(out, mine), f"{tag}: 산술 default-off != refit_lib 미러"
        # p=1.0 멱평균 ≈ 산술 (eps·재정규 오차만)
        base = cfgs["simple"]
        a = ad_lib.predict_conditional_probs("/fake", samples, base)
        pm1 = ad_lib.predict_conditional_probs(
            "/fake", samples, _base({"margin_th": 0.85, "cond_members": [1, 2],
                                     "combiner": {"kind": "powmean", "p": 1.0}}))
        assert np.abs(a - pm1).max() < 1e-6, "p=1.0 멱평균이 산술과 어긋남"
        print("  default-off byte-identity(array_equal) + p=1.0≈산술 PASS")
    finally:
        ad_lib.predict_logits, ad_lib.serialize = orig


def test_powmean_guards():
    """combiner+compress_tta 동시 가드(ValueError, python -O 생존) + kind 미지원 가드."""
    probs, samples, id2row = _mk_probs(n=200, seed=7)
    orig = _patch(probs, id2row)
    try:
        bad_tta = _base({"margin_th": 0.85, "cond_members": [1, 2],
                         "combiner": {"kind": "powmean", "p": 0.0}})
        bad_tta["compress_tta"] = {"lambda": 0.5, "margin_th": 0.5}
        caught = False
        try:
            ad_lib.predict_conditional_probs("/fake", samples, bad_tta)
        except ValueError:
            caught = True
        assert caught, "combiner+compress_tta 가드 미발동"
        # kind 미지원
        caught = False
        try:
            ad_lib.predict_conditional_probs(
                "/fake", samples, _base({"margin_th": 0.85, "cond_members": [1, 2],
                                         "combiner": {"kind": "median", "p": 0.0}}))
        except ValueError:
            caught = True
        assert caught, "combiner kind 미지원 가드 미발동"
        # refit_lib dict 형태 파싱
        d, _ = L.cascade_probs(probs, (0.45, 0.40, 0.15), 0.85, (1, 2),
                               combiner={"kind": "powmean", "p": 0.0})
        f, _ = L.cascade_probs(probs, (0.45, 0.40, 0.15), 0.85, (1, 2), combiner=0.0)
        assert np.array_equal(d, f), "refit_lib dict-combiner != float-combiner"
        print("  combiner 가드(tta/kind) + refit dict파싱 PASS")
    finally:
        ad_lib.predict_logits, ad_lib.serialize = orig


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        print(f"--- {fn.__name__} ---")
        fn()
        print("    PASS")
    print("=" * 60)
    print("ALL POWMEAN TESTS PASS")
