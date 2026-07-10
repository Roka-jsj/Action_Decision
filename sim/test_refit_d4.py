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


def _fake_ad_lib(probs, id2row):
    """ad_lib predict_logits/serialize monkeypatch 컨텍스트 헬퍼 — 호출 행수도 기록."""
    calls = {}

    def fake_predict_logits(model_dir, subset, *a, **kw):
        mi = int(os.path.basename(model_dir).replace("m", ""))
        rows = [id2row[s["id"]] for s in subset]
        calls[mi] = calls.get(mi, 0) + len(rows)
        assert kw.get("return_probs")
        return probs[mi][rows]

    return fake_predict_logits, calls


def test_cascade_stages_equals_ad_lib_tt30():
    """R51 tt30: 2단 조건부 가중 — ad_lib 신설 경로와 refit_lib 시뮬 배열 단위 동일성
    + 단일 stage == 기존 단일-th 경로 + cond 추론 행수 불변(시간 영향 0) 증명."""
    rng = np.random.default_rng(7)
    n = 1200
    probs = []
    for _ in range(3):
        z = rng.normal(size=(n, 14)).astype(np.float32) * 2
        e = np.exp(z - z.max(1, keepdims=True))
        probs.append((e / e.sum(1, keepdims=True)).astype(np.float32))
    samples = [{"id": f"r{i}"} for i in range(n)]
    id2row = {f"r{i}": i for i in range(n)}
    fake, calls = _fake_ad_lib(probs, id2row)
    orig_pl, orig_ser = ad_lib.predict_logits, ad_lib.serialize
    ad_lib.predict_logits, ad_lib.serialize = fake, (lambda s, ver: s["id"])
    try:
        w, th, cond = (0.45, 0.35, 0.20), 0.6, (1, 2)
        base = {"ensemble": [{"dir": f"m{i}"} for i in range(3)],
                "weights": list(w), "version": "v6", "max_len": 320, "batch_size": 128}
        stage_sets = [
            list(L.TT30_STAGES),                                    # tt30 본좌표
            [{"th": 0.6, "weights": [0.45, 0.35, 0.20]},
             {"th": 0.2, "weights": [0.30, 0.40, 0.30]}],           # 변형 2단
            [{"th": 0.3, "weights": [0.40, 0.35, 0.25]}],           # 부분단(얕은 행은 1단 가중? -> 미커버 행 p_full 유지)
        ]
        for stages in stage_sets:
            meta = dict(base, conditional={"margin_th": th, "cond_members": list(cond),
                                           "stages": [dict(s) for s in stages]})
            ref = ad_lib.predict_conditional_probs("/fake", samples, meta)
            mine, _ = L.cascade_probs(probs, w, th, cond, stages=stages)
            assert np.allclose(ref, mine, atol=1e-7, rtol=0), f"stages 불일치: {stages}"
        # 단일 stage(th=margin_th, weights=top-level) == 기존 단일-th 경로 (수치 동일)
        meta1 = dict(base, conditional={"margin_th": th, "cond_members": list(cond),
                                        "stages": [{"th": th, "weights": list(w)}]})
        meta0 = dict(base, conditional={"margin_th": th, "cond_members": list(cond)})
        p1 = ad_lib.predict_conditional_probs("/fake", samples, meta1)
        p0 = ad_lib.predict_conditional_probs("/fake", samples, meta0)
        assert np.allclose(p0, p1, atol=1e-7, rtol=0), "단일 stage != 기존 경로"
        # cond 추론 행수 불변 (staged vs plain): m1은 전행, cond 멤버는 게이트행만 — 동일
        calls.clear()
        ad_lib.predict_conditional_probs("/fake", samples, meta0)
        plain_calls = dict(calls)
        calls.clear()
        metaT = dict(base, conditional={"margin_th": th, "cond_members": list(cond),
                                        "stages": [dict(s) for s in L.TT30_STAGES]})
        ad_lib.predict_conditional_probs("/fake", samples, metaT)
        assert calls == plain_calls, f"추론 행수 변화(시간 영향): {plain_calls} -> {calls}"
    finally:
        ad_lib.predict_logits, ad_lib.serialize = orig_pl, orig_ser


def test_stages_th_guard():
    """stage th > margin_th 는 게이트 밖 행(cond 미추론)이라 거부되어야 함."""
    rng = np.random.default_rng(3)
    P = [rng.dirichlet(np.ones(14), size=50).astype(np.float32) for _ in range(3)]
    bad = [{"th": 0.7, "weights": [0.45, 0.35, 0.20]}]
    caught = False
    try:
        L.cascade_probs(P, (0.45, 0.35, 0.20), 0.6, (1, 2), stages=bad)
    except AssertionError:
        caught = True
    assert caught, "stage th > margin_th 가 통과됨"


def test_tt30_fold0_redteam_reproduction():
    """tt30 클린 델타(레드팀 R51 실측 +0.00091 vs cw45) 재현 + 실데이터에서
    ad_lib 신설 경로와 시뮬의 예측 완전 일치."""
    y, folds, fmap, members, old_bias = _materials()
    rows = np.sort(folds[0][1])
    mems = [m.oof[rows] for m in members]
    yb = y[rows]
    P_cw, _ = L.cascade_probs(mems, (0.45, 0.35, 0.20), 0.6)
    P_tt, _ = L.cascade_probs(mems, L.TT30_W, L.TT30_TH, stages=L.TT30_STAGES)
    f_cw = L.score(P_cw, old_bias, yb)
    f_tt = L.score(P_tt, old_bias, yb)
    d = f_tt - f_cw
    assert 0.0005 <= d <= 0.0015, f"tt30 fold0 Δ{d:+.5f} — 레드팀 +0.00091 재현 실패"
    # 실데이터 배열로 ad_lib 신설 경로 대조 (monkeypatch)
    samples = [{"id": f"r{i}"} for i in range(len(rows))]
    id2row = {f"r{i}": i for i in range(len(rows))}
    fake, _ = _fake_ad_lib(mems, id2row)
    orig_pl, orig_ser = ad_lib.predict_logits, ad_lib.serialize
    ad_lib.predict_logits, ad_lib.serialize = fake, (lambda s, ver: s["id"])
    try:
        meta = {"ensemble": [{"dir": f"m{i}"} for i in range(3)],
                "weights": list(L.TT30_W), "version": "v6",
                "conditional": {"margin_th": L.TT30_TH, "cond_members": [1, 2],
                                "stages": [dict(s) for s in L.TT30_STAGES]}}
        ref = ad_lib.predict_conditional_probs("/fake", samples, meta)
    finally:
        ad_lib.predict_logits, ad_lib.serialize = orig_pl, orig_ser
    assert np.allclose(ref, P_tt, atol=1e-7, rtol=0)
    assert np.array_equal(L.bias_argmax(ref, old_bias), L.bias_argmax(P_tt, old_bias))


def test_gen_rescue_byte_identity_and_targets():
    """GEN-rescue(R53): ①비대상 행 input_ids 완전 동일(byte-identity) ②대상 행은
    [GEN] 헤더 보존 + len<=max_len + 꼬리([CUR] 끝) 유지 ③v8/무[GEN] 텍스트 무대상."""
    ckpt = os.path.join(L.ROOT, "work", "foldckpt_largev6_f0ckpt_f0")
    if not os.path.isdir(ckpt):
        print("skip (fold0 ckpt 없음)")
        return
    from transformers import AutoTokenizer
    from common.io_utils import load_train
    tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
    tok.truncation_side = "left"
    MAXLEN = 320
    samples = load_train()[0]
    folds, _, _ = L.load_splits()
    va0 = np.sort(folds[0][1])
    # 대상 밀집 확보: 앞 2000행 + history 길이 상위 1500행
    sub = list(va0[:2000]) + sorted(va0, key=lambda i: -len(str(samples[i].get("history", ""))))[:1500]
    sub = sorted(set(int(i) for i in sub))
    texts = [ad_lib.serialize(samples[i], "v6") for i in sub]
    rescued = ad_lib._gen_rescue_ids(tok, texts, MAXLEN)
    assert rescued, "GEN삭제 대상 행 미검출 — 탐지 로직 확인 필요"
    n_special = tok.num_special_tokens_to_add(False)
    ids_nosp = tok(texts, add_special_tokens=False)["input_ids"]
    for i, t in enumerate(texts):
        old = tok(t, truncation=True, max_length=MAXLEN)["input_ids"]
        if i in rescued:
            new = rescued[i]
            assert len(new) <= MAXLEN
            dec = tok.decode(new)
            assert "[GEN]" in dec, "rescued 행에 [GEN] 부재"
            assert "[CUR]" in dec, "rescued 행에 [CUR](꼬리) 부재"
            assert new[-5:] == old[-5:], "꼬리 끝 토큰 불일치(최근 문맥 소실)"
            # 대상 조건 재검증: 절단 && 유지창에 [GEN] 없음
            assert len(ids_nosp[i]) + n_special > MAXLEN
            assert "[GEN]" not in tok.decode(ids_nosp[i][-(MAXLEN - n_special):])
        elif "[GEN]" in t and len(ids_nosp[i]) + n_special > MAXLEN:
            # 절단이지만 [GEN] 생존 → 무대상이 맞는지
            assert "[GEN]" in tok.decode(ids_nosp[i][-(MAXLEN - n_special):])
    # 혼합 배치 byte-identity: 배치 토크나이즈 vs 단건 토크나이즈(신형 경로) ids 동일
    bt = tok(texts[:64], padding=True, truncation=True, max_length=MAXLEN,
             pad_to_multiple_of=8)["input_ids"]
    for j in range(64):
        single = tok(texts[j], truncation=True, max_length=MAXLEN)["input_ids"]
        assert bt[j][:len(single)] == single, "배치 vs 단건 토크나이즈 불일치"
    # v8(헤더 꼬리 배치)·[GEN] 제거 텍스트 → 무대상(자기가드)
    t8 = [ad_lib.serialize(samples[i], "v8") for i in sub[:500]]
    assert not ad_lib._gen_rescue_ids(tok, t8, MAXLEN), "v8 텍스트 오검출"
    tn = [tx.replace("[GEN] sim ", "").replace("[GEN] au ", "") for tx in texts[:500]]
    assert not ad_lib._gen_rescue_ids(tok, tn, MAXLEN), "[GEN] 무존재 텍스트 오검출"
    print(f"    (대상 {len(rescued)}/{len(texts)}행, byte-identity OK)")


def test_compress_tta_equals_mirror():
    """R55 D1: compress_tta 배포 경로 — numpy 미러와 배열 동일 + 비대상 행 불변(회귀)
    + members 옵션 + tta margin_th 가드."""
    import transformers
    rng = np.random.default_rng(55)
    n = 600
    NB = [rng.dirichlet(np.ones(14), size=n).astype(np.float32) for _ in range(3)]  # 일반뷰
    TB = [rng.dirichlet(np.ones(14), size=n).astype(np.float32) for _ in range(3)]  # 압축뷰
    samples = [{"id": f"r{i}"} for i in range(n)]
    id2row = {f"r{i}": i for i in range(n)}

    class FakeTok:
        truncation_side = "left"

        def num_special_tokens_to_add(self, pair=False):
            return 2

    def _is_tgt(text):                                  # 행 고유(id) 기준 — 위치 무관 일관 판정
        return int(str(text).replace("COMP::", "")[1:]) % 3 == 0

    def fake_rescue_ids(tok, texts, max_len):
        return {j: [0] for j, t in enumerate(texts) if _is_tgt(t)}

    REUSE = {"on": False}

    def fake_predict_logits(model_dir, subset, *a, **kw):
        mi = int(os.path.basename(model_dir).replace("m", ""))
        bank = TB if (kw.get("texts") and str(kw["texts"][0]).startswith("COMP::")) else NB
        if REUSE["on"] and kw.get("rescue_rows_out") is not None:
            kw["rescue_rows_out"]["rows"] = [id2row[s["id"]] for s in subset
                                             if _is_tgt(s["id"])]   # 스캔 재사용 경로
        return bank[mi][[id2row[s["id"]] for s in subset]]

    orig = (ad_lib.predict_logits, ad_lib.serialize, ad_lib.serialize_compress,
            ad_lib._gen_rescue_ids, transformers.AutoTokenizer.from_pretrained)
    ad_lib.predict_logits = fake_predict_logits
    ad_lib.serialize = lambda s, ver, mht=8: s["id"]
    ad_lib.serialize_compress = lambda s, tok, keep, **kw: f"COMP::{s['id']}"
    ad_lib._gen_rescue_ids = fake_rescue_ids
    transformers.AutoTokenizer.from_pretrained = staticmethod(lambda *a, **kw: FakeTok())
    try:
        w, th, lam, tth = (0.45, 0.40, 0.15), 0.75, 0.5, 0.5
        base = {"ensemble": [{"dir": f"m{i}"} for i in range(3)], "weights": list(w),
                "version": "v6", "conditional": {"margin_th": th, "cond_members": [1, 2]}}
        meta_t = dict(base, compress_tta={"lambda": lam, "margin_th": tth})
        out_t = ad_lib.predict_conditional_probs("/fake", samples, meta_t)
        out_0 = ad_lib.predict_conditional_probs("/fake", samples, base)
        # numpy 미러
        p0 = (w[0] * NB[0]) / w[0]
        srt = np.sort(p0, axis=1)
        marg = srt[:, -1] - srt[:, -2]
        sel = marg < th
        cand = np.where(marg < tth)[0]
        rows_tta = np.array([i for i in cand if i % 3 == 0])
        blend = []
        for mi in range(3):
            B = (NB[mi] if mi else p0).copy()
            B[rows_tta] = (1 - lam) * B[rows_tta] + lam * TB[mi][rows_tta]
            blend.append(B)
        exp = blend[0].copy()
        acc = w[0] * blend[0][sel]
        for mi in (1, 2):
            acc = acc + w[mi] * blend[mi][sel]
        exp[sel] = acc / sum(w)
        assert np.allclose(out_t, exp, atol=1e-6, rtol=0), "compress_tta 미러 불일치"
        # 비대상 행: TTA 유무 무관 동일
        tta_mask = np.zeros(n, bool)
        tta_mask[rows_tta] = True
        assert np.allclose(out_t[~tta_mask], out_0[~tta_mask], atol=1e-7, rtol=0), \
            "비대상 행 변경 — byte-identity 위반"
        assert not np.allclose(out_t[tta_mask], out_0[tta_mask], atol=1e-7), "TTA 미적용 의심"
        # 스캔 재사용 경로(rescue_rows_out) == 스캔 폴백 경로 (배열 동일)
        REUSE["on"] = True
        out_r = ad_lib.predict_conditional_probs("/fake", samples, meta_t)
        REUSE["on"] = False
        assert np.allclose(out_r, out_t, atol=1e-7, rtol=0), "재사용 경로 != 스캔 경로"
        # members 옵션: [0] 만 — cond 멤버 확률 불변 → m1 채널만 반영된 미러
        meta_m = dict(base, compress_tta={"lambda": lam, "margin_th": tth, "members": [0]})
        out_m = ad_lib.predict_conditional_probs("/fake", samples, meta_m)
        exp_m = blend[0].copy()
        acc = w[0] * blend[0][sel] + w[1] * NB[1][sel] + w[2] * NB[2][sel]
        exp_m[sel] = acc / sum(w)
        assert np.allclose(out_m, exp_m, atol=1e-6, rtol=0), "members 옵션 미러 불일치"
        # tta margin_th 가드
        caught = False
        try:
            ad_lib.predict_conditional_probs(
                "/fake", samples, dict(base, compress_tta={"lambda": 0.5, "margin_th": 0.9}))
        except AssertionError:
            caught = True
        assert caught, "tta margin_th > margin_th 가 통과됨"
    finally:
        (ad_lib.predict_logits, ad_lib.serialize, ad_lib.serialize_compress,
         ad_lib._gen_rescue_ids, transformers.AutoTokenizer.from_pretrained) = orig


def test_member_max_len_override():
    """R57: conditional 경로 멤버별 max_len 오버라이드 — 지정 멤버만 변경, 미지정=글로벌(회귀)."""
    rng = np.random.default_rng(9)
    n = 300
    probs = [rng.dirichlet(np.ones(14), size=n).astype(np.float32) for _ in range(3)]
    samples = [{"id": f"r{i}"} for i in range(n)]
    id2row = {f"r{i}": i for i in range(n)}
    seen = {}

    def fake_predict_logits(model_dir, subset, *a, **kw):
        mi = int(os.path.basename(model_dir).replace("m", ""))
        seen.setdefault(mi, set()).add(kw.get("max_len"))
        return probs[mi][[id2row[s["id"]] for s in subset]]

    orig_pl, orig_ser = ad_lib.predict_logits, ad_lib.serialize
    ad_lib.predict_logits, ad_lib.serialize = fake_predict_logits, (lambda s, ver, mht=8: s["id"])
    try:
        base = {"ensemble": [{"dir": "m0"}, {"dir": "m1", "max_len": 384}, {"dir": "m2"}],
                "weights": [0.45, 0.40, 0.15], "version": "v6", "max_len": 320,
                "conditional": {"margin_th": 0.75, "cond_members": [1, 2]}}
        out_o = ad_lib.predict_conditional_probs("/fake", samples, base)
        assert seen[0] == {320} and seen[1] == {384} and seen[2] == {320}, f"max_len 전달 오류: {seen}"
        seen.clear()
        no = {**base, "ensemble": [{"dir": "m0"}, {"dir": "m1"}, {"dir": "m2"}]}
        out_n = ad_lib.predict_conditional_probs("/fake", samples, no)
        assert seen[0] == seen[1] == seen[2] == {320}
        assert np.array_equal(out_o, out_n)   # fake 확률은 max_len 무관 — 경로 회귀
    finally:
        ad_lib.predict_logits, ad_lib.serialize = orig_pl, orig_ser


def test_member_th_asymmetric_coverage():
    """R60 S2: member_th 비대칭 커버리지 — ①ad_lib==refit_lib 미러 ②mid-stage 재정규
    식 (0.45L+0.40D)/0.85 배열 일치(레드팀 식) ③klue 추론행 절감 ④균일 th==기존 경로."""
    rng = np.random.default_rng(60)
    n = 900
    # m1 마진이 0~1 전대역에 퍼지도록 로짓 스케일 부여 (low/mid/게이트밖 3구간 모두 채움)
    z = rng.normal(size=(n, 14)) * rng.uniform(0.5, 6.0, size=(n, 1))
    e = np.exp(z - z.max(1, keepdims=True))
    probs = [(e / e.sum(1, keepdims=True)).astype(np.float32),
             rng.dirichlet(np.ones(14), size=n).astype(np.float32),
             rng.dirichlet(np.ones(14), size=n).astype(np.float32)]
    samples = [{"id": f"r{i}"} for i in range(n)]
    id2row = {f"r{i}": i for i in range(n)}
    calls = {}

    def fake_predict_logits(model_dir, subset, *a, **kw):
        mi = int(os.path.basename(model_dir).replace("m", ""))
        calls[mi] = calls.get(mi, 0) + len(subset)
        return probs[mi][[id2row[s["id"]] for s in subset]]

    orig_pl, orig_ser = ad_lib.predict_logits, ad_lib.serialize
    ad_lib.predict_logits, ad_lib.serialize = fake_predict_logits, (lambda s, ver, mht=8: s["id"])
    try:
        w, TH = (0.45, 0.40, 0.15), 0.95
        mth = {"1": 0.95, "2": 0.75}
        meta = {"ensemble": [{"dir": f"m{i}"} for i in range(3)], "weights": list(w),
                "version": "v6",
                "conditional": {"margin_th": TH, "cond_members": [1, 2], "member_th": mth}}
        calls.clear()
        out = ad_lib.predict_conditional_probs("/fake", samples, meta)
        mine, _ = L.cascade_probs(probs, w, TH, (1, 2), member_th={1: 0.95, 2: 0.75})
        assert np.allclose(out, mine, atol=1e-7, rtol=0), "ad_lib != refit_lib 미러"
        # 레드팀 식 직접 대조: mid(0.75<=m<0.95) = (wL*L+wD*D)/0.85, low(<0.75) = 3멤버/1.0
        p0 = (w[0] * probs[0]) / w[0]
        srt = np.sort(p0, axis=1)
        marg = srt[:, -1] - srt[:, -2]
        mid = (marg >= 0.75) & (marg < 0.95)
        low = marg < 0.75
        exp_mid = (w[0] * p0[mid] + w[1] * probs[1][mid]) / (w[0] + w[1])
        exp_low = (w[0] * p0[low] + w[1] * probs[1][low] + w[2] * probs[2][low]) / sum(w)
        assert np.allclose(out[mid], exp_mid, atol=1e-7, rtol=0), "mid-stage 재정규 불일치"
        assert np.allclose(out[low], exp_low, atol=1e-7, rtol=0), "low-stage 3멤버 혼합 불일치"
        assert np.allclose(out[marg >= 0.95], p0[marg >= 0.95], atol=1e-7), "게이트 밖 행 변경"
        # klue(멤버2) 추론행 = low 행수만 (< mdeb 행수 = sel 행수) — 시간 절감 원천
        assert calls[2] == int(low.sum()) and calls[1] == int((marg < 0.95).sum())
        assert calls[2] < calls[1]
        # 균일 member_th == 기존 경로 (수치 동일 수준)
        meta_u = {"ensemble": [{"dir": f"m{i}"} for i in range(3)], "weights": list(w),
                  "version": "v6",
                  "conditional": {"margin_th": TH, "cond_members": [1, 2],
                                  "member_th": {"1": TH, "2": TH}}}
        meta_0 = {"ensemble": [{"dir": f"m{i}"} for i in range(3)], "weights": list(w),
                  "version": "v6", "conditional": {"margin_th": TH, "cond_members": [1, 2]}}
        ou = ad_lib.predict_conditional_probs("/fake", samples, meta_u)
        o0 = ad_lib.predict_conditional_probs("/fake", samples, meta_0)
        assert np.allclose(ou, o0, atol=1e-6, rtol=0), "균일 member_th != 기존 경로"
        # 가드: member_th > margin_th
        bad = {"ensemble": [{"dir": f"m{i}"} for i in range(3)], "weights": list(w),
               "version": "v6",
               "conditional": {"margin_th": 0.9, "cond_members": [1, 2],
                               "member_th": {"1": 0.95}}}
        caught = False
        try:
            ad_lib.predict_conditional_probs("/fake", samples, bad)
        except AssertionError:
            caught = True
        assert caught, "member_th > margin_th 통과됨"
    finally:
        ad_lib.predict_logits, ad_lib.serialize = orig_pl, orig_ser


def test_serialize_compress_real():
    """serialize_compress 실데이터: <=keep 보장·[GEN]/[CUR]/[SEQ] 보존 (이분탐색 경계 검증)."""
    ckpt = os.path.join(L.ROOT, "work", "foldckpt_largev6_f0ckpt_f0")
    if not os.path.isdir(ckpt):
        print("skip (ckpt 없음)")
        return
    from transformers import AutoTokenizer
    from common.io_utils import load_train
    tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
    tok.truncation_side = "left"
    samples = load_train()[0]
    long_rows = sorted(range(20000), key=lambda i: -len(str(samples[i].get("history", ""))))[:120]
    for i in long_rows:
        c = ad_lib.serialize_compress(samples[i], tok, 318)
        ids = tok(c, add_special_tokens=False)["input_ids"]
        assert len(ids) <= 318, f"row {i}: {len(ids)} > 318"
        assert "[GEN]" in c and "[CUR]" in c and "[SEQ]" in c


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
    assert L.rescue_grid_hash() == "a619e74ed3a75dfb", (
        "R54 rescue 그리드 변경 감지 — 사전등록(레드팀 T5) 위반 여부 확인 필요")


def test_rescue_two_stage_smoke():
    """R54 T5-light 2단 게이트 스모크 — rescue OOF 존재 시 전체 경로 구동.

    ①m1 rescue/old npz 로더 통과(행순서·폴드·F1 정합) ②변경행 == rescue 대상만
    ③Stage1 표(15좌표)·anchor 좌표 Δ=0 ④Stage2 veto 폴드 자동인식."""
    p_old = os.path.join(L.ROOT, D.RESCUE_M1_OLD)
    p_new = os.path.join(L.ROOT, D.RESCUE_M1_NEW)
    if not (os.path.exists(p_old) and os.path.exists(p_new)):
        print("skip (rescue OOF 없음 — sim/gen_rescue_oof.py 선행)")
        return
    y, folds, fmap, members, old_bias = _materials()
    out = D.run_rescue(y, folds, fmap, old_bias)
    # anchor 좌표 자체의 Δ는 정확히 0 (표에서 확인)
    anchor_line = [ln for ln in out["lines"]
                   if f"w={L.RESCUE_ANCHOR_W} th={L.RESCUE_ANCHOR_TH}" in ln]
    assert anchor_line and "+0.00000" in anchor_line[0], "anchor 좌표 Δ != 0"
    # rescue OOF 무결성: 변경행 ⊆ fold0 va, old/new 커버 동일
    o = np.load(p_old, allow_pickle=True)["oof"]
    n = np.load(p_new, allow_pickle=True)["oof"]
    diff = np.where(np.abs(o - n).max(1) > 0)[0]
    va0 = set(np.asarray(folds[0][1]).tolist())
    assert set(diff.tolist()) <= va0, "fold0 밖 행 변경"
    assert 1000 < len(diff) < 2500, f"변경행 수 이상: {len(diff)} (기대 ~1603)"


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
