#!/usr/bin/env python3
"""합성 배치 3-게이트 스크린 + dedup (문샷 R57b/R65 §5-2) — 순수 CPU.

게이트 (전부 실측 수치로 보고, 임계 초과 시 FAIL):
  gate0 provenance : 모든 소스 행이 fold0-train 안 (va0/holdout 무접촉) + id 무충돌.
  gate1 proximity  : 클래스별 JS divergence (v6 직렬화 길이 / 토큰 유니그램 /
                     마지막액션 분포) synth vs 실데이터 같은 클래스. 낮을수록
                     "훈련분포 복원" 교리 정합(신규 입력형태=OOD 금지).
  gate2 targeting  : synth 가 실제로 타겟(rare-slot) 클래스를 증량하는가 —
                     타겟 클래스 비중 + 클래스별 상대 증량률(uplift).
  gate3 prior-drift: (tr0+synth) vs tr0 클래스 사전확률 이동 — TVD / max|Δp|.
                     (히든 사전확률은 R30 프로브 실측치가 별도 존재 — 과이동 금지)
  dedup            : (a) T1 공여자쌍 유사도 — work/emb_v6_70k.npy(중심화 코사인,
                     retrieval_diag2 이방성 보정과 동일) : 쌍이 거의 동일하면 스왑이
                     사실상 복제. (b) 어휘 5-gram 코사인 — synth vs {공여자 ∪ 공여자의
                     저장공간 top-K NN} : 표면 복제(patch/paraphrase) 직접 검출.
                     (c) q4 CPU 인코더 스팟체크(동일공간) — synth↔공여자 코사인 분포가
                     "가깝지만 동일 아님" 밴드에 있는지. (a)(b)는 컷 적용, (c)는 정보성.

  주의: CPU(q4) 임베딩과 저장 emb(fp16 원본 모델)의 교차공간 중심화 코사인은
  self-cos 0.84 수준으로 잡음이 커서(실측) dedup 판정에 부적합 — 그래서 (a)는
  저장공간 내부 연산만, (c)는 q4 동일공간 내부 연산만 쓴다.

사용:
  PYTHONPATH=/root/Action_Decision python3 sim/synth_gates.py \
      --synth work/synth_batch0.jsonl --report work/synth_gates_b0.json \
      --emit-pass work/synth_batch0_pass.jsonl
"""
from __future__ import annotations
import argparse
import collections
import json
import math
import os
import re
import sys
import tempfile
import time
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from common.io_utils import load_train, CLASSES, CLASS_TO_IDX  # noqa: E402
from common import parse as P  # noqa: E402
from common.serialize import serialize  # noqa: E402
import refit_lib  # noqa: E402

EMB_PATH = os.path.join(ROOT, "work", "emb_v6_70k.npy")
_WORD = re.compile(r"\w+")


# ----------------------------- 유틸 -----------------------------
def js_div(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence, base-2, [0,1]."""
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-15))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def len_hist(lengths, bins):
    h, _ = np.histogram(lengths, bins=bins)
    return h.astype(np.float64)


def tok_counts(texts):
    c = collections.Counter()
    for t in texts:
        c.update(_WORD.findall(t.lower()))
    return c


def char5_vec(text: str, dim: int = 1 << 18) -> dict:
    import zlib
    v: dict[int, int] = {}
    t = text.lower().encode("utf-8", "ignore")
    for i in range(len(t) - 4):
        h = zlib.crc32(t[i:i + 5]) & (dim - 1)   # 결정적 해시(재현성)
        v[h] = v.get(h, 0) + 1
    return v


def cos_sparse(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / max(na * nb, 1e-12)


def pct(a, qs=(50, 90, 95, 99)):
    a = np.asarray(a, dtype=np.float64)
    if len(a) == 0:
        return {}
    d = {f"p{q}": round(float(np.percentile(a, q)), 4) for q in qs}
    d["max"] = round(float(a.max()), 4)
    d["mean"] = round(float(a.mean()), 4)
    return d


# ----------------------------- 메인 -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--emit-pass", default="", help="dedup 컷 통과 행만 저장할 jsonl")
    ap.add_argument("--mht", type=int, default=12, help="게이트 직렬화 mht (캐리어=mdeb T3=12)")
    # 임계값 (사전 제안치 — 3자 라운드에서 확정)
    # gate1 은 null 보정: 같은 n 의 실데이터 부분표본 JS(null) 대비 비율로 판정
    # (소표본 JS 양의 편향 실측: n=75 에서 null js_tok 0.092 — 절대 임계는 오판).
    # pass 조건: js <= max(floor, ratio_cap * null_mean)
    ap.add_argument("--cap-ratio-len", type=float, default=4.0,
                    help="길이 JS 는 T2 절단의 설계된 이동 포함 — 완화 캡")
    ap.add_argument("--cap-ratio-tok", type=float, default=1.5)
    ap.add_argument("--cap-ratio-last", type=float, default=2.0)
    ap.add_argument("--floor-js", type=float, default=0.03)
    ap.add_argument("--null-draws", type=int, default=6)
    ap.add_argument("--th-share", type=float, default=0.98)
    ap.add_argument("--min-uplift", type=float, default=0.005)
    ap.add_argument("--th-tvd", type=float, default=0.015)
    ap.add_argument("--th-maxdp", type=float, default=0.010)
    ap.add_argument("--cut-donor-cos", type=float, default=0.90, help="T1 공여자쌍 중심화cos 컷")
    ap.add_argument("--cut-lex-t1", type=float, default=0.95)
    ap.add_argument("--cut-lex-t2", type=float, default=0.985)
    ap.add_argument("--max-reject", type=float, default=0.20)
    ap.add_argument("--nn-k", type=int, default=8)
    ap.add_argument("--embed-check", type=int, default=48, help="q4 스팟체크 행수(0=생략)")
    ap.add_argument("--emb-model", default=os.path.join(ROOT, "work", "member_largefullv6_q4.zip"))
    args = ap.parse_args()

    t_start = time.time()
    rep: dict = {"synth": args.synth, "mht": args.mht}

    # ---------- 로드 ----------
    syn = []
    with open(args.synth, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                syn.append(json.loads(line))
    assert syn, "synth 파일 비어있음"
    samples, y, ids = load_train()
    y = np.asarray(y)
    folds, dev_idx, hold_idx = refit_lib.load_splits()
    tr0, va0 = folds[0]
    pool = np.asarray(tr0)
    pool_set = set(int(i) for i in pool)
    va0_set = set(int(i) for i in va0)
    hold_set = set(int(i) for i in hold_idx)
    id_to_row = {v: k for k, v in enumerate(ids)}
    print(f"[load] synth {len(syn)} / pool(tr0) {len(pool)}", flush=True)

    # ---------- gate0 provenance ----------
    bad_src, bad_id = 0, 0
    real_ids = set(ids)
    seen = set()
    for s in syn:
        if s["id"] in real_ids or s["id"] in seen:
            bad_id += 1
        seen.add(s["id"])
        for src in s["_synth"]["src_ids"]:
            ridx = id_to_row.get(src, -1)
            if ridx < 0 or ridx not in pool_set or ridx in va0_set or ridx in hold_set:
                bad_src += 1
    g0 = {"bad_src": bad_src, "bad_id": bad_id, "pass": bad_src == 0 and bad_id == 0}
    rep["gate0_provenance"] = g0
    print(f"[gate0] provenance bad_src={bad_src} bad_id={bad_id} -> {'PASS' if g0['pass'] else 'FAIL'}", flush=True)

    # ---------- 직렬화 (게이트 공용) ----------
    syn_cls = np.array([CLASS_TO_IDX[s["label"]] for s in syn])
    tgt_classes = sorted(set(syn_cls.tolist()))
    syn_txt = [serialize(s, "v6", args.mht) for s in syn]
    pool_by_cls = {c: pool[y[pool] == c] for c in tgt_classes}
    real_txt_by_cls = {c: [serialize(samples[int(i)], "v6", args.mht) for i in pool_by_cls[c]]
                       for c in tgt_classes}

    # ---------- gate1 proximity (null 보정) ----------
    bins = list(range(0, 3200, 160)) + [10 ** 9]
    g1 = {}
    g1_pass = True
    rng_null = np.random.default_rng(20260711)

    def last_key(s_):
        nm, _, _, stt = P.last_action(s_)
        return f"{nm}|{stt}"

    is_crop = np.array([s["_synth"]["template"].startswith("T2") for s in syn])
    for c in tgt_classes:
        mask = syn_cls == c
        st = [t for t, m in zip(syn_txt, mask) if m]
        # 길이 JS 는 재조합(T1/T3)만 판정 — T2 절단의 길이 이동은 설계된 것
        # (histdrop R24 선례, LB 검증)이라 근접성 위반이 아님. T2 는 정보성 보고.
        st_recomb = [t for t, m, cr in zip(syn_txt, mask, is_crop) if m and not cr]
        st_crop = [t for t, m, cr in zip(syn_txt, mask, is_crop) if m and cr]
        rt = real_txt_by_cls[c]
        n_c = len(st)
        rows_c = pool_by_cls[c]
        # 실데이터(풀) 기준 분포
        rc = tok_counts(rt)
        vocab = [w for w, _ in rc.most_common(5000)]
        pv = np.array([rc.get(w, 0) for w in vocab], np.float64) + 1e-9
        ph = len_hist([len(t) for t in rt], bins) + 1e-9
        rl = collections.Counter(last_key(samples[int(i)]) for i in rows_c)

        def three_js(sub_txts, sub_samples):
            sc = tok_counts(sub_txts)
            qv = np.array([sc.get(w, 0) for w in vocab], np.float64) + 1e-9
            qh = len_hist([len(t) for t in sub_txts], bins) + 1e-9
            sl = collections.Counter(last_key(s_) for s_ in sub_samples)
            keys = sorted(set(rl) | set(sl))
            return (js_div(ph, qh), js_div(pv, qv),
                    js_div(np.array([rl.get(k, 0) for k in keys], np.float64) + 1e-9,
                           np.array([sl.get(k, 0) for k in keys], np.float64) + 1e-9))

        _, j_tok, j_last = three_js(st, [s for s, m in zip(syn, mask) if m])
        j_len = js_div(len_hist([len(t) for t in st_recomb], bins) + 1e-9, ph) \
            if st_recomb else 0.0
        j_len_t2 = js_div(len_hist([len(t) for t in st_crop], bins) + 1e-9, ph) \
            if st_crop else 0.0
        # null: 같은 n 의 실데이터 부분표본 (소표본 JS 편향 보정)
        nulls = []
        n_rc = max(len(st_recomb), 2)
        for _ in range(args.null_draws):
            sub = rng_null.choice(len(rows_c), size=min(n_c, len(rows_c)), replace=False)
            jl, jt, jla = three_js([rt[j] for j in sub],
                                   [samples[int(rows_c[j])] for j in sub])
            sub2 = rng_null.choice(len(rows_c), size=min(n_rc, len(rows_c)), replace=False)
            jl2 = js_div(len_hist([len(rt[j]) for j in sub2], bins) + 1e-9, ph)
            nulls.append((jl2, jt, jla))
        nl = np.array(nulls).mean(0)
        lim_len = max(args.floor_js, args.cap_ratio_len * nl[0])
        lim_tok = max(args.floor_js, args.cap_ratio_tok * nl[1])
        lim_last = max(args.floor_js, args.cap_ratio_last * nl[2])
        ok = bool(j_len <= lim_len and j_tok <= lim_tok and j_last <= lim_last)
        g1_pass = bool(g1_pass and ok)
        g1[CLASSES[c]] = {
            "js_len_recomb": round(j_len, 4), "null_len": round(float(nl[0]), 4),
            "js_len_t2_designed": round(j_len_t2, 4),
            "js_tok": round(j_tok, 4), "null_tok": round(float(nl[1]), 4),
            "js_last": round(j_last, 4), "null_last": round(float(nl[2]), 4),
            "ratios": [round(j_len / max(nl[0], 1e-9), 2), round(j_tok / max(nl[1], 1e-9), 2),
                       round(j_last / max(nl[2], 1e-9), 2)],
            "pass": ok}
        print(f"[gate1] {CLASSES[c]:18s} len(재조합) {j_len:.4f}/null {nl[0]:.4f} "
              f"(T2설계이동 {j_len_t2:.4f}) | tok {j_tok:.4f}/null {nl[1]:.4f} | "
              f"last {j_last:.4f}/null {nl[2]:.4f} -> {'PASS' if ok else 'FAIL'}", flush=True)
    rep["gate1_proximity"] = {"per_class": g1,
                              "caps": {"ratio_len": args.cap_ratio_len, "ratio_tok": args.cap_ratio_tok,
                                       "ratio_last": args.cap_ratio_last, "floor": args.floor_js},
                              "pass": g1_pass}
    print(f"[gate1] -> {'PASS' if g1_pass else 'FAIL'}", flush=True)

    # ---------- gate2 targeting ----------
    share = float(np.mean([CLASSES[c] in
                           {CLASSES[t] for t in tgt_classes} for c in syn_cls]))  # 정의상 1.0
    uplift = {}
    for c in tgt_classes:
        n_s = int((syn_cls == c).sum())
        n_r = int(len(pool_by_cls[c]))
        uplift[CLASSES[c]] = {"n_synth": n_s, "n_pool": n_r, "uplift": round(n_s / max(n_r, 1), 4)}
    g2_pass = share >= args.th_share and all(v["uplift"] >= args.min_uplift for v in uplift.values())
    rep["gate2_targeting"] = {"target_share": round(share, 4), "uplift": uplift,
                              "min_uplift": args.min_uplift, "pass": g2_pass}
    print(f"[gate2] share={share:.3f} uplift=" +
          " ".join(f"{k}:{v['uplift']:.3f}" for k, v in uplift.items()) +
          f" -> {'PASS' if g2_pass else 'FAIL'}", flush=True)

    # ---------- gate3 prior drift ----------
    p_before = np.bincount(y[pool], minlength=len(CLASSES)).astype(np.float64)
    p_after = p_before + np.bincount(syn_cls, minlength=len(CLASSES)).astype(np.float64)
    p_b = p_before / p_before.sum()
    p_a = p_after / p_after.sum()
    dp = p_a - p_b
    tvd = 0.5 * float(np.abs(dp).sum())
    maxdp = float(np.abs(dp).max())
    g3_pass = tvd <= args.th_tvd and maxdp <= args.th_maxdp
    rep["gate3_prior_drift"] = {
        "tvd": round(tvd, 5), "max_abs_dp": round(maxdp, 5),
        "per_class_dp": {CLASSES[i]: round(float(dp[i]), 5) for i in range(len(CLASSES))
                         if abs(dp[i]) > 1e-5},
        "thresholds": {"tvd": args.th_tvd, "max_abs_dp": args.th_maxdp}, "pass": g3_pass}
    print(f"[gate3] TVD={tvd:.5f} max|dp|={maxdp:.5f} -> {'PASS' if g3_pass else 'FAIL'}", flush=True)

    # ---------- dedup (a): T1 공여자쌍 저장공간 중심화 cos ----------
    print("[dedup] emb_v6_70k 로드/중심화...", flush=True)
    emb = np.load(EMB_PATH, mmap_mode="r")
    mu = np.zeros(emb.shape[1], np.float64)
    for b in range(0, emb.shape[0], 10000):
        mu += np.asarray(emb[b:b + 10000], np.float64).sum(0)
    mu = (mu / emb.shape[0]).astype(np.float32)

    def centered_rows(idx_list):
        E = np.asarray(emb[np.asarray(idx_list)], np.float32) - mu
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        return E

    pair_rows = [i for i, s in enumerate(syn) if len(s["_synth"]["src_ids"]) == 2]
    donor_pairs = [(id_to_row[syn[i]["_synth"]["src_ids"][0]],
                    id_to_row[syn[i]["_synth"]["src_ids"][1]]) for i in pair_rows]
    donor_cos = np.zeros(len(donor_pairs), np.float32)
    if donor_pairs:
        ai = centered_rows([p[0] for p in donor_pairs])
        bi = centered_rows([p[1] for p in donor_pairs])
        donor_cos = (ai * bi).sum(1)
    cut_a = {i for i, dcos in zip(pair_rows, donor_cos) if dcos > args.cut_donor_cos}
    rep["dedup_donor_pair_cos"] = {"dist": pct(donor_cos), "cutoff": args.cut_donor_cos,
                                   "n_reject": len(cut_a)}
    print(f"[dedup-a] 공여자쌍(T1/T3) 중심화cos {pct(donor_cos)} cut>{args.cut_donor_cos} "
          f"reject={len(cut_a)}/{len(pair_rows)}", flush=True)

    # ---------- dedup (b): 어휘 5-gram — synth vs 공여자∪공여자NN ----------
    prompt_donor = [id_to_row[s["_synth"]["src_ids"][0]] for s in syn]
    uniq_donor = sorted(set(prompt_donor))
    E_pool = centered_rows(pool)          # (n_pool, d)
    E_d = centered_rows(uniq_donor)
    K = args.nn_k
    donor_nn: dict[int, list[int]] = {}
    for b in range(0, len(uniq_donor), 256):
        S = E_d[b:b + 256] @ E_pool.T
        part = np.argpartition(-S, K + 1, axis=1)[:, :K + 1]
        for r_ in range(part.shape[0]):
            drow = uniq_donor[b + r_]
            cand = [int(pool[j]) for j in part[r_] if int(pool[j]) != drow][:K]
            donor_nn[drow] = cand
    del E_pool, E_d

    lex_cache: dict[int, dict] = {}

    def real_vec(ridx: int) -> dict:
        if ridx not in lex_cache:
            lex_cache[ridx] = char5_vec(serialize(samples[ridx], "v6", args.mht))
        return lex_cache[ridx]

    lex_max = np.zeros(len(syn), np.float32)
    for i, s in enumerate(syn):
        sv = char5_vec(syn_txt[i])
        cands = [id_to_row[x] for x in s["_synth"]["src_ids"]] + donor_nn.get(prompt_donor[i], [])
        lex_max[i] = max(cos_sparse(sv, real_vec(r_)) for r_ in cands)
    # T1(재조합)은 신규성 요구 — 낮은 컷. T2/T3(섭동, histdrop 계열)는 원본 근접이
    # 설계 그 자체 — 완전복제만 차단하는 높은 컷.
    is_t1 = np.array([s["_synth"]["template"].startswith("T1") for s in syn])
    tmpl_arr = np.array([s["_synth"]["template"][:2] for s in syn])
    cut_b = {i for i in range(len(syn))
             if lex_max[i] > (args.cut_lex_t1 if is_t1[i] else args.cut_lex_t2)}
    rep["dedup_lexical"] = {
        "per_template_dist": {t: pct(lex_max[tmpl_arr == t]) for t in sorted(set(tmpl_arr))},
        "cutoffs": {"T1": args.cut_lex_t1, "T2/T3": args.cut_lex_t2}, "n_reject": len(cut_b)}
    for t in sorted(set(tmpl_arr)):
        print(f"[dedup-b] lex5gram {t} {pct(lex_max[tmpl_arr == t])}", flush=True)
    print(f"[dedup-b] reject(T1>{args.cut_lex_t1}, T2/T3>{args.cut_lex_t2})"
          f"={len(cut_b)}/{len(syn)}", flush=True)

    # ---------- dedup (c): q4 동일공간 스팟체크 (정보성) ----------
    if args.embed_check > 0:
        try:
            mdir = args.emb_model
            if mdir.endswith(".zip"):
                tmp = tempfile.mkdtemp(prefix="m1emb_")
                with zipfile.ZipFile(mdir) as zf:
                    zf.extractall(tmp)
                mdir = tmp
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            import torch
            from transformers import AutoTokenizer
            from common import ad_lib
            torch.set_num_threads(max(os.cpu_count() - 4, 8))
            tok = AutoTokenizer.from_pretrained(mdir, local_files_only=True)
            tok.truncation_side = "left"
            model = ad_lib._load_model_maybe_quant(mdir).float().eval()
            idm_p = os.path.join(mdir, "id_map.npy")
            id_map = np.load(idm_p) if os.path.exists(idm_p) else None

            rng = np.random.default_rng(0)
            pick = rng.choice(len(syn), size=min(args.embed_check, len(syn)), replace=False)
            texts, owners = [], []
            for i in pick:
                texts.append(serialize(syn[int(i)], "v6", 8))          # 저장공간과 동일 mht=8
                texts.append(serialize(samples[prompt_donor[int(i)]], "v6", 8))
            embs = np.zeros((len(texts), model.config.hidden_size), np.float32)
            order = sorted(range(len(texts)), key=lambda k: len(texts[k]))
            with torch.no_grad():
                for b in range(0, len(order), 16):
                    idxb = order[b:b + 16]
                    enc = tok([texts[k] for k in idxb], padding=True, truncation=True,
                              max_length=320, pad_to_multiple_of=8, return_tensors="pt")
                    if id_map is not None:
                        enc["input_ids"] = torch.from_numpy(
                            id_map[enc["input_ids"].numpy()]).to(enc["input_ids"].dtype)
                    out = model.base_model(**enc).last_hidden_state
                    m_ = enc["attention_mask"].unsqueeze(-1).float()
                    mean = ((out * m_).sum(1) / m_.sum(1).clamp(min=1)).numpy()
                    for j, k in enumerate(idxb):
                        embs[k] = mean[j]
            en = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
            sd_cos = (en[0::2] * en[1::2]).sum(1)                       # synth ↔ 자기 공여자
            dn = en[1::2]
            rng2 = np.random.default_rng(1)
            ri = rng2.integers(0, len(dn), 200)
            rj = rng2.integers(0, len(dn), 200)
            keep = ri != rj
            base_cos = (dn[ri[keep]] * dn[rj[keep]]).sum(1)             # 무관쌍 기준선
            rep["dedup_q4_spotcheck"] = {
                "n": int(len(sd_cos)),
                "synth_vs_donor_rawcos": pct(sd_cos),
                "random_pair_rawcos_baseline": pct(base_cos),
                "note": "동일 q4 공간. 기대: synth-donor가 기준선보다 높되 1.0 미포화"}
            print(f"[dedup-c] q4 synth↔donor {pct(sd_cos)} / random-pair {pct(base_cos)}", flush=True)
        except Exception as e:  # 스팟체크 실패는 게이트 판정에 비영향(정보성)
            rep["dedup_q4_spotcheck"] = {"error": str(e)}
            print(f"[dedup-c] 스팟체크 실패(비치명): {e}", flush=True)

    # ---------- 종합 ----------
    cut_all = cut_a | cut_b
    reject_frac = len(cut_all) / len(syn)
    dedup_pass = reject_frac <= args.max_reject
    rep["dedup_summary"] = {"n_reject": len(cut_all), "reject_frac": round(reject_frac, 4),
                            "max_reject": args.max_reject, "pass": dedup_pass}
    overall = g0["pass"] and g1_pass and g2_pass and g3_pass and dedup_pass
    rep["overall_pass"] = overall
    rep["runtime_sec"] = round(time.time() - t_start, 1)
    print(f"[dedup] 총 reject {len(cut_all)}/{len(syn)} ({reject_frac:.1%}) -> "
          f"{'PASS' if dedup_pass else 'FAIL'}", flush=True)
    print(f"[overall] {'PASS' if overall else 'FAIL'} ({rep['runtime_sec']}s)", flush=True)

    if args.emit_pass:
        kept = [s for i, s in enumerate(syn) if i not in cut_all]
        with open(args.emit_pass, "w", encoding="utf-8") as f:
            for s in kept:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        cnt = collections.Counter(s["label"] for s in kept)
        rep["emitted_pass"] = {"path": args.emit_pass, "n": len(kept), "per_class": dict(cnt)}
        print(f"[emit] dedup 통과 {len(kept)}행 → {args.emit_pass} {dict(cnt)}", flush=True)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
        print(f"[report] {args.report}", flush=True)


if __name__ == "__main__":
    main()
