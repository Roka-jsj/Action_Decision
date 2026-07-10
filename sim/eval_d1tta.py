#!/usr/bin/env python3
"""D1 CompressView-TTA 게이트 실측 (codex R55) — fold0-val, 8ep f0 ckpt.

TTA = GEN삭제∩(게이트마진<0.5) 행만 압축 재직렬화(u:60/rs:30/12아이템/턴경계 ≤320)
2패스 후 λ=0.5 확률 평균. 게이트(사전등록):
  ① fold0 sim∩GEN삭제 슬라이스 Δacc >= +0.004
  ② 전략가 probe_r54_out.npz 교차부호 (독립 구현 대조 + 12ep OOF 스플라이스 방향, 프록시 캐비앗)
  ③ 비대상 byte-identity
  ④ 캐스케이드 클린(th75 좌표) 비음수
  ⑤ 총 시간 추정 <= 525s (LB 앵커: genrescue 495s)

한계: mdeb/klue f0 ckpt 부재 — 계기판은 m1 채널만(배포는 3멤버 TTA, 과소추정 방향).
GPU 소배치(농사 병행). 사용: python3 sim/eval_d1tta.py
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import refit_lib as L  # noqa: E402
from common import ad_lib, postproc  # noqa: E402
from common.io_utils import load_train  # noqa: E402

ROOT = L.ROOT
SCRATCH = "/tmp/claude-0/-root-Action-Decision/4ad02a7b-69d3-4625-b907-25b364030498/scratchpad"
CKPT = os.path.join(ROOT, "work", "foldckpt_largev6_f0ckpt_f0")
LAM, TTH = 0.5, 0.5
GATE_SLICE = +0.004
GATE_TIME = 525.0
LB_ANCHOR_S = 495.0     # genrescue 실측
# LB 행당 추정단가(초/행): 495s 분해(r_L(1+0.37)+0.3r_L*0.37 ≈ (495-60)/30000) — 문서화된 가정
R_L, R_D = 0.0102, 0.0031


def main():
    samples, y_all, _ = load_train()
    y_all = np.asarray(y_all)
    folds, _, _ = L.load_splits()
    va0 = np.sort(folds[0][1])
    sub = [samples[i] for i in va0]
    yb = y_all[va0]
    texts = [ad_lib.serialize(s, "v6") for s in sub]
    is_sim = np.array([str(s["id"]).startswith("sess_sim") for s in sub])
    OLD = postproc.load(L.OLD_BIAS_PATH)

    # 기준선: m1 rescue OOF (배포 등가 m1) + 구입력 D/K
    M1r = np.load(os.path.join(ROOT, "work", "m1_f0ckpt_rescue.npz"), allow_pickle=True)["oof"][va0]
    D12 = np.load(os.path.join(ROOT, "work", "mdeb12ep_f0.npz"), allow_pickle=True)["oof"][va0]
    K = np.load(os.path.join(ROOT, "work", "klue_f0.npz"), allow_pickle=True)["oof"][va0]

    # 대상 산정: 게이트마진(m1 rescue) < 0.5 ∧ GEN삭제(m1 토크나이저)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CKPT, local_files_only=True)
    tok.truncation_side = "left"
    t0 = time.time()
    gen_del_map = ad_lib._gen_rescue_ids(tok, texts, 320)
    t_scan = time.time() - t0
    gen_del = np.zeros(len(sub), bool)
    gen_del[sorted(gen_del_map.keys())] = True
    srt = np.sort(M1r, axis=1)
    marg = srt[:, -1] - srt[:, -2]
    tta_mask = gen_del & (marg < TTH)
    rows_tta = np.where(tta_mask)[0]
    f_tta = len(rows_tta) / len(sub)
    print(f"[d1tta] fold0-val {len(sub)}행 | GEN삭제 {gen_del.sum()} | 대상(∩marg<{TTH}) "
          f"{len(rows_tta)} ({f_tta*100:.1f}%) | 스캔 {t_scan:.1f}s")

    # 압축 직렬화 (이분탐색) + m1 TTA 패스
    n_sp = tok.num_special_tokens_to_add(False)
    t0 = time.time()
    comp = [ad_lib.serialize_compress(sub[i], tok, 320 - n_sp) for i in rows_tta]
    t_comp = time.time() - t0
    t0 = time.time()
    Pt = ad_lib.predict_logits(CKPT, [sub[i] for i in rows_tta], version="v6", max_len=320,
                               batch_size=16, texts=comp, return_probs=True, gen_rescue=True)
    t_inf = time.time() - t0
    print(f"  압축 {t_comp:.1f}s ({t_comp/max(len(rows_tta),1)*1000:.1f}ms/행) | m1 TTA 추론 {t_inf:.1f}s")

    M1b = M1r.copy()
    M1b[rows_tta] = (1 - LAM) * M1r[rows_tta] + LAM * Pt

    # ③ 비대상 byte-identity
    nt = ~tta_mask
    b_ident = bool(np.array_equal(M1b[nt], M1r[nt]))

    # ① 슬라이스
    def acc(P, m):
        return float((P.argmax(1)[m] == yb[m]).mean())

    sl_sim = is_sim & gen_del
    a_base = acc(M1r, sl_sim)
    a_tta = acc(M1b, sl_sim)
    d_slice = a_tta - a_base
    # 상세: 대상 교집합/대조군
    rows = [("sim∩GEN삭제 (게이트)", sl_sim), ("sim∩대상(∩marg<0.5)", is_sim & tta_mask),
            ("au∩GEN삭제", (~is_sim) & gen_del), ("비대상 전체", nt)]
    print(f"  {'슬라이스':<22} {'n':>6} {'rescue':>8} {'+TTA':>8} {'Δ':>9}")
    lines = []
    for name, m in rows:
        r = f"{name:<22} {int(m.sum()):>6} {acc(M1r, m):>8.4f} {acc(M1b, m):>8.4f} {acc(M1b, m)-acc(M1r, m):>+9.4f}"
        print("  " + r)
        lines.append(r)

    # ② 전략가 probe 교차 (독립 구현 대조 + 12ep 스플라이스 방향[세대혼합 프록시])
    x_lines, x_agree, d12_sign = [], None, None
    px = os.path.join(SCRATCH, "probe_r54_out.npz")
    if os.path.exists(px):
        d = np.load(px, allow_pickle=True)
        tgt_idx = np.asarray(d["tgt_idx"])
        V1 = np.asarray(d["tgt_V1"], np.float32)
        V5 = np.asarray(d["tgt_V5"], np.float32)
        common = [j for j, r in enumerate(tgt_idx) if tta_mask[r]]
        mine_pos = {int(r): k for k, r in enumerate(rows_tta)}
        mp = np.array([mine_pos[int(tgt_idx[j])] for j in common])
        x_agree = float((Pt[mp].argmax(1) == V5[common].argmax(1)).mean())
        x_lines.append(f"probe V5 vs 본구현 압축뷰 argmax 일치 {x_agree:.4f} (n={len(common)})")
        # 12ep OOF 스플라이스 방향 (V1/V5는 8ep 산출 — 세대혼합 프록시, 방향 참고만)
        M12 = np.load(os.path.join(ROOT, "work", "teacher_largev6_12ep_f0.npz"),
                      allow_pickle=True)["oof"][va0]
        s12 = np.sort(M12, axis=1)
        m12 = (s12[:, -1] - s12[:, -2] < TTH)
        spl = [j for j, r in enumerate(tgt_idx) if m12[r]]
        Ma, Mb_ = M12.copy(), M12.copy()
        Ma[tgt_idx[spl]] = V1[spl]
        Mb_[tgt_idx[spl]] = (1 - LAM) * V1[spl] + LAM * V5[spl]
        Pa, _ = L.cascade_probs([Ma, D12, K], L.RESCUE_ANCHOR_W, L.RESCUE_ANCHOR_TH)
        Pb, _ = L.cascade_probs([Mb_, D12, K], L.RESCUE_ANCHOR_W, L.RESCUE_ANCHOR_TH)
        d12 = L.score(Pb, OLD, yb) - L.score(Pa, OLD, yb)
        d12_sign = d12 > 0
        x_lines.append(f"12ep OOF 스플라이스 캐스케이드 방향(프록시): Δ{d12:+.5f}")
    for r in x_lines:
        print("  ② " + r)

    # ④ 캐스케이드 클린 (th75 좌표, m1만 TTA — 과소추정 방향)
    Pc_r, _ = L.cascade_probs([M1r, D12, K], L.RESCUE_ANCHOR_W, L.RESCUE_ANCHOR_TH)
    Pc_b, _ = L.cascade_probs([M1b, D12, K], L.RESCUE_ANCHOR_W, L.RESCUE_ANCHOR_TH)
    f_r = L.score(Pc_r, OLD, yb)
    f_b = L.score(Pc_b, OLD, yb)
    d_casc = f_b - f_r
    print(f"  ④ 캐스케이드 th75: {f_r:.5f} → {f_b:.5f} (Δ{d_casc:+.5f})")

    # ⑤ 시간 추정 (30k, LB 앵커 495s + 추가분)
    # 스캔: gen_rescue 패키지는 m1 full 패스의 rescue 스캔을 재사용(ad_lib rescue_rows_out) → 0s
    # 추론: TTA 행은 ~320토큰 만장행 — 평균행 단가 대비 ×1.3 보정. 토크나이저 로드 +2s.
    n30 = 30000
    LEN_CORR = 1.3
    t_comp30 = (t_comp / max(len(rows_tta), 1)) * (f_tta * n30)
    est = {}
    for tag, rates in [("3멤버(m1+mdeb+klue)", (R_L + R_D + R_L)),
                       ("2멤버(m1+mdeb, klue 제외 옵션)", (R_L + R_D))]:
        t_inf30 = f_tta * n30 * rates * LEN_CORR
        est[tag] = LB_ANCHOR_S + 2.0 + t_comp30 + t_inf30
        print(f"  ⑤ 시간추정[{tag}]: 앵커 {LB_ANCHOR_S:.0f}s + tok 2s + 압축 {t_comp30:.0f}s "
              f"+ TTA추론 {t_inf30:.0f}s = {est[tag]:.0f}s (한도 {GATE_TIME:.0f}s, 하드 570s)")
    t_total = est["3멤버(m1+mdeb+klue)"]

    gates = [
        ("① sim∩GEN삭제 Δacc >= +0.004", d_slice >= GATE_SLICE, f"{d_slice:+.4f}"),
        ("② probe 교차부호", (x_agree or 0) > 0.9 and (d12_sign is not False),
         f"일치 {x_agree} / 12ep방향 {'+' if d12_sign else ('측정불가' if d12_sign is None else '-')}"),
        ("③ 비대상 byte-identity", b_ident, str(b_ident)),
        ("④ 캐스케이드 클린 비음수", d_casc >= 0, f"Δ{d_casc:+.5f}"),
        ("⑤ 총시간 <= 525s", t_total <= GATE_TIME, f"{t_total:.0f}s"),
    ]
    all_ok = True
    print("  --- 게이트(사전등록) ---")
    out_lines = lines + x_lines
    for name, ok, detail in gates:
        all_ok &= bool(ok)
        r = f"{'PASS' if ok else 'FAIL'}  {name}: {detail}"
        print("  " + r)
        out_lines.append("게이트 " + r)
    verdict = "ALL PASS — submit_d1tta 조립 가능 (발사는 3자)" if all_ok else "FAIL — 발사 보류"
    print(f"  => {verdict}")
    out_lines.append(f"판정: {verdict}")
    out_lines.append("한계: mdeb/klue f0 ckpt 부재 — m1 채널만 반영(배포는 3멤버 TTA, 과소추정 방향)")

    with open(os.path.join(ROOT, "work", "d1tta_gate.md"), "w", encoding="utf-8") as f:
        f.write("# D1 CompressView-TTA 게이트 (R55, fold0-val)\n\n")
        for ln in out_lines:
            f.write(f"- {ln}\n")
    json.dump({"slice_delta": d_slice, "cascade_delta": d_casc, "time_est": t_total,
               "rows_tta_frac": f_tta, "pass": all_ok},
              open(os.path.join(ROOT, "work", "d1tta_gate.json"), "w"))
    print(f"[report] work/d1tta_gate.md")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
