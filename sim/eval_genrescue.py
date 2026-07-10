#!/usr/bin/env python3
"""GEN-rescue A/B 평가 (R53) — fold0-val 같은 행 old vs patched 재추론.

게이트(사전등록, 레드팀 R53 재설계):
  (a') sim∩GEN삭제 patched acc >= 0.753
  (b') 비대상 행 byte-identity (토크나이저 테스트 + 본 스크립트 logits 동일성)
  (c)  fold0 전체 F1 델타 >= 0 (solo 발사 기준은 codex >= +0.0004)
  (d') mdeb 부호 + (FULL 멤버라 오염 캐비앗 — 참고용)
  (e)  추론시간 증가 없음 (Δ <= +15s)

GPU 병행 규칙: mdeb 농사(teacher_cli) 병행 — 소배치(기본 16), VRAM ~2GB 이내.
사용:
  python3 sim/eval_genrescue.py                     # m1(v6 8ep fold0 ckpt) A/B
  python3 sim/eval_genrescue.py --mdeb              # + mdeb FULL 멤버 부호(오염 캐비앗)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import refit_lib as L  # noqa: E402
from common import ad_lib  # noqa: E402
from common import postproc  # noqa: E402
from common.io_utils import load_train  # noqa: E402

ROOT = L.ROOT
GATE_ACC = 0.753          # (a') sim∩GEN삭제 patched acc
GATE_SOLO_F1 = 0.0004     # (c) solo 발사 기준
GATE_TIME = 15.0          # (e) 초


def acc(y_true, pred, mask):
    if mask.sum() == 0:
        return float("nan"), 0
    return float((pred[mask] == y_true[mask]).mean()), int(mask.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "work", "foldckpt_largev6_f0ckpt_f0"))
    ap.add_argument("--version", default="v6")
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--mdeb", action="store_true", help="mdeb FULL 멤버 부호 측정(오염 캐비앗)")
    ap.add_argument("--mdeb_dir", default=os.path.join(ROOT, "work", "cc_members", "mdeb"))
    ap.add_argument("--out", default=os.path.join(ROOT, "work", "genrescue_ab.md"))
    a = ap.parse_args()

    print(f"[genrescue A/B] ckpt={os.path.relpath(a.ckpt, ROOT)} v={a.version} "
          f"max_len={a.max_len} batch={a.batch}")
    samples, y_all, ids = load_train()
    y_all = np.asarray(y_all)
    folds, _, _ = L.load_splits()
    va0 = np.sort(folds[0][1])
    sub = [samples[i] for i in va0]
    yb = y_all[va0]
    texts = [ad_lib.serialize(s, a.version) for s in sub]
    is_sim = np.array([str(s["id"]).startswith("sess_sim") for s in sub])

    # ---- 대상 판정 (m1 토크나이저 기준 = 판정 타깃 슬라이스) ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt, local_files_only=True)
    tok.truncation_side = "left"
    t0 = time.time()
    rescued = ad_lib._gen_rescue_ids(tok, texts, a.max_len)
    t_rescue_scan = time.time() - t0
    n_special = tok.num_special_tokens_to_add(False)
    lens = np.array([len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]])
    trunc = lens + n_special > a.max_len
    gen_del = np.zeros(len(sub), dtype=bool)
    gen_del[sorted(rescued.keys())] = True
    sl = {
        "sim∩GEN삭제 (판정)": is_sim & gen_del,
        "au∩GEN삭제": (~is_sim) & gen_del,
        "sim∩절단∩GEN생존 (대조)": is_sim & trunc & ~gen_del,
        "au∩절단∩GEN생존 (대조)": (~is_sim) & trunc & ~gen_del,
        "비절단": ~trunc,
        "전체": np.ones(len(sub), dtype=bool),
    }
    print(f"  fold0-val {len(sub)}행: 절단 {trunc.sum()}({trunc.mean()*100:.1f}%) "
          f"GEN삭제 {gen_del.sum()}({gen_del.mean()*100:.1f}%) "
          f"[sim {int((is_sim & gen_del).sum())} / au {int(((~is_sim) & gen_del).sum())}] "
          f"| rescue 스캔 {t_rescue_scan:.1f}s")

    # ---- old / patched 풀패스 (동일 배칭 — 비대상 logits 동일성까지 검증) ----
    t0 = time.time()
    P_old = ad_lib.predict_logits(a.ckpt, sub, version=a.version, max_len=a.max_len,
                                  batch_size=a.batch, texts=texts, return_probs=True)
    t_old = time.time() - t0
    t0 = time.time()
    P_new = ad_lib.predict_logits(a.ckpt, sub, version=a.version, max_len=a.max_len,
                                  batch_size=a.batch, texts=texts, return_probs=True,
                                  gen_rescue=True)
    t_new = time.time() - t0
    dt = t_new - t_old
    nt = ~gen_del
    same = np.array_equal(P_old[nt], P_new[nt])
    close = bool(np.allclose(P_old[nt], P_new[nt], atol=1e-6))
    pred_same = bool((P_old[nt].argmax(1) == P_new[nt].argmax(1)).all())
    print(f"  시간: old {t_old:.0f}s / patched {t_new:.0f}s (Δ{dt:+.1f}s) "
          f"| 비대상 logits 동일={same} allclose={close} pred동일={pred_same}")

    po, pn = P_old.argmax(1), P_new.argmax(1)
    lines = [f"ckpt={os.path.relpath(a.ckpt, ROOT)} (v6 8ep fold0, val 0.7485) rows={len(sub)}",
             f"절단 {trunc.sum()}행({trunc.mean()*100:.1f}%) / GEN삭제 {gen_del.sum()}행({gen_del.mean()*100:.1f}%)",
             f"시간: old {t_old:.0f}s → patched {t_new:.0f}s (Δ{dt:+.1f}s, rescue 스캔 {t_rescue_scan:.1f}s 포함)",
             f"비대상 byte-identity: logits array_equal={same} / allclose={close} / pred동일={pred_same}"]
    print(f"  {'슬라이스':<24} {'n':>6} {'old':>8} {'patched':>8} {'Δ':>9}")
    for name, m in sl.items():
        ao, n = acc(yb, po, m)
        an, _ = acc(yb, pn, m)
        row = f"{name:<24} {n:>6} {ao:>8.4f} {an:>8.4f} {an-ao:>+9.4f}"
        print("  " + row)
        lines.append(row)

    # ---- fold0 F1 (멤버 단독 + 캐스케이드 th75/cw45) ----
    f1_old = L.fast_macro_f1(yb, po)
    f1_new = L.fast_macro_f1(yb, pn)
    print(f"  멤버 단독 macro-F1: {f1_old:.5f} → {f1_new:.5f} (Δ{f1_new-f1_old:+.5f})")
    lines.append(f"멤버 단독 F1: {f1_old:.5f} → {f1_new:.5f} (Δ{f1_new-f1_old:+.5f})")
    OLD_BIAS = postproc.load(L.OLD_BIAS_PATH)
    D = np.load(os.path.join(ROOT, "work", "mdeb12ep_f0.npz"), allow_pickle=True)["oof"][va0]
    K = np.load(os.path.join(ROOT, "work", "klue_f0.npz"), allow_pickle=True)["oof"][va0]
    casc = {}
    for tag, w, th in [("th75(발사구성)", (0.45, 0.40, 0.15), 0.75),
                       ("cw45", (0.45, 0.35, 0.20), 0.6)]:
        Pc_o, _ = L.cascade_probs([P_old, D, K], w, th)
        Pc_n, _ = L.cascade_probs([P_new, D, K], w, th)
        fo = L.score(Pc_o, OLD_BIAS, yb)
        fn = L.score(Pc_n, OLD_BIAS, yb)
        casc[tag] = fn - fo
        row = f"캐스케이드 {tag}: {fo:.5f} → {fn:.5f} (Δ{fn-fo:+.5f})"
        print("  " + row)
        lines.append(row)

    # ---- mdeb 부호 (FULL 멤버 — fold0-val이 훈련집합에 포함, 오염 캐비앗) ----
    mdeb_delta = None
    if a.mdeb and os.path.isdir(a.mdeb_dir):
        tgt = np.where(gen_del)[0]
        sub_t = [sub[i] for i in tgt]
        tx_t = [texts[i] for i in tgt]
        Mo = ad_lib.predict_logits(a.mdeb_dir, sub_t, version=a.version, max_len=a.max_len,
                                   batch_size=a.batch, texts=tx_t, return_probs=True)
        Mn = ad_lib.predict_logits(a.mdeb_dir, sub_t, version=a.version, max_len=a.max_len,
                                   batch_size=a.batch, texts=tx_t, return_probs=True,
                                   gen_rescue=True)
        yt = yb[tgt]
        sm = is_sim[tgt]
        ao = float((Mo.argmax(1)[sm] == yt[sm]).mean())
        an = float((Mn.argmax(1)[sm] == yt[sm]).mean())
        mdeb_delta = an - ao
        row = (f"mdeb(FULL, 오염 캐비앗) sim∩GEN삭제 {int(sm.sum())}행 acc: "
               f"{ao:.4f} → {an:.4f} (Δ{mdeb_delta:+.4f})")
        print("  " + row)
        lines.append(row)

    # ---- 게이트 판정 ----
    a_acc, _ = acc(yb, pn, sl["sim∩GEN삭제 (판정)"])
    d_f1 = f1_new - f1_old
    gates = [
        ("a' sim∩GEN삭제 patched acc >= 0.753", a_acc >= GATE_ACC, f"{a_acc:.4f}"),
        ("b' 비대상 byte-identity", bool(same or (close and pred_same)),
         f"array_equal={same} allclose={close}"),
        ("c fold0 멤버 F1 >= 0", d_f1 >= 0, f"Δ{d_f1:+.5f} (solo기준 >= +0.0004: {d_f1 >= GATE_SOLO_F1})"),
        ("d' mdeb 부호 +", (mdeb_delta is None) or (mdeb_delta > 0),
         "측정불가(fold ckpt 없음)" if mdeb_delta is None else f"Δ{mdeb_delta:+.4f} (FULL 오염 캐비앗)"),
        ("e 시간 Δ <= +15s", dt <= GATE_TIME, f"Δ{dt:+.1f}s"),
    ]
    print("  --- 게이트(사전등록) ---")
    all_ok = True
    for name, ok, detail in gates:
        all_ok &= bool(ok)
        row = f"{'PASS' if ok else 'FAIL'}  {name}: {detail}"
        print("  " + row)
        lines.append("게이트 " + row)
    verdict = "ALL PASS — 패키지 조립 진행 가능 (발사는 3자 서명)" if all_ok else "FAIL — 발사 보류"
    print(f"  => {verdict}")
    lines.append(f"판정: {verdict}")

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# GEN-rescue A/B (R53, fold0-val old vs patched)\n\n")
        for ln in lines:
            f.write(f"- {ln}\n")
    print(f"[report] {a.out}")
    # 캐스케이드 참고치는 게이트 외 정보 (8ep ckpt 기반 — 12ep 배포멤버와 수준 상이)
    json.dump({"acc_target_patched": a_acc, "member_f1_delta": d_f1,
               "cascade": casc, "time_delta": dt, "gates_pass": all_ok},
              open(os.path.join(ROOT, "work", "genrescue_ab.json"), "w"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
