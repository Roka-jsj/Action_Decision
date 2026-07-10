#!/usr/bin/env python3
"""rescue OOF 재생성 (R54 T5-light) — fold ckpt로 fold-val을 gen_rescue 입력으로 재추론.

genrescue 배포(LB 0.78985) 후 저마진 행 구성이 변해 구입력 OOF 기반 그리드는 낡음.
이 스크립트는 보존된 fold ckpt 로 old(구입력)/rescue(헤더보존) 두 패스를 같은 추론경로
(ad_lib.predict_logits)로 재생성해 로더 호환 npz 로 저장한다 — 제출용 아닌 계기판.

주의: folds1-4 ckpt 미보존(save_w=False) — fold0 만 가능. mdeb12/klue f0 ckpt 는
전 컨테이너 수색 결과 부재(2026-07-10 실측) → m1(8ep f0ckpt)만 재생성 가능.
GPU 소배치(기본 16, 농사 병행 규율).

사용:
  python3 sim/gen_rescue_oof.py                          # m1 기본 (old+rescue 둘 다)
  python3 sim/gen_rescue_oof.py --ckpt <dir> --prefix work/mdeb12ep_f0  # ckpt 확보 시
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import refit_lib as L  # noqa: E402
from common import ad_lib  # noqa: E402
from common.io_utils import load_train  # noqa: E402

ROOT = L.ROOT
SCRATCH = "/tmp/claude-0/-root-Action-Decision/4ad02a7b-69d3-4625-b907-25b364030498/scratchpad"


def save_npz(path, oof, rows, y, fold, model, version, max_len):
    f1 = L.fast_macro_f1(y[rows], oof[rows].argmax(1))
    np.savez_compressed(path, oof=oof.astype(np.float32),
                        hold=np.zeros((5810, L.NUM_CLASSES), np.float32),
                        scores=np.array([f1]), fold_lo=fold, fold_hi=fold + 1,
                        model=model, version=version, max_len=max_len)
    print(f"  [저장] {os.path.relpath(path, ROOT)} covered-F1={f1:.5f}")
    return f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "work", "foldckpt_largev6_f0ckpt_f0"))
    ap.add_argument("--prefix", default=os.path.join(ROOT, "work", "m1_f0ckpt"),
                    help="산출 접두 — {prefix}_old.npz / {prefix}_rescue.npz")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--version", default="v6")
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--xcheck", action="store_true",
                    help="레드팀 probe_r54_out.npz(tgt_V1)와 교차검증 (m1 fold0 전용)")
    a = ap.parse_args()
    assert os.path.isdir(a.ckpt), f"ckpt 없음: {a.ckpt}"

    samples, y_all, _ = load_train()
    y_all = np.asarray(y_all)
    folds, _, _ = L.load_splits()
    va = np.sort(folds[a.fold][1])
    sub = [samples[i] for i in va]
    texts = [ad_lib.serialize(s, a.version) for s in sub]
    print(f"[gen_rescue_oof] ckpt={os.path.relpath(a.ckpt, ROOT)} fold{a.fold} rows={len(va)} "
          f"batch={a.batch}")

    t0 = time.time()
    P_old = ad_lib.predict_logits(a.ckpt, sub, version=a.version, max_len=a.max_len,
                                  batch_size=a.batch, texts=texts, return_probs=True)
    t_old = time.time() - t0
    t0 = time.time()
    P_new = ad_lib.predict_logits(a.ckpt, sub, version=a.version, max_len=a.max_len,
                                  batch_size=a.batch, texts=texts, return_probs=True,
                                  gen_rescue=True)
    t_new = time.time() - t0
    print(f"  old {t_old:.0f}s / rescue {t_new:.0f}s (Δ{t_new-t_old:+.1f}s)")

    oof_o = np.zeros((len(samples), L.NUM_CLASSES), np.float32)
    oof_r = np.zeros((len(samples), L.NUM_CLASSES), np.float32)
    oof_o[va] = P_old
    oof_r[va] = P_new
    f_old = save_npz(f"{a.prefix}_old.npz", oof_o, va, y_all, a.fold, a.ckpt, a.version, a.max_len)
    f_new = save_npz(f"{a.prefix}_rescue.npz", oof_r, va, y_all, a.fold, a.ckpt, a.version, a.max_len)
    print(f"  멤버 F1: old {f_old:.5f} → rescue {f_new:.5f} (Δ{f_new-f_old:+.5f})")

    # 변경 행 = rescue 대상만인지 확인 (byte-identity 시험의 실전판)
    changed = np.where(np.abs(P_old - P_new).max(1) > 0)[0]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt, local_files_only=True)
    tok.truncation_side = "left"
    rescued = ad_lib._gen_rescue_ids(tok, texts, a.max_len)
    tgt = np.array(sorted(rescued.keys()))
    ok = set(changed.tolist()) <= set(tgt.tolist())
    print(f"  변경행 {len(changed)} ⊆ rescue대상 {len(tgt)}: {ok}")
    assert ok, "비대상 행 변경 감지 — byte-identity 위반"

    if a.xcheck:
        px = os.path.join(SCRATCH, "probe_r54_out.npz")
        if os.path.exists(px):
            d = np.load(px, allow_pickle=True)
            tgt_idx, V1 = np.asarray(d["tgt_idx"]), np.asarray(d["tgt_V1"], dtype=np.float32)
            mine = P_new[tgt_idx]
            mad = float(np.abs(mine - V1).max())
            agree = float((mine.argmax(1) == V1.argmax(1)).mean())
            print(f"  [xcheck 레드팀 tgt_V1] n={len(tgt_idx)} max|Δ|={mad:.4f}(fp16 저장) "
                  f"argmax 일치={agree:.4f}")
        else:
            print("  [xcheck] probe_r54_out.npz 없음 — 생략")
    print("[gen_rescue_oof] 완료")


if __name__ == "__main__":
    main()
