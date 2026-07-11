#!/usr/bin/env python3
"""provenance preflight — 인수 환경이 원본 파이프라인을 재현하는지 판정 게이트.

배경(codex R63c 반증): 대조군 m1h8full을 '한 번만 완주'하면 δ/τ가 분리돼 m1t3 제출판단이
된다는 결론은, 재발진한 대조군이 **원본과 동일한 데이터·토크나이저·모델·런타임**에서 나왔을
때에만 성립한다. 새 HF 캐시/라이브러리/토크나이저로 재학습하면 측정된 τ가 T3 효과가 아니라
재현 실패일 수 있다(대조군만 다른 분포). 이 스크립트는 eval_tau_delta / 대조군 재발진 **전에**
그 재현성을 강제검증하고, 불일치면 중단(exit 1)시킨다.

레퍼런스: work/provenance_ref.json (원본으로 믿어지는 컨테이너에서 캡처 — uid1000 산출물·
sklearn1.8.0 지뢰환경 일치). 이 파일과 대조.

단계(각 PASS/FAIL/SKIP):
  1 ENV      런타임 버전(transformers·numpy=하드, tokenizers=하드, torch=경고, sklearn=경고)
  2 ANCHOR   splits sha·grid 해시·autopsy sha·멤버 토크나이저 지문 (전부 하드)
  3 DATA     data/train.jsonl + IDS_SHA16 일치 + 14클래스 커버 (data/ 부재면 SKIP)
  4 HFBASE   xlm-roberta-large HF 캐시 (대조군 '재학습'에만 필요; 부재면 SKIP-blocked)
  5 REPRO    ★핵심★ th85/m1(=autopsy p_old 출처)을 autopsy 5k행에 재추론→p_old와 대조.
             원본 추론 파이프라인 재현 실증(추론은 HF 베이스 불필요). data/ 부재면 SKIP.

종료코드: 0=GREEN(재현 신뢰·판정 진행 가능) / 2=BLOCKED(입력 미복구, FAIL 없음) / 1=RED(재현 오염).
사용: python3 sim/preflight_provenance.py && python3 sim/eval_tau_delta.py
"""
import os, sys, json, hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
REF_PATH = os.path.join(ROOT, "work", "provenance_ref.json")

# 재현대조 허용오차 (int8 양자화 잡음 상한 — handoff §9-2 max 0.0045 참조, 보수 0.01)
REPRO_ATOL = 0.01
REPRO_ARGMAX_MIN = 0.999   # argmax 일치율 하한
N_REPRO = 512              # 재추론 표본(전체 5k 대신 앞 N행 — 속도)

C = {"P": "\033[0m", "G": "", "R": "", "Y": ""}  # 색 없이(--color never 관행)

results = []  # (phase, name, status, detail)  status ∈ PASS/FAIL/SKIP


def rec(phase, name, status, detail=""):
    results.append((phase, name, status, detail))
    tag = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[status]
    print(f"  [{tag}] {phase}/{name}  {detail}")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha16_file(p):
    return sha256_file(p)[:16]


def main():
    if not os.path.exists(REF_PATH):
        print(f"[치명] 레퍼런스 없음: {REF_PATH} — 원본 환경에서 먼저 생성해야 함")
        return 1
    ref = json.load(open(REF_PATH))
    print(f"provenance preflight — ref={os.path.relpath(REF_PATH, ROOT)}")

    # ---------- 1. ENV ----------
    print("\n[1] ENV — 런타임 버전")
    rv = ref["versions"]
    try:
        import numpy, transformers, tokenizers
        cur = {"transformers": transformers.__version__, "numpy": numpy.__version__,
               "tokenizers": tokenizers.__version__}
        for k in ("transformers", "numpy", "tokenizers"):  # 하드(requirements 핀 + 산출물 정합)
            rec("ENV", k, "PASS" if cur[k] == rv[k] else "FAIL",
                f"{cur[k]} (기대 {rv[k]})")
        try:
            import torch
            tv = torch.__version__
            same_mm = tv.split("+")[0].rsplit(".", 1)[0] == rv["torch"].split("+")[0].rsplit(".", 1)[0]
            rec("ENV", "torch", "PASS" if same_mm else "FAIL", f"{tv} (기대 {rv['torch']})")
            cu = torch.version.cuda
            rec("ENV", "cuda", "PASS" if cu == rv.get("cuda") else "SKIP", f"{cu} (기대 {rv.get('cuda')})")
        except Exception as e:
            rec("ENV", "torch", "SKIP", f"임포트 실패: {e}")
        try:
            import sklearn
            # sklearn은 splits READ에 영향 없음(파일 고정). make_splits 재생성만 위험 → 경고만.
            rec("ENV", "sklearn", "PASS" if sklearn.__version__ == rv["sklearn"] else "SKIP",
                f"{sklearn.__version__} (기대 {rv['sklearn']}; splits 재생성 절대금지)")
        except Exception as e:
            rec("ENV", "sklearn", "SKIP", str(e))
    except Exception as e:
        rec("ENV", "imports", "FAIL", str(e))

    # ---------- 2. ANCHOR ----------
    print("\n[2] ANCHOR — 정적 앵커(하드)")
    a = ref["anchors"]
    sp = os.path.join(ROOT, "splits", "splits.npz")
    if os.path.exists(sp):
        got = sha256_file(sp)
        rec("ANCHOR", "splits.npz", "PASS" if got == a["splits_sha256"] else "FAIL",
            got[:16] + f" (기대 {a['splits_sha256'][:16]})")
    else:
        rec("ANCHOR", "splits.npz", "FAIL", "부재 — git checkout splits/splits.npz")
    aut = os.path.join(ROOT, "work", "autopsy_m1t3_5k.npz")
    if os.path.exists(aut):
        got = sha256_file(aut)
        rec("ANCHOR", "autopsy", "PASS" if got == a["autopsy_sha256"] else "FAIL", got[:16])
    else:
        rec("ANCHOR", "autopsy", "FAIL", "부재 — 게이트C 부검 산출물 미이전")
    try:
        from sim import refit_lib as L
        for fn, key in (("grid_hash", "grid_hash"), ("rescue_grid_hash", "rescue_grid_hash")):
            got = getattr(L, fn)()
            rec("ANCHOR", fn, "PASS" if got == a[key] else "FAIL", f"{got} (기대 {a[key]})")
    except Exception as e:
        rec("ANCHOR", "grid_hash", "FAIL", str(e))
    # 멤버 토크나이저 지문 (추론경로 정합의 핵심)
    for name, m in ref["member_tok"].items():
        d = os.path.join(ROOT, m["dir"])
        if not os.path.isdir(d):
            rec("ANCHOR", f"tok:{name}", "SKIP", f"{m['dir']} 부재")
            continue
        bad = [fn for fn, h in m["files"].items()
               if not os.path.exists(os.path.join(d, fn)) or sha16_file(os.path.join(d, fn)) != h]
        rec("ANCHOR", f"tok:{name}", "PASS" if not bad else "FAIL",
            "일치" if not bad else f"불일치/부재: {bad}")

    # ---------- 3. DATA ----------
    print("\n[3] DATA — 원본 데이터 정합")
    train_p = os.path.join(ROOT, "data", "train.jsonl")
    if not os.path.exists(train_p):
        rec("DATA", "train.jsonl", "SKIP", "부재 — 운영자 docker cp 필요(마스터 차단)")
    else:
        try:
            from sim import refit_lib as L
            ids, y, groups, ids_hash = L.load_ids_labels()  # 내부에서 IDS_SHA16 assert + 14클래스 assert
            rec("DATA", "IDS_SHA16", "PASS", f"{len(ids)}행 로드·해시 {ids_hash} 일치·14클래스 커버")
        except AssertionError as e:
            rec("DATA", "IDS_SHA16", "FAIL", f"데이터 정합 실패: {e}")
        except Exception as e:
            rec("DATA", "load", "FAIL", str(e))

    # ---------- 4. HFBASE ----------
    print("\n[4] HFBASE — xlm-roberta-large 캐시(대조군 재학습 전용)")
    hits = []
    for base in (os.path.expanduser("~/.cache/huggingface/hub"),
                 os.environ.get("HF_HOME", ""), os.path.join(ROOT, ".cache")):
        if base and os.path.isdir(base):
            for d in os.listdir(base):
                if "xlm-roberta-large" in d and os.path.isdir(os.path.join(base, d)):
                    hits.append(os.path.join(base, d))
    if hits:
        rec("HFBASE", "xlm-roberta-large", "PASS", hits[0])
    else:
        rec("HFBASE", "xlm-roberta-large", "SKIP",
            "캐시 부재 — 재학습만 차단(추론·재현대조는 불필요). 정확한 revision snapshot 복구 필요")

    # ---------- 5. REPRO (핵심) ----------
    print("\n[5] REPRO — 원본 추론 파이프라인 재현대조(th85/m1 → autopsy p_old)")
    if not os.path.exists(train_p):
        rec("REPRO", "p_old", "SKIP", "data/ 부재 — 복구 후 자동 활성")
    elif not os.path.exists(aut):
        rec("REPRO", "p_old", "SKIP", "autopsy 부재")
    else:
        m1dir = os.path.join(ROOT, "packages", "submit_th85", "model", "m1")
        if not os.path.isdir(m1dir):
            rec("REPRO", "p_old", "SKIP", "th85/m1 부재")
        else:
            try:
                import numpy as np
                from common import ad_lib
                from common.io_utils import load_train
                d = np.load(aut)
                rows = d["rows"][:N_REPRO]
                p_old = d["p_old"][:N_REPRO]
                samples, _, _ = load_train()
                sub = [samples[i] for i in rows]
                tx8 = [ad_lib.serialize(s, "v6", 8) for s in sub]
                p_rep = ad_lib.predict_logits(m1dir, sub, version="v6", max_len=320,
                                              batch_size=128, texts=tx8,
                                              return_probs=True, gen_rescue=True)
                mad = float(np.abs(p_rep - p_old).max())
                agree = float((p_rep.argmax(1) == p_old.argmax(1)).mean())
                ok = mad <= REPRO_ATOL and agree >= REPRO_ARGMAX_MIN
                rec("REPRO", "p_old", "PASS" if ok else "FAIL",
                    f"max|Δ|={mad:.5f}(≤{REPRO_ATOL}) argmax일치={agree:.4f}(≥{REPRO_ARGMAX_MIN}) N={len(rows)}")
            except Exception as e:
                rec("REPRO", "p_old", "FAIL", f"재추론 실패: {e}")

    # ---------- 판정 ----------
    print("\n" + "=" * 60)
    fails = [r for r in results if r[2] == "FAIL"]
    skips = [r for r in results if r[2] == "SKIP"]
    blocking_skips = [r for r in skips if r[0] in ("DATA", "REPRO")
                      or (r[0] == "HFBASE")]
    if fails:
        print("판정: RED — 환경이 원본을 재현하지 못함. 재학습·판정 금지.")
        for p, n, s, det in fails:
            print(f"  ✗ {p}/{n}: {det}")
        print("→ 원본과 동일 데이터/토크나이저/모델/런타임을 복구할 것. 불일치 항목 해소 전 eval_tau_delta 무효.")
        return 1
    data_missing = any(r[0] == "DATA" and r[2] == "SKIP" for r in results)
    repro_ran = any(r[0] == "REPRO" and r[2] in ("PASS", "FAIL") for r in results)
    hf_missing = any(r[0] == "HFBASE" and r[2] == "SKIP" for r in results)
    if data_missing or not repro_ran or hf_missing:
        print("판정: BLOCKED — FAIL은 없으나 입력 미복구. 아래 해소 후 재실행:")
        if data_missing:
            print("  · data/ (train.jsonl 등) 복구 → DATA·REPRO 활성")
        if hf_missing:
            print("  · xlm-roberta-large HF snapshot(정확 revision) 복구 → 대조군 재학습 가능")
        print("  (추론·재현대조는 data/만으로 가능 — HF는 재학습에만 필요)")
        return 2
    print("판정: GREEN — 정적 앵커·데이터·재현대조 전부 통과. eval_tau_delta 신뢰 가능.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
