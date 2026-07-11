#!/usr/bin/env python3
"""합성 증강 생성기 (문샷, codex R57b + 전략가 R65 §5 설계) — CPU 전용.

원칙: "이미 확인된 신호를 더 많은 표본으로 강화"만 한다(신규 정보 주입 금지).
생성 = 실데이터 재조합/섭동 (LLM 없음, 오프라인):
  T1 histswap : 같은 (클래스, gen, 마지막액션명, 결과상태, turn_bin) 버킷 안에서
                프롬프트 공여자 a 의 meta+current_prompt 에 히스토리 공여자 b 의
                history 를 결합. 라벨 결정 신호(프롬프트·마지막 액션·상태)는 보존,
                문맥(히스토리 본문·경로·요약)만 다양화. 앵커포라(again/다시 등)
                프롬프트는 히스토리 결합 부적합으로 제외.
  T2 tailcrop : histdrop(R24, LB 검증 선례)와 동일한 섭동 패밀리를 타겟 클래스에만
                적용 — history 절단(최근2-4턴/랜덤 prefix) + 필드 드롭아웃
                (result_summary 12%, args 10%, 마지막 액션턴 면제).
  T3 metaswap : 히스토리 없는(none|na) 버킷 전용 — 같은 버킷의 두 행에서 a 의
                프롬프트+히스토리에 b 의 session_meta 를 결합(메타-라벨 허위상관
                탈착). 실데이터 none-share(list 41.6%!)를 클래스별로 정확 재현 —
                batch0 실측에서 none 행 부재가 js_last 왜곡(null 대비 9배)의 주범.

오염 차단 (교리):
  - 소스 행은 fold0-train(tr0)만. fold0-val·holdout 유래 행 사용 시 assert 사망.
  - splits 는 refit_lib.load_splits() 로만 로드(재생성 금지 교리 준수).
  - synth id 는 신규 세션 네임스페이스(sess_{gen}_syn*) — 실세션 그룹과 절대 불교차,
    세션-스텝 누출(tb/416939) 원천 차단. gen(au/sim)은 공여자 것을 보존([GEN] 태그 정합).

출력: JSONL. 각 행 = 표준 샘플 스키마(id, session_meta, history, current_prompt)
  + "label"(문자열) + "_synth"(provenance: template, src_ids, bucket, seed).
  주입 시 "label"/"_synth" 는 pop 하고 사용(설계: work/moonshot_plan.md §4).

사용:
  PYTHONPATH=/root/Action_Decision python3 sim/synth_gen.py \
      --n 500 --tag b0 --seed 20260711 --out work/synth_batch0.jsonl
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from common.io_utils import load_train, CLASSES, CLASS_TO_IDX  # noqa: E402
from common import parse as P  # noqa: E402
from common.serialize import serialize, _turn_bin  # noqa: E402
import refit_lib  # noqa: E402

# 기본 타겟: fold0-val per-class F1 최약 + 단일 혼동 클러스터(내비게이션 4중주).
# mdebr_f0(캐리어 기준선 0.7640) 실측: list 0.480(P 0.364) / read 0.577 / grep 0.621 / glob 0.682.
# 상위 혼동 5셀 전부 이 4중주 내부 (read->list 428, grep->read 370, grep->list 356, ...).
DEFAULT_TARGETS = "list_directory,read_file,grep_search,glob_pattern"
DEFAULT_SHARES = "0.35,0.30,0.20,0.15"

# 프롬프트가 자기 히스토리에 강결합(앵커포라)된 행은 T1 프롬프트 공여자에서 제외.
_ANAPHORA = re.compile(
    r"\b(again|same|retry|re-?run|once more|as before|like before|previous(ly)?|"
    r"that (file|test|dir|folder|one|patch|diff)|those|the above|earlier)\b"
    r"|다시|재시도|재실행|같은\s|아까|방금|위에서|이전(에|처럼)?",
    re.IGNORECASE,
)


def bucket_key(sample: dict, y_cls: int) -> tuple:
    nm, _args, _rs, st = P.last_action(sample)
    m = P.meta_fields(sample)
    return (y_cls, sample["gen"], nm or "none", st, _turn_bin(m["turn_index"]))


def valid_v6(sample: dict) -> bool:
    """v6 직렬화 유효성: 예외 없이 생성되고 [CUR] 포함 (mht 8/12 양쪽)."""
    try:
        for mht in (8, 12):
            t = serialize(sample, "v6", mht)
            if "[CUR] " not in t:
                return False
    except Exception:
        return False
    return True


def make_id(gen: str, tag: str, seq: int, step: int) -> str:
    # generator()/session_id()/step_num() 파싱 정합: sess_{gen}_syn{tag}{seq}-step_{NN}
    return f"sess_{gen}_syn{tag}{seq:05d}-step_{max(step, 0):02d}"


def t1_histswap(a: dict, b: dict, gen_id: str) -> dict:
    s = {
        "id": gen_id,
        "session_meta": copy.deepcopy(a.get("session_meta") or {}),
        "history": copy.deepcopy(b.get("history") or []),
        "current_prompt": a.get("current_prompt") or "",
    }
    return s


def t2_tailcrop(a: dict, gen_id: str, r: random.Random) -> dict:
    """histdrop(R24, train_full_cli.py:69-104) 섭동 패밀리의 클래스 타겟판.

    원본 histdrop 대비 2개 수정(게이트1 실측 반영 — batch0 에서 list js_last 가
    null 대비 7배 이탈):
      - 빈 히스토리 모드 제거(클래스-조건 [NOHIST]/last-action 분포 왜곡 방지)
      - 마지막 assistant_action 턴은 필드 드롭아웃 면제(라벨 결정 신호 보존)
    """
    h = list(a.get("history") or [])
    u = r.random()
    if u < 0.375:
        nh = h[-r.choice([2, 3, 4]):]
    else:
        k = r.randint(1, max(1, len(h) - 1))
        nh = h[-k:]
    last_act = max((j for j, t in enumerate(nh) if t.get("role") == "assistant_action"),
                   default=-1)
    out = []
    for j, t in enumerate(nh):
        if t.get("role") == "assistant_action" and j != last_act:
            t = dict(t)
            if r.random() < 0.12:
                t["result_summary"] = ""
            if r.random() < 0.10:
                t["args"] = {}
        out.append(t)
    return {
        "id": gen_id,
        "session_meta": copy.deepcopy(a.get("session_meta") or {}),
        "history": out,
        "current_prompt": a.get("current_prompt") or "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="목표 생성 행수")
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--shares", default=DEFAULT_SHARES)
    ap.add_argument("--t1-frac", type=float, default=0.7, help="T1 histswap 비중(잔여=T2)")
    ap.add_argument("--seed", type=int, default=20260711)
    ap.add_argument("--tag", default="b0", help="id 네임스페이스 태그")
    ap.add_argument("--out", default=os.path.join(ROOT, "work", "synth_batch0.jsonl"))
    ap.add_argument("--source", choices=["fold0train"], default="fold0train",
                    help="소스 풀. fold0 게이트 사이클은 tr0 고정(va0/holdout 무접촉).")
    args = ap.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    shares = [float(x) for x in args.shares.split(",")]
    assert len(targets) == len(shares) and abs(sum(shares) - 1.0) < 1e-6, "targets/shares 불일치"
    for t in targets:
        assert t in CLASS_TO_IDX, f"미지 클래스 {t}"
    tgt_idx = [CLASS_TO_IDX[t] for t in targets]

    r = random.Random(args.seed)

    print("[load] train + splits (refit_lib.load_splits — 재생성 금지 교리)", flush=True)
    samples, y, ids = load_train()
    folds, dev_idx, hold_idx = refit_lib.load_splits()
    tr0, va0 = folds[0]
    va0_set = set(int(i) for i in va0)
    hold_set = set(int(i) for i in hold_idx)
    pool = [int(i) for i in tr0]
    pool_set = set(pool)
    assert not (pool_set & va0_set), "소스풀-va0 교차 — 게이트 계기 오염"
    assert not (pool_set & hold_set), "소스풀-holdout 교차 — 계기 오염"
    print(f"[pool] fold0-train {len(pool)}행 (va0 {len(va0_set)} / holdout {len(hold_set)} 무접촉)", flush=True)

    # 버킷 구축 (타겟 클래스 × 소스풀만)
    buckets: dict[tuple, list[int]] = {}
    per_cls_pool: dict[int, list[int]] = {c: [] for c in tgt_idx}
    for i in pool:
        c = y[i]
        if c not in per_cls_pool:
            continue
        per_cls_pool[c].append(i)
        buckets.setdefault(bucket_key(samples[i], c), []).append(i)

    anaph_cache: dict[int, bool] = {}

    def anaphoric(i: int) -> bool:
        if i not in anaph_cache:
            anaph_cache[i] = bool(_ANAPHORA.search(samples[i].get("current_prompt") or ""))
        return anaph_cache[i]

    # 클래스별 생성 목표
    n_per_cls = {c: int(round(args.n * s)) for c, s in zip(tgt_idx, shares)}
    # 반올림 잔차 보정
    diff = args.n - sum(n_per_cls.values())
    n_per_cls[tgt_idx[0]] += diff

    out_rows = []
    seq = 0
    stats = {"T1": 0, "T2": 0, "T3": 0, "anaphora_skipped": 0, "invalid_v6": 0,
             "bucket_exhausted": 0}
    used_pairs: set[tuple] = set()

    for c, share in zip(tgt_idx, shares):
        goal = n_per_cls[c]
        # 실데이터 none-share(히스토리 무액션 행 비율)를 클래스별 정확 재현
        none_share = sum(1 for i in per_cls_pool[c]
                         if P.last_action(samples[i])[0] is None) / max(len(per_cls_pool[c]), 1)
        n_t3 = int(round(goal * none_share))
        n_rest = goal - n_t3
        n_t1 = int(round(n_rest * args.t1_frac))
        n_t2 = n_rest - n_t1
        act_buckets = [(k, v) for k, v in buckets.items()
                       if k[0] == c and k[2] != "none" and len(v) >= 2]
        none_buckets = [(k, v) for k, v in buckets.items()
                        if k[0] == c and k[2] == "none" and len(v) >= 2]
        assert act_buckets, f"클래스 {CLASSES[c]} 스왑 가능 액션버킷 없음"
        bw_act = [len(v) for _, v in act_buckets]
        bw_none = [len(v) for _, v in none_buckets]

        def recombine(bucket_list, bweights, n_goal, allow_metaswap):
            made, attempts = 0, 0
            while made < n_goal and attempts < n_goal * 60:
                attempts += 1
                bi = r.choices(range(len(bucket_list)), weights=bweights, k=1)[0]
                key, rows = bucket_list[bi]
                a_i, b_i = r.sample(rows, 2)
                if anaphoric(a_i):
                    stats["anaphora_skipped"] += 1
                    continue
                pair = (a_i, b_i)
                if pair in used_pairs or a_i == b_i:
                    continue
                used_pairs.add(pair)
                gen = samples[a_i]["gen"]
                sid = make_id(gen, args.tag, seq_ref[0], samples[a_i]["step"])
                hb = samples[b_i].get("history") or []
                ha = samples[a_i].get("history") or []
                if hb and hb != ha:
                    s = t1_histswap(samples[a_i], samples[b_i], sid)
                    tmpl = "T1_histswap"
                elif allow_metaswap:
                    # 양쪽 다 (사실상) 빈 히스토리 — b 의 메타 + a 의 프롬프트/히스토리
                    s = {"id": sid,
                         "session_meta": copy.deepcopy(samples[b_i].get("session_meta") or {}),
                         "history": copy.deepcopy(ha),
                         "current_prompt": samples[a_i].get("current_prompt") or ""}
                    tmpl = "T3_metaswap"
                else:
                    continue
                if not valid_v6(s):
                    stats["invalid_v6"] += 1
                    continue
                s["label"] = CLASSES[c]
                s["_synth"] = {"template": tmpl, "src_ids": [ids[a_i], ids[b_i]],
                               "bucket": "|".join(map(str, key)), "seed": args.seed}
                out_rows.append(s)
                seq_ref[0] += 1
                made += 1
                stats["T3" if tmpl == "T3_metaswap" else "T1"] += 1
            return made

        seq_ref = [seq]
        made_t3 = recombine(none_buckets, bw_none, n_t3, True) if none_buckets else 0
        made_t1 = recombine(act_buckets, bw_act, n_t1 + (n_t3 - made_t3), False)
        seq = seq_ref[0]
        made = made_t1 + made_t3
        if made < n_t1 + n_t3:
            stats["bucket_exhausted"] += n_t1 + n_t3 - made
            n_t2 += n_t1 + n_t3 - made  # 부족분 T2 로 이월

        # --- T2 tailcrop ---
        cand = [i for i in per_cls_pool[c] if len(P.action_turns(samples[i])) >= 2]
        made2, attempts = 0, 0
        import collections as _coll
        used_src: _coll.Counter = _coll.Counter()
        while made2 < n_t2 and attempts < n_t2 * 60:
            attempts += 1
            a_i = r.choice(cand)
            # 동일 소스 반복 허용은 2회까지(다양성)
            if used_src[a_i] >= 2:
                continue
            gen = samples[a_i]["gen"]
            sid = make_id(gen, args.tag, seq, samples[a_i]["step"])
            s = t2_tailcrop(samples[a_i], sid, r)
            # 절단이 실제 일어났는지(원본과 동일 히스토리면 무의미 복제)
            if len(s["history"]) >= len(samples[a_i].get("history") or []):
                continue
            if not valid_v6(s):
                stats["invalid_v6"] += 1
                continue
            s["label"] = CLASSES[c]
            s["_synth"] = {"template": "T2_tailcrop", "src_ids": [ids[a_i]],
                           "bucket": "", "seed": args.seed}
            out_rows.append(s)
            used_src[a_i] += 1
            seq += 1
            made2 += 1

        stats["T2"] += made2
        print(f"[gen] {CLASSES[c]:18s} T1={made_t1} T3={made_t3}(none목표 {n_t3}) "
              f"T2={made2} (목표 {goal})", flush=True)

    # ---- 최종 오염/유효성 assert ----
    id_to_row = {v: k for k, v in enumerate(ids)}
    all_ids = set()
    for s in out_rows:
        assert s["id"] not in all_ids, "synth id 중복"
        all_ids.add(s["id"])
        assert s["id"] not in id_to_row, "synth id 가 실데이터 id 와 충돌"
        for src in s["_synth"]["src_ids"]:
            ridx = id_to_row[src]
            assert ridx in pool_set, f"소스 {src} 가 fold0-train 밖 — 오염"
            assert ridx not in va0_set and ridx not in hold_set, f"소스 {src} 계기 오염"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in out_rows:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    import collections
    lab_cnt = collections.Counter(s["label"] for s in out_rows)
    gen_cnt = collections.Counter(s["id"].split("_")[1] for s in out_rows)
    print(f"[done] {len(out_rows)}행 → {args.out}")
    print(f"  per-class: {dict(lab_cnt)}")
    print(f"  per-gen  : {dict(gen_cnt)}  stats={stats}")


if __name__ == "__main__":
    main()
