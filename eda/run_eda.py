"""Phase 1 EDA + 시뮬레이터 역설계.

산출: 콘솔 요약 + eda/eda_report.md + eda/*.csv
핵심: 구조-only '오라클 다수결 예측기'의 accuracy/macro-F1 = GBDT 상한 추정.
"""
from __future__ import annotations
import os, sys, csv, collections, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.io_utils import load_train, CLASSES, CLASS_TO_IDX
from common import parse as P

OUT = os.path.dirname(os.path.abspath(__file__))
samples, y, ids = load_train()
y = np.array(y)
N = len(samples)
lines = []  # markdown report


def w(s=""):
    print(s); lines.append(s)


def macro_f1(y_true, y_pred, n=14):
    """의존성 없이 macro-F1 직접 계산."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    f1s = []
    for c in range(n):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)), f1s


def oracle(sig_fn):
    """시그니처별 다수결 오라클 예측기의 (accuracy, macro_f1, #groups)."""
    groups = collections.defaultdict(list)
    for i, s in enumerate(samples):
        groups[sig_fn(s)].append(i)
    pred = np.empty(N, dtype=int)
    for g, idxs in groups.items():
        yy = y[idxs]
        maj = np.bincount(yy, minlength=14).argmax()
        for i in idxs:
            pred[i] = maj
    acc = float(np.mean(pred == y))
    mf1, _ = macro_f1(y, pred)
    return acc, mf1, len(groups)


# 파생 헬퍼
def turn_bucket(s):
    ti = P.meta_fields(s)["turn_index"]
    return "t1" if ti <= 1 else ("t2-3" if ti <= 3 else ("t4-7" if ti <= 7 else "t8+"))


def sig_last_action(s):
    return P.last_action(s)[0]


def sig_last_act_status(s):
    nm, _, _, st = P.last_action(s)
    return (nm, st)


def sig_struct(s):
    nm, _, _, st = P.last_action(s)
    m = P.meta_fields(s)
    return (nm, st, m["last_ci_status"], m["git_dirty"], turn_bucket(s))


def sig_struct_gen(s):
    return sig_struct(s) + (P.generator_tag(s),) if hasattr(P, "generator_tag") else sig_struct(s)


def sig_struct_rich(s):
    nm, args, _, st = P.last_action(s)
    m = P.meta_fields(s)
    ext = P.path_ext(P.arg_path_or_pattern(nm, args)) if nm else ""
    return (nm, st, m["last_ci_status"], m["git_dirty"], turn_bucket(s),
            m["user_tier"], m["language_pref"], ext)


def sig_prompt(s):
    return (s.get("current_prompt") or "").strip().lower()


def sig_struct_prompt(s):
    return sig_struct(s) + (sig_prompt(s),)


# ============================ 리포트 ============================
w("# EDA & 시뮬레이터 역설계 리포트 — Dacon 236694\n")
w(f"- 총 샘플: **{N:,}**  | 클래스: 14 | 세션(universal): "
  f"**{len(set(P.__dict__.get('session_id', lambda x:x)(i) if False else i.rsplit('-step_',1)[0] for i in ids)):,}**\n")

# 1) 클래스 분포 (전체 / sim / au)
w("## 1. 클래스 분포 (전체 / sim / au)")
from collections import Counter
gen = [("au" if i.startswith("sess_au_") else "sim") for i in ids]
c_all = Counter(samples[i]["label"] for i in range(N))
c_sim = Counter(samples[i]["label"] for i in range(N) if gen[i] == "sim")
c_au = Counter(samples[i]["label"] for i in range(N) if gen[i] == "au")
tot, ts, ta = sum(c_all.values()), sum(c_sim.values()), sum(c_au.values())
w("| class | all | all% | sim% | au% |")
w("|---|---:|---:|---:|---:|")
for cls in sorted(CLASSES, key=lambda c: -c_all[c]):
    w(f"| {cls} | {c_all[cls]} | {100*c_all[cls]/tot:.2f} | {100*c_sim[cls]/ts:.2f} | {100*c_au[cls]/ta:.2f} |")
rare = sorted(c_all.items(), key=lambda kv: kv[1])[:5]
w(f"\n- 최희귀 5: {rare}")
w(f"- 불균형 max/min = {max(c_all.values())/min(c_all.values()):.1f} (완만) | 5-fold 시 최소 클래스 fold당 ~{min(c_all.values())//5}건 → 0셀 없음\n")

# 2) 구조-only 오라클 상한 (Bayes error 프록시)
w("## 2. 구조-only 오라클 상한 (다수결 예측기 = GBDT 상한 추정)")
w("| 시그니처 | #groups | accuracy | macro-F1 |")
w("|---|---:|---:|---:|")
for name, fn in [
    ("prior(전체 다수결)", lambda s: 0),
    ("last_action", sig_last_action),
    ("last_action+status", sig_last_act_status),
    ("struct(last_act,status,ci,git,turn)", sig_struct),
    ("struct_rich(+tier,langpref,ext)", sig_struct_rich),
    ("prompt(정확일치)", sig_prompt),
    ("struct+prompt", sig_struct_prompt),
]:
    acc, mf1, ng = oracle(fn)
    w(f"| {name} | {ng} | {acc:.4f} | {mf1:.4f} |")
w("\n> struct-only가 높을수록 규칙기반 시뮬 → GBDT 강세. prompt가 크게 더 얹으면 텍스트(트랜스포머) 투자 가치.\n")

# 3) 전이행렬 P(label | last_action)
w("## 3. 전이행렬 P(다음 label | 마지막 action)")
trans = collections.defaultdict(Counter)
for i, s in enumerate(samples):
    trans[sig_last_action(s)][samples[i]["label"]] += 1
with open(os.path.join(OUT, "transition_matrix.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["last_action"] + CLASSES + ["N", "top1", "top1%"])
    for la in [None] + CLASSES:
        row = trans.get(la)
        if not row: continue
        n = sum(row.values()); top, tc = row.most_common(1)[0]
        wr.writerow([la] + [row.get(c, 0) for c in CLASSES] + [n, top, f"{100*tc/n:.1f}"])
w("- 저장: eda/transition_matrix.csv")
w("- 마지막 action별 최빈 다음 action (support≥200):")
w("\n| last_action | N | top-next | purity% | 2nd |")
w("|---|---:|---|---:|---|")
for la in [None] + CLASSES:
    row = trans.get(la)
    if not row: continue
    n = sum(row.values())
    if n < 200: continue
    mc = row.most_common(2)
    top, tc = mc[0]; second = mc[1][0] if len(mc) > 1 else "-"
    w(f"| {la} | {n} | {top} | {100*tc/n:.1f} | {second} |")

# 4) 결정규칙 스캔 (단일/쌍 조건, purity≥85%, support≥100)
w("\n## 4. 거의 결정적 규칙 (purity≥85%, support≥100)")
def scan(sig_fn, label):
    groups = collections.defaultdict(Counter)
    for i, s in enumerate(samples):
        groups[sig_fn(s)][samples[i]["label"]] += 1
    hits = []
    for g, cc in groups.items():
        n = sum(cc.values())
        if n < 100: continue
        top, tc = cc.most_common(1)[0]
        if tc / n >= 0.85:
            hits.append((tc / n, n, g, top))
    hits.sort(reverse=True)
    return hits
allhits = scan(sig_struct, "struct")[:20]
w("| purity | support | (last_act,status,ci,git,turn) | -> label |")
w("|---:|---:|---|---|")
for pur, n, g, top in allhits:
    w(f"| {pur:.2f} | {n} | {g} | {top} |")

# 5) result_status → 다음 label
w("\n## 5. 마지막 result_status → 다음 label")
st_trans = collections.defaultdict(Counter)
for i, s in enumerate(samples):
    st_trans[P.last_action(s)[3]][samples[i]["label"]] += 1
w("| status | N | top-next | % |")
w("|---|---:|---|---:|")
for st, cc in sorted(st_trans.items(), key=lambda kv: -sum(kv[1].values())):
    n = sum(cc.values()); top, tc = cc.most_common(1)[0]
    w(f"| {st} | {n} | {top} | {100*tc/n:.1f} |")

# 6) respond_only 터미널성 확인
in_hist = set()
for s in samples:
    for nm in P.action_sequence(s):
        in_hist.add(nm)
w(f"\n## 6. respond_only 특이성\n- history에 등장하는 action 종류: {len(in_hist)}/14. respond_only ∈ history: **{'respond_only' in in_hist}** (없으면 터미널 행동).")

# 7) current_prompt 언어/길이
w("\n## 7. current_prompt 언어/길이")
langs = Counter(P.lang_of(s.get("current_prompt", "")) for s in samples)
pref = Counter(P.meta_fields(s)["language_pref"] for s in samples)
w(f"- 내용기반 언어: {dict(langs)}")
w(f"- language_pref 필드: {dict(pref)}")
clen = np.array([len(s.get("current_prompt", "") or "") for s in samples])
w(f"- prompt 문자수: mean={clen.mean():.0f} p50={np.percentile(clen,50):.0f} p90={np.percentile(clen,90):.0f} p99={np.percentile(clen,99):.0f} max={clen.max()}")

# 8) 직렬화 길이(근사) → max_len 근거 : 대략적인 char 길이
def serialize_len(s):
    m = P.meta_fields(s)
    parts = [m["user_tier"], m["language_pref"], m["last_ci_status"], str(m["loc"])]
    for t in s.get("history", [])[-8:]:
        if t.get("role") == "user":
            parts.append(t.get("content", "")[:200])
        else:
            parts.append(f"{t.get('name')} {t.get('result_summary','')[:120]}")
    parts.append(s.get("current_prompt", ""))
    return len(" ".join(parts))
slen = np.array([serialize_len(s) for s in samples])
w(f"\n## 8. 직렬화(전체 history) 근사 문자길이: mean={slen.mean():.0f} p50={np.percentile(slen,50):.0f} p90={np.percentile(slen,90):.0f} p99={np.percentile(slen,99):.0f} max={slen.max()}")
w("> 서브워드 토큰 ≈ 문자/2~3 (ko/en 혼재). max_len 256~384 검토 근거.")

# 9) 메타 ↔ 라벨 (last_ci_status, git_dirty, turn_index)
w("\n## 9. 메타 신호 ↔ 라벨 (top-next per value)")
for field, fn in [("last_ci_status", lambda s: P.meta_fields(s)["last_ci_status"]),
                  ("git_dirty", lambda s: P.meta_fields(s)["git_dirty"]),
                  ("user_tier", lambda s: P.meta_fields(s)["user_tier"]),
                  ("turn_bucket", turn_bucket)]:
    cc = collections.defaultdict(Counter)
    for i, s in enumerate(samples):
        cc[fn(s)][samples[i]["label"]] += 1
    w(f"\n**{field}**: " + " | ".join(
        f"{k}→{v.most_common(1)[0][0]}({100*v.most_common(1)[0][1]/sum(v.values()):.0f}%,n={sum(v.values())})"
        for k, v in sorted(cc.items(), key=lambda x: -sum(x[1].values()))))

with open(os.path.join(OUT, "eda_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n[saved] eda/eda_report.md , eda/transition_matrix.csv")
