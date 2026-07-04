"""템플릿 역설계 프로브 — 0.8+ 승부수 검증.

가설: 시뮬레이터가 유한 템플릿으로 prompt 생성 → 변수슬롯(경로/식별자/따옴표/숫자)을
플레이스홀더로 치환하면 train↔test 템플릿 매칭 가능.

정직한 측정(세션 단위 5-fold): val 샘플의 정규화 prompt가 train측에 존재하는 비율(coverage)과
매칭 시 다수결 라벨 정확도(purity), 그리고 (커버리지×순도) 기여 상한.
컨텍스트 결합 변형: 정규화prompt + 직전action(+status) 키도 비교.
"""
from __future__ import annotations
import os, sys, re, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.io_utils import load_train
from common.cv import make_splits
from common import ad_lib

samples, y, ids = load_train()
y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups)
folds = sp["folds"]; dev_idx = sp["dev_idx"]

_QUOTE = re.compile(r"(['\"`])(?:(?!\1).)*\1")
_PATH = re.compile(r"[\w.\-/]+/[\w.\-/]*|\b[\w\-]+\.(?:py|ts|tsx|js|jsx|rs|go|java|rb|css|html|json|yml|yaml|toml|md|sh|sql|txt|cfg|ini|lock|xml|proto|dockerfile)\b", re.I)
_CAMEL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b|\b[a-z0-9]+(?:_[a-z0-9]+)+\b|\b[A-Z]{2,}[A-Z0-9_]*\b")
_NUM = re.compile(r"\d+")
_WS = re.compile(r"\s+")


def normalize(p: str) -> str:
    t = p or ""
    t = _QUOTE.sub("<Q>", t)
    t = _PATH.sub("<P>", t)
    t = _CAMEL.sub("<ID>", t)
    t = _NUM.sub("<N>", t)
    t = _WS.sub(" ", t).strip().lower()
    return t


def probe(key_fn, name):
    """세션단위 fold로 정직 측정: coverage, matched-accuracy, 기대 기여."""
    covs, accs = [], []
    for tr, va in folds:
        table = collections.defaultdict(collections.Counter)
        for i in tr:
            table[key_fn(samples[i])][y[i]] += 1
        n_match = n_correct = 0
        for i in va:
            k = key_fn(samples[i])
            if k in table:
                n_match += 1
                if table[k].most_common(1)[0][0] == y[i]:
                    n_correct += 1
        covs.append(n_match / len(va))
        accs.append(n_correct / max(n_match, 1))
    cov, acc = float(np.mean(covs)), float(np.mean(accs))
    print(f"{name:34} coverage={cov:.3f}  matched-acc={acc:.3f}  (cov×acc={cov*acc:.3f})")
    return cov, acc


print(f"folds={len(folds)}  dev={len(dev_idx)}")
print("\n=== 키 변형별 정직 측정 (val의 키가 train측에 존재?) ===")
probe(lambda s: (s.get("current_prompt") or "").strip().lower(), "raw prompt (lower)")
probe(lambda s: normalize(s.get("current_prompt")), "normalized prompt")
probe(lambda s: (normalize(s.get("current_prompt")), ad_lib.last_action(s)[0]), "norm prompt + last_action")
probe(lambda s: (normalize(s.get("current_prompt")), ad_lib.last_action(s)[0], ad_lib.last_action(s)[3]), "norm prompt + last_act + status")

# 템플릿 다양성 지표
norms = [normalize(s.get("current_prompt")) for s in samples]
cnt = collections.Counter(norms)
sizes = np.array(sorted(cnt.values(), reverse=True))
print(f"\n정규화 후 고유 템플릿 수: {len(cnt):,} (원본 70,000)")
print(f"상위 10 템플릿 크기: {sizes[:10].tolist()}")
print(f"2회 이상 등장 템플릿이 커버하는 샘플: {sizes[sizes>=2].sum():,} ({100*sizes[sizes>=2].sum()/len(samples):.1f}%)")

# 순도 분포 (전체 train 기준, 참고용)
pur = []
tbl = collections.defaultdict(collections.Counter)
for i, k in enumerate(norms):
    tbl[k][y[i]] += 1
for k, c in tbl.items():
    n = sum(c.values())
    if n >= 3:
        pur.append(c.most_common(1)[0][1] / n)
pur = np.array(pur)
print(f"\n(참고) n>=3 템플릿 순도: mean={pur.mean():.3f}  >=0.9 비율={float((pur>=0.9).mean()):.3f}")
