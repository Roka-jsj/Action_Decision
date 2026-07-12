#!/usr/bin/env python
"""R74 검증 — (A) AWP/EMA/R-Drop 로직 CPU 스모크(5스텝, 랜덤 텐서, 크래시·복원 확인)
                (B) train_full_cli.py 소스레벨 default-off 회귀검사(AD_*=0 → 기존경로 불변).

GPU 불필요. 실행: PYTHONPATH=/root/Action_Decision python3 sim/test_train_techniques.py
"""
import os, sys, re
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sim.train_techniques import AWP, EMA, rdrop_kl   # noqa: E402

torch.manual_seed(0)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(20, 8)
        self.fc1 = nn.Linear(8, 16)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(16, 5)

    def forward(self, x):
        h = self.emb(x).mean(1)
        h = torch.relu(self.fc1(h))
        return self.fc2(self.drop(h))


def _batch():
    x = torch.randint(0, 20, (12, 6))
    y = torch.randint(0, 5, (12,))
    return x, y


def test_awp():
    model = Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    awp = AWP(model, adv_lr=1.0, adv_eps=0.01)
    x, y = _batch()
    for step in range(5):
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        snap = {n: p.detach().clone() for n, p in model.named_parameters()}
        awp.perturb()
        # 최소 한 2D weight 는 실제로 교란됐어야 함
        changed = [n for n, p in model.named_parameters() if not torch.equal(p.detach(), snap[n])]
        assert changed, f"step{step}: AWP 가 아무 가중치도 교란 안 함"
        # 교란은 eps-공 안(|Δw| ≤ eps·|w| + 미세여유)이어야, 그리고 bias(1D)는 불변
        for n, p in model.named_parameters():
            if p.dim() > 1 and "weight" in n:
                d = (p.detach() - snap[n]).abs()
                assert (d <= 0.01 * snap[n].abs() + 1e-6).all(), f"step{step}:{n} eps-공 위반"
            else:
                assert torch.equal(p.detach(), snap[n]), f"step{step}:{n} 1D/bias 는 불변이어야"
        # 적대 backward(누적) 후 복원
        F.cross_entropy(model(x), y).backward()
        awp.restore()
        for n, p in model.named_parameters():
            assert torch.equal(p.detach(), snap[n]), f"step{step}:{n} 복원 실패"
        assert not awp.backup, "restore 후 backup 비어야"
        opt.step()
    print("  [ok] AWP: 5스텝 교란·eps-공·bias불변·정확복원")


def test_ema():
    model = Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-1)
    ema = EMA(model, decay=0.9)
    init_shadow = {k: v.clone() for k, v in ema.shadow.items()}
    x, y = _batch()
    for _ in range(5):
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
        ema.update(model)
    assert any(not torch.equal(ema.shadow[k], init_shadow[k]) for k in ema.shadow), "shadow 미갱신"
    # shadow 는 raw 와 달라야(수렴궤적 평균)
    assert any(not torch.allclose(ema.shadow[n], p.detach().float())
               for n, p in model.named_parameters() if n in ema.shadow), "shadow==raw (평균 무의미)"
    pre = {n: p.detach().clone() for n, p in model.named_parameters()}
    ema.apply_shadow(model)
    for n, p in model.named_parameters():
        if n in ema.shadow:
            assert torch.allclose(p.detach().float(), ema.shadow[n], atol=1e-6), f"{n} shadow 미적용"
    ema.restore(model)
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), pre[n]), f"{n} EMA restore 실패"
    print("  [ok] EMA: shadow 갱신·raw와 상이·apply/restore 왕복 정확")


def test_rdrop():
    l1 = torch.randn(4, 5)
    assert abs(rdrop_kl(l1, l1.clone(), 1.0).item()) < 1e-6, "동일 로짓 KL≠0"
    v = rdrop_kl(l1, torch.randn(4, 5), 1.0)
    assert torch.isfinite(v) and v.item() > 0, "상이 로짓 KL 비양수/비유한"
    # 실제 dropout 이중패스 — 유한·양수
    model = Tiny(); model.train()
    x, _ = _batch()
    k = rdrop_kl(model(x), model(x), 0.5)
    assert torch.isfinite(k), "dropout 이중패스 KL 비유한"
    print("  [ok] R-Drop: 동일→0·상이→양수·dropout 이중패스 유한")


def test_source_default_off():
    """train_full_cli.py 소스가 AD_*=0 에서 기존경로와 byte-동일함을 정적 보증."""
    src = open(os.path.join(ROOT, "action_decision_maximum/src/train_full_cli.py"), encoding="utf-8").read()
    checks = {
        "AD_AWP default off": 'os.environ.get("AD_AWP", "0") == "1"' in src,
        "AD_EMA default off": 'os.environ.get("AD_EMA", "0") == "1"' in src,
        "AD_RDROP default off": 'os.environ.get("AD_RDROP", "0") == "1"' in src,
        "AWP off→None guard": re.search(r"awp\s*=\s*AWP\(.*\)\s*if\s*AWP_ON\s*else\s*None", src) is not None,
        "EMA off→None guard": re.search(r"ema\s*=\s*EMA\(.*\)\s*if\s*EMA_ON\s*else\s*None", src) is not None,
        "loop AWP guarded": "if awp is not None and ep >= AWP_START_EP" in src,
        "loop EMA guarded": "if ema is not None:" in src,
        "FGM adv pass rdrop=False": "aloss = total_loss(enc, lb, bi, rdrop=False)" in src,
        "rdrop param defaults None": "def total_loss(enc, lb, bi, rdrop=None):" in src,
        "rdrop off→orig branch": "use_rd = RDROP_ON if rdrop is None else rdrop" in src,
        # RDROP off 시 실행되는 원본 단일패스 3줄이 그대로 보존
        "orig single-forward preserved": "    logits = model(**enc).logits\n    loss = ce_loss(logits, lb, bi)" in src,
        "EMA/SWA 상호배제": 'raise ValueError("AD_EMA 와 AD_SWA_K 동시 사용 불가' in src,
    }
    bad = [k for k, v in checks.items() if not v]
    assert not bad, "default-off 회귀검사 실패: " + "; ".join(bad)
    print(f"  [ok] 소스 default-off 회귀검사 {len(checks)}/{len(checks)}항 통과")


if __name__ == "__main__":
    print("R74 검증 시작 (CPU)")
    test_awp()
    test_ema()
    test_rdrop()
    test_source_default_off()
    print("=== ALL PASS ===")
