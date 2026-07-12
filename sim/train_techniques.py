#!/usr/bin/env python
"""R74 opt-in 학습기법 업그레이드 — train_full_cli.py 에 배선(전부 default-off).

배포 조원 대비 우리 from-scratch 재현이 -0.008~-0.018 solo. 지금까지 시도한 레버는
FGM(입력 임베딩 교란, +0.010 solo) 와 SWA(해악) 둘뿐. 이 모듈은 아직 한 번도 당기지
않은 세 개의 표준 레버를 추가한다. 전부 순수 로직(import 부작용 없음) — 대응하는 AD_*
플래그가 켜질 때만 인스턴스화되므로 AD_AWP=AD_EMA=AD_RDROP=0 이면 기존 루프와 byte-동일.

  AWP       : Adversarial Weight Perturbation (Wu et al. 2020). 가중치를 교란(FGM의 입력
              교란보다 강). 트랜스포머 분류에서 통상 +0.002~0.005. FGM 위에 합성 가능.
  EMA       : 가중치 지수이동평균. SWA(미수렴 에폭 평균 → 이 프로젝트서 해악)와 달리
              수렴 궤적을 추종. 통상 +0.001~0.003. eval/save 에 shadow 사용.
  rdrop_kl  : R-Drop(Liang et al. 2021). 두 드롭아웃 패스 출력분포 간 대칭 KL 일관성항.
              분류에서 통상 +0.002~0.004.

설계상 합성 원칙(FGM 과 조합):
  normal.backward() → [FGM: perturb emb, backward(누적), restore]
                    → [AWP: perturb weight, backward(누적), restore]
                    → optimizer.step() → [EMA.update()]
  AWP 는 기존 grad(정상[+FGM])을 zero 하지 않고 적대 grad 를 그 위에 누적한다. 이는
  이 저장소의 FGM 관습(교란 지점 grad 를 정상 grad 에 더하고 복원)과 동일한 철학이며
  fp16 GradScaler 하에서 안전하다(교란 방향이 grad-정규화 → loss-scale 불변).
"""
import torch
import torch.nn.functional as F


class AWP:
    """Adversarial Weight Perturbation.

    perturb() 는 loss.backward() 직후(누적 grad 존재) 호출. 매칭된 2D 가중치행렬을
    grad 상승방향으로 Frobenius-정규화 스텝만큼 이동 후 per-element L∞ eps-공
    (|Δw| ≤ adv_eps·|w|) 로 사영한다. 교란량이 grad-정규화라 loss-scale 불변 →
    scaler.unscale_ 없이 fp16 안전.

    adv_lr    : 가중치 상승 스텝(상대, ~1.0). eps-clamp 를 포화시키는 역할.
    adv_eps   : per-weight 최대 상대 교란(~0.01, 범위 [0.005,0.1]).
    adv_param : 교란 대상 파라미터 이름 부분문자열('weight' → 모든 weight 행렬).
                dim>1 필터로 bias/LayerNorm(1D)은 제외(표준 AWP 관행).
    """

    def __init__(self, model, adv_lr=1.0, adv_eps=0.01, adv_param="weight"):
        self.model = model
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.adv_param = adv_param
        self.backup = {}

    def _match(self, name, param):
        return (param.requires_grad and param.grad is not None
                and self.adv_param in name and param.dim() > 1)

    @torch.no_grad()
    def perturb(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if not self._match(name, param):
                continue
            gnorm = torch.norm(param.grad)
            if gnorm == 0 or not torch.isfinite(gnorm):
                continue                      # grad 0/inf/nan → 이 텐서 건너뜀(fp16 오버플로 보호)
            clean = param.data.clone()
            self.backup[name] = clean
            wnorm = torch.norm(param.data.detach())
            param.data.add_(self.adv_lr * param.grad / (gnorm + e) * (wnorm + e))
            lo = clean - self.adv_eps * clean.abs()
            hi = clean + self.adv_eps * clean.abs()
            param.data = torch.min(torch.max(param.data, lo), hi)

    @torch.no_grad()
    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class EMA:
    """부동소수 가중치(파라미터+float 버퍼)의 지수이동평균.

    shadow 는 모델 device 에 fp32 로 상주. optimizer.step() 마다 update().
    apply_shadow()/restore() 로 eval/save 시 EMA 가중치 교체. 정수 버퍼
    (position_ids 등)는 건드리지 않는다(is_floating_point 필터).
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            sh = self.shadow.get(k)
            if sh is not None:
                sh.mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def apply_shadow(self, model):
        msd = model.state_dict()
        self.backup = {}
        for k, sh in self.shadow.items():
            self.backup[k] = msd[k].detach().clone()
            msd[k].copy_(sh.to(msd[k].dtype))

    @torch.no_grad()
    def restore(self, model):
        msd = model.state_dict()
        for k, b in self.backup.items():
            msd[k].copy_(b)
        self.backup = {}


def rdrop_kl(logits1, logits2, alpha):
    """두 드롭아웃 패스 로짓 간 대칭 KL 일관성항(R-Drop). fp32 로 계산(autocast 안전)."""
    p = F.log_softmax(logits1.float(), dim=-1)
    q = F.log_softmax(logits2.float(), dim=-1)
    kl_pq = F.kl_div(p, q, log_target=True, reduction="batchmean")
    kl_qp = F.kl_div(q, p, log_target=True, reduction="batchmean")
    return alpha * 0.5 * (kl_pq + kl_qp)
