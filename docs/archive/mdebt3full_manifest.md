# mdeb-T3 FULL 발진 manifest — R64 3자 서명(운영자+codex#3+레드팀), 발진 전 동결

작성: 2026-07-11 10:35 KST. **이 manifest는 발진 전 커밋으로 동결한다(레드팀 조건①). 결과를 보고 하이퍼파라미터 변경·재발진 금지(codex#3 B).**

## 발진 명령 (레드팀 검증: train_full_cli env 스키마 + mdebr_f0 프로브 헤더 대조)
```
cd /root/Action_Decision && env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
 PYTHONPATH=/root/Action_Decision AD_WORK=/root/Action_Decision/work \
 AD_MODEL=microsoft/mdeberta-v3-base AD_VERSION=v6 AD_MAXLEN=320 AD_EPOCHS=10 \
 AD_LR=2e-5 AD_BATCH=64 AD_LLRD=1 AD_FGM=1 AD_SEED=1234 AD_PRUNE=1 \
 AD_GEN_RESCUE=1 AD_MHT=12 AD_TAG=mdebt3full \
 python3 action_decision_maximum/src/train_full_cli.py > work/mdebt3full.log 2>&1 &
```
- 깊이 10ep = 프로브 best ep12 × 56/70 업데이트 등가(9.6→10) — R62후속 1차 후보의 3자 서명 이행.
- AD_FGM=1 필수(프로브 fgm=True·R36 mdeberta 레시피), AD_MAXLEN=320(프로브와 동일 — 384 아님).
- 예상 소요 ~3.6-3.7h(21.0min/ep×10+저장), 디스크 ~1.5GB(112GB 여유).

## 발진 5분 체크(불일치 = 즉시 kill)
1. 헤더: `[full] mdebt3full: microsoft/mdeberta-v3-base v=v6 len=320 ep=10 lr=2e-05 b=64 prune=True ... fgm=True gen_rescue=True mht=12`
2. `[gen_rescue] 30845/70000 rows header-preserved (mht=12)` — 프로브와 정확 일치(동일 70k·토크나이저)

## 격리(quarantine)
체크포인트·예측·조립·제출 전면 격리. 조립·제출 개방 조건(사전등록):
- codex#3 C: 원래 "A 강양성 ∧ τ 통과". A는 음성(-0.00375)으로 판명 → 레드팀 밴드(≤−0.0005) 발동:
  **paired-control 설계로만 제출 가능.** 그 paired-control = δ_mdeb 프로브(2026-07-11 10:30 실측):
  우리 mdebfull vs 배포 조원-mdeb, 같은 5k·같은 서빙(384+rescue) — solo +0.00207 / casc +0.00065
  CI95 [-0.00332,+0.00453] → **m1형 δ세금 부재 실증**(m1은 동일조건 solo -0.0116).
- 조립 전 필수 게이트(전부 3자 재서명): ①parity A/B/C(vs th85-m2, 구조증명+변경행 비율+F1 경보)
  ②시간 리플레이(서버환산 ≤555s; m1t3 581s 실측 교훈 — 570s 운영캡 엄수) ③멤버앵커 폴드 계기
  (teacher_mdeb12_f14/mdeb12ep_f0로 δ′ 재확인) ④τ 판정 결과 반영(τ=축폐쇄면 3자 kill-vs-keep).
- 판정 불지연: DONE_m1h8full 등장 즉시 GPU1에서 eval_tau_delta 실행(본 학습과 무간섭).

## 근거 기록
- mdeb-T3 프로브: best ep12 val 0.7640(+0.015, 이중계상 보정 후 +0.0115 순양수), work/mdebr_f0.log
- m1t3 LB 실측: 0.78651(-0.00375, 581s) — m1축 T3-FULL 폐쇄, 홀드아웃→LB 전이비 ~0.42 캘리브
- δ_mdeb: work/delta_mdeb_5k.npz + work/delta_mdeb.log
