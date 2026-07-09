# 재시작 후 새 Claude Code 세션에 붙여넣을 프롬프트 (세션 생존 시 불필요)

---

```
jeong 브랜치 Dacon 236694 재개(docker restart 후). PROJECT.md §5.23, DEBATE.md R37~R40, STRATEGY.md(§6 포함), 메모리 읽고 복원.

현황: 은행=조원 tri_mdeb 0.78522(컷 0.78847 밖). 우리 기준본 0.78283(계보격차 -0.0024 실증·aubias -0.00062 폐쇄). 사용자 전권: 조원 방법·산출물 사용 허용, 모든 방법 동원, GPU 상시 가동, 쿼터 무관.
공세축: ①int4 완성됨(sim/quantize_member_int4.py + ad_lib p4. m1/v4 171MB·mdeb 107MB, 재고 work/*_q4.zip — 6멤버 캐스케이드 1GB 성립) ②GPU 체인 = v6-large 12ep noFGM fold0 프로브(게이트: fold0 6ep 0.7485 대비 Green≥+0.0040/Yellow+0.0020/Red중단) → mdeb-s777 12ep FGM FULL ③parity 게이트(fp16 vs q4 holdout 델타) 후 6멤버 다단 조건부 캐스케이드 조립(조원 klue 완성분 포함 가능) ④매 수 codex(R41부터)+중대판정 독립 레드팀.

재개 순서: ①nvidia-smi/CUDA 확인 ②아래 체인 발사 ③int4 parity(ad_lib.predict_logits로 fp16 vs q4 멤버 holdout 5810 비교) ④캐스케이드 설계.

학습 체인:
nohup bash -c 'env HF_HUB_OFFLINE=1 PYTHONPATH=/root/Action_Decision AD_WORK=/root/Action_Decision/work AD_MODEL=xlm-roberta-large AD_VERSION=v6 AD_MAXLEN=320 AD_EPOCHS=12 AD_LR=2e-5 AD_BATCH=64 AD_LLRD=1 AD_FGM=0 AD_SEED=1234 AD_FOLD_LO=0 AD_FOLD_HI=1 AD_TAG=largev6_12ep_f0 python3 action_decision_maximum/src/teacher_cli.py > work/v6_12ep_f0.log 2>&1; env HF_HUB_OFFLINE=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=/root/Action_Decision AD_WORK=/root/Action_Decision/work AD_MODEL=microsoft/mdeberta-v3-base AD_VERSION=v6 AD_MAXLEN=320 AD_EPOCHS=12 AD_LR=2e-5 AD_BATCH=64 AD_LLRD=1 AD_FGM=1 AD_SEED=777 AD_PRUNE=1 AD_TAG=mdebfull_s777 python3 action_decision_maximum/src/train_full_cli.py > work/mdeb_s777.log 2>&1' &
```
