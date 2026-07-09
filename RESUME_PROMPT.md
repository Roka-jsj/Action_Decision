# 재시작 후 새 Claude Code 세션에 붙여넣을 프롬프트 (세션 생존 시 불필요)

---

```
jeong 브랜치 Dacon 236694 재개(재시작 후). PROJECT.md §5.23, DEBATE.md R41~R44, STRATEGY.md(§6 포함), 메모리 읽고 복원.

현황(07-09 심야): **은행 = wd30 0.78621**(30350 submit_c0_wd30.zip: 조원m1+조원mdeb+우리klue7ep_q8, cond[1,2], **w 0.55/0.30/0.15**, th0.5, 구bias, 382s). 계보 0.78522→0.78567(C0)→0.78621(wd30). 컷 0.78847(−0.0023). 0.78554는 유령기록(정정됨). R44: 0.80 불가, 목표=컷. **D12 하네스**(조원 mdeb-12ep OOF 기반 fold0 캐스케이드 시뮬, 레드팀 산출) 부호 3/3 검증·전이율 0.26 — D-4 공식 시뮬레이터. 사용자 전권: 조원 산출물 사용 허용(컨테이너 수정 절대금지)·**GPU0 Claude 무조건 1순위**·모든 발사는 3자 사전검증 의무(R47 사고 교훈)·타계정 LB 인용 시 운영자 제출내역 대조.

죽은 축(재론금지): prior/transductive·aubias·S5패치·specialist·384(v6)·sb·koelectra·naive int4·kfdeb(0.7299+ρ0.954+델타-0.0006)·v4 12ep(용량슬롯 없음)·증류(현행 게이트).

진행 중이던 것:
① C1 종결: LB 0.77751 파국(-0.0082) → 사전등록대로 m1 교체 폐기, **C0 0.78567 앵커 확정**. 교훈(M6 확장): FULL 최종에폭 끝점은 fold 곡선에서 외삽 금지 — 신규 FULL 재학습 멤버 금지, fold-앙상블 또는 검증된 기존 FULL만.
② klue folds1-4 OOF 농사 웨이터 장전됨(7ep b32 FGM, ~15.5h) — 죽었으면 재발사:
   bash sim/gpu_when_idle.sh 21600 -- env HF_HUB_OFFLINE=1 PYTHONPATH=/root/Action_Decision AD_WORK=/root/Action_Decision/work AD_MODEL=klue/roberta-large AD_VERSION=v6 AD_MAXLEN=320 AD_EPOCHS=7 AD_LR=2e-5 AD_BATCH=32 AD_LLRD=1 AD_FGM=1 AD_SEED=1234 AD_FOLD_LO=1 AD_FOLD_HI=5 AD_TAG=klue_f14 python3 action_decision_maximum/src/teacher_cli.py > work/klue_f14.log 2>&1
③ 로드맵: klue농사 → mdeb f1-4(13h) → v6-12ep f1-4(8.5h) → D-4(7/11) 클린 앙상블 OOF 재적합(bias/가중/th/w2/coverage 그리드→사전등록 최적점 1발) → D-3 klue@384 fold0(조건부) → D-2 동결 → D-1 이중은행(은행1=최고LB, 은행2=사전등록 구조상이본). LB예산 ~5-7발.
주의: 재적합 시 배포멤버 OOF 근사 문제(조원 m1·FULL모델은 클린 OOF 부재 — fold 계열로 프록시) 사전등록에 명시. 매 수 codex+중대판정 독립 레드팀(3자).
```
