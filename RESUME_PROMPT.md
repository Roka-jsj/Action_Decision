# 재시작 후 새 Claude Code 세션에 붙여넣을 프롬프트 (세션 생존 시 불필요)

---

```
jeong 브랜치 Dacon 236694 재개(재시작 후). PROJECT.md §5.23, DEBATE.md R41~R44, STRATEGY.md(§6 포함), 메모리 읽고 복원.

현황(07-09 저녁): 은행 = 우리 C0 0.78567(30110 submit_c0_klue.zip: 조원m1+조원mdeb+우리klue7ep_q8, cond[1,2], w 0.6/0.15/0.25, 구bias, 서버 380s — 실캡 600s 확인). 컷 0.78847(-0.0028). R44 3자 판정: 0.80 불가(P<1%), 목표=컷(P 35-45%). 사용자 전권: 조원 방법·산출물 사용 허용(컨테이너 수정만 절대금지), 쿼터 무관, GPU 유휴시만 사용(gpu_when_idle.sh).

죽은 축(재론금지): prior/transductive·aubias·S5패치·specialist·384(v6)·sb·koelectra·naive int4·kfdeb(0.7299+ρ0.954+델타-0.0006)·v4 12ep(용량슬롯 없음)·증류(현행 게이트).

진행 중이던 것:
① C1: v6 FULL 11ep(work/v6_11ep_full.log, 16:51 실시작·~19:35 zip) → sim/quantize_member.py로 q8 → sim/parity_int4.py 게이트 → bash sim/assemble_c1.sh → 발사. 사전등록: 중앙 0.7864 밴드 [0.7853,0.7880], <0.78567이면 m1 교체 폐기·후속조립은 C0 앵커.
② klue folds1-4 OOF 농사 웨이터 장전됨(7ep b32 FGM, ~15.5h) — 죽었으면 재발사:
   bash sim/gpu_when_idle.sh 21600 -- env HF_HUB_OFFLINE=1 PYTHONPATH=/root/Action_Decision AD_WORK=/root/Action_Decision/work AD_MODEL=klue/roberta-large AD_VERSION=v6 AD_MAXLEN=320 AD_EPOCHS=7 AD_LR=2e-5 AD_BATCH=32 AD_LLRD=1 AD_FGM=1 AD_SEED=1234 AD_FOLD_LO=1 AD_FOLD_HI=5 AD_TAG=klue_f14 python3 action_decision_maximum/src/teacher_cli.py > work/klue_f14.log 2>&1
③ 로드맵: klue농사 → mdeb f1-4(13h) → v6-12ep f1-4(8.5h) → D-4(7/11) 클린 앙상블 OOF 재적합(bias/가중/th/w2/coverage 그리드→사전등록 최적점 1발) → D-3 klue@384 fold0(조건부) → D-2 동결 → D-1 이중은행(은행1=최고LB, 은행2=사전등록 구조상이본). LB예산 ~5-7발.
주의: 재적합 시 배포멤버 OOF 근사 문제(조원 m1·FULL모델은 클린 OOF 부재 — fold 계열로 프록시) 사전등록에 명시. 매 수 codex+중대판정 독립 레드팀(3자).
```
