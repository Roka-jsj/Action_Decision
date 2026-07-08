# 재시작 후 새 Claude Code 세션에 붙여넣을 프롬프트

아래 블록을 새 세션 첫 입력으로 그대로 붙여넣으세요.

---

```
jeong 브랜치에서 Dacon 236694 작업 재개. docker restart(NVML 복구)로 세션이 끊겼다. PROJECT.md §5.23, DEBATE.md R33~R37, STRATEGY.md(통일이론 M1~M6/S1~S5), 메모리를 읽고 복원해라.

핵심: 계정 은행 = 조원 tri_mdeb 0.78522(20위, 컷 0.78847 밖). 조원(dev 브랜치, mun-train GPU1, 자료공유 없음·쿼터만 공유)은 klue-large/koelectra/int4/증류로 컷 추격 중. 우리(jeong)=보조트랙: 자체 mdeberta 재현 + 후처리 변형(aubias-on-tri_mdeb 1순위 → th 0.45/0.40 ≤2발 → w2 0.20 후순위) + 동결설계. 발사기준 기대LB ≥0.78545, proxy<+0.0005 폐기. 1차 5발(tc 스윕)은 보류 확정(R37). 프록시→LB 전이율 40%.

재개 순서: ①nvidia-smi 정상 확인 ②mdebfull 재발사(아래 명령, gpu_when_idle 래퍼 필수 — GPU 공유 정책) ③완료(~4.5h)시 aubias-on-tri_mdeb 조립→검증→발사안 보고 ④매 수 codex 토론(R38부터)·DEBATE 기록.

mdebfull 재발사:
bash sim/gpu_when_idle.sh 7200 -- env HF_HUB_OFFLINE=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=/root/Action_Decision AD_WORK=/root/Action_Decision/work AD_MODEL=microsoft/mdeberta-v3-base AD_VERSION=v6 AD_MAXLEN=320 AD_EPOCHS=12 AD_LR=2e-5 AD_BATCH=64 AD_LLRD=1 AD_FGM=1 AD_SEED=1234 AD_PRUNE=1 AD_TAG=mdebfull python3 action_decision_maximum/src/train_full_cli.py > work/mdeb_full.log 2>&1
(b64 필수 — b128+FGM은 자기 OOM. protobuf==3.20.3 설치돼 있음. NVML 죽어도 로그 epoch 라인으로 진행 확인.)
```

---

## 참고: 재시작 절차 (호스트 터미널)
```
docker restart mun-jtrain mun-jtest mun-train mun-test
```
- 파일 전부 보존. 코드·문서는 origin/jeong push됨(R37까지).
- packages/ 준비물: tc 스윕 10종+sb_tri(전부 보류 상태), aubias bias는 eda 스크립트 재계산 가능.

## 현재 in-flight (재개 시 확인)
- mdebfull 학습이 restart로 사망 — 위 명령으로 재발사 (유일한 GPU 작업).
- 조원 트랙은 origin/dev fetch로 확인(klue-large fold0 진행 중이었음). 컨테이너 읽기전용·무수정 원칙.
- codex R37까지 완료(work/r37_codex.md). R38부터.
