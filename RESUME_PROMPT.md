# 재시작 후 새 Claude Code 세션에 붙여넣을 프롬프트

아래 블록을 새 세션 첫 입력으로 그대로 붙여넣으세요.

---

```
지금 브랜치 jeong에서 Dacon 236694(Action Decision) 작업을 이어간다. 컨테이너 재시작/세션전환으로 이전 작업이 끊겼다. 먼저 PROJECT.md의 §5.23~§5.26(재개 스냅샷 — 최우선), DEBATE.md의 R22~R29, work/scoreboard.md, 그리고 메모리를 읽고 전체 맥락을 복원해라.

핵심만: 외부/조원 자료 공유는 없음. 조원과는 제출쿼터만 공유한다. 우리 자체 최고 LB는 tri_cond 0.78266. 목표는 방어가 아니라 우리 로그/구조/파라미터를 조합해 새 기록을 만드는 것. probe는 제출하지 않음(쿼터 보존). session-balanced R27은 짧은세션/hist0 양성이나 nnQ1(low) 악화로 FULL/LB 보류. R28 compact retrieval p384는 제출되어 LB 0.78261로 평탄 종료. R29에서 probe 없이 hidden/test batch 확률만 쓰는 transductive EM label-shift 후보를 구현했다.

재개 순서(§5.23~5.26): ①nvidia-smi로 GPU 상태 확인 ②새 1순위 후보 `packages/submit_tri_cond_em075.zip` 상태 확인(0.943GB, check_zip PASS, A6000 210s, VRAM10868, 30k holdout 0.80907 = rebuild 0.80702 대비 +0.00205, 최종 changed 373/30000=1.24%). ③`em05`는 보수 후보(holdout 0.80768). ④LB probe는 제출하지 말고, 제출 슬롯을 쓸지는 사용자 결정 필요. ⑤추가 개선은 EM shrink/clip/gate의 보수적 검증 또는 전혀 다른 구조 축만. 매 수마다 DEBATE.md·experiments_master.csv 기록.

지금 GPU/컨테이너 상태부터 점검하고, 위 순서로 이어서 진행해라.
```

---

## 참고: 재시작 절차 (호스트 터미널)
```
docker restart mun-jtrain mun-jtest mun-train mun-test    # 데이터 보존(rm 아님), NVML cgroup 복구
```
- 재시작 후 VS Code에서 mun-jtrain에 재접속 → 폴더 /root/Action_Decision → Claude Code 새 세션 → 위 프롬프트.
- 파일 전부 보존됨(restart는 파일시스템 유지). 코드·문서는 origin/jeong에도 push됨(2중 안전).
- packages/(33개 zip)·action_decision_maximum/experiments/(19 member) 보존.

## 현재 in-flight (재개 시 확인)
- 실행중 GPU작업 없음이어야 함. `nvidia-smi`로 확인.
- LB probe zip 14개는 준비돼 있지만 제출하지 않음(쿼터 보존).
- codex/R29까지 완료. session-balanced 산출물: work/foldckpt_sbwt_f0_f0, work/foldckpt_sbsp_f0_f0, work/sessbal_*_f0.log, sim/eval_stress.py. compact retrieval 산출물: work/retrieval_pack_p384, work/retrieval_pack_p256, packages/submit_tri_cond_retr_p384.zip, packages/submit_tri_cond_retr_p256.zip, packages/submit_tri_cond_rebuild.zip. EM 산출물: packages/submit_tri_cond_em05.zip, packages/submit_tri_cond_em075.zip, work/verify_tri_cond_em05.log, work/verify_tri_cond_em075.log. 앙상블 retrieval + transductive EM 지원은 common/ad_lib.py + sim/package_ensemble.py + sim/package_single.py에 구현됨.
