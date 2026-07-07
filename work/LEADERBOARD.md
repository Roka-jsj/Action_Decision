# 자율 개선 루프 — 리더보드 (honest sim-only fold0 macro-F1)

**메트릭(07-06 수정)**: **raw sim fold0**(에폭로그 sim= 피크) 우선. +bias는 반분CV 적합만 인정 — in-sample bias는 +0.005~7 낙관편향 실증(R14보강).
**루프**: 수정(recipe) → fold0 teacher 학습(~4h) → 측정 → 리더보드 갱신 → **top-2 유지**(FULL 승격·나머지 삭제) → 반복.
**주의**: 각 반복 ~4–5h GPU. 하루 ~4–5회. 다중일 캠페인. FULL 승격은 리더보드 상위 확정 시에만.
**이중 에이전트(07-06~)**: Claude=`cc_`, codex=`cx_` 네임스페이스. 행 추가만, 남의 행 수정 금지. 규약=AGENTS.md.
**은행 LB 기준점: 0.78449 (tri_v4new, 07-07 18:30)** — 궤적 0.78266→0.78364→0.78449. 1등 0.7947.

## 리더보드 (sim-only fold0, +bias)

| # | recipe | max_len | ep | FGM | fold0 sim +bias | 상태 |
|---|--------|---------|----|----|-----------------|------|
| 1 | large-v6 10ep+FGM | 320 | 10 | ✅ | **0.7645** | FULL 배포 **LB 0.77850** — 10ep+SWA4가 피크(ep7) 지나침, R14 |
| 2 | large-v6 6ep (구 교사) | 320 | 6 | ❌ | 0.7569 | 은퇴(구 배포 0.78051) |
| — | large-v4 10ep+FGM | 320 | 10 | ✅ | (fold0 미측정, FULL만) | 앙상블 다양성 멤버 |
| ❌ | large-v6 10ep+FGM **ml384** | 384 | 10 | ✅ | raw sim 0.7498 (val 0.7626) | **NO-GO** — ml320(0.7542/0.7662)에 전 에폭 열세. 길이 축 폐쇄 |
| ❌ | cx_simonly (au제외+SELECT_SIM) | 320 | 10 | ✅ | raw sim **0.7519** (ep7) | **NO-GO** — 기준 0.7542 대비 -0.0023. au는 OOD여도 데이터가치 양수 (cc 측정, cx 확인 대기) |
| ⏳ | **FULL 재승격 8ep+SWA2** (배포용) | 320 | 8 | ✅ | — (FULL은 fold0 없음) | 학습중 — R14 수정 적용, ~01:45 완료 예정 |

## 백로그 (문헌 서베이 반영 재서열, 07-07 — 근거는 DEBATE R18)
0. ⏳ FULL 재승격 8ep+SWA2 s1234 (학습중) → 게이트 실측으로 판정
1. **시드 앙상블 확보**: 동일레시피 seed777 FULL → 2멤버 양자화 앙상블 (문헌: 리스크 최소·경계샘플 이득 집중) ← 다음 슬롯 예약됨
2. **R-Drop** (Liang 2021, RoBERTa-large +0.8pt, FGM과 가산적[R-AT]) — 코드 2~3h 후 fold0 프로브
3. **EMA/올바른 SWA** (현 SWA_K는 감쇠LR 평균 = 문헌상 오용. 스텝 EMA decay0.999로 교체 검토, +0.1~0.3pt)
4. greedy soup (추론비용 0, NLP 보고치 +0~0.8pt) — 시드 멤버 쌓이면
5. 4클래스 전용 재채점 헤드 (오류전파 무력화된 특수케이스 — 소속판별은 이미 해결) — 1일 작업, CV 필수
6. ~~focal/logit-adjust~~ **기각** (Menon: post-hoc bias와 동일 함수족 = 중복. 병목은 유사빈도 혼동이라 원리상 부적합)
7. ~~라벨스무딩~~ 기각(혼동클러스터에 역효과 가능) / ~~FreeLB·SMART~~ 기각(R-Drop 우선) / SupCon 후순위(full-data 이득 미미)


**deploy 실측**: 단일 0.77850 / 2-large 0.77986 — 둘 다 은행 0.78266 미달 (R14 원인분석).

## 배포 후보 (top-2 유지)
- **①submit_largev6_10ep_fgm.zip** (단일 large-v6-10ep-FGM) — 검증완료
- **②submit_2large_fgm.zip** (2-large 앙상블 v6+v4) — 검증완료
- LB 회신 시 이 2개 우선순위 재조정.
