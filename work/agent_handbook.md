# Agent 핸드북 (증류판 — 신규 agent는 이것부터. 갱신: 2026-07-10 21:15)

## 임무 공통 수칙
- 실측 > 추정. 발사는 3자(지휘+codex+레드팀) 서명 후만. 조원 컨테이너(mun-*) 수정 절대 금지(읽기/복사만). GPU는 진행 중 학습 보호(소배치 병행만). splits/splits.npz 재생성 절대 금지(sklearn 버전 비재현 — sha256 고정). 산출물은 파일로 박제(스크래치패드·work/), 보고는 요약+수치표.

## 대회·목표
Dacon 236694, 14클래스 macro-F1, 히든 30k(=Public=최종, 셰이크업 없음·제출 하방 없음), zip≤1GB, T4 600s(운영 hard 570s·soft 540s), 마감 7/15 10:00. **은행 th85 0.79026(30881)**. 컷(12위) 0.79064 인플레 중(+0.001/일, D-0 ~0.792-3). 사용자 목표 우승(0.8) — 실측 상한 0.796(R57 오라클/선택기 분석), P(top-12) ~25-45%.

## 확정 교리 (전부 LB/클린 실측 근거)
1. **입력축**: "훈련분포 복원만 액면 전이"(rescue 전이 0.82). 신규 입력형태 = OOD(compress 0.10, 448/hist12 음수). 추론시 길이확장 전면 폐쇄(절대·상대위치 공히).
2. **결정규칙 압착 종점**: 가중/th/bias 미세조정 전이 0.04~0.26, th축 종결(thasym95 flat). 선택기 상한 honest 음수 — 관측신호 소진.
3. **T3(훈련-서빙 정합 재학습)가 유일 대형 레버**: 이중계상 주의 — 프로브 val은 기적립 rescue 몫(+0.0035 val등가)을 차감(공정 기준선 = 구피크+0.0035).
4. 게이트 표준: 사전등록(밴드·분기)·교차폴드(이중 L-프록시 v8/v9)·후보풀 평균 ≥0(선택노이즈 스크린)·"이득이 GEN삭제행 집중이면 인공물 자동기각"·발사 전 30k 리플레이(경합·구식 calib 2.62 무효 — 실효 경합비 1.25~1.28 2점).
5. FULL 끝점 외삽 금지(C1 −0.0082) — FULL 에폭 = fold 프로브 best-epoch 고정.

## 계기·자산 인벤토리
- 하네스: sim/refit_lib.py·refit_d4.py(fold0 EXACT·2단 게이트·rescue 모드), sim/gen_rescue_oof.py, ad_lib: gen_rescue/stages/member_th/compress_tta/멤버별 max_len 전부 opt-in 구현.
- OOF: v6-12ep f0, klue f0(조원)+f1-4, mdeb12 f0(조원)+f1-4(work/teacher_mdeb12_f14.npz), v8/v9 f0+f14(L-프록시), m1 f0-rescue(work/m1_f0ckpt_rescue.npz). mdeb/klue fold ckpt 부재(rescue 재추론 불가 — 2단 구조 강제).
- 프로브 로그: work/v6r_f0.log(T3-320: rescue+mht12, ep10 0.7622 배포GO 돌파). 배포 계보 패키지: packages/submit_{genrescue,d1tta,mdeb384,th85}. postproc 원천: packages/submit_tri_cond_rebuild/model/postproc.json.
- 시간 모델: 커버리지 비용 620s/100%(±15%), 실측점 495/496/520/558/553s. 슬로프: m1 1.05 / mdeb 0.108@th0.85 / klue 0.023.
- 사망 축 대장: experiments_master.csv + DEBATE.md R44~R61 (재론 시 신논거 필수).

## 현행 작전 (R61b)
21:40 T3-320 판독(≥0.7605 배포GO — ep10 이미 돌파) → m1-T3 FULL 밤샘(기대 LB +0.001~25) → mdeb-T3 프로브(기준선 0.749) → 아침 양자화·parity·조립·캐너리 → D-3 klue-T3·문샷(합성증강, A-lock 0.7915+ 후 GPU) → D-2 동결.

## (07-11 갱신) 단독 운영 체제
- **조원은 없다**(운영자 확정). 문서의 "조원 m1/컨테이너/레시피"는 과거 산출물 지칭 관습 — 요청할 인간 없음. 조원 역할(독립 검증·대안 파이프라인)이 필요하면 서브에이전트 신설로 대체. GPU 2개·일일 10발 전부 운영자 단독 사용.
- σ_LB=0 실증(th85 재제출 소수10자리 동일) — 동일파일 재제출 정보 0, 모든 점수차는 순수신호. 시간 서버분산 ±7s → 후보 시간예산 중앙 ≤555s.
- 프로세스 kill/생존확인은 기록된 PID(+starttime)로만 — env변수는 cmdline에 없어 pkill/pgrep -f 불가·self-match 위험.
- CI 안의 점추정은 발사·폐쇄 근거 불가(Δ의 런간CI 하한이 0∧+0.003 마진을 넘을 때만 액션).
