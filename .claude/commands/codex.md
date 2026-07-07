---
description: codex(GPT-5.5)와 적대적 토론 라운드 발사 — 표준 절차로 브리핑 인라인·발사·감시
---

codex와 상시 비판 토론(me⇄codex)을 한 라운드 진행한다. 사용자 인자(`$ARGUMENTS`)가 이번 라운드 쟁점이다.

## 절차 (반드시 지킬 것)
1. **브리핑 작성**: `work/rNN_brief.md`에 이번 쟁점 + 실측 수치 + "동의만 하지 말고 수치로 반박하라" + 질문별 [승인/반박/수정] 형식 요구.
2. **자료 인라인 (필수)**: codex는 컨테이너 bwrap 샌드박스 때문에 파일을 못 읽는다. 참고자료(DEBATE.md 해당 라운드, experiments_master.csv tail 등)를 브리핑에 **직접 붙여** `work/rNN_brief_full.md` 생성.
3. **발사**:
   ```
   CODEX_HOME=/root/.codex nohup /root/.vscode-server/extensions/openai.chatgpt-*/bin/linux-x86_64/codex exec \
     --sandbox read-only --skip-git-repo-check --ephemeral -C /root/Action_Decision \
     -c model_reasoning_effort='"xhigh"' --color never - < work/rNN_brief_full.md > work/rNN_codex.md 2>&1 & disown
   ```
4. **감시**: `pgrep -f "[c]odex exec"`로 종료 확인(브래킷 패턴 필수 — 자기매칭 자살 방지). 완료 후 `sed -n '/^codex$/,$p' work/rNN_codex.md | grep -vE "web search|tokens used|bwrap"` 로 답변 읽기.
5. **기록**: 판정을 DEBATE.md에 라운드로 요약, experiments_master.csv에 관련 실측 추가.

## 주의
- `pkill -f "codex exec"`는 자기 셸을 죽인다 → `pkill -f "[c]odex exec"` 브래킷 패턴.
- 하네스 엔지니어링(학습데이터·테스트·직렬화·평가·제출루프)을 항상 후보 공간에 포함.
- 판정 규칙: LB 실측만 신뢰(OOF는 방향성 없음이 다수 실증), Δ문턱·선택편향 예산 명시.
