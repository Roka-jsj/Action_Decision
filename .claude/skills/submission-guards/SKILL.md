---
name: submission-guards
description: 대회 제출물(코드+모델 zip) 조립 파이프라인에 6종 안전가드와 런타임 캐너리를 구축. 제출 파이프라인 첫 구축, 제출 오류 발생 후 재발방지, 마감 임박 자동조립 셋업 시 사용.
---

# 제출물 조립 6종 가드 + 런타임 캐너리

전부 실제 사고 후 도입된 가드들이다. 새 대회의 조립 스크립트에 처음부터 전부 넣는다.

## 가드 6종 (bash 조립 스크립트 골격)

```bash
set -euo pipefail
# 1) 자기파괴 가드 — 출력이 원본(은행)을 덮지 않게
[ "$DST" != "$SRC" ] || { echo "FATAL: DST==SRC"; exit 1; }

# 2) 캐시 신선도 — 파생물(양자화본 등)이 원본보다 오래되면 재생성
if [ -f "$CACHE" ] && [ "$CACHE" -nt "$SRC_FILE" ]; then echo reuse; else regenerate; fi

# 3) 메타 assert — 서빙 설정파일의 핵심 플래그를 조립마다 검증
python3 -c "import json; m=json.load(open('$DST/run_meta.json')); assert m[...]==expected"

# 4) byte-diff 가드 — 새 티켓이 기존 제출물과 바이트동일이면 FATAL(슬롯 낭비 방지)
[ "$(md5sum A)" != "$(md5sum B)" ] || exit 1

# 5) 용량 가드 — 사이트 기준(=SI 바이트)으로 검증. du의 MiB와 혼동 금지(879MiB=920.9MB!)
[ "$(stat -c%s $ZIP)" -lt 1000000000 ] || exit 1

# 6) 런타임 캐너리(필수) — 실제 제출 zip을 풀어 진입점을 끝까지 실행
#    - 소형 목데이터 N행으로 script.py 완주 (import 누락·경로오류를 잡는 유일한 방법)
#    - 출력 행수·필수 마커(missing=0)·라벨 유효성(허용 클래스 집합) 전부 assert
```

## 캐너리가 잡는 것 / 못 잡는 것
- 잡음: import 누락(실사고: 개발용 모듈 import 잔존으로 서버 ModuleNotFoundError → 슬롯 2개 소각), 파일경로, 로더 크래시, 출력 스키마
- 못 잡음: 서버 시간초과(연산량 변화는 N행으로 안 보임 — 연산량이 늘어난 변경은 별도 시간 추정), 점수 품질
- zip 완결성: 다른 프로세스가 쓰는 중인 zip을 열지 마라 — 크기 안정(2회 폴링 동일) + `zipfile.ZipFile(f).testzip()` 통과 후 사용 (BadZipFile 레이스 실사고)

## 운영 수칙
- 캐너리 통과본만 사용자에게 핸드오버, 통과 로그를 함께 제시
- 마감 임박 시 새 조립경로 도입 금지 — 검증된 스크립트의 인자만 바꾼다
- 평가 전용/폐기 패키지는 quarantine 디렉토리로 격리 (오업로드 방지)
