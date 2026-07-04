# EDA & 시뮬레이터 역설계 리포트 — Dacon 236694

- 총 샘플: **70,000**  | 클래스: 14 | 세션(universal): **9,429**

## 1. 클래스 분포 (전체 / sim / au)
| class | all | all% | sim% | au% |
|---|---:|---:|---:|---:|
| edit_file | 11171 | 15.96 | 15.84 | 17.55 |
| grep_search | 9912 | 14.16 | 14.41 | 10.95 |
| read_file | 9257 | 13.22 | 12.26 | 25.69 |
| glob_pattern | 5284 | 7.55 | 8.00 | 1.77 |
| respond_only | 5178 | 7.40 | 7.44 | 6.89 |
| run_bash | 5068 | 7.24 | 7.04 | 9.87 |
| apply_patch | 4823 | 6.89 | 7.13 | 3.80 |
| run_tests | 4561 | 6.52 | 6.51 | 6.59 |
| list_directory | 4329 | 6.18 | 6.49 | 2.17 |
| ask_user | 2701 | 3.86 | 4.01 | 1.87 |
| plan_task | 2679 | 3.83 | 3.86 | 3.36 |
| lint_or_typecheck | 2283 | 3.26 | 3.08 | 5.59 |
| write_file | 1481 | 2.12 | 2.16 | 1.55 |
| web_search | 1273 | 1.82 | 1.78 | 2.35 |

- 최희귀 5: [('web_search', 1273), ('write_file', 1481), ('lint_or_typecheck', 2283), ('plan_task', 2679), ('ask_user', 2701)]
- 불균형 max/min = 8.8 (완만) | 5-fold 시 최소 클래스 fold당 ~254건 → 0셀 없음

## 2. 구조-only 오라클 상한 (다수결 예측기 = GBDT 상한 추정)
| 시그니처 | #groups | accuracy | macro-F1 |
|---|---:|---:|---:|
| prior(전체 다수결) | 1 | 0.1596 | 0.0197 |
| last_action | 14 | 0.2294 | 0.1371 |
| last_action+status | 29 | 0.2338 | 0.1460 |
| struct(last_act,status,ci,git,turn) | 379 | 0.2557 | 0.1719 |
| struct_rich(+tier,langpref,ext) | 11264 | 0.3718 | 0.3116 |
| prompt(정확일치) | 63250 | 0.9624 | 0.9554 |
| struct+prompt | 69344 | 0.9992 | 0.9990 |

> struct-only가 높을수록 규칙기반 시뮬 → GBDT 강세. prompt가 크게 더 얹으면 텍스트(트랜스포머) 투자 가치.

## 3. 전이행렬 P(다음 label | 마지막 action)
- 저장: eda/transition_matrix.csv
- 마지막 action별 최빈 다음 action (support≥200):

| last_action | N | top-next | purity% | 2nd |
|---|---:|---|---:|---|
| None | 9000 | list_directory | 20.2 | read_file |
| read_file | 8887 | edit_file | 29.0 | grep_search |
| grep_search | 9412 | edit_file | 21.9 | read_file |
| list_directory | 4223 | read_file | 25.1 | grep_search |
| glob_pattern | 4967 | grep_search | 22.2 | glob_pattern |
| edit_file | 10620 | run_tests | 23.0 | edit_file |
| write_file | 1446 | edit_file | 40.4 | run_bash |
| apply_patch | 4417 | lint_or_typecheck | 17.1 | respond_only |
| run_bash | 4797 | run_bash | 20.4 | edit_file |
| run_tests | 4251 | edit_file | 24.5 | respond_only |
| lint_or_typecheck | 2016 | apply_patch | 24.9 | edit_file |
| ask_user | 2192 | grep_search | 20.0 | read_file |
| plan_task | 2584 | apply_patch | 18.0 | read_file |
| web_search | 1188 | edit_file | 19.0 | grep_search |

## 4. 거의 결정적 규칙 (purity≥85%, support≥100)
| purity | support | (last_act,status,ci,git,turn) | -> label |
|---:|---:|---|---|

## 5. 마지막 result_status → 다음 label
| status | N | top-next | % |
|---|---:|---|---:|
| success | 51350 | edit_file | 17.5 |
| na | 9000 | list_directory | 20.2 |
| error | 2957 | grep_search | 15.2 |
| test_pass | 2901 | respond_only | 26.3 |
| zero | 1882 | glob_pattern | 19.7 |
| test_fail | 1351 | edit_file | 33.4 |
| nonzero_exit | 559 | edit_file | 21.1 |

## 6. respond_only 특이성
- history에 등장하는 action 종류: 13/14. respond_only ∈ history: **False** (없으면 터미널 행동).

## 7. current_prompt 언어/길이
- 내용기반 언어: {'en': 19208, 'mixed': 40416, 'ko': 10376}
- language_pref 필드: {'en': 17802, 'ko': 45028, 'mixed': 7170}
- prompt 문자수: mean=61 p50=56 p90=103 p99=169 max=346

## 8. 직렬화(전체 history) 근사 문자길이: mean=376 p50=397 p90=605 p99=770 max=1024
> 서브워드 토큰 ≈ 문자/2~3 (ko/en 혼재). max_len 256~384 검토 근거.

## 9. 메타 신호 ↔ 라벨 (top-next per value)

**last_ci_status**: passed→edit_file(15%,n=28035) | failed→edit_file(18%,n=22623) | none→grep_search(15%,n=19342)

**git_dirty**: True→edit_file(16%,n=53701) | False→read_file(19%,n=16299)

**user_tier**: pro→edit_file(16%,n=37733) | free→edit_file(16%,n=20948) | enterprise→edit_file(16%,n=11319)

**turn_bucket**: t4-7→edit_file(19%,n=27606) | t2-3→edit_file(21%,n=17243) | t8+→grep_search(14%,n=16151) | t1→list_directory(20%,n=9000)
