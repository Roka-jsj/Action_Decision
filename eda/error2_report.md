# 에러분석 2차 (large-4ep 5fold OOF, pooled=0.7043)

## 클래스별 F1 (오름차순)

| class | support | P | R | F1 |
|---|---|---|---|---|
| list_directory | 3952 | 0.3676 | 0.6771 | 0.4765 |
| read_file | 8522 | 0.5239 | 0.4654 | 0.4929 |
| grep_search | 9099 | 0.648 | 0.4993 | 0.564 |
| lint_or_typecheck | 2061 | 0.5077 | 0.657 | 0.5728 |
| ask_user | 2467 | 0.6721 | 0.5209 | 0.5869 |
| glob_pattern | 4844 | 0.6503 | 0.5882 | 0.6177 |
| web_search | 1144 | 0.5151 | 0.7911 | 0.6239 |
| plan_task | 2479 | 0.6608 | 0.6539 | 0.6573 |
| run_tests | 4183 | 0.7232 | 0.704 | 0.7135 |
| run_bash | 4673 | 0.816 | 0.7458 | 0.7793 |
| apply_patch | 4418 | 0.8325 | 0.9088 | 0.869 |
| edit_file | 10199 | 0.9517 | 0.9135 | 0.9322 |
| write_file | 1373 | 0.9572 | 0.9934 | 0.975 |
| respond_only | 4776 | 0.9992 | 0.9996 | 0.9994 |

## 혼동쌍 상위 15 (정답→오답, 건수)

- grep_search → read_file: 2190
- read_file → list_directory: 2139
- read_file → grep_search: 1606
- grep_search → list_directory: 1598
- glob_pattern → list_directory: 817
- run_tests → lint_or_typecheck: 777
- edit_file → apply_patch: 767
- glob_pattern → read_file: 679
- list_directory → read_file: 678
- read_file → glob_pattern: 674
- ask_user → plan_task: 665
- run_bash → run_tests: 629
- grep_search → glob_pattern: 627
- run_bash → lint_or_typecheck: 482
- ask_user → web_search: 479

## 슬라이스별 오류율

- history=0(세션시작): n=8276, err=0.439, macroF1=0.4304
- history 1-2: n=15853, err=0.310, macroF1=0.6757
- history 3+: n=40061, err=0.263, macroF1=0.6806
- gen=sim: n=59568, err=0.286, macroF1=0.7131
- gen=au: n=4622, err=0.441, macroF1=0.5334

## fold0: 6ep − 4ep 클래스별 F1 변화 (내림차순)

- lint_or_typecheck: +0.0473
- web_search: +0.0453
- read_file: +0.0355
- plan_task: +0.0317
- run_tests: +0.0272
- list_directory: +0.0248
- run_bash: +0.0246
- apply_patch: +0.0155
- glob_pattern: +0.0146
- grep_search: +0.0142
- edit_file: +0.0103
- write_file: +0.0037
- respond_only: +0.0000
- ask_user: -0.0105

## rank-2 정답 비율(오류 중): 10874/19091 = 0.570