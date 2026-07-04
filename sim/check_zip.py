"""제출 zip 최종 검증기 — 구조/용량/컬럼/클래스/파일 점검.

사용: python sim/check_zip.py <submit.zip>
"""
from __future__ import annotations
import sys, zipfile, os

CLASSES = {"read_file", "grep_search", "list_directory", "glob_pattern", "edit_file",
           "write_file", "apply_patch", "run_bash", "run_tests", "lint_or_typecheck",
           "ask_user", "plan_task", "web_search", "respond_only"}


def check(path):
    ok = True
    def bad(m):
        nonlocal ok; ok = False; print("  ❌", m)
    def good(m): print("  ✅", m)

    z = zipfile.ZipFile(path)
    names = z.namelist()
    tops = sorted(set(n.split("/")[0] for n in names if n.strip("/")))
    print(f"[zip] {path}  ({os.path.getsize(path)/1e6:.1f} MB)")

    # 1) 최상위 구조
    if set(tops) <= {"model", "script.py", "requirements.txt"} and "script.py" in tops and "model" in tops:
        good(f"최상위 = {tops}")
    else:
        bad(f"최상위 구조 오류: {tops} (model/, script.py, requirements.txt 만 허용)")

    # 2) 용량
    sz = os.path.getsize(path) / 1e9
    good(f"용량 {sz:.2f} GB (<1GB)") if sz < 1.0 else bad(f"용량 {sz:.2f} GB ≥ 1GB")

    # 3) 필수 파일
    need = ["script.py", "requirements.txt"]
    model_files = [n for n in names if n.startswith("model/") and not n.endswith("/")]
    for f in need:
        good(f"{f} 존재") if f in names else bad(f"{f} 누락")
    has_w = any(n.endswith((".safetensors", ".bin", ".pkl", ".txt", ".pt")) for n in model_files)
    good(f"model/ 파일 {len(model_files)}개") if model_files else bad("model/ 비어있음")

    # 4) 토크나이저/코드 파일 존재(트랜스포머 제출인 경우)
    if any("ad_lib.py" in n for n in names):
        for req in ["model/ad_lib.py", "model/config.json", "model/run_meta.json"]:
            good(f"{req} 존재") if req in names else bad(f"{req} 누락")
        spm = any(n.endswith((".model", ".json")) and "token" in n.lower() for n in names) or \
              any(n.endswith("sentencepiece.bpe.model") for n in names)
        good("토크나이저 파일 포함") if spm else bad("토크나이저 파일(spm/tokenizer.json) 누락 위험")
        st = any(n.endswith(".safetensors") for n in names)
        good("safetensors 가중치 포함") if st else bad("safetensors 없음(용량/호환 위험)")

    print("=> 결과:", "PASS ✅" if ok else "FAIL ❌")
    return ok


if __name__ == "__main__":
    sys.exit(0 if check(sys.argv[1]) else 1)
