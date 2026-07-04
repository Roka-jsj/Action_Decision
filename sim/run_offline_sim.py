"""오프라인 평가서버 시뮬레이터.

- 격리 run 디렉터리 구성: model/ + script.py + requirements.txt + data/(test,sample_sub)
- 네트워크 차단(sitecustomize 로 socket.connect 외부 연결 차단)
- pip install 시간 + script.py 추론 시간 + child peak RSS(메모리) 측정
- output/submission.csv 스키마 검증
사용:
  python sim/run_offline_sim.py --model <model_dir> --script <script.py> [--n 30000] [--real]
"""
from __future__ import annotations
import argparse, os, sys, shutil, subprocess, time, resource, csv, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = {"read_file","grep_search","list_directory","glob_pattern","edit_file","write_file",
           "apply_patch","run_bash","run_tests","lint_or_typecheck","ask_user","plan_task",
           "web_search","respond_only"}

NETBLOCK = '''# 외부 네트워크 차단 (오프라인 검증)
import socket as _s
_orig = _s.socket.connect
def _blocked(self, addr):
    try:
        host = addr[0] if isinstance(addr, tuple) else str(addr)
    except Exception:
        host = ""
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise OSError("offline-sim: 외부 네트워크 차단됨 -> " + str(addr))
    return _orig(self, addr)
_s.socket.connect = _blocked
'''

def build_rundir(model_dir, script_path, data_dir, run):
    if os.path.exists(run): shutil.rmtree(run)
    os.makedirs(os.path.join(run, "data"))
    shutil.copytree(model_dir, os.path.join(run, "model"))
    shutil.copy(script_path, os.path.join(run, "script.py"))
    req = os.path.join(os.path.dirname(script_path), "requirements.txt")
    shutil.copy(req, os.path.join(run, "requirements.txt")) if os.path.exists(req) else open(os.path.join(run,"requirements.txt"),"w").close()
    for f in ("test.jsonl", "sample_submission.csv"):
        shutil.copy(os.path.join(data_dir, f), os.path.join(run, "data", f))
    open(os.path.join(run, "sitecustomize.py"), "w").write(NETBLOCK)

def run(cmd, cwd, env):
    t = time.time()
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    return time.time() - t, r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--script", default=os.path.join(ROOT, "action_decision_balance/submission/script.py"))
    ap.add_argument("--n", type=int, default=0, help="합성 test 크기(0=실제 test.jsonl 사용)")
    ap.add_argument("--run", default="/tmp/ad_sim/run")
    args = ap.parse_args()

    # 데이터 준비
    if args.n > 0:
        data_dir = "/tmp/ad_sim/data"
        subprocess.run([sys.executable, os.path.join(ROOT,"sim/make_synth_test.py"), str(args.n), data_dir], check=True)
    else:
        data_dir = os.path.join(ROOT, "data")

    build_rundir(args.model, args.script, data_dir, args.run)
    env = dict(os.environ)
    env["PYTHONPATH"] = args.run + os.pathsep + env.get("PYTHONPATH", "")  # sitecustomize 로드
    env["TRANSFORMERS_OFFLINE"] = "1"; env["HF_HUB_OFFLINE"] = "1"

    n_test = sum(1 for l in open(os.path.join(args.run,"data","test.jsonl")) if l.strip())
    print(f"[sim] run={args.run}  test rows={n_test}")

    # 1) 설치 시간
    t_ins, r_ins = run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], args.run, env)
    print(f"[install] {t_ins:.1f}s (제한 600s) {'OK' if t_ins<600 else 'SLOW'}  rc={r_ins.returncode}")

    # 2) 추론 시간 + 메모리
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    t_inf, r_inf = run([sys.executable, "script.py"], args.run, env)
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_mb = max(after, before) / 1024
    print(f"[infer] {t_inf:.1f}s  child_peak_RSS≈{peak_mb:.0f}MB (CPU 기준; T4는 노트북 벤치 참고)")
    print("STDOUT:", r_inf.stdout.strip()[-400:])
    if r_inf.returncode != 0:
        print("STDERR:", r_inf.stderr.strip()[-1500:]); print("=> FAIL ❌"); sys.exit(1)

    # 3) 출력 스키마
    out = os.path.join(args.run, "output", "submission.csv")
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    hdr = list(rows[0].keys()) if rows else []
    ok = (hdr == ["id","action"]) and (len(rows) == n_test) and all(r["action"] in CLASSES for r in rows)
    print(f"[schema] header={hdr} rows={len(rows)}/{n_test} all_valid={all(r['action'] in CLASSES for r in rows)}")
    print("=> PASS ✅" if ok else "=> FAIL ❌")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
