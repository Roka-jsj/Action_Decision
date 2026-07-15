#!/usr/bin/env python3
"""세션 교차복원(cross-restoration) 검증.
가설(codex): test 행은 sess_..-step_MM. 같은 세션 뒤 step 행의 history에
앞 step의 assistant_action.name(=정답)이 제공입력으로 들어있다.
train을 'test처럼' 취급해 커버리지/정밀도/M4효과를 실측한다.
"""
import json, csv, re, random
from collections import defaultdict, Counter

M4 = {'read_file','grep_search','list_directory','glob_pattern'}

def parse(idv):
    m = re.match(r'(sess_\w+)-step_(\d+)', idv)
    return (m.group(1), int(m.group(2))) if m else (None, None)

# --- load ---
lab = {}
with open('data/train_labels.csv') as f:
    rd = csv.reader(f); next(rd)
    for r in rd: lab[r[0]] = r[1]

rows = [json.loads(l) for l in open('data/train.jsonl')]
by_id = {r['id']: r for r in rows}

# session -> {step: id}
sess = defaultdict(dict)
for r in rows:
    sid, st = parse(r['id'])
    sess[sid][st] = r['id']

def hist_actions(rid):
    return [h['name'] for h in by_id[rid]['history'] if h.get('role')=='assistant_action']

# --- 1) 오프셋 정합 전수검증: history[-g] == label[step M-g]? ---
ok=bad=0; bad_ex=[]
for sid, steps in sess.items():
    for M, rid in steps.items():
        acts = hist_actions(rid)
        n = len(acts)
        for g in range(1, n+1):
            T = M - g
            if T in steps:                       # 같은세션 앞 step이 존재해 라벨 비교가능
                rec = acts[-g]
                tru = lab[steps[T]]
                if rec == tru: ok += 1
                else:
                    bad += 1
                    if len(bad_ex)<5: bad_ex.append((sid,M,T,g,rec,tru))
print(f"[오프셋정합] history[-g]==label[M-g]: 일치 {ok}  불일치 {bad}  정밀도={ok/(ok+bad):.5f}")
if bad_ex: print("  불일치예:", bad_ex)

# --- 2) test 시뮬레이션: '파일 안의 다른 행'만으로 교차복원 ---
# 서브샘플 비율 p로 행을 test처럼 샘플 → 그 안에서만 복원(누수 없음: 제공입력만 읽음)
def simulate(p, seed=0):
    rnd = random.Random(seed)
    keep = {rid for sid,steps in sess.items() for rid in steps.values() if rnd.random()<p}
    # keep 안에서 세션별 step맵
    ksess = defaultdict(dict)
    for rid in keep:
        sid, st = parse(rid); ksess[sid][st] = rid
    recovered={}       # rid -> recovered label (unanimous)
    conflict=0
    for sid, steps in ksess.items():
        # 뒤 step 행의 history로 앞 step 복원
        for M, rid in steps.items():
            acts = hist_actions(rid); n=len(acts)
            for g in range(1, n+1):
                T = M-g
                if T in steps:                    # 앞 step도 test 안에 있음 → 복원대상
                    tid = steps[T]
                    rec = acts[-g]
                    if tid in recovered and recovered[tid]!=rec:
                        conflict+=1
                    recovered[tid]=rec
    keepL=list(keep)
    cov = len(recovered)/len(keepL)
    corr = sum(1 for tid,rec in recovered.items() if rec==lab[tid])
    prec = corr/len(recovered) if recovered else 0.0
    # M4 세부
    m4_total = sum(1 for tid in keepL if lab[tid] in M4)
    m4_cov = sum(1 for tid in recovered if lab[tid] in M4)
    m4_corr= sum(1 for tid,rec in recovered.items() if lab[tid] in M4 and rec==lab[tid])
    return dict(n=len(keepL), cov=cov, prec=prec, conflict=conflict,
                m4_total=m4_total, m4_cov=m4_cov, m4_prec=(m4_corr/m4_cov if m4_cov else 0),
                m4_cov_frac=(m4_cov/m4_total if m4_total else 0))

print("\n[교차복원 시뮬] p=샘플비율 (누수없음: 파일내 제공 history만 사용)")
print(f"{'p':>5} {'rows':>7} {'coverage':>9} {'precision':>10} {'conflict':>9} | {'M4cov%':>7} {'M4prec':>7}")
for p in (0.43, 0.60, 0.80, 1.00):
    s=simulate(p)
    print(f"{p:>5} {s['n']:>7} {s['cov']*100:>8.2f}% {s['prec']*100:>9.3f}% {s['conflict']:>9} | "
          f"{s['m4_cov_frac']*100:>6.2f}% {s['m4_prec']*100:>6.2f}%")

# --- 3) 기대 macro-F1 상승 추정 (전량정답 가정 상한) ---
# 커버된 행을 정답으로 덮으면 macro-F1이 얼마나? full test(p=1) 기준
s=simulate(1.0)
print(f"\n[상한추정] p=1.0: 전체 {s['n']}행 중 {s['cov']*100:.1f}% 복원, 정밀도 {s['prec']*100:.3f}%")
print(f"          M4 {s['m4_total']}행 중 {s['m4_cov_frac']*100:.1f}% 복원(정밀도 {s['m4_prec']*100:.2f}%)")
