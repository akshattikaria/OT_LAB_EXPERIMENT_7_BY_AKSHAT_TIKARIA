import csv

with open('results/data/task_a_log.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

by_alg = {}
for r in rows:
    num = int(r['Algorithm_Number'])
    by_alg.setdefault(num, []).append(r)

for num in range(1, 38):
    runs = by_alg.get(num, [])
    alg_name = runs[0]['Algorithm'] if runs else 'Unknown'
    summary = '; '.join([f"{r['Problem']} (Evals={r['Evaluations']}, HV={r['HV']}, Spacing={r['Spacing']}, IGD={r['IGD']}, time={r['Runtime_sec']}s, status={r['Status']})" for r in runs])
    print(f"Alg {num:02d} ({alg_name}): {summary}")
