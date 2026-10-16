import csv

with open('results/data/task_a_log.csv', mode='r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

alg_dict = {}
for r in rows:
    num = int(r['Algorithm_Number'])
    if num not in alg_dict:
        alg_dict[num] = []
    alg_dict[num].append(r)

for num in range(1, 38):
    runs = alg_dict.get(num, [])
    run_str = ", ".join([f"{x['Problem']}: HV={x['HV']}, IGD={x['IGD']}, time={x['Runtime_sec']}s" for x in runs])
    print(f"Alg {num:02d} ({runs[0]['Algorithm']}): {run_str}")
