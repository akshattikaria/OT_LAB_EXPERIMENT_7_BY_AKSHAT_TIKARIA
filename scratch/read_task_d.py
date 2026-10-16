import csv

files = [
    'results/data/task_d1_p3_ranked.csv',
    'results/data/task_d1_p5_ranked.csv',
    'results/data/task_d2_p10_m5_ranked.csv',
    'results/data/task_d3_p8_tsp_ranked.csv',
    'results/data/task_d4_mcdm.csv',
    'results/data/task_d5_mtl.csv'
]

for fpath in files:
    print("=" * 60)
    print("FILE:", fpath)
    with open(fpath, 'r', encoding='utf-8') as f:
        reader = list(csv.reader(f))
        for row in reader:
            print(" | ".join(row))
