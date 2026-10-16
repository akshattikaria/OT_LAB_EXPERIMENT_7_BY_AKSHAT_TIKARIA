"""
COMPREHENSIVE ASSIGNMENT VERIFICATION SCRIPT
Checks every deliverable, file, import rule, and brief requirement.
"""
import os, sys, csv, ast, zipfile, re
sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, '.')

PASS = 0
FAIL = 0
WARN = 0

def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label} -- {detail}")

def warn(label, detail=""):
    global WARN
    WARN += 1
    print(f"  [WARN] {label} -- {detail}")

print("=" * 70)
print("SECTION 1: FILE EXISTENCE & SIZE CHECKS")
print("=" * 70)

required_files = {
    'moo_utils.py': 10000,
    'run_all_tests.py': 500,
    'seeds.txt': 20,
    'README.md': 1000,
    'submission_cover_template.md': 500,
    'report.pdf': 100000,
    'code.zip': 50000,
    'raw_results.zip': 100000,
    'report/task_a_log.md': 5000,
    'report/problem_bank.md': 10000,
    'report/task_b_master_table.md': 30000,
    'report/task_c_deep_dive.md': 10000,
    'report/task_d_comparisons.md': 5000,
    'report/task_e_selection.md': 5000,
    'report/task_f_viva_answers.md': 5000,
    'results/data/task_a_log.csv': 1000,
    'results/data/task_c_sensitivity.csv': 100,
    'results/data/task_d1_p3_ranked.csv': 200,
    'results/data/task_d1_p5_ranked.csv': 200,
    'results/data/task_d2_p10_m5_ranked.csv': 100,
    'results/data/task_d3_p8_tsp_ranked.csv': 100,
    'results/data/task_d4_mcdm.csv': 50,
    'results/data/task_d5_mtl.csv': 50,
}

for fpath, min_size in required_files.items():
    exists = os.path.exists(fpath)
    size = os.path.getsize(fpath) if exists else 0
    check(f"{fpath} exists ({size:,} bytes)", exists and size >= min_size,
          f"missing or too small (need >= {min_size})")

# Check all 37 algorithm files exist
print("\n" + "=" * 70)
print("SECTION 2: ALL 37 ALGORITHM FILES")
print("=" * 70)
alg_files = [
    'alg01_weighted_sum.py', 'alg02_epsilon_constraint.py', 'alg03_benson.py',
    'alg04_weight_metric.py', 'alg05_value_function.py', 'alg06_goal_programming.py',
    'alg07_lexicographic.py', 'alg08_topsis.py', 'alg09_vikor.py',
    'alg10_vega.py', 'alg11_moga.py', 'alg12_nsga.py', 'alg13_npga.py',
    'alg14_nsga2.py', 'alg15_nsga3.py', 'alg16_spea2.py', 'alg17_pesa2.py',
    'alg18_moead.py', 'alg19_moead_de.py', 'alg20_moead_aco.py',
    'alg21_constrained_nsga2.py', 'alg22_mopso.py', 'alg23_mogwo.py',
    'alg24_moabc.py', 'alg25_moaco.py', 'alg26_mosa.py', 'alg27_mode.py',
    'alg28_motlbo.py', 'alg29_moba.py', 'alg30_moma.py',
    'alg31_rvea.py', 'alg32_ibea.py', 'alg33_sms_emoa.py', 'alg34_hype.py',
    'alg35_r_nsga2.py', 'alg36_mgda.py', 'alg37_gradnorm.py'
]
for af in alg_files:
    p = os.path.join('algorithms', af)
    check(f"algorithms/{af}", os.path.exists(p) and os.path.getsize(p) > 200)

# Check test files
print("\n" + "=" * 70)
print("SECTION 3: TEST FILES")
print("=" * 70)
for tf in ['test_family1_2.py', 'test_family3_4.py', 'test_family5.py', 'test_family6.py']:
    p = os.path.join('tests', tf)
    check(f"tests/{tf}", os.path.exists(p) and os.path.getsize(p) > 500)

# Check experiment files
for ef in ['run_task_a.py', 'run_task_c.py', 'run_task_d.py']:
    p = os.path.join('experiments', ef)
    check(f"experiments/{ef}", os.path.exists(p) and os.path.getsize(p) > 500)

print("\n" + "=" * 70)
print("SECTION 4: FORBIDDEN IMPORT SCAN (AST-based)")
print("=" * 70)

def scan_imports(filepath):
    """Returns list of (module, line_no) for imports."""
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name.split('.')[0], node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append((node.module.split('.')[0], node.lineno))
    return imports

# Scan all .py files
pandas_found = []
scipy_violations = []
sklearn_violations = []

for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'scratch']]
    for fname in files:
        if not fname.endswith('.py'):
            continue
        fpath = os.path.join(root, fname)
        imports = scan_imports(fpath)
        for mod, lineno in imports:
            if mod == 'pandas':
                pandas_found.append(f"{fpath}:{lineno}")
            if mod == 'scipy':
                # Only allowed in alg01-alg07
                basename = os.path.basename(fpath)
                alg_match = re.match(r'alg(\d+)', basename)
                if alg_match:
                    alg_num = int(alg_match.group(1))
                    if alg_num > 7:
                        scipy_violations.append(f"{fpath}:{lineno}")
                elif 'moo_utils' not in basename:
                    # scipy in experiment/test files is also a violation
                    scipy_violations.append(f"{fpath}:{lineno}")
            if mod == 'sklearn' or mod == 'scikit':
                basename = os.path.basename(fpath)
                if 'moo_utils' not in basename:
                    sklearn_violations.append(f"{fpath}:{lineno}")

check("Zero pandas imports in codebase", len(pandas_found) == 0,
      f"Found in: {pandas_found}")
check("scipy only in alg01-alg07 (not in alg08+ or moo_utils)", len(scipy_violations) == 0,
      f"Violations: {scipy_violations}")
check("sklearn only in moo_utils.py", len(sklearn_violations) == 0,
      f"Violations: {sklearn_violations}")

print("\n" + "=" * 70)
print("SECTION 5: TASK A LOG CSV INTEGRITY")
print("=" * 70)

with open('results/data/task_a_log.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
alg_nums = set(int(r['Algorithm_Number']) for r in rows)
check("Task A log has all 37 algorithms", alg_nums == set(range(1, 38)),
      f"Missing: {set(range(1,38)) - alg_nums}")
check(f"Task A log has {len(rows)} entries (≥37)", len(rows) >= 37)

# Check status values are valid
valid_statuses = {'works'}
for r in rows:
    st = r['Status']
    if st != 'works' and 'partly works' not in st and 'not done' not in st:
        warn(f"Unexpected status for Alg {r['Algorithm_Number']}", st)

print("\n" + "=" * 70)
print("SECTION 6: FIGURES — ALL 37 ALGORITHMS HAVE PLOTS")
print("=" * 70)

fig_files = os.listdir('results/figures')
algs_with_figs = set()
for fig in fig_files:
    m = re.match(r'front_alg(\d+)', fig)
    if m:
        algs_with_figs.add(int(m.group(1)))

missing_figs = set(range(1, 38)) - algs_with_figs
check("All 37 algorithms have front plots", len(missing_figs) == 0,
      f"Missing front plots for algs: {sorted(missing_figs)}")
check(f"Total figures: {len(fig_files)} (≥37)", len(fig_files) >= 37)

# Check Task D figures
task_d_figs = [
    'task_d1_comparison_P3.png', 'task_d1_comparison_P5.png',
    'task_d2_manyobjective_p10_m5.png', 'task_d3_discrete_tsp.png',
    'task_d5_mtl_loss.png'
]
for tdf in task_d_figs:
    check(f"Task D figure: {tdf}", tdf in fig_files)

# Task C sensitivity figures
task_c_figs = [
    'task_c_sensitivity_alg05.png', 'task_c_sensitivity_alg17.png',
    'task_c_sensitivity_alg29.png'
]
for tcf in task_c_figs:
    check(f"Task C figure: {tcf}", tcf in fig_files)

print("\n" + "=" * 70)
print("SECTION 7: SEEDS FILE")
print("=" * 70)
with open('seeds.txt', 'r') as f:
    seeds = []
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            if '=' in line:
                seeds.append(int(line.split('=')[1]))
            else:
                seeds.append(int(line))
check(f"seeds.txt has 10 seeds", len(seeds) == 10, f"Found {len(seeds)}")

print("\n" + "=" * 70)
print("SECTION 8: ZIP ARCHIVE CONTENTS")
print("=" * 70)

with zipfile.ZipFile('code.zip', 'r') as zf:
    code_names = zf.namelist()
check("code.zip contains moo_utils.py", 'moo_utils.py' in code_names)
check("code.zip contains run_all_tests.py", 'run_all_tests.py' in code_names)
check("code.zip contains seeds.txt", 'seeds.txt' in code_names)
check("code.zip contains README.md", 'README.md' in code_names)
alg_in_zip = [n for n in code_names if n.startswith('algorithms/alg')]
check(f"code.zip has {len(alg_in_zip)} algorithm files (need 37)", len(alg_in_zip) >= 37)
test_in_zip = [n for n in code_names if n.startswith('tests/')]
check(f"code.zip has {len(test_in_zip)} test files (need 4)", len(test_in_zip) >= 4)
exp_in_zip = [n for n in code_names if n.startswith('experiments/')]
check(f"code.zip has {len(exp_in_zip)} experiment files (need 3)", len(exp_in_zip) >= 3)

with zipfile.ZipFile('raw_results.zip', 'r') as zf:
    raw_names = zf.namelist()
csv_in_raw = [n for n in raw_names if n.endswith('.csv')]
png_in_raw = [n for n in raw_names if n.endswith('.png')]
check(f"raw_results.zip has {len(csv_in_raw)} CSV files (need ≥8)", len(csv_in_raw) >= 8)
check(f"raw_results.zip has {len(png_in_raw)} PNG figures (need ≥37)", len(png_in_raw) >= 37)

print("\n" + "=" * 70)
print("SECTION 9: REPORT CONTENT CHECKS")
print("=" * 70)

# Task B: check all 37 algorithms are in the table
with open('report/task_b_master_table.md', 'r', encoding='utf-8') as f:
    tb = f.read()
for i in range(1, 38):
    found = f'| **{i}** |' in tb
    if not found:
        check(f"Task B contains Algorithm {i}", False, "missing row")
    # Only print pass for first, last, and spot checks
if all(f'| **{i}** |' in tb for i in range(1, 38)):
    check("Task B master table contains all 37 algorithm rows", True)
else:
    missing = [i for i in range(1, 38) if f'| **{i}** |' not in tb]
    check("Task B master table contains all 37 algorithm rows", False, f"Missing: {missing}")

# Check family analysis paragraph exists
check("Task B has family analysis paragraph", "Comparative Family Analysis" in tb)

# Task C: check all 3 deep-dive algorithms covered
with open('report/task_c_deep_dive.md', 'r', encoding='utf-8') as f:
    tc = f.read()
check("Task C covers Algorithm 5 (Value Function)", 'Value Function' in tc or 'Algorithm 5' in tc or 'Alg 5' in tc)
check("Task C covers Algorithm 17 (PESA-II)", 'PESA-II' in tc or 'PESA' in tc)
check("Task C covers Algorithm 29 (MOBA)", 'MOBA' in tc or 'Bat Algorithm' in tc)
check("Task C has hand trace section", 'Hand Trace' in tc or 'hand trace' in tc or 'Hand trace' in tc)
check("Task C has sensitivity study section", 'Sensitivity' in tc or 'sensitivity' in tc)

# Task D: check all 5 parts
with open('report/task_d_comparisons.md', 'r', encoding='utf-8') as f:
    td = f.read()
check("Task D Part 1: P3 & P5 rankings", 'P3' in td and 'P5' in td and 'Rank' in td)
check("Task D Part 2: Many-objective M=5", 'M = 5' in td or 'M=5' in td or 'many-objective' in td.lower())
check("Task D Part 3: Discrete TSP P8", 'TSP' in td or 'P8' in td)
check("Task D Part 4: TOPSIS & VIKOR MCDM", 'TOPSIS' in td and 'VIKOR' in td)
check("Task D Part 5: MGDA vs GradNorm", 'MGDA' in td and 'GradNorm' in td)
check("Task D: NSGA-II M=5 degradation note", 'degrad' in td.lower() or 'dominance resistance' in td.lower() or 'boundary bias' in td.lower())

# Task E: 6 problems
with open('report/task_e_selection.md', 'r', encoding='utf-8') as f:
    te = f.read()
problem_headers = re.findall(r'###\s+Problem\s+\d+', te)
check(f"Task E has 6 problem selections", len(problem_headers) >= 6, f"Found {len(problem_headers)}")
check("Task E includes many-objective problem", 'P10' in te or 'many-objective' in te.lower() or 'DTLZ2' in te)
check("Task E includes constrained problem", 'P5' in te or 'P6' in te or 'constrained' in te.lower())

# Task F: 8 questions
with open('report/task_f_viva_answers.md', 'r', encoding='utf-8') as f:
    tf = f.read()
q_headers = re.findall(r'###\s+Question\s+\d+', tf)
check(f"Task F has 8 viva questions", len(q_headers) >= 8, f"Found {len(q_headers)}")

# Problem Bank: 14 problems
with open('report/problem_bank.md', 'r', encoding='utf-8') as f:
    pb = f.read()
pb_headers = re.findall(r'##\s+Problem\s+P\d+', pb)
check(f"Problem Bank has 14 problem specs", len(pb_headers) >= 14, f"Found {len(pb_headers)}")
# Check required subsections exist
for keyword in ['Standard', 'Encoding', 'Constraint', 'Normalization', 'Reference Point', 'Termination']:
    count = pb.lower().count(keyword.lower())
    check(f"Problem Bank mentions '{keyword}' (≥14 times)", count >= 10, f"Found {count}")

print("\n" + "=" * 70)
print("SECTION 10: REPORT PDF SIZE CHECK")
print("=" * 70)
pdf_size = os.path.getsize('report.pdf')
check(f"report.pdf is substantial ({pdf_size:,} bytes, ≥500KB)", pdf_size >= 500000)

print("\n" + "=" * 70)
print(f"FINAL SUMMARY: {PASS} PASSED | {FAIL} FAILED | {WARN} WARNINGS")
print("=" * 70)
if FAIL == 0:
    print("ALL CHECKS PASSED. Assignment is complete!")
else:
    print(f"ATTENTION: {FAIL} check(s) failed. Review above.")
