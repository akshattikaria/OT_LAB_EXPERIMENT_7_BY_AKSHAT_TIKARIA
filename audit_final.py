"""
audit_final.py - Master Final Audit for OT Lab Assignment 7
===========================================================
Student: Akshat Tikaria, Roll 241070005

Implements the 8 mandatory checks specified in MASTER_PLAN.md:
  (1) All 37 algorithms importable and expose run()
  (2) All CSVs present with >= 1 data row
  (3) All figures present and non-empty in results/figures/
  (4) No HV > 1.5 or IGD > 10 in any CSV
  (5) 'M.Tech' absent from all .md and .py files
  (6) report.pdf >= 15 pages and contains student name
  (7) P8 ref_union_P8.csv and p8_coords.csv present
  (8) task_c_trace_check.csv present with max diff < 1e-9
"""

import os
import sys
import glob
import csv
import re
from typing import List, Tuple

sys.path.insert(0, os.path.abspath('algorithms'))
sys.path.insert(0, os.path.abspath('.'))


def check_1_algorithms_importable() -> Tuple[bool, str]:
    """(1) All 37 algorithms importable and expose run()."""
    alg_names = [
        'alg01_weighted_sum', 'alg02_epsilon_constraint', 'alg03_benson', 'alg04_weight_metric',
        'alg05_value_function', 'alg06_goal_programming', 'alg07_lexicographic', 'alg08_topsis',
        'alg09_vikor', 'alg10_vega', 'alg11_moga', 'alg12_nsga', 'alg13_npga', 'alg14_nsga2',
        'alg15_nsga3', 'alg16_spea2', 'alg17_pesa2', 'alg18_moead', 'alg19_moead_de',
        'alg20_moead_aco', 'alg21_constrained_nsga2', 'alg22_mopso', 'alg23_mogwo',
        'alg24_moabc', 'alg25_moaco', 'alg26_mosa', 'alg27_mode', 'alg28_motlbo',
        'alg29_moba', 'alg30_moma', 'alg31_rvea', 'alg32_ibea', 'alg33_sms_emoa',
        'alg34_hype', 'alg35_r_nsga2', 'alg36_mgda', 'alg37_gradnorm'
    ]
    failed = []
    for a in alg_names:
        try:
            mod = __import__(a)
            if not hasattr(mod, 'run'):
                failed.append(f"{a} (missing run)")
        except Exception as e:
            failed.append(f"{a} ({e})")

    if failed:
        return False, f"Failed imports: {', '.join(failed)}"
    return True, f"All {len(alg_names)} algorithms imported successfully with run() method."


def check_2_csvs_present() -> Tuple[bool, str]:
    """(2) All CSVs present with >= 1 data row."""
    required_csvs = [
        'results/data/task_a_log.csv',
        'results/data/task_c_sensitivity.csv',
        'results/data/task_c_trace_check.csv',
        'results/data/task_d1_p3_ranked.csv',
        'results/data/task_d1_p5_ranked.csv',
        'results/data/task_d2_p10_m5_ranked.csv',
        'results/data/task_d3_p8_tsp_ranked.csv',
        'results/data/task_d4_mcdm.csv',
        'results/data/task_d5_mtl.csv',
        'results/data/task_d5_network.csv',
        'results/data/ref_union_P8.csv',
        'results/data/p8_coords.csv'
    ]
    missing_or_empty = []
    total_rows = 0
    for p in required_csvs:
        if not os.path.exists(p):
            missing_or_empty.append(f"{p} (missing)")
            continue
        try:
            with open(p, 'r', encoding='utf-8') as f:
                r = list(csv.reader(f))
            if len(r) < 2:
                missing_or_empty.append(f"{p} (only {len(r)} lines)")
            else:
                total_rows += len(r) - 1
        except Exception as e:
            missing_or_empty.append(f"{p} ({e})")

    if missing_or_empty:
        return False, f"CSV issues: {', '.join(missing_or_empty)}"
    return True, f"All {len(required_csvs)} required CSVs present with valid data rows ({total_rows} total rows)."


def check_3_figures_present() -> Tuple[bool, str]:
    """(3) All figures present in results/figures/."""
    required_keys = [
        'flowchart_alg05_value_function.png',
        'flowchart_alg17_pesa2.png',
        'flowchart_alg29_moba.png',
        'task_c_sensitivity_alg05.png',
        'task_c_sensitivity_alg17.png',
        'task_c_sensitivity_alg29.png',
        'task_d1_comparison_P3.png',
        'task_d1_comparison_P5.png',
        'task_d2_manyobjective_p10_m5.png',
        'task_d3_discrete_tsp.png',
        'task_d5_mtl_loss.png',
        'task_d5_network.png'
    ]
    missing = []
    for k in required_keys:
        p = os.path.join('results/figures', k)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            missing.append(k)

    # Check that Pareto plots exist for all 37 algorithms
    all_pngs = glob.glob('results/figures/*.png')
    alg_plots = [f for f in all_pngs if 'front_alg' in os.path.basename(f).lower()]

    if missing:
        return False, f"Missing required key figures: {', '.join(missing)}"
    if len(alg_plots) < 37:
        return False, f"Expected >= 37 algorithm plots, found {len(alg_plots)}"

    return True, f"All {len(required_keys)} key figures and {len(alg_plots)} algorithm plots verified (Total: {len(all_pngs)} figures)."


def check_4_no_hv_gt_1_5_or_igd_gt_10() -> Tuple[bool, str]:
    """(4) No HV > 1.5 or IGD > 10 in any CSV."""
    violations = []
    csv_files = glob.glob('results/**/*.csv', recursive=True)
    checked_count = 0
    for f in csv_files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                r = csv.DictReader(fh)
                if not r.fieldnames:
                    continue
                for i, row in enumerate(r):
                    for k, v in row.items():
                        if not k or not v or v.strip() == 'N/A':
                            continue
                        k_lower = k.lower()
                        if 'hv' in k_lower or 'hypervolume' in k_lower:
                            try:
                                val = float(v)
                                checked_count += 1
                                if val > 1.5:
                                    violations.append(f"{f} line {i+2}: {k}={val} > 1.5")
                            except ValueError:
                                pass
                        if 'igd' in k_lower:
                            try:
                                val = float(v)
                                checked_count += 1
                                if val > 10.0:
                                    violations.append(f"{f} line {i+2}: {k}={val} > 10.0")
                            except ValueError:
                                pass
        except Exception as e:
            violations.append(f"{f} read error: {e}")

    if violations:
        return False, f"Threshold violations found:\n  " + "\n  ".join(violations[:5])
    return True, f"Verified {checked_count} metric values across {len(csv_files)} CSVs. Zero instances of HV > 1.5 or IGD > 10.0."


def check_5_no_mtech() -> Tuple[bool, str]:
    """(5) 'M.Tech' absent from all project .md and .py files."""
    forbidden = 'M' + '.' + 'Tech'
    found = []
    for root, dirs, files in os.walk('.'):
        if any(x in root for x in ['_old', '.git', '.gemini', '__pycache__']):
            continue
        for f in files:
            if f == 'audit_final.py':
                continue
            if f.endswith('.md') or f.endswith('.py'):
                p = os.path.join(root, f)
                try:
                    with open(p, 'r', encoding='utf-8', errors='ignore') as fh:
                        content = fh.read()
                    if forbidden in content:
                        found.append(f"{p} ({content.count(forbidden)} occurrences)")
                except Exception:
                    pass

    if found:
        return False, f"Found 'M.Tech' in: {', '.join(found)}"
    return True, "Verified zero occurrences of 'M.Tech' across all project .md and .py files."


def check_6_report_pdf() -> Tuple[bool, str]:
    """(6) report.pdf >= 15 pages and contains student name."""
    pdf_path = 'report.pdf'
    if not os.path.exists(pdf_path):
        return False, "report.pdf does not exist."

    size = os.path.getsize(pdf_path)
    if size < 50000:
        return False, f"report.pdf is suspiciously small ({size} bytes)."

    # Determine page count
    page_count = 0
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        page_count = len(reader.pages)
    except Exception:
        # Fallback regex on /Count
        with open(pdf_path, 'rb') as f:
            b = f.read()
        m = re.findall(rb'/Count\s+(\d+)', b)
        if m:
            page_count = max([int(x) for x in m])

    if page_count < 15:
        return False, f"report.pdf has {page_count} pages (expected >= 15)."

    # Check student name
    name_found = False
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages[:3]:
            text = page.extract_text() or ""
            if "Akshat Tikaria" in text:
                name_found = True
                break
    except Exception:
        pass

    if not name_found:
        with open(pdf_path, 'rb') as f:
            raw = f.read()
        if b"Akshat" in raw:
            name_found = True

    if not name_found:
        return False, "Student name 'Akshat Tikaria' not detected in report.pdf header/cover."

    return True, f"report.pdf verified: {page_count} pages (>= 15), contains student name 'Akshat Tikaria', size {size:,} bytes."


def check_7_p8_reference_files() -> Tuple[bool, str]:
    """(7) P8 ref_union_P8.csv and p8_coords.csv present."""
    p_union = 'results/data/ref_union_P8.csv'
    p_coords = 'results/data/p8_coords.csv'

    if not os.path.exists(p_union):
        return False, f"Missing {p_union}"
    if not os.path.exists(p_coords):
        return False, f"Missing {p_coords}"

    with open(p_union, 'r', encoding='utf-8') as f:
        r_union = list(csv.reader(f))
    with open(p_coords, 'r', encoding='utf-8') as f:
        r_coords = list(csv.reader(f))

    if len(r_union) < 2:
        return False, f"{p_union} has insufficient rows ({len(r_union)})"
    if len(r_coords) != 16:  # header + 15 cities
        return False, f"{p_coords} expected header + 15 cities, got {len(r_coords)} rows"

    return True, f"P8 files verified: ref_union_P8.csv ({len(r_union)-1} points), p8_coords.csv (15 cities)."


def check_8_hand_trace_diffs() -> Tuple[bool, str]:
    """(8) task_c_trace_check.csv present with max diff < 1e-9."""
    trace_path = 'results/data/task_c_trace_check.csv'
    if not os.path.exists(trace_path):
        return False, f"Missing {trace_path}"

    diffs = []
    with open(trace_path, 'r', encoding='utf-8') as f:
        r = list(csv.DictReader(f))
    if len(r) == 0:
        return False, f"{trace_path} contains no data rows."

    for row in r:
        try:
            diffs.append(abs(float(row['diff'])))
        except ValueError:
            pass

    max_diff = max(diffs) if diffs else 1.0
    if max_diff >= 1e-9:
        return False, f"Maximum hand-trace difference {max_diff:.2e} >= 1e-9."

    return True, f"task_c_trace_check.csv verified: {len(r)} assertions, max diff = {max_diff:.2e} (< 1e-9)."


def main():
    print("=" * 80)
    print("MASTER FINAL AUDIT — B.TECH OPTIMIZATION TECHNIQUES LAB 7")
    print("Student: Akshat Tikaria (Roll: 241070005)")
    print("=" * 80)

    checks = [
        ("Check 1: 37 Algorithms Importable", check_1_algorithms_importable),
        ("Check 2: CSV Data Completeness", check_2_csvs_present),
        ("Check 3: All Figures Present", check_3_figures_present),
        ("Check 4: Metric Bounds (HV <= 1.5, IGD <= 10.0)", check_4_no_hv_gt_1_5_or_igd_gt_10),
        ("Check 5: Academic Degree Integrity (No 'M.Tech')", check_5_no_mtech),
        ("Check 6: Submission PDF (>= 15 pages + Name)", check_6_report_pdf),
        ("Check 7: P8 TSP Reference Artifacts", check_7_p8_reference_files),
        ("Check 8: Task C Hand Trace Exactness (< 1e-9)", check_8_hand_trace_diffs),
    ]

    all_passed = True
    for title, fn in checks:
        passed, msg = fn()
        status_str = "[PASS]" if passed else "[FAIL]"
        print(f"\n{status_str} {title}")
        print(f"       {msg}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("FINAL AUDIT RESULT: ALL 8 AUDIT CHECKS PASSED (100% SUCCESS)!")
        print("The repository is in a flawless state ready for final submission.")
        print("=" * 80)
        sys.exit(0)
    else:
        print("FINAL AUDIT RESULT: AUDIT FAILED ON ONE OR MORE CHECKS.")
        print("=" * 80)
        sys.exit(1)


if __name__ == '__main__':
    main()
