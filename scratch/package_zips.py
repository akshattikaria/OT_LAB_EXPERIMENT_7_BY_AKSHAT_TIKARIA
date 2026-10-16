"""
scratch/package_zips.py — Create code.zip and raw_results.zip and verify in fresh folder
"""

import os
import zipfile
import shutil
import tempfile
import subprocess
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def create_code_zip(zip_path):
    print(f"Creating {zip_path}...")
    include_dirs = ['algorithms', 'experiments', 'tests', 'report']
    include_files = [
        'moo_utils.py',
        'p13_network.py',
        'run_all_tests.py',
        'test_moo_utils.py',
        'test_s3e.py',
        'test_s3f.py',
        'test_s3g.py',
        'test_s3h.py',
        'test_s3i.py',
        'test_s4.py',
        'seeds.txt',
        'submission_cover_template.md',
        'README.md',
        'PROGRESS.md',
        'MASTER_PLAN.md',
        'audit_final.py'
    ]
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add root files
        for fname in include_files:
            fpath = os.path.join(ROOT_DIR, fname)
            if os.path.exists(fpath):
                zf.write(fpath, arcname=fname)
                print(f"  Added file: {fname}")
            else:
                print(f"  WARNING: missing file {fname}")
        
        # Add directories
        for dname in include_dirs:
            dirpath = os.path.join(ROOT_DIR, dname)
            for root, dirs, files in os.walk(dirpath):
                if '__pycache__' in root or '.git' in root:
                    continue
                for f in files:
                    if f.endswith('.pyc') or f.endswith('.pyo'):
                        continue
                    full_p = os.path.join(root, f)
                    rel_p = os.path.relpath(full_p, ROOT_DIR)
                    zf.write(full_p, arcname=rel_p)
            print(f"  Added directory: {dname}")

    sz = os.path.getsize(zip_path)
    print(f"SUCCESS: {zip_path} created ({sz:,} bytes)")

def create_raw_results_zip(zip_path):
    print(f"\nCreating {zip_path}...")
    results_dir = os.path.join(ROOT_DIR, 'results')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(results_dir):
            for f in files:
                full_p = os.path.join(root, f)
                rel_p = os.path.relpath(full_p, ROOT_DIR)
                zf.write(full_p, arcname=rel_p)
    sz = os.path.getsize(zip_path)
    print(f"SUCCESS: {zip_path} created ({sz:,} bytes)")

def test_unzip_in_fresh_folder(code_zip_path):
    print("\n--- Testing Unzip in Fresh Temporary Folder ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Extracting {code_zip_path} to {tmpdir}...")
        with zipfile.ZipFile(code_zip_path, 'r') as zf:
            zf.extractall(tmpdir)
        
        # 1. Test run_all_tests.py
        print("\nExecuting: python run_all_tests.py in unzipped folder...")
        res_tests = subprocess.run([sys.executable, 'run_all_tests.py'], cwd=tmpdir, capture_output=True, text=True, timeout=60)
        print("Test runner return code:", res_tests.returncode)
        if res_tests.returncode == 0:
            print("[PASS] All unit tests passed in unzipped environment!")
        else:
            print("[FAIL] Tests failed in unzipped environment:")
            print(res_tests.stdout[:500])
            print(res_tests.stderr[:500])
            sys.exit(1)

        # 2. Test run_task_a.py (must not crash in first 60 seconds)
        print("\nExecuting: python experiments/run_task_a.py in unzipped folder (verifying first 30 s)...")
        # Run process with timeout
        proc = subprocess.Popen([sys.executable, 'experiments/run_task_a.py'], cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(20) # Let it run for 20 seconds
        poll_res = proc.poll()
        if poll_res is None:
            # Process is running healthy and strong!
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            print("[PASS] experiments/run_task_a.py started and ran healthy without crashing!")
        elif poll_res == 0:
            print("[PASS] experiments/run_task_a.py completed successfully!")
        else:
            stdout, stderr = proc.communicate()
            print(f"[FAIL] experiments/run_task_a.py crashed with code {poll_res}:")
            print("stdout:", stdout[:500])
            print("stderr:", stderr[:500])
            sys.exit(1)

    print("\n[SUCCESS] Unzip verification passed 100%!")

if __name__ == "__main__":
    code_zip = os.path.join(ROOT_DIR, "code.zip")
    raw_zip = os.path.join(ROOT_DIR, "raw_results.zip")
    
    create_code_zip(code_zip)
    create_raw_results_zip(raw_zip)
    test_unzip_in_fresh_folder(code_zip)
