import zipfile
import os

def create_code_zip():
    zip_path = 'code.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # root files
        for f in ['moo_utils.py', 'run_all_tests.py', 'seeds.txt', 'README.md', 'submission_cover_template.md']:
            if os.path.exists(f):
                zf.write(f, arcname=f)
                print(f"Added to code.zip: {f}")
        
        # directories
        for folder in ['algorithms', 'experiments', 'tests']:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.endswith('.py'):
                        full_path = os.path.join(root, file)
                        arcname = os.path.relpath(full_path, '.')
                        zf.write(full_path, arcname=arcname)
                        print(f"Added to code.zip: {arcname}")
    print(f"Successfully generated {zip_path} (Size: {os.path.getsize(zip_path):,} bytes)")

def create_raw_results_zip():
    zip_path = 'raw_results.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for folder in [os.path.join('results', 'data'), os.path.join('results', 'figures')]:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, '.')
                    zf.write(full_path, arcname=arcname)
    print(f"Successfully generated {zip_path} (Size: {os.path.getsize(zip_path):,} bytes)")

if __name__ == '__main__':
    create_code_zip()
    create_raw_results_zip()
