with open('report/build_claims.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if ('f"' in line or "f'" in line or 'f"""' in line) and ('\\text{' in line or '\\rm{' in line or '\\mu' in line or '\\sigma' in line or '\\lambda' in line):
        print(f"Line {i+1}: {line.strip()[:100]}")
