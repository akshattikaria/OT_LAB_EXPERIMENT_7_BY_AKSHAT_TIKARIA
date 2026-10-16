import os, re, subprocess

report_dir = 'report'
figures_dir = os.path.abspath('results/figures')

# Order of markdown files for the combined report
md_files = [
    ('Cover Page', None),  # We'll inline the cover
    ('Task A — Implementation Log & Unit Test Summary', 'task_a_log.md'),
    ('Problem Bank — P1 to P14 Specifications', 'problem_bank.md'),
    ('Task B — Master Comparison Table (37 Algorithms)', 'task_b_master_table.md'),
    ('Task C — Deep Dive (Algorithms 5, 17, 29)', 'task_c_deep_dive.md'),
    ('Task D — Empirical Comparative Benchmarking', 'task_d_comparisons.md'),
    ('Task E — Algorithm Selection Reasoning', 'task_e_selection.md'),
    ('Task F — Viva & Code Walkthrough Answers', 'task_f_viva_answers.md'),
]

def md_to_html_basic(md_text):
    """Minimal markdown to HTML. Handles headers, bold, italic, tables, code blocks, lists, paragraphs."""
    lines = md_text.split('\n')
    html_parts = []
    in_table = False
    in_code = False
    in_list = False
    code_buf = []
    
    for line in lines:
        # Code blocks
        if line.strip().startswith('```'):
            if in_code:
                html_parts.append('<pre><code>' + '\n'.join(code_buf) + '</code></pre>')
                code_buf = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_buf.append(line.replace('<', '&lt;').replace('>', '&gt;'))
            continue
        
        # Close list if line is not a list item
        if in_list and not re.match(r'^\s*[-*\d]+[.)]\s', line) and line.strip() != '':
            html_parts.append('</ul>')
            in_list = False

        # Empty line
        if line.strip() == '':
            if in_table:
                html_parts.append('</tbody></table>')
                in_table = False
            continue
        
        # Headers
        hm = re.match(r'^(#{1,6})\s+(.*)', line)
        if hm:
            level = len(hm.group(1))
            text = hm.group(2)
            text = inline_format(text)
            html_parts.append(f'<h{level}>{text}</h{level}>')
            continue
        
        # Horizontal rule
        if re.match(r'^---+$', line.strip()):
            html_parts.append('<hr>')
            continue
        
        # Table
        if '|' in line and not line.strip().startswith('```'):
            cells = [c.strip() for c in line.split('|')]
            cells = [c for c in cells if c != '']
            # Skip separator rows
            if all(re.match(r'^[-:]+$', c) for c in cells):
                continue
            if not in_table:
                html_parts.append('<table><thead><tr>')
                for c in cells:
                    html_parts.append(f'<th>{inline_format(c)}</th>')
                html_parts.append('</tr></thead><tbody>')
                in_table = True
            else:
                html_parts.append('<tr>')
                for c in cells:
                    html_parts.append(f'<td>{inline_format(c)}</td>')
                html_parts.append('</tr>')
            continue
        
        # List items
        lm = re.match(r'^\s*[-*]\s+(.*)', line)
        if not lm:
            lm = re.match(r'^\s*\d+[.)]\s+(.*)', line)
        if lm:
            if not in_list:
                html_parts.append('<ul>')
                in_list = True
            html_parts.append(f'<li>{inline_format(lm.group(1))}</li>')
            continue
        
        # Paragraph
        html_parts.append(f'<p>{inline_format(line)}</p>')
    
    if in_table:
        html_parts.append('</tbody></table>')
    if in_list:
        html_parts.append('</ul>')
    if in_code:
        html_parts.append('<pre><code>' + '\n'.join(code_buf) + '</code></pre>')
    
    return '\n'.join(html_parts)

def inline_format(text):
    """Handle inline markdown: bold, italic, code, math, links, <br>."""
    # Escape HTML angle brackets in non-tag contexts (but keep <br>)
    text = text.replace('<br>', '§BR§')
    
    # Display math $$...$$
    text = re.sub(r'\$\$(.*?)\$\$', r'\\[\1\\]', text)
    
    # Inline math $...$  (but not \$)
    text = re.sub(r'(?<!\\)\$(.+?)(?<!\\)\$', r'\\(\1\\)', text)
    
    # Bold **text**
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic *text*
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    # Inline code `text`
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    
    # Image references to figures
    text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img src="\2" alt="\1" style="max-width:100%">', text)
    
    text = text.replace('§BR§', '<br>')
    return text

# Build combined HTML
cover_html = """
<div style="text-align:center; padding-top:120px;">
<h1 style="font-size:28px;">Multi-Objective Optimization:<br>37 Algorithms, Trade-offs and Applications</h1>
<h2 style="font-size:20px; color:#555;">M.Tech Optimization Techniques — Lab Assignment 7</h2>
<br><br>
<p style="font-size:16px;"><strong>Student Name:</strong> Akshat Tikaria</p>
<p style="font-size:16px;"><strong>Roll Number:</strong> 241070005 (R = 05)</p>
<p style="font-size:16px;"><strong>Deep-Dive Algorithms:</strong><br>
Algorithm 5 (Value Function) · Algorithm 17 (PESA-II) · Algorithm 29 (MOBA)</p>
<br><br>
<p style="font-size:14px; color:#777;">Complete report covering Tasks A through F:<br>
Implementation of 37 algorithms, 14 problem benchmarks, 47 unit tests,<br>
empirical comparisons, and algorithm selection analysis.</p>
</div>
<div style="page-break-after: always;"></div>
"""

body_parts = [cover_html]

for title, fname in md_files:
    if fname is None:
        continue
    fpath = os.path.join(report_dir, fname)
    with open(fpath, 'r', encoding='utf-8') as f:
        md_text = f.read()
    
    section_html = md_to_html_basic(md_text)
    # Replace figure references with absolute paths for Edge
    section_html = section_html.replace('results/figures/', figures_dir.replace('\\', '/') + '/')
    
    body_parts.append(f'<div style="page-break-before: always;">')
    body_parts.append(section_html)
    body_parts.append('</div>')

full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script>
MathJax = {{
  tex: {{ inlineMath: [['\\\\(','\\\\)']], displayMath: [['\\\\[','\\\\]']] }},
  startup: {{ pageReady: () => {{ return MathJax.startup.defaultPageReady(); }} }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
<style>
@page {{ margin: 20mm 15mm; }}
body {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 11px; line-height: 1.5; color: #222; max-width: 100%; }}
h1 {{ font-size: 20px; border-bottom: 2px solid #333; padding-bottom: 4px; margin-top: 20px; }}
h2 {{ font-size: 16px; color: #444; margin-top: 16px; }}
h3 {{ font-size: 14px; color: #555; margin-top: 12px; }}
h4 {{ font-size: 12px; color: #666; }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9px; }}
th {{ background: #f0f0f0; border: 1px solid #ccc; padding: 4px 6px; text-align: left; font-weight: bold; }}
td {{ border: 1px solid #ddd; padding: 3px 6px; vertical-align: top; }}
tr:nth-child(even) {{ background: #fafafa; }}
code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 10px; }}
pre {{ background: #f8f8f8; border: 1px solid #ddd; padding: 8px; overflow-x: auto; font-size: 9px; }}
hr {{ border: none; border-top: 1px solid #ccc; margin: 16px 0; }}
img {{ max-width: 90%; height: auto; display: block; margin: 8px auto; }}
ul, ol {{ margin: 4px 0; padding-left: 20px; }}
li {{ margin: 2px 0; }}
p {{ margin: 4px 0; }}
strong {{ color: #111; }}
</style>
</head>
<body>
{''.join(body_parts)}
</body>
</html>"""

html_path = os.path.abspath('report_combined.html')
pdf_path = os.path.abspath('report.pdf')

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(full_html)
print(f"Generated combined HTML: {html_path} ({len(full_html):,} bytes)")

# Convert to PDF via Edge headless
cmd = [
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    '--headless',
    '--disable-gpu',
    '--run-all-compositor-stages-before-draw',
    '--virtual-time-budget=5000',
    f'--print-to-pdf={pdf_path}',
    '--no-pdf-header-footer',
    html_path
]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
print(f"Edge exit code: {result.returncode}")
if os.path.exists(pdf_path):
    print(f"SUCCESS: report.pdf generated ({os.path.getsize(pdf_path):,} bytes)")
else:
    print("FAILED: report.pdf not found")
    print("stderr:", result.stderr[:500] if result.stderr else "none")
