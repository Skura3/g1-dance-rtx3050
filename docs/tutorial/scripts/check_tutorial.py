"""Static tutorial checks. Does not execute any documented workload."""
import ast
import json
import re
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'docs/tutorial'
errors = []
links = blocks = command_count = 0
for doc in sorted(BASE.glob('*.md')):
    content = doc.read_text()
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', content):
        if '://' in target or target.startswith('#'):
            continue
        links += 1
        if not (doc.parent / target.split('#')[0]).exists():
            errors.append({'doc': doc.name, 'missing_link': target})
    for language, code in re.findall(r'```([^\n]*)\n(.*?)```', content, re.S):
        if language != 'bash':
            continue
        blocks += 1
        result = subprocess.run(['bash', '-n'], input=code, text=True, capture_output=True)
        if result.returncode:
            errors.append({'doc': doc.name, 'shell_syntax': result.stderr})
        for py in re.findall("<<'PY'\n(.*?)\nPY", code, re.S):
            try:
                ast.parse(py)
            except SyntaxError as exc:
                errors.append({'doc': doc.name, 'python_syntax': str(exc)})
        joined = code.replace(chr(92) + chr(10), ' ')
        for line in joined.splitlines():
            try:
                tokens = shlex.split(line, comments=True)
            except ValueError:
                continue
            scripts = [(i, t) for i, t in enumerate(tokens) if t.endswith('.py')
                       and t.startswith(('tools/', 'docs/tutorial/scripts/'))]
            for index, script in scripts:
                if script == 'tools/refine_contacts_tutorial.py':
                    script = 'tools/refine_contacts.py'
                path = ROOT / script
                if not path.exists():
                    errors.append({'doc': doc.name, 'missing_script': script})
                    continue
                tree = ast.parse(path.read_text())
                allowed = {'--help'}
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
                        allowed.update(a.value for a in node.args if isinstance(a, ast.Constant)
                                       and isinstance(a.value, str) and a.value.startswith('--'))
                used = {t.split('=')[0] for t in tokens[index + 1:] if t.startswith('--')}
                if used - allowed:
                    errors.append({'doc': doc.name, 'script': script, 'unknown_flags': sorted(used - allowed)})
                command_count += 1
for path in (BASE / 'scripts').glob('*.py'):
    ast.parse(path.read_text())
if links < 20 or blocks < 20 or command_count < 20:
    errors.append({'coverage': 'Unexpectedly few links, blocks or commands parsed; investigate the checker'})
report = {'markdown_files': len(list(BASE.glob('*.md'))), 'local_links_checked': links,
          'bash_blocks_syntax_checked': blocks, 'script_commands_checked': command_count,
          'errors': errors,
          'scope': 'Static only: no installation, reconstruction, simulation, training, service start or GUI.'}
(BASE / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10))
print(json.dumps(report, ensure_ascii=False, indent=2))
raise SystemExit(bool(errors))
