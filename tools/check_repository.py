import ast
import hashlib
import json
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.local', '.git', '.pytest_cache', '.ruff_cache', '__pycache__', 'outputs',
            'checkpoints', 'build', 'dist', 'data'}


def public_files():
    for current, children, names in os.walk(ROOT, followlinks=False):
        children[:] = [name for name in children if name not in EXCLUDED
                       and not name.startswith('.venv') and not name.endswith('.egg-info')
                       and not (Path(current) / name).is_symlink()]
        for name in sorted(names):
            path = Path(current) / name
            if path.is_file() and not path.is_symlink():
                yield path
    yield ROOT / 'data/README.md'


def check():
    errors = []
    for name in ('.local', 'crossinterest_formal', 'handoff_records'):
        if (ROOT / name).exists():
            errors.append(f'Private workspace embedded in standalone repository: {name}')
    for name in ('src', 'tests', 'configs', 'docs', 'tools', '.github'):
        directory = ROOT / name
        for path in (directory, *directory.rglob('*')):
            if path.is_symlink():
                errors.append(f'Source tree symlink: {path.relative_to(ROOT)}')
    files = list(public_files())
    content_paths = {}
    link_count = 0
    python_count = 0
    for path in files:
        relative = path.relative_to(ROOT)
        if path.stat().st_size and path.name != '__init__.py':
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            if checksum in content_paths:
                errors.append(f'Duplicate public content: {content_paths[checksum]} and {relative}')
            else:
                content_paths[checksum] = relative
        if path.stat().st_size > 5 * 1024 * 1024:
            errors.append(f'Large public file: {relative}')
        if path.suffix in {'.parquet', '.safetensors', '.pt', '.pth', '.bin', '.jsonl'}:
            errors.append(f'Private artifact in public tree: {relative}')
        if path.suffix not in {'.py', '.md', '.toml', '.yaml', '.yml', '.txt'}:
            continue
        text = path.read_text()
        if path.name != 'check_repository.py':
            for pattern in (r'/home/[A-Za-z0-9_-]+/', r'/subscriptions/[0-9a-f-]{36}',
                            r'https://[a-z0-9]+\.blob\.core\.windows\.net',
                            r'(?i)(?:sig|api_key|access_token)=[A-Za-z0-9%+/]{20,}'):
                if re.search(pattern, text):
                    errors.append(f'Machine-specific or sensitive value: {relative}')
        if path.suffix == '.py':
            python_count += 1
            try:
                tree = ast.parse(text, filename=str(relative))
            except SyntaxError as error:
                errors.append(str(error))
                continue
            for node in ast.walk(tree):
                modules = [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
                if isinstance(node, ast.Import):
                    modules += [alias.name for alias in node.names]
                for module in modules:
                    if module.startswith('crossinterest.'):
                        target = ROOT / 'src' / module.replace('.', '/')
                        if not target.with_suffix('.py').is_file() and not (target / '__init__.py').is_file():
                            errors.append(f'Unknown project import: {relative}: {module}')
                    if module.startswith(('eval_amazon_', 'eval_interest_id_minionerec', 'prep.')):
                        errors.append(f'Unmigrated import: {relative}: {module}')
        if path.suffix == '.md':
            if text.count('```') % 2:
                errors.append(f'Unbalanced code fence: {relative}')
            for target in re.findall(r'\]\(([^)]+)\)', text):
                if target.startswith(('https:', 'http:', '#', 'mailto:')):
                    continue
                link_count += 1
                destination = (path.parent / target.split('#')[0].replace('%20', ' ')).resolve()
                if not destination.is_relative_to(ROOT):
                    errors.append(f'External local link: {relative}: {target}')
                elif '.local' in destination.relative_to(ROOT).parts:
                    errors.append(f'Private archive link: {relative}: {target}')
                elif not destination.exists():
                    errors.append(f'Broken document link: {relative}: {target}')
    return {'public_files': len(files), 'python_files': python_count, 'document_links': link_count,
            'errors': errors}


if __name__ == '__main__':
    result = check()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(bool(result['errors']))