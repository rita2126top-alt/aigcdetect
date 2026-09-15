#!/usr/bin/env python3
"""Read-only source, YAML, shell, import and CLI checks; never write upstream pyc."""
from __future__ import annotations
import argparse
import ast
import contextlib
import importlib
import io
import json
import subprocess
import sys
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional JSON report outside the source tree')
    args = parser.parse_args()
    import yaml
    counts = {'python': 0, 'shell': 0, 'yaml': 0, 'modules': 0, 'cli_commands': 0}
    errors, notices = [], []
    excluded = {'.git', '.pytest_cache', '__pycache__', '.delivery', 'build', 'dist', '.venv'}
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(x in excluded for x in relative.parts):
            continue
        try:
            if path.suffix == '.py':
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter('always')
                    source = path.read_text(encoding='utf-8')
                    compile(source, str(relative), 'exec', dont_inherit=True)
                    ast.parse(source, filename=str(relative))
                for w in seen:
                    message = f'{relative}:{w.lineno}: {w.category.__name__}: {w.message}'
                    if message not in notices:
                        notices.append(message)
                counts['python'] += 1
            elif path.suffix == '.sh':
                subprocess.run(['bash', '-n', str(path)], check=True, capture_output=True, text=True)
                counts['shell'] += 1
            elif path.suffix in ('.yaml', '.yml'):
                yaml.safe_load(path.read_text(encoding='utf-8'))
                counts['yaml'] += 1
        except Exception as exc:
            detail = getattr(exc, 'stderr', None) or str(exc)
            errors.append(f'{relative}: {detail}')
    for path in sorted((ROOT/'cadp').glob('*.py')):
        name = 'cadp' if path.stem == '__init__' else f'cadp.{path.stem}'
        try:
            importlib.import_module(name)
            counts['modules'] += 1
        except Exception as exc:
            errors.append(f'import {name}: {exc}')
    from cadp.cli import parser as cli_parser, verify_upstream
    cli = cli_parser()
    commands = next(a for a in cli._actions if isinstance(a, argparse._SubParsersAction)).choices
    for name in commands:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                cli.parse_args([name, '--help'])
        except SystemExit as exc:
            if exc.code != 0:
                errors.append(f'CLI {name} --help: exit {exc.code}')
            else:
                counts['cli_commands'] += 1
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            upstream = verify_upstream()
    except Exception as exc:
        errors.append(f'upstream integrity: {exc}')
        upstream = {'passed': False}
    report = {'passed': not errors, 'counts': counts, 'upstream': upstream,
              'warnings': notices, 'errors': errors,
              'note': 'Source compilation and import checks do not execute full CUDA experiments.'}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
