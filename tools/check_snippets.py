"""Check syntax of Python-labelled examples; this is not a runtime guarantee."""
import ast
from pathlib import Path
import sys
from markdown_source import split_source

ROOT = Path(__file__).resolve().parents[1]


def main():
    count, errors = 0, []
    for path in sorted((ROOT / 'src').rglob('*.md')):
        try:
            _, blocks = split_source(path.read_text())
        except ValueError as exc:
            errors.append(f'{path.relative_to(ROOT)}: {exc}')
            continue
        for language, line, code in blocks:
            if language != 'python':
                continue
            count += 1
            try:
                ast.parse(code)
            except SyntaxError as exc:
                errors.append(f'{path.relative_to(ROOT)}:{line + (exc.lineno or 0)}: {exc.msg}')
    for error in errors:
        print(error, file=sys.stderr)
    print(f'Parsed {count} Python code blocks; errors: {len(errors)}')
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
