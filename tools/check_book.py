"""Check chapter links and literal companion paths without network or dependencies.

Fenced examples are excluded: paths inside them may describe hypothetical projects.
Static image paths are checked alongside chapter links.
"""
from pathlib import Path
import re
import sys
from urllib.parse import unquote
if __package__:
    from .build_books import publication_asset_errors, publication_files, publication_svgs
    from .markdown_source import split_source
else:
    from build_books import publication_asset_errors, publication_files, publication_svgs
    from markdown_source import split_source

ROOT = Path(__file__).resolve().parents[1]


def check():
    errors = []
    chapter_counts = []
    navigation = []
    files = [path for path in publication_files(ROOT) if path.suffix.lower() == '.md']
    for source in files:
        try:
            text, _ = split_source(source.read_text())
        except ValueError as exc:
            errors.append(f'{source.relative_to(ROOT)}: {exc}')
            continue
        for match in re.finditer(r'!?\[[^\]\n]*\]\(([^\s)]+)\)', text):
            target = unquote(match[1].split('#')[0])
            if not target or re.match(r'[a-zA-Z][\w+.-]*:', target):
                continue
            destination = source.parent / target
            if not destination.exists():
                errors.append(f'{source.relative_to(ROOT)}: missing chapter or image link {target}')
            elif destination.suffix == '.md' and destination.is_file():
                label = match[0].split('](', 1)[0]
                actual = re.search(r'(?:第\s*|Chapter\s+)(\d+)\s*(?:章|\b)', destination.read_text().splitlines()[0], re.I)
                stated = re.search(r'(?:第\s*|Chapter\s+)(\d+)\s*(?:章|\b)', label, re.I)
                if actual and stated and actual[1] != stated[1]:
                    errors.append(f'{source.relative_to(ROOT)}: chapter label {stated[1]} points to chapter {actual[1]}')
        for match in re.finditer(r'`((?:examples|reference-agent|reference_code)/[\w./-]+)`', text):
            if not (ROOT / match[1]).exists():
                errors.append(f'{source.relative_to(ROOT)}: missing companion path {match[1]}')
    for lang in ('zh', 'en'):
        summary = (ROOT / f'src/{lang}/SUMMARY.md').read_text()
        navigation.append(re.findall(r"\]\(([^)]+)\)", summary))
        chapter_counts.append(len(re.findall(r'^\s*-\s*\[', summary, re.M)))
    if chapter_counts[0] != chapter_counts[1]:
        errors.append(f'SUMMARY entry counts differ: {chapter_counts}')
    if navigation[0] != navigation[1]:
        errors.append("SUMMARY chapter paths or order differ between languages")
    assets = publication_svgs(ROOT)
    errors.extend(publication_asset_errors(ROOT))
    for error in errors:
        print(error, file=sys.stderr)
    print(f'Checked {len(files)} Markdown files and {len(assets)} SVG files; '
          f'SUMMARY entries: {chapter_counts}; errors: {len(errors)}')
    return bool(errors)


if __name__ == '__main__':
    sys.exit(check())
