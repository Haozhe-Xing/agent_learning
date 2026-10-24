"""Build both languages without swapping or modifying repository configuration."""
from __future__ import annotations

import argparse
from html import unescape
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import tomllib
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

if __package__:
    from .check_rendered_book import check_rendered_book
    from .markdown_source import split_source
else:
    from check_rendered_book import check_rendered_book
    from markdown_source import split_source

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = {'mdbook': '0.5.4', 'mdbook-katex': '0.10.0'}
# Exact repository-relative paths, reviewed separately from publication sources.
# This unreferenced pre-repair backup contains 240 historical SVG copies. Keep
# the source files, but never copy this directory into a publication build.
# Do not replace this list with a wildcard that could hide legitimate assets.
EXCLUDED_SOURCE_DIRECTORIES = (Path('src/en/svg_backup_broken'),)


def source_roots(root: Path = ROOT) -> dict[str, Path]:
    configs = {'zh': 'book.toml', 'en': 'book-en.toml'}
    return {language: root / tomllib.loads((root / config).read_text(encoding='utf-8'))['book']['src']
            for language, config in configs.items()}


def is_excluded_source(path: Path, root: Path = ROOT) -> bool:
    path = path.resolve()
    return any(path.is_relative_to((root / directory).resolve())
               for directory in EXCLUDED_SOURCE_DIRECTORIES)


def publication_files(root: Path = ROOT) -> list[Path]:
    """Files mdBook may copy from its configured sources, including nested assets."""
    return sorted({path for source in source_roots(root).values() for path in source.rglob('*')
                   if path.is_file() and not is_excluded_source(path, root)})


def publication_svgs(root: Path = ROOT) -> list[Path]:
    return [path for path in publication_files(root) if path.suffix.lower() == '.svg']


def resource_references(source: Path) -> list[str]:
    """Read local resource/link syntax, not paths in hypothetical fenced examples."""
    text = source.read_text(encoding='utf-8')
    if source.suffix.lower() == '.md':
        text, _ = split_source(text)
    patterns = (
        r'!?\[[^\]\n]*\]\(\s*<?([^\s)>]+)>?',
        r'(?m)^\s*\[[^\]\n]+\]:\s*<?([^\s>]+)>?',
        r'''(?i)\b(?:href|src|xlink:href)\s*=\s*["']([^"']+)["']''',
        r'''(?i)url\(\s*["']?([^\s"')]+)["']?\s*\)''',
    )
    return [match[1] for pattern in patterns for match in re.finditer(pattern, text)]


def publication_asset_errors(root: Path = ROOT) -> list[str]:
    errors = []
    roots = list(source_roots(root).values())
    for source in publication_files(root):
        if source.suffix.lower() in {'.md', '.html', '.htm', '.css', '.svg', '.js'}:
            try:
                references = resource_references(source)
            except (ValueError, OSError) as exc:
                errors.append(f'{source.relative_to(root)}: cannot read resource references: {exc}')
                references = []
            for target in references:
                url = urlsplit(unescape(target))
                if url.scheme or url.netloc or not url.path:
                    continue
                path = unquote(url.path)
                if path.startswith('/'):
                    book_source = next(directory for directory in roots if source.is_relative_to(directory))
                    destination = book_source / path.lstrip('/')
                else:
                    destination = source.parent / path
                if is_excluded_source(destination, root):
                    errors.append(f'{source.relative_to(root)}: references excluded publication resource {target}')
        if source.suffix.lower() == '.svg':
            try:
                ElementTree.parse(source)
            except (ElementTree.ParseError, OSError) as exc:
                errors.append(f'{source.relative_to(root)}: invalid SVG XML: {exc}')
    return errors


def copy_publication_sources(root: Path, stage: Path) -> None:
    def ignore(directory, names):
        return [name for name in names if is_excluded_source(Path(directory) / name, root)]

    shutil.copytree(root / 'src', stage / 'src', ignore=ignore)


def copy_preprocessor_tools(root: Path, stage: Path) -> None:
    """Copy only explicitly configured local tools; commands run in stage root."""
    for name in ('book.toml', 'book-en.toml'):
        config = tomllib.loads((root / name).read_text(encoding='utf-8'))
        if 'readme-links' in config.get('preprocessor', {}):
            (stage / 'tools').mkdir(exist_ok=True)
            shutil.copyfile(root / 'tools/readme_links.py', stage / 'tools/readme_links.py')
            return


def validate_output_assets(output: Path, languages: list[str], root: Path = ROOT) -> tuple[dict, list[str]]:
    """Check the actual build output; XML validity does not establish visual quality."""
    counts, errors = {}, []
    for language in languages:
        destination = output / language
        source = source_roots(root)[language]
        for excluded in EXCLUDED_SOURCE_DIRECTORIES:
            path = root / excluded
            if path.is_relative_to(source) and (destination / path.relative_to(source)).exists():
                errors.append(f'{destination}: excluded directory was published: {path.relative_to(source)}')
        assets = [path for path in destination.rglob('*') if path.is_file() and path.suffix.lower() == '.svg']
        counts[language] = len(assets)
        expected = {path.relative_to(source) for path in publication_svgs(root) if path.is_relative_to(source)}
        actual = {path.relative_to(destination) for path in assets}
        for missing in sorted(expected - actual):
            errors.append(f'{destination}: missing published SVG: {missing}')
        for asset in assets:
            try:
                ElementTree.parse(asset)
            except (ElementTree.ParseError, OSError) as exc:
                errors.append(f'{asset}: invalid output SVG XML: {exc}')
    return counts, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'book')
    parser.add_argument('--language', choices=['zh', 'en', 'both'], default='both')
    args = parser.parse_args()
    errors = publication_asset_errors(ROOT)
    if errors:
        raise SystemExit('\n'.join(errors))
    for tool, version in VERSIONS.items():
        path = shutil.which(tool)
        if path is None:
            raise SystemExit(f'Missing {tool}; install version {version}. See tools/README.md.')
        actual = subprocess.check_output([path, '--version'], text=True).strip()
        if actual.removeprefix(tool + ' ').removeprefix('v') != version:
            raise SystemExit(f'Expected {tool} {version}; found {actual}. See tools/README.md.')
    output = args.output.resolve()
    languages = ['zh', 'en'] if args.language == 'both' else [args.language]
    with tempfile.TemporaryDirectory(prefix='agent-book-') as temp:
        stage = Path(temp)
        copy_publication_sources(ROOT, stage)
        copy_preprocessor_tools(ROOT, stage)
        shutil.copytree(ROOT / 'theme', stage / 'theme')
        for lang in languages:
            shutil.copyfile(ROOT / ('book.toml' if lang == 'zh' else 'book-en.toml'), stage / 'book.toml')
            subprocess.run(['mdbook', 'build', str(stage), '-d', str(output / lang)], check=True)
    counts, errors = validate_output_assets(output, languages, ROOT)
    if errors:
        raise SystemExit('\n'.join(errors))
    link_counts, errors = check_rendered_book(output, languages, root=ROOT)
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Built {args.language}: {output}; output SVG XML checked: {counts}; '
          f'rendered link checks: {link_counts}')


if __name__ == '__main__':
    main()
