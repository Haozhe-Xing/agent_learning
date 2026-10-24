"""Check local links, images and HTML fragment IDs in the published book.

Only SUMMARY-listed pages and references inside their <main> are inspected.
This offline check establishes target existence, not external-link validity,
content correctness, accessibility, or visual quality.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
import re
from typing import Sequence
import tomllib
from urllib.parse import unquote, urlsplit

try:
    from .markdown_source import split_source
except ImportError:  # Direct script execution.
    from markdown_source import split_source

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ('zh', 'en')


def external_reference(value: str) -> bool:
    """Do not validate even the syntax of URLs outside the offline scope."""
    value = value.lstrip()
    return value.startswith('//') or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', value) is not None


def summary_urls(text: str) -> list[str]:
    """Keep raw navigation URLs so percent-encoded paths are decoded once."""
    prose, _ = split_source(text)
    return [match[1] for line in prose.splitlines()
            if (match := re.fullmatch(r'\s*(?:-\s+)?\[[^\]\n]+\]\(([^\s)]+)\)\s*', line))]


def rendered_path(source: str) -> Path:
    """Map a SUMMARY source to mdBook's actual page route, including README."""
    path = PurePosixPath(unquote(urlsplit(source).path))
    name = 'index.html' if path.stem.lower() == 'readme' else path.with_suffix('.html').name
    return Path(*path.parent.parts) / name


@dataclass
class RenderedPage:
    references: list[tuple[int, str, str]] = field(default_factory=list)
    ids: set[str] = field(default_factory=set)
    main_count: int = 0


class PageParser(HTMLParser):
    """HTMLParser ignores escaped code examples instead of regexing their text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.page = RenderedPage()
        self.in_main = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get('id') is not None:
            self.page.ids.add(attributes['id'])
        if tag == 'a' and attributes.get('name') is not None:
            self.page.ids.add(attributes['name'])  # Legacy named HTML anchors.
        if tag == 'main':
            self.in_main = True
            self.page.main_count += 1
        if self.in_main:
            attribute = 'href' if tag == 'a' else 'src' if tag == 'img' else None
            if attribute and attributes.get(attribute) is not None:
                self.page.references.append((self.getpos()[0], tag, attributes[attribute]))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag == 'main':
            self.in_main = False

    def handle_endtag(self, tag):
        if tag == 'main':
            self.in_main = False


def read_page(path: Path) -> RenderedPage:
    parser = PageParser()
    parser.feed(path.read_text(encoding='utf-8'))
    parser.close()
    return parser.page


def check_rendered_book(output: Path, languages: Sequence[str] = LANGUAGES,
                        root: Path = ROOT) -> tuple[dict, list[str]]:
    """Validate actual output; return per-language scope counts and errors.

    ``output`` is the common parent of zh/ and en/. ``root`` supplies the book
    configs and SUMMARY sources, which makes small temporary fixtures testable.
    Fragments are checked against IDs/named anchors in HTML targets, including
    same-page IDs. Browser text-directive matching is not evaluated.
    Query strings do not change filesystem destinations. Other schemes and
    protocol-relative URLs are outside this check's scope.
    """
    output, root = Path(output).resolve(), Path(root).resolve()
    counts, errors, cache = {}, [], {}

    def load(path: Path) -> RenderedPage:
        if path not in cache:
            cache[path] = read_page(path)
        return cache[path]

    for language in languages:
        if language not in LANGUAGES:
            raise ValueError(f'Unsupported book language: {language}')
        stats = dict(expected_pages=0, pages_checked=0, links=0, images=0,
                     fragments=0, external_references_skipped=0)
        counts[language] = stats
        config_path = root / ('book.toml' if language == 'zh' else 'book-en.toml')
        try:
            config = tomllib.loads(config_path.read_text(encoding='utf-8'))
            summary = root / config['book']['src'] / 'SUMMARY.md'
            entries = summary_urls(summary.read_text(encoding='utf-8'))
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f'{config_path}: cannot read publication SUMMARY: {exc}')
            continue
        pages = []
        for entry in entries:
            if external_reference(entry):
                continue
            try:
                url = urlsplit(entry)
            except ValueError as exc:
                errors.append(f'{summary}: invalid navigation URL {entry!r}: {exc}')
                continue
            # SUMMARY is navigation input rather than a published content page.
            if url.scheme or url.netloc or not url.path or PurePosixPath(url.path).name == 'SUMMARY.md':
                continue
            try:
                page = (output / language / rendered_path(entry)).resolve()
            except (OSError, ValueError, RuntimeError) as exc:
                errors.append(f'{summary}: invalid navigation target {entry!r}: {exc}')
                continue
            if page not in pages:
                pages.append(page)
        stats['expected_pages'] = len(pages)
        for page in pages:
            if not page.is_relative_to(output):
                errors.append(f'{summary}: page route escapes published output: {page}')
                continue
            if not page.is_file():
                errors.append(f'{page}: missing SUMMARY-listed output page')
                continue
            try:
                parsed = load(page)
            except (OSError, UnicodeError) as exc:
                errors.append(f'{page}: cannot read output HTML: {exc}')
                continue
            stats['pages_checked'] += 1
            if parsed.main_count != 1:
                errors.append(f'{page}: expected one <main>, found {parsed.main_count}')
            for line, tag, reference in parsed.references:
                location = f'{page}:{line}'
                if external_reference(reference):
                    stats['external_references_skipped'] += 1
                    continue
                try:
                    url = urlsplit(reference)
                except ValueError as exc:
                    errors.append(f'{location}: invalid local reference {reference!r}: {exc}')
                    continue
                if url.scheme or url.netloc:
                    stats['external_references_skipped'] += 1
                    continue
                stats['links' if tag == 'a' else 'images'] += 1
                path = unquote(url.path)
                try:
                    target = (output / path.lstrip('/')) if path.startswith('/') else (page.parent / path)
                    if not path:
                        target = page
                    target = target.resolve()
                    if path.endswith('/') or target.is_dir():
                        target /= 'index.html'
                except (OSError, ValueError, RuntimeError) as exc:
                    errors.append(f'{location}: invalid local target {reference!r}: {exc}')
                    continue
                if not target.is_relative_to(output):
                    errors.append(f'{location}: local target escapes published output: {reference!r}')
                    continue
                if not target.is_file():
                    errors.append(f'{location}: missing local {"image" if tag == "img" else "file"} '
                                  f'{reference!r} -> {target}')
                    continue
                # A browser text directive is not an element ID. If it also
                # includes an ID, only that preceding element target is checked.
                fragment = unquote(url.fragment).partition(':~:')[0]
                if fragment and target.suffix.lower() in ('.html', '.htm'):
                    stats['fragments'] += 1
                    try:
                        target_page = load(target)
                    except (OSError, UnicodeError) as exc:
                        errors.append(f'{location}: cannot read fragment target {target}: {exc}')
                        continue
                    if fragment not in target_page.ids:
                        errors.append(f'{location}: missing HTML fragment {reference!r} '
                                      f'-> {target}#{fragment}')
    return counts, errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'book')
    parser.add_argument('--language', choices=(*LANGUAGES, 'both'), default='both')
    args = parser.parse_args(argv)
    languages = LANGUAGES if args.language == 'both' else (args.language,)
    counts, errors = check_rendered_book(args.output, languages)
    for language, stats in counts.items():
        print(f'{language}: {stats}')
    for error in errors:
        print(error)
    print(f'Rendered local files / HTML fragment IDs: {len(errors)} error(s). '
          'External links and visual quality were not checked.')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
