"""Check reader order, numbered navigation and chapter iteration contracts.

Run with ``python tools/check_learning_path.py``. Checks are offline and read
only. Unlinked prose, diagram text, and anchor destinations need separate review;
this checker deliberately does not guess the meaning of arbitrary numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import posixpath
import re
import sys
from typing import Mapping, Sequence
from urllib.parse import unquote

try:
    from .book_structure import CHAPTERS, CHAPTER_BY_SLUG, ITERATION_SECTIONS, LANGUAGES, PARTS
    from .markdown_source import split_source
except ImportError:  # Direct script execution.
    from book_structure import CHAPTERS, CHAPTER_BY_SLUG, ITERATION_SECTIONS, LANGUAGES, PARTS
    from markdown_source import split_source

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r'(?<!!)\[([^\]\n]+)\]\(([^\s)]+)\)')
ROMAN = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7}
CHINESE = dict(zip('一二三四五六七', range(1, 8)))


@dataclass(frozen=True)
class NavigationEntry:
    label: str
    path: str
    line: int
    depth: int


def number_identifier(text: str) -> str | None:
    """Read an explicit chapter, section, part or appendix identifier."""
    text = re.sub(r'[*_`]', '', text.lstrip('# *_`[')).strip()
    chapter = re.match(r'(?:第\s*(\d+)\s*章|Chapter\s+(\d+)(?![\d.]))', text, re.I)
    if chapter:
        return chapter[1] or chapter[2]
    section = re.match(r'(?:Section\s+|第\s*)?(\d+(?:\.\d+[a-z]?)+|[A-Z]\.\d+(?:\.\d+)*)'
                       r"(?=\s|[节:：'’]|$)", text, re.I)
    if section:
        return section[1]
    part = re.match(r'(?:第([一二三四五六七])部分|Part\s+([IVX]+|\d+))(?=\s|[:：]|$)', text, re.I)
    if part:
        value = CHINESE[part[1]] if part[1] else ROMAN.get(part[2].upper(), part[2])
        return f'part:{value}'
    appendix = re.match(r'(?:附录\s*|Appendix\s+)([A-Z])(?=\s|[:：]|$)', text, re.I)
    return f'appendix:{appendix[1].upper()}' if appendix else None


def link_number_identifier(label: str) -> str | None:
    """Also recognize navigation captions and explicit references inside labels."""
    direct = number_identifier(label)
    if direct is not None:
        return direct
    explicit = re.search(r'(?:\b(?:Chapter|Section)\s+\d+(?:\.\d+[a-z]?)*|第\s*\d+(?:\.\d+)*\s*[章节])', label, re.I)
    if explicit:
        return number_identifier(explicit[0])
    caption = re.search(r'(?:Next|Previous|下一节|上一节|下一章|上一章)\s*[:：]\s*(.+)', label, re.I)
    return number_identifier(caption[1]) if caption else None


def parse_summary(text: str) -> list[NavigationEntry]:
    """Extract standalone SUMMARY entries, keeping stable normalized paths."""
    prose, _ = split_source(text)
    entries = []
    for line_number, line in enumerate(prose.splitlines(), 1):
        match = re.fullmatch(r'(\s*)(?:-\s+)?\[([^\]\n]+)\]\(([^\s)]+)\)\s*', line)
        if match:
            entries.append(NavigationEntry(match[2], posixpath.normpath(unquote(match[3])),
                                           line_number, len(match[1].expandtabs(4))))
    return entries


def check_summary(text: str, language: str) -> list[str]:
    """Pure structure checks; no filesystem or global mutable state."""
    entries = parse_summary(text)
    errors, observed_parts, observed_chapters = [], [], []
    current_part, current_chapter = None, None
    for entry in entries:
        location = f'{language}/SUMMARY.md:{entry.line}'
        part_match = re.fullmatch(r'part(\d+)\.md', entry.path)
        if part_match:
            current_part, current_chapter = int(part_match[1]), None
            observed_parts.append(current_part)
            if number_identifier(entry.label) != f'part:{current_part}':
                errors.append(f'{location}: part label must identify part {current_part}')
            if entry.depth:
                errors.append(f'{location}: part entries must be top-level')
            continue
        parts = PurePosixPath(entry.path).parts
        chapter = CHAPTER_BY_SLUG.get(parts[0]) if parts else None
        if chapter is None:
            identifier = number_identifier(entry.label)
            if identifier and identifier[0].isdigit():
                errors.append(f'{location}: numbered navigation page is absent from the chapter manifest: {entry.path}')
            continue  # Front matter and appendices are outside the numbered chapters.
        if entry.path == chapter.path:
            observed_chapters.append(chapter.slug)
            current_chapter = chapter.slug
            if entry.depth:
                errors.append(f'{location}: chapter entries must be top-level')
            if number_identifier(entry.label) != str(chapter.number):
                errors.append(f'{location}: {chapter.slug} must be chapter {chapter.number}')
            if current_part != chapter.part:
                errors.append(f'{location}: chapter {chapter.number} belongs to part {chapter.part}, got {current_part}')
        else:
            identifier = number_identifier(entry.label)
            if current_chapter != chapter.slug or not entry.depth:
                errors.append(f'{location}: section must be nested under {chapter.slug}')
            if not identifier or not identifier.startswith(f'{chapter.number}.'):
                errors.append(f'{location}: section number must start with {chapter.number}.')
    if observed_parts != [part.number for part in PARTS]:
        errors.append(f'{language}/SUMMARY.md: part sequence must be 1–7, got {observed_parts}')
    if observed_chapters != [chapter.slug for chapter in CHAPTERS]:
        errors.append(f'{language}/SUMMARY.md: chapter paths must follow the canonical 25-chapter order')
    paths = [entry.path for entry in entries]
    if len(paths) != len(set(paths)):
        errors.append(f'{language}/SUMMARY.md: duplicate navigation page')
    numbers = [number_identifier(entry.label) for entry in entries]
    numbers = [number for number in numbers if number is not None]
    if len(numbers) != len(set(numbers)):
        errors.append(f'{language}/SUMMARY.md: duplicate navigation number')
    return errors


def check_navigation(entries: Sequence[NavigationEntry], pages: Mapping[str, str], language: str) -> list[str]:
    """Compare every numbered navigation label to its page's first real H1."""
    errors = []
    for entry in entries:
        location = f'{language}/{entry.path}'
        if entry.path not in pages:
            errors.append(f'{location}: navigation page does not exist')
            continue
        prose, _ = split_source(pages[entry.path])
        headings = re.findall(r'^#\s+(.+)$', prose, re.M)
        if not headings:
            errors.append(f'{location}: navigation page has no H1')
            continue
        expected = number_identifier(entry.label)
        if expected and number_identifier(headings[0]) != expected:
            errors.append(f'{location}: H1 number must match navigation {expected}, got {number_identifier(headings[0])}')
    return errors


def check_readme(text: str, language: str, section_numbers: set[str]) -> list[str]:
    """Check local section tables and concrete iteration/next-step fields."""
    prose, _ = split_source(text)
    title, fields = ITERATION_SECTIONS[language]
    errors = []
    heading = re.search(r'^##\s+' + re.escape(title) + r'\s*$', prose, re.M)
    if heading is None:
        errors.append(f'missing ## {title}')
    else:
        body = prose[heading.end():]
        next_heading = re.search(r'^#{1,2}\s', body, re.M)
        body = body[:next_heading.start()] if next_heading else body
        for field in fields:
            values = re.findall(r'^\s*-\s+\*\*' + re.escape(field) + r'\*\*\s*[:：]\s*(.*)$', body, re.M)
            if len(values) != 1:
                errors.append(f'{title}: requires exactly one {field} field')
                continue
            value = values[0].strip()
            if len(re.sub(r'\s|[*_`]', '', value)) < 8 or re.fullmatch(
                    r'(?:TODO|TBD|待补充|待完善|同上|[.。…\s])+', value, re.I):
                errors.append(f'{title}: {field} needs concrete non-placeholder text')
            if field == fields[-1] and not any(
                    '#' not in link[2] and not re.match(r'[A-Za-z][\w+.-]*:', link[2])
                    and (link[2].endswith('.md') or link[2].endswith('/')) for link in LINK.finditer(value)):
                errors.append(f'{title}: {field} must link to a local next learning page')
    for line_number, line in enumerate(prose.splitlines(), 1):
        if not line.lstrip().startswith('|'):
            continue
        first_cell = line.strip().split('|')[1].strip()
        identifier = number_identifier(first_cell)
        if identifier and '.' in identifier and identifier not in section_numbers:
            errors.append(f'line {line_number}: local table section {identifier} is absent from this chapter navigation')
    return errors


def check_link_labels(text: str, source_path: str, pages: Mapping[str, str]) -> list[str]:
    """Resolve numbered link labels against local target H1s.

    Anchor links may name a deeper subsection; skip them until rendered-anchor
    validation is available. Numeric versions and unlinked prose are not guessed.
    """
    prose, _ = split_source(text)
    errors = []
    for match in LINK.finditer(prose):
        target = unquote(match[2])
        if '#' in target or re.match(r'[A-Za-z][\w+.-]*:', target):
            continue
        path = posixpath.normpath(posixpath.join(posixpath.dirname(source_path), target))
        if path not in pages and posixpath.join(path, 'README.md') in pages:
            path = posixpath.join(path, 'README.md')
        if path not in pages or not path.endswith('.md'):
            continue
        identifier = link_number_identifier(match[1])
        if identifier is None:
            before = prose[:match.start()].rsplit('\n', 1)[-1]
            prefix = re.search(r'((?:Chapter|Section)\s+\d+(?:\.\d+[a-z]?)*|第\s*\d+\s*章)\s*$', before, re.I)
            identifier = number_identifier(prefix[1]) if prefix else None
        if identifier is None:
            continue
        target_prose, _ = split_source(pages[path])
        header = re.search(r'^#\s+(.+)$', target_prose, re.M)
        actual = number_identifier(header[1]) if header else None
        # A whole-chapter reference may legitimately jump to one of its pages.
        parent_reference = identifier.isdigit() and actual and actual.startswith(identifier + '.')
        if identifier != actual and not parent_reference:
            errors.append(f'{source_path}: link label {identifier} points to {path} ({actual})')
    return errors


def check_heading_numbers(text: str) -> list[str]:
    """Reject collapsed sibling numbers without imposing book numbers on locals.

    Algorithm pages may use local headings such as 1.1, 1.2 and 1.8.1 under a
    page numbered 24.3. A local sequence may restart under a different parent.
    Reusing one number for different sibling headings is the migration error
    this check guards against; it does not renumber or infer heading semantics.
    """
    prose, _ = split_source(text)
    errors, stack, seen = [], [], {}
    for line_number, line in enumerate(prose.splitlines(), 1):
        heading = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading is None:
            continue
        level, title = len(heading[1]), heading[2]
        while stack and stack[-1][0] >= level:
            stack.pop()
        identifier = number_identifier(title)
        if level > 1 and identifier and re.fullmatch(r'\d+(?:\.\d+[a-z]?)+', identifier):
            scope = tuple(parent[1] for parent in stack)
            key = (scope, identifier)
            if key in seen:
                previous_line, previous_title = seen[key]
                errors.append(f'line {line_number}: duplicate sibling heading number {identifier}: '
                              f'{title!r}; first used at line {previous_line}: {previous_title!r}')
            else:
                seen[key] = (line_number, title)
        stack.append((level, line_number))
    return errors


def collect_errors(root: Path = ROOT) -> list[str]:
    """Read a checkout and return actionable errors; does not print or mutate."""
    errors, navigation = [], {}
    for language in LANGUAGES:
        directory = root / 'src' / language
        pages = {}
        for path in sorted(directory.rglob('*.md')):
            try:
                text = path.read_text(encoding='utf-8')
                split_source(text)
                pages[str(path.relative_to(directory))] = text
            except (ValueError, OSError) as exc:
                errors.append(f'{language}/{path.relative_to(directory)}: {exc}')
        summary = pages.get('SUMMARY.md', '')
        try:
            errors.extend(check_summary(summary, language))
            entries = parse_summary(summary)
            navigation[language] = [(entry.path, number_identifier(entry.label)) for entry in entries]
            errors.extend(check_navigation(entries, pages, language))
        except ValueError as exc:
            errors.append(f'{language}/SUMMARY.md: {exc}')
            continue
        for chapter in CHAPTERS:
            if chapter.path not in pages:
                errors.append(f'{language}/{chapter.path}: chapter README is missing')
                continue
            section_numbers = {number_identifier(entry.label) for entry in entries
                               if entry.path.startswith(chapter.slug + '/') and entry.path != chapter.path}
            try:
                errors.extend(f'{language}/{chapter.path}: {error}'
                              for error in check_readme(pages[chapter.path], language, section_numbers))
            except ValueError as exc:
                errors.append(f'{language}/{chapter.path}: {exc}')
        for path, text in pages.items():
            try:
                errors.extend(f'{language}/{error}' for error in check_link_labels(text, path, pages))
                errors.extend(f'{language}/{path}: {error}' for error in check_heading_numbers(text))
            except ValueError as exc:
                errors.append(f'{language}/{path}: {exc}')
        # Next-step links must resolve, including when the label has no number.
        for chapter in CHAPTERS:
            if chapter.path not in pages:
                continue
            prose, _ = split_source(pages[chapter.path])
            field = ITERATION_SECTIONS[language][1][-1]
            line = re.search(r'^\s*-\s+\*\*' + re.escape(field) + r'\*\*\s*[:：]\s*(.*)$', prose, re.M)
            if line:
                for link in LINK.finditer(line[1]):
                    target = unquote(link[2])
                    if re.match(r'[A-Za-z][\w+.-]*:', target):
                        continue
                    target = posixpath.normpath(posixpath.join(chapter.slug, target.split('#')[0]))
                    if target not in pages and target + '/README.md' in pages:
                        target += '/README.md'
                    if target not in pages:
                        errors.append(f'{language}/{chapter.path}: next learning page does not exist: {target}')
    if navigation.get('zh') != navigation.get('en'):
        errors.append('SUMMARY numbered paths or order differ between zh and en')
    return errors


def check(root: Path = ROOT) -> bool:
    errors = collect_errors(root)
    for error in errors:
        print(error, file=sys.stderr)
    print(f'Checked {len(CHAPTERS)} chapters, {len(PARTS)} parts and two languages; errors: {len(errors)}')
    return bool(errors)


if __name__ == '__main__':
    sys.exit(check())
