"""Map declared README chapter links to mdBook's actual rendered chapter paths.

mdBook 0.5.4's index preprocessor changes Chapter.path, but its HTML renderer
only changes .md extensions in body links. Run this after index and before
KaTeX. Changes apply to the in-memory book only; source_path is retained.
This command deliberately uses only Python 3.9-compatible standard libraries
so native mdbook build/serve and temporary staging can invoke system python3.
"""
from __future__ import annotations

from html import escape, unescape
import json
import posixpath
import re
import sys
from urllib.parse import quote, unquote, urlsplit, urlunsplit


def chapters(items):
    for item in items:
        if isinstance(item, dict) and isinstance(item.get('Chapter'), dict):
            chapter = item['Chapter']
            yield chapter
            yield from chapters(chapter.get('sub_items', []))


def chapter_routes(book):
    """Use index's actual source→destination metadata, not a README wildcard."""
    routes = {}
    for chapter in chapters(book['items']):
        source, destination = chapter.get('source_path'), chapter.get('path')
        if not source or not destination:
            continue
        source = posixpath.normpath(source.replace('\\', '/'))
        destination = posixpath.normpath(destination.replace('\\', '/'))
        stem = posixpath.splitext(posixpath.basename(source))[0]
        if stem.casefold() == 'readme' and source != destination:
            routes[source] = posixpath.splitext(destination)[0] + '.html'
    return routes


def rewrite_target(target, source_path, routes):
    decoded = unescape(target).lstrip()
    # External URL syntax is outside this chapter-routing tool's scope.
    # Decide that scope before urlsplit, which rejects malformed IPv6 hosts.
    if decoded.startswith('//') or re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', decoded):
        return target
    url = urlsplit(decoded)
    if url.scheme or url.netloc or not url.path:
        return target
    path = unquote(url.path)
    parent = posixpath.dirname(source_path.replace('\\', '/'))
    resolved = posixpath.normpath(path.lstrip('/') if path.startswith('/')
                                  else posixpath.join(parent, path))
    destination = routes.get(resolved)
    if destination is None:
        return target
    if path.startswith('/'):
        rewritten = '/' + destination
    else:
        rewritten = posixpath.relpath(destination, parent or '.')
        if path.startswith('./') and not rewritten.startswith('.'):
            rewritten = './' + rewritten
    return urlunsplit(('', '', quote(rewritten, safe='/-._~'), url.query, url.fragment))


def rewrite_markup(text, source_path, routes):
    def markdown_target(match):
        target = match['target']
        angled = target.startswith('<') and target.endswith('>')
        value = target[1:-1] if angled else target
        rewritten = rewrite_target(value, source_path, routes)
        if rewritten == value:
            return match[0]
        if angled:
            rewritten = '<' + rewritten + '>'
        return match['prefix'] + rewritten

    text = re.sub(r'(?P<prefix>!?\[[^\]\n]*\]\(\s*)(?P<target><[^>\n]+>|[^\s)]+)',
                  markdown_target, text)
    text = re.sub(r'(?m)(?P<prefix>^ {0,3}\[[^\]\n]+\]:\s*)(?P<target><[^>\n]+>|\S+)',
                  markdown_target, text)

    def html_target(match):
        target = match['target']
        rewritten = rewrite_target(target, source_path, routes)
        return match[0] if rewritten == target else match['prefix'] + match['quote'] + escape(rewritten, quote=True) + match['quote']

    return re.sub(r'''(?i)(?P<prefix>\bhref\s*=\s*)(?P<quote>["'])(?P<target>.*?)(?P=quote)''',
                  html_target, text)


def rewrite_prose(text, source_path, routes):
    # A code span's delimiter must match in length. Its contents are examples,
    # not navigable links; multiline code spans are kept unchanged as well.
    code = re.compile(r'(?<!`)(?P<ticks>`+)(?!`)(.*?)(?<!`)(?P=ticks)(?!`)', re.DOTALL)
    # Mask spans while retaining surrounding markup. Splitting at every span
    # would miss a real link such as [`Skill`](chapter_skill/README.md).
    prefix = '\x00mdbook-code-'
    while prefix in text:
        prefix += '_'
    spans = []

    def mask(match):
        token = prefix + str(len(spans)) + '\x00'
        spans.append((token, match[0]))
        return token

    rewritten = rewrite_markup(code.sub(mask, text), source_path, routes)
    for token, original in spans:
        rewritten = rewritten.replace(token, original)
    return rewritten


def rewrite_content(content, source_path, routes):
    output, prose, fence = [], [], None
    for line in content.splitlines(keepends=True):
        if fence is not None:
            output.append(line)
            if re.fullmatch(r' {0,3}' + re.escape(fence[0]) + '{' + str(len(fence)) + r',}\s*', line):
                fence = None
            continue
        opening = re.match(r' {0,3}(`{3,}|~{3,})', line)
        if opening:
            output.append(rewrite_prose(''.join(prose), source_path, routes))
            prose = []
            fence = opening[1]
            output.append(line)
        else:
            prose.append(line)
    output.append(rewrite_prose(''.join(prose), source_path, routes))
    return ''.join(output)


def preprocess(book):
    if not isinstance(book, dict) or not isinstance(book.get('items'), list):
        raise ValueError('readme-links requires the mdBook 0.5 book.items protocol')
    routes = chapter_routes(book)
    for chapter in chapters(book['items']):
        source = chapter.get('source_path')
        if source and isinstance(chapter.get('content'), str):
            chapter['content'] = rewrite_content(chapter['content'], source, routes)
    return book


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == 'supports':
        return int(argv[1:] != ['html'])
    _context, book = json.load(sys.stdin)
    json.dump(preprocess(book), sys.stdout, ensure_ascii=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
