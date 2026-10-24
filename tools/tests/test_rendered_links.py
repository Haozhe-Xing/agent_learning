"""Actual mdBook/KaTeX regression for declared README chapter routes.

Discovery can skip the real-renderer class when binaries are absent. CI must
also run this file with --require-tools after installing the pinned tools;
that entry point fails instead of silently skipping missing dependencies.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import build_books, readme_links
from tools.check_rendered_book import check_rendered_book, read_page


REQUIRE_TOOLS = False


def chapter(source, destination=None, content='', children=None):
    return {'Chapter': {'source_path': source, 'path': destination or source,
                        'content': content, 'sub_items': children or []}}


class ReadmePreprocessorTests(unittest.TestCase):
    def setUp(self):
        self.book = {'items': [chapter('guide/page.md'),
                                 chapter('chapter/README.md', 'chapter/index.md'),
                                 chapter('章节/README.md', '章节/index.md')]}

    def test_declared_routes_preserve_relative_fragments_queries_and_html(self):
        content = ('[plain](../chapter/README.md)\n'
                   '[dot](./../chapter/README.md#target)\n'
                   '[cn](../%E7%AB%A0%E8%8A%82/README.md#target)\n'
                   '[reference][id]\n\n[id]: ../chapter/README.md#target "title"\n'
                   '<a href="../chapter/README.md?q=1&amp;v=2#target">html</a>\n')
        self.book['items'][0]['Chapter']['content'] = content
        original_sources = [item['Chapter']['source_path'] for item in self.book['items']]
        result = readme_links.preprocess(self.book)
        rewritten = result['items'][0]['Chapter']['content']
        self.assertIn('[plain](../chapter/index.html)', rewritten)
        self.assertIn('[dot](../chapter/index.html#target)', rewritten)
        self.assertIn('../%E7%AB%A0%E8%8A%82/index.html#target', rewritten)
        self.assertIn('[id]: ../chapter/index.html#target "title"', rewritten)
        self.assertIn('href="../chapter/index.html?q=1&amp;v=2#target"', rewritten)
        self.assertEqual([item['Chapter']['source_path'] for item in result['items']], original_sources)

    def test_external_and_undeclared_readme_targets_are_not_globally_renamed(self):
        content = ('[external](https://example.com/chapter/README.md#target)\n'
                   '[undeclared](../notes/README.md)\n'
                   '[same page](#target)\n')
        routes = readme_links.chapter_routes(self.book)
        self.assertEqual(readme_links.rewrite_content(content, 'guide/page.md', routes), content)
        no_index = {'items': [chapter('chapter/README.md')]}
        self.assertEqual(readme_links.chapter_routes(no_index), {})

    def test_malformed_external_hosts_remain_outside_local_routing_scope(self):
        content = ('[external](https://[)\n'
                   '[protocol-relative](//[/chapter/README.md)\n'
                   '<a href="HTTPS://[">external syntax is preserved</a>\n'
                   '[local](../chapter/README.md)\n')
        rewritten = readme_links.rewrite_content(
            content, 'guide/page.md', readme_links.chapter_routes(self.book))
        self.assertEqual(rewritten, content.replace(
            '[local](../chapter/README.md)', '[local](../chapter/index.html)'))

    def test_fenced_and_inline_examples_remain_unchanged(self):
        content = ('[real](../chapter/README.md)\n'
                   '`[inline](../chapter/README.md)`\n'
                   '``[longer ` code](../chapter/README.md)``\n'
                   '````markdown\n[example](../chapter/README.md)\n```\n````\n'
                   '~~~html\n<a href="../chapter/README.md">example</a>\n~~~\n')
        rewritten = readme_links.rewrite_content(content, 'guide/page.md', readme_links.chapter_routes(self.book))
        self.assertEqual(rewritten, content.replace('[real](../chapter/README.md)', '[real](../chapter/index.html)'))

    def test_code_in_a_real_link_label_does_not_hide_the_link(self):
        content = '[`Skill`](../chapter/README.md#target) and `[example](../chapter/README.md)`'
        rewritten = readme_links.rewrite_content(content, 'guide/page.md', readme_links.chapter_routes(self.book))
        self.assertEqual(rewritten, '[`Skill`](../chapter/index.html#target) and `[example](../chapter/README.md)`')

    def test_standalone_protocol_is_compatible_with_system_python3(self):
        program = ROOT / 'tools/readme_links.py'
        result = subprocess.run(['python3', str(program), 'supports', 'html'], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(['python3', str(program)], input=json.dumps([{}, copy.deepcopy(self.book)]),
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), self.book)

    def test_required_renderer_gate_cannot_skip_missing_or_wrong_versions(self):
        for binary, actual in ((None, ''), ('/fixture/mdbook', 'mdbook v0.0.0')):
            with self.subTest(binary=binary), \
                    patch.object(shutil, 'which', return_value=binary), \
                    patch.object(subprocess, 'check_output', return_value=actual):
                with patch.dict(globals(), REQUIRE_TOOLS=False), self.assertRaises(unittest.SkipTest):
                    RealRenderedLinkTests.setUpClass()
                with patch.dict(globals(), REQUIRE_TOOLS=True), self.assertRaises(RuntimeError):
                    RealRenderedLinkTests.setUpClass()


class RealRenderedLinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for tool, version in build_books.VERSIONS.items():
            binary = shutil.which(tool)
            if binary is None:
                if REQUIRE_TOOLS:
                    raise RuntimeError(f'Required rendering tool missing: {tool} {version}')
                raise unittest.SkipTest(f'{tool} missing; run with --require-tools after installation')
            actual = subprocess.check_output([binary, '--version'], text=True).strip()
            expected = actual.removeprefix(tool + ' ').removeprefix('v')
            if expected != version:
                message = f'Expected {tool} {version}; found {actual}'
                if REQUIRE_TOOLS:
                    raise RuntimeError(message)
                raise unittest.SkipTest(message + '; run --require-tools after installation')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='rendered-readme-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / 'output'
        (self.root / 'tools').mkdir()
        shutil.copyfile(ROOT / 'tools/readme_links.py', self.root / 'tools/readme_links.py')
        (self.root / 'theme').mkdir()
        (self.root / 'theme/custom.css').write_text('body { color: black; }', encoding='utf-8')
        for language, config in (('zh', 'book.toml'), ('en', 'book-en.toml')):
            shutil.copyfile(ROOT / config, self.root / config)
            directory = '章节' if language == 'zh' else 'chapter'
            source = self.root / 'src' / language
            (source / directory).mkdir(parents=True)
            (source / 'SUMMARY.md').write_text(
                f'# Summary\n\n- [Start](start.md)\n- [Chapter]({directory}/README.md)\n'
                f'  - [Nested]({directory}/nested.md)\n', encoding='utf-8')
            other = 'en' if language == 'zh' else 'zh'
            (source / 'start.md').write_text(
                '# Start\n\n'
                f'[plain]({directory}/README.md)\n\n[dot](./{directory}/README.md#target)\n\n'
                f'[reference][chapter]\n\n[chapter]: ./{directory}/README.md#target "chapter"\n\n'
                f'[`code label`](./{directory}/README.md#target)\n\n'
                f'<a href="./{directory}/README.md?mode=small&amp;view=1#target">html</a>\n\n'
                f'[other language](../{other}/start.html#start)\n\n'
                f'`[literal](./{directory}/README.md)`\n\n'
                f'```markdown\n[example](./{directory}/README.md)\n```\n\n'
                'Inline math: $x^2$.\n\n$$y = x + 1$$\n', encoding='utf-8')
            (source / directory / 'README.md').write_text(
                '# Chapter\n\n## Target\n\n[child](nested.md)\n', encoding='utf-8')
            (source / directory / 'nested.md').write_text(
                f'# Nested\n\n[parent](./README.md#target)\n\n'
                f'[relative](../{directory}/README.md#target)\n\n', encoding='utf-8')

    def assert_rendered_contract(self):
        counts, errors = check_rendered_book(self.output, root=self.root)
        self.assertEqual(errors, [], '\n'.join(errors))
        for language in ('zh', 'en'):
            self.assertEqual(counts[language]['pages_checked'], 3)
            self.assertGreaterEqual(counts[language]['fragments'], 6)
            page = self.output / language / 'start.html'
            parsed = read_page(page)
            links = [target for _, tag, target in parsed.references if tag == 'a']
            self.assertFalse(any('README.html' in target for target in links), links)
            content = page.read_text(encoding='utf-8')
            self.assertIn('class="katex"', content)
            self.assertIn('katex-display', content)
            # Literal teaching examples retain README.md, even though actual
            # navigable links use the metadata-defined index.html route.
            self.assertIn('README.md', content)
            self.assertIn('index.html', content)

    def test_native_mdbook_build_keeps_body_links_and_katex_working(self):
        for language, config in (('zh', 'book.toml'), ('en', 'book-en.toml')):
            (self.root / 'active').mkdir(exist_ok=True)
            # Native mdBook uses book.toml; use isolated roots sharing no cwd or
            # PYTHONPATH assumptions, while retaining each repository config.
            book = self.root / 'active' / language
            book.mkdir()
            shutil.copyfile(self.root / config, book / 'book.toml')
            shutil.copytree(self.root / 'src', book / 'src')
            shutil.copytree(self.root / 'tools', book / 'tools')
            shutil.copytree(self.root / 'theme', book / 'theme')
            result = subprocess.run(['mdbook', 'build', str(book), '-d', str(self.output / language)],
                                    cwd=self.root.parent, capture_output=True, text=True, timeout=30,
                                    env={key: value for key, value in os.environ.items() if key != 'PYTHONPATH'})
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_rendered_contract()

    def test_temporary_staging_copies_the_configured_preprocessor_and_enforces_links(self):
        originals = {name: (self.root / name).read_bytes() for name in ('book.toml', 'book-en.toml')}
        with patch.object(build_books, 'ROOT', self.root), \
                patch.object(sys, 'argv', ['build_books.py', '--output', str(self.output)]), \
                contextlib.chdir(self.root.parent), contextlib.redirect_stdout(io.StringIO()) as log:
            build_books.main()
        self.assertIn('rendered link checks:', log.getvalue())
        self.assert_rendered_contract()
        for name, original in originals.items():
            self.assertEqual((self.root / name).read_bytes(), original)

    def test_order_alone_does_not_fix_the_pinned_native_readme_body_links(self):
        for name, katex in (('links', '[preprocessor.katex]\nafter=["links"]\noptional=false\n'),
                            ('ordered', '[preprocessor.katex]\nafter=["links","index"]\noptional=false\n'),
                            ('plain', '')):
            with self.subTest(configuration=name):
                book = self.root / name
                (book / 'src/chapter').mkdir(parents=True)
                (book / 'book.toml').write_text('[book]\nsrc="src"\n' + katex, encoding='utf-8')
                (book / 'src/SUMMARY.md').write_text('- [Start](start.md)\n- [Chapter](chapter/README.md)\n')
                (book / 'src/start.md').write_text('# Start\n\n[link](./chapter/README.md#target)\n\n$x^2$\n')
                (book / 'src/chapter/README.md').write_text('# Chapter\n\n## Target\n')
                result = subprocess.run(['mdbook', 'build', str(book)],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                content = (book / 'book/start.html').read_text()
                self.assertIn('href="./chapter/README.html#target"', content)
                self.assertTrue((book / 'book/chapter/index.html').exists())
                self.assertFalse((book / 'book/chapter/README.html').exists())


if __name__ == '__main__':
    REQUIRE_TOOLS = '--require-tools' in sys.argv
    if REQUIRE_TOOLS:
        sys.argv.remove('--require-tools')
    unittest.main()
