"""Publication staging and XML coverage, independent of visual semantics."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools import build_books, check_book


SVG = '<svg xmlns="http://www.w3.org/2000/svg"><title>Fixture</title></svg>'


class PublicationAssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for language, config in (('zh', 'book.toml'), ('en', 'book-en.toml')):
            (self.root / config).write_text(
                f'[book]\nsrc = "src/{language}"\nlanguage = "{language}"\n'
                '[preprocessor.katex]\noptional = false\n', encoding='utf-8')
            self.write(f'src/{language}/SUMMARY.md', '- [Chapter](chapter.md)\n')
            self.write(f'src/{language}/chapter.md', '# Chapter\n')
            self.write(f'src/{language}/svg/main.svg', SVG)
        self.write('theme/custom.css', 'body { color: black; }')
        # A broken historical source must be preserved, not parsed or published.
        self.backup = self.write('src/en/svg_backup_broken/old.svg', '<broken historical copy')

    def write(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return path

    def test_recursive_svg_inventory_includes_nested_and_uppercase_assets(self):
        nested = self.write('src/en/chapter/diagrams/nested.SVG', SVG)
        self.assertIn(nested, build_books.publication_svgs(self.root))
        self.assertEqual(len(build_books.publication_svgs(self.root)), 3)
        self.assertNotIn(self.backup, build_books.publication_svgs(self.root))
        self.assertEqual(build_books.publication_asset_errors(self.root), [])

    def test_explicit_exclusion_does_not_hide_similarly_named_directories(self):
        ordinary = self.write('src/en/svg_backup_broken_copy/keep.svg', SVG)
        self.assertFalse(build_books.is_excluded_source(ordinary, self.root))
        stage = self.root / 'stage'
        stage.mkdir()
        before = self.backup.read_bytes()
        build_books.copy_publication_sources(self.root, stage)
        self.assertFalse((stage / 'src/en/svg_backup_broken').exists())
        self.assertTrue((stage / ordinary.relative_to(self.root)).exists())
        self.assertEqual(self.backup.read_bytes(), before)

    def test_invalid_nested_published_svg_is_rejected_but_archive_is_ignored(self):
        invalid = self.write('src/zh/chapter/figures/bad.svg', '<svg><g></svg>')
        errors = build_books.publication_asset_errors(self.root)
        self.assertEqual(len(errors), 1)
        self.assertIn(str(invalid.relative_to(self.root)), errors[0])
        self.assertIn('invalid SVG XML', errors[0])

    def test_reference_to_excluded_resource_is_rejected_across_supported_syntax(self):
        cases = (
            ('src/en/chapter.md', '![diagram](svg_backup_broken/old.svg)'),
            ('src/en/chapter.md', '![diagram][old]\n\n[old]: svg_backup_broken/old.svg'),
            ('src/en/chapter.md', '<img src="svg_backup_broken/old.svg">'),
            ('src/en/nested/page.html', '<a href="../svg_backup_broken/old.svg">old</a>'),
            ('src/en/theme.css', 'body { background: url("svg_backup_broken/old.svg"); }'),
            ('src/en/chapter.md', '![diagram](/svg_backup_broken/old.svg)'),
            ('src/en/chapter.md', '![diagram](svg%5Fbackup%5Fbroken/old.svg?rev=1#figure)'),
            ('src/en/icon.svg', '<svg xmlns="http://www.w3.org/2000/svg"><use href="svg_backup_broken/old.svg#icon"/></svg>'),
        )
        for relative, content in cases:
            with self.subTest(relative=relative, content=content):
                path = self.root / relative
                before = path.read_bytes() if path.exists() else None
                self.write(relative, content)
                errors = build_books.publication_asset_errors(self.root)
                self.assertTrue(any('references excluded publication resource' in error for error in errors), errors)
                if before is None:
                    path.unlink()
                else:
                    path.write_bytes(before)

    def test_external_links_and_fenced_hypothetical_paths_are_not_excluded_references(self):
        self.write('src/en/chapter.md',
                   '# Chapter\n\n![remote](https://example.com/svg_backup_broken/old.svg)\n\n'
                   '```markdown\n![hypothetical](svg_backup_broken/old.svg)\n```\n')
        self.assertEqual(build_books.publication_asset_errors(self.root), [])

    def test_check_book_uses_recursive_publication_inventory(self):
        self.write('src/en/nested/figure.svg', SVG)
        with patch.object(check_book, 'ROOT', self.root), contextlib.redirect_stdout(io.StringIO()) as output:
            with contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertFalse(check_book.check(), error.getvalue())
        self.assertIn('and 3 SVG files', output.getvalue())

    def test_check_book_rejects_an_existing_but_excluded_image(self):
        self.write('src/en/chapter.md', '# Chapter\n\n![old](svg_backup_broken/old.svg)')
        with patch.object(check_book, 'ROOT', self.root), contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertTrue(check_book.check())
        self.assertIn('references excluded publication resource', error.getvalue())

    def test_output_validation_checks_missing_and_invalid_assets_and_archive_leaks(self):
        nested = self.write('src/en/figures/deep/nested.svg', SVG)
        output = self.root / 'output'
        for language in ('zh', 'en'):
            shutil.copytree(self.root / 'src' / language, output / language,
                            ignore=shutil.ignore_patterns('svg_backup_broken'))
        counts, errors = build_books.validate_output_assets(output, ['zh', 'en'], self.root)
        self.assertEqual(counts, {'zh': 1, 'en': 2})
        self.assertEqual(errors, [])
        (output / 'en' / nested.relative_to(self.root / 'src/en')).unlink()
        self.write('output/en/generated/invalid.svg', '<svg>')
        self.write('output/en/svg_backup_broken/old.svg', SVG)
        _, errors = build_books.validate_output_assets(output, ['zh', 'en'], self.root)
        self.assertTrue(any('missing published SVG: figures/deep/nested.svg' in error for error in errors), errors)
        self.assertTrue(any('invalid output SVG XML' in error for error in errors), errors)
        self.assertTrue(any('excluded directory was published' in error for error in errors), errors)

    def test_build_main_stages_filtered_sources_and_preserves_original_configs(self):
        originals = {name: (self.root / name).read_bytes() for name in ('book.toml', 'book-en.toml')}
        backup = self.backup.read_bytes()
        output = self.root / 'result'
        built = []

        def version(command, **_kwargs):
            name = Path(command[0]).name
            return f'{name} {build_books.VERSIONS[name]}'

        def build(command, **kwargs):
            self.assertTrue(kwargs['check'])
            stage, destination = Path(command[2]), Path(command[4])
            language = destination.name
            self.assertEqual((stage / 'book.toml').read_bytes(),
                             originals['book.toml' if language == 'zh' else 'book-en.toml'])
            self.assertFalse((stage / 'src/en/svg_backup_broken').exists())
            self.assertTrue((stage / 'theme/custom.css').exists())
            shutil.copytree(stage / 'src' / language, destination)
            # The builder now also enforces the rendered-page contract. This
            # mock must represent HTML output, not just copied Markdown assets.
            (destination / 'chapter.html').write_text('<main><h1 id="chapter">Chapter</h1></main>', encoding='utf-8')
            built.append(language)

        with patch.object(build_books, 'ROOT', self.root), \
                patch.object(build_books.shutil, 'which', side_effect=lambda name: '/tools/' + name), \
                patch.object(build_books.subprocess, 'check_output', side_effect=version), \
                patch.object(build_books.subprocess, 'run', side_effect=build), \
                patch.object(sys, 'argv', ['build_books.py', '--output', str(output)]), \
                contextlib.redirect_stdout(io.StringIO()):
            build_books.main()
        self.assertEqual(built, ['zh', 'en'])
        for name, content in originals.items():
            self.assertEqual((self.root / name).read_bytes(), content)
        self.assertEqual(self.backup.read_bytes(), backup)

    def test_build_refuses_excluded_reference_before_staging(self):
        self.write('src/en/chapter.md', '![old](svg_backup_broken/old.svg)')
        with patch.object(build_books, 'ROOT', self.root), \
                patch.object(build_books, 'copy_publication_sources') as copy, \
                patch.object(sys, 'argv', ['build_books.py']), \
                self.assertRaisesRegex(SystemExit, 'references excluded publication resource'):
            build_books.main()
        copy.assert_not_called()


if __name__ == '__main__':
    unittest.main()
