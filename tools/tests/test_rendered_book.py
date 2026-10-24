"""Regression checks for the published routes, rather than source existence."""
from pathlib import Path
import tempfile
import unittest

from tools.check_rendered_book import check_rendered_book, rendered_path


class RenderedBookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'output'
        for language, config in (('zh', 'book.toml'), ('en', 'book-en.toml')):
            (self.root / config).write_text(f'[book]\nsrc = "src/{language}"\n')
            source = self.root / 'src' / language
            source.mkdir(parents=True)
            (source / 'SUMMARY.md').write_text('[Intro](./chapter/README.md)\n[Details](./chapter/details.md)\n')
            self.write(f'{language}/chapter/index.html', '<main><h1 id="intro">Intro</h1></main>')
            self.write(f'{language}/chapter/details.html', '<main><h1 id="细节">Details</h1></main>')

    def write(self, path, content):
        destination = self.output / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding='utf-8')
        return destination

    def check(self, languages=('zh', 'en')):
        return check_rendered_book(self.output, languages, self.root)

    def test_readme_route_and_summary_are_not_assumed_output_files(self):
        self.assertEqual(rendered_path('./chapter/README.md'), Path('chapter/index.html'))
        self.assertEqual(rendered_path('chapter/Readme.md'), Path('chapter/index.html'))
        counts, errors = self.check()
        self.assertEqual(errors, [])
        self.assertEqual(counts['zh']['pages_checked'], 2)
        self.assertFalse((self.output / 'zh/chapter/README.html').exists())

    def test_relative_cross_language_query_and_encoded_fragments(self):
        self.write('zh/chapter/index.html', '<main id="top"><a href="details.html?view=1#%E7%BB%86%E8%8A%82">one</a>'
                   '<a href="../../en/chapter/details.html#细节">two</a>'
                   '<a href="/en/chapter/details.html#细节">three</a>'
                   '<a href="./?x=1#top">directory</a><a href="#top">same page</a></main>')
        counts, errors = self.check()
        self.assertEqual(errors, [])
        self.assertEqual(counts['zh']['fragments'], 5)

    def test_missing_readme_html_is_not_hidden_by_existing_source_or_index(self):
        self.write('zh/chapter/index.html', '<main><a href="README.html">bad route</a></main>')
        _, errors = self.check(('zh',))
        self.assertEqual(len(errors), 1)
        self.assertIn("missing local file 'README.html'", errors[0])

    def test_missing_summary_listed_page_is_an_error(self):
        (self.output / 'zh/chapter/details.html').unlink()
        counts, errors = self.check(('zh',))
        self.assertEqual(counts['zh']['expected_pages'], 2)
        self.assertEqual(counts['zh']['pages_checked'], 1)
        self.assertIn('missing SUMMARY-listed output page', errors[0])

    def test_missing_cross_page_and_same_page_fragment(self):
        self.write('zh/chapter/index.html', '<main><a href="details.html#absent">cross</a>'
                   '<a href="#gone">same</a></main>')
        counts, errors = self.check(('zh',))
        self.assertEqual(counts['zh']['fragments'], 2)
        self.assertEqual(len(errors), 2)
        self.assertTrue(all('missing HTML fragment' in error for error in errors))

    def test_existing_and_missing_images(self):
        self.write('zh/picture space.svg', '<svg/>')
        self.write('zh/chapter/index.html', '<main><img src="../picture%20space.svg?x=1">'
                   '<img src="missing.svg"></main>')
        counts, errors = self.check(('zh',))
        self.assertEqual(counts['zh']['images'], 2)
        self.assertEqual(len(errors), 1)
        self.assertIn('missing local image', errors[0])

    def test_external_schemes_and_escaped_code_are_out_of_scope(self):
        self.write('zh/chapter/index.html', '<nav><a href="sidebar-missing.html">sidebar</a></nav>'
                   '<main><a href="https://example.test/no-page">web</a><a href="//example.test/a">relative web</a>'
                   '<a href="mailto:reader@example.test">mail</a><img src="data:image/png;base64,abc">'
                   '<a href="https://[">unverified external syntax</a><a href="//[">external syntax</a>'
                   '<pre><code>&lt;a href="missing.html#no"&gt;example&lt;/a&gt;'
                   '&lt;img src="absent.svg"&gt;</code></pre></main>')
        counts, errors = self.check(('zh',))
        self.assertEqual(errors, [])
        self.assertEqual(counts['zh']['external_references_skipped'], 6)
        self.assertEqual(counts['zh']['links'], 0)
        self.assertEqual(counts['zh']['images'], 0)

    def test_summary_link_is_not_a_published_directory_entry(self):
        self.write('zh/chapter/index.html', '<main><a href="../SUMMARY.html">Contents</a></main>')
        _, errors = self.check(('zh',))
        self.assertEqual(len(errors), 1)
        self.assertIn('SUMMARY.html', errors[0])

    def test_directory_index_must_exist(self):
        (self.output / 'zh/empty').mkdir()
        self.write('zh/chapter/index.html', '<main><a href="../empty/">empty</a></main>')
        _, errors = self.check(('zh',))
        self.assertEqual(len(errors), 1)
        self.assertIn('empty/index.html', errors[0])

    def test_existing_file_outside_publication_does_not_make_valid_route(self):
        (self.root / 'outside.html').write_text('<h1>Not published</h1>')
        self.write('zh/chapter/index.html', '<main><a href="../../../outside.html">escape</a></main>')
        _, errors = self.check(('zh',))
        self.assertEqual(len(errors), 1)
        self.assertIn('escapes published output', errors[0])

    def test_main_is_required_for_summary_content(self):
        self.write('zh/chapter/index.html', '<h1>Missing template main</h1>')
        _, errors = self.check(('zh',))
        self.assertEqual(len(errors), 1)
        self.assertIn('expected one <main>, found 0', errors[0])

    def test_invalid_decoded_target_is_an_error_instead_of_a_crash(self):
        self.write('zh/chapter/index.html', '<main><a href="%00.html">bad</a></main>')
        _, errors = self.check(('zh',))
        self.assertEqual(len(errors), 1)
        self.assertIn('invalid local target', errors[0])

    def test_encoded_summary_path_is_decoded_once(self):
        (self.root / 'src/zh/SUMMARY.md').write_text('[Percent](./literal%2520.md)\n')
        self.write('zh/literal%20.html', '<main>Literal percent in filename</main>')
        counts, errors = self.check(('zh',))
        self.assertEqual(errors, [])
        self.assertEqual(counts['zh']['pages_checked'], 1)

    def test_named_anchors_and_text_directives_are_not_missing_ids(self):
        self.write('zh/chapter/index.html', '<main><a name="legacy"></a><a href="#legacy">old</a>'
                   '<h1 id="intro">Intro</h1><a href="#:~:text=Intro">text</a>'
                   '<a href="#intro:~:text=Intro">id and text</a></main>')
        counts, errors = self.check(('zh',))
        self.assertEqual(errors, [])
        self.assertEqual(counts['zh']['fragments'], 2)


if __name__ == '__main__':
    unittest.main()
