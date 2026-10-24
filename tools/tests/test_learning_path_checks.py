"""Regression cases for the book's structural editing contract; no build needed."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.book_structure import CHAPTERS, CHAPTER_BY_NUMBER, ITERATION_SECTIONS, OLD_TO_NEW, PARTS
from tools.check_learning_path import (check_heading_numbers, check_link_labels, check_navigation, check_readme,
                                      check_summary, collect_errors, number_identifier, parse_summary)


def summary(language='en'):
    lines = ['# Contents', '', '[Preface](./preface.md)', '']
    for part in PARTS:
        label = f'第{"一二三四五六七"[part.number - 1]}部分：学习' if language == 'zh' else f'Part {part.number}: Learning'
        lines.append(f'- [{label}](./{part.path})')
        for chapter in (item for item in CHAPTERS if item.part == part.number):
            label = f'第{chapter.number}章 主题' if language == 'zh' else f'Chapter {chapter.number}: Topic'
            lines += [f'- [{label}](./{chapter.path})',
                      f'  - [{chapter.number}.1 Section](./{chapter.slug}/01_topic.md)']
    return '\n'.join(lines)


def readme(number=10, language='en'):
    title, fields = ITERATION_SECTIONS[language]
    heading = f'第{number}章 主题' if language == 'zh' else f'Chapter {number}: Topic'
    lines = [f'# {heading}', '', f'| {number}.1 Section | Topic |', '', f'## {title}']
    lines += [f'- **{field}**: Specific chapter {number} behavior and acceptance evidence.' for field in fields[:-1]]
    lines += [f'- **{fields[-1]}**: Continue with [the project](../learning_project.md).']
    return '\n'.join(lines)


class LearningPathChecksTest(unittest.TestCase):
    def test_manifest_covers_new_order_without_renaming_urls(self):
        self.assertEqual([chapter.number for chapter in CHAPTERS], list(range(1, 26)))
        self.assertEqual(CHAPTER_BY_NUMBER[10].slug, 'chapter_langchain')
        self.assertEqual(CHAPTER_BY_NUMBER[16].slug, 'chapter_18_multi_agent')
        self.assertEqual(CHAPTER_BY_NUMBER[24].slug, 'chapter_agentic_rl')
        self.assertEqual(CHAPTER_BY_NUMBER[25].slug, 'chapter_self_evolving')
        self.assertEqual([chapter.part for chapter in CHAPTERS],
                         [1]*2 + [2]*7 + [3]*6 + [4]*2 + [5]*3 + [6]*3 + [7]*2)
        self.assertEqual(OLD_TO_NEW[12], 10)
        self.assertEqual(OLD_TO_NEW[10], 24)

    def test_bilingual_summaries_follow_same_manifest(self):
        for language in ('zh', 'en'):
            self.assertEqual(check_summary(summary(language), language), [])
        self.assertEqual([entry.path for entry in parse_summary(summary('zh'))],
                         [entry.path for entry in parse_summary(summary('en'))])

    def test_wrong_chapter_number_and_part_are_rejected(self):
        changed = summary().replace('Chapter 10: Topic', 'Chapter 12: Topic')
        self.assertTrue(any('must be chapter 10' in error for error in check_summary(changed, 'en')))
        changed = summary().replace('- [Part 7: Learning](./part7.md)', '- [Part 6: Learning](./part6.md)')
        errors = check_summary(changed, 'en')
        self.assertTrue(any('belongs to part 7' in error for error in errors))
        self.assertTrue(any('part sequence' in error for error in errors))

    def test_duplicate_and_reordered_chapters_are_rejected(self):
        changed = summary().replace('chapter_langchain/README.md', 'chapter_langgraph/README.md')
        errors = check_summary(changed, 'en')
        self.assertTrue(any('canonical 25-chapter order' in error for error in errors))
        self.assertTrue(any('duplicate navigation page' in error for error in errors))

    def test_section_must_be_under_its_chapter_with_new_number(self):
        changed = summary().replace('  - [10.1 Section]', '  - [12.1 Section]')
        self.assertTrue(any('section number must start with 10.' in error for error in check_summary(changed, 'en')))
        changed = summary().replace('  - [10.1 Section]', '- [10.1 Section]')
        self.assertTrue(any('section must be nested' in error for error in check_summary(changed, 'en')))

    def test_unregistered_chapters_and_duplicate_section_numbers_are_rejected(self):
        changed = summary() + '\n- [Chapter 26: Extra](./chapter_extra/README.md)'
        self.assertTrue(any('absent from the chapter manifest' in error for error in check_summary(changed, 'en')))
        changed = summary().replace('  - [10.1 Section](./chapter_langchain/01_topic.md)',
                                    '  - [10.1 Section](./chapter_langchain/01_topic.md)\n'
                                    '  - [10.1 Other](./chapter_langchain/02_topic.md)')
        self.assertTrue(any('duplicate navigation number' in error for error in check_summary(changed, 'en')))

    def test_navigation_h1_uses_labels_instead_of_historical_directory_digits(self):
        entries = parse_summary('- [Chapter 16: Teams](./chapter_18_multi_agent/README.md)\n'
                                '  - [16.2b Distributed](./chapter_18_multi_agent/02b_topic.md)')
        pages = {'chapter_18_multi_agent/README.md': '# Chapter 16: Teams',
                 'chapter_18_multi_agent/02b_topic.md': '# 16.2b Distributed'}
        self.assertEqual(check_navigation(entries, pages, 'en'), [])
        pages['chapter_18_multi_agent/02b_topic.md'] = '# 18.2b Distributed'
        self.assertTrue(any('H1 number must match' in error for error in check_navigation(entries, pages, 'en')))

    def test_identifiers_cover_parts_appendices_and_section_suffixes(self):
        for text, expected in [('Part VII: Research', 'part:7'), ('第七部分：研究', 'part:7'),
                               ('Appendix F: Setup', 'appendix:F'), ('F.1 Python', 'F.1'),
                               ('**24.2b** | Distributed', '24.2b'),
                               ('第 25.3 节的数据飞轮', '25.3'),
                               ("Section 24.2's Graduation Criteria", '24.2')]:
            self.assertEqual(number_identifier(text), expected)

    def test_readme_requires_concrete_fields_inside_iteration_heading(self):
        for language in ('zh', 'en'):
            self.assertEqual(check_readme(readme(language=language), language, {'10.1'}), [])
        changed = readme().replace('## This chapter\'s iteration', '## Overview')
        self.assertTrue(any('missing ##' in error for error in check_readme(changed, 'en', {'10.1'})))
        changed = readme().replace('Specific chapter 10 behavior and acceptance evidence.', 'TODO', 1)
        self.assertTrue(any('non-placeholder' in error for error in check_readme(changed, 'en', {'10.1'})))
        changed = readme().replace('- **How to verify**:', '## Unrelated\n- **How to verify**:')
        self.assertTrue(any('How to verify field' in error for error in check_readme(changed, 'en', {'10.1'})))

    def test_duplicate_iteration_fields_and_missing_next_link_are_rejected(self):
        changed = readme() + '\n- **Previous limitation**: Duplicate chapter description.'
        self.assertTrue(any('exactly one Previous limitation' in error
                            for error in check_readme(changed, 'en', {'10.1'})))
        changed = readme().replace('[the project](../learning_project.md)', 'the next learning page')
        self.assertTrue(any('must link' in error for error in check_readme(changed, 'en', {'10.1'})))

    def test_local_table_checks_section_membership(self):
        changed = readme().replace('| 10.1 Section', '| **12.1** Section')
        self.assertTrue(any('local table section 12.1' in error
                            for error in check_readme(changed, 'en', {'10.1'})))

    def test_section_link_numbers_are_checked_and_anchors_are_deliberately_skipped(self):
        pages = {'chapter_agentic_rl/02_sft_lora.md': '# 24.2 SFT'}
        text = "See [Section 12.2's SFT Criteria](./02_sft_lora.md)."
        self.assertTrue(any('label 12.2' in error
                            for error in check_link_labels(text, 'chapter_agentic_rl/01_topic.md', pages)))
        text = '[24.2 SFT](./02_sft_lora.md) and [24.2.1 Deep](./02_sft_lora.md#deep)'
        self.assertEqual(check_link_labels(text, 'chapter_agentic_rl/01_topic.md', pages), [])

    def test_next_captions_and_sections_inside_link_labels_are_checked(self):
        pages = {'chapter_25_multimodal/02_topic.md': '# 23.2 Images',
                 'chapter_self_evolving/03_topic.md': '# 25.3 Data'}
        text = '[Next: 21.2 Images →](./02_topic.md)'
        self.assertTrue(any('label 21.2' in error
                            for error in check_link_labels(text, 'chapter_25_multimodal/01_topic.md', pages)))
        text = '[the data flywheel in Section 11.3](./03_topic.md)'
        self.assertTrue(any('label 11.3' in error
                            for error in check_link_labels(text, 'chapter_self_evolving/02_topic.md', pages)))
        text = '[the data flywheel in Chapter 25](./03_topic.md)'
        self.assertEqual(check_link_labels(text, 'chapter_self_evolving/02_topic.md', pages), [])

    def test_chapter_number_before_link_and_directory_targets_are_checked(self):
        pages = {'chapter_20_evaluation/README.md': '# Chapter 18: Evaluation'}
        source = 'chapter_langchain/README.md'
        self.assertTrue(check_link_labels('Chapter 20 [Evaluation](../chapter_20_evaluation/README.md)', source, pages))
        self.assertEqual(check_link_labels('[Chapter 18 Evaluation](../chapter_20_evaluation/)', source, pages), [])

    def test_fenced_examples_are_not_treated_as_navigation_or_links(self):
        text = '```markdown\n[Chapter 12](./02_sft_lora.md)\n```'
        self.assertEqual(parse_summary(text), [])
        self.assertEqual(check_link_labels(text, 'chapter_agentic_rl/01_topic.md',
                                          {'chapter_agentic_rl/02_sft_lora.md': '# 24.2 SFT'}), [])

    def test_algorithm_local_heading_numbers_must_not_collapse_to_page_number(self):
        correct = ('# 24.3 PPO\n## Prerequisites\n### 1.1 Motivation\n### 1.2 Sampling\n'
                   '### 1.8 Implementation\n#### 1.8.1 Advantage estimation\n'
                   '#### 1.8.2 Clip loss\n')
        self.assertEqual(check_heading_numbers(correct), [])
        collapsed = correct.replace('### 1.1 ', '### 24.3 ').replace('### 1.2 ', '### 24.3 ')
        errors = check_heading_numbers(collapsed)
        self.assertEqual(len(errors), 1)
        self.assertIn('duplicate sibling heading number 24.3', errors[0])
        self.assertIn('Sampling', errors[0])

    def test_local_sequences_may_restart_in_a_different_parent_and_use_letter_suffixes(self):
        text = ('# 24.5 GRPO\n## Rewards\n### 5.3 Accuracy\n### 5.3b Persona\n'
                '## Another example\n### 5.3 Accuracy\n### 5.3b Persona\n'
                '```markdown\n### 5.3b Duplicate inside an example\n```\n')
        self.assertEqual(check_heading_numbers(text), [])
        text += '### 5.3b Another persona\n'
        self.assertIn('duplicate sibling heading number 5.3b', check_heading_numbers(text)[0])

    def test_global_subheadings_keep_their_full_unique_prefixes(self):
        text = '# 25.1 Prompt evolution\n## 25.1.1 Feedback\n## 25.1.2 Held-out tasks\n'
        self.assertEqual(check_heading_numbers(text), [])

    def test_checkout_check_catches_broken_next_link_and_language_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for language in ('zh', 'en'):
                directory = root / 'src' / language
                directory.mkdir(parents=True)
                (directory / 'SUMMARY.md').write_text(summary(language))
                (directory / 'preface.md').write_text('# Preface')
                (directory / 'learning_project.md').write_text('# Project')
                for part in PARTS:
                    label = f'第{"一二三四五六七"[part.number-1]}部分：学习' if language == 'zh' else f'Part {part.number}: Learning'
                    (directory / part.path).write_text('# ' + label)
                for chapter in CHAPTERS:
                    (directory / chapter.slug).mkdir()
                    (directory / chapter.path).write_text(readme(chapter.number, language))
                    (directory / chapter.slug / '01_topic.md').write_text(f'# {chapter.number}.1 Section')
            self.assertEqual(collect_errors(root), [])
            path = root / 'src/en/chapter_langchain/README.md'
            path.write_text(path.read_text().replace('../learning_project.md', '../missing.md'))
            self.assertTrue(any('next learning page does not exist' in error for error in collect_errors(root)))
            path = root / 'src/en/SUMMARY.md'
            path.write_text(path.read_text().replace('Chapter 10: Topic', 'Chapter 12: Topic'))
            self.assertTrue(any('differ between zh and en' in error for error in collect_errors(root)))
            path = root / 'src/en/chapter_langgraph/README.md'
            path.write_text('# Chapter 11\n```python\nunterminated')
            self.assertTrue(any('unclosed python fence' in error for error in collect_errors(root)))


if __name__ == '__main__':
    unittest.main()
