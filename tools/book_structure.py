"""Canonical reader order; directory names are stable publication URLs.

Chapter numbers belong to this manifest, not to the historical digits embedded
in some directory names. The two SUMMARY files must follow the same structure.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chapter:
    number: int
    slug: str
    part: int

    @property
    def path(self) -> str:
        return f'{self.slug}/README.md'


@dataclass(frozen=True)
class Part:
    number: int
    slugs: tuple[str, ...]

    @property
    def path(self) -> str:
        return f'part{self.number}.md'


PARTS = (
    Part(1, ('chapter_intro', 'chapter_llm')),
    Part(2, ('chapter_tools', 'chapter_memory', 'chapter_planning', 'chapter_rag',
             'chapter_context_engineering', 'chapter_harness', 'chapter_skill')),
    Part(3, ('chapter_langchain', 'chapter_langgraph', 'chapter_openclaw',
             'chapter_hermes', 'chapter_claude_code', 'chapter_deepseek_harness')),
    Part(4, ('chapter_18_multi_agent', 'chapter_19_protocol')),
    Part(5, ('chapter_20_evaluation', 'chapter_21_security', 'chapter_22_deployment')),
    Part(6, ('chapter_23_coding_agent', 'chapter_24_data_agent', 'chapter_25_multimodal')),
    Part(7, ('chapter_agentic_rl', 'chapter_self_evolving')),
)
CHAPTERS = tuple(
    Chapter(number, slug, part.number)
    for number, (part, slug) in enumerate(
        ((part, slug) for part in PARTS for slug in part.slugs), 1)
)
CHAPTER_BY_SLUG = {chapter.slug: chapter for chapter in CHAPTERS}
CHAPTER_BY_NUMBER = {chapter.number: chapter for chapter in CHAPTERS}
LANGUAGES = ('zh', 'en')

# For one-time migrations only: apply to original tokens in a single pass.
OLD_TO_NEW = {**{number: number for number in range(1, 10)},
              10: 24, 11: 25,
              **{number: number - 2 for number in range(12, 26)}}

ITERATION_SECTIONS = {
    'zh': ('本章迭代', ('上一版的不足', '本章增加', '怎样验证', '下一步')),
    'en': ("This chapter's iteration",
           ('Previous limitation', 'This chapter adds', 'How to verify', 'Next step')),
}
