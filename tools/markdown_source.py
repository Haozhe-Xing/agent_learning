"""Read Markdown fences with matching delimiter lengths, including nested examples."""
import re


def split_source(text):
    prose, blocks = [], []
    fence, language, body, start = None, None, [], 0
    for number, line in enumerate(text.splitlines(), 1):
        opening = re.fullmatch(r'(`{3,}|~{3,})(\w*)\s*', line)
        if fence is None and opening:
            fence, language, start, body = opening[1], opening[2], number, []
        elif fence is not None and line.strip() == fence:
            blocks.append((language, start, '\n'.join(body)))
            fence = None
        elif fence is None:
            prose.append(line)
        else:
            body.append(line)
    if fence is not None:
        raise ValueError(f'unclosed {language or "text"} fence at line {start}')
    return '\n'.join(prose), blocks
