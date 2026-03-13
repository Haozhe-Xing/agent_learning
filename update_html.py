#!/usr/bin/env python3
"""
Update mdBook HTML files from Markdown source.
Usage: python3 update_html.py <src_md_file> <book_html_file>
   or: python3 update_html.py  (auto-detect changed files)
"""

import sys
import re
import html
import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from pathlib import Path


def md_to_mdbook_html(md_content: str) -> str:
    """Convert Markdown content to mdBook-style HTML."""
    # Use Python-Markdown with extensions
    md = markdown.Markdown(extensions=[
        TableExtension(),
        FencedCodeExtension(),
        'markdown.extensions.nl2br',
    ])
    raw_html = md.convert(md_content)

    # Post-process: mdBook wraps tables in <div class="table-wrapper">
    raw_html = re.sub(
        r'<table>',
        r'<div class="table-wrapper"><table>',
        raw_html
    )
    raw_html = re.sub(
        r'</table>',
        r'</table>\n</div>',
        raw_html
    )

    # Post-process: mdBook adds anchor links to headings
    def add_anchor(m):
        level = m.group(1)
        inner = m.group(2)
        # Generate id: lowercase, replace spaces with hyphens, remove non-alphanumeric (keep CJK)
        heading_id = re.sub(r'[^\w\u4e00-\u9fff\u3000-\u303f]', '', inner.lower().replace(' ', '-'))
        heading_id = heading_id.strip('-')
        return (f'<h{level} id="{heading_id}">'
                f'<a class="header" href="#{heading_id}">{inner}</a>'
                f'</h{level}>')

    raw_html = re.sub(r'<h([1-6])>(.*?)</h\1>', add_anchor, raw_html)

    # Post-process: convert <code class="language-xxx"> style (already done by fenced_code)
    # Python-Markdown uses class="language-xxx", mdBook uses class="language-xxx" too — OK

    return raw_html


def update_html_file(md_path: str, html_path: str):
    """Replace the <main> content in an HTML file with converted Markdown."""
    md_file = Path(md_path)
    html_file = Path(html_path)

    if not md_file.exists():
        print(f"ERROR: Markdown file not found: {md_path}")
        sys.exit(1)
    if not html_file.exists():
        print(f"ERROR: HTML file not found: {html_path}")
        sys.exit(1)

    md_content = md_file.read_text(encoding='utf-8')
    html_content = html_file.read_text(encoding='utf-8')

    # Convert markdown to HTML
    new_main_content = md_to_mdbook_html(md_content)

    # Replace content between <main> and </main>
    pattern = r'(<main>\s*)(.*?)(\s*</main>)'
    replacement = r'\1' + new_main_content + r'\3'
    new_html, count = re.subn(pattern, replacement, html_content, flags=re.DOTALL)

    if count == 0:
        print(f"ERROR: Could not find <main>...</main> in {html_path}")
        sys.exit(1)

    html_file.write_text(new_html, encoding='utf-8')
    print(f"✅ Updated: {html_path}")
    print(f"   Source:  {md_path}")


def auto_detect_and_update():
    """Auto-detect src/*.md -> book/*.html mapping and update all."""
    project_root = Path(__file__).parent
    src_dir = project_root / 'src'
    book_dir = project_root / 'book'

    md_files = list(src_dir.rglob('*.md'))
    updated = 0
    skipped = 0

    for md_file in md_files:
        # Map src/chapter_xxx/foo.md -> book/chapter_xxx/foo.html
        rel = md_file.relative_to(src_dir)
        html_file = book_dir / rel.with_suffix('.html')

        if not html_file.exists():
            print(f"⚠️  No HTML target for: {md_file.relative_to(project_root)}")
            skipped += 1
            continue

        update_html_file(str(md_file), str(html_file))
        updated += 1

    print(f"\nDone: {updated} updated, {skipped} skipped.")


if __name__ == '__main__':
    if len(sys.argv) == 3:
        update_html_file(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 1:
        auto_detect_and_update()
    else:
        print("Usage:")
        print("  python3 update_html.py <src.md> <book.html>   # update single file")
        print("  python3 update_html.py                         # update all files")
        sys.exit(1)
