"""Convert docs/looped_minimind_paper.md into a formatted academic PDF.

Pipeline: Markdown -> HTML (MathJax + academic CSS) -> Chrome headless print-to-pdf.
Requires: python markdown lib + a local Chrome/Edge binary (auto-detected).
"""
import markdown
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_PATH = os.path.join(REPO, "docs", "looped_minimind_paper.md")
HTML_PATH = os.path.join(REPO, "docs", "looped_minimind_paper.html")
PDF_PATH = os.path.join(REPO, "docs", "looped_minimind_paper.pdf")

TITLE = "面向小语言模型的动态循环与奖励驱动早退机制"
EN_TITLE = "Dynamic Looping with Reward-Driven Early Exit for Small Language Models"

CSS = """
@page { size: A4; margin: 2.2cm 2.0cm 2.4cm 2.0cm; }
@page :first { margin-top: 2.0cm; }
html { font-size: 11pt; }
body {
  font-family: "Latin Modern Roman", "Times New Roman", "Songti SC", "SimSun", serif;
  line-height: 1.55; color: #111; max-width: 100%; margin: 0;
  text-align: justify; hyphens: auto;
}
/* ── Title block ── */
.title { text-align: center; margin-bottom: 0.4em; }
.title h1 { font-size: 16.5pt; font-weight: 700; line-height: 1.35; margin: 0 0 0.15em 0; }
.title .en { font-size: 12.5pt; font-weight: 600; color: #333; margin-bottom: 0.5em; }
.title .meta { font-size: 9.5pt; color: #555; margin-top: 0.4em; }
.rule { border: none; border-top: 0.8pt solid #444; margin: 0.8em 0 1.2em 0; }
/* ── Abstract ── */
.abstract { font-size: 10pt; line-height: 1.5; padding: 0.6em 1em; margin: 0 0 1.2em 0;
  border-left: 2.5pt solid #999; background: #fafafa; }
.abstract .k { font-weight: 700; }
.abstract .kw { margin-top: 0.6em; font-size: 9.5pt; }
.abstract .kw b { font-weight: 700; }
/* ── Headings ── */
h1, h2, h3 { font-weight: 700; color: #000; page-break-after: avoid; }
h2 { font-size: 13pt; margin: 1.3em 0 0.55em 0; border-bottom: 0.6pt solid #bbb; padding-bottom: 0.2em; }
h3 { font-size: 11.5pt; margin: 1.1em 0 0.45em 0; }
p { margin: 0.45em 0; }
/* ── Math ── */
mjx-container { font-size: 105%; }
/* display math */
mjx-container[display="true"] { margin: 0.7em 0; }
/* ── Tables ── */
table { border-collapse: collapse; margin: 0.9em auto; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 0.6pt solid #999; padding: 0.3em 0.55em; text-align: center; }
th { background: #f0f0f0; font-weight: 700; }
table + blockquote, p > em, .tablenote { font-size: 9pt; color: #444; text-align: center; margin-top: -0.5em; }
blockquote { margin: 0.3em auto 0.8em auto; padding: 0.2em 0; font-size: 9pt; color: #444; text-align: center; border: none; }
/* ── Lists ── */
ul, ol { margin: 0.4em 0; padding-left: 1.8em; }
li { margin: 0.25em 0; }
/* ── Code ── */
code { font-family: "Consolas", "Menlo", monospace; font-size: 0.88em; background: #f5f5f5; padding: 0.05em 0.25em; border-radius: 2px; }
pre { background: #f8f8f8; border: 0.5pt solid #ddd; padding: 0.6em; font-size: 9pt; overflow-x: auto; }
pre code { background: none; padding: 0; }
/* ── References ── */
h2#references, h2:has(+ ol) { }
ol, ul { }
.refs { font-size: 9.5pt; }
.refs li { margin: 0.28em 0; }
/* ── Footer page numbers (Chrome prints via @page counters not fully; handled by paged css if needed) ── */
"""


def detect_chrome():
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def split_front_matter(src):
    """Extract title lines (# ...) at top; rest is body."""
    lines = src.split("\n")
    body_lines = []
    for ln in lines:
        body_lines.append(ln)
    return "\n".join(body_lines)


def main():
    src = open(MD_PATH, encoding="utf-8").read()

    # Pull the title line and the English subtitle out of the body
    # (they are the first two '#' / '##' headings).
    title = TITLE
    en = EN_TITLE
    # remove original title headings so they are rendered by .title block
    src = re.sub(r"^# .*$", "", src, count=1, flags=re.M)
    src = re.sub(r"^## Dynamic Looping.*$", "", src, count=1, flags=re.M)
    src = re.sub(r"^---\s*$", "", src, count=1, flags=re.M)
    src = re.sub(r"^## 摘要\s*$", "", src, count=1, flags=re.M)

    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists"])
    body = md.convert(src)

    # Rebuild the reference list: markdown renders each ref as a <p> because they
    # are blank-line separated. Anchor strictly: a ref paragraph STARTS with
    # "[N] Author" (a citation number followed by a proper-noun author), which
    # inline body citations like "…[3]" never satisfy.
    ref_pattern = re.compile(r"<p>\[(\d+)\] [A-Z]\..*?</p>", re.S)
    segments = list(ref_pattern.finditer(body))
    if segments:
        first = segments[0].start()
        last = segments[-1].end()
        ref_items = []
        for m in segments:
            content = m.group(0)[3:-4]  # strip <p>...</p>
            ref_items.append(f"<li>{content}</li>")
        body = (
            body[:first]
            + '<ol class="refs">' + "".join(ref_items) + "</ol>"
            + body[last:]
        )

    # Abstract block: the first paragraph after 摘要 is the abstract; wrap it.
    # Since we stripped the 摘要 heading, the first <p> is the abstract.
    body = body.lstrip()
    m = re.match(r"^(<p>.*?</p>)", body, re.S)
    if m:
        abstract_text = m.group(1)
        # Insert keyword line marker: find the 关键词 paragraph
        rest = body[m.end():]
        km = re.search(r"(<p><strong>关键词</strong>.*?</p>)", rest, re.S)
        kw_html = ""
        if km:
            kw_html = km.group(1)
            rest = rest[:km.start()] + rest[km.end():]
        abstract_html = (
            '<div class="abstract">'
            f'<span class="k">摘要：</span>{abstract_text}'
            f'<div class="kw">{kw_html}</div>'
            "</div>"
        )
        body = abstract_html + "\n" + rest

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{CSS}</style>
<script>
window.MathJax = {{
  tex: {{ inlineMath: [['$','$'], ['\\\\(','\\\\)']], displayMath: [['$$','$$'], ['\\\\[','\\\\]']] }},
  svg: {{ fontCache: 'global' }},
  options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre'] }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>
</head>
<body>
<div class="title">
  <h1>{title}</h1>
  <div class="en">{en}</div>
  <div class="meta">MiniMind Project &mdash; Technical Report</div>
</div>
<hr class="rule">
{body}
</body>
</html>"""
    open(HTML_PATH, "w", encoding="utf-8").write(html)
    print(f"HTML written: {HTML_PATH}")

    chrome = detect_chrome()
    if chrome is None:
        print("ERROR: no Chrome/Edge found for PDF printing.", file=sys.stderr)
        sys.exit(1)

    # --print-to-pdf with prefer-css-page-size; disable header/footer default
    cmd = [
        chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
        "--print-to-pdf-no-header",
        "--print-to-pdf=" + PDF_PATH,
        "file:///" + HTML_PATH.replace("\\", "/"),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if os.path.exists(PDF_PATH) and os.path.getsize(PDF_PATH) > 0:
        print(f"PDF written: {PDF_PATH} ({os.path.getsize(PDF_PATH):,} bytes)")
    else:
        print("ERROR: PDF not produced.", file=sys.stderr)
        print(r.stdout[-2000:], file=sys.stderr)
        print(r.stderr[-2000:], file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
