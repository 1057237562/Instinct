# -*- coding: utf-8 -*-
"""HTML -> plain text conversion for competitive programming statements.

Shared by clean_code_contests.py / clean_taco.py. Codeforces/GeeksforGeeks
statements use a small HTML subset; MathJax ($...$) is kept verbatim as text.
<pre> blocks are preserved exactly (code samples must not be re-wrapped).
"""
from html.parser import HTMLParser
import re

_BLOCK = {"p", "div", "center", "ul", "ol", "table", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "br", "blockquote", "dl"}
_SKIP = {"script", "style", "head"}


class _StatementHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.images = 0
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self._skip_depth += 1
            return
        if tag == "img":
            self.images += 1
            return
        if tag == "li":
            self.parts.append("\n- ")
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "pre":
            self.parts.append("\n")
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(data)

    def text(self):
        out = "".join(self.parts)
        out = re.sub(r"[ \t]+\n", "\n", out)
        out = re.sub(r"\n{3,}", "\n\n", out)
        return out.strip()


def html_to_text(raw: str):
    """Returns (text, n_images). n_images counts <img> tags (dropped)."""
    if not raw:
        return "", 0
    p = _StatementHTML()
    p.feed(raw)
    p.close()
    return p.text(), p.images


def norm_hash_key(text: str) -> str:
    """Whitespace-collapsed key for exact within/cross-dataset dedup."""
    import hashlib
    return hashlib.md5(re.sub(r"\s+", " ", text or "").strip().encode("utf-8")).hexdigest()
