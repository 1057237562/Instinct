"""Small, side-effect-free helpers shared by the chat WebUI and its tests."""

import base64
import html
import re
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path


def animated_stream_html(previous, chunk):
    """Animate only fresh text in the browser, with no Python sleeps or token polling."""
    count = max(1, len(chunk))
    tail = ''.join(f'<span style="animation-delay:{min(i * 12, i * 240 / count):.0f}ms">{html.escape(char)}</span>'
                   for i, char in enumerate(chunk))
    return ('<style>@keyframes instinct-reveal{from{opacity:0}to{opacity:1}}'
            '.instinct-stream span{opacity:0;animation:instinct-reveal 60ms linear forwards;}'
            '@media(prefers-reduced-motion:reduce){.instinct-stream span{animation:none;opacity:1;}}'
            '</style><div class="instinct-stream" style="white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6">'
            + html.escape(previous) + tail + '</div>')


def render_animated_stream(placeholder, previous, chunk):
    """Bypass Markdown entirely: backticks and blank lines must remain text."""
    import streamlit as st
    with placeholder.container():
        st.html(animated_stream_html(previous, chunk))


def _thinking_parts(content, implicit=False):
    """Split control tags, preserving literal tags inside fenced/inline code."""
    parts, buffer = [], []
    # UI toggles / a lone closing tag cannot manufacture a thinking region.
    thinking = False
    fence = None
    inline = 0
    cursor = 0
    for match in re.finditer(r'`+|~{3,}|</?think>', content):
        token = match.group()
        prefix = content[content.rfind('\n', 0, match.start()) + 1:match.start()]
        is_fence = token[0] in '`~' and len(token) >= 3 and len(prefix) <= 3 and not prefix.strip()
        if fence:
            if is_fence and token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if is_fence:
            fence = token
            continue
        if token[0] == '`':
            inline = 0 if inline == len(token) else (len(token) if not inline else inline)
            continue
        if inline or token[0] != '<':
            continue
        buffer.append(content[cursor:match.start()])
        text = ''.join(buffer)
        if text or (thinking and token == '</think>'):
            parts.append((thinking, text, token == '</think>'))
        buffer = []
        thinking = token == '<think>'
        cursor = match.end()
    buffer.append(content[cursor:])
    text = ''.join(buffer)
    # A control tag can be split between two 16-token transfers.
    text = re.sub(r'</?t(?:h(?:i(?:n(?:k)?)?)?)?$', '', text)
    if text or thinking:
        parts.append((thinking, text, False))
    return parts


class _AnimateTextNodes(HTMLParser):
    def __init__(self, start):
        super().__init__(convert_charrefs=True)
        self.position = 0
        self.start = start
        self.output = []
        self.spans = 0

    def handle_starttag(self, tag, attrs):
        self.output.append(self.get_starttag_text())

    def handle_startendtag(self, tag, attrs):
        self.output.append(self.get_starttag_text())

    def handle_endtag(self, tag):
        self.output.append(f'</{tag}>')

    def handle_data(self, data):
        # Stable text is appended in one operation. Only the fresh tail gets
        # animated nodes, bounded to 64 for the entire update.
        stable = len(data) if self.start < 0 else min(len(data), max(0, self.start - self.position))
        self.output.append(html.escape(data[:stable]))
        self.position += stable
        tail = data[stable:]
        group_size = max(1, (len(tail) + 31) // 32)
        for offset in range(0, len(tail), group_size):
            part = tail[offset:offset + group_size]
            escaped = html.escape(part)
            if self.spans < 64 and not part.isspace():
                delay = min(self.spans * 6, 180)
                escaped = f'<span class="instinct-new" style="animation-delay:{delay}ms">{escaped}</span>'
                self.spans += 1
            self.output.append(escaped)
            self.position += len(part)


@lru_cache(maxsize=1)
def _markdown_parser():
    from markdown_it import MarkdownIt
    return MarkdownIt('commonmark', {'html': False}).enable('table')


@lru_cache(maxsize=8)
def _render_markdown_body(text, thinking, streaming):
    output = []
    for is_think, body, closed in _thinking_parts(text, implicit=thinking):
        rendered = _markdown_parser().render(body)
        if is_think:
            label = '思考中…' if streaming and not closed else '思考过程'
            rendered = ('<details open class="instinct-think"><summary>' + label
                        + '</summary>' + (rendered or '<p>等待思考内容…</p>') + '</details>')
        output.append(rendered)
    return ''.join(output)


def markdown_stream_html(content, previous='', thinking=False, streaming=True):
    def render(text):
        return _render_markdown_body(text, thinking, streaming)
    rendered = render(content)
    # Compare visible text; animate text nodes after Markdown parsing, never
    # insert animation HTML into a Markdown code fence.
    old = html.unescape(re.sub('<[^>]+>', '', render(previous)))
    new = html.unescape(re.sub('<[^>]+>', '', rendered))
    common = 0
    for a, b in zip(old, new):
        if a != b:
            break
        common += 1
    # A changed think label / newly closed Markdown delimiter must not animate
    # the entire existing response again.
    common = max(common, len(new) - max(0, len(content) - len(previous)))
    animated = _AnimateTextNodes(common if streaming and len(content) < 50000 else -1)
    animated.feed(rendered)
    return ('<style>@keyframes instinct-reveal{from{opacity:0}to{opacity:1}}'
            '.instinct-new{animation:instinct-reveal 60ms linear both;}'
            '.instinct-markdown pre{overflow-x:auto;padding:12px;background:rgba(128,128,128,.12);border-radius:8px;}'
            '.instinct-markdown pre code{white-space:pre;}'
            '.instinct-think{border-left:2px solid #888;padding-left:12px;margin:8px 0;}'
            '.instinct-think summary{color:#888;cursor:pointer;}'
            '@media(prefers-reduced-motion:reduce){.instinct-new{animation:none;}}'
            '</style><div class="instinct-markdown">' + ''.join(animated.output) + '</div>')


@lru_cache(maxsize=8)
def _contains_code_block(body):
    return any(token.type in ('fence', 'code_block') for token in _markdown_parser().parse(body))


def generation_loading_html():
    return ('<style>@keyframes instinct-loading{0%,80%,100%{opacity:.25;transform:translateY(0)}'
            '40%{opacity:1;transform:translateY(-3px)}}'
            '.instinct-loading i{display:inline-block;width:5px;height:5px;border-radius:50%;'
            'background:currentColor;margin:0 3px;animation:instinct-loading 1.2s infinite;}'
            '.instinct-loading i:nth-child(2){animation-delay:.15s}'
            '.instinct-loading i:nth-child(3){animation-delay:.3s}'
            '@media(prefers-reduced-motion:reduce){.instinct-loading i{animation:none;opacity:.7;}}'
            '</style><div class="instinct-loading" role="status" aria-label="正在生成" '
            'style="opacity:.75;font-size:13px;padding:6px 0">'
            '<span>正在生成</span> <i></i><i></i><i></i></div>')


@lru_cache(maxsize=8)
def _code_segments(body):
    tokens = _markdown_parser().parse(body)
    code = [token for token in tokens if token.type in ('fence', 'code_block')]
    # Keep nested list/blockquote structure under the native Markdown renderer.
    if any(token.level != 0 for token in code):
        return [('code', body, 0)]
    lines = body.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    parts, cursor = [], 0
    for token in code:
        start, end = (offsets[index] for index in token.map)
        if start > cursor:
            parts.append(('text', body[cursor:start], cursor))
        parts.append(('code', body[start:end], start))
        cursor = end
    if cursor < len(body):
        parts.append(('text', body[cursor:], cursor))
    return parts


def render_markdown_stream(placeholder, content, previous='', thinking=False, streaming=True):
    import streamlit as st
    parts = _thinking_parts(content)
    with placeholder.container():
        if not any(_contains_code_block(body) for _, body, _ in parts):
            st.html(markdown_stream_html(content, previous, thinking, streaming))
            return
        # Native Markdown owns code rendering, syntax highlighting and clipboard
        # controls. Never inject animated spans into its Markdown input.
        old_parts = _thinking_parts(previous)
        for index, (is_think, body, closed) in enumerate(parts):
            def render_body():
                old = old_parts[index][1] if index < len(old_parts) and old_parts[index][0] == is_think else ''
                if _contains_code_block(body):
                    for kind, text, start in _code_segments(body):
                        if kind == 'code':
                            st.markdown(text)
                        elif text.strip():
                            st.html(markdown_stream_html(text, old[start:start + len(text)], streaming=streaming))
                else:
                    st.html(markdown_stream_html(body, old, streaming=streaming))
            if is_think:
                with st.expander('思考中…' if streaming and not closed else '思考过程', expanded=True):
                    render_body()
            else:
                render_body()


CONVERSATION_STATE_KEYS = (
    "messages",
    "chat_messages",
    "regenerate",
    "last_user_message",
    "regenerate_index",
    "confirm_clear_chat",
)

MODEL_STATE_KEYS = (
    "model",
    "tokenizer",
    "model_loaded",
    "loaded_weight_path",
    "loaded_config_path",
)


def clear_conversation_state(state):
    """Remove conversation-only state without unloading the model or settings."""
    for key in CONVERSATION_STATE_KEYS:
        state.pop(key, None)


def detach_model_state(state):
    """Detach model resources and chat history from one WebUI session.

    The returned objects remain alive until the caller explicitly drops them.
    This lets the WebUI move a CUDA model to CPU before garbage collection while
    keeping this helper independent from PyTorch and Streamlit.
    """
    model = state.pop("model", None)
    tokenizer = state.pop("tokenizer", None)
    for key in MODEL_STATE_KEYS[2:]:
        state.pop(key, None)
    clear_conversation_state(state)
    return model, tokenizer


def encode_clipboard_text(text):
    """Encode arbitrary Unicode safely for embedding in component JavaScript."""
    return base64.b64encode(str(text).encode("utf-8")).decode("ascii")


def queue_last_response_regeneration(state):
    """Remove the last turn and queue its user prompt for fresh generation.

    The display history contains only user/final-assistant messages, while the
    model history can additionally contain system, tool-call, and tool-result
    messages.  Both histories are rewound to just before the last user prompt.
    """
    messages = state.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    if messages[-1].get("role") != "assistant":
        return False

    user_index = next(
        (
            index
            for index in range(len(messages) - 2, -1, -1)
            if messages[index].get("role") == "user"
        ),
        None,
    )
    if user_index is None:
        return False
    prompt = messages[user_index].get("content")
    if not isinstance(prompt, str) or not prompt:
        return False

    del messages[user_index:]
    model_messages = state.get("chat_messages")
    if isinstance(model_messages, list):
        model_user_index = next(
            (
                index
                for index in range(len(model_messages) - 1, -1, -1)
                if model_messages[index].get("role") == "user"
            ),
            None,
        )
        if model_user_index is not None:
            del model_messages[model_user_index:]

    state["regenerate"] = True
    state["last_user_message"] = prompt
    state.pop("regenerate_index", None)
    return True


def resolve_model_config_path(weight_path: str, repo_root: str) -> str:
    """Resolve the config matching a raw weight file.

    Training writes lightweight weights to ``out/`` and the matching config to
    ``checkpoints/``.  Prefer a config beside the selected weight (for custom
    exports), then the checkpoint copy, and only then the legacy pretrain
    fallback.
    """
    root = Path(repo_root)
    weight = Path(weight_path)
    config_name = weight.with_suffix(".json").name
    candidates = (
        weight.with_suffix(".json"),
        root / "checkpoints" / config_name,
        root / "trainer" / "config_pretrain.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ""
