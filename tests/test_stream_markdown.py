import html
import re

from scripts.web_demo_utils import markdown_stream_html, _thinking_parts


def without_animation(markup):
    return re.sub(r'</?span\b[^>]*>', '', markup)


def test_markdown_is_rendered_during_generation():
    text = '# 标题\n\n**加粗**\n\n- item\n\n```python\ndef gcd(a, b):\n    return a\n'
    output = without_animation(markdown_stream_html(text))
    assert '<h1>标题</h1>' in output
    assert '<strong>加粗</strong>' in output
    assert '<li>item</li>' in output
    assert '<pre><code class="language-python">' in output
    assert 'def gcd(a, b):' in output
    assert '&lt;span' not in output


def test_thinking_requires_explicit_open_tag_and_shows_unclosed_content():
    initial = without_animation(markdown_stream_html('', thinking=True))
    assert '<details' not in initial
    assert '<details' not in markdown_stream_html('分析</think>答案', thinking=True)
    output = without_animation(markdown_stream_html('<think>**分析**', thinking=True))
    assert output.count('<details open') == 1
    assert '<strong>分析</strong>' in output
    done = without_animation(markdown_stream_html('<think>分析</think>\n\n## 答案'))
    assert '思考过程' in done
    assert '<h2>答案</h2>' in done
    assert done.index('</details>') < done.index('<h2>')


def test_code_tags_and_partial_control_tokens():
    assert _thinking_parts('`<think>`\n\n```xml\n<think>\n```')[0][0] is False
    parts = _thinking_parts('<think>分析</thi')
    assert parts == [(True, '分析', False)]
    assert _thinking_parts('分析</think>答案') == [(False, '分析', True), (False, '答案', False)]


def test_model_html_stays_escaped():
    output = without_animation(markdown_stream_html('<script>alert(1)</script>'))
    assert '<script>' not in output
    assert '&lt;script&gt;' in output


def test_animation_nodes_are_bounded_for_large_updates():
    text = '内容很多' * 3000
    output = markdown_stream_html(text)
    assert output.count('class="instinct-new"') <= 64
    assert without_animation(output).count('内容很多') == 3000
    very_long = markdown_stream_html('长内容' * 20000)
    assert 'class="instinct-new"' not in very_long


def test_live_and_completed_code_use_native_markdown():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string('''
import streamlit as st
from scripts.web_demo_utils import render_markdown_stream
render_markdown_stream(st.empty(), "### Example\\n\\n```python\\nprint('x')\\n", streaming=True)
render_markdown_stream(st.empty(), "```python\\nprint('done')\\n```", streaming=False)
''').run()
    assert not app.exception
    assert len(app.markdown) == 2
    assert app.markdown[0].value == "```python\nprint('x')"
    assert app.markdown[1].value == "```python\nprint('done')\n```"
    assert all('<span' not in item.value for item in app.markdown)
    assert any('<h3>' in item.proto.body for item in app.get('html'))


def test_loading_and_mixed_text_keep_animation_without_losing_code():
    from scripts.web_demo_utils import generation_loading_html, _code_segments
    assert 'instinct-loading' in generation_loading_html()
    assert 'role="status"' in generation_loading_html()
    text = 'Before\n\n```python\nx = 1\n```\n\nAfter'
    parts = _code_segments(text)
    assert [part[0] for part in parts] == ['text', 'code', 'text']
    assert ''.join(part[1] for part in parts) == text


def test_code_inside_think_has_native_copy_controls():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string('''
import streamlit as st
from scripts.web_demo_utils import render_markdown_stream
render_markdown_stream(st.empty(), "<think>```python\\nx = 1\\n```", streaming=True)
''').run()
    assert not app.exception
    assert app.expander[0].label == '思考中…'
    assert app.markdown[0].value == '```python\nx = 1\n```'
