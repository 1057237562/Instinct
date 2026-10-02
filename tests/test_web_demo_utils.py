import base64
from pathlib import Path

from scripts.web_demo_utils import (
    clear_conversation_state,
    detach_model_state,
    encode_clipboard_text,
    queue_last_response_regeneration,
    resolve_model_config_path,
    user_message_html,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_resolve_model_config_prefers_config_beside_weight(tmp_path):
    weight = _touch(tmp_path / "out" / "model.pth")
    adjacent = _touch(tmp_path / "out" / "model.json")
    _touch(tmp_path / "checkpoints" / "model.json")

    assert resolve_model_config_path(str(weight), str(tmp_path)) == str(adjacent)


def test_resolve_model_config_finds_checkpoint_config_for_out_weight(tmp_path):
    weight = _touch(tmp_path / "out" / "model.pth")
    checkpoint_config = _touch(tmp_path / "checkpoints" / "model.json")
    _touch(tmp_path / "trainer" / "config_pretrain.json")

    assert resolve_model_config_path(str(weight), str(tmp_path)) == str(checkpoint_config)


def test_resolve_model_config_uses_legacy_fallback_last(tmp_path):
    weight = _touch(tmp_path / "out" / "model.pth")
    fallback = _touch(tmp_path / "trainer" / "config_pretrain.json")

    assert resolve_model_config_path(str(weight), str(tmp_path)) == str(fallback)


def test_clear_conversation_state_keeps_model_and_settings():
    state = {
        "messages": [{"role": "user", "content": "old"}],
        "chat_messages": [{"role": "user", "content": "old"}],
        "regenerate": True,
        "last_user_message": "old",
        "regenerate_index": 0,
        "model": object(),
        "temperature": 0.9,
    }

    clear_conversation_state(state)
    clear_conversation_state(state)  # Clearing an already clean chat is safe.

    assert state.keys() == {"model", "temperature"}


def test_detach_model_state_returns_resources_and_clears_model_chat_state():
    model = object()
    tokenizer = object()
    state = {
        "model": model,
        "tokenizer": tokenizer,
        "model_loaded": True,
        "loaded_weight_path": "out/model.pth",
        "messages": [{"role": "user", "content": "old"}],
        "chat_messages": [{"role": "user", "content": "old"}],
        "temperature": 0.9,
    }

    detached_model, detached_tokenizer = detach_model_state(state)

    assert detached_model is model
    assert detached_tokenizer is tokenizer
    assert state == {"temperature": 0.9}


def test_encode_clipboard_text_round_trips_unicode_and_code():
    answer = "中文回答\n```python\nprint('hello')\n```\n</script>"

    encoded = encode_clipboard_text(answer)

    assert base64.b64decode(encoded).decode("utf-8") == answer
    assert "</script>" not in encoded


def test_queue_last_response_regeneration_rewinds_tool_chain():
    state = {
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "latest question"},
            {"role": "assistant", "content": "latest answer"},
        ],
        "chat_messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "latest question"},
            {"role": "assistant", "content": "tool call"},
            {"role": "tool", "content": "old tool result"},
            {"role": "assistant", "content": "latest answer"},
        ],
        "model": object(),
    }

    assert queue_last_response_regeneration(state) is True
    assert [item["content"] for item in state["messages"]] == [
        "first", "first answer",
    ]
    assert [item["content"] for item in state["chat_messages"]] == [
        "system", "first", "first answer",
    ]
    assert state["last_user_message"] == "latest question"
    assert state["regenerate"] is True
    assert "model" in state


def test_queue_last_response_regeneration_requires_completed_answer():
    state = {
        "messages": [{"role": "user", "content": "not answered yet"}],
        "chat_messages": [],
    }

    assert queue_last_response_regeneration(state) is False
    assert state["messages"] == [
        {"role": "user", "content": "not answered yet"},
    ]


def test_user_message_html_closes_unclosed_code_fence():
    html_out = user_message_html("看看这段代码:\n\n```python\nprint(1)")

    # The wrapper's closing tags must stay outside the parsed content.
    assert html_out.count("</div>") == 2
    assert html_out.endswith("</div></div>")
    assert '<pre><code class="language-python">print(1)</code></pre>' in html_out


def test_user_message_html_keeps_blank_lines_inside_code_verbatim():
    # A blank line inside a fence would end st.markdown's raw-HTML block and
    # mangle the rest, so the bubble ships parsed HTML untouched via st.html.
    html_out = user_message_html("```\na\n\nb\n```")

    assert "<pre><code>a\n\nb\n</code></pre>" in html_out


def test_user_message_html_escapes_pasted_markup():
    html_out = user_message_html("<b>bold</b> <script>alert(1)</script>")

    assert "&lt;b&gt;bold&lt;/b&gt;" in html_out
    assert "&lt;script&gt;" in html_out
    assert "<script>" not in html_out


def test_user_message_html_wraps_plain_text_in_bubble():
    html_out = user_message_html("你好")

    assert '<div class="instinct-user-row"><div class="instinct-user-bubble"><p>你好</p>' in html_out
    assert html_out.endswith("</div></div>")
