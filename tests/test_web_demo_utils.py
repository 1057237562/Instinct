from pathlib import Path

from scripts.web_demo_utils import (
    clear_conversation_state,
    queue_last_response_regeneration,
    resolve_model_config_path,
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
