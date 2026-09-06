"""Small, side-effect-free helpers shared by the chat WebUI and its tests."""

from pathlib import Path


CONVERSATION_STATE_KEYS = (
    "messages",
    "chat_messages",
    "regenerate",
    "last_user_message",
    "regenerate_index",
)


def clear_conversation_state(state):
    """Remove conversation-only state without unloading the model or settings."""
    for key in CONVERSATION_STATE_KEYS:
        state.pop(key, None)


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
