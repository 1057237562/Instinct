"""Tests for instruction-dataset normalization."""

from scripts.prepare_sft_data import normalize_row


def test_normalize_codealpaca_prompt_completion():
    result = normalize_row(
        {"prompt": "Write Python", "completion": "print('ok')"},
        "prompt_completion",
    )

    assert result == {
        "conversations": [
            {"role": "user", "content": "Write Python"},
            {"role": "assistant", "content": "print('ok')"},
        ]
    }


def test_normalize_messages_preserves_system_and_multiple_turns():
    messages = [
        {"role": "system", "content": "Follow exactly"},
        {"role": "user", "content": "One word"},
        {"role": "assistant", "content": "Done"},
    ]

    assert normalize_row({"messages": messages}, "messages") == {
        "conversations": messages
    }

