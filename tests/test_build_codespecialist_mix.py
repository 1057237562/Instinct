from scripts.build_codespecialist_mix import MIX_PLAN, render_messages, selected_by_hash


def test_code_related_share_is_sixty_percent():
    code = sum(
        MIX_PLAN[name]
        for name in (
            "open_repository_code_and_docs",
            "competitive_problem_reasoning",
            "verified_competitive_submissions",
            "code_instruction",
            "text_to_sql",
            "exercism_software_tasks",
        )
    )
    assert code / sum(MIX_PLAN.values()) == 0.60


def test_render_messages_preserves_all_content_and_chat_boundaries():
    rendered = render_messages(
        [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "problem"},
            {"role": "assistant", "content": "solution"},
        ]
    )
    assert rendered.count("<|im_start|>") == 3
    assert rendered.count("<|im_end|>") == 3
    assert all(value in rendered for value in ("rules", "problem", "solution"))


def test_hash_selection_is_deterministic():
    values = [selected_by_hash(f"row-{i}".encode(), 0.5, 42) for i in range(100)]
    assert values == [selected_by_hash(f"row-{i}".encode(), 0.5, 42) for i in range(100)]
    assert any(values) and not all(values)
