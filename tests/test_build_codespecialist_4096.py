from dataset.scripts.build_codespecialist_4096 import CODE_COMPONENTS, MAX_TOKENS, MIX_PLAN


def test_token_plan_is_1_6b_and_sixty_percent_code():
    total = sum(MIX_PLAN.values())
    assert total == 1_600_000_000
    assert sum(MIX_PLAN[name] for name in CODE_COMPONENTS) / total == 0.60


def test_cap_includes_bos_and_eos():
    assert MAX_TOKENS == 4096
