from agentic.guard import Guard


def test_blocks_foreign_domain():
    g = Guard({"example.com"})
    assert g.allow_url("https://example.com/x")
    assert not g.allow_url("https://evil.com")


def test_rate_limit():
    g = Guard({"example.com"}, max_actions_per_min=2)
    assert g.allow_action()
    assert g.allow_action()
    assert not g.allow_action()
