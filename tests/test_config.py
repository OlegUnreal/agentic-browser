from agentic.config import Settings


def test_settings_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("AGENT_MODEL", "gpt-4o")
    monkeypatch.setenv("AGENT_ALLOWED_DOMAINS", "a.com, b.com")
    s = Settings()
    assert s.openai_api_key == "sk-x"
    assert s.model == "gpt-4o"
    assert s.allowed_domains == ("a.com", "b.com")
    assert s.has_key


def test_settings_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert Settings().has_key is False
