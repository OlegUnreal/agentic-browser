# agentic-browser

An LLM agent that drives a real browser: reads the page (text + screenshot), decides which tool to call (click, type, scroll), observes the result, repeats.

## Architecture

```
config.py        -> Settings from env / .env
logging_config.py-> structured JSON logs (no secrets)
browser.py       -> Playwright wrapper, every action is a tool
vision.py        -> page description via LLM (JSON elements)
agent.py         -> observe -> choose tool -> act loop
guard.py         -> domain allow-list + rate limit
llm.py           -> OpenAI tool-calling brain with retries
__main__.py      -> CLI: python -m agentic "<goal>" "<start_url>"
```

## Why this is interesting

It combines three hard things: browser automation, multimodal perception, and a safe action policy. Most "agent" demos skip the safety layer — this one has it.

## Run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
playwright install chromium
cp .env.example .env   # put your OPENAI_API_KEY in .env

# offline demo (no key, uses stub LLM)
python -m agentic.demo

# live run
python -m agentic "Find the pricing page" "https://example.com"

# tests
pytest -q
```

## Config

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | required for live mode |
| `AGENT_MODEL` | `gpt-4o-mini` | model name |
| `AGENT_MAX_STEPS` | `10` | safety cap on loop iterations |
| `AGENT_ALLOWED_DOMAINS` | `example.com` | comma-separated allow-list |
| `AGENT_MAX_ACTIONS_PER_MIN` | `30` | rate limit |
