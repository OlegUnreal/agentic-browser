# agentic-browser

An LLM agent that drives a real browser: reads the page (text + screenshot), decides which tool to call (click, type, scroll), observes the result, repeats — with a safety policy that most agent demos skip.

## The idea

Browser automation (Playwright) is deterministic. An LLM on top makes it *adaptive*: it can handle pages it has never seen, recover from layout changes, and decide its own next step. The hard part is not the clicking — it is **perception** (turning a messy DOM into something the model can reason about) and **safety** (stopping the agent from wandering off to malicious domains or looping forever).

This repo has both.

## Architecture

```
agentic/
├── config.py          # Settings from env / .env
├── logging_config.py  # structured JSON logs (no secrets)
├── browser.py         # Playwright wrapper; every action is a typed tool
├── vision.py          # page description via LLM (structured JSON elements)
├── agent.py           # observe → choose tool → act loop
├── guard.py           # domain allow-list + rate limit
├── llm.py             # OpenAI tool-calling brain with retries
├── __main__.py        # CLI: python -m agentic "<goal>" "<start_url>"
├── demo.py            # offline stub (no key)
└── demo_llm.py        # live run with real OpenAI
```

### The loop

```
while steps < MAX:
    observe  = browser.snapshot() + vision.describe(page)   # text + structured elements
    action   = llm.choose(goal, observe)                   # tool call: goto/click/type/scroll/done
    guard.check(action)                                     # domain + rate limit
    result   = browser.execute(action)                     # Playwright does it
    if action == done: break
```

## Why this is interesting

It combines three hard things that most "agent" demos skip:

1. **Browser automation** — real Playwright, real pages, real failures.
2. **Multimodal perception** — the model sees both the DOM text and a structured element list, so it can target by role/text rather than brittle CSS selectors.
3. **A safety policy** — domain allow-list and rate limit, so a hallucinating model can't `goto https://evil.com` or click itself into an infinite loop.

## How to run

```bash
git clone https://github.com/OlegUnreal/agentic-browser.git
cd agentic-browser

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
playwright install chromium        # downloads the browser binary

cp .env.example .env               # put OPENAI_API_KEY=sk-... in .env

# offline demo (no key, uses stub LLM)
python -m agentic.demo

# live run
python -m agentic "Find the pricing page" "https://example.com"

# tests
pytest -q
```

Windows notes:

- Activate with `.venv\Scripts\activate`.
- `playwright install chromium` downloads a bundled browser — no system Chrome/Edge required, but the first run needs network access.
- On Windows the bundled Chromium runs headless by default; pass `--headed` (or set `AGENT_HEADED=1`) to watch it.
- If `playwright install` fails behind a proxy, set `PLAYWRIGHT_DOWNLOAD_HOST` to a mirror.

## Libraries used and why

| Library | Version | Why it is here |
|---|---|---|
| `openai` | `>=1.40` | Tool-calling client. The agent emits structured actions (`goto`, `click`, `type`, `scroll`, `done`) via the function-calling API, so the model can't emit free-text garbage that the parser has to guess at. |
| `playwright` | `>=1.40` | Real browser automation. Auto-waits, role-based selectors, and a bundled Chromium mean tests don't depend on whatever Chrome the interviewer happens to have installed. |
| `python-dotenv` | `>=1.0` | Loads `.env` for the API key and the domain allow-list. |
| `pytest` | `>=8.0` | (dev) Tests for the agent loop, the safety guard, and vision parsing — with a fake browser so CI stays green without downloading Chromium. |

Why Playwright over Selenium: Playwright's selector engine is built around accessibility roles and text, which maps directly onto what an LLM can reason about. Selenium's CSS/XPath-first model forces brittle selectors that break on every layout change — exactly the failure mode this agent is meant to avoid.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | required for live mode |
| `AGENT_MODEL` | `gpt-4o-mini` | model (vision-capable recommended) |
| `AGENT_MAX_STEPS` | `10` | safety cap on loop iterations |
| `AGENT_ALLOWED_DOMAINS` | `example.com` | comma-separated allow-list |
| `AGENT_MAX_ACTIONS_PER_MIN` | `30` | rate limit |
| `AGENT_TIMEOUT` | `30` | per-LLM-call timeout (seconds) |
| `AGENT_HEADED` | `0` | `1` to show the browser window |

## Testing

```bash
pytest -q
pytest -v tests/test_agent_loop.py   # observe → act loop + termination
pytest -v tests/test_guard.py        # domain allow-list + rate limit
pytest -v tests/test_vision.py       # structured element parsing
```

Browser-dependent tests are skipped automatically when Playwright isn't installed, so `pytest -q` is green in CI without a browser.

## Design decisions (interview notes)

1. **Why structured elements instead of raw HTML?**
   Raw HTML is huge and full of noise (scripts, styles, tracking pixels). A structured list of `{role, text, selector}` fits in a prompt and lets the model target by meaning, not by brittle CSS.

2. **Why a domain allow-list?**
   An LLM can hallucinate a URL. Without a guard, one bad completion navigates to an attacker-controlled page. The allow-list is a cheap, deterministic backstop.

3. **Why a rate limit?**
   A stuck agent in a redirect loop can burn hundreds of actions per minute. The cap turns a runaway into a clean termination.

4. **Why tool-calling over free-text actions?**
   `response_format` with a tool schema forces the model to emit a valid action object. Free-text parsing ("click the blue button") is fragile and drifts between model versions.

5. **Why skip browser tests in CI?**
   Playwright needs a real browser binary. Skipping when absent keeps CI fast and green; the logic is still unit-tested with a fake browser.

## Project status

Working prototype with real LLM tool-calling, Playwright integration, safety guards, and tests. Not production — no CAPTCHA handling, no multi-tab, no persistent sessions. Strong portfolio piece for agent + browser-automation interviews.

## License

MIT.
