# agentic-browser

[![CI](https://github.com/OlegUnreal/agentic-browser/actions/workflows/ci.yml/badge.svg)](https://github.com/OlegUnreal/agentic-browser/actions/workflows/ci.yml)

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
├── snapshot.py        # accessibility-tree page snapshot
├── vision.py          # page description via LLM (structured JSON elements)
├── agent.py           # observe → retrieve → choose tool → act loop + traces
├── guard.py           # domain allow-list + rate limit
├── llm.py             # OpenAI tool-calling brain with retries
├── offline.py         # scripted LLM for keyless demos/tests
├── train.py           # CLI: train the selector ranker from memory / synthetic data
├── perception/        # element retrieval + learning
│   ├── corpus.py      # element corpus + 19 numeric features
│   ├── embed.py       # hashed n-gram embeddings for elements
│   ├── index.py       # MMR re-ranking (relevance vs diversity)
│   ├── ranker.py      # SelectorRanker: learned logistic ranking, persisted via joblib
│   ├── memory.py      # EpisodicMemory: SQLite of states, episodes, selections
│   ├── feedback.py    # turn click outcomes into training rows
│   └── selectors.py   # selector brittleness scoring
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

It combines four hard things that most "agent" demos skip:

1. **Browser automation** — real Playwright, real pages, real failures.
2. **Multimodal perception** — the model sees both the DOM text and a structured element list, so it can target by role/text rather than brittle CSS selectors.
3. **A safety policy** — domain allow-list and rate limit, so a hallucinating model can't `goto https://evil.com` or click itself into an infinite loop.
4. **Learning from its own clicks** — a semantic retrieval layer over page elements plus a selector ranker trained on what actually worked.

## Perception and learning

A page can hold hundreds of elements; the LLM prompt fits eight. Choosing *which* eight is a retrieval problem, solved in `agentic/perception/`:

- **Semantic candidates.** Element text and labels are embedded (hashed n-grams, no network) and scored against the goal; MMR re-ranking balances relevance against diversity so the shortlist isn't five near-identical buttons.
- **Learned ranking.** `SelectorRanker` is a logistic model over 19 features per element (text match, role, viewport, DOM order, selector brittleness, prior success from memory…). Trained with `python -m agentic.train`, it beats the hand-tuned heuristic on held-out groups: **MRR 0.98 vs 0.94, top-1 96.6% vs 89.7%**. The artifact ships in `models/`; a missing file falls back to the heuristic, so the agent never hard-fails.
- **Episodic memory.** Every state, episode, and click lands in a local SQLite database (`EpisodicMemory`). Previous selections for a similar page state are recalled and folded back into ranking — the agent gets measurably better at pages it has seen before.
- **Feedback loop.** Click outcomes become training rows (`perception/feedback.py`); `python -m agentic.train` re-fits the ranker from real history or from synthetic pages when memory is empty (deterministic seeds, reproducible metrics).

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

# (re)train the selector ranker — synthetic data by default, --groups for real history
python -m agentic.train --model logreg --json

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
| `numpy` | `>=1.26` | Vector math for element embeddings and feature arrays. |
| `scikit-learn` | `>=1.4` | The learned selector ranker (logistic regression over 19 element features) and its metrics. |
| `joblib` | `>=1.3` | Persists the ranker artifact (`models/selector_ranker.joblib`) with its metadata. |
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
| `AGENT_TOP_K` | `8` | elements handed to the model per step after retrieval |
| `AGENT_MAX_SCROLLS_PER_MIN` | `12` | separate cap for scrolling (cheap, abusable) |
| `AGENT_SELECTOR_MODEL` | `models/selector_ranker.joblib` | ranker artifact path; missing file → heuristic fallback |
| `AGENT_MEMORY_DB` | — | episodic-memory SQLite path (in-memory when unset) |

## Testing

```bash
pytest -q                              # 17 passed
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

6. **Why a learned ranker instead of a bigger prompt?**
   Prompt size is the real budget: a 300-element page can't be pasted into the LLM, and a bad shortlist caps the agent no matter how good the model is. A 19-feature logistic ranker is trained from click outcomes (real or synthetic), beats the hand-written heuristic on held-out groups, and degrades gracefully — no artifact, no problem, the heuristic takes over.

## Project status

Working agent with real LLM tool-calling, Playwright integration, safety guards, semantic element retrieval, a trained selector ranker with episodic memory, and tests. Not production — no CAPTCHA handling, no multi-tab, no persistent sessions. Strong portfolio piece for agent + browser-automation interviews.

## License

MIT.

*Last updated: 2026-09-19*
