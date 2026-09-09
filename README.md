# agentic-browser

An LLM agent that drives a real browser: reads the page (text + screenshot), decides which tool to call (click, type, scroll), observes the result, repeats.

## Architecture

- `browser/` — Playwright wrapper. Every action is a tool the agent can invoke. Sandboxed: runs in a throwaway context, records a trace.
- `vision/` — takes a screenshot, sends it to a vision model, returns a structured description of clickable elements.
- `agent/` — the loop: observe → choose tool → act → observe. Termination on goal reached or max steps.
- `guard/` — policy layer: blocks navigation to disallowed domains, rate-limits clicks.

## Why this is interesting

It combines three hard things: browser automation, multimodal perception, and a safe action policy. Most "agent" demos skip the safety layer — this one has it.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
python -m agentic.demo
```
