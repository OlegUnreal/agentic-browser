# Changelog

## 0.3.0 — 2026-09-18

- Semantic element retrieval: hashed n-gram embeddings over element text/labels + MMR re-ranking pick the `AGENT_TOP_K` candidates handed to the LLM each step.
- Learned selector ranking: `SelectorRanker` (logistic regression, 19 features incl. selector brittleness and prior success) trained via `python -m agentic.train` on synthetic pages or real click history. Held-out: MRR 0.98 vs 0.94 heuristic, top-1 96.6% vs 89.7%. Artifact ships in `models/`; missing artifact falls back to the heuristic.
- Episodic memory: SQLite-backed states/episodes/selections; past selections for similar page states are recalled and folded back into ranking.
- Feedback pipeline: click outcomes become training rows; `agentic.train` supports `--model logreg|ridge|gbm`, `--dry-run`, `--json`.
- Accessibility-tree snapshots (`agentic/snapshot.py`) and a scripted offline LLM (`agentic/offline.py`) so demos and tests run without an API key.
- Traced agent loop (`run_traced`) with per-step observations and element provenance (`origin_dom/attr/history/text`).
- Manifest: `numpy`, `scikit-learn`, `joblib` added to `requirements.txt` / `pyproject.toml`.
