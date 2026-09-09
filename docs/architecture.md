# Architecture

```mermaid
graph TD
    U[User goal] --> A[Agent]
    A --> V[Vision: screenshot to text]
    V --> L[LLM decides action]
    L --> B[Browser: Playwright]
    B --> G[Guard: domain + rate limit]
    G --> B
    L --> D[Done]
```

Four layers:

| Layer | Responsibility |
|---|---|
| **Vision** | Converts page screenshots to structured observations |
| **Agent** | Chooses next action: goto, click, type, done |
| **Browser** | Executes actions via Playwright |
| **Guard** | Blocks foreign domains, enforces rate limits |
