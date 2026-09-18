"""Agent loop: observe → retrieve → choose tool → act → record.

Three things happen between "look at the page" and "ask the model", and they
are the difference between a demo and a system:

1. **Retrieval.** The snapshot's elements are embedded and the top-k most
   relevant to the goal *plus the last observation* are selected with MMR, so
   the prompt carries eight distinct widgets instead of forty near-copies.
2. **Episodic memory.** The state gets a structural signature. If we have been
   here before, the planner is told what was already tried, and an action that
   memory knows to be dead end is refused and replaced.
3. **Learned ranking.** Candidate selectors for the retrieved elements are
   scored by the trained selector ranker (heuristic when no model file exists)
   before the shortlist reaches the model.

`run` keeps its original signature and still returns `list[Step]`; `run_traced`
returns the full `Run` object when you want the memory stats and guard denials.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .browser import BrowserDriver, PageState, SCROLL_STEP
from .config import SETTINGS
from .guard import KNOWN_TOOLS, Guard
from .logging_config import get_logger
from .perception.index import DEFAULT_LAMBDA_MMR, ElementIndex, Selection
from .perception.memory import EpisodicMemory, LoopWarning, StateSignature, action_key
from .perception.ranker import SelectorRanker
from .perception.selectors import (
    SelectorCandidate,
    SelectionContext,
    build_candidates,
)
from .snapshot import clean_label
from .vision import describe as describe_page

log = get_logger(__name__)


@dataclass
class Step:
    n: int
    action: str
    result: str
    #: Why this step looks the way it does: 'ok', 'loop-break: ...', the guard's
    #: refusal, 'rate-limited'. Surfaced in the CLI trace and asserted in tests.
    reason: str = ""
    #: Structural signature of the state this step started from.
    state_hash: str = ""
    #: Selectors retrieval actually put in front of the model.
    offered: tuple[str, ...] = ()
    #: Selector/URL the action was aimed at (empty for `done`). Lets a reader —
    #: or the offline policy — see which targets are already spent.
    target: str = ""


@dataclass
class Run:
    """Everything one `run_traced` produced."""

    steps: list[Step] = field(default_factory=list)
    goal: str = ""
    memory_stats: dict[str, Any] = field(default_factory=dict)
    denials: dict[str, int] = field(default_factory=dict)
    stopped_reason: str = ""

    @property
    def trace(self) -> list[str]:
        return [
            f"{s.n}. {s.action}{(' ' + s.target) if s.target else ''} -> {s.result} [{s.reason}]"
            for s in self.steps
        ]

    def __len__(self) -> int:
        return len(self.steps)


def _execute(driver: BrowserDriver, decision: dict, guard: Guard) -> PageState:
    """Dispatch one validated decision onto the driver."""
    name = str(decision.get("name", "")).lower()
    args = decision.get("args") or {}
    if name == "goto":
        url = str(args.get("url", ""))
        if guard.allow_url(url):
            return driver.goto(url)
        return _blocked(url)
    if name == "click":
        return driver.click(str(args.get("selector", "")))
    if name == "type":
        return driver.type(str(args.get("selector", "")), str(args.get("text", "")))
    if name == "scroll":
        return driver.scroll(
            str(args.get("selector", "") or ""),
            int(args.get("amount", SCROLL_STEP) or SCROLL_STEP),
        )
    if name == "done":
        return driver.snapshot()
    return driver.snapshot()


def _blocked(url: str) -> PageState:
    return PageState(url=url, title="blocked", text="domain not allowed", screenshot_b64="")


def _ask(
    llm: Callable,
    goal: str,
    url: str,
    elements: Sequence[dict],
    history: Sequence[Step],
    context: dict,
) -> dict:
    """Call the planner, passing the memory context only if it accepts it."""
    try:
        return llm(goal, url, list(elements), list(history), context=context)
    except TypeError as exc:
        if "context" not in str(exc):
            raise
        return llm(goal, url, list(elements), list(history))


def _perceive(state: PageState, goal: str, observation: str, llm: Callable | None, top_k: int, lambda_mm: float) -> Selection:
    """Embed + retrieve the elements worth showing the model.

    Falls back to the LLM-backed vision reader only when the driver produced no
    structured elements, so the decision LLM is never invoked as a describer.
    """
    elements = list(state.elements or [])
    if not elements and llm is not None and state.text:
        elements = describe_page(state, llm)
    index = ElementIndex(elements)
    return index.retrieve(goal, k=top_k, lambda_mm=lambda_mm, prior=observation)


def _shortlist(
    selection: Selection,
    goal: str,
    observation: str,
    action: str,
    ranker: SelectorRanker,
    memory: EpisodicMemory,
    state: PageState,
) -> tuple[list[SelectorCandidate], SelectionContext]:
    """Ranked candidate selectors for this observation."""
    history_targets = [e.target for e in memory.recall_for_goal(goal, limit=6) if e.ok and e.target]
    candidates = build_candidates(
        goal=goal,
        elements=[hit.element for hit in selection.hits],
        observation=observation,
        action=action,
        history_targets=history_targets,
    )
    context = SelectionContext(
        goal=goal,
        observation=observation,
        action=action,
        n_candidates=len(candidates),
        n_elements=max(len(selection.hits), 1),
        viewport_height=getattr(state, "viewport_height", 900) or 900,
    )
    prior: dict[str, tuple[int, int]] = {}
    for candidate in candidates:
        successes, attempts = memory.prior_success(action, candidate.selector)
        prior[candidate.key()] = (successes, attempts)
    context.prior_success = prior
    return ranker.rank(candidates, context), context


def _force_alternative(
    warning: LoopWarning,
    decision: dict,
    ranked: Sequence[SelectorCandidate],
    signature: StateSignature,
) -> dict | None:
    """Pick something the memory has not already burned at this state.

    Escalation is deterministic: next-best untried selector for the same verb,
    then a scroll to reveal new content, then give up. Refusing to improvise is
    worse than stopping — an agent that keeps clicking dead buttons spends the
    user's budget on nothing.
    """
    tried = signature.tried | {warning.action_key_name}
    name = str(decision.get("name", "")).lower()
    args = dict(decision.get("args") or {})
    if name in {"click", "type", "scroll"}:
        for candidate in ranked:
            key = action_key(name, candidate.selector, args)
            if key not in tried:
                replacement = dict(args)
                replacement["selector"] = candidate.selector
                return {"name": name, "args": replacement}
        if name != "scroll":
            # Nothing left to target: change what the viewport exposes so the
            # next retrieval has new material to work with.
            amount = int(args.get("amount", SCROLL_STEP) or SCROLL_STEP)
            return {"name": "scroll", "args": {"selector": "", "amount": amount * 2}}
    return None


def run_traced(
    goal: str,
    start_url: str,
    llm,
    max_steps: int = 10,
    allowed_domains: set[str] | None = None,
    driver: Any | None = None,
    memory: EpisodicMemory | None = None,
    ranker: SelectorRanker | None = None,
    top_k: int | None = None,
    lambda_mm: float = DEFAULT_LAMBDA_MMR,
    vision_llm: Callable | None = None,
    feedback: Any | None = None,
    owns_driver: bool | None = None,
    headed: bool | None = None,
) -> Run:
    """Run the agent and return the full trace plus memory/denial stats.

    Lifecycle: the loop opens and closes a browser only when it built one. A
    caller-supplied `driver` stays the caller's, which is what lets a test reuse
    one recording double across phases. `headed=None` defers to `AGENT_HEADED`.
    """
    owns = driver is None if owns_driver is None else bool(owns_driver)
    if driver is None:
        driver = BrowserDriver(headed=headed)
    if owns:
        driver.start()
    memory = memory if memory is not None else EpisodicMemory(SETTINGS.memory_db or ":memory:")
    ranker = ranker if ranker is not None else SelectorRanker.load_or_default(SETTINGS.selector_model_path)
    guard = Guard(
        allowed_domains or set(SETTINGS.allowed_domains),
        max_actions_per_min=SETTINGS.max_actions_per_min,
        max_scrolls_per_min=SETTINGS.max_scrolls_per_min,
    )
    top_k = int(top_k or SETTINGS.top_k)
    result = Run(goal=goal)
    observation = "starting observation: navigated to the start url"
    previous_hash = ""
    try:
        if not guard.allow_url(start_url):
            result.stopped_reason = "start-url-blocked"
            result.steps.append(Step(0, "goto", "blocked", reason="domain not allowed", offered=()))
            return result
        state = driver.goto(start_url)

        for n in range(1, max_steps + 1):
            signature = memory.observe(state.structure_hash(), state.url, state.elements, goal, title=state.title)
            selection = _perceive(state, goal, observation, vision_llm or (llm if _is_vision_callable(llm) else None), top_k, lambda_mm)
            ranked, ranking_context = _shortlist(selection, goal, observation, "click", ranker, memory, state)
            memory.note_selection(signature.state_hash, goal, selection.as_dicts())

            context = {
                "state_hash": signature.state_hash[:12],
                "visits": signature.visits,
                "memory_hint": clean_label(signature.hint(), 300),
                "ranked_selectors": [c.as_dict() for c in ranked[:top_k]],
                "guard": guard.remaining(),
                "embedder": selection.embedder,
            }
            try:
                decision = _ask(llm, goal, state.url, selection.as_dicts(), result.steps, context)
            except Exception as exc:  # noqa: BLE001 - the loop must survive the planner
                result.steps.append(Step(n, "error", str(exc), reason="llm-failure", state_hash=signature.state_hash))
                observation = f"last observation: the planner raised {type(exc).__name__}; pick a different approach"
                continue
            if not isinstance(decision, dict):
                decision = {"name": str(decision), "args": {}}
            name = str(decision.get("name", "")).lower()
            args = decision.get("args") or {}
            selector = str(args.get("selector", "") or "")

            if name == "done":
                memory.record_action(signature.state_hash, goal, n, "done", outcome="ok", detail="goal reached")
                result.steps.append(Step(n, "DONE", "goal reached", reason="done", state_hash=signature.state_hash))
                result.stopped_reason = "done"
                break

            if not guard.allow_tool(name):
                memory.record_action(signature.state_hash, goal, n, name, selector, args, outcome="blocked", detail="unknown tool")
                refused = sorted(KNOWN_TOOLS)
                result.steps.append(
                    Step(
                        n,
                        name or "action",
                        "refused",
                        reason=f"unknown-tool: {name!r} is not one of {refused}",
                        state_hash=signature.state_hash,
                        target=selector,
                    )
                )
                observation = f"last observation: {name!r} is not a tool; choose one of {refused}"
                continue

            warning = memory.detect_loop(signature, name, selector, args)
            reason = "ok"
            if warning is not None:
                forced = _force_alternative(warning, decision, ranked, signature)
                if forced is None:
                    memory.record_action(signature.state_hash, goal, n, name, selector, args, outcome="no-effect", detail=warning.reason)
                    result.steps.append(
                        Step(
                            n,
                            name or "action",
                            "stopped",
                            reason=f"loop-break: {warning.reason}",
                            state_hash=signature.state_hash,
                            target=selector,
                        )
                    )
                    result.stopped_reason = f"loop:{warning.kind}"
                    break
                reason = f"loop-break: {warning.reason}; forced {action_key(forced['name'], str((forced.get('args') or {}).get('selector', '')))}"
                decision = forced
                name = str(decision["name"]).lower()
                args = decision.get("args") or {}
                selector = str(args.get("selector", "") or "")

            if not guard.allow_action(name):
                memory.record_action(signature.state_hash, goal, n, name, selector, args, outcome="rate-limited", detail="action budget exhausted")
                result.steps.append(
                    Step(n, name, "rate-limited", reason="rate-limited", state_hash=signature.state_hash, target=selector)
                )
                observation = "last observation: rate limited, no action taken"
                continue

            try:
                state = _execute(driver, decision, guard)
                execution = "ok"
                detail = state.url
                if state.title == "blocked":
                    execution, reason = "blocked", f"guard refused: {state.text}"
            except Exception as exc:  # noqa: BLE001 - a dead selector must not kill the run
                state = PageState(state.url, state.title, f"action failed: {exc}", state.screenshot_b64, list(state.elements))
                execution, detail = "failed", f"{type(exc).__name__}: {exc}"

            current_hash = state.structure_hash()
            progressed = execution == "ok" and current_hash != previous_hash
            memory_outcome = execution if execution != "ok" else ("ok" if progressed else "no-effect")
            memory.record_action(
                signature.state_hash, goal, n, name, selector, args,
                outcome=memory_outcome, detail=clean_label(detail or reason, 200),
            )
            _log_feedback(feedback, ranked, ranking_context, signature.state_hash, goal, n, name, selector, execution)
            previous_hash = current_hash
            observation = (
                f"last observation: {name} on {selector or '-'} -> {memory_outcome}"
                + ("" if memory_outcome == "ok" else f" ({clean_label(detail, 120)})")
            )
            result.steps.append(
                Step(
                    n=n,
                    action=name,
                    result=state.url,
                    reason=reason if reason != "ok" else ("executed" if progressed else "no-effect"),
                    state_hash=signature.state_hash,
                    offered=tuple(h.selector for h in selection.hits[:3]),
                    target=selector or str(args.get("url", "")),
                )
            )
        else:
            result.stopped_reason = "max-steps"
    finally:
        if owns:
            try:
                driver.close()
            except Exception:  # noqa: BLE001 - closing a browser must not mask the trace
                log.warning("driver_close_failed")
        result.memory_stats = memory.stats()
        result.denials = dict(guard.denials)
    return result


def _is_vision_callable(llm) -> bool:
    """True for `describe(state) -> raw` style callables (the vision reader)."""
    if llm is None or not callable(llm):
        return False
    try:
        parameters = [p for p in inspect.signature(llm).parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    except (TypeError, ValueError):
        return False
    required = [p for p in parameters if p.default is p.empty]
    return len(required) == 1


def _log_feedback(store, ranked, context, state_hash, goal, step, action, selector, execution) -> None:
    """Append the offered shortlist plus its outcome to the training log.

    Deliberately a different label from episodic memory: feedback scores the
    *selector* (did the driver resolve it), memory scores the *strategy* (did
    the page change). A click on a correct but already-current element executes
    fine and makes no progress; conflating those two would train the ranker to
    distrust selectors that are actually healthy. Best-effort: never fatal.
    """
    if store is None or not ranked:
        return
    try:
        store.record_offer(ranked, context, state_hash, step, chosen_selector=selector)
        store.record_outcome(state_hash, goal, step, action, selector, "ok" if execution == "ok" else "failed")
    except Exception:  # noqa: BLE001 - logging the run must never break the run
        log.warning("feedback_log_failed")


def run(
    goal: str,
    start_url: str,
    llm,
    max_steps: int = 10,
    allowed_domains: set[str] | None = None,
    driver: Any | None = None,
    memory: EpisodicMemory | None = None,
    ranker: SelectorRanker | None = None,
    top_k: int | None = None,
    lambda_mm: float = DEFAULT_LAMBDA_MMR,
    vision_llm: Callable | None = None,
    feedback: Any | None = None,
    headed: bool | None = None,
) -> list[Step]:
    """Backwards-compatible entry point: the same loop, returning `list[Step]`."""
    return run_traced(
        goal,
        start_url,
        llm,
        max_steps=max_steps,
        allowed_domains=allowed_domains,
        driver=driver,
        memory=memory,
        ranker=ranker,
        top_k=top_k,
        lambda_mm=lambda_mm,
        vision_llm=vision_llm,
        feedback=feedback,
        headed=headed,
    ).steps


__all__ = ["Run", "Step", "run", "run_traced", "_execute"]
