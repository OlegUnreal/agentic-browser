"""Snapshot normalisation: the shared vocabulary between driver and perception.

Kept deliberately dependency-free (stdlib only) so that `browser.py` can import
it without pulling in numpy/scikit-learn on machines that only want the driver,
and so that the perception layer has no import cycle back into the browser.

An *element* is a plain dict with a normalised shape::

    {"selector": str, "label": str, "role": str, "confidence": float}

plus optional extras contributed by the driver: `tag`, `attrs`, `rect`,
`visible`, `in_viewport`, `index`.  Everything downstream (embedding, MMR
retrieval, selector ranking, memory) reads these keys and tolerates absence.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable

# Query parameters that never change the *meaning* of a page.
_TRACKING_PARAMS = re.compile(r"^(utm_|ga_|gclid|fbclid|ref$|source$)", re.I)
_WHITESPACE = re.compile(r"\s+")

VALID_ROLES = frozenset(
    {
        "button", "link", "input", "textbox", "searchbox", "combobox", "checkbox",
        "radio", "menuitem", "tab", "option", "slider", "switch", "submit",
        "file", "range", "listbox", "treeitem", "generic", "unknown",
    }
)


def normalize_url(url: str) -> str:
    """Lower-case host, drop credentials/fragments/tracking params, keep the rest.

    Stable across the churn that would otherwise make every snapshot look like a
    brand-new state to the episodic memory (session ids in fragments, `utm_*`).
    """
    raw = str(url or "").strip()
    if not raw:
        return ""
    scheme, sep, rest = raw.partition("://")
    if not sep:
        scheme, rest = "", raw
    authority, _, remainder = rest.partition("/")
    authority = authority.lower().rsplit("@", 1)[-1]  # drop credentials
    host, port_sep, port = authority.partition(":")
    path, _, raw_query = remainder.partition("?")
    path = path.split("#", 1)[0]
    path = re.sub(r"/{2,}", "/", path).rstrip("/") or "/"
    kept = sorted(
        pair
        for pair in (p for p in raw_query.split("#", 1)[0].split("&") if p)
        if not _TRACKING_PARAMS.match(pair.split("=", 1)[0])
    )
    body = f"{host}{port_sep}{port}{path}"
    if kept:
        body += "?" + "&".join(kept)
    return f"{scheme}://{body}" if scheme else body


def normalize_text(text: str) -> str:
    return _WHITESPACE.sub(" ", str(text or "")).strip()


def clean_label(text: str, limit: int = 160) -> str:
    text = normalize_text(text)
    return text[:limit]


def element_text(element: dict) -> str:
    """The string we embed for an element: label plus its semantic attributes.

    Attribute values (placeholder, aria-label, href anchor text) carry most of
    the intent on real pages, so they belong in the embedding text.
    """
    parts: list[str] = [str(element.get("label") or ""), str(element.get("role") or "")]
    attrs = element.get("attrs") or {}
    if isinstance(attrs, dict):
        for key in ("placeholder", "aria-label", "name", "title", "alt", "value", "type"):
            value = attrs.get(key)
            if value:
                parts.append(str(value))
        href = attrs.get("href")
        if href:
            parts.append(str(href).rsplit("/", 1)[-1].replace(".html", "").replace("-", " "))
    return normalize_text(" ".join(p for p in parts if p))


def selector_shape(selector: str) -> str:
    """A selector generalised to a stable key for feedback lookup.

    `#login > input:nth-child(3)` -> `#id > input:nth-child(n)`. Used as the
    join key for prior-success statistics so that sibling-count churn does not
    create a new identity for the same widget.
    """
    s = normalize_text(selector)
    s = re.sub(r":nth-(child|of-type)\((\d+)\)", r":nth-\2", s)
    s = re.sub(r"#[\w-]+", "#id", s)
    s = re.sub(r"\.[\w-]+", ".cls", s)
    s = re.sub(r'="[^"]*"', '="v"', s)
    s = s.lower()
    return s[:200]


def coerce_elements(raw: Iterable[Any]) -> list[dict]:
    """Normalise heterogeneous element payloads into the shared shape."""
    out: list[dict] = []
    for i, item in enumerate(raw or []):
        if isinstance(item, str):
            out.append({"selector": item, "label": item, "role": "unknown", "confidence": 0.0, "index": i})
            continue
        if not isinstance(item, dict):
            continue
        selector = normalize_text(str(item.get("selector") or item.get("css") or ""))
        label = clean_label(item.get("label") or item.get("text") or item.get("name") or "")
        role = str(item.get("role") or item.get("type") or "").strip().lower() or "unknown"
        if role not in VALID_ROLES:
            role = "unknown"
        element = {
            "selector": selector or f"auto:{i}",
            "label": label,
            "role": role,
            "confidence": float(item.get("confidence", 0.0) or 0.0),
            "index": int(item.get("index", i) or i),
        }
        for key in ("tag", "attrs", "rect", "visible", "in_viewport"):
            if key in item:
                element[key] = item[key]
        out.append(element)
    return out


def structure_hash(url: str, elements: list[dict], title: str = "") -> str:
    """Content hash of the *interactive structure* of a page state.

    Deliberately excludes free body text so that timestamps, A/B jitter and
    rotating banners do not defeat loop detection: two states with the same
    URL and the same ordered role/selector shape are the same state.
    """
    shape = [
        (
            str(e.get("role", "unknown")),
            selector_shape(str(e.get("selector", ""))),
            len(str(e.get("label", ""))) // 12,
        )
        for e in (elements or [])
    ]
    payload = json.dumps(
        {"url": normalize_url(url), "title": normalize_text(title)[:80], "elements": shape},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def parse_dom_json(raw: str | None, limit: int = 400) -> list[dict]:
    """Turn the driver's accessibility/DOM extraction script output into elements.

    Accepts either a JSON list or a `{"elements": [...]}` envelope. Malformed
    payloads degrade to an empty list rather than raising, because the agent
    loop must survive a partial page.
    """
    if not raw or not isinstance(raw, str):
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = data.get("elements") or data.get("nodes") or []
    if not isinstance(data, list):
        return []
    normalised: list[dict] = []
    for node in data[:limit]:
        if not isinstance(node, dict):
            continue
        rect = node.get("rect") or {}
        attrs = {k: v for k, v in node.items() if k in ("id", "name", "type", "placeholder", "aria-label", "title", "alt", "href", "role", "value") and v not in (None, "")}
        width = float(rect.get("width", 0) or 0)
        height = float(rect.get("height", 0) or 0)
        normalised.append(
            {
                "selector": normalize_text(str(node.get("selector") or "")),
                "label": clean_label(node.get("text") or node.get("aria-label") or node.get("placeholder") or ""),
                "role": str(node.get("role") or attrs.get("type") or node.get("tag") or "unknown").lower(),
                "tag": str(node.get("tag") or "").lower(),
                "attrs": attrs,
                "rect": {
                    "x": float(rect.get("x", 0) or 0),
                    "y": float(rect.get("y", 0) or 0),
                    "width": width,
                    "height": height,
                },
                "visible": bool(node.get("visible", width > 0 and height > 0)),
                "in_viewport": bool(node.get("inViewport", False)),
                "confidence": 0.0,
            }
        )
    return coerce_elements(normalised)


__all__ = [
    "clean_label",
    "coerce_elements",
    "element_text",
    "normalize_text",
    "normalize_url",
    "parse_dom_json",
    "selector_shape",
    "structure_hash",
    "VALID_ROLES",
]
