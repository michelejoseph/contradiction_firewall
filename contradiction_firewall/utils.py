"""
Shared utilities for the Contradiction Firewall.
"""
from __future__ import annotations

import hashlib
import uuid
from typing import Any, Dict, List, Optional


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())


def truncate(text: str, max_chars: int = 200) -> str:
    """Truncate text for display, adding ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def content_hash(text: str) -> str:
    """Short hash of text content, useful for deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def flatten_list(nested: List[List[Any]]) -> List[Any]:
    """Flatten a list of lists."""
    return [item for sublist in nested for item in sublist]


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Clamp a float value between minimum and maximum."""
    return max(minimum, min(maximum, value))


def weighted_average(values: List[float], weights: List[float]) -> float:
    """Compute weighted average of values."""
    if not values or not weights or len(values) != len(weights):
        return 0.0
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def safe_get(d: Dict, *keys: str, default: Any = None) -> Any:
    """Safely navigate nested dicts."""
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def format_claim_for_display(claim: Any) -> str:
    """Format a Claim object for human display."""
    parts = [f"'{claim.text[:80]}'"]
    if claim.source:
        parts.append(f"[{claim.source}]")
    if claim.time_scope:
        parts.append(f"(time: {claim.time_scope})")
    if claim.geo_scope:
        parts.append(f"(geo: {claim.geo_scope})")
    return " ".join(parts)


def merge_explanations(explanations: List[Optional[str]], separator: str = " | ") -> str:
    """Merge multiple explanation strings, filtering out None/empty."""
    valid = [e for e in explanations if e and e.strip()]
    return separator.join(valid) if valid else ""


class Timer:
    """Simple context manager timer."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        import time
        self._start = time.monotonic()
        return self

    def __exit__(self, *args: Any) -> None:
        import time
        self.elapsed_ms = (time.monotonic() - self._start) * 1000
