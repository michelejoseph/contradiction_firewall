"""
Numeric and Temporal Consistency Checker.

Specializes in detecting conflicts between:
  - Numeric values with the same unit (30 days vs 14 days)
  - Temporal states (active vs deprecated, v1 vs v2)
  - Version conflicts
  - Date range conflicts

These are caught more reliably here than by NLI, which sometimes
treats different numbers as semantically neutral.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..models import (
    Claim,
    ContradictionType,
    DetectorName,
    DetectorResult,
)


# ---------------------------------------------------------------------------
# Numeric extraction
# ---------------------------------------------------------------------------

@dataclass
class NumericFact:
    """A quantity extracted from a claim."""
    value: float
    unit: str          # normalized unit (singular, lowercase)
    raw: str           # original text fragment
    context: str       # surrounding words for disambiguation
    is_upper_bound: bool = False   # "at most 30 days"
    is_lower_bound: bool = False   # "at least 30 days"


_UNIT_ALIASES: Dict[str, str] = {
    "day": "day", "days": "day",
    "week": "week", "weeks": "week",
    "month": "month", "months": "month",
    "year": "year", "years": "year",
    "hour": "hour", "hours": "hour",
    "minute": "minute", "minutes": "minute",
    "second": "second", "seconds": "second",
    "percent": "percent", "%": "percent",
    "dollar": "dollar", "dollars": "dollar", "$": "dollar", "usd": "dollar",
    "euro": "euro", "euros": "euro", "eur": "euro",
    "gbp": "gbp", "pound": "gbp", "pounds": "gbp",
    "user": "user", "users": "user",
    "account": "account", "accounts": "account",
    "request": "request", "requests": "request",
    "item": "item", "items": "item",
    "token": "token", "tokens": "token",
    "call": "call", "calls": "call",
    "mb": "mb", "gb": "gb", "tb": "tb",
    "char": "char", "chars": "char", "character": "char", "characters": "char",
}

_NUMERIC_RE = re.compile(
    r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*"
    r"(days?|weeks?|months?|years?|hours?|minutes?|seconds?|"
    r"percent|%|dollars?|\$|euros?|gbp|pounds?|usd|eur|"
    r"users?|accounts?|requests?|items?|tokens?|calls?|"
    r"mb|gb|tb|chars?|characters?)",
    re.IGNORECASE,
)

_BOUND_WORDS_UPPER = {"at most", "up to", "maximum", "max", "no more than", "within", "up to"}
_BOUND_WORDS_LOWER = {"at least", "minimum", "min", "no less than", "more than", "over"}


def _extract_numeric_facts(text: str) -> List[NumericFact]:
    facts = []
    text_lower = text.lower()
    
    for m in _NUMERIC_RE.finditer(text):
        raw_num = m.group(1).replace(",", "")
        try:
            value = float(raw_num)
        except ValueError:
            continue
        
        raw_unit = m.group(2).lower()
        unit = _UNIT_ALIASES.get(raw_unit, raw_unit)
        
        # Check for bound modifiers in the 4 words before the number
        start = max(0, m.start() - 30)
        context_before = text_lower[start:m.start()].strip()
        
        is_upper = any(b in context_before for b in _BOUND_WORDS_UPPER)
        is_lower = any(b in context_before for b in _BOUND_WORDS_LOWER)
        
        facts.append(NumericFact(
            value=value,
            unit=unit,
            raw=m.group(0),
            context=context_before[-20:],
            is_upper_bound=is_upper,
            is_lower_bound=is_lower,
        ))
    
    return facts


def _numeric_conflict(
    facts_a: List[NumericFact],
    facts_b: List[NumericFact],
) -> Optional[Tuple[float, str]]:
    """
    Compare two lists of numeric facts for conflicts.
    Returns (probability, explanation) or None if no conflict.
    """
    # Group by unit
    units_a: Dict[str, List[NumericFact]] = {}
    for f in facts_a:
        units_a.setdefault(f.unit, []).append(f)
    
    for fact_b in facts_b:
        unit = fact_b.unit
        if unit not in units_a:
            continue
        
        for fact_a in units_a[unit]:
            # Same unit, different value
            if fact_a.value == fact_b.value:
                continue
            
            # Check if bounds make them compatible
            # "up to 30 days" vs "14 days" -- the 14 days is within the 30-day window
            if fact_a.is_upper_bound and fact_b.value <= fact_a.value:
                continue
            if fact_b.is_upper_bound and fact_a.value <= fact_b.value:
                continue
            if fact_a.is_lower_bound and fact_b.value >= fact_a.value:
                continue
            if fact_b.is_lower_bound and fact_a.value >= fact_b.value:
                continue
            
            # Real conflict
            explanation = (
                f"Numeric conflict on unit '{unit}': "
                f"{fact_a.raw!r} vs {fact_b.raw!r}"
            )
            # High confidence for exact same unit with different value
            probability = 0.92
            return (probability, explanation)
    
    return None


# ---------------------------------------------------------------------------
# Temporal conflict
# ---------------------------------------------------------------------------

_TEMPORAL_ACTIVE = {"currently", "active", "available", "enabled", "supported", "live", "now"}
_TEMPORAL_INACTIVE = {
    "deprecated", "discontinued", "legacy", "removed", "disabled",
    "unavailable", "unsupported", "retired", "obsolete", "decommissioned",
    "no longer", "end of life", "eol",
}

_VERSION_RE = re.compile(r"v(\d+(?:\.\d+)*)", re.IGNORECASE)


def _temporal_conflict(text_a: str, text_b: str) -> Optional[Tuple[float, str]]:
    """Detect active/deprecated state contradictions."""
    lower_a = text_a.lower()
    lower_b = text_b.lower()
    
    # Active vs inactive
    a_active = any(w in lower_a for w in _TEMPORAL_ACTIVE)
    a_inactive = any(w in lower_a for w in _TEMPORAL_INACTIVE)
    b_active = any(w in lower_b for w in _TEMPORAL_ACTIVE)
    b_inactive = any(w in lower_b for w in _TEMPORAL_INACTIVE)
    
    if (a_active and b_inactive) or (a_inactive and b_active):
        return (0.75, f"Temporal state conflict: one claim indicates active, other indicates inactive")
    
    # Version conflict
    versions_a = _VERSION_RE.findall(text_a)
    versions_b = _VERSION_RE.findall(text_b)
    
    if versions_a and versions_b and set(versions_a) != set(versions_b):
        return (0.70, f"Version conflict: {versions_a} vs {versions_b}")
    
    return None


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class NumericTemporalDetector:
    """
    Specialized detector for numeric and temporal consistency.

    Runs fast (pure Python, no model), catches conflicts that NLI
    models sometimes miss when numbers are involved.
    """

    def check_pair(self, candidate: Claim, prior: Claim) -> DetectorResult:
        t0 = time.monotonic()
        
        facts_candidate = _extract_numeric_facts(candidate.text)
        facts_prior = _extract_numeric_facts(prior.text)
        
        result = _numeric_conflict(facts_prior, facts_candidate)
        if result:
            prob, explanation = result
            # Reduce if scopes differ
            if candidate.geo_scope and prior.geo_scope:
                if candidate.geo_scope.lower() != prior.geo_scope.lower():
                    prob *= 0.4  # Different geo -- much less likely a real conflict
                    explanation += f" (but geo scopes differ: {prior.geo_scope} vs {candidate.geo_scope})"
            
            return DetectorResult(
                detector=DetectorName.NUMERIC,
                contradiction_probability=prob,
                contradiction_type=ContradictionType.NUMERIC_CONFLICT,
                explanation=explanation,
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        
        # Check temporal
        temp_result = _temporal_conflict(prior.text, candidate.text)
        if temp_result:
            prob, explanation = temp_result
            return DetectorResult(
                detector=DetectorName.NUMERIC,
                contradiction_probability=prob,
                contradiction_type=ContradictionType.TEMPORAL_CONFLICT,
                explanation=explanation,
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        
        return DetectorResult(
            detector=DetectorName.NUMERIC,
            contradiction_probability=0.0,
            contradiction_type=None,
            explanation="No numeric or temporal conflict detected",
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    def check_against_many(
        self, candidate: Claim, priors: List[Claim]
    ) -> List[DetectorResult]:
        return [self.check_pair(candidate, prior) for prior in priors]
