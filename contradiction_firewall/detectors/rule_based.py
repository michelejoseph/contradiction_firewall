"""
Rule-Based Contradiction Detector.

Performs fast, symbolic checks before involving NLI or LLM judges.
Catches the clearest and most expensive violations with zero API cost.

Checks performed:
  1. Hard constraint violations -- claim contradicts a ledger entry
  2. Numeric conflicts -- different numbers for the same quantity
  3. Direct negation -- "X is allowed" vs "X is not allowed"
  4. Qualifier conflicts -- "only" / "never" / "always" violations
"""
from __future__ import annotations

import re
import time
from typing import List, Optional, Tuple

from ..models import (
    Claim,
    ContradictionType,
    DetectorName,
    DetectorResult,
)


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_numbers(text: str) -> List[Tuple[float, str]]:
    """Extract all (value, unit) pairs from text."""
    pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*(days?|weeks?|months?|years?|hours?|percent|%|dollars?|\$)",
        re.IGNORECASE,
    )
    results = []
    for m in pattern.finditer(text):
        try:
            val = float(m.group(1))
            unit = m.group(2).lower().rstrip("s")  # normalize plural
            results.append((val, unit))
        except ValueError:
            pass
    return results


def _check_negation_conflict(claim_a: Claim, claim_b: Claim) -> Optional[float]:
    """
    Returns conflict probability if claims appear to be direct negations.

    Simple heuristic: same subject+object but one is negated and the other is not.
    """
    if claim_a.subject and claim_b.subject:
        subj_match = _normalize(claim_a.subject) == _normalize(claim_b.subject)
        if not subj_match:
            return None
    
    # Check negation flip
    if claim_a.is_negated != claim_b.is_negated:
        # Same predicate or object -- higher confidence
        if claim_a.predicate and claim_b.predicate:
            if _normalize(claim_a.predicate) == _normalize(claim_b.predicate):
                return 0.85
        # Fall back to text overlap
        words_a = set(_normalize(claim_a.text).split())
        words_b = set(_normalize(claim_b.text).split())
        negation_words = {"not", "never", "no", "cannot"}
        # Remove negation words to compare core content
        core_a = words_a - negation_words
        core_b = words_b - negation_words
        if core_a and core_b:
            overlap = len(core_a & core_b) / max(len(core_a), len(core_b))
            if overlap > 0.5:
                return 0.75
    return None


def _check_numeric_conflict(claim_a: Claim, claim_b: Claim) -> Optional[float]:
    """
    Returns conflict probability if claims state different numbers for the same unit.
    """
    nums_a = _extract_numbers(claim_a.text)
    nums_b = _extract_numbers(claim_b.text)
    
    if not nums_a or not nums_b:
        return None
    
    # Build unit -> value map for each claim
    units_a = {unit: val for val, unit in nums_a}
    units_b = {unit: val for val, unit in nums_b}
    
    for unit in units_a:
        if unit in units_b and units_a[unit] != units_b[unit]:
            # Same unit, different value -- potential conflict
            # But check scope first: US vs EU rules can legitimately differ
            if _scopes_overlap(claim_a, claim_b):
                return 0.90
    return None


def _scopes_overlap(claim_a: Claim, claim_b: Claim) -> bool:
    """
    Returns True if the two claims could apply to the same context.
    If scopes are mutually exclusive (e.g. US vs EU), they do not overlap.
    """
    # Geo scope check
    if claim_a.geo_scope and claim_b.geo_scope:
        if _normalize(claim_a.geo_scope) != _normalize(claim_b.geo_scope):
            # Different geos -- scopes don't overlap, not a contradiction
            return False
    # User scope check
    if claim_a.user_scope and claim_b.user_scope:
        if _normalize(claim_a.user_scope) != _normalize(claim_b.user_scope):
            return False
    # Condition check
    if claim_a.condition and claim_b.condition:
        if _normalize(claim_a.condition) != _normalize(claim_b.condition):
            return False
    return True


def _check_qualifier_conflict(claim_a: Claim, claim_b: Claim) -> Optional[float]:
    """
    Detect conflicts arising from universal qualifiers like "only", "always", "never".
    """
    qual_a = claim_a.qualifier or ""
    qual_b = claim_b.qualifier or ""
    
    # "only X" + "also Y (where Y != X)" is a conflict
    # "always" + "never" for the same subject is a conflict
    conflicting_pairs = {
        ("always", "never"),
        ("never", "always"),
        ("only", "also"),
        ("always", "not always"),
    }
    
    for qa, qb in conflicting_pairs:
        if qa in qual_a.lower() and qb in qual_b.lower():
            if claim_a.subject and claim_b.subject:
                if _normalize(claim_a.subject) == _normalize(claim_b.subject):
                    return 0.80
    return None


def _check_ledger_violation(
    candidate: Claim, ledger_claim: Claim
) -> Optional[Tuple[float, ContradictionType]]:
    """
    Check if candidate violates a hard ledger constraint.
    This has the highest priority -- ledger violations are always surfaced.
    """
    # If the ledger claim is negated and candidate is not (or vice versa), flag it
    neg_conflict = _check_negation_conflict(candidate, ledger_claim)
    if neg_conflict:
        return (min(neg_conflict + 0.05, 1.0), ContradictionType.POLICY_CONFLICT)
    
    # Numeric conflict against ledger
    num_conflict = _check_numeric_conflict(candidate, ledger_claim)
    if num_conflict:
        return (min(num_conflict + 0.05, 1.0), ContradictionType.NUMERIC_CONFLICT)
    
    # Keyword overlap with opposite meaning
    words_candidate = set(_normalize(candidate.text).split())
    words_ledger = set(_normalize(ledger_claim.text).split())
    
    # Meaningful overlap (not stop words)
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                  "to", "of", "in", "on", "at", "for", "with", "by"}
    content_a = words_candidate - stop_words
    content_l = words_ledger - stop_words
    
    if not content_a or not content_l:
        return None
    
    overlap = len(content_a & content_l) / max(len(content_a), len(content_l))
    if overlap > 0.4 and candidate.is_negated != ledger_claim.is_negated:
        return (0.78, ContradictionType.POLICY_CONFLICT)
    
    return None


class RuleBasedDetector:
    """
    Fast, zero-cost symbolic contradiction checker.

    Returns DetectorResult for each candidate/prior claim pair.
    High precision, lower recall -- designed to catch the obvious violations
    before spending tokens on NLI or LLM judges.
    """

    def check_pair(
        self,
        candidate: Claim,
        prior: Claim,
    ) -> DetectorResult:
        """
        Compare candidate claim against a single prior claim or ledger entry.
        Returns a DetectorResult with contradiction_probability.
        """
        t0 = time.monotonic()
        prob = 0.0
        ctype = ContradictionType.UNKNOWN
        explanation = None

        # Scope check first -- if scopes are exclusive, no contradiction possible
        if not _scopes_overlap(candidate, prior):
            return DetectorResult(
                detector=DetectorName.RULE_BASED,
                contradiction_probability=0.0,
                contradiction_type=None,
                explanation="Scopes are mutually exclusive -- not a contradiction",
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        # Priority 1: Ledger violations
        if prior.source == "ledger":
            result = _check_ledger_violation(candidate, prior)
            if result:
                prob, ctype = result
                explanation = (
                    f"Candidate claim may violate ledger rule '{prior.rule_id}': "
                    f"'{candidate.text}' vs '{prior.text}'"
                )

        # Priority 2: Numeric conflicts
        if prob < 0.5:
            num_prob = _check_numeric_conflict(candidate, prior)
            if num_prob and num_prob > prob:
                prob = num_prob
                ctype = ContradictionType.NUMERIC_CONFLICT
                explanation = (
                    f"Numeric conflict: '{candidate.text}' states different "
                    f"number than '{prior.text}'"
                )

        # Priority 3: Direct negation
        if prob < 0.5:
            neg_prob = _check_negation_conflict(candidate, prior)
            if neg_prob and neg_prob > prob:
                prob = neg_prob
                ctype = ContradictionType.DIRECT_NEGATION
                explanation = (
                    f"Negation conflict: '{candidate.text}' vs '{prior.text}'"
                )

        # Priority 4: Qualifier conflict
        if prob < 0.5:
            qual_prob = _check_qualifier_conflict(candidate, prior)
            if qual_prob and qual_prob > prob:
                prob = qual_prob
                ctype = ContradictionType.CONDITIONAL_CONFLICT
                explanation = (
                    f"Qualifier conflict: '{candidate.qualifier}' vs '{prior.qualifier}'"
                )

        latency = (time.monotonic() - t0) * 1000
        return DetectorResult(
            detector=DetectorName.RULE_BASED,
            contradiction_probability=prob,
            contradiction_type=ctype if prob > 0.1 else None,
            explanation=explanation,
            latency_ms=latency,
        )

    def check_against_many(
        self,
        candidate: Claim,
        priors: List[Claim],
    ) -> List[DetectorResult]:
        """Check candidate against a list of prior/ledger claims."""
        return [self.check_pair(candidate, prior) for prior in priors]
