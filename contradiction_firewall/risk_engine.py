"""
Risk Engine -- aggregates multi-judge results and decides what to do.

The risk engine is the decision layer that:
  1. Aggregates contradiction scores from multiple detectors
  2. Weights by detector reliability and contradiction type
  3. Determines severity and business risk
  4. Maps to an action: ALLOW | LOG_ONLY | REPAIR | BLOCK | ESCALATE

Design principles:
  - Precision over recall: default to allowing when uncertain
  - Ledger violations always critical regardless of confidence
  - Multi-detector agreement raises confidence (ensemble effect)
  - Contradictions in the same session conversation are highest risk
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .models import (
    ActionDecision,
    Claim,
    ContraEvent,
    ContradictionType,
    DetectorName,
    DetectorResult,
    Priority,
)


# ---------------------------------------------------------------------------
# Detector weights for ensemble scoring
# ---------------------------------------------------------------------------

_DETECTOR_WEIGHTS: Dict[DetectorName, float] = {
    DetectorName.RULE_BASED: 1.0,   # High precision, trust fully
    DetectorName.NUMERIC: 1.1,      # Numeric conflicts are very objective
    DetectorName.NLI: 0.85,         # Good but can have false positives
    DetectorName.LLM_JUDGE: 0.95,   # High quality but costs tokens
}

# Contradiction type severity multipliers
_TYPE_SEVERITY: Dict[ContradictionType, float] = {
    ContradictionType.POLICY_CONFLICT: 1.4,       # Always escalate policy
    ContradictionType.NUMERIC_CONFLICT: 1.2,      # Precise, hard to explain away
    ContradictionType.DIRECT_NEGATION: 1.1,
    ContradictionType.CROSS_TURN_MEMORY: 1.0,
    ContradictionType.CONDITIONAL_CONFLICT: 0.9,
    ContradictionType.SCOPE_CONFLICT: 0.7,        # Often not a real contradiction
    ContradictionType.TEMPORAL_CONFLICT: 0.8,
    ContradictionType.UNKNOWN: 0.6,
}


class RiskConfig:
    """
    Configurable thresholds for the risk engine.

    All thresholds are against the combined_confidence score (0.0-1.0).
    """

    def __init__(
        self,
        log_threshold: float = 0.30,         # Log but allow above this
        repair_threshold: float = 0.55,      # Attempt repair above this
        block_threshold: float = 0.85,       # Hard block above this
        escalate_on_policy: bool = True,     # Always escalate policy violations
        fail_safe: str = "repair",           # "allow" | "repair" | "block"
        min_detector_agreement: int = 1,     # Min detectors that must flag
    ) -> None:
        self.log_threshold = log_threshold
        self.repair_threshold = repair_threshold
        self.block_threshold = block_threshold
        self.escalate_on_policy = escalate_on_policy
        self.fail_safe = fail_safe
        self.min_detector_agreement = min_detector_agreement


def _compute_ensemble_score(
    detector_results: List[DetectorResult],
) -> Tuple[float, int]:
    """
    Compute weighted ensemble contradiction score.
    Returns (combined_score, number_of_flagging_detectors).
    """
    if not detector_results:
        return 0.0, 0

    total_weight = 0.0
    weighted_sum = 0.0
    flagged_count = 0

    for result in detector_results:
        weight = _DETECTOR_WEIGHTS.get(result.detector, 1.0)
        weighted_sum += result.contradiction_probability * weight
        total_weight += weight
        if result.contradiction_probability >= 0.4:
            flagged_count += 1

    raw_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    # Agreement multiplier: when multiple detectors agree, boost confidence
    if flagged_count >= 2:
        agreement_boost = 1.0 + (0.08 * (flagged_count - 1))
        raw_score = min(1.0, raw_score * agreement_boost)

    return raw_score, flagged_count


def _infer_dominant_contradiction_type(
    results: List[DetectorResult],
) -> ContradictionType:
    """Return the most common non-null contradiction type from detector results."""
    type_counts: Dict[ContradictionType, float] = {}
    for r in results:
        if r.contradiction_type and r.contradiction_probability >= 0.3:
            score = r.contradiction_probability * _DETECTOR_WEIGHTS.get(r.detector, 1.0)
            type_counts[r.contradiction_type] = type_counts.get(r.contradiction_type, 0) + score

    if not type_counts:
        return ContradictionType.UNKNOWN
    return max(type_counts, key=type_counts.get)  # type: ignore[arg-type]


def _assign_severity(
    contradiction_type: ContradictionType,
    combined_confidence: float,
    candidate_source: str,
    conflicting_source: str,
) -> Priority:
    """Assign a priority/severity level to the contradiction event."""
    type_mult = _TYPE_SEVERITY.get(contradiction_type, 1.0)
    adjusted = combined_confidence * type_mult

    # Policy and ledger violations are always at least HIGH
    if conflicting_source == "ledger":
        if adjusted >= 0.6:
            return Priority.CRITICAL
        return Priority.HIGH

    if adjusted >= 0.85:
        return Priority.CRITICAL
    elif adjusted >= 0.65:
        return Priority.HIGH
    elif adjusted >= 0.45:
        return Priority.MEDIUM
    else:
        return Priority.LOW


class RiskEngine:
    """
    Aggregates multi-judge results and determines the action for a
    (candidate_claim, prior_claim) contradiction event.
    """

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.config = config or RiskConfig()

    def adjudicate(
        self,
        candidate: Claim,
        conflicting: Claim,
        detector_results: List[DetectorResult],
        turn: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> ContraEvent:
        """
        Take all detector results for a (candidate, conflicting) pair and
        produce a ContraEvent with action decision.
        """
        combined_confidence, flagged_count = _compute_ensemble_score(detector_results)
        
        # Require minimum detector agreement
        if flagged_count < self.config.min_detector_agreement and combined_confidence < 0.7:
            # Not enough evidence -- lower confidence
            combined_confidence *= 0.7

        contradiction_type = _infer_dominant_contradiction_type(detector_results)
        severity = _assign_severity(
            contradiction_type,
            combined_confidence,
            candidate.source,
            conflicting.source,
        )
        business_risk = self._compute_business_risk(
            combined_confidence, severity, contradiction_type, conflicting.source
        )

        action = self._decide_action(
            combined_confidence=combined_confidence,
            severity=severity,
            contradiction_type=contradiction_type,
            conflicting_source=conflicting.source,
        )

        explanation = self._build_explanation(
            candidate=candidate,
            conflicting=conflicting,
            contradiction_type=contradiction_type,
            combined_confidence=combined_confidence,
            action=action,
            detector_results=detector_results,
        )

        return ContraEvent(
            session_id=session_id,
            turn=turn,
            candidate_claim=candidate,
            conflicting_claim=conflicting,
            detector_results=detector_results,
            contradiction_type=contradiction_type,
            combined_confidence=combined_confidence,
            severity=severity,
            business_risk=business_risk,
            action=action,
            explanation=explanation,
        )

    def _compute_business_risk(
        self,
        confidence: float,
        severity: Priority,
        ctype: ContradictionType,
        conflicting_source: str,
    ) -> float:
        """
        Business risk score (0.0-1.0) combines contradiction confidence,
        severity, and whether this is a policy violation.
        """
        severity_weights = {
            Priority.CRITICAL: 1.0,
            Priority.HIGH: 0.75,
            Priority.MEDIUM: 0.5,
            Priority.LOW: 0.25,
        }
        sev_w = severity_weights.get(severity, 0.5)
        type_mult = _TYPE_SEVERITY.get(ctype, 1.0)
        ledger_boost = 1.3 if conflicting_source == "ledger" else 1.0
        return min(1.0, confidence * sev_w * type_mult * ledger_boost)

    def _decide_action(
        self,
        combined_confidence: float,
        severity: Priority,
        contradiction_type: ContradictionType,
        conflicting_source: str,
    ) -> ActionDecision:
        """Map confidence + severity to an action."""
        # Hard escalate for critical policy violations
        if (
            self.config.escalate_on_policy
            and contradiction_type == ContradictionType.POLICY_CONFLICT
            and severity == Priority.CRITICAL
        ):
            return ActionDecision.ESCALATE

        # Hard block
        if combined_confidence >= self.config.block_threshold:
            return ActionDecision.BLOCK

        # Repair band
        if combined_confidence >= self.config.repair_threshold:
            return ActionDecision.REPAIR

        # Log band
        if combined_confidence >= self.config.log_threshold:
            return ActionDecision.LOG_ONLY

        # Below all thresholds -- allow
        return ActionDecision.ALLOW

    def _build_explanation(
        self,
        candidate: Claim,
        conflicting: Claim,
        contradiction_type: ContradictionType,
        combined_confidence: float,
        action: ActionDecision,
        detector_results: List[DetectorResult],
    ) -> str:
        """Build a human-readable explanation for the audit trail."""
        lines = [
            f"Contradiction detected ({combined_confidence:.2f} confidence):",
            f"  Candidate: '{candidate.text[:100]}'",
            f"  Conflicts with: '{conflicting.text[:100]}' (source: {conflicting.source})",
            f"  Type: {contradiction_type.value}",
            f"  Action: {action.value}",
        ]

        # Add detector breakdown
        flagged = [r for r in detector_results if r.contradiction_probability >= 0.3]
        if flagged:
            lines.append("  Detectors flagged:")
            for r in flagged:
                lines.append(
                    f"    - {r.detector.value}: {r.contradiction_probability:.2f}"
                    + (f" ({r.explanation[:60]})" if r.explanation else "")
                )
        
        return "\n".join(lines)

    def adjudicate_batch(
        self,
        candidate: Claim,
        conflicting_claims: List[Claim],
        results_per_claim: List[List[DetectorResult]],
        turn: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> List[ContraEvent]:
        """Adjudicate multiple (candidate, conflicting) pairs."""
        events = []
        for conflicting, det_results in zip(conflicting_claims, results_per_claim):
            event = self.adjudicate(
                candidate=candidate,
                conflicting=conflicting,
                detector_results=det_results,
                turn=turn,
                session_id=session_id,
            )
            events.append(event)
        return events
