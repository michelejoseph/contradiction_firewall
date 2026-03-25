"""
Data models for the Contradiction Firewall.
All core dataclasses and enums used across the system.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ContradictionType(str, Enum):
    """Taxonomy of contradiction kinds the firewall can detect."""
    DIRECT_NEGATION = "direct_negation"          # "allowed" vs "not allowed"
    NUMERIC_CONFLICT = "numeric_conflict"         # "30 days" vs "14 days"
    CONDITIONAL_CONFLICT = "conditional_conflict" # "if X then Y" vs "if X then not Y"
    SCOPE_CONFLICT = "scope_conflict"             # "all users" vs "enterprise only"
    TEMPORAL_CONFLICT = "temporal_conflict"       # "currently" vs "deprecated"
    POLICY_CONFLICT = "policy_conflict"           # violates system rules / ledger
    CROSS_TURN_MEMORY = "cross_turn_memory"       # contradicts prior session answer
    UNKNOWN = "unknown"


class ActionDecision(str, Enum):
    """What the firewall decides to do with a flagged response."""
    ALLOW = "allow"
    REPAIR = "repair"
    BLOCK = "block"
    ESCALATE = "escalate"
    LOG_ONLY = "log_only"


class Priority(str, Enum):
    """Constraint / contradiction priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DetectorName(str, Enum):
    """Detectors available in the multi-judge layer."""
    RULE_BASED = "rule_based"
    NLI = "nli"
    LLM_JUDGE = "llm_judge"
    NUMERIC = "numeric"


# ---------------------------------------------------------------------------
# Atomic Claim
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    """
    An atomic, structured assertion extracted from text.

    A claim is the fundamental unit of contradiction detection.
    Contradictions are detected at the claim level, never at the paragraph level.
    """
    claim_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""                        # natural language form
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    qualifier: Optional[str] = None       # "only", "never", "always", etc.
    # Scope metadata — critical to avoid false positives
    time_scope: Optional[str] = None      # "within 30 days", "as of Jan 2026"
    geo_scope: Optional[str] = None       # "US", "EU", "global"
    user_scope: Optional[str] = None      # "enterprise users", "all users"
    condition: Optional[str] = None       # "if X", "when Y"
    # Provenance
    source: str = "response"              # "response" | "system_prompt" | "ledger" | "memory"
    source_turn: Optional[int] = None     # which conversation turn
    rule_id: Optional[str] = None         # if from the constraint ledger
    # Confidence in the extraction itself
    extraction_confidence: float = 1.0
    extracted_at: datetime = field(default_factory=datetime.utcnow)
    # Raw negation flag for quick rule-based checks
    is_negated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "qualifier": self.qualifier,
            "time_scope": self.time_scope,
            "geo_scope": self.geo_scope,
            "user_scope": self.user_scope,
            "condition": self.condition,
            "source": self.source,
            "source_turn": self.source_turn,
            "rule_id": self.rule_id,
            "extraction_confidence": self.extraction_confidence,
            "is_negated": self.is_negated,
        }


# ---------------------------------------------------------------------------
# Contradiction Event
# ---------------------------------------------------------------------------

@dataclass
class DetectorResult:
    """Result from a single judge/detector."""
    detector: DetectorName
    contradiction_probability: float          # 0.0 – 1.0
    contradiction_type: Optional[ContradictionType] = None
    explanation: Optional[str] = None
    latency_ms: Optional[float] = None

    def is_flagged(self, threshold: float = 0.5) -> bool:
        return self.contradiction_probability >= threshold


@dataclass
class ContraEvent:
    """
    A fully adjudicated contradiction event.
    Recorded in the audit log for every flagged pair of claims.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None
    turn: Optional[int] = None

    # The two claims in conflict
    candidate_claim: Optional[Claim] = None       # from the new response
    conflicting_claim: Optional[Claim] = None     # from memory / ledger / system prompt

    # Multi-judge results
    detector_results: List[DetectorResult] = field(default_factory=list)

    # Aggregated risk assessment
    contradiction_type: ContradictionType = ContradictionType.UNKNOWN
    combined_confidence: float = 0.0              # weighted ensemble score
    severity: Priority = Priority.MEDIUM
    business_risk: float = 0.0                    # 0.0 – 1.0

    # Action taken
    action: ActionDecision = ActionDecision.LOG_ONLY
    repair_attempted: bool = False
    repair_succeeded: bool = False
    repair_explanation: Optional[str] = None

    # Human-readable explanation of the flag
    explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "turn": self.turn,
            "candidate_claim": self.candidate_claim.to_dict() if self.candidate_claim else None,
            "conflicting_claim": self.conflicting_claim.to_dict() if self.conflicting_claim else None,
            "detector_results": [
                {
                    "detector": r.detector.value,
                    "probability": r.contradiction_probability,
                    "type": r.contradiction_type.value if r.contradiction_type else None,
                    "explanation": r.explanation,
                }
                for r in self.detector_results
            ],
            "contradiction_type": self.contradiction_type.value,
            "combined_confidence": self.combined_confidence,
            "severity": self.severity.value,
            "business_risk": self.business_risk,
            "action": self.action.value,
            "repair_attempted": self.repair_attempted,
            "repair_succeeded": self.repair_succeeded,
            "repair_explanation": self.repair_explanation,
            "explanation": self.explanation,
        }


# ---------------------------------------------------------------------------
# Firewall Response
# ---------------------------------------------------------------------------

@dataclass
class FirewallResponse:
    """
    The object returned to the caller after the firewall has processed a response.
    Wraps the (possibly repaired) content with a full audit report.
    """
    # Final content delivered to the user
    content: str = ""
    # Whether the content was modified from the raw model output
    was_repaired: bool = False
    was_blocked: bool = False
    # Raw model output before any repair
    raw_content: Optional[str] = None
    # All contradiction events detected in this turn
    contra_events: List[ContraEvent] = field(default_factory=list)
    # Overall action taken for this turn
    action: ActionDecision = ActionDecision.ALLOW
    # Latency breakdown
    model_latency_ms: Optional[float] = None
    firewall_latency_ms: Optional[float] = None
    # Session context
    session_id: Optional[str] = None
    turn: Optional[int] = None

    @property
    def firewall_report(self) -> Dict[str, Any]:
        """Human-auditable summary of firewall activity this turn."""
        return {
            "session_id": self.session_id,
            "turn": self.turn,
            "action": self.action.value,
            "was_repaired": self.was_repaired,
            "was_blocked": self.was_blocked,
            "contradictions_detected": len(self.contra_events),
            "events": [e.to_dict() for e in self.contra_events],
            "latency": {
                "model_ms": self.model_latency_ms,
                "firewall_ms": self.firewall_latency_ms,
            },
        }

    def __str__(self) -> str:
        return self.content
