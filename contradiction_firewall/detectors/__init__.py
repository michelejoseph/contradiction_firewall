"""
Detectors package -- multi-judge contradiction detection layer.

Exposes all detector classes for import convenience.
"""
from .rule_based import RuleBasedDetector
from .nli import NLIDetector, APIBasedNLIDetector
from .llm_judge import LLMJudge
from .numeric import NumericTemporalDetector

__all__ = [
    "RuleBasedDetector",
    "NLIDetector",
    "APIBasedNLIDetector",
    "LLMJudge",
    "NumericTemporalDetector",
]
