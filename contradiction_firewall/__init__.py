"""
Contradiction Firewall package.

Main entry point:
  from contradiction_firewall import FirewallMiddleware
  from contradiction_firewall.ledger import ConstraintLedger
"""
from .middleware import FirewallMiddleware, FirewallConfig
from .ledger import ConstraintLedger
from .models import (
    Claim,
    ContraEvent,
    FirewallResponse,
    ActionDecision,
    ContradictionType,
    Priority,
)
from .logging_layer import (
    FirewallLogger,
    InMemoryLogSink,
    JSONLinesSink,
    SQLiteSink,
    WebhookSink,
)

__version__ = "0.1.0"
__all__ = [
    "FirewallMiddleware",
    "FirewallConfig",
    "ConstraintLedger",
    "Claim",
    "ContraEvent",
    "FirewallResponse",
    "ActionDecision",
    "ContradictionType",
    "Priority",
    "FirewallLogger",
    "InMemoryLogSink",
    "JSONLinesSink",
    "SQLiteSink",
    "WebhookSink",
]
