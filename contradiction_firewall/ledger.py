"""
Constraint Ledger — machine-readable store of hard constraints.

The ledger is the authoritative source for non-negotiable rules:
  - system rules
  - company / product policies
  - legal constraints
  - user-specified hard requirements

Every entry is a Claim with priority metadata. The firewall treats
ledger violations as highest-severity events regardless of NLI confidence.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .models import Claim, Priority


class LedgerEntry:
    """A single hard-constraint entry in the ledger."""

    def __init__(
        self,
        rule_id: str,
        statement: str,
        rule_type: str = "hard_constraint",
        priority: Priority | str = Priority.CRITICAL,
        geo_scope: Optional[str] = None,
        user_scope: Optional[str] = None,
        time_scope: Optional[str] = None,
        condition: Optional[str] = None,
        tags: Optional[List[str]] = None,
        version: str = "1.0",
        created_at: Optional[str] = None,
        source: str = "ledger",
    ) -> None:
        self.rule_id = rule_id
        self.statement = statement
        self.rule_type = rule_type
        self.priority = Priority(priority) if isinstance(priority, str) else priority
        self.geo_scope = geo_scope
        self.user_scope = user_scope
        self.time_scope = time_scope
        self.condition = condition
        self.tags = tags or []
        self.version = version
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.source = source

    def to_claim(self) -> Claim:
        """Convert this ledger entry into a Claim for comparison."""
        return Claim(
            text=self.statement,
            source="ledger",
            rule_id=self.rule_id,
            geo_scope=self.geo_scope,
            user_scope=self.user_scope,
            time_scope=self.time_scope,
            condition=self.condition,
            extraction_confidence=1.0,  # Ledger claims are ground truth
        )

    def to_dict(self) -> Dict:
        return {
            "rule_id": self.rule_id,
            "statement": self.statement,
            "rule_type": self.rule_type,
            "priority": self.priority.value,
            "geo_scope": self.geo_scope,
            "user_scope": self.user_scope,
            "time_scope": self.time_scope,
            "condition": self.condition,
            "tags": self.tags,
            "version": self.version,
            "created_at": self.created_at,
        }


class ConstraintLedger:
    """
    The central store for hard constraints.

    Usage
    -----
    ledger = ConstraintLedger()
    ledger.add_rule(
        rule_id="refund_policy_001",
        statement="Refunds are allowed only within 30 days of purchase",
        rule_type="hard_constraint",
        priority="critical",
        tags=["refund", "policy"],
    )

    # Load from JSON
    ledger = ConstraintLedger.from_json("constraints/example_ledger.json")
    """

    def __init__(self) -> None:
        self._entries: Dict[str, LedgerEntry] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_rule(
        self,
        rule_id: str,
        statement: str,
        rule_type: str = "hard_constraint",
        priority: str | Priority = Priority.CRITICAL,
        geo_scope: Optional[str] = None,
        user_scope: Optional[str] = None,
        time_scope: Optional[str] = None,
        condition: Optional[str] = None,
        tags: Optional[List[str]] = None,
        version: str = "1.0",
    ) -> "ConstraintLedger":
        """Add a rule and return self (fluent interface)."""
        if rule_id in self._entries:
            raise ValueError(f"Rule '{rule_id}' already exists. Use update_rule() to change it.")
        self._entries[rule_id] = LedgerEntry(
            rule_id=rule_id,
            statement=statement,
            rule_type=rule_type,
            priority=Priority(priority) if isinstance(priority, str) else priority,
            geo_scope=geo_scope,
            user_scope=user_scope,
            time_scope=time_scope,
            condition=condition,
            tags=tags or [],
            version=version,
        )
        return self

    def update_rule(self, rule_id: str, **kwargs) -> "ConstraintLedger":
        """Update fields on an existing entry."""
        if rule_id not in self._entries:
            raise KeyError(f"Rule '{rule_id}' not found.")
        entry = self._entries[rule_id]
        for k, v in kwargs.items():
            if hasattr(entry, k):
                setattr(entry, k, v)
        return self

    def remove_rule(self, rule_id: str) -> "ConstraintLedger":
        self._entries.pop(rule_id, None)
        return self

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, rule_id: str) -> Optional[LedgerEntry]:
        return self._entries.get(rule_id)

    def all_entries(self) -> List[LedgerEntry]:
        return list(self._entries.values())

    def all_claims(self) -> List[Claim]:
        return [e.to_claim() for e in self._entries.values()]

    def by_priority(self, priority: Priority) -> List[LedgerEntry]:
        return [e for e in self._entries.values() if e.priority == priority]

    def by_tag(self, tag: str) -> List[LedgerEntry]:
        return [e for e in self._entries.values() if tag in e.tags]

    def search(self, query: str) -> List[LedgerEntry]:
        """
        Lightweight keyword search over rule statements.
        For semantic search, pass ledger claims through the retriever.
        """
        q = query.lower()
        results = []
        for entry in self._entries.values():
            if re.search(q, entry.statement.lower()):
                results.append(entry)
        return results

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"ConstraintLedger({len(self._entries)} rules)"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self, path: Optional[str] = None) -> str:
        data = {"version": "1.0", "rules": [e.to_dict() for e in self._entries.values()]}
        serialized = json.dumps(data, indent=2)
        if path:
            Path(path).write_text(serialized)
        return serialized

    @classmethod
    def from_json(cls, path_or_str: str) -> "ConstraintLedger":
        """Load from a JSON file path or a raw JSON string."""
        ledger = cls()
        try:
            data = json.loads(path_or_str)
        except (json.JSONDecodeError, ValueError):
            data = json.loads(Path(path_or_str).read_text())
        for rule in data.get("rules", []):
            ledger.add_rule(**{k: v for k, v in rule.items() if v is not None})
        return ledger
