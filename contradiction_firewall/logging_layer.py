"""
Structured Audit Logging Layer.

Every contradiction event is logged with:
  - Which claim was problematic
  - Which prior claim or rule it conflicts with
  - Contradiction type and confidence
  - Detectors that flagged it (and their scores)
  - Action taken (allow/repair/block/escalate)
  - Repair outcome if repair was attempted

Supports multiple sinks:
  - Python standard logging (default)
  - JSON lines file
  - In-memory buffer (for testing)
  - Webhook (Slack, PagerDuty, etc.)
  - SQLite (lightweight persistent store)

The audit trail must be human-readable AND machine-queryable.
If the system cannot explain a flag, it should be trusted less.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .models import ActionDecision, ContraEvent, FirewallResponse, Priority


logger = logging.getLogger("contradiction_firewall")


# ---------------------------------------------------------------------------
# Log record
# ---------------------------------------------------------------------------

@dataclass
class FirewallLogRecord:
    """A complete log entry for one firewall turn."""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    session_id: Optional[str] = None
    turn: Optional[int] = None
    action: str = ActionDecision.ALLOW.value
    was_repaired: bool = False
    was_blocked: bool = False
    contradiction_count: int = 0
    events: List[Dict] = field(default_factory=list)
    model_latency_ms: Optional[float] = None
    firewall_latency_ms: Optional[float] = None
    # For human review queue
    flagged_for_review: bool = False
    review_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "turn": self.turn,
            "action": self.action,
            "was_repaired": self.was_repaired,
            "was_blocked": self.was_blocked,
            "contradiction_count": self.contradiction_count,
            "events": self.events,
            "model_latency_ms": self.model_latency_ms,
            "firewall_latency_ms": self.firewall_latency_ms,
            "flagged_for_review": self.flagged_for_review,
            "review_reason": self.review_reason,
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict())


# ---------------------------------------------------------------------------
# Log sinks
# ---------------------------------------------------------------------------

class InMemoryLogSink:
    """In-memory log sink for testing and development."""

    def __init__(self, max_records: int = 10_000) -> None:
        self.records: List[FirewallLogRecord] = []
        self.max_records = max_records
        self._lock = threading.Lock()

    def write(self, record: FirewallLogRecord) -> None:
        with self._lock:
            if len(self.records) >= self.max_records:
                self.records.pop(0)
            self.records.append(record)

    def query(
        self,
        session_id: Optional[str] = None,
        action: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> List[FirewallLogRecord]:
        results = self.records
        if session_id:
            results = [r for r in results if r.session_id == session_id]
        if action:
            results = [r for r in results if r.action == action]
        return results

    def clear(self) -> None:
        with self._lock:
            self.records.clear()


class JSONLinesSink:
    """Writes one JSON line per record to a log file."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: FirewallLogRecord) -> None:
        line = record.to_json_line() + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)


class SQLiteSink:
    """
    Lightweight persistent store using SQLite.
    Good for single-process deployments or development.
    """

    def __init__(self, path: str = "firewall_audit.db") -> None:
        self.path = path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS firewall_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    turn INTEGER,
                    action TEXT NOT NULL,
                    was_repaired INTEGER NOT NULL DEFAULT 0,
                    was_blocked INTEGER NOT NULL DEFAULT 0,
                    contradiction_count INTEGER NOT NULL DEFAULT 0,
                    firewall_latency_ms REAL,
                    events_json TEXT,
                    flagged_for_review INTEGER NOT NULL DEFAULT 0,
                    review_reason TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON firewall_log(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_action ON firewall_log(action)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON firewall_log(timestamp)")
            conn.commit()

    def write(self, record: FirewallLogRecord) -> None:
        with self._lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute(
                    """
                    INSERT INTO firewall_log
                        (timestamp, session_id, turn, action, was_repaired,
                         was_blocked, contradiction_count, firewall_latency_ms,
                         events_json, flagged_for_review, review_reason)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        record.timestamp,
                        record.session_id,
                        record.turn,
                        record.action,
                        int(record.was_repaired),
                        int(record.was_blocked),
                        record.contradiction_count,
                        record.firewall_latency_ms,
                        json.dumps(record.events),
                        int(record.flagged_for_review),
                        record.review_reason,
                    ),
                )
                conn.commit()


class WebhookSink:
    """
    HTTP webhook sink. Useful for Slack, PagerDuty, or custom review queues.
    Only fires for events that meet the severity threshold.
    """

    def __init__(
        self,
        webhook_url: str,
        min_action: str = ActionDecision.BLOCK.value,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 5.0,
    ) -> None:
        self.webhook_url = webhook_url
        self.min_action = min_action
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self._action_severity = {
            ActionDecision.ALLOW.value: 0,
            ActionDecision.LOG_ONLY.value: 1,
            ActionDecision.REPAIR.value: 2,
            ActionDecision.BLOCK.value: 3,
            ActionDecision.ESCALATE.value: 4,
        }

    def write(self, record: FirewallLogRecord) -> None:
        record_sev = self._action_severity.get(record.action, 0)
        min_sev = self._action_severity.get(self.min_action, 3)
        if record_sev < min_sev:
            return
        
        import urllib.request
        payload = json.dumps(record.to_dict()).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url,
            data=payload,
            headers=self.headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout):
                pass
        except Exception as e:
            logger.warning(f"Webhook delivery failed: {e}")


# ---------------------------------------------------------------------------
# Main logging layer
# ---------------------------------------------------------------------------

class FirewallLogger:
    """
    Central logging layer. Supports multiple sinks simultaneously.

    Parameters
    ----------
    sinks : list
        List of sink objects (InMemoryLogSink, JSONLinesSink, SQLiteSink, WebhookSink).
    emit_to_python_logger : bool
        Also emit to the Python standard logger at INFO/WARNING level.
    log_all_turns : bool
        Log even clean (no-contradiction) turns. Useful for analytics.
    """

    def __init__(
        self,
        sinks: Optional[List[Any]] = None,
        emit_to_python_logger: bool = True,
        log_all_turns: bool = False,
    ) -> None:
        self.sinks = sinks or [InMemoryLogSink()]
        self.emit_to_python_logger = emit_to_python_logger
        self.log_all_turns = log_all_turns

    def log_response(self, response: FirewallResponse) -> Optional[FirewallLogRecord]:
        """
        Log a complete firewall response.
        Returns the created log record (useful for testing).
        """
        # Skip clean turns if not logging all
        if not self.log_all_turns and not response.contra_events:
            return None
        if not self.log_all_turns and response.action == ActionDecision.ALLOW:
            return None

        record = FirewallLogRecord(
            session_id=response.session_id,
            turn=response.turn,
            action=response.action.value,
            was_repaired=response.was_repaired,
            was_blocked=response.was_blocked,
            contradiction_count=len(response.contra_events),
            events=[e.to_dict() for e in response.contra_events],
            model_latency_ms=response.model_latency_ms,
            firewall_latency_ms=response.firewall_latency_ms,
            flagged_for_review=response.action == ActionDecision.ESCALATE,
            review_reason=(
                response.contra_events[0].explanation
                if response.action == ActionDecision.ESCALATE and response.contra_events
                else None
            ),
        )

        for sink in self.sinks:
            try:
                sink.write(record)
            except Exception as e:
                logger.error(f"Log sink {type(sink).__name__} failed: {e}")

        if self.emit_to_python_logger:
            self._emit_to_stdlib(response, record)

        return record

    def _emit_to_stdlib(
        self, response: FirewallResponse, record: FirewallLogRecord
    ) -> None:
        action = response.action
        if action in (ActionDecision.BLOCK, ActionDecision.ESCALATE):
            level = logging.WARNING
        elif action in (ActionDecision.REPAIR, ActionDecision.LOG_ONLY):
            level = logging.INFO
        else:
            level = logging.DEBUG

        msg = (
            f"[Firewall] session={record.session_id} turn={record.turn} "
            f"action={record.action} contradictions={record.contradiction_count} "
            f"repaired={record.was_repaired} blocked={record.was_blocked} "
            f"fw_latency={record.firewall_latency_ms:.1f}ms"
            if record.firewall_latency_ms else
            f"[Firewall] session={record.session_id} turn={record.turn} "
            f"action={record.action} contradictions={record.contradiction_count}"
        )
        logger.log(level, msg)

    def add_sink(self, sink: Any) -> "FirewallLogger":
        self.sinks.append(sink)
        return self

    def get_memory_sink(self) -> Optional[InMemoryLogSink]:
        """Returns the first InMemoryLogSink if one is registered."""
        for sink in self.sinks:
            if isinstance(sink, InMemoryLogSink):
                return sink
        return None
