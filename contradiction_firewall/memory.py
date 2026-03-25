"""
Conversation Memory Manager.

Maintains a rolling window of atomic claims extracted from prior turns
so the firewall can compare new responses against recent conversation history.

Design decisions:
  - Store claims, not raw text -- contradiction detection is claim-vs-claim
  - Scope-aware: claims carry time/geo/user metadata to prevent false positives
  - Configurable window: default 10 turns, keeps memory bounded
  - Indexed by topic + entity for efficient retrieval
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from .models import Claim


class TurnRecord:
    """A record of one conversation turn with its extracted claims."""

    def __init__(
        self,
        turn: int,
        role: str,  # "assistant" | "user" | "system"
        raw_text: str,
        claims: Optional[List[Claim]] = None,
    ) -> None:
        self.turn = turn
        self.role = role
        self.raw_text = raw_text
        self.claims: List[Claim] = claims or []

    def __repr__(self) -> str:
        return f"TurnRecord(turn={self.turn}, role={self.role}, claims={len(self.claims)})"


class ConversationMemory:
    """
    Rolling window memory store for a single session.

    Parameters
    ----------
    window : int
        Maximum number of turns to retain.
    store_user_turns : bool
        Also extract and store claims from user messages.
    """

    def __init__(self, window: int = 10, store_user_turns: bool = False) -> None:
        self.window = window
        self.store_user_turns = store_user_turns
        self._turns: Deque[TurnRecord] = deque(maxlen=window)
        self._turn_counter = 0
        self.session_id: Optional[str] = None

    def add_turn(self, role: str, raw_text: str, claims: Optional[List[Claim]] = None) -> int:
        """Add a new turn. Returns the turn number assigned."""
        if role == "user" and not self.store_user_turns:
            return self._turn_counter
        self._turn_counter += 1
        record = TurnRecord(
            turn=self._turn_counter,
            role=role,
            raw_text=raw_text,
            claims=claims or [],
        )
        self._turns.append(record)
        return self._turn_counter

    def clear(self) -> None:
        self._turns.clear()
        self._turn_counter = 0

    def all_claims(self, role_filter: Optional[str] = None) -> List[Claim]:
        """Return all stored claims, optionally filtered by role."""
        claims: List[Claim] = []
        for record in self._turns:
            if role_filter and record.role != role_filter:
                continue
            claims.extend(record.claims)
        return claims

    def recent_claims(self, n_turns: int = 5, role_filter: Optional[str] = None) -> List[Claim]:
        """Return claims from the last n_turns turns."""
        recent = list(self._turns)[-n_turns:]
        claims: List[Claim] = []
        for record in recent:
            if role_filter and record.role != role_filter:
                continue
            claims.extend(record.claims)
        return claims

    def get_turn(self, turn: int) -> Optional[TurnRecord]:
        for record in self._turns:
            if record.turn == turn:
                return record
        return None

    def candidate_claims_for(self, new_claim: Claim, top_k: int = 20) -> List[Claim]:
        """
        Return stored claims that share subject/topic with the new claim.
        Lightweight keyword-based pre-filter before semantic retrieval.
        """
        all_c = self.all_claims(role_filter="assistant")
        if not new_claim.subject:
            return all_c[:top_k]

        subject_lower = new_claim.subject.lower()
        scored: List[Tuple[int, Claim]] = []
        for c in all_c:
            score = 0
            if c.subject and subject_lower in c.subject.lower():
                score += 3
            if c.predicate and c.predicate == new_claim.predicate:
                score += 2
            if c.object and new_claim.object and new_claim.object.lower() in c.object.lower():
                score += 1
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def snapshot(self) -> Dict:
        return {
            "session_id": self.session_id,
            "turn_counter": self._turn_counter,
            "window": self.window,
            "turns": [
                {
                    "turn": r.turn,
                    "role": r.role,
                    "text_preview": r.raw_text[:100],
                    "claim_count": len(r.claims),
                }
                for r in self._turns
            ],
        }

    def __len__(self) -> int:
        return len(self._turns)

    def __repr__(self) -> str:
        return f"ConversationMemory(turns={len(self._turns)}/{self.window}, session={self.session_id})"
