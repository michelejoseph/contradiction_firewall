"""
Repair Layer -- self-repairing contradiction resolution.

When the risk engine decides to REPAIR (not block), this module:
  1. Builds a targeted repair prompt explaining what contradicts what
  2. Asks the LLM to rewrite the response to resolve the conflict
  3. Re-checks the repaired response through the full pipeline
  4. If repair succeeds: returns repaired content
  5. If repair fails after max attempts: escalates to BLOCK

The best firewall is not just a cop. It is a correction layer.
"""
from __future__ import annotations

import time
from typing import Any, List, Optional, Tuple

from .models import (
    ActionDecision,
    ContraEvent,
    FirewallResponse,
)


_REPAIR_SYSTEM_PROMPT = """You are a careful AI assistant. You have been asked to revise your previous response because it may contradict an established rule or prior statement.

Your job: rewrite the response to resolve the contradiction while preserving its helpfulness.
- Keep the same tone and intent
- Respect the constraint exactly
- Do not introduce new contradictions
- If the constraint limits what you can say, say so clearly and honestly"""


_REPAIR_USER_PROMPT = """Your previous response contained a potential contradiction.

ORIGINAL RESPONSE:
{original}

CONTRADICTION DETECTED:
{contradiction_explanation}

CONSTRAINT TO RESPECT:
{conflicting_claim}
(Source: {conflicting_source})

Please rewrite the original response to resolve this contradiction.
The rewritten response should:
1. Not contradict the constraint above
2. Be clear and helpful to the user
3. Acknowledge the constraint if it limits the answer

REWRITTEN RESPONSE:"""


def _build_repair_prompt(
    original_response: str,
    event: ContraEvent,
) -> Tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for a repair call.
    """
    conflicting = event.conflicting_claim
    if conflicting is None:
        return _REPAIR_SYSTEM_PROMPT, original_response

    constraint_text = conflicting.text
    source = conflicting.source
    explanation = event.explanation or f"Type: {event.contradiction_type.value}"

    user = _REPAIR_USER_PROMPT.format(
        original=original_response,
        contradiction_explanation=explanation,
        conflicting_claim=constraint_text,
        conflicting_source=source,
    )
    return _REPAIR_SYSTEM_PROMPT, user


class RepairLayer:
    """
    Self-repair loop for contradicted responses.

    Parameters
    ----------
    llm_client : Any
        Initialized OpenAI or Anthropic client.
    model : str
        Model to use for repair (typically same as generation model).
    provider : str
        "openai" or "anthropic".
    max_attempts : int
        Maximum repair attempts before giving up and blocking.
    temperature : float
        Temperature for repair generation (slightly > 0 helps rephrase).
    max_tokens : int
        Maximum tokens for repaired response.
    """

    def __init__(
        self,
        llm_client: Any,
        model: str = "gpt-4o",
        provider: str = "openai",
        max_attempts: int = 2,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self.client = llm_client
        self.model = model
        self.provider = provider
        self.max_attempts = max_attempts
        self.temperature = temperature
        self.max_tokens = max_tokens

    def repair(
        self,
        original_response: str,
        events: List[ContraEvent],
        recheck_fn: Optional[Any] = None,
    ) -> Tuple[str, bool, str]:
        """
        Attempt to repair a contradicted response.

        Parameters
        ----------
        original_response : str
            The raw model response that was flagged.
        events : List[ContraEvent]
            The contradiction events detected in this response.
        recheck_fn : Optional[Callable]
            If provided, called with (repaired_text) -> List[ContraEvent].
            If no events are returned, repair is considered successful.

        Returns
        -------
        (repaired_text, success, explanation)
        """
        if not events:
            return original_response, True, "No contradictions to repair"

        # Use the highest-confidence event for the repair prompt
        primary_event = max(events, key=lambda e: e.combined_confidence)

        current_response = original_response
        last_explanation = ""

        for attempt in range(1, self.max_attempts + 1):
            t0 = time.monotonic()
            system, user = _build_repair_prompt(current_response, primary_event)

            try:
                repaired = self._call_llm(system, user)
            except Exception as e:
                last_explanation = f"Repair LLM call failed (attempt {attempt}): {e}"
                continue

            repair_latency = (time.monotonic() - t0) * 1000

            # Re-check if a recheck function is provided
            if recheck_fn is not None:
                try:
                    remaining_events = recheck_fn(repaired)
                    if not remaining_events or all(
                        e.action == ActionDecision.ALLOW
                        or e.action == ActionDecision.LOG_ONLY
                        for e in remaining_events
                    ):
                        last_explanation = (
                            f"Repair succeeded on attempt {attempt} "
                            f"({repair_latency:.0f}ms). "
                            f"Resolved: {primary_event.explanation[:80] if primary_event.explanation else ''}"
                        )
                        return repaired, True, last_explanation
                    else:
                        current_response = repaired  # try to repair further
                        last_explanation = f"Attempt {attempt} still contradictory, retrying"
                        # Use the new primary event for next attempt
                        primary_event = max(remaining_events, key=lambda e: e.combined_confidence)
                except Exception as e:
                    last_explanation = f"Recheck failed (attempt {attempt}): {e}"
                    return repaired, False, last_explanation
            else:
                # No recheck -- trust the repair
                last_explanation = (
                    f"Repair applied (attempt {attempt}, no recheck). "
                    f"Resolved: {primary_event.explanation[:80] if primary_event.explanation else ''}"
                )
                return repaired, True, last_explanation

        # All attempts exhausted
        last_explanation = (
            f"Repair failed after {self.max_attempts} attempts. "
            f"Original contradiction: {primary_event.explanation[:100] if primary_event.explanation else ''}"
        )
        return original_response, False, last_explanation

    def _call_llm(self, system: str, user: str) -> str:
        """Call the LLM for repair generation."""
        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content or ""

        elif self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def build_repair_explanation(
        self,
        original: str,
        repaired: str,
        events: List[ContraEvent],
        success: bool,
    ) -> str:
        """Build a human-readable explanation of what the repair did."""
        if not success:
            return "Repair failed -- response blocked"

        event_summaries = []
        for e in events[:3]:  # Top 3 events
            if e.conflicting_claim:
                event_summaries.append(
                    f"  - Conflict with '{e.conflicting_claim.text[:60]}' "
                    f"(conf: {e.combined_confidence:.2f})"
                )

        summary = [
            "Response repaired to resolve contradiction(s):",
            *event_summaries,
            f"  Original length: {len(original)} chars",
            f"  Repaired length: {len(repaired)} chars",
        ]
        return "\n".join(summary)
