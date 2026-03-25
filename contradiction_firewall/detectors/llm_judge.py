"""
LLM Judge -- nuanced semantic contradiction detector.

The LLM judge is the third layer in the multi-judge stack.
It is invoked only when:
  1. Rule-based + NLI detectors disagree, OR
  2. Rule/NLI confidence is in the uncertainty band [0.4, 0.75]
  3. The response is flagged as high-stakes

The LLM judge provides:
  - Nuanced reasoning over ambiguous pairs
  - Contradiction type classification
  - Scope-aware adjudication
  - Repair suggestion if contradiction is confirmed

Prompt is structured as a zero-shot JSON-output chain-of-thought.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from ..models import (
    Claim,
    ContradictionType,
    DetectorName,
    DetectorResult,
)


_JUDGE_SYSTEM_PROMPT = """You are a contradiction detection expert. Your job is to determine whether two claims contradict each other.

Be precise. Many apparent contradictions are NOT contradictions because:
- They apply to different geographic regions (US vs EU)
- They apply to different user types (free vs enterprise)
- They apply at different times (current policy vs deprecated policy)
- One is an exception to the other
- Different conditions apply (if X vs if Y)

You must output ONLY valid JSON, no commentary."""


_JUDGE_USER_PROMPT = """Analyze these two claims for contradiction.

CLAIM A (prior / established):
"{prior}"
Source: {prior_source}
{prior_scope}

CLAIM B (new / candidate):
"{candidate}"
Source: {candidate_source}
{candidate_scope}

Task: Determine if Claim B contradicts Claim A.

Output this JSON object:
{{
  "is_contradiction": true/false,
  "confidence": 0.0-1.0,
  "contradiction_type": "direct_negation" | "numeric_conflict" | "conditional_conflict" | "scope_conflict" | "temporal_conflict" | "policy_conflict" | "cross_turn_memory" | "none",
  "reasoning": "brief explanation (2-3 sentences)",
  "scope_note": "explain any scope differences that prevent contradiction (or null)",
  "repair_suggestion": "how to rewrite Claim B to resolve the conflict (or null if no contradiction)"
}}"""


def _format_scope(claim: Claim) -> str:
    parts = []
    if claim.time_scope:
        parts.append(f"Time scope: {claim.time_scope}")
    if claim.geo_scope:
        parts.append(f"Geographic scope: {claim.geo_scope}")
    if claim.user_scope:
        parts.append(f"User scope: {claim.user_scope}")
    if claim.condition:
        parts.append(f"Condition: {claim.condition}")
    return "\n".join(parts) if parts else "(no scope specified)"


def _parse_judge_response(raw: str) -> Dict:
    """Parse LLM JSON output with fallback extraction."""
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON object from response
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    
    # Return safe fallback
    return {
        "is_contradiction": False,
        "confidence": 0.0,
        "contradiction_type": "none",
        "reasoning": "Failed to parse LLM response",
        "scope_note": None,
        "repair_suggestion": None,
    }


class LLMJudge:
    """
    LLM-based contradiction judge for nuanced semantic conflicts.

    Parameters
    ----------
    llm_client : Any
        Initialized OpenAI or Anthropic client.
    model : str
        Model to use. GPT-4o or Claude recommended for best precision.
    provider : str
        "openai" or "anthropic".
    temperature : float
        Keep at 0 for deterministic judgment.
    max_tokens : int
        Max tokens for the judge response.
    """

    def __init__(
        self,
        llm_client: Any,
        model: str = "gpt-4o",
        provider: str = "openai",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self.client = llm_client
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens

    def check_pair(self, candidate: Claim, prior: Claim) -> DetectorResult:
        """Adjudicate a single claim pair."""
        t0 = time.monotonic()
        
        user_content = _JUDGE_USER_PROMPT.format(
            prior=prior.text,
            prior_source=prior.source,
            prior_scope=_format_scope(prior),
            candidate=candidate.text,
            candidate_source=candidate.source,
            candidate_scope=_format_scope(candidate),
        )
        
        try:
            raw = self._call_llm(user_content)
        except Exception as e:
            return DetectorResult(
                detector=DetectorName.LLM_JUDGE,
                contradiction_probability=0.0,
                explanation=f"LLM judge call failed: {e}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        
        parsed = _parse_judge_response(raw)
        
        confidence = float(parsed.get("confidence", 0.0))
        if not parsed.get("is_contradiction", False):
            confidence = 0.0
        
        ctype_str = parsed.get("contradiction_type", "none")
        try:
            ctype = ContradictionType(ctype_str) if ctype_str != "none" else None
        except ValueError:
            ctype = ContradictionType.UNKNOWN
        
        reasoning = parsed.get("reasoning", "")
        scope_note = parsed.get("scope_note")
        repair = parsed.get("repair_suggestion")
        
        explanation_parts = [f"LLM judge: {reasoning}"]
        if scope_note:
            explanation_parts.append(f"Scope note: {scope_note}")
        if repair:
            explanation_parts.append(f"Suggested repair: {repair}")
        
        latency = (time.monotonic() - t0) * 1000
        result = DetectorResult(
            detector=DetectorName.LLM_JUDGE,
            contradiction_probability=confidence,
            contradiction_type=ctype,
            explanation=" | ".join(explanation_parts),
            latency_ms=latency,
        )
        # Attach repair suggestion as extra attribute for the repair layer
        result._repair_suggestion = repair  # type: ignore[attr-defined]
        return result

    def check_against_many(
        self,
        candidate: Claim,
        priors: List[Claim],
        max_pairs: int = 5,
    ) -> List[DetectorResult]:
        """
        Judge candidate against up to max_pairs priors.
        LLM calls are expensive -- only pass the most promising candidates
        (those already flagged by rule/NLI layers).
        """
        # Only check top max_pairs to bound cost
        return [self.check_pair(candidate, prior) for prior in priors[:max_pairs]]

    def _call_llm(self, user_content: str) -> str:
        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or "{}"
        
        elif self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            return resp.content[0].text
        
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
