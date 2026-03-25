"""
Atomic Claim Extractor.

Breaks natural language text into structured atomic claims.
Each claim carries subject/predicate/object + scope metadata so that
the contradiction detectors operate at the claim level, not paragraph level.

Two extraction modes:
  1. heuristic  -- fast, regex + spaCy-style parsing, zero API cost
  2. llm        -- LLM-powered extraction, higher quality, costs tokens

The heuristic mode is used as a pre-filter; LLM mode is used when
the response is high-stakes or when heuristic confidence is low.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .models import Claim


# ---------------------------------------------------------------------------
# Heuristic extractor (fast, no API cost)
# ---------------------------------------------------------------------------

# Patterns for detecting numbers with units (for numeric conflict detection)
_NUMBER_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(days?|weeks?|months?|years?|hours?|minutes?|seconds?|"
    r"percent|%|dollars?|\$|euros?|GBP|USD|EUR|"
    r"users?|customers?|accounts?|items?|requests?)",
    re.IGNORECASE,
)

# Negation cues
_NEGATION_CUES = {
    "not", "never", "no", "cannot", "can't", "won't", "don't",
    "doesn't", "isn't", "aren't", "wasn't", "weren't", "prohibited",
    "forbidden", "denied", "blocked", "refused", "disallowed",
}

# Scope words
_GEO_TERMS = {"us", "eu", "uk", "united states", "europe", "global", "worldwide"}
_USER_TERMS = {"enterprise", "free", "premium", "admin", "all users", "new users", "existing"}
_TIME_TERMS = {"currently", "now", "today", "as of", "starting", "deprecated", "legacy", "v2", "v1"}


def _contains_negation(text: str) -> bool:
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    return bool(tokens & _NEGATION_CUES)


def _extract_geo_scope(text: str) -> Optional[str]:
    lower = text.lower()
    for term in _GEO_TERMS:
        if term in lower:
            return term.upper() if len(term) <= 3 else term.title()
    return None


def _extract_user_scope(text: str) -> Optional[str]:
    lower = text.lower()
    for term in _USER_TERMS:
        if term in lower:
            return term
    return None


def _extract_time_scope(text: str) -> Optional[str]:
    lower = text.lower()
    # Look for "within X days/months" etc.
    m = _NUMBER_PATTERN.search(text)
    if m:
        return m.group(0).strip()
    for term in _TIME_TERMS:
        if term in lower:
            return term
    return None


def _split_into_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter.
    For production, replace with spaCy sentencizer or similar.
    """
    # Split on . ! ? followed by whitespace or end of string
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    # Also split on semicolons and conjunctions that introduce new claims
    result = []
    for sent in sentences:
        # Split on "; " or " but " or " however "
        sub = re.split(r";\s+| but | however ,? | although | whereas ", sent, flags=re.IGNORECASE)
        result.extend([s.strip() for s in sub if len(s.strip()) > 10])
    return result


class HeuristicExtractor:
    """
    Fast, zero-cost claim extractor using regex and simple NLP heuristics.

    Output quality is lower than LLM extractor but suitable for pre-filtering
    and low-stakes checks.
    """

    def extract(self, text: str, source: str = "response", turn: Optional[int] = None) -> List[Claim]:
        sentences = _split_into_sentences(text)
        claims: List[Claim] = []
        for sent in sentences:
            claim = self._parse_sentence(sent, source=source, turn=turn)
            if claim:
                claims.append(claim)
        return claims

    def _parse_sentence(
        self, sentence: str, source: str, turn: Optional[int]
    ) -> Optional[Claim]:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 8:
            return None

        # Very simple SVO extraction: first noun phrase = subject, verb = predicate
        # This is intentionally lightweight; use LLM extractor for high-stakes
        subject = self._extract_subject(sentence)
        predicate = self._extract_predicate(sentence)
        obj = self._extract_object(sentence)

        return Claim(
            text=sentence,
            subject=subject,
            predicate=predicate,
            object=obj,
            qualifier=self._extract_qualifier(sentence),
            time_scope=_extract_time_scope(sentence),
            geo_scope=_extract_geo_scope(sentence),
            user_scope=_extract_user_scope(sentence),
            source=source,
            source_turn=turn,
            extraction_confidence=0.6,  # Heuristic extraction is lower confidence
            is_negated=_contains_negation(sentence),
        )

    def _extract_subject(self, text: str) -> Optional[str]:
        # Naive: take first capitalized word or "we/you/it/they"
        m = re.match(r"^(\w+(?:\s+\w+)?)", text)
        return m.group(1).lower() if m else None

    def _extract_predicate(self, text: str) -> Optional[str]:
        # Look for common modal + verb patterns
        m = re.search(
            r"\b(are|is|are not|is not|can|cannot|will|will not|must|should|"
            r"allows?|denies?|blocks?|requires?|supports?|provides?|returns?)\b",
            text, re.IGNORECASE
        )
        return m.group(0).lower() if m else None

    def _extract_object(self, text: str) -> Optional[str]:
        # Take the last noun phrase fragment (very rough)
        parts = text.split()
        if len(parts) > 3:
            return " ".join(parts[-3:]).lower()
        return None

    def _extract_qualifier(self, text: str) -> Optional[str]:
        qualifiers = ["only", "always", "never", "at most", "at least", "exactly", "up to"]
        lower = text.lower()
        for q in qualifiers:
            if q in lower:
                return q
        return None


# ---------------------------------------------------------------------------
# LLM extractor (high quality, costs tokens)
# ---------------------------------------------------------------------------

_LLM_EXTRACTION_PROMPT = """You are a precise claim extractor for a contradiction detection system.

Given the following text, extract ALL atomic factual or policy claims.
Return a JSON array. Each element is an object with these fields:
  - text: the claim in full as a complete sentence
  - subject: the main entity (string or null)
  - predicate: the main verb/relation (string or null)
  - object: the main object (string or null)
  - qualifier: limiting words like "only", "always", "never" (string or null)
  - time_scope: when this applies, e.g. "within 30 days", "as of 2024" (string or null)
  - geo_scope: geographic restriction, e.g. "US", "EU" (string or null)
  - user_scope: user group restriction, e.g. "enterprise users" (string or null)
  - condition: conditional scope, e.g. "if the item is unused" (string or null)
  - is_negated: true if the claim is a negation (boolean)

Rules:
- One atomic claim per object (split compound claims)
- Preserve scope qualifiers precisely -- they are critical for avoiding false contradictions
- Do not infer; extract only what is stated
- Return ONLY the JSON array, no commentary

Text to extract from:
{text}"""


class LLMClaimExtractor:
    """
    High-quality claim extractor using an LLM call.

    Parameters
    ----------
    llm_client : Any
        An initialized OpenAI or Anthropic client.
    model : str
        Model to use for extraction.
    provider : str
        "openai" or "anthropic".
    """

    def __init__(
        self,
        llm_client: Any,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
    ) -> None:
        self.client = llm_client
        self.model = model
        self.provider = provider

    def extract(self, text: str, source: str = "response", turn: Optional[int] = None) -> List[Claim]:
        import json

        prompt = _LLM_EXTRACTION_PROMPT.format(text=text)
        raw = self._call_llm(prompt)

        try:
            parsed: List[Dict] = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON array from response
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    return []
            else:
                return []

        claims = []
        for item in parsed:
            claims.append(
                Claim(
                    text=item.get("text", ""),
                    subject=item.get("subject"),
                    predicate=item.get("predicate"),
                    object=item.get("object"),
                    qualifier=item.get("qualifier"),
                    time_scope=item.get("time_scope"),
                    geo_scope=item.get("geo_scope"),
                    user_scope=item.get("user_scope"),
                    condition=item.get("condition"),
                    source=source,
                    source_turn=turn,
                    extraction_confidence=0.9,
                    is_negated=bool(item.get("is_negated", False)),
                )
            )
        return claims

    def _call_llm(self, prompt: str) -> str:
        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"} if "gpt-4" in self.model else None,
            )
            content = resp.choices[0].message.content
            # Wrap in array if the model returned a dict with a key
            if content and content.strip().startswith("{"):
                import json
                d = json.loads(content)
                # Try common keys
                for key in ("claims", "results", "extracted", "items"):
                    if key in d:
                        return json.dumps(d[key])
                return json.dumps(list(d.values())[0] if d else [])
            return content or "[]"
        elif self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        else:
            raise ValueError(f"Unknown provider: {self.provider}")


# ---------------------------------------------------------------------------
# Unified extractor (heuristic + optional LLM upgrade)
# ---------------------------------------------------------------------------

class ClaimExtractor:
    """
    Unified claim extractor.

    Uses heuristic extraction by default. Optionally upgrades to LLM
    extraction for high-stakes responses or when heuristic confidence is low.

    Parameters
    ----------
    use_llm : bool
        Always use LLM extractor (slower, costs tokens).
    llm_client : Any
        Required when use_llm=True.
    model : str
        LLM model name.
    provider : str
        "openai" or "anthropic".
    llm_upgrade_threshold : float
        If heuristic extraction confidence average falls below this,
        automatically upgrade to LLM extractor.
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm_client: Optional[Any] = None,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        llm_upgrade_threshold: float = 0.5,
    ) -> None:
        self._heuristic = HeuristicExtractor()
        self._llm: Optional[LLMClaimExtractor] = None
        self.use_llm = use_llm
        self.llm_upgrade_threshold = llm_upgrade_threshold

        if use_llm or llm_client:
            if llm_client is None:
                raise ValueError("llm_client required when use_llm=True")
            self._llm = LLMClaimExtractor(llm_client, model=model, provider=provider)

    def extract(
        self,
        text: str,
        source: str = "response",
        turn: Optional[int] = None,
        force_llm: bool = False,
    ) -> List[Claim]:
        """Extract atomic claims from text."""
        if force_llm and self._llm:
            return self._llm.extract(text, source=source, turn=turn)

        if self.use_llm and self._llm:
            return self._llm.extract(text, source=source, turn=turn)

        # Heuristic mode with optional LLM upgrade
        heuristic_claims = self._heuristic.extract(text, source=source, turn=turn)

        if self._llm and heuristic_claims:
            avg_conf = sum(c.extraction_confidence for c in heuristic_claims) / len(heuristic_claims)
            if avg_conf < self.llm_upgrade_threshold:
                return self._llm.extract(text, source=source, turn=turn)

        return heuristic_claims
