"""
NLI (Natural Language Inference) Contradiction Detector.

Uses a cross-encoder NLI model to classify claim pairs as:
  - entailment    (claim B follows from claim A)
  - neutral       (no strong relationship)
  - contradiction (claim B contradicts claim A)

Default model: cross-encoder/nli-deberta-v3-small
  - Fast (~20ms on CPU, ~5ms on GPU)
  - Strong performance on semantic contradiction
  - Runs locally, no API cost after model download

For environments where local ML is not available, a fallback
API-based NLI endpoint can be configured.
"""
from __future__ import annotations

import time
from typing import Any, List, Optional, Tuple

from ..models import (
    Claim,
    ContradictionType,
    DetectorName,
    DetectorResult,
)

# NLI label indices (HuggingFace cross-encoder convention)
_LABEL_CONTRADICTION = 0
_LABEL_ENTAILMENT = 1
_LABEL_NEUTRAL = 2


def _infer_contradiction_type(premise: str, hypothesis: str) -> ContradictionType:
    """Heuristically infer contradiction type from text for richer logging."""
    import re
    if re.search(r"\d+", premise) and re.search(r"\d+", hypothesis):
        nums_p = re.findall(r"\d+(?:\.\d+)?", premise)
        nums_h = re.findall(r"\d+(?:\.\d+)?", hypothesis)
        if nums_p and nums_h and set(nums_p) != set(nums_h):
            return ContradictionType.NUMERIC_CONFLICT
    if any(w in hypothesis.lower() for w in ["not", "never", "no longer", "cannot"]):
        return ContradictionType.DIRECT_NEGATION
    return ContradictionType.UNKNOWN


class NLIDetector:
    """
    NLI-based contradiction detector using a local cross-encoder model.

    Parameters
    ----------
    model_name : str
        HuggingFace model name. Defaults to DeBERTa-v3-small cross-encoder.
    device : str
        "cpu" or "cuda". Auto-detected if None.
    batch_size : int
        Number of pairs to score in one forward pass.
    contradiction_threshold : float
        Minimum contradiction score to report a detection.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        device: Optional[str] = None,
        batch_size: int = 16,
        contradiction_threshold: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.contradiction_threshold = contradiction_threshold
        self._model: Optional[Any] = None
        self._device = device
        self._loaded = False

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the model on first use."""
        if self._loaded:
            return
        try:
            from sentence_transformers import CrossEncoder
            import torch
            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = CrossEncoder(self.model_name, device=device)
            self._loaded = True
        except ImportError:
            raise ImportError(
                "NLIDetector requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )

    def _score_pair(self, premise: str, hypothesis: str) -> Tuple[float, float, float]:
        """Returns (contradiction_score, entailment_score, neutral_score)."""
        import numpy as np
        self._ensure_model_loaded()
        scores = self._model.predict(
            [(premise, hypothesis)],
            apply_softmax=True,
        )[0]
        # scores is array of [contradiction, entailment, neutral]
        return float(scores[_LABEL_CONTRADICTION]), float(scores[_LABEL_ENTAILMENT]), float(scores[_LABEL_NEUTRAL])

    def check_pair(self, candidate: Claim, prior: Claim) -> DetectorResult:
        """
        Score a single claim pair.
        The NLI model treats prior as premise, candidate as hypothesis.
        """
        t0 = time.monotonic()
        
        premise = prior.text
        hypothesis = candidate.text
        
        if not premise or not hypothesis:
            return DetectorResult(
                detector=DetectorName.NLI,
                contradiction_probability=0.0,
                explanation="Empty claim text -- skipped",
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        try:
            contra_score, entail_score, neutral_score = self._score_pair(premise, hypothesis)
        except Exception as e:
            return DetectorResult(
                detector=DetectorName.NLI,
                contradiction_probability=0.0,
                explanation=f"NLI model error: {e}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        ctype = (
            _infer_contradiction_type(premise, hypothesis)
            if contra_score >= self.contradiction_threshold
            else None
        )

        latency = (time.monotonic() - t0) * 1000
        return DetectorResult(
            detector=DetectorName.NLI,
            contradiction_probability=contra_score,
            contradiction_type=ctype,
            explanation=(
                f"NLI scores — contradiction: {contra_score:.3f}, "
                f"entailment: {entail_score:.3f}, neutral: {neutral_score:.3f}"
            ),
            latency_ms=latency,
        )

    def check_against_many(
        self,
        candidate: Claim,
        priors: List[Claim],
    ) -> List[DetectorResult]:
        """
        Batch-score candidate against multiple prior claims.
        More efficient than calling check_pair in a loop.
        """
        if not priors:
            return []
        
        t0 = time.monotonic()
        self._ensure_model_loaded()

        pairs = [(prior.text, candidate.text) for prior in priors]
        
        try:
            import numpy as np
            all_scores = self._model.predict(pairs, apply_softmax=True)
        except Exception as e:
            return [
                DetectorResult(
                    detector=DetectorName.NLI,
                    contradiction_probability=0.0,
                    explanation=f"Batch NLI error: {e}",
                )
                for _ in priors
            ]

        results = []
        for i, (prior, scores) in enumerate(zip(priors, all_scores)):
            contra_score = float(scores[_LABEL_CONTRADICTION])
            entail_score = float(scores[_LABEL_ENTAILMENT])
            neutral_score = float(scores[_LABEL_NEUTRAL])
            
            ctype = (
                _infer_contradiction_type(prior.text, candidate.text)
                if contra_score >= self.contradiction_threshold
                else None
            )
            
            results.append(
                DetectorResult(
                    detector=DetectorName.NLI,
                    contradiction_probability=contra_score,
                    contradiction_type=ctype,
                    explanation=(
                        f"NLI scores — contra: {contra_score:.3f}, "
                        f"entail: {entail_score:.3f}, neutral: {neutral_score:.3f}"
                    ),
                    latency_ms=(time.monotonic() - t0) * 1000 / len(priors),
                )
            )
        return results


class APIBasedNLIDetector:
    """
    Fallback NLI detector using a hosted NLI API endpoint.
    Useful when local ML inference is not available.

    Parameters
    ----------
    api_url : str
        URL of the NLI inference endpoint (e.g., HuggingFace Inference API).
    api_key : str
        API key for the endpoint.
    model_id : str
        Model identifier on the endpoint.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_id: str = "cross-encoder/nli-deberta-v3-small",
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.model_id = model_id

    def check_pair(self, candidate: Claim, prior: Claim) -> DetectorResult:
        import requests
        import json
        t0 = time.monotonic()
        
        try:
            payload = {
                "inputs": {
                    "source_sentence": prior.text,
                    "sentences": [candidate.text],
                }
            }
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.post(
                f"{self.api_url}/models/{self.model_id}",
                headers=headers,
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            scores = resp.json()
            # Parse according to HuggingFace API format
            contra_score = 0.0
            if isinstance(scores, list):
                for item in scores[0] if isinstance(scores[0], list) else scores:
                    if item.get("label", "").lower() == "contradiction":
                        contra_score = item.get("score", 0.0)
        except Exception as e:
            return DetectorResult(
                detector=DetectorName.NLI,
                contradiction_probability=0.0,
                explanation=f"API NLI error: {e}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        latency = (time.monotonic() - t0) * 1000
        return DetectorResult(
            detector=DetectorName.NLI,
            contradiction_probability=contra_score,
            explanation=f"API NLI contradiction score: {contra_score:.3f}",
            latency_ms=latency,
        )
