"""
Candidate Retriever.

Finds prior claims and ledger entries that are most relevant to compare
against a new candidate claim.

Critical design principle: only compare new claims against relevant prior claims.
Comparing everything against everything would cause false positive explosion.

Retrieval pipeline:
  1. Keyword pre-filter (fast, zero cost)
  2. BM25 or TF-IDF scoring (optional, improves recall)
  3. Embedding similarity (optional, best quality, costs compute)
  4. Scope-aware filtering (remove mutually exclusive scopes)

The retriever is the gatekeeper. The quality of what it surfaces
directly determines the quality of contradiction detection.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from .models import Claim


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase, remove punctuation, split on whitespace."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in text.split() if len(w) > 2]


_STOP_WORDS = {
    "the", "and", "for", "with", "are", "was", "has", "have",
    "that", "this", "its", "our", "not", "but", "from",
}


def _keyword_overlap_score(claim_a: Claim, claim_b: Claim) -> float:
    """
    Returns a 0.0-1.0 score based on keyword overlap between two claims.
    Used as a fast pre-filter.
    """
    tokens_a = set(_tokenize(claim_a.text)) - _STOP_WORDS
    tokens_b = set(_tokenize(claim_b.text)) - _STOP_WORDS
    
    if not tokens_a or not tokens_b:
        return 0.0
    
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    jaccard = intersection / union if union > 0 else 0.0
    
    # Boost if same subject
    subject_boost = 0.0
    if claim_a.subject and claim_b.subject:
        s_a = set(_tokenize(claim_a.subject))
        s_b = set(_tokenize(claim_b.subject))
        if s_a & s_b:
            subject_boost = 0.2
    
    return min(1.0, jaccard + subject_boost)


def _scope_compatible(claim_a: Claim, claim_b: Claim) -> bool:
    """
    Returns False if the scopes are mutually exclusive.
    Compatible claims may overlap; incompatible claims cannot contradict.
    """
    if claim_a.geo_scope and claim_b.geo_scope:
        if claim_a.geo_scope.lower() != claim_b.geo_scope.lower():
            return False
    if claim_a.user_scope and claim_b.user_scope:
        if claim_a.user_scope.lower() != claim_b.user_scope.lower():
            return False
    return True


class CandidateRetriever:
    """
    Retrieves prior claims and ledger entries relevant to a new candidate claim.

    Parameters
    ----------
    top_k : int
        Maximum number of candidates to return per new claim.
    keyword_threshold : float
        Minimum keyword overlap score to include a candidate.
    use_embeddings : bool
        Enable embedding-based semantic similarity.
    embedding_model : str
        Sentence embedding model (requires sentence-transformers).
    """

    def __init__(
        self,
        top_k: int = 10,
        keyword_threshold: float = 0.05,
        use_embeddings: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.top_k = top_k
        self.keyword_threshold = keyword_threshold
        self.use_embeddings = use_embeddings
        self.embedding_model = embedding_model
        self._embedder: Optional[Any] = None

    def _ensure_embedder_loaded(self) -> None:
        if self._embedder is None and self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embedding_model)
            except ImportError:
                raise ImportError(
                    "Embedding retrieval requires sentence-transformers. "
                    "Install with: pip install sentence-transformers"
                )

    def _embedding_similarity(
        self, candidate: Claim, priors: List[Claim]
    ) -> List[float]:
        """Returns cosine similarity scores for candidate vs each prior."""
        import numpy as np
        self._ensure_embedder_loaded()
        
        texts = [candidate.text] + [p.text for p in priors]
        embeddings = self._embedder.encode(texts, normalize_embeddings=True)
        
        candidate_emb = embeddings[0]
        prior_embs = embeddings[1:]
        
        # Cosine similarity (embeddings are already normalized)
        similarities = (prior_embs @ candidate_emb).tolist()
        return similarities

    def retrieve(
        self,
        candidate: Claim,
        memory_claims: List[Claim],
        ledger_claims: List[Claim],
    ) -> List[Claim]:
        """
        Return the most relevant prior claims + ledger entries to compare
        against the candidate claim.

        Ledger entries are always included if they have any keyword overlap
        (we never want to miss a policy violation).
        Memory claims are filtered by relevance.
        """
        results: List[Tuple[float, Claim]] = []

        # --- Ledger claims: include with lower threshold ---
        for lc in ledger_claims:
            if not _scope_compatible(candidate, lc):
                continue
            score = _keyword_overlap_score(candidate, lc)
            if score >= (self.keyword_threshold * 0.5):  # lower bar for ledger
                results.append((score + 0.1, lc))  # small priority boost

        # --- Memory claims: standard threshold ---
        memory_to_score = [c for c in memory_claims if _scope_compatible(candidate, c)]

        if self.use_embeddings and memory_to_score:
            try:
                similarities = self._embedding_similarity(candidate, memory_to_score)
                for sim, mc in zip(similarities, memory_to_score):
                    if sim >= self.keyword_threshold:
                        results.append((sim, mc))
            except Exception:
                # Fall back to keyword if embeddings fail
                for mc in memory_to_score:
                    score = _keyword_overlap_score(candidate, mc)
                    if score >= self.keyword_threshold:
                        results.append((score, mc))
        else:
            for mc in memory_to_score:
                score = _keyword_overlap_score(candidate, mc)
                if score >= self.keyword_threshold:
                    results.append((score, mc))

        # Sort by score descending, deduplicate, return top_k
        results.sort(key=lambda x: x[0], reverse=True)
        
        seen_ids = set()
        final: List[Claim] = []
        for _, claim in results:
            if claim.claim_id not in seen_ids:
                seen_ids.add(claim.claim_id)
                final.append(claim)
                if len(final) >= self.top_k:
                    break

        return final

    def retrieve_for_batch(
        self,
        candidates: List[Claim],
        memory_claims: List[Claim],
        ledger_claims: List[Claim],
    ) -> List[List[Claim]]:
        """Retrieve candidates for a batch of new claims."""
        return [
            self.retrieve(c, memory_claims, ledger_claims)
            for c in candidates
        ]
