"""
Tests for the claim extractor.
"""
import pytest
from contradiction_firewall.extractor import ClaimExtractor, HeuristicExtractor
from contradiction_firewall.models import Claim


class TestHeuristicExtractor:

    def setup_method(self):
        self.extractor = HeuristicExtractor()

    def test_basic_extraction(self):
        text = "Refunds are allowed within 30 days of purchase."
        claims = self.extractor.extract(text, source="test")
        assert len(claims) >= 1
        assert all(isinstance(c, Claim) for c in claims)

    def test_negation_detection(self):
        text = "Refunds are not allowed after 30 days."
        claims = self.extractor.extract(text, source="test")
        assert len(claims) >= 1
        assert any(c.is_negated for c in claims)

    def test_time_scope_extraction(self):
        text = "You must request a refund within 30 days."
        claims = self.extractor.extract(text, source="test")
        assert len(claims) >= 1
        assert any(c.time_scope is not None for c in claims)

    def test_geo_scope_extraction(self):
        text = "EU customers have a 14-day withdrawal period."
        claims = self.extractor.extract(text, source="test")
        assert len(claims) >= 1
        assert any(c.geo_scope is not None for c in claims)

    def test_empty_text(self):
        claims = self.extractor.extract("", source="test")
        assert claims == []

    def test_short_text(self):
        claims = self.extractor.extract("ok", source="test")
        assert claims == []

    def test_multiple_sentences(self):
        text = (
            "Refunds are available within 30 days. "
            "Final sale items cannot be returned. "
            "EU customers have special rights."
        )
        claims = self.extractor.extract(text, source="test")
        assert len(claims) >= 2

    def test_source_propagation(self):
        text = "Refunds are allowed within 30 days."
        claims = self.extractor.extract(text, source="ledger", turn=3)
        assert all(c.source == "ledger" for c in claims)
        assert all(c.source_turn == 3 for c in claims)

    def test_claim_has_text(self):
        text = "The service is available 24 hours a day."
        claims = self.extractor.extract(text, source="test")
        assert all(len(c.text) > 0 for c in claims)

    def test_qualifier_extraction(self):
        text = "Only enterprise users can access this feature."
        claims = self.extractor.extract(text, source="test")
        assert len(claims) >= 1
        # Should detect "only" qualifier
        qualifiers = [c.qualifier for c in claims if c.qualifier]
        assert len(qualifiers) >= 1


class TestClaimExtractor:

    def test_heuristic_mode(self):
        extractor = ClaimExtractor(use_llm=False)
        text = "Refunds are allowed within 30 days of purchase."
        claims = extractor.extract(text)
        assert len(claims) >= 1

    def test_extractor_returns_claims(self):
        extractor = ClaimExtractor()
        text = "The product is available in 3 colors. It costs $99."
        claims = extractor.extract(text)
        assert isinstance(claims, list)
        assert all(isinstance(c, Claim) for c in claims)
