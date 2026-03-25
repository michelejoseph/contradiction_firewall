"""
Tests for contradiction detectors.
"""
import pytest
from contradiction_firewall.models import Claim, DetectorName
from contradiction_firewall.detectors.rule_based import RuleBasedDetector
from contradiction_firewall.detectors.numeric import NumericTemporalDetector


def make_claim(text, source="response", is_negated=False, geo_scope=None,
               user_scope=None, time_scope=None, subject=None, predicate=None,
               obj=None, rule_id=None):
    return Claim(
        text=text,
        source=source,
        is_negated=is_negated,
        geo_scope=geo_scope,
        user_scope=user_scope,
        time_scope=time_scope,
        subject=subject,
        predicate=predicate,
        object=obj,
        rule_id=rule_id,
    )


class TestRuleBasedDetector:

    def setup_method(self):
        self.detector = RuleBasedDetector()

    def test_no_contradiction_on_same_claim(self):
        claim = make_claim("Refunds are allowed within 30 days.")
        result = self.detector.check_pair(claim, claim)
        # Same claim should not self-contradict at high confidence
        assert result.contradiction_probability < 0.7

    def test_negation_conflict_detected(self):
        claim_a = make_claim(
            "Refunds are allowed.",
            subject="refunds",
            predicate="are",
            is_negated=False,
        )
        claim_b = make_claim(
            "Refunds are not allowed.",
            subject="refunds",
            predicate="are not",
            is_negated=True,
        )
        result = self.detector.check_pair(claim_b, claim_a)
        assert result.contradiction_probability >= 0.5

    def test_scope_exclusion_prevents_false_positive(self):
        us_claim = make_claim("Refunds are allowed within 30 days.", geo_scope="US")
        eu_claim = make_claim("Refunds are allowed within 14 days.", geo_scope="EU")
        result = self.detector.check_pair(eu_claim, us_claim)
        # Different geo scopes -- should not be flagged as contradiction
        assert result.contradiction_probability < 0.3

    def test_ledger_violation_detected(self):
        ledger_claim = make_claim(
            "Refunds are allowed only within 30 days of purchase.",
            source="ledger",
            rule_id="refund_001",
            is_negated=False,
        )
        candidate = make_claim(
            "We can process your refund. It has been 60 days.",
            source="response",
            is_negated=False,
        )
        result = self.detector.check_pair(candidate, ledger_claim)
        assert result.detector == DetectorName.RULE_BASED

    def test_returns_detector_result(self):
        from contradiction_firewall.models import DetectorResult
        claim_a = make_claim("Service is available.")
        claim_b = make_claim("Service is not available.", is_negated=True)
        result = self.detector.check_pair(claim_b, claim_a)
        assert isinstance(result, DetectorResult)
        assert result.detector == DetectorName.RULE_BASED
        assert 0.0 <= result.contradiction_probability <= 1.0

    def test_check_against_many(self):
        candidate = make_claim("Refunds are not allowed.", is_negated=True)
        priors = [
            make_claim("Refunds are allowed.", is_negated=False),
            make_claim("Service is available."),
            make_claim("The price is $99."),
        ]
        results = self.detector.check_against_many(candidate, priors)
        assert len(results) == 3


class TestNumericTemporalDetector:

    def setup_method(self):
        self.detector = NumericTemporalDetector()

    def test_numeric_conflict_detected(self):
        claim_a = make_claim("Refunds are allowed within 30 days.")
        claim_b = make_claim("Refunds are available up to 60 days after purchase.")
        result = self.detector.check_pair(claim_b, claim_a)
        assert result.contradiction_probability >= 0.5

    def test_no_numeric_conflict_same_value(self):
        claim_a = make_claim("The policy covers 30 days.")
        claim_b = make_claim("You have 30 days to return items.")
        result = self.detector.check_pair(claim_b, claim_a)
        assert result.contradiction_probability < 0.5

    def test_different_units_no_conflict(self):
        claim_a = make_claim("Response time is 24 hours.")
        claim_b = make_claim("Delivery takes 7 days.")
        result = self.detector.check_pair(claim_b, claim_a)
        # Different units -- no conflict
        assert result.contradiction_probability < 0.3

    def test_temporal_active_vs_deprecated(self):
        claim_a = make_claim("The v1 API is currently active and supported.")
        claim_b = make_claim("The v1 API is deprecated and no longer supported.")
        result = self.detector.check_pair(claim_b, claim_a)
        assert result.contradiction_probability >= 0.5

    def test_geo_scope_reduces_numeric_conflict(self):
        claim_a = make_claim("Refunds within 30 days in the US.", geo_scope="US")
        claim_b = make_claim("EU customers have 14 days to return items.", geo_scope="EU")
        result = self.detector.check_pair(claim_b, claim_a)
        # Different geo scopes -- conflict probability should be reduced
        assert result.contradiction_probability < 0.7

    def test_returns_detector_result(self):
        from contradiction_firewall.models import DetectorResult
        claim_a = make_claim("The free tier allows 100 calls per day.")
        claim_b = make_claim("Free users can make 200 calls daily.")
        result = self.detector.check_pair(claim_b, claim_a)
        assert isinstance(result, DetectorResult)
        assert result.detector == DetectorName.NUMERIC


class TestLedgerIntegration:
    """Integration tests: detectors + ledger."""

    def test_ledger_claim_violation_via_rule_detector(self):
        from contradiction_firewall.ledger import ConstraintLedger
        ledger = ConstraintLedger()
        ledger.add_rule(
            rule_id="refund_30d",
            statement="Refunds are allowed only within 30 days of purchase",
            priority="critical",
        )
        ledger_claims = ledger.all_claims()
        candidate = make_claim(
            "We are happy to process a refund after 60 days for valued customers.",
        )
        detector = RuleBasedDetector()
        results = detector.check_against_many(candidate, ledger_claims)
        assert len(results) == 1

    def test_no_conflict_with_compatible_ledger(self):
        from contradiction_firewall.ledger import ConstraintLedger
        ledger = ConstraintLedger()
        ledger.add_rule(
            rule_id="refund_30d",
            statement="Refunds are allowed only within 30 days of purchase",
            priority="critical",
        )
        ledger_claims = ledger.all_claims()
        candidate = make_claim(
            "You can get a refund within 30 days of your purchase.",
        )
        detector = NumericTemporalDetector()
        results = detector.check_against_many(candidate, ledger_claims)
        # Same 30-day figure -- no numeric conflict
        assert all(r.contradiction_probability < 0.5 for r in results)
