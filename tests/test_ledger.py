"""
Tests for the Constraint Ledger.
"""
import json
import pytest
from contradiction_firewall.ledger import ConstraintLedger, LedgerEntry
from contradiction_firewall.models import Priority, Claim


class TestConstraintLedger:

    def test_add_rule(self):
        ledger = ConstraintLedger()
        ledger.add_rule(
            rule_id="test_001",
            statement="Test rule statement",
            priority="critical",
        )
        assert len(ledger) == 1

    def test_add_duplicate_rule_raises(self):
        ledger = ConstraintLedger()
        ledger.add_rule(rule_id="dup_001", statement="Rule A")
        with pytest.raises(ValueError):
            ledger.add_rule(rule_id="dup_001", statement="Rule B")

    def test_get_rule(self):
        ledger = ConstraintLedger()
        ledger.add_rule(rule_id="get_001", statement="Get this rule", priority="high")
        entry = ledger.get("get_001")
        assert entry is not None
        assert entry.statement == "Get this rule"
        assert entry.priority == Priority.HIGH

    def test_get_nonexistent_returns_none(self):
        ledger = ConstraintLedger()
        assert ledger.get("nonexistent") is None

    def test_remove_rule(self):
        ledger = ConstraintLedger()
        ledger.add_rule(rule_id="rm_001", statement="Remove me")
        ledger.remove_rule("rm_001")
        assert ledger.get("rm_001") is None
        assert len(ledger) == 0

    def test_update_rule(self):
        ledger = ConstraintLedger()
        ledger.add_rule(rule_id="upd_001", statement="Original")
        ledger.update_rule("upd_001", statement="Updated")
        assert ledger.get("upd_001").statement == "Updated"

    def test_all_claims_returns_claims(self):
        ledger = ConstraintLedger()
        ledger.add_rule(rule_id="c1", statement="Claim one")
        ledger.add_rule(rule_id="c2", statement="Claim two")
        claims = ledger.all_claims()
        assert len(claims) == 2
        assert all(isinstance(c, Claim) for c in claims)

    def test_claims_have_ledger_source(self):
        ledger = ConstraintLedger()
        ledger.add_rule(rule_id="src_001", statement="Source test")
        claims = ledger.all_claims()
        assert all(c.source == "ledger" for c in claims)

    def test_claims_have_rule_id(self):
        ledger = ConstraintLedger()
        ledger.add_rule(rule_id="rid_001", statement="Rule ID test")
        claims = ledger.all_claims()
        assert any(c.rule_id == "rid_001" for c in claims)

    def test_by_priority(self):
        ledger = ConstraintLedger()
        ledger.add_rule(rule_id="crit_001", statement="Critical", priority="critical")
        ledger.add_rule(rule_id="high_001", statement="High", priority="high")
        ledger.add_rule(rule_id="med_001", statement="Medium", priority="medium")
        
        critical = ledger.by_priority(Priority.CRITICAL)
        assert len(critical) == 1
        assert critical[0].rule_id == "crit_001"

    def test_by_tag(self):
        ledger = ConstraintLedger()
        ledger.add_rule(rule_id="tag_001", statement="Tagged", tags=["refund", "policy"])
        ledger.add_rule(rule_id="tag_002", statement="Other", tags=["security"])
        
        refund_rules = ledger.by_tag("refund")
        assert len(refund_rules) == 1

    def test_search(self):
        ledger = ConstraintLedger()
        ledger.add_rule(rule_id="s1", statement="Refunds are allowed within 30 days")
        ledger.add_rule(rule_id="s2", statement="Password must be 12 characters")
        
        results = ledger.search("refund")
        assert len(results) == 1
        assert results[0].rule_id == "s1"

    def test_scope_metadata_preserved(self):
        ledger = ConstraintLedger()
        ledger.add_rule(
            rule_id="scope_001",
            statement="EU withdrawal right",
            geo_scope="EU",
            user_scope="consumers",
            time_scope="within 14 days",
        )
        entry = ledger.get("scope_001")
        assert entry.geo_scope == "EU"
        assert entry.user_scope == "consumers"
        assert entry.time_scope == "within 14 days"

    def test_to_claim_preserves_scope(self):
        ledger = ConstraintLedger()
        ledger.add_rule(
            rule_id="scope_002",
            statement="EU rule",
            geo_scope="EU",
        )
        claim = ledger.get("scope_002").to_claim()
        assert claim.geo_scope == "EU"

    def test_serialization_roundtrip(self):
        ledger = ConstraintLedger()
        ledger.add_rule(rule_id="r1", statement="Rule one", priority="critical", tags=["a"])
        ledger.add_rule(rule_id="r2", statement="Rule two", priority="medium", geo_scope="US")
        
        json_str = ledger.to_json()
        data = json.loads(json_str)
        assert "rules" in data
        assert len(data["rules"]) == 2
        
        # Reload
        loaded = ConstraintLedger.from_json(json_str)
        assert len(loaded) == 2
        assert loaded.get("r1").statement == "Rule one"
        assert loaded.get("r2").geo_scope == "US"

    def test_fluent_interface(self):
        ledger = (
            ConstraintLedger()
            .add_rule(rule_id="f1", statement="Fluent one")
            .add_rule(rule_id="f2", statement="Fluent two")
        )
        assert len(ledger) == 2

    def test_repr(self):
        ledger = ConstraintLedger()
        ledger.add_rule(rule_id="repr_001", statement="Test")
        assert "1" in repr(ledger)
