"""
Basic OpenAI example: contradiction firewall in action.

Demonstrates:
  - Setting up a constraint ledger
  - Wrapping OpenAI calls with FirewallMiddleware
  - Reading the firewall audit report
"""
from contradiction_firewall import FirewallMiddleware
from contradiction_firewall.ledger import ConstraintLedger


def main():
    # 1. Define hard constraints
    ledger = ConstraintLedger()
    ledger.add_rule(
        rule_id="refund_policy_001",
        statement="Refunds are allowed only within 30 days of purchase",
        rule_type="hard_constraint",
        priority="critical",
        tags=["refund", "policy"],
    )
    ledger.add_rule(
        rule_id="final_sale_001",
        statement="Final sale items are not eligible for refunds",
        rule_type="hard_constraint",
        priority="critical",
        tags=["refund", "final-sale"],
    )

    # 2. Create the firewall middleware
    firewall = FirewallMiddleware(
        provider="openai",
        model="gpt-4o",
        ledger=ledger,
        block_threshold=0.85,
        repair_threshold=0.55,
        memory_window=10,
        use_llm_judge=True,   # Enable LLM judge for nuanced conflicts
        use_nli=True,          # Enable NLI model (requires sentence-transformers)
    )

    print(f"Session ID: {firewall.session_id}")

    # 3. First turn -- normal response
    response1 = firewall.chat(
        system="You are a helpful customer support agent for TechStore.",
        messages=[
            {"role": "user", "content": "What is your refund policy?"}
        ],
    )
    print("\n=== Turn 1 ===")
    print(f"Response: {response1.content}")
    print(f"Action: {response1.action.value}")
    print(f"Contradictions: {len(response1.contra_events)}")

    # 4. Second turn -- tries to contradict prior policy
    response2 = firewall.chat(
        system="You are a helpful customer support agent for TechStore.",
        messages=[
            {"role": "user", "content": "What is your refund policy?"},
            {"role": "assistant", "content": response1.content},
            {"role": "user", "content": "My purchase was 45 days ago, can I still get a refund?"},
        ],
    )
    print("\n=== Turn 2 ===")
    print(f"Response: {response2.content}")
    print(f"Action: {response2.action.value}")
    print(f"Was repaired: {response2.was_repaired}")
    print(f"Was blocked: {response2.was_blocked}")
    if response2.contra_events:
        print(f"Contradiction detected: {response2.contra_events[0].explanation[:200]}")

    # 5. Show audit trail
    print("\n=== Firewall Report (Turn 2) ===")
    report = response2.firewall_report
    print(f"Total contradictions: {report['contradictions_detected']}")
    print(f"Firewall latency: {report['latency']['firewall_ms']:.1f}ms")


if __name__ == "__main__":
    main()
