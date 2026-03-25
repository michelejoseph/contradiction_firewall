"""
Anthropic (Claude) example with advanced ledger configuration.

Shows:
  - Loading a ledger from JSON
  - Configuring multiple log sinks
  - Custom block threshold
  - Querying the audit log after the conversation
"""
from contradiction_firewall import FirewallMiddleware, FirewallLogger
from contradiction_firewall.ledger import ConstraintLedger
from contradiction_firewall.logging_layer import InMemoryLogSink, JSONLinesSink


def main():
    # Load ledger from JSON file
    ledger = ConstraintLedger.from_json("constraints/example_ledger.json")
    print(f"Loaded {len(ledger)} rules from ledger")

    # Configure logging: in-memory for queries + JSON lines for persistence
    memory_sink = InMemoryLogSink()
    json_sink = JSONLinesSink("logs/firewall_audit.jsonl")
    logger = FirewallLogger(
        sinks=[memory_sink, json_sink],
        emit_to_python_logger=True,
        log_all_turns=True,
    )

    # Create firewall with Anthropic/Claude
    firewall = FirewallMiddleware(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        ledger=ledger,
        block_threshold=0.85,
        repair_threshold=0.60,
        memory_window=15,
        use_llm_judge=True,
        use_nli=False,    # Skip NLI to reduce latency for this example
        logger=logger,
    )

    system = """You are a customer support agent for TechStore.
Always follow company policies exactly. Do not make exceptions to written policies."""

    # Simulate a multi-turn conversation
    conversation = []

    turns = [
        "What is your return policy?",
        "I bought something 45 days ago. Can I return it?",
        "But I have a special circumstance. Can you make an exception for 60-day returns?",
        "What about EU customers? What are their rights?",
    ]

    for user_message in turns:
        conversation.append({"role": "user", "content": user_message})

        response = firewall.chat(
            system=system,
            messages=conversation,
        )

        print(f"\nUser: {user_message}")
        print(f"Assistant: {response.content[:300]}")
        print(f"Action: {response.action.value} | Repaired: {response.was_repaired} | Blocked: {response.was_blocked}")

        conversation.append({"role": "assistant", "content": response.content})

    # Query the audit log
    print("\n=== Audit Summary ===")
    all_records = memory_sink.records
    print(f"Total turns logged: {len(all_records)}")
    flagged = [r for r in all_records if r.contradiction_count > 0]
    print(f"Turns with contradictions: {len(flagged)}")
    repaired = [r for r in all_records if r.was_repaired]
    print(f"Turns repaired: {len(repaired)}")
    blocked = [r for r in all_records if r.was_blocked]
    print(f"Turns blocked: {len(blocked)}")


if __name__ == "__main__":
    main()
