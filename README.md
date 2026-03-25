# Contradiction Firewall

> Infrastructure-layer coherence enforcement for LLM applications. Sits between user ↔ LLM and detects, blocks, or repairs contradictory outputs before they reach the user.

---

## What It Does

The Contradiction Firewall is a **middleware wrapper** around OpenAI/Anthropic API calls that:

- **Detects** when a new LLM output contradicts prior outputs or system-defined rules
- **Blocks or repairs** unstable responses before the user sees them
- **Explains** every flag with a human-auditable trail
- **Learns** from borderline cases via structured logging

This is infrastructure, not a tool. Like Stripe for payments — but for **coherence enforcement**.

---

## Architecture

```
Input Layer
  ├── System prompt + developer rules
  ├── Constraint ledger (hard rules)
  ├── Retrieved docs / RAG context
  └── Recent conversation memory

        ↓

Claim Extraction Layer
  └── Breaks responses into atomic claims
      with: subject, predicate, object, qualifier, time, scope, confidence

        ↓

Candidate Retrieval Layer
  └── Finds prior claims/rules most relevant to compare

        ↓

Multi-Judge Contradiction Layer
  ├── Rule-based checker (exact constraints, numbers, prohibited states)
  ├── NLI model (entailment / contradiction classification)
  ├── LLM adjudicator (nuanced semantic conflicts)
  └── Temporal/numeric consistency checker

        ↓

Risk Engine
  └── Contradiction severity × confidence × policy criticality → action

        ↓

Action Layer
  ├── ALLOW   — pass through
  ├── REPAIR  — inject correction prompt, retry, re-check
  ├── BLOCK   — hard fail with explanation
  └── ESCALATE — human review queue

        ↓

Logging Layer
  └── Contradiction event, detector agreement, repair outcome, audit trail
```

---

## Contradiction Taxonomy

| Type | Example |
|------|---------|
| **Direct negation** | "allowed" vs "not allowed" |
| **Numeric conflict** | "30 days" vs "14 days" |
| **Conditional conflict** | "if X then Y" vs "if X then not Y" |
| **Scope conflict** | "all users" vs "enterprise only" |
| **Temporal conflict** | "currently enabled" vs "deprecated Jan 2026" |
| **Policy conflict** | Response violates system rules |
| **Cross-turn memory** | Contradicts prior answer in same session |

---

## Quick Start

```bash
pip install contradiction-firewall
```

```python
from contradiction_firewall import FirewallMiddleware
from contradiction_firewall.ledger import ConstraintLedger

ledger = ConstraintLedger()
ledger.add_rule(
    rule_id="refund_policy_001",
    statement="Refunds are allowed only within 30 days of purchase",
    rule_type="hard_constraint",
    priority="critical"
)

firewall = FirewallMiddleware(
    provider="openai",           # or "anthropic"
    model="gpt-4o",
    ledger=ledger,
    memory_window=10,
    block_threshold=0.85,
    repair_threshold=0.55,
    max_repair_attempts=2,
)

response = firewall.chat(
    system="You are a helpful customer support agent.",
    messages=[{"role": "user", "content": "Can I get a refund after 60 days?"}]
)

print(response.content)
print(response.firewall_report)
```

---

## Project Structure

```
contradiction_firewall/
├── __init__.py
├── middleware.py
├── extractor.py
├── retriever.py
├── detectors/
│   ├── __init__.py
│   ├── rule_based.py
│   ├── nli.py
│   ├── llm_judge.py
│   └── numeric.py
├── risk_engine.py
├── repair.py
├── ledger.py
├── memory.py
├── models.py
├── logging_layer.py
└── utils.py
```

---

## Design Principles

1. **Claim-level comparison** — never compare whole paragraphs
2. **Multi-judge consensus** — rule engine + NLI + LLM adjudicator must agree before blocking
3. **Repair before block** — correction layer, not just a cop
4. **Time and scope as first-class** — "30 days in US" vs "14 days in EU" is not a contradiction
5. **Confidence-gated actions** — low confidence → log; medium → repair; high → block
6. **Human-auditable trail** — every flag explains which claim, which rule, why, what was done
7. **Precision over recall** — start high-precision, earn trust, then expand

---

## License

MIT
