# Architecture Deep Dive

## The Core Problem

LLM outputs are probabilistic. A model might say "refunds allowed within 30 days" in turn 3 and then say "we can process your 60-day refund" in turn 8. Without a firewall, this contradiction reaches the user.

The Contradiction Firewall solves this at the **infrastructure layer** — not by prompting the model to be consistent, but by detecting and repairing inconsistencies before they ship.

---

## Why Claim-Level Detection Matters

Most naive approaches compare whole responses. This fails because:

- "Refunds are available" and "No returns after 30 days" don't share many words but contradict depending on context
- "The EU has a 14-day rule" and "US customers get 30 days" look like a contradiction but aren't

The firewall extracts **atomic claims** first. A claim is a single, structured assertion:

```
{
  "text": "Refunds are allowed only within 30 days of purchase",
  "subject": "refunds",
  "predicate": "are allowed",
  "qualifier": "only",
  "time_scope": "within 30 days",
  "geo_scope": null,
  "user_scope": null,
  "is_negated": false
}
```

Then contradiction detection is **claim vs claim**, not response vs response. This dramatically reduces false positives.

---

## The Multi-Judge Layer

No single detector is reliable enough to trust alone.

### Judge 1: Rule-Based Checker
- Zero API cost
- Catches exact constraint violations, direct negations, numeric conflicts
- Runs first — if it flags with high confidence, other judges may be skipped

### Judge 2: NLI Model (DeBERTa-v3)
- Runs locally after model download
- Classifies claim pairs as entailment/neutral/contradiction
- Strong on semantic contradiction, catches paraphrases
- ~20ms on CPU, ~5ms on GPU

### Judge 3: LLM Adjudicator
- Most expensive, highest quality
- Called only when rule+NLI disagree or confidence is in the uncertainty band [0.4, 0.75]
- Returns structured JSON with reasoning, scope notes, and repair suggestion
- Uses GPT-4o or Claude for best precision

### Judge 4: Numeric/Temporal Checker
- Specialized for number and unit conflicts
- Handles bounds ("up to 30 days" vs "within 20 days" — compatible)
- Detects active/deprecated state conflicts
- Zero API cost, fast regex-based

### Ensemble Scoring

```python
combined_score = weighted_average(
    [rule_score, nli_score, llm_score, numeric_score],
    weights=[1.0, 0.85, 0.95, 1.1]
)

# Multi-detector agreement boosts confidence
if detectors_flagged >= 2:
    combined_score = min(1.0, combined_score * (1.0 + 0.08 * (detectors_flagged - 1)))
```

---

## The Constraint Ledger

Hard constraints are stored outside the model in a machine-readable ledger:

```json
{
  "rule_id": "refund_policy_001",
  "statement": "Refunds are allowed only within 30 days of purchase",
  "rule_type": "hard_constraint",
  "priority": "critical",
  "geo_scope": null,
  "user_scope": null,
  "time_scope": null,
  "tags": ["refund", "policy"]
}
```

Ledger violations are always surfaced regardless of NLI confidence. The ledger is version-controlled so you can trace any change to a policy.

---

## Scope-Aware Comparison

Before any comparison, the firewall checks if the scopes are compatible:

| Claim A scope | Claim B scope | Result |
|---------------|---------------|--------|
| geo: US       | geo: EU       | Skip — exclusive scopes |
| user: free    | user: enterprise | Skip — different users |
| no scope      | geo: US       | Compare — candidate may apply globally |
| same scope    | same scope    | Compare |

This prevents fake contradictions like "US: 30 days, EU: 14 days" being flagged.

---

## The Repair Loop

When contradiction confidence is in the repair band (0.55–0.85 by default):

1. Build a targeted repair prompt explaining:
   - The original response
   - Which claim is contradictory
   - Which rule/prior it conflicts with
   
2. Ask the LLM to rewrite while resolving the conflict

3. Re-check the repaired response through the full pipeline

4. If still contradictory: retry (up to max_attempts)

5. If all retries exhausted: escalate to BLOCK

The repair prompt is structured to preserve helpfulness while enforcing constraints. A repaired response for "can I get a refund after 60 days?" might become:

> "Our standard policy only allows refunds within 30 days of purchase, so unfortunately a 60-day refund isn't possible under this policy. If you have extenuating circumstances, please contact our support team who may be able to assist."

---

## Action Decision Matrix

| Combined Confidence | Severity      | Action   |
|---------------------|---------------|----------|
| < 0.30              | any           | ALLOW    |
| 0.30 – 0.55         | any           | LOG_ONLY |
| 0.55 – 0.85         | any           | REPAIR   |
| >= 0.85             | non-policy    | BLOCK    |
| >= 0.60             | policy/critical | ESCALATE |

---

## Audit Trail

Every flagged turn produces a structured log entry:

```json
{
  "event_id": "uuid",
  "session_id": "session-uuid",
  "turn": 3,
  "candidate_claim": { "text": "...", "subject": "...", ... },
  "conflicting_claim": { "text": "...", "source": "ledger", "rule_id": "..." },
  "detector_results": [
    { "detector": "rule_based", "probability": 0.85, "type": "policy_conflict" },
    { "detector": "nli", "probability": 0.78, "type": "direct_negation" }
  ],
  "combined_confidence": 0.88,
  "contradiction_type": "policy_conflict",
  "severity": "critical",
  "action": "block",
  "repair_attempted": false,
  "explanation": "Candidate claim may violate ledger rule..."
}
```

This is stored in any configured sink: in-memory, JSON lines, SQLite, or a webhook.

---

## Latency Profile

Typical per-turn overhead (excluding model latency):

| Mode              | Overhead |
|-------------------|----------|
| Rule + numeric only | 2–5ms   |
| + NLI (CPU)       | 20–40ms  |
| + NLI (GPU)       | 5–10ms   |
| + LLM judge       | 200–800ms |

For production, the LLM judge should be reserved for high-stakes or uncertain cases. The rule + numeric + NLI stack is fast enough to run on every turn.

---

## Fail-Safe Behavior

When the system is uncertain, it should fail safely. The `fail_safe` config controls this:

- `"allow"` — pass through uncertain cases (higher availability, lower safety)
- `"repair"` — attempt repair on uncertain cases (default, balanced)
- `"block"` — block uncertain cases (safest, may frustrate users)

A system that says "This answer may conflict with prior policy — revising to the narrower safe interpretation" is more valuable than one that either pretends certainty or blocks everything.
