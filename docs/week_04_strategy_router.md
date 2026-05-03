# Week 4: Strategy Router — Design Notes

## Problem Statement

Every AI request has a cost, a latency, and a risk profile. A naive implementation
routes every request through the most capable (and most expensive) model. The Strategy
Router solves this by deciding the **minimum capability required** to safely answer
a request, then selecting the cheapest execution path that meets that bar.

## Design Decisions

### 1. Zero LLM Calls in the Router

The router makes no LLM calls. All scoring is heuristic. This is intentional:

- **Latency**: LLM calls add 300–1500 ms before the actual work begins. A routing
  decision must be faster than the cheapest execution path it might choose.
- **Cost**: Spending $0.01 to decide whether to spend $0.002 is economically absurd
  at scale.
- **Reliability**: Heuristic routers have no inference-time failure modes. An LLM
  router can itself hallucinate, be prompt-injected, or rate-limited.
- **Determinism**: The same request always produces the same routing decision, which
  is critical for debugging production issues.

The trade-off: heuristic routers can misclassify subtle queries. The `constrained`
fallback strategy provides a safety net — uncertain queries get more model capacity,
not less.

### 2. Priority-Ordered Strategy Rules

Rules are evaluated in a fixed priority order (risk > action > knowledge > complexity > default).
This is a deliberate choice over a weighted scoring system because:

- It is **transparent**: you can explain exactly why a request was escalated.
- It is **auditable**: compliance teams can verify that high-risk requests always
  escalate, regardless of other signals.
- It is **easy to modify**: inserting a new rule is a one-line change with a clear
  priority position.

### 3. Budget Optimizer: Greedy Descending Tier Search

The optimizer tries the best tier first and works down. This maximises quality
within constraints rather than minimising cost. The rationale: users specify budgets
as hard upper bounds, not as targets. Spending $0.05 when the budget is $1.00 is
valid but wasteful in terms of quality.

The latency shrinkage logic (shrink token budget to fit latency) handles a real
production scenario: a request with a generous cost budget but a tight latency SLA
(e.g., a real-time assistant UI) should get a high-quality model with fewer tokens
rather than falling through to a cheap model with more tokens.

### 4. HandoffPayload as Module Contract

The `HandoffPayload` is the contract between the router and the next module. It
answers three questions downstream modules always ask:
1. "What am I?" (`next_module`)
2. "What am I allowed to do?" (`allowed_tools`)
3. "What are my resource limits?" (`constraints`)

This design keeps router knowledge minimal. The router does not know how retrieval
works or how agent orchestration works — it only knows which module to invoke and
under what constraints.

### 5. RouterTrace for Observability

Every routing decision is fully logged. The `fallback_strategy` field records what
would have happened if risk were one level lower. This is a debugging aid: if
production shows too many escalations, the fallback strategy tells you whether
lowering the risk threshold by one step would have routed those requests differently.

## Signal Design: Why These Four?

- **complexity_score**: Proxy for "how hard is this to answer in one shot?" High
  complexity = constrained mode (more careful reasoning with limits).
- **knowledge_need**: Proxy for "does answering require external information?" High
  knowledge need = retrieval mode (fetch before generate).
- **action_need**: Proxy for "does answering require changing state?" High action
  need = agent mode (tool use allowed).
- **risk_score**: Proxy for "could a wrong answer cause serious harm?" High risk =
  escalate (human in the loop).

These four signals are orthogonal enough that each maps to a distinct strategy. A
future version might add a `freshness_need` signal (for time-sensitive queries) or
a `multi_modal_need` signal (for image/audio inputs).

## Extensibility Notes

### Adding a New Strategy

1. Add a value to `Strategy` enum in `core/types.py`.
2. Add a threshold constant in `core/config.py`.
3. Add a branch in `strategy_selector.py` at the correct priority position.
4. Add a routing table entry in `handoff_manager.py`.

### Adding a New Model Tier

1. Add a value to `ModelTier` enum in `core/types.py`.
2. Add a `ModelProfile` entry in `core/config.py`.
3. Add the tier to `_TIER_PREFERENCE_ORDER` in `budget_optimizer.py`.

### Replacing Heuristic Scoring with ML

Each `score_*` function in `intent_classifier.py` has a well-defined signature and
contract. Replacing a heuristic with a small classifier (e.g., a fine-tuned
sentence-transformer) requires only changing the function body — the rest of the
pipeline is unaffected.

## Future Module Integration Points

| Module | Router Hook | Week |
|---|---|---|
| Retrieval Module | `handoff.next_module == "retrieval_module"` | 5 |
| Agent Orchestrator | `handoff.next_module == "agent_orchestrator"` | 6 |
| Evals Framework | Consumes `RouterTrace` JSON for offline analysis | 7 |
| Observability | `core/tracing.py` is the OpenTelemetry injection point | 8 |
