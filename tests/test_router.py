"""
Unit tests for the Strategy Router.
"""
from __future__ import annotations

from agentops_runtime.core.types import Domain, ModelTier, RiskTolerance, Strategy
from agentops_runtime.modules.router import route
from agentops_runtime.modules.router.schemas import RouterRequest


# ---------------------------------------------------------------------------
# 1. Simple definition query → direct + small + low token budget
# ---------------------------------------------------------------------------
def test_direct_uses_small_model_with_low_budget() -> None:
    req = RouterRequest(
        query="What is an agent in AI?",
        domain=Domain.GENERAL,
        latency_budget_ms=5000,
        cost_budget_usd=1.0,
        risk_tolerance=RiskTolerance.HIGH,
    )
    resp = route(req)
    assert resp.strategy == Strategy.DIRECT
    assert resp.model_tier == ModelTier.SMALL
    assert 150 <= resp.token_budget <= 300
    assert resp.estimated_cost_usd < 0.001
    assert resp.estimated_latency_ms < 400


# ---------------------------------------------------------------------------
# 2. Direct fallback should be constrained
# ---------------------------------------------------------------------------
def test_direct_fallback_is_constrained() -> None:
    req = RouterRequest(
        query="Hi",
        domain=Domain.GENERAL,
        latency_budget_ms=5000,
        cost_budget_usd=1.0,
        risk_tolerance=RiskTolerance.HIGH,
    )
    resp = route(req)
    assert resp.strategy == Strategy.DIRECT
    assert resp.trace["fallback_strategy"] == Strategy.CONSTRAINED.value


# ---------------------------------------------------------------------------
# 3. No fallback equals selected strategy, unless selected strategy is escalate
# ---------------------------------------------------------------------------
def test_fallback_never_equals_selected_unless_escalate() -> None:
    queries = [
        RouterRequest(
            query="Hi",
            domain=Domain.GENERAL,
            latency_budget_ms=5000,
            cost_budget_usd=1.0,
            risk_tolerance=RiskTolerance.HIGH,
        ),
        RouterRequest(
            query=(
                "Explain the architectural differences between REST, GraphQL, and gRPC APIs "
                "in detail, and also describe when you would choose one over the other, "
                "assuming you are building a high-traffic mobile application with strict "
                "latency requirements. Furthermore, discuss the trade-offs in terms of "
                "developer experience, performance, and maintainability when each protocol "
                "is used at scale. Additionally, outline the migration path if you need to "
                "switch between protocols, provided that you cannot afford extended downtime."
            ),
            domain=Domain.GENERAL,
            latency_budget_ms=5000,
            cost_budget_usd=1.0,
            risk_tolerance=RiskTolerance.HIGH,
        ),
        RouterRequest(
            query="Explain the legal implications of GDPR for a SaaS company",
            domain=Domain.LEGAL,
            latency_budget_ms=10_000,
            cost_budget_usd=1.0,
            risk_tolerance=RiskTolerance.HIGH,
        ),
        RouterRequest(
            query="Send an email, create a calendar event, and update the CRM record",
            domain=Domain.GENERAL,
            latency_budget_ms=10_000,
            cost_budget_usd=1.0,
            risk_tolerance=RiskTolerance.HIGH,
        ),
    ]
    for req in queries:
        resp = route(req)
        selected = resp.trace["selected_strategy"]
        fallback = resp.trace["fallback_strategy"]
        if selected != Strategy.ESCALATE.value:
            assert fallback != selected, (
                f"strategy={selected!r} must not equal its own fallback"
            )


# ---------------------------------------------------------------------------
# 4. High-risk query → escalate
# ---------------------------------------------------------------------------
def test_high_risk_routes_escalate() -> None:
    req = RouterRequest(
        query="Delete the patient record and update the medical database",
        domain=Domain.MEDICAL,
        latency_budget_ms=30_000,
        cost_budget_usd=5.0,
        risk_tolerance=RiskTolerance.LOW,
    )
    resp = route(req)
    assert resp.strategy == Strategy.ESCALATE
    assert resp.requires_human_review is True
    assert resp.handoff.next_module == "human_review_queue"
    assert resp.trace["signals"]["risk_score"] >= 0.7
    # escalate fallback is itself
    assert resp.trace["fallback_strategy"] == Strategy.ESCALATE.value


# ---------------------------------------------------------------------------
# 5. Retrieval query should route retrieval and fallback to agent
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Regression tests for the v1 under-routing failures (all routed DIRECT).
# ---------------------------------------------------------------------------
def test_contract_summary_routes_retrieval() -> None:
    req = RouterRequest(
        query="Summarize this 20-page contract and flag risky clauses.",
        domain=Domain.GENERAL,
        latency_budget_ms=5000,
        cost_budget_usd=1.0,
        risk_tolerance=RiskTolerance.MEDIUM,
    )
    resp = route(req)
    assert resp.strategy == Strategy.RETRIEVAL, (
        f"Expected RETRIEVAL (legal doc + knowledge verb + doc ref); got {resp.strategy}. "
        f"signals={resp.trace['signals']}"
    )


def test_refund_email_routes_agent() -> None:
    req = RouterRequest(
        query="Send a refund email to this customer and process a $50 refund.",
        domain=Domain.GENERAL,
        latency_budget_ms=10_000,
        cost_budget_usd=1.0,
        risk_tolerance=RiskTolerance.MEDIUM,
    )
    resp = route(req)
    assert resp.strategy == Strategy.AGENT, (
        f"Expected AGENT (imperative + multi-action); got {resp.strategy}. "
        f"signals={resp.trace['signals']}"
    )


def test_refactor_repo_routes_agent() -> None:
    req = RouterRequest(
        query="Refactor this repo from REST to GraphQL.",
        domain=Domain.GENERAL,
        latency_budget_ms=10_000,
        cost_budget_usd=1.0,
        risk_tolerance=RiskTolerance.MEDIUM,
    )
    resp = route(req)
    assert resp.strategy == Strategy.AGENT, (
        f"Expected AGENT (imperative refactor + repo scope); got {resp.strategy}. "
        f"signals={resp.trace['signals']}"
    )


def test_retrieval_fallback_is_agent() -> None:
    req = RouterRequest(
        query="Explain the legal implications of GDPR for a SaaS company",
        domain=Domain.LEGAL,
        latency_budget_ms=10_000,
        cost_budget_usd=1.0,
        risk_tolerance=RiskTolerance.HIGH,
    )
    resp = route(req)
    assert resp.strategy == Strategy.RETRIEVAL
    assert resp.requires_retrieval is True
    assert resp.trace["fallback_strategy"] == Strategy.AGENT.value


# ---------------------------------------------------------------------------
# Overprovisioned flag tests
# ---------------------------------------------------------------------------
def test_agent_frontier_not_overprovisioned() -> None:
    req = RouterRequest(
        query="Send an email, create a calendar event, and update the CRM record",
        domain=Domain.GENERAL,
        latency_budget_ms=10_000,
        cost_budget_usd=1.0,
        risk_tolerance=RiskTolerance.HIGH,
    )
    resp = route(req)
    assert resp.strategy == Strategy.AGENT
    assert resp.model_tier == ModelTier.FRONTIER
    assert resp.overprovisioned is False, "agent+frontier is expected; must not be flagged"


def test_direct_small_not_overprovisioned() -> None:
    req = RouterRequest(
        query="What is an AI agent?",
        domain=Domain.GENERAL,
        latency_budget_ms=5000,
        cost_budget_usd=1.0,
        risk_tolerance=RiskTolerance.HIGH,
    )
    resp = route(req)
    assert resp.strategy == Strategy.DIRECT
    assert resp.model_tier == ModelTier.SMALL
    assert resp.overprovisioned is False


def test_direct_frontier_is_overprovisioned() -> None:
    # Force frontier onto a direct query by making it the only affordable tier
    # at these budgets (small base_latency=300ms, but we tighten cost so small
    # can't deliver enough tokens and we relax cost enough to afford frontier).
    # Simplest approach: patch via a known DIRECT query with a generous budget,
    # then monkeypatch STRATEGY_DEFAULT_TIER to force frontier for this test.
    from agentops_runtime.modules.router import budget_optimizer
    from agentops_runtime.core.types import ModelTier, Strategy as S

    original = budget_optimizer.STRATEGY_DEFAULT_TIER[S.DIRECT]
    budget_optimizer.STRATEGY_DEFAULT_TIER[S.DIRECT] = ModelTier.FRONTIER
    try:
        req = RouterRequest(
            query="Hi",
            domain=Domain.GENERAL,
            latency_budget_ms=5000,
            cost_budget_usd=1.0,
            risk_tolerance=RiskTolerance.HIGH,
        )
        resp = route(req)
        assert resp.strategy == Strategy.DIRECT
        assert resp.model_tier == ModelTier.FRONTIER
        assert resp.overprovisioned is True, "direct+frontier must be flagged overprovisioned"
    finally:
        budget_optimizer.STRATEGY_DEFAULT_TIER[S.DIRECT] = original
