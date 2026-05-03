from __future__ import annotations

from agentops_runtime.core.types import ModelTier, Strategy
from agentops_runtime.modules.router import (
    budget_optimizer,
    handoff_manager,
    intent_classifier,
    router_trace,
    strategy_selector,
)
from agentops_runtime.modules.router.schemas import RouterRequest, RouterResponse


def route(request: RouterRequest) -> RouterResponse:
    signals = intent_classifier.classify(
        query=request.query,
        domain=request.domain.value,
        risk_tolerance=request.risk_tolerance.value,
    )

    strategy, reason = strategy_selector.select(
        complexity_score=signals["complexity_score"],
        knowledge_need=signals["knowledge_need"],
        action_need=signals["action_need"],
        risk_score=signals["risk_score"],
    )

    # Budget optimizer now receives strategy so it can pick the cheapest
    # sufficient tier rather than the best affordable tier.
    budget = budget_optimizer.optimise(
        cost_budget_usd=request.cost_budget_usd,
        latency_budget_ms=request.latency_budget_ms,
        strategy=strategy,
    )

    requires_retrieval = strategy == Strategy.RETRIEVAL
    requires_tools = strategy == Strategy.AGENT
    requires_human_review = strategy == Strategy.ESCALATE

    # Flag requests where the allocated resources are disproportionate to the
    # strategy's expected tier — useful for cost-governance dashboards (Week 9).
    # retrieval/agent/escalate using reasoning/frontier is expected and correct.
    overprovisioned = (
        (strategy == Strategy.DIRECT and budget.model_tier == ModelTier.FRONTIER)
        or (
            strategy == Strategy.CONSTRAINED
            and budget.model_tier == ModelTier.FRONTIER
            and signals["complexity_score"] < 0.3
        )
        or budget.estimated_cost_usd > request.cost_budget_usd
    )

    handoff = handoff_manager.build_handoff(
        strategy=strategy,
        estimated_cost_usd=budget.estimated_cost_usd,
        estimated_latency_ms=budget.estimated_latency_ms,
        requires_human_review=requires_human_review,
        cost_budget_usd=request.cost_budget_usd,
        latency_budget_ms=request.latency_budget_ms,
    )

    trace = router_trace.build(
        query=request.query,
        signals=signals,
        selected_strategy=strategy,
        model_tier=budget.model_tier.value,
        token_budget=budget.token_budget,
        estimated_cost=budget.estimated_cost_usd,
        estimated_latency=budget.estimated_latency_ms,
        reason=reason,
        overprovisioned=overprovisioned,
    )

    return RouterResponse(
        strategy=strategy,
        model_tier=budget.model_tier,
        token_budget=budget.token_budget,
        requires_retrieval=requires_retrieval,
        requires_tools=requires_tools,
        requires_human_review=requires_human_review,
        estimated_cost_usd=budget.estimated_cost_usd,
        estimated_latency_ms=budget.estimated_latency_ms,
        overprovisioned=overprovisioned,
        reason=reason,
        trace=trace.to_dict(),
        handoff=handoff,
    )
