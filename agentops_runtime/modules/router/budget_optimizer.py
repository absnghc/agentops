from __future__ import annotations

from dataclasses import dataclass

from agentops_runtime.core.config import MIN_TOKEN_BUDGET, MODEL_PROFILES
from agentops_runtime.core.types import ModelTier, Strategy


@dataclass(frozen=True)
class BudgetResult:
    model_tier: ModelTier
    token_budget: int
    estimated_cost_usd: float
    estimated_latency_ms: int


# Default (minimum sufficient) model tier per strategy.
STRATEGY_DEFAULT_TIER: dict[Strategy, ModelTier] = {
    Strategy.DIRECT: ModelTier.SMALL,
    Strategy.CONSTRAINED: ModelTier.REASONING,
    Strategy.RETRIEVAL: ModelTier.REASONING,
    Strategy.AGENT: ModelTier.FRONTIER,
    Strategy.ESCALATE: ModelTier.SMALL,
}

# Natural token allocation per strategy — what a typical request of this type needs.
# This is a ceiling, not a floor; budget constraints can reduce it further.
STRATEGY_NATURAL_TOKENS: dict[Strategy, int] = {
    Strategy.DIRECT: 256,
    Strategy.CONSTRAINED: 768,
    Strategy.RETRIEVAL: 1500,
    Strategy.AGENT: 3000,
    Strategy.ESCALATE: 128,
}

# Ordered downgrade path: if the default tier doesn't fit, try cheaper ones.
_DOWNGRADE_ORDER: dict[ModelTier, ModelTier | None] = {
    ModelTier.FRONTIER: ModelTier.REASONING,
    ModelTier.REASONING: ModelTier.SMALL,
    ModelTier.SMALL: None,
}


def _tokens_from_cost(cost_usd: float, cost_per_1k: float) -> int:
    if cost_per_1k <= 0:
        return MIN_TOKEN_BUDGET
    return int((cost_usd / cost_per_1k) * 1000)


def _estimate_latency(profile, token_budget: int) -> int:
    return int(profile.base_latency_ms + (token_budget * profile.latency_per_token_ms))


def optimise(cost_budget_usd: float, latency_budget_ms: int, strategy: Strategy) -> BudgetResult:
    """
    Select the cheapest sufficient model tier for the given strategy, then
    compute a token budget that respects both the strategy's natural needs
    and the caller's cost/latency constraints.
    """
    natural_tokens = STRATEGY_NATURAL_TOKENS[strategy]
    tier: ModelTier | None = STRATEGY_DEFAULT_TIER[strategy]

    while tier is not None:
        profile = MODEL_PROFILES[tier]

        # Can we afford at least MIN_TOKEN_BUDGET with this tier?
        min_cost = (MIN_TOKEN_BUDGET / 1000) * profile.cost_per_1k_tokens_usd
        if min_cost > cost_budget_usd:
            tier = _DOWNGRADE_ORDER[tier]
            continue

        # Does the base latency alone already blow the budget?
        if profile.base_latency_ms > latency_budget_ms:
            tier = _DOWNGRADE_ORDER[tier]
            continue

        # Token budget = natural need, capped by affordability and tier limit.
        affordable = _tokens_from_cost(cost_budget_usd, profile.cost_per_1k_tokens_usd)
        token_budget = max(MIN_TOKEN_BUDGET, min(natural_tokens, affordable, profile.max_tokens))

        # If worst-case latency still exceeds the budget, shrink tokens to fit.
        est_latency = _estimate_latency(profile, token_budget)
        if est_latency > latency_budget_ms:
            available_ms = latency_budget_ms - profile.base_latency_ms
            if available_ms <= 0 or profile.latency_per_token_ms <= 0:
                tier = _DOWNGRADE_ORDER[tier]
                continue
            token_budget = max(MIN_TOKEN_BUDGET, int(available_ms / profile.latency_per_token_ms))
            est_latency = _estimate_latency(profile, token_budget)
            if est_latency > latency_budget_ms:
                tier = _DOWNGRADE_ORDER[tier]
                continue

        estimated_cost = (token_budget / 1000) * profile.cost_per_1k_tokens_usd
        return BudgetResult(
            model_tier=tier,
            token_budget=token_budget,
            estimated_cost_usd=round(estimated_cost, 6),
            estimated_latency_ms=est_latency,
        )

    # Absolute fallback: SMALL at minimum budget.
    profile = MODEL_PROFILES[ModelTier.SMALL]
    return BudgetResult(
        model_tier=ModelTier.SMALL,
        token_budget=MIN_TOKEN_BUDGET,
        estimated_cost_usd=round((MIN_TOKEN_BUDGET / 1000) * profile.cost_per_1k_tokens_usd, 6),
        estimated_latency_ms=profile.base_latency_ms,
    )
