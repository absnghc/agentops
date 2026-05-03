from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentops_runtime.core import tracing
from agentops_runtime.core.types import Strategy

# Fixed fallback map: always escalates in capability, never matches selected strategy
# except when already at the top of the escalation ladder.
_FALLBACK_MAP: dict[Strategy, Strategy] = {
    Strategy.DIRECT: Strategy.CONSTRAINED,
    Strategy.CONSTRAINED: Strategy.RETRIEVAL,
    Strategy.RETRIEVAL: Strategy.AGENT,
    Strategy.AGENT: Strategy.ESCALATE,
    Strategy.ESCALATE: Strategy.ESCALATE,
}


@dataclass(frozen=True)
class RouterTrace:
    query_preview: str
    signals: dict[str, Any]
    selected_strategy: str
    model_tier: str
    token_budget: int
    estimated_cost_usd: float
    estimated_latency_ms: int
    reason: str
    fallback_strategy: str
    overprovisioned: bool
    trace_ts_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_preview": self.query_preview,
            "signals": self.signals,
            "selected_strategy": self.selected_strategy,
            "model_tier": self.model_tier,
            "token_budget": self.token_budget,
            "estimated_cost_usd": self.estimated_cost_usd,
            "estimated_latency_ms": self.estimated_latency_ms,
            "reason": self.reason,
            "fallback_strategy": self.fallback_strategy,
            "overprovisioned": self.overprovisioned,
            "trace_ts_ms": self.trace_ts_ms,
        }


def build(
    query: str,
    signals: dict[str, Any],
    selected_strategy: Strategy,
    model_tier: str,
    token_budget: int,
    estimated_cost: float,
    estimated_latency: int,
    reason: str,
    overprovisioned: bool,
) -> RouterTrace:
    fallback = _FALLBACK_MAP[selected_strategy].value
    return RouterTrace(
        query_preview=query[:120],
        signals=signals,
        selected_strategy=selected_strategy.value,
        model_tier=model_tier,
        token_budget=token_budget,
        estimated_cost_usd=round(estimated_cost, 6),
        estimated_latency_ms=estimated_latency,
        reason=reason,
        fallback_strategy=fallback,
        overprovisioned=overprovisioned,
        trace_ts_ms=tracing.now_ms(),
    )
