from __future__ import annotations

from agentops_runtime.core.types import Strategy
from agentops_runtime.modules.router.schemas import HandoffPayload

_STRATEGY_ROUTING_TABLE: dict[Strategy, tuple[str, list[str]]] = {
    Strategy.DIRECT: ("llm_executor", []),
    Strategy.CONSTRAINED: ("llm_executor", ["calculator", "formatter"]),
    Strategy.RETRIEVAL: ("retrieval_module", ["vector_search", "bm25_search", "reranker"]),
    Strategy.AGENT: ("agent_orchestrator", [
        "web_search", "code_executor", "calendar_api",
        "email_sender", "database_query", "file_reader",
    ]),
    Strategy.ESCALATE: ("human_review_queue", []),
}


def build_handoff(
    strategy: Strategy,
    estimated_cost_usd: float,
    estimated_latency_ms: int,
    requires_human_review: bool,
    cost_budget_usd: float,
    latency_budget_ms: int,
) -> HandoffPayload:
    next_module, allowed_tools = _STRATEGY_ROUTING_TABLE[strategy]
    constraints = {
        "max_cost_usd": cost_budget_usd,
        "max_latency_ms": latency_budget_ms,
        "human_review_required": requires_human_review,
        "projected_cost_usd": estimated_cost_usd,
        "projected_latency_ms": estimated_latency_ms,
    }
    return HandoffPayload(
        next_module=next_module,
        allowed_tools=allowed_tools,
        constraints=constraints,
    )
