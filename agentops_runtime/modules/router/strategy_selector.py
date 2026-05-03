from __future__ import annotations

from agentops_runtime.core.config import (
    ACTION_AGENT_THRESHOLD,
    COMPLEXITY_CONSTRAINED_THRESHOLD,
    KNOWLEDGE_RETRIEVAL_THRESHOLD,
    RISK_ESCALATE_THRESHOLD,
)
from agentops_runtime.core.types import Strategy


def select(
    complexity_score: float,
    knowledge_need: float,
    action_need: float,
    risk_score: float,
) -> tuple[Strategy, str]:
    if risk_score >= RISK_ESCALATE_THRESHOLD:
        return (
            Strategy.ESCALATE,
            f"risk_score={risk_score:.3f} exceeds threshold={RISK_ESCALATE_THRESHOLD}; "
            "escalating to human review.",
        )
    if action_need >= ACTION_AGENT_THRESHOLD:
        return (
            Strategy.AGENT,
            f"action_need={action_need:.3f} indicates tool-use required "
            f"(threshold={ACTION_AGENT_THRESHOLD}).",
        )
    if knowledge_need >= KNOWLEDGE_RETRIEVAL_THRESHOLD:
        return (
            Strategy.RETRIEVAL,
            f"knowledge_need={knowledge_need:.3f} indicates external retrieval required "
            f"(threshold={KNOWLEDGE_RETRIEVAL_THRESHOLD}).",
        )
    if complexity_score >= COMPLEXITY_CONSTRAINED_THRESHOLD:
        return (
            Strategy.CONSTRAINED,
            f"complexity_score={complexity_score:.3f} warrants constrained reasoning "
            f"(threshold={COMPLEXITY_CONSTRAINED_THRESHOLD}).",
        )
    return (
        Strategy.DIRECT,
        f"All signals below thresholds — direct single-shot response is sufficient. "
        f"(complexity={complexity_score:.3f}, knowledge={knowledge_need:.3f}, "
        f"action={action_need:.3f}, risk={risk_score:.3f})",
    )
