from __future__ import annotations

from dataclasses import dataclass

from agentops_runtime.core.types import ModelTier


@dataclass(frozen=True)
class ModelProfile:
    tier: ModelTier
    cost_per_1k_tokens_usd: float
    base_latency_ms: int
    latency_per_token_ms: float
    max_tokens: int


MODEL_PROFILES: dict[ModelTier, ModelProfile] = {
    ModelTier.SMALL: ModelProfile(
        tier=ModelTier.SMALL,
        cost_per_1k_tokens_usd=0.002,
        base_latency_ms=300,
        latency_per_token_ms=0.05,
        max_tokens=4096,
    ),
    ModelTier.REASONING: ModelProfile(
        tier=ModelTier.REASONING,
        cost_per_1k_tokens_usd=0.015,
        base_latency_ms=800,
        latency_per_token_ms=0.12,
        max_tokens=8192,
    ),
    ModelTier.FRONTIER: ModelProfile(
        tier=ModelTier.FRONTIER,
        cost_per_1k_tokens_usd=0.060,
        base_latency_ms=1500,
        latency_per_token_ms=0.25,
        max_tokens=16384,
    ),
}

# Strategy selection thresholds — centralised so they can be tuned without
# touching module logic.
RISK_ESCALATE_THRESHOLD: float = 0.7
ACTION_AGENT_THRESHOLD: float = 0.6
KNOWLEDGE_RETRIEVAL_THRESHOLD: float = 0.6
COMPLEXITY_CONSTRAINED_THRESHOLD: float = 0.5

# Domain sensitivity scores used by the intent classifier.
DOMAIN_SENSITIVITY: dict[str, float] = {
    "general": 0.1,
    "legal": 0.8,
    "finance": 0.7,
    "medical": 0.9,
    "support": 0.2,
}

# Risk tolerance multipliers.
RISK_TOLERANCE_MULTIPLIER: dict[str, float] = {
    "low": 1.2,
    "medium": 1.0,
    "high": 0.7,
}

MIN_TOKEN_BUDGET: int = 256
