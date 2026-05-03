from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from agentops_runtime.core.types import Domain, ModelTier, RiskTolerance, Strategy


class RouterRequest(BaseModel):
    model_config = {"frozen": True}

    query: str = Field(..., min_length=1, max_length=8192)
    domain: Domain = Field(default=Domain.GENERAL)
    latency_budget_ms: int = Field(..., ge=100, le=60_000)
    cost_budget_usd: float = Field(..., ge=0.0001, le=10.0)
    risk_tolerance: RiskTolerance = Field(default=RiskTolerance.MEDIUM)

    @field_validator("query")
    @classmethod
    def query_must_not_be_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must contain non-whitespace characters")
        return v.strip()


class HandoffPayload(BaseModel):
    model_config = {"frozen": True}

    next_module: str
    allowed_tools: list[str]
    constraints: dict[str, Any]


class RouterResponse(BaseModel):
    model_config = {"frozen": True}

    strategy: Strategy
    model_tier: ModelTier
    token_budget: int
    requires_retrieval: bool
    requires_tools: bool
    requires_human_review: bool
    estimated_cost_usd: float
    estimated_latency_ms: int
    overprovisioned: bool
    reason: str
    trace: dict[str, Any]
    handoff: HandoffPayload
