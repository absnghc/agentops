from __future__ import annotations

import time
from typing import Any


def now_ms() -> int:
    return int(time.time() * 1000)


def make_trace_event(stage: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"stage": stage, "ts_ms": now_ms(), **data}


def build_trace(
    query: str,
    signals: dict[str, float],
    selected_strategy: str,
    model_tier: str,
    token_budget: int,
    estimated_cost: float,
    estimated_latency: int,
    reason: str,
    fallback_strategy: str,
) -> dict[str, Any]:
    return {
        "query_preview": query[:120],
        "signals": signals,
        "selected_strategy": selected_strategy,
        "model_tier": model_tier,
        "token_budget": token_budget,
        "estimated_cost_usd": round(estimated_cost, 6),
        "estimated_latency_ms": estimated_latency,
        "reason": reason,
        "fallback_strategy": fallback_strategy,
        "trace_ts_ms": now_ms(),
    }
