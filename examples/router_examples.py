"""
Runnable examples demonstrating the Strategy Router across all five paths.

Usage:
    python examples/router_examples.py
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from agentops_runtime.modules.router import route
from agentops_runtime.modules.router.schemas import RouterRequest


def load_sample_requests() -> list[dict]:
    samples_path = pathlib.Path(__file__).parent / "sample_requests.json"
    with samples_path.open() as f:
        return json.load(f)


def print_response_summary(req_meta: dict, response) -> None:
    sep = "-" * 70
    print(sep)
    print(f"[{req_meta['id']}] {req_meta['description']}")
    query_preview = req_meta["request"]["query"]
    print(f"  Query        : {query_preview[:80]}{'...' if len(query_preview) > 80 else ''}")
    print(f"  Strategy     : {response.strategy.value.upper()}")
    print(f"  Model Tier   : {response.model_tier.value}")
    print(f"  Token Budget : {response.token_budget:,}")
    print(f"  Est. Cost    : ${response.estimated_cost_usd:.6f}")
    print(f"  Est. Latency : {response.estimated_latency_ms} ms")
    print(f"  Retrieval    : {response.requires_retrieval}")
    print(f"  Tools        : {response.requires_tools}")
    print(f"  Human Review : {response.requires_human_review}")
    print(f"  Next Module  : {response.handoff.next_module}")
    print(f"  Reason       : {response.reason}")
    signals = response.trace["signals"]
    print(
        f"  Signals      : complexity={signals['complexity_score']:.3f}  "
        f"knowledge={signals['knowledge_need']:.3f}  "
        f"action={signals['action_need']:.3f}  "
        f"risk={signals['risk_score']:.3f}"
    )


def main() -> None:
    samples = load_sample_requests()
    print("\n=== AgentOps Runtime — Strategy Router Examples ===\n")
    for sample in samples:
        req = RouterRequest(**sample["request"])
        resp = route(req)
        print_response_summary(sample, resp)
    print("-" * 70)
    print("\nAll examples completed.")


if __name__ == "__main__":
    main()
