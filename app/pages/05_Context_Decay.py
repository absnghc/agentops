"""
Streamlit page: Context Decay Middleware.

The strategy router decides if retrieval is needed and sets a token budget.
The context decay middleware decides which stored memories deserve that budget.

Run from repo root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from agentops_runtime.core.types import Domain, RiskTolerance
from agentops_runtime.modules.context_decay import filter_context
from agentops_runtime.modules.router import route
from agentops_runtime.modules.router.schemas import RouterRequest

MEMORY_FIXTURE = REPO_ROOT / "examples" / "sample_memory.json"

DEFAULT_QUERY = "What is the refund policy for Acme Corp given the failed API sync?"

st.set_page_config(
    page_title="Context Decay Middleware",
    page_icon="🧠",
    layout="wide",
)

st.title("Context Decay Middleware")
st.caption(
    "The strategy router decides if retrieval is needed and how many tokens it can spend. "
    "The context decay middleware decides which stored memories deserve that token budget."
)

query = st.text_area(
    "Request",
    value=DEFAULT_QUERY,
    height=100,
    placeholder="Ask something that requires retrieval, e.g. a refund or policy question.",
)

col_ctx, col_btn = st.columns([3, 1])
with col_ctx:
    domain = st.selectbox(
        "Use case context",
        options=[d.value for d in Domain],
        index=list(Domain).index(Domain.FINANCE),
        help="Used as a hint. The router can also infer domain from the request.",
    )
with col_btn:
    st.write("")
    st.write("")
    run_button = st.button("Run Pipeline", type="primary", use_container_width=True)

with st.expander("Advanced settings", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_tolerance = st.selectbox(
            "Automation risk tolerance",
            options=[r.value for r in RiskTolerance],
            index=1,
        )
    with col2:
        latency_budget_ms = st.slider(
            "Max latency allowed",
            min_value=200,
            max_value=30_000,
            value=5_000,
            step=100,
            format="%d ms",
        )
    with col3:
        cost_budget_usd = st.slider(
            "Max cost per request",
            min_value=0.001,
            max_value=5.0,
            value=0.50,
            step=0.001,
            format="$%.3f",
        )

    st.divider()
    col4, col5 = st.columns(2)
    with col4:
        half_life_hours = st.slider(
            "Memory half-life (hours)",
            min_value=1,
            max_value=168,
            value=24,
            step=1,
            help="How quickly memory relevance decays with age. Lower = more aggressive decay.",
        )
    with col5:
        memory_context_budget = st.slider(
            "Memory context budget (tokens)",
            min_value=20,
            max_value=500,
            value=120,
            step=10,
            help="Maximum tokens the middleware can select. Capped by the router's token budget.",
        )

if not run_button:
    st.stop()

# --- Run router ---
try:
    req = RouterRequest(
        query=query,
        domain=domain,
        latency_budget_ms=latency_budget_ms,
        cost_budget_usd=cost_budget_usd,
        risk_tolerance=risk_tolerance,
    )
    resp = route(req)
except Exception as exc:
    st.error(f"Router failed: {exc}")
    st.stop()

st.divider()

# Pipeline banner
st.info(
    "**Pipeline:** User Query → Strategy Router → Context Decay Middleware → Final Context"
)

# Router decision
st.subheader("Router Decision")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Strategy", resp.strategy.value.upper())
c2.metric("Model Tier", resp.model_tier.value.upper())
c3.metric("Router Token Budget", f"{resp.token_budget:,}")
c4.metric("Requires Retrieval", "Yes" if resp.requires_retrieval else "No")
st.markdown(f"**Why:** {resp.reason}")

# Gate
if not resp.requires_retrieval:
    st.warning(
        "Router decided retrieval is not needed. Context Decay middleware skipped."
    )
    st.stop()

# --- Run context decay middleware ---
effective_token_budget = min(memory_context_budget, resp.token_budget)

try:
    chunks = json.loads(MEMORY_FIXTURE.read_text())
    result = filter_context(
        query=query,
        chunks=chunks,
        token_budget=effective_token_budget,
        half_life_hours=float(half_life_hours),
    )
except Exception as exc:
    st.error(f"Context decay failed: {exc}")
    st.stop()

st.divider()

# Context decay summary
st.subheader("Context Decay Result")
d1, d2, d3, d4 = st.columns(4)
d1.metric("Effective Budget", f"{effective_token_budget} tokens",
          help="min(memory context budget, router token budget)")
d2.metric("Selected", len(result["selected_chunks"]))
d3.metric("Dropped", len(result["dropped_chunks"]))
d4.metric("Tokens Used", f"{result['total_tokens_used']} / {effective_token_budget}")

if effective_token_budget < resp.token_budget:
    st.caption(
        f"Router token budget: {resp.token_budget:,} tokens  |  "
        f"Memory context budget (slider): {memory_context_budget} tokens  |  "
        f"Effective: {effective_token_budget} tokens"
    )

# Selected memories
st.subheader(f"Selected Memories ({len(result['selected_chunks'])})")
if result["selected_chunks"]:
    rows = [
        {
            "source": c["source"],
            "age_hours": c["age_hours"],
            "final_score": round(c["final_score"], 3),
            "tokens": c["tokens"],
            "reason": c["reason"],
            "content": c["content"],
        }
        for c in result["selected_chunks"]
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
else:
    st.write("No chunks selected.")

# Dropped memories
st.subheader(f"Dropped Memories ({len(result['dropped_chunks'])})")
if result["dropped_chunks"]:
    rows = [
        {
            "source": c["source"],
            "age_hours": c["age_hours"],
            "final_score": round(c["final_score"], 3),
            "tokens": c["tokens"],
            "reason": c["reason"],
            "content": c["content"],
        }
        for c in result["dropped_chunks"]
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
else:
    st.write("No chunks dropped.")

# Score breakdown
st.subheader("Score Breakdown")
all_chunks = result["selected_chunks"] + result["dropped_chunks"]
all_chunks.sort(key=lambda c: c["final_score"], reverse=True)
breakdown = [
    {
        "id": c["id"],
        "source": c["source"],
        "age_hours": c["age_hours"],
        "relevance": round(c["relevance_score"], 3),
        "recency": round(c["recency_score"], 3),
        "importance": round(c["importance_score"], 3),
        "final_score": round(c["final_score"], 3),
        "tokens": c["tokens"],
        "decision": c["decision"],
        "reason": c["reason"],
    }
    for c in all_chunks
]
st.dataframe(breakdown, use_container_width=True, hide_index=True)
