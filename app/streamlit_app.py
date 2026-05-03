"""
Streamlit UI for the AgentOps Runtime Strategy Router.

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st

from agentops_runtime.core.types import Domain, RiskTolerance
from agentops_runtime.modules.router import route
from agentops_runtime.modules.router.schemas import RouterRequest


DEFAULT_QUERY = "What is an AI agent?"


def format_bool(value: bool) -> str:
    return "Yes" if value else "No"


st.set_page_config(
    page_title="AgentOps Strategy Router",
    page_icon="🧭",
    layout="wide",
)

st.title("AgentOps Strategy Router")
st.caption("Routes a request to the cheapest safe execution path.")

query = st.text_area(
    "Request",
    value=DEFAULT_QUERY,
    height=120,
    placeholder="Ask something like: Summarize this contract and flag risky clauses.",
)

col_context, col_button = st.columns([3, 1])

with col_context:
    domain = st.selectbox(
        "Use case context",
        options=[d.value for d in Domain],
        index=0,
        help="Used as a hint. The router can also infer domain from the request.",
    )

with col_button:
    st.write("")
    st.write("")
    run_button = st.button("Route request", type="primary", use_container_width=True)

with st.expander("Advanced settings", expanded=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        risk_tolerance = st.selectbox(
            "Automation risk tolerance",
            options=[r.value for r in RiskTolerance],
            index=1,
            help="Lower tolerance escalates more often. Higher tolerance automates more often.",
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

if not run_button:
    st.stop()

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
    st.error(f"Routing failed: {exc}")
    st.stop()

st.divider()

# Summary
st.subheader("Routing Decision")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Strategy", resp.strategy.value.upper())
col2.metric("Model Tier", resp.model_tier.value.upper())
col3.metric("Token Budget", f"{resp.token_budget:,}")
col4.metric("Est. Cost", f"${resp.estimated_cost_usd:.6f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Est. Latency", f"{resp.estimated_latency_ms} ms")
col6.metric("Retrieval", format_bool(resp.requires_retrieval))
col7.metric("Tools", format_bool(resp.requires_tools))
col8.metric("Human Review", format_bool(resp.requires_human_review))

st.markdown(f"**Why:** {resp.reason}")

# Main details
tab_signals, tab_handoff, tab_trace = st.tabs(
    ["Signals", "Handoff", "Trace"]
)

with tab_signals:
    signals = resp.trace.get("signals", {})

    cols = st.columns(4)
    numeric_signals = [
        ("Complexity", signals.get("complexity_score")),
        ("Knowledge Need", signals.get("knowledge_need")),
        ("Action Need", signals.get("action_need")),
        ("Risk", signals.get("risk_score")),
    ]

    for col, (label, value) in zip(cols, numeric_signals):
        if isinstance(value, (int, float)):
            col.metric(label, f"{value:.3f}")
            col.progress(min(max(float(value), 0.0), 1.0))

    st.write("**Inferred domain:**", signals.get("inferred_domain", "unknown"))

with tab_handoff:
    st.write("**Next module:**", resp.handoff.next_module)
    st.write("**Allowed tools:**", resp.handoff.allowed_tools or ["(none)"])
    st.write("**Constraints:**")
    st.json(resp.handoff.constraints)

with tab_trace:
    st.json(resp.trace)