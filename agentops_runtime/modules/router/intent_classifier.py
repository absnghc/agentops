"""
Intent classifier using two orthogonal primary signals:
  - execution_need: does this query ask for something to BE DONE (actions, tools)?
  - knowledge_need: does this query ask for something to BE KNOWN (retrieval)?

Inspired by RouteLLM (Ong et al., UC Berkeley 2024): a single well-calibrated
signal axis outperforms four weakly-correlated proxy scores. Here we use two
orthogonal axes (do / know) plus a risk gate instead of four entangled features.

Key fix over v1: the bigrams self-match bug (unioning _KNOWLEDGE_BIGRAMS into
all_terms then checking membership in all_terms created a permanent 0.55 floor
on knowledge_need for every query regardless of content).
"""
from __future__ import annotations

import re
from typing import Any

from agentops_runtime.core.config import DOMAIN_SENSITIVITY, RISK_TOLERANCE_MULTIPLIER

# ── Vocabulary ────────────────────────────────────────────────────────────────

# Verbs that imply external state changes: tool use, API calls, file ops.
# Kept separate from knowledge verbs — "explain" ≠ "send".
_EXECUTION_VERBS: frozenset[str] = frozenset({
    "send", "create", "update", "delete", "deploy", "migrate", "refactor",
    "build", "write", "fix", "process", "run", "execute", "schedule",
    "generate", "extract", "convert", "book", "cancel", "post", "submit",
    "upload", "download", "provision", "rollback", "transfer", "pay",
    "order", "buy", "install", "configure", "enable", "disable",
    "restart", "start", "stop", "set", "publish", "register", "remove",
    "merge", "split", "copy", "move", "rename", "archive", "backup",
    "flag", "tag", "assign", "approve", "reject", "close", "refund",
})

# Verbs that imply understanding/retrieval — NOT action verbs.
_KNOWLEDGE_VERBS: frozenset[str] = frozenset({
    "summarize", "summarise", "explain", "describe", "define", "compare",
    "analyse", "analyze", "review", "assess", "evaluate", "research",
    "find", "show", "list", "identify", "highlight", "outline",
})

# Only matched against actual query bigrams, never unioned into all_terms.
_KNOWLEDGE_BIGRAMS: frozenset[str] = frozenset({
    "what is", "what are", "how does", "how do", "tell me",
})

# Query targets on large/critical things amplify execution risk.
_SCOPE_AMPLIFIERS: frozenset[str] = frozenset({
    "all", "every", "entire", "whole", "complete", "full",
    "production", "live", "codebase", "repository", "repo",
    "database", "system", "environment", "cluster", "fleet",
})

# Document/artifact nouns signal the query needs external context.
_DOC_NOUNS: frozenset[str] = frozenset({
    "contract", "document", "report", "file", "code", "paper",
    "article", "email", "transcript", "specification", "spec",
    "dataset", "records", "logs", "history", "data", "invoice",
})

# Irreversible operations drive risk independent of domain.
_HIGH_RISK_VERBS: frozenset[str] = frozenset({
    "delete", "remove", "drop", "destroy", "wipe", "purge",
    "deploy", "migrate", "rollback", "provision", "terminate",
    "transfer", "withdraw",  # financial ops with irreversible/high-value consequences
})

# Structural complexity connectors (for constrained vs direct distinction).
_COMPLEXITY_SIGNALS: tuple[str, ...] = (
    " and also ", " but also ", " additionally ", " furthermore ",
    " however ", " on the other hand ", " in addition to ",
    "if ", "when ", "unless ", "provided that ", "assuming ",
)

# Implicit domain inference from query content — used when domain="general".
_DOMAIN_KEYWORDS: dict[str, frozenset[str]] = {
    "legal": frozenset({
        "contract", "clause", "compliance", "gdpr", "liability", "lawsuit",
        "attorney", "court", "regulation", "regulatory", "legal", "law",
        "terms", "agreement", "indemnity", "copyright", "patent",
    }),
    "medical": frozenset({
        "patient", "medication", "dosage", "diagnosis", "treatment",
        "clinical", "drug", "symptom", "therapy", "health", "medical",
        "doctor", "prescription", "surgery", "disease",
    }),
    "finance": frozenset({
        "payment", "transaction", "invoice", "refund", "bank", "financial",
        "credit", "debit", "portfolio", "investment", "revenue", "budget",
        "profit", "loss", "tax", "accounting", "audit", "billing",
    }),
}


# ── Tokeniser ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


# ── Domain inference ──────────────────────────────────────────────────────────

def infer_domain(query: str, declared_domain: str) -> str:
    """Upgrade 'general' to a specific domain when query content warrants it."""
    if declared_domain != "general":
        return declared_domain
    tokens = set(_tokenize(query))
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if tokens & keywords:
            return domain
    return declared_domain


# ── Signal scorers ────────────────────────────────────────────────────────────

def score_execution_need(query: str) -> float:
    """
    Measures whether the query asks for external actions (tool use, state changes).

    Primary signal: imperative framing — first token is an execution verb.
    This is the key insight from RouteLLM: task type (DO vs KNOW) is more
    predictive than token count or connective complexity.
    """
    tokens = _tokenize(query)
    if not tokens:
        return 0.0

    token_set = set(tokens)
    imperative = tokens[0] in _EXECUTION_VERBS
    action_hits = len(token_set & _EXECUTION_VERBS)
    scope_hits = len(token_set & _SCOPE_AMPLIFIERS)

    base = 0.5 if imperative else 0.1
    verb_score = min(action_hits / 3, 0.3)
    scope_score = min(scope_hits / 2, 0.2)

    return round(min(base + verb_score + scope_score, 1.0), 4)


def score_knowledge_need(query: str, domain: str) -> float:
    """
    Measures whether the query needs external knowledge retrieval.

    v1 bug: _KNOWLEDGE_BIGRAMS was unioned into all_terms, making every bigram
    trivially self-match → permanent 0.55 floor on every query.
    Fix: only check bigrams that actually appear in the query.
    """
    tokens = _tokenize(query)
    token_set = set(tokens)
    # Only bigrams present in the actual query — never union the vocabulary set in.
    query_bigrams = {f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)}

    domain_score = DOMAIN_SENSITIVITY.get(domain, 0.1) * 0.4
    knowledge_verb_hits = len(token_set & _KNOWLEDGE_VERBS)
    bigram_hits = len(query_bigrams & _KNOWLEDGE_BIGRAMS)
    verb_score = min((knowledge_verb_hits + bigram_hits) / 2, 0.4)
    doc_ref_score = 0.2 if (token_set & _DOC_NOUNS) else 0.0

    return round(min(domain_score + verb_score + doc_ref_score, 1.0), 4)


def score_complexity(query: str) -> float:
    """Structural complexity: distinguishes constrained from direct when other signals are low."""
    lower = query.lower()
    tokens = _tokenize(query)

    length_score = min(len(tokens) / 300, 1.0) * 0.4
    signal_hits = sum(1 for sig in _COMPLEXITY_SIGNALS if sig in lower)
    signal_score = min(signal_hits / 4, 1.0) * 0.35
    depth_score = min(query.count("?") / 3, 1.0) * 0.25

    return round(min(length_score + signal_score + depth_score, 1.0), 4)


def score_risk(
    query: str,
    domain: str,
    risk_tolerance: str,
    execution_need: float,
) -> float:
    """
    Risk = inferred domain sensitivity + execution consequence + irreversibility.

    All three components are derived from query content, not just the declared
    domain metadata — so "delete the production database" on domain=general
    still scores high risk.
    """
    effective_domain = infer_domain(query, domain)
    domain_risk = DOMAIN_SENSITIVITY.get(effective_domain, 0.1)
    tokens = set(_tokenize(query))
    irreversibility = min(len(tokens & _HIGH_RISK_VERBS) / 2, 0.4)

    raw = (domain_risk * 0.4) + (execution_need * 0.3) + irreversibility
    multiplier = RISK_TOLERANCE_MULTIPLIER.get(risk_tolerance, 1.0)

    return round(min(raw * multiplier, 1.0), 4)


# ── Public entry point ────────────────────────────────────────────────────────

def classify(query: str, domain: str, risk_tolerance: str) -> dict[str, Any]:
    effective_domain = infer_domain(query, domain)
    execution_need = score_execution_need(query)
    return {
        "complexity_score": score_complexity(query),
        "knowledge_need": score_knowledge_need(query, effective_domain),
        "action_need": execution_need,
        "risk_score": score_risk(query, domain, risk_tolerance, execution_need),
        "inferred_domain": effective_domain,
    }
