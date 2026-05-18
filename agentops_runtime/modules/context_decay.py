"""
Context Decay Middleware.

Scores stored memory chunks by relevance, recency, and importance, then
packs the highest-scoring chunks into the available token budget.
"""
from __future__ import annotations

import math
import re

_STOPWORDS = {
    "a", "an", "and", "are", "as", "be", "for", "has", "have",
    "if", "in", "is", "it", "not", "of", "on", "or", "our",
    "that", "the", "this", "to", "was", "we", "with",
}

_DEFAULT_WEIGHTS = {"relevance": 0.55, "recency": 0.25, "importance": 0.20}


def _stem(word: str) -> str:
    if len(word) >= 6 and word.endswith("ing"):
        return word[:-3]
    if len(word) >= 5 and word.endswith("ed"):
        return word[:-2]
    if len(word) >= 5 and word.endswith("ly"):
        return word[:-2]
    if len(word) >= 4 and word.endswith("s"):
        return word[:-1]
    return word


def _tokenize(text: str) -> set[str]:
    raw = re.split(r"\W+", text.lower())
    return {_stem(t) for t in raw if t and t not in _STOPWORDS}


def _score_relevance(query_stems: set[str], content: str) -> float:
    if not query_stems:
        return 0.0
    chunk_stems = _tokenize(content)
    overlap = len(query_stems & chunk_stems)
    return min(overlap / len(query_stems), 1.0)


def _score_recency(age_hours: float, half_life_hours: float) -> float:
    return math.pow(0.5, age_hours / half_life_hours)


def _assign_reason(
    decision: str,
    rel: float,
    rec: float,
    imp: float,
    final: float,
    tokens: int,
    remaining: int,
) -> str:
    if decision == "selected":
        if rel >= 0.3 and rec >= 0.3:
            return "Selected: high relevance and recent enough to fit the token budget."
        if rec < 0.3 and imp >= 0.7:
            return "Selected: old, but high importance kept it alive."
        return "Selected: fits budget with moderate scores."
    # Dropped — score-based reasons take priority.
    if rel >= 0.3 and rec < 0.15:
        return "Dropped: high relevance, but stale under current half-life."
    if rec >= 0.5 and rel < 0.1 and imp < 0.3:
        return "Dropped: recent, but low relevance and low importance."
    if final >= 0.4 and tokens > remaining:
        return "Dropped: would exceed token budget."
    return "Dropped: lower ranked and did not fit the remaining budget."


def filter_context(
    query: str,
    chunks: list[dict],
    token_budget: int,
    half_life_hours: float = 24.0,
    weights: dict | None = None,
) -> dict:
    """Score chunks, rank by final score, and pack into token_budget.

    Returns selected and dropped chunks with full score breakdowns.
    """
    w = {**_DEFAULT_WEIGHTS, **(weights or {})}
    query_stems = _tokenize(query)

    scored: list[dict] = []
    for chunk in chunks:
        rel = _score_relevance(query_stems, chunk["content"])
        rec = _score_recency(chunk["age_hours"], half_life_hours)
        imp = float(chunk["importance"])
        final = w["relevance"] * rel + w["recency"] * rec + w["importance"] * imp
        scored.append({**chunk, "relevance_score": rel, "recency_score": rec,
                       "importance_score": imp, "final_score": final})

    scored.sort(key=lambda c: c["final_score"], reverse=True)

    selected: list[dict] = []
    dropped: list[dict] = []
    remaining = token_budget

    for c in scored:
        tokens = c["tokens"]
        remaining_before = remaining
        if remaining >= tokens:
            decision = "selected"
            remaining -= tokens
        else:
            decision = "dropped"
        reason = _assign_reason(
            decision, c["relevance_score"], c["recency_score"],
            c["importance_score"], c["final_score"], tokens, remaining_before,
        )
        out = {**c, "decision": decision, "reason": reason}
        (selected if decision == "selected" else dropped).append(out)

    return {
        "selected_chunks": selected,
        "dropped_chunks": dropped,
        "total_tokens_used": token_budget - remaining,
        "token_budget": token_budget,
        "evaluated_chunk_count": len(scored),
    }
