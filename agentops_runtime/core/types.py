from enum import StrEnum


class Domain(StrEnum):
    GENERAL = "general"
    LEGAL = "legal"
    FINANCE = "finance"
    MEDICAL = "medical"
    SUPPORT = "support"


class RiskTolerance(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Strategy(StrEnum):
    DIRECT = "direct"
    CONSTRAINED = "constrained"
    RETRIEVAL = "retrieval"
    AGENT = "agent"
    ESCALATE = "escalate"


class ModelTier(StrEnum):
    SMALL = "small"
    REASONING = "reasoning"
    FRONTIER = "frontier"
