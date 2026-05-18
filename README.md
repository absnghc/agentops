**AgentOps Runtime**

A minimal system for routing AI requests to the cheapest safe execution path.

**The idea** - Most AI systems default to the strongest model for every request.

**Strategy Router** A zero-LLM-call decision engine that selects the cheapest safe execution path for every user request.
This project explores a different approach: decide how much intelligence a request actually needs.

What this does - Given a request, the router selects one of five execution strategies:
direct —> cheap model, fast response
constrained —> stronger model with limited reasoning
retrieval —> fetch context before answering
agent —> tools and multi-step execution
escalate —> human review or refusal

**How it works**

The router uses a small set of signals:

- task framing (does the query ask for something to be done or explained)
- implicit domain inference (legal, finance, code, etc.)
- knowledge need (does it require external context)
- risk (based on consequences and domain)
- budget (cost and latency constraints)

These signals determine both:

- the execution path
- how much compute to allocate (model + token budget)

**Run locally**
git clone https://github.com/absnghc/agentops.git
cd agentops
pip install -r requirements.txt
streamlit run app/streamlit_app.py

**Example queries**

Try these in the UI:

What is an AI agent?
Summarize this 20-page contract and flag risky clauses.
Send a refund email and process a $50 refund.
Refactor this repo from REST to GraphQL.
Delete the production database.

Example output
{
  "strategy": "retrieval",
  "model_tier": "reasoning",
  "token_budget": 1500,
  "estimated_cost_usd": 0.0225
}

**Architecture**

```
RouterRequest
    → IntentClassifier  (complexity, knowledge, action, risk scores)
    → BudgetOptimizer   (model tier + token budget within cost/latency budgets)
    → StrategySelector  (direct / constrained / retrieval / agent / escalate)
    → HandoffManager    (next_module, allowed_tools, constraints)
    → RouterTrace       (full decision log)
    → RouterResponse
```


**Week 5: Context Decay Middleware**

Week 4's router decides whether retrieval is needed and how many tokens it can spend.
Week 5 decides which stored memories deserve that token budget.

This demo uses a JSON memory fixture and transparent lexical scoring instead of a
vector DB so the retrieval policy is easy to inspect.

```
RouterResponse (requires_retrieval=true, token_budget=N)
    → ContextDecayMiddleware (relevance × recency × importance → ranked pack)
    → Final Context (selected chunks within effective token budget)
```

**Notes**

- This is a deterministic, explainable router.
- In production systems, this layer is typically extended with:

    semantic routing (embeddings)
    learned routing models
    policy and safety systems
    feedback loops from real usage

**Context Decay Middleware**
  
  When the router selects the retrieval strategy, the context decay middleware
  decides which stored memories deserve the token budget. It scores each chunk
  by lexical relevance, recency decay, and importance — then packs the
  highest-scoring chunks into the available tokens.

  RouterResponse (requires_retrieval=true, token_budget=N)
      → ContextDecayMiddleware (relevance × recency × importance → ranked pack)
      → Final Context (selected chunks within token budget)


**Project Structure**

```
agentops_runtime/
    core/         — shared types, config, tracing utilities
    modules/
      router/          — Strategy Router
      context_decay.py — Context Decay Middleware
  examples/       — runnable demonstrations and memory fixtures
  tests/          — pytest unit tests
  app/            — Streamlit UI
  docs/           — design documentation
```

