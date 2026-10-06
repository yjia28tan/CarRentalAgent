# CarRentalAgent

## Orchestration: why not LangChain/LangGraph?

The orchestrator is a deliberately small, explicit tool-calling loop in `src/agent/orchestrator.py`. We chose this over a framework because:

- **Full visibility.** Tool selection, argument validation, per-tool timeouts, the step limit and error handling are all plain code we can read, log and test.
- **Security stays in our code.** `customer_id` is injected from the verified JWT and is never a tool argument; writes are two-phase and are only executed when the customer presses Confirm, via the `/actions` endpoint.
- **Fewer dependencies** and less version churn.
- **Deterministic tests** through the `FakeLLM` gateway.

If the team standardises on LangGraph, the loop maps across directly, and this mapping is implemented in `src/agent/langgraph_orchestrator.py`:

| Ours | LangGraph |
|---|---|
| agent step | agent node |
| `execute_tool` | tool node |
| "has tool calls?" | conditional edge |
| `AGENT_MAX_STEPS` | recursion limit (the step limit is also enforced in the agent node) |
| Redis trip context | graph state (Redis/Mongo remain the store; no checkpointer) |
| pending action + Confirm button | still the `/actions` endpoint; no `interrupt()`, so the write keeps its JWT ownership checks |
| `RequestContext` | runtime config (never tool arguments) |

The tools, the LLM gateway, storage and the mock services are shared by both implementations.

### Switching to LangGraph

Set `AGENT_ORCHESTRATOR=langgraph` (in `secrets.env` or `src/config/config.yaml`); the default is `custom`. Replies, cards, sources, tool trace, fallbacks and audit events are the same either way. The API, scenario and two-phase-write tests run against both implementations.