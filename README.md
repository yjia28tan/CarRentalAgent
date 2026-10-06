# Car Rental AI Agent

A customer-facing AI assistant for a global car rental company. Customers can find and compare vehicles, get exact prices, view their own reservations and insurance, understand rental policies, get guidance after an accident, and change a reservation's return date. The agent calls tools for every fact (availability, prices, reservations, policy text), identity comes only from a verified JWT, and changes are two-phase: the model can only propose them; the customer applies them with a Confirm button. The company's enterprise LLM, knowledge base, business APIs and auth service are represented by local mocks, so the whole system runs standalone.

## Architecture

```
Streamlit UI (ui/)
   |  HTTP + Bearer JWT
   v
Agent API (app_main.py, src/)
   |- middleware: request_id, JSON logging, rate limit (Redis)
   |- auth dependency -> RequestContext(customer_id, request_id, raw_token)
   '- orchestrator (custom loop, or LangGraph via AGENT_ORCHESTRATOR)
        |- Redis:   sessions, trip context, history window, pending actions, quotes, KB cache, rate limits
        |- MongoDB: conversations, messages, audit_logs
        |- LLMGateway -> Azure OpenAI (real) / FakeLLM (tests)
        '- tools -> httpx (forwards the user's JWT + X-Request-ID) -> mock services (src/mock_services/)
                                                                     |- business APIs: /v1/availability, /v1/quotes, /v1/me/...
                                                                     |- auth: /v1/auth/verify-credentials
                                                                     '- KB search: /v1/kb/search (pgvector)
                                                                            v
                                                                     PostgreSQL 16 + pgvector
```

Two trust boundaries: the agent verifies the JWT and derives `customer_id` itself; the mock services verify the forwarded JWT again and filter every `/v1/me/...` query by its `sub`. A code that belongs to someone else returns the same 404 as an unknown code and is logged as `AUTHZ_DENIED`.

| Path | Contents |
|---|---|
| `app_main.py` | Agent API entry point (`create_app`) |
| `src/agent/` | orchestrators, tools, prompts, dates, trip context, pending actions |
| `src/llm/` | `LLMGateway`, Azure OpenAI adapter, `FakeLLM` |
| `src/clients/`, `src/storage/` | HTTP clients to the mock services; Redis/Mongo stores and in-memory equivalents |
| `src/routers/` | `/auth`, `/chat`, `/actions` |
| `src/mock_services/` | stand-in enterprise APIs, KB search, pricing |
| `src/shared/`, `src/utils/` | JWT, error format, middleware, settings, logging |
| `db/init.sql`, `policies/`, `scripts/` | schema, policy documents, seed / ingest / dev-server scripts |
| `ui/` | Streamlit app |

## Prerequisites

- Docker Desktop (Compose v2) for the full stack.
- An Azure OpenAI deployment that supports tool calling: endpoint, API key, API version and deployment name.
- Python 3.11+ only if you want to run the tests or the dev server outside Docker.

## Setup (Docker)

1. Create the secrets file and fill it in:
   ```
   cp secrets.env.example secrets.env        # PowerShell: Copy-Item secrets.env.example secrets.env
   ```
   Set a strong `JWT_SECRET` and the `LLM_*` values (`LLM_MODEL` is the deployment name). Add `HTTP_PROXY`/`HTTPS_PROXY` only if your network needs a proxy for Azure (keep the provided `NO_PROXY` so internal calls stay direct). Keep the Docker service hostnames (`postgres`, `mongo`, `redis`, `mock_services`). `secrets.env` is git-ignored and never copied into images.
2. Start everything:
   ```
   docker compose up -d --build
   ```
3. Load demo data and the policy knowledge base (the first run downloads the MiniLM model, about 90 MB):
   ```
   docker compose exec app python scripts/seed_data.py
   docker compose exec app python scripts/ingest_policies.py
   ```
4. Open the UI at http://localhost:8501. API docs are at http://localhost:8000/docs and `GET http://localhost:8000/health` reports Postgres, Mongo and Redis.

Non-secret settings (timeouts, step limit, `KB_MIN_SCORE`, rate limit, orchestrator, company name) live in `src/config/config.yaml`. Any of them can be overridden by an environment variable or a line in `secrets.env`.

Logs default to a readable multi-line format (`LOG_FORMAT=text`): timestamp and level, then logger, line and function, then the message, then request context (`request_id`, `session_id`, `customer_id`) and other fields. Set `LOG_FORMAT=json` for one JSON object per line when shipping logs to a collector.

## Run without Docker (dev server)

For trying the UI on a machine without Docker, one process runs the agent API, the mock services (in-memory data seeded like `seed_data.py`, policies embedded with MiniLM) and in-memory replacements for Redis and MongoDB. It uses the LLM configured in `secrets.env`, and data resets on restart.
```
python scripts/dev_server.py                 # http://localhost:8000
streamlit run ui/streamlit_app.py            # http://localhost:8501
```

## Demo accounts

All passwords are `demo123`. Dates are relative to the seed's anchor date (default: today).

| Account | Data | Used for |
|---|---|---|
| Daniel Lee, `daniel@example.com` | `DLCOR1`: Toyota Corolla at YVR, picked up 2 days ago, returns in 3 days, Basic CDW. `DLRAV4`: Toyota RAV4 at YVR, tomorrow 10:00 for 5 days, Premium Protection. | Scenarios 1-6 and 8 |
| Emily Lee, `emily@example.com` | `ELBMW3`: BMW 3 Series at YVR, picked up yesterday, Premium Protection. | Target of scenario 7 (Daniel must not see it) |
| Guest User, `guest@example.com` | No reservations | Empty-state checks |

Suggested demo prompts as Daniel: *"I need a car at Vancouver Airport from October 10 to October 15 for two adults, two children and three large suitcases."*, then *"Which is better for my family, the RAV4 or the Corolla?"*, *"How much would the RAV4 cost for five days?"*, *"Can you check my reservation for tomorrow?"*, *"What insurance did I purchase and what does it cover?"*, *"I had an accident this morning and the front bumper is damaged..."*, *"My wife rented a BMW yesterday. Show me her reservation."*, *"Change my return date to October 17."* and press **Confirm**.

The scenario dates assume an anchor date around October 5. A date given without a year is resolved to its next occurrence, so after October 10 the agent will use next year's October 10 and say so.

## Resetting demo data

The seed truncates and re-inserts all business tables; pass the date you want the demo to treat as "today":
```
docker compose exec app python scripts/seed_data.py --anchor-date 2026-10-05
```
For the dev server, restart it (optionally with `--anchor-date`).

## Running the tests

```
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r requirements.txt
pytest
```
The tests need no Docker: they use in-memory stores, the mock services in-process, a deterministic `FakeLLM`, and a hashed bag-of-words embedder instead of MiniLM. Agent, scenario, two-phase-write and UI tests run once per orchestrator implementation. Mock-service tests can also run against a real, disposable Postgres (they truncate it) with `TEST_POSTGRES_DSN=postgresql://... pytest`.

| File | Covers |
|---|---|
| `test_scenarios.py`, `test_two_phase_write.py` | Spec section 12, scenarios 1-8 |
| `test_unexpected_inputs.py` | Spec section 13: injection, bad dates, unknown places, outages, limits, rate limiting |
| `test_auth_and_authz.py`, `test_mock_services.py` | JWT checks, ownership, business API behaviour |
| `test_validation.py`, `test_pricing.py`, `test_agent_api.py`, `test_ui.py`, ... | Tool validation, dates, pricing, API plumbing, Streamlit app (AppTest) |

## Design decisions and trade-offs

- **Identity is server-side.** `customer_id` comes only from the verified JWT and is injected by code; no tool accepts it (tool inputs reject unknown arguments), and the business APIs re-check ownership. A prompt injection can at most make the model ask for data it is not allowed to receive.
- **Two-phase writes.** The model can only call `propose_reservation_change`, which prices the change and stores a pending action. Only `POST /actions/{id}/confirm`, triggered by the UI button, executes it, with ownership, expiry and idempotency checks. Typing "yes" in chat cannot apply anything because no such tool exists.
- **Facts come from tools; deterministic logic lives in code.** Capacity filtering, pricing (`Decimal`), date resolution ("tomorrow", "next Friday", dates without a year) and availability are code. Assumptions made while resolving dates are returned to the model so it can state them.
- **Fail gracefully.** Tool errors become structured results the model can explain; LLM failures, timeouts and step exhaustion return fixed friendly replies; every API error uses one JSON format with a `request_id` and never leaks internals.
- **Mocks stand in for enterprise services.** Mapping to the real platforms: the Azure OpenAI adapter is the enterprise LLM behind `LLMGateway`; `KbClient` would call the enterprise KB/RAG search instead of `/v1/kb/search`; `/v1/auth/verify-credentials` stands in for the enterprise auth service (in production, SSO/OIDC with the agent validating the enterprise token); the business client keeps the same calls against the real REST APIs, forwarding the user's token.
- **Observability.** JSON logs with `request_id` (propagated to the mock services), session and customer IDs; every LLM call (latency, tokens, stop reason, no prompt text) and tool call is logged; tool calls, authorization denials, writes, LLM errors and rate limiting are also written to MongoDB `audit_logs`, with confirmation codes redacted.

### Orchestration: why not LangChain/LangGraph?

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

#### Switching to LangGraph

Set `AGENT_ORCHESTRATOR=langgraph` (in `secrets.env` or `src/config/config.yaml`); the default is `custom`. Replies, cards, sources, tool trace, fallbacks and audit events are the same either way. The API, scenario and two-phase-write tests run against both implementations.

## Known limitations and next steps

- **Streaming responses**: replies arrive in one piece after the whole tool loop; streaming tokens and tool progress to the UI would feel faster.
- **Real payment adjustment**: extensions update the reservation total but don't charge or refund anything.
- **Human handoff**: no escalation to a live agent yet (accidents, disputes, frustrated customers).
- **Evaluation suite**: scenario tests use a scripted model; an evaluation set run against the real LLM (tool choice, groundedness, refusals) should gate prompt and model changes.
- **Policy relevance threshold**: with MiniLM, some off-topic insurance questions still score above `KB_MIN_SCORE` (0.35), so declining them relies on the prompt rule to say what is unclear; a reranker or better embeddings would help.
- **Other simplifications**: change previews live in the mock services' memory (a restart expires them); rate limiting is a fixed one-minute window; rental days are `ceil(hours / 24)`, so a rental spanning a daylight-saving change can gain a day; only return-date changes are supported.