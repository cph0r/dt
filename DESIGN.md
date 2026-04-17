# Agentic Customer Support System

## 1. High-Level Design

### Architecture Diagram

```text
Client
  -> FastAPI API Layer
    -> Agent Planner/Executor
      -> Prompt Registry
      -> LLM Wrapper (LiteLLM-style routing, retry, timeout)
      -> Tool Layer
          -> search_docs(query)
              -> Retriever
                  -> Vector Store
                  -> Metadata Filter
                  -> Optional Reranker
          -> create_ticket(issue)
              -> Ticketing System / CRM
      -> Evaluator (LLM-as-judge + heuristic scoring)
      -> Observability (structured JSON logs)
      -> Conversation Store
```

### Components

- API layer: exposes `/health` and `/v1/chat`.
- Agent: planner-executor loop with max steps, confidence threshold, and fallback.
- Tool layer: stateless tool interface with an allow-list.
- Retriever: embedding search plus metadata filtering.
- LLM wrapper: provider abstraction with retry and timeout.
- Prompt registry: versioned prompts for A/B testing.
- Evaluation module: offline scoring and feedback capture.
- Storage: vector DB plus chunk metadata.

### Data Flow

1. Client sends a support question and conversation id.
2. API forwards the request to the agent.
3. Agent loads the selected prompt version and conversation context.
4. Agent asks the LLM to decide whether to answer, search, or escalate.
5. If search is selected, the retriever returns top-k chunks with source and section metadata.
6. If confidence is high enough, the agent synthesizes an answer grounded in retrieved evidence.
7. If confidence is low or evidence is weak, the ticket tool creates a support ticket.
8. The agent logs query, prompt version, tool calls, latency, and errors.
9. The evaluator scores the response and stores feedback for iteration.

### Key Tradeoffs

- Deterministic control flow vs fully autonomous reasoning: this design favors reliability.
- Lightweight in-memory storage vs distributed persistence: feasible for implementation, but replaceable.
- Simple embedding baseline vs managed vector search: easier to ship first, easier to swap later.
- Strict confidence threshold vs answer coverage: fewer hallucinations, more escalations.

## 2. Low-Level Design

### Agent

Responsibilities:
- Run the planner-executor loop.
- Enforce max steps and retries.
- Apply fallback when confidence is low.
- Persist conversation context.

Method shape:
- `run(conversation_id: str, query: str, prompt_version: str) -> ChatResponse`

Extension points:
- custom stopping conditions
- tool policy enforcement
- reranking/evidence selection

### Tool Interface

Responsibilities:
- Stateless execution contract.
- Safe, allow-listed tool invocation.

Method shape:
- `execute(**kwargs) -> ToolResult`

Extension points:
- search docs
- ticket creation
- account lookup
- order status checks

### LLMClient

Responsibilities:
- Provider abstraction.
- Retry and timeout.
- Parse structured output.

Method shape:
- `complete(messages: list[dict[str, str]], temperature: float = 0.0) -> dict[str, Any]`

Extension points:
- LiteLLM backend
- fallback provider
- JSON schema validation

### Retriever

Responsibilities:
- Embed query and chunks.
- Perform similarity search.
- Apply metadata filters.
- Optionally rerank.

Method shape:
- `index(chunks)`
- `search(query, top_k, metadata_filter, rerank)`

Extension points:
- pgvector/Pinecone backends
- hybrid lexical + dense retrieval
- reranker model

### ChunkingPipeline

Responsibilities:
- Parse structure-aware blocks.
- Preserve headings and sections.
- Attach keywords and metadata.

Method shape:
- `chunk(source: str, text: str) -> list[DocumentChunk]`

Extension points:
- markdown parser
- HTML parser
- PDF/docx ingestion

### PromptRegistry

Responsibilities:
- Version prompts.
- Support A/B testing.
- Log prompt version used.

Method shape:
- `get_prompt(version: str) -> PromptArtifact`

### Evaluator

Responsibilities:
- LLM-as-judge scoring.
- Lightweight heuristic scoring.
- Feedback capture.

Method shape:
- `judge(question, answer, evidence) -> EvaluationResult`

### Config Manager

Responsibilities:
- Centralized runtime config.
- Environment override support.
- Safe defaults.

## 3. Project Structure

```text
app/
├── main.py
├── api/
├── services/
│     ├── agent.py
│     ├── llm.py
│     ├── retriever.py
├── tools/
├── ingestion/
├── evaluation/
├── prompts/
├── config/
├── models/
├── utils/
└── tests/
```

Purpose:
- `main.py`: app bootstrap and dependency wiring.
- `api/`: HTTP routing.
- `services/`: agent, LLM, retriever orchestration.
- `tools/`: stateless external actions.
- `ingestion/`: document parsing and chunking.
- `evaluation/`: scoring and feedback.
- `prompts/`: versioned prompt assets.
- `config/`: settings and environment config.
- `models/`: request/response and domain schemas.
- `utils/`: logging and shared helpers.
- `tests/`: unit and integration coverage.

## 4. Core Implementation

Implemented in the repository:
- FastAPI endpoint at `/v1/chat`.
- Agent loop with planner -> tool -> observation -> repeat.
- `search_docs` and `create_ticket` tools.
- LLM wrapper with timeout/retry/provider adapter.
- Retriever with similarity search and metadata filtering.
- Structure-aware chunking.
- Structured JSON logging.

## 5. Prompt Versioning System

- Prompts are stored in `PROMPTS = {"v1": ..., "v2": ...}`.
- Requests can select a version explicitly.
- The chosen version is logged and returned in the response.
- The system can A/B test versions by routing traffic per request.

## 6. RAG Design

### Chunking Strategy

- Structure-aware parsing keeps headings and content together.
- Semantic chunking can be added later by splitting on topical similarity.

### Metadata Schema

- `source`
- `section`
- `keywords`
- `chunk_id`

### Retrieval Pipeline

1. Embed query.
2. Apply metadata filter.
3. Retrieve top-k nearest chunks.
4. Optionally rerank results.
5. Pass evidence to the agent.

Tradeoffs:
- Strong metadata improves precision but can reduce recall if too strict.
- Reranking improves quality but adds latency and cost.

## 7. Agent Design

- Planner-executor loop with bounded iterations.
- Tool selection via model output plus allow-list.
- Failure handling via retries and fallback ticket creation.
- Loop termination when confidence is sufficient or when escalation is required.

## 8. Observability & Logging

Logged fields:
- query
- prompt version
- tool calls
- latency
- errors

Format:
- structured JSON logs for ingestion into ELK, Datadog, or Cloud Logging.

## 9. Reliability & Safety

- Retries for transient LLM failures.
- Timeout enforcement.
- Maximum agent iterations.
- Fallback responses and ticket escalation.
- Prompt injection protection by ignoring conflicting retrieved instructions.
- Tool allow-listing through the registry.

## 10. Evaluation & Feedback Loop

- LLM-as-judge scoring for groundedness and completeness.
- Thumbs up/down feedback capture can be stored per interaction.
- Offline metrics: answer accuracy, retrieval precision/recall, escalation rate, and ticket deflection.
- A/B testing compares prompt versions and routing policies.

## 11. Scaling Strategy

- Stateless API layer.
- Async workers and queue for ingestion and evaluation.
- Redis for cache and session state.
- pgvector or Pinecone for vector storage.
- Load balancing across API replicas.
- Cost optimization via model routing and caching.

## 12. Testing Strategy

- Unit tests for tools, retriever, and chunking.
- Integration tests for agent decision-making and fallback.
- Load testing is conceptual here but should cover concurrency, latency, and ticketing backpressure.

## 13. Tradeoffs & Extensions

- FAISS: simple and fast local index.
- pgvector: production-friendly SQL-backed vector search.
- Custom agent: more control and less framework overhead than LangChain.
- LiteLLM: provider routing and portability.
- DSPy: future optimization for prompt and policy tuning.
- Reranking strategies: cross-encoder rerankers or LLM-based reranking.
- Multi-modal extension: add OCR, image understanding, and document screenshots later.

## Run Notes

- Install dependencies with `pip install -e .`.
- Run the app with `uvicorn app.main:app --reload`.
- Run tests with `pytest`.
