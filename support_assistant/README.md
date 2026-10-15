# Module 3 — Support Assistant (`/support_assistant`)

A grounded RAG support assistant for Zepto's policies: local embeddings (`all-MiniLM-L6-v2`) in ChromaDB,
a LangGraph intent router, a Pydantic-enforced JSON answer schema, and a FastAPI `POST /ask` endpoint,
packaged with a Dockerfile.

**The graded baseline is fully offline.** With `MOCK_LLM` unset (or `MOCK_LLM=1`), no LLM provider is
contacted, no signup or API key is needed, and every answer is deterministic. The optional real-LLM path
(`MOCK_LLM=0`, Groq free tier) is implemented but not needed for anything below.

## Files

| File | Role |
|---|---|
| `docs/doc_01.txt … doc_08.txt` | The 8-document Zepto policy corpus, copied verbatim from the brief |
| `ingest.py` | **Ingestion + embedding + vector store**: `load_documents`, `chunk_documents`, `embed`, `build_index`, `retrieve` |
| `prompts.py` | Structured prompt templates (role, context, task, format, length, negative constraints, few-shot examples) used only when `MOCK_LLM=0` |
| `schemas.py` | Pydantic models: `AskRequest {query}` and `AskResponse {answer, sources, confidence}` |
| `graph.py` | LangGraph `StateGraph`: `classify_intent` → conditional edge → `retrieve_and_answer` / `direct_answer`, the `MOCK_LLM` toggle, and the retry-on-invalid-JSON logic |
| `main.py` | FastAPI app (`POST /ask`, `GET /health`); builds the index on startup if it is empty |
| `demo.py` | Runs 8 example queries through the graph and checks routing, retrieval and output format |
| `test_graph.py` | pytest: mock routing, no-network guarantee, and the retry / give-up behaviour of the real-LLM path (using a fake LLM) |
| `Dockerfile`, `.dockerignore` | Container serving `/ask` on port 7860 |

## Run locally

```bash
# from the repository root
python -m venv .venv && source .venv/bin/activate
pip install -r support_assistant/requirements.txt
cd support_assistant

python ingest.py                         # embed the 8 docs into ChromaDB (downloads the 90 MB MiniLM model once)
python demo.py                           # 8/8 routing + retrieval checks (mock mode)
python -m pytest -q test_graph.py        # 4 passed
uvicorn main:app --port 7860             # serve the API (MOCK_LLM defaults to mock)
```

*macOS note:* if the python.org build of Python fails the first model download with `CERTIFICATE_VERIFY_FAILED`,
run `export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")` first. After that one download,
`HF_HUB_OFFLINE=1` lets everything run with no network at all.

## Example calls (recorded with `MOCK_LLM` left at its default)

Server started with `uvicorn main:app --port 7860` and **no** `MOCK_LLM` variable set
(`GET /health` → `{"status":"ok","mock_llm":true}`).

**1. Policy question → `classify_intent` → `retrieve_and_answer`** (contains the keyword "delivery")

```bash
curl -s -X POST localhost:7860/ask -H 'Content-Type: application/json' \
     -d '{"query": "What is the delivery fee for an order below INR 149?"}'
```

```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del...","sources":["doc_01_chunk_0","doc_05_chunk_0","doc_07_chunk_0"],"confidence":1.0}
```

The top retrieved chunk is `doc_01` (Delivery Policy, cosine similarity 0.506), which is the document that
states the INR 149 threshold and the INR 25 fee.

**2. General question → `classify_intent` → `direct_answer`** (no policy keyword, so no retrieval)

```bash
curl -s -X POST localhost:7860/ask -H 'Content-Type: application/json' \
     -d '{"query": "What is the capital of France?"}'
```

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

**3. Invalid request (Pydantic request validation)** — `{"query": ""}` → HTTP 422
`{"detail":[{"type":"string_too_short","loc":["body","query"],"msg":"String should have at least 1 character",...}]}`

### Retrieval quality (from `python demo.py`, mock mode)

| Query | Route | Top-1 chunk (cosine sim) | Correct doc? |
|---|---|---|---|
| What is the delivery fee for an order below INR 149? | retrieve_and_answer | doc_01 Delivery Policy (0.506) | ✅ |
| How long do refunds take to reach my bank account? | retrieve_and_answer | doc_02 Returns & Refunds (0.480) | ✅ |
| How much does the Zepto Pass+ membership cost? | retrieve_and_answer | doc_03 Membership Tiers (0.732) | ✅ |
| Can I cancel my order after it has been packed? | retrieve_and_answer | doc_05 Order Cancellation (0.677) | ✅ |
| Can a gift card be combined with another gift card? | retrieve_and_answer | doc_07 Gift Cards (0.523) | ✅ |
| What are your customer support hours? | retrieve_and_answer | doc_08 Customer Support Hours (0.493) | ✅ |
| What is the capital of France? | direct_answer | — | — |
| Tell me a joke about cats. | direct_answer | — | — |

## Docker

```bash
cd support_assistant
docker build -t zepto-support .
docker run --rm -p 7860:7860 zepto-support
# in another terminal:
curl -s -X POST localhost:7860/ask -H 'Content-Type: application/json' -d '{"query": "Can I cancel after packing?"}'
```

**Verified locally** (Docker 29 via Colima on Apple Silicon). `docker build` succeeded, producing a 2.7 GB image.
`docker run -p 7860:7860 zepto-support` served `POST /ask` with `MOCK_LLM` at its default:

```bash
$ curl -s -X POST localhost:7860/ask -H 'Content-Type: application/json' -d '{"query": "Can I cancel my order after it has been packed?"}'
{"answer":"Based on the retrieved context: Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been packed, it can no longer be...","sources":["doc_05_chunk_0","doc_02_chunk_0","doc_06_chunk_0"],"confidence":1.0}

$ curl -s -X POST localhost:7860/ask -H 'Content-Type: application/json' -d '{"query": "What is the capital of France?"}'
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

The container also answers correctly when started with `--network none`, which confirms it needs no network
access at runtime.

The image installs CPU-only PyTorch (about 2 GB smaller than the default CUDA wheel). At build time it runs
`python ingest.py`, so the MiniLM model and the ChromaDB index are baked into the image. It then sets
`HF_HUB_OFFLINE=1` and `MOCK_LLM=1`, so the running container needs no network access. It runs as a non-root
user (uid 1000, as Hugging Face Spaces expects) and starts
`uvicorn main:app --host 0.0.0.0 --port 7860`. The optional Hugging Face Spaces deployment was not attempted.

## Architecture: ingestion → embedding → retrieval → generation

```
 docs/doc_01..08.txt
        │  ingest.load_documents()
        ▼
 ┌──────────────┐  ingest.chunk_documents()   1 chunk per doc, id "doc_0X_chunk_0",
 │  INGESTION   │──────────────────────────▶ metadata {doc_id, title, chunk_index}
 └──────────────┘
        ▼
 ┌──────────────┐  ingest.embed(): SentenceTransformer("all-MiniLM-L6-v2"),
 │  EMBEDDING   │  384-d, L2-normalised ──▶ ChromaDB PersistentClient("chroma_db/"),
 └──────────────┘  collection "zepto_policies" (hnsw:space = cosine)
        ▼
  POST /ask {"query"} ── main.py ──▶ graph.assistant_graph.invoke({"query": ...})
        ▼
 ┌─────────────────┐  MOCK: keyword heuristic   |  MOCK_LLM=0: LLM + CLASSIFY_PROMPT_TEMPLATE
 │ classify_intent │
 └─────────────────┘
     │ route_by_intent (conditional edge; does not depend on MOCK_LLM)
     ├── policy_question ──▶ ┌─────────────────────┐ RETRIEVAL (always real): embed query,
     │                       │ retrieve_and_answer │ Chroma top-3 by cosine similarity
     │                       └─────────────────────┘ GENERATION: MOCK = "Based on the retrieved
     │                                                context: <first 200 chars of top chunk>"
     │                                                | MOCK_LLM=0: ANSWER_PROMPT_TEMPLATE → LLM
     └── general_question ─▶ ┌───────────────┐       GENERATION: MOCK = fixed canned string
                             │ direct_answer │       | MOCK_LLM=0: DIRECT_PROMPT_TEMPLATE → LLM
                             └───────────────┘
        ▼
  AskResponse(answer, sources, confidence)  ── Pydantic-validated JSON ──▶ HTTP 200
```

1. **Ingestion** (`ingest.py`: `load_documents`, `chunk_documents`). The 8 `.txt` files are read from `docs/`.
   Each policy is a single paragraph of 333–561 characters, so each document becomes **one chunk**
   (`doc_01_chunk_0` … `doc_08_chunk_0`) with `doc_id` and `title` metadata. Splitting further would separate
   facts that belong together, such as a fee and the order threshold it applies to.
2. **Embedding** (`ingest.py`: `embed`, `build_index`). Each chunk is encoded locally by
   `sentence-transformers/all-MiniLM-L6-v2` into a normalised 384-dimensional vector and upserted, with its
   text and metadata, into the persistent ChromaDB collection **`zepto_policies`** (`chroma_db/`, cosine
   distance). `main.py` calls `ensure_index()` at startup, so the index is rebuilt automatically when missing
   (`chroma_db/` is git-ignored; the Docker image builds it at build time).
3. **Retrieval** (`graph.retrieve_and_answer` → `ingest.retrieve`). Runs only for `policy_question` queries,
   which the `classify_intent` node decides and the `route_by_intent` conditional edge routes. The query is
   embedded with the same model and ChromaDB returns the **top-3** chunks; similarity = 1 − cosine distance.
   **This stage never branches on `MOCK_LLM`.** It always runs for real, and it needs no key or network.
4. **Generation** (`graph.retrieve_and_answer` / `graph.direct_answer`). Produces the `AskResponse`.
   **This is the stage that branches on `MOCK_LLM`, together with `classify_intent`'s classification step.**
   - **Default / `MOCK_LLM=1` (graded):**
     - `classify_intent` uses the keyword heuristic: a query containing delivery, return, refund, membership,
       tracking, cancel, gift card or support hours is a `policy_question`; anything else is a `general_question`.
     - `retrieve_and_answer` returns `"Based on the retrieved context: <first 200 chars of the top chunk>..."`,
       with `sources` = the 3 retrieved chunk ids and `confidence = 1.0`.
     - `direct_answer` returns the fixed string `"I can only answer questions about Zepto policies right now."`,
       with `sources = []` and `confidence = 1.0`.
     - The schema is filled deterministically by our own code, so there is no LLM output that could fail
       validation, and no network call is made (`test_graph.py` patches the HTTP client to fail and still passes).
   - **Optional `MOCK_LLM=0`:**
     - `classify_intent` asks the LLM using `CLASSIFY_PROMPT_TEMPLATE`.
     - `retrieve_and_answer` sends the retrieved chunks, labelled by id, inside `ANSWER_PROMPT_TEMPLATE`.
     - `direct_answer` uses `DIRECT_PROMPT_TEMPLATE`.
     - `graph.generate_validated` parses the reply into `AskResponse` and also rejects sources that were not
       actually retrieved. On failure it **retries up to 2 more times**, appending `CORRECTIVE_INSTRUCTION`
       with the validation error. After 3 failed attempts it returns a clearly marked
       `"[ERROR] LLM output failed schema validation ..."` response with `confidence = 0.0`.

## Prompt template (`prompts.ANSWER_PROMPT_TEMPLATE`)

The full text is in `prompts.py`. It has the five skeleton sections plus the constraints and a worked example:

- **ROLE:** Zepto's customer-support assistant.
- **CONTEXT:** the retrieved excerpts, each prefixed with `[chunk_id]`.
- **TASK:** answer only from the excerpts. **Negative constraints:** *"Do NOT answer using information that is
  not present in the provided context."* and *"Do NOT invent fees, time limits, phone numbers or policies."*
  If the answer is missing, the model must say so and use low confidence.
- **FORMAT:** a single JSON object `{"answer", "sources", "confidence"}`, with no markdown fences.
- **LENGTH:** at most 3 sentences / 80 words.
- **EXAMPLE (few-shot):** a gift-card context, a question and the exact expected JSON answer.

## Optional real-LLM extension (not required, not used for the results above)

```bash
export MOCK_LLM=0
export GROQ_API_KEY=...          # free Groq account; never commit it (.env is git-ignored)
export GROQ_MODEL=llama-3.1-8b-instant   # optional override
uvicorn main:app --port 7860
```

The request goes to Groq's OpenAI-compatible `chat/completions` endpoint via `httpx`.
