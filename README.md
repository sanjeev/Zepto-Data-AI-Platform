# Zepto Data & AI Platform

One repository with three linked modules. Together they tell one story: clean structured data for analysts,
an end-to-end analytics and modeling workflow, and a grounded GenAI service in front of Zepto's own policies.

| Module | Folder | Marks | What it does |
|---|---|---|---|
| Data Pipeline | [`data_pipeline/`](data_pipeline/) | 25 | Scrape books.toscrape.com → clean → GBP→INR at a fixed rate → normalized SQLite → SQL + pandas |
| Analytics Pipeline | [`analytics/`](analytics/) | 50 | Titanic: profile → clean → EDA story → stratified ML pipeline → tuning → saved joblib pipeline |
| Support Assistant | [`support_assistant/`](support_assistant/) | 25 | MiniLM + ChromaDB RAG, LangGraph intent router, Pydantic JSON, FastAPI `/ask`, Dockerfile |

Each module has its own README with the full write-up: every required interpretation, the query outputs,
the model comparison table, the example API calls and the architecture description.

## Setup

**Requirements approach: one `requirements.txt` per module** (`data_pipeline/requirements.txt`,
`analytics/requirements.txt`, `support_assistant/requirements.txt`). The modules have very different
dependency weights (the support assistant pulls in PyTorch for sentence-transformers), and the Dockerfile
needs its own list anyway. Python 3.11+ is required (developed on 3.13).

```bash
git clone <this repo> && cd zepto-data-ai-platform
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r data_pipeline/requirements.txt \
            -r analytics/requirements.txt \
            -r support_assistant/requirements.txt
```

*macOS note:* with the python.org installer, HTTPS downloads from Python (seaborn's dataset, the Hugging Face
model) can fail with `CERTIFICATE_VERIFY_FAILED`. Fix it with
`export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")` or by running the
"Install Certificates.command" that ships with that Python.

## Running each module end to end

### 1. Data pipeline
```bash
python data_pipeline/pipeline.py         # scrapes (internet), rebuilds data_pipeline/books.db from scratch,
                                         # writes data_pipeline/outputs/query_results.md
python data_pipeline/test_cleaning.py    # offline messy-row test
```

### 2. Analytics
```bash
cd analytics
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb       # the ONE dataset load + saves titanic.csv
jupyter nbconvert --to notebook --execute --inplace 02_modeling.ipynb  # reads titanic.csv; saves the joblib pipeline
python predict_demo.py                                                 # reload pipeline, predict on raw rows
cd ..
```
(Or open the notebooks in Jupyter and run all cells in order: 01, then 02.)

### 3. Support assistant (graded in default mock mode: no API key, no LLM network calls)
```bash
cd support_assistant
python ingest.py                  # embed 8 docs → ChromaDB collection "zepto_policies"
python demo.py                    # 8/8 routing + retrieval checks
python -m pytest -q test_graph.py
uvicorn main:app --port 7860      # then: curl -X POST localhost:7860/ask -H 'Content-Type: application/json' -d '{"query":"Can I cancel after packing?"}'
# Docker:
docker build -t zepto-support . && docker run --rm -p 7860:7860 zepto-support
```

## Design decisions

### Data pipeline
- **Scope:** four categories (Mystery, Historical Fiction, Science Fiction, Travel), following pagination,
  giving **85 books across 4 categories** (requirement: ≥ 60 across ≥ 3).
- **Robust fetching:** session reuse, a timeout, explicit `status_code` checks and retries. Pages are parsed
  from raw bytes so `£` decodes correctly.
- **Messy rows:** unparseable *numeric* fields (price, rating) are **median-imputed**, so one bad value does not
  lose the row, and the median resists price skew. Unparseable *non-numeric* fields (title, category, stock
  status) cause the row to be **dropped**, because there is no defensible "median title" and a guessed stock
  status would mislead analysts. A unit test proves this on synthetic messy rows.
- **Currency:** `price_inr = price_gbp × 105.50`, using the project's fixed baseline rate
  **1 GBP = 105.50 INR**. This is an artificial, project-defined constant, not a market rate, and needs no
  lookup, network access or date. The optional live-API stretch was not attempted.
- **Schema:** `categories(category_id PK, category_name UNIQUE)` ← `books(book_id PK, …, category_id FK)`,
  with CHECK constraints on `rating` (1–5) and `in_stock` (0/1), and foreign keys enforced.
- **Queries:** six SQL queries cover SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, IN, BETWEEN and two JOINs.
  The top-3-per-category JOIN is reproduced with `pd.merge` and proven identical with
  `pd.testing.assert_frame_equal`.

### Analytics
- **Load once:** `sns.load_dataset('titanic')` appears exactly once (first cell of `01_eda.ipynb`) and is
  immediately saved to the committed `analytics/titanic.csv`. `02_modeling.ipynb` reads only that CSV.
- **Shared cleaning (`cleaning.py`):** the threshold rule gives `embarked`/`embark_town` (0.22%) → drop rows,
  `age` (19.87%) → impute, and `deck` (77.22%) → an `Unknown` category. Both notebooks import the same rules.
  Only data-independent steps run before the split. Everything that learns from data (imputer, encoder,
  scaler) sits inside a `ColumnTransformer` + `Pipeline` fit on the training split, so test leakage is
  prevented by the structure itself.
- **Model selection without peeking:** the deployed model is chosen by 5-fold CV on the training split, and the
  test set is only a final check. The **tuned Random Forest** (`max_depth=6`, 400 trees, `oob_score=True`) has
  CV ROC AUC 0.878, test accuracy 0.837, test F1 0.760 and OOB score 0.813.
- **Imbalance:** SMOTE runs inside an `imblearn` pipeline, so it only ever resamples the training fold.
- **Persistence:** the **whole** fitted pipeline (preprocessing + estimator) is saved with `joblib.dump`, then
  reloaded and run on raw rows with missing values.

### Support assistant
- **Offline by default:** `MOCK_LLM` unset or `1` uses deterministic mock logic. Only `MOCK_LLM=0` calls
  Groq (optional). ChromaDB telemetry is disabled, and the Docker image bakes in the model and index and sets
  `HF_HUB_OFFLINE=1`.
- **Chunking:** one chunk per document. Each policy is a short single paragraph, so splitting it would separate
  related facts (e.g., a fee from its threshold).
- **Retrieval is always real:** local MiniLM embeddings plus a cosine-space ChromaDB collection, top-3. All 6
  policy test queries retrieve the correct source document first.
- **Graph:** `classify_intent` → conditional edge (`route_by_intent`, independent of `MOCK_LLM`) →
  `retrieve_and_answer` | `direct_answer`, over a `TypedDict` state.
- **Structured output:** the Pydantic `AskResponse(answer, sources, confidence ∈ [0,1])` is the FastAPI
  `response_model`. On the real-LLM path, invalid JSON triggers up to 2 corrective retries and then a marked
  `[ERROR]` response. This is covered by tests with a fake LLM.

## Git workflow

Each module was built on its own feature branch (`feature/data-pipeline`, `feature/analytics`,
`feature/support-assistant`) with at least two commits, then merged into `main` with `--no-ff`. The merges
are visible with `git log --graph --all --oneline`.

## Repository layout

```
├── README.md                      ← this file
├── data_pipeline/   pipeline.py, test_cleaning.py, books.db, outputs/, README.md, requirements.txt
├── analytics/       01_eda.ipynb, 02_modeling.ipynb, cleaning.py, titanic.csv,
│                    titanic_best_pipeline.joblib, predict_demo.py, figures/, README.md, requirements.txt
└── support_assistant/  docs/doc_01..08.txt, ingest.py, prompts.py, schemas.py, graph.py, main.py,
                        demo.py, test_graph.py, Dockerfile, README.md, requirements.txt
```
