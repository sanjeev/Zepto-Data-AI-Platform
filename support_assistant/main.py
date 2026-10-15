"""FastAPI wrapper: POST /ask -> LangGraph assistant -> validated AskResponse.

Run locally:  uvicorn main:app --port 7860      (from this folder)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from graph import answer_query, mock_llm_enabled
from ingest import ensure_index
from schemas import AskRequest, AskResponse


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_index()  # embed + store the 8 docs if the Chroma collection is empty
    yield


app = FastAPI(title="Zepto Support Assistant", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mock_llm": mock_llm_enabled()}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    response, _state = answer_query(request.query)
    return response
