"""LangGraph orchestration: classify_intent -> (retrieve_and_answer | direct_answer).

Every node's *generation* step branches on MOCK_LLM:
  * MOCK_LLM unset or "1" (default, graded): deterministic rule-based logic, no LLM call.
  * MOCK_LLM="0" (optional): calls a real LLM (Groq free tier, OpenAI-compatible API).
Retrieval in retrieve_and_answer always runs for real (local embeddings + ChromaDB).
"""

from __future__ import annotations

import json
import os
import re
from typing import Literal, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from ingest import retrieve
from prompts import (
    ANSWER_PROMPT_TEMPLATE,
    CLASSIFY_PROMPT_TEMPLATE,
    CORRECTIVE_INSTRUCTION,
    DIRECT_PROMPT_TEMPLATE,
)
from schemas import AskResponse

POLICY_KEYWORDS = (
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
)
DIRECT_ANSWER_CANNED = "I can only answer questions about Zepto policies right now."
SNIPPET_CHARS = 200
MAX_EXTRA_RETRIES = 2  # real-LLM path: 1 attempt + up to 2 corrective retries

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


class AssistantState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved: list[dict]
    response: dict  # AskResponse.model_dump()


def mock_llm_enabled() -> bool:
    """Mock is the default: only an explicit MOCK_LLM=0 turns the real LLM on."""
    return os.getenv("MOCK_LLM", "1").strip() != "0"


# --------------------------------------------------------------------------- #
# Real-LLM helpers (optional MOCK_LLM=0 path only)
# --------------------------------------------------------------------------- #
def call_llm(messages: list[dict], temperature: float = 0.0) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("MOCK_LLM=0 requires GROQ_API_KEY to be set in the environment")
    resp = httpx.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": GROQ_MODEL, "messages": messages, "temperature": temperature},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _extract_json(text: str) -> str:
    """Strip markdown fences / chatter around the first {...} block, if any."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else text


def generate_validated(prompt: str, allowed_sources: set[str]) -> AskResponse:
    """Call the LLM and validate against AskResponse, retrying up to 2 more times
    with a corrective instruction; return a clearly marked error response if all fail."""
    messages = [{"role": "user", "content": prompt}]
    last_error = "unknown error"
    for _attempt in range(1 + MAX_EXTRA_RETRIES):
        raw = call_llm(messages)
        try:
            parsed = AskResponse.model_validate(json.loads(_extract_json(raw)))
            unknown = set(parsed.sources) - allowed_sources
            if unknown:
                raise ValueError(f"sources not in retrieved context: {sorted(unknown)}")
            return parsed
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = str(exc)
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": CORRECTIVE_INSTRUCTION.format(error=last_error)},
            ]
    return AskResponse(
        answer=f"[ERROR] LLM output failed schema validation after {1 + MAX_EXTRA_RETRIES} attempts: {last_error}",
        sources=[],
        confidence=0.0,
    )


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def classify_intent(state: AssistantState) -> AssistantState:
    query = state["query"]
    if mock_llm_enabled():
        q = query.lower()
        intent = "policy_question" if any(k in q for k in POLICY_KEYWORDS) else "general_question"
    else:
        label = call_llm([{"role": "user", "content": CLASSIFY_PROMPT_TEMPLATE.format(question=query)}])
        intent = "policy_question" if "policy_question" in label.lower() else "general_question"
    return {"intent": intent}


def retrieve_and_answer(state: AssistantState) -> AssistantState:
    # Retrieval always runs for real: local MiniLM embedding + ChromaDB cosine search.
    hits = retrieve(state["query"], k=3)
    source_ids = [h["id"] for h in hits]

    if mock_llm_enabled():
        top_text = hits[0]["text"]
        snippet = top_text if len(top_text) <= SNIPPET_CHARS else top_text[:SNIPPET_CHARS].rstrip() + "..."
        response = AskResponse(
            answer=f"Based on the retrieved context: {snippet}",
            sources=source_ids,
            confidence=1.0,
        )
    else:
        context = "\n\n".join(f"[{h['id']}] {h['text']}" for h in hits)
        prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, question=state["query"])
        response = generate_validated(prompt, allowed_sources=set(source_ids))
    return {"retrieved": hits, "response": response.model_dump()}


def direct_answer(state: AssistantState) -> AssistantState:
    if mock_llm_enabled():
        response = AskResponse(answer=DIRECT_ANSWER_CANNED, sources=[], confidence=1.0)
    else:
        prompt = DIRECT_PROMPT_TEMPLATE.format(question=state["query"])
        response = generate_validated(prompt, allowed_sources=set())
    return {"retrieved": [], "response": response.model_dump()}


def route_by_intent(state: AssistantState) -> str:
    # Routing depends only on the classification, never on MOCK_LLM.
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def build_graph():
    graph = StateGraph(AssistantState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)
    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()


assistant_graph = build_graph()


def answer_query(query: str) -> tuple[AskResponse, AssistantState]:
    final_state = assistant_graph.invoke({"query": query})
    return AskResponse.model_validate(final_state["response"]), final_state
