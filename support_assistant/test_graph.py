"""Offline tests for routing, mock mode and the real-LLM retry logic.

Run: pytest support_assistant  (or: cd support_assistant && python -m pytest)
No LLM provider is ever contacted: the network client is patched to fail loudly.
"""

import httpx
import pytest

import graph
from ingest import ensure_index


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("network call attempted")

    monkeypatch.setattr(httpx, "post", boom)
    ensure_index()


def test_mock_policy_query_routes_to_retrieval(monkeypatch):
    monkeypatch.delenv("MOCK_LLM", raising=False)  # default = mock
    resp, state = graph.answer_query("How do refunds work for a return?")
    assert state["intent"] == "policy_question"
    assert state["retrieved"][0]["metadata"]["doc_id"] == "doc_02"
    assert resp.answer.startswith("Based on the retrieved context: Grocery and perishable items")
    assert resp.sources == [h["id"] for h in state["retrieved"]] and len(resp.sources) == 3
    assert resp.confidence == 1.0


def test_mock_general_query_routes_to_direct_answer(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")
    resp, state = graph.answer_query("Who won the cricket match yesterday?")
    assert state["intent"] == "general_question"
    assert resp.answer == graph.DIRECT_ANSWER_CANNED
    assert resp.sources == []


def test_real_llm_retries_then_succeeds(monkeypatch):
    replies = iter(["not json at all", '{"answer": "ok", "sources": ["doc_01_chunk_0"], "confidence": 0.9}'])
    calls = []
    monkeypatch.setattr(graph, "call_llm", lambda messages: calls.append(messages) or next(replies))
    resp = graph.generate_validated("prompt", allowed_sources={"doc_01_chunk_0"})
    assert resp.answer == "ok" and len(calls) == 2
    assert "could not be parsed" in calls[1][-1]["content"]  # corrective instruction was sent


def test_real_llm_gives_up_after_two_retries(monkeypatch):
    calls = []
    monkeypatch.setattr(graph, "call_llm", lambda messages: calls.append(1) or '{"answer": "x", "confidence": 7}')
    resp = graph.generate_validated("prompt", allowed_sources=set())
    assert len(calls) == 3  # 1 attempt + 2 retries
    assert resp.answer.startswith("[ERROR]") and resp.confidence == 0.0
