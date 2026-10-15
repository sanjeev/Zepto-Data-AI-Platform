"""Offline demo / self-check of the graph in default mock mode (no network to any LLM).

Run: python demo.py
Shows routing, retrieved chunks and the validated JSON for a set of example queries.
"""

import json
import os

os.environ.setdefault("MOCK_LLM", "1")

from graph import DIRECT_ANSWER_CANNED, answer_query  # noqa: E402
from ingest import ensure_index  # noqa: E402

# (query, expected intent, expected top source document or None)
EXAMPLES = [
    ("What is the delivery fee for an order below INR 149?", "policy_question", "doc_01"),
    ("How long do refunds take to reach my bank account?", "policy_question", "doc_02"),
    ("How much does the Zepto Pass+ membership cost?", "policy_question", "doc_03"),
    ("Can I cancel my order after it has been packed?", "policy_question", "doc_05"),
    ("Can a gift card be combined with another gift card?", "policy_question", "doc_07"),
    ("What are your customer support hours?", "policy_question", "doc_08"),
    ("What is the capital of France?", "general_question", None),
    ("Tell me a joke about cats.", "general_question", None),
]


def main() -> None:
    ensure_index()
    failures = 0
    for query, want_intent, want_doc in EXAMPLES:
        resp, state = answer_query(query)
        top_doc = state["retrieved"][0]["metadata"]["doc_id"] if state["retrieved"] else None
        ok = state["intent"] == want_intent and top_doc == want_doc
        if want_intent == "policy_question":
            ok &= resp.answer.startswith("Based on the retrieved context: ")
        else:
            ok &= resp.answer == DIRECT_ANSWER_CANNED and resp.sources == []
        failures += not ok
        node = "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"
        print(f"{'PASS' if ok else 'FAIL'}  [{state['intent']} -> {node}]  {query}")
        for h in state["retrieved"]:
            print(f"        {h['id']:<15} sim={h['similarity']:.3f}  {h['metadata']['title']}")
        print("        " + json.dumps(resp.model_dump()))
    print(f"\n{len(EXAMPLES) - failures}/{len(EXAMPLES)} examples passed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
