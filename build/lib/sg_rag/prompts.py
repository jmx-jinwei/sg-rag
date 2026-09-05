"""Prompt templates shared by SG-RAG's lightweight baselines."""

from __future__ import annotations


def build_qa_prompt(question: str, context_text: str) -> str:
    """Build the short-answer prompt used by non-SG-RAG baselines."""
    return (
        "Use only the provided context to answer the question.\n"
        "Return only the shortest final answer phrase or one sentence.\n"
        "Do not include references, markdown, source names, or explanations.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

