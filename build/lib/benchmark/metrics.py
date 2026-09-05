"""Model-free answer and retrieval metrics used by the SG-RAG runner."""

from __future__ import annotations

import math
import re
import string
from typing import Any, Iterable


def normalize_answer(text: str) -> str:
    """Match the answer normalization used by ``run_benchmark.py``."""
    without_articles = re.sub(r"\b(a|an|the)\b", " ", (text or "").lower())
    without_punctuation = "".join(
        character for character in without_articles if character not in set(string.punctuation)
    )
    return " ".join(without_punctuation.split())


def normalize_ultradomain_row(row: dict[str, Any], sample_index: int) -> dict[str, Any]:
    """Normalize one UltraDomain JSONL row without dropping its metadata."""
    question = row.get("input", row.get("question"))
    if question is None or not str(question).strip():
        raise ValueError(f"UltraDomain row {sample_index} has no input/question")

    answers = row.get("answers", row.get("answer", row.get("gold_answer", [])))
    if isinstance(answers, str):
        answers = [answers]
    elif answers is None:
        answers = []
    else:
        answers = [str(answer) for answer in answers]

    context = row.get("context", "")
    if context is None:
        context = ""

    normalized = dict(row)
    normalized["_id"] = row.get("_id") or f"sample-{sample_index:06d}"
    normalized["input"] = str(question)
    normalized["answers"] = answers
    normalized["context"] = str(context)
    return normalized


def _answer_relevance(answer_tokens: set[str], text: str) -> float:
    if not answer_tokens or not text:
        return 0.0
    retrieved_tokens = set(normalize_answer(text).split())
    if not retrieved_tokens:
        return 0.0
    return len(answer_tokens & retrieved_tokens) / len(answer_tokens)


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _rouge_l(prediction_tokens: list[str], gold_tokens: list[str]) -> float:
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    lcs = _lcs_length(prediction_tokens, gold_tokens)
    if not lcs:
        return 0.0
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_quality_metrics(prediction: str, answers: Iterable[str]) -> dict[str, Any]:
    """Return additional reference-based QA metrics shared by all runners."""
    prediction_tokens = normalize_answer(prediction).split()
    candidates = [normalize_answer(answer).split() for answer in answers if str(answer).strip()]
    if not candidates:
        return {
            "exact_match": 0.0,
            "rouge_l": 0.0,
            "prediction_word_count": len(prediction_tokens),
            "gold_word_count": 0,
            "prediction_to_gold_length_ratio": None,
        }

    scores = [
        (float(prediction_tokens == gold_tokens), _rouge_l(prediction_tokens, gold_tokens), gold_tokens)
        for gold_tokens in candidates
    ]
    exact_match, rouge_l, best_gold_tokens = max(scores, key=lambda item: (item[0], item[1]))
    return {
        "exact_match": exact_match,
        "rouge_l": rouge_l,
        "prediction_word_count": len(prediction_tokens),
        "gold_word_count": len(best_gold_tokens),
        "prediction_to_gold_length_ratio": (
            len(prediction_tokens) / len(best_gold_tokens) if best_gold_tokens else None
        ),
    }


def retrieval_proxy_metrics(
    answers: Iterable[str], retrieved_chunks: list[dict[str, Any]], ks: tuple[int, ...] = (3, 10, 50)
) -> dict[str, Any]:
    """Compute answer-token-overlap retrieval proxies from ranked text chunks."""
    chunks_with_text = [
        chunk
        for chunk in retrieved_chunks
        if isinstance(chunk, dict) and str(chunk.get("text") or "").strip()
    ]
    result: dict[str, Any] = {
        "relevance_definition": "answer_token_overlap_proxy",
        "available": bool(chunks_with_text),
    }
    if not chunks_with_text:
        for k in ks:
            result[f"acc_at_{k}"] = None
            result[f"hit_at_{k}"] = None
            result[f"ndcg_at_{k}"] = None
        for k in (1, 5):
            result[f"hit_at_{k}"] = None
        result["ndcg_at_1"] = None
        result["ndcg_at_5"] = None
        result["mrr"] = None
        result["answer_token_coverage"] = None
        return result

    answer_token_sets = [
        set(normalize_answer(answer).split())
        for answer in answers
        if str(answer).strip()
    ]
    if not answer_token_sets:
        for k in ks:
            result[f"acc_at_{k}"] = None
            result[f"hit_at_{k}"] = None
            result[f"ndcg_at_{k}"] = None
        for k in (1, 5):
            result[f"hit_at_{k}"] = None
        result["ndcg_at_1"] = None
        result["ndcg_at_5"] = None
        result["mrr"] = None
        result["answer_token_coverage"] = None
        return result

    relevance = [
        max((_answer_relevance(tokens, str(chunk["text"])) for tokens in answer_token_sets), default=0.0)
        for chunk in chunks_with_text
    ]
    all_ks = tuple(sorted(set(ks) | {1, 5}))
    for k in all_ks:
        cutoff = relevance[:k]
        hit = float(any(value > 0 for value in cutoff))
        result[f"acc_at_{k}"] = float(any(value > 0 for value in cutoff))
        result[f"hit_at_{k}"] = hit
        dcg = sum(value / math.log2(index + 2) for index, value in enumerate(cutoff))
        ideal = sorted(relevance, reverse=True)[:k]
        idcg = sum(value / math.log2(index + 2) for index, value in enumerate(ideal))
        result[f"ndcg_at_{k}"] = float(dcg / idcg) if idcg else 0.0
    first_relevant = next((index + 1 for index, value in enumerate(relevance) if value > 0), None)
    result["mrr"] = float(1 / first_relevant) if first_relevant else 0.0
    answer_tokens = set().union(*answer_token_sets)
    retrieved_tokens = set(
        token
        for chunk in chunks_with_text
        for token in normalize_answer(str(chunk["text"])).split()
    )
    result["answer_token_coverage"] = (
        len(answer_tokens & retrieved_tokens) / len(answer_tokens) if answer_tokens else 0.0
    )
    return result
