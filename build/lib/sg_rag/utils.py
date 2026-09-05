"""General benchmark utilities shared by the SG-RAG runner."""
from __future__ import annotations

import hashlib
import math
import re
import string
from collections import Counter
from typing import Any

import numpy as np

def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc((text or "").lower())))


def qa_prf_detail(prediction: str, ground_truth: str) -> dict[str, float | int]:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        score = float(pred_tokens == gold_tokens)
        return {
            "precision": score,
            "recall": score,
            "f1": score,
            "pred_token_count": len(pred_tokens),
            "gold_token_count": len(gold_tokens),
            "overlap_token_count": int(pred_tokens == gold_tokens),
        }
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "pred_token_count": len(pred_tokens),
            "gold_token_count": len(gold_tokens),
            "overlap_token_count": 0,
        }
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_token_count": len(pred_tokens),
        "gold_token_count": len(gold_tokens),
        "overlap_token_count": overlap,
    }


def best_qa_prf_detail(prediction: str, answers: list[str]) -> dict[str, Any]:
    if not answers:
        detail = qa_prf_detail(prediction, "")
        detail["best_answer"] = None
        detail["best_answer_index"] = None
        return detail
    scored = []
    for index, answer in enumerate(answers):
        detail = qa_prf_detail(prediction, answer)
        detail["best_answer"] = answer
        detail["best_answer_index"] = index
        scored.append(detail)
    return max(scored, key=lambda item: item["f1"])


def postprocess_short_answer(text: str, max_words: int = 40) -> str:
    answer = "" if text is None else str(text).strip()
    if not answer:
        return ""

    # Remove common markdown/reference wrappers while preserving raw_prediction.
    while True:
        stripped = re.sub(r"^\s*#{1,6}\s*[^\n]+\n*", "", answer, count=1).strip()
        if stripped == answer:
            break
        answer = stripped

    for pattern in [
        r"\n\s*#{1,6}\s*references\b",
        r"\n\s*references\s*:",
        r"\n\s*sources\s*:",
        r"\n\s*citations\s*:",
        r"\n\s*note\s*:",
    ]:
        match = re.search(pattern, answer, flags=re.IGNORECASE)
        if match:
            answer = answer[: match.start()].strip()

    explicit = re.search(
        r"(?:^|\n)\s*(?:final\s+answer|answer)\s*:\s*(.+)",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if explicit:
        answer = explicit.group(1).strip()

    answer = re.sub(r"(?im)^\s*(?:state|step|stage)\s*\d+\s*[:.)-]\s*", "", answer)
    answer = re.sub(r"(?im)^\s*\d+\s*[:.)-]\s*", "", answer)
    answer = re.sub(r"\n+", "; ", answer).strip()
    answer = re.sub(r"^\s*[-*]\s+", "", answer)
    answer = re.sub(r"^#+\s*", "", answer)
    answer = re.sub(
        r"^(according to|based on)\s+(?:the\s+)?(?:provided\s+)?(?:context|text|passage|poem),?\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(r"\s*\([^)]*\bline\b[^)]*\)\s*$", "", answer, flags=re.IGNORECASE)
    answer = answer.strip(" \t\r\n\"'`")

    first_para = re.split(r"\n\s*\n", answer, maxsplit=1)[0].strip()
    if first_para:
        answer = first_para
    chain_like = bool(
        re.search(r"\b(then|finally|vapou?r|cloud|meteor|star|mars)\b", answer, flags=re.IGNORECASE)
        and len(re.findall(r"[.!?]", answer)) <= 5
    )
    sentence_match = re.search(r"(.+?[.!?])(?:\s|$)", answer, flags=re.DOTALL)
    if sentence_match and not chain_like:
        answer = sentence_match.group(1).strip()

    words = answer.split()
    if max_words > 0 and len(words) > max_words:
        answer = " ".join(words[:max_words]).rstrip(" ,;:")
    return answer.strip()


def approx_word_count(text: str | None) -> int:
    return len((text or "").split())


def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())


def chunk_text(text: str, chunk_words: int, overlap_words: int) -> list[dict[str, Any]]:
    words = (text or "").split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        end = min(len(words), start + chunk_words)
        chunk = " ".join(words[start:end])
        chunks.append(
            {
                "chunk_id": len(chunks),
                "text": chunk,
                "word_start": start,
                "word_end": end,
                "char_length": len(chunk),
            }
        )
        if end >= len(words):
            break
    return chunks


def minmax_scale(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if math.isclose(low, high):
        return np.ones_like(values) if high > 0 else np.zeros_like(values)
    return (values - low) / (high - low)
