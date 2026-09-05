"""Retrieval contracts and primitives shared by SG-RAG and baselines."""
from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .prompts import build_qa_prompt
from .utils import chunk_text, minmax_scale, tokenize_for_bm25

if TYPE_CHECKING:
    from .models import LocalCausalLLM, LocalEmbeddingModel

class SimpleBM25:
    """Small BM25 implementation to avoid depending on rank_bm25."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.doc_lens = np.array([len(doc) for doc in corpus_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) if len(self.doc_lens) else 0.0
        self.doc_freq: Counter[str] = Counter()
        self.term_freqs: list[Counter[str]] = []
        for doc in corpus_tokens:
            counter = Counter(doc)
            self.term_freqs.append(counter)
            self.doc_freq.update(counter.keys())
        self.n_docs = len(corpus_tokens)
        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in self.doc_freq.items()
        }

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if self.n_docs == 0 or self.avgdl == 0:
            return scores
        for index, term_freq in enumerate(self.term_freqs):
            doc_len = self.doc_lens[index]
            score = 0.0
            for term in query_tokens:
                freq = term_freq.get(term, 0)
                if freq <= 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * freq * (self.k1 + 1) / denom
            scores[index] = score
        return scores

@dataclass
class RetrievalResult:
    chunks: list[dict[str, Any]]
    meta: dict[str, Any]


class BaseMethod:
    def __init__(self, args: argparse.Namespace, llm: LocalCausalLLM | None, embedder: LocalEmbeddingModel | None):
        self.args = args
        self.llm = llm
        self.embedder = embedder

    def retrieve(self, question: str, context: str) -> RetrievalResult:
        raise NotImplementedError

    def build_prompt(self, question: str, retrieved: RetrievalResult, context: str) -> str:
        context_text = "\n\n".join(chunk["text"] for chunk in retrieved.chunks)
        return build_qa_prompt(question, context_text)

    def answer(self, question: str, context: str) -> tuple[str, RetrievalResult, float]:
        retrieved = self.retrieve(question, context)
        prompt = self.build_prompt(question, retrieved, context)
        start = time.perf_counter()
        assert self.llm is not None
        raw = self.llm.generate(prompt)
        return raw, retrieved, time.perf_counter() - start
