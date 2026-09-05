"""Lightweight local baselines retained for SG-RAG paper comparisons."""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from sg_rag.prompts import build_qa_prompt
from sg_rag.retrieval import BaseMethod, RetrievalResult, SimpleBM25
from sg_rag.utils import chunk_text, minmax_scale, tokenize_for_bm25

if TYPE_CHECKING:
    from sg_rag.models import LocalCausalLLM, LocalEmbeddingModel

class DirectMethod(BaseMethod):
    def retrieve(self, question: str, context: str) -> RetrievalResult:
        return RetrievalResult(chunks=[], meta={"retrieval": "none"})

    def build_prompt(self, question: str, retrieved: RetrievalResult, context: str) -> str:
        return (
            "Answer the question directly. Return only the shortest final answer phrase or one sentence.\n\n"
            f"Question: {question}\nAnswer:"
        )


class FullContextMethod(BaseMethod):
    def retrieve(self, question: str, context: str) -> RetrievalResult:
        chunk = {
            "chunk_id": 0,
            "text": context,
            "score": None,
            "source": "full_context",
            "char_length": len(context),
        }
        return RetrievalResult(chunks=[chunk], meta={"retrieval": "full_context"})


class BM25Method(BaseMethod):
    def retrieve(self, question: str, context: str) -> RetrievalResult:
        chunks = chunk_text(context, self.args.chunk_words, self.args.chunk_overlap)
        tokenized = [tokenize_for_bm25(chunk["text"]) for chunk in chunks]
        bm25 = SimpleBM25(tokenized)
        scores = bm25.get_scores(tokenize_for_bm25(question))
        order = np.argsort(-scores)[: self.args.top_k]
        selected = []
        for rank, index in enumerate(order, start=1):
            chunk = dict(chunks[int(index)])
            chunk["score"] = float(scores[int(index)])
            chunk["rank"] = rank
            chunk["source"] = "bm25"
            selected.append(chunk)
        return RetrievalResult(chunks=selected, meta={"retrieval": "bm25", "candidate_chunks": len(chunks)})


class VectorMethod(BaseMethod):
    def retrieve(self, question: str, context: str) -> RetrievalResult:
        assert self.embedder is not None
        chunks = chunk_text(context, self.args.chunk_words, self.args.chunk_overlap)
        texts = [chunk["text"] for chunk in chunks]
        if not texts:
            return RetrievalResult(chunks=[], meta={"retrieval": "vector", "candidate_chunks": 0})
        chunk_vectors = self.embedder.encode(texts)
        query_vector = self.embedder.encode([question])[0]
        scores = chunk_vectors @ query_vector
        order = np.argsort(-scores)[: self.args.top_k]
        selected = []
        for rank, index in enumerate(order, start=1):
            chunk = dict(chunks[int(index)])
            chunk["score"] = float(scores[int(index)])
            chunk["rank"] = rank
            chunk["source"] = "bge_m3"
            selected.append(chunk)
        return RetrievalResult(chunks=selected, meta={"retrieval": "vector", "candidate_chunks": len(chunks)})


class HybridMethod(BaseMethod):
    def retrieve(self, question: str, context: str) -> RetrievalResult:
        assert self.embedder is not None
        chunks = chunk_text(context, self.args.chunk_words, self.args.chunk_overlap)
        texts = [chunk["text"] for chunk in chunks]
        if not texts:
            return RetrievalResult(chunks=[], meta={"retrieval": "hybrid", "candidate_chunks": 0})
        bm25 = SimpleBM25([tokenize_for_bm25(text) for text in texts])
        bm25_scores = bm25.get_scores(tokenize_for_bm25(question))
        chunk_vectors = self.embedder.encode(texts)
        query_vector = self.embedder.encode([question])[0]
        dense_scores = chunk_vectors @ query_vector
        alpha = self.args.hybrid_alpha
        scores = alpha * minmax_scale(dense_scores) + (1 - alpha) * minmax_scale(bm25_scores)
        order = np.argsort(-scores)[: self.args.top_k]
        selected = []
        for rank, index in enumerate(order, start=1):
            chunk = dict(chunks[int(index)])
            chunk["score"] = float(scores[int(index)])
            chunk["dense_score"] = float(dense_scores[int(index)])
            chunk["bm25_score"] = float(bm25_scores[int(index)])
            chunk["rank"] = rank
            chunk["source"] = "hybrid_bge_bm25"
            selected.append(chunk)
        return RetrievalResult(
            chunks=selected,
            meta={"retrieval": "hybrid", "candidate_chunks": len(chunks), "hybrid_alpha": alpha},
        )


class HyDEMethod(VectorMethod):
    def retrieve(self, question: str, context: str) -> RetrievalResult:
        assert self.llm is not None
        assert self.embedder is not None
        hyde_prompt = (
            "Write a concise hypothetical passage that would answer the question. "
            "Do not say you are guessing. Include likely key terms and facts.\n\n"
            f"Question: {question}\nHypothetical passage:"
        )
        hypothetical = self.llm.generate(hyde_prompt, max_new_tokens=self.args.hyde_max_new_tokens)
        chunks = chunk_text(context, self.args.chunk_words, self.args.chunk_overlap)
        texts = [chunk["text"] for chunk in chunks]
        if not texts:
            return RetrievalResult(
                chunks=[],
                meta={"retrieval": "hyde", "candidate_chunks": 0, "hypothetical_document": hypothetical},
            )
        chunk_vectors = self.embedder.encode(texts)
        query_vector = self.embedder.encode([hypothetical])[0]
        scores = chunk_vectors @ query_vector
        order = np.argsort(-scores)[: self.args.top_k]
        selected = []
        for rank, index in enumerate(order, start=1):
            chunk = dict(chunks[int(index)])
            chunk["score"] = float(scores[int(index)])
            chunk["rank"] = rank
            chunk["source"] = "hyde_bge_m3"
            selected.append(chunk)
        return RetrievalResult(
            chunks=selected,
            meta={
                "retrieval": "hyde",
                "candidate_chunks": len(chunks),
                "hypothetical_document": hypothetical,
            },
        )


class SimpleGraphMethod(BaseMethod):
    def retrieve(self, question: str, context: str) -> RetrievalResult:
        chunks = chunk_text(context, self.args.chunk_words, self.args.chunk_overlap)
        graph = self._build_graph(chunks)
        query_terms = self._extract_terms(question)
        matched_nodes = [term for term in query_terms if term in graph["node_to_chunks"]]

        chunk_scores: defaultdict[int, float] = defaultdict(float)
        for node in matched_nodes:
            for chunk_id in graph["node_to_chunks"][node]:
                chunk_scores[chunk_id] += 2.0
            for neighbor in graph["edges"].get(node, {}):
                for chunk_id in graph["node_to_chunks"].get(neighbor, []):
                    chunk_scores[chunk_id] += 1.0

        # Fallback to BM25 if graph matching is sparse.
        if len(chunk_scores) < max(1, self.args.top_k // 2):
            bm25 = SimpleBM25([tokenize_for_bm25(chunk["text"]) for chunk in chunks])
            bm25_scores = bm25.get_scores(tokenize_for_bm25(question))
            for chunk_id, score in enumerate(bm25_scores):
                chunk_scores[chunk_id] += float(score) * 0.25

        ranked = sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True)[: self.args.top_k]
        selected = []
        for rank, (chunk_id, score) in enumerate(ranked, start=1):
            chunk = dict(chunks[int(chunk_id)])
            chunk["score"] = float(score)
            chunk["rank"] = rank
            chunk["source"] = "simple_graph"
            selected.append(chunk)
        return RetrievalResult(
            chunks=selected,
            meta={
                "retrieval": "simple_graph",
                "candidate_chunks": len(chunks),
                "node_count": len(graph["node_to_chunks"]),
                "edge_count": sum(len(v) for v in graph["edges"].values()) // 2,
                "matched_nodes": matched_nodes[:30],
                "graph_extractor": self.args.graph_extractor,
            },
        )

    def _build_graph(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        node_to_chunks: defaultdict[str, set[int]] = defaultdict(set)
        edges: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for chunk in chunks:
            terms = self._extract_terms(chunk["text"])
            chunk_id = int(chunk["chunk_id"])
            for term in terms:
                node_to_chunks[term].add(chunk_id)
            window_terms = terms[: self.args.graph_max_terms_per_chunk]
            for i, left in enumerate(window_terms):
                for right in window_terms[i + 1 : i + 1 + self.args.graph_edge_window]:
                    if left == right:
                        continue
                    edges[left][right] += 1
                    edges[right][left] += 1
        return {
            "node_to_chunks": {key: sorted(value) for key, value in node_to_chunks.items()},
            "edges": edges,
        }

    def _extract_terms(self, text: str) -> list[str]:
        lower_terms = tokenize_for_bm25(text)
        stop = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "have",
            "what",
            "when",
            "where",
            "which",
            "into",
            "about",
            "because",
            "according",
            "question",
            "answer",
        }
        terms = [term for term in lower_terms if len(term) >= 4 and term not in stop]
        proper = re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\b", text or "")
        terms.extend(term.lower() for term in proper)
        seen = set()
        unique = []
        for term in terms:
            if term not in seen:
                unique.append(term)
                seen.add(term)
        return unique
