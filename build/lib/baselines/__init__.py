"""Lightweight baselines included for controlled SG-RAG comparisons."""

from .methods import (
    BM25Method,
    DirectMethod,
    FullContextMethod,
    HybridMethod,
    HyDEMethod,
    SimpleGraphMethod,
    VectorMethod,
)

__all__ = [
    "BM25Method",
    "DirectMethod",
    "FullContextMethod",
    "HybridMethod",
    "HyDEMethod",
    "SimpleGraphMethod",
    "VectorMethod",
]

