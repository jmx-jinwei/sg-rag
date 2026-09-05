"""Public SG-RAG algorithm package with lazy model imports."""

__all__ = [
    "BaseMethod",
    "build_qa_prompt",
    "LocalCausalLLM",
    "LocalEmbeddingModel",
    "RetrievalResult",
    "SimpleBM25",
    "SGRAGMethod",
    "SkillGraphMethod",
]


def __getattr__(name: str):
    """Load optional model-backed classes only when they are requested."""
    if name in {"SGRAGMethod", "SkillGraphMethod"}:
        from .method import SGRAGMethod, SkillGraphMethod

        return {"SGRAGMethod": SGRAGMethod, "SkillGraphMethod": SkillGraphMethod}[name]
    if name in {"LocalCausalLLM", "LocalEmbeddingModel"}:
        from .models import LocalCausalLLM, LocalEmbeddingModel

        return {"LocalCausalLLM": LocalCausalLLM, "LocalEmbeddingModel": LocalEmbeddingModel}[name]
    if name in {"BaseMethod", "RetrievalResult", "SimpleBM25"}:
        from .retrieval import BaseMethod, RetrievalResult, SimpleBM25

        return {"BaseMethod": BaseMethod, "RetrievalResult": RetrievalResult, "SimpleBM25": SimpleBM25}[name]
    if name == "build_qa_prompt":
        from .prompts import build_qa_prompt

        return build_qa_prompt
    raise AttributeError(name)
