import pytest

from benchmark.metrics import answer_quality_metrics, retrieval_proxy_metrics


def test_answer_quality_metrics_computes_exact_match_and_rouge():
    metrics = answer_quality_metrics("Ada Lovelace", ["Ada Lovelace"])

    assert metrics["exact_match"] == 1.0
    assert metrics["rouge_l"] == pytest.approx(1.0)


def test_retrieval_proxy_metrics_uses_ranked_chunk_text():
    metrics = retrieval_proxy_metrics(
        ["Ada"],
        [{"rank": 1, "text": "Ada wrote the note."}, {"rank": 2, "text": "Another note."}],
    )

    assert metrics["available"] is True
    assert metrics["hit_at_1"] == 1.0
    assert metrics["mrr"] == 1.0
