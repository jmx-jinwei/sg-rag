from sg_rag.retrieval import SimpleBM25
from sg_rag.utils import chunk_text, tokenize_for_bm25


def test_chunk_text_preserves_overlap_and_order():
    chunks = chunk_text("one two three four five", chunk_words=3, overlap_words=1)

    assert [chunk["text"] for chunk in chunks] == ["one two three", "three four five"]


def test_simple_bm25_ranks_matching_document_first():
    bm25 = SimpleBM25(
        [
            tokenize_for_bm25("apple orchard harvest"),
            tokenize_for_bm25("quantum field theory"),
        ]
    )

    scores = bm25.get_scores(tokenize_for_bm25("orchard harvest"))

    assert scores[0] > scores[1]
