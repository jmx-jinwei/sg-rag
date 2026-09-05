from pathlib import Path

from benchmark.dataset import load_domain_rows, select_domains


def test_load_domain_rows_normalizes_common_question_and_answer_keys(tmp_path: Path):
    (tmp_path / "agriculture.jsonl").write_text(
        '{"question":"Who found it?","answer":"Ada","context":"Ada found it."}\n',
        encoding="utf-8",
    )

    rows = load_domain_rows(tmp_path, "agriculture")

    assert rows[0]["_id"] == "sample-000001"
    assert rows[0]["input"] == "Who found it?"
    assert rows[0]["answers"] == ["Ada"]
    assert select_domains(tmp_path, ["all"]) == ["agriculture"]
