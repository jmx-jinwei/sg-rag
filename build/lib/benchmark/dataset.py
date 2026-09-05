"""UltraDomain-compatible dataset loading and normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def normalize_row(row: dict[str, Any], sample_index: int) -> dict[str, Any]:
    """Normalize one JSONL row while preserving extra metadata fields."""
    question = row.get("input", row.get("question"))
    if question is None or not str(question).strip():
        raise ValueError(f"dataset row {sample_index} has no input/question")

    answers = row.get("answers", row.get("answer", row.get("gold_answer", [])))
    if isinstance(answers, str):
        answers = [answers]
    elif answers is None:
        answers = []
    else:
        answers = [str(answer) for answer in answers]

    context = row.get("context", "")
    normalized = dict(row)
    normalized["_id"] = row.get("_id") or f"sample-{sample_index:06d}"
    normalized["input"] = str(question)
    normalized["answers"] = answers
    normalized["context"] = "" if context is None else str(context)
    return normalized


def load_domain_rows(dataset_dir: str | Path, domain: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Load one ``<domain>.jsonl`` file and normalize its rows."""
    path = Path(dataset_dir) / f"{domain}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(normalize_row(json.loads(line), len(rows) + 1))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def select_domains(dataset_dir: str | Path, domains: list[str]) -> list[str]:
    """Return requested domains, or all JSONL stems for the special ``all`` value."""
    if domains == ["all"]:
        return sorted(path.stem for path in Path(dataset_dir).glob("*.jsonl"))
    return list(dict.fromkeys(domains))

