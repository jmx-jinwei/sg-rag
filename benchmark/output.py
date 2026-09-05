"""Output and runtime metadata helpers for benchmark runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def write_run_metadata(
    output_dir: str | Path,
    started_at: datetime,
    finished_at: datetime,
    elapsed_sec: float,
    status: str,
    methods: list[str],
    domains: list[str],
    limit: int | None,
    error: str | None = None,
) -> None:
    """Write wall-clock runtime separately from per-query and metric timings."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "elapsed_sec": round(elapsed_sec, 3),
        "status": status,
        "methods": methods,
        "domains": domains,
        "limit": limit,
    }
    if error:
        metadata["error"] = error
    with (directory / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
