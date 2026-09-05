import json
from datetime import datetime, timezone

from benchmark.output import write_run_metadata


def test_runtime_metadata_records_wall_clock_fields(tmp_path):
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    finished = datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)

    write_run_metadata(tmp_path, started, finished, 5.0, "completed", ["skillgraph"], ["mix"], 3)

    metadata = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["elapsed_sec"] == 5.0
    assert metadata["status"] == "completed"
    assert metadata["methods"] == ["skillgraph"]
