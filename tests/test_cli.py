import subprocess
import sys
from pathlib import Path


def test_runner_help_is_available_without_model_paths():
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "benchmark.run", "--help"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "sg_rag" in completed.stdout
    assert "lightrag" not in completed.stdout.lower()
