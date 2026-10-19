"""The README's demo script runs and produces its figure."""

import subprocess
import sys
from pathlib import Path

DEMO = Path(__file__).parents[1] / "examples" / "demo.py"


def test_demo_runs(tmp_path):
    """examples/demo.py runs end to end and saves demo.png."""
    subprocess.run([sys.executable, str(DEMO), "--output", str(tmp_path)], check=True)
    assert (tmp_path / "demo.png").stat().st_size > 0
