import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "train_oracle_teacher.py"


def test_train_oracle_teacher_help_runs():
    proc = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=True)
    assert "Teach a student policy" in proc.stdout


def test_train_oracle_teacher_search_mode_outputs_json(tmp_path):
    output = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--search",
            "--seed",
            "1",
            "--num-seeds",
            "1",
            "--round-sizes",
            "1",
            "--iterations-candidates",
            "1,2",
            "--rollouts-candidates",
            "1",
            "--output-json",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rendered = json.loads(proc.stdout)
    assert rendered["iterations"] in {1, 2}
    assert rendered["rollouts_per_action"] == 1
    assert output.exists()
