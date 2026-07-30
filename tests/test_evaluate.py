import json
from pathlib import Path

import pytest

from desi_fm.evaluate import write_metrics_json


def test_write_metrics_json_is_atomic_and_machine_readable(tmp_path: Path):
    output = tmp_path / "nested" / "metrics.json"
    metrics = {
        "reconstruction_rmse_masked": 0.854,
        "eta15_map": 0.273,
    }

    write_metrics_json(metrics, output)

    assert json.loads(output.read_text()) == metrics
    assert not output.with_name(f".{output.name}.tmp").exists()


def test_write_metrics_json_rejects_nonfinite_values(tmp_path: Path):
    output = tmp_path / "metrics.json"

    with pytest.raises(ValueError):
        write_metrics_json({"eta15_map": float("nan")}, output)

    assert not output.exists()
