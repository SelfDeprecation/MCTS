import json

import pytest

from azparallel.experiments import benchmark
from azparallel.network import NetworkConfig, PolicyValueNet, save_model
from azparallel.training import seed_all


def test_benchmark_preserves_raw_data_and_reports_paired_comparison(tmp_path):
    seed_all(7)
    checkpoint = tmp_path / "network.pt"
    save_model(PolicyValueNet(NetworkConfig(width=8, blocks=1)), checkpoint)
    spec = {
        "game": "tictactoe",
        "evaluator": "neural",
        "methods": ["serial", "async"],
        "workers": [2],
        "batch_sizes": [2],
        "positions": 2,
        "seeds": [3],
        "repeats": 1,
        "simulations": [8],
    }
    out = tmp_path / "study"
    benchmark(spec, out, checkpoint)
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary) == 2
    assert summary[0]["latency_speedup_vs_serial"] == pytest.approx(1)
    assert summary[1]["n_positions"] == 2
    assert summary[1]["speedup_cluster_ci95"][0] > 0
    assert (out / "raw.csv").exists()
    assert (out / "REPORT.md").exists()
    assert (out / "throughput.png").stat().st_size > 0
