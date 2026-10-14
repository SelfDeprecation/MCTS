"""Thread-safe instrumentation. Concurrent service times are NOT wall-time shares."""

import json
import os
import platform
import sys
import threading
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.counts = Counter()
        self.times = defaultdict(float)
        self.samples = defaultdict(list)

    def add(self, key: str, value: float = 1):
        with self.lock:
            self.counts[key] += value

    def time(self, key: str, seconds: float):
        with self.lock:
            self.times[key] += seconds

    def sample(self, key: str, value: float):
        with self.lock:
            self.samples[key].append(value)

    def snapshot(self) -> dict:
        with self.lock:
            result = {**self.counts, **{f"{k}_s": v for k, v in self.times.items()}}
            for key, values in self.samples.items():
                result[f"{key}_mean"] = float(np.mean(values))
                result[f"{key}_p95"] = float(np.percentile(values, 95))
                result[f"{key}_max"] = float(max(values))
            return result

    def reset(self):
        with self.lock:
            self.counts.clear()
            self.times.clear()
            self.samples.clear()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(value):
        value = asdict(value)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def environment() -> dict:
    import torch

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cpu_logical_count": os.cpu_count(),
        "processor": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
