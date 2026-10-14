"""Run the CPU study sequentially; concurrent experiments contaminate timings."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/cpu_study"))
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.out.resolve()
    train_dir = output / "training"
    checkpoint = train_dir / "latest.pt"

    def run(*arguments):
        subprocess.run(
            [sys.executable, "-m", "azparallel", *map(str, arguments)], check=True, cwd=root
        )

    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        planned = json.loads((root / "configs/train_cpu.json").read_text())["iterations"]
        if saved["iteration"] < planned:
            run(
                "train",
                "--config",
                root / "configs/train_cpu.json",
                "--out",
                train_dir,
                "--resume",
                checkpoint,
            )
    else:
        run("train", "--config", root / "configs/train_cpu.json", "--out", train_dir)
    jobs = [
        ("benchmark", "benchmark_cpu.json", "search"),
        ("benchmark", "benchmark_rollout.json", "rollout"),
        ("benchmark", "benchmark_time.json", "time"),
        ("inference", "inference_cpu.json", "inference"),
        ("optimizers", "optimizers_cpu.json", "optimizers"),
    ]
    if args.ablation:
        jobs.append(("benchmark", "ablation_cpu.json", "ablation"))
    for command, config, folder in jobs:
        target = output / folder
        marker = (
            "raw.json"
            if command == "optimizers"
            else ("inference.csv" if command == "inference" else "summary.json")
        )
        if (target / marker).exists():
            if command == "optimizers":
                parameters = json.loads((root / "configs" / config).read_text())
                expected = len(parameters["optimizers"]) * len(parameters["seeds"])
                if len(json.loads((target / marker).read_text())) != expected:
                    raise RuntimeError(
                        f"Incomplete optimizer experiment: {target}. "
                        "Preserve it and use a new output directory."
                    )
            print(f"Skip completed experiment: {target}", flush=True)
            continue
        arguments = [command, "--config", root / "configs" / config, "--out", target]
        if command == "optimizers":
            arguments += ["--replay", checkpoint]
        elif command == "benchmark" and "rollout" not in config:
            arguments += ["--checkpoint", checkpoint]
        run(*arguments)


if __name__ == "__main__":
    main()
