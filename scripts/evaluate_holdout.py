"""Assess the initial and trained networks on a separate corpus and arenas."""

import argparse
from pathlib import Path

from azparallel.evaluator import NeuralEvaluator
from azparallel.metrics import write_json
from azparallel.network import load_model
from azparallel.search import SearchConfig, SearchEngine
from azparallel.training import arena, evaluate_policy, seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, default=Path("results/cpu_training"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--positions", type=int, default=500)
    parser.add_argument("--simulations", type=int, default=128)
    args = parser.parse_args()
    seed_all(7)
    summary = {
        "seed": 404,
        "positions": args.positions,
        "description": "Separate evaluation corpus; not used for training or model selection",
    }
    for label, name in (("initial", "initial.pt"), ("trained", "latest.pt")):
        model = load_model(args.training / name)
        summary[label] = evaluate_policy(model, model.config.game, seed=404, count=args.positions)
        with SearchEngine(
            NeuralEvaluator(model, symmetry=False), SearchConfig(simulations=args.simulations)
        ) as engine:
            cases = [("random", True), ("random", False), ("minimax", False)]
            if model.config.game != "tictactoe":
                cases = cases[:2]
            for opponent, policy_only in cases:
                result = arena(
                    engine,
                    model.config.game,
                    args.games,
                    seed=4040,
                    opponent=opponent,
                    policy_only=policy_only,
                )
                key = f"{label}_{opponent}_{'policy' if policy_only else 'search'}"
                write_json(
                    args.out / "arenas" / f"{key}.json",
                    {"result": result, "simulations": args.simulations, "seed": 4040},
                )
                print(key, result["wins"], result["draws"], result["losses"], flush=True)
    write_json(args.out / "cpu_holdout.json", summary)


if __name__ == "__main__":
    main()
