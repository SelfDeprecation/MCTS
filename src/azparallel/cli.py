"""One portable command surface for Windows, Linux and Google Colab."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .evaluator import NeuralEvaluator
from .experiments import analyze, benchmark, inference_benchmark, optimizer_study
from .game import initial_state
from .metrics import environment, write_json
from .network import load_model
from .search import SearchConfig, SearchEngine
from .training import TrainConfig, arena, resolve_device, seed_all, train


def read_config(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Parallel AlphaZero study")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Environment and CUDA availability")
    doctor.add_argument("--device", default="auto")
    for name in ("train", "benchmark", "inference", "optimizers"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--device", choices=("cpu", "cuda", "auto"))
        if name == "train":
            p.add_argument("--resume", type=Path)
        elif name == "optimizers":
            p.add_argument("--replay", type=Path, required=True)
        else:
            p.add_argument("--checkpoint", type=Path)
    p = sub.add_parser("analyze")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("arena")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--opponent", choices=("random", "minimax"), default="random")
    p.add_argument(
        "--method", choices=("serial", "leaf", "root", "tree", "async"), default="serial"
    )
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--simulations", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--policy-only", action="store_true")
    p.add_argument("--seed", type=int, default=9001)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path)
    p = sub.add_parser("play")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--human-colour", type=int, choices=(-1, 1), default=1)
    p.add_argument("--simulations", type=int, default=128)
    p.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "doctor":
        print(
            json.dumps({**environment(), "selected_device": resolve_device(args.device)}, indent=2)
        )
    elif args.command in ("train", "benchmark", "inference", "optimizers"):
        config = read_config(args.config)
        if args.device:
            config["device"] = args.device
        if args.command == "train":
            train(TrainConfig(**config), args.out, args.resume)
        elif args.command == "benchmark":
            benchmark(config, args.out, args.checkpoint)
        elif args.command == "inference":
            inference_benchmark(config, args.out, args.checkpoint)
        else:
            optimizer_study(config, args.replay, args.out)
    elif args.command == "analyze":
        analyze(args.out)
    else:
        seed_all(getattr(args, "seed", 9001))
        device = resolve_device(args.device)
        model = load_model(args.checkpoint, device)
        config = SearchConfig(simulations=args.simulations, workers=getattr(args, "workers", 4))
        with SearchEngine(
            NeuralEvaluator(model, device, symmetry=False),
            config,
            getattr(args, "method", "serial"),
        ) as engine:
            if args.command == "arena":
                result = arena(
                    engine,
                    model.config.game,
                    args.games,
                    args.seed,
                    args.opponent,
                    args.policy_only,
                )
                print(json.dumps(result, indent=2))
                if args.out:
                    write_json(
                        args.out,
                        {"result": result, "search": asdict(config), "environment": environment()},
                    )
            else:
                state = initial_state(model.config.game)
                while state.outcome() is None:
                    print(state.render())
                    if state.player == args.human_colour:
                        try:
                            action = int(input(f"Legal actions {state.legal_actions()}: "))
                            state = state.play(action)
                        except ValueError as exc:
                            print(exc)
                    else:
                        action = engine.search(
                            state, 9001 + sum(x != 0 for x in state.board)
                        ).action()
                        print(f"AlphaZero plays {action}")
                        state = state.play(action)
                print(state.render(), "Outcome:", state.outcome())


if __name__ == "__main__":
    main()
