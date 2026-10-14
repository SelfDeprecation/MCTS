"""Self-play -> replay -> policy/value optimisation, with resumable checkpoints."""

import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .evaluator import NeuralEvaluator
from .game import State, action_permutation, initial_state, optimal_actions, transform
from .metrics import environment, write_json
from .network import NetworkConfig, PolicyValueNet, save_model
from .optim import OptimizerBundle
from .search import SearchConfig, SearchEngine


@dataclass
class TrainConfig:
    game: str = "tictactoe"
    seed: int = 7
    device: str = "cpu"
    iterations: int = 12
    games_per_iteration: int = 24
    updates_per_iteration: int = 80
    batch_size: int = 64
    replay_capacity: int = 20_000
    width: int = 64
    blocks: int = 2
    simulations: int = 48
    method: str = "serial"
    workers: int = 4
    inference_batch: int = 16
    batch_timeout_ms: float = 1.0
    c_puct: float = 1.5
    virtual_loss: float = 1.0
    noise_fraction: float = 0.25
    dirichlet_alpha: float = 0.3
    temperature_moves: int = 6
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    asam_rho: float = 0.1
    asam_eta: float = 0.01
    muon_lr: float = 0.02
    torch_threads: int = 1
    eval_games: int = 24
    eval_positions: int = 100
    eval_every: int = 2


@dataclass
class Example:
    state: State
    policy: np.ndarray
    value: float


def seed_all(seed: int, torch_threads: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(torch_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use CPU or a GPU Colab runtime.")
    if device not in ("cpu", "cuda"):
        raise ValueError("Device must be cpu, cuda, or auto")
    return device


def positions(game: str, count: int, seed: int = 2026) -> list[State]:
    """Reachable nonterminal positions, sampled independently of training."""
    rng = np.random.default_rng(seed)
    result = []
    seen = set()
    while len(result) < count:
        state = initial_state(game)
        moves = int(rng.integers(1, 7 if game == "tictactoe" else 22))
        for _ in range(moves):
            if state.outcome() is not None:
                break
            state = state.play(int(rng.choice(state.legal_actions())))
        if state.outcome() is None and state not in seen:
            seen.add(state)
            result.append(state)
    return result


def self_play(config: TrainConfig, engine: SearchEngine, rng) -> list[Example]:
    state = initial_state(config.game)
    history = []
    while state.outcome() is None:
        result = engine.search(state, int(rng.integers(0, 2**31)))
        history.append((state, result.policy.astype(np.float32)))
        if len(history) <= config.temperature_moves:
            action = int(rng.choice(state.action_size, p=result.policy))
        else:
            tied = np.flatnonzero(result.visits == result.visits.max())
            action = int(rng.choice(tied))
        state = state.play(action)
    return [Example(s, p, float(state.outcome() * s.player)) for s, p in history]


def sample_batch(examples: list[Example], rng, size: int, device: str, augment=True):
    indices = rng.integers(0, len(examples), size=size)
    xs, ps, zs, masks = [], [], [], []
    for index in indices:
        example = examples[int(index)]
        sym = int(rng.integers(8 if example.state.game == "tictactoe" else 2)) if augment else 0
        permutation = action_permutation(example.state, sym)
        state = transform(example.state, sym)
        policy = np.empty_like(example.policy)
        policy[permutation] = example.policy
        mask = np.zeros(state.action_size, dtype=bool)
        mask[list(state.legal_actions())] = True
        xs.append(state.encode())
        ps.append(policy)
        zs.append(example.value)
        masks.append(mask)
    return (
        torch.tensor(np.stack(xs), device=device),
        torch.tensor(np.stack(ps), device=device),
        torch.tensor(zs, dtype=torch.float32, device=device),
        torch.tensor(np.stack(masks), device=device),
    )


def objective(model, batch):
    x, target_policy, target_value, legal = batch
    logits, values = model(x)
    # Legal-only policy targets and probabilities; illegal logits cannot affect CE.
    logp = logits.masked_fill(~legal, -1e9).log_softmax(-1)
    policy_error = -(target_policy * logp).sum(-1).mean()
    value_error = (values - target_value).square().mean()
    loss = policy_error + value_error  # Weight decay is handled by optimizer.
    return loss, {
        "policy_error": float(policy_error.detach()),
        "value_error": float(value_error.detach()),
    }


def evaluate_policy(model, game: str, seed=8080, count=100, device="cpu") -> dict:
    evaluator = NeuralEvaluator(model, device, symmetry=False)
    states = positions(game, count, seed)
    outputs = evaluator.evaluate(states, [seed + i for i in range(len(states))])
    if game != "tictactoe":
        return {"positions": count, "oracle_available": False}
    correct, regret, value_errors = [], [], []
    from .game import minimax

    for state, output in zip(states, outputs, strict=True):
        action = int(np.argmax(output.policy))
        correct.append(action in optimal_actions(state))
        regret.append(minimax(state) + minimax(state.play(action)))
        value_errors.append((output.value - minimax(state)) ** 2)
    return {
        "positions": count,
        "optimal_action_rate": float(np.mean(correct)),
        "mean_minimax_regret": float(np.mean(regret)),
        "value_mse_vs_minimax": float(np.mean(value_errors)),
    }


def arena(engine, game: str, games=24, seed=9001, opponent="random", policy_only=False):
    if opponent == "minimax" and game != "tictactoe":
        raise ValueError("Minimax arena is available only for tic-tac-toe")
    rng = np.random.default_rng(seed)
    outcomes = []
    for i in range(games):
        state = initial_state(game)
        colour = 1 if i % 2 == 0 else -1
        while state.outcome() is None:
            if state.player == colour:
                if policy_only:
                    action = int(
                        np.argmax(engine.evaluator.evaluate([state], [seed + i])[0].policy)
                    )
                else:
                    action = engine.search(state, int(rng.integers(0, 2**31))).action()
            else:
                choices = optimal_actions(state) if opponent == "minimax" else state.legal_actions()
                action = int(rng.choice(choices))
            state = state.play(action)
        outcomes.append(int(state.outcome() * colour))
    wins = outcomes.count(1)
    draws = outcomes.count(0)
    return {
        "games": games,
        "opponent": opponent,
        "policy_only": policy_only,
        "wins": wins,
        "draws": draws,
        "losses": outcomes.count(-1),
        "score_rate": (wins + draws / 2) / games,
        "mean_reward": float(np.mean(outcomes)),
        "outcomes": outcomes,
    }


def serialize_replay(replay):
    return [
        {
            "board": list(x.state.board),
            "player": x.state.player,
            "game": x.state.game,
            "policy": x.policy.tolist(),
            "value": x.value,
        }
        for x in replay
    ]


def deserialize_replay(payload):
    return [
        Example(
            State(tuple(x["board"]), x["player"], x["game"]),
            np.array(x["policy"], dtype=np.float32),
            x["value"],
        )
        for x in payload
    ]


def train(config: TrainConfig, out: Path, resume: Path | None = None):
    if (
        min(
            config.iterations,
            config.games_per_iteration,
            config.updates_per_iteration,
            config.batch_size,
            config.replay_capacity,
            config.eval_games,
            config.eval_positions,
            config.eval_every,
            config.torch_threads,
        )
        < 1
    ):
        raise ValueError("Training sizes must be positive")
    if out.exists() and (out / "train.jsonl").exists() and resume is None:
        raise FileExistsError("Training output exists. Use another --out or --resume.")
    device = resolve_device(config.device)
    seed_all(config.seed, config.torch_threads)
    rng = np.random.default_rng(config.seed)
    out.mkdir(parents=True, exist_ok=True)
    model = PolicyValueNet(NetworkConfig(config.game, config.width, config.blocks)).to(device)
    optimizer = OptimizerBundle(
        model,
        config.optimizer,
        config.lr,
        config.weight_decay,
        config.grad_clip,
        config.asam_rho,
        config.asam_eta,
        config.muon_lr,
    )
    replay = deque(maxlen=config.replay_capacity)
    first_iteration = 0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        saved_config = checkpoint["train_config"]
        for key, value in asdict(config).items():
            if key not in ("iterations", "device") and saved_config[key] != value:
                raise ValueError(f"Resume mismatch: {key}")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        replay.extend(deserialize_replay(checkpoint["replay"]))
        rng.bit_generator.state = checkpoint["numpy_rng"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint["cuda_rng"]])
        first_iteration = checkpoint["iteration"]
    else:
        save_model(model, out / "initial.pt")
        write_json(
            out / "initial_quality.json",
            evaluate_policy(model, config.game, count=config.eval_positions, device=device),
        )
    write_json(out / "config.json", config)
    write_json(out / "environment.json", environment())
    search_config = SearchConfig(
        simulations=config.simulations,
        workers=config.workers,
        c_puct=config.c_puct,
        virtual_loss=config.virtual_loss,
        batch_size=config.inference_batch,
        batch_timeout_ms=config.batch_timeout_ms,
        root_noise=config.noise_fraction,
        dirichlet_alpha=config.dirichlet_alpha,
    )
    evaluator = NeuralEvaluator(model, device, symmetry=True)
    with SearchEngine(evaluator, search_config, config.method) as engine:
        for iteration in range(first_iteration, config.iterations):
            started = perf_counter()
            model.eval()
            added = 0
            for _ in range(config.games_per_iteration):
                examples = self_play(config, engine, rng)
                replay.extend(examples)
                added += len(examples)
            selfplay_s = perf_counter() - started
            examples = list(replay)
            model.train()
            updates = []
            train_start = perf_counter()
            for _ in range(config.updates_per_iteration):
                batch = sample_batch(examples, rng, config.batch_size, device)
                updates.append(
                    optimizer.step(lambda model=model, batch=batch: objective(model, batch))
                )
            model.eval()
            record = {
                "iteration": iteration + 1,
                "replay_size": len(replay),
                "new_examples": added,
                "selfplay_s": selfplay_s,
                "training_s": perf_counter() - train_start,
                **{k: float(np.mean([u[k] for u in updates])) for k in updates[0]},
            }
            if (iteration + 1) % config.eval_every == 0 or iteration + 1 == config.iterations:
                record["policy_quality"] = evaluate_policy(
                    model, config.game, count=config.eval_positions, device=device
                )
                eval_config = replace_search_without_noise(search_config)
                with SearchEngine(
                    NeuralEvaluator(model, device, symmetry=False), eval_config, "serial"
                ) as eval_engine:
                    record["arena_random"] = arena(eval_engine, config.game, config.eval_games)
                    if config.game == "tictactoe":
                        record["arena_minimax"] = arena(
                            eval_engine, config.game, config.eval_games, opponent="minimax"
                        )
            with (out / "train.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            save_model(
                model,
                out / "latest.pt",
                iteration=iteration + 1,
                train_config=asdict(config),
                optimizer_state=optimizer.state_dict(),
                replay=serialize_replay(replay),
                numpy_rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device == "cuda" else None,
            )
            print(json.dumps(record, ensure_ascii=False), flush=True)
    return model


def replace_search_without_noise(config):
    from dataclasses import replace

    return replace(config, root_noise=0.0)
