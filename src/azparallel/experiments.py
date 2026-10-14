"""Paired benchmark, isolated inference sweep and controlled optimiser study."""

import csv
import json
from dataclasses import asdict
from hashlib import sha256
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .evaluator import NeuralEvaluator, RolloutEvaluator
from .game import minimax, optimal_actions, transform
from .metrics import environment, write_json
from .network import NetworkConfig, PolicyValueNet, load_model, save_model
from .optim import OptimizerBundle
from .search import SearchConfig, SearchEngine
from .training import (
    arena,
    deserialize_replay,
    evaluate_policy,
    objective,
    positions,
    resolve_device,
    sample_batch,
    seed_all,
)


def file_hash(path: Path | None):
    return sha256(path.read_bytes()).hexdigest() if path else None


def append_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def benchmark(spec: dict, out: Path, checkpoint: Path | None = None):
    if (out / "raw.csv").exists():
        raise FileExistsError("Benchmark output already exists; choose another --out")
    out.mkdir(parents=True, exist_ok=True)
    game = spec.get("game", "tictactoe")
    device = resolve_device(spec.get("device", "cpu"))
    seed_all(spec.get("model_seed", 7), spec.get("torch_threads", 1))
    kind = spec.get("evaluator", "neural")
    if kind not in ("rollout", "neural"):
        raise ValueError("Evaluator must be neural or rollout")
    model = None
    if kind == "neural":
        model = (
            load_model(checkpoint, device)
            if checkpoint
            else PolicyValueNet(NetworkConfig(game, spec.get("width", 64), spec.get("blocks", 2)))
            .to(device)
            .eval()
        )
        if model.config.game != game:
            raise ValueError("Benchmark game differs from checkpoint")
        # Preserve the actual weights even when this is an untrained scaling probe.
        save_model(model, out / "evaluated_model.pt")
    corpus = positions(game, spec.get("positions", 12), spec.get("position_seed", 2026))
    write_json(
        out / "positions.json",
        [{"board": s.board, "player": s.player, "game": s.game} for s in corpus],
    )
    write_json(out / "config.json", spec)
    write_json(out / "environment.json", environment())
    write_json(
        out / "model_metadata.json",
        {
            "source_checkpoint": str(checkpoint) if checkpoint else None,
            "source_sha256": file_hash(checkpoint),
            "model_status": "checkpoint" if checkpoint else "random",
            "network": asdict(model.config) if model else None,
        },
    )
    rows = []
    methods = spec.get("methods", ["serial", "leaf", "root", "tree", "async"])
    configs = []
    for simulations, seconds, delay in product(
        spec.get("simulations", [128]),
        spec.get("time_budgets_s", [None]),
        spec.get("synthetic_delay_ms", [0]),
    ):
        for method in methods:
            for workers in [1] if method == "serial" else spec.get("workers", [4]):
                batches = spec.get("batch_sizes", [4, 16]) if method == "async" else [1]
                losses = spec.get("virtual_losses", [1.0]) if method in ("tree", "async") else [0.0]
                backends = (
                    spec.get("rollout_backends", ["threads"])
                    if kind == "rollout" and method in ("leaf", "root")
                    else ["threads"]
                )
                for batch, vl, backend in product(batches, losses, backends):
                    if seconds is not None and backend == "processes":
                        continue
                    configs.append(
                        (
                            method,
                            SearchConfig(
                                simulations=simulations,
                                workers=workers,
                                c_puct=spec.get("c_puct", 1.5),
                                virtual_loss=vl,
                                selection="uct" if kind == "rollout" else "puct",
                                batch_size=batch,
                                batch_timeout_ms=spec.get("batch_timeout_ms", 1),
                                backend=backend,
                            ),
                            seconds,
                            delay,
                        )
                    )
    # Alternate order across repeat blocks to reduce systematic thermal/order bias.
    repeats = spec.get("repeats", 2)
    seeds = spec.get("seeds", [11, 29, 47])
    if not configs or repeats < 1 or not seeds:
        raise ValueError("Empty experiment configuration")
    for repeat in range(repeats):
        order = configs if repeat % 2 == 0 else list(reversed(configs))
        for method, config, seconds, delay in order:
            if kind == "neural" and delay:
                raise ValueError("Synthetic delay ablations apply only to rollout evaluator")
            if config.backend == "processes" and delay:
                raise ValueError("Process backend does not emulate latency")
            evaluator = (
                RolloutEvaluator(delay_ms=delay)
                if kind == "rollout"
                else NeuralEvaluator(model, device, symmetry=spec.get("symmetry", True))
            )
            with SearchEngine(evaluator, config, method) as engine:
                # Pool creation/first kernels do not enter steady-state timings.
                engine.search(corpus[0], seed=123456)
                for seed in seeds:
                    for index, state in enumerate(corpus):
                        result = engine.search(state, seed=seed + 997 * index, seconds=seconds)
                        action = result.action()
                        row = {
                            **result.stats,
                            "game": game,
                            "evaluator": kind,
                            "seed": seed,
                            "repeat": repeat,
                            "position_id": index,
                            "workers": config.workers,
                            "batch_size_limit": config.batch_size,
                            "virtual_loss": config.virtual_loss,
                            "budget": config.simulations,
                            "time_budget_s": seconds,
                            "synthetic_delay_ms": delay,
                            "action": action,
                            "checkpoint_sha256": file_hash(checkpoint) or "untrained",
                            "batch_timeout_ms": config.batch_timeout_ms,
                        }
                        if game == "tictactoe":
                            row["optimal_action"] = int(action in optimal_actions(state))
                            row["minimax_regret"] = minimax(state) + minimax(state.play(action))
                        else:
                            wins = [
                                a
                                for a in state.legal_actions()
                                if state.play(a).outcome() == state.player
                            ]
                            row["has_immediate_win"] = int(bool(wins))
                            row["takes_immediate_win"] = int(action in wins) if wins else ""
                        rows.append(row)
            append_csv(out / "raw.csv", rows)  # Flush each completed configuration.
            print(
                f"benchmark repeat={repeat} method={method} workers={config.workers} "
                f"batch={config.batch_size} backend={config.backend} time={seconds}",
                flush=True,
            )
    analyze(out)


def inference_benchmark(spec: dict, out: Path, checkpoint: Path | None = None):
    if (out / "inference.csv").exists():
        raise FileExistsError("Inference output exists")
    out.mkdir(parents=True, exist_ok=True)
    seed_all(spec.get("model_seed", 7), spec.get("torch_threads", 1))
    device = resolve_device(spec.get("device", "cpu"))
    game = spec.get("game", "connect4")
    model = (
        load_model(checkpoint, device)
        if checkpoint
        else PolicyValueNet(NetworkConfig(game, spec.get("width", 256), spec.get("blocks", 4)))
        .to(device)
        .eval()
    )
    if model.config.game != game:
        raise ValueError("Game/checkpoint mismatch")
    write_json(out / "environment.json", environment())
    write_json(out / "config.json", spec)
    save_model(model, out / "evaluated_model.pt")
    corpus = positions(game, 64)
    rows = []
    total_samples = spec.get("total_samples", 512)
    repeats = spec.get("repeats", 5)
    evaluator = NeuralEvaluator(model, device, symmetry=False)
    for size in spec.get("batch_sizes", [1, 2, 4, 8, 16, 32, 64]):
        evaluator.evaluate([corpus[0]] * size, [0] * size)
        for repeat in range(repeats):
            evaluator.metrics.reset()
            start = perf_counter()
            for offset in range(0, total_samples, size):
                count = min(size, total_samples - offset)
                evaluator.evaluate(
                    [corpus[(offset + j) % len(corpus)] for j in range(count)],
                    list(range(offset, offset + count)),
                )
            elapsed = perf_counter() - start
            rows.append(
                {
                    "batch_size": size,
                    "repeat": repeat,
                    "samples": total_samples,
                    "elapsed_s": elapsed,
                    "states_per_s": total_samples / elapsed,
                    **evaluator.metrics.snapshot(),
                }
            )
    append_csv(out / "inference.csv", rows)
    print(f"inference benchmark saved to {out}", flush=True)


def _group_key(row):
    fields = (
        "method",
        "workers",
        "batch_size_limit",
        "virtual_loss",
        "backend",
        "budget",
        "time_budget_s",
        "synthetic_delay_ms",
    )
    return tuple(row.get(k, "") for k in fields)


def analyze(out: Path):
    """Position-cluster bootstrap avoids treating repeated positions as independent."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (out / "raw.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    groups = {}
    baseline = {}

    def paired_key(r):
        return tuple(
            r.get(k, "")
            for k in (
                "budget",
                "time_budget_s",
                "synthetic_delay_ms",
                "seed",
                "repeat",
                "position_id",
            )
        )

    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)
        if row["method"] == "serial":
            baseline[paired_key(row)] = row
    summary = []
    rng = np.random.default_rng(51)
    for key, group in groups.items():
        elapsed = np.array([float(r["elapsed_s"]) for r in group])
        simulations = np.array([float(r["root_visits"]) for r in group])
        pair = [
            (float(baseline[paired_key(r)]["elapsed_s"]), float(r["elapsed_s"]))
            for r in group
            if paired_key(r) in baseline
        ]
        speedup = (sum(x for x, _ in pair) / sum(y for _, y in pair)) if pair else None
        positions_ids = sorted({r["position_id"] for r in group})
        cluster = []
        for pid in positions_ids:
            subset = [r for r in group if r["position_id"] == pid]
            base = sum(
                float(baseline[paired_key(r)]["elapsed_s"])
                for r in subset
                if paired_key(r) in baseline
            )
            current = sum(float(r["elapsed_s"]) for r in subset if paired_key(r) in baseline)
            cluster.append((base, current))
        if pair:
            samples = np.array(cluster)[rng.integers(0, len(cluster), size=(2000, len(cluster)))]
            ratios = samples[:, :, 0].sum(1) / samples[:, :, 1].sum(1)
            low, high = np.percentile(ratios, [2.5, 97.5]).tolist()
        else:
            low = high = None
        fixed_time = bool(key[6])
        entry = dict(
            zip(
                (
                    "method",
                    "workers",
                    "batch_size_limit",
                    "virtual_loss",
                    "backend",
                    "budget",
                    "time_budget_s",
                    "synthetic_delay_ms",
                ),
                key,
                strict=True,
            )
        )
        entry.update(
            n_measurements=len(group),
            n_positions=len(positions_ids),
            mean_elapsed_ms=float(elapsed.mean() * 1000),
            p95_elapsed_ms=float(np.percentile(elapsed, 95) * 1000),
            simulations_per_s=float(simulations.sum() / elapsed.sum()),
            mean_completed_simulations=float(simulations.mean()),
            latency_speedup_vs_serial=speedup,
            speedup_cluster_ci95=[low, high],
            fixed_time=fixed_time,
        )
        if fixed_time and pair:
            entry["throughput_speedup_vs_serial"] = (simulations.sum() / elapsed.sum()) / (
                sum(float(baseline[paired_key(r)]["root_visits"]) for r in group)
                / sum(float(baseline[paired_key(r)]["elapsed_s"]) for r in group)
            )
        for column in (
            "optimal_action",
            "minimax_regret",
            "evaluation_samples",
            "evaluation_calls",
            "expanded_nodes",
            "expansion_collisions",
            "batch_size_mean",
            "queue_wait_ms_mean",
            "selection_s",
            "backup_s",
            "evaluation_service_s",
            "network_forward_s",
            "tree_lock_wait_s",
            "collision_wait_s",
        ):
            vals = [float(r[column]) for r in group if r.get(column, "") != ""]
            if vals:
                entry[f"mean_{column}"] = float(np.mean(vals))
        summary.append(entry)
    write_json(out / "summary.json", summary)
    simple = [{**s, "speedup_cluster_ci95": json.dumps(s["speedup_cluster_ci95"])} for s in summary]
    append_csv(out / "summary.csv", simple)
    labels = [
        f"{s['method']} w={s['workers']} b={s['batch_size_limit']} "
        f"vl={s['virtual_loss']} {s['backend']} N={s['budget']} t={s['time_budget_s']}"
        for s in summary
    ]
    fig, ax = plt.subplots(figsize=(11, max(4, len(summary) * 0.35)))
    ax.barh(labels, [s["simulations_per_s"] for s in summary])
    ax.set_xlabel("Completed backup units per second")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "throughput.png", dpi=160)
    plt.close(fig)
    if summary and "mean_optimal_action" in summary[0]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, s in zip(labels, summary, strict=True):
            ax.scatter(s["simulations_per_s"], s["mean_optimal_action"])
            ax.annotate(label, (s["simulations_per_s"], s["mean_optimal_action"]), fontsize=6)
        ax.set_xlabel("Completed backup units per second")
        ax.set_ylabel("Fraction of minimax-optimal actions")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(out / "speed_quality.png", dpi=160)
        plt.close(fig)
    text = [
        "# Результаты измерений",
        "",
        "Каждая конфигурация проверена на одинаковых достижимых позициях и начальных числах генератора.",
        "У Root бюджет распределён между деревьями, у Leaf учитываются все повторные оценки листа.",
        "Число фактических оценок сохраняется отдельно: терминальные возвраты не требуют нейросети.",
        "Время включает инициализацию корня и завершение уже принятых запросов; прогрев исключён.",
        "95% интервал ускорения получен бутстрэпом по позициям. Он не описывает вариацию между машинами.",
        "Суммарные времена параллельных фаз могут перекрываться и не являются долями общего времени.",
        "",
        "| Метод | Работники | Пакет | Бюджет | Время, мс | Возвратов/с | Ускорение | Оптимальные ходы |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for s in summary:
        rate = f"{s['mean_optimal_action']:.3f}" if "mean_optimal_action" in s else "—"
        acceleration = (
            f"{s['latency_speedup_vs_serial']:.2f}"
            if not s["fixed_time"] and s["latency_speedup_vs_serial"]
            else f"{s.get('throughput_speedup_vs_serial', 1):.2f} (по пропускной способности)"
        )
        text.append(
            f"| {s['method']} ({s['backend']}, VL={s['virtual_loss']}) | {s['workers']} | "
            f"{s['batch_size_limit']} | {s['time_budget_s'] or s['budget']} | "
            f"{s['mean_elapsed_ms']:.2f} | {s['simulations_per_s']:.0f} | "
            f"{acceleration} | {rate} |"
        )
    (out / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def optimizer_study(spec: dict, replay_checkpoint: Path, out: Path):
    if (out / "raw.json").exists():
        raise FileExistsError("Optimizer output exists")
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(spec.get("device", "cpu"))
    payload = torch.load(replay_checkpoint, map_location="cpu", weights_only=True)
    examples = deserialize_replay(payload["replay"])
    net_config = NetworkConfig(**payload["network_config"])

    # Group all symmetry-equivalent states, so train/validation do not share a board.
    def canonical(example):
        state = example.state
        count = 8 if state.game == "tictactoe" else 2
        return min(tuple(transform(state, s).encode().ravel().tolist()) for s in range(count))

    keys = sorted({canonical(x) for x in examples})
    split_rng = np.random.default_rng(2027)
    split_rng.shuffle(keys)
    held_out = set(keys[: max(1, len(keys) // 5)])
    train_examples = [x for x in examples if canonical(x) not in held_out]
    valid_examples = [x for x in examples if canonical(x) in held_out]
    if not train_examples or not valid_examples:
        raise ValueError("Replay too small for a grouped validation split")
    write_json(out / "environment.json", environment())
    write_json(
        out / "config.json",
        {
            **spec,
            "protocol": "offline_fixed_replay",
            "replay_sha256": file_hash(replay_checkpoint),
            "train_examples": len(train_examples),
            "validation_examples": len(valid_examples),
            "held_out_canonical_positions": len(held_out),
        },
    )
    rows = []
    curves = []
    for seed in spec.get("seeds", [11, 29, 47]):
        for name in spec.get("optimizers", ["adamw", "asam", "muon"]):
            seed_all(seed, spec.get("torch_threads", 1))
            model = PolicyValueNet(net_config).to(device)
            optimizer = OptimizerBundle(
                model,
                name,
                spec.get("lr", 1e-3),
                spec.get("weight_decay", 1e-4),
                spec.get("grad_clip", 1),
                spec.get("asam_rho", 0.1),
                spec.get("asam_eta", 0.01),
                spec.get("muon_lr", 0.02),
            )
            rng = np.random.default_rng(seed)
            validation_rng = np.random.default_rng(909)
            valid_batch = sample_batch(
                valid_examples,
                validation_rng,
                min(2048, len(valid_examples)),
                device,
                augment=False,
            )
            started = perf_counter()
            for step in range(spec.get("steps", 200)):
                batch = sample_batch(train_examples, rng, spec.get("batch_size", 64), device)
                stats = optimizer.step(lambda model=model, batch=batch: objective(model, batch))
                if step % 20 == 0:
                    with torch.no_grad():
                        val, _ = objective(model, valid_batch)
                    curves.append(
                        {
                            "seed": seed,
                            "optimizer": name,
                            "step": step,
                            **stats,
                            "validation_objective": float(val),
                        }
                    )
            training_s = perf_counter() - started
            model.eval()
            with torch.no_grad():
                val, parts = objective(model, valid_batch)
            row = {
                "seed": seed,
                "optimizer": name,
                "training_s": training_s,
                "validation_objective": float(val),
                "validation_parts": parts,
                "policy_quality": evaluate_policy(
                    model, net_config.game, count=spec.get("eval_positions", 100), device=device
                ),
            }
            with SearchEngine(
                NeuralEvaluator(model, device, symmetry=False),
                SearchConfig(simulations=spec.get("eval_simulations", 32)),
                "serial",
            ) as engine:
                row["arena_policy_random"] = arena(
                    engine, net_config.game, spec.get("eval_games", 40), policy_only=True
                )
                row["arena_search_random"] = arena(
                    engine, net_config.game, spec.get("eval_games", 40)
                )
            save_model(model, out / f"{name}_seed{seed}.pt")
            rows.append(row)
            write_json(out / "raw.json", rows)
            append_csv(out / "curves.csv", curves)
            print(json.dumps(row, ensure_ascii=False), flush=True)
