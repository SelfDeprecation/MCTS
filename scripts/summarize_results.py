"""Combine completed artifacts into readable tables and export research figures."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def wilson(k, n):
    if n == 0:
        return 0, 1
    z = 1.959964
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(center - radius), float(center + radius)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results"))
    args = parser.parse_args()
    root = args.root
    lines = [
        "# Выполненные эксперименты",
        "",
        "Этот файл и графики построены из сохранённых измерений. GPU-результаты сюда не включены, если GPU-запуск не выполнен.",
        "",
    ]
    train_dir = root / "cpu_training"
    if (train_dir / "train.jsonl").exists():
        records = [
            json.loads(line) for line in (train_dir / "train.jsonl").read_text().splitlines()
        ]
        initial = read(train_dir / "initial_quality.json")
        final = records[-1]
        lines += [
            "## Обучение",
            "",
            f"Итераций: {len(records)}. Партии: {len(records) * read(train_dir / 'config.json')['games_per_iteration']}. "
            f"Примеров в буфере: {final['replay_size']}.",
            f"Доля оптимальных ходов чистой сети на мониторинговых 100 позициях: "
            f"{initial['optimal_action_rate']:.0%} → {final['policy_quality']['optimal_action_rate']:.0%}.",
            "Точный решатель не участвовал в обучении. Один запуск не оценивает разброс результатов самостоятельной игры.",
            "",
        ]
        evaluated = [r for r in records if "policy_quality" in r]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(
            [0] + [r["iteration"] for r in evaluated],
            [initial["optimal_action_rate"]]
            + [r["policy_quality"]["optimal_action_rate"] for r in evaluated],
            marker="o",
        )
        axes[0].set(
            xlabel="Self-play iteration", ylabel="Minimax-optimal policy actions", ylim=(0, 1)
        )
        axes[1].plot(
            [r["iteration"] for r in records],
            [r["policy_error"] for r in records],
            label="Policy CE",
        )
        axes[1].plot(
            [r["iteration"] for r in records],
            [r["value_error"] for r in records],
            label="Value MSE",
        )
        axes[1].set(xlabel="Self-play iteration", ylabel="Training error")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(root / "training.png", dpi=160)
        plt.close(fig)
    for folder, title in (
        ("cpu_search", "Нейросетевой поиск при одинаковом объёме"),
        ("cpu_rollout", "Классические доигрывания"),
        ("cpu_time", "Поиск при одинаковом времени"),
    ):
        path = root / folder / "summary.json"
        if not path.exists():
            continue
        lines += [
            f"## {title}",
            "",
            "| Метод | Работники | Пакет | мс | Возвратов/с | Ускорение | Оптимальные действия |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for s in read(path):
            acceleration = (
                s.get("throughput_speedup_vs_serial")
                if s["fixed_time"]
                else s["latency_speedup_vs_serial"]
            )
            lines.append(
                f"| {s['method']} ({s['backend']}) | {s['workers']} | {s['batch_size_limit']} | "
                f"{s['mean_elapsed_ms']:.2f} | {s['simulations_per_s']:.0f} | "
                f"{acceleration:.2f} | {s.get('mean_optimal_action', 0):.3f} |"
            )
        lines += [
            "",
            "Подробности, интервалы и исходные данные находятся в каталоге этого эксперимента.",
            "",
        ]
    path = root / "cpu_optimizers" / "raw.json"
    if path.exists():
        rows = read(path)
        names = sorted({r["optimizer"] for r in rows})
        lines += [
            "## Оптимизаторы на фиксированном буфере",
            "",
            "Начальные веса и мини-пакеты совпадают внутри каждого начального числа генератора. "
            "Проверка отделена по позициям и симметриям. ASAM выполняет два прохода на шаг.",
            "",
            "| Оптимизатор | Проверочная цель (среднее ± std) | Оптимальные действия | Время обучения, с | Награда без поиска |",
            "|---|---:|---:|---:|---:|",
        ]
        objectives, errors = [], []
        for name in names:
            subset = [r for r in rows if r["optimizer"] == name]
            values = [r["validation_objective"] for r in subset]
            objectives.append(np.mean(values))
            errors.append(np.std(values, ddof=1) if len(values) > 1 else 0)
            lines.append(
                f"| {name} | {np.mean(values):.3f} ± {errors[-1]:.3f} | "
                f"{np.mean([r['policy_quality']['optimal_action_rate'] for r in subset]):.3f} | "
                f"{np.mean([r['training_s'] for r in subset]):.2f} | "
                f"{np.mean([r['arena_policy_random']['mean_reward'] for r in subset]):.3f} |"
            )
        lines += [
            "",
            "Это предварительное сравнение при выбранных параметрах, без поиска оптимальной величины шага каждого метода. "
            "Стандартное отклонение трёх запусков не является доказательством превосходства. "
            "Рост награды на 15% не предполагается и не гарантируется.",
            "",
        ]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(names, objectives, yerr=errors, capsize=5)
        ax.set(
            ylabel="Validation policy CE + value MSE",
            title="Fixed-replay training: mean ± sample std",
        )
        fig.tight_layout()
        fig.savefig(root / "optimizers.png", dpi=160)
        plt.close(fig)
    lines += ["## Независимая оценка", ""]
    for path in sorted((root / "arenas").glob("*.json")) if (root / "arenas").exists() else []:
        r = read(path)["result"]
        lines.append(
            f"- {path.stem}: {r['games']} партий, побед/ничьих/поражений "
            f"{r['wins']}/{r['draws']}/{r['losses']}, доля очков {r['score_rate']:.3f}, "
            f"награда {r['mean_reward']:.3f}."
        )
    if (root / "cpu_holdout.json").exists():
        holdout = read(root / "cpu_holdout.json")
        lines += [
            "",
            f"Отдельный корпус: {holdout['positions']} позиций, генератор {holdout['seed']}. "
            f"Доля оптимальных действий сети до/после: "
            f"{holdout['initial']['optimal_action_rate']:.3f}/{holdout['trained']['optimal_action_rate']:.3f}.",
        ]
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Isolated inference plot; the measured latency includes encoding and transfers.
    path = root / "cpu_inference_isolated" / "inference.csv"
    if path.exists():
        with path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        sizes = sorted({int(r["batch_size"]) for r in rows})
        speeds = [
            np.mean([float(r["states_per_s"]) for r in rows if int(r["batch_size"]) == size])
            for size in sizes
        ]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(sizes, speeds, marker="o")
        ax.set(
            xlabel="Actual batch size",
            ylabel="Evaluated states per second",
            title="CPU inference sweep",
        )
        ax.set_xscale("log", base=2)
        fig.tight_layout()
        fig.savefig(root / "inference.png", dpi=160)
        plt.close(fig)
    print(root / "SUMMARY.md")


if __name__ == "__main__":
    main()
