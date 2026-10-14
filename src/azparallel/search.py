"""PUCT/UCT search with explicitly different Leaf, Root and Tree strategies."""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from math import log, sqrt
from threading import Event, Lock
from time import perf_counter

import numpy as np

from .evaluator import Evaluation, InferenceServer, RolloutEvaluator, process_rollout
from .game import State
from .metrics import Metrics

METHODS = ("serial", "leaf", "root", "tree", "async")


@dataclass
class SearchConfig:
    simulations: int = 64
    workers: int = 4
    c_puct: float = 1.5
    virtual_loss: float = 1.0
    selection: str = "puct"
    batch_size: int = 16
    batch_timeout_ms: float = 1.0
    root_noise: float = 0.0
    dirichlet_alpha: float = 0.3
    backend: str = "threads"

    def validate(self):
        if self.simulations < 1 or self.workers < 1 or self.batch_size < 1:
            raise ValueError("Budgets, workers and batch size must be positive")
        if self.selection not in ("puct", "uct") or self.backend not in ("threads", "processes"):
            raise ValueError("Invalid selection/backend")
        if self.virtual_loss < 0 or self.c_puct <= 0 or self.batch_timeout_ms < 0:
            raise ValueError("Invalid search parameters")
        if not 0 <= self.root_noise <= 1 or self.dirichlet_alpha <= 0:
            raise ValueError("Invalid root noise")


@dataclass
class Edge:
    prior: float
    child: "Node"
    visits: int = 0
    value_sum: float = 0.0  # Parent player's perspective.
    pending: int = 0  # Separate from completed statistics.


@dataclass
class Node:
    state: State
    edges: dict[int, Edge] = field(default_factory=dict)
    expanded: bool = False
    expanding: bool = False
    ready: Event = field(default_factory=Event)


@dataclass
class SearchResult:
    policy: np.ndarray
    visits: np.ndarray
    value_sums: np.ndarray
    stats: dict

    def action(self) -> int:
        return int(np.argmax(self.visits if self.visits.sum() else self.policy))


class Tree:
    """Shared tree with short, coarse-grained protected selection/backup sections.

    Evaluation happens OUTSIDE the lock. Colliding expansion requests wait on an
    event, then reselect. All reservation changes are reverted even on exceptions.
    """

    def __init__(self, state: State, evaluator, config: SearchConfig, seed: int, metrics: Metrics):
        self.root = Node(state)
        self.evaluator = evaluator
        self.config = config
        self.metrics = metrics
        self.lock = Lock()
        evaluation = evaluator.evaluate([state], [seed])[0]
        self.expand(self.root, evaluation)
        if config.root_noise:
            rng = np.random.default_rng(seed)
            noise = rng.dirichlet(np.full(len(self.root.edges), config.dirichlet_alpha))
            for edge, n in zip(self.root.edges.values(), noise, strict=True):
                edge.prior = (1 - config.root_noise) * edge.prior + config.root_noise * float(n)

    def expand(self, node: Node, evaluation: Evaluation):
        legal = node.state.legal_actions()
        priors = np.array([evaluation.policy[a] for a in legal], dtype=np.float64)
        if not np.isfinite(priors).all() or np.any(priors < 0):
            raise ValueError("Invalid policy")
        priors = priors / priors.sum() if priors.sum() else np.full(len(legal), 1 / len(legal))
        node.edges = {
            a: Edge(float(p), Node(node.state.play(a))) for a, p in zip(legal, priors, strict=True)
        }
        node.expanded = True
        node.expanding = False
        node.ready.set()
        self.metrics.add("expanded_nodes")

    def reserve(self, rng, weight: int):
        t0 = perf_counter()
        with self.lock:
            self.metrics.time("tree_lock_wait", perf_counter() - t0)
            start = perf_counter()
            node = self.root
            path = []
            while node.expanded and node.edges:
                total = sum(e.visits + e.pending for e in node.edges.values())
                scored = []
                for action, edge in node.edges.items():
                    count = edge.visits + edge.pending
                    q = (
                        (edge.value_sum - self.config.virtual_loss * edge.pending) / count
                        if count
                        else 0.0
                    )
                    if self.config.selection == "uct":
                        score = (
                            q + self.config.c_puct * sqrt(log(total + 1) / count)
                            if count
                            else float("inf")
                        )
                    else:
                        score = q + self.config.c_puct * edge.prior * sqrt(total + 1) / (1 + count)
                    scored.append((action, score))
                best = max(v for _, v in scored)
                candidates = [a for a, v in scored if v == best or abs(v - best) < 1e-12]
                action = int(rng.choice(candidates))
                edge = node.edges[action]
                edge.pending += weight
                path.append(edge)
                node = edge.child
            if node.expanding:
                for edge in path:
                    edge.pending -= weight
                self.metrics.add("expansion_collisions")
                self.metrics.time("selection", perf_counter() - start)
                return None, node.ready
            terminal = node.state.terminal_value()
            if terminal is None:
                node.expanding = True
                node.ready.clear()
            self.metrics.sample("depth", len(path))
            self.metrics.time("selection", perf_counter() - start)
            return (node, path, terminal), None

    def finish(self, reservation, evaluation: Evaluation | None, weight: int, failed=False):
        node, path, terminal = reservation
        t0 = perf_counter()
        with self.lock:
            self.metrics.time("tree_lock_wait", perf_counter() - t0)
            start = perf_counter()
            try:
                if failed:
                    return
                value = terminal if terminal is not None else evaluation.value
                if not np.isfinite(value) or abs(value) > 1.000001:
                    raise ValueError("Invalid value")
                if terminal is None:
                    self.expand(node, evaluation)
                else:
                    self.metrics.add("terminal_backups", weight)
                for edge in reversed(path):
                    value = -value  # Child -> parent, one sign change per move.
                    edge.visits += weight
                    edge.value_sum += weight * value
                self.metrics.add("completed_simulations", weight)
            finally:
                for edge in path:
                    edge.pending -= weight
                node.expanding = False
                node.ready.set()
                self.metrics.time("backup", perf_counter() - start)

    def simulation(self, seed: int, weight: int = 1, leaf_evaluate=None):
        rng = np.random.default_rng(seed)
        while True:
            reservation, event = self.reserve(rng, weight)
            if reservation is not None:
                break
            start = perf_counter()
            event.wait()
            self.metrics.time("collision_wait", perf_counter() - start)
        node, _, terminal = reservation
        evaluation = None
        try:
            if terminal is None:
                seeds = [int(x) for x in rng.integers(0, 2**31, size=weight)]
                outputs = (leaf_evaluate or self.evaluator.evaluate)([node.state] * weight, seeds)
                evaluation = Evaluation(
                    np.mean([x.policy for x in outputs], axis=0),
                    float(np.mean([x.value for x in outputs])),
                )
        except BaseException:
            self.finish(reservation, None, weight, failed=True)
            raise
        self.finish(reservation, evaluation, weight)

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        visits = np.zeros(self.root.state.action_size, dtype=np.int64)
        values = np.zeros_like(visits, dtype=np.float64)
        for action, edge in self.root.edges.items():
            visits[action] = edge.visits
            values[action] = edge.value_sum
        return visits, values


def process_root(state: State, config: SearchConfig, seed: int, budget: int):
    metrics = Metrics()
    evaluator = RolloutEvaluator(metrics)
    tree = Tree(state, evaluator, config, seed, metrics)
    for i in range(budget):
        tree.simulation(seed + i + 1)
    return (*tree.result(), metrics.snapshot())


class SearchEngine:
    """Persistent pools/server. Reused for all moves, never recreated per simulation."""

    def __init__(self, evaluator, config: SearchConfig, method: str = "serial"):
        config.validate()
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        if config.backend == "processes" and (
            method not in ("leaf", "root") or not isinstance(evaluator, RolloutEvaluator)
        ):
            raise ValueError("Process backend supports classic rollout Leaf/Root only")
        self.config, self.method = config, method
        self.raw_evaluator = evaluator
        self.evaluator = evaluator
        self.metrics = evaluator.metrics
        self.pool = None
        self.server = None
        if method == "async":
            self.server = InferenceServer(evaluator, config.batch_size, config.batch_timeout_ms)
            self.evaluator = self.server
        if method != "serial":
            if config.backend == "processes":
                self.pool = ProcessPoolExecutor(config.workers, mp_context=mp.get_context("spawn"))
            else:
                self.pool = ThreadPoolExecutor(config.workers, thread_name_prefix="mcts")

    def _leaf_evaluate(self, states, seeds):
        if self.config.backend == "processes":
            futures = [
                self.pool.submit(process_rollout, [s], [seed])
                for s, seed in zip(states, seeds, strict=True)
            ]
            outputs = []
            for future in futures:
                out, snapshot = future.result()
                self.merge(snapshot)
                outputs.extend(out)
            return outputs
        futures = [
            self.pool.submit(self.evaluator.evaluate, [s], [seed])
            for s, seed in zip(states, seeds, strict=True)
        ]
        return [f.result()[0] for f in futures]

    def merge(self, snapshot):
        """Process snapshots: sums/counters only, process batch histograms excluded."""
        for key, value in snapshot.items():
            if key.endswith("_s"):
                self.metrics.time(key[:-2], value)
            elif not key.endswith(("_mean", "_p95", "_max")):
                self.metrics.add(key, value)

    def search(self, state: State, seed: int = 0, seconds: float | None = None) -> SearchResult:
        if state.outcome() is not None:
            raise ValueError("Cannot search a terminal state")
        if seconds is not None and (seconds <= 0 or self.config.backend == "processes"):
            raise ValueError("Time budgets must be positive; process runs use fixed work")
        self.metrics.reset()
        start = perf_counter()
        deadline = start + seconds if seconds is not None else None
        config = self.config
        if self.method == "root" and config.backend == "processes":
            active = min(config.workers, config.simulations)
            budgets = [
                config.simulations // active + (i < config.simulations % active)
                for i in range(active)
            ]
            futures = [
                self.pool.submit(process_root, state, config, seed + 100_003 * i, budget)
                for i, budget in enumerate(budgets)
            ]
            parts = []
            for future in futures:
                n, w, snapshot = future.result()
                self.merge(snapshot)
                parts.append((n, w))
        elif self.method == "root":
            active = min(config.workers, config.simulations)

            def worker(i):
                tree = Tree(state, self.evaluator, config, seed + 100_003 * i, self.metrics)
                budget = config.simulations // active + (i < config.simulations % active)
                j = 0
                while j < budget if deadline is None else perf_counter() < deadline:
                    tree.simulation(seed + 100_003 * i + j + 1)
                    j += 1
                return tree.result()

            futures = [self.pool.submit(worker, i) for i in range(active)]
            parts, errors = [], []
            for future in futures:
                try:
                    parts.append(future.result())
                except BaseException as exc:
                    errors.append(exc)
            if errors:
                raise errors[0]
        else:
            tree = Tree(state, self.evaluator, config, seed, self.metrics)
            if self.method in ("serial", "leaf"):
                i = 0
                while i < config.simulations if deadline is None else perf_counter() < deadline:
                    weight = (
                        (
                            min(config.workers, config.simulations - i)
                            if deadline is None
                            else config.workers
                        )
                        if self.method == "leaf"
                        else 1
                    )
                    tree.simulation(
                        seed + i + 1, weight, self._leaf_evaluate if self.method == "leaf" else None
                    )
                    i += weight
            else:
                counter = 0
                ticket_lock = Lock()

                def worker():
                    nonlocal counter
                    while True:
                        with ticket_lock:
                            if (
                                counter >= config.simulations
                                if deadline is None
                                else perf_counter() >= deadline
                            ):
                                return
                            ticket = counter
                            counter += 1
                        tree.simulation(seed + ticket + 1)

                futures = [self.pool.submit(worker) for _ in range(config.workers)]
                # Join ALL workers before propagating any exception/resetting metrics.
                errors = []
                for future in futures:
                    try:
                        future.result()
                    except BaseException as exc:
                        errors.append(exc)
                if errors:
                    raise errors[0]
            parts = [tree.result()]
        visits = sum((p[0] for p in parts), np.zeros(state.action_size, dtype=np.int64))
        values = sum((p[1] for p in parts), np.zeros(state.action_size, dtype=np.float64))
        elapsed = perf_counter() - start
        policy = (
            visits / visits.sum()
            if visits.sum()
            else np.array([a in state.legal_actions() for a in range(state.action_size)], float)
            / len(state.legal_actions())
        )
        stats = self.metrics.snapshot()
        stats.update(
            elapsed_s=elapsed,
            simulations_per_s=float(visits.sum() / elapsed),
            method=self.method,
            backend=config.backend,
            root_visits=int(visits.sum()),
            requested_simulations=config.simulations if seconds is None else None,
            requested_seconds=seconds,
        )
        return SearchResult(policy, visits, values, stats)

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)
        if self.server:
            self.server.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
