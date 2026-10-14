"""Evaluators and an asynchronous, bounded, micro-batching inference service."""

from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Lock, Thread
from time import perf_counter

import numpy as np

from .game import State, action_permutation, transform
from .metrics import Metrics


@dataclass
class Evaluation:
    policy: np.ndarray
    value: float


def uniform_policy(state: State) -> np.ndarray:
    p = np.zeros(state.action_size, dtype=np.float64)
    legal = state.legal_actions()
    if legal:
        p[list(legal)] = 1 / len(legal)
    return p


class RolloutEvaluator:
    """A single independent random playout per evaluation; no neural guidance."""

    def __init__(self, metrics: Metrics | None = None, delay_ms: float = 0):
        self.metrics = metrics or Metrics()
        self.delay_ms = delay_ms

    def evaluate(self, states: list[State], seeds: list[int]) -> list[Evaluation]:
        from time import sleep

        t0 = perf_counter()
        if self.delay_ms:
            sleep(self.delay_ms / 1000)  # Explicit synthetic latency ablation only.
        out = []
        for state, seed in zip(states, seeds, strict=True):
            rng = np.random.default_rng(seed)
            current = state
            while current.outcome() is None:
                current = current.play(int(rng.choice(current.legal_actions())))
            out.append(Evaluation(uniform_policy(state), float(current.outcome() * state.player)))
        self.metrics.add("evaluation_samples", len(out))
        self.metrics.add("evaluation_calls")
        self.metrics.time("evaluation_service", perf_counter() - t0)
        self.metrics.sample("batch_size", len(out))
        return out


class NeuralEvaluator:
    """Immutable inference model, serialized calls or one shared batch service.

    symmetry=True uses random valid board symmetries and restores action coordinates.
    This gives meaningful repeated-leaf estimates, unlike identical deterministic calls.
    """

    def __init__(
        self, model, device: str = "cpu", symmetry: bool = True, metrics: Metrics | None = None
    ):
        import torch

        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.symmetry = symmetry
        self.metrics = metrics or Metrics()
        self.lock = Lock()

    def evaluate(self, states: list[State], seeds: list[int]) -> list[Evaluation]:
        import torch

        if not states:
            return []
        if any(s.game != self.model.config.game for s in states):
            raise ValueError("Checkpoint game does not match evaluation game")
        t0 = perf_counter()
        with self.lock:
            self.metrics.time("inference_lock_wait", perf_counter() - t0)
            start = perf_counter()
            symmetries = [
                (
                    int(np.random.default_rng(seed).integers(8 if s.game == "tictactoe" else 2))
                    if self.symmetry
                    else 0
                )
                for s, seed in zip(states, seeds, strict=True)
            ]
            array = np.stack(
                [transform(s, sym).encode() for s, sym in zip(states, symmetries, strict=True)]
            )
            x = torch.from_numpy(array).to(self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            forward_start = perf_counter()
            with torch.inference_mode():
                logits, values = self.model(x)
                policies = logits.softmax(-1).cpu().numpy()
                values = values.cpu().numpy()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.metrics.time("network_forward", perf_counter() - forward_start)
            result = []
            for s, sym, p, value in zip(states, symmetries, policies, values, strict=True):
                restored = p[action_permutation(s, sym)].astype(np.float64)
                mask = np.zeros(s.action_size, dtype=bool)
                mask[list(s.legal_actions())] = True
                restored[~mask] = 0
                total = restored.sum()
                restored = restored / total if total > 0 else uniform_policy(s)
                result.append(Evaluation(restored, float(value)))
            self.metrics.add("evaluation_samples", len(states))
            self.metrics.add("evaluation_calls")
            self.metrics.sample("batch_size", len(states))
            self.metrics.time("evaluation_service", perf_counter() - start)
            return result


class InferenceServer:
    """CPU selection workers submit futures; one consumer evaluates NN batches.

    Each worker waits for its own result while other workers continue selecting.
    Batches flush on size OR timeout. Queue is bounded and close drains accepted work.
    Exceptions propagate to every future in the affected batch.
    """

    def __init__(self, evaluator, batch_size: int = 16, timeout_ms: float = 1.0):
        if batch_size < 1 or timeout_ms < 0:
            raise ValueError("Invalid batching parameters")
        self.evaluator = evaluator
        self.metrics = evaluator.metrics
        self.batch_size = batch_size
        self.timeout = timeout_ms / 1000
        self.queue = Queue(maxsize=max(64, batch_size * 8))
        self.closed = False
        self.lifecycle = Lock()
        self.thread = Thread(target=self._run, name="inference-server", daemon=True)
        self.thread.start()

    def submit(self, state: State, seed: int) -> Future:
        future = Future()
        with self.lifecycle:
            if self.closed:
                raise RuntimeError("Inference server is closed")
            self.queue.put((state, seed, future, perf_counter()))
        return future

    def evaluate(self, states: list[State], seeds: list[int]) -> list[Evaluation]:
        futures = [self.submit(s, seed) for s, seed in zip(states, seeds, strict=True)]
        return [f.result() for f in futures]

    def _run(self):
        stopping = False
        while not stopping:
            first = self.queue.get()
            if first is None:
                break
            batch = [first]
            deadline = perf_counter() + self.timeout
            while len(batch) < self.batch_size:
                try:
                    remaining = deadline - perf_counter()
                    item = self.queue.get(timeout=max(0, remaining))
                except Empty:
                    break
                if item is None:
                    stopping = True
                    break
                batch.append(item)
            batch = [item for item in batch if item[2].set_running_or_notify_cancel()]
            if not batch:
                continue
            now = perf_counter()
            for item in batch:
                self.metrics.sample("queue_wait_ms", (now - item[3]) * 1000)
            try:
                outputs = self.evaluator.evaluate([x[0] for x in batch], [x[1] for x in batch])
                if len(outputs) != len(batch):
                    raise RuntimeError("Evaluator returned wrong batch length")
                for item, output in zip(batch, outputs, strict=True):
                    item[2].set_result(output)
            except Exception as exc:
                for item in batch:
                    item[2].set_exception(exc)

    def close(self):
        with self.lifecycle:
            if self.closed:
                return
            self.closed = True
            self.queue.put(None)
        self.thread.join()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def process_rollout(states: list[State], seeds: list[int]) -> tuple[list[Evaluation], dict]:
    """Top-level picklable entry point for Windows spawn."""
    evaluator = RolloutEvaluator()
    outputs = evaluator.evaluate(states, seeds)
    return outputs, evaluator.metrics.snapshot()
