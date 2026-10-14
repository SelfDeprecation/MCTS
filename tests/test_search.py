from threading import Lock

import numpy as np
import pytest

from azparallel.evaluator import Evaluation, InferenceServer, RolloutEvaluator, uniform_policy
from azparallel.game import State, initial_state, minimax
from azparallel.metrics import Metrics
from azparallel.search import METHODS, SearchConfig, SearchEngine, Tree


class OracleEvaluator:
    def __init__(self):
        self.metrics = Metrics()

    def evaluate(self, states, seeds):
        self.metrics.add("evaluation_samples", len(states))
        return [Evaluation(uniform_policy(s), minimax(s)) for s in states]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("budget", [1, 7, 31])
def test_exact_work_budget_policy_mask_and_finite_values(method, budget):
    state = initial_state().play(4).play(0)
    with SearchEngine(
        OracleEvaluator(), SearchConfig(simulations=budget, workers=4, batch_size=4), method
    ) as engine:
        result = engine.search(state, 123)
    assert result.visits.sum() == budget
    assert result.stats["completed_simulations"] == budget
    assert result.policy[0] == result.policy[4] == 0
    assert result.policy.sum() == pytest.approx(1)
    assert np.isfinite(result.value_sums).all()
    assert result.action() in state.legal_actions()


@pytest.mark.parametrize("method", METHODS)
def test_immediate_win_and_parent_value_sign(method):
    state = initial_state()
    for a in (0, 3, 1, 4):
        state = state.play(a)
    with SearchEngine(OracleEvaluator(), SearchConfig(simulations=64, workers=4), method) as engine:
        result = engine.search(state, 20)
    assert result.action() == 2
    assert result.value_sums[2] > 0


def test_single_worker_equivalence():
    state = initial_state().play(4)
    results = []
    for method in METHODS:
        with SearchEngine(
            RolloutEvaluator(), SearchConfig(simulations=64, workers=1, batch_size=1), method
        ) as engine:
            results.append(engine.search(state, 5))
    for result in results[1:]:
        np.testing.assert_array_equal(result.visits, results[0].visits)
        np.testing.assert_allclose(result.value_sums, results[0].value_sums)


def test_no_reservations_after_failed_evaluation():
    metrics = Metrics()
    tree = Tree(initial_state(), OracleEvaluator(), SearchConfig(), 0, metrics)

    def fail(*_):
        raise RuntimeError("test inference failure")

    with pytest.raises(RuntimeError):
        tree.simulation(1, leaf_evaluate=fail)

    def check(node):
        assert not node.expanding
        for edge in node.edges.values():
            assert edge.pending == 0
            check(edge.child)

    check(tree.root)
    tree.simulation(2)
    assert tree.result()[0].sum() == 1


def test_parallel_failure_does_not_deadlock():
    class Broken(OracleEvaluator):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.lock = Lock()

        def evaluate(self, states, seeds):
            with self.lock:
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("broken model")
            return super().evaluate(states, seeds)

    for method in ("root", "tree", "async"):
        with SearchEngine(Broken(), SearchConfig(simulations=16), method) as engine:
            with pytest.raises(RuntimeError, match="broken model"):
                engine.search(initial_state())


def test_server_batches_and_drains_then_rejects_new_work():
    evaluator = RolloutEvaluator()
    server = InferenceServer(evaluator, batch_size=4, timeout_ms=20)
    futures = [server.submit(initial_state(), seed) for seed in range(7)]
    server.close()
    assert all(f.result().policy.sum() == pytest.approx(1) for f in futures)
    assert evaluator.metrics.snapshot()["batch_size_max"] == 4
    assert evaluator.metrics.snapshot()["evaluation_samples"] == 7
    with pytest.raises(RuntimeError):
        server.submit(initial_state(), 42)


@pytest.mark.parametrize("method", ["leaf", "root"])
def test_windows_spawn_process_backend(method):
    with SearchEngine(
        RolloutEvaluator(), SearchConfig(simulations=13, workers=2, backend="processes"), method
    ) as engine:
        result = engine.search(initial_state(), 9)
    assert result.visits.sum() == 13
    assert result.stats["completed_simulations"] == 13


def test_cancelled_request_does_not_kill_batch_server():
    evaluator = RolloutEvaluator()
    server = InferenceServer(evaluator, batch_size=4, timeout_ms=20)
    cancelled = server.submit(initial_state(), 1)
    cancelled.cancel()
    future = server.submit(initial_state(), 2)
    server.close()
    assert future.result().policy.sum() == pytest.approx(1)


def test_time_budget_returns_legal_action_even_when_init_exhausts_budget():
    state = initial_state().play(0)
    with SearchEngine(RolloutEvaluator(delay_ms=2), SearchConfig(), "serial") as engine:
        result = engine.search(state, seconds=0.00001)
    assert result.visits.sum() == 0
    assert result.action() in state.legal_actions()


def test_terminal_search_rejected():
    state = State((1, 1, 1, -1, -1, 0, 0, 0, 0), -1)
    with SearchEngine(OracleEvaluator(), SearchConfig()) as engine:
        with pytest.raises(ValueError):
            engine.search(state)
