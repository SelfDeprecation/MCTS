from dataclasses import replace

import numpy as np
import pytest
import torch

from azparallel.evaluator import NeuralEvaluator
from azparallel.game import initial_state
from azparallel.network import NetworkConfig, PolicyValueNet, load_model, save_model
from azparallel.optim import OptimizerBundle
from azparallel.search import SearchConfig, SearchEngine
from azparallel.training import (
    Example,
    TrainConfig,
    deserialize_replay,
    objective,
    sample_batch,
    seed_all,
    self_play,
    serialize_replay,
    train,
)


@pytest.fixture(autouse=True)
def seeded():
    seed_all(7)


def test_selfplay_targets_sign_policy_and_replay_roundtrip():
    model = PolicyValueNet(NetworkConfig(width=16, blocks=1))
    config = TrainConfig(simulations=8, width=16, blocks=1)
    with SearchEngine(NeuralEvaluator(model), SearchConfig(simulations=8)) as engine:
        examples = self_play(config, engine, np.random.default_rng(7))
    assert len(examples) <= 9
    for index, example in enumerate(examples):
        assert example.policy.sum() == pytest.approx(1)
        assert example.value in (-1, 0, 1)
        if index:
            assert example.value == -examples[index - 1].value
    restored = deserialize_replay(serialize_replay(examples))
    assert restored[0].state == examples[0].state
    np.testing.assert_array_equal(restored[0].policy, examples[0].policy)


@pytest.mark.parametrize("name", ["adamw", "asam", "muon"])
def test_optimizer_updates_finite_parameters_and_clips(name):
    if name == "muon" and not hasattr(torch.optim, "Muon"):
        pytest.skip("Colab runtime has PyTorch older than 2.9; Muon is unavailable")
    model = PolicyValueNet(NetworkConfig(width=16, blocks=1))
    optimizer = OptimizerBundle(model, name, grad_clip=0.1)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    state = initial_state().play(0)
    policy = np.zeros(9, dtype=np.float32)
    policy[4] = 1
    batch = sample_batch([Example(state, policy, 1)], np.random.default_rng(7), 8, "cpu")
    result = optimizer.step(lambda: objective(model, batch))
    assert np.isfinite(result["objective"])
    assert any(not torch.equal(v, original[k]) for k, v in model.state_dict().items())
    assert all(torch.isfinite(p).all() for p in model.parameters())
    norm = torch.linalg.vector_norm(
        torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])
    )
    assert float(norm) <= 0.10001


def test_asam_restores_weights_when_second_pass_fails():
    model = PolicyValueNet(NetworkConfig(width=8, blocks=1))
    optimizer = OptimizerBundle(model, "asam")
    originals = [p.detach().clone() for p in model.parameters()]
    count = 0

    def closure():
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("second pass")
        logits, values = model(torch.zeros(2, 2, 3, 3))
        return logits.square().sum() + values.square().sum(), {}

    with pytest.raises(RuntimeError):
        optimizer.step(closure)
    for original, parameter in zip(originals, model.parameters(), strict=True):
        assert torch.equal(original, parameter)


def test_model_checkpoint_and_game_mismatch(tmp_path):
    model = PolicyValueNet(NetworkConfig(width=16, blocks=1))
    path = tmp_path / "model.pt"
    save_model(model, path)
    loaded = load_model(path)
    for p, q in zip(model.parameters(), loaded.parameters(), strict=True):
        assert torch.equal(p, q)
    with pytest.raises(ValueError):
        NeuralEvaluator(loaded).evaluate([initial_state("connect4")], [0])


def test_training_resume_matches_uninterrupted_run(tmp_path):
    config = TrainConfig(
        iterations=2,
        games_per_iteration=2,
        updates_per_iteration=2,
        simulations=4,
        width=8,
        blocks=1,
        batch_size=4,
        eval_games=2,
        eval_positions=4,
        eval_every=2,
    )
    full = train(config, tmp_path / "full")
    split_dir = tmp_path / "split"
    train(replace(config, iterations=1), split_dir)
    resumed = train(config, split_dir, split_dir / "latest.pt")
    for name, parameter in full.state_dict().items():
        assert torch.equal(parameter, resumed.state_dict()[name]), name
