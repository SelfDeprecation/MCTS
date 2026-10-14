import numpy as np
import pytest

from azparallel.game import action_permutation, initial_state, minimax, optimal_actions, transform


def test_terminal_perspective_and_illegal_moves():
    state = initial_state()
    for a in (0, 3, 1, 4, 2):
        state = state.play(a)
    assert state.outcome() == 1
    assert state.player == -1 and state.terminal_value() == -1
    assert state.legal_actions() == ()
    with pytest.raises(ValueError):
        state.play(8)


def test_connect4_gravity_and_horizontal_win():
    state = initial_state("connect4")
    for a in (0, 0, 1, 1, 2, 2, 3):
        state = state.play(a)
    assert state.outcome() == 1
    assert state.board[35:39] == (1, 1, 1, 1)


@pytest.mark.parametrize("game", ["tictactoe", "connect4"])
def test_symmetries_commute_with_moves(game):
    state = initial_state(game).play(0).play(1)
    for symmetry in range(8 if game == "tictactoe" else 2):
        perm = action_permutation(state, symmetry)
        for action in state.legal_actions():
            assert transform(state.play(action), symmetry) == transform(state, symmetry).play(
                int(perm[action])
            )
        assert transform(state, symmetry).outcome() == state.outcome()


def test_exhaustive_reachable_tictactoe_oracle():
    seen = set()
    stack = [initial_state()]
    while stack:
        state = stack.pop()
        if state in seen:
            continue
        seen.add(state)
        v = state.terminal_value()
        if v is None:
            children = [state.play(a) for a in state.legal_actions()]
            assert minimax(state) == max(-minimax(s) for s in children)
            assert optimal_actions(state)
            assert np.array_equal(
                state.encode()[0], (np.array(state.board).reshape(3, 3) == state.player)
            )
            stack.extend(children)
        else:
            assert minimax(state) == v
    assert len(seen) == 5478
    assert minimax(initial_state()) == 0
