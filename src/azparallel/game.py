"""Immutable, alternating, two-player zero-sum games.

Terminal values and neural values ALWAYS refer to the player to move.
A move switches that player even when the move ends the game.
"""

from dataclasses import dataclass
from functools import cache, lru_cache

import numpy as np


@dataclass(frozen=True)
class State:
    board: tuple[int, ...]
    player: int = 1
    game: str = "tictactoe"

    @property
    def shape(self) -> tuple[int, int]:
        return (3, 3) if self.game == "tictactoe" else (6, 7)

    @property
    def action_size(self) -> int:
        return 9 if self.game == "tictactoe" else 7

    def legal_actions(self) -> tuple[int, ...]:
        if self.outcome() is not None:
            return ()
        if self.game == "tictactoe":
            return tuple(i for i, x in enumerate(self.board) if x == 0)
        return tuple(c for c in range(7) if self.board[c] == 0)

    def play(self, action: int) -> "State":
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action {action}")
        cells = list(self.board)
        index = action
        if self.game == "connect4":
            index = next(r * 7 + action for r in range(5, -1, -1) if cells[r * 7 + action] == 0)
        cells[index] = self.player
        return State(tuple(cells), -self.player, self.game)

    @lru_cache(maxsize=200_000)
    def outcome(self) -> int | None:
        """Winner in absolute colours (+1/-1), draw 0, ongoing None."""
        h, w = self.shape
        k = 3 if self.game == "tictactoe" else 4
        for line in winning_lines(h, w, k):
            colour = self.board[line[0]]
            if colour and all(self.board[i] == colour for i in line[1:]):
                return colour
        return None if 0 in self.board else 0

    def terminal_value(self) -> float | None:
        result = self.outcome()
        return None if result is None else float(result * self.player)

    def encode(self) -> np.ndarray:
        cells = np.array(self.board, dtype=np.int8).reshape(self.shape)
        return np.stack((cells == self.player, cells == -self.player)).astype(np.float32)

    def render(self) -> str:
        h, w = self.shape
        chars = {0: ".", 1: "X", -1: "O"}
        return (
            "\n".join(" ".join(chars[x] for x in self.board[r * w : (r + 1) * w]) for r in range(h))
            + "\n"
            + " ".join(map(str, range(w)))
        )


@lru_cache(maxsize=4)
def winning_lines(h: int, w: int, k: int) -> tuple[tuple[int, ...], ...]:
    lines = []
    for r in range(h):
        for c in range(w):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if 0 <= r + (k - 1) * dr < h and 0 <= c + (k - 1) * dc < w:
                    lines.append(tuple((r + i * dr) * w + c + i * dc for i in range(k)))
    return tuple(lines)


def initial_state(game: str = "tictactoe") -> State:
    if game not in ("tictactoe", "connect4"):
        raise ValueError(f"Unknown game: {game}")
    return State((0,) * (9 if game == "tictactoe" else 42), 1, game)


def action_permutation(state: State, symmetry: int) -> np.ndarray:
    """For each ORIGINAL action give its TRANSFORMED index."""
    if state.game == "connect4":
        if symmetry not in (0, 1):
            raise ValueError("Gravity permits identity and horizontal reflection only")
        return np.arange(7) if symmetry == 0 else np.arange(6, -1, -1)
    if not 0 <= symmetry < 8:
        raise ValueError("Tic-tac-toe symmetry must be in [0, 7]")
    grid = np.arange(9).reshape(3, 3)
    transformed = np.rot90(grid, symmetry % 4)
    if symmetry >= 4:
        transformed = np.fliplr(transformed)
    return np.argsort(transformed.ravel())


def transform(state: State, symmetry: int) -> State:
    permutation = action_permutation(state, symmetry)
    board = np.array(state.board).reshape(state.shape)
    if state.game == "connect4":
        board = board if symmetry == 0 else np.fliplr(board)
    else:
        cells = np.empty(9, dtype=np.int8)
        cells[permutation] = board.ravel()
        board = cells
    return State(tuple(int(x) for x in board.ravel()), state.player, state.game)


@cache
def minimax(state: State) -> float:
    """Exact oracle for the small verification game; never used for training."""
    if state.game != "tictactoe":
        raise ValueError("Exact oracle is restricted to tic-tac-toe")
    value = state.terminal_value()
    if value is not None:
        return value
    return max(-minimax(state.play(a)) for a in state.legal_actions())


def optimal_actions(state: State) -> tuple[int, ...]:
    best = minimax(state)
    return tuple(a for a in state.legal_actions() if -minimax(state.play(a)) == best)
