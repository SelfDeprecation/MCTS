"""Compact residual MLP with policy and value heads (scaled AlphaZero variant)."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class NetworkConfig:
    game: str = "tictactoe"
    width: int = 64
    blocks: int = 2


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x):
        return x + self.fc2(torch.relu(self.fc1(self.norm(x))))


class PolicyValueNet(nn.Module):
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        cells = 9 if config.game == "tictactoe" else 42
        actions = 9 if config.game == "tictactoe" else 7
        self.input = nn.Linear(2 * cells, config.width)
        self.trunk = nn.Sequential(*(ResidualBlock(config.width) for _ in range(config.blocks)))
        self.norm = nn.LayerNorm(config.width)
        self.policy = nn.Linear(config.width, actions)
        self.value = nn.Linear(config.width, 1)

    def forward(self, x):
        x = torch.relu(self.input(x.flatten(1)))
        x = self.norm(self.trunk(x))
        return self.policy(x), torch.tanh(self.value(x)).squeeze(-1)


def save_model(model: PolicyValueNet, path: Path, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {"network_config": asdict(model.config), "state_dict": model.state_dict(), **extra},
        temporary,
    )
    temporary.replace(path)


def load_model(path: Path, device: str = "cpu") -> PolicyValueNet:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = PolicyValueNet(NetworkConfig(**payload["network_config"]))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()
