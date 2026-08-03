from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


def _orthogonal_init(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)


def build_mlp(
    input_size: int,
    hidden_sizes: Sequence[int],
    output_size: int,
    output_gain: float,
) -> nn.Sequential:
    modules: list[nn.Module] = []
    previous = int(input_size)
    for width in hidden_sizes:
        layer = nn.Linear(previous, int(width))
        _orthogonal_init(layer, math.sqrt(2.0))
        modules.extend((layer, nn.Tanh()))
        previous = int(width)
    output = nn.Linear(previous, int(output_size))
    _orthogonal_init(output, output_gain)
    modules.append(output)
    return nn.Sequential(*modules)


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        self.actor = build_mlp(observation_size, hidden_sizes, action_size, 0.01)
        self.critic = build_mlp(observation_size, hidden_sizes, 1, 1.0)
        self.log_std = nn.Parameter(torch.full((action_size,), -0.5))

    def distribution(self, observation: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(observation)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation).squeeze(-1)

    @staticmethod
    def squash_log_probability(
        distribution: torch.distributions.Normal,
        pre_tanh_action: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        correction = torch.log(1.0 - action.pow(2) + 1.0e-6)
        return (distribution.log_prob(pre_tanh_action) - correction).sum(dim=-1)

    def sample(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        pre_tanh = distribution.rsample()
        action = torch.tanh(pre_tanh)
        log_probability = self.squash_log_probability(
            distribution, pre_tanh, action
        )
        return action, pre_tanh, log_probability, self.value(observation)

    def deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(observation))

    def evaluate_pre_tanh(
        self, observation: torch.Tensor, pre_tanh_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        action = torch.tanh(pre_tanh_action)
        log_probability = self.squash_log_probability(
            distribution, pre_tanh_action, action
        )
        base_entropy = distribution.entropy().sum(dim=-1)
        return log_probability, base_entropy, self.value(observation)

