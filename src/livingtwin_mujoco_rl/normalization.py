from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RunningMeanStd:
    shape: tuple[int, ...]
    epsilon: float = 1.0e-4
    clip: float = 10.0

    def __post_init__(self) -> None:
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = float(self.epsilon)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == len(self.shape):
            array = array[None, ...]
        batch_mean = np.mean(array, axis=0)
        batch_var = np.var(array, axis=0)
        batch_count = array.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = float(total)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        normalized = (array - self.mean.astype(np.float32)) / np.sqrt(
            self.var.astype(np.float32) + 1.0e-8
        )
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "epsilon": self.epsilon,
            "clip": self.clip,
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if tuple(state["shape"]) != self.shape:
            raise ValueError("normalizer shape mismatch")
        self.epsilon = float(state["epsilon"])
        self.clip = float(state["clip"])
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.var = np.asarray(state["var"], dtype=np.float64).copy()
        self.count = float(state["count"])

