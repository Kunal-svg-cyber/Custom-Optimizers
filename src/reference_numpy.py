"""NumPy reference implementation of the coordinate-wise SVRG update rule.

Purpose
-------
1. **Oracle for differential testing.** The PyTorch and JAX engines are validated
   step-for-step against this deliberately naive, loop-free-of-tricks
   implementation. Three independent implementations agreeing is much stronger
   evidence than one implementation agreeing with its own tests.
2. **Research harness.** It runs the 2x2 ablation used in ``experiments/``:

   ==========================  ==============================  ==========================
   ``variance_reduction``      ``adaptive``                    resulting method
   ==========================  ==============================  ==========================
   False                       False                           SGD with momentum
   False                       True                            Adam (with clamped denom.)
   True                        False                           SVRG with momentum
   True                        True                            Coordinate SVRG (this repo)
   ==========================  ==============================  ==========================

The class is gradient-oracle agnostic: it consumes gradients, it never computes
them, so environment and math stay decoupled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]


@dataclass(frozen=True)
class ReferenceConfig:
    """Hyper-parameters plus the two ablation switches."""

    lr: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-15
    weight_decay: float = 0.0
    update_clip: float = 10.0
    snapshot_interval: int = 50
    variance_reduction: bool = True
    adaptive: bool = True

    def __post_init__(self) -> None:
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if not (0.0 <= self.beta1 < 1.0 and 0.0 <= self.beta2 < 1.0):
            raise ValueError("beta1 and beta2 must lie in [0, 1)")
        if self.eps <= 0.0 or self.update_clip <= 0.0:
            raise ValueError("eps and update_clip must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0")
        if self.snapshot_interval < 1:
            raise ValueError("snapshot_interval must be >= 1")


class ReferenceSVRG:
    """Stateful NumPy optimizer: ``w_{t+1} = step(w_t, g(w_t), g(w~))``."""

    def __init__(self, config: ReferenceConfig, dim: int, dtype: type = np.float64) -> None:
        self.config: ReferenceConfig = config
        self.dtype: type = dtype
        self.exp_avg: FloatArray = np.zeros(dim, dtype=dtype)
        self.exp_avg_sq: FloatArray = np.zeros(dim, dtype=dtype)
        self.snapshot: Optional[FloatArray] = None
        self.full_grad: Optional[FloatArray] = None
        self.step_count: int = 0
        self.steps_since_snapshot: int = 0
        self.lr_scale: float = 1.0  # external learning-rate schedule hook

    @property
    def needs_snapshot(self) -> bool:
        """True when variance reduction is on and the snapshot is missing/stale."""
        if not self.config.variance_reduction:
            return False
        if self.snapshot is None:
            return True
        return self.steps_since_snapshot >= self.config.snapshot_interval

    def refresh_snapshot(self, weights: FloatArray, full_grad: FloatArray) -> None:
        """Freeze ``w~ <- weights`` and store ``mu~ = full_grad``."""
        self.snapshot = np.array(weights, dtype=self.dtype, copy=True)
        self.full_grad = np.array(full_grad, dtype=self.dtype, copy=True)
        self.steps_since_snapshot = 0

    def variance_reduced_gradient(
        self,
        weights: FloatArray,
        grad_live: FloatArray,
        grad_snapshot: Optional[FloatArray],
    ) -> FloatArray:
        """``g_hat = g(w_t) - g(w~) + mu~`` (or plain ``g(w_t)`` if VR is off)."""
        g: FloatArray = np.asarray(grad_live, dtype=self.dtype)
        if self.config.variance_reduction:
            if self.snapshot is None or self.full_grad is None or grad_snapshot is None:
                raise RuntimeError("variance reduction needs a snapshot and its gradient")
            g = g - np.asarray(grad_snapshot, dtype=self.dtype) + self.full_grad
        if self.config.weight_decay != 0.0:
            g = g + self.config.weight_decay * np.asarray(weights, dtype=self.dtype)
        return g

    def step(
        self,
        weights: FloatArray,
        grad_live: FloatArray,
        grad_snapshot: Optional[FloatArray] = None,
    ) -> FloatArray:
        """Return the next weights; updates moments and counters in place."""
        cfg: ReferenceConfig = self.config
        finfo = np.finfo(self.dtype)
        grad_bound: float = 0.25 * math.sqrt(float(finfo.max))
        denom_floor: float = max(cfg.eps, 4.0 * math.sqrt(float(finfo.tiny)))

        g_hat: FloatArray = self.variance_reduced_gradient(weights, grad_live, grad_snapshot)
        g_hat = np.nan_to_num(g_hat, nan=0.0, posinf=grad_bound, neginf=-grad_bound)
        g_hat = np.clip(g_hat, -grad_bound, grad_bound)

        self.step_count += 1
        self.steps_since_snapshot += 1
        t: int = self.step_count

        self.exp_avg = cfg.beta1 * self.exp_avg + (1.0 - cfg.beta1) * g_hat
        self.exp_avg_sq = cfg.beta2 * self.exp_avg_sq + (1.0 - cfg.beta2) * g_hat * g_hat
        bias1: float = 1.0 - cfg.beta1 ** t
        bias2: float = 1.0 - cfg.beta2 ** t

        m_hat: FloatArray = self.exp_avg / bias1
        if cfg.adaptive:
            denom: FloatArray = np.maximum(np.sqrt(self.exp_avg_sq / bias2), denom_floor)
            update: FloatArray = np.clip(m_hat / denom, -cfg.update_clip, cfg.update_clip)
        else:
            update = m_hat
        return np.asarray(weights, dtype=self.dtype) - cfg.lr * self.lr_scale * update
