"""Stateless JAX backend: coordinate-wise SVRG as a pure ``state_t -> state_t+1`` map.

The math is identical to ``torch_optimizer.CoordinateSVRG``::

    g_hat = g_t(w_t) - g_t(w~) + mu~
    m     = b1 * m + (1 - b1) * g_hat
    v     = b2 * v + (1 - b2) * g_hat^2
    step  = clip( (m / (1 - b1^t)) / max(sqrt(v / (1 - b2^t)), floor), +-update_clip )
    w    <- w - lr * step

Nothing here holds Python-side mutable state. ``SVRGState`` is registered as a
custom JAX PyTree, so it flows through ``jax.jit`` / ``jax.lax`` transforms and
XLA fuses the whole update into a few kernels.

The four invariants
-------------------
1. Variance reduction: ``g_hat`` is built from a snapshot control variate;
   ``variance_reduced_gradient`` exposes it so the law can be checked directly.
2. Deterministic sample alignment: the step function takes ONE ``batch`` and
   evaluates both ``g_t(w_t)`` and ``g_t(w~)`` on that very object. Misalignment
   is impossible by construction.
3. Hardware contiguity: XLA owns buffer layout, so there is no fragmented
   Python-side tensor zoo to flatten; the state is a handful of pytrees that
   XLA fuses element-wise.
4. Coordinate boundary safeguards: NaN/Inf sanitised, gradients bounded so
   ``g^2`` stays finite, denominator clamped (not offset), ratio clipped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

Array = jnp.ndarray
PyTree = Any
Batch = Any
LossFn = Callable[[PyTree, Batch], Array]
StepFn = Callable[[PyTree, "SVRGState", Batch], Tuple[PyTree, "SVRGState", Dict[str, Array]]]


@dataclass(frozen=True)
class SVRGConfig:
    """Static (hashable) hyper-parameters; closed over at jit time."""

    lr: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-15
    weight_decay: float = 0.0
    snapshot_interval: int = 50
    update_clip: float = 10.0

    def __post_init__(self) -> None:
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if not (0.0 <= self.beta1 < 1.0 and 0.0 <= self.beta2 < 1.0):
            raise ValueError("beta1 and beta2 must lie in [0, 1)")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0")
        if self.snapshot_interval < 1:
            raise ValueError("snapshot_interval must be >= 1")
        if self.update_clip <= 0.0:
            raise ValueError("update_clip must be positive")


@register_pytree_node_class
class SVRGState:
    """Optimizer state as a custom PyTree (all fields are children/leaves).

    Attributes:
        step: int32 scalar, number of completed steps.
        steps_since_snapshot: int32 scalar, inner steps since the last refresh.
        snapshot_ready: bool scalar, False until ``refresh_snapshot`` runs.
        snapshot_params: the frozen snapshot ``w~`` (same structure as params).
        full_grad: ``mu~``, the full gradient at ``w~``.
        exp_avg: first moment ``m``.
        exp_avg_sq: second moment ``v``.
    """

    def __init__(
        self,
        step: Array,
        steps_since_snapshot: Array,
        snapshot_ready: Array,
        snapshot_params: PyTree,
        full_grad: PyTree,
        exp_avg: PyTree,
        exp_avg_sq: PyTree,
    ) -> None:
        # No validation here: JAX rebuilds this class with placeholder leaves.
        self.step = step
        self.steps_since_snapshot = steps_since_snapshot
        self.snapshot_ready = snapshot_ready
        self.snapshot_params = snapshot_params
        self.full_grad = full_grad
        self.exp_avg = exp_avg
        self.exp_avg_sq = exp_avg_sq

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], None]:
        children: Tuple[Any, ...] = (
            self.step,
            self.steps_since_snapshot,
            self.snapshot_ready,
            self.snapshot_params,
            self.full_grad,
            self.exp_avg,
            self.exp_avg_sq,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: Tuple[Any, ...]) -> "SVRGState":
        return cls(*children)


def init_state(params: PyTree) -> SVRGState:
    """Fresh state: zero moments, snapshot = params, no snapshot committed yet."""
    zeros: PyTree = jax.tree_util.tree_map(jnp.zeros_like, params)
    copy: PyTree = jax.tree_util.tree_map(jnp.array, params)
    return SVRGState(
        step=jnp.zeros((), dtype=jnp.int32),
        steps_since_snapshot=jnp.zeros((), dtype=jnp.int32),
        snapshot_ready=jnp.zeros((), dtype=jnp.bool_),
        snapshot_params=copy,
        full_grad=zeros,
        exp_avg=zeros,
        exp_avg_sq=zeros,
    )


def compute_full_gradient(loss_fn: LossFn, params: PyTree, chunks: Batch) -> PyTree:
    """Exact full gradient as the mean of per-chunk mean-loss gradients.

    ``chunks`` is a pytree whose leaves share a leading ``num_chunks`` axis;
    chunks must be equal-sized and cover the data once. Chunks are processed
    sequentially with ``jax.lax.map`` to keep memory flat.
    """

    def one_chunk(chunk: Batch) -> PyTree:
        return jax.grad(loss_fn)(params, chunk)

    stacked: PyTree = jax.lax.map(one_chunk, chunks)
    return jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), stacked)


def refresh_snapshot(params: PyTree, state: SVRGState, full_grad: PyTree) -> SVRGState:
    """Freeze ``w~ <- params`` and install the freshly computed ``mu~``."""
    return SVRGState(
        step=state.step,
        steps_since_snapshot=jnp.zeros_like(state.steps_since_snapshot),
        snapshot_ready=jnp.ones_like(state.snapshot_ready),
        snapshot_params=jax.tree_util.tree_map(jnp.array, params),
        full_grad=full_grad,
        exp_avg=state.exp_avg,
        exp_avg_sq=state.exp_avg_sq,
    )


def needs_snapshot(state: SVRGState, config: SVRGConfig) -> bool:
    """Host-side check (forces a device sync; prefer counting steps in Python)."""
    if not bool(state.snapshot_ready):
        return True
    return int(state.steps_since_snapshot) >= config.snapshot_interval


def _leaf_bounds(dtype: Any, eps: float) -> Tuple[float, float]:
    """Dtype-aware (gradient bound, denominator floor)."""
    finfo = jnp.finfo(dtype)
    grad_bound: float = 0.25 * math.sqrt(float(finfo.max))
    denom_floor: float = max(eps, 4.0 * math.sqrt(float(finfo.tiny)))
    return grad_bound, denom_floor


def _sanitize(g_hat: Array, grad_bound: float) -> Array:
    cleaned: Array = jnp.nan_to_num(g_hat, nan=0.0, posinf=grad_bound, neginf=-grad_bound)
    return jnp.clip(cleaned, -grad_bound, grad_bound)


def _tree_norm(leaves: List[Array]) -> Array:
    total: Array = jnp.zeros((), dtype=jnp.float32)
    for leaf in leaves:
        total = total + jnp.sum(jnp.square(leaf.astype(jnp.float32)))
    return jnp.sqrt(total)


def variance_reduced_gradient(
    loss_fn: LossFn, params: PyTree, state: SVRGState, batch: Batch
) -> PyTree:
    """``g_hat = g(w_t) - g(w~) + mu~`` on ONE shared batch (no update applied)."""
    g_live: PyTree = jax.grad(loss_fn)(params, batch)
    g_snap: PyTree = jax.grad(loss_fn)(state.snapshot_params, batch)
    return jax.tree_util.tree_map(
        lambda gl, gs, mu: gl - gs + mu, g_live, g_snap, state.full_grad
    )


def make_svrg_step(loss_fn: LossFn, config: SVRGConfig) -> StepFn:
    """Build a jitted pure step ``(params, state, batch) -> (params', state', stats)``.

    ``loss_fn(params, batch)`` must return the mean scalar loss on ``batch``.
    If no snapshot has been committed yet, the step degrades safely to the plain
    mini-batch gradient instead of using a meaningless control variate.
    """
    lr: float = config.lr
    beta1: float = config.beta1
    beta2: float = config.beta2

    def _step(
        params: PyTree, state: SVRGState, batch: Batch
    ) -> Tuple[PyTree, SVRGState, Dict[str, Array]]:
        loss: Array
        g_live: PyTree
        loss, g_live = jax.value_and_grad(loss_fn)(params, batch)
        g_snap: PyTree = jax.grad(loss_fn)(state.snapshot_params, batch)  # same batch

        new_step: Array = state.step + 1
        t: Array = new_step.astype(jnp.float32)

        p_leaves, treedef = jax.tree_util.tree_flatten(params)
        snap_leaves = jax.tree_util.tree_leaves(state.snapshot_params)
        mu_leaves = jax.tree_util.tree_leaves(state.full_grad)
        m_leaves = jax.tree_util.tree_leaves(state.exp_avg)
        v_leaves = jax.tree_util.tree_leaves(state.exp_avg_sq)
        gl_leaves = jax.tree_util.tree_leaves(g_live)
        gs_leaves = jax.tree_util.tree_leaves(g_snap)

        new_p: List[Array] = []
        new_m: List[Array] = []
        new_v: List[Array] = []
        hats: List[Array] = []
        updates: List[Array] = []
        corrections: List[Array] = []
        denoms_min: List[Array] = []
        nonfinite: Array = jnp.zeros((), dtype=jnp.float32)

        for p, snap, mu, m, v, gl, gs in zip(
            p_leaves, snap_leaves, mu_leaves, m_leaves, v_leaves, gl_leaves, gs_leaves
        ):
            grad_bound, denom_floor = _leaf_bounds(p.dtype, config.eps)
            vr: Array = gl - gs + mu
            g_hat: Array = jnp.where(state.snapshot_ready, vr, gl)
            if config.weight_decay != 0.0:
                g_hat = g_hat + config.weight_decay * p
            nonfinite = nonfinite + jnp.sum(~jnp.isfinite(g_hat)).astype(jnp.float32)
            g_hat = _sanitize(g_hat, grad_bound)

            m_new: Array = beta1 * m + (1.0 - beta1) * g_hat
            v_new: Array = beta2 * v + (1.0 - beta2) * g_hat * g_hat

            bias1: Array = (1.0 - beta1 ** t).astype(p.dtype)
            bias2: Array = (1.0 - beta2 ** t).astype(p.dtype)
            denom: Array = jnp.maximum(jnp.sqrt(v_new) / jnp.sqrt(bias2), denom_floor)
            update: Array = jnp.clip(
                (m_new / bias1) / denom, -config.update_clip, config.update_clip
            )

            new_p.append(p - lr * update)
            new_m.append(m_new)
            new_v.append(v_new)
            hats.append(g_hat)
            updates.append(update)
            corrections.append(gl - gs)
            denoms_min.append(jnp.min(denom).astype(jnp.float32))

        new_params: PyTree = jax.tree_util.tree_unflatten(treedef, new_p)
        new_state: SVRGState = SVRGState(
            step=new_step,
            steps_since_snapshot=state.steps_since_snapshot + 1,
            snapshot_ready=state.snapshot_ready,
            snapshot_params=state.snapshot_params,
            full_grad=state.full_grad,
            exp_avg=jax.tree_util.tree_unflatten(treedef, new_m),
            exp_avg_sq=jax.tree_util.tree_unflatten(treedef, new_v),
        )
        stats: Dict[str, Array] = {
            "loss": loss.astype(jnp.float32),
            "live_grad_norm": _tree_norm(gl_leaves),
            "correction_norm": _tree_norm(corrections),
            "full_grad_norm": _tree_norm(mu_leaves),
            "snapshot_distance": _tree_norm([p - s for p, s in zip(p_leaves, snap_leaves)]),
            "vr_grad_norm": _tree_norm(hats),
            "update_norm": _tree_norm(updates),
            "nonfinite_count": nonfinite,
            "min_denominator": jnp.min(jnp.stack(denoms_min)),
        }
        return new_params, new_state, stats

    return jax.jit(_step)
