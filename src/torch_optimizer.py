"""Stateful PyTorch backend: coordinate-wise SVRG with adaptive moment scaling.

Update rule, per coordinate, with ``w~`` the frozen snapshot and ``mu~`` the
full gradient at ``w~``::

    g_hat = g_t(w_t) - g_t(w~) + mu~            variance-reduced gradient
    m     = b1 * m + (1 - b1) * g_hat           first moment (momentum)
    v     = b2 * v + (1 - b2) * g_hat^2         second moment
    step  = clip( (m / (1 - b1^t)) / max(sqrt(v / (1 - b2^t)), floor), +-update_clip )
    w    <- w - lr * step

The four invariants and where they are enforced
-----------------------------------------------
1. Variance reduction: ``g_hat`` uses a snapshot control variate, so
   ``g_hat - grad f(w_t)`` shrinks as ``w_t -> w~``. Telemetry exposes
   ``snapshot_distance`` (``||w_t - w~||``) and ``correction_norm`` to watch it.
2. Deterministic sample alignment: ``step`` calls the closure twice with the
   *same* batch id (live weights, then snapshot weights). The closure must be a
   pure function of that id, so both gradients see identical samples.
3. Hardware contiguity: on first use, every parameter of a group is re-pointed
   into one flat buffer, and the snapshot / full gradient / both moments live in
   one flat ``(4, N)`` block. Swapping to the snapshot weights is a single
   ``copy_``.
4. Coordinate boundary safeguards: gradients are sanitised and bounded (the
   operative guard against overflow), the second-moment denominator is floored,
   and the final ratio is clipped, so no division can produce NaN or Inf.
   Experiment 14 shows the gradient bound is what prevents failures; the floor
   alone does not.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.optim import Optimizer

ParamsT = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]
BatchClosure = Callable[[int], Tensor]

# Rows of the persistent (4, N) state block.
_ROW_SNAPSHOT: int = 0
_ROW_MU: int = 1
_ROW_EXP_AVG: int = 2
_ROW_EXP_AVG_SQ: int = 3
_STATE_KEYS: Tuple[str, ...] = ("snapshot", "mu", "exp_avg", "exp_avg_sq")

# Rows of the transient (3, N) scratch block (never checkpointed).
_SCRATCH_G_LIVE: int = 0
_SCRATCH_G_SNAP: int = 1
_SCRATCH_BACKUP: int = 2
_N_SCRATCH_ROWS: int = 3


@dataclass
class _FlatBlock:
    """Flat, contiguous storage backing one parameter group."""

    params: List[Tensor]
    offsets: List[int]
    numels: List[int]
    total: int
    flat_params: Tensor  # (N,)  parameters live here
    state: Tensor        # (4, N) snapshot, mu, exp_avg, exp_avg_sq
    scratch: Tensor      # (3, N) live grad, snapshot grad / denominator, backup / update
    precond: Optional[Tensor] = None  # frozen diagonal D (Proposition 5), or None for adaptive scaling


class CoordinateSVRG(Optimizer):
    """SVRG with coordinate-wise adaptive momentum for low-SNR objectives.

    Usage::

        opt = CoordinateSVRG(model.parameters(), lr=0.02, snapshot_interval=64)

        def batch_closure(batch_id: int) -> Tensor:      # pure in batch_id
            rows = sampler.batch(batch_id)
            return loss_fn(model(X[rows]), y[rows])

        def chunk_closure(chunk_id: int) -> Tensor:      # mean loss over one chunk
            rows = chunks[chunk_id]
            return loss_fn(model(X[rows]), y[rows])

        for step in range(steps):
            if opt.needs_snapshot:
                opt.refresh_snapshot(chunk_closure, num_chunks=len(chunks))
            opt.step(batch_closure)                        # batch id = step count

    Notes:
        * Construct the optimizer *after* moving the model to its final device
          and dtype. Parameters are re-pointed into a flat buffer on first use.
        * Mutate parameters in place (``p.copy_``, ``p.add_``); reassigning
          ``p.data`` would detach them from the flat buffer.
        * All parameters within one group must share device and dtype.
        * Closures must be deterministic in their id argument (seed any dropout
          from the id), otherwise the alignment invariant cannot hold.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-15,
        weight_decay: float = 0.0,
        snapshot_interval: int = 50,
        update_clip: float = 10.0,
        telemetry: bool = True,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"betas must lie in [0, 1), got {betas}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if snapshot_interval < 1:
            raise ValueError(f"snapshot_interval must be >= 1, got {snapshot_interval}")
        if update_clip <= 0.0:
            raise ValueError(f"update_clip must be positive, got {update_clip}")
        defaults: Dict[str, Any] = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            snapshot_interval=snapshot_interval,
            update_clip=update_clip,
            step=0,
            steps_since_snapshot=0,
            snapshot_ready=False,
        )
        super().__init__(params, defaults)
        self.telemetry: bool = telemetry
        self.last_stats: Dict[str, float] = {}
        self.last_batch_id: Optional[int] = None
        self._blocks: Dict[int, _FlatBlock] = {}
        self._preconditioners: Dict[int, Tensor] = {}

    # ------------------------------------------------------------------ #
    # Public introspection
    # ------------------------------------------------------------------ #
    @property
    def step_count(self) -> int:
        """Number of completed optimizer steps (also the default batch id)."""
        return int(self.param_groups[0]["step"])

    @property
    def needs_snapshot(self) -> bool:
        """True if any group has no snapshot yet or its snapshot is stale."""
        for group in self.param_groups:
            if not group["snapshot_ready"]:
                return True
            if group["steps_since_snapshot"] >= group["snapshot_interval"]:
                return True
        return False

    def flat_parameters(self, group_index: int = 0) -> Tensor:
        """The flat buffer that backs every parameter of ``group_index``."""
        return self._ensure_block(group_index).flat_params

    def flat_state(self, group_index: int = 0) -> Dict[str, Tensor]:
        """Row views (snapshot, mu, exp_avg, exp_avg_sq) of the flat state block."""
        block: _FlatBlock = self._ensure_block(group_index)
        return {key: block.state[row] for row, key in enumerate(_STATE_KEYS)}

    # ------------------------------------------------------------------ #
    # Frozen diagonal preconditioner (Proposition 5)
    # ------------------------------------------------------------------ #
    def set_frozen_preconditioner(
        self, diag: Union[Tensor, Sequence[Tensor]], group_index: int = 0
    ) -> None:
        """Replace adaptive scaling by a FIXED positive diagonal ``D`` for one parameter group.

        The step becomes ``w <- w - lr * m_hat / D`` with no ratio clipping, which is the setting
        of Proposition 5 / the Johnson-Zhang theorem (use ``betas=(0.0, ...)`` for the exact
        theorem). ``diag`` is either one flat tensor with one entry per parameter of the group, or
        a sequence of tensors shaped like the group's parameters. ``D`` is clamped below at the
        same floor as the adaptive denominator. The preconditioner is not part of ``state_dict``;
        it is kept across ``load_state_dict`` on the same object, but a freshly constructed
        optimizer must be given it again.
        """
        block: _FlatBlock = self._ensure_block(group_index)
        if isinstance(diag, Tensor):
            flat: Tensor = diag.detach().reshape(-1)
        else:
            flat = torch.cat([d.detach().reshape(-1) for d in diag])
        if flat.numel() != block.total:
            raise ValueError(
                f"preconditioner has {flat.numel()} entries but group {group_index} has {block.total} parameters"
            )
        flat = flat.to(dtype=block.flat_params.dtype, device=block.flat_params.device)
        if not bool(torch.isfinite(flat).all()) or bool((flat <= 0).any()):
            raise ValueError("preconditioner must be finite and strictly positive")
        finfo = torch.finfo(block.flat_params.dtype)
        floor: float = max(float(self.param_groups[group_index]["eps"]), 4.0 * math.sqrt(finfo.tiny))
        stored: Tensor = flat.clamp(min=floor).clone()
        self._preconditioners[group_index] = stored
        block.precond = stored

    def clear_frozen_preconditioner(self, group_index: int = 0) -> None:
        """Return the group to adaptive (second-moment) scaling."""
        self._preconditioners.pop(group_index, None)
        if group_index in self._blocks:
            self._blocks[group_index].precond = None

    # ------------------------------------------------------------------ #
    # Lazy state / flat memory construction
    # ------------------------------------------------------------------ #
    def _ensure_block(self, group_index: int) -> _FlatBlock:
        existing: Optional[_FlatBlock] = self._blocks.get(group_index)
        if existing is not None:
            return existing

        group: Dict[str, Any] = self.param_groups[group_index]
        params: List[Tensor] = list(group["params"])
        if len(params) == 0:
            raise ValueError(f"param group {group_index} is empty")
        device: torch.device = params[0].device
        dtype: torch.dtype = params[0].dtype
        if not dtype.is_floating_point:
            raise ValueError(f"parameters must be floating point, got {dtype}")
        for p in params:
            if p.device != device or p.dtype != dtype:
                raise ValueError(
                    "all parameters in a group must share device and dtype; "
                    "split them into separate param groups"
                )

        numels: List[int] = [int(p.numel()) for p in params]
        offsets: List[int] = []
        running: int = 0
        for n in numels:
            offsets.append(running)
            running += n
        total: int = running

        with torch.no_grad():
            flat_params: Tensor = torch.empty(total, dtype=dtype, device=device)
            state_block: Tensor = torch.zeros(len(_STATE_KEYS), total, dtype=dtype, device=device)
            scratch: Tensor = torch.zeros(_N_SCRATCH_ROWS, total, dtype=dtype, device=device)
            for p, offset, n in zip(params, offsets, numels):
                flat_params[offset:offset + n].copy_(p.detach().reshape(-1))
                p.data = flat_params[offset:offset + n].view(p.shape)
                per_param: Dict[str, Any] = self.state[p]
                for row, key in enumerate(_STATE_KEYS):
                    view: Tensor = state_block[row, offset:offset + n]
                    if key in per_param:  # restored from a checkpoint
                        view.copy_(per_param[key].reshape(-1))
                    per_param[key] = view.view(p.shape)

        block: _FlatBlock = _FlatBlock(
            params=params,
            offsets=offsets,
            numels=numels,
            total=total,
            flat_params=flat_params,
            state=state_block,
            scratch=scratch,
        )
        block.precond = self._preconditioners.get(group_index)
        self._blocks[group_index] = block
        return block

    def _prepared_blocks(self) -> List[_FlatBlock]:
        """All blocks, verifying every group already has a snapshot."""
        blocks: List[_FlatBlock] = [
            self._ensure_block(i) for i in range(len(self.param_groups))
        ]
        for group in self.param_groups:
            if not group["snapshot_ready"]:
                raise RuntimeError(
                    "no snapshot available: call refresh_snapshot(closure, num_chunks) "
                    "before step() (check optimizer.needs_snapshot)"
                )
        return blocks

    # ------------------------------------------------------------------ #
    # Gradient evaluation helpers
    # ------------------------------------------------------------------ #
    def _evaluate(
        self,
        closure: BatchClosure,
        closure_id: int,
        blocks: List[_FlatBlock],
        scratch_row: int,
    ) -> Tensor:
        """Run ``closure(closure_id)``, backprop, gather grads into ``scratch_row``."""
        all_params: List[Tensor] = [p for block in blocks for p in block.params]
        for p in all_params:
            p.grad = None
        with torch.enable_grad():
            loss: Tensor = closure(closure_id)
            if not isinstance(loss, Tensor) or not loss.requires_grad:
                raise RuntimeError(
                    "closure must return a scalar loss tensor connected to the parameters"
                )
            loss.backward()
        with torch.no_grad():
            for block in blocks:
                target: Tensor = block.scratch[scratch_row]
                for p, offset, n in zip(block.params, block.offsets, block.numels):
                    grad: Optional[Tensor] = p.grad
                    if grad is None:
                        target[offset:offset + n].zero_()
                    else:
                        if grad.is_sparse:
                            raise RuntimeError("sparse gradients are not supported")
                        target[offset:offset + n].copy_(grad.reshape(-1))
        for p in all_params:
            p.grad = None
        return loss.detach()

    def _gradient_pair(
        self,
        closure: BatchClosure,
        batch_id: int,
        blocks: List[_FlatBlock],
    ) -> Tuple[Tensor, Tensor]:
        """Evaluate g(w_t) then g(w~) on the *same* batch id.

        Live gradients land in scratch row ``_SCRATCH_G_LIVE`` and snapshot
        gradients in ``_SCRATCH_G_SNAP``. The live weights are restored even if
        the second evaluation raises.
        """
        loss_live: Tensor = self._evaluate(closure, batch_id, blocks, _SCRATCH_G_LIVE)
        with torch.no_grad():
            for block in blocks:
                block.scratch[_SCRATCH_BACKUP].copy_(block.flat_params)
                block.flat_params.copy_(block.state[_ROW_SNAPSHOT])
        try:
            loss_snap: Tensor = self._evaluate(closure, batch_id, blocks, _SCRATCH_G_SNAP)
        finally:
            with torch.no_grad():
                for block in blocks:
                    block.flat_params.copy_(block.scratch[_SCRATCH_BACKUP])
        return loss_live, loss_snap

    # ------------------------------------------------------------------ #
    # Snapshot management
    # ------------------------------------------------------------------ #
    def refresh_snapshot(self, closure: BatchClosure, num_chunks: int = 1) -> float:
        """Freeze ``w~ <- w`` and recompute the full gradient ``mu~``.

        ``closure(chunk_id)`` must return the *mean* loss over chunk
        ``chunk_id``; chunks must be equal-sized and cover the data exactly once
        so the mean of chunk gradients equals the full gradient. Returns the
        mean loss over the pass. State is only committed after every chunk
        succeeds.
        """
        if num_chunks < 1:
            raise ValueError("num_chunks must be >= 1")
        blocks: List[_FlatBlock] = [
            self._ensure_block(i) for i in range(len(self.param_groups))
        ]
        with torch.no_grad():
            for block in blocks:
                block.scratch[_SCRATCH_BACKUP].zero_()  # accumulator
        losses: List[Tensor] = []
        for chunk_id in range(num_chunks):
            losses.append(self._evaluate(closure, chunk_id, blocks, _SCRATCH_G_LIVE))
            with torch.no_grad():
                for block in blocks:
                    block.scratch[_SCRATCH_BACKUP].add_(
                        block.scratch[_SCRATCH_G_LIVE], alpha=1.0 / num_chunks
                    )
        with torch.no_grad():
            for block in blocks:
                block.state[_ROW_MU].copy_(block.scratch[_SCRATCH_BACKUP])
                block.state[_ROW_SNAPSHOT].copy_(block.flat_params)
        for group in self.param_groups:
            group["steps_since_snapshot"] = 0
            group["snapshot_ready"] = True
        return float(torch.stack(losses).mean().item())

    def variance_reduced_gradient(
        self, closure: BatchClosure, batch_id: int
    ) -> List[Tensor]:
        """Return ``g_hat`` (one flat tensor per group) without updating anything.

        Useful to verify the Variance Reduction Law: compare against the exact
        gradient at the current weights. Includes the L2 weight-decay term when
        configured. Weights are unchanged afterwards.
        """
        blocks: List[_FlatBlock] = self._prepared_blocks()
        self._gradient_pair(closure, int(batch_id), blocks)
        results: List[Tensor] = []
        with torch.no_grad():
            for group, block in zip(self.param_groups, blocks):
                g_hat: Tensor = (
                    block.scratch[_SCRATCH_G_LIVE]
                    - block.scratch[_SCRATCH_G_SNAP]
                    + block.state[_ROW_MU]
                )
                weight_decay: float = float(group["weight_decay"])
                if weight_decay != 0.0:
                    g_hat = g_hat + weight_decay * block.flat_params
                results.append(g_hat)
        return results

    def verify_closure_determinism(
        self,
        closure: BatchClosure,
        batch_id: int = 0,
        repeats: int = 2,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        """Check that ``closure(batch_id)`` is a deterministic function of its id.

        SVRG's variance reduction needs the live and snapshot gradients to see identical data. The engine
        passes the same id to both, but if the closure itself is not a pure function of that id (a shuffling
        data loader, dropout, augmentation, a stateful counter), the two evaluations differ and the control
        variate silently stops working (Experiment 14). This evaluates the closure ``repeats`` times at the
        CURRENT weights and compares losses and gradients. Small tolerances absorb benign floating-point
        summation-order noise on GPUs. Weights and optimizer state are not modified.

        Returns True if every repeat agrees with the first, False otherwise.
        """
        if repeats < 2:
            raise ValueError("repeats must be at least 2")
        blocks: List[_FlatBlock] = [self._ensure_block(i) for i in range(len(self.param_groups))]
        first_loss: Optional[Tensor] = None
        first_grads: List[Tensor] = []
        for _ in range(repeats):
            loss: Tensor = self._evaluate(closure, int(batch_id), blocks, _SCRATCH_G_LIVE)
            grads: List[Tensor] = [block.scratch[_SCRATCH_G_LIVE].clone() for block in blocks]
            if first_loss is None:
                first_loss, first_grads = loss.clone(), grads
                continue
            if not torch.allclose(loss, first_loss, rtol=rtol, atol=atol):
                return False
            for current, reference in zip(grads, first_grads):
                if not torch.allclose(current, reference, rtol=rtol, atol=atol):
                    return False
        return True

    # ------------------------------------------------------------------ #
    # The optimisation step
    # ------------------------------------------------------------------ #
    def step(  # type: ignore[override]
        self,
        closure: Optional[BatchClosure] = None,
        batch_id: Optional[int] = None,
    ) -> Optional[Tensor]:
        """One SVRG inner step. Returns the live mini-batch loss at ``w_t``.

        ``closure(batch_id)`` is invoked twice with the same id: once at the
        live weights and once at the snapshot weights. ``batch_id`` defaults to
        the global step counter, so successive steps shift to the next batch.
        """
        if closure is None:
            raise ValueError("CoordinateSVRG.step requires a closure(batch_id) -> loss")
        start_ns: int = time.perf_counter_ns()
        blocks: List[_FlatBlock] = self._prepared_blocks()
        active_batch: int = self.step_count if batch_id is None else int(batch_id)

        loss_live, _ = self._gradient_pair(closure, active_batch, blocks)

        per_block_stats: List[Optional[List[float]]] = []
        with torch.no_grad():
            for group, block in zip(self.param_groups, blocks):
                per_block_stats.append(self._update_block(group, block))
        self.last_batch_id = active_batch

        if self.telemetry:
            self._record_stats(per_block_stats, loss_live, active_batch, start_ns)
        return loss_live

    def _update_block(self, group: Dict[str, Any], block: _FlatBlock) -> Optional[List[float]]:
        beta1, beta2 = group["betas"]
        lr: float = float(group["lr"])
        eps: float = float(group["eps"])
        weight_decay: float = float(group["weight_decay"])
        update_clip: float = float(group["update_clip"])

        group["step"] += 1
        group["steps_since_snapshot"] += 1
        t: int = int(group["step"])

        dtype: torch.dtype = block.flat_params.dtype
        finfo = torch.finfo(dtype)
        grad_bound: float = 0.25 * math.sqrt(finfo.max)   # keeps g^2 finite
        denom_floor: float = max(eps, 4.0 * math.sqrt(finfo.tiny))
        bias1: float = 1.0 - beta1 ** t
        bias2: float = 1.0 - beta2 ** t

        params: Tensor = block.flat_params
        snapshot: Tensor = block.state[_ROW_SNAPSHOT]
        mu: Tensor = block.state[_ROW_MU]
        exp_avg: Tensor = block.state[_ROW_EXP_AVG]
        exp_avg_sq: Tensor = block.state[_ROW_EXP_AVG_SQ]
        g_hat: Tensor = block.scratch[_SCRATCH_G_LIVE]
        g_snap: Tensor = block.scratch[_SCRATCH_G_SNAP]

        live_norm: Optional[Tensor] = None
        correction_norm: Optional[Tensor] = None
        mu_norm: Optional[Tensor] = None
        snapshot_distance: Optional[Tensor] = None
        if self.telemetry:
            live_norm = torch.linalg.vector_norm(g_hat)
            correction_norm = torch.linalg.vector_norm(g_hat - g_snap)
            mu_norm = torch.linalg.vector_norm(mu)
            snapshot_distance = torch.linalg.vector_norm(params - snapshot)

        # g_hat = g(w_t) - g(w~) + mu~   (in place in the live-gradient row)
        g_hat.sub_(g_snap).add_(mu)
        if weight_decay != 0.0:
            g_hat.add_(params, alpha=weight_decay)

        nonfinite: Optional[Tensor] = None
        if self.telemetry:
            nonfinite = (~torch.isfinite(g_hat)).sum().to(dtype)
        g_hat.nan_to_num_(nan=0.0, posinf=grad_bound, neginf=-grad_bound)
        g_hat.clamp_(-grad_bound, grad_bound)

        exp_avg.mul_(beta1).add_(g_hat, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(g_hat, g_hat, value=1.0 - beta2)

        update: Tensor = block.scratch[_SCRATCH_BACKUP]  # backup was restored already
        denom: Tensor
        if block.precond is not None:
            # Frozen diagonal preconditioner: w <- w - lr * m_hat / D (no ratio clip, theory setting).
            denom = block.precond
            torch.div(exp_avg, denom, out=update)
            update.div_(bias1)
        else:
            # Coordinate boundary safeguard: floor the denominator (the gradient bound above is what prevents overflow).
            denom = g_snap  # snapshot gradients are no longer needed
            torch.sqrt(exp_avg_sq, out=denom)
            denom.div_(math.sqrt(bias2))
            denom.clamp_(min=denom_floor)
            torch.div(exp_avg, denom, out=update)
            update.div_(bias1)
            update.clamp_(-update_clip, update_clip)
        params.add_(update, alpha=-lr)

        if not self.telemetry:
            return None
        assert live_norm is not None and correction_norm is not None
        assert mu_norm is not None and snapshot_distance is not None and nonfinite is not None
        packed: Tensor = torch.stack(
            [
                live_norm,
                correction_norm,
                mu_norm,
                snapshot_distance,
                torch.linalg.vector_norm(g_hat),
                torch.linalg.vector_norm(update),
                nonfinite,
                denom.min(),
            ]
        )
        return [float(x) for x in packed.tolist()]

    def _record_stats(
        self,
        per_block_stats: List[Optional[List[float]]],
        loss_live: Tensor,
        batch_id: int,
        start_ns: int,
    ) -> None:
        """Fold per-group telemetry into ``self.last_stats`` (microsecond timing)."""
        sq: List[float] = [0.0] * 6
        nonfinite_total: float = 0.0
        min_denominator: float = math.inf
        for stats in per_block_stats:
            if stats is None:
                continue
            for i in range(6):
                sq[i] += stats[i] * stats[i]
            nonfinite_total += stats[6]
            min_denominator = min(min_denominator, stats[7])
        loss_value: float = float(loss_live.item())
        end_ns: int = time.perf_counter_ns()  # .tolist()/.item() already synchronised
        self.last_stats = {
            "step": float(self.step_count),
            "batch_id": float(batch_id),
            "loss": loss_value,
            "live_grad_norm": math.sqrt(sq[0]),
            "correction_norm": math.sqrt(sq[1]),
            "full_grad_norm": math.sqrt(sq[2]),
            "snapshot_distance": math.sqrt(sq[3]),
            "vr_grad_norm": math.sqrt(sq[4]),
            "update_norm": math.sqrt(sq[5]),
            "nonfinite_count": nonfinite_total,
            "min_denominator": min_denominator,
            "steps_since_snapshot": float(self.param_groups[0]["steps_since_snapshot"]),
            "step_ns": float(end_ns - start_ns),
            "step_us": float(end_ns - start_ns) / 1000.0,
        }

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load a checkpoint, then rebuild the flat blocks from the loaded state.

        ``Optimizer.load_state_dict`` replaces per-parameter state with fresh
        tensors. Dropping the cached blocks forces ``_ensure_block`` to
        re-flatten and copy those values back into contiguous storage.
        """
        super().load_state_dict(state_dict)
        self._blocks.clear()
