"""Benchmark: CoordinateSVRG vs Adam vs SGD-momentum on the toxic-market landscape.

All three optimizers see the same data, the same initial weights and the same
deterministic batch sequence. Progress is logged against *sample-gradient
evaluations*, not steps, because an SVRG step costs two mini-batch gradients
plus a periodic full pass; plotting per step would flatter it.

Run from the repo root:
    python -m benchmarks.compare_optimizers --wandb-mode offline
    python -m benchmarks.compare_optimizers --device cuda --wandb-mode online
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import (  # noqa: E402
    DeterministicBatchSampler,
    MarketDataset,
    RegimeConfig,
    generate_market,
    load_config,
)
from src.torch_optimizer import CoordinateSVRG  # noqa: E402

try:
    import wandb

    HAS_WANDB: bool = True
except ImportError:  # pragma: no cover
    HAS_WANDB = False


class Problem:
    """Least-squares landscape on one device, with float64 reference metrics."""

    def __init__(self, dataset: MarketDataset, batch_size: int, chunks: int, device: str) -> None:
        self.dataset: MarketDataset = dataset
        self.device: str = device
        self.features: torch.Tensor = torch.as_tensor(
            dataset.features, dtype=torch.float32, device=device
        )
        self.targets: torch.Tensor = torch.as_tensor(
            dataset.targets, dtype=torch.float32, device=device
        )
        self.sampler = DeterministicBatchSampler(
            dataset.n_samples, batch_size, dataset.config.seed
        )
        self.chunk_rows: List[np.ndarray] = self.sampler.full_pass_chunks(chunks)
        self.batch_size: int = batch_size
        self.optimal_loss: float = dataset.loss(dataset.optimal_weights)

    def loss_on(self, w: torch.Tensor, rows: np.ndarray) -> torch.Tensor:
        idx: torch.Tensor = torch.as_tensor(rows, device=self.device)
        residual: torch.Tensor = self.features[idx] @ w - self.targets[idx]
        return 0.5 * (residual * residual).mean()

    def metrics(self, w: torch.Tensor) -> Dict[str, float]:
        w64: np.ndarray = w.detach().cpu().double().numpy()
        return {
            "loss_gap": max(self.dataset.loss(w64) - self.optimal_loss, 0.0),
            "dist_to_optimum": float(np.linalg.norm(w64 - self.dataset.optimal_weights)),
        }


def _start_run(name: str, cfg: Dict[str, Any], mode: str, extra: Dict[str, Any]) -> Optional[Any]:
    if not HAS_WANDB or mode == "disabled":
        return None
    tag: str = str(extra.get("tag", ""))
    return wandb.init(
        project=str(cfg["wandb"]["project"]),
        group=f"optimizer-comparison/{tag}" if tag else "optimizer-comparison",
        name=f"{name}/{tag}" if tag else name,
        mode=mode,
        config=extra,
        reinit=True,
    )


def run_baseline(
    name: str,
    problem: Problem,
    make_opt: Callable[[List[torch.nn.Parameter]], torch.optim.Optimizer],
    steps: int,
    log_every: int,
    cfg: Dict[str, Any],
    mode: str,
    tag: str = "",
) -> Dict[str, float]:
    w = torch.nn.Parameter(torch.zeros(problem.dataset.n_features, device=problem.device))
    opt: torch.optim.Optimizer = make_opt([w])
    run = _start_run(name, cfg, mode, {"optimizer": name, "steps": steps, "tag": tag})
    for k in range(steps):
        opt.zero_grad(set_to_none=True)
        loss: torch.Tensor = problem.loss_on(w, problem.sampler.batch(k))
        loss.backward()
        opt.step()
        if (k + 1) % log_every == 0 or k + 1 == steps:
            row: Dict[str, float] = problem.metrics(w)
            row["grad_evals"] = float((k + 1) * problem.batch_size)
            if run is not None:
                wandb.log(row, step=k + 1)
    final: Dict[str, float] = problem.metrics(w)
    if run is not None:
        run.finish()
    return final


def run_svrg(
    problem: Problem,
    steps: int,
    log_every: int,
    cfg: Dict[str, Any],
    mode: str,
    tag: str = "",
) -> Dict[str, float]:
    opt_cfg: Dict[str, Any] = dict(cfg["optimizer"])
    w = torch.nn.Parameter(torch.zeros(problem.dataset.n_features, device=problem.device))
    opt = CoordinateSVRG(
        [w],
        lr=float(opt_cfg["lr"]),
        betas=(float(opt_cfg["betas"][0]), float(opt_cfg["betas"][1])),
        eps=float(opt_cfg["eps"]),
        weight_decay=float(opt_cfg["weight_decay"]),
        snapshot_interval=int(opt_cfg["snapshot_interval"]),
        update_clip=float(opt_cfg["update_clip"]),
        telemetry=bool(opt_cfg["telemetry"]),
    )
    num_chunks: int = len(problem.chunk_rows)
    run = _start_run("svrg", cfg, mode, {"optimizer": "svrg", "steps": steps, "tag": tag, **opt_cfg})

    def batch_closure(batch_id: int) -> torch.Tensor:
        return problem.loss_on(w, problem.sampler.batch(batch_id))

    def chunk_closure(chunk_id: int) -> torch.Tensor:
        return problem.loss_on(w, problem.chunk_rows[chunk_id])

    grad_evals: float = 0.0
    for k in range(steps):
        if opt.needs_snapshot:
            opt.refresh_snapshot(chunk_closure, num_chunks=num_chunks)
            grad_evals += float(problem.dataset.n_samples)
        opt.step(batch_closure)
        grad_evals += 2.0 * problem.batch_size          # live + snapshot gradient
        if (k + 1) % log_every == 0 or k + 1 == steps:
            row: Dict[str, float] = problem.metrics(w)
            row["grad_evals"] = grad_evals
            row.update({f"svrg/{key}": val for key, val in opt.last_stats.items()
                        if math.isfinite(val)})
            if run is not None:
                wandb.log(row, step=k + 1)
    final: Dict[str, float] = problem.metrics(w)
    final["grad_evals"] = grad_evals
    if run is not None:
        run.finish()
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "stochastic_regime.yaml"))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument(
        "--noise-regimes", nargs="+", default=None,
        help="names from noise_sigma_levels (default: the config's noise_regime)",
    )
    parser.add_argument("--seeds", type=int, default=1, help="number of dataset seeds per regime")
    args = parser.parse_args()

    cfg: Dict[str, Any] = load_config(args.config)
    mode: str = args.wandb_mode or str(cfg["wandb"]["mode"])
    steps: int = args.steps or int(cfg["training"]["inner_steps"])
    levels: Dict[str, float] = dict(cfg["environment"]["noise_sigma_levels"])
    regimes: List[str] = args.noise_regimes or [str(cfg["environment"]["noise_regime"])]
    for regime in regimes:
        if regime not in levels:
            raise SystemExit(f"unknown regime '{regime}', choose from {sorted(levels)}")

    lr: float = float(cfg["optimizer"]["lr"])
    base_seed: int = int(cfg["seed"])
    optimizers: List[str] = ["svrg", "adam", "sgd_momentum"]
    summary: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

    for regime in regimes:
        summary[regime] = {name: [] for name in optimizers}
        for seed_offset in range(args.seeds):
            run_cfg: Dict[str, Any] = dict(cfg)
            run_cfg["seed"] = base_seed + seed_offset
            run_cfg["environment"] = {**cfg["environment"], "noise_regime": regime}
            dataset: MarketDataset = generate_market(RegimeConfig.from_dict(run_cfg))
            problem = Problem(
                dataset,
                batch_size=int(cfg["training"]["batch_size"]),
                chunks=int(cfg["training"]["snapshot_chunks"]),
                device=args.device,
            )
            tag: str = f"{regime}/seed{base_seed + seed_offset}"
            print(f"[{tag}] device={args.device}  SNR={dataset.empirical_snr_db:.2f} dB  steps={steps}")
            summary[regime]["svrg"].append(run_svrg(problem, steps, args.log_every, run_cfg, mode, tag))
            summary[regime]["adam"].append(run_baseline(
                "adam", problem, lambda p: torch.optim.Adam(p, lr=lr),
                steps, args.log_every, run_cfg, mode, tag,
            ))
            summary[regime]["sgd_momentum"].append(run_baseline(
                "sgd_momentum", problem, lambda p: torch.optim.SGD(p, lr=lr, momentum=0.9),
                steps, args.log_every, run_cfg, mode, tag,
            ))

    print(f"\nmedian over {args.seeds} seed(s)")
    print(f"{'regime':<10}{'optimizer':<14}{'loss gap':>14}{'dist to w*':>14}")
    for regime in regimes:
        for name in optimizers:
            gaps: List[float] = [r["loss_gap"] for r in summary[regime][name]]
            dists: List[float] = [r["dist_to_optimum"] for r in summary[regime][name]]
            print(f"{regime:<10}{name:<14}{float(np.median(gaps)):>14.3e}{float(np.median(dists)):>14.3e}")


if __name__ == "__main__":
    main()
