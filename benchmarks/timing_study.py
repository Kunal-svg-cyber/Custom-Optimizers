"""GPU timing study: when, if ever, does SVRG beat a direct solve in WALL-CLOCK time?

Everything else in this repository counts sample-gradient evaluations on one CPU core. This script measures
seconds on a GPU (float32) for least squares with Gaussian features, using the repository's own PyTorch engine
for the SVRG runs, and reports:

  1. time to reach a target excess loss for: direct normal equations, direct ``lstsq``, Jacobi-preconditioned
     SVRG (``CoordinateSVRG.set_frozen_preconditioner``, telemetry off), and Adam (untuned, lr = 0.01);
  2. the per-step cost of the engine with telemetry on versus off (host-synchronisation overhead);
  3. peak extra GPU memory (beyond the data) of each method.

Design choices that keep the comparison honest
  * Rows of ``X`` are already in random order, so a mini-batch is a contiguous slice (zero-copy, no host-to-device
    index traffic) and the closure is a pure function of its batch id (alignment invariant holds by construction).
  * Convergence checks (one extra pass) are excluded from the timed region; the cap on epochs is reported.
  * SVRG hyper-parameters are fixed by a simple rule, not tuned: ``lr = 0.2 / (1 + sqrt(d / b))^2`` and one
    snapshot per pass, with ``b = max(256, d)``. Adam is also untuned. Treat both as indicative, not optimal.
  * Gaussian features are well conditioned, which is the best case for SVRG; badly conditioned problems are not covered.

Usage on a Colab T4 (repo root):
    python -m benchmarks.timing_study --sizes 100000x256 200000x1024 200000x2048
    python -m benchmarks.timing_study --device cpu --sizes 20000x64     # slow smoke test, no GPU needed

Nothing here has been run by the author's sandbox (no torch available); it is provided to be run and reported.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.torch_optimizer import CoordinateSVRG  # noqa: E402


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def make_problem(n: int, d: int, device: str, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn(n, d, device=device, generator=g)
    w_true = torch.randn(d, device=device, generator=g) / math.sqrt(d)
    y = x @ w_true + torch.randn(n, device=device, generator=g)      # unit-variance noise: low signal-to-noise
    return x, y


def loss_full(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> float:
    r = x @ w - y
    return float(0.5 * (r * r).mean().item())


def timed(device: str, fn: Callable[[], Any]) -> Tuple[Any, float, float]:
    """Run ``fn``; return (output, seconds, peak extra memory in MB)."""
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    else:
        base = 0
    sync(device)
    t0 = time.perf_counter()
    out = fn()
    sync(device)
    seconds = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20 if device.startswith("cuda") else float("nan")
    return out, seconds, peak


def run_iterative(
    name: str, x: torch.Tensor, y: torch.Tensor, w_ref_loss: float, loss0: float, target: float,
    batch: int, max_epochs: int, device: str,
) -> Dict[str, Any]:
    n, d = x.shape
    nb = n // batch
    w = torch.nn.Parameter(torch.zeros(d, device=device))
    if name == "svrg_jacobi":
        lr = 0.2 / (1.0 + math.sqrt(d / batch)) ** 2
        opt = CoordinateSVRG([w], lr=lr, betas=(0.0, 0.999), snapshot_interval=nb, telemetry=False)
        opt.set_frozen_preconditioner((x ** 2).mean(dim=0))
        chunks = 8
        csize = n // chunks

        def batch_closure(i: int) -> torch.Tensor:
            s = (i % nb) * batch
            r = x[s:s + batch] @ w - y[s:s + batch]
            return 0.5 * (r * r).mean()

        def chunk_closure(c: int) -> torch.Tensor:
            r = x[c * csize:(c + 1) * csize] @ w - y[c * csize:(c + 1) * csize]
            return 0.5 * (r * r).mean()

        def one_epoch() -> None:
            opt.refresh_snapshot(chunk_closure, num_chunks=chunks)
            for _ in range(nb):
                opt.step(batch_closure)
    else:
        lr = 0.01
        opt_adam = torch.optim.Adam([w], lr=lr)

        def one_epoch() -> None:
            for i in range(nb):
                s = i * batch
                opt_adam.zero_grad(set_to_none=True)
                r = x[s:s + batch] @ w - y[s:s + batch]
                (0.5 * (r * r).mean()).backward()
                opt_adam.step()

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    else:
        base = 0
    elapsed, epochs, reached = 0.0, 0, False
    while epochs < max_epochs:
        sync(device)
        t0 = time.perf_counter()
        one_epoch()
        sync(device)
        elapsed += time.perf_counter() - t0                       # checks below are NOT timed
        epochs += 1
        gap = loss_full(x, y, w.detach()) - w_ref_loss
        if not math.isfinite(gap):
            break
        if gap <= target * loss0:
            reached = True
            break
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20 if device.startswith("cuda") else float("nan")
    return {"method": name, "seconds": elapsed, "epochs": epochs, "reached": reached, "peak_extra_mb": peak, "lr": lr}


def step_cost_microbenchmark(x: torch.Tensor, y: torch.Tensor, batch: int, device: str, steps: int = 200) -> Dict[str, float]:
    """Microseconds per SVRG step with telemetry on versus off (isolates host-synchronisation overhead)."""
    n, d = x.shape
    nb = n // batch
    out: Dict[str, float] = {}
    for telemetry in (True, False):
        w = torch.nn.Parameter(torch.zeros(d, device=device))
        opt = CoordinateSVRG([w], lr=1e-3, betas=(0.0, 0.999), snapshot_interval=10 ** 9, telemetry=telemetry)

        def batch_closure(i: int) -> torch.Tensor:
            s = (i % nb) * batch
            r = x[s:s + batch] @ w - y[s:s + batch]
            return 0.5 * (r * r).mean()

        opt.refresh_snapshot(lambda c: batch_closure(c), num_chunks=1)
        for _ in range(10):                                           # warm-up
            opt.step(batch_closure)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(steps):
            opt.step(batch_closure)
        sync(device)
        out["telemetry_on_us" if telemetry else "telemetry_off_us"] = (time.perf_counter() - t0) / steps * 1e6
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="+", default=["100000x256", "200000x1024", "200000x2048"],
                        help="n x d pairs; n must be divisible by 8")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target", type=float, default=1e-4, help="target excess loss as a fraction of the initial excess")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "timing"))
    args = parser.parse_args()

    device: str = args.device
    gpu_name = torch.cuda.get_device_name(0) if device.startswith("cuda") else "CPU"
    print(f"device: {device} ({gpu_name}), torch {torch.__version__}")
    rows: List[Dict[str, Any]] = []
    micro: Dict[str, Dict[str, float]] = {}
    for spec in args.sizes:
        n, d = (int(v) for v in spec.lower().split("x"))
        if n % 8 != 0:
            raise SystemExit(f"n={n} must be divisible by 8 (snapshot chunks)")
        batch = max(256, d)
        print(f"\n== n={n}, d={d}, batch={batch}, target={args.target:g} of the initial excess loss ==")
        try:
            x, y = make_problem(n, d, device, args.seed)
        except RuntimeError as err:
            print(f"  could not allocate the problem ({err}); skipping")
            continue
        loss0 = loss_full(x, y, torch.zeros(d, device=device))

        def normal_equations() -> torch.Tensor:
            a = x.T @ x / n
            b = x.T @ y / n
            return torch.linalg.solve(a, b)

        w_ref, t_normal, mem_normal = timed(device, normal_equations)
        w_ref_loss = loss_full(x, y, w_ref)
        loss0 -= w_ref_loss                                            # initial excess loss
        rows.append({"n": n, "d": d, "method": "direct_normal_equations", "seconds": t_normal, "epochs": None,
                     "reached": True, "peak_extra_mb": mem_normal})
        try:
            _, t_lstsq, mem_lstsq = timed(device, lambda: torch.linalg.lstsq(x, y.unsqueeze(1)).solution)
            rows.append({"n": n, "d": d, "method": "direct_lstsq", "seconds": t_lstsq, "epochs": None,
                         "reached": True, "peak_extra_mb": mem_lstsq})
        except RuntimeError as err:
            print(f"  lstsq failed ({str(err)[:80]}); skipping")
        for name in ("svrg_jacobi", "adam"):
            res = run_iterative(name, x, y, w_ref_loss, loss0, args.target, batch, args.max_epochs, device)
            rows.append({"n": n, "d": d, **res})
        micro[spec] = step_cost_microbenchmark(x, y, batch, device)
        del x, y
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    lines = [f"# GPU timing study ({gpu_name}, torch {torch.__version__})", "",
             f"Target: excess loss at most {args.target:g} x the initial excess; epoch cap {args.max_epochs}. "
             "Convergence checks are excluded from the timed region. SVRG and Adam hyper-parameters are fixed by a simple rule, not tuned.", "",
             "| n | d | method | seconds | epochs | reached target | peak extra memory (MB) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['n']} | {r['d']} | {r['method']} | {r['seconds']:.3f} | {r['epochs'] if r['epochs'] is not None else '-'} | "
                     f"{'yes' if r['reached'] else 'NO'} | {r['peak_extra_mb']:.0f} |")
    lines += ["", "Engine step cost in its default adaptive mode (microseconds per SVRG step, 200 steps after warm-up):", "",
              "| size | telemetry on | telemetry off | ratio |", "|---|---|---|---|"]
    for spec, m in micro.items():
        lines.append(f"| {spec} | {m['telemetry_on_us']:.0f} | {m['telemetry_off_us']:.0f} | {m['telemetry_on_us'] / m['telemetry_off_us']:.1f}x |")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "timing.json").write_text(json.dumps({"rows": rows, "step_cost": micro, "device": gpu_name}, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}/report.md and timing.json")


if __name__ == "__main__":
    main()
