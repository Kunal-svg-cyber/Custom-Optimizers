"""Compiled-kernel latency study: what do the algorithms cost per operation, without Python or GPU overhead?

An external review rated the repository poorly for high-frequency trading, citing a 2 ms step on a GPU. That number
measures PyTorch with host synchronisation on a 32-parameter problem, not the algorithm. This script compiles
``benchmarks/svrg_kernel.c`` and measures, in nanoseconds per operation on one CPU core:

  * the Jacobi-preconditioned SVRG inner step (batch sizes 1, 8, 64) and the snapshot (full-gradient) pass;
  * recursive least squares (RLS), the usual per-tick online estimator, and a plain SGD step;
  * the signal-inference path of a linear model (one dot product).

It first runs a differential test: the C kernel must reproduce the NumPy oracle's weights.

What this does and does not show
  * It shows the cost of the *algorithm* in compiled code. It does NOT make this repository a production
    low-latency system (no thread pinning, lock-free queues, fixed-point arithmetic or kernel bypass).
  * Model *fitting* (this optimizer) and signal *inference* (a dot product) are different paths; latency budgets of
    tens of microseconds apply to inference. A refit on a trailing window can run off the critical path.
  * Numbers are specific to the machine that ran them (recorded in the output) and are noisy on shared hardware.

Usage (repo root):
    python -m benchmarks.latency_kernel                  # writes results/latency.json and prints a table
    python -m benchmarks.latency_kernel --quick          # small settings
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reference_numpy import ReferenceConfig, ReferenceSVRG  # noqa: E402

DoubleArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int32]
C_SOURCE: Path = REPO_ROOT / "benchmarks" / "svrg_kernel.c"


class Kernel:
    """ctypes wrapper around the compiled kernel."""

    def __init__(self, library: Path) -> None:
        self.lib = ctypes.CDLL(str(library))
        dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self.lib.svrg_run.argtypes = [dp, dp, dp, ctypes.c_int, ctypes.c_int, ip, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_double, dp]
        self.lib.svrg_run.restype = None
        self.lib.time_svrg_step_ns.argtypes = [dp, dp, dp, ctypes.c_int, ctypes.c_int, ip, ctypes.c_int,
                                               ctypes.c_int, ctypes.c_double, ctypes.c_int]
        self.lib.time_svrg_step_ns.restype = ctypes.c_double
        self.lib.time_snapshot_ns.argtypes = [dp, dp, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.time_snapshot_ns.restype = ctypes.c_double
        self.lib.time_sgd_step_ns.argtypes = [dp, dp, ctypes.c_int, ip, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.time_sgd_step_ns.restype = ctypes.c_double
        self.lib.time_rls_update_ns.argtypes = [dp, dp, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int]
        self.lib.time_rls_update_ns.restype = ctypes.c_double
        self.lib.time_dot_ns.argtypes = [dp, dp, ctypes.c_int, ctypes.c_int, ctypes.c_int, dp]
        self.lib.time_dot_ns.restype = ctypes.c_double

    def run(self, x: DoubleArray, y: DoubleArray, inv_diag: DoubleArray, idx: IntArray, interval: int,
            lr: float) -> DoubleArray:
        n, d = x.shape
        k, b = idx.shape
        w = np.zeros(d)
        self.lib.svrg_run(x, y, inv_diag, n, d, idx, k, b, interval, lr, w)
        return w


def build(out_dir: Path) -> Path:
    """Compile the kernel; raises RuntimeError if no C compiler is available."""
    compiler = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    if compiler is None:
        raise RuntimeError("no C compiler found (gcc/cc/clang)")
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    target = out_dir / f"libsvrg{suffix}"
    subprocess.run([compiler, "-O3", "-march=native", "-shared", "-fPIC", "-o", str(target), str(C_SOURCE), "-lm"],
                   check=True, capture_output=True, text=True)
    return target


def differential_check(kernel: Kernel, seed: int = 0) -> float:
    """Maximum absolute difference between the C kernel and the NumPy oracle after 200 steps."""
    rng = np.random.default_rng(seed)
    n, d, b, steps, interval, lr = 2048, 16, 8, 200, 25, 0.05
    x = rng.standard_normal((n, d)) * np.exp(rng.uniform(-1.0, 1.0, d))      # unequal column scales
    y = x @ rng.standard_normal(d) + rng.standard_normal(n)
    diag = np.mean(x ** 2, axis=0)
    idx = rng.integers(0, n, size=(steps, b)).astype(np.int32)
    w_c = kernel.run(np.ascontiguousarray(x), y, 1.0 / diag, idx, interval, lr)

    ref = ReferenceSVRG(
        ReferenceConfig(lr=lr, beta1=0.0, snapshot_interval=interval, variance_reduction=True, adaptive=False), d
    )
    ref.set_preconditioner(diag)
    w = np.zeros(d)
    for k in range(steps):
        if ref.needs_snapshot:
            ref.refresh_snapshot(w, x.T @ (x @ w - y) / n)
        rows = idx[k].astype(np.int64)
        g = lambda v: x[rows].T @ (x[rows] @ v - y[rows]) / b          # noqa: E731
        w = ref.step(w, g(w), g(ref.snapshot))                         # type: ignore[arg-type]
    return float(np.max(np.abs(w_c - w)))


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def benchmark(kernel: Kernel, dims: List[int], window: int, steps: int, trials: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for d in dims:
        x = np.ascontiguousarray(rng.standard_normal((window, d)))
        y = x @ rng.standard_normal(d) + rng.standard_normal(window)
        inv_diag = 1.0 / np.mean(x ** 2, axis=0)
        w = rng.standard_normal(d)
        sink = np.zeros(1)
        row: Dict[str, Any] = {"d": d, "window": window}
        for b in (1, 8, 64):
            idx = rng.integers(0, window, size=(steps, b)).astype(np.int32)
            row[f"svrg_step_b{b}_ns"] = kernel.lib.time_svrg_step_ns(x, y, inv_diag, window, d, idx, steps, b, 1.0, trials)
            row[f"sgd_step_b{b}_ns"] = kernel.lib.time_sgd_step_ns(x, y, d, idx, steps, b, trials)
        row["snapshot_ns"] = kernel.lib.time_snapshot_ns(x, y, window, d, trials)
        row["rls_update_ns"] = kernel.lib.time_rls_update_ns(x, y, d, steps, 0.999, trials)
        row["dot_inference_ns"] = kernel.lib.time_dot_ns(x, w, d, steps, trials, sink)
        rows.append(row)
    return {"rows": rows, "cpu": cpu_model(), "compiler_flags": "-O3 -march=native", "python": platform.python_version(),
            "steps_per_timing": steps, "trials": trials}


def format_table(result: Dict[str, Any], max_err: float) -> List[str]:
    lines = [f"# Compiled-kernel latency ({result['cpu']}, one core)", "",
             f"Differential test against the NumPy oracle: max |C - oracle| = {max_err:.2e} after 200 steps.", "",
             f"Best of {result['trials']} timings of {result['steps_per_timing']} operations each; "
             "nanoseconds per operation unless stated. Machine-specific and noisy.", "",
             "| d | dot product (inference) | RLS update | SGD step b=1 | SVRG step b=1 | SVRG step b=8 | SVRG step b=64 | snapshot over window (ms) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in result["rows"]:
        lines.append(f"| {r['d']} | {r['dot_inference_ns']:.0f} | {r['rls_update_ns']:.0f} | {r['sgd_step_b1_ns']:.0f} | "
                     f"{r['svrg_step_b1_ns']:.0f} | {r['svrg_step_b8_ns']:.0f} | {r['svrg_step_b64_ns']:.0f} | "
                     f"{r['snapshot_ns'] / 1e6:.2f} (window {r['window']:,}) |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dims", nargs="+", type=int, default=[8, 32, 128])
    parser.add_argument("--window", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=str(REPO_ROOT / "results"))
    args = parser.parse_args()
    if args.quick:
        args.window, args.steps, args.trials, args.dims = 5000, 2000, 3, [8, 32]
    with tempfile.TemporaryDirectory() as tmp:
        kernel = Kernel(build(Path(tmp)))
        max_err = differential_check(kernel, args.seed)
        print(f"differential test vs NumPy oracle: max |C - oracle| = {max_err:.2e}")
        if max_err > 1e-9:
            raise SystemExit("C kernel disagrees with the oracle; refusing to report timings")
        result = benchmark(kernel, args.dims, args.window, args.steps, args.trials, args.seed)
    result["differential_max_abs_error"] = max_err
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "latency.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    lines = format_table(result, max_err)
    (out / "latency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
