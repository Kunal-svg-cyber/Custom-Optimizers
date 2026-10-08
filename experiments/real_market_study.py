"""Walk-forward study on REAL market data (or a synthetic stand-in for smoke tests).

Question: when a linear signal model is re-fitted on noisy daily returns, does the choice of
optimizer (closed-form OLS, ridge, budget-limited Adam, budget-limited Jacobi-SVRG) change the
out-of-sample result? The honest prior is that predictability of daily returns is tiny, so most
differences will be statistically indistinguishable from zero; the script reports confidence
intervals and a placebo (circularly shifted targets) so that a null is reported as a null.

Leakage controls
    * features at day t use closes up to day t only (``build_features``; unit-tested for causality);
    * the target for day t is the log return from close t to close t+1;
    * a refit on day R trains on rows t <= R-1 (their targets are known by close R) and predicts
      rows R .. R+H-1; feature standardisation and target clipping use training rows only;
    * everything is deterministic given the seed.

Data sources (first that applies)
    --synthetic signal|null   generated stand-in (smoke test; ``null`` has no predictability)
    --csv-dir DIR             one CSV per asset with Date and Close (Adj Close preferred) and Volume columns
    default                   Yahoo Finance via ``yfinance`` (``pip install yfinance``; needs internet)

Usage (repo root):
    python -m experiments.real_market_study --synthetic signal --out /tmp/rm        # smoke test
    python -m experiments.real_market_study --tickers SPY QQQ IWM EFA EEM TLT GLD XLF XLE XLK \\
        --start 2008-01-01 --end 2025-12-31 --placebo
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import DeterministicBatchSampler  # noqa: E402
from src.preconditioning import jacobi_diag_least_squares  # noqa: E402
from src.reference_numpy import ReferenceConfig, ReferenceSVRG  # noqa: E402

FloatArray = npt.NDArray[np.float64]
LAGS: Tuple[int, ...] = (1, 2, 3, 5, 10, 21)
FEATURE_NAMES: List[str] = [f"ret_{k}d" for k in LAGS] + ["vol_5d", "vol_21d", "log_volume_z"]
METHODS: List[str] = ["ols", "ridge", "adam_budget", "svrg_jacobi_budget", "mom21"]
LABELS: Dict[str, str] = {
    "ols": "OLS (closed form)",
    "ridge": "Ridge (validated)",
    "adam_budget": "Adam, budget-limited",
    "svrg_jacobi_budget": "Jacobi SVRG, budget-limited",
    "mom21": "21-day momentum (no fitting)",
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def synthetic_panel(
    kind: str, n_assets: int, n_days: int, seed: int, strength: float = 1.0
) -> Tuple[FloatArray, FloatArray]:
    """Heavy-tailed daily returns; ``signal`` adds weak, genuine predictability from lagged returns
    (``strength`` scales it; 1.0 is deliberately weak, like real daily data)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_t(4, size=(n_days, n_assets)) * 0.01 / math.sqrt(2.0)
    rets = np.zeros((n_days, n_assets))
    for t in range(1, n_days):
        if kind == "signal":
            lag5 = rets[max(0, t - 5):t].mean(axis=0)
            rets[t] = strength * (0.04 * rets[t - 1] - 0.10 * lag5) + noise[t]
        else:
            rets[t] = noise[t]
    close = 100.0 * np.exp(np.cumsum(rets, axis=0))
    volume = np.exp(rng.normal(15.0, 0.4, size=(n_days, n_assets)))
    return close, volume


def _read_csv_series(path: Path) -> Tuple[List[str], FloatArray, FloatArray]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    cols = {name.lower().strip(): name for name in rows[0].keys()}
    date_key = cols.get("date") or cols.get("datetime")
    close_key = cols.get("adj close") or cols.get("adj_close") or cols.get("close")
    vol_key = cols.get("volume")
    if date_key is None or close_key is None:
        raise ValueError(f"{path} needs Date and Close columns")
    dates = [r[date_key][:10] for r in rows]
    close = np.array([float(r[close_key]) for r in rows])
    volume = np.array([float(r[vol_key]) if vol_key and r[vol_key] else 1.0 for r in rows])
    return dates, close, volume


def load_csv_panel(directory: Path) -> Tuple[List[str], FloatArray, FloatArray]:
    series = {p.stem: _read_csv_series(p) for p in sorted(directory.glob("*.csv"))}
    if len(series) < 5:
        raise ValueError("need at least 5 CSV files for a cross-sectional study")
    common = sorted(set.intersection(*[set(s[0]) for s in series.values()]))
    index = {name: {d: i for i, d in enumerate(s[0])} for name, s in series.items()}
    close = np.column_stack([[series[n][1][index[n][d]] for d in common] for n in series])
    volume = np.column_stack([[series[n][2][index[n][d]] for d in common] for n in series])
    return list(series.keys()), close, volume


def load_yfinance_panel(tickers: Sequence[str], start: str, end: str) -> Tuple[List[str], FloatArray, FloatArray]:
    import yfinance as yf  # imported lazily: only needed for live downloads

    closes, vols, names = [], [], []
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df is None or len(df) == 0:
            print(f"  warning: no data for {ticker}, skipping")
            continue
        c, v = df["Close"], df["Volume"]
        if hasattr(c, "columns"):
            c, v = c.iloc[:, 0], v.iloc[:, 0]
        closes.append(c.rename(ticker))
        vols.append(v.rename(ticker))
        names.append(ticker)
    if len(names) < 5:
        raise RuntimeError("fewer than 5 tickers downloaded; check the network or use --csv-dir")
    import pandas as pd  # pandas is a dependency of yfinance

    close_df = pd.concat(closes, axis=1).dropna()
    vol_df = pd.concat(vols, axis=1).loc[close_df.index].fillna(1.0)
    return names, close_df.to_numpy(dtype=np.float64), vol_df.to_numpy(dtype=np.float64)


# --------------------------------------------------------------------------- #
# Features and targets (causal)
# --------------------------------------------------------------------------- #
def build_features(close: FloatArray, volume: FloatArray) -> Tuple[FloatArray, FloatArray]:
    """Return features (T, N, K) and next-day log-return targets (T, N).

    Row t uses closes/volumes up to and including day t. The target for row t is
    log(close[t+1] / close[t]) and is NaN on the last row.
    """
    t_len, n_assets = close.shape
    logc = np.log(close)
    ret1 = np.full((t_len, n_assets), np.nan)
    ret1[1:] = logc[1:] - logc[:-1]
    feats = np.full((t_len, n_assets, len(FEATURE_NAMES)), np.nan)
    for j, lag in enumerate(LAGS):
        feats[lag:, :, j] = logc[lag:] - logc[:-lag]
    for j, win in enumerate((5, 21)):
        for t in range(win, t_len):
            feats[t, :, len(LAGS) + j] = ret1[t - win + 1:t + 1].std(axis=0)
    logv = np.log1p(volume)
    for t in range(20, t_len):
        window = logv[t - 20:t + 1]
        feats[t, :, len(LAGS) + 2] = (logv[t] - window.mean(axis=0)) / (window.std(axis=0) + 1e-9)
    target = np.full((t_len, n_assets), np.nan)
    target[:-1] = logc[1:] - logc[:-1]
    return feats, target


# --------------------------------------------------------------------------- #
# Fitters
# --------------------------------------------------------------------------- #
def _grad(x: FloatArray, y: FloatArray, w: FloatArray, rows: npt.NDArray[np.int64]) -> FloatArray:
    xb = x[rows]
    return xb.T @ (xb @ w - y[rows]) / float(len(rows))


def fit_ols(x: FloatArray, y: FloatArray) -> FloatArray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def fit_ridge(x: FloatArray, y: FloatArray) -> FloatArray:
    n = len(y)
    split = int(0.8 * n)
    best_lam, best_err = 1e-4, math.inf
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        w = np.linalg.solve(x[:split].T @ x[:split] / split + lam * np.eye(x.shape[1]), x[:split].T @ y[:split] / split)
        err = float(np.mean((x[split:] @ w - y[split:]) ** 2))
        if err < best_err:
            best_lam, best_err = lam, err
    return np.linalg.solve(x.T @ x / n + best_lam * np.eye(x.shape[1]), x.T @ y / n)


def fit_iterative(
    kind: str, x: FloatArray, y: FloatArray, w0: FloatArray, budget_evals: int, batch: int,
    interval: int, seed: int,
) -> FloatArray:
    """Budget-limited warm-started Adam (``adam``) or Jacobi-preconditioned SVRG (``svrg``)."""
    n, d = x.shape
    sampler = DeterministicBatchSampler(n, min(batch, n), seed)
    if kind == "adam":
        ref = ReferenceSVRG(ReferenceConfig(lr=0.01, variance_reduction=False, adaptive=True), d)
    else:
        ref = ReferenceSVRG(
            ReferenceConfig(lr=0.1, beta1=0.0, snapshot_interval=interval, variance_reduction=True, adaptive=False), d
        )
        ref.set_preconditioner(jacobi_diag_least_squares(x))
    w = w0.copy()
    evals = 0
    step = 0
    while evals < budget_evals:
        if kind == "svrg" and ref.needs_snapshot:
            ref.refresh_snapshot(w, x.T @ (x @ w - y) / float(n))
            evals += n
        rows = sampler.batch(step)
        g_live = _grad(x, y, w, rows)
        if kind == "svrg":
            w = ref.step(w, g_live, _grad(x, y, ref.snapshot, rows))  # type: ignore[arg-type]
            evals += 2 * len(rows)
        else:
            w = ref.step(w, g_live)
            evals += len(rows)
        step += 1
    return w


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
def walk_forward_predictions(
    feats: FloatArray, target: FloatArray, window: int, horizon: int, budget_epochs: float,
    batch: int, seed: int,
) -> Dict[str, FloatArray]:
    """Predictions (T, N) for every method; NaN where no prediction was made."""
    t_len, n_assets, k = feats.shape
    preds = {m: np.full((t_len, n_assets), np.nan) for m in METHODS}
    start = max(LAGS[-1], 21) + 1 + window                    # first refit day with a full window
    warm: Dict[str, FloatArray] = {"adam": np.zeros(k), "svrg": np.zeros(k)}
    refit = start
    while refit < t_len - 1:
        tr = slice(refit - window, refit)                     # rows t <= refit-1
        te = slice(refit, min(refit + horizon, t_len - 1))    # rows with a known next-day target
        ftr = feats[tr].reshape(-1, k)
        ytr = target[tr].reshape(-1)
        keep = np.isfinite(ftr).all(axis=1) & np.isfinite(ytr)
        ftr, ytr = ftr[keep], ytr[keep]
        if len(ytr) < 200:
            refit += horizon
            continue
        mu, sd = ftr.mean(axis=0), ftr.std(axis=0) + 1e-12
        x = np.clip((ftr - mu) / sd, -5.0, 5.0)
        y_mean, y_sd = float(ytr.mean()), float(ytr.std() + 1e-12)
        y = np.clip(ytr - y_mean, -5.0 * y_sd, 5.0 * y_sd) / y_sd     # centred, clipped, unit scale
        budget = int(budget_epochs * len(y))
        interval = max(8, int(2 * len(y) / batch))
        weights = {
            "ols": fit_ols(x, y),
            "ridge": fit_ridge(x, y),
        }
        warm["adam"] = fit_iterative("adam", x, y, warm["adam"], budget, batch, interval, seed + refit)
        warm["svrg"] = fit_iterative("svrg", x, y, warm["svrg"], budget, batch, interval, seed + refit)
        weights["adam_budget"] = warm["adam"]
        weights["svrg_jacobi_budget"] = warm["svrg"]
        fte = feats[te]
        valid = np.isfinite(fte).all(axis=2)
        xte = np.clip((np.nan_to_num(fte) - mu) / sd, -5.0, 5.0)
        for name, w in weights.items():
            out = xte @ w
            out[~valid] = np.nan
            preds[name][te] = out
        mom = fte[:, :, LAGS.index(21)].copy()
        mom[~valid] = np.nan
        preds["mom21"][te] = mom
        refit += horizon
    return preds


# --------------------------------------------------------------------------- #
# Metrics and bootstrap
# --------------------------------------------------------------------------- #
def _rank(a: FloatArray) -> FloatArray:
    return np.argsort(np.argsort(a)).astype(np.float64)


def daily_series(pred: FloatArray, target: FloatArray, cost_bps: float) -> Dict[str, FloatArray]:
    """Daily cross-sectional IC, dollar-neutral gross and net P&L and turnover (NaN days dropped later)."""
    t_len, n_assets = pred.shape
    ic = np.full(t_len, np.nan)
    pnl = np.full(t_len, np.nan)
    turnover = np.full(t_len, np.nan)
    prev_w = np.zeros(n_assets)
    for t in range(t_len):
        p, y = pred[t], target[t]
        ok = np.isfinite(p) & np.isfinite(y)
        if ok.sum() < 5:
            prev_w = np.zeros(n_assets)
            continue
        pc = p[ok] - p[ok].mean()
        if np.std(pc) > 0 and np.std(y[ok]) > 0:
            ic[t] = float(np.corrcoef(_rank(pc), _rank(y[ok]))[0, 1])
        gross = np.abs(pc).sum()
        w = np.zeros(n_assets)
        if gross > 0:
            w[ok] = pc / gross
        pnl[t] = float(w @ np.where(ok, y, 0.0))
        turnover[t] = float(np.abs(w - prev_w).sum())
        prev_w = w
    net = pnl - turnover * cost_bps / 1e4
    return {"ic": ic, "pnl_gross": pnl, "pnl_net": net, "turnover": turnover}


def sharpe(x: FloatArray) -> float:
    s = float(np.std(x))
    return float(np.mean(x) / s * math.sqrt(252.0)) if s > 0 else 0.0


def block_indices(n: int, block: int, rng: np.random.Generator) -> npt.NDArray[np.int64]:
    starts = rng.integers(0, max(n - block + 1, 1), size=int(math.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n] % n


def paired_block_bootstrap(
    series: Dict[str, FloatArray], a: str, b: str, stat: Callable[[FloatArray], float],
    block: int = 21, draws: int = 2000, seed: int = 0,
) -> Tuple[float, float, float]:
    """Point estimate and 95% interval of stat(a) - stat(b) with a moving-block bootstrap over days."""
    xa, xb = series[a], series[b]
    ok = np.isfinite(xa) & np.isfinite(xb)
    xa, xb = xa[ok], xb[ok]
    rng = np.random.default_rng(seed)
    diffs = np.empty(draws)
    for i in range(draws):
        idx = block_indices(len(xa), block, rng)
        diffs[i] = stat(xa[idx]) - stat(xb[idx])
    return stat(xa) - stat(xb), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def summarise(
    preds: Dict[str, FloatArray], target: FloatArray, cost_bps: float, seed: int
) -> Dict[str, object]:
    series = {m: daily_series(preds[m], target, cost_bps) for m in METHODS}
    table: Dict[str, Dict[str, float]] = {}
    for m, s in series.items():
        ok_ic = s["ic"][np.isfinite(s["ic"])]
        ok_p = s["pnl_gross"][np.isfinite(s["pnl_gross"])]
        ok_n = s["pnl_net"][np.isfinite(s["pnl_net"])]
        flat_p = preds[m][np.isfinite(preds[m]) & np.isfinite(target)]
        flat_y = target[np.isfinite(preds[m]) & np.isfinite(target)]
        table[m] = {
            "days": float(len(ok_ic)),
            "mean_daily_ic": float(ok_ic.mean()) if len(ok_ic) else float("nan"),
            "pooled_ic": float(np.corrcoef(flat_p, flat_y)[0, 1]) if len(flat_p) > 2 else float("nan"),
            "sharpe_gross": sharpe(ok_p) if len(ok_p) else float("nan"),
            "sharpe_net": sharpe(ok_n) if len(ok_n) else float("nan"),
            "mean_turnover": float(np.nanmean(s["turnover"])),
        }
    paired: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    ic_series = {m: series[m]["ic"] for m in METHODS}
    net_series = {m: series[m]["pnl_net"] for m in METHODS}
    for m in ("ridge", "adam_budget", "svrg_jacobi_budget", "mom21"):
        paired[m] = {
            "ic_vs_ols": paired_block_bootstrap(ic_series, m, "ols", lambda v: float(np.mean(v)), seed=seed),
            "sharpe_net_vs_ols": paired_block_bootstrap(net_series, m, "ols", sharpe, seed=seed),
        }
    paired["svrg_jacobi_budget_vs_adam"] = {
        "ic": paired_block_bootstrap(ic_series, "svrg_jacobi_budget", "adam_budget", lambda v: float(np.mean(v)), seed=seed),
    }
    return {"table": table, "paired": paired, "series": {m: {k: v.tolist() for k, v in s.items()} for m, s in series.items()}}


def placebo_target(target: FloatArray, seed: int, min_shift: int = 63) -> FloatArray:
    """Circularly shift each asset's target series by a random lag: keeps its marginal law and
    autocorrelation, destroys its alignment with the features (a null for 'is there signal?')."""
    rng = np.random.default_rng(seed)
    t_len, n_assets = target.shape
    out = np.empty_like(target)
    for j in range(n_assets):
        out[:, j] = np.roll(target[:, j], int(rng.integers(min_shift, t_len - min_shift)))
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def markdown_report(title: str, summary: Dict[str, object]) -> List[str]:
    table = summary["table"]  # type: ignore[index]
    paired = summary["paired"]  # type: ignore[index]
    lines = [f"### {title}", "", "| Method | days | mean daily rank IC | pooled IC | Sharpe (gross) | Sharpe (net of costs) | mean turnover |",
             "|---|---|---|---|---|---|---|"]
    for m in METHODS:
        r = table[m]
        lines.append(f"| {LABELS[m]} | {r['days']:.0f} | {r['mean_daily_ic']:+.4f} | {r['pooled_ic']:+.4f} | "
                     f"{r['sharpe_gross']:+.2f} | {r['sharpe_net']:+.2f} | {r['mean_turnover']:.2f} |")
    lines += ["", "Paired differences against OLS (moving-block bootstrap over days, block 21, 95% interval):", "",
              "| Method | mean daily IC difference | net Sharpe difference |", "|---|---|---|"]
    for m in ("ridge", "adam_budget", "svrg_jacobi_budget", "mom21"):
        a, b = paired[m]["ic_vs_ols"], paired[m]["sharpe_net_vs_ols"]
        lines.append(f"| {LABELS[m]} | {a[0]:+.4f} [{a[1]:+.4f}, {a[2]:+.4f}] | {b[0]:+.2f} [{b[1]:+.2f}, {b[2]:+.2f}] |")
    c = paired["svrg_jacobi_budget_vs_adam"]["ic"]
    lines += ["", f"Jacobi SVRG minus Adam, mean daily IC: {c[0]:+.4f} [{c[1]:+.4f}, {c[2]:+.4f}].", ""]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "XLF", "XLE", "XLK"])
    parser.add_argument("--start", default="2008-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--csv-dir", default=None)
    parser.add_argument("--synthetic", choices=["signal", "null"], default=None)
    parser.add_argument("--window", type=int, default=756)
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--budget-epochs", type=float, default=20.0)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--placebo", action="store_true", help="also run on circularly shifted targets")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "real_market"))
    args = parser.parse_args()

    if args.synthetic:
        close, volume = synthetic_panel(args.synthetic, 10, 3000, args.seed)
        names = [f"SYN{i}" for i in range(close.shape[1])]
        source = f"synthetic ({args.synthetic})"
    elif args.csv_dir:
        names, close, volume = load_csv_panel(Path(args.csv_dir))
        source = f"CSV files in {args.csv_dir}"
    else:
        names, close, volume = load_yfinance_panel(args.tickers, args.start, args.end)
        source = f"Yahoo Finance via yfinance, {args.start} to {args.end}"
    print(f"{source}: {close.shape[0]} days x {close.shape[1]} assets ({', '.join(names)})")

    feats, target = build_features(close, volume)
    preds = walk_forward_predictions(feats, target, args.window, args.horizon, args.budget_epochs, args.batch, args.seed)
    summary = summarise(preds, target, args.cost_bps, args.seed)
    lines = [f"# Walk-forward study: {source}", "",
             f"{close.shape[0]} days, {close.shape[1]} assets, window {args.window} days, refit every {args.horizon} days, "
             f"{args.budget_epochs:g} epochs of gradient budget per refit for the iterative methods, cost {args.cost_bps:g} bps per unit traded.", ""]
    lines += markdown_report("Real run" if not args.synthetic else "Synthetic run", summary)
    if args.placebo:
        placebo_summary = summarise(preds, placebo_target(target, args.seed + 1), args.cost_bps, args.seed)
        lines += markdown_report("Placebo (circularly shifted targets: no signal by construction)", placebo_summary)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({k: v for k, v in summary.items() if k != "series"}, indent=1), encoding="utf-8")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        for m in METHODS:
            pnl = np.array(summary["series"][m]["pnl_net"], dtype=float)  # type: ignore[index]
            ax.plot(np.nancumsum(np.nan_to_num(pnl)), label=LABELS[m])
        ax.set_xlabel("trading day"); ax.set_ylabel("cumulative net P&L (sum of daily returns)")
        ax.set_title("Dollar-neutral long-short, net of costs"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / "cumulative_pnl.png", dpi=150); plt.close(fig)
    except ImportError:
        pass
    print("\n".join(lines))
    print(f"\nwrote {out}/report.md, summary.json, cumulative_pnl.png")


if __name__ == "__main__":
    main()
