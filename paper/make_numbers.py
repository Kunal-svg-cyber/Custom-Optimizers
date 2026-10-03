"""Generate ``numbers.tex`` and ``generated_tables.tex`` from ``results/experiments.json``.

Every number quoted in ``technical_report.tex`` is a macro defined here, so the report
cannot drift from the stored results. Run from anywhere:

    python paper/make_numbers.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_experiments import VARIANT_LABELS, paired_bootstrap  # noqa: E402

BASELINES: List[str] = ["sgd_momentum", "sgd_momentum_cosine", "adam", "adam_cosine"]


def sci(x: float) -> str:
    """LaTeX scientific notation."""
    if x == 0.0:
        return "0"
    if math.isinf(x):
        return r"\infty"
    mantissa, exponent = f"{x:.1e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def signed(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}".replace("-", "{-}")


def log_diff(variants: Dict[str, Any], a: str, b: str) -> Dict[str, float]:
    la = np.log10(np.maximum(np.asarray(variants[a]["final_gaps"], dtype=float), 1e-40))
    lb = np.log10(np.maximum(np.asarray(variants[b]["final_gaps"], dtype=float), 1e-40))
    diff = la - lb
    med, lo, hi = paired_bootstrap(diff)
    return {"med": med, "lo": lo, "hi": hi, "wins": int(np.sum(diff < 0)), "n": len(diff)}


def best_baseline(variants: Dict[str, Any]) -> str:
    return min(BASELINES, key=lambda b: variants[b]["final_gap_median"])


def main() -> None:
    r: Dict[str, Any] = json.loads((REPO_ROOT / "results" / "experiments.json").read_text(encoding="utf-8"))
    macros: Dict[str, str] = {}

    # ---- Experiment 1 (least squares, volatile) ----
    v1 = r["ablation"]["regimes"]["volatile"]["variants"]
    bb = best_baseline(v1)
    d = log_diff(v1, "svrg_adam", bb)
    macros["ExpOneSeeds"] = str(d["n"])
    macros["ExpOneWins"] = f"{d['wins']}/{d['n']}"
    macros["ExpOneBestBaseline"] = VARIANT_LABELS[bb]
    macros["ExpOneBaselineGap"] = sci(v1[bb]["final_gap_median"])
    macros["ExpOneSvrgGap"] = sci(v1["svrg_adam"]["final_gap_median"])
    lo_b = min(r["ablation"]["regimes"][g]["variants"][best_baseline(r["ablation"]["regimes"][g]["variants"])]["final_gap_median"]
               for g in r["ablation"]["regimes"])
    hi_b = max(r["ablation"]["regimes"][g]["variants"][best_baseline(r["ablation"]["regimes"][g]["variants"])]["final_gap_median"]
               for g in r["ablation"]["regimes"])
    macros["ExpOneBaselineRange"] = rf"${sci(lo_b)}$ to ${sci(hi_b)}$"

    # ---- Experiment 4 (logistic, volatile) ----
    v4 = r["logistic"]["regimes"]["volatile"]["variants"]
    d4 = log_diff(v4, "svrg_adam", best_baseline(v4))
    macros["ExpFourWins"] = f"{d4['wins']}/{d4['n']}"

    # ---- Experiment 6 (bad scaling) ----
    v6 = r["hetero"]["regimes"]["volatile"]["variants"]
    d6a = log_diff(v6, "svrg_adam", "svrg_momentum")
    d6b = log_diff(v6, "svrg_adam_cosine", "adam_cosine")
    macros["ExpSixAdaptiveVsMomentum"] = signed(d6a["med"], 1)
    macros["ExpSixAdaptiveCI"] = f"[{signed(d6a['lo'], 1)}, {signed(d6a['hi'], 1)}]"
    macros["ExpSixAdaptiveWins"] = f"{d6a['wins']}/{d6a['n']}"
    macros["ExpSixCosineVsAdamCosine"] = signed(d6b["med"], 1)
    macros["ExpSixCosineCI"] = f"[{signed(d6b['lo'], 1)}, {signed(d6b['hi'], 1)}]"
    macros["ExpSixCosineWins"] = f"{d6b['wins']}/{d6b['n']}"
    macros["ExpSixConstGap"] = sci(v6["svrg_adam"]["final_gap_median"])
    macros["ExpSixCosineGap"] = sci(v6["svrg_adam_cosine"]["final_gap_median"])
    macros["ExpSixAdamCosineGap"] = sci(v6["adam_cosine"]["final_gap_median"])

    # ---- Experiment 5 (larger problem, wall clock) ----
    b5 = r["highdim"]["regimes"]["volatile"]
    macros["ExpFiveN"] = str(b5["n"])
    macros["ExpFiveD"] = str(b5["d"])
    macros["ExpFiveSvrgSeconds"] = f"{b5['variants']['svrg_momentum']['seconds_to_target_median']:.2f}"
    macros["ExpFiveNormalSeconds"] = f"{b5['normal_equations_seconds']:.2f}"
    macros["ExpFiveLstsqSeconds"] = f"{b5['direct_solve_seconds']:.2f}"

    # ---- Experiment 7 and 8 (non-convex) ----
    v7 = r["nonconvex"]["regimes"]["volatile"]["variants"]
    bb7 = best_baseline(v7)
    d7 = log_diff(v7, "svrg_adam_cosine", bb7)
    macros["ExpSevenDecades"] = signed(d7["med"], 1)
    macros["ExpSevenWins"] = f"{d7['wins']}/{d7['n']}"
    h = r["heldout"]["nonconvex"]
    test_best = min(BASELINES, key=lambda b: float(np.median(h[b]["test_loss"])))
    diff8 = np.asarray(h["svrg_adam_cosine"]["test_loss"]) - np.asarray(h[test_best]["test_loss"])
    med8, lo8, hi8 = paired_bootstrap(diff8)
    macros["ExpEightGap"] = signed(med8, 4)
    macros["ExpEightCI"] = f"[{signed(lo8, 4)}, {signed(hi8, 4)}]"
    macros["ExpEightSvrgTest"] = f"{float(np.median(h['svrg_adam_cosine']['test_loss'])):.4f}"
    macros["ExpEightBaselineTest"] = f"{float(np.median(h[test_best]['test_loss'])):.4f}"
    macros["ExpEightSvrgTrain"] = f"{float(np.median(h['svrg_adam_cosine']['train_loss'])):.4f}"
    macros["ExpEightBaselineTrain"] = f"{float(np.median(h[test_best]['train_loss'])):.4f}"
    macros["ExpEightBaselineName"] = VARIANT_LABELS[test_best]
    lg = r["heldout"]["logistic"]
    macros["ExpEightLogisticLow"] = f"{min(float(np.median(lg[v]['test_logloss'])) for v in lg):.5f}"
    macros["ExpEightLogisticHigh"] = f"{max(float(np.median(lg[v]['test_logloss'])) for v in lg):.5f}"
    macros["ExpEightAccuracy"] = f"{100 * float(np.median(lg['svrg_adam']['test_acc'])):.0f}"

    # ---- Experiment 2 (variance) ----
    rows = r["variance"]["rows"]
    macros["ExpTwoVrStart"] = sci(rows[0]["var_vr"])
    macros["ExpTwoVrEnd"] = sci(rows[-1]["var_vr"])
    macros["ExpTwoSgdMin"] = f"{min(x['var_sgd'] for x in rows):.1f}"
    macros["ExpTwoSgdMax"] = f"{max(x['var_sgd'] for x in rows):.1f}"
    macros["ExpTwoProbes"] = str(len(rows))
    macros["ExpTwoWithinLemma"] = f"{100 * r['variance']['fraction_within_lemma2']:.0f}"
    macros["ExpTwoWithinProp"] = f"{100 * r['variance']['fraction_within_prop3']:.0f}"

    # ---- Experiment 9 (snapshot interval) ----
    si = r["snapshot_interval"]["rows"]
    finite = [x for x in si if math.isfinite(x["evals_to_target_median"])]
    best = min(finite, key=lambda x: x["evals_to_target_median"])
    macros["ExpNineBestK"] = str(best["interval"])
    macros["ExpNineBestEvals"] = f"{best['evals_to_target_median']:,.0f}".replace(",", "{,}")
    worst_ratio = max(x["prop3_ratio_max"] for x in si)
    macros["ExpNineMaxRatio"] = sci(worst_ratio)
    macros["ExpNineKmin"] = str(min(x["interval"] for x in si))
    macros["ExpNineKmax"] = str(max(x["interval"] for x in si))
    within2 = [x["interval"] for x in finite if x["evals_to_target_median"] <= 2.0 * best["evals_to_target_median"]]
    macros["ExpNineWithinTwoLow"] = str(min(within2))
    macros["ExpNineWithinTwoHigh"] = str(max(within2))

    mid = [x["evals_to_target_median"] / best["evals_to_target_median"] for x in finite if 16 <= x["interval"] <= 256]
    macros["ExpNineMidFactor"] = f"{max(mid):.1f}"
    by_k = {x["interval"]: x["evals_to_target_median"] / best["evals_to_target_median"] for x in finite}
    macros["ExpNineSmallFactor"] = f"{by_k[min(by_k)]:.1f}"
    macros["ExpNineLargeFactor"] = f"{by_k[max(by_k)]:.1f}"

    # ---- Experiment 10 (conditioning) ----
    cond = r["conditioning"]["problems"]
    bad = [v for k, v in cond.items() if k.startswith("badly")][0]
    good = [v for k, v in cond.items() if k.startswith("well")][0]
    macros["ExpTenBadRaw"] = sci(bad["kappa"])
    macros["ExpTenBadJacobi"] = f"{bad['kappa_jacobi']:.0f}"
    macros["ExpTenGoodRaw"] = f"{good['kappa']:.0f}"
    macros["ExpTenGoodJacobi"] = f"{good['kappa_jacobi']:.0f}"

    # ---- Experiment 11 (Jacobi SVRG, theory-prescribed) ----
    jp = r["jacobi"]["problems"]
    jbad = [v for k, v in jp.items() if k.startswith("badly")][0]
    jgood = [v for k, v in jp.items() if k.startswith("well")][0]
    macros["ExpElevenBadTheoryGap"] = sci(jbad["theory"]["final_gap_median"])
    macros["ExpElevenBadTunedGap"] = sci(jbad["tuned"]["final_gap_median"])
    macros["ExpElevenGoodTheoryGap"] = sci(jgood["theory"]["final_gap_median"])
    macros["ExpElevenAlpha"] = f"{jbad['theory']['alpha_predicted']:.2f}"
    macros["ExpElevenContractionBad"] = f"{jbad['theory']['mean_epoch_contraction']:.2f}"
    macros["ExpElevenContractionGood"] = f"{jgood['theory']['mean_epoch_contraction']:.2f}"
    macros["ExpElevenInner"] = f"{jbad['theory']['inner_steps']:,}".replace(",", "{,}")
    coord = r["hetero"]["regimes"]["volatile"]["variants"]["svrg_adam_cosine"]
    macros["ExpElevenEvalsTheory"] = f"{jbad['theory']['evals_to_target_median']:,.0f}".replace(",", "{,}")
    macros["ExpElevenEvalsTuned"] = f"{jbad['tuned']['evals_to_target_median']:,.0f}".replace(",", "{,}")
    macros["ExpElevenEvalsCoord"] = f"{coord['evals_to_target_median']:,.0f}".replace(",", "{,}")
    macros["ExpElevenSpeedup"] = f"{coord['evals_to_target_median'] / jbad['theory']['evals_to_target_median']:.1f}"
    mine = np.log10(np.maximum(np.asarray(jbad["theory"]["final_gaps"], dtype=float), 1e-40))
    theirs = np.log10(np.maximum(np.asarray(coord["final_gaps"], dtype=float), 1e-40))
    macros["ExpElevenBadWins"] = f"{int(np.sum(mine < theirs))}/{len(mine)}"

    # ---- Experiment 3 (walk-forward) ----
    wf = r["walk_forward"]["by_budget"]
    diffs: List[float] = []
    for summ in wf.values():
        ols = np.asarray(summ["ols"]["ic_clean"]["values"])
        for m in ("adam", "svrg_adam"):
            diffs.append(float(np.median(np.asarray(summ[m]["ic_clean"]["values"]) - ols)))
    macros["ExpThreeMaxICDiff"] = f"{max(abs(x) for x in diffs):.4f}"
    spread = [float(np.percentile(s["ols"]["ic_clean"]["values"], 75) - np.percentile(s["ols"]["ic_clean"]["values"], 25))
              for s in wf.values()]
    macros["ExpThreeIQR"] = f"{float(np.median(spread)):.2f}"

    lines = [r"\newcommand{\%s}{%s}" % (k, v) for k, v in macros.items()]
    (REPO_ROOT / "paper" / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- tables ----
    def table(key: str, regime: str, caption: str, label: str) -> str:
        block = r[key]["regimes"][regime]["variants"]
        out = [r"\begin{table}[h]", r"\centering", r"\small",
               r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
               r"Method & tuned lr & median final gap & evals to target \\", r"\midrule"]
        for name, x in block.items():
            ev = x["evals_to_target_median"]
            ev_txt = "not reached" if math.isinf(ev) else f"{ev:,.0f}".replace(",", r"\,")
            dagger = r"$^\dagger$" if x.get("best_lr_at_grid_edge") else ""
            out.append(rf"{VARIANT_LABELS[name]} & ${sci(x['best_lr'])}${dagger} & ${sci(x['final_gap_median'])}$ & {ev_txt} \\")
        out += [r"\bottomrule", r"\end{tabular}",
                rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
        return "\n".join(out)

    jt = [r"\begin{table}[h]", r"\centering", r"\small", r"\begin{tabular}{@{}llrr@{}}", r"\toprule",
          r"Problem & Method & median final gap & evals to target \\", r"\midrule"]
    for pname, pr in r["jacobi"]["problems"].items():
        short = "well-scaled" if pname.startswith("well") else "badly scaled"
        for label, key in (("Jacobi SVRG, theory-prescribed", "theory"), ("Jacobi SVRG, tuned lr", "tuned")):
            ev = pr[key]["evals_to_target_median"]
            ev_txt = "not reached" if math.isinf(ev) else f"{ev:,.0f}".replace(",", r"\,")
            jt.append(rf"{short} & {label} & ${sci(pr[key]['final_gap_median'])}$ & {ev_txt} \\")
    jt += [r"\bottomrule", r"\end{tabular}",
           r"\caption{Jacobi-preconditioned SVRG on the datasets and seeds of Tables~\ref{tab:ls} and~\ref{tab:hetero}. The theory-prescribed row uses no tuning at all.}",
           r"\label{tab:jacobi}", r"\end{table}"]
    tables = [
        "\n".join(jt),
        table("ablation", "volatile",
              r"Least squares, volatile regime. Median final loss gap and sample-gradient evaluations to reach $10^{-8}$ of the initial gap, held-out seeds. $\dagger$: best learning rate on the edge of the grid.",
              "tab:ls"),
        table("hetero", "volatile",
              r"Least squares with feature scales spread over three decades. Same protocol as Table~\ref{tab:ls}.",
              "tab:hetero"),
    ]
    (REPO_ROOT / "paper" / "generated_tables.tex").write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    print(f"wrote {len(macros)} macros and {len(tables)} tables")


if __name__ == "__main__":
    main()
