"""Phase 0.5 - power analysis. Sizes the eval split BEFORE the freeze (D-008).

CLAUDE.md rule 7 makes the frozen split one-shot: written once, hashed, never
edited. So the question "how many items do we need?" has to be answered before
any data is written, not after. Discovering underpowered CIs afterwards would
cost the whole dataset.

Generative model
----------------
A latent common-cause mixture, which is the natural data-generating process for
the hypothesis under test:

    with probability lam : ALL layers fail together   (the shared base fires)
    otherwise            : each layer fails independently with rate q

Choosing q = (p - lam) / (1 - lam) keeps the marginal failure rate at exactly p
for any lam in [0, p], so lam moves the COUPLING while holding the MARGINALS
fixed. That is precisely the manipulation the paper claims per-layer monitoring
cannot see, so it is the right thing to simulate.

Under this model the analytic values are

    P_all = lam + (1 - lam) * q**n
    beta  = (P_all - p**n) / (p - p**n)        [equal marginals, so both forms agree]

What this script reports
------------------------
For each (n_items, p, n_layers) cell: the standard error of beta-hat estimated
from repeated simulation, the implied minimum detectable delta-beta at 80% and
95% power, and whether the cell is estimable at all. The headline output is the
smallest n_items that can distinguish a given delta-beta from zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
from provenance import (  # noqa: E402
    canonical_hash,
    ensure_utf8_stdout,
    provenance_block,
    write_results,
)

Z_95 = 1.959963984540054  # two-sided alpha = 0.05
Z_80_POWER = 0.8416212335729143  # one-sided z for 80% power
Z_95_POWER = 1.6448536269514722  # one-sided z for 95% power


def q_for(p: float, lam: float) -> float:
    """Independent-component rate that keeps the marginal at p."""
    if lam >= p:
        raise ValueError(f"lam ({lam}) must be < p ({p}) for a valid mixture")
    return (p - lam) / (1.0 - lam)


def analytic_beta(p: float, n_layers: int, lam: float) -> float:
    q = q_for(p, lam)
    p_all = lam + (1.0 - lam) * q**n_layers
    return (p_all - p**n_layers) / (p - p**n_layers)


def lambda_for_beta(p: float, n_layers: int, target_beta: float) -> float:
    """Invert analytic_beta by bisection. beta is monotone increasing in lam."""
    lo, hi = 0.0, p * (1 - 1e-9)
    if target_beta <= 0:
        return 0.0
    if analytic_beta(p, n_layers, hi) < target_beta:
        return hi
    for _ in range(200):
        mid = (lo + hi) / 2
        if analytic_beta(p, n_layers, mid) < target_beta:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def simulate(n_items: int, p: float, n_layers: int, lam: float, rng) -> np.ndarray:
    """Draw one error matrix from the common-cause mixture."""
    q = q_for(p, lam)
    common = rng.random(n_items) < lam
    indep = rng.random((n_items, n_layers)) < q
    return np.where(common[:, None], True, indep)


def se_of_beta(
    n_items: int, p: float, n_layers: int, lam: float, n_rep: int, seed: int
) -> dict[str, Any]:
    """Standard error of beta-hat, from repeated simulation of the whole experiment."""
    rng = np.random.default_rng(seed)
    vals, n_undef = [], 0
    for _ in range(n_rep):
        e = simulate(n_items, p, n_layers, lam, rng)
        b = M.beta(e)
        if b.ok:
            vals.append(b.value)
        else:
            n_undef += 1
    if len(vals) < 2:
        return {
            "se": None,
            "reason": "insufficient_defined_replicates",
            "n_undefined": n_undef,
            "n_rep": n_rep,
        }
    arr = np.asarray(vals)
    truth = analytic_beta(p, n_layers, lam)
    return {
        "se": float(arr.std(ddof=1)),
        "mean_beta_hat": float(arr.mean()),
        "analytic_beta": float(truth),
        "bias": float(arr.mean() - truth),
        "n_undefined": n_undef,
        "undefined_rate": n_undef / n_rep,
        "n_rep": n_rep,
    }


def joint_event_count(n_items: int, p: float, n_layers: int, lam: float) -> float:
    """Expected number of all-layers-fail items. The real driver of estimability."""
    q = q_for(p, lam)
    return n_items * (lam + (1 - lam) * q**n_layers)


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Phase 0.5 power analysis")
    ap.add_argument("--n-rep", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "results" / "phase0_power_analysis.json"),
    )
    args = ap.parse_args()

    # Grid. p values span the plausible operating range:
    #   harmful axis at 1% FPR -> miss rate roughly 0.2-0.5
    #   over-refusal axis      -> false-refusal rate near the 0.01 calibration point
    n_items_grid = [500, 1000, 2000, 5000, 10000, 20000]
    p_grid = [0.01, 0.05, 0.10, 0.20, 0.40]
    layer_grid = [2, 3]
    baseline_beta = 0.20  # a modest pre-update coupling, per arXiv:2608.28327's phi range

    print("Phase 0.5 power analysis - sizing the eval split before the freeze")
    print(f"replicates per cell: {args.n_rep}   seed: {args.seed}")
    print(f"baseline beta: {baseline_beta}\n")

    cells: list[dict[str, Any]] = []
    for n_layers in layer_grid:
        for p in p_grid:
            lam = lambda_for_beta(p, n_layers, baseline_beta)
            for n_items in n_items_grid:
                r = se_of_beta(n_items, p, n_layers, lam, args.n_rep, args.seed)
                exp_joint = joint_event_count(n_items, p, n_layers, lam)
                cell = {
                    "n_items": n_items,
                    "p": p,
                    "n_layers": n_layers,
                    "lam": lam,
                    "expected_joint_failures": exp_joint,
                    **r,
                }
                if r["se"]:
                    # Two independent arms (pre, post); paired data would do better,
                    # so this is the conservative direction.
                    se_delta = r["se"] * np.sqrt(2)
                    cell["se_delta_beta"] = float(se_delta)
                    cell["mdes_80"] = float((Z_95 + Z_80_POWER) * se_delta)
                    cell["mdes_95"] = float((Z_95 + Z_95_POWER) * se_delta)
                cells.append(cell)

    # --- printed summary ---
    for n_layers in layer_grid:
        print(f"\n{'=' * 78}\nn_layers = {n_layers}\n{'=' * 78}")
        print(f"{'p':>6} {'n_items':>8} {'E[joint]':>9} {'SE(beta)':>10} "
              f"{'MDES@80%':>9} {'undef':>7}")
        print("-" * 78)
        for p in p_grid:
            for n_items in n_items_grid:
                c = next(
                    x for x in cells
                    if x["n_layers"] == n_layers and x["p"] == p and x["n_items"] == n_items
                )
                se = f"{c['se']:.4f}" if c.get("se") else "  n/a"
                mdes = f"{c['mdes_80']:.3f}" if c.get("mdes_80") else "  n/a"
                und = f"{c.get('undefined_rate', 0):.1%}"
                print(f"{p:>6.2f} {n_items:>8} {c['expected_joint_failures']:>9.1f} "
                      f"{se:>10} {mdes:>9} {und:>7}")

    # --- the actionable answer ---
    target_delta = 0.20
    print(f"\n{'=' * 78}")
    print(f"SMALLEST n_items TO DETECT delta-beta = {target_delta} AT 80% POWER")
    print(f"{'=' * 78}")
    recommendations = []
    for n_layers in layer_grid:
        for p in p_grid:
            ok = [
                c for c in cells
                if c["n_layers"] == n_layers and c["p"] == p
                and c.get("mdes_80") is not None and c["mdes_80"] <= target_delta
            ]
            need = min((c["n_items"] for c in ok), default=None)
            recommendations.append(
                {"n_layers": n_layers, "p": p, "target_delta": target_delta, "n_items_required": need}
            )
            label = f"{need}" if need else f"> {max(n_items_grid)} (NOT ACHIEVABLE on this grid)"
            print(f"  n_layers={n_layers}  p={p:<5}  ->  n_items = {label}")

    payload = {
        **provenance_block("phase0_power_analysis"),
        "config": {
            "n_items_grid": n_items_grid,
            "p_grid": p_grid,
            "layer_grid": layer_grid,
            "baseline_beta": baseline_beta,
            "n_rep": args.n_rep,
            "seed": args.seed,
            "target_delta_for_recommendation": target_delta,
        },
        "config_hash": canonical_hash(
            {"n": n_items_grid, "p": p_grid, "L": layer_grid, "b": baseline_beta, "rep": args.n_rep}
        ),
        "generative_model": "latent common-cause mixture; lam moves coupling at fixed marginals",
        "recommendations": recommendations,
        "cells": cells,
    }
    digest = write_results(args.out, payload)
    print(f"\nWritten: {args.out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
