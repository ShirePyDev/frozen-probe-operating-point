"""4b fits under D-156: how well do the moment predictions track the realised paired damage?

Reads the nine per-pair files results/twomoment_<pair>.json written by src/twomoment_cells.py
for the pairs of UPDATE_PAIRS_REPAIR (declared, never globbed), and their per-draw arrays
(results/twomoment_<pair>_draws.npz, sha256 and size checked against the JSON). Refuses unless
all nine ran the registered parameters with identical code, libraries, BLAS threads and
selection file, and each matched results/damage_d156_r2.json at its selected layer at both
levels, with the selection of record and the damage_d156_r2 of D-173. Every fit is computed
twice: on the item-level re-partitions (D-156) and on the cluster-level ones (D-164, the npz
keys ending '__cluster'). The cluster level is a sensitivity reported beside, never instead:
its rows are never the headline, the headline's cluster-level counterpart sits inside the
headline block, and in its within-draw fields the 'frozen' value belongs to the frozen split,
which is not a cluster-level draw. Each row counts its cells carrying the D-170 pin-only flag
(D-172: such a row waits for D-168); the cache bytes behind every pair file are listed so the
provenance audit counts this file too.

For each probe set (selected C, C=1) and each predictor (translation / moment_full /
moment_tail, each in-sample and out-of-sample; roles as recorded by twomoment_cells) it fits

  log(realised paired ratio) = intercept + slope * log(predicted paired ratio)

by OLS with Pearson r, and the same fit on the FPR scale (log realised FPR_post on log
predicted FPR_post), the shared-denominator check: both ratios divide by the same realised
pre-update FPR, and that alone correlates them. Cells: all layer cells, or the deployer's
(selected) layer only (one cell per pair).

HEADLINE: exactly ONE row (the author's decisions, D-163, recorded before any 4b number
existed; D-152 4b), copied with its coordinates into the output's 'headline' block:
  row        selected C, all layer cells, ratio scale (C=1 is the sensitivity row, the
             deployer's layer its own row, the FPR scale a check);
  estimate   the fit on each cell's D-156 estimate (log of the mean over the draws in which
             the realised and the predicted value are both defined);
  interval   bootstrap over the nine pairs as clusters (B = 5000, fixed seed), as D-152 4b
             names it; other values of B or the seed cannot write the final name;
  cells      every cell, censored draws floored at 1/n_neg and kept; each cell's fraction of
             floored draws is reported ('cells'); leaving out cells floored in more than 2.5%
             of their defined draws is a SENSITIVITY only - one cell set per predictor row,
             used in both intervals and every draw;
  predictor  in-sample full-SD moment matching (roles as recorded by twomoment_cells).
PARTITION SENSITIVITY: the same fit WITHIN each draw over every cell's value in that draw
(draw r is the same item split for every cell on an axis; the two axes are partitioned
independently, so draw r is a draw of the joint re-partition), reported as the mean over the
1000 draws, its 2.5-97.5% band, and the frozen split as one draw with its percentile. Within a
draw the predicted and realised ratios share the pre-update FPR and the threshold, so their
errors are correlated and the within-draw slope is pulled toward the shared-denominator slope
(about 1 on the log scale); it is a partition sensitivity only (D-166, correcting D-163 (1)).

Also the Pearson r of the two-moment (moment_full) minus the one-moment (translation)
prediction, in both forms on the same cells: those whose two predictions are defined in every
draw. Rows of different predictors can differ in cells (undefined predictions; the per-predictor
sensitivity cut).

Cluster dependence not modelled: the nine pairs are seven distinct model updates.
WG_mistral_base2instruct is the same update as mistral_base2instruct and
WG_llama3_base2instruct the same as llama3_base2instruct (other dataset), and
llama3_base2instruct and llama3_to_31 share their pre-update model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
from provenance import (ensure_utf8_stdout, env_state, provenance_block, read_result, reissue_payload,  # noqa: E402
                        sha256_bytes, sha256_file, write_results)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/twomoment_fit.py", "src/paired.py", "src/populations.py", "src/provenance.py",
              "src/superseded_provenance.py"]
PRED_KEYS = [f"{p}_{s}" for s in ("in_sample", "out_of_sample") for p in ("translation", "moment_full", "moment_tail")]
REGISTERED = {"n_draws": P.N_DRAWS, "target_fpr": P.TARGET_FPR, "seeds": [0, 1, 2, 3, 4], "tail_quantile": 0.90}
BOOT = {"n_boot": 5000, "boot_seed": 520260925}         # D-163 (1): B = 5000; the seed is fixed here
# D-300: the record --r2 re-issues (it names no repo ID or revision for its cache inputs, D-297 (1))
FIRST_RECORD = ("results/twomoment_fit.json", "515faab949465d0852cd61dbc1c7805cf3d800c5e2222029e4622261727a3b96")
SHARED_MODULES = ("src/paired.py", "src/populations.py", "src/provenance.py", "src/superseded_provenance.py")
SAME_ACROSS_PAIRS = ("predictors", "moment_sources", "split", "selection_sha256", "sd_eps", "selection_file",
                     "cluster_manifest", "input_registry_sources_sha256")
# D-163 (2): no exclusion in the headline; leaving out cells floored in more than 2.5% of the
# draws is a sensitivity only.
SENSITIVITY_FLOORED_SHARE = 0.025
PARTITION_CAVEAT = ("within a draw the predicted and realised ratios share the pre-update FPR and the threshold, "
                    "so their errors are correlated and the within-draw slope is pulled toward the "
                    "shared-denominator slope (about 1 on the log scale); it is a partition sensitivity only "
                    "(D-166, correcting D-163 (1)); applies to every 'partition_sensitivity_within_draw' field")
SELECTION_OF_RECORD = "results/selection_sweep_d165.json"          # D-171
D156_R2_SHA256 = "9c6cd0d6ee2700f60626adca034eae45ace6c18f336fcd2cffc306b59e3919f6"   # D-173


def wols(X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> dict[str, np.ndarray]:
    """OLS of Y on X with intercept, one fit per column. X, Y, W: (cells, K); W holds 0/1
    inclusion or cluster multiplicities (a weight k is the cell repeated k times). Cells with a
    non-finite X or Y are left out. NaN where fewer than 3 DISTINCT cells (a replicate that
    repeats two cells is still a line through two points), or no spread in X or Y (the draft's
    np.var < 1e-15 rule). 'n' is the number of distinct cells used."""
    W = np.where(np.isfinite(X) & np.isfinite(Y), W, 0.0).astype(float)
    X0, Y0 = np.where(W > 0, X, 0.0), np.where(W > 0, Y, 0.0)
    n = W.sum(axis=0)
    n_distinct = (W > 0).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mx, my = (W * X0).sum(axis=0) / n, (W * Y0).sum(axis=0) / n
        dx, dy = np.where(W > 0, X0 - mx, 0.0), np.where(W > 0, Y0 - my, 0.0)
        sxx, syy, sxy = (W * dx * dx).sum(axis=0), (W * dy * dy).sum(axis=0), (W * dx * dy).sum(axis=0)
        slope = sxy / sxx
        out = {"slope": slope, "intercept": my - slope * mx,
               "pearson_r": np.clip(sxy / np.sqrt(sxx * syy), -1.0, 1.0)}
        bad = (n_distinct < 3) | ~(sxx / n >= 1e-15) | ~(syy / n >= 1e-15)
    for k in out:
        out[k] = np.where(bad, np.nan, out[k])
    out["n"] = n_distinct
    return out


def d156_estimates(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per cell, the log of the mean over the draws (columns 1..) of exp(X) and exp(Y), taken on
    the SAME draws: those where both are finite. Returns (x, y, n_common_draws); NaN where a cell
    has no common draw."""
    ok = np.isfinite(X[:, 1:]) & np.isfinite(Y[:, 1:])
    n = ok.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        mx = np.where(ok, np.exp(np.where(ok, X[:, 1:], 0.0)), 0.0).sum(axis=1) / n
        my = np.where(ok, np.exp(np.where(ok, Y[:, 1:], 0.0)), 0.0).sum(axis=1) / n
        return np.where(n > 0, np.log(mx), np.nan), np.where(n > 0, np.log(my), np.nan), n


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="4b under D-156: fit realised vs predicted paired damage")
    ap.add_argument("--n-boot", type=int, default=BOOT["n_boot"])
    ap.add_argument("--boot-seed", type=int, default=BOOT["boot_seed"])
    ap.add_argument("--allow-unregistered", action="store_true",
                    help="test runs only: accept per-pair files with non-registered parameters")
    ap.add_argument("--fpr5", action="store_true",
                    help="4e (D-152, D-248): fit the 5%% per-pair records twomoment_<pair>_fpr5.json into "
                         "twomoment_fit_fpr5.json; each must carry a complete 1%% self-anchor")
    ap.add_argument("--r2", action="store_true",
                    help="re-issue twomoment_fit.json naming each cache input's repo ID and revision, checked equal (D-300)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.r2 and args.fpr5:
        raise SystemExit("REFUSING: --r2 re-issues the 1% record only (D-300)")
    sfx = "_fpr5" if args.fpr5 else ""
    registered_terms = {**REGISTERED, "target_fpr": 0.05} if args.fpr5 else REGISTERED
    final_path = (RESULTS / f"twomoment_fit{sfx}{'_r2' if args.r2 else ''}.json").resolve()
    out_path = Path(args.out).resolve() if args.out else final_path
    if out_path.exists():
        raise SystemExit(f"REFUSING: {out_path} exists")
    final = out_path == final_path
    if out_path.parent == RESULTS.resolve() and out_path.name.startswith("twomoment_fit") and not final:
        raise SystemExit("REFUSING: only this mode's own final name may be written under results/")
    if final and (args.allow_unregistered or args.n_boot != BOOT["n_boot"] or args.boot_seed != BOOT["boot_seed"]):
        raise SystemExit("REFUSING: the final name needs the registered B and seed and no --allow-unregistered")
    if args.r2:   # D-300: the fits equal the first record only in its environment (libraries, BLAS threads); check first
        r1_env, now = read_result(ROOT / FIRST_RECORD[0])["env"], env_state()
        odd = [k for k in ("libraries", "blas_threads") if r1_env.get(k) != now.get(k)]
        if odd:
            raise SystemExit(f"REFUSING: --r2 needs {FIRST_RECORD[0]}'s environment; {odd} differ")
    code0 = P.code_block(CODE_FILES)

    # the damage file each per-pair record was cross-checked against: at 1% the pinned D-173 file; at 5% the 5%
    # damage file, which must itself be complete and self-anchored on that pinned 1% file (D-248)
    dmg_rel_want, dmg_sha_want = "results/damage_d156_r2.json", D156_R2_SHA256
    if args.fpr5:
        dmg_rel_want = "results/damage_d156_r2_fpr5.json"
        db = (ROOT / dmg_rel_want).read_bytes()
        dmg_sha_want = sha256_bytes(db)
        sa = json.loads(db.decode("utf-8")).get("fpr5_self_anchor") or {}
        if sa.get("sha256") != D156_R2_SHA256 or not sa.get("n_cells_identical_at_1pct"):
            raise SystemExit(f"REFUSING: {dmg_rel_want} is not self-anchored on the pinned 1% damage file (D-248)")
    pairs = list(POP.UPDATE_PAIRS_REPAIR)
    files = {s: RESULTS / f"twomoment_{s}{sfx}.json" for s in pairs}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise SystemExit(f"REFUSING: per-pair files missing: {missing}")
    docs = {s: json.load(open(p, encoding="utf-8")) for s, p in files.items()}
    file_sha = {f"results/{p.name}": sha256_file(p) for p in files.values()}

    # Every pair on the same, registered terms; every axis on one partition sequence (D-156).
    ref = docs[pairs[0]]
    axis_ids: dict[str, tuple] = {}
    r2_module_diffs: dict[str, list[str]] = {}
    for s, d in docs.items():
        c = d["config"]
        problems = [f"{k}={c[k]!r}" for k, v in registered_terms.items() if c[k] != v]
        if d.get("run_kind") != "twomoment_cells" or c["pair"] != s:
            problems.append("not this pair's twomoment_cells record")
        if args.fpr5:
            sa = d.get("fpr5_self_anchor") or {}
            if sa.get("n_cells_identical_at_1pct") != len(d["cells"]):
                problems.append(f"incomplete 1% self-anchor (D-248): {sa.get('n_cells_identical_at_1pct')} of {len(d['cells'])}")
        if problems and not args.allow_unregistered:
            raise SystemExit(f"REFUSING: {s}: non-registered or wrong record: {problems}")
        if not args.allow_unregistered and not all(d["damage_d156_check"].get(p, {}).get("identical") and
                                                   d["damage_d156_check"].get(p, {}).get("cluster_level_identical")
                                                   for p in ("selected", "C=1")):
            raise SystemExit(f"REFUSING: {s} did not reproduce its damage file ({d['damage_d156_check'].get('damage_d156_file')}) at its selected layer "
                             f"at both levels")
        diff = [k for k in SAME_ACROSS_PAIRS if c[k] != ref["config"][k]]
        if d["code"]["sha256"] != ref["code"]["sha256"]:
            diff.append("code.sha256")
        if d["damage_d156_check"].get("damage_d156_sha256") != ref["damage_d156_check"].get("damage_d156_sha256"):
            diff.append("damage_d156_check.damage_d156_sha256")
        # The fit's own shared modules (band_summary, ci, N_DRAWS, AXIS_DRAW_SEED, ...) must be the
        # ones the pair jobs ran: a push between the last pair job and the fit would change them.
        own = [f for f in SHARED_MODULES if code0["sha256"].get(f) != d["code"]["sha256"].get(f)]
        if own and args.r2:
            # D-300: the re-issue's guarantee is stronger - every compared key must equal the first record, which
            # was fitted when the modules matched; the differing modules are recorded in reissue_of
            r2_module_diffs[s] = own
        elif own:
            diff.append(f"shared modules differ from this fit's: {own}")
        for k in ("blas_threads", "libraries"):
            if d["env"].get(k) != ref["env"].get(k):
                diff.append(f"env.{k}")
        if diff:
            raise SystemExit(f"REFUSING: {s} differs from {pairs[0]} in {diff}")
        if c["draw_seed"] != P.AXIS_DRAW_SEED[c["axis"]]:
            raise SystemExit(f"REFUSING: {s} was partitioned with seed {c['draw_seed']}, not the axis seed")
        if not args.allow_unregistered and (c["selection_file"] != SELECTION_OF_RECORD or
                                            d["damage_d156_check"].get("damage_d156_sha256") != dmg_sha_want):
            raise SystemExit(f"REFUSING: {s} did not use the selection of record and the damage file of record "
                             f"({dmg_rel_want}, D-173; at 5% D-248)")
        # one item-level and one cluster-level partition sequence per axis (D-156, D-164)
        key = (c["pooled_negatives"]["item_ids_sha256"], c["draw_seed"], c["cluster_draw_seed"],
               c["cluster_manifest"]["sha256"])
        if axis_ids.setdefault(c["axis"], key) != key:
            raise SystemExit(f"REFUSING: {s} pooled negatives differ from other pairs on {c['axis']}")
    roles = ref["config"]["predictor_roles"]

    # Stack every cell of every pair: arrays (cells, 1 + n_draws), column 0 the frozen split.
    arrays: dict[str, list[np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for s, d in docs.items():
        rec = d["draw_arrays"]
        npz = ROOT / rec["file"]
        if not npz.exists() or npz.stat().st_size != rec["bytes"] or sha256_file(npz) != rec["sha256"]:
            raise SystemExit(f"REFUSING: {rec['file']} is not the recorded bytes")
        file_sha[rec["file"]] = rec["sha256"]
        with np.load(npz, allow_pickle=False) as z:
            if [int(v) for v in z["cell_layer"]] != [c["layer"] for c in d["cells"]] or \
                    [str(v) for v in z["cell_probe_set"]] != [c["probe_set"] for c in d["cells"]]:
                raise SystemExit(f"REFUSING: {rec['file']} rows do not match the cells of {s}")
            for k in z.files:
                if not k.startswith("cell_"):
                    arrays.setdefault(k, []).append(z[k])
        rows += [{"pair": s, "layer": c["layer"], "probe_set": c["probe_set"], "deployer": c["is_selected_layer"],
                  "input_verification_flag": c["input_verification"]["flag"]}      # fail closed: required key
                 for c in d["cells"]]
    A = {k: np.vstack(v) for k, v in arrays.items()}
    # D-170/D-172: a fit row whose cells include a pin-only input carries the flag, with the count.
    flagged = np.array([r["input_verification_flag"] is not None for r in rows])
    pin_flag = next((r["input_verification_flag"] for r in rows if r["input_verification_flag"]), None)
    n_draws = int(ref["config"]["n_draws"])
    if A["log_real_ratio"].shape != (len(rows), 1 + n_draws):
        raise SystemExit("REFUSING: per-draw arrays have the wrong shape")
    pair_of = np.array([pairs.index(r["pair"]) for r in rows])
    pset_of = np.array([r["probe_set"] for r in rows])
    deployer = np.array([r["deployer"] for r in rows])

    # Pair-cluster bootstrap replicates: multiplicity of each pair in each replicate.
    rng = np.random.default_rng(args.boot_seed)
    idx = rng.integers(0, len(pairs), size=(args.n_boot, len(pairs)))
    mult = np.stack([np.bincount(r, minlength=len(pairs)) for r in idx])        # (B, pairs)

    def within_draw(X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> dict[str, Any]:
        f = wols(X, Y, W)
        out = {k: P.band_summary(f[k][1:], f[k][0]) for k in ("slope", "intercept", "pearson_r")}
        n = f["n"][1:]
        out["n_cells_per_draw"] = {"min": int(n.min()), "mean": float(n.mean()), "max": int(n.max()),
                                   "frozen": int(f["n"][0])}
        out["n_draws_undefined"] = int(np.isnan(f["slope"][1:]).sum())
        return out

    def pair_bootstrap(x: np.ndarray, y: np.ndarray, keep: np.ndarray, pairs_idx: np.ndarray) -> dict[str, Any]:
        pt = wols(x[:, None], y[:, None], keep[:, None].astype(float))
        Wb = (mult[:, pairs_idx] * keep[None, :]).T                                # (cells, B)
        rep = wols(np.repeat(x[:, None], args.n_boot, 1), np.repeat(y[:, None], args.n_boot, 1), Wb)
        return {"n_cells_used": int(pt["n"][0]), "n_cells_kept": int(keep.sum()),
                "n_cells_kept_but_undefined": int((keep & ~(np.isfinite(x) & np.isfinite(y))).sum()),
                "point": {k: (None if np.isnan(pt[k][0]) else float(pt[k][0])) for k in ("slope", "intercept", "pearson_r")},
                **{f"{k}_ci95": P.ci(rep[k])["ci95"] for k in ("slope", "intercept", "pearson_r")},
                "n_replicates_undefined": int(np.isnan(rep["slope"]).sum())}

    def fraction_floored(cens_draw: np.ndarray, defined: np.ndarray) -> np.ndarray:
        """Per cell, the fraction of the draws (columns 1..) in which the prediction is defined
        that were floored (realised OR predicted); NaN for a cell with no defined draw. The same
        definition twomoment_cells records as predicted[*].fraction_of_draws_floored."""
        c, d = cens_draw[:, 1:], defined[:, 1:]
        n = d.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, (c & d).sum(axis=1) / n, np.nan)

    # Cells: every cell kept, floored draws at 1/n_neg (the headline cell set, D-163 (2)); or, as
    # a sensitivity only, cells floored in more than SENSITIVITY_FLOORED_SHARE of their defined
    # draws left out. Within one predictor row that cell set is the same in both intervals and in
    # every draw; it is cut per predictor, so rows of different predictors can differ in cells.
    CELLSETS = ("all_cells_floored_draws_kept", f"sensitivity_excluding_cells_floored_over_{SENSITIVITY_FLOORED_SHARE:g}")
    CELLSET_ROLE = {CELLSETS[0]: "headline cell set (D-163 (2))", CELLSETS[1]: "sensitivity only (D-163 (2))"}
    # The ONE headline row (D-163 (1), (2), (4); D-152 4b: all layer cells at the selected C, the
    # ratio scale; C=1 is the sensitivity row, the deployer subset its own row, the FPR scale a check).
    HEADLINE = {"probe_set": "selected", "cells": "all", "cell_set": CELLSETS[0],
                "predictor": "moment_full_in_sample", "scale": "ratio_scale"}
    def level_arrays(sfx: str) -> dict[str, np.ndarray]:
        """One level's per-draw arrays under the item-level names: sfx '' is the item level
        (D-156), '__cluster' the cluster level (D-164)."""
        if not sfx:
            return {k: v for k, v in A.items() if not k.endswith("__cluster")}
        return {k[:-len(sfx)]: v for k, v in A.items() if k.endswith(sfx)}

    def flags(sel: np.ndarray, keep: np.ndarray) -> dict[str, Any]:
        n = int(flagged[sel][keep].sum())
        return {"n_cells_flagged": n, "flag": pin_flag if n else None}

    def run_fits(G: dict[str, np.ndarray], item_level: bool) -> tuple[list, list, dict]:
        """Every fit row on one level. Only the ITEM level has a headline row (D-163 (1)); at the
        cluster level the row at the same coordinates is the headline's D-164 sensitivity."""
        fits: list[dict[str, Any]] = []
        comparisons: list[dict[str, Any]] = []
        headline_row: dict[str, Any] = {}
        for pset in ("selected", "C=1"):
            for subset in ("all", "deployer"):
                sel = (pset_of == pset) & (deployer if subset == "deployer" else True)
                rc = G["real_censored"][sel]
                for pred in PRED_KEYS:
                    cens_draw = rc | G[f"pred_censored__{pred}"][sel]                   # (cells, 1 + n_draws)
                    share = fraction_floored(cens_draw, np.isfinite(G[f"log_pred_ratio__{pred}"][sel]))
                    for cellset in CELLSETS:
                        keep = np.ones(int(sel.sum()), bool) if cellset == CELLSETS[0] else share <= SENSITIVITY_FLOORED_SHARE
                        W = np.repeat(keep[:, None], cens_draw.shape[1], 1).astype(float)
                        row: dict[str, Any] = {"probe_set": pset, "cells": subset, "cell_set": cellset,
                                               "cell_set_role": CELLSET_ROLE[cellset],
                                               "predictor": pred, "role": roles[pred], "n_cells": int(sel.sum()),
                                               "n_cells_kept": int(keep.sum()), "input_verification": flags(sel, keep),
                                               "fraction_of_draws_floored_per_cell": {"min": float(np.nanmin(share)) if np.isfinite(share).any() else None,
                                                                                      "median": float(np.nanmedian(share)) if np.isfinite(share).any() else None,
                                                                                      "max": float(np.nanmax(share)) if np.isfinite(share).any() else None,
                                                                                      "n_cells_over_sensitivity_cut": int((share > SENSITIVITY_FLOORED_SHARE).sum())},
                                               "n_cells_floored_frozen_split": int(cens_draw[:, 0].sum())}
                        for scale, xk, yk in (("ratio_scale", f"log_pred_ratio__{pred}", "log_real_ratio"),
                                              ("fpr_post_scale", f"log_fpr_pred__{pred}", "log_fpr_post")):
                            X, Y = G[xk][sel], G[yk][sel]
                            x, y, _ = d156_estimates(X, Y)
                            at_head = (pset, subset, cellset, pred, scale) == tuple(HEADLINE.values())
                            row[scale] = {"is_headline": at_head and item_level,
                                          "fit_on_d156_estimates_pair_cluster_ci": pair_bootstrap(x, y, keep, pair_of[sel]),
                                          "partition_sensitivity_within_draw": within_draw(X, Y, W)}
                            if at_head:
                                headline_row = {**row[scale]["fit_on_d156_estimates_pair_cluster_ci"],
                                                "input_verification": row["input_verification"],
                                                "within_draw_slope": row[scale]["partition_sensitivity_within_draw"]["slope"]}
                        if not item_level:
                            row["cell_set_role"] = f"cluster-level sensitivity (D-164) of: {row['cell_set_role']}"
                        fits.append(row)
                # two-moment minus one-moment Pearson r on identical cells
                for src in ("in_sample", "out_of_sample"):
                    two, one = f"moment_full_{src}", f"translation_{src}"
                    X2, X1, Y = G[f"log_pred_ratio__{two}"][sel], G[f"log_pred_ratio__{one}"][sel], G["log_real_ratio"][sel]
                    both = np.isfinite(X2) & np.isfinite(X1)
                    share = fraction_floored(rc | G[f"pred_censored__{two}"][sel] | G[f"pred_censored__{one}"][sel], both)
                    for cellset in CELLSETS:
                        cells_ok = np.ones(int(sel.sum()), bool) if cellset == CELLSETS[0] else share <= SENSITIVITY_FLOORED_SHARE
                        # identical cells in both forms: those whose two predictions are defined in every draw
                        keep = both[:, 1:].all(axis=1) & cells_ok
                        W = np.repeat(keep[:, None], both.shape[1], 1).astype(float)
                        rd = wols(X2, Y, W)["pearson_r"] - wols(X1, Y, W)["pearson_r"]
                        x2, y2, _ = d156_estimates(X2, Y)
                        x1, y1, _ = d156_estimates(X1, Y)
                        pt = wols(x2[:, None], y2[:, None], keep[:, None].astype(float))["pearson_r"][0] - \
                            wols(x1[:, None], y1[:, None], keep[:, None].astype(float))["pearson_r"][0]
                        Wb = (mult[:, pair_of[sel]] * keep[None, :]).T
                        rep = wols(np.repeat(x2[:, None], args.n_boot, 1), np.repeat(y2[:, None], args.n_boot, 1), Wb)["pearson_r"] - \
                            wols(np.repeat(x1[:, None], args.n_boot, 1), np.repeat(y1[:, None], args.n_boot, 1), Wb)["pearson_r"]
                        comparisons.append({"probe_set": pset, "cells": subset, "cell_set": cellset,
                                            "cell_set_role": CELLSET_ROLE[cellset], "source": src,
                                            "roles": {"two_moment": roles[two], "one_moment": roles[one]},
                                            "input_verification": flags(sel, keep),
                                            "fit_on_d156_estimates_pair_cluster_ci": {"n_cells_kept": int(keep.sum()),
                                                                                      "point": None if np.isnan(pt) else float(pt),
                                                                                      **P.ci(rep)},
                                            "partition_sensitivity_within_draw": P.band_summary(rd[1:], rd[0])})
        if not headline_row:
            raise SystemExit("REFUSING: no fit row matched the headline coordinates")
        return fits, comparisons, headline_row

    fits, comparisons, headline_row = run_fits(level_arrays(""), item_level=True)
    fits_c, comparisons_c, headline_row_c = run_fits(level_arrays("__cluster"), item_level=False)
    CLUSTER_FIT_NOTE = ("cluster-level re-partitions (D-164): a sensitivity reported beside the item level, never "
                        "instead of it. In 'partition_sensitivity_within_draw' the 'frozen' value, its "
                        "frozen_percentile and n_cells_floored_frozen_split belong to the frozen split, which is NOT "
                        "a cluster-level draw (it splits 36 of 67 Aegis and 7 of 11 WildGuardMix clusters)")

    # D-163 (2): every cell's fraction of floored draws, realised and per predictor, and the
    # number of draws its per-cell D-156 estimate used (realised and predicted both defined).
    cell_table = []
    for i, r in enumerate(rows):
        entry = dict(r)
        for lvl_tag, sfx in (("", ""), ("__cluster_level_D164", "__cluster")):
            entry[f"fraction_of_draws_floored_realised{lvl_tag}"] = float(A[f"real_censored{sfx}"][i, 1:].mean())
            for p in PRED_KEYS:
                dfd = np.isfinite(A[f"log_pred_ratio__{p}{sfx}"][i:i + 1])
                fl = fraction_floored(A[f"real_censored{sfx}"][i:i + 1] | A[f"pred_censored__{p}{sfx}"][i:i + 1], dfd)[0]
                entry[f"fraction_of_draws_floored__{p}{lvl_tag}"] = None if np.isnan(fl) else float(fl)
                entry[f"n_draws_in_d156_estimate__{p}{lvl_tag}"] = int(d156_estimates(
                    A[f"log_pred_ratio__{p}{sfx}"][i:i + 1], A[f"log_real_ratio{sfx}"][i:i + 1])[2][0])
        cell_table.append(entry)
    # the cache bytes behind every pair file, so the D-170 audit (counted by bytes) sees this file too
    input_cache = {f"{s}:{k}": h for s, d in docs.items() for k, h in d["input_cache_sha256"].items()}

    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"] or any(sha256_file(ROOT / f) != h for f, h in file_sha.items()):
        raise SystemExit("REFUSING to write: source or an input file changed during the run")
    payload = {**provenance_block("twomoment_fit"),
               "preregistration": "D-152 (4b), D-156, D-163, D-164, D-166" + ("; 4e: D-152, D-248 (the 5% arm)" if args.fpr5 else ""),
               "code": code0,
               "input_verification": {"flag": pin_flag, "n_cells_flagged": int(flagged.sum()), "n_cells": len(rows),
                                      "rule": "D-170/D-172: a flagged row enters no paper table until D-168 has passed"},
               "headline": {"coordinates": HEADLINE,
                            "estimate": "OLS of log realised on log predicted paired ratio, fitted on the per-cell "
                                        "D-156 estimates (every layer cell at the selected C)",
                            "interval": f"pair-cluster bootstrap, B = {args.n_boot}, seed {args.boot_seed}, 95% percentile",
                            "cells": "every cell, floored draws kept at 1/n_neg",
                            "result": headline_row,
                            "source": "D-163 (1), (2), (4); D-152 4b",
                            "partition_sensitivity_caveat": PARTITION_CAVEAT,
                            "cluster_level_sensitivity_D164": {"result": headline_row_c, "note": CLUSTER_FIT_NOTE}},
               "partition_sensitivity_caveat": PARTITION_CAVEAT,
               "input_cache_sha256": input_cache,
               "config": {"n_boot": args.n_boot, "boot_seed": args.boot_seed, "cluster": "update pair",
                          "n_draws": n_draws, "axis_partitions": {a: {"item_ids_sha256": v[0], "draw_seed": v[1],
                                                                      "cluster_draw_seed": v[2],
                                                                      "cluster_manifest_sha256": v[3]}
                                                                  for a, v in axis_ids.items()},
                          "predictors": PRED_KEYS, "predictor_roles": roles,
                          "allow_unregistered": args.allow_unregistered,
                          "per_pair_code_commits": {s: d["code"]["commit"] for s, d in docs.items()},
                          "d156_cell_estimate": "log of the mean over the draws where realised and predicted are both defined",
                          "sensitivity_floored_share": SENSITIVITY_FLOORED_SHARE},
               "population": {"declared_in": "src/populations.py", "group": "UPDATE_PAIRS_REPAIR", "pairs": pairs},
               "input_files_sha256": file_sha, "n_cells": len(rows), "fits": fits,
               "two_vs_one_moment": comparisons,
               "cluster_level_D164_note": CLUSTER_FIT_NOTE,
               "fits_cluster_level_D164": fits_c, "two_vs_one_moment_cluster_level_D164": comparisons_c,
               "cells": cell_table}
    if args.r2:
        payload = reissue_payload(payload, *FIRST_RECORD)
        payload["reissue_of"]["shared_modules_differing_from_the_pair_records"] = r2_module_diffs
    digest = write_results(out_path, payload, exclusive=True)
    rel = str(out_path.relative_to(ROOT)).replace("\\", "/") if out_path.is_relative_to(ROOT) else out_path.name
    print(f"Written: {rel}\nsha256:  {digest}")
    sl = lambda h: "undefined" if h["point"]["slope"] is None else f"{h['point']['slope']:+.3f}"  # noqa: E731
    print(f"HEADLINE (D-163): slope {sl(headline_row)} {headline_row['slope_ci95']} n={headline_row['n_cells_used']} "
          f"| cluster-level sensitivity (D-164): slope {sl(headline_row_c)} {headline_row_c['slope_ci95']} "
          f"| D-170: {int(flagged.sum())} of {len(rows)} cells flagged, headline flag {headline_row['input_verification']['flag']}")

    def fmt(b: dict) -> str:
        if b["mean"] is None:
            return "undefined"
        fz = "none" if b["frozen"] is None else f"{b['frozen']:+.3f}"
        return f"{b['mean']:+.3f} [{b['band95'][0]:+.3f},{b['band95'][1]:+.3f}] fz {fz}"
    print("\nlog realised on log predicted paired ratio: fit on per-cell D-156 estimates with pair-cluster CI "
          "| within-draw (partition sensitivity) mean [band], frozen split. ONE row is the headline (D-163).")
    for f in fits:
        pb = f["ratio_scale"]["fit_on_d156_estimates_pair_cluster_ci"]
        rs = f["ratio_scale"]["partition_sensitivity_within_draw"]
        ci = pb["slope_ci95"] or [np.nan, np.nan]
        rci = pb["pearson_r_ci95"] or [np.nan, np.nan]
        pts = "undefined" if pb["point"]["slope"] is None else f"{pb['point']['slope']:+.3f}"
        ptr = "undefined" if pb["point"]["pearson_r"] is None else f"{pb['point']['pearson_r']:+.3f}"
        tag = "HEADLINE" if f["ratio_scale"]["is_headline"] else ("all-cells" if f["cell_set"] == CELLSETS[0] else "sens>2.5%")
        print(f"  {tag:<9} [{f['probe_set']:<8}] {f['cells']:<8} {f['predictor']:<26} slope {pts} [{ci[0]:+.3f},{ci[1]:+.3f}] "
              f"r {ptr} [{rci[0]:+.3f},{rci[1]:+.3f}] n={pb['n_cells_used']} | within-draw slope {fmt(rs['slope'])}")
    print("\nPearson r, two-moment minus one-moment, same cells (no headline among these)")
    for c in comparisons:
        pb, b = c["fit_on_d156_estimates_pair_cluster_ci"], c["partition_sensitivity_within_draw"]
        ci = pb["ci95"] or [np.nan, np.nan]
        pt = "undefined" if pb["point"] is None else f"{pb['point']:+.3f}"
        tag = "all-cells" if c["cell_set"] == CELLSETS[0] else "sens>2.5%"
        print(f"  {tag:<9} [{c['probe_set']:<8}] {c['cells']:<8} {c['source']:<13} {pt} [{ci[0]:+.3f},{ci[1]:+.3f}] "
              f"n={pb['n_cells_kept']} | within-draw {fmt(b)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
