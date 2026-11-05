"""4b supplement (D-175): the unflagged-cell sensitivity and prediction-accuracy summaries,
computed from the existing per-pair 4b outputs without a re-run.

Pre-registered in D-175 before either number existed.

  (a) UNFLAGGED SENSITIVITY  the D-163 headline fit (selected C, every layer cell,
      moment_full_in_sample, ratio scale, OLS on the per-cell D-156 estimates, pair-cluster
      bootstrap B = 5000, seed 520260925) restricted to the cells whose inputs carry no D-170
      flag; the bootstrap resamples those pairs. Slope, Pearson r and intercept with 95%
      intervals, beside the headline, and the same at the D-164 cluster level.
  (b) PREDICTION ACCURACY    per cell |log(realised / predicted)| from the same per-cell D-156
      estimates; its median and the share of cells predicted within a factor of 1.5, for all
      selected-C cells and the deployer's layer, for every predictor (moment_full_in_sample
      primary), over all cells and over the unflagged cells, at both levels.

  (c) PER-PAIR OFFSETS (D-177) the mean log(realised / predicted) of the headline predictor per
      pair, with its D-156 partition band (a single pair cannot be resampled at the pair level),
      and pooled rows (all pairs; by dataset; by family; by dataset within the unflagged pairs)
      with pair-cluster bootstrap intervals where a row has at least 3 pairs; the dataset
      contrast on the two matched model pairs, which is aliased with the D-170 flag. Every row
      carries the accuracy line (D-177 (3)) and its flagged-cell count.

Inputs: results/twomoment_<pair>.json and _draws.npz for the nine pairs of UPDATE_PAIRS_REPAIR,
each of whose sha256 must equal what results/twomoment_fit.json recorded; the fit's own headline
is copied beside for comparison. Writes results/twomoment_supplement_r3.json (never overwritten),
whose (a) and (b) must equal those of results/twomoment_supplement_r2.json (D-176), which it
supersedes; the first run, results/twomoment_supplement.json, lacked the per-file and per-row
D-170 flag (D-172).
--fpr5 (D-271 (3) item 9, 4e): the same blocks at 5% from results/twomoment_fit_fpr5.json and the nine
results/twomoment_<pair>_fpr5.json (+ _draws.npz); writes results/twomoment_supplement_r3_fpr5.json. The 5% arm
has no r2 to reproduce; the all-cells fit must reproduce the 5% fit's headline.
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
from provenance import ensure_utf8_stdout, provenance_block, reissue_payload, sha256_file, write_results  # noqa: E402
from twomoment_fit import BOOT, PRED_KEYS, d156_estimates, wols  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/twomoment_supplement.py", "src/twomoment_fit.py", "src/paired.py", "src/populations.py",
              "src/provenance.py", "src/superseded_provenance.py"]
FACTOR = 1.5                       # D-175 (b): "predicted within a factor of 1.5"
HEADLINE_PRED = "moment_full_in_sample"
LEVELS = {"item_level_D156": "", "cluster_level_D164": "__cluster"}
# D-175 (a) names the unflagged cell set before any number: 90 cells, these five pairs.
PREREG_UNFLAGGED = {"n_cells": 90, "pairs": ["repair_gemma2_base2instruct", "repair_llama31_base2instruct",
                                             "repair_olmo_base2instruct", "repair_WG_mistral_base2instruct",
                                             "repair_WG_llama3_base2instruct"]}
# The fit's headline and its cluster-level sensitivity, which the all-cells fit here must reproduce.
FIT_RESULT = {"item_level_D156": lambda f: f["headline"]["result"],
              "cluster_level_D164": lambda f: f["headline"]["cluster_level_sensitivity_D164"]["result"]}
PREVIOUS = RESULTS / "twomoment_supplement_r2.json"                  # D-176; r3 must reproduce its blocks
# D-300: the record --r2 re-issues as twomoment_supplement_r4.json (r3 names no repo ID or revision, D-297 (1))
FIRST_RECORD = ("results/twomoment_supplement_r3.json", "b7ecaf97d9caf3dd004aaee860ca8008a38d7600e441ae850a781076c9b14d74")
# D-177 (2): the groupings, named before any number.
DATASET = {"safety": "Aegis", "safety_wildguard": "WildGuardMix"}
FAMILY = {"repair_mistral_base2instruct": "Mistral", "repair_WG_mistral_base2instruct": "Mistral",
          "repair_llama3_base2instruct": "Llama-3/3.1", "repair_llama31_base2instruct": "Llama-3/3.1",
          "repair_llama3_to_31": "Llama-3/3.1", "repair_WG_llama3_base2instruct": "Llama-3/3.1",
          "repair_gemma2_base2instruct": "Gemma-2", "repair_qwen_base2instruct": "Qwen2.5",
          "repair_olmo_base2instruct": "OLMo-2"}
MATCHED = [("repair_mistral_base2instruct", "repair_WG_mistral_base2instruct"),    # (Aegis, WildGuardMix)
           ("repair_llama3_base2instruct", "repair_WG_llama3_base2instruct")]
MIN_PAIRS_FOR_CI = 3
EXPECTED_GROUP_PAIRS = {"all_pairs": 9, "dataset:Aegis": 7, "dataset:WildGuardMix": 2, "family:Llama-3/3.1": 4,
                        "family:Mistral": 2, "family:Gemma-2": 1, "family:Qwen2.5": 1, "family:OLMo-2": 1,
                        "unflagged_pairs_only:dataset:Aegis": 3, "unflagged_pairs_only:dataset:WildGuardMix": 2}
PARTITION_NOTE = ("95% band over the 1000 D-156 draws of the within-draw mean offset over the pair's cells, with the "
                  "frozen value; a partition sensitivity only (D-166): within a draw, predicted and realised share "
                  "the pre-update FPR and the threshold")
MATCHED_NOTE = ("on both matched pairs the Aegis side is D-170-flagged and the WildGuardMix side is not: this "
                "contrast is aliased with the flag and is not interpreted before D-168 (D-177 (1), (2c))")


def subset_equal(old: Any, new: Any) -> bool:
    """Every key of `old` is in `new` with an equal value, recursively (new keys allowed)."""
    if isinstance(old, dict):
        return isinstance(new, dict) and all(k in new and subset_equal(v, new[k]) for k, v in old.items())
    return old == new


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="4b supplement (D-175)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fpr5", action="store_true",
                    help="D-271 (3) item 9: the same blocks for the 5%% arm (4e, D-248), from results/twomoment_fit_fpr5.json "
                         "and the nine _fpr5 4b records; writes twomoment_supplement_r3_fpr5.json (no r2 to reproduce)")
    ap.add_argument("--r2", action="store_true",
                    help="re-issue twomoment_supplement_r3.json as _r4, naming each cache input's repo ID and "
                         "revision, checked equal (D-300)")
    args = ap.parse_args()
    if args.r2 and args.fpr5:
        raise SystemExit("REFUSING: --r2 re-issues the 1% record only (D-300)")
    sfx = "_fpr5" if args.fpr5 else ""
    final_out = RESULTS / ("twomoment_supplement_r4.json" if args.r2 else f"twomoment_supplement_r3{sfx}.json")
    out_path = Path(args.out).resolve() if args.out else final_out.resolve()
    if out_path.exists():
        raise SystemExit(f"REFUSING: {out_path} exists")
    if args.r2 and out_path.is_relative_to(RESULTS.resolve()) and out_path != final_out.resolve():
        raise SystemExit("REFUSING: under results/, --r2 writes only twomoment_supplement_r4.json")
    if not args.fpr5 and out_path == final_out.resolve() and not PREVIOUS.exists():
        raise SystemExit(f"REFUSING: the final name needs {PREVIOUS.name} to check against (D-177)")
    code0 = P.code_block(CODE_FILES)

    fit_path = RESULTS / f"twomoment_fit{sfx}.json"
    fit_sha = sha256_file(fit_path)
    fit = json.load(open(fit_path, encoding="utf-8"))
    pairs = list(POP.UPDATE_PAIRS_REPAIR)
    arrays: dict[str, list[np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for s in pairs:
        jf = f"results/twomoment_{s}{sfx}.json"
        if sha256_file(ROOT / jf) != fit["input_files_sha256"][jf]:
            raise SystemExit(f"REFUSING: {jf} is not the file {fit_path.name} read")
        d = json.load(open(ROOT / jf, encoding="utf-8"))
        rec = d["draw_arrays"]
        if sha256_file(ROOT / rec["file"]) != fit["input_files_sha256"][rec["file"]]:
            raise SystemExit(f"REFUSING: {rec['file']} is not the file {fit_path.name} read")
        with np.load(ROOT / rec["file"], allow_pickle=False) as z:
            if [int(v) for v in z["cell_layer"]] != [c["layer"] for c in d["cells"]] or \
                    [str(v) for v in z["cell_probe_set"]] != [c["probe_set"] for c in d["cells"]]:
                raise SystemExit(f"REFUSING: {rec['file']} rows do not match the cells of {s}")
            for k in z.files:
                if not k.startswith("cell_"):
                    arrays.setdefault(k, []).append(z[k])
        rows += [{"pair": s, "layer": c["layer"], "probe_set": c["probe_set"], "deployer": c["is_selected_layer"],
                  "flag": c["input_verification"]["flag"],       # fail closed: required key
                  "dataset": DATASET[d["config"]["axis"]], "family": FAMILY[s]} for c in d["cells"]]
    A = {k: np.vstack(v) for k, v in arrays.items()}
    pset = np.array([r["probe_set"] for r in rows])
    dep = np.array([r["deployer"] for r in rows])
    flagged = np.array([r["flag"] is not None for r in rows])
    pair_of = np.array([pairs.index(r["pair"]) for r in rows])
    selected = pset == "selected"
    # D-170/D-172: the flag is carried per file and per row, with the count of flagged cells.
    pin_flag = next((r["flag"] for r in rows if r["flag"]), None)

    def flags(keep_sel: np.ndarray) -> dict[str, Any]:
        n = int(flagged[selected][keep_sel].sum())
        return {"n_cells_flagged": n, "flag": pin_flag if n else None}

    unflagged_pairs = sorted({pairs[p] for p in pair_of[selected & ~flagged]}, key=pairs.index)
    if int((selected & ~flagged).sum()) != PREREG_UNFLAGGED["n_cells"] or unflagged_pairs != PREREG_UNFLAGGED["pairs"]:
        raise SystemExit(f"REFUSING: the unflagged cells are not D-175's: {int((selected & ~flagged).sum())} cells, "
                         f"pairs {unflagged_pairs}")
    if flagged[selected & np.isin(pair_of, [pairs.index(s) for s in unflagged_pairs])].any():
        raise SystemExit("REFUSING: a pair of the unflagged set also has flagged cells (the bootstrap resamples whole pairs)")

    def boot_fit(x: np.ndarray, y: np.ndarray, sub: np.ndarray) -> dict[str, Any]:
        """The headline's estimator (twomoment_fit's pair_bootstrap, every kept cell) on the cells
        `sub`, bootstrapping the pairs present in `sub`: rows sliced first and the replicates drawn
        as the fit draws them, so on the headline's cells it repeats the fit's arithmetic exactly."""
        xs, ys, po = x[sub], y[sub], pair_of[selected][sub]
        pk = sorted(set(po.tolist()))
        local = np.array([pk.index(p) for p in po])
        rng = np.random.default_rng(BOOT["boot_seed"])
        idx = rng.integers(0, len(pk), size=(BOOT["n_boot"], len(pk)))
        mult = np.stack([np.bincount(r, minlength=len(pk)) for r in idx])        # (B, pairs in sub)
        keep = np.ones(len(xs), bool)
        pt = wols(xs[:, None], ys[:, None], keep[:, None].astype(float))
        rep = wols(np.repeat(xs[:, None], BOOT["n_boot"], 1), np.repeat(ys[:, None], BOOT["n_boot"], 1),
                   (mult[:, local] * keep[None, :]).T)
        return {"n_cells": int(pt["n"][0]), "n_pairs_bootstrapped": len(pk),
                "pairs": [pairs[p] for p in pk], "input_verification": flags(sub),
                "point": {k: (None if np.isnan(pt[k][0]) else float(pt[k][0])) for k in ("slope", "intercept", "pearson_r")},
                **{f"{k}_ci95": P.ci(rep[k])["ci95"] for k in ("slope", "intercept", "pearson_r")},
                "n_replicates_undefined": int(np.isnan(rep["slope"]).sum()),
                "accuracy": accuracy(x, y, sub)}                                          # D-177 (3)

    def accuracy(x: np.ndarray, y: np.ndarray, keep: np.ndarray) -> dict[str, Any]:
        """|log realised - log predicted| per cell over `keep` (selected-C rows); undefined cells counted."""
        e = np.abs(y - x)[keep]
        n_undef = int((~np.isfinite(e)).sum())
        e = e[np.isfinite(e)]
        return {"n_cells": int(e.size), "n_cells_undefined": n_undef, "input_verification": flags(keep),
                "median_abs_log_realised_over_predicted": float(np.median(e)) if e.size else None,
                "share_within_factor_1_5": float((e <= np.log(FACTOR)).mean()) if e.size else None}

    pair_sel = pair_of[selected]
    pair_dataset = {r["pair"]: r["dataset"] for r in rows}
    group_masks = {"all_pairs": np.ones(int(selected.sum()), bool)}
    for key, attr in (("dataset", "dataset"), ("family", "family")):
        for v in dict.fromkeys(r[attr] for r in rows):
            group_masks[f"{key}:{v}"] = np.array([r[attr] == v for r in rows])[selected]
    for v in ("Aegis", "WildGuardMix"):
        group_masks[f"unflagged_pairs_only:dataset:{v}"] = group_masks[f"dataset:{v}"] & ~flagged[selected]
    got = {g: len(set(pair_sel[m].tolist())) for g, m in group_masks.items()}
    if got != EXPECTED_GROUP_PAIRS:
        raise SystemExit(f"REFUSING: the groups are not D-177's: {got}")

    def within_draw_mean(Xd: np.ndarray, Yd: np.ndarray, sub: np.ndarray) -> np.ndarray:
        """Per draw (column), the mean offset log realised - log predicted over the cells `sub`."""
        E = Yd[sub] - Xd[sub]
        ok = np.isfinite(E)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(ok.sum(axis=0) > 0, np.where(ok, E, 0.0).sum(axis=0) / ok.sum(axis=0), np.nan)

    def offset_row(x: np.ndarray, y: np.ndarray, sub: np.ndarray, Xd: np.ndarray, Yd: np.ndarray) -> dict[str, Any]:
        """Mean offset over the cells `sub` (selected-C rows), cell-weighted as the headline fit. One
        pair: its D-156 partition band. Several: a pair-cluster bootstrap interval (>= 3 pairs)."""
        e = (y - x)[sub]
        fin = np.isfinite(e)
        po = pair_sel[sub]
        pk = sorted(set(po.tolist()))
        row: dict[str, Any] = {"pairs": [pairs[p] for p in pk], "n_pairs": len(pk), "n_cells": int(fin.sum()),
                               "n_cells_undefined": int((~fin).sum()),
                               "mean_log_realised_over_predicted": float(e[fin].mean()) if fin.any() else None,
                               "accuracy": accuracy(x, y, sub), "input_verification": flags(sub)}
        if len(pk) == 1:
            v = within_draw_mean(Xd, Yd, sub)
            row["partition_band_D156"] = {**P.band_summary(v[1:], v[0]), "note": PARTITION_NOTE}
        elif len(pk) >= MIN_PAIRS_FOR_CI:
            local = np.array([pk.index(p) for p in po])
            rng = np.random.default_rng(BOOT["boot_seed"])
            idx = rng.integers(0, len(pk), size=(BOOT["n_boot"], len(pk)))
            mult = np.stack([np.bincount(r, minlength=len(pk)) for r in idx])
            W = mult[:, local] * fin[None, :]
            with np.errstate(invalid="ignore", divide="ignore"):
                rep = (W * np.where(fin, e, 0.0)[None, :]).sum(axis=1) / W.sum(axis=1)
            row["pair_cluster_bootstrap"] = {**P.ci(rep), **BOOT}
        else:
            row["pair_cluster_bootstrap"] = None
            row["no_interval"] = f"{len(pk)} pairs: no pair-cluster interval below {MIN_PAIRS_FOR_CI} pairs (D-177)"
        return row

    out: dict[str, Any] = {}
    for lvl, sfx in LEVELS.items():
        # per-cell estimates on the selected-C rows (d156_estimates is row-wise; sliced as the fit slices)
        X = {p: d156_estimates(A[f"log_pred_ratio__{p}{sfx}"][selected], A[f"log_real_ratio{sfx}"][selected])
             for p in PRED_KEYS}
        xh, yh, _ = X[HEADLINE_PRED]
        sens = {"all_cells": boot_fit(xh, yh, np.ones(int(selected.sum()), bool)),
                "unflagged_cells": boot_fit(xh, yh, ~flagged[selected])}
        d_s, u_s = dep[selected], ~flagged[selected]
        acc = {p: {scope: accuracy(X[p][0], X[p][1], m) for scope, m in
                   (("all_cells", np.ones(int(selected.sum()), bool)), ("deployer_layer", d_s),
                    ("unflagged_all_cells", u_s), ("unflagged_deployer_layer", d_s & u_s))}
               for p in PRED_KEYS}
        # The all-cells fit is the fit's own headline (same cells, estimator, bootstrap draws): it must
        # reproduce it exactly, or this script is not computing the same thing on the unflagged cells.
        ref = FIT_RESULT[lvl](fit)
        mine = sens["all_cells"]
        if mine["point"] != ref["point"] or mine["n_cells"] != ref["n_cells_used"] or \
                any(mine[f"{k}_ci95"] != ref[f"{k}_ci95"] for k in ("slope", "intercept", "pearson_r")):
            raise SystemExit(f"REFUSING: the all-cells fit does not reproduce {fit_path.name}'s {lvl} headline")
        sens["all_cells"]["reproduces_twomoment_fit_headline"] = True
        # D-177 (2): per-pair offsets of the headline predictor, pooled rows, matched dataset contrast.
        Xd, Yd = A[f"log_pred_ratio__{HEADLINE_PRED}{sfx}"][selected], A[f"log_real_ratio{sfx}"][selected]
        key = "mean_log_realised_over_predicted"
        per_pair = {s: {"dataset": pair_dataset[s], "family": FAMILY[s],
                        **offset_row(xh, yh, pair_sel == pairs.index(s), Xd, Yd)} for s in pairs}
        pooled = {g: offset_row(xh, yh, m, Xd, Yd) for g, m in group_masks.items()}
        matched = []
        for a, w in MATCHED:
            ma, mw = pair_sel == pairs.index(a), pair_sel == pairs.index(w)
            dv = within_draw_mean(Xd, Yd, mw) - within_draw_mean(Xd, Yd, ma)
            matched.append({"aegis_pair": a, "wildguardmix_pair": w,
                            "difference_wildguardmix_minus_aegis": per_pair[w][key] - per_pair[a][key],
                            "partition_band_D156": {**P.band_summary(dv[1:], dv[0]),
                                                    "note": "draw d of one axis against draw d of the other; " + PARTITION_NOTE},
                            "input_verification": flags(ma | mw), "note": MATCHED_NOTE})
        out[lvl] = {"headline_fit": sens, "prediction_accuracy": acc,
                    "pair_offsets": {"predictor": HEADLINE_PRED, "offset": "log(realised / predicted), per-cell D-156 estimates",
                                     "per_pair": per_pair, "pooled": pooled, "matched_dataset_contrast": matched}}

    # (a) and (b) must be exactly those of r2 (D-176): r3 only adds the accuracy line and (c).
    prev_check = None
    if PREVIOUS.exists() and not args.fpr5:                   # r2 exists only at 1% (the 5% arm has no predecessor)
        prev = json.load(open(PREVIOUS, encoding="utf-8"))
        if prev["input_files_sha256"] != fit["input_files_sha256"] or \
                not all(subset_equal(prev["results"][lvl], out[lvl]) for lvl in LEVELS):
            raise SystemExit(f"REFUSING: the D-175 blocks differ from {PREVIOUS.name}")
        prev_check = {"file": f"results/{PREVIOUS.name}", "sha256": sha256_file(PREVIOUS),
                      "d175_blocks_equal": True, "status": "superseded by this file (D-177)"}

    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"] or sha256_file(fit_path) != fit_sha:
        raise SystemExit(f"REFUSING to write: source or {fit_path.name} changed during the run")
    # the base run_kind in both arms, as every _fpr5 record keeps it (target_fpr tells them apart); note that `sfx` is
    # rebound by the LEVELS loop above, so it must not be used here (review of the --fpr5 mode)
    payload = {**provenance_block("twomoment_supplement"),
               "preregistration": "D-175, D-177" + ("; D-248 (4e, the 5% arm, a sensitivity arm), D-271 (3) item 9" if args.fpr5 else ""),
               "code": code0, "target_fpr": 0.05 if args.fpr5 else 0.01,
               "supersedes": prev_check,
               "input_verification": {"flag": pin_flag, "n_cells_flagged": int(flagged[selected].sum()),
                                      "n_cells": int(selected.sum()),
                                      "rule": "D-170/D-172: a flagged row enters no paper table until D-168 has passed; "
                                              "each row carries its own flag and count"},
               "config": {"headline_predictor": HEADLINE_PRED, "probe_set": "selected", "scale": "ratio",
                          "estimate": "per-cell D-156 estimates (log mean over draws where both are defined)",
                          "bootstrap": BOOT, "factor": FACTOR,
                          "unflagged_rule": "cells whose input_verification flag is null (D-170/D-172)",
                          "twomoment_fit": {"file": f"results/{fit_path.name}", "sha256": fit_sha},
                          "headline_as_reported_by_twomoment_fit": fit["headline"]["result"],
                          "pair_offsets_D177": {"datasets": DATASET, "families": FAMILY, "matched_pairs": MATCHED,
                                                "min_pairs_for_interval": MIN_PAIRS_FOR_CI,
                                                "pooled_weighting": "cell-weighted, as the headline fit"}},
               "input_files_sha256": fit["input_files_sha256"],
               "input_cache_sha256": fit["input_cache_sha256"],
               "n_cells_selected": int(selected.sum()), "n_cells_selected_unflagged": int((selected & ~flagged).sum()),
               "results": out}
    if args.r2:   # target_fpr was added to the payload with the 5% mode (D-271 (3) item 9), after r3
        payload = reissue_payload(payload, *FIRST_RECORD, new_keys={"target_fpr": 0.01})
    digest = write_results(out_path, payload, exclusive=True)
    rel = str(out_path.relative_to(ROOT)).replace("\\", "/") if out_path.is_relative_to(ROOT) else out_path.name
    print(f"Written: {rel}\nsha256:  {digest}")
    for lvl in LEVELS:
        h = out[lvl]["headline_fit"]
        for scope in ("all_cells", "unflagged_cells"):
            f = h[scope]; p = f["point"]
            print(f"[{lvl}] headline fit, {scope:<16} n={f['n_cells']:<4} pairs={f['n_pairs_bootstrapped']} "
                  f"slope {p['slope']:+.3f} {f['slope_ci95']} r {p['pearson_r']:+.3f} {f['pearson_r_ci95']} "
                  f"intercept {p['intercept']:+.3f} {f['intercept_ci95']}")
        for scope in ("all_cells", "deployer_layer", "unflagged_all_cells", "unflagged_deployer_layer"):
            a = out[lvl]["prediction_accuracy"][HEADLINE_PRED][scope]
            print(f"[{lvl}] accuracy {HEADLINE_PRED}, {scope:<26} n={a['n_cells']:<4} median |log r/p| "
                  f"{a['median_abs_log_realised_over_predicted']:.4f}  within x1.5 {a['share_within_factor_1_5']:.3f}")
        po = out[lvl]["pair_offsets"]
        for name, r in [*po["per_pair"].items(), *po["pooled"].items()]:
            iv = r.get("partition_band_D156") or r.get("pair_cluster_bootstrap")
            ci = (f"band {iv['band95']}" if "band95" in iv else f"ci {iv['ci95']}") if iv else "no interval"
            a = r["accuracy"]
            print(f"[{lvl}] offset {name:<42} pairs={r['n_pairs']} n={r['n_cells']:<4} flagged={r['input_verification']['n_cells_flagged']:<3} "
                  f"mean {r['mean_log_realised_over_predicted']:+.4f} {ci}  median|e| "
                  f"{a['median_abs_log_realised_over_predicted']:.4f} within x1.5 {a['share_within_factor_1_5']:.3f}")
        for m in po["matched_dataset_contrast"]:
            print(f"[{lvl}] matched WG - Aegis {m['wildguardmix_pair']}: {m['difference_wildguardmix_minus_aegis']:+.4f} "
                  f"band {m['partition_band_D156']['band95']} (aliased with the D-170 flag)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
