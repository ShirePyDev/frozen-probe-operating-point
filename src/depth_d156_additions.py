"""Depth under D-156: additions 1 and 3 of D-191, from the ten results/depth_d156_<pair>.json only
(no re-run; addition 2 is src/depth_d156_twox.py).

  (1) LAYER-LEVEL CORRELATION over the 156 layer cells of the nine pairs of UPDATE_PAIRS_REPAIR: per
      cell, the AUROC drop (drop_frozen, mean over seeds, frozen eval split) against the paired
      ratio (D-156 mean). Spearman (primary, beside D-190's n = 10 Spearman) and Pearson of the
      drop against log paired ratio (secondary). 95% interval by pair-cluster bootstrap (B = 5000,
      seed 520260925, pairs resampled; a resample repeats a pair's cells). Selected C and C=1, item
      level and D-164 cluster level; a sensitivity row adds Mistral v0.1 -> v0.2 (173 cells).
  (3) APPENDIX TABLE, frozen split vs D-156 at the deployer's layer (D-117 selected layer,
      selected C) for the nine pairs: paired ratio, ratio to target, floor FPR_pre/target, TPR pre
      and post, each with frozen value, D-156 mean and band, frozen percentile, and the cluster-level
      mean; the WildGuardMix pairs marked as the median-split controls (D-155, D-159).
Writes results/depth_d156_additions.json (never overwritten).
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
from provenance import ensure_utf8_stdout, provenance_block, sha256_bytes, sha256_file, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/depth_d156_additions.py", "src/paired.py", "src/populations.py", "src/provenance.py",
              "src/superseded_provenance.py"]
LEVELS = ("item_level_D156", "cluster_level_D164")
PROBE_SETS = ("selected", "C=1")
BOOT = {"n_boot": 5000, "boot_seed": 520260925}                        # the 4b pair-cluster bootstrap (D-163)
NINE = [f"depth_{s[len('repair_'):]}" for s in POP.UPDATE_PAIRS_REPAIR]
DENSE = "depth_mistral_v01_v02_dense"
MEDIAN_SPLIT_AXIS = "safety_wildguard"                                # D-155, D-159: the median-split controls
SUMMARY_OF_RECORD_SHA256 = "efabb2181946617d944c10e9deec3251073da28b84476b98df3c7a4bfb3de3b7"   # D-190
TABLE3_QUANT = ("paired_ratio", "ratio_to_target", "floor_ratio_to_target", "tpr_pre", "tpr_post")


def rank(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Average ranks of v where each item counts w times (w integer >= 0): the ranks the expanded
    sample would give its copies, so a weighted Spearman is the Spearman of the resample."""
    order = np.argsort(v, kind="mergesort")
    vs, ws = v[order], w[order].astype(float)
    cum = np.cumsum(ws)
    r = np.empty(len(v))
    i = 0
    while i < len(vs):
        j = i
        while j + 1 < len(vs) and vs[j + 1] == vs[i]:
            j += 1
        lo = cum[i] - ws[i]
        hi = cum[j]
        r[order[i:j + 1]] = (lo + 1 + hi) / 2.0        # the mean rank of the tied block's copies
        i = j + 1
    return r


def wcorr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson correlation (w = multiplicities)."""
    m = w > 0
    x, y, w = x[m], y[m], w[m].astype(float)
    if len(x) < 3:
        return float("nan")
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    sxy = np.sum(w * (x - mx) * (y - my))
    sxx, syy = np.sum(w * (x - mx) ** 2), np.sum(w * (y - my) ** 2)
    return float(sxy / np.sqrt(sxx * syy)) if sxx > 0 and syy > 0 else float("nan")


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Depth additions 1 and 3 (D-191)")
    ap.add_argument("--out", default=str(RESULTS / "depth_d156_additions.json"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"REFUSING: {out} exists; results are never overwritten")
    final = out.resolve() == (RESULTS / "depth_d156_additions.json").resolve()
    if not final and out.resolve().is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test output must not be under results/")
    code0 = P.code_block(CODE_FILES)
    files = {s: RESULTS / f"depth_d156_{s[len('depth_'):]}.json" for s in NINE + [DENSE]}
    raw = {s: f.read_bytes() for s, f in files.items()}
    sha = {s: sha256_bytes(b) for s, b in raw.items()}
    docs = {s: json.loads(b.decode("utf-8")) for s, b in raw.items()}
    summ_path = RESULTS / "depth_d156_summary.json"
    summ_sha = sha256_file(summ_path)
    if final and summ_sha != SUMMARY_OF_RECORD_SHA256:
        raise SystemExit(f"REFUSING the final name: depth_d156_summary.json is not D-190's ({summ_sha[:16]})")
    summ = json.loads(summ_path.read_text(encoding="utf-8"))
    for s, d in docs.items():
        if f"results/{files[s].name}" not in summ["input_files_sha256"] or summ["input_files_sha256"][f"results/{files[s].name}"] != sha[s]:
            raise SystemExit(f"REFUSING: {files[s].name} is not the file depth_d156_summary.json read")

    def at(c: dict, lvl: str) -> dict:
        return c if lvl == LEVELS[0] else c[lvl]

    # (1) layer-level correlation with a pair-cluster bootstrap
    corr: dict[str, Any] = {}
    # the pairs with no flagged cell, beside (D-184's unflagged row; D-170: only rows without flagged cells
    # can enter a paper table before D-168)
    unflagged = [s for s in NINE if not any(c["input_verification"]["flag"] is not None for c in docs[s]["cells"])]
    for label, pairs in (("nine_pairs_156_cells", NINE), ("sensitivity_ten_pairs_with_v01_v02", NINE + [DENSE]),
                         ("unflagged_pairs_of_the_nine", unflagged)):
        corr[label] = {}
        for pset in PROBE_SETS:
            corr[label][pset] = {}
            for lvl in LEVELS:
                rows = [(pairs.index(s), c["auroc"]["drop_frozen"]["mean_over_seeds"], at(c, lvl)["operating_point"]["paired_ratio"]["mean"],
                         c["input_verification"]["flag"] is not None)
                        for s in pairs for c in docs[s]["cells"] if c["probe_set"] == pset]
                pid = np.array([r[0] for r in rows])
                drop, ratio = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
                ones = np.ones(len(rows))
                sp = wcorr(rank(drop, ones), rank(ratio, ones), ones)
                pe = wcorr(drop, np.log(ratio), ones)
                rng = np.random.default_rng(BOOT["boot_seed"])
                idx = rng.integers(0, len(pairs), size=(BOOT["n_boot"], len(pairs)))
                rep_sp, rep_pe = [], []
                for r in idx:
                    w = np.bincount(r, minlength=len(pairs))[pid]
                    rep_sp.append(wcorr(rank(drop, w), rank(ratio, w), w))
                    rep_pe.append(wcorr(drop, np.log(ratio), w))
                corr[label][pset][lvl] = {"n_cells": len(rows), "n_pairs": len(pairs), "pairs": list(pairs),
                                          "n_cells_flagged": int(sum(r[3] for r in rows)),
                                          "spearman_auroc_drop_vs_paired_ratio": {"point": sp, **P.ci(np.array(rep_sp))},
                                          "pearson_auroc_drop_vs_log_paired_ratio": {"point": pe, **P.ci(np.array(rep_pe))}}
    beside = {pset: {lvl: {k: summ["across_pairs"][pset][lvl][k]["spearman_sweep_maximum_paired_ratio_vs_max_auroc_drop"]
                           for k in summ["across_pairs"][pset][lvl]} for lvl in LEVELS} for pset in PROBE_SETS}

    # (3) frozen split vs D-156 at the deployer's layer, nine pairs
    table3 = {}
    for s in NINE:
        d = docs[s]
        c = next(x for x in d["cells"] if x["probe_set"] == "selected" and x["is_selected_layer"])
        table3[s] = {"layer": c["layer"], "axis": d["config"]["axis"],
                     "role": "median-split control (WildGuardMix; D-155, D-159)" if d["config"]["axis"] == MEDIAN_SPLIT_AXIS else "Aegis",
                     "input_verification": {"flag": c["input_verification"]["flag"]},
                     **{q: {"frozen_split": c["operating_point"][q]["frozen"], "d156_mean": c["operating_point"][q]["mean"],
                            "d156_band95": c["operating_point"][q]["band95"],
                            "frozen_percentile": c["operating_point"][q]["frozen_percentile"],
                            "cluster_level_mean": c[LEVELS[1]]["operating_point"][q]["mean"]} for q in TABLE3_QUANT}}

    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"] or any(sha256_file(f) != sha[s] for s, f in files.items()) or sha256_file(summ_path) != summ_sha:
        raise SystemExit("REFUSING to write: code or an input file changed during the run")
    digest = write_results(out, {**provenance_block("depth_d156_additions"), "preregistration": "D-191 (1), (3)", "code": code0,
                                 "input_files_sha256": {f"results/{files[s].name}": h for s, h in sha.items()}
                                 | {"results/depth_d156_summary.json": summ_sha},
                                 "input_verification": {"flag": "input verified by pin only, pending D-168",
                                                        "rule": "rows with flagged cells or pairs enter no paper table before D-168; "
                                                                "each row carries its own count",
                                                        "unflagged_pairs_of_the_nine": unflagged},
                                 "note_pair_cluster_bootstrap": {
                                     "text": "pairs are resampled as independent units although pre models repeat across pairs; "
                                             "as pre-registered, the same as 4b",
                                     "pairs_per_pre_model_of_the_nine": {m: sum(docs[s]["config"]["pre_repo"] == m for s in NINE)
                                                                         for m in sorted({docs[s]["config"]["pre_repo"] for s in NINE})}},
                                 "bootstrap": BOOT,
                                 "addition_1_layer_level_correlation": corr,
                                 "beside_d190_n10_spearman_sweep_max_vs_max_drop": beside,
                                 "addition_3_frozen_split_vs_d156_deployer_layer": table3}, exclusive=True)
    print(f"Written: {out}\nsha256:  {digest}")
    for label in ("nine_pairs_156_cells", "unflagged_pairs_of_the_nine"):
        for pset in PROBE_SETS:
            for lvl in LEVELS:
                c = corr[label][pset][lvl]
                s_, p_ = c["spearman_auroc_drop_vs_paired_ratio"], c["pearson_auroc_drop_vs_log_paired_ratio"]
                print(f"[{label} {pset} {lvl}] {c['n_cells']} cells, {c['n_pairs']} pairs: Spearman {s_['point']:+.3f} "
                      f"{s_['ci95'] and [round(v, 3) for v in s_['ci95']]}; Pearson(log) {p_['point']:+.3f} "
                      f"{p_['ci95'] and [round(v, 3) for v in p_['ci95']]}; flagged cells {c['n_cells_flagged']}; "
                      f"non-finite replicates {s_['n_nonfinite']}/{p_['n_nonfinite']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
