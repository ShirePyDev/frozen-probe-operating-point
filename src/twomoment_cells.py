"""4b per pair, under D-156: realised paired damage and its one- and two-moment predictions,
per layer cell, over 1000 re-partitions of the pooled negatives.

Pre-registered in D-152 (item 4b; P1-P6) and D-156 (the operating-point estimator, which 4b
uses from the start). Run once per released update pair of UPDATE_PAIRS_REPAIR. Writes
results/twomoment_<pair>.json and its per-draw arrays results/twomoment_<pair>_draws.npz (the
npz's sha256 and size are recorded in the JSON); the fits across cells are
src/twomoment_fit.py.

For every layer of the pair's cache grid, for the D-117 SELECTED C (per seed on the D-117 80%
split, permutation order) and for C=1 (same split, file order, library-default tolerance: the
sensitivity row, F-059), five seeds:

  PARTITIONS  the 1000 threshold_calib and 2000 eval negatives are pooled (pre- and post-update
              scores of the same items) and re-partitioned 1000 times at the original sizes by
              paired.repartition_draws with the axis's fixed seed (paired.AXIS_DRAW_SEED). The
              frozen split is evaluated as one more draw (column 0 of every per-draw array).
              The eval positives are fixed. Every quantity is computed twice: over the
              item-level re-partitions (D-156) and over the cluster-level ones of D-164 (no
              near-duplicate cluster of manifests/negative_clusters.json straddles; own seeds;
              npz keys ending '__cluster'; the frozen column is not a cluster-level draw). At the
              selected layer the realised reports at BOTH levels must equal
              results/damage_d156_r2.json's for the same pair and probe set, exactly; the
              selected layer is computed FIRST so a mismatch refuses before the other layers.
  INPUTS      D-165: the pair, revisions and expected cache bytes come from
              src/input_registry.py (it reads the original extraction records as records of what
              was extracted, and the committed pins), and the selection of record is
              results/selection_sweep_d165.json (D-171; a final name refuses any other). No
              superseded RESULT is read as an input. Every cell and the file carry the D-170
              input_verification flag (D-172), and each input names its file, the source of its
              expected bytes and whether it is pin-only. A final name also needs the 6-thread
              BLAS pin (D-152 P6).
  LEVELS      Every D-156 quantity - realised and predicted operating points, the moment bands
              and the reversed-order C=1 check - is reported at the item level and, beside it, at
              the D-164 cluster level ('cluster_level_D164' blocks; see CLUSTER_NOTE).
  REALISED    paired.repartition_estimate: each seed's threshold at 1% on the draw's
              calibration part of the PRE scores; realised FPR pre and post on its evaluation
              part (floored at 1/2000), the paired ratio (mean over seeds of per-seed floored
              ratios), ratio to target, the floor, TPR pre and post at the frozen threshold on
              the positives; exact McNemar on the frozen split with the minimum achievable p,
              and the share of draws with p < 0.05.
  PREDICTED   in each draw, the evaluation part's pre-update negative scores a, mapped and
              re-thresholded at the same per-seed thresholds; predicted ratio = mean over seeds
              of floored FPR(mapped) / floored FPR(a):
                translation  a + (mu_b - mu_a)                                    (D-121 form)
                moment_full  mean(a) + (mu_b - mu_a) + (a - mean(a)) * sd_b / sd_a
                moment_tail  as moment_full with the SD of the upper decile
              IN-SAMPLE: (mu, sd) of the draw's evaluation-part negatives through the pre and
              post models (a decomposition, D-076). OUT-OF-SAMPLE: (mu, sd) of the draw's
              calibration-part negatives (a prediction). Scaling is about the pre negative mean
              (moment matching, D-152 4b). Roles (D-163 (4), PREDICTOR_ROLE): in-sample full SD
              primary, out-of-sample full SD the variant, upper-decile SD the sensitivity,
              translation a comparator.
  REPORTED    every realised and predicted operating-point quantity as the mean over the draws
              and its 2.5-97.5% band, the frozen split as one draw with its percentile (D-156,
              paired.band_summary); per-seed flag counts k/n on the frozen split (P1, P5).
              Censored draws are floored at the resolution 1/n_neg and kept (D-163 (2)). Per
              cell and predictor, 'fraction_of_draws_floored' is the share of the draws in which
              that prediction is defined where any seed's realised pre or post FPR, or its
              predicted FPR, was 0 before flooring (the definition twomoment_fit uses);
              realised_censoring gives the realised-only share over all draws. A prediction
              whose pre SD is 0 (below SD_EPS relative to the mean) is undefined in that draw
              and counted. Per-draw arrays also carry the moment changes and, on C=1 rows, the
              reversed-order paired ratio.
  C=1 CHECK   at EVERY layer the C=1 probes are refitted in reversed file order (D-163 (3)):
              cos(w_file, w_reversed) per seed and the per-draw paired-ratio difference
              (D-152 P2, F-059).

Moments are reported with their D-156 band across draws (D-163 (5); no item bootstrap): the
mean shift and log SD ratios per draw, with the frozen split's four parts per seed and the
pooled negatives (all 3000, partition-free) as description. Supersedes results/mechanism_selected.json (D-152 P4).

Only the registered parameters (1000 draws, 1% target, seeds 0-4) may write the final names;
any other values need --out elsewhere (a test run must not take a final name).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import input_registry as REG  # noqa: E402
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
from provenance import (  # noqa: E402
    commit_staged, ensure_utf8_stdout, provenance_block, sha256_bytes, sha256_file, stage_bytes,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/twomoment_cells.py", "src/input_registry.py", "src/paired.py", "src/populations.py",
              "src/provenance.py", "src/superseded_provenance.py"]
D156_R2 = RESULTS / "damage_d156_r2.json"          # the D-156 run of record for the cross-check (D-173)
D156_R2_FPR5 = RESULTS / "damage_d156_r2_fpr5.json"   # 4e (D-248): the 5% damage file, the cross-check at 5%
TWOMOMENT_FIT = RESULTS / "twomoment_fit.json"      # names the 1% twomoment records of record (D-174)
TWOMOMENT_FIT_SHA256 = "515faab949465d0852cd61dbc1c7805cf3d800c5e2222029e4622261727a3b96"
TARGET_5 = 0.05
CLUSTERS = ROOT / "manifests" / "negative_clusters.json"
SELECTION_OF_RECORD = RESULTS / "selection_sweep_d165.json"   # D-171/D-172 (1)
LEVELS = ("item_level_D156", "cluster_level_D164")
CLUSTER_NOTE = ("cluster-level re-partitions (D-164), reported beside the item level, never instead. Their "
                "'frozen' value is the frozen split, which is NOT a cluster-level draw (it splits 36 of 67 Aegis "
                "and 7 of 11 WildGuardMix clusters), so a frozen_percentile here places a split this distribution "
                "cannot produce; the two levels use independent seeds, so their means differ by Monte Carlo noise "
                "of about sqrt(2) * sd_over_draws / sqrt(n_draws)")
REGISTERED = {"n_draws": P.N_DRAWS, "target_fpr": P.TARGET_FPR, "seeds": [0, 1, 2, 3, 4]}
TAIL_Q = 0.90
SD_EPS = 1e-12
PREDICTORS = ("translation", "moment_full", "moment_tail")
SOURCES = ("in_sample", "out_of_sample")
PRED_KEYS = tuple(f"{p}_{s}" for s in SOURCES for p in PREDICTORS)
# Roles fixed by the author in D-163 (4), reading D-152 item 4b.
PREDICTOR_ROLE = {
    "moment_full_in_sample": "PRIMARY (D-163): full-SD moment matching, in-sample - the mechanism decomposition (D-076)",
    "moment_full_out_of_sample": "VARIANT (D-163): full-SD moment matching, out-of-sample - the health-check basis",
    "moment_tail_in_sample": "SENSITIVITY (D-163): upper-decile SD, in-sample",
    "moment_tail_out_of_sample": "SENSITIVITY (D-163): upper-decile SD, out-of-sample",
    "translation_in_sample": "COMPARATOR (D-163): one-moment translation (D-121 form), in-sample",
    "translation_out_of_sample": "COMPARATOR (D-163): one-moment translation (D-121 form), out-of-sample",
}
D156_COMPARED = ("frozen_split_significance", "share_of_draws_mcnemar_neg_p_max_below_0_05",
                 "share_of_draws_mcnemar_pos_p_max_below_0_05", "frozen_split_counts", "n_draws")


def moments(x: np.ndarray) -> dict[str, np.ndarray]:
    """Per-seed mean, SD and upper-decile SD (population SDs) of scores x (items, seeds)."""
    q = np.quantile(x, TAIL_Q, axis=0)
    in_tail = x >= q
    n_tail = in_tail.sum(axis=0)
    tail_mean = (x * in_tail).sum(axis=0) / n_tail
    sd_tail = np.sqrt((((x - tail_mean) ** 2) * in_tail).sum(axis=0) / n_tail)
    return {"mean": x.mean(axis=0), "sd": x.std(axis=0), "sd_tail": np.where(n_tail > 1, sd_tail, np.nan)}


def mapped(a: np.ndarray, pre_m: dict, post_m: dict, predictor: str) -> np.ndarray:
    """The pre-update negative scores a (items, seeds) moved by the moment change pre_m -> post_m:
    translated by the mean shift, or moment-matched about a's own mean (D-152 4b). A pre SD at or
    below SD_EPS relative to the mean is treated as 0 (a constant column's SD comes out as float
    noise, not 0), and the SD ratio is then undefined (NaN)."""
    shift = post_m["mean"] - pre_m["mean"]
    if predictor == "translation":
        return a + shift
    k = sd_ratio(pre_m, post_m, "sd" if predictor == "moment_full" else "sd_tail")
    mu_a = a.mean(axis=0)
    return mu_a + shift + (a - mu_a) * k


def sd_ratio(pre_m: dict, post_m: dict, key: str) -> np.ndarray:
    """Per-seed post/pre SD ratio; NaN where the pre SD is 0 (at or below SD_EPS relative to the
    mean: a constant column's SD comes out as float noise, not 0)."""
    sd_pre = pre_m[key]
    zero = sd_pre <= SD_EPS * np.maximum(1.0, np.abs(pre_m["mean"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(zero, np.nan, post_m[key] / np.where(zero, 1.0, sd_pre))


def log_sd_ratio(pre_m: dict, post_m: dict, key: str) -> np.ndarray:
    """log of sd_ratio, NaN where it is undefined or not positive."""
    r = sd_ratio(pre_m, post_m, key)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(r) & (r > 0), np.log(np.where(r > 0, r, 1.0)), np.nan)


def one_partition(pool_pre: np.ndarray, pool_post: np.ndarray, cal: np.ndarray, ev: np.ndarray,
                  target: float) -> dict[str, Any]:
    """Realised flags and every prediction for ONE partition, at the per-seed thresholds that
    paired.operating_point sets for the same partition (same function on the same scores; the
    caller checks the realised ratio against the shared estimator's, draw by draw)."""
    T = np.array([P.thr_at_fpr(pool_pre[cal, s], target) for s in range(pool_pre.shape[1])])
    A, B = pool_pre[ev], pool_post[ev]
    F_pre, F_post = A >= T, B >= T
    zr = (P.rate(F_pre) == 0) | (P.rate(F_post) == 0)
    src = {"in_sample": (moments(A), moments(B)),
           "out_of_sample": (moments(pool_pre[cal]), moments(pool_post[cal]))}
    out: dict[str, Any] = {"ratio": P.ratio_from_flags(F_pre, F_post),
                           "fpr_post": float(P.floored_fpr(F_post).mean()),
                           "n_censored_seeds": int(zr.sum()),
                           "n_flagged_pre": F_pre.sum(axis=0).tolist(), "n_flagged_post": F_post.sum(axis=0).tolist(),
                           "moments": src}
    out["moment_change"] = {s: {"mean_shift": float((qm["mean"] - pm["mean"]).mean()),
                                "log_sd_ratio": float(log_sd_ratio(pm, qm, "sd").mean()),
                                "log_sd_tail_ratio": float(log_sd_ratio(pm, qm, "sd_tail").mean())}
                            for s, (pm, qm) in src.items()}
    for s, (pm, qm) in src.items():
        for pr in PREDICTORS:
            M = mapped(A, pm, qm, pr)
            if not np.isfinite(M).all():
                out[f"{pr}_{s}"] = None
                continue
            F_pred = M >= T
            zp = P.rate(F_pred) == 0
            out[f"{pr}_{s}"] = {"ratio": P.ratio_from_flags(F_pre, F_pred),
                                "ratio_per_seed": P.per_seed_ratio(F_pre, F_pred).tolist(),
                                "fpr_pred": float(P.floored_fpr(F_pred).mean()),
                                "n_flagged": F_pred.sum(axis=0).tolist(),
                                "n_censored_seeds": int(zp.sum())}
    return out


def diff_from_d156(report: dict[str, Any], ref: dict[str, Any]) -> list[str]:
    """Keys of a realised report that differ from results/damage_d156_r2.json's report for the
    same cell and level. Both come from the same paired.repartition_estimate on the same probes,
    partitions and inputs, so every value must be identical."""
    diffs = [m for m in P.D156_METRICS if report[m] != ref[m]]
    return diffs + [k for k in D156_COMPARED if report[k] != ref[k]]


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="4b per pair under D-156: realised and predicted paired damage")
    ap.add_argument("--pair", required=True, choices=POP.UPDATE_PAIRS_REPAIR)
    ap.add_argument("--selection", default=str(RESULTS / "selection_sweep_d165.json"))   # of record (D-171)
    ap.add_argument("--seeds", type=int, nargs="*", default=REGISTERED["seeds"])
    ap.add_argument("--n-draws", type=int, default=REGISTERED["n_draws"])
    ap.add_argument("--target-fpr", type=float, default=None)
    ap.add_argument("--fpr5", action="store_true",
                    help="4e (D-152, D-248): the registered 5%% arm, written as twomoment_<pair>_fpr5; every cell is also "
                         "computed at 1%% and must equal the 1%% record exactly (the self-anchor)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.fpr5 and args.target_fpr not in (None, TARGET_5):
        raise SystemExit("REFUSING: --fpr5 fixes the target at 0.05")
    args.target_fpr = TARGET_5 if args.fpr5 else (REGISTERED["target_fpr"] if args.target_fpr is None else args.target_fpr)
    stem = args.pair
    registered = (args.n_draws == REGISTERED["n_draws"] and list(args.seeds) == REGISTERED["seeds"]
                  and args.target_fpr == (TARGET_5 if args.fpr5 else REGISTERED["target_fpr"]))
    final_json = (RESULTS / f"twomoment_{stem}{'_fpr5' if args.fpr5 else ''}.json").resolve()
    rec1_cells: dict[tuple[int, str], dict[str, Any]] = {}
    rec1_info: dict[str, Any] = {}
    if args.fpr5:
        # D-248 self-anchor: the 1% record of record, named by the pinned 4b fit
        fb = TWOMOMENT_FIT.read_bytes()
        if sha256_bytes(fb) != TWOMOMENT_FIT_SHA256:
            raise SystemExit(f"REFUSING: {TWOMOMENT_FIT.name} is not the fit of record")
        rel1 = f"results/twomoment_{stem}.json"
        want1 = json.loads(fb.decode("utf-8"))["input_files_sha256"].get(rel1)
        rb = (ROOT / rel1).read_bytes()
        if want1 is None or sha256_bytes(rb) != want1:
            raise SystemExit(f"REFUSING: {rel1} is not the 1% record the fit of record names")
        rec1_cells = {(c["layer"], c["probe_set"]): c for c in json.loads(rb.decode("utf-8"))["cells"]}
        rec1_info = {"file": rel1, "sha256": want1, "named_by": f"results/{TWOMOMENT_FIT.name} ({TWOMOMENT_FIT_SHA256[:16]}...)"}
    n_self_anchor = 0
    out_path = Path(args.out).resolve() if args.out else final_json
    npz_path = out_path.with_name(f"{out_path.stem}_draws.npz")
    if out_path.parent == final_json.parent and out_path.name.startswith("twomoment_") and \
            (not registered or out_path != final_json):
        raise SystemExit("REFUSING: only the registered parameters of this mode may write a twomoment_* name in results/, "
                         "and only this mode's own final name (use --out elsewhere)")
    if npz_path.exists() and not out_path.exists():
        raise SystemExit(f"REFUSING: {npz_path} exists without {out_path.name}: the orphan of a failed write. "
                         f"Move it aside, never delete it, to {npz_path.stem}.orphan-<utc>.npz (ignored by git, "
                         f"not matched by *_draws.npz), and rerun.")
    if out_path.exists():
        raise SystemExit(f"REFUSING: {out_path} exists; results are never overwritten (D-152 P5)")
    sel_path = Path(args.selection).resolve()
    final = out_path == final_json
    if final and sel_path != SELECTION_OF_RECORD.resolve():
        raise SystemExit(f"REFUSING: a final name needs the selection of record {SELECTION_OF_RECORD.name} (D-171); "
                         f"got {sel_path.name}")
    import os
    threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    if final and any(t != "6" for t in threads.values()):
        raise SystemExit(f"REFUSING: a final name needs the 6-thread BLAS pin (D-152 P6, D-172 (5)); got {threads}. "
                         f"Launch through scripts/detach.sh, which sources scripts/project_env.sh")
    sel_sha = sha256_file(sel_path)
    cl_sha = sha256_file(CLUSTERS)
    code0 = P.code_block(CODE_FILES)

    sel_doc = json.load(open(sel_path, encoding="utf-8"))
    if list(sel_doc["config"]["seeds"]) != list(args.seeds) or float(sel_doc["config"]["holdout"]) != P.HOLDOUT:
        raise SystemExit("REFUSING: selection seeds/holdout differ from this run's")
    sel = next((m for m in sel_doc["models"] if m["pair"] == stem), None)
    if sel is None or not sel["selected"]["arm"].startswith("C="):
        raise SystemExit(f"REFUSING: no logistic-C selection for {stem} in {sel_path.name}")
    C_sel, L_sel = float(sel["selected"]["arm"].split("=")[1]), int(sel["selected"]["layer"])

    # The D-156 cross-check: this pair's selected-layer cells in results/damage_d156_r2.json, at
    # BOTH levels (item-level and the D-164 cluster level).
    d156_ref: dict[str, dict] = {}
    d156_sha = None
    d156_path = D156_R2_FPR5 if args.fpr5 else D156_R2       # D-248: at 5% the cross-check is the 5% damage file
    if registered:
        d156_sha = sha256_file(d156_path)
        d156_doc = json.load(open(d156_path, encoding="utf-8"))
        if args.fpr5 and (d156_doc["config"]["target_fpr"] != TARGET_5
                          or (d156_doc.get("fpr5_self_anchor") or {}).get("n_cells_identical_at_1pct") != len(d156_doc["cells"])):
            raise SystemExit(f"REFUSING: {d156_path.name} is not a complete, self-anchored 5% damage file (D-248)")
        for c in d156_doc["cells"]:
            if c["scope"] == "nine-pair table" and c["cell"] == stem:
                if c["layer"] != L_sel or (c["probe_set"] == "selected" and c["probe_C"] != C_sel):
                    raise SystemExit(f"REFUSING: {d156_path.name}'s {stem} cell is not this selection")
                d156_ref[c["probe_set"]] = c
        if set(d156_ref) != {"selected", "C=1"}:
            raise SystemExit(f"REFUSING: {d156_path.name} lacks {stem}'s selected and C=1 cells")

    cfg = REG.run(stem)                  # D-165: declared run, revisions from extraction records
    axis = cfg["axis"]
    if axis not in P.AXIS_DRAW_SEED:
        raise SystemExit(f"REFUSING: no D-156 partition seed for axis {axis}")
    needed = [("pre", "probe_train"), ("pre", "threshold_calib"), ("pre", "eval"),
              ("post", "threshold_calib"), ("post", "eval")]      # post/probe_train is never read
    data: dict[str, dict[str, Any]] = {"pre": {}, "post": {}}
    input_sha: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    for side, part in needed:
        z, r = REG.load(cfg[f"{side}_repo"], axis, part)          # expected bytes checked there
        data[side][part] = z
        input_sha[f"{side}/{part}"] = r["sha256"]
        input_recs[f"{side}/{part}"] = r
    checks = [(f"selection {k}", input_sha[k], sel_doc["input_cache_sha256"].get(f"{stem}:{k}"))
              for k in ("pre/probe_train", "pre/threshold_calib")]
    bad = [f"{w}: have {h[:16]}, expected {(e or 'NOT RECORDED')[:16]}" for w, h, e in checks if h != e]
    if bad:
        raise SystemExit("REFUSING: the selection was made on other bytes:\n  " + "\n  ".join(bad))
    verification = REG.verification(input_recs)
    for part in ("threshold_calib", "eval"):
        if not np.array_equal(data["pre"][part]["item_ids"], data["post"][part]["item_ids"]) or \
                not np.array_equal(data["pre"][part]["labels"], data["post"][part]["labels"]):
            raise SystemExit(f"REFUSING: {part} items or labels differ between pre and post")
    for side in data:
        for part, z in data[side].items():
            if "nonfinite" in z.files and z["nonfinite"].any():
                raise SystemExit(f"REFUSING: {side}/{part} has non-finite items")
    grids = [sorted(int(k.split("_")[1]) for k in z.files if k.startswith("layer_"))
             for side in data for z in data[side].values()]
    if any(g != grids[0] for g in grids):
        raise SystemExit(f"REFUSING: the input caches have different layer grids: {grids}")
    layers = grids[0]
    if L_sel not in layers:
        raise SystemExit(f"REFUSING: selected layer {L_sel} is not in the caches' grid")

    y_tr = data["pre"]["probe_train"]["labels"].astype(bool)
    ncal = ~data["pre"]["threshold_calib"]["labels"].astype(bool)
    y_ev = data["pre"]["eval"]["labels"].astype(bool)
    neg, pos = ~y_ev, y_ev
    pooled_ids = [str(i) for i in data["pre"]["threshold_calib"]["item_ids"][ncal]] + \
                 [str(i) for i in data["pre"]["eval"]["item_ids"][neg]]
    n_cal, n_pool = int(ncal.sum()), len(pooled_ids)
    frozen_part = (np.arange(n_cal), np.arange(n_cal, n_pool))
    draws = {LEVELS[0]: P.repartition_draws(n_cal, n_pool - n_cal, args.n_draws, P.AXIS_DRAW_SEED[axis])}
    cl = json.load(open(CLUSTERS, encoding="utf-8"))["axes"][axis]
    if cl["n_pooled_negatives"] != n_pool or cl["n_calibration"] != n_cal or \
            cl["pool_item_ids_sha256"] != sha256_bytes("\n".join(pooled_ids).encode("utf-8")):
        raise SystemExit(f"REFUSING: manifests/negative_clusters.json does not describe the {axis} pool")
    draws[LEVELS[1]] = P.cluster_repartition_draws(P.cluster_units(pooled_ids, cl["clusters"]), n_cal,
                                                   args.n_draws, P.AXIS_CLUSTER_DRAW_SEED[axis])
    parts = {lvl: [frozen_part] + [(d[:n_cal], d[n_cal:]) for d in dr] for lvl, dr in draws.items()}
    print(f"{stem} [{axis}]: selected C={C_sel:g} at layer {L_sel}; {len(layers)} layers; pooled negatives "
          f"{n_cal} + {n_pool - n_cal}, {int(pos.sum())} positives; {args.n_draws} draws per level; "
          f"inputs from the registry, {len(checks)} selection checks passed; D-170 flag: {verification['flag']}; "
          f"registered parameters: {registered}", flush=True)

    done: dict[tuple[int, str], tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
    d156_check: dict[str, Any] = {"checked": registered, "damage_d156_file": f"results/{d156_path.name}",
                                  "damage_d156_sha256": d156_sha}
    for li in [L_sel] + [x for x in layers if x != L_sel]:
        k = f"layer_{li}"
        X = {s: {p: data[s][p][k] for p in ("threshold_calib", "eval")} for s in data}
        X_tr = data["pre"]["probe_train"][k]
        sets: dict[str, list[P.Probe]] = {"selected": [], "C=1": []}
        c1_rev: list[P.Probe] = []
        for seed in args.seeds:
            _, fit = P.d117_split(len(y_tr), seed)
            sets["selected"].append(P.fit_probe(X_tr[fit], y_tr[fit], seed, C_sel, "d117_permutation"))
            fo = np.sort(fit)
            sets["C=1"].append(P.fit_probe(X_tr[fo], y_tr[fo], seed, 1.0, "file_order"))
            c1_rev.append(P.fit_probe(X_tr[fo[::-1]], y_tr[fo[::-1]], seed, 1.0, "reversed_file_order"))

        def scores(probes: list[P.Probe]) -> dict[str, np.ndarray]:
            S = lambda side, part, m: np.column_stack([q.score(X[side][part][m]) for q in probes])  # noqa: E731
            return {"cal_pre": S("pre", "threshold_calib", ncal), "ev_pre": S("pre", "eval", neg),
                    "cal_post": S("post", "threshold_calib", ncal), "ev_post": S("post", "eval", neg),
                    "pos_pre": S("pre", "eval", pos), "pos_post": S("post", "eval", pos)}

        def estimate(sc: dict[str, np.ndarray], lvl: str = LEVELS[0]) -> tuple[dict, dict]:
            return P.repartition_estimate(sc["cal_pre"], sc["ev_pre"], sc["cal_post"], sc["ev_post"],
                                          sc["pos_pre"], sc["pos_post"], draws[lvl], args.target_fpr)

        def per_level(sc: dict[str, np.ndarray], report: dict, arr: dict, lvl: str, where: str):
            """Per-draw realised and predicted values for one level (column 0 = the frozen split),
            checked draw by draw against paired.repartition_estimate on the same partitions."""
            pool_pre = np.vstack([sc["cal_pre"], sc["ev_pre"]])
            pool_post = np.vstack([sc["cal_post"], sc["ev_post"]])
            per = [one_partition(pool_pre, pool_post, c, e, args.target_fpr) for c, e in parts[lvl]]
            real = np.array([x["ratio"] for x in per])
            fpost = np.array([x["fpr_post"] for x in per])
            if not (np.array_equal(real[1:], arr["paired_ratio"]) and real[0] == report["paired_ratio"]["frozen"]
                    and np.array_equal(fpost[1:], arr["fpr_post"]) and fpost[0] == report["fpr_post"]["frozen"]):
                raise SystemExit(f"REFUSING: {where} {lvl}: realised values differ from paired.repartition_estimate")
            rcens = np.array([x["n_censored_seeds"] > 0 for x in per])
            cols: dict[str, np.ndarray] = {"log_real_ratio": np.log(real), "log_fpr_post": np.log(fpost),
                                           "real_censored": rcens}
            preds: dict[str, Any] = {}
            for key in PRED_KEYS:
                ok = np.array([x[key] is not None for x in per])
                ratio = np.array([x[key]["ratio"] if x[key] else np.nan for x in per])
                fpred = np.array([x[key]["fpr_pred"] if x[key] else np.nan for x in per])
                pc = np.array([bool(x[key] and x[key]["n_censored_seeds"] > 0) for x in per])
                cols[f"log_pred_ratio__{key}"] = np.log(ratio)
                cols[f"log_fpr_pred__{key}"] = np.log(fpred)
                cols[f"pred_censored__{key}"] = pc
                fz = per[0][key]
                union = (rcens | pc)[1:][ok[1:]]
                preds[key] = {"role": PREDICTOR_ROLE[key],
                              "ratio": P.band_summary(ratio[1:], ratio[0]),
                              "fpr_pred": P.band_summary(fpred[1:], fpred[0]),
                              # D-163 (2), the fit's definition: a draw is floored when the realised
                              # OR the predicted FPR of any seed was 0; over the defined draws.
                              "fraction_of_draws_floored": float(union.mean()) if union.size else None,
                              "n_draws_undefined": int((~ok[1:]).sum()),
                              "frozen_split": ({"ratio_per_seed": fz["ratio_per_seed"], "n_flagged_per_seed": fz["n_flagged"],
                                                "n_eval_negatives": n_pool - n_cal, "n_censored_seeds": fz["n_censored_seeds"]}
                                               if fz else "undefined")}
            return per, cols, preds, rcens, pool_pre, pool_post

        for pset, probes in sets.items():
            sc = scores(probes)
            report, arr = estimate(sc, LEVELS[0])
            report_c, arr_c = estimate(sc, LEVELS[1])
            if registered and li == L_sel:
                diffs = diff_from_d156(report, d156_ref[pset])
                diffs_c = diff_from_d156(report_c, d156_ref[pset][LEVELS[1]])
                d156_check[pset] = {"identical": not diffs, "differing_keys": diffs,
                                    "cluster_level_identical": not diffs_c, "cluster_level_differing_keys": diffs_c}
                if diffs or diffs_c:
                    raise SystemExit(f"REFUSING: layer {li} {pset} realised report differs from results/{d156_path.name} "
                                     f"(item level {diffs}, cluster level {diffs_c}): the partitions, probes or "
                                     f"environment are not the ones D-156 used")
            if args.fpr5 and registered:
                # D-248 self-anchor: this cell's realised report at 1%, both levels, equals the 1% record's exactly
                rec = rec1_cells.get((li, pset))
                if rec is None:
                    raise SystemExit(f"REFUSING: {rec1_info['file']} has no cell L{li} {pset}")
                for lvl, want in ((LEVELS[0], rec["realised"]), (LEVELS[1], rec[LEVELS[1]]["realised"])):
                    rep1, _ = P.repartition_estimate(sc["cal_pre"], sc["ev_pre"], sc["cal_post"], sc["ev_post"],
                                                     sc["pos_pre"], sc["pos_post"], draws[lvl], P.TARGET_FPR)
                    if json.dumps(json.loads(json.dumps(rep1, default=str)), sort_keys=True) != json.dumps(want, sort_keys=True):
                        raise SystemExit(f"REFUSING: layer {li} {pset} {lvl}: the 1% twin is not {rec1_info['file']}'s (D-248)")
                n_self_anchor += 1
            per, cols, preds, rcens, pool_pre, pool_post = per_level(sc, report, arr, LEVELS[0], f"layer {li} {pset}")
            per_c, cols_c, preds_c, rcens_c, _, _ = per_level(sc, report_c, arr_c, LEVELS[1], f"layer {li} {pset}")
            for key, v in cols_c.items():                     # cluster-level per-draw arrays, suffixed
                cols[f"{key}__cluster"] = v
            mc, mc_c = {}, {}
            for s in SOURCES:
                mc[s], mc_c[s] = {}, {}
                for m in ("mean_shift", "log_sd_ratio", "log_sd_tail_ratio"):
                    v = np.array([x["moment_change"][s][m] for x in per])       # column 0 = frozen
                    cols[f"moment_change__{s}__{m}"] = v
                    mc[s][m] = P.band_summary(v[1:], v[0])
                    vc = np.array([x["moment_change"][s][m] for x in per_c])    # cluster level (D-164)
                    cols[f"moment_change__{s}__{m}__cluster"] = vc
                    mc_c[s][m] = P.band_summary(vc[1:], vc[0])
            fm = per[0]["moments"]
            mom = {"note": "D-163 (5): change_over_draws carries the D-156 band; no item bootstrap",
                   "pooled_negatives_pre": {m: v.tolist() for m, v in moments(pool_pre).items()},
                   "pooled_negatives_post": {m: v.tolist() for m, v in moments(pool_post).items()},
                   "frozen_split_per_seed": {f"{part}_{side}": {m: v.tolist() for m, v in fm[s][i].items()}
                                             for s, part in (("in_sample", "eval_part"), ("out_of_sample", "calibration_part"))
                                             for i, side in enumerate(("pre", "post"))},
                   "change_over_draws": mc}
            cell: dict[str, Any] = {
                "pair": stem, "axis": axis, "layer": li, "probe_set": pset,
                "probe_C": C_sel if pset == "selected" else 1.0,
                "fit_row_order": "d117_permutation" if pset == "selected" else "file_order",
                "is_selected_layer": li == L_sel, "input_verification": verification, "realised": report,
                "realised_censoring": {"share_of_draws_floored": float(rcens[1:].mean()),
                                       "frozen_n_censored_seeds": per[0]["n_censored_seeds"]},
                "predicted": preds,
                LEVELS[1]: {"note": CLUSTER_NOTE,
                            "realised": report_c,
                            "realised_censoring": {"share_of_draws_floored": float(rcens_c[1:].mean())},
                            "predicted": {k: {kk: v[kk] for kk in ("ratio", "fpr_pred", "fraction_of_draws_floored",
                                                                  "n_draws_undefined")}
                                          for k, v in preds_c.items()},
                            "moments_change_over_draws": mc_c},
                "moments": mom, "probe_fit_diagnostics": [q.diag for q in probes]}
            n_cols = len(parts[LEVELS[0]])
            cols["log_real_ratio_reversed"] = np.full(n_cols, np.nan)              # C=1 rows only
            cols["log_real_ratio_reversed__cluster"] = np.full(n_cols, np.nan)
            if pset == "C=1":
                sc_rev = scores(c1_rev)
                rep_r, arr_r = estimate(sc_rev, LEVELS[0])
                rep_rc, arr_rc = estimate(sc_rev, LEVELS[1])                         # D-164 beside
                cols["log_real_ratio_reversed"] = np.log(np.concatenate([[rep_r["paired_ratio"]["frozen"]],
                                                                         arr_r["paired_ratio"]]))
                cols["log_real_ratio_reversed__cluster"] = np.log(np.concatenate([[rep_rc["paired_ratio"]["frozen"]],
                                                                                  arr_rc["paired_ratio"]]))
                cosw = []
                for a, b in zip(probes, c1_rev):
                    na, nb = np.linalg.norm(a.w), np.linalg.norm(b.w)
                    cosw.append(float(a.w @ b.w / (na * nb)) if na > 0 and nb > 0 else None)
                cell["reversed_file_order_refit"] = {
                    "paired_ratio": rep_r["paired_ratio"],
                    "paired_ratio_file_minus_reversed": P.band_summary(
                        arr["paired_ratio"] - arr_r["paired_ratio"],
                        report["paired_ratio"]["frozen"] - rep_r["paired_ratio"]["frozen"]),
                    "cos_w_file_vs_reversed_per_seed": cosw,
                    "probe_fit_diagnostics": [q.diag for q in c1_rev]}
                cell[LEVELS[1]]["reversed_file_order_refit"] = {
                    "paired_ratio": rep_rc["paired_ratio"],
                    "paired_ratio_file_minus_reversed": P.band_summary(
                        arr_c["paired_ratio"] - arr_rc["paired_ratio"],
                        report_c["paired_ratio"]["frozen"] - rep_rc["paired_ratio"]["frozen"])}
            done[(li, pset)] = (cell, cols)
        c = done[(li, "selected")][0]
        r, rc = c["realised"]["paired_ratio"], c[LEVELS[1]]["realised"]["paired_ratio"]
        pin, pout = (c["predicted"][f"moment_full_{s}"]["ratio"]["mean"] for s in SOURCES)
        pin_c = c[LEVELS[1]]["predicted"]["moment_full_in_sample"]["ratio"]["mean"]
        tin = c["predicted"]["translation_in_sample"]["ratio"]["mean"]
        f3 = lambda v: "   undef" if v is None else f"{v:8.3f}"  # noqa: E731
        print(f"  layer {li:>2} [selected C]: realised {r['mean']:8.3f} [{r['band95'][0]:.3f}, {r['band95'][1]:.3f}] "
              f"| moment_full in {f3(pin)} oos {f3(pout)} | translation in {f3(tin)} "
              f"|| cluster-level realised {rc['mean']:8.3f} moment_full in {f3(pin_c)}", flush=True)

    order = [(li, pset) for li in layers for pset in ("selected", "C=1")]
    cells = [done[o][0] for o in order]
    arrays = {k: np.vstack([done[o][1][k] for o in order]) for k in done[order[0]][1]}
    arrays["cell_layer"] = np.array([li for li, _ in order], dtype=np.int64)
    arrays["cell_probe_set"] = np.array([pset for _, pset in order])
    arrays["cell_is_selected_layer"] = np.array([li == L_sel for li, _ in order])

    if args.fpr5 and registered and n_self_anchor != len(order):
        raise SystemExit(f"REFUSING to write: {n_self_anchor} self-anchored cells of {len(order)} (D-248)")
    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"] or sha256_file(sel_path) != sel_sha or sha256_file(CLUSTERS) != cl_sha \
            or (registered and sha256_file(d156_path) != d156_sha):
        raise SystemExit(f"REFUSING to write: source, selection, cluster manifest or {d156_path.name} changed during the run")
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    npz_bytes = buf.getvalue()
    rel = lambda p: str(p.relative_to(ROOT)).replace("\\", "/") if p.is_relative_to(ROOT) else str(p)  # noqa: E731
    payload = {**provenance_block("twomoment_cells"),
               "preregistration": "D-152 (4b), D-156, D-163, D-164, D-165" + ("; 4e: D-152, D-248 (the 5% arm)" if args.fpr5 else ""),
               **({"fpr5_self_anchor": {**rec1_info, "n_cells_identical_at_1pct": n_self_anchor,
                                        "kind": "every cell's realised report recomputed at 1% equals the 1% record at both "
                                                "levels (D-248)"}} if args.fpr5 else {}),
               "code": code0, "registered_parameters": registered, "input_verification": verification,
               "config": {"pair": stem, "axis": axis, "selection_file": rel(sel_path),
                          "selection_sha256": sel_sha, "selected_C": C_sel, "selected_layer": L_sel,
                          "layers": layers, "seeds": args.seeds, "split": f"D-117 (holdout {P.HOLDOUT})",
                          "target_fpr": args.target_fpr, "tail_quantile": TAIL_Q, "sd_eps": SD_EPS,
                          "predictors": list(PREDICTORS), "moment_sources": list(SOURCES),
                          "predictor_roles": PREDICTOR_ROLE,
                          "pre_repo": cfg["pre_repo"], "pre_rev": cfg["pre_rev"],
                          "post_repo": cfg["post_repo"], "post_rev": cfg["post_rev"],
                          "inputs_from": "src/input_registry.py (D-165)",
                          "input_registry_sources_sha256": REG.sources_sha256(),
                          "cluster_manifest": {"file": "manifests/negative_clusters.json", "sha256": cl_sha},
                          "cluster_draw_seed": P.AXIS_CLUSTER_DRAW_SEED[axis],
                          "levels": {LEVELS[0]: "item-level re-partitions (D-156); npz keys without suffix",
                                     LEVELS[1]: "cluster-level re-partitions (D-164); npz keys ending '__cluster'; "
                                                "column 0 is the frozen split, which is not a cluster-level draw"},
                          "n_eval_positives": int(pos.sum()),
                          "pooled_negatives": {"n_calibration": n_cal, "n_evaluation": n_pool - n_cal,
                                               "item_ids_sha256": hashlib.sha256("\n".join(pooled_ids).encode()).hexdigest(),
                                               "order": "threshold_calib negatives then eval negatives, cache order"},
                          "n_draws": args.n_draws, "draw_seed": P.AXIS_DRAW_SEED[axis],
                          "positives": "eval positives, fixed"},
               "damage_d156_check": d156_check,
               "population": {"declared_in": "src/populations.py", "group": "UPDATE_PAIRS_REPAIR", "pair": stem},
               "supersedes": [] if args.fpr5 else ["results/mechanism_selected.json"],
               "input_cache_sha256": input_sha,
               "input_cache_source": {k: {"file": r["file"], "expected_from": r["expected_from"],
                                          "pin_only": r["pin_only"]} for k, r in input_recs.items()},
               "draw_arrays": {"file": rel(npz_path), "sha256": sha256_bytes(npz_bytes), "bytes": len(npz_bytes),
                               "column_0": "the frozen split; columns 1..n_draws: the re-partitions in draw order",
                               "rows": "cells in the order of 'cells' below", "keys": sorted(arrays)},
               "cells": cells}
    json_bytes = json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8")
    json.loads(json_bytes.decode("utf-8"))
    # Stage BOTH outputs (written, fsynced, verified) before either final name exists; then link.
    staged: list[Path] = []
    try:
        t_npz, h_npz = stage_bytes(npz_path, npz_bytes)
        staged.append(t_npz)
        t_json, h_json = stage_bytes(out_path, json_bytes)
        staged.append(t_json)
        commit_staged(t_npz, npz_path, h_npz)
        commit_staged(t_json, out_path, h_json)
    finally:
        for t in staged:
            if t.exists():
                t.unlink()
    print(f"Written: {rel(npz_path)}\nsha256:  {h_npz}\nWritten: {rel(out_path)}\nsha256:  {h_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
