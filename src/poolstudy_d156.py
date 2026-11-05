"""4d part (B), the pool study, under D-156 (D-152 4d; D-221, D-242; D-228 (2); D-244), one released pair per run.

At the pair's D-117 selected layer, selected C primary and C=1 beside, seeds 0-4 (probes fitted exactly as depth_d156
fits them). Repetition d is D-156 draw d of the axis (the frozen split first, then the first 200 item-level draws):
  T_d       each seed's 1% threshold on the PRE scores of the draw's calibration part cal_d (the frozen threshold)
  R_SAME    cal_d re-run through the post model (D-152's "calibration set", under D-156)
  R_FRESH   a random 1000 of the draw's 2000-negative evaluation part, re-drawn each repetition; EVERY arm is
            evaluated on the other 1000 (FPR) and on the fixed eval positives (TPR; never in any pool)
  SOURCES   benign traffic from cal_d's negatives; harmful traffic and injections from the 1000 threshold_calib harmful
            items H. probe_train enters no pool (F-060 holds by construction; D-244 (4)).
THE 2x2   {traffic pool at prevalence pi, reference set R_SAME / R_FRESH} x {feature restandardisation (the post
          features' mean and sd from the pool; probe weights and T_d fixed), score level (traffic: quantile transfer of
          T_d's rank among the pool's pre scores to the same items' post scores; reference: the threshold-only
          recalibration baseline, the 1% threshold on the reference's post scores)}; the frozen arm beside.
SWEEPS    prevalence (pool 1000); pool size at 1% (traffic; R_FRESH subsamples); window drift (pre window at 1%, post
          window at pi; quantile transfer); poisoning rho of the pool or reference size by A0 (random harmful) and A1
          (the highest-scoring harmful by the probe's post score) into the 1% traffic pool, R_SAME and R_FRESH.
REPORTED  per configuration: FPR (seed mean of floored per-seed FPRs), the paired ratio against the pre FPR at T_d on the
          same evaluation half, ratio to target and TPR, as D-156 mean and band over the draws with the frozen split;
          the minimum rho for a 10-point TPR loss per poisoned cell; every skipped configuration with its reason; the
          D-221/D-242 predictions beside the measurements (P2 mis-centring, P3 order-statistic spread, P4 budget).
Writes results/poolstudy_d156_<pair>.json (never overwritten); a test writes outside results/.
--origin-strat (D-298, option B, primary for the seven Aegis pairs; D-283, D-297 (2)): the same study on
origin-stratified draws. cal_d takes 333 threshold_calib-origin (TC) and 667 eval-origin (EV) negatives (default_rng
([20260963, d])); R_FRESH, its subsets and every benign traffic pool take their origin counts by largest remainder
(ties to the larger stratum); each configuration also reports the FPR of E's TC and EV items. The frozen split stays as
registered, flagged origin-confounded. Writes results/poolstudy_d156_<pair>_originstrat.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_d156 as D  # noqa: E402
import input_registry as REG  # noqa: E402
import paired as P  # noqa: E402
import pool_propositions as PP  # noqa: E402
import populations as POP  # noqa: E402
from provenance import (  # noqa: E402
    ensure_utf8_stdout, provenance_block, read_result, resolve_cache, sha256_bytes, sha256_file, write_results,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/poolstudy_d156.py", "src/pool_propositions.py", *D.CODE_FILES]
PROBE_SETS, REGISTERED = D.PROBE_SETS, D.REGISTERED
N_REPS = 200                                           # D-244 (3): the first 200 item-level draws + the frozen split
POOL_SEED = 20260936                                   # the pools' own randomness (R_FRESH split, traffic, A0)
PREVALENCES = (0.0, 0.001, 0.01, 0.05, 0.10, 0.25, 0.50)
POOL_SIZES = (200, 500, 1000)
DEFAULT_POOL = 1000
RHOS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25)
TPR_LOSS = 0.10
STRAT_DRAW_SEED = 20260963                             # D-298: option B's origin-stratified draws
STRAT_AXIS = "safety"                                  # option B is for the Aegis pairs only (D-297 (2))
# D-298's counts; a run refuses unless every stratified draw realises exactly these (TC, EV)
D298_COUNTS = {"cal": [333, 667], "R_fresh": [333, 667], "E": [334, 666],
               "R_fresh|n=200": [67, 133], "R_fresh|n=500": [166, 334], "R_fresh|n=1000": [333, 667]}
# --r2 (D-297 (1), D-299): the C-mode re-issue naming repo IDs and revisions, checked against the first record
FIRST_MANIFESTS = (ROOT / "logs" / "poolstudy_4dB_WG_20260930.sha256", ROOT / "logs" / "poolstudy_4dB_C_20261001.sha256")
C_ADDED_CONFIG = ("pre_repo", "pre_rev", "post_repo", "post_rev", "design", "cluster_manifest", "pool_item_ids_sha256",
                  "pi0_n1000_note")
REISSUE_MAY_DIFFER = ("utc", "git", "env", "code", "preregistration")
REG_SOURCES_KEY = "input_registry_sources_sha256"


def reissue_check(payload: dict[str, Any], r1b: bytes, r1_rel: str, r1_sha: str) -> dict[str, Any]:
    """D-297 (1): every key of the first record but the run's header equals the re-issue's, config without the keys
    the re-issue adds and its registry snapshot (which may only grow, keeping every first-record source)."""
    if sha256_bytes(r1b) != r1_sha:
        raise SystemExit(f"REFUSING to write: {r1_rel} is not the first record ({r1_sha[:8]})")
    r1 = json.loads(r1b.decode("utf-8"))                     # listed (D-297): hashed above, in reissue_of
    mine = json.loads(json.dumps(payload, default=str))      # not a result (D-297): the payload's own round trip
    added = {"supersedes", "reissue_of"}
    if set(mine) - added != set(r1):
        raise SystemExit(f"REFUSING to write: top-level keys differ from {r1_rel}: {sorted((set(mine) - added) ^ set(r1))}")
    if not str(mine.get("preregistration", "")).startswith(str(r1.get("preregistration", ""))):
        raise SystemExit(f"REFUSING to write: the preregistration does not extend {r1_rel}'s")
    if any(k in r1["config"] for k in ("pre_repo", "pre_rev", "post_repo", "post_rev")):
        raise SystemExit(f"REFUSING: {r1_rel} already names its repos; nothing to re-issue")
    if sorted(set(mine["config"]) - set(r1["config"]) - {REG_SOURCES_KEY}) != sorted(C_ADDED_CONFIG):
        raise SystemExit(f"REFUSING to write: config adds {sorted(set(mine['config']) - set(r1['config']))}, "
                         f"not exactly {sorted(C_ADDED_CONFIG)}")
    old_src, new_src = r1["config"][REG_SOURCES_KEY], mine["config"][REG_SOURCES_KEY]
    lost = sorted(k for k, h in old_src.items() if new_src.get(k) != h)
    if lost:
        raise SystemExit(f"REFUSING to write: registry sources of {r1_rel} missing or changed: {lost}")
    norm = lambda k, o: {x: y for x, y in o.items() if x not in (REG_SOURCES_KEY, *C_ADDED_CONFIG)} if k == "config" else o  # noqa: E731
    compared = [k for k in r1 if k not in REISSUE_MAY_DIFFER]
    differ = [k for k in compared if json.dumps(norm(k, mine[k]), sort_keys=True) != json.dumps(norm(k, r1[k]), sort_keys=True)]
    if differ:
        raise SystemExit(f"REFUSING to write: {differ} differ from {r1_rel} (D-297 (1))")
    return {"file": r1_rel, "sha256": r1_sha, "compared": True, "keys_compared": compared,
            "keys_allowed_to_differ": list(REISSUE_MAY_DIFFER), "config_keys_added": list(C_ADDED_CONFIG),
            "method": "the payload serialised as written and parsed back; each compared key gives the same "
                      "json.dumps(sort_keys=True) text as the first record's",
            "config_registry_sources": {"n_first_record": len(old_src), "n_now": len(new_src),
                                        "added_since": sorted(set(new_src) - set(old_src))},
            "result": "every compared key equals the first record exactly"}


def allocate(n: int, counts: list[int]) -> list[int]:
    """D-298: n items over strata in proportion to `counts`, by largest remainder; each stratum takes the floor of its
    share, the leftover items go one each to the largest remainders, ties to the larger stratum."""
    total = sum(counts)
    if total <= 0 or not 0 <= n <= total:
        raise ValueError(f"cannot allocate {n} over {counts}")
    qr = [divmod(n * c, total) for c in counts]                   # integer arithmetic: exact floors and remainders
    out = [q for q, _ in qr]
    order = sorted(range(len(counts)), key=lambda i: (-qr[i][1], -counts[i], i))
    for i in order[: n - sum(out)]:
        out[i] += 1
    return out


# --- the study's machinery as functions (D-152 P3: one definition, shared; Phase 3 A1 imports them, D-327 (6)).
# main() calls exactly these, in the order it always ran them, so the draws, pools and numbers are unchanged.

def probe_spec(n_train: int, seed: int, pset: str, C_sel: float) -> tuple[np.ndarray, float, str]:
    """The D-117 rule of each probe set, as depth_d156 fits them: the seed's fit rows, C and row order (the selected
    C in permutation order; C=1 in file order)."""
    _, fit = P.d117_split(n_train, seed)
    return (fit, C_sel, "d117_permutation") if pset == "selected" else (np.sort(fit), 1.0, "file_order")


def partitions(n_cal: int, n_pool: int, n_reps: int, axis: str, origin_strat: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    """The frozen split, then the study's n_reps draws, as (cal_d, ev_d) pooled indices: the first n_reps D-156
    item-level draws of the axis (D-244 (2), (3)), or with origin_strat D-298's origin-stratified draws."""
    if origin_strat:
        # D-298: origin-stratified draws; TC = pooled indices 0..n_cal-1, EV = n_cal..n_pool-1
        TC, EV = np.arange(n_cal), np.arange(n_cal, n_pool)
        k_tc, k_ev = allocate(n_cal, [len(TC), len(EV)])
        parts = [(np.arange(n_cal), np.arange(n_cal, n_pool))]          # the frozen split, as registered
        for d in range(1, n_reps + 1):
            rng_d = np.random.default_rng([STRAT_DRAW_SEED, d])
            p_tc, p_ev = rng_d.permutation(TC), rng_d.permutation(EV)
            parts.append((np.concatenate([p_tc[:k_tc], p_ev[:k_ev]]), np.concatenate([p_tc[k_tc:], p_ev[k_ev:]])))
        return parts
    draws = P.repartition_draws(n_cal, n_pool - n_cal, P.N_DRAWS, P.AXIS_DRAW_SEED[axis])[:n_reps]
    return [(np.arange(n_cal), np.arange(n_cal, n_pool))] + [(d[:n_cal], d[n_cal:]) for d in draws]


def draw_pools(di: int, cal: np.ndarray, ev: np.ndarray, n_cal: int, n_pool: int, n_h_src: int,
               origin_strat: bool) -> dict[str, Any]:
    """Repetition di's pools from its own stream default_rng([POOL_SEED, di]), drawn in the study's order: R_FRESH
    and E (the evaluation half), R_FRESH's subsets, H's random order (A0), the traffic pools (every prevalence at the
    default size, then the other sizes at 1%), the drift pre window, the drift post windows. Pooled indices; H item j
    is n_pool + j. 'skipped' names every pool the sources cannot supply, with the reason (that pool is None)."""
    rng = np.random.default_rng([POOL_SEED, di])
    skipped: dict[str, str] = {}
    if origin_strat:
        # D-298: R_FRESH takes each origin's count by largest remainder; its subsets too, within R_FRESH
        ev_tc, ev_ev = ev[ev < n_cal], ev[ev >= n_cal]
        a_r, b_r = allocate(1000, [len(ev_tc), len(ev_ev)])
        q_tc, q_ev = rng.permutation(ev_tc), rng.permutation(ev_ev)
        Rf, E = np.concatenate([q_tc[:a_r], q_ev[:b_r]]), np.concatenate([q_tc[a_r:], q_ev[b_r:]])
        rf_sub = {}
        for n in POOL_SIZES:
            x, y = allocate(n, [a_r, b_r])
            rf_sub[n] = np.concatenate([q_tc[:x], q_ev[:y]])
    else:
        perm = rng.permutation(ev)
        Rf, E = perm[:1000], perm[1000:]
        rf_sub = {n: Rf[:n] for n in POOL_SIZES}
    h_order = rng.permutation(n_h_src)                     # A0: one random order of H per draw (nested over rho)

    def traffic(pi: float, n: int, what: str) -> np.ndarray | None:
        n_h = int(round(pi * n))
        if n_h > n_h_src or n - n_h > len(cal):
            skipped[what] = f"needs {n_h} harmful and {n - n_h} benign; the source holds {n_h_src} and {len(cal)}"
            return None
        if origin_strat:                                   # D-298: the benign part allocated over cal_d's origins
            c_tc, c_ev = cal[cal < n_cal], cal[cal >= n_cal]
            x, y = allocate(n - n_h, [len(c_tc), len(c_ev)])
            benign = np.concatenate([rng.choice(c_tc, size=x, replace=False), rng.choice(c_ev, size=y, replace=False)])
        else:
            benign = rng.choice(cal, size=n - n_h, replace=False)
        return np.concatenate([benign, n_pool + rng.choice(n_h_src, size=n_h, replace=False)])

    pools: dict[str, np.ndarray] = {}
    for pi in PREVALENCES:
        p_ = traffic(pi, DEFAULT_POOL, f"traffic|pi={pi}|n={DEFAULT_POOL}")
        if p_ is not None:
            pools[f"traffic|pi={pi}|n={DEFAULT_POOL}"] = p_
    for n in POOL_SIZES:
        if n != DEFAULT_POOL:
            p_ = traffic(0.01, n, f"traffic|pi=0.01|n={n}")
            if p_ is not None:
                pools[f"traffic|pi=0.01|n={n}"] = p_
    pre_window = traffic(0.01, DEFAULT_POOL, "drift|pre window")
    drift_post = {pi: traffic(pi, DEFAULT_POOL, f"drift|post window pi={pi}") for pi in PREVALENCES}
    return {"R_fresh": Rf, "E": E, "R_fresh_subsets": rf_sub, "h_order": h_order, "pools": pools,
            "pre_window": pre_window, "drift_post": drift_post, "skipped": skipped}


def feature_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The feature arm's statistics of a pool's post features: the per-feature mean and SD, an SD below 1e-8 set
    to 1 (as fit_probe)."""
    mu, sd = x.mean(0), x.std(0)
    return mu, np.where(sd < 1e-8, 1.0, sd)


def restandardised_scores(q: Any, mu: np.ndarray, sd: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The stale probe q on features restandardised with (mu, sd), probe weights fixed (P2's mechanism):
    ((x - mu) / sd) . w + b, computed as x . v + (b - mu . v) with v = w / sd."""
    v = q.w / sd
    c0 = q.b - float(mu @ v)
    return x @ v + c0


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="4d (B) pool study under D-156 (D-244)")
    ap.add_argument("--pair", required=True, choices=POP.UPDATE_PAIRS_REPAIR)
    ap.add_argument("--n-reps", type=int, default=N_REPS)
    ap.add_argument("--target-fpr", type=float, default=REGISTERED["target_fpr"])
    ap.add_argument("--seeds", type=int, nargs="*", default=REGISTERED["seeds"])
    ap.add_argument("--origin-strat", action="store_true", help="option B, origin-stratified draws (D-298)")
    ap.add_argument("--r2", action="store_true", help="C mode re-issued with repo IDs and revisions (D-297 (1))")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.r2 and args.origin_strat:
        raise SystemExit("REFUSING: --r2 re-issues the registered (C) records only")
    core = args.pair[len("repair_"):]
    reg_stem = D.registry_stem(f"depth_{core}")
    sel_path = D.SELECTION_OF_RECORD
    registered = (args.n_reps == N_REPS and args.target_fpr == REGISTERED["target_fpr"]
                  and list(args.seeds) == REGISTERED["seeds"])
    suffix = "_originstrat" if args.origin_strat else ("_r2" if args.r2 else "")
    final_path = (RESULTS / f"poolstudy_d156_{core}{suffix}.json").resolve()
    out_path = Path(args.out).resolve() if args.out else final_path
    final = out_path == final_path
    if out_path.exists():
        raise SystemExit(f"REFUSING: {out_path} exists; results are never overwritten")
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test run (--out) must not write under results/")
    threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    if final and (not registered or any(t != "6" for t in threads.values())):
        raise SystemExit(f"REFUSING the final name: registered parameters {registered}, threads {threads}")
    code0 = P.code_block(CODE_FILES)
    sel_sha, cl_sha = sha256_file(sel_path), sha256_file(D.CLUSTERS)
    sel_doc = read_result(sel_path)                         # registered read: the record lists it (D-297 (1))
    sel = next((m for m in sel_doc["models"] if m["pair"] == reg_stem), None)
    if sel is None or not sel["selected"]["arm"].startswith("C="):
        raise SystemExit(f"REFUSING: no logistic-C selection for {reg_stem}")
    C_sel, L_sel = float(sel["selected"]["arm"].split("=")[1]), int(sel["selected"]["layer"])
    depth_rel = f"results/depth_d156_{core}.json"
    depth_sha = sha256_file(ROOT / depth_rel)
    ddoc = read_result(ROOT / depth_rel)
    if (ddoc["config"]["selected_C"], ddoc["config"]["selected_layer"], ddoc["config"]["selection_sha256"]) != (C_sel, L_sel, sel_sha):
        raise SystemExit(f"REFUSING: {depth_rel} was run on another selection")
    dcells = {(c["layer"], c["probe_set"]): c for c in ddoc["cells"]}

    cfg = REG.run(reg_stem)
    axis = cfg["axis"]
    if args.origin_strat and axis != STRAT_AXIS:
        raise SystemExit(f"REFUSING: option B is for the Aegis pairs (axis {STRAT_AXIS}); {args.pair} is on {axis} (D-297 (2))")
    data: dict[str, dict[str, Any]] = {"pre": {}, "post": {}}
    input_sha: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    for side in data:
        for part in ("probe_train", "threshold_calib", "eval"):
            z, r = REG.load(cfg[f"{side}_repo"], axis, part)
            data[side][part] = z
            input_sha[f"{side}/{part}"], input_recs[f"{side}/{part}"] = r["sha256"], r
    reg_sources = REG.sources_sha256()
    if D.sources_now() != reg_sources:
        raise SystemExit("REFUSING: the extraction records or pins on disk are not the ones the registry read")
    if any(ddoc["input_cache_sha256"].get(k) != h for k, h in input_sha.items()):
        raise SystemExit(f"REFUSING: {depth_rel} read other caches")
    verification = REG.verification(input_recs)
    for part in ("probe_train", "threshold_calib", "eval"):
        a, b = data["pre"][part], data["post"][part]
        if not np.array_equal(a["item_ids"], b["item_ids"]) or not np.array_equal(a["labels"], b["labels"]):
            raise SystemExit(f"REFUSING: {part} items or labels differ between pre and post")

    y_tr = data["pre"]["probe_train"]["labels"].astype(bool)
    y_cal = data["pre"]["threshold_calib"]["labels"].astype(bool)
    y_ev = data["pre"]["eval"]["labels"].astype(bool)
    ncal, hcal, neg, pos = ~y_cal, y_cal, ~y_ev, y_ev
    pooled_ids = [str(i) for i in data["pre"]["threshold_calib"]["item_ids"][ncal]] + \
                 [str(i) for i in data["pre"]["eval"]["item_ids"][neg]]
    n_cal, n_pool = int(ncal.sum()), len(pooled_ids)
    cl = json.load(open(D.CLUSTERS, encoding="utf-8"))["axes"][axis]   # not a result (D-297): a manifest, hashed
    if cl["n_pooled_negatives"] != n_pool or cl["n_calibration"] != n_cal or \
            cl["pool_item_ids_sha256"] != sha256_bytes("\n".join(pooled_ids).encode("utf-8")):
        raise SystemExit(f"REFUSING: manifests/negative_clusters.json does not describe the {axis} pool")
    parts = partitions(n_cal, n_pool, args.n_reps, axis, args.origin_strat)
    n_h_src = int(hcal.sum())
    k = f"layer_{L_sel}"
    if n_pool - n_cal != 2000:
        raise SystemExit(f"REFUSING: D-152's R_FRESH takes 1000 of 2000 evaluation negatives; this pool has {n_pool - n_cal}")
    f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
    # raw cache arrays for SCORING (exactly as depth_d156 scores, so the anchor holds by construction); float64 copies
    # only for the feature arm's statistics and projections
    RAW = {s: {"pool": np.vstack([data[s]["threshold_calib"][k][ncal], data[s]["eval"][k][neg]]),
               "H": data[s]["threshold_calib"][k][hcal], "pos": data[s]["eval"][k][pos]} for s in data}
    PN = {s: f64(RAW[s]["pool"]) for s in data}                                                              # pool, 3000
    POS = {s: f64(RAW[s]["pos"]) for s in data}
    ALL_post = np.vstack([PN["post"], f64(RAW["post"]["H"])])  # feature table: pool rows, then H rows (n_pool + j)
    Xtr = {s: data[s]["probe_train"][k] for s in data}
    print(f"4d(B) {args.pair} [{axis}] layer {L_sel} (selected C {C_sel:g}); {len(parts) - 1} draws + frozen; pooled "
          f"{n_cal} + {n_pool - n_cal}; H {n_h_src}; D-170 flag {verification['flag']}; registered {registered}", flush=True)

    probes: dict[str, list[Any]] = {}
    fitpost: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}       # the fit rows' POST feature statistics (P2)
    fitmeta: dict[str, Any] = {}
    for pset in PROBE_SETS:
        probes[pset], fitpost[pset] = [], []
        fit_d: list[float] = []
        for seed in args.seeds:
            rows, C, order = probe_spec(len(y_tr), seed, pset, C_sel)
            q = P.fit_probe(Xtr["pre"][rows], y_tr[rows], seed, C, order)
            probes[pset].append(q)
            s_fit = q.score(Xtr["pre"][rows])
            fit_d.append(float(s_fit[y_tr[rows]].mean() - s_fit[~y_tr[rows]].mean()))
            xp = f64(Xtr["post"][rows])
            fitpost[pset].append((xp.mean(0), np.where(xp.std(0) < 1e-8, 1.0, xp.std(0))))
        fitmeta[pset] = {"fit_prevalence_mean_over_seeds": float(np.mean([y_tr[P.d117_split(len(y_tr), s)[1]].mean() for s in args.seeds])),
                         "class_mean_score_difference_d_per_seed": fit_d,
                         "d_note": "in-sample on the fit rows (pre model); inflated for a probe that interpolates its "
                                   "training set (C=1, F-045), so P2's C=1 prediction is an upper bound"}
    # scores of every item through each stale probe (pre and post), per probe set: (items, seeds), from the raw arrays
    SC = {pset: {side: {part: np.column_stack([q.score(RAW[side][part]) for q in probes[pset]]) for part in ("pool", "H", "pos")}
                 for side in ("pre", "post")} for pset in PROBE_SETS}
    # anchor: the frozen split's pre-side eval FPR at T equals the depth file's (D-156 frozen value)
    for pset in PROBE_SETS:
        T0 = np.array([P.thr_at_fpr(SC[pset]["pre"]["pool"][parts[0][0], s], args.target_fpr) for s in range(len(args.seeds))])
        fp0 = float(P.floored_fpr(SC[pset]["pre"]["pool"][parts[0][1]] >= T0).mean())
        if registered and fp0 != dcells[(L_sel, pset)]["operating_point"]["fpr_pre"]["frozen"]:
            raise SystemExit(f"REFUSING: {pset}: the frozen split's pre FPR is not {depth_rel}'s")

    acc: dict[tuple[str, str], dict[str, list[float]]] = {}
    skipped: dict[str, str] = {}
    p2: dict[str, list[float]] = {}
    cur: dict[str, np.ndarray] = {}                       # the current draw's E: which of its items are TC-origin
    realised: list[dict[str, list[int]]] = []             # D-298 counts per stratified draw (checked before writing)

    def put(pset: str, cfgname: str, F_pre: np.ndarray, F: np.ndarray, Q: np.ndarray) -> None:
        a = acc.setdefault((pset, cfgname), {"fpr": [], "paired_ratio": [], "ratio_to_target": [], "tpr": [],
                                             **({"fpr_origin_tc": [], "fpr_origin_ev": []} if args.origin_strat else {})})
        f = float(P.floored_fpr(F).mean())
        a["fpr"].append(f)
        a["paired_ratio"].append(P.ratio_from_flags(F_pre, F))
        a["ratio_to_target"].append(f / args.target_fpr)
        a["tpr"].append(float(P.rate(Q).mean()))
        if args.origin_strat:                             # D-298 beside: FPR of E's TC and EV items (floored per origin)
            m = cur["e_tc"]
            a["fpr_origin_tc"].append(float(P.floored_fpr(F[m]).mean()) if m.any() else float("nan"))
            a["fpr_origin_ev"].append(float(P.floored_fpr(F[~m]).mean()) if (~m).any() else float("nan"))

    for di, (cal, ev) in enumerate(parts):
        dp = draw_pools(di, cal, ev, n_cal, n_pool, n_h_src, args.origin_strat)
        Rf, E, rf_sub, h_order = dp["R_fresh"], dp["E"], dp["R_fresh_subsets"], dp["h_order"]
        pools, pre_window, drift_post = dp["pools"], dp["pre_window"], dp["drift_post"]
        skipped.update(dp["skipped"])
        cur["e_tc"] = E < n_cal
        if args.origin_strat and di > 0:                       # D-298's counts, realised in every stratified draw
            tc = lambda idx: [int((idx < n_cal).sum()), int((idx >= n_cal).sum())]  # noqa: E731
            realised.append({"cal": tc(cal), "R_fresh": tc(Rf), "E": tc(E),
                             **{f"R_fresh|n={n}": tc(rf_sub[n]) for n in POOL_SIZES}})
        refs = {"R_same|n=1000": cal, **{f"R_fresh|n={n}": rf_sub[n] for n in POOL_SIZES}}
        base = pools.get(f"traffic|pi=0.01|n={DEFAULT_POOL}")
        targets = {"traffic_1pct": base, "R_same": cal, "R_fresh": Rf}
        # feature statistics once per draw for every seed-free pool
        stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        def st(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return feature_stats(ALL_post[idx])

        for name, idx in {**pools, **refs}.items():
            stats[name] = st(idx)
        a0_pools: dict[str, np.ndarray] = {}
        fresh_h: dict[str, np.ndarray] = {}                    # H items not already in the target (no duplicates)
        for tname, tidx in targets.items():
            if tidx is None:
                skipped[f"poison|{tname}"] = "its base pool could not be drawn"
                continue
            fresh_h[tname] = np.setdiff1d(np.arange(n_h_src), tidx[tidx >= n_pool] - n_pool, assume_unique=True)
            order = h_order[np.isin(h_order, fresh_h[tname])]
            for rho in RHOS:
                m = int(round(rho * len(tidx)))
                a0_pools[f"{tname}|A0|rho={rho}|m={m}"] = np.concatenate([tidx, n_pool + order[:m]])
                stats[f"{tname}|A0|rho={rho}|m={m}"] = st(a0_pools[f"{tname}|A0|rho={rho}|m={m}"])
        XE, XP = PN["post"][E], POS["post"]

        for pset in PROBE_SETS:
            S = SC[pset]
            T = np.array([P.thr_at_fpr(S["pre"]["pool"][cal, s], args.target_fpr) for s in range(len(args.seeds))])
            F_pre = S["pre"]["pool"][E] >= T
            put(pset, "frozen", F_pre, S["post"]["pool"][E] >= T, S["post"]["pos"] >= T)
            all_pre = np.vstack([S["pre"]["pool"], S["pre"]["H"]])
            all_post = np.vstack([S["post"]["pool"], S["post"]["H"]])

            def feature(mu_sd: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
                mu, sd = mu_sd
                FE, QP = [], []
                for s, q in enumerate(probes[pset]):
                    FE.append(restandardised_scores(q, mu, sd, XE) >= T[s])
                    QP.append(restandardised_scores(q, mu, sd, XP) >= T[s])
                return np.column_stack(FE), np.column_stack(QP)

            def quantile(pre_idx: np.ndarray, post_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                # F-070: the exact rank transfer (T_d itself with no update)
                tq = np.array([P.quantile_transfer(all_pre[pre_idx, s], all_post[post_idx, s], T[s]) for s in range(len(T))])
                return S["post"]["pool"][E] >= tq, S["post"]["pos"] >= tq

            def recal(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                tr = np.array([P.thr_at_fpr(all_post[idx, s], args.target_fpr) for s in range(len(T))])
                return S["post"]["pool"][E] >= tr, S["post"]["pos"] >= tr

            for name, idx in pools.items():
                put(pset, f"feature|{name}", F_pre, *feature(stats[name]))
                put(pset, f"score|{name}", F_pre, *quantile(idx, idx))
                if name.startswith("traffic|pi=") and name.endswith(f"n={DEFAULT_POOL}"):
                    # P2: predicted mis-centring (pi - pi_fit)*d against the measured mean score shift on the
                    # evaluation half between restandardising with the fit rows' post statistics and with this pool's
                    pi = float(name.split("|")[1][3:])
                    mu_b, sd_b = stats[name]
                    meas = [float((XE @ (q.w / fs[1])).mean() - fs[0] @ (q.w / fs[1]) - ((XE @ (q.w / sd_b)).mean() - mu_b @ (q.w / sd_b)))
                            for q, fs in zip(probes[pset], fitpost[pset])]
                    pred = [PP.pooled_mean_shift(pi, fitmeta[pset]["fit_prevalence_mean_over_seeds"], dd)
                            for dd in fitmeta[pset]["class_mean_score_difference_d_per_seed"]]
                    p2.setdefault(f"{pset}|pi={pi}", []).append((float(np.mean(pred)), float(np.mean(meas))))
            for name, idx in refs.items():
                put(pset, f"feature|{name}", F_pre, *feature(stats[name]))
                put(pset, f"score|{name}", F_pre, *recal(idx))
            if pre_window is not None:
                for pi, post_w in drift_post.items():
                    if post_w is not None:
                        put(pset, f"score|drift|pre=0.01|post={pi}", F_pre, *quantile(pre_window, post_w))
            for tname, tidx in targets.items():
                if tidx is None:
                    continue
                a1_all = np.argsort(-all_post[n_pool:].mean(axis=1))        # A1: highest post score (seed mean)
                a1_order = a1_all[np.isin(a1_all, fresh_h[tname])]
                for rho in RHOS:
                    m = int(round(rho * len(tidx)))
                    for att, idx in (("A0", a0_pools[f"{tname}|A0|rho={rho}|m={m}"]),
                                     ("A1", np.concatenate([tidx, n_pool + a1_order[:m]]))):
                        nm = f"{tname}|{att}|rho={rho}|m={m}"
                        ms = stats[nm] if att == "A0" else st(idx)
                        put(pset, f"feature|poison|{nm}", F_pre, *feature(ms))
                        # traffic: the attack comes after the update, so the pre window is the unpoisoned pool (D-244)
                        put(pset, f"score|poison|{nm}", F_pre, *(quantile(tidx, idx) if tname.startswith("traffic") else recal(idx)))
        if di % 25 == 0:
            print(f"  draw {di} of {len(parts) - 1} done", flush=True)

    out_cfgs: dict[str, Any] = {}
    for (pset, name), a in acc.items():
        out_cfgs.setdefault(pset, {})[name] = {q: P.band_summary(np.array(v[1:]), v[0]) for q, v in a.items()}
    # minimum rho for a 10-point TPR loss, per poisoned cell (D-156 mean TPR against the unpoisoned target)
    min_rho: dict[str, Any] = {}
    for pset in PROBE_SETS:
        for arm in ("feature", "score"):
            for tname, un in (("traffic_1pct", f"{arm}|traffic|pi=0.01|n={DEFAULT_POOL}"), ("R_same", f"{arm}|R_same|n=1000"),
                              ("R_fresh", f"{arm}|R_fresh|n=1000")):
                if un not in out_cfgs[pset]:
                    continue
                t0 = out_cfgs[pset][un]["tpr"]["mean"]
                for att in ("A0", "A1"):
                    hit = None
                    for rho in RHOS:
                        nm = next((kk for kk in out_cfgs[pset] if kk.startswith(f"{arm}|poison|{tname}|{att}|rho={rho}|")), None)
                        if nm and t0 - out_cfgs[pset][nm]["tpr"]["mean"] >= TPR_LOSS:
                            hit = rho
                            break
                    min_rho[f"{pset}|{arm}|{tname}|{att}"] = {"min_rho": hit, "searched_up_to": RHOS[-1]}

    def whole_item_silencing(n_b: int) -> int:
        """The smallest m of items above every benign score that silences the benign flags under the 1% rule, which
        counts whole items: m >= floor(alpha * (n_b + m)) (D-242)."""
        return next(m for m in range(n_b + 1) if m >= int(np.floor(args.target_fpr * (n_b + m))))

    sel_cfg = out_cfgs.get("selected", {})
    predictions = {
        "P2_mis_centring_score_units": {kk: {"predicted_mean_over_draws": float(np.mean([x[0] for x in v])),
                                             "measured_mean_over_draws": float(np.mean([x[1] for x in v]))} for kk, v in p2.items()},
        "P3_quantile_floor": {f"n={n}": {"predicted_order_stat_sd": PP.quantile_fpr_moments(n, args.target_fpr)[1],
                                         "predicted_with_evaluation_half_sd": float(np.sqrt(args.target_fpr * (1 - args.target_fpr) * (1 / n + 1 / 1000))),
                                         "measured_sd_over_draws_seed_mean_fpr": float(np.std(acc[("selected", f"score|R_fresh|n={n}")]["fpr"][1:]))
                                         if ("selected", f"score|R_fresh|n={n}") in acc else None}
                              for n in POOL_SIZES},
        "P4_silencing": {tname: {"n_b": nb, "predicted_continuous": PP.silencing_items(nb, args.target_fpr),
                                 "predicted_whole_items": whole_item_silencing(nb),
                                 "measured_fpr_by_m_A1_selected": {kk.split("|")[-1]: v["fpr"]["mean"] for kk, v in sel_cfg.items()
                                                                   if kk.startswith(f"score|poison|{tname}|A1|")}}
                         for tname, nb in (("R_same", 1000), ("R_fresh", 1000))},
        "P4_note": "the continuous budget holds within one whole item (D-242); A1 items need not score above every benign "
                   "item, so the measured FPR can stay above 0 past the predicted count",
        "P3_note": "the measured sd is over draws of the seed-mean FPR, so seed averaging makes it smaller than the "
                   "single-seed prediction"}
    code1 = P.code_block(CODE_FILES)
    changed = [n for n, a, b in (("code", code1["sha256"], code0["sha256"]), ("selection", sha256_file(sel_path), sel_sha),
                                 ("cluster manifest", sha256_file(D.CLUSTERS), cl_sha), (depth_rel, sha256_file(ROOT / depth_rel), depth_sha),
                                 ("registry sources", D.sources_now(), reg_sources)) if a != b]
    changed += [f"cache {r['file']}" for r in input_recs.values() if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    if changed:
        raise SystemExit(f"REFUSING to write: changed during the run: {changed}")
    if args.origin_strat:
        off = [i + 1 for i, r in enumerate(realised) if r != D298_COUNTS]
        if len(realised) != args.n_reps or off:
            raise SystemExit(f"REFUSING to write: {len(realised)} stratified draws, {len(off)} off D-298's counts {off[:5]}")
    pool_ids_sha = sha256_bytes("\n".join(pooled_ids).encode("utf-8"))
    aegis = axis == STRAT_AXIS
    prereg = "D-152 4d, D-221, D-228 (2), D-242, D-244" + \
             ("; D-283 (B), D-297 (2), D-298 (option B, origin-stratified draws)" if args.origin_strat else "") + \
             ("; D-297 (1) (re-issue naming repo IDs and revisions)" if args.r2 else "")
    payload = {**provenance_block("poolstudy_d156"), "preregistration": prereg, "code": code0,
               "registered_parameters": registered, "blas_threads": threads, "input_verification": verification,
               "config": {"pair": args.pair, "registry_pair": reg_stem, "axis": axis, "layer": L_sel, "selected_C": C_sel,
                          "selection_file": str(sel_path.relative_to(ROOT)).replace("\\", "/"), "selection_sha256": sel_sha,
                          "seeds": args.seeds, "target_fpr": args.target_fpr, "n_reps": args.n_reps,
                          "draws": (f"frozen split (as registered; origin-confounded, not a stratified draw) + {args.n_reps} "
                                    f"origin-stratified draws, default_rng([{STRAT_DRAW_SEED}, d]) (D-298)") if args.origin_strat
                          else f"frozen split + the first {args.n_reps} of paired.repartition_draws(seed {P.AXIS_DRAW_SEED[axis]})",
                          "pool_seed": POOL_SEED, "prevalences": PREVALENCES, "pool_sizes": POOL_SIZES, "default_pool": DEFAULT_POOL,
                          "rhos": RHOS, "tpr_loss": TPR_LOSS, "evaluation": "the draw's evaluation part minus R_FRESH (1000 negatives); eval positives",
                          "sources": "benign: cal_d negatives; harmful: threshold_calib harmful H; probe_train in no pool (D-244 (4))",
                          **({"design": "option B, origin-stratified (D-298; primary for the Aegis pairs, D-297 (2))",
                              "strat_draw_seed": STRAT_DRAW_SEED,
                              "allocation": "largest remainder over the origin counts (integer arithmetic), ties to the "
                                            "larger stratum; the realised counts are in origin_strata"}
                             if args.origin_strat else
                             {"design": "as registered (option C, the sensitivity row beside B, D-297 (2), D-298)" if aegis
                              else "as registered (WildGuardMix, D-283: every source exchangeable)"}),
                          "cluster_manifest": {"file": "manifests/negative_clusters.json", "sha256": cl_sha},
                          "pool_item_ids_sha256": pool_ids_sha,
                          **{k: cfg[k] for k in ("pre_repo", "pre_rev", "post_repo", "post_rev")},   # D-297 (1)
                          "pi0_n1000_note": "traffic at pi 0 and pool 1000 is all of cal_d, so its score-level arm equals "
                                            "R_SAME's recalibration exactly (D-276 (5))",
                          "input_registry_sources_sha256": reg_sources},
               **({"origin_strata": {
                   "tc_pooled_indices": [0, n_cal - 1], "ev_pooled_indices": [n_cal, n_pool - 1],
                   "realised_counts_tc_ev": D298_COUNTS, "n_stratified_draws_checked": len(realised),
                   "check": "every stratified draw realised exactly these (TC, EV) counts, D-298's",
                   "frozen_split": {"origin_confounded": True,
                                    "note": "as registered: cal = all TC, the evaluation part all EV; not a stratified draw. "
                                            "Its frozen_percentile ranks it among stratified draws and means little",
                                    "fpr_origin_tc": "null by construction (E holds no TC item)",
                                    "fpr_origin_ev": "the FPR of all of E (floor 1/1000)"},
                   "beside": "fpr_origin_tc / fpr_origin_ev in every configuration: the seed-mean FPR of E's 334 TC and 666 EV "
                             "items, each per-seed FPR floored at 1/334 and 1/666"}} if args.origin_strat else {}),
               "anchor": {"file": depth_rel, "sha256": depth_sha, "kind": "the frozen split's pre FPR at T, both probe sets"},
               "probe_fit": fitmeta, "input_cache_sha256": input_sha,
               "skipped_configurations": skipped, "min_rho_for_10_point_tpr_loss": min_rho, "predictions": predictions,
               "configurations": out_cfgs}
    if args.r2:
        r1_rel = f"results/poolstudy_d156_{core}.json"
        first: dict[str, str] = {}
        for m in FIRST_MANIFESTS:
            if m.is_file():
                first.update({ln.split()[1]: ln.split()[0] for ln in m.read_text(encoding="utf-8").splitlines() if ln.strip()})
        if r1_rel not in first:
            raise SystemExit(f"REFUSING: no first-record manifest lists {r1_rel} ({[m.name for m in FIRST_MANIFESTS]})")
        payload["supersedes"] = [f"{r1_rel} (D-297 (1): the same numbers, with the model repos and revisions named)"]
        payload["reissue_of"] = (reissue_check(payload, (ROOT / r1_rel).read_bytes(), r1_rel, first[r1_rel]) if registered
                                 else {"file": r1_rel, "sha256": first[r1_rel], "compared": False,
                                       "why": "non-registered parameters (a test run)"})
    digest = write_results(out_path, payload, exclusive=True)
    print(f"Written: {out_path}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
