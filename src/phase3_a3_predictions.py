"""Phase 3 attacker A3, the reference-set poisoner (D-332), part 1 of 2: the P-A3.1 PREDICTIONS, one pair per run.

Written before any poisoned repair is computed (D-332 (5), (7)). From the clean pools' post scores and the injected
items' post scores ONLY, for every draw d (the frozen split first, then 4d B's 200 draws), seed s of the pair's D-117
selected probe at L*, pool, attacker in {A3-top, A3-benign} and rho:
  t'_ds     the k-th largest of the clean pool's n post scores together with the m injected items' post scores,
            k = floor(alpha * (n + m)), alpha = 0.01 (the 1% rule in whole items, D-242)       (kth_largest, whole_item_k)
  TPR'_ds   F1bar(t'_ds): the share of the eval positives' post scores at or above t'_ds          (predicted_tpr below)
  POOLS     R_same (cal_d's 1000 negatives), R_fresh (the draw's 1000 fresh benign negatives), traffic_1pct (4d B's 1%
            traffic pool: 990 benign from cal_d and 10 harmful from H); n = 1000 each (D-332 (1)). Poisoning R_same
            models contamination of the post-update re-scoring only: T_d (and A3-band's pre moments) come from the
            clean cal_d.
  ATTACKERS A3-top: the harmful items H (threshold_calib's harmful) not already in the pool, highest seed-mean post score
            first: 4d B's A1 order by the exact expression of poolstudy_d156.main() (np.argsort(-mean) over H, then
            np.isin to the items not in the pool), so the injected items are 4d's A1 items (D-332 (3), (6)).
            A3-benign: the pre side's probe_train negatives (label 0), highest seed-mean post score first (the same
            expression over them); disjoint from every pool by the split (checked on item IDs at load).
  rho       4d's grid (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25); m = int(round(rho * n)), 4d B's rounding.
No repair runs here: t' is the stated order statistic (np.sort), never paired.thr_at_fpr, quantile_transfer or a
restandardisation on a poisoned pool. The record is results/phase3_a3_predictions_<pair>.json (written once,
exclusive) with its per-draw arrays in results/phase3_a3_predictions_<pair>_draws.npz (tracked, D-163 (6); sha256 and
size in the JSON). The simulation (src/phase3_a3.py) refuses unless this record exists, names the same pair, draws,
inputs, injections and this file's sha256 (the prediction function), and its arrays equal the simulation's own
recomputation bit for bit.

MACHINERY (D-332 (1), (7); D-152 P3, one definition shared): 4d B's inputs, probes, partitions and pools.
src/poolstudy_d156.py's probe_spec, partitions and draw_pools give the D-117 probe rule, the frozen split plus 200 draws
(option B, D-298, for the seven Aegis pairs; the registered draws for the two WildGuardMix pairs: the 4d B record of
record, populations.POOLSTUDY_D156) and each draw's R_FRESH, E and 1% traffic pool; src/paired.py fits and scores; the
small output helpers are src/phase3_a1_predictions.py's (write_pair, record_bytes, npz_bytes, rel, core_of). setup()
below is shared with the simulation.

CHOICES BY THE CLOSEST D-ENTRY (D-181; also in the record's implementation_choices and the run report):
  (a) probe set: the selected C only (D-332 (1) "selected C"; D-346 (2)(a)).
  (b) P-A3.1's cells are pair x pool x attacker x rho (D-332 (5)) with the POOL'S SCORE-LEVEL REPAIR: the threshold
      re-set on R_same and on R_fresh, quantile transfer on traffic. Restandardisation on traffic is not predicted by
      P-A3.1: its threshold is T_d on restandardised scores, not an order statistic of the pool (D-221 P4 is the
      quantile-threshold budget; its pooled-mean offset is the restandardisation one). It is scored by P-A3.4.
      On the threshold re-set cells (32 of the 48, about two thirds) the prediction is the repair's own rule on the
      same pool, so they agree by construction up to ties at the k-th place. D-332's 1% rule is NOT quantile
      transfer's rule on traffic: the transfer carries the clean pre window's tail share at T_d (its 10 harmful items
      included) to the poisoned post window, not 1% of n + m, so its threshold sits lower than t' and the traffic cells
      are expected to be under-predicted (predicted TPR below the realised one).
  (c) F1bar(t) is the share of eval positives with post score >= t: the project's flag rule (score >= threshold, as
      paired.thr_at_fpr's exceedance and every 4d flag), D-221's survival notation.
  (d) k = floor(alpha * (n + m)) in floating point, int(math.floor(alpha * (n + m))); a test shows it equals the
      largest count c with c / (n + m) <= alpha (the count paired.thr_at_fpr allows) for every n + m of the grid.
      "The k-th largest" counts tied scores with multiplicity (np.sort(x)[N - k]); paired.thr_at_fpr skips a value
      tied at the k-th place, which the simulation's identity check allows for (src/phase3_a3.py).
  (e) A3-benign ranks by the seed-mean post score with the expression 4d B uses for A1 (np.argsort(-mean)), so the two
      attackers rank alike (D-244 (4): injections ranked by the probe set's seed-mean post score).
  (f) A3-benign's candidates are ALL the probe_train negatives, as D-332 (3) names the set; each seed's D-117 fit rows
      are among them (in-sample for that seed's probe; F-060, D-221). A CONFOUND, with its direction (D-352 (2), rule
      5): the fit rows are 80% of probe_train (D-117's 20% hold-out aside), so about 80% of A3-benign's candidates are,
      per seed, that seed's own fit rows, which score low under the probe fitted on them (the reason F-045 / F-060 keep
      fit rows out of every 4d pool). This weakens A3-benign against an attacker holding unseen benign text and biases
      P-A3.2 toward holding. The realised share among the injected items is recorded per seed
      (benign_fit_row_share_per_seed). A3-benign stays as D-332 (3) names it (D-352 (2)).
  (g) per draw the predicted TPR is the mean over seeds of TPR'_ds; a cell's prediction is its D-156 mean over the 200
      draws with the frozen split beside (as the realised TPR: D-156, D-346 (4)).
  (h) the injections are the attacker's choice, not a repair: their first m_max = round(0.25 n) items per pool and draw
      (A3-top) and per pair (A3-benign) are stored in the npz and bound bit for bit by the simulation.

The record carries its config's hash (config_hash) and D-352 (5)'s sandbox disclosure (sandbox_disclosure).
Only registered parameters (200 draws, 1% target, seeds 0-4) at the 6-thread BLAS pin write the final name; a test
writes outside results/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_d156 as D  # noqa: E402
import input_registry as REG  # noqa: E402
import paired as P  # noqa: E402
import phase3_a1_predictions as A1P  # noqa: E402
import populations as POP  # noqa: E402
import poolstudy_d156 as PS  # noqa: E402
from provenance import (  # noqa: E402
    canonical_hash, ensure_utf8_stdout, provenance_block, read_result, resolve_cache, sha256_bytes, sha256_file,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
THIS_FILE = "src/phase3_a3_predictions.py"
CODE_FILES = [THIS_FILE, *A1P.CODE_FILES]
PSET = "selected"                                                           # choice (a)
TRAFFIC = f"traffic|pi=0.01|n={PS.DEFAULT_POOL}"                            # 4d B's 1% traffic pool at n 1000
POOLS = ("R_same", "R_fresh", "traffic_1pct")                               # 4d B's poisoning targets (its tname)
POOL_OF_DRAW = {"R_same": "cal", "R_fresh": "R_fresh", "traffic_1pct": "traffic"}
PREDICTED = ("A3-top", "A3-benign")                                         # P-A3.1 (D-332 (5))
RHOS = PS.RHOS                                                              # 4d's grid (D-332 (2))
REGISTERED = {"n_reps": PS.N_REPS, "target_fpr": P.TARGET_FPR, "seeds": [0, 1, 2, 3, 4]}
PREREG = ("D-332 (the Phase 3 A3 pre-registration: (1) population and machinery, (2) threat model, (3) attackers, "
          "(5) P-A3.1, (7) predictions first); D-352 (the review outcome; A3-benign kept as registered, its confound "
          "stated); D-318 (9); docs/phase3_design.md revision 2")
SANDBOX = ("D-352 (5), as D-346 (3): at c220781, before the A3 code was reviewed, a node sandbox computed A3-top for "
           "llama31 at 200 draws (anchor-only: 4d B's A1 values, reproduced bit for bit) and the predictions plus the "
           "full simulation for llama31 at 5 draws; nobody read their A3-benign, A3-band or P-A3.x fields; this record "
           "is written fresh by the reviewed code, and no sandbox figure is quoted")
FORMULA = ("predicted TPR = F1bar(t'), t' = the k-th largest of the clean pool's post scores together with the m "
           "injected items' post scores, k = floor(alpha * (n + m)), alpha = 0.01; F1bar(t) = share of eval positives "
           "with post score >= t; per draw and seed (D-332 P-A3.1)")
CHOICES = {
    "a": "probe set: the selected C only (D-332 (1); D-346 (2)(a))",
    "b": "P-A3.1 cells: pair x pool x attacker x rho with the pool's score-level repair (threshold re-set on R_same "
         "and R_fresh, quantile transfer on traffic); restandardisation is not predicted by P-A3.1 (its threshold is "
         "T_d on restandardised scores, not an order statistic; D-221 P4) and is scored by P-A3.4 (D-332 (5)). On the "
         "32 threshold re-set cells of 48 (about two thirds) the prediction is the repair's own rule on the same pool, "
         "so they agree by construction up to ties at the k-th place. D-332's 1% rule is not quantile transfer's rule "
         "on traffic: the transfer carries the clean pre window's tail share at T_d (its harmful items included) to the "
         "poisoned post window, so its threshold sits below t' and the traffic cells are expected to be under-predicted",
    "c": "F1bar(t) = share of eval positives with post score >= t, the project's flag rule (paired.thr_at_fpr's "
         "exceedance; D-221's notation)",
    "d": "k = int(math.floor(alpha * (n + m))) (equal to the largest c with c/(n+m) <= alpha on the grid, tested); "
         "the k-th largest counts ties with multiplicity (np.sort(x)[N - k]) (D-242: whole items)",
    "e": "A3-benign ranked by the seed-mean post score with 4d B's A1 expression np.argsort(-mean) (D-244 (4))",
    "f": "A3-benign candidates: every probe_train negative, as D-332 (3) names them; each seed's fit rows are among "
         "them (F-060, D-221). A confound with a direction (D-352 (2), rule 5): the fit rows are 80% of probe_train "
         "(D-117's 20% hold-out aside), so about 80% of the candidates are, per seed, that seed's own fit rows, which "
         "score low under the probe fitted on them (why F-045 / F-060 keep fit rows out of every 4d pool); this weakens "
         "A3-benign and biases P-A3.2 toward holding. The realised share among the injected items is recorded per seed "
         "(benign_fit_row_share_per_seed)",
    "g": "a cell's prediction is the D-156 mean over the 200 draws of the per-draw mean over seeds, frozen split beside "
         "(D-156; D-346 (4))",
    "h": "the injections (first round(0.25 n) items per pool and draw for A3-top, per pair for A3-benign) are stored "
         "in the npz and bound bit for bit by the simulation",
}


def m_of(rho: float, n: int) -> int:
    """The injected count at rho: 4d B's int(round(rho * n)) exactly (D-244 (6), D-332 (2))."""
    return int(round(rho * n))


def whole_item_k(target: float, n_total: int) -> int:
    """Choice (d): k = floor(alpha * (n + m)), the 1% rule in whole items (D-242)."""
    return int(math.floor(target * n_total))


def kth_largest(scores: np.ndarray, k: int) -> np.ndarray:
    """The k-th largest of each column, ties counted with multiplicity (choice (d)); +inf when k is 0 (no item may
    be flagged). scores: (N, S)."""
    s = np.asarray(scores, dtype=np.float64)
    if k <= 0:
        return np.full(s.shape[1], np.inf)
    return np.sort(s, axis=0)[s.shape[0] - k]


def predicted_tpr(pos: np.ndarray, t_prime: np.ndarray) -> np.ndarray:
    """THE prediction function of P-A3.1 (D-332 (5)): F1bar(t'), the share of the eval positives' post scores at or
    above t', per seed (choice (c)). pos: (n_pos, S); t_prime: (S,)."""
    return (np.asarray(pos) >= np.asarray(t_prime)).mean(axis=0)


def check_benign_disjoint(data: dict[str, dict[str, Any]]) -> None:
    """D-332 (3): A3-benign's candidates (the pre side's probe_train negatives) are disjoint from cal_d, E and R_fresh,
    which are drawn from threshold_calib and eval: no probe_train item ID may be in either (F-045). Refuses otherwise."""
    other = {str(i) for part in ("threshold_calib", "eval") for i in data["pre"][part]["item_ids"]}
    if {str(i) for i in data["pre"]["probe_train"]["item_ids"]} & other:
        raise SystemExit("REFUSING: a probe_train item is in threshold_calib or eval (F-045; A3-benign must be disjoint "
                         "from every pool, D-332 (3))")


def load_inputs(pair: str) -> dict[str, Any]:
    """The pair's inputs, checked as 4d B and A1 check them (phase3_a1_predictions.load_inputs without A1's transfer
    surrogate, whose cache A3 does not read): the selection of record (C and L*), the depth_d156 record (same selection
    and caches), the six caches through the registry (bytes against the extraction records), items and labels equal pre
    and post, no non-finite item, the cluster manifest's pooled-negative layout, and no probe_train item ID in
    threshold_calib or eval (A3-benign's candidates are probe_train negatives, disjoint from cal_d, E and R_fresh,
    D-332 (3))."""
    core = A1P.core_of(pair)
    reg_stem = D.registry_stem(f"depth_{core}")
    sel_path = D.SELECTION_OF_RECORD
    sel_sha, cl_sha = sha256_file(sel_path), sha256_file(D.CLUSTERS)
    sel_doc = read_result(sel_path)                          # registered read: the record lists it (D-297 (1))
    sel = next((m for m in sel_doc["models"] if m["pair"] == reg_stem), None)
    if sel is None or not sel["selected"]["arm"].startswith("C="):
        raise SystemExit(f"REFUSING: no logistic-C selection for {reg_stem}")
    C_sel, L_sel = float(sel["selected"]["arm"].split("=")[1]), int(sel["selected"]["layer"])
    depth_rel = f"results/depth_d156_{core}.json"
    depth_sha = sha256_file(ROOT / depth_rel)
    ddoc = read_result(ROOT / depth_rel)
    if (ddoc["config"]["selected_C"], ddoc["config"]["selected_layer"], ddoc["config"]["selection_sha256"]) != (C_sel, L_sel, sel_sha):
        raise SystemExit(f"REFUSING: {depth_rel} was run on another selection")
    cfg = REG.run(reg_stem)
    axis = cfg["axis"]
    k = f"layer_{L_sel}"
    data: dict[str, dict[str, Any]] = {"pre": {}, "post": {}}
    input_sha: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    for side in data:
        for part in ("probe_train", "threshold_calib", "eval"):
            z, r = REG.load(cfg[f"{side}_repo"], axis, part)
            if "nonfinite" in z.files and z["nonfinite"].any():
                raise SystemExit(f"REFUSING: {side} {part} has non-finite items")
            data[side][part] = {"X": z[k], "labels": z["labels"], "item_ids": z["item_ids"]}
            input_sha[f"{side}/{part}"], input_recs[f"{side}/{part}"] = r["sha256"], r
    if any(ddoc["input_cache_sha256"].get(kk) != h for kk, h in input_sha.items()):
        raise SystemExit(f"REFUSING: {depth_rel} read other caches")
    for part in ("probe_train", "threshold_calib", "eval"):
        a, b = data["pre"][part], data["post"][part]
        if not np.array_equal(a["item_ids"], b["item_ids"]) or not np.array_equal(a["labels"], b["labels"]):
            raise SystemExit(f"REFUSING: {part} items or labels differ between pre and post")
    y_cal = data["pre"]["threshold_calib"]["labels"].astype(bool)
    y_ev = data["pre"]["eval"]["labels"].astype(bool)
    pooled_ids = [str(i) for i in data["pre"]["threshold_calib"]["item_ids"][~y_cal]] + \
                 [str(i) for i in data["pre"]["eval"]["item_ids"][~y_ev]]
    cl = json.load(open(D.CLUSTERS, encoding="utf-8"))["axes"][axis]   # not a result (D-297): a manifest, hashed
    if cl["n_pooled_negatives"] != len(pooled_ids) or cl["n_calibration"] != int((~y_cal).sum()) or \
            cl["pool_item_ids_sha256"] != sha256_bytes("\n".join(pooled_ids).encode("utf-8")):
        raise SystemExit(f"REFUSING: manifests/negative_clusters.json does not describe the {axis} pool")
    check_benign_disjoint(data)
    reg_sources = REG.sources_sha256()
    if D.sources_now() != reg_sources:
        raise SystemExit("REFUSING: the extraction records or pins on disk are not the ones the registry read")
    return {"pair": pair, "core": core, "reg_stem": reg_stem, "axis": axis, "origin_strat": axis == PS.STRAT_AXIS,
            "cfg": cfg, "C_sel": C_sel, "L_sel": L_sel, "sel_path": sel_path, "sel_sha": sel_sha, "cl_sha": cl_sha,
            "depth_rel": depth_rel, "depth_sha": depth_sha, "data": data,
            "input_sha": input_sha, "input_recs": input_recs, "reg_sources": reg_sources,
            "verification": REG.verification(input_recs)}


def build(inp: dict[str, Any], n_reps: int, seeds: list[int], target: float) -> dict[str, Any]:
    """From the loaded arrays only (no file is read): 4d B's pooled layout, partitions and pools (poolstudy_d156's
    functions; option B for Aegis, D-298), the D-117 selected probes, scores through the raw cache arrays (exactly as 4d
    B scores, so an A3-top number can equal its record's), T_d, and the attackers' orders (A3-top per pool and draw,
    A3-benign per pair). Score-table indices: pooled negatives 0..n_pool-1, H item j at n_pool + j (4d B's), A3-benign
    candidate i (the i-th probe_train negative in file order) at n_pool + n_h + i."""
    data = inp["data"]
    y_tr = data["pre"]["probe_train"]["labels"].astype(bool)
    y_cal = data["pre"]["threshold_calib"]["labels"].astype(bool)
    y_ev = data["pre"]["eval"]["labels"].astype(bool)
    ncal, hcal, neg, pos = ~y_cal, y_cal, ~y_ev, y_ev
    n_cal = int(ncal.sum())
    n_pool = n_cal + int(neg.sum())
    if n_pool - n_cal != 2000:
        raise SystemExit(f"REFUSING: 4d B's R_FRESH takes 1000 of 2000 evaluation negatives; this pool has {n_pool - n_cal}")
    n_h_src = int(hcal.sum())
    ptr_neg = np.flatnonzero(~y_tr)                          # A3-benign's candidates: probe_train negatives (file order)
    origin_strat = inp["origin_strat"]
    parts = PS.partitions(n_cal, n_pool, n_reps, inp["axis"], origin_strat)
    draws: list[dict[str, np.ndarray]] = []
    h = hashlib.sha256()
    for di, (cal, ev) in enumerate(parts):
        dp = PS.draw_pools(di, cal, ev, n_cal, n_pool, n_h_src, origin_strat)
        tr = dp["pools"].get(TRAFFIC)
        if tr is None:
            raise SystemExit(f"REFUSING: draw {di} has no {TRAFFIC} pool: {dp['skipped']}")
        dr = {"cal": cal, "ev": ev, "R_fresh": dp["R_fresh"], "E": dp["E"], "traffic": tr}
        if origin_strat and di > 0:                          # D-298's counts, as 4d B checks them
            tc = lambda idx: [int((idx < n_cal).sum()), int((idx >= n_cal).sum())]  # noqa: E731
            got = {"cal": tc(cal), "R_fresh": tc(dr["R_fresh"]), "E": tc(dr["E"])}
            if got != {kk: PS.D298_COUNTS[kk] for kk in got}:
                raise SystemExit(f"REFUSING: draw {di} is off D-298's counts: {got}")
        for kk in ("cal", "ev", "R_fresh", "E", "traffic"):
            h.update(np.asarray(dr[kk], dtype="<i8").tobytes())
        draws.append(dr)

    f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
    RAW = {s: {"pool": np.vstack([data[s]["threshold_calib"]["X"][ncal], data[s]["eval"]["X"][neg]]),
               "H": data[s]["threshold_calib"]["X"][hcal], "pos": data[s]["eval"]["X"][pos]} for s in ("pre", "post")}
    RAW["post"]["B"] = data["post"]["probe_train"]["X"][ptr_neg]        # A3-benign: the post model's features
    Xtr = {s: data[s]["probe_train"]["X"] for s in ("pre", "post")}
    probes, rows_of = [], []
    for seed in seeds:
        rows, C, order = PS.probe_spec(len(y_tr), seed, PSET, inp["C_sel"])
        probes.append(P.fit_probe(Xtr["pre"][rows], y_tr[rows], seed, C, order))
        rows_of.append(rows)
    SC = {side: {part: np.column_stack([q.score(RAW[side][part]) for q in probes]) for part in ("pool", "H", "pos")}
          for side in ("pre", "post")}
    SC["post"]["B"] = np.column_stack([q.score(RAW["post"]["B"]) for q in probes])
    pre_table = np.vstack([SC["pre"]["pool"], SC["pre"]["H"]])                       # 4d B's all_pre
    post_table = np.vstack([SC["post"]["pool"], SC["post"]["H"], SC["post"]["B"]])   # 4d B's all_post, then B
    nD, nS = len(draws), len(seeds)
    T = np.empty((nD, nS))
    for d, dr in enumerate(draws):
        for j in range(nS):
            T[d, j] = P.thr_at_fpr(SC["pre"]["pool"][dr["cal"], j], target)

    # the attackers' orders: the attacker's choice from post scores, no repair
    all_post_4d = np.vstack([SC["post"]["pool"], SC["post"]["H"]])                  # 4d B's all_post, as it was
    a1_all = np.argsort(-all_post_4d[n_pool:].mean(axis=1))                         # 4d B's A1: poolstudy_d156.main()
    b_all = np.argsort(-SC["post"]["B"].mean(axis=1))                               # choice (e): the same expression
    b0 = n_pool + n_h_src
    m_max = {pool: m_of(RHOS[-1], len(draws[0][POOL_OF_DRAW[pool]])) for pool in POOLS}
    top: dict[str, list[np.ndarray]] = {pool: [] for pool in POOLS}
    for dr in draws:
        for pool in POOLS:
            tidx = dr[POOL_OF_DRAW[pool]]
            if len(tidx) != PS.DEFAULT_POOL:
                raise SystemExit(f"REFUSING: {pool} holds {len(tidx)} items, not D-332's n = {PS.DEFAULT_POOL}")
            fresh = np.setdiff1d(np.arange(n_h_src), tidx[tidx >= n_pool] - n_pool, assume_unique=True)
            order = n_pool + a1_all[np.isin(a1_all, fresh)]                         # table indices, 4d B's A1 order
            if len(order) < m_max[pool]:
                raise SystemExit(f"REFUSING: {pool} has {len(order)} harmful items to inject; rho 0.25 needs {m_max[pool]}")
            top[pool].append(order)
    benign = b0 + b_all
    if len(benign) < max(m_max.values()):
        raise SystemExit(f"REFUSING: {len(benign)} probe_train negatives; rho 0.25 needs {max(m_max.values())}")
    hi = hashlib.sha256()
    for pool in POOLS:
        for o in top[pool]:
            hi.update(np.asarray(o, dtype="<i8").tobytes())
    hi.update(np.asarray(benign, dtype="<i8").tobytes())
    fit_share = [float(np.isin(ptr_neg[b_all[:max(m_max.values())]], rows).mean()) for rows in rows_of]
    return {**{kk: vv for kk, vv in inp.items() if kk != "data"},
            "seeds": list(seeds), "target": target, "n_reps": n_reps,
            "y_tr": y_tr, "n_cal": n_cal, "n_pool": n_pool, "n_h_src": n_h_src, "n_pos": int(pos.sum()),
            "n_benign_candidates": int(len(ptr_neg)), "ptr_neg": ptr_neg,
            "draws": draws, "draws_sha256": h.hexdigest(), "RAW": RAW, "Xtr": Xtr, "f64": f64, "probes": probes,
            "rows_of": rows_of, "SC": SC, "pre_table": pre_table, "post_table": post_table, "T": T,
            "a1_all": a1_all, "top_order": top, "benign_order": benign, "m_max": m_max,
            "injections_sha256": hi.hexdigest(), "benign_fit_row_share_per_seed": fit_share}


def setup(pair: str, n_reps: int, seeds: list[int], target: float) -> dict[str, Any]:
    """Everything the predictions and the simulation share (load_inputs, then build)."""
    return build(load_inputs(pair), n_reps, seeds, target)


def injected(S: dict[str, Any], attacker: str, pool: str, d: int) -> np.ndarray:
    """The attacker's injection order (table indices) for draw d and pool, highest seed-mean post score first."""
    if attacker == "A3-top":
        return S["top_order"][pool][d]
    if attacker == "A3-benign":
        return S["benign_order"]
    raise ValueError(f"no fixed order for {attacker}")


def ckey(pool: str, attacker: str, rho: float) -> str:
    return f"{pool}|{attacker}|rho={rho}"


def prediction_arrays(S: dict[str, Any]) -> dict[str, np.ndarray]:
    """Every per-draw array of the record (draws x seeds; row 0 the frozen split): T, and per pool, attacker and rho the
    predicted threshold t' and the predicted TPR F1bar(t'); the injections (first m_max per pool and draw for A3-top,
    per pair for A3-benign)."""
    out: dict[str, np.ndarray] = {"T": S["T"]}
    nD, nS = S["T"].shape
    pos = S["SC"]["post"]["pos"]
    for pool in POOLS:
        for att in PREDICTED:
            for rho in RHOS:
                tp, tpr, mm = np.empty((nD, nS)), np.empty((nD, nS)), set()
                for d, dr in enumerate(S["draws"]):
                    tidx = dr[POOL_OF_DRAW[pool]]
                    m = m_of(rho, len(tidx))
                    mm.add(m)
                    sc = S["post_table"][np.concatenate([tidx, injected(S, att, pool, d)[:m]])]
                    tp[d] = kth_largest(sc, whole_item_k(S["target"], len(tidx) + m))
                    tpr[d] = predicted_tpr(pos, tp[d])
                if len(mm) != 1:
                    raise SystemExit(f"REFUSING: {pool} gives different m across draws at rho {rho}: {sorted(mm)}")
                out[f"{ckey(pool, att, rho)}|t_prime"] = tp
                out[f"{ckey(pool, att, rho)}|predicted_tpr"] = tpr
        out[f"inject|{pool}|A3-top"] = np.stack([o[:S["m_max"][pool]] for o in S["top_order"][pool]])
    out["inject|A3-benign"] = S["benign_order"][:max(S["m_max"].values())]
    return {kk: (np.ascontiguousarray(vv, dtype="<i8") if kk.startswith("inject|") else
                 np.ascontiguousarray(vv, dtype="<f8")) for kk, vv in out.items()}


def predicted_cells(S: dict[str, Any], arr: dict[str, np.ndarray]) -> dict[str, Any]:
    """Per pool, attacker and rho: m, k and the predicted TPR (the per-draw mean over seeds) as the D-156 mean and band
    (choice (g))."""
    out: dict[str, Any] = {}
    n = PS.DEFAULT_POOL
    for pool in POOLS:
        for att in PREDICTED:
            for rho in RHOS:
                raw = arr[f"{ckey(pool, att, rho)}|predicted_tpr"].mean(axis=1)
                m = m_of(rho, n)
                out.setdefault(pool, {}).setdefault(att, {})[f"rho={rho}"] = {
                    "m": m, "k": whole_item_k(S["target"], n + m), "predicted_tpr": P.band_summary(raw[1:], raw[0])}
    return out


def input_block(S: dict[str, Any]) -> dict[str, Any]:
    """The config entries naming every cache input's repo ID and revision (D-297 (1)), and the run's population."""
    cfg = S["cfg"]
    return {"pair": S["pair"], "registry_pair": S["reg_stem"], "axis": S["axis"], "layer": S["L_sel"],
            "selected_C": S["C_sel"], "probe_set": PSET,
            "selection_file": A1P.rel(S["sel_path"]), "selection_sha256": S["sel_sha"],
            "seeds": S["seeds"], "target_fpr": S["target"], "n_reps": S["n_reps"],
            "draws": (f"frozen split + {S['n_reps']} origin-stratified draws (D-298, poolstudy_d156.partitions)"
                      if S["origin_strat"] else f"frozen split + the first {S['n_reps']} of paired.repartition_draws "
                                                f"(seed {P.AXIS_DRAW_SEED[S['axis']]})"),
            "design": "option B, origin-stratified (D-298)" if S["origin_strat"] else "as registered (D-283)",
            "pool_seed": PS.POOL_SEED, "evaluation_half": "E: the draw's evaluation part minus R_FRESH (1000 negatives)",
            "pools": {"R_same": "cal_d's 1000 negatives; the poisoning models contamination of the post-update "
                                "re-scoring only (T_d and the band's pre moments come from the clean cal_d)",
                      "R_fresh": "the draw's R_FRESH (1000)",
                      "traffic_1pct": f"4d B's {TRAFFIC}"},
            "pool_n": PS.DEFAULT_POOL, "rhos": list(RHOS), "m_by_rho": {f"rho={r}": m_of(r, PS.DEFAULT_POOL) for r in RHOS},
            "predicted_attackers": list(PREDICTED),
            "attacker_sources": {"A3-top": "threshold_calib harmful H not in the pool, 4d B's A1 order",
                                 "A3-benign": f"the {S['n_benign_candidates']} probe_train negatives (pre side's split)"},
            "draws_index_sha256": S["draws_sha256"], "injections_sha256": S["injections_sha256"],
            "cluster_manifest": {"file": "manifests/negative_clusters.json", "sha256": S["cl_sha"]},
            **{kk: cfg[kk] for kk in ("pre_repo", "pre_rev", "post_repo", "post_rev")},
            "input_registry_sources_sha256": S["reg_sources"]}


def changed_inputs(S: dict[str, Any], code0: dict[str, Any], files: list[str], extra: list[tuple[str, str, str]]) -> list[str]:
    """What changed on disk during the run: the code, the selection, the cluster manifest, the depth record, the
    registry's sources, every cache, and `extra` (name, now, then)."""
    code1 = P.code_block(files)
    changed = [n for n, a, b in (("code", code1["sha256"], code0["sha256"]), ("selection", sha256_file(S["sel_path"]), S["sel_sha"]),
                                 ("cluster manifest", sha256_file(D.CLUSTERS), S["cl_sha"]),
                                 (S["depth_rel"], sha256_file(ROOT / S["depth_rel"]), S["depth_sha"]),
                                 ("registry sources", D.sources_now(), S["reg_sources"]), *extra) if a != b]
    changed += [f"cache {r['file']}" for r in S["input_recs"].values()
                if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    return changed


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Phase 3 A3 predictions (D-332 P-A3.1), before any poisoned repair")
    ap.add_argument("--pair", required=True, choices=POP.UPDATE_PAIRS_REPAIR)
    ap.add_argument("--n-reps", type=int, default=REGISTERED["n_reps"])
    ap.add_argument("--target-fpr", type=float, default=REGISTERED["target_fpr"])
    ap.add_argument("--seeds", type=int, nargs="*", default=REGISTERED["seeds"])
    ap.add_argument("--out", default=None, help="TESTING: a JSON path outside results/")
    args = ap.parse_args()
    core = A1P.core_of(args.pair)
    registered = (args.n_reps == REGISTERED["n_reps"] and args.target_fpr == REGISTERED["target_fpr"]
                  and list(args.seeds) == REGISTERED["seeds"])
    final_path = (RESULTS / f"phase3_a3_predictions_{core}.json").resolve()
    out_path = Path(args.out).resolve() if args.out else final_path
    npz_path = out_path.with_name(out_path.stem + "_draws.npz")
    final = out_path == final_path
    for p in (out_path, npz_path):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists; results are never overwritten")
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test run (--out) must not write under results/")
    threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    if final and (not registered or any(t != "6" for t in threads.values())):
        raise SystemExit(f"REFUSING the final name: registered parameters {registered}, threads {threads}")
    code0 = P.code_block(CODE_FILES)
    S = setup(args.pair, args.n_reps, list(args.seeds), args.target_fpr)
    print(f"A3 predictions {args.pair} [{S['axis']}] L{S['L_sel']} C {S['C_sel']:g}: {len(S['draws']) - 1} draws + "
          f"frozen; pools {list(POOLS)}; attackers {list(PREDICTED)}; registered {registered}", flush=True)
    arr = prediction_arrays(S)
    nb = A1P.npz_bytes(arr)
    changed = changed_inputs(S, code0, CODE_FILES, [])
    if changed:
        raise SystemExit(f"REFUSING to write: changed during the run: {changed}")
    config = input_block(S)
    payload = {**provenance_block("phase3_a3_predictions"), "preregistration": PREREG, "code": code0,
               "registered_parameters": registered, "blas_threads": threads, "input_verification": S["verification"],
               "config": config, "config_hash": canonical_hash(config), "sandbox_disclosure": SANDBOX,
               "prediction_function": {"module": THIS_FILE, "function": "predicted_tpr", "formula": FORMULA,
                                       "sha256": code0["sha256"][THIS_FILE],
                                       "threshold": "kth_largest(clean pool + injected post scores, whole_item_k)"},
               "order": "computes no repair on a poisoned pool (t' is the stated order statistic by np.sort, not "
                        "paired.thr_at_fpr, quantile_transfer or restandardisation); written before this pair's "
                        "simulation, which refuses without it (D-332 (5), (7))",
               "implementation_choices": CHOICES,
               "depth_record": {"file": S["depth_rel"], "sha256": S["depth_sha"], "role": "selection and cache check"},
               "benign_fit_row_share_per_seed": S["benign_fit_row_share_per_seed"],
               "probe_fit": [q.diag for q in S["probes"]],
               "input_cache_sha256": S["input_sha"],
               "input_cache_source": {kk: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"]}
                                      for kk, r in S["input_recs"].items()},
               "draws_npz": {"file": A1P.rel(npz_path), "sha256": sha256_bytes(nb), "size": len(nb), "keys": sorted(arr),
                             "rows": "draw index (0 the frozen split); columns the seeds in config order; inject|* "
                                     "rows are draws (A3-top) and columns the order, highest seed-mean post score first"},
               "predicted": predicted_cells(S, arr)}
    for p, hh in A1P.write_pair(out_path, payload, npz_path, nb):
        print(f"Written: {p}\nsha256:  {hh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
