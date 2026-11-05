"""Phase 3 attacker A1, score translation in feature space (D-327), part 1 of 2: the PREDICTIONS, one pair per run.

Written before any attacked score exists (D-327 (4) P-A1.3, (6)). From c = 0 data and the attack geometry ONLY, for
every draw d (the frozen split first, then 4d B's 200 draws) and seed s of the pair's D-117 selected probe at L*:
  predicted FPR_ds(c) = 1 - Phi((T_ds - m_ds - Delta_ds(c)) / s_ds)                       (predicted_fpr below)
  T_ds     the frozen threshold: seed s's 1% threshold on the PRE scores of cal_d (as 4d B sets it)
  m, s     the mean and SD of the POST scores at c = 0 of the draw's evaluation half E (4d B's 1000 negatives)
  Delta    the score shift: white-box c * sigma_ds; transfer delta . v, delta = c * sigma_ds * (1 / ||v||) * v_sur /
           ||v_sur|| (the white-box delta's norm, the surrogate's direction), computed per draw and seed
  sigma_ds the SD of the pre scores of cal_d's negatives (D-327 (2))
The record is results/phase3_a1_predictions_<pair>.json (written once, exclusive) with its per-draw arrays in
results/phase3_a1_predictions_<pair>_draws.npz (tracked, D-163 (6); its sha256 and size in the JSON). The simulation
(src/phase3_a1.py) refuses unless this record exists, names the same pair, draws, inputs and this file's sha256 (the
prediction function), and its arrays equal the simulation's own recomputation bit for bit.

MACHINERY (D-327 (1), (6); D-152 P3, one definition shared): the inputs, probes, partitions and pools are 4d B's.
src/poolstudy_d156.py's probe_spec, partitions and draw_pools give the D-117 probe rule, the frozen split plus 200
draws (option B, D-298, for the seven Aegis pairs; the registered draws for the two WildGuardMix pairs, as the 4d B
record of record, populations.POOLSTUDY_D156) and each draw's R_FRESH, E and 1% traffic pool (n 1000); src/paired.py
fits and scores. setup() below is shared with the simulation.

CHOICES BY THE CLOSEST D-ENTRY (D-181; listed in the run report):
  (a) probe set: the selected C only (D-327 (1) "the selected C"); C=1 is not run.
  (b) sigma, m and s are population SDs (ddof 0), as the 4c health ratio's moments (healthcheck_d156.health, D-241 (2)).
  (c) white-box Delta in the prediction is c * sigma exactly, as D-327 writes it; the realised delta . v of the
      materialised delta is recorded beside, with its largest deviation from c * sigma.
  (d) TRANSFER population: every pair whose pre model has Phase 1 caches on both axes (D-327 (2): "where a pre model
      has caches on both axes"): the Aegis pairs mistral, llama3 and llama3_to_31 (pre Llama-3-8B) with a
      WildGuardMix surrogate, and the WildGuardMix pairs with an Aegis surrogate. The surrogate is fitted on the same
      pre model's OTHER-axis probe_train at the defender's L*, per seed by the D-117 rule (the seed's 80% fit rows in
      permutation order, as the selected probe; paired seeds), with C chosen by the D-117 criterion at that layer:
      the argmax over the selection record's arms of the mean ho20 AUROC at L* (results/selection_sweep_d165.json,
      the other-axis pair(s) of the same pre model; strict '>', record order, so ties go to the first arm, as
      regularisation_sweep selects); refused unless the chosen arm is a logistic C (as poolstudy refuses).
  (e) the prediction is unfloored as D-327 writes it; a beside convention floors each seed's predicted FPR at the
      realised resolution 1/|E| (D-163 (2), F-034), and both accuracy lines are reported (D-322 (4): both conventions).

Only registered parameters (200 draws, 1% target, seeds 0-4) at the 6-thread BLAS pin write the final name; a test
writes outside results/.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_d156 as D  # noqa: E402
import input_registry as REG  # noqa: E402
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
import poolstudy_d156 as PS  # noqa: E402
from provenance import (  # noqa: E402
    _repo_rel, commit_staged, ensure_utf8_stdout, input_identity_problems, provenance_block, read_result, resolve_cache,
    sha256_bytes, sha256_file, stage_bytes,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
THIS_FILE = "src/phase3_a1_predictions.py"
CODE_FILES = [THIS_FILE, *PS.CODE_FILES]
C_GRID = (-3.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0)    # benign-score SD units (D-318 (9), D-327 (2))
PSET = "selected"                                                          # choice (a)
TRAFFIC = f"traffic|pi=0.01|n={PS.DEFAULT_POOL}"                           # 4d B's 1% traffic pool at n 1000
REGISTERED = {"n_reps": PS.N_REPS, "target_fpr": P.TARGET_FPR, "seeds": [0, 1, 2, 3, 4]}
PREREG = ("D-327 (the Phase 3 A1 pre-registration: (1) population, (2) attack, (4) P-A1.3, (6) predictions first); "
          "D-318 (9); docs/phase3_design.md revision 2")
FORMULA = "predicted FPR = 1 - Phi((T_d - m - Delta) / s), per draw and seed (D-327 P-A1.3)"


def ckey(c: float) -> str:
    return f"c={c:g}"


def core_of(pair: str) -> str:
    return pair[len("repair_"):]


def record_of_record_stem(core: str) -> str:
    """The 4d B record of record for the pair (populations.POOLSTUDY_D156: option B for Aegis, _r2 for WildGuardMix)."""
    hit = [s for s in POP.POOLSTUDY_D156 if s in (f"poolstudy_d156_{core}_originstrat", f"poolstudy_d156_{core}_r2")]
    if len(hit) != 1:
        raise SystemExit(f"REFUSING: populations.POOLSTUDY_D156 names {len(hit)} records for {core}: {hit}")
    return hit[0]


def transfer_sources(pair: str) -> list[str]:
    """Choice (d): the declared pairs with this pair's pre model on the other axis (their caches are the surrogate's)."""
    pre, _, axis = REG.PAIRS[pair]
    return [p for p, (pr, _, ax) in REG.PAIRS.items() if pr == pre and ax != axis]


def surrogate_selection(sel_doc: dict[str, Any], stems: list[str], layer: int) -> dict[str, Any]:
    """Choice (d): the D-117 criterion at the fixed layer: argmax over the record's arms of the mean ho20 AUROC there
    (strict '>', record order). Every source stem must give the same arm (they share one pre model and axis)."""
    picks = []
    for st in stems:
        m = next((x for x in sel_doc["models"] if x["pair"] == st), None)
        if m is None:
            raise SystemExit(f"REFUSING: the selection record has no entry for {st}")
        best: tuple[str, float] | None = None
        for arm, per_layer in m["grid"].items():
            if str(layer) not in per_layer:
                raise SystemExit(f"REFUSING: {st}'s selection grid has no layer {layer} for {arm}")
            h = float(per_layer[str(layer)]["auroc_ho20"])
            if best is None or h > best[1]:
                best = (arm, h)
        picks.append(best)
    if len({p[0] for p in picks}) != 1:
        raise SystemExit(f"REFUSING: the surrogate's source pairs {stems} select different arms at layer {layer}: {picks}")
    arm, h = picks[0]
    if not arm.startswith("C="):
        raise SystemExit(f"REFUSING: the surrogate's D-117 arm at layer {layer} is {arm}, not a logistic C")
    return {"arm": arm, "C": float(arm.split("=")[1]), "layer": layer, "mean_ho20_auroc": h, "source_pairs": stems,
            "rule": "D-117 criterion at the defender's L*: argmax over the selection record's arms of the mean ho20 "
                    "AUROC at that layer, strict '>' in record order (ties to the first arm, as regularisation_sweep)"}


def predicted_fpr(T: np.ndarray, m: np.ndarray, s: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """THE prediction function of P-A1.3 (D-327 (4)): the two-moment (Gaussian) frozen FPR after a score shift delta,
    1 - Phi((T - m - delta) / s), elementwise (the survival function, so the far tail does not round to 0 early)."""
    return norm.sf((np.asarray(T, dtype=np.float64) - m - delta) / s)


def attack_vector(S: dict[str, Any], variant: str, c: float, d: int, s: int) -> np.ndarray:
    """delta for draw d and seed s (D-327 (2)): white-box c * sigma_ds * v / (v . v); transfer the same norm along the
    surrogate's direction, c * sigma_ds * v_sur / (||v_sur|| ||v||). Zero at c = 0."""
    return (c * float(S["sigma"][d, s])) * S["unit"][variant][s]


def load_inputs(pair: str) -> dict[str, Any]:
    """The pair's inputs, checked as 4d B checks them: the selection of record (C and L*), the depth_d156 record (same
    selection and caches; the AUC rule's source), the six caches through the registry (bytes against the extraction
    records), items and labels equal pre and post, no non-finite item, the cluster manifest's pooled-negative layout,
    and, where choice (d) applies, the surrogate's cache (the same pre model's other-axis probe_train) and its C."""
    core = core_of(pair)
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
    dcell = next((c for c in ddoc["cells"] if c["layer"] == L_sel and c["probe_set"] == PSET), None)
    if dcell is None:
        raise SystemExit(f"REFUSING: {depth_rel} has no {PSET} cell at layer {L_sel}")
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
    if {str(i) for i in data["pre"]["probe_train"]["item_ids"]} & set(pooled_ids):
        raise SystemExit("REFUSING: a probe_train item is in the pooled negatives (F-045)")
    sur: dict[str, Any] | None = None
    srcs = transfer_sources(reg_stem)
    if srcs:
        s_axis = REG.PAIRS[srcs[0]][2]
        zs, rs = REG.load(cfg["pre_repo"], s_axis, "probe_train")
        if k not in zs.files or ("nonfinite" in zs.files and zs["nonfinite"].any()):
            raise SystemExit(f"REFUSING: the surrogate's cache has no {k} or has non-finite items")
        input_sha["surrogate/probe_train"], input_recs["surrogate/probe_train"] = rs["sha256"], rs
        sur = {"X": zs[k], "labels": zs["labels"], "axis": s_axis, "cache": rs["file"],
               "selection": surrogate_selection(sel_doc, srcs, L_sel)}
    reg_sources = REG.sources_sha256()
    if D.sources_now() != reg_sources:
        raise SystemExit("REFUSING: the extraction records or pins on disk are not the ones the registry read")
    return {"pair": pair, "core": core, "reg_stem": reg_stem, "axis": axis, "origin_strat": axis == PS.STRAT_AXIS,
            "cfg": cfg, "C_sel": C_sel, "L_sel": L_sel, "sel_path": sel_path, "sel_sha": sel_sha, "cl_sha": cl_sha,
            "depth_rel": depth_rel, "depth_sha": depth_sha, "dcell": dcell, "data": data, "surrogate_in": sur,
            "input_sha": input_sha, "input_recs": input_recs, "reg_sources": reg_sources,
            "verification": REG.verification(input_recs)}


def build(inp: dict[str, Any], n_reps: int, seeds: list[int], target: float) -> dict[str, Any]:
    """From the loaded arrays only (no file is read): 4d B's pooled layout, partitions and pools (poolstudy_d156's
    functions; the record of record's design: option B for Aegis, D-298), the D-117 selected probes, scores through the
    raw cache arrays (exactly as 4d B scores, so a c = 0 number can equal its record's), T_d, sigma, the c = 0 moments
    of E and the attack geometry (white-box; the transfer surrogate where choice (d) applies)."""
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
    Xtr = {s: data[s]["probe_train"]["X"] for s in ("pre", "post")}
    probes, rows_of = [], []
    for seed in seeds:
        rows, C, order = PS.probe_spec(len(y_tr), seed, PSET, inp["C_sel"])
        probes.append(P.fit_probe(Xtr["pre"][rows], y_tr[rows], seed, C, order))
        rows_of.append(rows)
    SC = {side: {part: np.column_stack([q.score(RAW[side][part]) for q in probes]) for part in ("pool", "H", "pos")}
          for side in ("pre", "post")}
    v = [q.w / q.sd for q in probes]                         # the frozen probe's direction in raw feature space
    unit: dict[str, list[np.ndarray]] = {"white_box": [vv / float(vv @ vv) for vv in v]}
    geometry: dict[str, Any] = {"v_norm_per_seed": [float(np.linalg.norm(vv)) for vv in v]}
    sur: dict[str, Any] | None = None
    si = inp.get("surrogate_in")
    if si is not None:
        ys, Xs = si["labels"].astype(bool), si["X"]
        sprobes = []
        for seed in seeds:
            rows, C, order = PS.probe_spec(len(ys), seed, PSET, si["selection"]["C"])
            sprobes.append(P.fit_probe(Xs[rows], ys[rows], seed, C, order))
        vs = [q.w / q.sd for q in sprobes]
        unit["transfer"] = [vsur / (float(np.linalg.norm(vsur)) * float(np.linalg.norm(vv))) for vsur, vv in zip(vs, v)]
        geometry["cos_v_surrogate_per_seed"] = [float(vv @ vsur / (np.linalg.norm(vv) * np.linalg.norm(vsur)))
                                                for vv, vsur in zip(v, vs)]
        geometry["surrogate_v_norm_per_seed"] = [float(np.linalg.norm(vsur)) for vsur in vs]
        sur = {"axis": si["axis"], "selection": si["selection"], "cache": si["cache"],
               "fit": [q.diag for q in sprobes], "n_probe_train": int(len(ys))}
    nD, nS = len(draws), len(seeds)
    T, sigma, m, s_ = (np.empty((nD, nS)) for _ in range(4))
    for d, dr in enumerate(draws):
        pre_cal = SC["pre"]["pool"][dr["cal"]]
        post_E = SC["post"]["pool"][dr["E"]]
        for j in range(nS):
            T[d, j] = P.thr_at_fpr(pre_cal[:, j], target)
        sigma[d], m[d], s_[d] = pre_cal.std(axis=0), post_E.mean(axis=0), post_E.std(axis=0)   # choice (b): ddof 0
    return {**{kk: vv for kk, vv in inp.items() if kk not in ("data", "surrogate_in")},
            "seeds": list(seeds), "target": target, "n_reps": n_reps,
            "y_tr": y_tr, "n_cal": n_cal, "n_pool": n_pool, "n_h_src": n_h_src, "n_pos": int(pos.sum()),
            "draws": draws, "draws_sha256": h.hexdigest(), "RAW": RAW, "Xtr": Xtr, "f64": f64, "probes": probes,
            "rows_of": rows_of, "SC": SC, "v": v, "unit": unit, "geometry": geometry, "surrogate": sur,
            "variants": [x for x in ("white_box", "transfer") if x in unit], "T": T, "sigma": sigma, "m": m, "s": s_}


def setup(pair: str, n_reps: int, seeds: list[int], target: float) -> dict[str, Any]:
    """Everything the predictions and the simulation share (load_inputs, then build)."""
    return build(load_inputs(pair), n_reps, seeds, target)


def prediction_arrays(S: dict[str, Any]) -> dict[str, np.ndarray]:
    """Every per-draw array of the record: T, sigma, m, s, and per variant and c the score shift and the predicted FPR
    (draws x seeds; row 0 the frozen split); the realised white-box shift delta . v beside its stated c * sigma."""
    out = {"T": S["T"], "sigma": S["sigma"], "m": S["m"], "s": S["s"]}
    nD, nS = S["T"].shape
    for variant in S["variants"]:
        for c in C_GRID:
            real = np.array([[float(attack_vector(S, variant, c, d, j) @ S["v"][j]) for j in range(nS)] for d in range(nD)])
            shift = c * S["sigma"] if variant == "white_box" else real          # choice (c)
            out[f"{variant}|{ckey(c)}|delta_score_shift"] = shift
            out[f"{variant}|{ckey(c)}|predicted_fpr"] = predicted_fpr(S["T"], S["m"], S["s"], shift)
            if variant == "white_box":
                out[f"{variant}|{ckey(c)}|realised_delta_dot_v"] = real
    return {kk: np.ascontiguousarray(vv, dtype="<f8") for kk, vv in out.items()}


def predicted_cells(S: dict[str, Any], arr: dict[str, np.ndarray]) -> dict[str, Any]:
    """Per variant and c: the predicted FPR/target per draw (the seed mean of the per-seed predicted FPR, over the
    target), unfloored (D-327) and floored per seed at 1/|E| (choice (e)), as the D-156 mean and band."""
    n_e = len(S["draws"][0]["E"])
    out: dict[str, Any] = {}
    for variant in S["variants"]:
        for c in C_GRID:
            pf = arr[f"{variant}|{ckey(c)}|predicted_fpr"]
            raw = pf.mean(axis=1) / S["target"]
            fl = np.maximum(pf, 1.0 / n_e).mean(axis=1) / S["target"]
            sh = arr[f"{variant}|{ckey(c)}|delta_score_shift"].mean(axis=1)
            out.setdefault(variant, {})[ckey(c)] = {
                "predicted_ratio_to_target": P.band_summary(raw[1:], raw[0]),
                "predicted_ratio_to_target_floored_1_over_E": P.band_summary(fl[1:], fl[0]),
                "delta_score_shift_mean_over_seeds": P.band_summary(sh[1:], sh[0])}
    return out


def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def record_bytes(payload: dict[str, Any], name: str) -> bytes:
    """The JSON bytes exactly as provenance.write_results writes them, after its D-297 (1) results-writer check."""
    problems = input_identity_problems(payload)
    if problems:
        raise SystemExit(f"REFUSING to write {name} (D-297 (1) results-writer rule): {problems}")
    b = json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8")
    json.loads(b.decode("utf-8"))           # not a result (D-297): the round trip of the bytes about to be written
    return b


def write_pair(json_path: Path, payload: dict[str, Any], npz_path: Path, npz_b: bytes) -> list[tuple[Path, str]]:
    """Stage both outputs (written, fsynced, verified) before either final name exists, then link each (exclusive:
    an existing name refuses); the JSON's bytes pass the D-297 (1) rule first."""
    jb = record_bytes(payload, json_path.name)
    staged: list[tuple[Path, Path, str]] = []
    try:
        for p, b in ((npz_path, npz_b), (json_path, jb)):
            t, hh = stage_bytes(p, b)
            staged.append((t, p, hh))
        for t, p, hh in staged:
            commit_staged(t, p, hh)
    finally:
        for t, _, _ in staged:
            if t.exists():
                t.unlink()
    return [(p, hh) for _, p, hh in staged]


def rel(p: Path) -> str:
    """A path as the D-297 (1) results-writer rule names it (provenance._repo_rel): repo-relative inside the repo, else
    its file name, so a record read from outside the repo (a test's predictions record) is listed under the name the
    rule registered it by."""
    return _repo_rel(Path(p))


def input_block(S: dict[str, Any]) -> dict[str, Any]:
    """The config entries naming every cache input's repo ID and revision (D-297 (1)), and the run's population."""
    cfg = S["cfg"]
    out = {"pair": S["pair"], "registry_pair": S["reg_stem"], "axis": S["axis"], "layer": S["L_sel"],
           "selected_C": S["C_sel"], "probe_set": PSET,
           "selection_file": rel(S["sel_path"]), "selection_sha256": S["sel_sha"],
           "seeds": S["seeds"], "target_fpr": S["target"], "n_reps": S["n_reps"],
           "draws": (f"frozen split + {S['n_reps']} origin-stratified draws (D-298, poolstudy_d156.partitions)"
                     if S["origin_strat"] else f"frozen split + the first {S['n_reps']} of paired.repartition_draws "
                                               f"(seed {P.AXIS_DRAW_SEED[S['axis']]})"),
           "design": "option B, origin-stratified (D-298)" if S["origin_strat"] else "as registered (D-283)",
           "pool_seed": PS.POOL_SEED, "evaluation_half": "E: the draw's evaluation part minus R_FRESH (1000 negatives)",
           "traffic_pool": TRAFFIC, "draws_index_sha256": S["draws_sha256"],
           "c_grid": list(C_GRID), "variants": S["variants"],
           "cluster_manifest": {"file": "manifests/negative_clusters.json", "sha256": S["cl_sha"]},
           **{kk: cfg[kk] for kk in ("pre_repo", "pre_rev", "post_repo", "post_rev")},
           "input_registry_sources_sha256": S["reg_sources"]}
    if S["surrogate"] is not None:
        out.update({"surrogate_repo": cfg["pre_repo"], "surrogate_rev": cfg["pre_rev"],
                    "surrogate_axis": S["surrogate"]["axis"]})
    return out


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Phase 3 A1 predictions (D-327 P-A1.3), before any attacked score")
    ap.add_argument("--pair", required=True, choices=POP.UPDATE_PAIRS_REPAIR)
    ap.add_argument("--n-reps", type=int, default=REGISTERED["n_reps"])
    ap.add_argument("--target-fpr", type=float, default=REGISTERED["target_fpr"])
    ap.add_argument("--seeds", type=int, nargs="*", default=REGISTERED["seeds"])
    ap.add_argument("--out", default=None, help="TESTING: a JSON path outside results/")
    args = ap.parse_args()
    core = core_of(args.pair)
    registered = (args.n_reps == REGISTERED["n_reps"] and args.target_fpr == REGISTERED["target_fpr"]
                  and list(args.seeds) == REGISTERED["seeds"])
    final_path = (RESULTS / f"phase3_a1_predictions_{core}.json").resolve()
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
    print(f"A1 predictions {args.pair} [{S['axis']}] L{S['L_sel']} C {S['C_sel']:g}: {len(S['draws']) - 1} draws + "
          f"frozen; variants {S['variants']}; registered {registered}", flush=True)
    arr = prediction_arrays(S)
    wb_dev = max(float(np.max(np.abs(arr[f"white_box|{ckey(c)}|realised_delta_dot_v"] - c * S["sigma"])))
                 for c in C_GRID)
    nb = npz_bytes(arr)
    code1 = P.code_block(CODE_FILES)
    changed = [n for n, a, b in (("code", code1["sha256"], code0["sha256"]), ("selection", sha256_file(S["sel_path"]), S["sel_sha"]),
                                 ("cluster manifest", sha256_file(D.CLUSTERS), S["cl_sha"]),
                                 (S["depth_rel"], sha256_file(ROOT / S["depth_rel"]), S["depth_sha"]),
                                 ("registry sources", D.sources_now(), S["reg_sources"])) if a != b]
    changed += [f"cache {r['file']}" for r in S["input_recs"].values()
                if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    if changed:
        raise SystemExit(f"REFUSING to write: changed during the run: {changed}")
    payload = {**provenance_block("phase3_a1_predictions"), "preregistration": PREREG, "code": code0,
               "registered_parameters": registered, "blas_threads": threads, "input_verification": S["verification"],
               "config": input_block(S),
               "prediction_function": {"module": THIS_FILE, "function": "predicted_fpr", "formula": FORMULA,
                                       "sha256": code0["sha256"][THIS_FILE],
                                       "delta": "white-box c * sigma (D-327 (2); choice (c)); transfer delta . v per "
                                                "draw and seed",
                                       "moments": "m, s: mean and SD (ddof 0) of the post scores of E at c = 0; T_d and "
                                                  "sigma (ddof 0) from the pre scores of cal_d (choice (b))"},
               "order": "computes no attacked score (only the geometry delta . v); written before this pair's "
                        "simulation, which refuses without it (D-327 (6))",
               "depth_record": {"file": S["depth_rel"], "sha256": S["depth_sha"],
                                "role": "selection check and the cache identity (the 4c AUC rule's source, D-241 (4))"},
               "geometry": {**S["geometry"], "white_box_max_abs_delta_dot_v_minus_c_sigma": wb_dev,
                            "surrogate": S["surrogate"]},
               "probe_fit": [q.diag for q in S["probes"]],
               "input_cache_sha256": S["input_sha"],
               "input_cache_source": {kk: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"]}
                                      for kk, r in S["input_recs"].items()},
               "draws_npz": {"file": rel(npz_path), "sha256": sha256_bytes(nb), "size": len(nb), "keys": sorted(arr),
                             "rows": "draw index (0 the frozen split); columns the seeds in config order"},
               "predicted": predicted_cells(S, arr)}
    for p, hh in write_pair(out_path, payload, npz_path, nb):
        print(f"Written: {p}\nsha256:  {hh}")
    for variant, cells in payload["predicted"].items():
        line = "  ".join(f"{kk}:{v['predicted_ratio_to_target']['mean']:.3g}" for kk, v in cells.items())
        print(f"  [{variant}] predicted FPR/target (D-156 mean): {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
