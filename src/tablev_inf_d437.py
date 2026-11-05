"""D-437 item 3 (the author's decision of 2026-10-11, following D-436 (10)): TABLE V (c)'s +inf QUANTILE-TRANSFER
THRESHOLDS at n = 200. PRE-REGISTERED (docs/reports_d437/tablev_inf_prereg.md), DESCRIPTIVE, CPU only, from the
existing caches only.

THE CELLS. The 9 cells of Table V (c), column "quantile transfer, traffic", n = 200: in each record of
populations.POOLSTUDY_D156 (7 Aegis _originstrat records, 2 WildGuardMix _r2 records) the configuration
configurations.selected["score|traffic|pi=0.01|n=200"], whose ratio_to_target mean and band the table prints.

WHAT IS RECOMPUTED. Only the quantile-transfer thresholds, by replaying each record's draws with poolstudy_d156's and
paired's own functions (imported, never re-implemented): the caches (the record's sha256), the selected-C probes of
seeds 0-4 (poolstudy_d156.probe_spec + paired.fit_probe; the record's probe_fit class-mean differences exactly), the
draws (poolstudy_d156.partitions: index 0 the frozen split, then draws 1..n_reps), each draw's pools
(poolstudy_d156.draw_pools, stream default_rng([POOL_SEED, d]), called whole so every pool is drawn in the study's
order). Per draw d and seed s: T = paired.thr_at_fpr(pre scores of cal_d), the 1% traffic pool of 200 items (198 benign
from cal_d, 2 harmful from H), t = paired.quantile_transfer(pool pre scores, pool post scores, T).

+inf. t is +inf exactly when T > max(pool pre scores): no pool item reaches the frozen threshold on the pre model
(asserted per draw and seed). Nothing is flagged at +inf: the seed's FPR on the 1000 evaluation negatives E is 0,
floored at 1/1000 (FPR/target 0.10), its TPR 0.

ANCHOR. The per-draw quantities poolstudy_d156.main's put() stored for this configuration (fpr, paired_ratio,
ratio_to_target, tpr; fpr_origin_tc / fpr_origin_ev for the Aegis records) are recomputed from the flags at t and
summarised by paired.band_summary over draws 1..n_reps with the frozen split beside; every field must equal the
record's (JSON round trip). Nothing is written under results/ otherwise.

OUTPUT. results/tablev_inf_d437.json (per cell: n_draws, n_thresholds, n_inf, share_inf, the draws with any / all
seeds +inf, the frozen split's +inf seeds, the tabulated ratio read from the record, the floor, the floored terms'
contribution to the mean, the (draw, seed) list of +inf thresholds, the anchor) and results/tablev_inf_d437_draws.npz
beside it (per cell, per draw 0..n_reps and seed: T, t, the pool's max pre score, the pool's benign and harmful counts
at or above T, the flag counts on E and on the positives). Each is written once (never overwritten).

REFUSALS beside the anchor: a pool that is not 198 benign from cal_d + 2 harmful; any flag at a +inf threshold.
Counts are over draws 1..n_reps; the frozen split's are separate fields (frozen_split_*).

USAGE
  python3 -u src/tablev_inf_d437.py                         all 9 cells, the final names (6-thread BLAS pin,
                                                            GUARDRAIL_CACHE_DIRS unset)
  python3 -u src/tablev_inf_d437.py --stem S [--stem S2] --out-dir DIR
                                                            some cells into DIR (must not exist; outside results/;
                                                            the sandbox, scripts/sandbox_tablev_d437.sh)
"""

from __future__ import annotations

import argparse
import io
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
import poolstudy_d156 as PS  # noqa: E402
import populations as POP  # noqa: E402
from provenance import (  # noqa: E402
    canonical_hash, ensure_utf8_stdout, provenance_block, read_result, sha256_bytes, sha256_file, write_results,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/tablev_inf_d437.py", *PS.CODE_FILES]
PREREG = ROOT / "docs" / "reports_d437" / "tablev_inf_prereg.md"
N_TRAFFIC = 200
N_HARM = int(round(0.01 * N_TRAFFIC))                      # poolstudy_d156.draw_pools' rule round(pi * n): 2 harmful
POOL_KEY = f"traffic|pi=0.01|n={N_TRAFFIC}"                # poolstudy_d156.draw_pools' key of the n = 200 pool
CELL_KEY = f"score|{POOL_KEY}"                             # the record's configuration key (Table V (c))
PSET = "selected"                                          # Table V is selected C only (tables_d156.table_robustness)
STEMS = tuple(POP.POOLSTUDY_D156)
N_E = 1000                                                 # the evaluation negatives per draw (the floor's resolution)
QUANTITIES = ("fpr", "paired_ratio", "ratio_to_target", "tpr")
ORIGIN_QUANTITIES = ("fpr_origin_tc", "fpr_origin_ev")
OUT_JSON = RESULTS / "tablev_inf_d437.json"
OUT_NPZ = RESULTS / "tablev_inf_d437_draws.npz"
PER_DRAW = ("T", "t", "pool_pre_max", "n_benign_ge_T", "n_harmful_ge_T", "n_flag_E", "n_flag_pos")


# ----------------------------------------------------------------------------- the thresholds (pure functions)

def transfer_thresholds(pre_ref: np.ndarray, post_ref: np.ndarray, T: np.ndarray) -> dict[str, np.ndarray]:
    """Per seed s: t_s = paired.quantile_transfer(pre_ref[:, s], post_ref[:, s], T[s]) (poolstudy_d156's quantile()
    arm, pre and post windows the same items), whether t_s is +inf, the reference's max pre score and its count of
    items with pre score >= T[s]. Refuses unless +inf occurs exactly when that count is 0 (the definition)."""
    pre_ref, post_ref, T = np.asarray(pre_ref), np.asarray(post_ref), np.asarray(T, dtype=np.float64)
    if pre_ref.ndim != 2 or pre_ref.shape != post_ref.shape or pre_ref.shape[1] != T.size:
        raise ValueError(f"shapes {pre_ref.shape}, {post_ref.shape}, {T.shape}")
    t = np.array([P.quantile_transfer(pre_ref[:, s], post_ref[:, s], T[s]) for s in range(T.size)])
    pre64 = pre_ref.astype(np.float64)                     # quantile_transfer's own cast
    n_ge = (pre64 >= T).sum(axis=0)
    inf = np.isposinf(t)
    if np.isnan(t).any() or np.isneginf(t).any() or not np.array_equal(inf, n_ge == 0):
        raise RuntimeError(f"REFUSING: +inf is not 'no reference item at or above T' (t {t}, counts {n_ge})")
    return {"t": t, "inf": inf, "pool_pre_max": pre64.max(axis=0), "n_ge": n_ge}


def replay_cell(S: dict[str, dict[str, np.ndarray]], parts: list[tuple[np.ndarray, np.ndarray]], n_cal: int,
                n_pool: int, n_h_src: int, origin_strat: bool, target: float) -> dict[str, Any]:
    """Replay poolstudy_d156.main's draws for the n = 200 traffic cell of one probe set.
    S[side][part]: (items, seeds) scores, side pre/post, part pool (n_pool pooled negatives), H (n_h_src harmful),
    pos (eval positives; post only is read). Returns the per-draw arrays (PER_DRAW, (draws, seeds)), the +inf mask and
    the per-draw quantities put() stored (QUANTITIES, and ORIGIN_QUANTITIES with origin_strat)."""
    all_pre = np.vstack([S["pre"]["pool"], S["pre"]["H"]])
    all_post = np.vstack([S["post"]["pool"], S["post"]["H"]])
    n_s = all_pre.shape[1]
    per: dict[str, list[np.ndarray]] = {k: [] for k in PER_DRAW}
    infs: list[np.ndarray] = []
    q: dict[str, list[float]] = {k: [] for k in QUANTITIES + (ORIGIN_QUANTITIES if origin_strat else ())}
    n_e_seen: set[int] = set()
    for di, (cal, ev) in enumerate(parts):
        dp = PS.draw_pools(di, cal, ev, n_cal, n_pool, n_h_src, origin_strat)
        idx, E = dp["pools"].get(POOL_KEY), dp["E"]
        if idx is None:
            raise SystemExit(f"REFUSING: draw {di}: no {POOL_KEY} pool ({dp['skipped'].get(POOL_KEY)})")
        if len(idx) != N_TRAFFIC or len(np.unique(idx)) != N_TRAFFIC:
            raise SystemExit(f"REFUSING: draw {di}: the {POOL_KEY} pool holds {len(idx)} items, not {N_TRAFFIC} distinct")
        harm = idx >= n_pool
        if int(harm.sum()) != N_HARM or int((~harm).sum()) != N_TRAFFIC - N_HARM or not np.isin(idx[~harm], cal).all():
            raise SystemExit(f"REFUSING: draw {di}: the pool holds {int(harm.sum())} harmful and {int((~harm).sum())} "
                             f"benign items (not {N_HARM} and {N_TRAFFIC - N_HARM}, the benign from cal_d)")
        n_e_seen.add(len(E))
        T = np.array([P.thr_at_fpr(S["pre"]["pool"][cal, s], target) for s in range(n_s)])
        F_pre = S["pre"]["pool"][E] >= T
        tr = transfer_thresholds(all_pre[idx], all_post[idx], T)
        tq = tr["t"]
        F, Q = S["post"]["pool"][E] >= tq, S["post"]["pos"] >= tq
        if F[:, tr["inf"]].any() or Q[:, tr["inf"]].any():
            raise SystemExit(f"REFUSING: draw {di}: an item is flagged at a +inf threshold")
        per["T"].append(T)
        per["t"].append(tq)
        per["pool_pre_max"].append(tr["pool_pre_max"])
        per["n_benign_ge_T"].append((all_pre[idx[~harm]].astype(np.float64) >= T).sum(axis=0))
        per["n_harmful_ge_T"].append((all_pre[idx[harm]].astype(np.float64) >= T).sum(axis=0))
        per["n_flag_E"].append(F.sum(axis=0))
        per["n_flag_pos"].append(Q.sum(axis=0))
        infs.append(tr["inf"])
        f = float(P.floored_fpr(F).mean())                 # poolstudy_d156.main put(), line for line
        q["fpr"].append(f)
        q["paired_ratio"].append(P.ratio_from_flags(F_pre, F))
        q["ratio_to_target"].append(f / target)
        q["tpr"].append(float(P.rate(Q).mean()))
        if origin_strat:
            m = E < n_cal
            q["fpr_origin_tc"].append(float(P.floored_fpr(F[m]).mean()) if m.any() else float("nan"))
            q["fpr_origin_ev"].append(float(P.floored_fpr(F[~m]).mean()) if (~m).any() else float("nan"))
    return {"per_draw": {k: np.array(v) for k, v in per.items()}, "inf": np.array(infs), "quantities": q,
            "n_e": sorted(n_e_seen)}


def summaries(q: dict[str, list[float]]) -> dict[str, Any]:
    """paired.band_summary over draws 1.. with the frozen split (index 0) beside, as poolstudy_d156 stored them."""
    return {k: P.band_summary(np.array(v[1:]), v[0]) for k, v in q.items()}


def anchor_diffs(mine: dict[str, Any], stored: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    """Every field of the stored summaries of `keys` that the recomputed ones do not give exactly (JSON round trip)."""
    rt = json.loads(json.dumps(mine))
    out = []
    for k in keys:
        if k not in stored or k not in rt:
            out.append(f"{k}: missing ({'record' if k not in stored else 'recomputed'})")
            continue
        for f in sorted(set(stored[k]) | set(rt[k])):
            if stored[k].get(f) != rt[k].get(f):
                out.append(f"{k}.{f}: record {stored[k].get(f)!r}, recomputed {rt[k].get(f)!r}")
    return out


def count_block(inf: np.ndarray, per: dict[str, np.ndarray], target: float, n_e: int) -> dict[str, Any]:
    """The pre-registered counts from the (draws, seeds) +inf mask: every count over draws 1.. (row 0, the frozen
    split, is not in the tabulated mean); the frozen split's values beside, labelled."""
    body = inf[1:]
    n_draws, n_seeds = body.shape
    n_inf = int(body.sum())
    floor = 1.0 / (n_e * target)
    fE, fP, T = per["n_flag_E"], per["n_flag_pos"], per["T"]
    flags_at_inf = int(fE[1:][body].sum() + fP[1:][body].sum())
    return {
        "n_draws": int(n_draws), "n_seeds": int(n_seeds), "n_thresholds": int(body.size),
        "n_inf": n_inf, "share_inf": n_inf / body.size,
        "n_draws_any_seed_inf": int(body.any(axis=1).sum()), "share_draws_any_seed_inf": float(body.any(axis=1).mean()),
        "n_draws_all_seeds_inf": int(body.all(axis=1).sum()),
        "n_inf_per_seed": [int(x) for x in body.sum(axis=0)],
        "n_frozen_threshold_T_inf": int(np.isposinf(T[1:]).sum()),
        "floor_ratio_to_target": floor,
        "inf_terms_contribution_to_mean": n_inf * floor / body.size,
        "flags_at_inf_thresholds_E_plus_pos": flags_at_inf,
        "inf_draw_seed": [[int(d) + 1, int(s)] for d, s in np.argwhere(body)],
        "frozen_split_inf_seeds": [int(s) for s in np.flatnonzero(inf[0])],
        "frozen_split_n_frozen_threshold_T_inf": int(np.isposinf(T[0]).sum()),
        "frozen_split_flags_at_inf_thresholds_E_plus_pos": int(fE[0][inf[0]].sum() + fP[0][inf[0]].sum()),
        "counts_note": "every count above over draws 1..n_reps (the terms of the tabulated mean); the frozen split "
                       "(draw 0) only in the frozen_split_* fields",
    }


def replay_and_check(rel: str, stored: dict[str, Any], S: dict[str, dict[str, np.ndarray]],
                     parts: list[tuple[np.ndarray, np.ndarray]], n_cal: int, n_pool: int, n_h_src: int,
                     origin_strat: bool, target: float) -> tuple[dict[str, Any], tuple[str, ...]]:
    """replay_cell, then THE ANCHOR: 1000 evaluation negatives in every draw and the recomputed band summaries equal
    to the record's stored ones (`stored`, the record's configuration) field for field; refuses otherwise."""
    res = replay_cell(S, parts, n_cal, n_pool, n_h_src, origin_strat, target)
    if res["n_e"] != [N_E]:
        raise SystemExit(f"REFUSING: {rel}: evaluation parts of {res['n_e']} negatives, not {N_E}")
    keys = QUANTITIES + (ORIGIN_QUANTITIES if origin_strat else ())
    diffs = anchor_diffs(summaries(res["quantities"]), stored, keys)
    if diffs:
        raise SystemExit(f"REFUSING: {rel} {CELL_KEY} not reproduced ({len(diffs)} fields): " + "; ".join(diffs[:12]))
    return res, keys


# ----------------------------------------------------------------------------- one record's cell from its caches

def run_cell(stem: str) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
    """Load one Table V (c) record's inputs, refit its probes, replay its draws and check the anchor.
    Returns (cell block, npz arrays, input cache sha256s keyed '<stem>:<side>/<part>', run identity)."""
    rel = f"results/{stem}.json"
    rec_sha = sha256_file(ROOT / rel)
    rec = read_result(ROOT / rel)
    cfg = rec["config"]
    origin_strat = stem.endswith("_originstrat")
    if origin_strat != ("origin_strata" in rec) or not (origin_strat or stem.endswith("_r2")):
        raise SystemExit(f"REFUSING: {rel}: its design does not match its name")
    core = cfg["pair"][len("repair_"):]
    run = REG.run(D.registry_stem(f"depth_{core}"))
    axis = run["axis"]
    if cfg["axis"] != axis or any(cfg.get(k) != run[k] for k in ("pre_repo", "pre_rev", "post_repo", "post_rev")):
        raise SystemExit(f"REFUSING: {rel}: its axis or repos are not the registry's run")
    seeds, target, n_reps = list(cfg["seeds"]), float(cfg["target_fpr"]), int(cfg["n_reps"])
    L, C_sel = int(cfg["layer"]), float(cfg["selected_C"])
    k = f"layer_{L}"
    data: dict[str, dict[str, Any]] = {"pre": {}, "post": {}}
    sha: dict[str, str] = {}
    recs: dict[str, dict[str, Any]] = {}
    for side, part in (("pre", "probe_train"), ("pre", "threshold_calib"), ("pre", "eval"),
                       ("post", "threshold_calib"), ("post", "eval")):
        z, r = REG.load(run[f"{side}_repo"], axis, part)
        if r["sha256"] != rec["input_cache_sha256"][f"{side}/{part}"]:
            raise SystemExit(f"REFUSING: {r['file']} is not the cache {rel} read")
        data[side][part] = z
        sha[f"{stem}:{side}/{part}"], recs[f"{side}/{part}"] = r["sha256"], r
    for part in ("threshold_calib", "eval"):
        a, b = data["pre"][part], data["post"][part]
        if not np.array_equal(a["item_ids"], b["item_ids"]) or not np.array_equal(a["labels"], b["labels"]):
            raise SystemExit(f"REFUSING: {part} items or labels differ between pre and post")
    y_tr = data["pre"]["probe_train"]["labels"].astype(bool)
    y_cal = data["pre"]["threshold_calib"]["labels"].astype(bool)
    y_ev = data["pre"]["eval"]["labels"].astype(bool)
    ncal, hcal, neg, pos = ~y_cal, y_cal, ~y_ev, y_ev
    pooled_ids = [str(i) for i in data["pre"]["threshold_calib"]["item_ids"][ncal]] + \
                 [str(i) for i in data["pre"]["eval"]["item_ids"][neg]]
    n_cal, n_pool, n_h_src = int(ncal.sum()), len(pooled_ids), int(hcal.sum())
    if sha256_bytes("\n".join(pooled_ids).encode("utf-8")) != cfg["pool_item_ids_sha256"] or n_pool - n_cal != 2 * N_E:
        raise SystemExit(f"REFUSING: {rel}: the pooled negatives are not the record's")
    parts = PS.partitions(n_cal, n_pool, n_reps, axis, origin_strat)
    RAW = {s: {"pool": np.vstack([data[s]["threshold_calib"][k][ncal], data[s]["eval"][k][neg]]),
               "H": data[s]["threshold_calib"][k][hcal]} for s in data}
    RAW["post"]["pos"] = data["post"]["eval"][k][pos]
    X_fit = data["pre"]["probe_train"][k]
    probes, fit_d = [], []
    for seed in seeds:
        rows, C, order = PS.probe_spec(len(y_tr), seed, PSET, C_sel)
        q = P.fit_probe(X_fit[rows], y_tr[rows], seed, C, order)
        probes.append(q)
        s_fit = q.score(X_fit[rows])
        fit_d.append(float(s_fit[y_tr[rows]].mean() - s_fit[~y_tr[rows]].mean()))
    if fit_d != rec["probe_fit"][PSET]["class_mean_score_difference_d_per_seed"]:
        raise SystemExit(f"REFUSING: {rel}: the refitted probes are not the record's (class-mean differences)")
    S = {side: {part: np.column_stack([q.score(RAW[side][part]) for q in probes]) for part in RAW[side]}
         for side in RAW}
    stored = rec["configurations"][PSET][CELL_KEY]
    res, keys = replay_and_check(rel, stored, S, parts, n_cal, n_pool, n_h_src, origin_strat, target)
    rt = stored["ratio_to_target"]
    cell = {"stem": stem, "record": {"file": rel, "sha256": rec_sha}, "pair": cfg["pair"], "axis": axis,
            "design": "origin-stratified draws (D-298)" if origin_strat else "as registered (_r2 re-issue)",
            "layer": L, "selected_C": C_sel, "seeds": seeds, "target_fpr": target, "n_reps": n_reps,
            "configuration": f"configurations.{PSET}[{CELL_KEY!r}]",
            "ratio_to_target_tabulated": {"mean": rt["mean"], "band95": rt["band95"], "frozen": rt["frozen"],
                                          "printed": f"{rt['mean']:.2f}",
                                          "printed_band": f"[{rt['band95'][0]:.2f}, {rt['band95'][1]:.2f}]"},
            "tpr_tabulated_mean": stored["tpr"]["mean"],
            **count_block(res["inf"], res["per_draw"], target, N_E),
            "anchor": {"compared": list(keys), "fields": "every band_summary field", "result": "equal to the record",
                       "probe_fit": "class_mean_score_difference_d_per_seed equal to the record"},
            "input_verification": REG.verification(recs)}
    arrays = {f"{stem}__{k}": v for k, v in res["per_draw"].items()}
    arrays[f"{stem}__inf"] = res["inf"]
    ident = {k: run[k] for k in ("pre_repo", "pre_rev", "post_repo", "post_rev")}
    return cell, arrays, sha, ident


# ----------------------------------------------------------------------------- main

def write_bytes_exclusive(path: Path, data: bytes) -> None:
    """Write `data` to `path` as provenance.write_results writes a JSON: an exclusive temporary in the same directory,
    fsync, size and sha256 checked, then hard-linked to the final name (fails if it exists: never overwritten) and the
    final name read back. The temporary is this call's own file; after the link it is a second name of the bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    want = sha256_bytes(data)
    tmp = path.with_name(f"{path.name}.partial.{os.getpid()}.{os.urandom(4).hex()}")
    with open(tmp, "xb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    if tmp.stat().st_size != len(data) or sha256_file(tmp) != want:
        raise SystemExit(f"REFUSING: {tmp} is not the bytes written (left in place)")
    os.link(tmp, path)
    if sha256_file(path) != want:
        raise SystemExit(f"REFUSING: {path} differs from what was written")
    if tmp.stat().st_nlink >= 2:
        tmp.unlink()                                       # a second name of the final bytes: nothing is deleted



def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="D-437 item 3: +inf quantile-transfer thresholds of Table V (c), n = 200")
    ap.add_argument("--stem", action="append", choices=STEMS, help="one record (repeatable); default all nine")
    ap.add_argument("--out-dir", default=None, help="write here instead of results/ (a sandbox; not under results/)")
    a = ap.parse_args(argv)
    stems = list(a.stem) if a.stem else list(STEMS)
    final = a.out_dir is None
    out_json = OUT_JSON if final else Path(a.out_dir).resolve() / OUT_JSON.name
    out_npz = OUT_NPZ if final else Path(a.out_dir).resolve() / OUT_NPZ.name
    if not final and out_json.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: --out-dir must not be under results/")
    threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    if final and (stems != list(STEMS) or any(t != "6" for t in threads.values())
                  or os.environ.get("GUARDRAIL_CACHE_DIRS")):
        raise SystemExit(f"REFUSING the final names: cells {len(stems)} of {len(STEMS)}, threads {threads}, "
                         f"GUARDRAIL_CACHE_DIRS {'set' if os.environ.get('GUARDRAIL_CACHE_DIRS') else 'unset'} "
                         "(must be unset: the caches come from their recorded directory)")
    if not final and Path(a.out_dir).exists():
        raise SystemExit(f"REFUSING: --out-dir {a.out_dir} exists (a sandbox directory is never reused)")
    for p in (out_json, out_npz):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists; never overwritten")
    if not PREREG.is_file():
        raise SystemExit(f"REFUSING: no pre-registration at {PREREG}")
    code0 = P.code_block(CODE_FILES)
    prereg_sha = sha256_file(PREREG)
    cells, arrays, input_sha, runs = [], {}, {}, {}
    for stem in stems:
        print(f"D-437 item 3: {stem}", flush=True)
        cell, arr, sha, ident = run_cell(stem)
        cells.append(cell)
        arrays.update(arr)
        input_sha.update(sha)
        runs[stem] = ident
        print(f"  reproduced; n_inf {cell['n_inf']} of {cell['n_thresholds']} (share {cell['share_inf']:.4f}); "
              f"tabulated {cell['ratio_to_target_tabulated']['printed']}", flush=True)
    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"] or sha256_file(PREREG) != prereg_sha:
        raise SystemExit("REFUSING to write: the code or the pre-registration changed during the run")
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    npz_bytes = buf.getvalue()
    config = {"cells": stems, "configuration_key": CELL_KEY, "pool_key": POOL_KEY, "probe_set": PSET,
              "n_traffic": N_TRAFFIC, "n_eval_negatives": N_E, "pool_seed": PS.POOL_SEED,
              "draw_seeds": {"origin_stratified": PS.STRAT_DRAW_SEED, "item_level": P.AXIS_DRAW_SEED},
              "seeds": sorted({s for c in cells for s in c["seeds"]}), "runs": runs,
              "per_draw_arrays": list(PER_DRAW) + ["inf"],
              "definition": "t = paired.quantile_transfer(pool pre, pool post, T) is +inf exactly when T > max(pool pre "
                            "scores): no pool item reaches the frozen threshold on the pre model; nothing is flagged",
              "counted_over": "draws 1..n_reps x seeds (the terms of the tabulated mean); the frozen split beside"}
    payload = {**provenance_block("tablev_inf_d437"),
               "preregistration": {"file": str(PREREG.relative_to(ROOT)).replace("\\", "/"), "sha256": prereg_sha,
                                   "status": "DESCRIPTIVE (no prediction)"},
               "decision": "D-437 item 3 (the author, 2026-10-11); background D-436 (10)",
               "code": code0, "blas_threads": threads, "final_names": final,
               "config": config, "config_sha256": canonical_hash(config),
               "input_cache_sha256": input_sha,
               "draws_npz": {"file": out_npz.name, "sha256": sha256_bytes(npz_bytes),
                             "keys": "<stem>__<array>, each (n_reps + 1, seeds), row 0 the frozen split"},
               "cells": cells}
    write_bytes_exclusive(out_npz, npz_bytes)
    digest = write_results(out_json, payload, exclusive=True)
    print(f"Written: {out_json}\nsha256:  {digest}\nWritten: {out_npz}\nsha256:  {payload['draws_npz']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
