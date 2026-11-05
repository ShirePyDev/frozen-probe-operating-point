"""D-440 (8): the PHASE 2 XSTest over-refusal column with the PHASE 1 estimator (the author's optional item 8 of
2026-10-13; pre-registered in docs/reports_d440/item8_xstest_p2_prereg.md before any Phase 2 XSTest number existed).
DESCRIPTIVE: no prediction, no verdict, no D-177 line. CPU only; reads existing caches; no GPU work.

THE ESTIMATOR is Phase 1's, called, not copied (D-167, D-187, D-189): src/overrefusal_d156.py's xstest_point at each
partition's per-seed thresholds (set at 1% on the partition's calibration part of the pooled PRE-side safety-axis
negatives); XSTest itself is never re-partitioned; per level the summaries of overrefusal_d156.QUANT by
paired.band_summary over the draws with the frozen split as one more draw, the share of draws with McNemar p < 0.05 and
the paired refusal ratio's share of floored draws; the XSTest AUROC (contrast vs safe) with the D-152 P5 item bootstrap
(B = 5000, overrefusal_d156.ITEM_BOOT_SEED), checked against depth_d156.weighted_auroc with unit weights.

THE PHASE 2 INPUTS to it (each by the closest D-entry; the prereg's section 3):
  axis        WildGuardMix (p2wg), the only Phase 2 safety axis (D-334, D-361 (2)); OPEN QUESTION 1 of the prereg
  probes      the pre side's Phase 2 selection of record (its C and L*), seeds 0-4 (D-361 (2))
  draws       split v2: 5,000 calibration + 6,000 evaluation negatives pooled, 1,000 re-partitions
              (twomoment_p2_d361.P2_DRAW_SEED), the frozen v2 split first; D-164 cluster level beside
              (P2_CLUSTER_DRAW_SEED, the unit labels refsize_d382.p2_bind binds) (D-361 (2))
  cells       the nine D-361 headline pairs x the pair's D-361 layer set (GRID6 + the pre side's L*, 7 each); L* is the
              pair's primary row (D-206 (2)), every other layer beside (D-189 (1)); selected C only; 1% only
  poolings    user-span mean primary; last-token beside, as its own records (D-361 (2), (4); D-429's precedent)
  XSTest      each side's p2sub rest record, entry 'over_refusal/eval/<pooling>' (D-303 (c), D-325), split file
              data/frozen/over_refusal__eval.jsonl at sha256 1efd74eb... (results/dataset_manifest.json)
  read        only what the item reads (D-187 INPUTS): the PRE side's WildGuardMix probe_train and calibration (the
              selection's caches) and its eval and pool4d subsets (the v2 gather), and both sides' XSTest caches. No
              post-side WildGuardMix cache is opened.

CHECKS (nothing is written unless all pass): the D-361 anchor record passes (twomoment_p2_d361.passing_anchor); at L*
the probes are the selection's own fit bit for bit and reproduce its ho20 TPRs; at every layer and level the PRE-side
operating point (post = pre, paired.repartition_estimate) equals the pair's D-361 record on
overrefusal_d156.PRE_ONLY_METRICS and PRE_ONLY_COUNTS (D-187's CHECK); the XSTest thresholds are the shared
estimator's draw by draw: at them the WildGuardMix pre FPR on each draw's evaluation part equals the estimator's
per-draw value exactly (D-189 (4)); XSTest 250 safe + 200 contrast for a final name; pre and post hold the same items
and labels; n_tokens and truncation differences pre vs post recorded (D-189 (6)).

MODES (every output written once: staged, then named, never overwritten)
  pair --pair K [--pooling P]   results/overrefusal_p2_d440_<K>[_lastgen].json and its _scores.npz (per layer the
                                XSTest scores pre and post, 450 x 5, and the frozen split's per-seed thresholds, so the
                                frozen split's per-item flags are recomputable).
  summary [--pooling P]         results/overrefusal_p2_d440_summary[_lastgen].json: the nine L* rows and all 63 cells,
                                from the nine pair records (each bound by its sha256 at read, its npz by the sha256 the
                                record names; the frozen split's counts recomputed from the stored scores first).
  preflight [--pooling P]       every record check of the nine pair jobs and the presence of every cache file they
                                read; reads no cache array, computes nothing, writes nothing.
  queue [--write]               queues/laneA_20261013_xstest_p2.tsv (DO NOT LOAD before review and the sandbox).

Final names need the registered parameters (1000 draws, seeds 0-4 from the selection, B = 5000, 1%), the 6-thread
BLAS pin, GUARDRAIL_CACHE_DIRS unset and the reused files at their reviewed bytes (REUSED_SHA256). A test writes
outside results/ (--out).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import float16_check_d364 as F16  # noqa: E402  (imported, not changed: gather_split, V2_EVAL_FILE)
import metrics as M  # noqa: E402
import overrefusal_d156 as O  # noqa: E402  (imported, not changed: xstest_point and its constants; D-187)
import p2auroc_d424 as A  # noqa: E402  (imported, not changed: pooled, refuse, rel, canon)
import paired as P  # noqa: E402
import refsize_d382 as RS  # noqa: E402  (imported, not changed: p2_bind, tp2_code_block)
import selection_p2 as SP  # noqa: E402  (imported, not changed: load_ingested, CACHE, threads)
import selection_tpr_v2 as T  # noqa: E402  (imported, not changed: seed_probes, tprs_at, anchor_diffs)
import twomoment_p2_d361 as TP2  # noqa: E402  (imported, not changed: open_sub_cache, fit_probes, score_matrices)
from depth_d156 import weighted_auroc  # noqa: E402
from provenance import (  # noqa: E402
    _RESULT_READS, canonical_hash, ensure_utf8_stdout, input_identity_problems, provenance_block, read_result,
    sha256_bytes, sha256_file,
)

ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = ROOT
RESULTS = ROOT / "results"
QUEUES = ROOT / "queues"
OWN = "src/overrefusal_p2_d440.py"
CODE_FILES = list(dict.fromkeys([OWN, *O.CODE_FILES, *A.CODE_FILES]))
# the reviewed files whose functions this module calls, at the bytes it was written and tested against (2026-10-13,
# branch d440-xstest from 8c792b3); a final name refuses other bytes
REUSED_SHA256 = {
    "src/overrefusal_d156.py": "11bad9000dc6c70ee664d0784b167ad8bc08ee6a671be26b3a12e6be995d007f",
    "src/p2auroc_d424.py": "edc7346487c7d9369526602b0cd54176db04ffb9f319c4276ad4ae4f7d1c106f",
    **A.REUSED_SHA256,
}
RUN_KIND = {"pair": "overrefusal_p2_d440_pair", "summary": "overrefusal_p2_d440_summary"}
PREREG = ("D-440 (8) (the author's optional item 8 of 2026-10-13), pre-registered in "
          "docs/reports_d440/item8_xstest_p2_prereg.md before any Phase 2 XSTest number existed; the estimator of "
          "D-167, D-187 and D-189 (overrefusal_d156.xstest_point); the Phase 2 inputs by D-334 and D-361 (2); "
          "layers D-189 (1) and D-206 (2); DESCRIPTIVE, no prediction")

# ----------------------------------------------------------------------------- registered parameters
SEEDS = list(O.REGISTERED["seeds"])                     # 0-4 (D-187; the Phase 2 selections use the same)
N_BOOT = int(O.REGISTERED["n_boot"])                    # 5000 (D-187)
N_DRAWS = P.N_DRAWS                                     # 1000 (D-156; D-361 (2))
TARGET = P.TARGET_FPR                                   # 1% only (D-187)
REGISTERED = {"n_draws": N_DRAWS, "seeds": SEEDS, "n_boot": N_BOOT, "target_fpr": TARGET}
LEVELS = O.LEVELS                                       # item_level_D156, cluster_level_D164
ITEM_BOOT_SEED = dict(O.ITEM_BOOT_SEED)                 # xstest_safe 20260933, xstest_contrast 20260934 (D-187)
PAIRS = list(TP2.HEADLINE_PAIRS)                        # the nine D-361 headline pairs
FIRST_PAIR = "mistral_v01_v02"
PAIR_ORDER = [FIRST_PAIR] + [k for k in PAIRS if k != FIRST_PAIR]
LAYERS_PER_PAIR = 7
POOLING = TP2.PRIMARY_POOLING                           # usermean
POOLINGS = tuple(TP2.POOLINGS)                          # usermean, lastgen
XS_SPLIT = "over_refusal/eval"
XSTEST_FILE = "data/frozen/over_refusal__eval.jsonl"
XSTEST_SHA256 = "1efd74eb103ec0d3e7ac5c6e9634ebbae6e75dec1a560d1611a04bf3175058d8"   # results/dataset_manifest.json
N_SAFE, N_CONTRAST = 250, 200
TABLE_QUANT = ("safe_flagged_pre_mean", "safe_flagged_post_mean", "refusal_change", "paired_refusal_ratio",
               "contrast_flag_change")                 # overrefusal_d117_table.QUANT (Phase 1's table form)
NPZ_KEYS = ("L<layer>|xs_pre, L<layer>|xs_post: (450, 5) float64 XSTest scores (rows in the frozen split's order, "
            "one column per seed); L<layer>|thr_frozen: (5,) the frozen v2 split's per-seed thresholds at 1%; "
            "labels: (450,) 1 = contrast, 0 = safe; cell_layer: the layers in the order of 'cells'")


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def blas_threads() -> dict[str, str | None]:
    return {v: os.environ.get(v) for v in SP.THREAD_VARS}


def reused_now() -> dict[str, str | None]:
    return {f: (sha256_file(CODE_ROOT / f) if (CODE_ROOT / f).is_file() else None) for f in REUSED_SHA256}


def final_problems(registered: bool) -> list[str]:
    out: list[str] = []
    if not registered:
        out.append(f"non-registered parameters (the final names need {REGISTERED})")
    th = blas_threads()
    if any(v != SP.THREADS for v in th.values()):
        out.append(f"no 6-thread BLAS pin (D-152 P6): {th}")
    if os.environ.get("GUARDRAIL_CACHE_DIRS"):
        out.append("GUARDRAIL_CACHE_DIRS is set; the caches must come from their recorded directories")
    now = reused_now()
    for f, want in REUSED_SHA256.items():
        if now[f] is None:
            out.append(f"{f} is missing: this module reuses its reviewed functions")
        elif now[f] != want:
            out.append(f"{f} is not the reviewed bytes this module reuses ({now[f][:12]} vs {want[:12]})")
    return out


def status_block(registered: bool) -> dict[str, Any]:
    return {"status": "FINAL" if registered else "PROVISIONAL", "d_entries": [],
            "rule": "CLAUDE.md evidence standard: FINAL, FLAGGED (with the D-entry) or PROVISIONAL (not the registered "
                    f"parameters {REGISTERED}); DESCRIPTIVE (no verdict)"}


def sfx(pooling: str) -> str:
    if pooling not in POOLINGS:
        refuse(f"pooling must be one of {POOLINGS}, not {pooling!r}")
    return "" if pooling == POOLING else f"_{pooling}"


def out_paths(stem: str, out: str | Path | None, with_npz: bool = True) -> tuple[Path, Path | None, bool]:
    final_path = (RESULTS / f"{stem}.json").resolve()
    out_path = Path(out).resolve() if out else final_path
    final = out_path == final_path
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        refuse("a test run (--out) must not write under results/; only this job's own final name may")
    npz_path = out_path.with_name(out_path.stem + "_scores.npz") if with_npz else None
    if npz_path is not None and npz_path.exists() and not out_path.exists():
        refuse(f"{npz_path} exists without {out_path.name}: the orphan of a failed write; move it aside (never delete "
               "it) and rerun")
    for p in [out_path] + ([npz_path] if npz_path is not None else []):
        if p.exists() or p.is_symlink():
            refuse(f"{p} exists; results are never overwritten (D-152 P5)")
    return out_path, npz_path, final


def json_bytes_of(payload: dict[str, Any]) -> bytes:
    b = json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8")
    json.loads(b.decode("utf-8"))      # not a result (D-297): this run's own bytes, parsed before staging
    return b


def savez_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: np.ascontiguousarray(v) for k, v in arrays.items()})
    return buf.getvalue()


# ----------------------------------------------------------------------------- the arithmetic (Phase 1's lines)
def pre_only_diffs(pre_report: dict[str, Any], want: dict[str, Any]) -> list[str]:
    """overrefusal_d156's CHECK: the pre side through the shared estimator (post = pre) against the record's
    'realised' block, on PRE_ONLY_METRICS and the frozen split's PRE_ONLY_COUNTS (canonical JSON values)."""
    mine = json.loads(json.dumps(pre_report, default=str))
    want = want or {}
    return [q for q in O.PRE_ONLY_METRICS if mine.get(q) != want.get(q)] + \
           [q for q in O.PRE_ONLY_COUNTS
            if mine["frozen_split_counts"].get(q) != (want.get("frozen_split_counts") or {}).get(q)]


def level_block(sc: dict[str, np.ndarray], x_con: np.ndarray, draws: np.ndarray, want: dict[str, Any],
                target: float, where: str) -> tuple[dict[str, Any], int]:
    """One level of one cell: the pre-side check against the D-361 'realised' block, then overrefusal_d156.xstest_point
    at every partition (the frozen split first), the threshold tie (D-189 (4)) and overrefusal_d156's block. sc holds
    (n, seeds) score matrices cal_pre, ev_pre, pos_pre, xs_pre, xs_post. Returns (block, 1 tie checked)."""
    cal, ev, pos = sc["cal_pre"], sc["ev_pre"], sc["pos_pre"]
    pre_report, pre_arr = P.repartition_estimate(cal, ev, cal, ev, pos, pos, draws, target)
    diff = pre_only_diffs(pre_report, want)
    if diff:
        refuse(f"{where}: the pre side differs from the pair's D-361 record in {diff}")
    n_cal, n_pool = cal.shape[0], cal.shape[0] + ev.shape[0]
    pool_pre = np.vstack([cal, ev])
    parts = [(np.arange(n_cal), np.arange(n_cal, n_pool))] + [(d[:n_cal], d[n_cal:]) for d in draws]
    sa, sb = sc["xs_pre"], sc["xs_post"]
    safe = ~x_con
    per = [O.xstest_point(pool_pre, c, e, sa[safe], sb[safe], sa[x_con], sb[x_con], target) for c, e in parts]
    afp = np.array([p["aegis_fpr_pre"] for p in per])       # here the WildGuardMix pre FPR (Phase 1's key name)
    if afp[0] != pre_report["fpr_pre"]["frozen"] or not np.array_equal(afp[1:], pre_arr["fpr_pre"]):
        refuse(f"{where}: the XSTest thresholds are not the shared estimator's (D-189 (4))")
    blk: dict[str, Any] = {q: P.band_summary(np.array([p[q] for p in per[1:]]), per[0][q]) for q in O.QUANT}
    blk["share_of_draws_mcnemar_safe_p_max_below_0_05"] = float(np.mean([p["mcnemar_safe_p_max"] < 0.05
                                                                         for p in per[1:]]))
    blk["paired_refusal_ratio_share_of_draws_floored"] = float(np.mean([p["floored"] for p in per[1:]]))
    blk["frozen_split"] = per[0]["_frozen"]
    blk["safety_axis_fpr_pre_tie"] = {"axis": SP.AXIS, "frozen": float(afp[0]), "n_draws_equal": int(len(afp) - 1),
                                      "note": "xstest_point's 'aegis_fpr_pre' is the safety axis's pre FPR; in "
                                              "Phase 2 that axis is WildGuardMix"}
    return blk, 1


def auroc_block(sa: np.ndarray, sb: np.ndarray, x_con: np.ndarray, n_seeds: int, CNT_S: np.ndarray,
                CNT_C: np.ndarray, where: str) -> dict[str, Any]:
    """overrefusal_d156's XSTest AUROC block (contrast vs safe, pre, post, change), its lines repeated."""
    safe = ~x_con
    n_safe, n_con = int(safe.sum()), int(x_con.sum())
    au: dict[str, Any] = {"note": O.AUROC_NOTE}
    reps, pts = {}, {}
    for name, sc in (("pre", sa), ("post", sb)):
        pt = np.array([M.auroc(sc[x_con, s], sc[safe, s]).value for s in range(n_seeds)])
        unit = np.array([weighted_auroc(sc[x_con, s], sc[safe, s], np.ones((1, n_con)), np.ones((1, n_safe)))[0]
                         for s in range(n_seeds)])
        if not np.array_equal(pt, unit):
            refuse(f"{where} {name}: weighted AUROC is not metrics.auroc")
        pts[name] = pt
        reps[name] = np.mean([weighted_auroc(sc[x_con, s], sc[safe, s], CNT_C, CNT_S) for s in range(n_seeds)], axis=0)
    pts["change"], reps["change"] = pts["post"] - pts["pre"], reps["post"] - reps["pre"]
    for name in ("pre", "post", "change"):
        au[name] = {"mean_over_seeds": float(pts[name].mean()), "per_seed": pts[name].tolist(), **P.ci(reps[name])}
    return au


def compute_cell(key: str, li: int, sc: dict[str, np.ndarray], x_con: np.ndarray, draws: dict[str, np.ndarray],
                 d361_cell: dict[str, Any], meta: dict[str, Any], CNT_S: np.ndarray, CNT_C: np.ndarray,
                 target: float = TARGET) -> tuple[dict[str, Any], dict[str, np.ndarray], int]:
    """One (pair, layer) cell at both levels. meta: L_sel, g6, C, n_seeds. Returns (cell, npz arrays, ties)."""
    if not isinstance(d361_cell, dict):
        refuse(f"{key} L{li}: the pair's D-361 record holds no cell for this layer")
    n_seeds = int(meta["n_seeds"])
    cell: dict[str, Any] = {"pair": key, "layer": li, "grid_position": A.grid_position(li, meta["L_sel"], meta["g6"]),
                            "is_selected_layer": li == meta["L_sel"], "probe_set": "selected", "probe_C": meta["C"],
                            "input_verification": TP2.D170_P2}
    ties = 0
    for lvl in LEVELS:
        want = d361_cell.get("realised") if lvl == LEVELS[0] else (d361_cell.get(LEVELS[1]) or {}).get("realised")
        blk, t = level_block(sc, x_con, draws[lvl], want, target, f"{key} L{li} {lvl}")
        ties += t
        if lvl == LEVELS[1]:
            blk["note"] = ("cluster-level re-partitions of the WildGuardMix negatives (D-164); XSTest itself is never "
                           "re-partitioned; 'frozen' is the frozen split, not a cluster-level draw")
        cell[lvl] = blk
    cell["xstest_auroc"] = auroc_block(sc["xs_pre"], sc["xs_post"], x_con, n_seeds, CNT_S, CNT_C, f"{key} L{li}")
    thr = np.array([P.thr_at_fpr(sc["cal_pre"][:, s], target) for s in range(n_seeds)])
    arrays = {f"L{li}|xs_pre": np.ascontiguousarray(sc["xs_pre"], dtype="<f8"),
              f"L{li}|xs_post": np.ascontiguousarray(sc["xs_post"], dtype="<f8"),
              f"L{li}|thr_frozen": np.ascontiguousarray(thr, dtype="<f8")}
    fz = cell[LEVELS[0]]["frozen_split"]
    if [int(x) for x in (sc["xs_pre"][~x_con] >= thr).sum(axis=0)] != fz["safe_flagged_pre"] or \
            [int(x) for x in (sc["xs_post"][~x_con] >= thr).sum(axis=0)] != fz["safe_flagged_post"]:
        refuse(f"{key} L{li}: the stored frozen thresholds do not reproduce the frozen split's counts")
    cell["scores_sha256"] = {k.split("|")[1]: sha256_bytes(v.tobytes()) for k, v in arrays.items()}
    return cell, arrays, ties


# ----------------------------------------------------------------------------- records (no cache array read)
def xs_entry(sb: dict[str, Any], pooling: str, layers: list[int]) -> dict[str, Any]:
    """The XSTest entry of one side's p2sub rest record, checked on the records alone (selection_tpr_v2.subset_entry's
    checks for the split over_refusal/eval, which that function's p2wg key cannot name), plus the pinned split file."""
    key = f"{XS_SPLIT}/{pooling}"
    s = (sb["rec"].get("splits") or {}).get(key)
    if not isinstance(s, dict):
        refuse(f"{sb['rel']} has no split {key}")
    name, want = s.get("cache_file"), s.get("cache_sha256")
    if not isinstance(name, str) or "/" in name or "\\" in name or not isinstance(want, str) or len(want) != 64:
        refuse(f"{sb['rel']} split {key}: malformed cache_file or cache_sha256")
    if s.get("n_nonfinite") != 0 or s.get("n_truncated") != 0:
        refuse(f"{name}: the record counts {s.get('n_nonfinite')} non-finite and {s.get('n_truncated')} truncated items")
    if (s.get("split_file"), s.get("split_sha256")) != (XSTEST_FILE, XSTEST_SHA256):
        refuse(f"{name}: split {s.get('split_file')!r} at {str(s.get('split_sha256'))[:16]}, not {XSTEST_FILE} at "
               f"{XSTEST_SHA256[:16]} (results/dataset_manifest.json)")
    lay = s.get("layers_resolved")
    if not isinstance(lay, list) or any(li not in lay for li in layers):
        refuse(f"{name}: layers_resolved {lay} lack the pair's layers {layers}")
    src = (sb["rest"].get("splits") or {}).get(key) or {}
    if not isinstance(src.get("cache_sha256"), str) or \
            (s.get("source_cache_file"), s.get("source_cache_sha256")) != (src.get("cache_file"), src.get("cache_sha256")):
        refuse(f"{name}: its source cache is not the one {sb['rest_rel']} names for {key}")
    if (src.get("split_file"), src.get("split_sha256")) != (XSTEST_FILE, XSTEST_SHA256):
        refuse(f"{sb['rest_rel']}: its {key} split is not the pinned XSTest file")
    if sb["index"].get(name) != want:
        refuse(f"{name} is not in the ingest index {A.rel(sb['index_path'])} with the record's sha256")
    return s


def files_read(b: dict[str, Any], xs: dict[str, dict[str, Any]]) -> dict[str, str]:
    """{cache key: file name} of every cache a pair job opens (pre WildGuardMix four, XSTest two)."""
    pooling, sr = b["pooling"], b["sr"]
    out = {f"pre/{SP.AXIS}/{part}/{pooling}": ((sr["rec"].get("splits") or {}).get(f"{SP.AXIS}/{part}/{pooling}")
                                               or {}).get("cache_file") for part in ("probe_train", "calibration")}
    for part in ("eval", "pool4d"):
        out[f"pre/{T.SUB_TAG}/{SP.AXIS}/{part}/{pooling}"] = b["s_sub"]["pre"][part].get("cache_file")
    for side in ("pre", "post"):
        out[f"{side}/{T.SUB_TAG}/{XS_SPLIT}/{pooling}"] = xs[side]["cache_file"]
    return out


def absent(files: dict[str, str]) -> list[str]:
    return sorted(k for k, f in files.items()
                  if not isinstance(f, str) or (SP.CACHE / f).is_symlink() or not (SP.CACHE / f).is_file())


def bind(key: str, pooling: str, n_draws: int, n_boot: int, final: bool, records_only: bool,
         gates: bool = False) -> dict[str, Any]:
    """Every record the job reads before any cache. The final-name gates run when `final` (a pair job at its final
    name) or `gates` (the preflight, records only, at the registered parameters: a stale pin is caught before the
    first real job)."""
    if key not in PAIRS:
        refuse(f"{key!r} is not one of the nine D-361 headline pairs: {PAIRS}")
    code0 = P.code_block(CODE_FILES)
    with A.pooled(pooling):
        b = RS.p2_bind(key, n_draws, False, True, code0)
    if b.get("pooling") != pooling:
        refuse(f"{key}: the binding is for pooling {b.get('pooling')!r}, not {pooling!r}")
    if b.get("beyond") or len(b["layers"]) != LAYERS_PER_PAIR:
        refuse(f"{key}: the cell set is not D-361's {LAYERS_PER_PAIR} layers: {list(b['layers'])}")
    layers = list(b["layers"])
    xs = {side: xs_entry(b["sb"][side], pooling, layers) for side in ("pre", "post")}
    files = files_read(b, xs)
    if final or gates:
        registered = n_draws == N_DRAWS and list(b["seeds"]) == SEEDS and n_boot == N_BOOT
        bad = final_problems(registered) + TP2.final_problems(registered, b["pinned_keys"])
        have = {"v2/calibration": b["v2"]["sha256"], "v2/eval": b["ev2"]["sha256"]}
        bad += [f"{k} is not D-362's ({have[k][:16]})" for k in have if have[k] != TP2.PINNED[k]]
        if (b["n_cal"], b["n_ev"], b["n_pos"]) != TP2.V2_SIZES:
            bad.append(f"split v2 holds {(b['n_cal'], b['n_ev'], b['n_pos'])}, not {TP2.V2_SIZES}")
        if (b.get("d361") or {}).get("registered_parameters") is not True:
            bad.append(f"{b.get('d361_rel')} was not run with D-361's registered parameters")
        if bad:
            refuse(f"the final names need: {bad}")
    anchor_rec = None if records_only else TP2.passing_anchor(RS.tp2_code_block(code0), final=final)
    return {"b": b, "code0": code0, "xs": xs, "files": files, "absent": absent(files), "anchor_rec": anchor_rec}


# ----------------------------------------------------------------------------- loading
def load(b: dict[str, Any], xs: dict[str, dict[str, Any]], log: Callable[[str], None] = print) -> dict[str, Any]:
    """p2auroc_d424.load_scores' lines for the PRE side only (D-187 INPUTS), plus both sides' XSTest caches. Returns
    {sc: {layer: score matrices}, x_con, tokenisation, probe_anchor, caches, gathered, probe_diagnostics}."""
    pooling, sr, seeds, C_sel, L_sel = b["pooling"], b["sr"], list(b["seeds"]), b["C_sel"], b["L_sel"]
    v2, ev2, sb, layers = b["v2"], b["ev2"], b["sb"], list(b["layers"])
    z_tr, r_tr, _ = SP.load_ingested(sr["rec"], sr["rec_rel"], b["index_pre"][1], "probe_train", pooling)
    z_cal, r_cal, _ = SP.load_ingested(sr["rec"], sr["rec_rel"], b["index_pre"][1], "calibration", pooling)
    recorded = sr["sel"].get("input_cache_sha256") if isinstance(sr["sel"].get("input_cache_sha256"), dict) else {}
    for part, r in (("probe_train", r_tr), ("calibration", r_cal)):
        k_ = f"pre/{SP.AXIS}/{part}/{pooling}"
        if recorded.get(k_) != r["sha256"]:
            refuse(f"{r['file']} ({r['sha256'][:16]}) is not the cache {sr['sel_rel']} read for {k_}")
    z_sub, r_sub, meta_sub = {}, {}, {}
    for part in ("eval", "pool4d"):
        z_sub[part], ids, labs, r_sub[part] = TP2.open_sub_cache(sb["pre"], b["s_sub"]["pre"][part], part, pooling,
                                                                 layers)
        meta_sub[part] = (ids, labs)
    # XSTest: open_sub_cache's checks (data/hidden, mode 444, sha256, layers, no flagged item, item for item the
    # frozen split named by the entry); its messages name the p2wg key, the entry is the over_refusal one
    z_xs, r_xs, meta_xs = {}, {}, {}
    for side in ("pre", "post"):
        z_xs[side], ids, labs, r_xs[side] = TP2.open_sub_cache(sb[side], xs[side], "eval", pooling, layers)
        meta_xs[side] = (ids, labs)
    if meta_xs["pre"] != meta_xs["post"]:
        refuse("XSTest items or labels differ between pre and post")
    x_con = np.asarray(meta_xs["pre"][1]) == 1               # 1 = contrast (unsafe), 0 = safe-but-scary
    tok = {"n_items_n_tokens_differ": int((np.asarray(z_xs["pre"]["n_tokens"]) != np.asarray(z_xs["post"]["n_tokens"])).sum()),
           "n_items_truncated_differ": int((np.asarray(z_xs["pre"]["truncated"]) != np.asarray(z_xs["post"]["truncated"])).sum())}
    cal_meta = ([str(x) for x in z_cal["item_ids"]], [int(x) for x in z_cal["labels"]])
    y_tr = z_tr["labels"].astype(bool)
    if sr["sel"]["model"].get("n_probe_train") != len(y_tr):
        refuse(f"probe_train holds {len(y_tr)} items, {sr['sel_rel']} says {sr['sel']['model'].get('n_probe_train')}")
    y_cal_v1 = np.asarray(cal_meta[1]) == 1
    if y_cal_v1.all():
        refuse("v1 calibration holds no negative")

    # the L* probe anchor (p2auroc_d424.load_scores' lines)
    kL = f"layer_{L_sel}"
    probes_L = TP2.fit_probes(z_tr[kL], y_tr, seeds, C_sel)
    sel_probes = T.seed_probes(z_tr[kL], y_tr, b["arm"], seeds, b["holdout"])
    off = [q.diag.get("C") for q, s in zip(probes_L, sel_probes)
           if not (q.mu.dtype == s["mu"].dtype and q.mu.tobytes() == s["mu"].tobytes()
                   and q.sd.dtype == s["sd"].dtype and q.sd.tobytes() == s["sd"].tobytes()
                   and np.asarray(q.w).dtype == np.asarray(s["w"]).dtype
                   and np.asarray(q.w).tobytes() == np.asarray(s["w"]).tobytes() and q.b == s["b"])]
    if off or len(sel_probes) != len(probes_L):
        raise SystemExit(f"ANCHOR FAILED: {len(off)} of {len(probes_L)} probes are not the selection's own fit bit for "
                         "bit (selection_tpr_v2.seed_probes); nothing written")
    tprs = [r["tpr"] for r in T.tprs_at(sel_probes, z_cal[kL], ~y_cal_v1, sr["target_fpr"])]
    diffs, n_cmp = T.anchor_diffs(tprs, sr["selected"])
    if diffs:
        raise SystemExit(f"ANCHOR FAILED: {len(diffs)} of {n_cmp} values are not equal to {sr['sel_rel']}; nothing "
                         "written")
    probe_anchor = {"selection_record": {"file": sr["sel_rel"], "sha256": _RESULT_READS[sr["sel_rel"]]},
                    "layer": L_sel, "probes_equal_to_selection_fit_bit_for_bit": True,
                    "ho20_tpr_per_seed_recomputed": tprs, "values_compared": n_cmp, "values_differing": 0}
    log(f"  ANCHOR OK at L* {L_sel}: probes bit for bit the selection's; {n_cmp} of {n_cmp} ho20 values equal")

    sc_by: dict[int, dict[str, np.ndarray]] = {}
    diag: dict[int, list[dict[str, Any]]] = {}
    gathered: dict[str, Any] = {}
    for li in [L_sel] + [x for x in layers if x != L_sel]:
        k = f"layer_{li}"
        probes = probes_L if li == L_sel else TP2.fit_probes(z_tr[k], y_tr, seeds, C_sel)
        sources = {"calibration": (z_cal[k], *cal_meta), "eval": (z_sub["eval"][k], *meta_sub["eval"]),
                   "pool4d": (z_sub["pool4d"][k], *meta_sub["pool4d"])}
        Xc, Xc_pos, cc, _ = F16.gather_split("calibration", v2["ids"], [0] * v2["n"], sources)
        Xn, Xp, cn, cp = F16.gather_split("eval", ev2["ids"], ev2["labels"], sources)
        if Xc_pos.shape[0] or cc != v2["plan_counts"] or cn != ev2["expect_neg"] or cp != ev2["expect_pos"]:
            refuse(f"pre layer {li}: v2 items per v1 split (calibration {cc}, eval negatives {cn}, eval positives "
                   f"{cp}) are not the v2 record's plan")
        gathered = {"calibration": cc, "eval_negatives": cn, "eval_positives": cp}
        sc_by[li] = TP2.score_matrices(probes, {"cal_pre": Xc, "ev_pre": Xn, "pos_pre": Xp,
                                                "xs_pre": z_xs["pre"][k], "xs_post": z_xs["post"][k]})
        diag[li] = [q.diag for q in probes]
        del sources, Xc, Xn, Xp
        log(f"  layer {li:>2} scored")
    for zz in (z_tr, z_cal, *z_sub.values(), *z_xs.values()):
        zz.close()
    caches = {f"pre/{SP.AXIS}/probe_train/{pooling}": r_tr, f"pre/{SP.AXIS}/calibration/{pooling}": r_cal}
    caches.update({f"pre/{T.SUB_TAG}/{SP.AXIS}/{part}/{pooling}": r_sub[part] for part in ("eval", "pool4d")})
    caches.update({f"{side}/{T.SUB_TAG}/{XS_SPLIT}/{pooling}": r_xs[side] for side in ("pre", "post")})
    return {"sc": sc_by, "x_con": x_con, "labels": np.asarray(meta_xs["pre"][1], dtype=np.int64), "tok": tok,
            "probe_anchor": probe_anchor, "caches": caches, "gathered": gathered, "probe_diagnostics": diag}


# ----------------------------------------------------------------------------- the per-pair job
def pair_stem(key: str, pooling: str) -> str:
    return f"overrefusal_p2_d440_{key}{sfx(pooling)}"


def run_pair(key: str, pooling: str = POOLING, n_draws: int = N_DRAWS, n_boot: int = N_BOOT,
             out: str | Path | None = None, records_only: bool = False, log: Callable[[str], None] = print) -> Path:
    t0 = time.time()
    out_path, npz_path, final = out_paths(pair_stem(key, pooling), out)
    ctx = bind(key, pooling, n_draws, n_boot, final, records_only)
    b, code0, xs = ctx["b"], ctx["code0"], ctx["xs"]
    if ctx["absent"]:
        refuse(f"{key} [{pooling}]: cache files absent from {SP.rel(SP.CACHE)}: {ctx['absent']}")
    if records_only:
        log(f"RECORDS OK: {key} [{pooling}] (records only; no cache read, nothing written); "
            f"{len(ctx['files'])} of {len(ctx['files'])} cache files present")
        return out_path
    registered = n_draws == N_DRAWS and list(b["seeds"]) == SEEDS and n_boot == N_BOOT
    layers, L_sel = list(b["layers"]), b["L_sel"]
    log(f"D-440 (8) {key} [{SP.AXIS} probe, XSTest, {pooling}]: {b['pre_repo']} -> {b['post_repo']}; {b['arm']} at L* "
        f"{L_sel}; layers {layers}; v2 {b['n_cal']} + {b['n_ev']} pooled negatives; {n_draws} draws per level; B "
        f"{n_boot}; registered {registered}")
    loaded = load(b, xs, log)
    x_con = loaded["x_con"]
    n_safe, n_con = int((~x_con).sum()), int(x_con.sum())
    if final and (n_safe, n_con) != (N_SAFE, N_CONTRAST):
        refuse(f"XSTest holds {n_safe} safe and {n_con} contrast prompts, not {N_SAFE} and {N_CONTRAST} (D-187)")
    draws = {LEVELS[0]: P.repartition_draws(b["n_cal"], b["n_ev"], n_draws, TP2.P2_DRAW_SEED),
             LEVELS[1]: P.cluster_repartition_draws(b["unit"], b["n_cal"], n_draws, TP2.P2_CLUSTER_DRAW_SEED)}
    CNT_S = P.counts_matrix(n_safe, n_boot, ITEM_BOOT_SEED["xstest_safe"])
    CNT_C = P.counts_matrix(n_con, n_boot, ITEM_BOOT_SEED["xstest_contrast"])
    meta = {"L_sel": L_sel, "g6": list(b["g6"]), "C": b["C_sel"], "n_seeds": len(b["seeds"])}
    cells: dict[int, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {"labels": loaded["labels"], "cell_layer": np.array(layers, dtype=np.int64)}
    ties = 0
    for li in layers:
        cells[li], arr, t = compute_cell(key, li, loaded["sc"][li], x_con, draws, b["d361_cells"].get(li), meta,
                                         CNT_S, CNT_C)
        ties += t
        cells[li]["probe_fit_diagnostics"] = loaded["probe_diagnostics"].get(li)
        arrays.update(arr)
        h = cells[li][LEVELS[0]]
        log(f"  layer {li:>2}: safe prompts flagged pre {h['safe_flagged_pre_mean']['mean']:.1f} -> post "
            f"{h['safe_flagged_post_mean']['mean']:.1f} of {n_safe}; D-361 pre side and threshold tie OK")
    if P.code_block(CODE_FILES)["sha256"] != code0["sha256"]:
        refuse("source changed during the run")
    caches = loaded["caches"]
    moved = TP2.end_hash_problems({f"data/hidden/{r['file']}": (SP.CACHE / r["file"], r["sha256"])
                                   for r in caches.values()})
    moved += [k for k, h in b["index_sha0"].items() if sha256_file(b["index_read"][k]) != h]
    moved += [m for m, h in b["manifests"].items() if sha256_file(RS.ROOT / m) != h]
    if final:
        moved += TP2.end_hash_problems({k: (TP2.resolve(k), h) for k, h in _RESULT_READS.items()})
    if moved:
        refuse(f"changed during the run: {moved}")
    files_sha = {f: sha256_file(SP.SPLIT_ROOT / f) for f in (T.V2_CAL_FILE, F16.V2_EVAL_FILE, XSTEST_FILE)}
    split_moved = [f"{f} (want {w[:16]})" for f, w in ((T.V2_CAL_FILE, b["v2"]["sha256"]),
                                                       (F16.V2_EVAL_FILE, b["ev2"]["sha256"]),
                                                       (XSTEST_FILE, XSTEST_SHA256)) if files_sha[f] != w]
    if split_moved:
        refuse(f"split files are not the bytes the records name at the end of the run: {split_moved}")
    files_sha.update(dict(b["index_sha0"]))
    files_sha.update(dict(_RESULT_READS))
    sr, post_sel = b["sr"], b["post_sel"]
    config = {"item": "D-440 (8): Phase 2 XSTest over-refusal with the Phase 1 estimator; descriptive",
              "pooling_role": "primary (D-361 (2))" if pooling == POOLING else "beside (D-361 (4), D-429's precedent)",
              "pair": key, "pair_group": b["group"], "pair_note": b["note"], "pre_repo": b["pre_repo"],
              "pre_rev": sr["rev"], "post_repo": b["post_repo"], "post_rev": post_sel["rev"], "pooling": pooling,
              "safety_axis": SP.AXIS, "external_set": "XSTest (over_refusal/eval), never re-partitioned (D-167)",
              "xstest_split": {"file": XSTEST_FILE, "sha256": XSTEST_SHA256, "n_safe": n_safe, "n_contrast": n_con},
              "selection_record": {"file": sr["sel_rel"], "sha256": _RESULT_READS[sr["sel_rel"]]},
              "selected_arm": b["arm"], "selected_C": b["C_sel"], "selected_layer": L_sel, "probe_set": "selected",
              "primary_row": {"layer": L_sel, "rule": "the D-117 selected layer L* (D-206 (2)); every other layer of "
                                                      "the D-361 set beside (D-189 (1))"},
              "grid6": list(b["g6"]), "layers": layers, "seeds": list(b["seeds"]), "target_fpr": TARGET,
              "n_draws": n_draws, "draw_seed": TP2.P2_DRAW_SEED, "cluster_draw_seed": TP2.P2_CLUSTER_DRAW_SEED,
              "pooled_negatives": {"n_calibration": b["n_cal"], "n_evaluation": b["n_ev"],
                                   "item_ids_sha256": sha256_bytes("\n".join(b["pool_ids"]).encode("utf-8"))},
              "item_bootstrap": {"n_boot": n_boot, "seeds": ITEM_BOOT_SEED,
                                 "counts_sha256": {"safe": sha256_bytes(CNT_S.tobytes()),
                                                   "contrast": sha256_bytes(CNT_C.tobytes())}},
              "blas_threads": blas_threads()}
    npz_bytes = savez_bytes(arrays)
    payload = {**provenance_block(RUN_KIND["pair"]), "preregistration": PREREG, "code": code0,
               "reused_files_sha256": reused_now(), "registered_parameters": registered,
               "status": status_block(registered), "elapsed_s": round(time.time() - t0, 1),
               "config": config, "config_hash": canonical_hash(config),
               "checks": {"d361_anchor_record": ctx["anchor_rec"], "probe_anchor_at_L_star": loaded["probe_anchor"],
                          "d361_pair_record": {"file": b["d361_rel"], "sha256": b["d361_sha"]},
                          "pre_side_equal_to_d361_record": {"levels": list(LEVELS), "layers": layers,
                                                            "metrics": list(O.PRE_ONLY_METRICS),
                                                            "counts": list(O.PRE_ONLY_COUNTS)},
                          "n_level_blocks_xstest_thresholds_tied_every_draw": ties,
                          "manifests": b["manifests"]},
               "tokenisation_xstest_pre_vs_post": loaded["tok"],
               "input_verification": TP2.D170_P2, "bos_check": b["bos"], "layer_plans": b["plans"],
               "v2_gather_pre_side": loaded["gathered"],
               "inputs": dict(_RESULT_READS), "input_files_sha256": files_sha,
               "input_cache_sha256": {k: r["sha256"] for k, r in caches.items()},
               "input_cache_source": {k: f"data/hidden/{r['file']} (expected bytes from {r['expected_from']})"
                                      for k, r in caches.items()},
               "scores_npz": {"file": A.rel(npz_path), "sha256": sha256_bytes(npz_bytes), "size": len(npz_bytes),
                              "keys": NPZ_KEYS},
               "cells": [cells[li] for li in layers]}
    problems = input_identity_problems(payload)
    if problems:
        refuse(f"to write (D-297 (1) results-writer rule): {problems}")
    TP2.write_outputs([(npz_path, npz_bytes), (out_path, json_bytes_of(payload))])
    log(f"elapsed {payload['elapsed_s']} s; {len(layers)} cells")
    return out_path


# ----------------------------------------------------------------------------- the summary
def verify_record(key: str, pooling: str, doc: dict[str, Any], arrays: dict[str, np.ndarray], code0: dict[str, Any],
                  final: bool) -> list[str]:
    """What keeps a pair record from entering the summary (empty when it may): its identity, and the frozen split's
    safe-prompt counts recomputed from its stored scores and thresholds at every layer."""
    cfg = doc.get("config") or {}
    out = []
    if doc.get("run_kind") != RUN_KIND["pair"] or cfg.get("pair") != key or cfg.get("pooling") != pooling:
        out.append("not this module's record of this pair and pooling")
    if (doc.get("code") or {}).get("sha256") != code0["sha256"]:
        out.append("written by other code")
    if final and doc.get("registered_parameters") is not True:
        out.append("not a run of the registered parameters")
    x_con = np.asarray(arrays.get("labels")) == 1
    for c in doc.get("cells") or []:
        li = c["layer"]
        try:
            thr = arrays[f"L{li}|thr_frozen"]
            got = [[int(x) for x in (arrays[f"L{li}|{s}"][~x_con] >= thr).sum(axis=0)] for s in ("xs_pre", "xs_post")]
        except KeyError:
            out.append(f"L{li}: stored scores missing")
            continue
        fz = c[LEVELS[0]]["frozen_split"]
        if got != [fz["safe_flagged_pre"], fz["safe_flagged_post"]]:
            out.append(f"L{li}: the stored scores do not reproduce the frozen split's counts")
    return out


def row_of(c: dict[str, Any]) -> dict[str, Any]:
    return {"layer": c["layer"], "grid_position": c["grid_position"], "is_selected_layer": c["is_selected_layer"],
            **{lvl: {**{q: c[lvl][q] for q in TABLE_QUANT},
                     "share_of_draws_mcnemar_safe_p_max_below_0_05": c[lvl]["share_of_draws_mcnemar_safe_p_max_below_0_05"],
                     "paired_refusal_ratio_share_of_draws_floored": c[lvl]["paired_refusal_ratio_share_of_draws_floored"],
                     "frozen_split_safe_flagged_pre_post": [c[lvl]["frozen_split"]["safe_flagged_pre"],
                                                            c[lvl]["frozen_split"]["safe_flagged_post"]]}
               for lvl in LEVELS},
            "xstest_auroc_change": c["xstest_auroc"]["change"]}


def run_summary(pooling: str = POOLING, out: str | Path | None = None, records_dir: str | Path | None = None,
                log: Callable[[str], None] = print) -> Path:
    stem = f"overrefusal_p2_d440_summary{sfx(pooling)}"
    out_path, _, final = out_paths(stem, out, with_npz=False)
    rd = Path(records_dir).resolve() if records_dir else RESULTS
    if final and rd != RESULTS.resolve():
        refuse("the final summary reads the records under results/ only")
    code0 = P.code_block(CODE_FILES)
    rows, cells, problems = {}, [], {}
    for key in PAIR_ORDER:
        jp = rd / f"{pair_stem(key, pooling)}.json"
        if not jp.is_file():
            problems[key] = [f"{A.rel(jp)} is absent"]
            continue
        doc = read_result(jp)
        npz_rel = (doc.get("scores_npz") or {}).get("file")
        np_path = rd / Path(str(npz_rel)).name
        if not np_path.is_file() or sha256_file(np_path) != (doc.get("scores_npz") or {}).get("sha256"):
            problems[key] = [f"{np_path.name} is absent or not the bytes the record names"]
            continue
        with np.load(np_path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        bad = verify_record(key, pooling, doc, arrays, code0, final)
        if bad:
            problems[key] = bad
            continue
        L_sel = doc["config"]["selected_layer"]
        rows[key] = {"pre_repo": doc["config"]["pre_repo"], "post_repo": doc["config"]["post_repo"],
                     "selected_C": doc["config"]["selected_C"],
                     "L_star": row_of(next(c for c in doc["cells"] if c["layer"] == L_sel)),
                     "record": {"file": A.rel(jp), "sha256": _RESULT_READS[A.rel(jp)],
                                "npz_sha256": doc["scores_npz"]["sha256"]}}
        cells += [{"pair": key, **row_of(c)} for c in doc["cells"]]
    if problems:
        refuse(f"pair records not usable: {problems}")
    if len(cells) != len(PAIRS) * LAYERS_PER_PAIR:
        refuse(f"{len(cells)} cells, not {len(PAIRS) * LAYERS_PER_PAIR}")
    payload = {**provenance_block(RUN_KIND["summary"]), "preregistration": PREREG, "code": code0,
               "registered_parameters": final, "status": status_block(final),
               "config": {"pooling": pooling, "pairs": PAIR_ORDER, "levels": list(LEVELS), "quantities": list(TABLE_QUANT),
                          "primary": "per pair the L* row (D-206 (2)), selected C, item level, 1%; the cluster level "
                                     "and the other six layers beside"},
               "inputs": dict(_RESULT_READS), "input_files_sha256": dict(_RESULT_READS),
               "l_star_rows": rows, "cells": cells}
    TP2.write_outputs([(out_path, json_bytes_of(payload))])
    return out_path


# ----------------------------------------------------------------------------- preflight and the lane list
def run_preflight(pooling: str = POOLING, log: Callable[[str], None] = print) -> int:
    bad = 0
    for key in PAIR_ORDER:
        try:
            ctx = bind(key, pooling, N_DRAWS, N_BOOT, False, True, gates=True)
            if ctx["absent"]:
                raise SystemExit(f"REFUSING: cache files absent: {ctx['absent']}")
            log(f"RECORDS OK: {key} [{pooling}]; {len(ctx['files'])} of {len(ctx['files'])} cache files present")
        except SystemExit as e:
            bad += 1
            log(f"PREFLIGHT FAILED: {key} [{pooling}]: {e}")
    return 1 if bad else 0


QUEUE_NAME = "laneA_20261013_xstest_p2.tsv"
QUEUE_HEAD = [
    "# DO NOT LOAD before review and the sandbox",
    "# Lane A (CPU only, no network, no GPU; D-440 (8), the author's optional item 8 of 2026-10-13, PRE-REGISTERED in "
    "docs/reports_d440/item8_xstest_p2_prereg.md, DESCRIPTIVE: no prediction): the Phase 2 XSTest over-refusal column "
    "with the Phase 1 estimator (overrefusal_d156.xstest_point, D-167/D-187/D-189), by src/overrefusal_p2_d440.py. "
    "Nine D-361 headline pairs x 7 layers (GRID6 + the pre side's L*), the WildGuardMix probe of the pre side's "
    "selection, split v2, 1000 draws, item and cluster level, 1%; user-span mean primary, last-token beside.",
    "# BEFORE LOADING: open question 1 settled by D-334 (D-440 (8)): the WildGuardMix probe; one reviewer (sonnet) on "
    "the module and its test; the push that carries them and this list.",
    "# ORDER: (0) the sandbox lane job (scripts/sandbox_xstest_p2.sh: tests, both preflights, one pair job per pooling "
    "in a scratch tree); its failure stops the lane. Then per pooling, user-span mean first: its preflight (records and file presence only; exit 1 on a failure), "
    "the nine pair jobs (Mistral v0.1 -> v0.2 first), their summary; then the same for last-token. Each pair job refuses unless the D-361 "
    "anchor record passes, the L* probes are the selection's bit for bit, the pre side equals the pair's D-361 record "
    "at every layer and level, and the XSTest thresholds are tied to the shared estimator in every draw.",
    "# PRICE, AN ESTIMATE (not a measurement): Phase 1's two _xsfull re-runs (D-253: Qwen2.5 21:11:42Z-21:22:58Z, "
    "Mistral v0.1 -> v0.2 to 21:33:11Z) took about 11 and 10 min for 30 and 34 cells (two probe sets) at two levels; "
    "a Phase 2 pair is 7 cells at two levels, plus loading the pre side's four WildGuardMix caches and two XSTest "
    "caches, which D-424's pair jobs (which also load the post side and bootstrap 10,000 items) did in 364-540 s each "
    "including their arithmetic. Expect at most about 10 min a pair, about 3 h for both poolings; the sandbox measures it.",
    "# Final names need the registered parameters, the 6-thread BLAS pin (scripts/project_env.sh), GUARDRAIL_CACHE_DIRS "
    "unset and the reused files at REUSED_SHA256. Runs from the repo root through scripts/detach.sh on the NODE. "
    "Labels checked unique against queues/lane*_*.tsv; check them against the node's logs/queue/ before loading.",
]


def lane_jobs() -> list[tuple[str, str, str]]:
    jobs: list[tuple[str, str, str]] = [("xstest_p2_sandbox", "-", "bash scripts/sandbox_xstest_p2.sh "
                                         f"{os.environ.get('GUARDRAIL_NODE_HOME') or Path.home().as_posix()}/.cache/gccf_scratch/xstest_p2_sandbox_20261013")]
    for pooling in POOLINGS:
        tag = "" if pooling == POOLING else "_lastgen"
        jobs.append((f"xstest_p2_preflight{tag}", "-",
                     f"python3 -u src/overrefusal_p2_d440.py preflight --pooling {pooling}"))
        for key in PAIR_ORDER:
            st = pair_stem(key, pooling)
            jobs.append((f"xstest_p2_{key}{tag}", f"results/{st}.json results/{st}_scores.npz",
                         f"python3 -u src/overrefusal_p2_d440.py pair --pair {key} --pooling {pooling}"))
        jobs.append((f"xstest_p2_summary{tag}", f"results/overrefusal_p2_d440_summary{tag}.json",
                     f"python3 -u src/overrefusal_p2_d440.py summary --pooling {pooling}"))
    return jobs


def lane_list_text() -> str:
    return "\n".join(QUEUE_HEAD + ["\t".join(j) for j in lane_jobs()]) + "\n"


def label_conflicts(text: str, root: Path, name: str = QUEUE_NAME) -> list[str]:
    mine = [ln.split("\t")[0] for ln in text.splitlines() if ln and not ln.startswith("#")]
    out = [f"duplicate label {x}" for x in sorted({x for x in mine if mine.count(x) > 1})]
    for f in sorted((root / "queues").glob("lane*_*.tsv")):
        if f.name == name:
            continue
        theirs = {ln.split("\t")[0] for ln in f.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")}
        out += [f"{x} also in {f.name}" for x in mine if x in theirs]
    return out


def run_queue(write: bool = False, root: Path | None = None) -> int:
    root = root or ROOT
    text = lane_list_text()
    bad = label_conflicts(text, root)
    if bad:
        refuse(f"label conflicts: {bad}")
    p = root / "queues" / QUEUE_NAME
    if write:
        if p.exists():
            refuse(f"{p} exists; never overwritten")
        p.write_bytes(text.encode("utf-8"))
        print(f"Written: {A.rel(p)}")
    else:
        sys.stdout.write(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="D-440 (8): Phase 2 XSTest over-refusal with the Phase 1 estimator")
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("pair")
    p.add_argument("--pair", required=True, choices=PAIRS)
    p.add_argument("--pooling", default=POOLING, choices=list(POOLINGS))
    p.add_argument("--n-draws", type=int, default=N_DRAWS)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--out", default=None)
    p.add_argument("--records-only", action="store_true")
    s = sub.add_parser("summary")
    s.add_argument("--pooling", default=POOLING, choices=list(POOLINGS))
    s.add_argument("--out", default=None)
    s.add_argument("--records-dir", default=None)
    f = sub.add_parser("preflight")
    f.add_argument("--pooling", default=POOLING, choices=list(POOLINGS))
    q = sub.add_parser("queue")
    q.add_argument("--write", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    a = build_parser().parse_args(argv)
    if a.mode == "pair":
        run_pair(a.pair, a.pooling, a.n_draws, a.n_boot, a.out, a.records_only)
        return 0
    if a.mode == "summary":
        run_summary(a.pooling, a.out, a.records_dir)
        return 0
    if a.mode == "preflight":
        return run_preflight(a.pooling)
    return run_queue(a.write)


if __name__ == "__main__":
    raise SystemExit(main())
