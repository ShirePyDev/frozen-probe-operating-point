"""D-282 for the SECONDARY mean-difference probe (item 19; D-220, D-246, D-251; populations.MEANDIFF_D156): the combined
item-bootstrap x re-partition interval (D-281 (1)) for TPR pre, TPR post and delta TPR at the frozen threshold, beside
the D-156 band, as a SUPPLEMENT record (D-287's precedent: no source record is re-issued). It is the last open D-282 TPR
column, docs/phase1_exit.md section 3 item 3, "meandiff (secondary)".

WHAT THE RECORDS QUOTE. src/meandiff_d156.py --r2 wrote results/meandiff_d156_<pair>_r2.json for the nine released
repair pairs (populations.UPDATE_PAIRS_REPAIR; D-299 (4), (6); logs/meandiff_d156_r2_20261001.sha256). Each quotes TPR
pre, TPR post and delta TPR (paired.repartition_estimate's operating point, at 1% only) for
  - the mean_diff probe at its own layer L_md (the D-117 criterion restricted to the selection's mean_diff arm, the first
    maximum of the mean holdout AUROC), item level (D-156) and cluster level (D-164); and
  - beside it, the logistic selected-C probe at the logistic selected layer L* and at L_md (logistic_beside), copied
    from the pair's depth record results/depth_d156_<pair>.json, which the meandiff record names by sha256 (depth_file).
For each pair this script re-does meandiff_d156's work at L_md only (the same caches through the input registry, the
selection of record, the D-117 80% fit split per seed, meandiff_d156.meandiff_probe unchanged, and the same item-level
D-156 draws), and depth_d156's work for the logistic rows at the logistic_beside layers (table3_tpr_d282.probe_scores,
unchanged). Then, per block (pair, probe, layer):
  - ANCHOR: the item-level operating point equals the record's as canonical JSON text: the meandiff record's for the
    mean_diff block; for a logistic block the depth record's cell, with its per-draw ratio_to_target hash
    (table3_tpr_d282.anchored_replicates), and the meandiff record's logistic_beside copy. The draws are one array per
    pair and axis, made by the identical call P.repartition_draws(n_cal, n_pool - n_cal, n_draws,
    P.AXIS_DRAW_SEED[axis]) in meandiff_d156.py (line 180), depth_d156.py (line 375) and here. The meandiff record stores
    no per-draw array; the link for the draw ORDER, which the pairing of draw d_r with counts row r needs, is the depth
    record's per-draw ratio_to_target hash, checked on every logistic block computed on that same array, which the
    mean_diff block then uses. The mean_diff fits must also give the record's selection_check AUROCs exactly, and L_md
    must follow the record's rule from the selection of record;
  - ONES CHECK: counts of all ones give back the D-156 per-draw TPR arrays bit for bit (D-282 (4));
  - SHARED PRE SIDE: llama3_base2instruct and llama3_to_31 share the pre model and axis (so their selection and layers),
    and every block of the second must give the first's TPR-pre replicates;
  - TABLE 1 CROSS-CHECK: a logistic block that Table 1's record (results/table1_tpr_d282_r2.json, D-305) also holds, on
    the same depth record bytes, must have its replicate hashes (L*, its headline, always).
paired.combined_tpr_replicates gives the combined interval: replicate r = 1..5000 uses draw d_r = 1 + ((r - 1) mod 1000)
and row r of paired.counts_matrix(n positives, 5000, depth_d156.ITEM_BOOT_SEED["eval_positives"]); the same counts weight
the pre and post flags and every seed column (paired within item), rates take P.rate's weighted form, and the interval is
the 2.5-97.5 percentiles. The D-156 band, its mean and the frozen split's value are copied beside it from the record.

CHOICES by the closest D-entry (D-181), each also in the record's implementation_choices (CHOICES below):
  (1) SUPPLEMENT (D-287 (1)): results/meandiff_tpr_d282.json; the meandiff, depth and Table 1 records are untouched and
      anchored exactly.
  (2) SCOPE (D-281 (1), D-287 (2), D-251): tpr_pre, tpr_post and delta_tpr at every layer where a meandiff record quotes
      them: the mean_diff probe at L_md, and the logistic selected-C rows beside it at L* and L_md (its logistic_beside
      layers), at 1% only: meandiff_d156 has no 5% record to anchor on (D-251 ran the registered 1%).
  (3) NOT COVERED (D-282 (2), D-251): the cluster level keeps its D-156 band only; AUROC is not a TPR quantity; there is
      no between-probe contrast (D-251: descriptive only, D-220 registered none), so no mean_diff-minus-logistic interval.
  (4) DRAW ORDER (D-287, D-282 (2)): the draws are one array per pair and axis, from the identical repartition_draws
      call in meandiff_d156.py and depth_d156.py; the meandiff record has no per-draw hash, so the link is the depth
      record's per-draw ratio_to_target hash (the bytes the meandiff record names), checked on every logistic block
      computed on the array the mean_diff block uses. Each row records the src/paired.py sha256 the meandiff record ran
      with beside this run's (no refusal). The mean_diff block's per-draw hashes are recorded here, so later work can
      anchor on them.
  (5) COUNTS (D-287 (3)): depth_d156's eval_positives stream, a function of n positives; the eval positives are the same
      items in the same order on each axis (checked by an ordered item-id hash), so one matrix pairs by item across rows.
  (6) BINDING (D-299 (4), D-305, D-356 (2)(c)): each meandiff record by logs/meandiff_d156_r2_20261001.sha256, its depth
      record by the meandiff record's depth_file sha256, Table 1's record by logs/reissue_d305_20261001.sha256; the final
      name needs each manifest read to be the bytes pinned in MANIFEST_SHA256. The records must be the populations'
      records of record: the pairs' meandiff records are populations.MEANDIFF_D156, Table 1's is in DEPTH_D156_D282, and
      none of them nor any depth record read is in populations.SUPERSEDED (D-299 (6), D-305).
  (7) SHARED PRE (D-287 (5)): every (pre repo and revision, axis, probe, C, layer) seen again gives identical TPR-pre
      replicates, and a pair whose pre side (repo, revision, axis) an earlier pair had must have every block so checked
      (the selection reads the pre side only, so the layers are the same).
  (8) TABLE 1 CROSS-CHECK (D-282 (3), D-305): the logistic blocks Table 1's record holds (always L*) must have its
      replicate hashes, on the same depth record bytes; skipped, and said so, only when this run's parameters are not the
      Table 1 record's (a test run); the final name needs it.
  (9) FLAGS (D-172, D-331): the D-170 flag per row and per file from the input registry now; the meandiff record's and
      the depth record's flags are each compared with it and carried beside it (the depth record's as
      table3_tpr_d282.py does). A record's pin-only flag that the final D-168 record has since cleared (D-331) is
      accepted when the pin-only inputs are the same, and the row says so; any other difference refuses.
  (10) IDENTITY (D-297 (1), D-300): the cache inputs are keyed by the meandiff record's stem and named through
      provenance.identity_config (from that record's config); every results file is read through read_result and listed
      with its sha256 (input_files_sha256, with the manifests and, when the registry read it for a pin-only input, the
      final D-168 record at its pinned sha256); each cache's resolved path is recorded, and the final name refuses a set
      GUARDRAIL_CACHE_DIRS.
  (11) SECONDARY (D-220, D-251): the record says so; population group MEANDIFF_D156_D282, declared in src/populations.py
      when the record is pulled (D-287 (6)); it enters no primary table.
  (12) RELEASE (D-179; as table3_tpr_d282.py, D-287): RESEARCH.
  (13) REVIEW (D-271 (1), D-282 (4)): new analysis code, three reviewers before the run of record on a node lane.
It writes results/meandiff_tpr_d282.json once (provenance.write_results, exclusive: a verified temporary, then a hard
link); only the registered parameters, all nine pairs, the 6-thread BLAS pin, the pinned manifests and an unset
GUARDRAIL_CACHE_DIRS may write the final name. A test writes outside results/.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_d156 as D  # noqa: E402
import input_registry as REG  # noqa: E402
import meandiff_d156 as MD  # noqa: E402
import metrics as MET  # noqa: E402
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
import table1_tpr_d282 as T1  # noqa: E402
import table3_tpr_d282 as T3  # noqa: E402
from provenance import (  # noqa: E402
    _RESULT_READS, canonical_hash, ensure_utf8_stdout, identity_config, provenance_block, read_result, resolve_cache,
    sha256_bytes, sha256_file, write_results,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/meandiff_tpr_d282.py", "src/meandiff_d156.py", "src/table3_tpr_d282.py", "src/table1_tpr_d282.py",
              *D.CODE_FILES]
QUANT = T1.QUANT
PAIRS = list(POP.UPDATE_PAIRS_REPAIR)                         # the nine meandiff records' pairs, in that order
PARTS = ("probe_train", "threshold_calib", "eval")
IDENTITY_KEYS = ("pre_repo", "pre_rev", "post_repo", "post_rev")
SELECTION = D.SELECTION_OF_RECORD                             # D-171: the selection meandiff_d156 read
MEANDIFF_MANIFEST = ROOT / "logs" / "meandiff_d156_r2_20261001.sha256"        # D-299 (4): the nine _r2 records
TABLE1_MANIFEST = ROOT / "logs" / "reissue_d305_20261001.sha256"              # D-305: table1_tpr_d282_r2 among them
TABLE1_RECORD = "results/table1_tpr_d282_r2.json"                            # populations.DEPTH_D156_D282 (D-305)
# the final name needs these manifest bytes (as table3_tpr_d282, D-356 (2)(c); each value the committed file's sha256)
MANIFEST_SHA256 = {"meandiff_d156_r2_20261001.sha256": "dcf9194e36565030d500ac548045f46482df5ad6ccbf4afc1f110fd7813375f6",
                   "reissue_d305_20261001.sha256": "10f8e9d40e640584c19a34063f041f3c3895c7d36f3025dd1ba9b0ccdab702c6"}
SECONDARY = {"decision": "D-251",
             "status": "SECONDARY (D-220, D-251): the mean_diff standardisation check, populations.MEANDIFF_D156. This "
                       "supplement gives its TPR quantities D-282's combined interval; it enters no primary table, and "
                       "no between-probe contrast is formed (D-251: descriptive only)"}
NOTE = ("D-282 (D-281 (1)): combined 95% interval, the 2.5-97.5 percentiles over n_boot replicates. Replicate r is "
        "evaluated at item-level draw d_r = 1 + ((r - 1) mod n_draws) with row r of the axis's item-bootstrap counts over "
        "the eval positives, paired within item (the same counts weight pre and post and every seed). The D-156 band "
        "(partition variation only), its mean and the frozen split's value are copied beside it: for the mean_diff "
        "block from the meandiff record's operating point, for a logistic block from the depth record's cell")
CHOICES = (
    {"choice": "a supplement, not a re-issue: the meandiff, depth and Table 1 records are untouched and anchored exactly",
     "rule": "D-287 (1)"},
    {"choice": "tpr_pre, tpr_post and delta_tpr at every layer where a meandiff record quotes them: the mean_diff probe at "
               "L_md and the logistic selected-C rows beside it at L* and L_md (logistic_beside), at 1% only (no 5% "
               "meandiff record exists to anchor on)",
     "rule": "D-281 (1), D-287 (2), D-251"},
    {"choice": "the cluster level, AUROC and any between-probe contrast get no D-282 interval",
     "rule": "D-282 (2); D-251 (descriptive only, D-220 registered no contrast)"},
    {"choice": "the draws are one array per pair and axis from the identical repartition_draws call in meandiff_d156 and "
               "depth_d156; the meandiff record holds no per-draw array, so the link for the draw order is the depth "
               "record's per-draw ratio_to_target hash, checked on every logistic block on the array the mean_diff block "
               "uses; the meandiff record's src/paired.py sha256 is recorded beside this run's; the mean_diff block's "
               "per-draw hashes are recorded", "rule": "D-287, D-282 (2)"},
    {"choice": "counts: depth_d156's eval_positives stream, one matrix per number of positives (same positives, same "
               "order on each axis, checked)", "rule": "D-287 (3), D-282 (2)"},
    {"choice": "binding: each meandiff record by logs/meandiff_d156_r2_20261001.sha256, its depth record by its depth_file "
               "sha256, Table 1's record by logs/reissue_d305_20261001.sha256; the final name needs both manifests at the "
               "pinned bytes; the records are the populations' (MEANDIFF_D156, DEPTH_D156_D282) and none read is "
               "SUPERSEDED", "rule": "D-299 (4), D-299 (6), D-305, D-356 (2)(c)"},
    {"choice": "shared pre side: blocks with the same pre repo and revision, axis, probe, C and layer give identical TPR-pre "
               "replicates; a pair sharing an earlier pair's pre side must have every block so checked",
     "rule": "D-287 (5)"},
    {"choice": "Table 1 cross-check: a logistic block Table 1's record holds on the same depth record bytes has its "
               "replicate hashes (L* always); skipped only for a test run whose parameters are not Table 1's",
     "rule": "D-282 (3), D-305"},
    {"choice": "the D-170 flag per row and per file from the registry now, the meandiff record's and the depth record's "
               "compared and carried beside; a pin-only flag the final D-168 record has since cleared is accepted with "
               "the same pin-only inputs, any other difference refuses", "rule": "D-172, D-331"},
    {"choice": "cache inputs keyed by the meandiff record's stem and named through identity_config; every results file "
               "read through read_result and listed with its sha256, and the final D-168 record when the registry read "
               "it; each cache's resolved path recorded; the final name refuses a set GUARDRAIL_CACHE_DIRS",
     "rule": "D-297 (1), D-300"},
    {"choice": "secondary: population group MEANDIFF_D156_D282, declared when the record is pulled; no primary table",
     "rule": "D-220, D-251, D-287 (6)"},
    {"choice": "release class RESEARCH", "rule": "D-179, D-287"},
    {"choice": "new analysis code: three reviewers before the run of record", "rule": "D-271 (1), D-282 (4)"},
)


def rel_path(p: Path) -> str:
    p = Path(p)
    return str(p.relative_to(ROOT)).replace("\\", "/") if p.is_relative_to(ROOT) else p.name


def path_text(p: Path) -> str:
    """A path as recorded: repo-relative when under the repository, else absolute."""
    p = Path(p)
    return str(p.relative_to(ROOT)).replace("\\", "/") if p.is_relative_to(ROOT) else str(p)


def read_listed(rel: str) -> dict[str, Any]:
    """A results file read through read_result (D-297 (1)); refuses unless the read is registered under `rel`."""
    doc = read_result(ROOT / rel)
    if rel not in _RESULT_READS:
        raise SystemExit(f"REFUSING: {rel} was not registered under its repo-relative name")
    return doc


def record_stem(pair: str) -> str:
    """populations.MEANDIFF_D156's name of the pair's record."""
    return f"meandiff_d156_{pair[len('repair_'):]}_r2"


def population_check(run_pairs: list[str]) -> None:
    """CHOICES (6): the records read are the populations' records of record: the pairs' meandiff records are
    populations.MEANDIFF_D156 in order, Table 1's record is in DEPTH_D156_D282, and none of them nor any depth record
    read is SUPERSEDED."""
    if [record_stem(p) for p in PAIRS] != list(POP.MEANDIFF_D156):
        raise SystemExit("REFUSING: the pairs' meandiff records are not populations.MEANDIFF_D156")
    t1_stem = Path(TABLE1_RECORD).stem
    if t1_stem not in POP.DEPTH_D156_D282:
        raise SystemExit(f"REFUSING: {t1_stem} is not in populations.DEPTH_D156_D282")
    names = [t1_stem, *(record_stem(p) for p in run_pairs), *(D.out_name(f"depth_{p[len('repair_'):]}") for p in run_pairs)]
    superseded = [n for n in names if n in POP.SUPERSEDED]
    if superseded:
        raise SystemExit(f"REFUSING: records that populations.SUPERSEDED lists would be read: {superseded}")


def meandiff_layer(grid: dict[str, dict[str, float]]) -> int:
    """meandiff_d156's rule (its main()'s lines, repeated): the mean_diff arm's highest mean holdout AUROC, the first
    layer on a tie (the sweep's strict '>')."""
    L, best = None, None
    for lk in sorted(grid, key=int):
        if best is None or grid[lk]["auroc_ho20"] > best:
            L, best = int(lk), grid[lk]["auroc_ho20"]
    if L is None:
        raise SystemExit("REFUSING: the selection's mean_diff arm holds no layer")
    return L


def flag_status(now: dict[str, Any], rec: dict[str, Any], where: str) -> str:
    """CHOICES (9): how the D-170 flag of the inputs read now stands to the record's; refuses a difference other than
    a pin-only flag the final D-168 record has since cleared (D-331) with the same pin-only inputs."""
    if sorted(now.get("pin_only_inputs") or []) != sorted(rec.get("pin_only_inputs") or []):
        raise SystemExit(f"REFUSING: {where}: the pin-only inputs differ from the record's")
    if now.get("flag") == rec.get("flag"):
        return "equal to the record's"
    cleared = now.get("pin_flag_cleared_by") or {}
    if rec.get("flag") == REG.PIN_ONLY_FLAG and now.get("flag") is None and cleared.get("sha256"):
        return (f"the record's pin-only flag, cleared since the record by the final D-168 record "
                f"({cleared.get('decision')}, {cleared.get('file')})")
    raise SystemExit(f"REFUSING: {where}: the D-170 flag differs from the record's ({now.get('flag')!r} against "
                     f"{rec.get('flag')!r})")


def mean_diff_scores(X: dict[tuple[str, str], np.ndarray], y_tr: np.ndarray, masks: dict[str, np.ndarray],
                     seeds: list[int]) -> tuple[dict[str, np.ndarray], float, float]:
    """meandiff_d156's probes and scores at its layer: per seed the D-117 80% fit split and meandiff_probe (unchanged);
    the mean train and holdout AUROCs (its selection check); scores (items, seeds) as table3_tpr_d282.probe_scores
    returns them."""
    probes, fit_au, ho_au = [], [], []
    Xtr = X[("pre", "probe_train")]
    for seed in seeds:
        ho, fit = P.d117_split(len(y_tr), seed)
        q = MD.meandiff_probe(Xtr[fit], y_tr[fit])
        probes.append(q)
        s_fit, s_ho = q.score(Xtr[fit]), q.score(Xtr[ho])
        fit_au.append(MET.auroc(s_fit[y_tr[fit]], s_fit[~y_tr[fit]]).value)
        ho_au.append(MET.auroc(s_ho[y_tr[ho]], s_ho[~y_tr[ho]]).value)
    S = lambda side, part, m: np.column_stack([q.score(X[(side, part)][m]) for q in probes])  # noqa: E731
    sc = {"cal_pre": S("pre", "threshold_calib", masks["ncal"]), "ev_pre": S("pre", "eval", masks["neg"]),
          "cal_post": S("post", "threshold_calib", masks["ncal"]), "ev_post": S("post", "eval", masks["neg"]),
          "pos_pre": S("pre", "eval", masks["pos"]), "pos_post": S("post", "eval", masks["pos"])}
    return sc, float(np.mean(fit_au)), float(np.mean(ho_au))


def record_anchored(sc: dict[str, np.ndarray], draws: np.ndarray, target: float, op: dict[str, Any] | None,
                    counts: np.ndarray, where: str
                    ) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The mean_diff block: table3_tpr_d282.anchored_replicates without the per-draw hash, which the meandiff record does
    not hold (the draw order is pinned by the logistic block at the same layer, CHOICES (4)). The D-156 report must equal
    the record's operating point `op` as canonical JSON; counts of ones must give back the per-draw TPR arrays bit for
    bit; returns (the report, its per-draw arrays, {quantity: combined replicates})."""
    n_cal = sc["cal_pre"].shape[0]
    report, arr = P.repartition_estimate(sc["cal_pre"], sc["ev_pre"], sc["cal_post"], sc["ev_post"],
                                         sc["pos_pre"], sc["pos_post"], draws, target)
    if op is None or T1.canon(report) != T1.canon(op):
        raise SystemExit(f"REFUSING: {where}: the operating point is not the meandiff record's")
    pool_pre = np.vstack([sc["cal_pre"], sc["ev_pre"]])
    n_seeds = pool_pre.shape[1]
    T_dr = np.array([[P.thr_at_fpr(pool_pre[d[:n_cal], s], target) for s in range(n_seeds)] for d in draws])
    n_pos = sc["pos_pre"].shape[0]
    ones = P.combined_tpr_replicates(T_dr, sc["pos_pre"], sc["pos_post"], np.ones((len(draws), n_pos), dtype=np.float32))
    if any(not np.array_equal(ones[q], arr[q]) for q in QUANT):
        raise SystemExit(f"REFUSING: {where}: counts of ones do not give back the D-156 draws (D-282 (4))")
    return report, arr, P.combined_tpr_replicates(T_dr, sc["pos_pre"], sc["pos_post"], counts)


def per_draw_hashes(report: dict[str, Any], arr: dict[str, np.ndarray]) -> dict[str, str]:
    """CHOICES (4): sha256 of the mean_diff block's per-draw arrays (float64 little-endian, draws in order): the
    ratio_to_target array with the frozen split first, as depth_d156 hashes it, and each TPR quantity's draws."""
    rt = np.concatenate([[report["ratio_to_target"]["frozen"]], arr["ratio_to_target"]]).astype("<f8")
    return {"ratio_to_target_frozen_first": sha256_bytes(rt.tobytes()),
            **{q: sha256_bytes(np.asarray(arr[q]).astype("<f8").tobytes()) for q in QUANT}}


def tpr_block(op: dict[str, Any], reps: dict[str, np.ndarray]) -> dict[str, Any]:
    """table1/table3_tpr_d282's block: frozen, D-156 mean and band copied from `op`, the combined interval, the hash."""
    return {q: {"frozen": op[q]["frozen"], "mean_D156": op[q]["mean"], "band95_D156": op[q]["band95"],
                "combined": P.ci(reps[q]), "replicates_sha256": sha256_bytes(reps[q].astype("<f8").tobytes())}
            for q in QUANT}


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="D-282's combined TPR interval for the secondary mean_diff records")
    ap.add_argument("--pairs", nargs="+", default=None, choices=PAIRS, help="a subset of the pairs (a test)")
    ap.add_argument("--n-draws", type=int, default=D.REGISTERED["n_draws"])
    ap.add_argument("--n-boot", type=int, default=D.REGISTERED["n_boot"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    target, seeds = D.REGISTERED["target_fpr"], D.REGISTERED["seeds"]
    run_pairs = list(PAIRS) if args.pairs is None else list(args.pairs)
    if len(set(run_pairs)) != len(run_pairs):
        raise SystemExit(f"REFUSING: {run_pairs} are not distinct pairs")
    registered = (args.n_draws == D.REGISTERED["n_draws"] and args.n_boot == D.REGISTERED["n_boot"] and run_pairs == PAIRS)
    final_path = (RESULTS / "meandiff_tpr_d282.json").resolve()
    out_path = Path(args.out).resolve() if args.out else final_path
    final = out_path == final_path
    if out_path.exists():
        raise SystemExit(f"REFUSING: {rel_path(out_path)} exists; results are never overwritten")
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test run (--out) must not write under results/")
    threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    cache_dirs = os.environ.get("GUARDRAIL_CACHE_DIRS")
    (md_man, md_h), (t1_man, t1_h) = T3.manifest(MEANDIFF_MANIFEST), T3.manifest(TABLE1_MANIFEST)
    man_sha = {MEANDIFF_MANIFEST: md_h, TABLE1_MANIFEST: t1_h}
    if final:
        problems = []
        if not registered:
            problems.append(f"registered {registered} (the registered n_draws and n_boot and all nine pairs)")
        if any(t != "6" for t in threads.values()):
            problems.append(f"threads {threads} (the 6-thread BLAS pin)")
        if cache_dirs:
            problems.append(f"GUARDRAIL_CACHE_DIRS is set ({cache_dirs!r}); the caches must come from the registry's directory")
        unpinned = [rel_path(p) for p, h in man_sha.items() if MANIFEST_SHA256.get(p.name) != h]
        if unpinned:
            problems.append(f"manifests {unpinned} are not the pinned bytes")
        if problems:
            raise SystemExit(f"REFUSING the final name: {problems}")
    population_check(run_pairs)
    code0 = P.code_block(CODE_FILES)
    sel_rel = rel_path(SELECTION)
    sel_doc = read_listed(sel_rel)
    sel_sha = _RESULT_READS[sel_rel]
    if list(sel_doc["config"]["seeds"]) != seeds or float(sel_doc["config"]["holdout"]) != P.HOLDOUT:
        raise SystemExit("REFUSING: the selection's seeds or holdout differ from the registered ones")
    reg_sources = REG.sources_sha256()
    if D.sources_now() != reg_sources:
        raise SystemExit("REFUSING: the extraction records or pins on disk are not the ones the registry read")
    t1 = read_listed(TABLE1_RECORD)                           # CHOICES (8)
    t1_sha = _RESULT_READS[TABLE1_RECORD]
    if t1_man.get(TABLE1_RECORD) != t1_sha:
        raise SystemExit(f"REFUSING: {TABLE1_RECORD} is not the record {rel_path(TABLE1_MANIFEST)} pins (D-305)")
    t1c = t1.get("config") or {}
    t1_comparable = ([t1c.get("target_fpr"), t1c.get("seeds"), t1c.get("n_draws"), t1c.get("n_boot"), t1c.get("counts_seed")]
                     == [target, seeds, args.n_draws, args.n_boot, D.ITEM_BOOT_SEED["eval_positives"]])
    if registered and not t1_comparable:
        raise SystemExit(f"REFUSING: {TABLE1_RECORD} was made with other parameters; the registered run must cross-check it")

    counts: dict[int, np.ndarray] = {}
    positives_id: dict[str, str] = {}
    shared_pre: dict[tuple, tuple[str, str]] = {}
    pre_seen: dict[tuple[str, str, str], str] = {}
    first_read: dict[str, str] = {sel_rel: sel_sha, TABLE1_RECORD: t1_sha}
    input_recs: dict[str, dict[str, Any]] = {}
    resolved: dict[str, Path] = {}                            # input key -> the cache file read (CHOICES (10))
    out_rows: dict[str, Any] = {}
    n_md = n_log = n_ones = n_shared = n_t1 = 0
    want_log = 0                                              # from the meandiff records' logistic_beside, not this loop
    for pair in run_pairs:
        core = pair[len("repair_"):]
        stem, mstem = f"depth_{core}", record_stem(pair)
        reg_stem = D.registry_stem(stem)
        md_rel = f"results/{mstem}.json"
        md = read_listed(md_rel)
        md_sha = _RESULT_READS[md_rel]
        first_read[md_rel] = md_sha
        if md_man.get(md_rel) != md_sha:
            raise SystemExit(f"REFUSING: {md_rel} is not the meandiff record {rel_path(MEANDIFF_MANIFEST)} pins (D-299)")
        if md.get("run_kind") != "meandiff_d156" or (md.get("population") or {}).get("group") != "MEANDIFF_D156":
            raise SystemExit(f"REFUSING: {md_rel} is not a meandiff_d156 record of population MEANDIFF_D156")
        if registered and md.get("registered_parameters") is not True:
            raise SystemExit(f"REFUSING: {md_rel} was not run with the registered parameters")
        mc = md["config"]
        cfg = REG.run(reg_stem)
        axis = cfg["axis"]
        if (mc.get("pair"), mc.get("registry_pair"), mc.get("axis")) != (pair, reg_stem, axis) or \
                any(mc.get(k) != cfg[k] for k in IDENTITY_KEYS):
            raise SystemExit(f"REFUSING: {md_rel} names another pair, axis, repo or revision than the input registry's {reg_stem}")
        if [list(mc.get("seeds") or []), mc.get("target_fpr"), mc.get("n_draws"), mc.get("draw_seed")] != \
                [seeds, target, args.n_draws, P.AXIS_DRAW_SEED[axis]]:
            raise SystemExit(f"REFUSING: {md_rel} was run with other seeds, target, number of draws or draw seed")
        if (mc.get("selection_file"), mc.get("selection_sha256")) != (sel_rel, sel_sha):
            raise SystemExit(f"REFUSING: {md_rel} was run on another selection")
        sel = next((m for m in sel_doc["models"] if m["pair"] == reg_stem), None)
        if sel is None or not sel["selected"]["arm"].startswith("C=") or "mean_diff" not in (sel.get("grid") or {}):
            raise SystemExit(f"REFUSING: no logistic-C selection with a mean_diff arm for {reg_stem} in {sel_rel}")
        C_sel, L_log = float(sel["selected"]["arm"].split("=")[1]), int(sel["selected"]["layer"])
        L_md = meandiff_layer(sel["grid"]["mean_diff"])
        if [mc.get("meandiff_layer"), md["meandiff"].get("layer"), mc.get("logistic_selected_C"),
                mc.get("logistic_selected_layer")] != [L_md, L_md, C_sel, L_log]:
            raise SystemExit(f"REFUSING: {md_rel}'s layers or C are not the ones the selection gives (meandiff_d156's rule)")
        beside_layers = sorted({L_log, L_md})
        beside = md.get("logistic_beside") or {}
        want_log += len(beside)
        if set(beside) != {f"logistic_selected_C_at_layer_{li}" for li in beside_layers}:
            raise SystemExit(f"REFUSING: {md_rel}'s logistic rows are not at L{L_log} and L{L_md}")
        dep_rel = f"results/{D.out_name(stem)}.json"
        df = md.get("depth_file") or {}
        if df.get("file") != dep_rel:
            raise SystemExit(f"REFUSING: {md_rel} names {df.get('file')!r}, not {dep_rel}")
        ddoc = read_listed(dep_rel)
        dep_sha = _RESULT_READS[dep_rel]
        first_read[dep_rel] = dep_sha
        if df.get("sha256") != dep_sha:
            raise SystemExit(f"REFUSING: {dep_rel} is not the bytes {md_rel} names (its depth_file)")
        dc = ddoc["config"]
        if (dc["selected_C"], dc["selected_layer"], dc["selection_sha256"], dc["target_fpr"]) != (C_sel, L_log, sel_sha, target):
            raise SystemExit(f"REFUSING: {dep_rel} was run on another selection or target")
        if (dc.get("registry_pair"), dc.get("selection_of_pair")) != (reg_stem, reg_stem):
            raise SystemExit(f"REFUSING: {dep_rel} names another registry run or selection pair")
        dcells = {(c["layer"], c["probe_set"]): c for c in ddoc["cells"]}

        z: dict[tuple[str, str], Any] = {}
        row_recs: dict[str, dict[str, Any]] = {}
        for side in ("pre", "post"):
            for part in PARTS:
                z[(side, part)], r = REG.load(cfg[f"{side}_repo"], axis, part)
                key = f"{mstem}:{side}/{part}"
                row_recs[f"{side}/{part}"] = r
                input_recs[key] = r
                resolved[key] = resolve_cache(r["file"], REG.CACHE)
                if md["input_cache_sha256"].get(f"{side}/{part}") != r["sha256"] or \
                        ddoc["input_cache_sha256"].get(f"{side}/{part}") != r["sha256"]:
                    raise SystemExit(f"REFUSING: {md_rel} or {dep_rel} read other bytes for {side}/{part}")
        for part in PARTS:                                    # as meandiff_d156: the same items, labelled alike
            z_pre, z_post = z[("pre", part)], z[("post", part)]
            if not np.array_equal(z_pre["item_ids"], z_post["item_ids"]) or not np.array_equal(z_pre["labels"], z_post["labels"]):
                raise SystemExit(f"REFUSING: {pair} {part}: item ids or labels differ pre/post")
        verification = REG.verification(row_recs)
        flag_note = flag_status(verification, md["input_verification"], md_rel)
        depth_flag_note = flag_status(verification, ddoc["input_verification"], dep_rel)     # as table3_tpr_d282
        y_tr = z[("pre", "probe_train")]["labels"].astype(bool)
        ncal = ~z[("pre", "threshold_calib")]["labels"].astype(bool)
        y_ev = z[("pre", "eval")]["labels"].astype(bool)
        masks = {"ncal": ncal, "neg": ~y_ev, "pos": y_ev}
        n_cal, n_ev, n_pos = int(ncal.sum()), int((~y_ev).sum()), int(y_ev.sum())
        pooled = [str(i) for i in z[("pre", "threshold_calib")]["item_ids"][ncal]] + \
                 [str(i) for i in z[("pre", "eval")]["item_ids"][~y_ev]]
        if sha256_bytes("\n".join(pooled).encode("utf-8")) != dc["pooled_negatives"]["item_ids_sha256"] \
                or n_pos != dc["n_eval_positives"]:
            raise SystemExit(f"REFUSING: {pair}'s pooled negatives or eval positives are not {dep_rel}'s")
        pid = sha256_bytes("\n".join(str(i) for i in z[("pre", "eval")]["item_ids"][y_ev]).encode("utf-8"))
        if positives_id.setdefault(axis, pid) != pid:
            raise SystemExit(f"REFUSING: {pair}'s eval positives are not the other {axis} pairs' (the counts would not pair by item)")
        if n_pos not in counts:
            counts[n_pos] = P.counts_matrix(n_pos, args.n_boot, D.ITEM_BOOT_SEED["eval_positives"])
        CNT = counts[n_pos]
        draws = P.repartition_draws(n_cal, n_ev, args.n_draws, P.AXIS_DRAW_SEED[axis])
        pre_key = (cfg["pre_repo"], cfg["pre_rev"], axis)
        first_with_pre = pre_seen.setdefault(pre_key, mstem)

        def shared_check(key: tuple, h: str, where: str) -> int:
            """CHOICES (7): the shared-pre check, and a pair sharing an earlier pair's pre side has every block checked."""
            was = T3.check_shared_pre(shared_pre, key, mstem, h, where)
            if first_with_pre != mstem and not was:
                raise SystemExit(f"REFUSING: {where}: {mstem} shares its pre side with {first_with_pre}, which has no "
                                 f"such block (the selection reads the pre side only)")
            return int(was)

        t1_layers: dict[str, Any] = {}
        if t1_comparable:                                     # CHOICES (8)
            t1row = (t1.get("rows") or {}).get(stem)
            if t1row is None or (t1.get("anchor") or {}).get("depth_records_sha256", {}).get(dep_rel) != dep_sha:
                raise SystemExit(f"REFUSING: {TABLE1_RECORD} holds no row for {stem} on the depth record bytes {md_rel} names")
            t1sel = (t1row.get("probe_sets") or {}).get("selected") or {}
            t1_layers = t1sel.get("layers") or {}
            if t1row.get("headline_layer") != L_log or t1sel.get("probe_C") != C_sel or str(L_log) not in t1_layers:
                raise SystemExit(f"REFUSING: {TABLE1_RECORD}'s {stem} row holds no selected-C block at its headline L{L_log}")
        out_rows[mstem] = {"pair": pair, "registry_pair": reg_stem, "axis": axis, "meandiff_record": md_rel,
                           "depth_record": dep_rel, "meandiff_layer": L_md, "logistic_selected_layer": L_log,
                           "logistic_selected_C": C_sel, "n_positives": n_pos, "eval_positives_item_ids_sha256": pid,
                           "draws_sha256": sha256_bytes(np.ascontiguousarray(draws).astype("<i8").tobytes()),
                           **{f"{side}_{kk}": cfg[f"{side}_{kk}"] for side in ("pre", "post") for kk in ("repo", "rev")},
                           "shares_pre_side_with": None if first_with_pre == mstem else first_with_pre,
                           "input_verification": verification,
                           "input_verification_meandiff_record": md["input_verification"], "d170_flag": flag_note,
                           "input_verification_depth_record": ddoc["input_verification"],
                           "d170_flag_depth_record": depth_flag_note,
                           "paired_py_sha256": {"meandiff_record": ((md.get("code") or {}).get("sha256") or {}).get("src/paired.py"),
                                                "this_run": (code0.get("sha256") or {}).get("src/paired.py"),
                                                "note": "repartition_draws's code (CHOICES (4)); recorded, not a refusal"},
                           "input_cache_source": {k: {"file": r["file"], "expected_from": r["expected_from"],
                                                      "pin_only": r["pin_only"],
                                                      "resolved_path": path_text(resolved[f"{mstem}:{k}"])}
                                                  for k, r in row_recs.items()},
                           "logistic_beside": {}}
        # the logistic rows first: the block at L_md pins the draw order of the array the mean_diff block uses (CHOICES (4))
        for li in beside_layers:
            k = f"layer_{li}"
            X = {(s, p): z[(s, p)][k] for s in ("pre", "post") for p in PARTS}
            where = f"{mstem} logistic C={C_sel:g} L{li}"
            sc = T3.probe_scores(X, y_tr, masks, seeds, C_sel, "selected")
            report, _, reps = T3.anchored_replicates(sc, draws, target, dcells.get((li, "selected")), CNT, where)
            if T1.canon(report) != T1.canon(beside[f"logistic_selected_C_at_layer_{li}"]["operating_point"][D.LEVELS[0]]):
                raise SystemExit(f"REFUSING: {where}: the operating point is not {md_rel}'s logistic_beside")
            n_log += 1
            n_ones += 1
            blk = tpr_block(dcells[(li, "selected")]["operating_point"], reps)
            n_shared += shared_check((cfg["pre_repo"], cfg["pre_rev"], axis, C_sel, li, "selected"),
                                     blk["tpr_pre"]["replicates_sha256"], where)
            t1_note = "not compared: this run's parameters are not the Table 1 record's (a test run)"
            if t1_comparable:
                t1_note = f"not in {TABLE1_RECORD}"
                if str(li) in t1_layers:
                    if any(t1_layers[str(li)][q]["replicates_sha256"] != blk[q]["replicates_sha256"] for q in QUANT):
                        raise SystemExit(f"REFUSING: {where}: the replicates are not {TABLE1_RECORD}'s for this block")
                    n_t1 += 1
                    t1_note = f"replicate hashes identical to {TABLE1_RECORD}'s"
            out_rows[mstem]["logistic_beside"][str(li)] = {
                "layer": li, "probe_C": C_sel, "is_logistic_selected_layer": li == L_log, "is_meandiff_layer": li == L_md,
                **blk, "table1_cross_check": t1_note}
            print(f"  {mstem:<44} logistic L{li:<3} delta TPR frozen {blk['delta_tpr']['frozen']:+.4f}; D-156 band "
                  f"[{blk['delta_tpr']['band95_D156'][0]:+.4f}, {blk['delta_tpr']['band95_D156'][1]:+.4f}]; combined "
                  f"[{blk['delta_tpr']['combined']['ci95'][0]:+.4f}, {blk['delta_tpr']['combined']['ci95'][1]:+.4f}]",
                  flush=True)
        X = {(s, p): z[(s, p)][f"layer_{L_md}"] for s in ("pre", "post") for p in PARTS}
        where = f"{mstem} mean_diff L{L_md}"
        sc, au_fit, au_ho = mean_diff_scores(X, y_tr, masks, seeds)
        chk = md.get("selection_check") or {}
        if [chk.get("layer"), chk.get("auroc_fit_here"), chk.get("auroc_ho20_here")] != [L_md, au_fit, au_ho]:
            raise SystemExit(f"REFUSING: {where}: the mean_diff fits do not give {md_rel}'s selection_check AUROCs")
        op = md["meandiff"]["operating_point"].get(D.LEVELS[0])
        report, arr, reps = record_anchored(sc, draws, target, op, CNT, where)
        n_md += 1
        n_ones += 1
        blk = tpr_block(op, reps)
        n_shared += shared_check((cfg["pre_repo"], cfg["pre_rev"], axis, "mean_diff", L_md),
                                 blk["tpr_pre"]["replicates_sha256"], where)
        out_rows[mstem]["mean_diff"] = {"layer": L_md, "probe": "mean_diff (meandiff_d156.meandiff_probe)",
                                        "selection_check_auroc_identical": True,
                                        "per_draw_sha256": per_draw_hashes(report, arr), **blk}
        print(f"  {mstem:<44} mean_diff L{L_md:<2} delta TPR frozen {blk['delta_tpr']['frozen']:+.4f}; D-156 band "
              f"[{blk['delta_tpr']['band95_D156'][0]:+.4f}, {blk['delta_tpr']['band95_D156'][1]:+.4f}]; combined "
              f"[{blk['delta_tpr']['combined']['ci95'][0]:+.4f}, {blk['delta_tpr']['combined']['ci95'][1]:+.4f}]",
              flush=True)

    n_stored_md = sum("mean_diff" in r for r in out_rows.values())
    n_stored_log = sum(len(r["logistic_beside"]) for r in out_rows.values())
    if not (n_md == n_stored_md == len(run_pairs) and n_log == n_stored_log == want_log and n_ones == n_md + n_log):
        raise SystemExit(f"REFUSING to write: {n_md} mean_diff and {n_log} logistic blocks anchored ({n_stored_md}, "
                         f"{n_stored_log} stored), {n_ones} ones-checked; the records quote {len(run_pairs)} and {want_log}")
    if t1_comparable and n_t1 < len(run_pairs):
        raise SystemExit(f"REFUSING to write: {n_t1} blocks cross-checked against {TABLE1_RECORD}, fewer than the pairs")
    input_sha = {k: r["sha256"] for k, r in input_recs.items()}
    # CHOICES (10): the registry read the final D-168 record (input_registry.verification) for every row with pin-only
    # inputs; it is listed with its sha256, which must be the pinned one
    d168_read: dict[str, str] = {}
    if REG.D168_FINAL_SHA256 is not None and any(r["input_verification"]["pin_only_inputs"] for r in out_rows.values()):
        d168_read = {f"results/{REG.D168_FINAL.name}": sha256_file(REG.D168_FINAL)}
        if list(d168_read.values()) != [REG.D168_FINAL_SHA256]:
            raise SystemExit(f"REFUSING to write: {REG.D168_FINAL.name} is not the pinned final D-168 record")
    idc, id_sources = identity_config(list(input_sha))      # re-reads the meandiff records: a change refuses below
    read_now = {rel: h for rel, h in _RESULT_READS.items()}
    changed = [n for n, a, b in (("code", P.code_block(CODE_FILES)["sha256"], code0["sha256"]),
                                 ("registry sources", D.sources_now(), reg_sources)) if a != b]
    changed += [rel for rel, h in first_read.items() if read_now.get(rel) != h]
    changed += [rel for rel, h in read_now.items() if sha256_file(ROOT / rel) != h]
    changed += [rel_path(p) for p, h in man_sha.items() if sha256_file(p) != h]
    caches = {r["file"]: (resolved[k], r["sha256"]) for k, r in input_recs.items()}
    changed += [f"cache {f}" for f, (p, h) in caches.items() if resolve_cache(f, REG.CACHE) != p or sha256_file(p) != h]
    if changed:
        raise SystemExit(f"REFUSING to write: changed during the run: {changed}")
    config = {"target_fpr": target, "seeds": seeds, "n_draws": args.n_draws, "n_boot": args.n_boot, "pairs": run_pairs,
              "level": "item_level_D156 (the cluster level keeps its D-156 band only, D-282 (2))",
              "blocks": "per pair the mean_diff probe at its layer L_md (meandiff_d156's TPR rows) and the logistic "
                        "selected-C probe at the meandiff record's logistic_beside layers (L* and L_md)",
              "probes": {"mean_diff": "meandiff_d156.meandiff_probe on the D-117 80% fit split per seed (unchanged)",
                         "logistic": "table3_tpr_d282.probe_scores, probe set 'selected' (the D-117 selected C, "
                                     "permutation order), as depth_d156"},
              "cache_dirs": {"GUARDRAIL_CACHE_DIRS": cache_dirs, "registry_cache": path_text(REG.CACHE)},
              "draw_seeds": P.AXIS_DRAW_SEED, "draw_of_replicate": "d_r = 1 + ((r - 1) mod n_draws)",
              "draw_order": "one array per pair and axis from the identical repartition_draws call in meandiff_d156 and "
                            "depth_d156; linked by the depth cell's per_draw_ratio_to_target_sha256, checked on every "
                            "logistic block on the array the mean_diff block uses",
              "counts": "paired.counts_matrix(n positives, n_boot, depth_d156.ITEM_BOOT_SEED['eval_positives']) "
                        "(depth_d156's CNT_P, a function of n positives); every pair on an axis has the same eval "
                        "positives in the same order, so the matrix pairs by item across pairs",
              "counts_seed": D.ITEM_BOOT_SEED["eval_positives"],
              "eval_positives_item_ids_sha256_per_axis": positives_id,
              "replicates_sha256_of": "float64 little-endian bytes, r = 1..n_boot in order",
              "selection_file": sel_rel, "selection_sha256": sel_sha,
              "table1_record": {"file": TABLE1_RECORD, "sha256": t1_sha, "compared": t1_comparable},
              "manifests": [{"file": rel_path(p), "sha256": h} for p, h in man_sha.items()],
              "input_registry_sources_sha256": reg_sources,
              "identity_sources_sha256": id_sources, **idc}
    payload = {**provenance_block("meandiff_tpr_d282"),
               "preregistration": "D-281 (1), D-282 (3) (the other TPR columns when their tables are rebuilt); D-287 "
                                  "(the supplement); D-220, D-246, D-251 (the mean_diff check, SECONDARY); D-299 (the "
                                  "_r2 records)",
               "code": code0, "registered_parameters": registered, "blas_threads": threads, "secondary": SECONDARY,
               "note": NOTE, "implementation_choices": list(CHOICES), "config": config,
               "config_sha256": canonical_hash(config),
               "input_files_sha256": {**read_now, **{rel_path(p): h for p, h in man_sha.items()}, **d168_read},
               "anchor": {"meandiff_records_sha256": {r["meandiff_record"]: first_read[r["meandiff_record"]]
                                                      for r in out_rows.values()},
                          "depth_records_sha256": {r["depth_record"]: first_read[r["depth_record"]] for r in out_rows.values()},
                          "n_mean_diff_blocks_identical": n_md, "n_logistic_blocks_identical": n_log,
                          "kind": "the mean_diff block's item-level operating point equals the meandiff record's as "
                                  "canonical JSON text, its fits give the record's selection_check AUROCs exactly and its "
                                  "layer follows the record's rule; each logistic block's equals the depth record's cell "
                                  "(the bytes the meandiff record names) and the meandiff record's logistic_beside, and "
                                  "its per-draw ratio_to_target array hashes to the cell's per_draw_ratio_to_target_sha256, "
                                  "which pins the draw order of the array both blocks use"},
               "ones_check": {"n_blocks": n_ones, "kind": "counts of all ones give back the D-156 per-draw TPR arrays bit for bit"},
               "shared_pre_check": {"n_blocks": n_shared, "kind": "blocks sharing the pre model and revision, axis, probe, "
                                                                  "C and layer give identical TPR-pre replicates"},
               "table1_cross_check": {"file": TABLE1_RECORD, "sha256": t1_sha, "compared": t1_comparable,
                                      "n_blocks_identical": n_t1,
                                      "kind": "each logistic block Table 1's record holds on the same depth record bytes "
                                              "has its replicate hashes"},
               "population": {"declared_in": "src/populations.py", "group": "MEANDIFF_D156_D282",
                              "role": "secondary (D-220, D-251)", "replaces": None},
               "input_verification": {**REG.verification(input_recs),
                                      "n_rows_flagged": sum(r["input_verification"]["flag"] is not None for r in out_rows.values())},
               "input_cache_sha256": input_sha, "rows": out_rows}
    digest = write_results(out_path, payload, exclusive=True)
    print(f"Written: {rel_path(out_path)}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
