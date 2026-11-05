"""D-337: the cluster-grouped ho20 sensitivity of the Phase 2 D-117 selections. For every Phase 2 selection record
(results/phase2/selection_p2__<org>__<model>@<rev12>__<pooling>.json) the D-117 sweep of src/selection_p2.py runs
again on the same caches, seeds, C grid, arms, layers, holdout fraction and target FPR, with ONE change: per seed, the
ho20 part of probe_train is whole near-duplicate clusters (the D-164 clustering over the pool union, recorded in each
split row's 'cluster' field) instead of an item-level permutation. A SENSITIVITY ROW (D-337 (4)): the recorded
selections stay the selections of record whatever this gives (frozen splits kept, D-318 (7) item 1).

THE RULE (D-337 (2); its text is RULE below and is written into every record): rng = np.random.default_rng(seed); the
unique cluster ids in sorted order are permuted by rng; clusters are taken in that order into ho20 while the ho20 item
count stays at most int(holdout * n); a cluster that would exceed it is skipped and the next tried, until no cluster
fits (one pass over the order is exactly that: the ho20 count only grows, so a cluster skipped once never fits later).
The realised ho20 size and label counts are recorded per seed; a seed whose ho20 lacks either label refuses.

CODE PATH: sweep_split() is selection_p2.sweep copied with only the per-seed split taken from a function of the seed;
item_split() is the permutation written inline in selection_p2.sweep. So sweep_split(..., item_split) gives exactly
selection_p2.sweep's output (tests/test_selection_ho20_cluster.py checks it on synthetic data; the anchor checks it on
a real record), and grouped_split() is the only difference between the anchor and the grouped run. selection_p2.py is
imported and never changed (its code hash is bound by its anchor record): its thread check, its loaders
(select_record, ingest_index, load_ingested: data/hidden only, mode 444, in the ingest index, hashed, item for item
the frozen split, the split file hashed against the extraction record; the D-307 (1) durable copies), composition()
and compare() are used as they are, with regularisation_sweep's functions behind them.

Every mode but --summary refuses unless OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS are all "6"
(selection_p2.check_threads). Run through scripts/project_env.sh. CPU only.

INPUTS of a record (both computing modes): the one selection record for --repo and --pooling; its extraction record
(the model's select record, which must be the file and sha256 the selection record names, say ingest: yes and name the
same repo and revision), the ingest index (file and sha256 as the selection record names), and the probe_train and
calibration caches loaded by selection_p2.load_ingested, each refused unless its sha256 equals the selection record's
input_cache_sha256 and its split file and sha256 equal the record's split_files. probe_train's items are joined to the
split rows by item_id (every item matched; each row's 'cluster' a non-empty string); the composition recomputed from
the rows must equal the record's probe_train_composition. The record's libraries, BLAS threads, host facts and the
sha256 of src/selection_p2.py, src/regularisation_sweep.py and src/paired.py (the code that computed its numbers) must
equal this run's, so the anchor's bit-for-bit result on one record carries to every record.

ANCHOR MODE (--anchor --repo R --pooling P), D-337 (5): the grouping switched off (item_split) on that record's
inputs; selection_p2.compare (selected arm, layer and values, ho20 TPR and its range, n, holdout, every grid value in
the record's arm and layer order) must find no difference AND the whole model block (runner-up and per-seed TPRs too)
must equal the record's after the JSON round trip. Writes results/phase2/ho20cluster_anchor__<stem>.json only then;
otherwise prints every difference and writes nothing.

GROUPED MODE (--repo R --pooling P): refuses unless an anchor record was written by this exact code (every CODE_FILES
sha256), with the same libraries, host facts and BLAS threads, equal bit for bit, on a selection record that still
holds the bytes the anchor names. Writes
results/phase2/ho20cluster__<org>__<model>@<rev12>__<pooling>.json (exclusive, never overwritten) with the D-337 (3)
fields: the selected cell under grouping beside the recorded one; ho20 AUROC of both cells grouped and as recorded;
the leakage gap (recorded item-split ho20 AUROC minus grouped) of every (arm, layer) cell, its maximum and the cell;
the grouped selection's ho20 TPR at the target FPR (calibration negatives, reporting only, D-117); per seed the
realised ho20 size and label counts; the probe_train composition; inputs with sha256 (D-297 (1)); code; config; RULE.

SUMMARY MODE (--summary): reads only the per-record outputs results/phase2/ho20cluster__*.json (each listed with its
sha256); refuses unless every selection record on disk has exactly one output naming it with its current sha256 (the
selection records are hashed for that, not read), all outputs and the summary itself come from one code block, and all
outputs name one anchor still on disk with the sha256 they give (hashed, not read). Writes
results/phase2/ho20cluster_summary.json: in how many of the records the selected cell is unchanged, and per record
both cells, the margins, the recorded runner-up and the largest leakage gap.

READING THE GAPS (RESAMPLING below, written into the leakage-gap definition and the summary): the grouped and the item
split are independent draws per seed, so a gap or a cell change holds resampling variation as well as leakage; the
recorded runner-up margin is the scale a cell change is read against, and the maximum over cells is biased upward.
No null comparison is made here (that would need a new D-entry).

IMPLEMENTATION CHOICES that D-337 leaves open are written into every record (CHOICES below).

SST-2 (--axis p2sst; D-365 (4)(vi): D-337 covers every Phase 2 selection record, so the SST-2 control's too). The same
three modes, RULE, sweep_split, grouped_split, record equalities and comparisons, on the SST-2 selection records
results/phase2/selection_p2sst__<org>__<model>@<rev12>__<pooling>.json (src/selection_p2sst.py: run_kind
selection_p2sst, config.axis p2sst). Only where a record and its inputs are found differs (AXES below): the model's
p2sst select record (selection_p2sst.select_record), its caches through selection_p2sst.load_ingested (selection_p2's
checks plus every split row naming axis p2sst, D-340 (3)), the input keys pre/p2sst/<part>/<pooling>, and the names
  results/phase2/ho20cluster_p2sst_anchor__<stem>.json   --axis p2sst --anchor --repo R --pooling P
  results/phase2/ho20cluster_p2sst__<stem>.json          --axis p2sst --repo R --pooling P
  results/phase2/ho20cluster_p2sst_summary.json          --axis p2sst --summary (every selection_p2sst__*.json)
No p2wg glob (selection_p2__*, ho20cluster__*, ho20cluster_anchor__*) meets these names, so the WildGuardMix outputs,
their anchor and results/phase2/ho20cluster_summary.json (D-354) are neither read nor rewritten. An SST-2 grouped run
requires an SST-2 anchor written by this code; an anchor on a WildGuardMix record is not used for SST-2, nor the
reverse.
Without --axis (default p2wg) every name, check, message and payload is the one the WildGuardMix runs used; the code
block names one more file (src/selection_p2sst.py, imported for its loaders). Only the SST-2 records carry a
config_hash (provenance.canonical_hash of their config). tests/test_selection_ho20_cluster.py
(unchanged) and tests/test_selection_ho20_cluster_p2sst.py show it: the earlier tests pass; the AST of every function
and constant the extension did not touch equals e5aeda6's (the module that wrote the 20 WildGuardMix records, sha256
bdb6ea6b...); e5aeda6's module and this one write equal anchor, grouped and summary records on one WildGuardMix fixture
(the code block and utc aside) and refuse in the same words.

CHOICES (D-181) for SST-2, each by the closest D-entry (CHOICES_P2SST below, written into every SST-2 record beside
CHOICES):
  (a) inputs: selection_p2sst's own select_record and load_ingested, the functions that read the SST-2 record's caches
      (D-337 (2): 'the same caches it read'; D-340 (3): read by axis and split, never by a bare item_id).
  (b) clusters: the p2sst split rows' 'cluster' field, D-337 (2)'s source; the rows carry the clusters of
      manifests/phase2_pool_clusters_p2sst.json (D-340 (1)) and the split file is hashed against the extraction
      record, so the cluster manifest itself is not read. These clusters are not phrase-disjoint on SST-2 (D-338 (3),
      D-341: sub-phrases of one sentence fall in different clusters), so on SST-2 the item-split minus grouped gap
      understates phrase-level leakage (written into each SST-2 record).
  (c) mixed-label clusters (in probe_train only, D-340 (2)) are taken whole like any other cluster (D-337 rejects a
      label-stratified grouping); a seed whose ho20 or fit part lacks a label refuses as before.
  (d) labels: label 1 (positive sentiment) the positive class, the reported ho20 TPR's threshold on the label-0
      calibration items, as the selection record (D-338 (2)).
  (e) the anchor: a separate mode on the SST-2 usermean record (D-342 (2)'s rule; both SST-2 records hold the same
      layers, so 'fewest layers' does not choose; user-span mean is extract_p2's primary pooling, D-338 (4)); its own
      name prefix and run kind, so neither axis's anchor serves the other.
  (f) the summary: a separate SST-2 summary over every selection_p2sst__*.json on disk (D-337 (3) per axis), with the
      WildGuardMix summary's coverage, one-code-block and one-anchor rules; ho20cluster_summary.json is not rerun.
  (g) record equalities (CHOICES record_equalities) unchanged: src/selection_p2.py, src/regularisation_sweep.py and
      src/paired.py computed the SST-2 records' numbers too (selection_p2sst computes nothing: its own anchor rule).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import (  # noqa: E402
    ensure_utf8_stdout,
    provenance_block,
    read_result,
    sha256_file,
    write_results,
)
import paired as P  # noqa: E402
import provenance as PV  # noqa: E402
import regularisation_sweep as RS  # noqa: E402  (the Phase 1 selection's functions, imported, not copied)
import selection_p2 as S  # noqa: E402  (imported, never changed: its code hash is bound by its anchor record)
import selection_p2sst as SS  # noqa: E402  (the SST-2 records' loaders, D-365 (4)(vi); imported, not changed)

CODE_FILES = ["src/selection_ho20_cluster.py", "src/selection_p2sst.py", *S.CODE_FILES]
COMPUTE_FILES = ("src/selection_p2.py", "src/regularisation_sweep.py", "src/paired.py")
OUT_PREFIX = "ho20cluster__"
ANCHOR_PREFIX = "ho20cluster_anchor__"
SUMMARY_NAME = "ho20cluster_summary.json"
RUN_KIND = "selection_ho20_cluster"

RULE = ("D-337 (2): per seed, the ho20 part is whole clusters. The cluster ids of probe_train's items are read from "
        "the split file rows (the 'cluster' field), joined by item id, every item matched. Rule: rng = "
        "np.random.default_rng(seed); the unique cluster ids in sorted order are permuted by rng; clusters are taken "
        "in that order into ho20 while the ho20 item count stays at most int(holdout * n); a cluster that would "
        "exceed it is skipped and the next tried, until no cluster fits. The realised ho20 size and its label counts "
        "are recorded per seed; a seed whose ho20 holds no item of either label refuses.")

RESAMPLING = ("the two splits are independent draws per seed; gaps and cell changes include resampling variation as "
              "well as leakage; the recorded runner-up margin is the scale to compare a cell change against; the "
              "maximum over cells is biased upward")
LEAKAGE_GAP_DEFINITION = ("the record's item-split ho20 AUROC minus the grouped ho20 AUROC, each the mean over seeds, "
                          f"per (arm, layer) cell; {RESAMPLING}")

CHOICES = {
    "sorted_order": "Python's sorted() over the cluster id strings (code-point order); the permutation is "
                    "rng.permutation(number of clusters) applied to that sorted list",
    "item_order_within_parts": "ho20 = the taken clusters in the order taken, fit = the clusters not taken in the "
                               "permuted order; within a cluster, ascending probe_train position. With every item its "
                               "own cluster, numbered in item order, this is selection_p2.sweep's item split exactly "
                               "(order included); row order can move lbfgs fits and float sums in the last bits",
    "fit_part_labels": "a seed whose fit part lacks either label also refuses (D-337 names ho20 only; the logistic "
                       "fit cannot run on one class)",
    "leakage_gap_max_tie_rule": "the first maximum in the record's arm then layer order (the selection's own tie rule)",
    "margin": "margin_under_grouping = grouped ho20 AUROC of the grouped cell minus that of the recorded cell; "
              "margin_under_item_split = recorded ho20 AUROC of the recorded cell minus that of the grouped cell; "
              "both 0 when the cell is unchanged",
    "record_equalities": "a record is used only if its numpy, scipy and scikit-learn versions, BLAS threads, host facts "
                         "(python, platform, machine, cpu_count) and the sha256 of src/selection_p2.py, "
                         "src/regularisation_sweep.py and src/paired.py equal this run's, its composition recomputed "
                         "from the split rows equals its own, and its extraction record and ingest index are the "
                         "files and sha256 it names",
    "anchor_equality": "selection_p2.compare finds no difference and the whole model block (runner-up and per-seed "
                       "TPRs included) equals the record's after the JSON round trip",
    "summary_coverage": "the summary refuses unless every selection record on disk has exactly one output naming it "
                        "with its current sha256, all outputs and the summary share one code block, and all outputs "
                        "name one anchor that is on disk with the sha256 they give",
    "anchor_record_bytes": "a grouped run uses an anchor only if the selection record the anchor was run on still "
                           "hashes to the sha256 the anchor names",
}

CHOICES_P2SST = {      # D-365 (4)(vi), by the closest D-entry (D-181); module docstring, CHOICES (D-181) (a)-(g)
    "p2sst_inputs": "the SST-2 selection record's caches through selection_p2sst.select_record and "
                    "selection_p2sst.load_ingested (selection_p2's checks plus every split row naming axis p2sst), "
                    "keys pre/p2sst/<part>/<pooling> (D-337 (2): the caches the record read; D-340 (3): by axis and "
                    "split)",
    "p2sst_clusters": "probe_train's cluster ids from the p2sst split rows' 'cluster' field (D-337 (2)); the rows "
                      "carry the clusters of manifests/phase2_pool_clusters_p2sst.json and the split file is hashed "
                      "against the extraction record, so the cluster manifest is not read",
    "p2sst_clusters_not_phrase_disjoint": "the D-164 clusters are not phrase-disjoint on SST-2 (D-338 (3), D-341: "
                                          "sub-phrases of one sentence fall in different clusters and splits), so a "
                                          "grouped ho20 can still hold sub-phrases of fit items: on SST-2 the "
                                          "item-split minus grouped gap understates phrase-level leakage",
    "p2sst_mixed_label_clusters": "taken whole like any other cluster (D-337 rejects label-stratified grouping)",
    "p2sst_labels": "label 1 (positive sentiment) the positive class; the reported ho20 TPR's threshold on the label-0 "
                    "calibration items, as the selection record (D-338 (2))",
    "p2sst_anchor": "a separate mode on the SST-2 usermean record (D-342 (2); both SST-2 records hold the same layers; "
                    "user-span mean is extract_p2's primary pooling, D-338 (4)); its own name prefix and run kind, so "
                    "an anchor on a WildGuardMix record is not used for SST-2, nor the reverse",
    "p2sst_summary": "a separate summary over every selection_p2sst__*.json on disk with the same coverage, "
                     "one-code-block and one-anchor rules; results/phase2/ho20cluster_summary.json is neither read nor "
                     "rewritten",
    "p2sst_record_equalities": "record_equalities unchanged: selection_p2.py, regularisation_sweep.py and paired.py "
                               "computed the SST-2 records' numbers too (selection_p2sst computes nothing)",
}

# Where an axis's selection records and their inputs are, and what this module writes for it. 'p2wg' holds exactly the
# names, run kinds and texts the WildGuardMix runs used (D-342); 'p2sst' is D-365 (4)(vi)'s extension. The loaders are
# looked up when called, as the module's own calls 'S.select_record(...)' always were.
AXES: dict[str, dict[str, Any]] = {
    "p2wg": {
        "axis": S.AXIS, "selection_prefix": "selection_p2__", "selection_run_kind": "selection_p2",
        "select_record": lambda repo: S.select_record(repo),
        "load_ingested": lambda *a: S.load_ingested(*a),
        "anchor_prefix": ANCHOR_PREFIX, "out_prefix": OUT_PREFIX, "summary_name": SUMMARY_NAME, "run_kind": RUN_KIND,
        "cli": "", "choices": CHOICES, "config_hash": False,
        "anchor_preregistration": "D-337 (5): with the grouping switched off (selection_p2.sweep's item split) this "
                                  "module reproduces one selection record bit for bit before any grouped record is "
                                  "written (the D-280 standard)",
        "preregistration": "D-337 (the cluster-grouped ho20 sensitivity of the Phase 2 D-117 selections, a sensitivity "
                           "row: the recorded selections stay the selections of record, D-318 (7) item 1); D-117 (the "
                           "selection protocol otherwise unchanged); D-164 (cluster-level units)",
        "summary_preregistration": "D-337 (3) summary: in how many of the records the selected cell is unchanged; "
                                   "D-337 (4) a record whose grouped selection differs is reported with both cells and "
                                   "the margin",
    },
    "p2sst": {
        "axis": SS.AXIS, "selection_prefix": "selection_p2sst__", "selection_run_kind": SS.RUN_KIND,
        "select_record": lambda repo: SS.select_record(repo),
        "load_ingested": lambda *a: SS.load_ingested(*a),
        "anchor_prefix": "ho20cluster_p2sst_anchor__", "out_prefix": "ho20cluster_p2sst__",
        "summary_name": "ho20cluster_p2sst_summary.json", "run_kind": f"{RUN_KIND}_p2sst",
        "cli": " --axis p2sst", "choices": {**CHOICES, **CHOICES_P2SST}, "config_hash": True,
        "anchor_preregistration": "D-337 (5) for the SST-2 selection records (D-365 (4)(vi)): with the grouping "
                                  "switched off (selection_p2.sweep's item split) this module reproduces one SST-2 "
                                  "selection record bit for bit before any grouped SST-2 record is written (the D-280 "
                                  "standard)",
        "preregistration": "D-337 applied to the SST-2 selection records by D-365 (4)(vi) (the cluster-grouped ho20 "
                           "sensitivity, a sensitivity row: the recorded selections stay the selections of record, "
                           "D-318 (7) item 1); D-338 (2), (4) (the SST-2 control's selection, label 1 positive); D-117 "
                           "(the selection protocol otherwise unchanged); D-164 (cluster-level units)",
        "summary_preregistration": "D-337 (3) summary over the SST-2 selection records (D-365 (4)(vi)): in how many "
                                   "of the records the selected cell is unchanged; D-337 (4) a record whose grouped "
                                   "selection differs is reported with both cells and the margin",
    },
}

refuse = S.refuse
rel = S.rel


def _dict(x: Any) -> dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _need(d: Any, path: str, kinds: tuple[type, ...], where: str) -> Any:
    """d's value at the dotted `path`, refused in words unless every step exists and the value is one of `kinds` (a
    bool is accepted only where bool is named)."""
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            refuse(f"{where} has no {path}")
        cur = cur[k]
    if not isinstance(cur, kinds) or (isinstance(cur, bool) and bool not in kinds):
        refuse(f"{where}: {path} is a {type(cur).__name__}, not {'/'.join(x.__name__ for x in kinds)}")
    return cur


# --------------------------------------------------------------------------- the splits

Split = Callable[[int], "tuple[np.ndarray, np.ndarray, dict[str, Any]]"]


def item_split(n: int, holdout: float) -> Split:
    """selection_p2.sweep's split, as written inline there: a seeded permutation of the n items, the first
    int(holdout * n) are ho20."""
    def split(seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_ho = int(holdout * n)
        return perm[:n_ho], perm[n_ho:], {"split": "item", "cap": n_ho}
    return split


def grouped_split(clusters: list[Any], holdout: float) -> Split:
    """D-337 (2): whole clusters into ho20. clusters[i] is item i's cluster id. Per seed: the unique ids, sorted, are
    permuted by np.random.default_rng(seed); one pass in that order takes each cluster whose items still fit within
    int(holdout * n) and skips the rest (a skipped cluster never fits later: the count only grows). ho20 is the taken
    clusters in the order taken, fit the others in the permuted order; within a cluster, ascending item position."""
    n = len(clusters)
    members: dict[Any, list[int]] = {}
    for i, c in enumerate(clusters):
        members.setdefault(c, []).append(i)
    uniq = sorted(members)
    cap = int(holdout * n)

    def split(seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        rng = np.random.default_rng(seed)
        order = [uniq[j] for j in rng.permutation(len(uniq))]
        taken: set[Any] = set()
        ho: list[int] = []
        for c in order:
            if len(ho) + len(members[c]) <= cap:
                taken.add(c)
                ho.extend(members[c])
        fit = [i for c in order if c not in taken for i in members[c]]
        return (np.array(ho, dtype=np.int64), np.array(fit, dtype=np.int64),
                {"split": "grouped", "cap": cap, "n_clusters": len(uniq), "n_clusters_ho20": len(taken)})
    return split


def cluster_ids(item_ids: Any, rows: list[dict[str, Any]]) -> list[str]:
    """probe_train's cluster ids, item for item, joined to the split rows by item_id: every item matched, every row
    used once, every 'cluster' a non-empty string (D-337 (2))."""
    by_id: dict[str, str] = {}
    for r in rows:
        iid, c = r.get("item_id"), r.get("cluster")
        if not isinstance(iid, str) or iid in by_id:
            refuse(f"split row item_id {iid!r} is missing, not a string, or repeated")
        if not isinstance(c, str) or not c:
            refuse(f"split row {iid}: cluster {c!r} is not a non-empty string (D-337 (2) reads the 'cluster' field)")
        by_id[iid] = c
    ids = [str(x) for x in item_ids]
    if len(set(ids)) != len(ids):
        refuse("probe_train's item ids repeat")
    unmatched = [x for x in ids if x not in by_id]
    if unmatched:
        refuse(f"{len(unmatched)} probe_train items match no split row by item_id (first {unmatched[:3]}); every "
               "item must be matched (D-337 (2))")
    if len(by_id) != len(ids):
        refuse(f"{len(by_id) - len(ids)} split rows match no probe_train item")
    return [by_id[x] for x in ids]


# --------------------------------------------------------------------------- the D-117 sweep, split as a function

def sweep_split(z_tr: Any, z_cal: Any, seeds: list[int], holdout: float, target_fpr: float,
                split: Split) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """selection_p2.sweep with ONE change: each seed's (ho20, fit) indices come from split(seed) (computed once per
    seed) instead of the permutation written inline there. Everything else is copied line for line: standardisation
    on the fitting part, every arm (logistic at each RS.C_GRID value, then mean_diff), AUROC on both parts, means over
    seeds, the argmax over (arm, layer) on ho20 keeping the first maximum in arm then layer order, the runner-up, and
    the selected cell's ho20 TPR at the threshold set on z_cal's negatives (reporting only, D-117). Returns
    selection_p2.sweep's dict and, per seed, the realised part sizes and label counts. A seed whose ho20 or fit part
    lacks either label refuses before any fit."""
    y_all = z_tr["labels"].astype(bool)
    n = len(y_all)
    parts: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        ho, fit, info = split(seed)
        if len(ho) + len(fit) != n or not np.array_equal(np.union1d(ho, fit), np.arange(n)):
            refuse(f"seed {seed}: the split does not partition the {n} items")
        yh, yf = y_all[ho], y_all[fit]
        rec = {"seed": seed, **info, "n_ho20": int(len(ho)), "n_ho20_positive": int(yh.sum()),
               "n_ho20_negative": int((~yh).sum()), "n_fit": int(len(fit)), "n_fit_positive": int(yf.sum()),
               "n_fit_negative": int((~yf).sum())}
        if rec["n_ho20_positive"] == 0 or rec["n_ho20_negative"] == 0:
            refuse(f"seed {seed}: ho20 holds {rec['n_ho20_positive']} positive and {rec['n_ho20_negative']} negative "
                   "items; a seed whose ho20 lacks either label refuses (D-337 (2))")
        if rec["n_fit_positive"] == 0 or rec["n_fit_negative"] == 0:
            refuse(f"seed {seed}: the fit part holds {rec['n_fit_positive']} positive and {rec['n_fit_negative']} "
                   "negative items; the probes cannot be fitted")
        parts[seed] = (ho, fit)
        per_seed.append(rec)

    keys = sorted((k for k in z_tr.files if k.startswith("layer_")), key=lambda s: int(s.split("_")[1]))
    grid: dict[str, dict[int, list[tuple[float, float]]]] = {}
    for k in keys:
        li = int(k.split("_")[1])
        X = z_tr[k]
        if X.ndim != 2 or X.shape[0] != n:
            refuse(f"{k} has shape {X.shape}, not (n_items={n}, hidden)")
        for seed in seeds:
            ho, fit = parts[seed]
            xf, yf = X[fit], y_all[fit]
            mu, sd = xf.mean(0), xf.std(0)
            sd = np.where(sd < 1e-8, 1.0, sd)
            xf_s = (xf - mu) / sd
            xh_s = (X[ho] - mu) / sd
            yh = y_all[ho]
            for C in RS.C_GRID:
                w, b = RS.fit_logistic(xf_s, yf, seed, C)
                grid.setdefault(f"C={C:g}", {}).setdefault(li, []).append(
                    (RS.auroc(xf_s @ w + b, yf), RS.auroc(xh_s @ w + b, yh)))
            w, b = RS.fit_meandiff(xf_s, yf)
            grid.setdefault("mean_diff", {}).setdefault(li, []).append(
                (RS.auroc(xf_s @ w + b, yf), RS.auroc(xh_s @ w + b, yh)))

    table: dict[str, dict[int, dict[str, float]]] = {}
    best: dict[str, Any] | None = None
    for arm, per_layer in grid.items():
        table[arm] = {}
        for li, vals in per_layer.items():
            f = float(np.mean([v[0] for v in vals]))
            h = float(np.mean([v[1] for v in vals]))
            table[arm][li] = {"auroc_fit": f, "auroc_ho20": h, "gap": f - h}
            if best is None or h > best["auroc_ho20"]:
                best = {"arm": arm, "layer": li, "auroc_fit": f, "auroc_ho20": h, "gap": f - h}
    if best is None:
        refuse("no layer_* array in the probe_train cache")

    # REPORTING ONLY (D-117): the selected probe's ho20 TPR at the threshold set on the calibration negatives
    k = f"layer_{best['layer']}"
    y_cal = z_cal["labels"].astype(bool)
    tprs = []
    for seed in seeds:
        ho, fit = parts[seed]
        xf = z_tr[k][fit]
        mu, sd = xf.mean(0), xf.std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        xf_s = (xf - mu) / sd
        if best["arm"] == "mean_diff":
            w, b = RS.fit_meandiff(xf_s, y_all[fit])
        else:
            w, b = RS.fit_logistic(xf_s, y_all[fit], seed, float(best["arm"].split("=")[1]))
        t = RS.thr_at_fpr(((z_cal[k] - mu) / sd @ w + b)[~y_cal], target_fpr)
        s_ho = (z_tr[k][ho] - mu) / sd @ w + b
        tprs.append(float((s_ho[y_all[ho]] >= t).mean()))
    best["ho20_tpr_at_1pct"] = float(np.mean(tprs))       # selection_p2's key names, so like keys are compared
    best["ho20_tpr_range"] = [float(np.min(tprs)), float(np.max(tprs))]
    best["ho20_tpr_per_seed"] = tprs
    best["ho20_tpr_target_fpr"] = target_fpr
    others = [(c["auroc_ho20"], a, li) for a, d in table.items() for li, c in d.items()
              if (a, li) != (best["arm"], best["layer"])]
    if others:
        h2, a2, l2 = max(others, key=lambda t: t[0])
        best["runner_up"] = {"arm": a2, "layer": l2, "auroc_ho20": h2, "margin": best["auroc_ho20"] - h2}
    return ({"n_probe_train": n, "holdout_fraction": holdout, "selected": best,
             "grid": {a: {str(li): v for li, v in d.items()} for a, d in table.items()}}, per_seed)


# --------------------------------------------------------------------------- the selection record and its inputs

def axis_spec(axis: str) -> dict[str, Any]:
    """AXES[axis], refused in words for an axis this module does not know."""
    if axis not in AXES:
        refuse(f"axis must be one of {sorted(AXES)}, not {axis!r}")
    return AXES[axis]


def config_entries(X: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """A record's 'config', and on an axis that carries one its 'config_hash' (provenance.canonical_hash): p2sst only,
    so the WildGuardMix records keep the form the 20 records of D-354 have."""
    return {"config": cfg, **({"config_hash": PV.canonical_hash(cfg)} if X["config_hash"] else {})}


def selection_record(repo: str, pooling: str, axis: str = "p2wg") -> tuple[Path, dict[str, Any], str]:
    """The one selection record for repo and pooling (on the axis), read as an input (D-297 (1)); its stem
    '<org>__<model>@<rev12>__<pooling>'."""
    X = axis_spec(axis)
    if pooling not in S.POOLINGS:
        refuse(f"pooling must be one of {S.POOLINGS}")
    org_model = repo.replace("/", "__")
    found = sorted(S.P2.glob(f"{X['selection_prefix']}{org_model}@*__{pooling}.json"))
    if len(found) != 1:
        refuse(f"{len(found)} selection records for {repo} ({pooling}) under {rel(S.P2)} (want exactly one): "
               f"{[p.name for p in found]}")
    path = found[0]
    sel = read_result(path)
    if not isinstance(sel, dict):
        refuse(f"{rel(path)} is not a JSON object")
    cfg, model = _dict(sel.get("config")), _dict(sel.get("model"))
    rev = cfg.get("pre_rev")
    if sel.get("run_kind") != X["selection_run_kind"] or cfg.get("pre_repo") != repo or model.get("repo") != repo \
            or cfg.get("pooling") != pooling or model.get("pooling") != pooling or not isinstance(rev, str) \
            or len(rev) != 40 or model.get("revision") != rev or cfg.get("axis") != X["axis"]:
        refuse(f"{rel(path)} is not a {X['selection_run_kind']} record of {repo} at a full revision, pooling "
               f"{pooling}, axis {X['axis']}")
    stem = f"{org_model}@{rev[:12]}__{pooling}"
    if path.name != f"{X['selection_prefix']}{stem}.json":
        refuse(f"{rel(path)} is not named for its own revision ({stem})")
    if cfg.get("C_grid") != RS.C_GRID:
        refuse(f"{rel(path)}'s C grid {cfg.get('C_grid')} is not regularisation_sweep.C_GRID {RS.C_GRID}")
    if not isinstance(cfg.get("seeds"), list) or not cfg["seeds"] or not isinstance(cfg.get("holdout"), float) \
            or not isinstance(cfg.get("target_fpr"), float):
        refuse(f"{rel(path)} names no seeds, holdout or target FPR")
    if not isinstance(model.get("grid"), dict) or not isinstance(model.get("selected"), dict):
        refuse(f"{rel(path)} holds no selection grid")
    return path, sel, stem


def check_record_env(sel: dict[str, Any], path: Path, code0: dict[str, Any],
                     threads: dict[str, Any]) -> dict[str, Any]:
    """The record's numbers are what this run's code, libraries, BLAS threads and host class give: the anchor proves
    it bit for bit on one record, and these equalities carry that to each record (CHOICES record_equalities)."""
    name = rel(path)
    libs = {k: code0["libraries"].get(k) for k in S.LIBS}
    rec_libs = {k: ((sel.get("code") or {}).get("libraries") or {}).get(k) for k in S.LIBS}
    if rec_libs != libs:
        refuse(f"{name} was computed with {rec_libs}, this run has {libs}")
    if (sel.get("config") or {}).get("blas_threads") != threads:
        refuse(f"{name} names BLAS threads {(sel.get('config') or {}).get('blas_threads')}, this run has {threads}")
    host = {k: PV.env_state().get(k) for k in S.HOST_KEYS}
    rec_host = {k: (sel.get("env") or {}).get(k) for k in S.HOST_KEYS}
    if rec_host != host:
        refuse(f"{name} was computed on host {rec_host}, this run is on {host}")
    rec_code = (sel.get("code") or {}).get("sha256") or {}
    for f in COMPUTE_FILES:
        h = code0["sha256"].get(f)
        if not h or rec_code.get(f) != h:
            refuse(f"{name}'s numbers come from {f} at {str(rec_code.get(f))[:12]}, this run's is {str(h)[:12]}")
    return {"libraries": libs, "blas_threads": threads, "host": host,
            "compute_code_sha256": {f: code0["sha256"][f] for f in COMPUTE_FILES}, "equal_to_the_record": True}


def load_inputs(sel: dict[str, Any], sel_path: Path, repo: str, pooling: str, axis: str = "p2wg") -> dict[str, Any]:
    """The caches the selection record read, through selection_p2's loaders (module docstring, INPUTS); on p2sst
    through selection_p2sst's (CHOICES_P2SST p2sst_inputs)."""
    X = axis_spec(axis)
    name = rel(sel_path)
    rev = sel["config"]["pre_rev"]
    rec_path = X["select_record"](repo)
    rec_rel = rel(rec_path)
    ext = {"file": rec_rel, "sha256": sha256_file(rec_path)}
    if sel.get("extraction_record") != ext:
        refuse(f"the extraction record now is {ext}, not the one {name} names {sel.get('extraction_record')}")
    rec = read_result(rec_path)
    if rec.get("repo") != repo or rec.get("revision") != rev or rec.get("set_name") != "select" \
            or rec.get("ingest") != "yes":
        refuse(f"{rec_rel} names repo {rec.get('repo')!r} at {str(rec.get('revision'))[:12]}, set "
               f"{rec.get('set_name')!r}, ingest {rec.get('ingest')!r}; want {repo} at {rev[:12]}, select, yes")
    index_path, index = S.ingest_index(rec_path)
    idx = {"file": rel(index_path), "sha256": sha256_file(index_path)}
    if sel.get("ingest_index") != idx:
        refuse(f"the ingest index now is {idx}, not the one {name} names {sel.get('ingest_index')}")
    loaded: dict[str, Any] = {}
    for part in ("probe_train", "calibration"):
        key = f"pre/{X['axis']}/{part}/{pooling}"
        z, r, rows = X["load_ingested"](rec, rec_rel, index, part, pooling)
        want = (sel.get("input_cache_sha256") or {}).get(key)
        if r["sha256"] != want:
            refuse(f"{key}: the cache read now ({r['sha256'][:16]}) is not the one {name} read ({str(want)[:16]})")
        if r["split"] != (sel.get("split_files") or {}).get(key):
            refuse(f"{key}: the split file {r['split']} is not the one {name} names "
                   f"{(sel.get('split_files') or {}).get(key)}")
        loaded[part] = (z, r, rows)
    z_tr, r_tr, rows_tr = loaded["probe_train"]
    z_cal, r_cal, _ = loaded["calibration"]
    y_tr, y_cal = z_tr["labels"].astype(bool), z_cal["labels"].astype(bool)
    if y_tr.all() or not y_tr.any():
        refuse("probe_train holds one class only")
    if y_cal.all():
        refuse("calibration holds no negative")
    comp = S.composition(rows_tr)
    if comp != sel.get("probe_train_composition"):
        refuse(f"probe_train's composition from the split rows {comp} is not {name}'s "
               f"{sel.get('probe_train_composition')}")
    clusters = cluster_ids(z_tr["item_ids"], rows_tr)
    inputs = {f"pre/{X['axis']}/probe_train/{pooling}": r_tr, f"pre/{X['axis']}/calibration/{pooling}": r_cal}
    return {"z_tr": z_tr, "z_cal": z_cal, "clusters": clusters, "composition": comp,
            "n_calibration": int(len(y_cal)), "n_calibration_negative": int((~y_cal).sum()),
            "extraction_record": ext, "ingest_index": idx,
            "input_cache_sha256": {k: r["sha256"] for k, r in inputs.items()},
            "input_cache_source": {k: f"{r['file']} (expected bytes from {r['expected_from']})"
                                   for k, r in inputs.items()},
            "split_files": {k: r["split"] for k, r in inputs.items()}}


def common_config(sel: dict[str, Any], threads: dict[str, Any], split: str) -> dict[str, Any]:
    c = sel["config"]
    return {"pre_repo": c["pre_repo"], "pre_rev": c["pre_rev"], "axis": c["axis"], "pooling": c["pooling"],
            "C_grid": RS.C_GRID, "seeds": c["seeds"], "holdout": c["holdout"], "target_fpr": c["target_fpr"],
            "blas_threads": threads, "settings_from": "the selection record's config", "split": split,
            "eval_used_for_selection": False,
            "threshold_population": "the calibration split's negatives, for the reported ho20 TPR only (D-117)",
            "inputs_from": "the caches the selection record read (data/hidden, ingested, indexed), sha256 bound"}


def inputs_block(sel_path: Path, inp: dict[str, Any]) -> dict[str, Any]:
    return {"selection_record": {"file": rel(sel_path), "sha256": sha256_file(sel_path)},
            "extraction_record": inp["extraction_record"], "ingest_index": inp["ingest_index"],
            "input_cache_sha256": inp["input_cache_sha256"], "input_cache_source": inp["input_cache_source"],
            "split_files": inp["split_files"],
            "n_calibration": inp["n_calibration"], "n_calibration_negative": inp["n_calibration_negative"],
            "probe_train_composition": inp["composition"]}


# --------------------------------------------------------------------------- anchor (grouping switched off)

def model_block_diffs(m: dict[str, Any], old: dict[str, Any]) -> list[str]:
    """Keys of the model block (and of its 'selected') that differ from the record's after the JSON round trip."""
    got = json.loads(json.dumps(m))  # not a result (D-297)
    want = {k: v for k, v in old.items() if k not in ("repo", "revision", "pooling")}
    if got == want:
        return []
    out = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))
    gs, ws = _dict(got.get("selected")), _dict(want.get("selected"))
    out += sorted(f"selected.{k}" for k in set(gs) | set(ws) if gs.get(k) != ws.get(k))
    return out or ["<model block>"]                      # e.g. a key present as null on one side, absent on the other


def run_anchor(repo: str, pooling: str, axis: str = "p2wg") -> Path:
    X = axis_spec(axis)
    threads = S.check_threads()
    code0 = P.code_block(CODE_FILES)
    sel_path, sel, stem = selection_record(repo, pooling, axis)
    out = S.P2 / f"{X['anchor_prefix']}{stem}.json"
    if out.exists():
        refuse(f"{rel(out)} exists; results are never overwritten (D-152 P5)")
    checks = check_record_env(sel, sel_path, code0, threads)
    inp = load_inputs(sel, sel_path, repo, pooling, axis)
    c = sel["config"]
    print(f"ANCHOR {stem}: the item split (grouping switched off) against {rel(sel_path)}", flush=True)
    m, per_seed = sweep_split(inp["z_tr"], inp["z_cal"], c["seeds"], c["holdout"], c["target_fpr"],
                              item_split(len(inp["clusters"]), c["holdout"]))
    S.print_summary(f"{pooling} item split", m)
    diffs, nvals = S.compare(m, sel["model"])
    whole = model_block_diffs(m, sel["model"])
    if diffs or whole:
        for d in diffs:
            print(f"  DIFFERENT {d}")
        if whole:
            print(f"  the model block differs from the record's in {whole}")
        raise SystemExit(f"ANCHOR FAILED: {len(diffs)} of {nvals} compared values differ, model block keys differing "
                         f"{whole}; nothing written")
    print(f"  ANCHOR OK: {nvals} of {nvals} values equal and the whole model block equal, bit for bit; selected "
          f"{m['selected']['arm']} at layer {m['selected']['layer']} as recorded", flush=True)
    if P.code_block(CODE_FILES)["sha256"] != code0["sha256"]:
        refuse("source changed during the run")
    digest = write_results(out, {
        **provenance_block(f"{X['run_kind']}_anchor"),
        "preregistration": X["anchor_preregistration"],
        "rule": RULE, "implementation_choices": X["choices"],
        "code": code0,
        **config_entries(X, {**common_config(sel, threads, "item (grouping switched off)"), "equality": "exact"}),
        "record_checks": checks,
        **inputs_block(sel_path, inp),
        "per_seed_ho20": per_seed,
        "anchor": {"file": rel(sel_path), "sha256": sha256_file(sel_path), "values_compared": nvals,
                   "values_differing": 0, "model_block_equal": True, "exact_equal": True,
                   "selected": {k: m["selected"][k] for k in ("arm", "layer")}},
        "model": {"repo": repo, "revision": c["pre_rev"], "pooling": pooling, **m},
    }, exclusive=True)
    print(f"Written: {rel(out)}\nsha256:  {digest}")
    return out


# --------------------------------------------------------------------------- the grouped run

def passing_anchor(code0: dict[str, Any], threads: dict[str, Any],
                   axis: str = "p2wg") -> tuple[dict[str, Any], list[dict[str, str]]]:
    """An anchor record (of the axis) written by this exact code, libraries, host facts and threads, equal bit for bit,
    on a selection record that still holds the bytes the anchor names (re-hashed here); every anchor record read is
    listed."""
    X = axis_spec(axis)
    found = sorted(S.P2.glob(f"{X['anchor_prefix']}*.json"))
    if not found:
        refuse(f"no anchor record in {rel(S.P2)}: run --anchor{X['cli']} --repo R --pooling P with this code first "
               "(D-337 (5))")
    host = {k: PV.env_state().get(k) for k in S.HOST_KEYS}
    libs = {k: code0["libraries"].get(k) for k in S.LIBS}
    read, match, why = [], None, []
    for p in found:
        a = read_result(p)
        read.append({"file": rel(p), "sha256": sha256_file(p)})
        a = _dict(a)
        code, cfg, anc = _dict(a.get("code")), _dict(a.get("config")), _dict(a.get("anchor"))
        rec, rec_sha = anc.get("file"), anc.get("sha256")
        rec_path = S.P2 / Path(rec).name if isinstance(rec, str) and rec else None
        checks = {
            "run kind": a.get("run_kind") == f"{X['run_kind']}_anchor",
            "code": code.get("sha256") == code0["sha256"],
            "libraries": {k: _dict(code.get("libraries")).get(k) for k in S.LIBS} == libs,
            "host facts": {k: _dict(a.get("env")).get(k) for k in S.HOST_KEYS} == host,
            "BLAS threads": cfg.get("blas_threads") == threads,
            "bit-for-bit equality": anc.get("exact_equal") is True and anc.get("values_differing") == 0
            and anc.get("model_block_equal") is True,
            "the anchored selection record's bytes": rec_path is not None and rel(rec_path) == rec
            and rec_path.is_file() and sha256_file(rec_path) == rec_sha,
        }
        failed = [k for k, v in checks.items() if not v]
        if not failed and match is None:
            match = {"file": rel(p), "sha256": sha256_file(p), "anchored_record": {"file": rec, "sha256": rec_sha}}
        why.append(f"{p.name}: {', '.join(failed) if failed else 'passes'}")
    if match is None:
        refuse(f"none of the {len(found)} anchor records was written by this code with these libraries, threads and "
               f"host facts on a selection record still holding the bytes it names ({'; '.join(why)}); run "
               f"--anchor{X['cli']} with this code first")
    return match, read


def compare_cells(m: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """D-337 (3): the grouped selection beside the recorded one, both cells' ho20 AUROC grouped and as recorded, the
    leakage gap of every cell (recorded item-split ho20 AUROC minus grouped), its maximum and cell, and the grouped
    selection's ho20 TPR."""
    gg, rg = m["grid"], model["grid"]
    if any(not isinstance(rg[a], dict) for a in rg) or list(gg) != list(rg) \
            or any(list(gg[a]) != list(rg[a]) for a in rg):
        refuse("the grouped grid's arms or layers are not the record's, in the record's order")
    bad = [f"{a} {li}" for a in rg for li in rg[a] if not isinstance(_dict(rg[a][li]).get("auroc_ho20"), float)]
    if bad:
        refuse(f"the record's grid has no float auroc_ho20 at {bad[:3]}")
    gs, rs = m["selected"], model["selected"]
    if rs.get("arm") not in rg or str(rs.get("layer")) not in rg[rs["arm"]]:
        refuse(f"the record's selected cell {rs.get('arm')!r} {rs.get('layer')!r} is not in its grid")

    def cell(arm: str, layer: int) -> dict[str, Any]:
        return {"arm": arm, "layer": layer, "ho20_auroc_grouped": gg[arm][str(layer)]["auroc_ho20"],
                "ho20_auroc_recorded_item_split": rg[arm][str(layer)]["auroc_ho20"]}

    gc, rc = cell(gs["arm"], gs["layer"]), cell(rs["arm"], rs["layer"])
    per_cell: dict[str, dict[str, float]] = {}
    best: tuple[float, str, int] | None = None
    for arm in rg:
        per_cell[arm] = {}
        for li in rg[arm]:
            gap = rg[arm][li]["auroc_ho20"] - gg[arm][li]["auroc_ho20"]
            per_cell[arm][li] = gap
            if best is None or gap > best[0]:
                best = (gap, arm, int(li))
    if best is None:
        refuse("the record's grid is empty")
    return {
        "selected_recorded": {"arm": rs["arm"], "layer": rs["layer"]},
        "selected_grouped": {"arm": gs["arm"], "layer": gs["layer"]},
        "selected_cell_unchanged": (gs["arm"], gs["layer"]) == (rs["arm"], rs["layer"]),
        "grouped_cell": gc, "recorded_cell": rc,
        "margin_under_grouping": gc["ho20_auroc_grouped"] - rc["ho20_auroc_grouped"],
        "margin_under_item_split": rc["ho20_auroc_recorded_item_split"] - gc["ho20_auroc_recorded_item_split"],
        "recorded_runner_up": rs.get("runner_up"),
        "leakage_gap": {"definition": LEAKAGE_GAP_DEFINITION,
                        "per_cell": per_cell, "max": best[0], "max_cell": {"arm": best[1], "layer": best[2]},
                        "tie_rule": CHOICES["leakage_gap_max_tie_rule"]},
        "grouped_selection_ho20_tpr": {"target_fpr": gs["ho20_tpr_target_fpr"], "mean": gs["ho20_tpr_at_1pct"],
                                       "range": gs["ho20_tpr_range"], "per_seed": gs["ho20_tpr_per_seed"],
                                       "threshold_population": "the calibration split's negatives (reporting only, "
                                                               "D-117)"},
    }


def run_grouped(repo: str, pooling: str, axis: str = "p2wg") -> Path:
    X = axis_spec(axis)
    threads = S.check_threads()
    code0 = P.code_block(CODE_FILES)
    anchor, anchors_read = passing_anchor(code0, threads, axis)
    sel_path, sel, stem = selection_record(repo, pooling, axis)
    out = S.P2 / f"{X['out_prefix']}{stem}.json"
    if out.exists():
        refuse(f"{rel(out)} exists; results are never overwritten (D-152 P5)")
    checks = check_record_env(sel, sel_path, code0, threads)
    inp = load_inputs(sel, sel_path, repo, pooling, axis)
    c = sel["config"]
    print(f"{stem}: the D-117 sweep with cluster-grouped ho20 (D-337) against {rel(sel_path)}; probe_train "
          f"composition {inp['composition']}", flush=True)
    m, per_seed = sweep_split(inp["z_tr"], inp["z_cal"], c["seeds"], c["holdout"], c["target_fpr"],
                              grouped_split(inp["clusters"], c["holdout"]))
    S.print_summary(f"{pooling} grouped", m)
    cmp_ = compare_cells(m, sel["model"])
    print(f"  recorded {cmp_['selected_recorded']}, grouped {cmp_['selected_grouped']}, unchanged "
          f"{cmp_['selected_cell_unchanged']}; per seed ho20 sizes {[r['n_ho20'] for r in per_seed]}", flush=True)
    if P.code_block(CODE_FILES)["sha256"] != code0["sha256"]:
        refuse("source changed during the run")
    digest = write_results(out, {
        **provenance_block(X["run_kind"]),
        "preregistration": X["preregistration"],
        "rule": RULE, "implementation_choices": X["choices"],
        "code": code0,
        **config_entries(X, {**common_config(sel, threads, "grouped (D-337 (2): whole clusters in ho20)"),
                             "selection_population": "held-out part of probe_train per seed, whole near-duplicate "
                                                     "clusters, at most int(holdout * n) items"}),
        "anchor": anchor, "anchors_read": anchors_read,
        "record_checks": checks,
        **inputs_block(sel_path, inp),
        "per_seed_ho20": per_seed,
        "comparison": cmp_,
        "grouped": {"repo": repo, "revision": c["pre_rev"], "pooling": pooling, **m},
    }, exclusive=True)
    print(f"Written: {rel(out)}\nsha256:  {digest}")
    return out


# --------------------------------------------------------------------------- summary

def run_summary(axis: str = "p2wg") -> Path:
    X = axis_spec(axis)
    code0 = P.code_block(CODE_FILES)
    out = S.P2 / X["summary_name"]
    if out.exists():
        refuse(f"{rel(out)} exists; results are never overwritten (D-152 P5)")
    outs = sorted(S.P2.glob(f"{X['out_prefix']}*.json"))
    if not outs:
        refuse(f"no per-record output {X['out_prefix']}*.json in {rel(S.P2)}")
    on_disk = {rel(p): sha256_file(p) for p in sorted(S.P2.glob(f"{X['selection_prefix']}*.json"))}  # hashed, not read
    inputs, rows, named, codes, anchors = [], [], {}, set(), set()
    num = (float, int)
    for p in outs:
        d = read_result(p)
        name = rel(p)
        inputs.append({"file": name, "sha256": sha256_file(p)})
        if not isinstance(d, dict) or d.get("run_kind") != X["run_kind"]:
            refuse(f"{name} is not a {X['run_kind']} record")
        sr_file = _need(d, "selection_record.file", (str,), name)
        sr_sha = _need(d, "selection_record.sha256", (str,), name)
        if on_disk.get(sr_file) != sr_sha:
            refuse(f"{name} names selection record {sr_file} ({sr_sha[:12]}), which is not on disk with those bytes")
        if sr_file in named:
            refuse(f"{named[sr_file]} and {name} both name {sr_file}")
        named[sr_file] = name
        codes.add(json.dumps(_need(d, "code.sha256", (dict,), name), sort_keys=True))
        anchors.add((_need(d, "anchor.file", (str,), name), _need(d, "anchor.sha256", (str,), name)))
        per_seed = _need(d, "per_seed_ho20", (list,), name)
        if not per_seed:
            refuse(f"{name} records no seed in per_seed_ho20")
        sizes = [_need(r, "n_ho20", (int,), f"{name} per_seed_ho20[{i}]") for i, r in enumerate(per_seed)]
        rows.append({"output": name, "selection_record": sr_file,
                     "repo": _need(d, "grouped.repo", (str,), name),
                     "revision": _need(d, "grouped.revision", (str,), name),
                     "pooling": _need(d, "grouped.pooling", (str,), name),
                     "selected_recorded": _need(d, "comparison.selected_recorded", (dict,), name),
                     "selected_grouped": _need(d, "comparison.selected_grouped", (dict,), name),
                     "selected_cell_unchanged": _need(d, "comparison.selected_cell_unchanged", (bool,), name),
                     "margin_under_grouping": _need(d, "comparison.margin_under_grouping", num, name),
                     "margin_under_item_split": _need(d, "comparison.margin_under_item_split", num, name),
                     "recorded_runner_up": _need(d, "comparison.recorded_runner_up", (dict,), name),
                     "leakage_gap_max": _need(d, "comparison.leakage_gap.max", num, name),
                     "leakage_gap_max_cell": _need(d, "comparison.leakage_gap.max_cell", (dict,), name),
                     "grouped_selection_ho20_tpr_mean": _need(d, "comparison.grouped_selection_ho20_tpr.mean", num,
                                                              name),
                     "realised_ho20_size_range": [min(sizes), max(sizes)]})
    missing = sorted(set(on_disk) - set(named))
    if missing:
        refuse(f"{len(missing)} selection records have no per-record output: {missing}")
    if len(codes) != 1:
        refuse(f"the {len(outs)} outputs come from {len(codes)} code blocks; a summary is over one code block")
    if json.dumps(code0["sha256"], sort_keys=True) not in codes:
        refuse("the outputs' code block is not this run's: the outputs and the summary are one code block")
    if len(anchors) != 1:
        refuse(f"the {len(outs)} outputs name {len(anchors)} anchors {sorted(anchors)}; a summary is over one anchor")
    (anchor_file, anchor_sha), = anchors
    apath = S.P2 / Path(anchor_file).name
    if rel(apath) != anchor_file or not apath.is_file() or sha256_file(apath) != anchor_sha:
        refuse(f"the anchor the outputs name, {anchor_file} ({anchor_sha[:12]}), is not on disk with those bytes")
    n_unch = sum(bool(r["selected_cell_unchanged"]) for r in rows)
    print(f"ho20 cluster sensitivity over {len(rows)} selection records: the selected cell is unchanged in {n_unch}",
          flush=True)
    digest = write_results(out, {
        **provenance_block(f"{X['run_kind']}_summary"),
        "preregistration": X["summary_preregistration"],
        "rule": RULE, "implementation_choices": X["choices"],
        "code": code0,
        **config_entries(X, {
            "reads": f"the per-record outputs {X['out_prefix']}*.json only, each listed with its sha256",
            "coverage": "every selection record on disk has exactly one output naming it with its current sha256 (the "
                        "selection records are hashed for that, not read)",
            "definitions": {"leakage_gap": LEAKAGE_GAP_DEFINITION, "resampling": RESAMPLING,
                            "margin": CHOICES["margin"],
                            "recorded_runner_up": "the selection record's own runner-up cell and margin, copied: the "
                                                  "scale a cell change is read against"}}),
        "inputs": inputs,
        "selection_records_covered": [{"file": f, "sha256": h} for f, h in on_disk.items()],
        "anchor": {"file": anchor_file, "sha256": anchor_sha},
        "n_records": len(rows), "n_selected_cell_unchanged": n_unch, "n_selected_cell_changed": len(rows) - n_unch,
        "records": rows,
    }, exclusive=True)
    print(f"Written: {rel(out)}\nsha256:  {digest}")
    return out


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="D-337: the cluster-grouped ho20 sensitivity of one Phase 2 selection "
                                             "record (--repo, --pooling), its anchor (--anchor), or the summary")
    ap.add_argument("--repo", help="the selection record's model")
    ap.add_argument("--pooling", choices=S.POOLINGS)
    ap.add_argument("--anchor", action="store_true", help="the item split must reproduce the record bit for bit")
    ap.add_argument("--summary", action="store_true", help="over every per-record output")
    ap.add_argument("--axis", choices=sorted(AXES), default="p2wg",
                    help="the selection records' axis: p2wg (WildGuardMix, default) or p2sst (SST-2, D-365 (4)(vi))")
    args = ap.parse_args()
    if args.summary:
        if args.repo or args.pooling or args.anchor:
            refuse("--summary takes no other option")
        run_summary(args.axis)
    else:
        if not args.repo or not args.pooling:
            refuse("--repo and --pooling are required (or --summary)")
        (run_anchor if args.anchor else run_grouped)(args.repo, args.pooling, args.axis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
