"""Phase 2 data, step 1 (D-303): near-duplicate clusters over the WildGuardMix pool UNION every Phase 1 item.

D-303 (docs/phase2_design.md revision 3, section 4 and details 2, 4, 5):
  POOL      WildGuardMix wildguardtrain, every usable row (exact-text distinct, as src/freeze_splits.py's load_axis
            defines usable: placeholders, conflicting labels and repeated texts dropped), excluding nothing yet. It is
            read through that same loader, OFFLINE, from a local hub-cache snapshot of the revision pinned in the
            Phase 1 manifest (--local-hub, made by src/fetch_dataset_snapshot.py). This script has no Hub path. The
            loader's cleaning counts must equal the recorded ones. Aegis is not mixed in (D-303 (5)).
  UNION     the pool plus every item of every data/frozen/*.jsonl file the Phase 1 manifests list
            (results/dataset_manifest.json, results/dataset_manifest_wildguard.json). Each file is checked against
            the sha256 recorded there and against its committed index manifest (manifests/splits/: sha256 and item
            order), and re-hashed after it is parsed, as src/negative_clusters.py does. Every Phase 1 WildGuardMix
            item must be a pool row with the same fields, or the run refuses.
  SOURCES   each Phase 1 file's source is named from its manifest entry: WildGuardMix, Aegis or XSTest (any other
            source refuses). A cluster holding a Phase 1 item may enter probe_train only (D-303 (2)). A cluster holding
            an XSTest item enters NO Phase 2 split: XSTest is extracted and evaluated for every Phase 2 model
            (D-303 (c)), so by D-303 (2)'s principle it is evaluation (the coordinator's decision, 2026-10-01).
  METHOD    D-164's: exact all-pairs Jaccard >= J_MIN (src/split_diagnostics.py) on the word 3-gram shingles of
            src/paraphrase_overlap.py; clusters are the connected components of that graph. Here the all-pairs
            product is computed in row blocks, so memory stays bounded at ~60k texts. The graph is the same, so the
            partition equals split_diagnostics.clusters()'s; tests/test_phase2_splits.py checks that on samples.
  NODES     keyed by item_id (freeze_splits: sha256 of the stripped text, first 16 hex). A pool row and a Phase 1 item
            with the same id are one node, and their texts must be equal. A cluster's label is the smallest item_id
            among its nodes.
  PLAN      plan() is the deterministic cluster-level split rule (FILL_RULE; D-303 (2), (4)), shared with
            src/phase2_freeze_splits.py, which recomputes it and refuses unless the assignment's sha256 matches.
  P2SST     --axis p2sst (D-338): the same steps over GLUE's SST-2 train split (nyu-mll/glue, config sst2; validation
            and test are never used), the SST-2 non-safety control. Its revision is the one its fetch record names
            (src/fetch_dataset_snapshot.py --resolve p2sst: the commit the Hub resolved when it was fetched); no Phase 1
            manifest pins it, and the run refuses if any Phase 1 axis reads it (SST-2 has no Phase 1 items). Labels
            are GLUE's coding: label 1 (positive sentiment) is the positive class, label 0 the negatives (D-338 (2)).
            The union with every Phase 1 item, the per-source rules (a cluster holding a Phase 1 item: probe_train
            only; an XSTest item: no split) and the clustering are p2wg's. The plan is FILL_RULE without step (2):
            there is no pool4d (D-338 (3)); sizes SIZES_P2SST. When the plan is exact the record also reports, as
            composition and not as a split rule, D-338 (3)'s substring containment (containment(), CONTAINMENT_RULE).
            Writes manifests/phase2_pool_clusters_p2sst.json and results/phase2/pool_clusters_p2sst.json; what p2wg
            writes is unchanged (the p2wg defaults, names and record fields are as before).

ENVIRONMENT  hf_env_guard.assert_hf_env runs first (HF_ENDPOINT https or unset, HF_HOME on NVMe, no token in the
             environment). --local-hub is required. HF_HUB_OFFLINE=1 and HF_DATASETS_OFFLINE=1 are set before the
             loader imports datasets. The record names the snapshot, the sha256 of its files and the library versions.

Writes (both refused if present; never overwritten; nothing ever deleted):
  manifests/phase2_pool_clusters.json  item ids and cluster labels only (no prompt text)
  results/phase2/pool_clusters.json    counts after clustering against the required sizes, the plan's counts, the
                                       sha256 of the assignment and of the pool's bytes, and every input with its
                                       sha256 (the Phase 1 manifests read through read_result)
Both files are built first. The record is checked by the D-297 (1) rule (provenance.input_identity_problems, as
provenance.write_results(path, payload) checks it) and serialised as write_results writes it. Both are then staged
and committed by src/safe_outputs.py's write_all. On a failure no file is deleted: final-named outputs of this call
are renamed aside to '<name>.failed.<pid>', and the exit names every file left (removal is a human's
scripts/safe_rm.sh call). Exit 3 if any minimum cannot be met: both files are written and nothing is shrunk; the
author decides.

  python3 src/phase2_pool_clusters.py --local-hub <hub dir holding the pinned snapshot>
  python3 src/phase2_pool_clusters.py --axis p2sst --local-hub <hub dir holding the GLUE snapshot>      (D-338)
The defaults are the final names. Any other output path must lie outside the repository (a test), and --test-sizes
is refused for the final names.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import freeze_splits as FS  # noqa: E402
import hf_env_guard as HG  # noqa: E402
import paired as P  # noqa: E402
from paraphrase_overlap import shingles  # noqa: E402
from provenance import (  # noqa: E402
    _RESULT_READS, _repo_rel, ensure_utf8_stdout, input_identity_problems, provenance_block, read_jsonl, read_result,
    sha256_bytes, sha256_file,
)
from safe_outputs import write_all  # noqa: E402,F401  (shared with src/phase2_freeze_splits.py)
from split_diagnostics import J_MIN  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FROZEN = ROOT / "data" / "frozen"                # Phase 1's split files (inputs only; Phase 2 writes data/frozen_p2/)
SPLIT_MANIFESTS = ROOT / "manifests" / "splits"
PHASE1_MANIFESTS = (RESULTS / "dataset_manifest.json", RESULTS / "dataset_manifest_wildguard.json")
OUT_MANIFEST = ROOT / "manifests" / "phase2_pool_clusters.json"
OUT_RECORD = RESULTS / "phase2" / "pool_clusters.json"
CODE_FILES = ["src/phase2_pool_clusters.py", "src/freeze_splits.py", "src/split_diagnostics.py",
              "src/paraphrase_overlap.py", "src/paired.py", "src/populations.py", "src/provenance.py",
              "src/superseded_provenance.py", "src/hf_env_guard.py", "src/safe_outputs.py"]

AXIS = "p2wg"                                   # the Phase 2 WildGuardMix axis: data/frozen_p2/p2wg__<split>.jsonl
SPEC = FS.SPEC["safety_wildguard"]              # the same source, config, split, fields and cleaning as Phase 1
BLOCK_ROWS = 1000                               # rows per block of the all-pairs product (memory only, not results)
SPLIT_SEED = 20261001                           # the cluster-order shuffle of plan(); no other stream uses it
SPLITS = ("eval", "calibration", "pool4d", "probe_train")
SOURCE_NAMES = {"allenai/wildguardmix": "WildGuardMix", "nvidia/Aegis-AI-Content-Safety-Dataset-2.0": "Aegis",
                "natolambert/xstest-v2-copy": "XSTest"}
EXCLUDED_SOURCES = ("XSTest",)                  # a cluster holding one of these enters no split (see SOURCES above)
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
TEST_STANDIN_MARKER = "GCCF_TEST_STANDIN"       # carried only by tests/test_phase2_splits.py's stand-in modules

# D-303's sizes. Each is read from the record named beside it; none is chosen here.
SIZES = {"eval_pos": 4000, "eval_neg_min": 5000, "cal_neg_min": 5000, "probe_pos": 2000, "probe_neg": 2000,
         "pool4d_pos": 1000, "pool4d_neg": 1000}
SIZE_SOURCES = {
    "eval_pos": "DECISIONS.md:714 D-303 (b) 'evaluation at least 5,000 negatives and at least 4,000 positives'; "
                "(4) 'evaluation positives at least 4,000 are filled first' (docs/phase2_design.md:9, 34). The "
                "minimum is taken as the size (the coordinator confirmed it, 2026-10-01)",
    "eval_neg_min": "DECISIONS.md:714 D-303 (b) 'evaluation at least 5,000 negatives'; (4) 'calibration and "
                    "evaluation negatives split the rest equally, each at least 5,000' (docs/phase2_design.md:37)",
    "cal_neg_min": "DECISIONS.md:714 D-303 (b) 'calibration at least 5,000 negatives' (docs/phase2_design.md:8, 37)",
    "probe_pos": "DECISIONS.md:714 D-303 (b) 'probe_train 4,000 (2,000 benign, 2,000 harmful)'; (4) 'probe_train "
                 "takes 2,000/2,000' (docs/phase2_design.md:7, 36)",
    "probe_neg": "as probe_pos",
    "pool4d_pos": "DECISIONS.md:714 D-303 (4) 'pool4d takes Phase 1's 4d sizes (D-244)'; DECISIONS.md:655 D-244 (4) "
                  "'Supply: 1000 benign and 1000 harmful items cover every configuration (at pool 1000: 500/500 at "
                  "50%; 990 + 10 + 250 injected at 1% with 25% poisoning)'; src/poolstudy_d156.py:9-10 'benign traffic "
                  "from cal_d's negatives; harmful traffic and injections from the 1000 threshold_calib harmful items "
                  "H'; src/poolstudy_d156.py:58-59 POOL_SIZES (200, 500, 1000), DEFAULT_POOL = 1000",
    "pool4d_neg": "as pool4d_pos: the 1000 benign items of D-244 (4)'s supply (cal_d's 1000 negatives in Phase 1)",
}
FILL_RULE = (
    "Units are the clusters holding at least one pool item. A unit is 'clean' when its cluster holds no Phase 1 item, "
    "and 'excluded' when it holds an XSTest item (no split). Units are sorted by their smallest pool item_id, then "
    "shuffled by random.Random(SPLIT_SEED) (as freeze_splits partitions). A fill walks that order and takes a whole "
    "unit only if it fits every remaining count of the split; a unit that would overflow is skipped and stays free. "
    "(1) eval positives first: clean ALL-POSITIVE units, until exactly eval_pos; clean mixed-label units only if those "
    "do not reach it. A mixed unit's negatives come with it (b) and are benign near-duplicates of harmful items that "
    "calibration can never hold (a D-283-kind confound), so each split row records 'mixed_label_cluster' and the "
    "record counts them. (2) pool4d (evaluation-like, D-303 (2); its negatives are 4d's recalibration traffic): clean "
    "all-positive units for its positives; clean mixed-label units only while those leave positives short (the same "
    "confound; each brings its negatives; counted in the record); then clean all-negative units, until exactly "
    "pool4d_pos and pool4d_neg. Each single-class pass takes multi-item units first, then singletons (as step (4)), "
    "so the singletons complete it exactly whenever enough remain. (3) probe_train: FIRST the units "
    "holding a WildGuardMix or Aegis Phase 1 item (never an XSTest one), in the shuffled order, each taken if it fits "
    "both classes' remaining counts; THEN free clean units, in the same order, to complete exactly probe_pos and "
    "probe_neg. Phase 1 units may enter probe_train only (D-303 (2)), so using them there first costs calibration and "
    "evaluation nothing (the coordinator's decision, 2026-10-01). (4) The remaining "
    "negatives (the clean all-negative units still free, N_R) are split equally between calibration and evaluation "
    "(D-303 (4)): each holds T = floor((N_R + b) / 2) negatives, and calibration holds negatives only. Multi-item "
    "units are placed first, then singletons, each in the shuffled order and each into whichever split has more "
    "room (ties to eval). The fill is exact, with at most one negative (the parity) unused, unless a multi-item unit "
    "had to be skipped for want of room; then a split falls short of T and the run exits 3. Clean mixed-label units "
    "left after (1)-(3) have no split left for their positives and are recorded as unused. A split not at its exact "
    "target, or T below either minimum, is a shortfall: exit 3.")

# --- axis p2sst (D-338): GLUE SST-2, the non-safety control ------------------------------------------------------
SPEC_P2SST = {"source": "nyu-mll/glue", "config": "sst2", "split": "train", "text_field": "sentence",
              "label_field": "label", "positive_value": 1, "negative_value": 0}   # GLUE's coding (D-338 (2))
SPLITS_P2SST = ("eval", "calibration", "probe_train")                             # no pool4d (D-338 (3))
SIZES_P2SST = {"eval_pos": 4000, "eval_neg_min": 5000, "cal_neg_min": 5000, "probe_pos": 2000, "probe_neg": 2000}
SIZE_SOURCES_P2SST = {
    "eval_pos": "DECISIONS.md:749 D-338 (3) 'evaluation at least 5,000 negatives and at least 4,000 positives' "
                "(D-303 (b)'s sizes); the minimum is taken as the size, as p2wg's plan takes it (SIZE_SOURCES)",
    "eval_neg_min": "DECISIONS.md:749 D-338 (3) 'evaluation at least 5,000 negatives'; the remaining negatives are split "
                    "equally with calibration, as p2wg's plan splits them (D-303 (4))",
    "cal_neg_min": "DECISIONS.md:749 D-338 (3) 'calibration at least 5,000 negatives'; negatives only, as p2wg's",
    "probe_pos": "DECISIONS.md:749 D-338 (3) 'probe_train 2,000 + 2,000'",
    "probe_neg": "as probe_pos",
}
FILL_RULE_P2SST = ("Axis p2sst (D-338 (3)): the rule below with step (2) left out, because p2sst has no pool4d split; "
                   "steps (1), (3) and (4) as written, with SIZES_P2SST and the same SPLIT_SEED. " + FILL_RULE)
OUT_MANIFEST_P2SST = ROOT / "manifests" / "phase2_pool_clusters_p2sst.json"
OUT_RECORD_P2SST = RESULTS / "phase2" / "pool_clusters_p2sst.json"
CONTAINMENT_Q = 4                               # the gram length of containment()'s candidate index (speed only)
CONTAINMENT_RULE = (
    "D-338 (3): composition (rule 5), not a split rule. Normalised text: the loader's text (freeze_splits.load_axis "
    "strips it) lowercased with str.lower. 'substring': a is contained in b when Python's `a in b` holds on the "
    "normalised texts (character level). 'token_aligned': the same on ' ' + text + ' ' (whole space-separated tokens). "
    "Counted: calibration and evaluation items contained in some probe_train item (with their labels); across "
    "calibration and evaluation, the items of each contained in some item of the other, and the unordered pairs (c, e) "
    "with c in e or e in c: pairs = |c in e| + |e in c| - pairs whose normalised texts are equal (only those hold both "
    "ways). Exact: the candidates of a needle are the items holding its rarest 4-character gram (an inverted index), and "
    "every candidate is verified with `in`; a needle shorter than 4 characters is checked against every item.")
AXES: dict[str, dict[str, Any]] = {
    "p2wg": {"spec": SPEC, "splits": SPLITS, "sizes": SIZES, "size_sources": SIZE_SOURCES, "fill_rule": FILL_RULE,
             "decision": "D-303", "out_manifest": OUT_MANIFEST, "out_record": OUT_RECORD, "pinned_by_phase1": True,
             "containment": False, "class_names": ("unharmful", "harmful")},
    "p2sst": {"spec": SPEC_P2SST, "splits": SPLITS_P2SST, "sizes": SIZES_P2SST, "size_sources": SIZE_SOURCES_P2SST,
              "fill_rule": FILL_RULE_P2SST, "decision": "D-338", "out_manifest": OUT_MANIFEST_P2SST,
              "out_record": OUT_RECORD_P2SST, "pinned_by_phase1": False, "containment": True,
              "class_names": ("negative", "positive")},
}
P2SST_LABELS = ("GLUE's coding: label 1 (positive sentiment) is the probe's positive class and label 0 its negatives, "
                "so the threshold is set on label-0 calibration items (D-338 (2)); a sign convention, not a claim about "
                "either sentiment")


# --- clustering -------------------------------------------------------------------------------------------------

def blocked_clusters(texts: list[str], block_rows: int = BLOCK_ROWS) -> tuple[np.ndarray, dict[str, Any]]:
    """split_diagnostics.clusters() computed in row blocks: the same shingle matrix, the same exact Jaccard
    (n / (|A| + |B| - n), the same dtypes), the same J >= J_MIN graph. Each block of rows is multiplied only
    against itself and the rows after it (the upper triangle), and its edges are merged into the components at
    once, so no all-pairs matrix is ever held. Returns, per text, the smallest index in its component, and the
    summary split_diagnostics.clusters() reports."""
    from scipy.sparse import coo_matrix, csr_matrix
    from scipy.sparse.csgraph import connected_components

    if block_rows < 1:
        raise ValueError("block_rows must be >= 1")
    n = len(texts)
    sh = [shingles(t) for t in texts]
    vocab: dict[int, int] = {}
    rows, cols = [], []
    for i, s in enumerate(sh):
        for h in s:
            rows.append(i)
            cols.append(vocab.setdefault(h, len(vocab)))
    M = csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(n, len(vocab)))
    size = np.asarray(M.sum(axis=1)).ravel()
    root = np.arange(n, dtype=np.int64)          # root[x]: the smallest index in x's component so far (compressed)
    n_pairs = 0
    for s in range(0, n, block_rows):
        e = min(n, s + block_rows)
        B = (M[s:e] @ M[s:].T).tocoo()
        i = B.row.astype(np.int64) + s
        j = B.col.astype(np.int64) + s
        keep = i < j
        i, j, cnt = i[keep], j[keep], B.data[keep]
        jac = cnt / (size[i] + size[j] - cnt)
        edge = jac >= J_MIN
        n_pairs += int(edge.sum())
        a, b = root[i[edge]], root[j[edge]]
        m = a != b
        if not m.any():
            continue
        code = np.unique(np.minimum(a[m], b[m]) * n + np.maximum(a[m], b[m]))   # distinct root pairs
        a, b = code // n, code % n
        nodes = np.unique(np.concatenate([a, b]))
        g = coo_matrix((np.ones(a.size, dtype=np.int64), (np.searchsorted(nodes, a), np.searchsorted(nodes, b))),
                       shape=(nodes.size, nodes.size))
        _, comp = connected_components(g, directed=False)
        low = np.full(int(comp.max()) + 1, n, dtype=np.int64)
        np.minimum.at(low, comp, nodes)
        remap = np.arange(n, dtype=np.int64)
        remap[nodes] = low[comp]
        root = remap[root]
    comp_sizes = Counter(root.tolist())
    multi = [c for c in comp_sizes.values() if c >= 2]
    return root, {"n_items": n, "n_empty_shingle_sets": int((size == 0).sum()),
                  "n_pairs_jaccard_ge_0_5": n_pairs, "n_clusters": len(multi),
                  "items_in_clusters": int(sum(multi)),
                  "size_distribution": {str(k): v for k, v in sorted(Counter(multi).items())},
                  "largest_cluster_sizes": sorted(multi, reverse=True)[:10]}


# --- the split plan (shared with src/phase2_freeze_splits.py) --------------------------------------------------

def units_from(item_ids: list[str], labels: list[int], clusters: list[str], is_phase1: list[int],
               cluster_sources: dict[str, list[str]]) -> list[dict[str, Any]]:
    """One unit per cluster holding at least one pool item: its pool counts by class, the Phase 1 sources its
    cluster holds (any node of it, pool or not), whether it is excluded (an XSTest item), the pool items that are
    themselves Phase 1 items, and its order key (the smallest pool item_id)."""
    U: dict[str, dict[str, Any]] = {}
    for iid, y, c, f in zip(item_ids, labels, clusters, is_phase1):
        src = sorted(cluster_sources.get(c, []))
        u = U.setdefault(c, {"cluster": c, "pos": 0, "neg": 0, "phase1": c in cluster_sources, "sources": src,
                             "excluded": any(s in EXCLUDED_SOURCES for s in src), "p1_items": 0, "key": iid})
        u["pos" if int(y) == 1 else "neg"] += 1
        u["p1_items"] += int(bool(f))
        u["key"] = min(u["key"], iid)
    return list(U.values())


def _tot(us: list[dict[str, Any]]) -> dict[str, int]:
    return {"units": len(us), "pos": sum(u["pos"] for u in us), "neg": sum(u["neg"] for u in us),
            "multi_item_units": sum(1 for u in us if u["pos"] + u["neg"] >= 2)}


def availability(units: list[dict[str, Any]]) -> dict[str, Any]:
    """The counts after clustering that the sizes are checked against."""
    clean = [u for u in units if not u["phase1"]]
    p1 = [u for u in units if u["phase1"] and not u["excluded"]]
    exc = [u for u in units if u["excluded"]]
    mixed = [u for u in clean if u["pos"] and u["neg"]]
    return {"pool_usable": {"n": sum(u["pos"] + u["neg"] for u in units), "pos": sum(u["pos"] for u in units),
                            "neg": sum(u["neg"] for u in units)},
            "pool_items_that_are_phase1_items": sum(u["p1_items"] for u in units),
            "units_total": len(units),
            "clean_units": {**_tot(clean), "mixed_units": len(mixed), "pos_in_mixed_units": sum(u["pos"] for u in mixed),
                            "neg_in_mixed_units": sum(u["neg"] for u in mixed)},
            "phase1_cluster_units_probe_train_only": {**_tot(p1), "pool_items_not_themselves_phase1": sum(
                u["pos"] + u["neg"] - u["p1_items"] for u in p1)},
            "xstest_cluster_units_excluded": {**_tot(exc), "pool_items_not_themselves_phase1": sum(
                u["pos"] + u["neg"] - u["p1_items"] for u in exc)}}


def plan(units: list[dict[str, Any]], sizes: dict[str, int], seed: int = SPLIT_SEED,
         splits: tuple[str, ...] = SPLITS) -> dict[str, Any]:
    """The cluster-level split rule (FILL_RULE). Returns the assignment {cluster: split} under 'assign' and the
    counts, targets, unused items, per-source placement and shortfalls (everything else is JSON-ready and recorded).
    splits: SPLITS (p2wg) or SPLITS_P2SST (D-338: no pool4d, so step (2) is left out and nothing names pool4d)."""
    if tuple(splits) not in (SPLITS, SPLITS_P2SST):
        raise ValueError(f"splits must be {SPLITS} or {SPLITS_P2SST}, not {splits}")
    has4d = "pool4d" in splits
    order = sorted(units, key=lambda u: u["key"])
    random.Random(seed).shuffle(order)
    assign: dict[str, str] = {}
    got = {s: {"pos": 0, "neg": 0, "units": 0, "multi_item_units": 0, "phase1_units": 0, "phase1_items": 0}
           for s in splits}

    def take(u: dict[str, Any], s: str) -> None:
        assign[u["cluster"]] = s
        g = got[s]
        g["pos"] += u["pos"]
        g["neg"] += u["neg"]
        g["units"] += 1
        g["multi_item_units"] += int(u["pos"] + u["neg"] >= 2)
        g["phase1_units"] += int(u["phase1"])
        g["phase1_items"] += u["p1_items"]

    def free(u: dict[str, Any]) -> bool:
        return u["cluster"] not in assign

    def fill(s: str, tpos: int, tneg: int, ok: Any) -> None:
        g = got[s]
        for u in order:
            if g["pos"] == tpos and g["neg"] == tneg:
                break
            if free(u) and ok(u) and g["pos"] + u["pos"] <= tpos and g["neg"] + u["neg"] <= tneg:
                take(u, s)

    def clean(u: dict[str, Any]) -> bool:
        return not u["phase1"]

    EP = sizes["eval_pos"]
    for mixed_pass in (False, True):            # (1) eval positives first: all-positive units, mixed ones only if needed
        for u in order:
            if got["eval"]["pos"] == EP:
                break
            if free(u) and clean(u) and u["pos"] > 0 and bool(u["neg"]) == mixed_pass \
                    and got["eval"]["pos"] + u["pos"] <= EP:
                take(u, "eval")
    b = got["eval"]["neg"]
    eval_mixed = [u for u in order if assign.get(u["cluster"]) == "eval" and u["neg"] > 0]
    eval_mixed = {"units": len(eval_mixed), "pos": sum(u["pos"] for u in eval_mixed),
                  "neg": sum(u["neg"] for u in eval_mixed)}
    # (2) pool4d: all-positive units for its positives, mixed units only while positives are short (a mixed unit can
    # fit only while there is positive room), then all-negative units for its negatives. Each single-class pass takes
    # multi-item units first and singletons after (as step (4) does), so the singletons complete it exactly
    if has4d:                                   # p2sst has no pool4d (D-338 (3)): step (2) is left out
        P4, N4 = sizes["pool4d_pos"], sizes["pool4d_neg"]

        def multi_first(ok: Any) -> None:
            fill("pool4d", P4, N4, lambda u: ok(u) and u["pos"] + u["neg"] >= 2)
            fill("pool4d", P4, N4, lambda u: ok(u) and u["pos"] + u["neg"] == 1)
        multi_first(lambda u: clean(u) and u["pos"] > 0 and u["neg"] == 0)
        fill("pool4d", P4, N4, lambda u: clean(u) and u["pos"] > 0 and u["neg"] > 0)
        multi_first(lambda u: clean(u) and u["pos"] == 0)
        pool4d_mixed = [u for u in order if assign.get(u["cluster"]) == "pool4d" and u["pos"] and u["neg"]]
        pool4d_mixed = {"units": len(pool4d_mixed), "pos": sum(u["pos"] for u in pool4d_mixed),
                        "neg": sum(u["neg"] for u in pool4d_mixed)}
    fill("probe_train", sizes["probe_pos"], sizes["probe_neg"],                               # (3) Phase 1 units first
         lambda u: u["phase1"] and not u["excluded"])
    fill("probe_train", sizes["probe_pos"], sizes["probe_neg"], clean)                       # (3) then clean units
    rest = [u for u in order if free(u) and clean(u) and u["pos"] == 0]                      # (4) equal split
    n_rest = sum(u["neg"] for u in rest)
    T = (n_rest + b) // 2
    for group in ([u for u in rest if u["neg"] >= 2], [u for u in rest if u["neg"] == 1]):
        for u in group:
            room = {"eval": T - got["eval"]["neg"], "calibration": T - got["calibration"]["neg"]}
            if room["eval"] <= 0 and room["calibration"] <= 0:
                break
            for s in sorted(room, key=lambda k: -room[k]):      # more room first; a tie keeps eval first (stable)
                if u["neg"] <= room[s]:
                    take(u, s)
                    break

    targets = {"eval": {"pos": EP, "neg": T}, "calibration": {"pos": 0, "neg": T}}
    if has4d:
        targets["pool4d"] = {"pos": sizes["pool4d_pos"], "neg": sizes["pool4d_neg"]}
    targets["probe_train"] = {"pos": sizes["probe_pos"], "neg": sizes["probe_neg"]}
    shortfall = []
    if T < sizes["eval_neg_min"] or T < sizes["cal_neg_min"]:
        shortfall.append(f"eval and calibration negatives get T = {T} each, below the minimum "
                         f"{max(sizes['eval_neg_min'], sizes['cal_neg_min'])}")
    for s in splits:
        if (got[s]["pos"], got[s]["neg"]) != (targets[s]["pos"], targets[s]["neg"]):
            shortfall.append(f"{s} holds {got[s]['pos']} positives / {got[s]['neg']} negatives, target "
                             f"{targets[s]['pos']} / {targets[s]['neg']}")
    fr = [u for u in units if free(u)]
    unused = {"clean_mixed_label": _tot([u for u in fr if clean(u) and u["pos"] and u["neg"]]),
              "clean_all_negative_parity_or_packing": _tot([u for u in fr if clean(u) and not u["pos"]]),
              "clean_all_positive": _tot([u for u in fr if clean(u) and not u["neg"]]),
              "phase1_clusters_not_in_probe_train": _tot([u for u in fr if u["phase1"] and not u["excluded"]]),
              "excluded_xstest_clusters": _tot([u for u in fr if u["excluded"]])}
    by_source: dict[str, dict[str, Any]] = {}
    for u in units:
        for src in u["sources"]:
            d = by_source.setdefault(src, {"units_with_pool_items": 0, "probe_train": [], "excluded": [], "unused": []})
            d["units_with_pool_items"] += 1
            where = assign.get(u["cluster"])
            if u["excluded"]:
                if where is not None:
                    raise AssertionError("an excluded cluster was placed")
                d["excluded"].append(u["cluster"])
            elif where == "probe_train":
                d["probe_train"].append(u["cluster"])
            elif where is None:
                d["unused"].append(u["cluster"])
            else:
                raise AssertionError(f"a Phase 1 cluster was placed in {where}")
    for d in by_source.values():
        for k in ("probe_train", "excluded", "unused"):
            d[k] = sorted(d[k])
    in_probe = [u for u in units if assign.get(u["cluster"]) == "probe_train"]
    probe_comp = {"from_phase1_item_clusters": {k: sum(u[k] for u in in_probe if u["phase1"]) for k in ("pos", "neg")},
                  "from_clean_clusters": {k: sum(u[k] for u in in_probe if not u["phase1"]) for k in ("pos", "neg")},
                  "from_phase1_item_clusters_by_source": {
                      src: {k: sum(u[k] for u in in_probe if src in u["sources"]) for k in ("pos", "neg")}
                      for src in sorted({s for u in in_probe for s in u["sources"]})},
                  "by_source_note": "a cluster holding items of two sources counts under each"}
    fill_order = ["eval positives", "pool4d", "probe_train (Phase 1 units first)", "eval negatives + calibration"]
    out = {"seed": seed, "fill_order": fill_order if has4d else [x for x in fill_order if x != "pool4d"],
            "T": T, "T_inputs": {"N_R_clean_all_negative_free_after_1_to_3": n_rest,
                                 "b_eval_negatives_from_positive_fill": b},
            "targets": targets, "splits": got, "unused": unused, "phase1_by_source": by_source,
            "probe_train_composition": probe_comp,
            "eval_from_mixed_label_clusters": {
                **eval_mixed, "note": "eval's negatives from these clusters are benign near-duplicates of harmful "
                                      "items, which calibration (all-negative clusters only) can never hold; taken "
                                      "only when the all-positive clusters could not fill eval_pos"}}
    if has4d:
        out["pool4d_from_mixed_label_clusters"] = {
            **pool4d_mixed, "note": "pool4d is evaluation-like (D-303 (2)) and its negatives are 4d's recalibration "
                                    "traffic; mixed-label clusters are taken only when the single-class clusters "
                                    "could not fill its counts"}
    out.update(shortfall=shortfall, assign_sha256=assign_sha256(assign), assign=assign)
    return out


def assign_sha256(assign: dict[str, str]) -> str:
    """sha256 of json.dumps(sorted(assign.items())): the cluster -> split assignment, recorded and re-checked."""
    return sha256_bytes(json.dumps(sorted(assign.items())).encode("utf-8"))


def mixed_preference_violations(units: list[dict[str, Any]], assign: dict[str, str]) -> list[str]:
    """Rule 5 (a D-283-kind confound): a mixed-label unit may enter eval or pool4d only while their positives were
    short. When step (1) ended, eval's remaining positive room was exactly the positives its mixed units then brought
    (R). Every clean all-positive unit eval did not take must therefore hold more than R positives, or eval would have
    taken it (the room only shrinks during a pass). The same holds for pool4d among the units eval did not take, since
    pool4d is filled after eval. Returns one line per split that breaks this (empty when none does)."""
    out = []
    for s, before in (("eval", {"eval"}), ("pool4d", {"eval", "pool4d"})):
        R = sum(u["pos"] for u in units if assign.get(u["cluster"]) == s and u["pos"] and u["neg"])
        if not R:
            continue
        bad = sorted(u["cluster"] for u in units if not u["phase1"] and u["pos"] > 0 and u["neg"] == 0
                     and assign.get(u["cluster"]) not in before and u["pos"] <= R)
        if bad:
            out.append(f"{s} took mixed-label clusters ({R} positives) while {len(bad)} clean all-positive clusters "
                       f"that fit stayed out of it (first {bad[0]})")
    return out


def plan_record(p: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in p.items() if k != "assign"}


class SubstringIndex:
    """Which of `hay` contain a needle (Python's `in`), exactly: the candidates are the haystacks holding the needle's
    rarest q-character gram (an inverted index built once), and each candidate is verified with `in`; a needle shorter
    than q is checked against every haystack. Every haystack containing the needle holds each of its grams, so nothing
    is missed."""

    def __init__(self, hay: list[str], q: int = CONTAINMENT_Q):
        if q < 1:
            raise ValueError("q must be >= 1")
        self.hay, self.q = hay, q
        post: dict[str, list[int]] = {}
        for k, h in enumerate(hay):
            for g in {h[i:i + q] for i in range(len(h) - q + 1)}:
                post.setdefault(g, []).append(k)
        self.post = post

    def containing(self, needle: str, first_only: bool = False) -> list[int]:
        """The indices of the haystacks containing `needle`, ascending (only the first when first_only)."""
        q, hay = self.q, self.hay
        cand: Any = range(len(hay))
        if len(needle) >= q:
            best = None
            for i in range(len(needle) - q + 1):
                p = self.post.get(needle[i:i + q])
                if p is None:
                    return []
                if best is None or len(p) < len(best):
                    best = p
            cand = best
        out = []
        for k in cand:
            if needle in hay[k]:
                out.append(k)
                if first_only:
                    break
        return out


def containment(texts: dict[str, list[str]], labels: dict[str, list[int]], q: int = CONTAINMENT_Q) -> dict[str, Any]:
    """D-338 (3)'s composition (CONTAINMENT_RULE) of the splits probe_train, calibration and eval, given each split's
    texts (as the loader returns them) and labels in the same order. Counts only; no split is changed by it."""
    def by_label(hit: list[bool], ys: list[int]) -> dict[str, int]:
        return {"n": len(hit), "contained": sum(hit), "contained_pos": sum(1 for h, y in zip(hit, ys) if h and int(y) == 1),
                "contained_neg": sum(1 for h, y in zip(hit, ys) if h and int(y) == 0)}

    def one(norm: Any) -> dict[str, Any]:
        nt = {s: [norm(t) for t in texts[s]] for s in ("probe_train", "calibration", "eval")}
        probe = SubstringIndex(nt["probe_train"], q)
        in_probe = {s: by_label([bool(probe.containing(t, first_only=True)) for t in nt[s]], labels[s])
                    for s in ("calibration", "eval")}
        in_eval, in_cal = SubstringIndex(nt["eval"], q), SubstringIndex(nt["calibration"], q)
        c_in_e = [len(in_eval.containing(t)) for t in nt["calibration"]]    # eval items holding each calibration item
        e_in_c = [len(in_cal.containing(t)) for t in nt["eval"]]            # calibration items holding each eval item
        cc, ec = Counter(nt["calibration"]), Counter(nt["eval"])
        equal = sum(n * ec[t] for t, n in cc.items() if t in ec)
        return {"in_probe_train": in_probe,
                "calibration_vs_eval": {
                    "calibration_items_in_some_eval_item": by_label([x > 0 for x in c_in_e], labels["calibration"]),
                    "eval_items_in_some_calibration_item": by_label([x > 0 for x in e_in_c], labels["eval"]),
                    "pairs_calibration_in_eval": sum(c_in_e), "pairs_eval_in_calibration": sum(e_in_c),
                    "pairs_equal_normalised_text": equal, "pairs": sum(c_in_e) + sum(e_in_c) - equal}}
    return {"rule": CONTAINMENT_RULE, "substring": one(lambda t: t.lower()),
            "token_aligned": one(lambda t: " " + t.lower() + " ")}


def split_texts(pool: list[dict[str, Any]], clusters: list[str], assign: dict[str, str],
                splits: tuple[str, ...]) -> tuple[dict[str, list[str]], dict[str, list[int]]]:
    """Each split's texts and labels under an assignment {cluster: split}, in pool order."""
    texts: dict[str, list[str]] = {s: [] for s in splits}
    labels: dict[str, list[int]] = {s: [] for s in splits}
    for r, c in zip(pool, clusters):
        s = assign.get(c)
        if s is not None:
            texts[s].append(r["text"])
            labels[s].append(int(r["label"]))
    return texts, labels


def split_bytes(rows: list[dict[str, Any]]) -> bytes:
    """freeze_splits.write_jsonl's bytes: one json.dumps(sort_keys, ensure_ascii=False) line per record, LF."""
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def sizes_from(test_sizes: str | None, base: dict[str, int] = SIZES) -> dict[str, int]:
    if test_sizes is None:
        return dict(base)
    s = json.loads(test_sizes)  # not a result (D-297): a command-line argument
    if set(s) != set(base) or any(not isinstance(v, int) or v < 0 for v in s.values()):
        raise SystemExit(f"REFUSING: --test-sizes needs exactly the keys {sorted(base)}, as non-negative integers")
    return s


def outside_repo_or_refuse(paths: list[Path], what: str) -> None:
    inside = [str(p) for p in paths if p.resolve().is_relative_to(ROOT.resolve())]
    if inside:
        raise SystemExit(f"REFUSING: a {what} writes outside the repository; these are inside it: {inside}")


# --- environment and outputs (shared with src/phase2_freeze_splits.py) -----------------------------------------

def guard_and_offline(caller: str, local_hub: str | None) -> dict[str, str]:
    """hf_env_guard first; then --local-hub is required (no Hub path here), and the offline switches are set before
    datasets is imported (freeze_splits.load_axis imports it on first use)."""
    HG.assert_hf_env(caller)
    if not local_hub:
        raise SystemExit(f"REFUSING: {caller} reads the dataset only from a local snapshot; pass --local-hub (made by "
                         "src/fetch_dataset_snapshot.py)")
    os.environ.update(OFFLINE_ENV)
    return dict(OFFLINE_ENV)


DATASETS_CACHE_NOTE = (
    "HF_DATASETS_CACHE is a fresh directory made for this run on NVMe before datasets is imported. Any cleanup the "
    "datasets library does of its own Arrow cache (on a failed load its builder removes its '.incomplete' directory, "
    "freeze_splits.load_axis -> load_dataset) is therefore confined to this run's directory. That library-internal "
    "cleanup is the one remaining deletion in this pipeline, and it is the author's to accept. The directory is left "
    "after the run; removal is a human's scripts/safe_rm.sh call.")


def datasets_cache(caller: str, root: str | None) -> dict[str, Any]:
    """Make this run's own HF_DATASETS_CACHE: <root>/<caller>.<utc>.<pid>.<random>, on NVMe, and point datasets at it.
    The root is --datasets-cache-root, else $HF_HOME/gccf_datasets_cache (hf_env_guard has checked HF_HOME is on
    NVMe). Called after every output check, right before the loader runs, so a refused run makes no directory.
    Refuses if datasets or huggingface_hub is already imported: they read their cache settings at import, so the
    variable set afterwards would not hold. The only exception is a module carrying TEST_STANDIN_MARKER, which only
    the tests' stand-ins carry; their presence is recorded."""
    import time
    early = [m for m in ("datasets", "huggingface_hub")
             if m in sys.modules and not getattr(sys.modules[m], TEST_STANDIN_MARKER, False)]
    if early:
        raise SystemExit(f"REFUSING: {early} imported before this run's HF_DATASETS_CACHE was set")
    standins = sorted(m for m in ("datasets", "huggingface_hub") if m in sys.modules)
    if root:
        base = Path(root)
    elif os.environ.get("HF_HOME"):
        base = Path(os.environ["HF_HOME"]) / "gccf_datasets_cache"
    else:
        raise SystemExit("REFUSING: no --datasets-cache-root and no HF_HOME; the datasets cache needs an NVMe directory")
    base = base.resolve()
    try:
        mnt, src = HG.mount_of(base)
    except OSError as e:
        raise SystemExit(f"REFUSING: cannot tell which device holds {base} ({type(e).__name__})")
    if not src.startswith(HG.NVME_SOURCE_PREFIX):
        raise SystemExit(f"REFUSING: the datasets cache root {base} is on {src or 'an unknown device'}, not local NVMe")
    d = base / f"{Path(caller).stem}.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.{os.getpid()}.{os.urandom(4).hex()}"
    d.mkdir(parents=True, exist_ok=False)
    os.environ["HF_DATASETS_CACHE"] = str(d)
    return {"dir": str(d), "root": str(base), "mount_point": mnt, "mount_source": src,
            "test_standin_modules_present": standins, "note": DATASETS_CACHE_NOTE}


def lib_versions() -> dict[str, str | None]:
    from importlib import metadata
    out: dict[str, str | None] = {}
    for d in ("datasets", "huggingface_hub", "pyarrow", "fsspec"):
        try:
            out[d] = metadata.version(d)
        except metadata.PackageNotFoundError:
            out[d] = None
    return out


def record_bytes(payload: dict[str, Any], name: str) -> bytes:
    """The record's bytes exactly as provenance.write_results(path, payload) writes them (indent 2, default=str), after
    the D-297 (1) results-writer check write_results makes (input_identity_problems); refuses otherwise."""
    problems = input_identity_problems(payload)
    if problems:
        raise SystemExit(f"REFUSING to write {name} (D-297 (1) results-writer rule): {problems}")
    data = json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8")
    json.loads(data.decode("utf-8"))  # not a result (D-297): the round trip of the bytes about to be written
    return data


# --- inputs -----------------------------------------------------------------------------------------------------

def load_phase1(manifest_paths: list[Path], frozen_dir: Path, split_dir: Path, spec: dict[str, Any] = SPEC,
                own_axis: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, str]]:
    """Every split file the Phase 1 manifests list, verified and named by source; the Phase 1 axis entry that reads
    spec's source (p2wg: WildGuardMix, required; own_axis False, p2sst: there must be none, and None is returned);
    {input: sha256}."""
    files: list[dict[str, Any]] = []
    inputs: dict[str, str] = {}
    wg: dict[str, Any] | None = None
    for mp in manifest_paths:
        doc = read_result(mp)                                  # registered: listed in the record's inputs
        inputs[_repo_rel(mp)] = _RESULT_READS[_repo_rel(mp)]
        for axis, a in doc["axes"].items():
            if a["source"] not in SOURCE_NAMES:
                raise SystemExit(f"REFUSING: {_repo_rel(mp)} axis {axis} reads {a['source']}, a source this rule does "
                                 f"not name ({sorted(SOURCE_NAMES)})")
            if (a["source"], a["config"], a["split"]) == (spec["source"], spec["config"], spec["split"]):
                if wg is not None:
                    raise SystemExit(f"REFUSING: two Phase 1 axes read {spec['source']} {spec['config']}")
                wg = {"axis": axis, "entry": a, "manifest": _repo_rel(mp)}
            for part, e in a["partitions"].items():
                name = Path(e["file"]).name
                f = frozen_dir / name
                if not f.is_file():
                    raise SystemExit(f"REFUSING: {name}, listed in {_repo_rel(mp)}, is not in {frozen_dir}")
                h = sha256_file(f)
                if h != e["sha256"]:
                    raise SystemExit(f"REFUSING: {name} sha256 {h[:16]} != {e['sha256'][:16]} recorded in {_repo_rel(mp)}")
                mf = split_dir / f"{Path(name).stem}__manifest.json"
                mb = mf.read_bytes()
                man = json.loads(mb.decode("utf-8"))  # not a result (D-297): a committed index manifest, hashed and listed
                if man["sha256_of_original"] != h:
                    raise SystemExit(f"REFUSING: {name} sha256 {h[:16]} != committed manifest "
                                     f"{man['sha256_of_original'][:16]}")
                rows = read_jsonl(f)                           # the sanctioned reader (F-017)
                if sha256_file(f) != h:
                    raise SystemExit(f"REFUSING: {name} changed while it was being read")
                if [r["item_id"] for r in rows] != [x["item_id"] for x in man["items"]] or len(rows) != e["n"]:
                    raise SystemExit(f"REFUSING: {name} items differ from its committed manifest or its count")
                if any(x["file"] == name for x in files):
                    raise SystemExit(f"REFUSING: {name} is listed twice")
                inputs[_repo_rel(f)] = h
                inputs[_repo_rel(mf)] = sha256_bytes(mb)
                files.append({"file": name, "axis": axis, "part": part, "source": SOURCE_NAMES[a["source"]],
                              "sha256": h, "rows": rows})
    if own_axis and wg is None:
        raise SystemExit(f"REFUSING: no Phase 1 manifest names {spec['source']} {spec['config']}; no pinned revision")
    if not own_axis and wg is not None:
        raise SystemExit(f"REFUSING: Phase 1 axis {wg['axis']} ({wg['manifest']}) reads {spec['source']} "
                         f"{spec['config']}, which this axis says has no Phase 1 items (D-338)")
    return files, wg, inputs


def snapshot_dir(local_hub: str, rev: str, spec: dict[str, Any] = SPEC) -> Path:
    """Where freeze_splits.load_axis(local_hub=...) reads the pinned revision."""
    return Path(local_hub) / f"datasets--{spec['source'].replace('/', '--')}" / "snapshots" / rev


def load_pool(wg_entry: dict[str, Any], local_hub: str, cache: dict[str, Any] | None = None,
              expected_files: dict[str, str] | None = None, spec: dict[str, Any] = SPEC, axis: str = AXIS,
              check_recorded: bool = True) -> tuple[list[dict[str, Any]], str, dict[str, int], dict]:
    """The pool through freeze_splits.load_axis from the local snapshot of the pinned revision; refuses unless the
    snapshot's files are expected_files (when given: the fetch record's, or step 1's), the same before and after the
    loader ran, and the loader's cleaning counts and usable-row count equal the ones the Phase 1 freeze recorded (the
    same bytes were read). `cache` is this run's datasets_cache(); the datasets library must report it as its cache.
    Also returns what was read: the snapshot, its files' sha256, the cache and the library versions.
    check_recorded False only for p2sst's step 1 (D-338), whose first read has no recorded counts to equal; step 2
    then checks against the counts step 1 recorded."""
    rev = wg_entry["source_revision_sha"]
    if not isinstance(rev, str) or not re.fullmatch(r"[0-9a-f]{40}", rev):
        raise SystemExit(f"REFUSING: the pinned source revision {rev!r} is not a 40-hex commit")
    snap = snapshot_dir(local_hub, rev, spec)
    if not snap.is_dir():
        raise SystemExit(f"REFUSING: no local snapshot of {spec['source']}@{rev[:12]} under {local_hub}")

    def hashes() -> dict[str, str]:
        return {p.relative_to(snap).as_posix(): sha256_file(p) for p in sorted(snap.rglob("*")) if p.is_file()}
    files = hashes()
    if expected_files is not None and files != expected_files:
        raise SystemExit(f"REFUSING: the snapshot {snap}'s files differ from the ones recorded for it")
    records, _, stats = FS.load_axis(axis, spec, revision=rev, local_hub=local_hub)
    if hashes() != files:
        raise SystemExit(f"REFUSING: the snapshot {snap} changed while the loader read it")
    used = getattr(getattr(sys.modules.get("datasets"), "config", None), "HF_DATASETS_CACHE", None)
    if cache is not None and used is None and not cache.get("test_standin_modules_present"):
        raise SystemExit("REFUSING: the datasets library did not report its cache, so this run's cache cannot be confirmed")
    if cache is not None and used is not None and Path(str(used)).resolve() != Path(cache["dir"]).resolve():
        raise SystemExit(f"REFUSING: the datasets library used the cache {used}, not this run's {cache['dir']}")
    if check_recorded and (stats != wg_entry["cleaning"] or len(records) != wg_entry["total_usable_rows"]):
        raise SystemExit(f"REFUSING: cleaning {stats} / {len(records)} usable rows != recorded "
                         f"{wg_entry['cleaning']} / {wg_entry['total_usable_rows']}")
    read = {"source_was_local_snapshot": True, "snapshot": snap.relative_to(Path(local_hub)).as_posix(),
            "local_snapshot_path": str(snap), "snapshot_files_sha256": files, "libraries": lib_versions(),
            "datasets_cache": {**(cache or {}), "library_reported": None if used is None else str(used)}}
    return records, rev, stats, read


def fetch_record_path(rev: str, spec: dict[str, Any] = SPEC) -> Path:
    """The record src/fetch_dataset_snapshot.py writes for the pinned snapshot."""
    org, name = spec["source"].split("/")
    return RESULTS / "phase2" / f"fetch_dataset__{org}__{name}@{rev[:12]}.json"


def read_fetch_record(path: Path, rev: str, spec: dict[str, Any] = SPEC) -> tuple[dict[str, str], str, str]:
    """The fetch record's {file: sha256} for the pinned snapshot, read through read_result (listed by the caller);
    refuses if it is absent or names another repo or revision."""
    if not path.is_file():
        raise SystemExit(f"REFUSING: no dataset fetch record at {path}; run src/fetch_dataset_snapshot.py first")
    frec = read_result(path)                                   # registered: listed in the record's inputs
    rel = _repo_rel(path)
    if frec.get("repo") != spec["source"] or frec.get("revision") != rev:
        raise SystemExit(f"REFUSING: {rel} is for {frec.get('repo')}@{str(frec.get('revision'))[:12]}, not "
                         f"{spec['source']}@{rev[:12]}")
    return {f["name"]: f["sha256"] for f in frec["files"]}, rel, _RESULT_READS[rel]


def default_resolved_fetch_record(spec: dict[str, Any]) -> Path:
    """p2sst (D-338 (1)): the one results/phase2/fetch_dataset__<org>__<name>@*.json; refuses unless exactly one."""
    org, name = spec["source"].split("/")
    found = sorted((RESULTS / "phase2").glob(f"fetch_dataset__{org}__{name}@*.json"))
    if len(found) != 1:
        raise SystemExit(f"REFUSING: {len(found)} fetch records of {spec['source']} in results/phase2/; name one with "
                         "--fetch-record")
    return found[0]


def read_resolved_fetch_record(path: Path, spec: dict[str, Any], axis: str) -> tuple[str, dict[str, str], str, str]:
    """p2sst (D-338 (1)): the fetch record of a snapshot pinned when it was fetched (src/fetch_dataset_snapshot.py
    --resolve <axis>), read through read_result (listed by the caller): its revision, {file: sha256}, path and sha256.
    Refuses unless it names spec's repo, a 40-hex revision, and a pin resolved at fetch time for this axis."""
    if not path.is_file():
        raise SystemExit(f"REFUSING: no dataset fetch record at {path}; run src/fetch_dataset_snapshot.py --resolve "
                         f"{axis} first")
    frec = read_result(path)                                   # registered: listed in the record's inputs
    rel = _repo_rel(path)
    rev, pin = frec.get("revision"), frec.get("pin") or {}
    if frec.get("repo") != spec["source"] or not isinstance(rev, str) or not re.fullmatch(r"[0-9a-f]{40}", rev):
        raise SystemExit(f"REFUSING: {rel} is for {frec.get('repo')}@{str(rev)[:12]}, not {spec['source']} at a 40-hex "
                         "commit")
    if pin.get("resolved_at_fetch") is not True or pin.get("axis") != axis or pin.get("metadata_sha") != rev:
        raise SystemExit(f"REFUSING: {rel} is not a fetch of axis {axis} at the commit the Hub resolved (its pin: "
                         f"{json.dumps(pin, sort_keys=True)[:200]})")
    return rev, {f["name"]: f["sha256"] for f in frec["files"]}, rel, _RESULT_READS[rel]


def same_item(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """A pool record and a Phase 1 record of the same row: every field equal but the axis name."""
    return {k: v for k, v in a.items() if k != "axis"} == {k: v for k, v in b.items() if k != "axis"}


# --- main -------------------------------------------------------------------------------------------------------

def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Phase 2 near-duplicate clusters over the pool union Phase 1 (D-303)")
    ap.add_argument("--phase1-manifests", nargs="+", default=[str(p) for p in PHASE1_MANIFESTS])
    ap.add_argument("--frozen-dir", default=str(FROZEN), help="where the Phase 1 split files are")
    ap.add_argument("--split-manifests-dir", default=str(SPLIT_MANIFESTS))
    ap.add_argument("--local-hub", default=None, help="REQUIRED: hub cache holding the pinned snapshot (offline read)")
    ap.add_argument("--fetch-record", default=None,
                    help="the snapshot's fetch record (default results/phase2/fetch_dataset__<org>__<name>@<rev12>.json)")
    ap.add_argument("--datasets-cache-root", default=None,
                    help="an NVMe directory for this run's own HF_DATASETS_CACHE (default $HF_HOME/gccf_datasets_cache)")
    ap.add_argument("--block-rows", type=int, default=BLOCK_ROWS)
    ap.add_argument("--axis", choices=sorted(AXES), default=AXIS, help="p2wg (D-303; the default) or p2sst (D-338)")
    ap.add_argument("--out-manifest", default=None, help="default: the axis's final name")
    ap.add_argument("--out-record", default=None, help="default: the axis's final name")
    ap.add_argument("--test-sizes", default=None, help="TEST ONLY: JSON of the SIZES keys; refused for the final names")
    args = ap.parse_args()
    offline = guard_and_offline("phase2_pool_clusters.py", args.local_hub)
    ax = AXES[args.axis]
    spec = ax["spec"]
    axis_key = {"axis": args.axis} if args.axis != AXIS else {}   # p2wg's outputs keep their fields

    out_m = Path(args.out_manifest if args.out_manifest is not None else ax["out_manifest"]).resolve()
    out_r = Path(args.out_record if args.out_record is not None else ax["out_record"]).resolve()
    sizes = sizes_from(args.test_sizes, ax["sizes"])
    final = (out_m, out_r) == (ax["out_manifest"].resolve(), ax["out_record"].resolve())
    default_inputs = ([Path(p).resolve() for p in args.phase1_manifests] == [p.resolve() for p in PHASE1_MANIFESTS]
                      and Path(args.frozen_dir).resolve() == FROZEN.resolve()
                      and Path(args.split_manifests_dir).resolve() == SPLIT_MANIFESTS.resolve())
    if final and (sizes != ax["sizes"] or not default_inputs):
        raise SystemExit(f"REFUSING the final names: they need {ax['decision']}'s sizes and the Phase 1 inputs of record")
    if not final:
        outside_repo_or_refuse([out_m, out_r], "test run (non-final output names)")
    for p in (out_m, out_r):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists; it is never overwritten")
    code0 = P.code_block(CODE_FILES)

    files, wg, inputs = load_phase1([Path(p) for p in args.phase1_manifests], Path(args.frozen_dir),
                                    Path(args.split_manifests_dir), spec, own_axis=ax["pinned_by_phase1"])
    if ax["pinned_by_phase1"]:                                 # p2wg: the revision the Phase 1 freeze recorded
        rev0 = wg["entry"]["source_revision_sha"]
        if not isinstance(rev0, str) or not re.fullmatch(r"[0-9a-f]{40}", rev0):
            raise SystemExit(f"REFUSING: the pinned source revision {rev0!r} is not a 40-hex commit")
        fpath = Path(args.fetch_record) if args.fetch_record else fetch_record_path(rev0)
        if final and fpath.resolve() != fetch_record_path(rev0).resolve():
            raise SystemExit("REFUSING the final names: they need the fetch record of record")
        fetched, frel, fsha = read_fetch_record(fpath, rev0)
        entry, pinned_by = wg["entry"], wg["manifest"]
    else:                                                      # p2sst: the commit its fetch resolved (D-338 (1))
        fpath = Path(args.fetch_record) if args.fetch_record else default_resolved_fetch_record(spec)
        rev0, fetched, frel, fsha = read_resolved_fetch_record(fpath, spec, args.axis)
        if final and fpath.resolve() != fetch_record_path(rev0, spec).resolve():
            raise SystemExit("REFUSING the final names: they need the fetch record of record")
        entry, pinned_by = {"source_revision_sha": rev0}, frel
    inputs[frel] = fsha
    cache = datasets_cache("phase2_pool_clusters.py", args.datasets_cache_root)
    pool, rev, stats, read = load_pool(entry, args.local_hub, cache, expected_files=fetched, spec=spec, axis=args.axis,
                                       check_recorded=ax["pinned_by_phase1"])
    by_id = {r["item_id"]: r for r in pool}
    for f in files:
        if wg is not None and f["axis"] == wg["axis"]:
            bad = [r["item_id"] for r in f["rows"] if r["item_id"] not in by_id or not same_item(by_id[r["item_id"]], r)]
            if bad:
                raise SystemExit(f"REFUSING: {len(bad)} items of {f['file']} are not pool rows with the same fields "
                                 f"(first {bad[0]})")
    print(f"pool {spec['source']} {spec['config']}@{rev[:12]} (local snapshot): {len(pool)} usable rows "
          f"({sum(r['label'] for r in pool)} {ax['class_names'][1]}); Phase 1 files {len(files)}, "
          f"{sum(len(f['rows']) for f in files)} items", flush=True)

    node_ids = [r["item_id"] for r in pool]
    texts = [r["text"] for r in pool]
    idx = {iid: i for i, iid in enumerate(node_ids)}
    p1_src: dict[int, set[str]] = {}
    for f in files:
        for r in f["rows"]:
            i = idx.get(r["item_id"])
            if i is None:
                i = idx[r["item_id"]] = len(node_ids)
                node_ids.append(r["item_id"])
                texts.append(r["text"])
            elif texts[i] != r["text"]:
                raise SystemExit(f"REFUSING: item_id {r['item_id']} names two different texts")
            p1_src.setdefault(i, set()).add(f["source"])
    n_pool = len(pool)
    lab, summ = blocked_clusters(texts, args.block_rows)
    low: dict[int, str] = {}
    for i, c in enumerate(lab.tolist()):
        low[c] = min(low.get(c, node_ids[i]), node_ids[i])
    cl = [low[c] for c in lab.tolist()]
    cluster_sources: dict[str, set[str]] = {}
    for i, srcs in p1_src.items():
        cluster_sources.setdefault(cl[i], set()).update(srcs)
    cluster_sources_l = {c: sorted(s) for c, s in sorted(cluster_sources.items())}
    excluded = sorted(c for c, s in cluster_sources_l.items() if any(x in EXCLUDED_SOURCES for x in s))
    is_p1 = [int(i in p1_src) for i in range(n_pool)]
    print(f"union {len(node_ids)} nodes ({n_pool} pool, {len(node_ids) - n_pool} Phase 1 only): "
          f"{summ['n_clusters']} clusters (J>={J_MIN}) holding {summ['items_in_clusters']} items; "
          f"{len(cluster_sources_l)} components (singletons included) hold a Phase 1 item, {len(excluded)} an XSTest "
          f"item", flush=True)

    units = units_from(node_ids[:n_pool], [r["label"] for r in pool], cl[:n_pool], is_p1, cluster_sources_l)
    avail = availability(units)
    pl = plan(units, sizes, SPLIT_SEED, ax["splits"])
    status = "SHORTFALL" if pl["shortfall"] else "OK"
    pool_sha = sha256_bytes(split_bytes(pool))
    composition = None
    if ax["containment"]:                                      # D-338 (3): composition, not a split rule
        composition = {"containment": None if pl["shortfall"] else containment(
                           *split_texts(pool, cl[:n_pool], pl["assign"], ax["splits"])),
                       "note": "computed on the plan's splits when the plan is exact; null on a shortfall"}

    dataset = {"source": spec["source"], "config": spec["config"], "split": spec["split"],
               "source_revision_sha": rev, "cleaning": stats, "total_usable_rows": len(pool),
               "loader": "src/freeze_splits.py load_axis (the Phase 1 cleaning and item ids)",
               "pinned_by": pinned_by, "source_was_local_snapshot": True, "snapshot": read["snapshot"],
               "snapshot_files_sha256": read["snapshot_files_sha256"]}
    config = {"jaccard_min": J_MIN, "jaccard": "exact, all pairs, word 3-gram shingles (src/paraphrase_overlap.py), "
                                               "connected components (D-164), all-pairs product in row blocks",
              "block_rows": args.block_rows, "union": "pool rows + every item of every split file the Phase 1 "
                                                      "manifests list; nodes keyed by item_id",
              "cluster_label": "the smallest item_id among the cluster's nodes",
              "sources": SOURCE_NAMES, "excluded_sources": list(EXCLUDED_SOURCES),
              "split_seed": SPLIT_SEED, "fill_rule": ax["fill_rule"], "sizes_registered": sizes == ax["sizes"]}
    if args.axis != AXIS:
        config.update(axis=args.axis, splits=list(ax["splits"]), labels=P2SST_LABELS,
                      pin="the commit the Hub resolved when the snapshot was fetched (its fetch record; D-338 (1)); no "
                          "Phase 1 axis reads this source")
    manifest = {**provenance_block("phase2_pool_clusters"), "decision": ax["decision"], **axis_key, "code": code0,
                "config": config,
                "dataset": dataset,
                "phase1_files": {f["file"]: {"axis": f["axis"], "part": f["part"], "source": f["source"],
                                             "sha256": f["sha256"], "n": len(f["rows"])} for f in files},
                "pool": {"n": n_pool, "order": "load_axis order (source row order)",
                         "item_ids_sha256": sha256_bytes("\n".join(node_ids[:n_pool]).encode("utf-8")),
                         "item_id": node_ids[:n_pool], "label": [int(r["label"]) for r in pool],
                         "source_row": [int(r["source_row"]) for r in pool], "phase1_item": is_p1,
                         "cluster": cl[:n_pool]},
                "phase1_only": {"n": len(node_ids) - n_pool, "item_id": node_ids[n_pool:], "cluster": cl[n_pool:]},
                "phase1_clusters": sorted(cluster_sources_l), "phase1_cluster_sources": cluster_sources_l,
                "excluded_clusters": excluded, "summary": summ}
    mdata = json.dumps(manifest, separators=(",", ":"), default=str).encode("utf-8")
    record = {**provenance_block("phase2_pool_clusters"), "decision": ax["decision"], **axis_key, "code": code0,
              "config": config, "required_sizes": sizes, "size_sources": ax["size_sources"],
              "dataset": {**dataset, "local_snapshot_path": read["local_snapshot_path"], "libraries": read["libraries"],
                          "offline_env": offline, "datasets_cache": read["datasets_cache"],
                          "fetch_record": {"file": frel, "sha256": fsha,
                                           "check": "every snapshot file's sha256 equals the fetch record's"}},
              "notes": [DATASETS_CACHE_NOTE],
              "inputs": inputs,
              "outputs": {"manifest_file": _repo_rel(out_m), "manifest_sha256": sha256_bytes(mdata)},
              "pool_split_bytes_sha256": pool_sha, "plan_assign_sha256": pl["assign_sha256"],
              "clustering": {**summ, "n_pool_nodes": n_pool, "n_phase1_only_nodes": len(node_ids) - n_pool,
                             "n_phase1_items": sum(len(f["rows"]) for f in files), "n_phase1_nodes": len(p1_src),
                             "n_components_holding_phase1_singletons_included": len(cluster_sources_l),
                             "n_components_holding_xstest_singletons_included": len(excluded)},
              "availability": avail, "plan": plan_record(pl), "status": status, "shortfall": pl["shortfall"]}
    if composition is not None:
        record["composition"] = composition
    rdata = record_bytes(record, out_r.name)
    if P.code_block(CODE_FILES)["sha256"] != code0["sha256"]:
        raise SystemExit("REFUSING to write: source changed during the run")
    sha = write_all([(out_m, mdata), (out_r, rdata)])
    print(f"availability: clean {avail['clean_units']['pos']} pos / {avail['clean_units']['neg']} neg; Phase 1 "
          f"clusters (probe_train only) {avail['phase1_cluster_units_probe_train_only']['pos']} pos / "
          f"{avail['phase1_cluster_units_probe_train_only']['neg']} neg; XSTest clusters (excluded) "
          f"{avail['xstest_cluster_units_excluded']['pos']} pos / {avail['xstest_cluster_units_excluded']['neg']} neg")
    for s in ax["splits"]:
        g, t = pl["splits"][s], pl["targets"][s]
        print(f"  {s:<12} pos {g['pos']:>6} / {t['pos']:<6} neg {g['neg']:>6} / {t['neg']:<6} units {g['units']}")
    if composition is not None and composition["containment"] is not None:
        for kind in ("substring", "token_aligned"):
            c = composition["containment"][kind]
            print(f"  containment ({kind}): in some probe_train item: calibration "
                  f"{c['in_probe_train']['calibration']['contained']} / {c['in_probe_train']['calibration']['n']}, "
                  f"eval {c['in_probe_train']['eval']['contained']} / {c['in_probe_train']['eval']['n']}; "
                  f"calibration-eval pairs {c['calibration_vs_eval']['pairs']}")
    print(f"Written: {_repo_rel(out_m)}  sha256 {sha[out_m]}\nWritten: {_repo_rel(out_r)}  sha256 {sha[out_r]}")
    if pl["shortfall"]:
        print("SHORTFALL (the author decides; nothing was shrunk):\n  " + "\n  ".join(pl["shortfall"]))
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
