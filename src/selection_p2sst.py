"""Phase 2 D-117 selection for the SST-2 non-safety control (D-338 (2), (4); D-340 (3)): L* for SST-2, chosen for the
LoRA base on SST-2's probe_train ho20 by src/selection_p2.py's own computation, with the axis this module supplies.

src/selection_p2.py is imported and NOT changed. Its sha256 is bound three ways: by the selection anchor record, by
src/selection_tpr_v2.py's BOUND_FILES (each selection record's src/selection_p2.py must be the present file) and by
src/selection_ho20_cluster.py's COMPUTE_FILES. Changing it would make every existing selection record unusable to those
two modules. So this module follows their rule (D-324 (7), D-337: import selection_p2, never change it) and adds only
the reading of the p2sst caches and the output name.

  anchor   selection_p2.passing_anchor, given this run's sha256 of exactly selection_p2.CODE_FILES: an anchor record
           written by selection_p2's code with every one of those files at its present sha256 (the computation,
           selection_p2.sweep, is that code), with the same numpy, scipy and scikit-learn, BLAS threads and host facts.
           This module's own file is outside that comparison (it computes nothing; it is in the record's code block),
           the rule of selection_ho20_cluster's COMPUTE_FILES. No new anchor is needed while selection_p2's files are
           unchanged. Seeds, holdout and target FPR are the anchor record's (D-117's 0-4, 0.20, 0.01).
  inputs   the model's p2sst select record, results/phase2/extract__<org>__<model>__p2sst@<rev12>__select.json (exactly
           one: extract_p2's own tag for SST-2, cache_repo '<repo>__p2sst'), ingest yes, the full every-second-layer
           grid, at the D-303 (8) revision (selection_p2.check_revision). Its p2sst/probe_train and p2sst/calibration
           caches for the pooling are read by load_ingested below. That function runs selection_p2.load_ingested's
           checks for the split key '<axis>/<part>/<pooling>': data/hidden only, a regular file, mode 444, in the ingest
           index with the record's sha256 and hashing to it, the record's layers, n items, no non-finite item, and item
           for item the frozen split's ids and labels. On p2sst it also requires every split row to name axis p2sst.
           tests/test_selection_p2sst.py proves that load_ingested(axis='p2wg') returns exactly what
           selection_p2.load_ingested returns.
  selection  selection_p2.sweep on SST-2's probe_train. Label 1 (positive sentiment) is the probe's positive class
           (D-338 (2)). The threshold behind the reported ho20 TPR is set on the calibration split's label-0 items.
  record   results/phase2/selection_p2sst__<org>__<model>@<rev12>__<pooling>.json (run_kind selection_p2sst), written
           once. It has selection_p2's record form, with config.axis 'p2sst' and the anchor rule above. The name prefix
           is its own so that no p2wg reader's glob meets it: selection_ho20_cluster globs
           'selection_p2__<model>@*__<pooling>.json' and, in its summary, 'selection_p2__*.json'. src/subset_p2.py reads
           this record for the SST-2 L* (--source-tag p2sst).
Readers key SST-2 caches by axis and split, never by a bare item_id: an SST-2 item_id can equal a WildGuardMix one
(D-340 (3)). CPU only. The BLAS threads must be pinned at 6 (selection_p2.check_threads).

    python3 src/selection_p2sst.py --repo mistralai/Mistral-7B-Instruct-v0.2 --pooling usermean
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as P  # noqa: E402
import selection_p2 as S  # noqa: E402  (imported, never changed: its sha256 is bound, module docstring)
from provenance import ensure_utf8_stdout, provenance_block, read_result, sha256_file, write_results  # noqa: E402

CODE_FILES = ["src/selection_p2sst.py", *S.CODE_FILES]
AXIS = TAG = "p2sst"                        # the axis, and extract_p2's tag for it (extract_p2.SST_TAG)
RUN_KIND = "selection_p2sst"
refuse = S.refuse
ANCHOR_RULE = ("selection_p2.passing_anchor given this run's sha256 of exactly selection_p2.CODE_FILES: an anchor record "
               "written by that code at its present bytes, with these libraries, BLAS threads and host facts; this "
               "module (which reads the p2sst caches and computes nothing) is outside the comparison, as in "
               "selection_ho20_cluster's COMPUTE_FILES")


def out_name(repo: str, rev: str, pooling: str) -> str:
    return f"selection_p2sst__{repo.replace('/', '__')}@{rev[:12]}__{pooling}.json"


def select_record(repo: str) -> Path:
    found = sorted(S.P2.glob(f"extract__{repo.replace('/', '__')}__{TAG}@*__select.json"))
    if len(found) != 1:
        refuse(f"{len(found)} p2sst select records for {repo} under {S.rel(S.P2)} (want exactly one): "
               f"{[p.name for p in found]}")
    return found[0]


def load_ingested(rec: dict[str, Any], rec_rel: str, index: dict[str, str], part: str, pooling: str,
                  axis: str = AXIS) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    """selection_p2.load_ingested for the split key '<axis>/<part>/<pooling>' (its checks, in its order, on its module
    paths), plus, off p2wg, every split row naming the axis."""
    key = f"{axis}/{part}/{pooling}"
    s = (rec.get("splits") or {}).get(key)
    if not isinstance(s, dict):
        refuse(f"{rec_rel} has no split {key}")
    name, want = s.get("cache_file"), s.get("cache_sha256")
    if not isinstance(name, str) or "/" in name or "\\" in name or not isinstance(want, str) or len(want) != 64:
        refuse(f"{rec_rel} split {key}: malformed cache_file or cache_sha256")
    if s.get("n_nonfinite") != 0:
        refuse(f"{name}: the record counts {s.get('n_nonfinite')} non-finite items (primary inputs must be finite)")
    if index.get(name) != want:
        refuse(f"{name} is not in the ingest index with the record's sha256 (index {str(index.get(name))[:16]})")
    p = S.CACHE / name
    if p.is_symlink() or not p.is_file():
        refuse(f"{name} is not a regular file in {S.rel(S.CACHE)}: ingest the select record first (D-307 (1)); NVMe "
               "scratch is never read for a result")
    if (p.stat().st_mode & 0o777) != 0o444:
        refuse(f"{name} is mode {oct(p.stat().st_mode & 0o777)}, not 0o444 (an ingested cache is read-only)")
    have = sha256_file(p)
    if have != want:
        refuse(f"{name} is not the bytes {rec_rel} names ({have[:16]} vs {want[:16]})")
    z = np.load(p, allow_pickle=False)
    layers = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("layer_"))
    if layers != list(rec.get("layers_resolved") or []):
        refuse(f"{name} holds layers {layers}, not the record's {rec.get('layers_resolved')}")
    if "nonfinite" not in z.files or bool(np.asarray(z["nonfinite"]).any()):
        refuse(f"{name} has no nonfinite flag, or flags a non-finite item")
    n = len(z["labels"])
    if n != s.get("n"):
        refuse(f"{name} holds {n} items, the record says {s.get('n')}")
    rows = S.split_rows(s, key)
    if [r["item_id"] for r in rows] != [str(x) for x in z["item_ids"]] \
            or [int(r["label"]) for r in rows] != [int(x) for x in z["labels"]]:
        refuse(f"{name}: item ids or labels are not the frozen split's, item for item")
    if axis != S.AXIS and any(r.get("axis") != axis for r in rows):
        refuse(f"{s['split_file']}: a row does not name axis {axis} (D-340 (3): items are keyed by axis and split)")
    return z, {"file": name, "sha256": have, "expected_from": rec_rel,
               "split": {"file": s["split_file"], "sha256": s["split_sha256"]}}, rows


def run(repo: str, pooling: str) -> Path:
    if pooling not in S.POOLINGS:
        refuse(f"pooling must be one of {S.POOLINGS}")
    threads = S.check_threads()
    code0 = P.code_block(CODE_FILES)
    compute = {"sha256": {f: code0["sha256"][f] for f in S.CODE_FILES}, "libraries": code0["libraries"]}
    anchor, anchors_read = S.passing_anchor(compute, threads)
    seeds, holdout, target_fpr = anchor["config"]["seeds"], anchor["config"]["holdout"], anchor["config"]["target_fpr"]
    rec_path = select_record(repo)
    rec_rel = S.rel(rec_path)
    rec = read_result(rec_path)
    if rec.get("repo") != repo or rec.get("set_name") != "select":
        refuse(f"{rec_rel} names repo {rec.get('repo')!r}, set {rec.get('set_name')!r}")
    if rec.get("ingest") != "yes":
        refuse(f"{rec_rel} says ingest: {rec.get('ingest')!r}; a result reads only ingested caches (D-307 (1))")
    if rec.get("layers_every_second_grid") is not True:
        refuse(f"{rec_rel} does not hold the full every-second-layer grid the selection runs over (D-303 (1))")
    rev = rec.get("revision")
    if not isinstance(rev, str) or len(rev) != 40 or rec.get("cache_revision") != rev \
            or rec.get("cache_repo") != f"{repo}__{TAG}":
        refuse(f"{rec_rel}: revision, cache_revision or cache_repo is not the documented-tag form for {TAG}")
    cc = read_result(S.CONFIG_CHECK)
    rev_block = S.check_revision(repo, rev, cc)
    out = S.P2 / out_name(repo, rev, pooling)
    if out.exists():
        refuse(f"{S.rel(out)} exists; results are never overwritten (D-152 P5)")
    index_path, index = S.ingest_index(rec_path)

    z_tr, r_tr, rows_tr = load_ingested(rec, rec_rel, index, "probe_train", pooling)
    z_cal, r_cal, _ = load_ingested(rec, rec_rel, index, "calibration", pooling)
    y_tr, y_cal = z_tr["labels"].astype(bool), z_cal["labels"].astype(bool)
    if y_tr.all() or not y_tr.any():
        refuse("probe_train holds one class only")
    if y_cal.all():
        refuse("calibration holds no negative")
    comp = S.composition(rows_tr)
    print(f"{repo}@{rev[:12]} [{AXIS}, {pooling}]: probe_train {len(y_tr)} ({int(y_tr.sum())} positive), calibration "
          f"{len(y_cal)} ({int((~y_cal).sum())} negative); selection on ho20 only, eval not read; probe_train "
          f"composition {comp}", flush=True)
    m = S.sweep(z_tr, z_cal, seeds, holdout, target_fpr)
    S.print_summary(pooling, m)

    if P.code_block(CODE_FILES)["sha256"] != code0["sha256"]:
        refuse("source changed during the run")
    inputs = {f"pre/{AXIS}/probe_train/{pooling}": r_tr, f"pre/{AXIS}/calibration/{pooling}": r_cal}
    digest = write_results(out, {
        **provenance_block(RUN_KIND),
        "preregistration": "D-117 (the protocol, unchanged: src/selection_p2.py's sweep, itself src/regularisation_"
                           "sweep.py's functions, anchored bit for bit); D-338 (2), (4): L* of the SST-2 non-safety "
                           "control, selected on SST-2's probe_train ho20 by the same rule (label 1 the positive class; "
                           "the threshold on label-0 calibration items); D-303 (d): one selection per pooling; D-340 "
                           "(3): read by axis and split",
        "code": code0,
        "config": {"pre_repo": repo, "pre_rev": rev, "axis": AXIS, "pooling": pooling, "C_grid": S.RS.C_GRID,
                   "seeds": seeds, "holdout": holdout, "target_fpr": target_fpr, "blas_threads": threads,
                   "settings_from": "the anchor record's config (the Phase 1 selection record's)",
                   "selection_population": "held-out 20% of probe_train, per seed (item-level permutation, as Phase 1)",
                   "eval_used_for_selection": False,
                   "threshold_population": "the calibration split's negatives (label 0), for the reported ho20 TPR only",
                   "inputs_from": "the model's p2sst select record; caches read from data/hidden (ingested, indexed)"},
        "anchor": anchor, "anchor_rule": ANCHOR_RULE, "anchors_read": anchors_read,
        "revision_check": rev_block,
        "extraction_record": {"file": rec_rel, "sha256": sha256_file(rec_path)},
        "config_check": {"file": S.rel(S.CONFIG_CHECK), "sha256": sha256_file(S.CONFIG_CHECK)},
        "ingest_index": {"file": S.rel(index_path), "sha256": sha256_file(index_path)},
        "input_cache_sha256": {k: r["sha256"] for k, r in inputs.items()},
        "input_cache_source": {k: f"{r['file']} (expected bytes from {r['expected_from']})" for k, r in inputs.items()},
        "split_files": {k: r["split"] for k, r in inputs.items()},
        "n_calibration": int(len(y_cal)), "n_calibration_negative": int((~y_cal).sum()),
        "n_probe_train_positive": int(y_tr.sum()),
        "probe_train_composition": comp,
        "model": {"repo": repo, "revision": rev, "pooling": pooling, **m},
    }, exclusive=True)
    print(f"Written: {S.rel(out)}\nsha256:  {digest}")
    return out


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Phase 2 D-117 selection on SST-2 (one model, one pooling), through "
                                             "selection_p2's anchored computation")
    ap.add_argument("--repo", required=True, help="the pre-side model (the LoRA base, D-338 (4))")
    ap.add_argument("--pooling", required=True, choices=S.POOLINGS)
    args = ap.parse_args()
    run(args.repo, args.pooling)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
