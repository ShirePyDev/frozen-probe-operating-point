"""Phase 2 data, step 2 (D-303): write the cluster-level frozen splits of the WildGuardMix pool. Runs ONCE.

Reads the cluster manifest and its record from step 1 (src/phase2_pool_clusters.py). The record is read through
provenance.read_result and lists the manifest's sha256. Refuses unless:
  - the record's code sha256 of src/phase2_pool_clusters.py equals the current file's (the rule that made the plan is
    the rule recomputing it);
  - the pool, re-read through the same loader (src/freeze_splits.py load_axis) offline from the local snapshot of the
    revision the manifest recorded, has the manifest's item ids, labels and source rows, the recorded cleaning
    counts, and the recorded sha256 of its bytes (phase2_pool_clusters.split_bytes over the pool);
  - the plan recomputed with the shared rule (phase2_pool_clusters.plan, its seed and the recorded sizes) has the
    recorded assignment sha256 and equals the recorded plan exactly.
Then it writes:

  data/frozen_p2/p2wg__eval.jsonl, p2wg__calibration.jsonl, p2wg__pool4d.jsonl, p2wg__probe_train.jsonl
      one record per item, with freeze_splits' fields (item_id = sha256 of the stripped text, first 16 hex;
      source_row = the row in the pinned source) plus "cluster" (the cluster label of the manifest); sorted by
      item_id, LF endings, sorted keys, as freeze_splits writes them. A directory of their own, not data/frozen/:
      Phase 2 outputs never change a Phase 1 reader's inputs (D-303; several Phase 1 scripts read every
      data/frozen/*.jsonl, e.g. src/train_lora.py's exclusion set)
  manifests/splits_p2/p2wg__<split>__manifest.json
      the per-split index manifests (item ids, labels, source rows and the split file's sha256; no prompt text), in
      the format src/write_split_manifests.py writes to manifests/splits/ (json.dumps(indent=1) plus a newline), so
      the release can carry them
  results/phase2/dataset_manifest_p2.json
      each split's sha256, counts by class, cluster counts and Phase 1 item counts, in the shape of
      results/dataset_manifest*.json (under results/phase2/, so no Phase 1 reader's glob sees it), and each index
      manifest's sha256

Checked before anything is written: whole clusters only (no cluster in two splits); a cluster holding a Phase 1 item
only in probe_train (D-303 (2)); a cluster holding an XSTest item in no split; every split at its exact target.
ALL OR NOTHING, NOTHING DELETED: all nine files are built and the record is checked by the D-297 (1) rule (as
provenance.write_results(path, payload) checks it) and serialised as write_results writes it; all are staged, then
committed, by src/safe_outputs.py's write_all. On a failure no file is deleted: final-named outputs of this call are
renamed aside to '<name>.failed.<pid>', and the exit names every file left (removal is a human's scripts/safe_rm.sh
call). Any existing p2wg__* file in data/frozen_p2/ or manifests/splits_p2/, or an existing record, makes the
run refuse (CLAUDE.md rule 7). If step 1 recorded a shortfall, this exits 3 and writes nothing; the author decides.

AXIS p2sst (D-338; --axis p2sst): the same step for the SST-2 control, from step 1's p2sst manifest and record
(manifests/phase2_pool_clusters_p2sst.json, results/phase2/pool_clusters_p2sst.json; both must say axis p2sst, and a
p2wg record, which names no axis, is refused here as a p2wg run refuses a p2sst one). It writes
data/frozen_p2/p2sst__{eval,calibration,probe_train}.jsonl (no pool4d, D-338 (3)), manifests/splits_p2/p2sst__<split>__
manifest.json and its own record results/phase2/dataset_manifest_p2sst.json, so no p2wg file is read as an output or
touched (the clash check covers p2sst__* only). The containment composition of D-338 (3) is recomputed on the rows
about to be written, must equal the one step 1 recorded, and is written under axes.p2sst.composition.

ENVIRONMENT: hf_env_guard.assert_hf_env runs first; --local-hub is required (no Hub path); the loader runs offline.
  python3 src/phase2_freeze_splits.py --local-hub <hub dir with the pinned snapshot>
  python3 src/phase2_freeze_splits.py --axis p2sst --local-hub <hub dir with the GLUE snapshot>      (D-338)
The defaults are the final names. Any other output location must lie outside the repository (a test).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as P  # noqa: E402
import phase2_pool_clusters as PC  # noqa: E402
from provenance import _RESULT_READS, _repo_rel, ensure_utf8_stdout, provenance_block, read_result, sha256_bytes, sha256_file  # noqa: E402

ROOT = PC.ROOT
FROZEN_P2 = ROOT / "data" / "frozen_p2"
SPLIT_MANIFESTS_P2 = ROOT / "manifests" / "splits_p2"
OUT = PC.RESULTS / "phase2" / "dataset_manifest_p2.json"
CODE_FILES = ["src/phase2_freeze_splits.py", *PC.CODE_FILES]
AXIS, SPLITS, SPEC = PC.AXIS, PC.SPLITS, PC.SPEC
INDEX_NOTE = ("Prompt text is not redistributed. Rebuild with src/phase2_pool_clusters.py then "
              "src/phase2_freeze_splits.py against the dataset revision pinned in "
              "results/phase2/dataset_manifest_p2.json, then compare sha256.")
OUT_P2SST = PC.RESULTS / "phase2" / "dataset_manifest_p2sst.json"            # D-338: p2sst's own record
INDEX_NOTE_P2SST = ("Prompt text is not redistributed. Rebuild with src/phase2_pool_clusters.py --axis p2sst then "
                    "src/phase2_freeze_splits.py --axis p2sst against the dataset revision pinned in "
                    "results/phase2/dataset_manifest_p2sst.json, then compare sha256.")
OUTS = {"p2wg": (OUT, INDEX_NOTE), "p2sst": (OUT_P2SST, INDEX_NOTE_P2SST)}


def index_manifest_bytes(name: str, rows: list[dict[str, Any]], digest: str, note: str = INDEX_NOTE) -> bytes:
    """The index manifest of one split file, in write_split_manifests.manifest_from_split's format and key order
    (split_file, n, sha256_of_original, note, items of item_id, label, source_row, meta_type), written as
    write_split_manifests writes it: json.dumps(indent=1) plus a newline. The note names the Phase 2 rebuild."""
    items = [{k: r[k] for k in ("item_id", "label", "source_row") if k in r}
             | ({"meta_type": r["meta_type"]} if "meta_type" in r else {}) for r in rows]
    man = {"split_file": name, "n": len(rows), "sha256_of_original": digest, "note": note, "items": items}
    return (json.dumps(man, indent=1) + "\n").encode("utf-8")


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Phase 2 cluster-level frozen splits (D-303); runs once")
    ap.add_argument("--axis", choices=sorted(PC.AXES), default=PC.AXIS,
                    help="p2wg (D-303; the default) or p2sst (D-338)")
    ap.add_argument("--clusters-manifest", default=None, help="default: step 1's manifest for the axis")
    ap.add_argument("--clusters-record", default=None, help="default: step 1's record for the axis")
    ap.add_argument("--local-hub", default=None, help="REQUIRED: hub cache holding the pinned snapshot (offline read)")
    ap.add_argument("--datasets-cache-root", default=None,
                    help="an NVMe directory for this run's own HF_DATASETS_CACHE (default $HF_HOME/gccf_datasets_cache)")
    ap.add_argument("--frozen-dir", default=str(FROZEN_P2), help="where the axis's split files are written")
    ap.add_argument("--split-manifests-dir", default=str(SPLIT_MANIFESTS_P2),
                    help="where the per-split index manifests are written")
    ap.add_argument("--out", default=None, help="default: the axis's record (dataset_manifest_p2.json for p2wg, "
                                                "dataset_manifest_p2sst.json for p2sst)")
    args = ap.parse_args()
    offline = PC.guard_and_offline("phase2_freeze_splits.py", args.local_hub)
    ax = PC.AXES[args.axis]
    axis, splits, spec = args.axis, ax["splits"], ax["spec"]
    out_final, note = OUTS[axis]
    axis_key = {"axis": axis} if axis != PC.AXIS else {}      # p2wg's record keeps its fields

    frozen_dir, out = Path(args.frozen_dir).resolve(), Path(args.out if args.out is not None else out_final).resolve()
    index_dir = Path(args.split_manifests_dir).resolve()
    mpath = Path(args.clusters_manifest if args.clusters_manifest is not None else ax["out_manifest"])
    rpath = Path(args.clusters_record if args.clusters_record is not None else ax["out_record"])
    final = (frozen_dir, index_dir, out) == (FROZEN_P2.resolve(), SPLIT_MANIFESTS_P2.resolve(), out_final.resolve())
    if final and (mpath.resolve(), rpath.resolve()) != (ax["out_manifest"].resolve(), ax["out_record"].resolve()):
        raise SystemExit("REFUSING the final names: they need the cluster manifest and record of record")
    if not final:
        PC.outside_repo_or_refuse([frozen_dir, index_dir, out], "test run (non-final output names)")
    phase1_dirs = (PC.FROZEN.resolve(), PC.SPLIT_MANIFESTS.resolve())
    if frozen_dir in phase1_dirs or index_dir in phase1_dirs:
        raise SystemExit("REFUSING: Phase 2 outputs never go into Phase 1's data/frozen/ or manifests/splits/")
    paths = {s: frozen_dir / f"{axis}__{s}.jsonl" for s in splits}
    index_paths = {s: index_dir / f"{axis}__{s}__manifest.json" for s in splits}
    clash = sorted(frozen_dir.glob(f"{axis}__*")) + sorted(index_dir.glob(f"{axis}__*")) + \
        ([out] if out.exists() else [])
    if clash:
        print("REFUSING TO OVERWRITE (CLAUDE.md rule 7: a split is written once). These exist:")
        for p in clash:
            print(f"  {p}  sha256={sha256_file(p) if p.is_file() else '(not a file)'}")
        raise SystemExit("REFUSING: outputs exist")
    code0 = P.code_block(CODE_FILES)

    mb = mpath.read_bytes()
    msha = sha256_bytes(mb)
    man = json.loads(mb.decode("utf-8"))  # not a result (D-297): the cluster manifest, hashed and listed
    rec = read_result(rpath)                                   # registered: listed in the record's inputs
    rrel = _repo_rel(rpath)
    pc_file = "src/phase2_pool_clusters.py"
    if rec["code"]["sha256"].get(pc_file) != code0["sha256"][pc_file]:
        raise SystemExit(f"REFUSING: {rrel} was made by another {pc_file} (recorded "
                         f"{str(rec['code']['sha256'].get(pc_file))[:16]}, now {code0['sha256'][pc_file][:16]})")
    if rec["outputs"]["manifest_sha256"] != msha:
        raise SystemExit(f"REFUSING: {mpath.name} sha256 {msha[:16]} is not the manifest {rrel} records")
    if rec["config"]["split_seed"] != PC.SPLIT_SEED or man["config"]["split_seed"] != PC.SPLIT_SEED:
        raise SystemExit("REFUSING: the recorded split seed is not this code's")
    if rec.get("axis", PC.AXIS) != axis or man.get("axis", PC.AXIS) != axis:
        raise SystemExit(f"REFUSING: {rrel} and {mpath.name} are step 1's output for axis {rec.get('axis', PC.AXIS)} / "
                         f"{man.get('axis', PC.AXIS)}, not {axis}")
    sizes = rec["required_sizes"]
    if final and (sizes != ax["sizes"] or not rec["config"]["sizes_registered"]):
        raise SystemExit("REFUSING the final names: the cluster record was made with test sizes")
    if rec["status"] != "OK":
        print(f"{rrel} records a SHORTFALL; nothing is written (the author decides):\n  " + "\n  ".join(rec["shortfall"]))
        return 3

    ds = man["dataset"]
    if (ds["source"], ds["config"], ds["split"]) != (spec["source"], spec["config"], spec["split"]):
        raise SystemExit(f"REFUSING: the manifest's dataset is not axis {axis}'s source ({spec['source']} "
                         f"{spec['config']})")
    cache = PC.datasets_cache("phase2_freeze_splits.py", args.datasets_cache_root)
    pool, rev, stats, read = PC.load_pool(ds, args.local_hub, cache, expected_files=ds["snapshot_files_sha256"],
                                          spec=spec, axis=axis)
    if read["snapshot_files_sha256"] != ds["snapshot_files_sha256"]:
        raise SystemExit("REFUSING: the local snapshot's files differ from the ones step 1 read")
    mp = man["pool"]
    ids = [r["item_id"] for r in pool]
    if (ids != mp["item_id"] or [int(r["label"]) for r in pool] != mp["label"]
            or [int(r["source_row"]) for r in pool] != mp["source_row"]
            or sha256_bytes("\n".join(ids).encode("utf-8")) != mp["item_ids_sha256"]):
        raise SystemExit("REFUSING: the pool read now differs from the manifest's (ids, labels or source rows)")
    if sha256_bytes(PC.split_bytes(pool)) != rec["pool_split_bytes_sha256"]:
        raise SystemExit("REFUSING: the pool's bytes differ from the ones step 1 recorded")

    units = PC.units_from(mp["item_id"], mp["label"], mp["cluster"], mp["phase1_item"], man["phase1_cluster_sources"])
    pl = PC.plan(units, sizes, PC.SPLIT_SEED, splits)
    if PC.assign_sha256(pl["assign"]) != rec["plan_assign_sha256"]:
        raise SystemExit(f"REFUSING: the recomputed assignment's sha256 differs from the one {rrel} records")
    if json.dumps(PC.plan_record(pl), sort_keys=True) != json.dumps(rec["plan"], sort_keys=True):
        raise SystemExit(f"REFUSING: the recomputed plan differs from the plan {rrel} records")
    if pl["shortfall"]:
        print("SHORTFALL; nothing is written:\n  " + "\n  ".join(pl["shortfall"]))
        return 3

    mixed = {u["cluster"] for u in units if u["pos"] and u["neg"]}     # clusters holding both classes in the pool
    rows: dict[str, list[dict[str, Any]]] = {s: [] for s in splits}
    for r, c in zip(pool, mp["cluster"]):
        s = pl["assign"].get(c)
        if s is not None:
            rows[s].append({**r, "cluster": c, "mixed_label_cluster": c in mixed})
    # the checks the record states, on the rows about to be written
    where: dict[str, set[str]] = {}
    for s, rs in rows.items():
        for r in rs:
            where.setdefault(r["cluster"], set()).add(s)
    if any(len(v) > 1 for v in where.values()):
        raise SystemExit("REFUSING: a cluster straddles two splits")
    pool_size = Counter(mp["cluster"])
    in_split = Counter(r["cluster"] for rs in rows.values() for r in rs)
    if any(in_split[c] != pool_size[c] for c in in_split):
        raise SystemExit("REFUSING: a cluster entered a split only in part")
    p1 = set(man["phase1_clusters"])
    if any(r["cluster"] in p1 for s, rs in rows.items() if s != "probe_train" for r in rs):
        raise SystemExit("REFUSING: a cluster holding a Phase 1 item is outside probe_train (D-303 (2))")
    excluded = set(man["excluded_clusters"])
    if any(c in excluded for c in in_split):
        raise SystemExit("REFUSING: a cluster holding an XSTest item is in a split")
    for s in splits:
        npos = sum(int(r["label"]) == 1 for r in rows[s])
        if (npos, len(rows[s]) - npos) != (pl["targets"][s]["pos"], pl["targets"][s]["neg"]):
            raise SystemExit(f"REFUSING: {s} is not at its exact target")
    if any(r["mixed_label_cluster"] for r in rows["calibration"]):
        raise SystemExit("REFUSING: calibration holds an item of a mixed-label cluster")
    for s, key in (("eval", "eval_from_mixed_label_clusters"), ("pool4d", "pool4d_from_mixed_label_clusters")):
        if s not in splits:
            continue
        m = [r for r in rows[s] if r["mixed_label_cluster"]]
        got_m = {"units": len({r["cluster"] for r in m}), "pos": sum(1 for r in m if int(r["label"]) == 1),
                 "neg": sum(1 for r in m if int(r["label"]) == 0)}
        if got_m != {k: pl[key][k] for k in ("units", "pos", "neg")}:
            raise SystemExit(f"REFUSING: {s}'s units and items from mixed-label clusters {got_m} differ from the plan's")
    bad = PC.mixed_preference_violations(units, pl["assign"])
    if bad:
        raise SystemExit("REFUSING: " + "; ".join(bad))
    composition = None
    if ax["containment"]:                                      # D-338 (3): recomputed on the rows to be written
        got_c = PC.containment({s: [r["text"] for r in rows[s]] for s in splits},
                               {s: [int(r["label"]) for r in rows[s]] for s in splits})
        want_c = (rec.get("composition") or {}).get("containment")
        if want_c is None or json.dumps(got_c, sort_keys=True) != json.dumps(want_c, sort_keys=True):
            raise SystemExit(f"REFUSING: the containment of the rows to be written differs from the one {rrel} records")
        composition = rec["composition"]

    ordered = {s: sorted(rows[s], key=lambda r: r["item_id"]) for s in splits}
    data = {s: PC.split_bytes(ordered[s]) for s in splits}
    sha = {s: sha256_bytes(data[s]) for s in splits}
    index_data = {s: index_manifest_bytes(paths[s].name, ordered[s], sha[s], note) for s in splits}
    p1_ids = {iid for iid, f in zip(mp["item_id"], mp["phase1_item"]) if f}
    partitions = {}
    for s in splits:
        rs = rows[s]
        npos = sum(int(r["label"]) == 1 for r in rs)
        cl = Counter(r["cluster"] for r in rs)
        partitions[s] = {"file": _repo_rel(paths[s]), "sha256": sha[s], "n": len(rs), "n_pos": npos,
                         "n_neg": len(rs) - npos, "target_pos": pl["targets"][s]["pos"],
                         "target_neg": pl["targets"][s]["neg"], "n_clusters": len(cl),
                         "n_multi_item_clusters": sum(1 for v in cl.values() if v >= 2),
                         "n_items_in_multi_item_clusters": sum(v for v in cl.values() if v >= 2),
                         "n_clusters_holding_phase1": sum(1 for c in cl if c in p1),
                         "n_phase1_items": sum(1 for r in rs if r["item_id"] in p1_ids),
                         "n_from_mixed_label_clusters": {
                             "pos": sum(1 for r in rs if r["mixed_label_cluster"] and int(r["label"]) == 1),
                             "neg": sum(1 for r in rs if r["mixed_label_cluster"] and int(r["label"]) == 0)},
                         "index_manifest": {"file": _repo_rel(index_paths[s]),
                                            "sha256": sha256_bytes(index_data[s])}}
    payload = {**provenance_block("phase2_freeze_splits"), "decision": ax["decision"], **axis_key, "code": code0,
               "seed": PC.SPLIT_SEED,
               "config": {"fill_rule": ax["fill_rule"], "sizes": sizes,
                          "sizes_registered": rec["config"]["sizes_registered"],
                          "record_fields": "freeze_splits load_axis fields plus 'cluster' (the manifest's label) and "
                                           "'mixed_label_cluster' (the cluster holds both classes in the pool; for an "
                                           "eval negative: it came with a positive of a mixed cluster)",
                          "split_dir": "data/frozen_p2/ (not data/frozen/: Phase 2 outputs never change a Phase 1 "
                                       "reader's inputs, D-303)",
                          "index_manifests": "manifests/splits_p2/, in src/write_split_manifests.py's format"},
               "inputs": {_repo_rel(mpath): msha, rrel: _RESULT_READS[rrel]},
               "dataset_read": {"source_was_local_snapshot": True, "snapshot": read["snapshot"],
                                "local_snapshot_path": read["local_snapshot_path"],
                                "snapshot_files_sha256": read["snapshot_files_sha256"],
                                "libraries": read["libraries"], "offline_env": offline,
                                "datasets_cache": read["datasets_cache"]},
               "notes": [PC.DATASETS_CACHE_NOTE],
               "checks": {"no_cluster_in_two_splits": True, "whole_clusters_only": True,
                          "phase1_clusters_only_in_probe_train": True, "xstest_clusters_in_no_split": True,
                          "every_split_at_its_exact_target": True, "plan_equals_recorded_plan": True,
                          "assign_sha256_equals_recorded": pl["assign_sha256"],
                          "pool_split_bytes_sha256_equals_recorded": rec["pool_split_bytes_sha256"],
                          "phase2_pool_clusters_code_equals_recorded": code0["sha256"][pc_file]},
               "axes": {axis: {"source": spec["source"], "source_revision_sha": rev, "config": spec["config"],
                               "split": spec["split"], "cleaning": stats, "total_usable_rows": len(pool),
                               "partitions": partitions}},
               "T": pl["T"], "unused": pl["unused"]}
    if axis != PC.AXIS:                                        # p2sst (D-338); p2wg's record keeps its fields
        payload["config"].update(axis=axis, splits=list(splits), labels=PC.P2SST_LABELS,
                                 dataset_manifest="results/phase2/dataset_manifest_p2sst.json (p2sst's own record; "
                                                  "p2wg's dataset_manifest_p2.json is never touched)")
        payload["checks"]["containment_equals_recorded"] = True
        payload["axes"][axis]["composition"] = composition
    rdata = PC.record_bytes(payload, out.name)
    if P.code_block(CODE_FILES)["sha256"] != code0["sha256"]:
        raise SystemExit("REFUSING to write: source changed during the run")
    written = PC.write_all([(paths[s], data[s]) for s in splits] + [(index_paths[s], index_data[s]) for s in splits]
                           + [(out, rdata)])
    for s in splits:
        e = partitions[s]
        print(f"  {s:<12} n={e['n']:<6} pos={e['n_pos']:<6} neg={e['n_neg']:<6} clusters={e['n_clusters']:<6} "
              f"sha256={e['sha256'][:16]}...")
    print(f"Written: {_repo_rel(out)}\nsha256:  {written[out]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
