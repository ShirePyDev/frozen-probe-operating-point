"""Phase 2 data, split version 2 (D-318 (8), D-320): re-allocate the WildGuardMix benign items of the v1 frozen splits
as calibration 5,000 / eval 6,000 / pool4d the rest, positives unchanged. Runs ONCE.

The author's two conditions (D-318 (8)) were checked before this was written (D-320): no Phase 2 evaluation result
had been read, and every extracted model's caches hold every item of v1's calibration, eval and pool4d (so no
re-extraction is needed: a v2 split re-indexes items the caches already hold). The record re-checks the second one: it
reads every Phase 2 extraction record present and lists, per record, that its splits name v1's files and sha256.

INPUTS: results/phase2/dataset_manifest_p2.json (the v1 record, pinned by sha256 and run_kind) and the four v1 split
files data/frozen_p2/p2wg__<split>.jsonl, each checked against the sha256 the v1 record names.

RULE (RESPLIT_RULE below, recorded):
  - probe_train is not touched (v2 names v1's file and sha256 for it; no copy is written).
  - Every positive of v1 eval and pool4d stays in its split, and so does every item of a cluster holding a positive
    (whole clusters only; such a cluster's negatives stay with its positives). Calibration holds negatives only.
  - The movable units are the clusters whose items in v1 calibration, eval and pool4d are all negatives; each must lie
    wholly in one v1 split. They are sorted by their smallest item_id and shuffled by random.Random(RESPLIT_SEED).
  - Targets: calibration 5,000 negatives; eval 6,000 negatives in all (its fixed negatives count); pool4d every
    remaining negative (its fixed negatives count). Each split's ROOM is its target minus its fixed negatives.
    Multi-item units are placed first, then singletons, each in the shuffled order, each into the split with the
    largest share of its room still free that can hold it (ties: eval, calibration, pool4d). Every target is met
    exactly or the run exits 3 and writes nothing. (A greedy fill, not a packing solver: exit 3 covers a miss.)
  - EXCLUDED_ITEMS (recorded pre-analysis exclusions, D-324 (5)(iii), D-326) are removed before planning and enter no
    v2 split; the record lists each with its v1 split and reason.

OUTPUTS (distinct basenames from v1, so no copy by basename can confuse the versions):
  data/frozen_p2_v2/p2wg_v2__{calibration,eval,pool4d}.jsonl   rows byte-identical to their v1 lines (each row keeps
      its axis 'p2wg'); sorted by item_id; phase2_pool_clusters.split_bytes
  manifests/splits_p2_v2/p2wg_v2__<split>__manifest.json         the index manifests (phase2_freeze_splits' format,
      with v2's own rebuild note, the split version and the file's repo-relative path)
  results/phase2/dataset_manifest_p2_v2.json                       the record
All or nothing through safe_outputs.write_all; any existing p2wg* entry in either output directory, or an existing
record, makes the run refuse (CLAUDE.md rule 7; a failed run's '.failed' leftovers block a rerun until a human
removes them); nothing is deleted.

JOIN RULE for v2 readers (recorded): a v2 split's caches are its items gathered BY item_id from the model's v1 caches,
each item from the cache of the split v1 put it in (v1 calibration: the select record's p2wg/calibration cache; v1 eval
or pool4d: the rest record's p2wg/eval or p2wg/pool4d cache), whichever v2 split it is now in; a cache keyed
'p2wg/<split>' is a v1 cache and is never read as a v2 split.

  python3 src/phase2_resplit_v2.py                       # the final names
  python3 src/phase2_resplit_v2.py --out-root <dir>      # a test: every output under <dir>, outside the repository
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as P  # noqa: E402
import phase2_freeze_splits as FS2  # noqa: E402
import phase2_pool_clusters as PC  # noqa: E402
from provenance import (_RESULT_READS, _repo_rel, canonical_hash, ensure_utf8_stdout, provenance_block,  # noqa: E402
                        read_result, sha256_bytes)

ROOT = PC.ROOT
V1_RECORD = PC.RESULTS / "phase2" / "dataset_manifest_p2.json"
V1_RECORD_SHA256 = "b51e8419ccbf035581d7f4960d422a7a1a9d46eca35b3c47cfe46e6507d1621e"   # D-311, D-320
V1_DIR = FS2.FROZEN_P2
EXTRACT_DIR = PC.RESULTS / "phase2"
FROZEN_V2 = ROOT / "data" / "frozen_p2_v2"
MANIFESTS_V2 = ROOT / "manifests" / "splits_p2_v2"
OUT = PC.RESULTS / "phase2" / "dataset_manifest_p2_v2.json"
CODE_FILES = ["src/phase2_resplit_v2.py", *FS2.CODE_FILES]
AXIS = PC.AXIS
V2_STEM = f"{AXIS}_v2"                                # output basenames: p2wg_v2__<split>...
MOVED = ("eval", "calibration", "pool4d")             # placement order on ties
RESPLIT_SEED = 20261002                               # the unit-order shuffle of v2 only; fixed in D-320 before any run
TARGET_CAL_NEG, TARGET_EVAL_NEG = 5000, 6000          # D-318 (8); pool4d takes the rest
EXCLUDED_ITEMS = {                                    # recorded pre-analysis exclusions, kept out of every v2 split
    "ef72f52bd8c3ec85": "D-324 (5)(iii), D-323: its text begins with a literal '<s>[INST] <<SYS>>', which the Mistral "
                        "and Phi-3 tokenizers read as a BOS id inside the user span (a second BOS for Mistral); excluded "
                        "before any analysis",
    "52c52cdb43a59029": "D-329, D-357 (B)(3): its user text holds a literal '[/INST]', which Mistral-7B-Instruct-v0.3's "
                        "tokenizer reads as a control token inside the user span (the extracted v0.3 caches hold it so); "
                        "excluded from every v2 split before any analysis",
}
RESPLIT_RULE = (
    "v2 re-allocates v1's benign items (D-318 (8)): probe_train untouched; every positive of v1 eval and pool4d, and "
    "every item of a cluster holding a positive, stays in its v1 split; the movable units are the clusters whose items "
    "in v1 calibration, eval and pool4d are all negatives (each wholly in one v1 split), sorted by smallest item_id and "
    f"shuffled by random.Random({RESPLIT_SEED}); targets calibration {TARGET_CAL_NEG} negatives, eval "
    f"{TARGET_EVAL_NEG} negatives in all, pool4d the rest; a split's room is its target minus its fixed negatives; "
    "multi-item units first, then singletons, each in the shuffled order into the split with the largest share of "
    "its room still free that can hold it (ties eval, calibration, pool4d); every target exact or exit 3. Rows are "
    "v1's lines, byte for byte. The EXCLUDED_ITEMS (recorded pre-analysis exclusions, D-324 (5)(iii)) are removed "
    "before planning and enter no v2 split.")
JOIN_RULE = (
    "A v2 split's caches are its items gathered by item_id from the model's v1 caches, each item from the cache of the "
    "split v1 put it in (v1 calibration: the select record's p2wg/calibration cache; v1 eval or pool4d: the rest "
    "record's p2wg/eval or p2wg/pool4d cache), whichever v2 split it is now in; plan.items_from_v1_to_v2 counts the "
    "moved negatives by v1 and v2 split (positives stay in their v1 split). A cache keyed 'p2wg/<split>' is a v1 cache "
    "and is never read as a v2 split. Under D-307 (1) a result reads only ingested caches, so a v2 reader needs the rest "
    "caches (or their durable layer subsets) ingested and indexed first.")
INDEX_NOTE_V2 = ("Prompt text is not redistributed. Split version 2 (D-318 (8), D-320): rebuild v1 with "
                 "src/phase2_pool_clusters.py then src/phase2_freeze_splits.py (results/phase2/dataset_manifest_p2.json), "
                 "then run src/phase2_resplit_v2.py and compare sha256 with results/phase2/dataset_manifest_p2_v2.json.")


def read_split(path: Path, want_sha: str) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """A v1 split file, checked against its recorded sha256; its rows and {item_id: the line's bytes}."""
    b = path.read_bytes()
    if sha256_bytes(b) != want_sha:
        raise SystemExit(f"REFUSING: {path.name} is not the bytes the v1 record names ({sha256_bytes(b)[:16]} vs "
                         f"{want_sha[:16]})")
    lines = [ln for ln in b.split(b"\n") if ln]
    rows = [json.loads(ln.decode("utf-8")) for ln in lines]  # not a result (D-297): a split file, hashed above
    if PC.split_bytes(rows) != b:
        raise SystemExit(f"REFUSING: {path.name} does not round-trip through split_bytes")
    by_id = {r["item_id"]: ln + b"\n" for r, ln in zip(rows, lines)}
    if len(by_id) != len(rows):
        raise SystemExit(f"REFUSING: {path.name} repeats an item_id")
    return rows, by_id


def plan_v2(v1: dict[str, list[dict[str, Any]]], seed: int = RESPLIT_SEED) -> dict[str, Any]:
    """The v2 assignment over v1's rows of calibration, eval and pool4d (RESPLIT_RULE). Returns the rows per v2 split
    and the counts; 'shortfall' lists every target missed (empty when the plan is exact). EXCLUDED_ITEMS are removed
    first (each must be a negative found exactly once in v1 calibration, eval or pool4d)."""
    excluded = []
    for iid, why in EXCLUDED_ITEMS.items():
        where = [(s, r) for s in MOVED for r in v1[s] if r["item_id"] == iid]
        if len(where) != 1 or int(where[0][1]["label"]) != 0:
            raise SystemExit(f"REFUSING: excluded item {iid} is not exactly one negative of v1 calibration, eval or pool4d")
        excluded.append({"item_id": iid, "v1_split": where[0][0], "label": 0, "cluster": where[0][1]["cluster"],
                         "reason": why})
    v1 = {**v1, **{s: [r for r in v1[s] if r["item_id"] not in EXCLUDED_ITEMS] for s in MOVED}}
    by_cluster: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for s in MOVED:
        for r in v1[s]:
            by_cluster.setdefault(r["cluster"], []).append((s, r))
    fixed: dict[str, list[dict[str, Any]]] = {s: [] for s in MOVED}
    units: list[dict[str, Any]] = []
    for c, items in by_cluster.items():
        homes = {s for s, _ in items}
        if len(homes) != 1:
            raise SystemExit(f"REFUSING: cluster {c} lies in two v1 splits {sorted(homes)}")
        home = homes.pop()
        rows = [r for _, r in items]
        if any(int(r["label"]) == 1 for r in rows):
            if home == "calibration":
                raise SystemExit(f"REFUSING: v1 calibration holds a positive (cluster {c})")
            fixed[home].extend(rows)                         # a cluster holding a positive stays whole where it is
        else:
            units.append({"cluster": c, "rows": rows, "home": home, "key": min(r["item_id"] for r in rows)})
    n_neg = sum(int(r["label"]) == 0 for s in MOVED for r in v1[s])
    fixed_neg = {s: sum(int(r["label"]) == 0 for r in fixed[s]) for s in MOVED}
    targets = {"calibration": TARGET_CAL_NEG, "eval": TARGET_EVAL_NEG,
               "pool4d": n_neg - TARGET_CAL_NEG - TARGET_EVAL_NEG}
    if targets["pool4d"] <= 0:
        raise SystemExit(f"REFUSING: v1 holds {n_neg} negatives, too few for calibration {TARGET_CAL_NEG} and eval "
                         f"{TARGET_EVAL_NEG} with any left for pool4d")
    room = {s: targets[s] - fixed_neg[s] for s in MOVED}
    order = sorted(units, key=lambda u: u["key"])
    random.Random(seed).shuffle(order)
    got: dict[str, list[dict[str, Any]]] = {s: [] for s in MOVED}
    left = dict(room)
    unplaced = []
    for group in ([u for u in order if len(u["rows"]) >= 2], [u for u in order if len(u["rows"]) == 1]):
        for u in group:
            k = len(u["rows"])
            fits = [s for s in MOVED if room[s] > 0 and left[s] >= k]
            if not fits:
                unplaced.append(u)
                continue
            s = max(fits, key=lambda t: (left[t] / room[t], -MOVED.index(t)))   # share of room free; ties in order
            got[s].append(u)
            left[s] -= k
    rows = {s: fixed[s] + [r for u in got[s] for r in u["rows"]] for s in MOVED}
    shortfall = []
    for s in MOVED:
        neg = sum(int(r["label"]) == 0 for r in rows[s])
        if neg != targets[s]:
            shortfall.append(f"{s} holds {neg} negatives, target {targets[s]}")
    if unplaced:
        shortfall.append(f"{len(unplaced)} movable units ({sum(len(u['rows']) for u in unplaced)} items) found no room")
    moves = Counter((u["home"], s) for s in MOVED for u in got[s] for _ in u["rows"])
    return {"rows": rows, "targets": targets, "room": room, "fixed_negatives": fixed_neg,
            "fixed_positives": {s: sum(int(r["label"]) == 1 for r in fixed[s]) for s in MOVED},
            "movable": {"units": len(units), "items": sum(len(u["rows"]) for u in units),
                        "multi_item_units": sum(1 for u in units if len(u["rows"]) >= 2)},
            "units_placed": {s: {"units": len(got[s]), "multi_item_units": sum(1 for u in got[s] if len(u["rows"]) >= 2),
                                 "items": sum(len(u["rows"]) for u in got[s])} for s in MOVED},
            "items_from_v1_to_v2": {f"{a}->{b}": n for (a, b), n in sorted(moves.items())},
            "excluded_items": excluded, "shortfall": shortfall}


def check_v2(v1: dict[str, list[dict[str, Any]]], v2: dict[str, list[dict[str, Any]]]) -> None:
    """The checks the record states (except byte identity, checked on the bytes in main), on the rows to be written."""
    def ids(rs: list[dict[str, Any]], lab: int | None = None) -> list[str]:
        return sorted(r["item_id"] for r in rs if lab is None or int(r["label"]) == lab)
    if any(r["item_id"] in EXCLUDED_ITEMS for s in MOVED for r in v2[s]):
        raise SystemExit("REFUSING: an excluded item is in a v2 split")
    if ids([r for s in MOVED for r in v1[s] if r["item_id"] not in EXCLUDED_ITEMS]) != ids([r for s in MOVED for r in v2[s]]):
        raise SystemExit("REFUSING: v2's items are not exactly v1's calibration, eval and pool4d items minus the exclusions")
    if len({r["item_id"] for s in MOVED for r in v2[s]}) != sum(len(v2[s]) for s in MOVED):
        raise SystemExit("REFUSING: an item is in two v2 splits")
    for s in ("eval", "pool4d"):
        if ids(v1[s], 1) != ids(v2[s], 1):
            raise SystemExit(f"REFUSING: {s}'s positives differ from v1's")
    if ids(v2["calibration"], 1):
        raise SystemExit("REFUSING: v2 calibration holds a positive")
    if any(r["mixed_label_cluster"] for r in v2["calibration"]):
        raise SystemExit("REFUSING: v2 calibration holds an item of a mixed-label cluster")
    where: dict[str, set[str]] = {}
    for s, rs in [*((s, v2[s]) for s in MOVED), ("probe_train", v1.get("probe_train", []))]:
        for r in rs:
            where.setdefault(r["cluster"], set()).add(s)
    if any(len(v) > 1 for v in where.values()):
        raise SystemExit("REFUSING: a cluster straddles two splits (v2's three, or one of them and probe_train)")


def composition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per class (rule 5): items, adversarial items, items in multi-item clusters, median prompt length in characters."""
    cl = Counter(r["cluster"] for r in rows)
    out: dict[str, Any] = {}
    for lab, name in ((1, "pos"), (0, "neg")):
        rs = [r for r in rows if int(r["label"]) == lab]
        out[name] = {"n": len(rs), "adversarial": sum(1 for r in rs if str(r.get("adversarial")) == "True"),
                     "in_multi_item_clusters": sum(1 for r in rs if cl[r["cluster"]] >= 2),
                     "median_chars": float(statistics.median(len(r["text"]) for r in rs)) if rs else None}
    return out


def index_manifest_bytes_v2(path_rel: str, rows: list[dict[str, Any]], digest: str) -> bytes:
    """phase2_freeze_splits.index_manifest_bytes' format and key order, with v2's note, the split version and the
    file's repo-relative path (the format is that of FS2 at the code the v1 record binds; D-340 later changed
    phase2_freeze_splits.py for the p2sst axis without changing any p2wg output, checked byte-identical)."""
    items = [{k: r[k] for k in ("item_id", "label", "source_row") if k in r}
             | ({"meta_type": r["meta_type"]} if "meta_type" in r else {}) for r in rows]
    man = {"split_file": Path(path_rel).name, "split_path": path_rel, "split_version": 2, "n": len(rows),
           "sha256_of_original": digest, "note": INDEX_NOTE_V2, "items": items}
    return (json.dumps(man, indent=1) + "\n").encode("utf-8")


def extraction_coverage(v1_parts: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """D-318 (8)'s second condition, on the record: every Phase 2 extraction record present (run_kind extract_p2) names
    v1's calibration (a select record) or v1's eval and pool4d (a rest record) by file and sha256, with every item
    cached, finite and untruncated, and every model has both records (a sitting half-way refuses)."""
    want = {s: (v1_parts[s]["file"], v1_parts[s]["sha256"], v1_parts[s]["n"]) for s in MOVED}
    cover: dict[str, Any] = {}
    reads: dict[str, str] = {}
    for p in sorted(EXTRACT_DIR.glob("extract__*__p2tpl@*__*.json")):
        rec = read_result(p)
        rel = _repo_rel(p)
        reads[rel] = _RESULT_READS[rel]
        if rec.get("run_kind") != "extract_p2":
            raise SystemExit(f"REFUSING: {rel} has run_kind {rec.get('run_kind')!r}, not extract_p2")
        parts = {"select": ("calibration",), "rest": ("eval", "pool4d")}.get(rec.get("set_name"))
        if parts is None:
            raise SystemExit(f"REFUSING: {rel} has set_name {rec.get('set_name')!r}")
        ok = {}
        for s in parts:
            for pooling in ("usermean", "lastgen"):
                e = (rec.get("splits") or {}).get(f"{AXIS}/{s}/{pooling}") or {}
                ok[f"{s}/{pooling}"] = ((e.get("split_file"), e.get("split_sha256"), e.get("n")) == want[s]
                                        and e.get("n_nonfinite") == 0 and e.get("n_truncated") == 0)
        if not all(ok.values()):
            raise SystemExit(f"REFUSING: {rel} does not cover v1's splits with every item finite: {ok}")
        cover[rel] = {"repo": rec.get("repo"), "set_name": rec.get("set_name"), "splits_checked": sorted(ok)}
    if not cover:
        raise SystemExit("REFUSING: no Phase 2 extraction record found")
    sets: dict[str, set[str]] = {}
    for c in cover.values():
        sets.setdefault(str(c["repo"]), set()).add(str(c["set_name"]))
    half = sorted(r for r, v in sets.items() if v != {"select", "rest"})
    if half:
        raise SystemExit(f"REFUSING: models without both a select and a rest record (a sitting half-way?): {half}")
    return cover, reads


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Phase 2 split version 2 (D-318 (8)); runs once")
    ap.add_argument("--out-root", default=None,
                    help="a test: write every output under this directory (it must lie outside the repository)")
    args = ap.parse_args()
    final = args.out_root is None
    if final:
        frozen_dir, index_dir, out = FROZEN_V2, MANIFESTS_V2, OUT
    else:
        base = Path(args.out_root).resolve()
        frozen_dir, index_dir, out = base / "frozen_p2_v2", base / "splits_p2_v2", base / OUT.name
        PC.outside_repo_or_refuse([frozen_dir, index_dir, out], "test run (--out-root)")
    paths = {s: frozen_dir / f"{V2_STEM}__{s}.jsonl" for s in MOVED}
    index_paths = {s: index_dir / f"{V2_STEM}__{s}__manifest.json" for s in MOVED}
    clash = sorted(frozen_dir.glob(f"{AXIS}*")) + sorted(index_dir.glob(f"{AXIS}*")) + \
        ([out] if out.exists() or out.is_symlink() else [])
    if clash:
        raise SystemExit(f"REFUSING TO OVERWRITE (CLAUDE.md rule 7: a split is written once): {[str(p) for p in clash]}")
    if final:      # the record must name repo-relative paths; a symlinked root would silently reduce them to basenames
        for p, want in [*((paths[s], f"data/frozen_p2_v2/{paths[s].name}") for s in MOVED),
                        *((index_paths[s], f"manifests/splits_p2_v2/{index_paths[s].name}") for s in MOVED),
                        (OUT, "results/phase2/dataset_manifest_p2_v2.json"),
                        (V1_RECORD, "results/phase2/dataset_manifest_p2.json")]:
            if _repo_rel(p) != want:
                raise SystemExit(f"REFUSING: {p} resolves to {_repo_rel(p)!r} relative to the repository, not {want!r}")
    code0 = P.code_block(CODE_FILES)

    rec = read_result(V1_RECORD)                               # registered: listed in the record's inputs
    rrel = _repo_rel(V1_RECORD)
    if _RESULT_READS[rrel] != V1_RECORD_SHA256 or rec.get("run_kind") != "phase2_freeze_splits":
        raise SystemExit(f"REFUSING: {rrel} is not the pinned v1 record (sha256 {_RESULT_READS[rrel][:16]}, run_kind "
                         f"{rec.get('run_kind')!r})")
    parts = rec["axes"][AXIS]["partitions"]
    v1: dict[str, list[dict[str, Any]]] = {}
    v1_lines: dict[str, bytes] = {}
    for s in (*MOVED, "probe_train"):
        f = V1_DIR / f"{AXIS}__{s}.jsonl"
        if parts[s]["file"] != f"data/frozen_p2/{f.name}":
            raise SystemExit(f"REFUSING: the v1 record names {parts[s]['file']!r} for {s}, not data/frozen_p2/{f.name}")
        v1[s], lines = read_split(f, parts[s]["sha256"])
        if s in MOVED:
            v1_lines.update(lines)
    for s in MOVED:
        if (len(v1[s]), sum(int(r["label"]) == 1 for r in v1[s])) != (parts[s]["n"], parts[s]["n_pos"]):
            raise SystemExit(f"REFUSING: v1 {s}'s counts differ from the v1 record's")
    coverage, cover_reads = extraction_coverage(parts)
    pl = plan_v2(v1)
    if pl["shortfall"]:
        print("SHORTFALL; nothing is written:\n  " + "\n  ".join(pl["shortfall"]))
        return 3
    t = pl["targets"]
    if t["calibration"] < PC.SIZES["cal_neg_min"] or t["eval"] < PC.SIZES["eval_neg_min"] \
            or t["pool4d"] < PC.SIZES["pool4d_neg"]:
        raise SystemExit("REFUSING: a v2 target is below the registered minimum (D-303)")
    check_v2(v1, pl["rows"])

    ordered = {s: sorted(pl["rows"][s], key=lambda r: r["item_id"]) for s in MOVED}
    data = {s: PC.split_bytes(ordered[s]) for s in MOVED}
    for s in MOVED:                                            # byte identity with v1, on the bytes to be written
        if data[s] != b"".join(v1_lines[r["item_id"]] for r in ordered[s]):
            raise SystemExit(f"REFUSING: a v2 {s} line differs from its v1 line")
    sha = {s: sha256_bytes(data[s]) for s in MOVED}
    rel = {s: (f"data/frozen_p2_v2/{paths[s].name}" if final else paths[s].name) for s in MOVED}
    index_data = {s: index_manifest_bytes_v2(rel[s], ordered[s], sha[s]) for s in MOVED}
    partitions: dict[str, Any] = {}
    for s in MOVED:
        rs = ordered[s]
        npos = sum(int(r["label"]) == 1 for r in rs)
        cl = Counter(r["cluster"] for r in rs)
        partitions[s] = {"file": rel[s], "sha256": sha[s], "n": len(rs), "n_pos": npos, "n_neg": len(rs) - npos,
                         "target_neg": t[s], "n_clusters": len(cl),
                         "n_multi_item_clusters": sum(1 for v in cl.values() if v >= 2),
                         "composition_by_class": composition(rs),
                         "v1_composition_by_class": composition(v1[s]),
                         "n_from_mixed_label_clusters": {
                             "pos": sum(1 for r in rs if r["mixed_label_cluster"] and int(r["label"]) == 1),
                             "neg": sum(1 for r in rs if r["mixed_label_cluster"] and int(r["label"]) == 0)},
                         "index_manifest": {"file": (f"manifests/splits_p2_v2/{index_paths[s].name}" if final
                                                     else index_paths[s].name),
                                            "sha256": sha256_bytes(index_data[s])}}
    partitions["probe_train"] = {**{k: parts["probe_train"][k] for k in ("file", "sha256", "n", "n_pos", "n_neg")},
                                 "note": "v1's file, unchanged (no copy is written)"}
    config = {"rule": RESPLIT_RULE, "join_rule": JOIN_RULE, "excluded_items": dict(EXCLUDED_ITEMS),
              "target_calibration_neg": TARGET_CAL_NEG,
              "target_eval_neg": TARGET_EVAL_NEG, "pool4d": "every remaining negative", "seed": RESPLIT_SEED,
              "split_dir": "data/frozen_p2_v2/ (v1 stays in data/frozen_p2/; output basenames p2wg_v2__*)",
              "index_manifests": "manifests/splits_p2_v2/, in phase2_freeze_splits' format plus split_path and "
                                 "split_version, with v2's rebuild note"}
    v1_inputs = {parts[s]["file"]: parts[s]["sha256"] for s in (*MOVED, "probe_train")}
    payload = {**provenance_block("phase2_resplit_v2"), "decision": "D-318 (8), D-320, D-324 (5)(iii), D-326", "split_version": 2,
               "code": code0, "seed": RESPLIT_SEED, "config": config,
               "config_hash": canonical_hash(config),
               "inputs": {rrel: _RESULT_READS[rrel], **v1_inputs, **cover_reads},
               "preconditions": {
                   "no_phase2_evaluation_result_read": "stated in D-320 (no Phase 2 result reads eval; the selection "
                                                       "records read probe_train and calibration only)",
                   "caches_hold_every_item": coverage},
               "v1": {"record": rrel, "record_sha256": V1_RECORD_SHA256,
                      "partitions": {s: {k: parts[s][k] for k in ("file", "sha256", "n", "n_pos", "n_neg")}
                                     for s in (*MOVED, "probe_train")}},
               "checks": {"items_exactly_v1_calibration_eval_pool4d_minus_excluded": True, "no_item_in_two_splits": True,
                          "positives_unchanged_eval_pool4d": True, "calibration_negatives_only": True,
                          "no_mixed_label_item_in_calibration": True,
                          "no_cluster_in_two_splits_including_probe_train": True,
                          "lines_byte_identical_to_v1": True, "probe_train_untouched": True,
                          "every_target_exact": True, "minimums_met": True},
               "plan": {k: v for k, v in pl.items() if k != "rows"},
               "axes": {AXIS: {"source": rec["axes"][AXIS]["source"],
                               "source_revision_sha": rec["axes"][AXIS]["source_revision_sha"],
                               "partitions": partitions}}}
    rdata = PC.record_bytes(payload, out.name)
    if P.code_block(CODE_FILES)["sha256"] != code0["sha256"]:
        raise SystemExit("REFUSING to write: source changed during the run")
    written = PC.write_all([(paths[s], data[s]) for s in MOVED] + [(index_paths[s], index_data[s]) for s in MOVED]
                           + [(out, rdata)])
    for s in MOVED:
        e = partitions[s]
        print(f"  {s:<12} n={e['n']:<6} pos={e['n_pos']:<6} neg={e['n_neg']:<6} clusters={e['n_clusters']:<6} "
              f"multi={e['n_multi_item_clusters']:<4} sha256={e['sha256'][:16]}...")
    print(f"  moves {pl['items_from_v1_to_v2']}")
    print(f"  extraction records checked: {len(coverage)}")
    print(f"Written: {_repo_rel(out)}\nsha256:  {written[out]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
