"""Near-duplicate clusters of the pooled negatives, as a committed manifest of item ids (D-164).

D-164 adds a cluster-level re-partitioning sensitivity: no near-duplicate cluster may straddle
calibration and evaluation. The clusters are the D-157 ones (results/split_diagnostics.json:
exact all-pairs Jaccard >= 0.5 on the word 3-gram shingles of src/paraphrase_overlap.py, over
the threshold_calib negatives then the eval negatives of each axis), recomputed here with the
same function and written as item ids only - no prompt text - to manifests/negative_clusters.json.

The frozen split files are checked against the committed split manifests (manifests/splits/,
D-153) before use, and the cluster counts must equal those split_diagnostics.json recorded.
Runs where data/frozen exists (the node). Never overwrites the manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as P  # noqa: E402
from provenance import (  # noqa: E402
    commit_staged, ensure_utf8_stdout, provenance_block, read_jsonl, sha256_bytes, sha256_file, stage_bytes,
)
from split_diagnostics import J_MIN, clusters  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "data" / "frozen"
SPLIT_MANIFESTS = ROOT / "manifests" / "splits"
OUT = ROOT / "manifests" / "negative_clusters.json"
CODE_FILES = ["src/negative_clusters.py", "src/split_diagnostics.py", "src/paraphrase_overlap.py", "src/paired.py",
              "src/populations.py", "src/provenance.py", "src/superseded_provenance.py"]
AXES = ("safety", "safety_wildguard")


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Near-duplicate clusters of the pooled negatives (D-164)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"REFUSING: {out} exists; a manifest is never overwritten")
    code0 = P.code_block(CODE_FILES)
    # Reference files, hashed at start and re-checked at the end: the D-157 diagnostics whose
    # counts must be reproduced, and the committed split manifests the texts are checked against.
    refs = [ROOT / "results" / "split_diagnostics.json"] + \
           [SPLIT_MANIFESTS / f"{a}__{p}__manifest.json" for a in AXES for p in ("threshold_calib", "eval")]
    ref_sha = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in refs}
    diag = json.load(open(refs[0], encoding="utf-8"))["clusters"]

    axes: dict[str, dict] = {}
    for axis in AXES:
        recs, split_sha = {}, {}
        for part in ("threshold_calib", "eval"):
            f = FROZEN / f"{axis}__{part}.jsonl"
            man = json.load(open(SPLIT_MANIFESTS / f"{axis}__{part}__manifest.json", encoding="utf-8"))
            h = sha256_file(f)
            if h != man["sha256_of_original"]:
                raise SystemExit(f"REFUSING: {f.name} sha256 {h[:16]} != committed manifest {man['sha256_of_original'][:16]}")
            rows = read_jsonl(f)                        # the sanctioned reader (F-017)
            if sha256_file(f) != h:                     # the bytes parsed are the bytes hashed
                raise SystemExit(f"REFUSING: {f.name} changed while it was being read")
            if [r["item_id"] for r in rows] != [x["item_id"] for x in man["items"]]:
                raise SystemExit(f"REFUSING: {f.name} item order differs from its committed manifest")
            split_sha[f.name] = h
            recs[part] = [r for r in rows if r["label"] == 0]
        pool = [r["item_id"] for r in recs["threshold_calib"]] + [r["item_id"] for r in recs["eval"]]
        lab, summ = clusters([r["text"] for r in recs["threshold_calib"]] + [r["text"] for r in recs["eval"]])
        groups: dict[int, list[str]] = {}
        for iid, c in zip(pool, lab):
            groups.setdefault(int(c), []).append(iid)
        multi = sorted((sorted(g) for g in groups.values() if len(g) >= 2), key=lambda g: (-len(g), g[0]))
        for k in ("n_items", "n_clusters", "items_in_clusters", "n_pairs_jaccard_ge_0_5",
                  "size_distribution", "n_empty_shingle_sets", "largest_cluster_sizes"):
            if summ[k] != diag[axis][k]:
                raise SystemExit(f"REFUSING: {axis} {k} {summ[k]} != split_diagnostics.json {diag[axis][k]}")
        axes[axis] = {"n_pooled_negatives": len(pool), "n_calibration": len(recs["threshold_calib"]),
                      "n_evaluation": len(recs["eval"]),
                      "pool_order": "threshold_calib negatives then eval negatives, split-file order",
                      "pool_item_ids_sha256": sha256_bytes("\n".join(pool).encode("utf-8")),
                      "n_clusters": len(multi), "items_in_clusters": sum(len(g) for g in multi),
                      "clusters": multi, "split_file_sha256": split_sha}
        print(f"[{axis}] {len(pool)} pooled negatives: {len(multi)} clusters holding "
              f"{sum(len(g) for g in multi)} items (all D-157 statistics reproduced)", flush=True)

    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"] or any(sha256_file(ROOT / f) != h for f, h in ref_sha.items()):
        raise SystemExit("REFUSING to write: source or a reference file changed during the run")
    payload = {**provenance_block("negative_clusters"), "decision": "D-164", "code": code0,
               "config": {"jaccard_min": J_MIN, "jaccard": "exact, all pairs, word 3-gram shingles "
                                                           "(src/paraphrase_overlap.py), union-find components",
                          "singletons": "every negative not listed is its own unit"},
               "reference_files_sha256": ref_sha, "axes": axes}
    data = json.dumps(payload, indent=2, default=str).encode("utf-8")
    tmp, want = stage_bytes(out, data)
    commit_staged(tmp, out, want)
    rel = str(out.relative_to(ROOT)).replace("\\", "/") if out.is_relative_to(ROOT) else out.name
    print(f"Written: {rel}\nsha256:  {want}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
