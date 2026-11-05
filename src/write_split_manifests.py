"""Write the per-item index manifests of the frozen splits to manifests/splits/ (tracked).

Why (2026-09-24, D-153): data/ is not tracked (D-148), so the frozen splits' prompt text is
not in the repository. The index manifests - item id, label, source row, split, and the
sha256 of the original file, but no prompt text - are what make a split rebuildable from
the public source dataset (D-140). They used to be generated only inside the release tree
by make_public_release.py; they now live, tracked, outside data/, and the release copies
them from here.

Refuses to write a manifest for any split whose sha256 differs from the one recorded in
results/dataset_manifest*.json, so a manifest can only describe the frozen bytes.

  python src/write_split_manifests.py            # write, refusing on any hash mismatch
  python src/write_split_manifests.py --check    # verify the tracked manifests, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import ensure_utf8_stdout  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "manifests" / "splits"


def manifest_from_split(path: Path) -> dict:
    """An index manifest: enough to rebuild the split from the public source dataset,
    with no prompt text. Carries the sha256 of the original file so a rebuild can be
    checked byte for byte."""
    rows, digest = [], hashlib.sha256()
    with open(path, "rb") as fh:
        for raw in fh:
            digest.update(raw)
            r = json.loads(raw)
            rows.append({k: r[k] for k in ("item_id", "label", "source_row")
                         if k in r} | ({"meta_type": r["meta_type"]}
                                       if "meta_type" in r else {}))
    return {"split_file": path.name, "n": len(rows),
            "sha256_of_original": digest.hexdigest(),
            "note": "Prompt text is not redistributed. Rebuild with "
                    "src/freeze_splits.py against the dataset revision pinned in "
                    "results/dataset_manifest*.json, then compare sha256.",
            "items": rows}


def recorded_hashes() -> dict[str, str]:
    want = {}
    for f in sorted((ROOT / "results").glob("dataset_manifest*.json")):
        for axis in json.load(open(f, encoding="utf-8"))["axes"].values():
            for part in axis["partitions"].values():
                want[Path(part["file"]).name] = part["sha256"]
    return want


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify tracked manifests only")
    args = ap.parse_args()
    want = recorded_hashes()
    bad = 0
    for name, h in sorted(want.items()):
        dest = OUT / (Path(name).stem + "__manifest.json")
        if args.check:
            ok = dest.exists() and json.load(open(dest, encoding="utf-8"))["sha256_of_original"] == h
            print(f"  {'OK ' if ok else 'BAD'} {dest.relative_to(ROOT)}")
            bad += not ok
            continue
        src = ROOT / "data" / "frozen" / name
        man = manifest_from_split(src)
        if man["sha256_of_original"] != h:
            print(f"  REFUSING {name}: sha256 {man['sha256_of_original'][:16]} != recorded {h[:16]}")
            bad += 1
            continue
        OUT.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"  wrote {dest.relative_to(ROOT)}  n={man['n']}  sha256_of_original {h[:16]}")
    print(f"{len(want) - bad} of {len(want)} {'verified' if args.check else 'written'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
