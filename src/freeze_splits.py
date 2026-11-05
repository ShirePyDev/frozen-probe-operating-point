"""Write the frozen, SHA-256-hashed dataset splits. Runs ONCE.

CLAUDE.md rule 7: every split is written once, hashed, and the hash recorded in
the results file. This script REFUSES to overwrite an existing frozen split
unless --force is passed, because silently regenerating a split would invalidate
every result that cites its hash.

Partition scheme (D-003) - three parts, not two, because the primary arm refits
the probe on the post-update backbone (D-002):

    probe_train      fit the logistic probe. Reused BYTE-IDENTICALLY on both
                     sides of an update, so the refit is a control rather than
                     an uncontrolled variable.
    threshold_calib  used EXACTLY ONCE, on the pre-update model, to pick the
                     decision threshold at 1% FPR. Then frozen.
    eval             the measurement set. Sized from the power analysis (D-029).

Axes:
    safety        Aegis 2.0 prompt_label in {safe, unsafe}. Human-labelled.
    over_refusal  XSTest prompts. Per D-030 beta is NOT estimable here at the
                  1% operating point (450 items vs 10000 needed), so this axis
                  carries the UNION false-refusal reading, not beta.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import (  # noqa: E402
    canonical_hash,
    ensure_utf8_stdout,
    provenance_block,
    sha256_file,
    write_results,
)

FROZEN = Path(__file__).resolve().parent.parent / "data" / "frozen"

# Sizes per partition, per class. Eval is >= 2000 total per D-029.
SPEC = {
    "safety": {
        "source": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
        "config": None,
        "split": "train",
        "text_field": "prompt",
        "label_field": "prompt_label",
        "positive_value": "unsafe",  # "positive" = harmful = should be FLAGged
        "negative_value": "safe",
        "partitions": {
            "probe_train": {"pos": 2000, "neg": 2000},
            "threshold_calib": {"pos": 1000, "neg": 1000},
            "eval": {"pos": 2000, "neg": 2000},
        },
    },
    "safety_wildguard": {
        "source": "allenai/wildguardmix",
        "config": "wildguardtrain",
        "split": "train",
        "text_field": "prompt",
        "label_field": "prompt_harm_label",
        "positive_value": "harmful",
        "negative_value": "unharmful",
        # Carried into the frozen records so the adversarial subset can be
        # analysed later without re-deriving it from the source.
        "extra_fields": ["adversarial", "subcategory"],
        "partitions": {
            "probe_train": {"pos": 2000, "neg": 2000},
            "threshold_calib": {"pos": 1000, "neg": 1000},
            "eval": {"pos": 2000, "neg": 2000},
        },
    },
    "over_refusal": {
        "source": "natolambert/xstest-v2-copy",
        # "prompts" is a SPLIT of the single "default" config, not a config.
        # The other splits (gpt4, llama2new, ...) are model RESPONSES, not prompts.
        "config": None,
        "split": "prompts",
        "text_field": "prompt",
        "label_field": "type",
        # XSTest "safe" types are the over-refusal probes; "contrast_" types are
        # genuinely unsafe controls. Label resolution happens in load_axis().
        "positive_value": None,
        "negative_value": None,
        "partitions": {"eval": {"pos": None, "neg": None}},
    },
}

SEED = 20260914


# Prompt texts that carry no content and must never enter a split. Aegis 2.0
# contains 911 rows whose prompt field is the literal string "REDACTED".
PLACEHOLDER_TEXTS = {"redacted", "[redacted]", "n/a", "none", ""}


def load_axis(axis: str, spec: dict[str, Any], revision: str | None = None,
              local_hub: str | None = None) -> tuple[list[dict[str, Any]], str, dict[str, int]]:
    """Load one axis, clean it, and return records, revision SHA, and drop counts.

    Four cleaning steps, each of which was forced by a real defect measured in
    the source (see FAILURES.md F-015). They are applied in this order and the
    counts are recorded in the manifest so the cleaning is auditable:

      1. drop rows with empty or placeholder prompt text
      2. drop texts whose label is CONFLICTING across duplicate rows - the same
         prompt appearing as both safe and unsafe is unusable ground truth, and
         keeping it would inject label noise that lands differently in each
         partition
      3. deduplicate by normalised text, keeping the first in a deterministic
         order - without this, a prompt can appear in probe_train AND eval,
         which is straightforward train/test contamination
      4. assign a content-derived item_id, unique by construction after step 3
    """
    import collections
    import hashlib

    from datasets import load_dataset
    from huggingface_hub import HfApi

    # revision=None is the original freeze: whatever the Hub serves now, recorded.
    # A REBUILD (2026-09-25) passes the recorded revision, so the same bytes are read -
    # offline, from a local cache, with no Hub call and no token.
    rev = revision if revision else HfApi().dataset_info(spec["source"]).sha

    kwargs = {"split": spec["split"]}
    source = spec["source"]
    if revision and local_hub:
        # Offline rebuild: read the snapshot directory of the recorded revision directly.
        # load_dataset on a local directory applies the same README split configuration to
        # the same files; datasets 5 refuses to resolve a repo NAME offline.
        source = str(Path(local_hub) / f"datasets--{spec['source'].replace('/', '--')}"
                     / "snapshots" / revision)
        if not Path(source).is_dir():
            raise SystemExit(f"REFUSING: no local snapshot for {spec['source']}@{revision}: {source}")
    elif revision:
        kwargs["revision"] = revision
    if spec["config"]:
        ds = load_dataset(source, spec["config"], **kwargs)
    else:
        ds = load_dataset(source, **kwargs)

    stats = {
        "rows_in": 0,
        "dropped_placeholder": 0,
        "dropped_unlabelled": 0,
        "dropped_conflicting_label": 0,
        "dropped_duplicate_text": 0,
    }

    staged: list[tuple] = []  # (norm_text, label, row_idx, meta, source_row)
    for i, row in enumerate(ds):
        stats["rows_in"] += 1
        raw = row.get(spec["text_field"])
        text = str(raw).strip() if raw is not None else ""
        if text.lower() in PLACEHOLDER_TEXTS:
            stats["dropped_placeholder"] += 1
            continue
        meta = str(row.get(spec["label_field"], ""))
        # Keyed off the spec, not off the axis NAME: a new labelled axis must not
        # have to be added to a hard-coded list to be handled correctly.
        if spec.get("positive_value") is not None:
            lab = row.get(spec["label_field"])
            if lab == spec["positive_value"]:
                y = 1
            elif lab == spec["negative_value"]:
                y = 0
            else:
                stats["dropped_unlabelled"] += 1
                continue
        else:
            # XSTest: "type" values starting with "contrast_" are genuinely unsafe;
            # the rest are safe prompts that merely look unsafe (the over-refusal
            # probes, which are the point of the benchmark).
            y = 1 if meta.startswith("contrast_") else 0
        staged.append((text, y, i, meta, row))

    by_text: dict[str, set[int]] = collections.defaultdict(set)
    for text, y, _, _, _ in staged:
        by_text[text].add(y)
    conflicting = {t for t, labs in by_text.items() if len(labs) > 1}

    records: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for text, y, i, meta, row in staged:
        if text in conflicting:
            stats["dropped_conflicting_label"] += 1
            continue
        if text in seen_text:
            stats["dropped_duplicate_text"] += 1
            continue
        seen_text.add(text)
        item_id = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        rec = {"item_id": item_id, "text": text, "label": y, "axis": axis,
               "source_row": i, "meta_type": meta}
        for ef in spec.get("extra_fields", []):
            v = row.get(ef)
            rec[ef] = str(v) if v is not None else None
        records.append(rec)

    assert len({r["item_id"] for r in records}) == len(records), "item_id collision"
    return records, rev, stats


def partition(
    records: list[dict[str, Any]], partitions: dict[str, dict[str, int | None]], seed: int
) -> dict[str, list[dict[str, Any]]]:
    """Deterministic, disjoint, class-balanced partitioning.

    Sorting by item_id before shuffling makes the result independent of the order
    the source happened to yield rows in, so the same source revision always gives
    the same split regardless of library version or row iteration order.
    """
    import random

    pos = sorted([r for r in records if r["label"] == 1], key=lambda r: r["item_id"])
    neg = sorted([r for r in records if r["label"] == 0], key=lambda r: r["item_id"])
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    out: dict[str, list[dict[str, Any]]] = {}
    pi = ni = 0
    for name, sizes in partitions.items():
        want_pos, want_neg = sizes["pos"], sizes["neg"]
        if want_pos is None:  # take everything remaining
            take_pos, take_neg = pos[pi:], neg[ni:]
            pi, ni = len(pos), len(neg)
        else:
            take_pos = pos[pi : pi + want_pos]
            take_neg = neg[ni : ni + want_neg]
            pi += want_pos
            ni += want_neg
        chunk = take_pos + take_neg
        chunk.sort(key=lambda r: r["item_id"])  # stable on disk
        out[name] = chunk
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    """Write one partition and return its SHA-256. LF endings, sorted keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    return sha256_file(path)


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Freeze and hash dataset splits (runs once)")
    ap.add_argument("--force", action="store_true", help="overwrite existing frozen splits")
    ap.add_argument("--axes", nargs="*", default=list(SPEC.keys()))
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "results" / "dataset_manifest.json"),
    )
    ap.add_argument("--out-dir", default=None,
                    help="REBUILD ONLY: write the split files here instead of data/frozen")
    ap.add_argument("--revisions-from", default=None,
                    help="REBUILD ONLY: a dataset_manifest*.json whose source_revision_sha "
                         "pins each axis, so the recorded bytes are read, not today's")
    ap.add_argument("--local-hub", default=None,
                    help="REBUILD ONLY: a hub cache directory holding the recorded snapshots; "
                         "read directly, offline (requires --revisions-from)")
    args = ap.parse_args()
    global FROZEN
    if args.out_dir:
        real_manifests = {(Path(__file__).resolve().parent.parent / "results" / n).resolve()
                          for n in ("dataset_manifest.json", "dataset_manifest_wildguard.json")}
        if Path(args.out).resolve() in real_manifests:
            print("REFUSING: a rebuild (--out-dir) must write its manifest elsewhere (--out)")
            return 2
        FROZEN = Path(args.out_dir).resolve()
        FROZEN.mkdir(parents=True, exist_ok=True)
    pinned: dict[str, str] = {}
    if args.revisions_from:
        for ax, a in json.load(open(args.revisions_from, encoding="utf-8"))["axes"].items():
            pinned[ax] = a["source_revision_sha"]

    # Rule 7 is one-shot PER FILE, not per directory: adding a NEW axis must not
    # be blocked by axes already frozen, and must never touch them.
    clash = [p for ax in args.axes for p in sorted(FROZEN.glob(f"{ax}__*.jsonl"))]
    if clash and not args.force:
        print("REFUSING TO OVERWRITE. These requested axes are already frozen:")
        for p in clash:
            print(f"  {p.name}  sha256={sha256_file(p)}")
        print("\nRule 7 makes the freeze one-shot. Re-run with --axes naming only NEW")
        print("axes, or pass --force to invalidate every result citing these hashes.")
        return 2

    manifest: dict[str, Any] = {
        **provenance_block("dataset_freeze"),
        "seed": SEED,
        "spec_hash": canonical_hash(SPEC),
        "axes": {},
    }

    for axis in args.axes:
        spec = SPEC[axis]
        print(f"\n=== {axis} <- {spec['source']} ===")
        records, rev, stats = load_axis(axis, spec, revision=pinned.get(axis), local_hub=args.local_hub)
        n_pos = sum(r["label"] == 1 for r in records)
        print(f"  rows in source: {stats['rows_in']}")
        for k, v in stats.items():
            if k != "rows_in" and v:
                print(f"    dropped {k.replace('dropped_', ''):<20} {v}")
        print(f"  usable after cleaning: {len(records)}  (pos={n_pos}, neg={len(records) - n_pos})")
        print(f"  source revision sha: {rev}")

        parts = partition(records, spec["partitions"], SEED)
        axis_entry: dict[str, Any] = {
            "source": spec["source"],
            "source_revision_sha": rev,
            "config": spec["config"],
            "split": spec["split"],
            "cleaning": stats,
            "total_usable_rows": len(records),
            "partitions": {},
        }
        seen: set[str] = set()
        for name, rows in parts.items():
            path = FROZEN / f"{axis}__{name}.jsonl"
            digest = write_jsonl(path, rows)
            ids = {r["item_id"] for r in rows}
            overlap = seen & ids
            if overlap:
                raise RuntimeError(f"partition leak: {len(overlap)} ids repeat in {name}")
            seen |= ids
            npos = sum(r["label"] == 1 for r in rows)
            axis_entry["partitions"][name] = {
                "file": str(path.relative_to(FROZEN.parent.parent)),
                "sha256": digest,
                "n": len(rows),
                "n_pos": npos,
                "n_neg": len(rows) - npos,
            }
            print(f"  {name:<16} n={len(rows):<6} pos={npos:<5} neg={len(rows) - npos:<5} sha256={digest[:16]}...")
        manifest["axes"][axis] = axis_entry

    digest = write_results(args.out, manifest)
    print(f"\nWritten: {args.out}\nsha256:  {digest}")
    print("\nThese hashes are now load-bearing. Every run must record them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
