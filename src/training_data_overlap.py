"""Would training on a candidate dataset contaminate the frozen evaluation splits?

Needed before a SAFETY LoRA can be trained. Safety fine-tuning corpora and safety
benchmarks are frequently built from the same red-teaming sources - Aegis 2.0's
prompts derive in part from Anthropic HH-RLHF, for instance - so an update trained
on such data could have seen evaluation prompts, which would confound any change
in the probe with memorisation of the eval set.

Prompts are hashed exactly as freeze_splits.py hashes them, sha256(stripped
text)[:16], and compared against every frozen item_id on every axis. Exact-match
overlap is a LOWER BOUND on contamination: paraphrases are not detected, which is
recorded as a limitation, not ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import ensure_utf8_stdout, provenance_block, read_jsonl, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "data" / "frozen"


def first_human_turn(dialog: str) -> str:
    """Extract the first Human turn from an HH-RLHF transcript."""
    marker_h, marker_a = "Human:", "Assistant:"
    i = dialog.find(marker_h)
    if i < 0:
        return dialog.strip()
    rest = dialog[i + len(marker_h):]
    j = rest.find(marker_a)
    return (rest[:j] if j >= 0 else rest).strip()


SOURCES: dict[str, dict[str, Any]] = {
    "hh-rlhf-harmless": {"repo": "Anthropic/hh-rlhf", "data_dir": "harmless-base", "split": "train",
                         "extract": lambda r: first_human_turn(r["chosen"])},
    "pku-saferlhf": {"repo": "PKU-Alignment/PKU-SafeRLHF", "data_dir": None, "split": "train",
                     "extract": lambda r: r["prompt"]},
}


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Exact-match overlap of training prompts with frozen splits")
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi

    frozen: dict[str, set[str]] = {}
    for p in sorted(FROZEN.glob("*.jsonl")):
        axis = p.name.split("__")[0]
        frozen.setdefault(axis, set()).update(r["item_id"] for r in read_jsonl(p))

    report: dict[str, Any] = {}
    for name in args.sources:
        spec = SOURCES[name]
        entry: dict[str, Any] = {"repo": spec["repo"], "data_dir": spec["data_dir"]}
        try:
            kw = {"split": spec["split"]}
            if spec["data_dir"]:
                kw["data_dir"] = spec["data_dir"]
            ds = load_dataset(spec["repo"], **kw)
            entry["revision"] = HfApi().dataset_info(spec["repo"]).sha
            prompts = [spec["extract"](r) for r in ds]
            ids = {hashlib.sha256(t.strip().encode("utf-8")).hexdigest()[:16] for t in prompts if t.strip()}
            entry["n_rows"] = len(ds)
            entry["n_unique_prompts"] = len(ids)
            entry["overlap_by_axis"] = {ax: len(ids & s) for ax, s in frozen.items()}
            entry["accessible"] = True
        except Exception as exc:
            entry["accessible"] = False
            entry["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        report[name] = entry
        if entry["accessible"]:
            print(f"{name:<18} rows {entry['n_rows']:>7}  unique prompts {entry['n_unique_prompts']:>7}  "
                  f"exact overlap with frozen: {entry['overlap_by_axis']}")
        else:
            print(f"{name:<18} NOT ACCESSIBLE: {entry['error']}")

    payload = {**provenance_block("training_data_overlap"), "limitation":
               "exact-match only; paraphrased or lightly edited prompts are not detected",
               "frozen_axis_sizes": {ax: len(s) for ax, s in frozen.items()}, "sources": report}
    out = ROOT / "results" / "training_data_overlap.json"
    digest = write_results(out, payload)
    print(f"\nWritten: {out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
