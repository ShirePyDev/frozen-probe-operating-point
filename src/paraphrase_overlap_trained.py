"""Overlap with the examples a run ACTUALLY trained on, not with the whole corpus (D-108).

results/paraphrase_overlap.json measures each corpus against the frozen splits, and for
the safety corpora most of that overlap is EXACT matches - which train_lora.py excluded
before training (--exclude-frozen). The quantity that can actually confound the
safety-versus-general comparison is the overlap with the 6400 examples the run consumed
AFTER that exclusion.

Those examples are reproducible exactly: train_lora.py permutes the dataset with
numpy default_rng(seed), drops every example whose prompt hashes to a frozen item_id,
and consumes the first n_needed of what remains. This script replays that selection from
the training record's own config and measures near-duplicate overlap for the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paraphrase_overlap import N_BANDS, N_PERM, hash_str, normalise, shingles, signatures  # noqa: E402
from provenance import ensure_utf8_stdout, provenance_block, read_jsonl, write_results  # noqa: E402
from train_lora import text_and_prompt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "data" / "frozen"
RES = ROOT / "results"


def frozen_item_ids() -> set[str]:
    ids: set[str] = set()
    for p in FROZEN.glob("*.jsonl"):
        ids.update(r["item_id"] for r in read_jsonl(p))
    return ids


def replay(record: dict[str, Any]) -> list[str]:
    """Reproduce the exact prompts a training run consumed."""
    from datasets import load_dataset

    c = record["config"]
    ds = (load_dataset(c["dataset"], data_dir=c["data_dir"], split="train") if c.get("data_dir")
          else load_dataset(c["dataset"], split="train"))
    rng = np.random.default_rng(c["seed"])
    order = [int(i) for i in rng.permutation(len(ds))]
    frozen = frozen_item_ids()
    fmt = c.get("format", "alpaca")   # the first (alpaca) run predates the --format flag

    def phash(i: int) -> str:
        return hashlib.sha256(text_and_prompt(ds[i], fmt)[1].encode("utf-8")).hexdigest()[:16]

    if c.get("exclude_frozen"):
        order = [i for i in order if phash(i) not in frozen]
    n_needed = max(c["save_steps"]) * c["micro_bs"] * c["grad_accum"]
    used = order[:n_needed]
    return [text_and_prompt(ds[i], fmt)[1] for i in used]


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Overlap with the actually-trained examples (D-108)")
    ap.add_argument("--records", nargs="+", default=["mistral7b-v01-alpaca-r16",
                                                    "mistral7b-v01-hhharmless-r16",
                                                    "mistral7b-v01-pkusafe-r16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RES / "paraphrase_overlap_trained.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    a = rng.integers(1, 1 << 31, size=N_PERM, dtype=np.uint64)
    b = rng.integers(0, 1 << 31, size=N_PERM, dtype=np.uint64)

    axes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in sorted(FROZEN.glob("*.jsonl")):
        axis = p.name.split("__")[0]
        axes[axis].extend({"item_id": r["item_id"], "text": r["text"], "label": r["label"]}
                          for r in read_jsonl(p))

    report: dict[str, Any] = {}
    per_item: dict[str, Any] = {}
    for name in args.records:
        rec_path = RES / f"lora_train__{name}.json"
        if not rec_path.exists():
            print(f"  skip {name}: no training record yet")
            continue
        rec = json.load(open(rec_path, encoding="utf-8"))
        prompts = replay(rec)
        print(f"\n{name}: replayed {len(prompts)} trained prompts "
              f"(record says {rec['n_examples_used']}, excluded {rec.get('n_excluded_for_frozen_overlap')})",
              flush=True)
        c_sh = [shingles(t) for t in prompts]
        c_sig = signatures(c_sh, a, b)
        rpb = N_PERM // N_BANDS
        buckets: list[dict[bytes, list[int]]] = [defaultdict(list) for _ in range(N_BANDS)]
        for i in range(len(c_sh)):
            for bi in range(N_BANDS):
                buckets[bi][c_sig[i, bi * rpb:(bi + 1) * rpb].tobytes()].append(i)
        exact_set = {hash_str(normalise(t)) for t in prompts}

        entry: dict[str, Any] = {"n_trained_prompts": len(prompts),
                                 "dataset": rec["config"]["dataset"],
                                 "n_examples_used_recorded": rec["n_examples_used"]}
        for axis, items in axes.items():
            f_sh = [shingles(it["text"]) for it in items]
            f_sig = signatures(f_sh, a, b)
            best = np.zeros(len(items))
            n_exact = 0
            for j, sh in enumerate(f_sh):
                if hash_str(normalise(items[j]["text"])) in exact_set:
                    n_exact += 1
                cand: set[int] = set()
                for bi in range(N_BANDS):
                    cand.update(buckets[bi].get(f_sig[j, bi * rpb:(bi + 1) * rpb].tobytes(), ()))
                for i in cand:
                    inter = len(sh & c_sh[i])
                    if inter:
                        sim = inter / len(sh | c_sh[i])
                        if sim > best[j]:
                            best[j] = sim
            neg = np.array([not it["label"] for it in items])
            entry[axis] = {
                "n_frozen_prompts": len(items),
                "exact_normalised_matches": n_exact,
                "frac_jaccard_ge_0.5": float((best >= 0.5).mean()),
                "frac_jaccard_ge_0.8": float((best >= 0.8).mean()),
                "frac_benign_jaccard_ge_0.5": float((best[neg] >= 0.5).mean()) if neg.any() else None,
                "p95_max_jaccard": float(np.quantile(best, 0.95)),
            }
            e = entry[axis]
            print(f"   {axis:<17} exact {n_exact:<5} >=0.5 {e['frac_jaccard_ge_0.5']:>7.2%}  "
                  f">=0.8 {e['frac_jaccard_ge_0.8']:>7.2%}  benign-only >=0.5 "
                  f"{(e['frac_benign_jaccard_ge_0.5'] or 0):>7.2%}", flush=True)
            if axis == "safety":
                per_item[name] = {items[j]["item_id"]: float(best[j]) for j in range(len(items))}
        report[name] = entry

    base_key = next((k for k in report if "alpaca" in k), None)
    verdict = "NOT EVALUATED"
    if base_key:
        base = report[base_key]["safety"]["frac_jaccard_ge_0.5"]
        gaps = {k: report[k]["safety"]["frac_jaccard_ge_0.5"] - base for k in report if k != base_key}
        verdict = ("CONFOUND REMAINS AFTER EXCLUSION (gap > 5 points)" if any(g > 0.05 for g in gaps.values())
                   else "CONFOUND REMOVED BY EXCLUSION: all arms within 5 points of the general arm")
        print(f"\n  safety-axis >=0.5 after exclusion: {base_key} {base:.2%}, " +
              ", ".join(f"{k} {report[k]['safety']['frac_jaccard_ge_0.5']:.2%}" for k in gaps))
        print(f"  VERDICT: {verdict}")

    payload = {**provenance_block("paraphrase_overlap_trained"), "preregistration": "D-107",
               "arms": report, "verdict": verdict,
               "note": "overlap measured against the examples actually consumed by each run, "
                       "replayed from its recorded seed, filter and dose"}
    digest = write_results(args.out, payload)
    (RES / "paraphrase_per_item_safety.json").write_text(json.dumps(per_item), encoding="utf-8")
    print(f"\nWritten: {args.out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
