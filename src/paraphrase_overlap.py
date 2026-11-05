"""Near-duplicate overlap between training corpora and the frozen evaluation splits (D-107).

D-080 removed training examples whose prompt is VERBATIM in a frozen split. Paraphrases
survive that filter, and if a safety corpus carries many lightly-edited copies of our
evaluation prompts while the general corpus does not, the safety-versus-general damage
gap (D-098, D-105) would be partly contamination rather than update intent.

Method, deliberately model-free so it cannot smuggle in a representation:
  normalise -> word 3-gram shingles -> MinHash (64 permutations) -> LSH banding for
  candidates -> EXACT Jaccard on the candidates, so reported numbers are exact, not
  estimated. For every frozen prompt we keep the maximum Jaccard against the corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import ensure_utf8_stdout, provenance_block, read_jsonl, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "data" / "frozen"
N_PERM, N_BANDS, SHINGLE = 64, 16, 3
MERSENNE = (1 << 61) - 1  # prime modulus; inputs are 32-bit so products fit in uint64


def normalise(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", t.lower())).strip()


def shingles(t: str) -> set[int]:
    w = normalise(t).split()
    if len(w) < SHINGLE:
        return {hash_str(" ".join(w))} if w else set()
    return {hash_str(" ".join(w[i:i + SHINGLE])) for i in range(len(w) - SHINGLE + 1)}


def hash_str(s: str) -> int:
    # 32-bit: keeps a*x + b inside uint64 without overflow in the MinHash step
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "big")


def signatures(docs: list[set[int]], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    sig = np.full((len(docs), N_PERM), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, sh in enumerate(docs):
        if not sh:
            continue
        x = np.fromiter(sh, dtype=np.uint64, count=len(sh))
        h = ((a[None, :] * x[:, None] + b[None, :]) % MERSENNE)
        sig[i] = h.min(axis=0)
    return sig


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Paraphrase overlap, training corpora vs frozen splits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "paraphrase_overlap.json"))
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi

    rng = np.random.default_rng(args.seed)
    a = rng.integers(1, 1 << 31, size=N_PERM, dtype=np.uint64)
    b = rng.integers(0, 1 << 31, size=N_PERM, dtype=np.uint64)

    # frozen prompts, by axis
    axes: dict[str, list[str]] = defaultdict(list)
    for p in sorted(FROZEN.glob("*.jsonl")):
        axis = p.name.split("__")[0]
        axes[axis].extend(r["text"] for r in read_jsonl(p))
    print({k: len(v) for k, v in axes.items()})

    corpora = {
        "alpaca (general)": ("yahma/alpaca-cleaned", None, "train", lambda r: r["instruction"]),
        "hh-rlhf harmless (safety)": ("Anthropic/hh-rlhf", "harmless-base", "train",
                                      lambda r: first_human(r["chosen"])),
        "pku-saferlhf (safety)": ("PKU-Alignment/PKU-SafeRLHF", None, "train", lambda r: r["prompt"]),
    }

    report: dict[str, Any] = {}
    for label, (repo, ddir, split, get) in corpora.items():
        kw = {"split": split}
        if ddir:
            kw["data_dir"] = ddir
        ds = load_dataset(repo, **kw)
        rev = HfApi().dataset_info(repo).sha
        texts = [get(r) for r in ds]
        texts = [t for t in texts if t and t.strip()]
        print(f"\n{label}: {len(texts)} prompts (revision {rev[:12]})", flush=True)
        c_sh = [shingles(t) for t in texts]
        c_sig = signatures(c_sh, a, b)
        rows_per_band = N_PERM // N_BANDS
        buckets: list[dict[bytes, list[int]]] = [defaultdict(list) for _ in range(N_BANDS)]
        for i in range(len(c_sh)):
            for bi in range(N_BANDS):
                key = c_sig[i, bi * rows_per_band:(bi + 1) * rows_per_band].tobytes()
                buckets[bi][key].append(i)

        entry: dict[str, Any] = {"repo": repo, "data_dir": ddir, "revision": rev,
                                 "n_train_prompts": len(texts)}
        for axis, prompts in axes.items():
            f_sh = [shingles(t) for t in prompts]
            f_sig = signatures(f_sh, a, b)
            best = np.zeros(len(prompts))
            n_exact = 0
            exact_set = {hash_str(normalise(t)) for t in texts}
            for j, sh in enumerate(f_sh):
                if hash_str(normalise(prompts[j])) in exact_set:
                    n_exact += 1
                cand: set[int] = set()
                for bi in range(N_BANDS):
                    key = f_sig[j, bi * rows_per_band:(bi + 1) * rows_per_band].tobytes()
                    cand.update(buckets[bi].get(key, ()))
                for i in cand:  # exact Jaccard on candidates only
                    if not sh and not c_sh[i]:
                        continue
                    inter = len(sh & c_sh[i])
                    if inter:
                        j_sim = inter / len(sh | c_sh[i])
                        if j_sim > best[j]:
                            best[j] = j_sim
            entry[axis] = {
                "n_frozen_prompts": len(prompts),
                "exact_normalised_matches": n_exact,
                "frac_jaccard_ge_0.5": float((best >= 0.5).mean()),
                "frac_jaccard_ge_0.8": float((best >= 0.8).mean()),
                "median_max_jaccard": float(np.median(best)),
                "p95_max_jaccard": float(np.quantile(best, 0.95)),
            }
            e = entry[axis]
            print(f"   {axis:<18} exact {n_exact:>5}  >=0.5 {e['frac_jaccard_ge_0.5']:>7.2%}  "
                  f">=0.8 {e['frac_jaccard_ge_0.8']:>7.2%}  median max J {e['median_max_jaccard']:.3f}",
                  flush=True)
        report[label] = entry

    # D-107 rule: compare the safety corpora against alpaca on the safety axis
    key = "safety"
    base = report["alpaca (general)"][key]["frac_jaccard_ge_0.5"]
    gaps = {lab: report[lab][key]["frac_jaccard_ge_0.5"] - base
            for lab in report if lab != "alpaca (general)"}
    confound = any(g > 0.05 for g in gaps.values())
    verdict = ("PARAPHRASE CONTAMINATION IS A LIVE CONFOUND (rule 1)" if confound
               else "R9 CLOSED for this comparison: all corpora within 5 points (rule 2)")
    print(f"\n  safety-axis >=0.5 rates: alpaca {base:.2%}, " +
          ", ".join(f"{lab.split()[0]} {report[lab][key]['frac_jaccard_ge_0.5']:.2%}" for lab in gaps))
    print(f"  VERDICT (D-107): {verdict}")

    payload = {**provenance_block("paraphrase_overlap"), "preregistration": "D-107",
               "params": {"n_perm": N_PERM, "n_bands": N_BANDS, "shingle": SHINGLE, "seed": args.seed},
               "corpora": report, "safety_axis_gap_vs_alpaca": gaps, "verdict": verdict}
    digest = write_results(args.out, payload)
    print(f"\nWritten: {args.out}\nsha256:  {digest}")
    return 0


def first_human(dialog: str) -> str:
    i = dialog.find("Human:")
    if i < 0:
        return dialog.strip()
    rest = dialog[i + len("Human:"):]
    j = rest.find("Assistant:")
    return (rest[:j] if j >= 0 else rest).strip()


if __name__ == "__main__":
    raise SystemExit(main())
