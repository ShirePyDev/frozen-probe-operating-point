"""Standing access audit: what can this token actually DOWNLOAD right now?

Written as a reusable script rather than an ad-hoc command because gate status
changes over time (manual review queues) and the answer decides what can be run
next.

Probe discipline (FAILURES.md F-012)
------------------------------------
Access is tested by downloading a file the gate ACTUALLY protects:
  models   -> config.json
  datasets -> a real .parquet / .jsonl / .csv from the repo listing
NOT README.md, which HuggingFace serves to everyone even on gated repos. An
earlier version of this check used README.md and reported WildGuardMix as
accessible when it was not.

Each entry also records WHAT IT UNLOCKS, so the output is a decision aid rather
than a list of booleans.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import (  # noqa: E402
    ensure_utf8_stdout,
    provenance_block,
    write_results,
)

# (kind, repo_id, role, unlocks)
TARGETS: list[tuple[str, str, str, str]] = [
    # --- previously blocked, re-checking ---
    ("model", "meta-llama/Meta-Llama-3-8B", "base", "Llama-3 -> 3.1 version pair (same width)"),
    ("model", "meta-llama/Meta-Llama-3-8B-Instruct", "instruct", "Llama-3 base->instruct pair"),
    ("model", "google/gemma-2-9b", "base", "Gemma-2 base->instruct pair"),
    ("model", "google/gemma-2-9b-it", "instruct", "Gemma-2 base->instruct pair"),
    ("model", "google/gemma-3-12b-pt", "base", "Gemma-2 -> 3 (caveated, size change)"),
    ("model", "google/gemma-3-12b-it", "instruct", "Gemma-3 base->instruct pair"),
    ("dataset", "allenai/wildguardmix", "safety axis", "the charter's primary safety dataset"),
    # --- known accessible, confirming still so ---
    ("model", "meta-llama/Llama-3.1-8B", "base", "Llama-3.1 base->instruct pair (3rd family)"),
    ("model", "meta-llama/Llama-3.1-8B-Instruct", "instruct", "Llama-3.1 base->instruct pair (3rd family)"),
    # --- ungated candidates for further base->instruct replication ---
    ("model", "allenai/OLMo-2-1124-7B", "base", "OLMo-2 base->instruct pair (4th family)"),
    ("model", "allenai/OLMo-2-1124-7B-Instruct", "instruct", "OLMo-2 base->instruct pair (4th family)"),
    ("model", "Qwen/Qwen3-8B-Base", "base", "Qwen3 base->instruct pair (5th family)"),
    ("model", "Qwen/Qwen3-8B", "instruct", "Qwen3 base->instruct pair (5th family)"),
    ("model", "microsoft/Phi-3-mini-4k-instruct", "instruct", "Phi-3 (instruct only, no base released)"),
    ("model", "microsoft/Phi-3.5-mini-instruct", "instruct", "Phi-3 -> 3.5 instruct-to-instruct pair"),
    # --- already in use, regression check ---
    ("model", "mistralai/Mistral-7B-v0.1", "base", "in use: Mistral base->instruct"),
    ("model", "mistralai/Mistral-7B-Instruct-v0.1", "instruct", "in use: Mistral base->instruct"),
    ("model", "Qwen/Qwen2.5-7B", "base", "in use: Qwen2.5 base->instruct"),
    ("model", "Qwen/Qwen2.5-7B-Instruct", "instruct", "in use: Qwen2.5 base->instruct"),
    ("dataset", "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "safety axis", "in use: frozen splits"),
    ("dataset", "natolambert/xstest-v2-copy", "over-refusal axis", "in use: frozen splits"),
]


def probe(kind: str, repo_id: str, cache_dir: str) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    rec: dict[str, Any] = {"kind": kind, "repo_id": repo_id, "accessible": None,
                           "probe_file": None, "error": None, "gated": None}
    try:
        info = (api.model_info(repo_id) if kind == "model"
                else api.dataset_info(repo_id))
        rec["gated"] = getattr(info, "gated", None)
    except Exception as exc:
        rec["error"] = f"metadata: {type(exc).__name__}"

    try:
        if kind == "model":
            target = "config.json"
        else:
            files = api.list_repo_files(repo_id, repo_type="dataset")
            data = [f for f in files if f.endswith((".parquet", ".jsonl", ".csv", ".json"))
                    and "README" not in f]
            if not data:
                rec["accessible"] = None
                rec["error"] = "no data file found to probe"
                return rec
            target = sorted(data)[0]
        rec["probe_file"] = target
        hf_hub_download(repo_id, target,
                        repo_type=None if kind == "model" else "dataset",
                        cache_dir=cache_dir)
        rec["accessible"] = True
    except Exception as exc:
        rec["accessible"] = False
        rec["error"] = type(exc).__name__
    return rec


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Audit what the current token can download")
    ap.add_argument("--cache-dir", default="/tmp/access_probe")
    ap.add_argument("--out", default=str(
        Path(__file__).resolve().parent.parent / "results" / "access_check.json"))
    args = ap.parse_args()

    try:
        from huggingface_hub import get_token, whoami

        tok = get_token()
        who = whoami(token=tok).get("name") if tok else None
    except Exception:
        tok, who = None, None
    print(f"token present: {bool(tok)}   authenticated as: {who}\n")

    rows = []
    for kind, repo, role, unlocks in TARGETS:
        rec = probe(kind, repo, args.cache_dir)
        rec.update({"role": role, "unlocks": unlocks})
        rows.append(rec)
        mark = "OK      " if rec["accessible"] else ("BLOCKED " if rec["accessible"] is False else "UNKNOWN ")
        err = f"  ({rec['error']})" if rec["error"] and not rec["accessible"] else ""
        print(f"  {mark} {kind:<7} {repo:<44} gated={str(rec['gated']):<6}{err}")

    ok = {r["repo_id"] for r in rows if r["accessible"]}

    # A base->instruct pair is only usable if BOTH sides are accessible.
    pairs = [
        ("Mistral-7B v0.1", "mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.1"),
        ("Qwen2.5-7B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"),
        ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"),
        ("OLMo-2-7B", "allenai/OLMo-2-1124-7B", "allenai/OLMo-2-1124-7B-Instruct"),
        ("Qwen3-8B", "Qwen/Qwen3-8B-Base", "Qwen/Qwen3-8B"),
        ("Llama-3-8B", "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct"),
        ("Gemma-2-9B", "google/gemma-2-9b", "google/gemma-2-9b-it"),
        ("Gemma-3-12B", "google/gemma-3-12b-pt", "google/gemma-3-12b-it"),
    ]
    usable = [(n, b, i) for n, b, i in pairs if b in ok and i in ok]
    blocked = [(n, b, i) for n, b, i in pairs if not (b in ok and i in ok)]

    print("\n" + "=" * 78)
    print("BASE -> INSTRUCT PAIRS (the current experiment needs BOTH sides)")
    print("=" * 78)
    for n, b, i in usable:
        print(f"  USABLE   {n}")
    for n, b, i in blocked:
        miss = [x for x in (b, i) if x not in ok]
        print(f"  blocked  {n:<16} missing: {', '.join(miss)}")
    print(f"\n  usable base->instruct families: {len(usable)} of {len(pairs)}")

    payload = {
        **provenance_block("access_check"),
        "authenticated_as": who,
        "summary": {
            "accessible": sum(1 for r in rows if r["accessible"]),
            "blocked": sum(1 for r in rows if r["accessible"] is False),
            "total": len(rows),
            "usable_base_instruct_pairs": [n for n, _, _ in usable],
            "blocked_base_instruct_pairs": [n for n, _, _ in blocked],
        },
        "targets": rows,
    }
    digest = write_results(args.out, payload)
    print(f"\nWritten: {args.out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
