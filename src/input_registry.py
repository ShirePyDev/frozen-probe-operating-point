"""Where a primary run's inputs come from (D-165): never from a results file D-161 superseded.

  PAIRS / LORA_RUNS   the model pairs, DECLARED here: pre and post cache repo and axis. Transcribed
                      on 2026-09-25 from the run records (results/repair_*.json, now superseded) and
                      verified against the extraction records: every repo named here has exactly one
                      recorded revision.
  DEPTH_ONLY          the one depth pair with no repair run (Mistral v0.1 -> v0.2, D-180), declared
                      and verified the same way on 2026-09-26.
  revision(repo)      the full revision from the extraction records (results/extract__*.json:
                      cache_repo/cache_revision where present - a LoRA arm's adapter sha256, an
                      __a100 re-extraction's tagged repo - else repo/revision). Refuses unless every
                      record of that repo names one and the same revision.
  expected_sha256()   the bytes a cache must have, from every source that names it - the original
                      extraction records and the committed pins (MANIFESTS) - which must agree;
                      refuses when there is none or two disagree.
  is_pin_only(), verification(), PIN_ONLY_FLAG
                      D-170/D-172: a cache no original extraction record lists is vouched for only
                      by its pin; results reading one carry PIN_ONLY_FLAG (per cell, per file).
  sources_sha256()    the sha256 of every extraction record and pin manifest the registry read, so
                      a result can name the exact records its expectations came from.

Only ORIGINAL extraction records are read: a '__rerun_' record (a re-extraction's summary, e.g.
the D-168 V100 rerun) never changes an expectation or clears a pin until the author accepts it
and it is promoted by a decision. The extraction records are read as records of what was
extracted, not as claims (D-161 marked 42 of them; D-162 reads them the same way).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from provenance import resolve_cache, sha256_file

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "hidden"
# Committed manifests of cache bytes, in sha256sum format ("<sha256>  <cache file name>").
# D-170: caches_without_extraction_record.sha256 pins the 16 primary caches whose extraction record
# was lost (F-024) to the bytes the node audit hashed on 2026-09-25; their identity is verified by
# the D-162/D-168 re-extraction.
MANIFESTS = (ROOT / "manifests" / "caches_without_extraction_record.sha256",)
# D-170 as approved (D-172): a result reading such a cache carries this machine-readable flag, the
# provenance audit counts those results, and none enters the paper until D-168 has passed.
PIN_ONLY_FLAG = "input verified by pin only, pending D-168"
# D-300 (3): the D-168 verdicts are made final on 2026-10-03 by src/d168_final.py, which writes D168_FINAL. The flag is
# cleared ONLY when D168_FINAL_SHA256 is pinned here (a commit after that record is pulled) and the record has exactly
# those bytes, names the committed pin manifest byte for byte, and passes every one of its caches. Until then (None)
# nothing changes; a pinned hash whose file is missing or differs refuses.
D168_FINAL = RESULTS / "d168_final_verdicts.json"
D168_FINAL_SHA256: str | None = "1612cd9f783c802abd7b4e59369cb15b5aa5b92a46b29f8667e093caa11f44e7"   # D-331
D168_FINAL_DECISION: str | None = "D-331"

# stem: (pre cache repo, post cache repo, axis)
PAIRS = {
    "repair_mistral_base2instruct": ("mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.1", "safety"),
    "repair_gemma2_base2instruct": ("google/gemma-2-9b", "google/gemma-2-9b-it", "safety"),
    "repair_llama3_base2instruct": ("meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct", "safety"),
    "repair_llama31_base2instruct": ("meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct", "safety"),
    "repair_qwen_base2instruct": ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct", "safety"),
    "repair_olmo_base2instruct": ("allenai/OLMo-2-1124-7B", "allenai/OLMo-2-1124-7B-Instruct", "safety"),
    "repair_llama3_to_31": ("meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-3.1-8B", "safety"),
    "repair_WG_mistral_base2instruct": ("mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.1",
                                        "safety_wildguard"),
    "repair_WG_llama3_base2instruct": ("meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct",
                                       "safety_wildguard"),
}
_A100_BASE = "mistralai/Mistral-7B-v0.1__a100"
LORA_RUNS = {
    "repair_loraA100_mistral_s25": (_A100_BASE, "lora/mistral7b-v01-alpaca-r16-s25__a100", "safety"),
    "repair_loraA100_mistral_s100": (_A100_BASE, "lora/mistral7b-v01-alpaca-r16-s100", "safety"),
    "repair_loraA100_mistral_s400": (_A100_BASE, "lora/mistral7b-v01-alpaca-r16-s400", "safety"),
    "repair_lorasafe_mistral_s25": (_A100_BASE, "lora/mistral7b-v01-hhharmless-r16-s25", "safety"),
    "repair_lorasafe_mistral_s100": (_A100_BASE, "lora/mistral7b-v01-hhharmless-r16-s100", "safety"),
    "repair_lorasafe_mistral_s400": (_A100_BASE, "lora/mistral7b-v01-hhharmless-r16-s400", "safety"),
    "repair_lorapku_mistral_s25": (_A100_BASE, "lora/mistral7b-v01-pkusafe-r16-s25", "safety"),
    "repair_lorapku_mistral_s100": (_A100_BASE, "lora/mistral7b-v01-pkusafe-r16-s100", "safety"),
    "repair_lorapku_mistral_s400": (_A100_BASE, "lora/mistral7b-v01-pkusafe-r16-s400", "safety"),
}
# D-180: the depth curves add one pair with no repair run (Mistral v0.1 -> v0.2, UPDATE_PAIRS_DEPTH).
# Transcribed 2026-09-26 from its depth record's config (pre/post repo and revision, axis) and
# verified against the extraction records: v0.2's three caches are listed in its original record
# with the bytes that record names; the pre side is the same v0.1 caches as the Mistral pairs.
DEPTH_ONLY = {
    "depth_mistral_v01_v02_dense": ("mistralai/Mistral-7B-v0.1", "mistral-community/Mistral-7B-v0.2", "safety"),
    # D-224: the GPU-change control, a no-update hardware null (the same model on the V100 and on the A100)
    "depth_hw_v100_a100": ("mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-v0.1__a100", "safety"),
}
ALL_RUNS = {**PAIRS, **LORA_RUNS, **DEPTH_ONLY}

_records: list[dict[str, Any]] | None = None
_manifest: dict[str, tuple[str, str]] | None = None


def _load_records() -> list[dict[str, Any]]:
    global _records
    if _records is None:
        _records = []
        for f in sorted(RESULTS.glob("extract__*.json")):
            if "__rerun_" in f.name:
                continue          # a rerun never changes an expectation until promoted (docstring)
            e = json.load(open(f, encoding="utf-8"))
            _records.append({"summary": f.name, "sha256": sha256_file(f),
                             "repo": e.get("cache_repo") or e.get("repo"),
                             "revision": e.get("cache_revision") or e.get("revision"),
                             "splits": {s["cache_file"]: s["cache_sha256"] for s in (e.get("splits") or {}).values()}})
    return _records


def _load_manifests() -> dict[str, tuple[str, str]]:
    global _manifest
    if _manifest is None:
        _manifest = {}
        for m in MANIFESTS:
            if not m.exists():
                continue
            for line in m.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                h, name = line.split(None, 1)
                name = name.strip()
                if name in _manifest and _manifest[name][0] != h:
                    raise SystemExit(f"REFUSING: {name} has two different hashes in the committed manifests")
                _manifest[name] = (h, str(m.relative_to(ROOT)).replace("\\", "/"))
    return _manifest


def sources_sha256() -> dict[str, str]:
    """sha256 of every extraction record and committed pin manifest the registry read."""
    out = {f"results/{r['summary']}": r["sha256"] for r in _load_records()}
    for m in MANIFESTS:
        if m.exists():
            out[str(m.relative_to(ROOT)).replace("\\", "/")] = sha256_file(m)
    return out


def revision(repo: str) -> str:
    revs = {r["revision"] for r in _load_records() if r["repo"] == repo}
    if len(revs) != 1:
        raise SystemExit(f"REFUSING: extraction records name {len(revs)} revisions for {repo}: {sorted(revs)}")
    return revs.pop()


def cache_name(repo: str, axis: str, part: str) -> str:
    return f"{repo.replace('/', '__')}@{revision(repo)[:12]}__{axis}__{part}__mean.npz"


def expected_sha256(name: str) -> tuple[str, str]:
    """(sha256, source) for a cache file name, from an extraction record or a committed manifest."""
    rec = {(r["splits"][name], r["summary"]) for r in _load_records() if name in r["splits"]}
    hashes = {h for h, _ in rec}
    man = _load_manifests().get(name)
    if man:
        hashes.add(man[0])
    if len(hashes) != 1:
        raise SystemExit(f"REFUSING: {name}: {'no' if not hashes else 'conflicting'} expected sha256 "
                         f"in the extraction records and committed manifests ({sorted(hashes)})")
    h = hashes.pop()
    src = sorted({f"extraction record {s}" for _, s in rec} | ({f"manifest {man[1]}"} if man else set()))
    return h, "; ".join(src)


def is_pin_only(name: str) -> bool:
    """True when no extraction record lists the cache, so its bytes are vouched for only by a
    committed pin (D-170): its identity awaits the D-168 re-extraction."""
    return not any(name in r["splits"] for r in _load_records())


def load(repo: str, axis: str, part: str) -> tuple[Any, dict[str, Any]]:
    """Open a cache and check its bytes against the registry. Returns (npz, record); the record's
    'pin_only' says whether only a committed pin vouches for it (D-170)."""
    name = cache_name(repo, axis, part)
    p = resolve_cache(name, CACHE)
    if not p.exists():
        raise FileNotFoundError(f"missing cache {name}")
    want, src = expected_sha256(name)
    have = sha256_file(p)
    if have != want:
        raise SystemExit(f"REFUSING: {name} is not the expected bytes ({have[:16]} vs {want[:16]} from {src})")
    return np.load(p, allow_pickle=False), {"file": name, "sha256": have, "expected_from": src,
                                            "pin_only": is_pin_only(name)}


D168_FINAL_TEXT = "PASSES both parts: FINAL"                    # src/d168_final.py's final verdict text begins so
D168_NOT_BEFORE_UTC = "2026-10-02T15:00:00+00:00"                  # 2026-10-03 00:00 at UTC+9 (D-300 (3))


def _d168_record() -> dict[str, Any] | None:
    """The validated final D-168 record's cache names, or None while D168_FINAL_SHA256 is not pinned. Validated afresh
    on every call (a small file; nothing cached that could go stale). Refuses a pinned record that is missing, not the
    pinned bytes, not a d168_final record issued on or after 2026-10-03 (UTC+9), not for the one committed pin manifest
    byte for byte, or not passing every cache with the FINAL text; and a pinned hash without a decision."""
    if D168_FINAL_SHA256 is None:
        return None
    if not D168_FINAL_DECISION:
        raise SystemExit("REFUSING: D168_FINAL_SHA256 is pinned without D168_FINAL_DECISION")
    if len(MANIFESTS) != 1:
        raise SystemExit(f"REFUSING: the final D-168 record covers one pin manifest; MANIFESTS lists {len(MANIFESTS)}")
    if not D168_FINAL.is_file():
        raise SystemExit(f"REFUSING: {D168_FINAL.name} is pinned ({D168_FINAL_SHA256[:16]}) but missing: pull or push "
                         f"results/{D168_FINAL.name} with the commit that pinned it")
    b = D168_FINAL.read_bytes()
    if hashlib.sha256(b).hexdigest() != D168_FINAL_SHA256:
        raise SystemExit(f"REFUSING: {D168_FINAL.name} is not the pinned bytes {D168_FINAL_SHA256[:16]}")
    man = MANIFESTS[0]
    if not man.is_file():
        raise SystemExit(f"REFUSING: the committed pin manifest {man.name} is missing")
    names = {line.split(None, 1)[1].strip() for line in man.read_text(encoding="utf-8").splitlines() if line.strip()}
    try:                                       # any malformed record refuses in words, never with a bare traceback
        rec = json.loads(b.decode("utf-8"))
        caches = rec.get("caches") or {}
        pm = rec.get("pin_manifest") or {}
        issued_ok = datetime.fromisoformat(str(rec.get("issued_utc"))) >= datetime.fromisoformat(D168_NOT_BEFORE_UTC)
        ok = (rec.get("run_kind") == "d168_final" and issued_ok and pm.get("sha256") == sha256_file(man)
              and set(caches) == names and rec.get("all_final_pass") is True
              and all(str(c.get("final_verdict", "")).startswith(D168_FINAL_TEXT) for c in caches.values()))
    except (ValueError, TypeError, AttributeError):
        ok = False
    if not ok:
        raise SystemExit(f"REFUSING: {D168_FINAL.name} is not a final D-168 record issued on or after 2026-10-03 that "
                         f"passes every cache of the committed pin manifest {man.name} byte for byte")
    return {"caches": frozenset(caches)}


def d168_final() -> dict[str, Any] | None:
    """The final D-168 record as {file, sha256, decision, n_caches} when D168_FINAL_SHA256 is pinned and the record
    meets every condition of _d168_record(); None while nothing is pinned."""
    r = _d168_record()
    if r is None:
        return None
    return {"file": f"results/{D168_FINAL.name}", "sha256": D168_FINAL_SHA256, "decision": D168_FINAL_DECISION,
            "n_caches": len(r["caches"])}


def d168_final_caches() -> frozenset[str]:
    """The cache names the pinned final record judged (empty while nothing is pinned)."""
    r = _d168_record()
    return r["caches"] if r else frozenset()


def verification(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The machine-readable D-170 flag for a set of inputs {key: load() record}: 'flag' is PIN_ONLY_FLAG when any input
    is pin-only, unless the pinned final D-168 record judged every pin-only input's cache file (then None, with the
    record named); the pin-only inputs are listed either way."""
    pin = sorted(k for k, r in records.items() if r.get("pin_only"))
    cleared = d168_final() if pin else None
    if cleared and not all(records[k].get("file") in d168_final_caches() for k in pin):
        cleared = None                         # a pin-only cache the final record did not judge keeps the flag
    # unchanged output while nothing is cleared (re-issues compare this block key for key, D-305)
    return {"flag": PIN_ONLY_FLAG if pin and cleared is None else None, "pin_only_inputs": pin,
            **({"pin_flag_cleared_by": cleared} if cleared else {}),
            "rule": "D-170/D-172: a result with a flag enters no paper table until D-168 has passed"}


def run(stem: str) -> dict[str, str]:
    """The declared run: pre/post repo and revision, and the axis."""
    if stem not in ALL_RUNS:
        raise SystemExit(f"REFUSING: {stem} is not a declared run in src/input_registry.py")
    pre, post, axis = ALL_RUNS[stem]
    return {"pre_repo": pre, "pre_rev": revision(pre), "post_repo": post, "post_rev": revision(post), "axis": axis}
