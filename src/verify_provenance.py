"""Provenance audit: can every reported result still be traced to the exact inputs it read?

Why this exists (FAILURES.md F-024)
-----------------------------------
extract_hidden.py named its summary file by model only, `extract__{repo}@{rev}.json`.
Extracting the same model again on a different axis, or through the chat-template
arm, silently OVERWROTE the earlier summary. Verified on 2026-09-16: the summaries
for Meta-Llama-3-8B, Meta-Llama-3-8B-Instruct, Mistral-7B-v0.1 and
Mistral-7B-Instruct-v0.1 list only their WildGuard/over-refusal splits although
their safety caches are on disk, and Qwen2.5-7B-Instruct's lists only the chat arm.

Two separate questions follow, and only the second threatens the science:

1. Is the per-cache record recoverable?  Yes - each .npz stores its own item_ids,
   labels, nonfinite and truncated flags. This script rebuilds a manifest from the
   caches themselves.

2. Was any cache overwritten AFTER an analysis used it?  Caches ARE keyed by axis
   and partition, but a denser re-extraction of the same model and axis writes the
   same filename. Every analysis records `input_cache_sha256`. If a recorded hash
   no longer matches any cache on disk, that result can no longer be reproduced from
   disk and must be labelled as such - it is not automatically wrong, but it is no
   longer verifiable.

Reads caches, results and the committed pin manifest (D-170). Writes one audit file.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import ensure_utf8_stdout, provenance_block, sha256_file, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "hidden"
RESULTS = ROOT / "results"
# D-256 (9), D-252, D-286: the extraction records with an empty git block, audited from the node's run logs
# D-308: the extended audit (D-286's 24 commit-less records, identical, plus the 7 D-168 A100 re-run records)
EXTRACTION_COMMIT_AUDIT = RESULTS / "extraction_commit_audit_r2.json"
EXTRACTION_COMMIT_AUDIT_SHA256 = "6aad4012dddde3fc2094cdb9ffed71151b40d01f2be3ff5d7db87427c302df96"


def cache_record(path: Path) -> dict[str, Any]:
    rec: dict[str, Any] = {"file": path.name, "bytes": path.stat().st_size,
                           "sha256": sha256_file(path)}
    try:
        with np.load(path, allow_pickle=False) as z:
            layers = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("layer_"))
            nf = z["nonfinite"] if "nonfinite" in z.files else None
            rec.update({
                "n_items": int(len(z["item_ids"])),
                "layers": layers,
                "n_nonfinite": int(nf.sum()) if nf is not None else None,
                # Which items were replaced by nan_to_num at extraction (F-016 addendum).
                "nonfinite_item_indices": np.flatnonzero(nf).tolist() if nf is not None else None,
                "n_truncated": int(z["truncated"].sum()) if "truncated" in z.files else None,
                "readable": True,
            })
    except Exception as exc:  # a corrupt cache is itself a finding
        rec.update({"readable": False, "error": f"{type(exc).__name__}: {exc}"})
    return rec


# F-035 (author, 2026-09-25, D-162): bytes recorded only under ANOTHER model's extraction
# satisfy the revision check only through an identity documented in the ledger, matched here
# by explicit conditions. Any other such alias is undocumented and fails like a mismatch.
DOCUMENTED_ALIASES = (
    {"id": "D-081 step-0 gate",
     "doc": "D-081: the step-0 LoRA adapter reproduces the base model's cache bit-for-bit",
     "match": lambda repo, rev, r: re.fullmatch(r"lora/.+-s0", r["repo"] or "") is not None
     and r["base_repo"] == repo and r["base_revision"] == rev},
    {"id": "D-034 v0.2 = v0.3",
     "doc": "D-034: all 289 transformer body tensors are bit-identical between Mistral-7B v0.2 and v0.3",
     "match": lambda repo, rev, r: (repo, rev) == ("mistral-community/Mistral-7B-v0.2",
                                                   "80cf49b18de6354fd8a6d1e81c177dd830e05eea")
     and (r["repo"], r["revision"]) == ("mistralai/Mistral-7B-v0.3", "caa1feb0e54d415e2df31207e5f4e273e33509b1")},
)
FAILING_REVISION_STATUSES = ("MISMATCH", "UNDOCUMENTED_ALIAS")
TAG = re.compile(r"[A-Za-z0-9]+")
_SNAPSHOT: list[Path] | None = None          # D-297 (1): the results files main() listed; every pass reads only these


def phase2_dir() -> Path:
    """D-303 (12): Phase 2 records live in results/phase2/ and the audit reads them too (computed at call time, so a
    test that repoints RESULTS repoints this as well)."""
    return RESULTS / "phase2"


def result_path(ref: str) -> Path:
    """The results file a record names as its source. Only an explicit 'phase2/<stem>' (how Phase 2 records name a
    Phase 2 source) is results/phase2/<stem>.json; any other reference is read exactly as before, by file name in
    results/ (no fallback, so a Phase 2 file can never change a Phase 1 resolution)."""
    r = str(ref).replace("\\", "/").removesuffix(".json").removeprefix("results/")
    if r.startswith("phase2/"):
        return RESULTS / f"{r}.json"
    return RESULTS / f"{r.rsplit('/', 1)[-1]}.json"


def rel_name(p: Path) -> str:
    """A results file's name relative to results/ ('x.json', or 'phase2/x.json' for a Phase 2 record)."""
    p = Path(p)
    return p.relative_to(RESULTS).as_posix() if p.is_relative_to(RESULTS) else p.name


def results_files(pattern: str = "*.json", top_only: bool = False) -> list[Path]:
    """As below; top_only keeps results/ itself (the Phase 1 checks: the registry, D-286)."""
    return [p for p in _results_files(pattern) if not top_only or p.parent == RESULTS]


def _results_files(pattern: str = "*.json") -> list[Path]:
    """The results files a pass reads: main()'s snapshot when set (so a record written during the audit is never read
    unlisted), else the directory as it is now (the functions used on their own, as in the tests)."""
    if _SNAPSHOT is not None:
        return [p for p in _SNAPSHOT if fnmatch.fnmatch(p.name, pattern)]
    return sorted(RESULTS.glob(pattern)) + sorted(phase2_dir().glob(pattern))


def readable_result(path: Path) -> bool:
    """A results file a pass may read by path: in main()'s snapshot when set (a file written after it is absent),
    else on disk (the functions used on their own)."""
    if _SNAPSHOT is not None:
        return path in _SNAPSHOT
    return path.exists()


def pin_flag_status(d: dict, pin_only_caches: list[str], final_sha: str | None,
                    final_caches: frozenset[str]) -> tuple[bool, bool]:
    """(carries the D-170 flag, cleared by the final D-168 record) for one result that read the pin-only caches named.
    Cleared only when a final record is pinned (final_sha), the result names exactly that sha256 in its
    input_verification.pin_flag_cleared_by, and every pin-only cache it read is one the record judged (D-300 (3))."""
    import input_registry as REG                       # imported where used, as main() does
    iv = d.get("input_verification") if isinstance(d.get("input_verification"), dict) else {}
    cb = iv.get("pin_flag_cleared_by") if isinstance(iv.get("pin_flag_cleared_by"), dict) else {}
    cleared = final_sha is not None and cb.get("sha256") == final_sha and set(pin_only_caches) <= set(final_caches)
    return iv.get("flag") == REG.PIN_ONLY_FLAG, cleared


def documented_tag(r: dict) -> str | None:
    """D-297 (1): the tag an extraction record documents for its cache, or None. A record documents a tag when
    its cache identity is its model's repo plus '__<tag>' (the D-252 xsfull and the __a100 caches) and its cache
    revision is a prefix of the model's full revision. Nothing else may stand between a repo and '@'."""
    repo, base, rev, base_rev = r.get("repo"), r.get("base_repo"), r.get("revision"), r.get("base_revision")
    if not (repo and base and rev and base_rev) or not repo.startswith(base + "__"):
        return None
    tag = repo[len(base) + 2:]
    return tag if TAG.fullmatch(tag) and base_rev.startswith(rev) else None


def single_model_repo_rev(cfg: dict) -> tuple[str | None, str | None]:
    """D-297 (1): a single-model record names its one model as config.repo / config.rev, read as the pre side;
    only when no side-specific repo or revision is named."""
    if "rev" in cfg and not any(k in cfg for k in ("pre_repo", "post_repo", "pre_rev", "post_rev")):
        return cfg.get("repo"), cfg["rev"]
    return None, None


def extraction_index() -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Every extraction record in results/extract__*.json, by cache sha256 and by file name.

    The identity a record gives a cache is cache_repo/cache_revision where present (LoRA
    arms: the adapter's sha256; __a100 re-extractions: the tagged repo name), otherwise
    repo/revision - the same identity the analyses write as {side}_repo / {side}_rev.
    """
    by_sha: dict[str, list[dict]] = {}
    by_name: dict[str, list[dict]] = {}
    for f in results_files("extract__*.json"):
        if "__rerun_" in f.name:
            # a re-extraction's record (D-168 V100 job, D-182 A100 fallback) describes a REBUILT
            # cache, not the original input; it never gives an input its identity or clears a
            # status until the author accepts it and it is promoted (as input_registry.py)
            continue
        e = json.load(open(f, encoding="utf-8"))
        repo = e.get("cache_repo") or e.get("repo")
        rev = e.get("cache_revision") or e.get("revision")
        for split, s in (e.get("splits") or {}).items():
            rec = {"summary": rel_name(f), "split": split, "cache_file": s.get("cache_file"),
                   "cache_sha256": s.get("cache_sha256"), "repo": repo, "revision": rev,
                   "base_repo": e.get("repo"), "base_revision": e.get("revision")}
            by_sha.setdefault(s.get("cache_sha256"), []).append(rec)
            by_name.setdefault(s.get("cache_file"), []).append(rec)
    return by_sha, by_name


def named_identity(d: dict, key: str) -> tuple[str | None, str | None, str]:
    """(repo, revision, where it was read) that result `d` names for input `key`.

    Keys are "side/part", "stem:side/part" (the input of a named results file) or a bare
    part (single-model files, config repo/rev). A file that does not carry the revision
    itself names it through the result it analyses: the stem in the key, or audited_result.
    """
    cfg = d.get("config") or {}
    stem, tail = key.split(":", 1) if ":" in key else (None, key)
    side = tail.split("/")[0]
    # Files written through src/input_registry.py (D-165) nest the revisions per run under
    # config.runs, keyed by the stem in the input key (the 4a cells use the LoRA run name).
    runs = cfg.get("runs") if isinstance(cfg.get("runs"), dict) else {}
    for rk in ([stem, f"repair_{stem}"] if stem else []):
        r = runs.get(rk)
        if isinstance(r, dict) and f"{side}_rev" in r:
            return r.get(f"{side}_repo"), r[f"{side}_rev"], "config.runs (input registry)"
    for pfx in ([f"{stem}:"] if stem else []) + [""]:
        if f"{pfx}{side}_rev" in cfg:
            return cfg.get(f"{pfx}{side}_repo"), cfg[f"{pfx}{side}_rev"], "config"
    if side not in ("pre", "post") and "rev" in cfg:
        return cfg.get("repo"), cfg["rev"], "config (single model)"
    if side == "pre" and not stem:
        s_repo, s_rev = single_model_repo_rev(cfg)
        if s_rev is not None:
            return s_repo, s_rev, "config (single model, read as the pre side, D-297)"
    ref = stem or d.get("audited_result")
    if ref:
        ref_path = result_path(ref)
        if readable_result(ref_path):
            rcfg = json.load(open(ref_path, encoding="utf-8")).get("config") or {}
            if f"{side}_rev" in rcfg:
                return rcfg.get(f"{side}_repo"), rcfg[f"{side}_rev"], f"via {rel_name(ref_path)}"
    return None, None, "no revision named"


def revision_check(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    """F-035, second half: the revision a result names for each cache it read must equal the
    revision in the extraction record of that cache.

    Cache file names carry only the first 12 characters of the revision, so a wrong full
    revision loads the right file and passes every other check (F-035). The extraction
    summary is the only place the full revision is stored.

    A cache's record is the record of the NAMED model whose cache_sha256 is the bytes the
    result recorded; failing that, the record of the named model's file on disk holding
    those bytes (the summary then lists a different hash for that name). Byte-identical
    caches of two models exist (D-034: Mistral v0.2/v0.3 sparse caches; a step-0 LoRA
    adapter is the identity, so its caches equal the base model's), so records of OTHER
    models with the same bytes are reported separately, never counted as the cache's record.
    Summaries overwritten before F-024 was fixed leave some caches with no record at all.

    MISMATCH is the F-035 failure: a record of the named model exists and none of its
    records has the named revision.
    """
    ext_sha, ext_name = extraction_index()
    disk_files: dict[str, list[str]] = {}
    for m in manifest:
        disk_files.setdefault(m["sha256"], []).append(m["file"])
    rows = []
    for f in results_files("*.json"):
        name = rel_name(f)
        if Path(f).name.startswith("extract__"):
            continue
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        refs = d.get("input_cache_sha256") if isinstance(d, dict) else None
        if not isinstance(refs, dict) or not refs:
            continue
        for key, h in sorted(refs.items()):
            repo, rev, where = named_identity(d, key)
            by_content = ext_sha.get(h, [])
            own, found = [r for r in by_content if r["repo"] == repo], "by content"
            if not own:
                own = [r for fn in disk_files.get(h, []) for r in ext_name.get(fn, [])
                       if r["repo"] == repo]
                found = "by name of the file on disk holding these bytes (record hash differs)"
            # D-297 (1): a cache whose record documents a tag ('<repo>__<tag>@', D-252) is the named model's own
            # cache when the record's model is the named one; its FULL revision is then compared. Only a fallback,
            # so no existing match changes.
            tagged = [] if own else [r for r in by_content if r["base_repo"] == repo and documented_tag(r)]
            other = [r for r in by_content if r["repo"] != repo and r not in tagged]
            aliases = sorted({a["id"] for a in DOCUMENTED_ALIASES for r in other if a["match"](repo, rev, r)})
            if rev is None:
                status = "NO_NAMED_REVISION"
            elif own:
                status = "MATCH" if any(r["revision"] == rev for r in own) else "MISMATCH"
            elif tagged:
                status = "MATCH_DOCUMENTED_TAG" if any(r["base_revision"] == rev for r in tagged) else "MISMATCH"
                own, found = tagged, f"by content, cache tag documented by its record ({documented_tag(tagged[0])})"
            elif other:
                status = "MATCH_DOCUMENTED_ALIAS" if aliases else "UNDOCUMENTED_ALIAS"
            else:
                status = "NO_EXTRACTION_RECORD"
            keep = ("summary", "split", "repo", "revision", "base_repo", "base_revision")
            rows.append({"result": name, "input": key, "cache_sha256": h, "named_repo": repo,
                         "named_revision": rev, "named_where": where, "status": status,
                         "record_found": found if own else None,
                         "records": [{k: r[k] for k in keep} for r in own],
                         "other_model_records": [{k: r[k] for k in keep} for r in other],
                         "other_record_base_revision_equals_named":
                             (any(r["base_revision"] == rev for r in other) if other else None),
                         "documented_alias": aliases or None})
    counts: dict[str, int] = {}
    by_tree: dict[str, dict[str, int]] = {"results": {}, "phase2": {}}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        t = by_tree["phase2" if r["result"].startswith("phase2/") else "results"]
        t[r["status"]] = t.get(r["status"], 0) + 1
    return {"inputs_checked": len(rows),
            "results_checked": len({r["result"] for r in rows}),
            "counts": counts,
            "counts_by_tree": by_tree,                # Phase 1 (results/) and Phase 2 (results/phase2/) apart
            "mismatches": [r for r in rows if r["status"] in FAILING_REVISION_STATUSES],
            "results_with_mismatch": sorted({r["result"] for r in rows if r["status"] in FAILING_REVISION_STATUSES}),
            "documented_aliases": [{k: a[k] for k in ("id", "doc")} for a in DOCUMENTED_ALIASES],
            "results_with_unrecorded_input": sorted({r["result"] for r in rows
                                                     if r["status"] == "NO_EXTRACTION_RECORD"}),
            "rows": rows}


_SOURCE_CFG: dict[str, dict] = {}


def expected_repo(key: str, cfg: dict) -> str | None:
    """The model repo an input key names: "<side>/<part>" reads cfg["<side>_repo"]; "<stem>:<side>/<part>"
    reads cfg["<stem>:<side>_repo"], else the config of results/<stem>.json (a source result), else the
    registry's declared run <stem>. None when none of them names it (then the match counts as by content)."""
    stem, sep, sp = key.rpartition(":")
    side = sp.split("/")[0]
    if not sep:
        if side == "pre" and not cfg.get("pre_repo"):
            s_repo, s_rev = single_model_repo_rev(cfg)          # D-297 (1)
            if s_rev is not None:
                return s_repo
        return cfg.get(f"{side}_repo")
    if cfg.get(f"{stem}:{side}_repo"):
        return cfg[f"{stem}:{side}_repo"]
    if stem not in _SOURCE_CFG:
        # exactly as before for every Phase 1 stem; only an explicit 'phase2/' stem resolves under results/phase2/
        src = result_path(stem) if str(stem).startswith("phase2/") else RESULTS / f"{stem}.json"
        try:
            _SOURCE_CFG[stem] = (json.load(open(src, encoding="utf-8")).get("config") or {}) if readable_result(src) else {}
        except Exception:  # noqa: BLE001 - an unreadable source names nothing
            _SOURCE_CFG[stem] = {}
    if _SOURCE_CFG[stem].get(f"{side}_repo"):
        return _SOURCE_CFG[stem][f"{side}_repo"]
    import input_registry as REG
    try:
        return REG.run(stem)[f"{side}_repo"]
    except (SystemExit, KeyError):
        return None


def tagged_cache_files() -> dict[str, str]:
    """D-297 (1): cache file name -> its model's repo, for every cache whose extraction record documents a tag."""
    _, ext_name = extraction_index()
    return {fn: r["base_repo"] for fn, rs in ext_name.items() for r in rs if fn and documented_tag(r)}


def name_hit(key: str, cfg: dict, candidates: list[str], tagged_files: dict[str, str]) -> str | None:
    """The file, among those holding the bytes, that carries the name of the model the key means: '<repo>@...',
    or a tagged cache ('<repo>__<tag>@...') whose extraction record documents the tag for that model (D-297 (1))."""
    if key.endswith(".npz"):
        return next((c for c in candidates if c == key), None)
    repo = expected_repo(key, cfg)
    if not repo:
        return None
    return next((c for c in candidates
                 if c.startswith(repo.replace("/", "__") + "@") or tagged_files.get(c) == repo), None)


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Audit result -> input-cache traceability")
    ap.add_argument("--out", required=True, help="a new, dated name: results/provenance_audit_<date>.json")
    args = ap.parse_args()
    if Path(args.out).exists():
        raise SystemExit(f"REFUSING: {args.out} exists; results are never overwritten (name each audit by its date)")
    # D-256 (9): the audit includes the commit-less extraction records, through D-286's record (hash-pinned)
    if not EXTRACTION_COMMIT_AUDIT.is_file():
        raise SystemExit(f"REFUSING: {EXTRACTION_COMMIT_AUDIT.name} (D-286) is not on disk")
    eca_bytes = EXTRACTION_COMMIT_AUDIT.read_bytes()
    if hashlib.sha256(eca_bytes).hexdigest() != EXTRACTION_COMMIT_AUDIT_SHA256:
        raise SystemExit(f"REFUSING: {EXTRACTION_COMMIT_AUDIT.name} is not the pinned record (D-286, D-308)")
    eca = json.loads(eca_bytes.decode("utf-8"))
    # D-297 (1) writer rule: the audit reads every results file, so it lists each with its sha256, hashed here before
    # any is read, and refuses to write if any changed by the end
    global _SNAPSHOT
    out_abs = Path(args.out).resolve()
    _SNAPSHOT = [p for p in sorted(RESULTS.glob("*.json")) + sorted(phase2_dir().glob("*.json")) if p.resolve() != out_abs]
    try:
        inputs_listed = {f"results/{rel_name(p)}": sha256_file(p) for p in _SNAPSHOT}
    except FileNotFoundError as e:
        raise SystemExit(f"REFUSING: a results file vanished while the audit listed them: {e.filename}") from None
    # the registry reads the extraction records through its own glob (input_registry._load_records, cached once): load
    # it now and require exactly the snapshot's non-rerun extract__ records, byte for byte, so it never reads one unlisted
    import input_registry as REG
    reg_seen = {r["summary"]: r["sha256"] for r in REG._load_records()}
    reg_want = {p.name: inputs_listed[f"results/{p.name}"] for p in results_files("extract__*.json", top_only=True)
                if "__rerun_" not in p.name}
    if reg_seen != reg_want:
        raise SystemExit("REFUSING: the registry's extraction records are not the snapshot's: "
                         f"{sorted(set(reg_seen.items()) ^ set(reg_want.items()))[:10]}")

    # F-032: caches now live in two places - the repo's data/hidden and the scratch
    # directory named by GUARDRAIL_CACHE_DIRS - so the audit must look in both or it
    # will report every scratch-resident cache as missing.
    import os

    dirs = [Path(d) for d in os.environ.get("GUARDRAIL_CACHE_DIRS", "").split(os.pathsep) if d]
    caches = sorted({p for d in [*dirs, CACHE] if d.exists() for p in d.glob("*.npz")},
                    key=lambda p: p.name)
    print(f"hashing {len(caches)} caches ...", flush=True)
    manifest = []
    for i, p in enumerate(caches, 1):
        manifest.append(cache_record(p))
        if i % 10 == 0:
            print(f"  {i}/{len(caches)}", flush=True)
    by_hash: dict[str, str] = {m["sha256"]: m["file"] for m in manifest}
    # every file name holding each sha256: a byte-identical twin (the D-034 kind) must not hide the name a
    # result actually read, or the D-217 (d) gate would fail a correct result
    names_by_hash: dict[str, list[str]] = {}
    for m in manifest:
        names_by_hash.setdefault(m["sha256"], []).append(m["file"])

    audits = []
    tagged_files = tagged_cache_files()                 # D-297 (1)
    for f in results_files("*.json"):
        name = rel_name(f)
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception as exc:
            audits.append({"result": name, "status": "UNREADABLE", "error": str(exc)})
            continue
        # A results file need not be a JSON object (results/hub_resolution.json is a list);
        # such a file has no cache inputs to trace. Found 2026-09-24 when that file first
        # reached the node and the audit died on it (exit 1, recorded by detach.sh).
        refs = d.get("input_cache_sha256") if isinstance(d, dict) else None
        if not isinstance(refs, dict) or not refs:
            continue  # audits, manifests, syntheses: no cache inputs to trace
        matched = {k: by_hash[h] for k, h in refs.items() if h in by_hash}
        missing = {k: h for k, h in refs.items() if h not in by_hash}

        # Content-addressed matching is the right reproducibility test - the exact
        # bytes are available - but it can mislead: found 2026-09-16 that
        # pilot_e6.json's Mistral-v0.2 inputs "matched" the Mistral-v0.3 cache
        # files, because the two sparse caches were byte-identical (D-034, v0.3
        # is a vocabulary-only change). So each match is labelled by whether the
        # file carries the model name the result says it read.
        cfg = d.get("config", {}) or {}
        by_name, by_content = {}, {}
        for k, fname in matched.items():
            # D-217 (d) review: derived results name their inputs as "<stem>:<side>/<part>" (the stem is a
            # declared run or a source result) or by the cache file name itself; resolve the model the key
            # means before judging the match, so a correct derived result is not failed by construction.
            # Any file holding those bytes may carry the expected name (byte-identical twins, D-034).
            candidates = names_by_hash.get(refs[k], [fname])
            hit = name_hit(k, cfg, candidates, tagged_files)
            if hit:
                by_name[k] = hit
            else:
                by_content[k] = fname
        if missing:
            status = "NOT_REPRODUCIBLE_FROM_DISK"
        elif by_content:
            status = "TRACEABLE_BY_CONTENT_ONLY"
        else:
            status = "TRACEABLE"
        audits.append({
            "result": name, "status": status,
            "n_inputs": len(refs), "n_matched": len(matched),
            "matched_by_name": by_name, "matched_by_content_other_file": by_content,
            "missing": missing,
        })

    ok = [a for a in audits if a["status"] == "TRACEABLE"]
    content_only = [a for a in audits if a["status"] == "TRACEABLE_BY_CONTENT_ONLY"]
    bad = [a for a in audits if a["status"] == "NOT_REPRODUCIBLE_FROM_DISK"]
    unreadable = [m for m in manifest if not m.get("readable")]

    print("\n" + "=" * 78)
    print(f"caches on disk: {len(manifest)}   unreadable: {len(unreadable)}   "
          f"with non-finite items: {sum(1 for m in manifest if (m.get('n_nonfinite') or 0) > 0)}")
    print(f"results with cache inputs: {len(audits)}   TRACEABLE: {len(ok)}   "
          f"by content only: {len(content_only)}   NOT reproducible from disk: {len(bad)}")
    print("=" * 78)
    for a in content_only + bad:
        print(f"  {a['status']:<27} {a['result']}  "
              f"({a.get('n_matched', 0)}/{a.get('n_inputs', 0)} input bytes still on disk)")
        for k, f in sorted(a.get("matched_by_content_other_file", {}).items()):
            print(f"      {k:<22} bytes found under a DIFFERENT file: {f}")
        for k in sorted(a.get("missing", {})):
            print(f"      {k:<22} bytes GONE")
    for m in unreadable:
        print(f"  UNREADABLE CACHE  {m['file']}  {m.get('error')}")

    # D-153: a primary cache (read by any result a declared population rests on) must have
    # no non-finite item. Such an item is stored as zeros by extraction, so it would enter
    # a headline number silently. This gate FAILS the audit, it does not just report.
    import populations as POP
    # a primary file written after the snapshot (primary_result_stems tests LoRA files by exists()) is left out, unread
    primary = {s for s in POP.primary_result_stems()
               if readable_result(RESULTS / f"{s}.json") or not (RESULTS / f"{s}.json").exists()}
    by_sha = {m["sha256"]: m for m in manifest}
    nonfinite_primary = []
    for a in audits:
        stem = a["result"][:-5] if a["result"].endswith(".json") else a["result"]
        if stem not in primary:
            continue
        refs = json.load(open(RESULTS / a["result"], encoding="utf-8")).get("input_cache_sha256", {})
        for key, h in refs.items():
            m = by_sha.get(h)
            if m and (m.get("n_nonfinite") or 0) > 0:
                nonfinite_primary.append({"result": a["result"], "input": key, "cache": m["file"],
                                          "n_nonfinite": m["n_nonfinite"],
                                          "item_indices": m.get("nonfinite_item_indices")})
    nonfinite_all = [{"cache": m["file"], "n_nonfinite": m["n_nonfinite"],
                      "item_indices": m.get("nonfinite_item_indices")}
                     for m in manifest if (m.get("n_nonfinite") or 0) > 0]
    print(f"non-finite: {len(nonfinite_all)} cache(s) with any non-finite item; "
          f"{len(nonfinite_primary)} primary input(s) affected over {len(primary)} primary results")
    for c in nonfinite_all:
        print(f"  {c['cache']}: {c['n_nonfinite']} item(s) at {c['item_indices']}")
    for v in nonfinite_primary:
        print(f"  FAIL primary {v['result']} {v['input']} reads {v['cache']} ({v['n_nonfinite']} non-finite)")

    rc = revision_check(manifest)
    print(f"revision check (F-035): {rc['inputs_checked']} inputs of {rc['results_checked']} "
          f"results: {rc['counts']}")
    for r in rc["mismatches"]:
        print(f"  {r['status']} {r['result']} {r['input']}: names {r['named_revision']} "
              f"({r['named_where']}); extraction records: "
              f"{sorted({x['revision'] for x in r['records'] + r['other_model_records']})}")
    # D-162: a revision mismatch (or an undocumented alias) on any PRIMARY input fails the audit.
    revision_primary = [r for r in rc["mismatches"] if r["result"].removesuffix(".json") in primary]
    print(f"revision gate (D-162): {len(revision_primary)} failing input(s) among {len(primary)} primary results")
    for r in revision_primary:
        print(f"  FAIL primary {r['result']} {r['input']} ({r['status']})")

    # D-170/D-172: results that read a cache vouched for only by a committed pin (no extraction
    # record) carry "input verified by pin only, pending D-168" and enter no paper table until
    # D-168 passes. Counted here by bytes, so results written before the flag existed count too.
    import input_registry as REG
    missing_pins = [str(m) for m in REG.MANIFESTS if not m.exists()]
    if missing_pins:
        raise SystemExit(f"REFUSING: the committed pin manifest is missing ({missing_pins}); the pin-only "
                         f"count would silently read 0")
    pins = {h: name for name, (h, _) in REG._load_manifests().items()}   # same parser, same conflict check
    final = REG.d168_final()                  # None until pinned; refuses a pinned record that is missing or wrong
    final_sha = final["sha256"] if final else None
    final_caches = REG.d168_final_caches()
    pin_only_results = []
    for f in results_files("*.json"):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        refs = d.get("input_cache_sha256") if isinstance(d, dict) else None
        if isinstance(refs, dict):
            hit = sorted({pins[h] for h in refs.values() if h in pins})
            if hit:
                stem = rel_name(f)[:-5]
                carries, cleared = pin_flag_status(d, hit, final_sha, final_caches)
                pin_only_results.append({"result": rel_name(f), "primary": stem in primary,
                                         "carries_flag": carries,
                                         # D-300 (3): written after the final D-168 record was pinned, naming it, and
                                         # reading only caches it judged
                                         "cleared_by_final": cleared,
                                         "pin_only_caches": hit})
    n_cleared = sum(r["cleared_by_final"] for r in pin_only_results)
    print(f"pin-only inputs (D-170): {len(pin_only_results)} results read a cache vouched for only by a pin "
          f"({sum(r['primary'] for r in pin_only_results)} primary; "
          f"{sum(r['carries_flag'] for r in pin_only_results)} carry the flag; {n_cleared} name the final D-168 "
          "record); "
          + (f"D-168 is final ({final['decision']}, {final['file']} {final['sha256'][:16]}): results written after the "
             f"pin carry no flag ({n_cleared}); earlier ones still store it in their bytes"
             if final else "none enters the paper before D-168"))

    # D-217 (d): ANY mismatch on a primary result fails the audit, not only non-finite items (D-153) and
    # revisions (D-162): input bytes gone, bytes found only under another file's name, an unreadable
    # result, or a pin-only input read without the D-170 flag
    trace_primary = [{"result": a["result"], "status": a["status"]} for a in audits
                     if a["result"].removesuffix(".json") in primary and a["status"] != "TRACEABLE"]
    flag_primary = [{"result": r["result"], "pin_only_caches": r["pin_only_caches"]}
                    for r in pin_only_results if r["primary"] and not (r["carries_flag"] or r["cleared_by_final"])]
    print(f"traceability gate (D-217 (d)): {len(trace_primary)} primary result(s) not TRACEABLE; "
          f"{len(flag_primary)} primary result(s) read a pin-only cache without the D-170 flag")
    for v in trace_primary:
        print(f"  FAIL primary {v['result']} ({v['status']})")
    for v in flag_primary:
        print(f"  FAIL primary {v['result']} reads pin-only {v['pin_only_caches']} without the D-170 flag")

    payload = {**provenance_block("provenance_audit"),
               "summary": {"caches": len(manifest), "unreadable_caches": len(unreadable),
                           "results_audited": len(audits), "traceable": len(ok),
                           "traceable_by_content_only": [a["result"] for a in content_only],
                           "not_reproducible_from_disk": [a["result"] for a in bad],
                           "primary_results": len(primary),
                           "caches_with_nonfinite": nonfinite_all,
                           "primary_inputs_with_nonfinite": nonfinite_primary,
                           "revision_mismatches": len(rc["mismatches"]),
                           "revision_gate_failing_primary_inputs": revision_primary,
                           "traceability_gate_failing_primary_results": trace_primary,
                           "flag_gate_failing_primary_results": flag_primary,
                           "results_reading_pin_only_caches": len(pin_only_results),
                           "pin_only_flag": REG.PIN_ONLY_FLAG,
                           "d168_final": final,     # D-300 (3): the pinned final record that clears the flag, or None
                           "revision_check_counts": rc["counts"]},
               "cache_manifest": manifest, "result_audits": audits, "revision_check": rc,
               "pin_only_results": pin_only_results,
               "extraction_records_without_commit": {
                   "source": {"file": f"results/{EXTRACTION_COMMIT_AUDIT.name}", "sha256": EXTRACTION_COMMIT_AUDIT_SHA256,
                              "decision": "D-286, extended by D-308"},
                   "n_records": eca["n_records"], "counts": eca["counts"],
                   "recovered": {n: r["commit"] for n, r in eca["records"].items() if r["verdict"] == "RECOVERED"},
                   "not_recovered_git_upper_bound": {n: r["git_bound"]["at_most"]["commit"]
                                                     for n, r in eca["records"].items() if r["verdict"] == "NOT RECOVERED"},
                   "reading": "the extraction code version of a NOT RECOVERED record is bounded above only; its caches' "
                              "evidence is their bytes (pins, D-170; the D-168 rebuild comparisons), not the code"}}
    # coverage: every commit-less extraction record on disk now must be one D-286 audited, with the same bytes
    commitless_now: dict[str, str] = {}
    for f in results_files("extract__*.json", top_only=True):       # D-286 audited Phase 1's records only
        fb = f.read_bytes()
        if not (json.loads(fb.decode("utf-8")).get("git") or {}).get("commit"):
            commitless_now[f.name] = hashlib.sha256(fb).hexdigest()
    payload["extraction_records_without_commit"]["coverage"] = {
        "commit_less_records_on_disk": len(commitless_now),
        "not_covered_by_D286": sorted(set(commitless_now) - set(eca["records"])),
        "missing_or_changed_since_D286": sorted(n for n, r in eca["records"].items()
                                                if commitless_now.get(n) != r["record_sha256"])}
    # Phase 2 extraction records (results/phase2/, D-303 (12)) are outside D-286: list any that names no commit in either
    # its git block or its code block (a node run has no git; its code block names the synced commit)
    p2_commitless = []
    for f in results_files("extract__*.json"):
        if f.parent == RESULTS:
            continue
        e = json.loads(f.read_bytes().decode("utf-8"))  # listed (D-297): a snapshot file, in input_files_sha256
        if not ((e.get("git") or {}).get("commit") or (e.get("code") or {}).get("commit")):
            p2_commitless.append(rel_name(f))
    payload["extraction_records_without_commit"]["phase2_records_without_commit"] = sorted(p2_commitless)
    payload["summary"]["gates_passed"] = not (nonfinite_primary or revision_primary or trace_primary or flag_primary)
    # only a LISTED file that changed or disappeared refuses; a record written during the audit was never read
    def unchanged(rel: str, h: str) -> bool:
        try:
            return sha256_file(ROOT / rel) == h
        except FileNotFoundError:
            return False
    changed = sorted(rel for rel, h in inputs_listed.items() if not unchanged(rel, h))
    if changed:
        raise SystemExit(f"REFUSING to write: listed results changed or vanished during the audit: {changed[:10]}")
    payload["input_files_sha256"] = inputs_listed
    cov = payload["extraction_records_without_commit"]["coverage"]
    print(f"extraction records without a commit (D-286): {eca['counts']}; on disk now {cov['commit_less_records_on_disk']}, "
          f"not covered {cov['not_covered_by_D286']}, missing or changed {cov['missing_or_changed_since_D286']}")
    digest = write_results(args.out, payload, exclusive=True)
    print(f"\nWritten: {args.out}\nsha256:  {digest}")
    if nonfinite_primary:
        print("AUDIT FAILED: a primary cache holds non-finite items (D-153)")
    if revision_primary:
        print("AUDIT FAILED: a primary input's named revision is not its extraction record's (D-162)")
    if trace_primary:
        print("AUDIT FAILED: a primary result's input bytes are gone, found only under another name, or unreadable (D-217 (d))")
    if flag_primary:
        print("AUDIT FAILED: a primary result reads a pin-only cache without the D-170 flag (D-217 (d))")
    # 3: the audit is written and a gate failed; a refusal exits 1, so a queue can tell them apart (as compare_caches)
    return 3 if nonfinite_primary or revision_primary or trace_primary or flag_primary else 0


if __name__ == "__main__":
    raise SystemExit(main())
