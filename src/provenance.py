"""Provenance stamping for every results file in this repo.

CLAUDE.md requires each results JSON to record git commit, config hash, dataset
SHA-256s, model repo IDs and revisions, and seeds. This module supplies the parts
that are identical across every run so no runner has to reimplement them.

Rule 2 discipline: every field here is either read from the real environment or
set to None with an explicit error string. Nothing is guessed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

SCHEMA_VERSION = 1


def ensure_utf8_stdout() -> None:
    """Force UTF-8 on stdout/stderr.

    The development machine's console codepage is cp949, which raises
    UnicodeEncodeError on any non-ASCII character in a print(). See FAILURES.md
    F-004. Source files in this repo are kept ASCII-only as the primary defence;
    this is the belt-and-braces second line, because model names, dataset cards
    and API error strings are outside our control and routinely contain non-ASCII.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _git_raw(*args: str) -> tuple[bool, str]:
    """Run a git command in the repo root. Returns (succeeded, stripped stdout).

    The success flag is separate from the output because `git status --porcelain`
    returns EMPTY output on a clean tree. Collapsing empty-to-None would report a
    clean tree as "unknown", which is a different and much weaker claim.
    """
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False, ""
    if out.returncode != 0:
        return False, ""
    return True, out.stdout.strip()


def _git(*args: str) -> str | None:
    """Stripped stdout, or None when the command failed or produced nothing."""
    ok, text = _git_raw(*args)
    return text if (ok and text) else None


def git_state() -> dict[str, Any]:
    """Current git commit and whether the tree is dirty.

    A dirty tree is recorded, not hidden: a result produced from uncommitted code
    is not reproducible and the results file must say so.
    """
    ok, status = _git_raw("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if ok else None,
        "dirty_file_count": len(status.splitlines()) if ok else None,
        "dirty_files": status.splitlines()[:50] if ok and status else [],
    }


def env_state() -> dict[str, Any]:
    """Interpreter and platform facts that can change a numeric result.

    BLAS thread count is recorded because it IS one of them: F-037 measured that thread
    count alone moves the summary metric by up to 0.008, and the paper says so in its
    method section. A results file that does not name its thread count cannot be
    re-derived exactly, so the count is part of the record, not of the environment.
    """
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "blas_threads": {v: os.environ.get(v) for v in
                         ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                          "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "cpu_count": os.cpu_count(),
        # lbfgs iterations, convergence and row-order sensitivity (F-059) depend on these.
        "libraries": _library_versions(),
    }


def _library_versions() -> dict[str, str | None]:
    from importlib import metadata

    out: dict[str, str | None] = {}
    for dist in ("numpy", "scipy", "scikit-learn", "torch", "transformers"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: str | Path) -> list[Any]:
    """Read a JSONL file. The ONLY sanctioned way to read frozen splits.

    Splits on "\\n" and never on str.splitlines(). json.dumps escapes \\n and \\r
    but does NOT escape U+2028 LINE SEPARATOR, U+2029, U+0085 or U+000B/U+000C,
    while str.splitlines() treats all of them as line breaks. The frozen safety
    split contains 21 U+2028 characters, so splitlines() yields 4021 fragments
    for 4000 records and the first fragment fails to parse. See FAILURES.md F-017.
    """
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def canonical_hash(obj: Any) -> str:
    """Stable SHA-256 of a JSON-serialisable object.

    Used for config hashes. sort_keys + fixed separators so the hash depends on
    content, not on dict insertion order or formatting.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_bytes(payload.encode("utf-8"))


def provenance_block(run_kind: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The standard header for every results JSON written by this repo."""
    block: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_kind": run_kind,
        "utc": utc_now(),
        "git": git_state(),
        "env": env_state(),
    }
    if extra:
        block.update(extra)
    return block


def cache_write_dir(default: Path) -> Path:
    """Where NEW hidden-state caches are written.

    F-032: the NAS quota filled mid-experiment, so a run must be able to put new
    caches on scratch storage while the frozen ones stay where they are. Unset,
    behaviour is exactly as before.
    """
    d = os.environ.get("GUARDRAIL_CACHE_WRITE_DIR")
    return Path(d) if d else default


def resolve_cache(name: str, default: Path) -> Path:
    """First existing cache file named `name`, searching GUARDRAIL_CACHE_DIRS
    (os.pathsep-separated) before the default directory. Falls back to the write
    location so the caller reports a missing file at the path it would be written."""
    for d in [Path(p) for p in os.environ.get("GUARDRAIL_CACHE_DIRS", "").split(os.pathsep) if p]:
        if (d / name).exists():
            return d / name
    if (default / name).exists():
        return default / name
    return cache_write_dir(default) / name


def stage_bytes(path: str | Path, data: bytes) -> tuple[Path, str]:
    """Stage `data` for `path`: write <path>.partial.<pid>.<random> (exclusive create), fsync,
    and verify its size and sha256 (F-027, F-032). Nothing exists under the final name yet, so
    a run with several outputs can stage them all and only then name them (commit_staged): a
    quota failure then happens before any final name exists, since naming only adds a link
    (the directory itself may still grow; an orphan that leaves is refused on rerun). The
    random suffix means a stale staged file from a killed run with a recycled pid is never
    touched; only a file this call created is ever removed. The caller removes the staged file
    if it is not committed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    want = sha256_bytes(data)
    tmp = path.with_name(f"{path.name}.partial.{os.getpid()}.{os.urandom(4).hex()}")
    created = False
    try:
        with open(tmp, "xb") as fh:
            created = True
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        got = tmp.stat().st_size
        if got != len(data):
            raise OSError(f"stage verify: wrote {got} bytes of {len(data)} to {tmp.name}")
        if sha256_file(tmp) != want:
            raise OSError(f"stage verify: {tmp.name} hash differs after write")
    except BaseException:
        if created and tmp.exists():
            tmp.unlink()
        raise
    return tmp, want


def commit_staged(tmp: Path, path: str | Path, want: str) -> None:
    """Name a staged file: hard link to `path` (FileExistsError if it exists, D-152 P5: never
    overwrite), read the final name back and check its sha256, then remove the staged name."""
    path = Path(path)
    try:
        os.link(tmp, path)
        if sha256_file(path) != want:
            raise OSError(f"commit verify: {path.name} differs from what was staged")
    finally:
        if tmp.exists():
            tmp.unlink()


# --- D-297 (1): the results-writer rule ------------------------------------------------
# A record that reads a cache must name the repo ID and revision of every cache input; a record that reads other
# results must list each of them with its sha256. write_results refuses otherwise. Results read through read_result()
# are registered here, so the rule can check them.
_RESULT_READS: dict[str, str] = {}
_SIDE_KEYS = ("pre_repo", "post_repo", "pre_rev", "post_rev")


def _repo_rel(p: Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO_ROOT)).replace("\\", "/") if p.is_relative_to(REPO_ROOT) else p.name


def read_result(path: str | Path) -> dict[str, Any]:
    """Read a results JSON as an INPUT (D-297 (1)): the bytes are hashed once and the read is registered, and
    write_results then refuses a record that does not list this file with that sha256."""
    b = Path(path).read_bytes()
    _RESULT_READS[_repo_rel(Path(path))] = sha256_bytes(b)
    return json.loads(b.decode("utf-8"))


def _names_repo_rev(key: str, cfg: dict[str, Any], ident: dict[str, Any]) -> bool:
    """Whether the record names a repo ID and a revision for cache input `key`: an explicit input_identity entry,
    or one of the conventions in use: config <side>_repo/_rev, config '<stem>:<side>_repo/_rev', config.runs[<stem>]
    (the input registry's form), or a single-model config.repo/rev read as the pre side."""
    i = ident.get(key)
    if isinstance(i, dict) and i.get("repo") and i.get("rev"):
        return True
    stem, _, tail = key.rpartition(":")
    side = tail.split("/")[0]
    if cfg.get(f"{side}_repo") and cfg.get(f"{side}_rev"):
        return True
    if stem and cfg.get(f"{stem}:{side}_repo") and cfg.get(f"{stem}:{side}_rev"):
        return True
    runs = cfg.get("runs") if isinstance(cfg.get("runs"), dict) else {}
    for rk in ([stem, f"repair_{stem}"] if stem else []):
        r = runs.get(rk)
        if isinstance(r, dict) and r.get(f"{side}_repo") and r.get(f"{side}_rev"):
            return True
    if not stem and side not in ("pre", "post") and cfg.get("repo") and cfg.get("rev"):
        return True                                       # a bare part of a single-model file, as the audit reads it
    return bool(not stem and side == "pre" and cfg.get("repo") and cfg.get("rev")
                and not any(k in cfg for k in _SIDE_KEYS))


def _listed_files(o: Any, out: set[tuple[str, str]]) -> set[tuple[str, str]]:
    """Every (file, sha256) a payload lists: {"file": f, "sha256": h} blocks, {path: sha256} mappings, and the
    '<name>_file' / '<name>_sha256' sibling pairs (e.g. selection_file and selection_sha256)."""
    if isinstance(o, dict):
        if isinstance(o.get("file"), str) and isinstance(o.get("sha256"), str):
            out.add((o["file"].replace("\\", "/"), o["sha256"]))
        for k, v in o.items():
            if isinstance(k, str) and ("/" in k or "." in k) and isinstance(v, str) and len(v) == 64:
                out.add((k.replace("\\", "/"), v))
            if isinstance(k, str) and k.endswith("_file") and isinstance(v, str):
                h = o.get(k[:-len("_file")] + "_sha256")
                if isinstance(h, str):
                    out.add((v.replace("\\", "/"), h))
            _listed_files(v, out)
    elif isinstance(o, list):
        for v in o:
            _listed_files(v, out)
    return out


def input_identity_problems(payload: dict[str, Any]) -> list[str]:
    """D-297 (1): what keeps `payload` from meeting the results-writer rule (empty when it meets it)."""
    problems: list[str] = []
    refs = payload.get("input_cache_sha256") if isinstance(payload, dict) else None
    if isinstance(refs, dict) and refs:
        cfg = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        ident = payload.get("input_identity") if isinstance(payload.get("input_identity"), dict) else {}
        missing = sorted(k for k in refs if not _names_repo_rev(k, cfg, ident))
        if missing:
            problems.append(f"cache inputs with no repo ID and revision named: {missing}")
    listed = _listed_files(payload, set())
    unlisted = sorted(f"{rel} ({h[:12]})" for rel, h in _RESULT_READS.items() if (rel, h) not in listed)
    if unlisted:
        problems.append(f"results read but not listed with their sha256: {unlisted}")
    return problems


REISSUE_MAY_DIFFER = ("utc", "git", "env", "code", "preregistration")
RESULTS_DIR = REPO_ROOT / "results"


def _registry_sources(REG: Any, repo: str) -> dict[str, str]:
    """The registry module and the extraction records that give `repo`'s revision, as {path: sha256}."""
    out = {"src/input_registry.py": sha256_file(Path(REG.__file__))}
    out.update({f"results/{r['summary']}": r["sha256"] for r in REG._load_records() if r["repo"] == repo})
    return out


def identity_config(keys: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """D-297 (1), D-300: config entries naming the repo ID and revision of each cache input key, in forms the audit
    (src/verify_provenance.py, named_identity) already reads, and the results files read to name them ({path: sha256}).
      '<stem>:<side>/<part>': '<stem>:<side>_repo' and '<stem>:<side>_rev', copied from results/<stem>.json's config
          <side>_repo/_rev when that file exists (the audit's own 'via' path), else from the input registry's declared
          run <stem>;
      a bare cache file name '<org>__<model>@<rev12>__...npz': '<name>_repo' and '<name>_rev' (the audit reads a key
          without '/' as its own side), the registry's revision of <org>/<model>, which must start with <rev12>.
    Refuses any key it cannot name."""
    import input_registry as REG                            # deferred: input_registry imports this module
    cfg: dict[str, str] = {}
    sources: dict[str, str] = {}
    for key in keys:
        stem, tail = key.split(":", 1) if ":" in key else ("", key)       # as the audit's named_identity splits
        if stem:
            side = tail.split("/")[0]
            src = RESULTS_DIR / f"{stem}.json"
            if src.is_file():
                scfg = read_result(src).get("config") or {}
                repo, rev = scfg.get(f"{side}_repo"), scfg.get(f"{side}_rev")
                sources[_repo_rel(src)] = _RESULT_READS[_repo_rel(src)]
            else:
                run = REG.run(stem)
                repo, rev = run.get(f"{side}_repo"), run.get(f"{side}_rev")
                sources.update(_registry_sources(REG, repo))
            name = f"{stem}:{side}"
        elif key.endswith(".npz") and "@" in key:
            left, right = key.split("@", 1)
            parts = left.split("__")
            if len(parts) != 2:
                raise SystemExit(f"REFUSING: {key}: not '<org>__<model>@<rev>__...' (a tagged cache is not named here)")
            repo, rev = f"{parts[0]}/{parts[1]}", REG.revision(f"{parts[0]}/{parts[1]}")
            if not rev.startswith(right.split("__", 1)[0]):
                raise SystemExit(f"REFUSING: {key}: the registry's revision of {repo} is {rev[:12]}, not the name's")
            sources.update(_registry_sources(REG, repo))
            name = key
        else:
            raise SystemExit(f"REFUSING: cannot name the repo and revision of cache input {key!r}")
        if not (repo and rev and len(str(rev)) in (40, 64)):              # a commit, or a LoRA adapter's sha256
            raise SystemExit(f"REFUSING: {key}: no full repo ID and revision found ({repo!r}, {rev!r})")
        for k, v in ((f"{name}_repo", repo), (f"{name}_rev", rev)):
            if cfg.setdefault(k, v) != v:
                raise SystemExit(f"REFUSING: {k} named twice with different values")
    return cfg, sources


def reissue_compare(payload: dict[str, Any], first_rel: str, first_sha: str, added_config: list[str],
                    added_top: tuple[str, ...] = ("reissue_of",)) -> dict[str, Any]:
    """D-297 (1), D-300: a re-issue equals its first record in every key but the run's header (REISSUE_MAY_DIFFER;
    the preregistration must extend the first record's), the config keys it adds (the identity entries, absent from
    the first record) and `added_top`; a 'config' block is new only when the first record had none."""
    b = (REPO_ROOT / first_rel).read_bytes()
    if sha256_bytes(b) != first_sha:
        raise SystemExit(f"REFUSING to write: {first_rel} is not the first record ({first_sha[:8]})")
    r1 = json.loads(b.decode("utf-8"))
    mine = json.loads(json.dumps(payload, default=str))
    top = set(added_top) | ({"config"} if "config" not in r1 else set())
    if set(mine) - top != set(r1):
        raise SystemExit(f"REFUSING to write: top-level keys differ from {first_rel}: {sorted((set(mine) - top) ^ set(r1))}")
    r1cfg, cfg = r1.get("config") or {}, mine.get("config") or {}
    if any(k in r1cfg for k in added_config):
        raise SystemExit(f"REFUSING: {first_rel} already names some of these inputs; nothing to re-issue")
    if set(cfg) - set(added_config) != set(r1cfg):
        raise SystemExit(f"REFUSING to write: config keys differ from {first_rel} beyond the identity entries")
    if not str(mine.get("preregistration", "")).startswith(str(r1.get("preregistration", ""))):
        raise SystemExit(f"REFUSING to write: the preregistration does not extend {first_rel}'s")
    mine["config"] = {k: v for k, v in cfg.items() if k not in added_config}
    compared = [k for k in r1 if k not in REISSUE_MAY_DIFFER]
    differ = [k for k in compared if json.dumps(mine[k], sort_keys=True) != json.dumps(r1[k], sort_keys=True)]
    if differ:
        raise SystemExit(f"REFUSING to write: {differ} differ from {first_rel} (D-297 (1), D-300)")
    return {"file": first_rel, "sha256": first_sha, "compared": True, "keys_compared": compared,
            "keys_allowed_to_differ": list(REISSUE_MAY_DIFFER), "config_keys_added": sorted(added_config),
            "method": "the payload serialised as written and parsed back; each compared key gives the same "
                      "json.dumps(sort_keys=True) text as the first record's, config without the added identity entries",
            "result": "every compared key equals the first record exactly"}


def reissue_payload(payload: dict[str, Any], first_rel: str, first_sha: str,
                    new_keys: dict[str, Any] | None = None) -> dict[str, Any]:
    """D-300: `payload` re-issued as the record `first_rel` (sha256 `first_sha`) plus config entries naming every cache
    input's repo ID and revision (identity_config); refuses unless every other key equals the first record's.
    `new_keys` names top-level keys the current code writes that the first record predates, each with the only value
    it may hold here (checked)."""
    for k, v in (new_keys or {}).items():
        if json.dumps(payload.get(k), sort_keys=True) != json.dumps(v, sort_keys=True):
            raise SystemExit(f"REFUSING: {k} is {payload.get(k)!r}, not {v!r}")
    idc, sources = identity_config(list(payload["input_cache_sha256"]))
    out = dict(payload)
    out["preregistration"] = (f"{payload.get('preregistration', '')}; re-issued naming each cache input's repo ID and "
                              "revision (D-297 (1), D-300)")
    out["config"] = {**(payload.get("config") or {}), **idc}
    out["reissue_of"] = {**reissue_compare(out, first_rel, first_sha, list(idc), ("reissue_of", *(new_keys or {}))),
                         "identity_sources_sha256": sources, "top_level_keys_added_by_newer_code": dict(new_keys or {})}
    return out


def write_results(path: str | Path, payload: dict[str, Any], exclusive: bool = False) -> str:
    """Write a results JSON and return its own SHA-256.

    The file's hash is printed by callers so a run can always be referred to by
    the checksum of its output, not just its filename.

    exclusive=True (D-152 P5, "nothing on disk is overwritten"): the verified temporary
    file is HARD-LINKED to the final name, which fails if that name exists - including
    one created while the run was going - instead of os.replace, which would overwrite it.

    D-297 (1): refuses, before any byte is written, a record that reads a cache without naming every cache
    input's repo ID and revision, or that read results (through read_result) without listing them with their
    sha256.
    """
    problems = input_identity_problems(payload)
    if problems:
        raise SystemExit(f"REFUSING to write {Path(path).name} (D-297 (1) results-writer rule): {problems}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False, default=str)
    # F-027: write-then-rename, so a failed write (quota, lost link) never leaves a
    # half-written record under the real name.
    data = text.encode("utf-8")
    want = sha256_bytes(data)
    # exclusive create with a random suffix: never truncates or later removes a file this call did not create
    tmp = path.with_name(f"{path.name}.partial.{os.getpid()}.{os.urandom(4).hex()}")
    linked = False
    created = False          # set only once THIS call's exclusive open succeeded: the cleanup never touches another file
    try:
        with open(tmp, "xb") as fh:
            created = True
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        # F-032: on this NAS a quota failure CREATES the file and writes nothing, so a
        # zero exit status is not evidence the bytes landed. Check the size first, then
        # the hash, then that the final path still parses as the JSON we meant to write.
        got = tmp.stat().st_size
        if got != len(data):
            raise OSError(f"results verify: wrote {got} bytes of {len(data)} to {tmp.name} "
                          f"-- the filesystem accepted the write but did not store it")
        if sha256_file(tmp) != want:
            raise OSError("results verify: hash differs after write")
        if exclusive:
            os.link(tmp, path)   # FileExistsError if the name exists: never overwrite
            linked = tmp.stat().st_nlink >= 2     # the link landed: the bytes now have a second name
        else:
            os.replace(tmp, path)
        # Read back the FINAL path. os.replace can succeed while the directory entry is
        # not durable, and a truncated file with the right size is still possible on a
        # stale NFS handle, so this parses what a later reader would actually get.
        final = path.read_bytes()
        if sha256_bytes(final) != want:
            raise OSError(f"results verify: {path.name} differs from what was written")
        json.loads(final.decode("utf-8"))
    finally:
        # The temporary is this call's own file (exclusive create). After a successful hard link its name is a second
        # name of the final bytes, so removing it deletes nothing. A never-named temporary is removed only up to 1 MB:
        # a larger one is left and named, because files over 1 MB are removed only by scripts/safe_rm.sh (D-307 review).
        if created and tmp.exists():
            if linked or tmp.stat().st_size < 1_000_000:        # the same boundary as safe_outputs.DELETE_LIMIT
                tmp.unlink()
            else:
                print(f"LEFT IN PLACE (over 1 MB, never named): {tmp} {tmp.stat().st_size} bytes; remove with "
                      f"scripts/safe_rm.sh", file=sys.stderr)
    return sha256_bytes(data)
