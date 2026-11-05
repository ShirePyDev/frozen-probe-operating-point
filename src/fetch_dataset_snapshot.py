"""Fetch one dataset repo at a pinned revision into an NVMe directory laid out as a hub cache (D-303).

src/phase2_pool_clusters.py and src/phase2_freeze_splits.py read WildGuardMix only through
freeze_splits.load_axis(local_hub=...), offline. They have no Hub path of their own, because from the node's login
shell `datasets` would follow HF_ENDPOINT to a plain-HTTP mirror that is not ours. This script makes that local
snapshot:
    <hub-dir>/datasets--<org>--<name>/snapshots/<rev>/<path>      (plain files; load_dataset reads the directory)

PIN     the revision the Phase 1 freeze recorded: results/dataset_manifest_wildguard.json (read through
        provenance.read_result), axis safety_wildguard, its source and source_revision_sha. The Hub's file list for
        that commit (https://huggingface.co/api/datasets/<repo>/revision/<rev>?blobs=true) must resolve exactly it and
        name each file once.
        --lora-corpus alpaca|hh|pku (D-343 (2)): a Phase 2 LoRA corpus at the revision its Phase 1 training record holds
        (LORA_CORPORA: results/lora_train__<run>.json, read through provenance.read_result; repo = config.dataset,
        revision = dataset_revision, run_kind lora_train). Phase 1 recorded the revision but no file hashes; this fetch
        adds them. The corpora are public: a gated answer refuses before any token is read (as --resolve), so no token
        is read or sent. The record's pin names the training record with its sha256.
        --resolve p2sst (D-338 (1)): GLUE's SST-2 has no Phase 1 pin, so its revision is the commit the Hub resolves
        when it is fetched. The public metadata of RESOLVED['p2sst'] (nyu-mll/glue, the GLUE repo whose sst2
        configuration is parquet files under sst2/), read with no token and no redirect followed, must name that repo,
        a 40-hex sha and no gating; the file list for that sha must then resolve exactly it, as above. A gated answer
        refuses before any token is read, so no token is ever read or sent for it. The record's pin block says so.
FILES   every file the commit lists. An LFS file is checked against the sha256 the listing publishes; a small file
        against its git blob sha1 (recorded as its blob_id); an LFS pointer committed as a plain blob (F-068) against
        the sha256 it names. Any mismatch refuses.
FETCH   the safe fetch layer already in the repo:
  - every request to huggingface.co goes through phase2_config_check.get(): https://huggingface.co only, redirects
    refused, trust_env=False (no credential from the environment or ~/.netrc);
  - the token is fetch_weights.load_token() (.env only), attached by fetch_weights.auth_headers() (to
    https://huggingface.co only), and only when the repo's public metadata says it is gated. It is never printed
    or written;
  - an LFS object's redirect to storage is ONE request on a fresh requests.Session (trust_env False) with no token.
    It must be https, to a host under .hf.co or .huggingface.co, with no userinfo and no port but 443. The URL may hold
    no backslash, whitespace, control or non-ASCII character. The host urllib3 parses must equal the host urlsplit
    parses, and an unparseable URL is refused without being printed. Redirects are refused. Only its host is
    recorded, never the presigned URL.
TARGET  the hub directory, and the staging directory made under it, must be on local NVMe (hf_env_guard.mount_of: the
        filesystem's source is an NVMe device), and the snapshot directory must not exist. Files are written into the
        staging directory, re-hashed, and the directory is renamed into place by a no-replace rename
        (src/safe_outputs.py: renameat2 RENAME_NOREPLACE on Linux), then its files are verified under the final name.
RECORD  results/phase2/fetch_dataset__<org>__<name>@<rev12>.json, written by provenance.write_results(out, payload,
        exclusive=True) after the snapshot is in place: the files with sizes, hashes and blob ids, the seconds, the MB/s
        and token_sent per request family. Never the token, never a presigned URL.
NOTHING IS DELETED (the author's standing rules; this runs as a background queue job). On a failure before the snapshot
        is in place, the staging directory is left where it is. On a failure after, the snapshot directory, if its
        files are this call's, is renamed aside to '<rev>.failed.<pid>'. The exit names every file and directory left;
        removal is a human's scripts/safe_rm.sh call.

  python3 src/fetch_dataset_snapshot.py --hub-dir <NVMe dir>      (hf_env_guard runs first)
  python3 src/fetch_dataset_snapshot.py --hub-dir <NVMe dir> --resolve p2sst      (D-338: GLUE SST-2, public, no token)
  python3 src/fetch_dataset_snapshot.py --hub-dir <NVMe dir> --lora-corpus hh      (D-343 (2): Phase 1's pinned revision)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_weights as FW  # noqa: E402
import hf_env_guard as HG  # noqa: E402
import paired as P  # noqa: E402
import phase2_config_check as CC  # noqa: E402
import safe_outputs as SO  # noqa: E402
from provenance import (  # noqa: E402
    _RESULT_READS, _repo_rel, ensure_utf8_stdout, input_identity_problems, provenance_block, read_result,
    sha256_bytes, sha256_file, write_results,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PIN_MANIFEST = RESULTS / "dataset_manifest_wildguard.json"
PIN_AXIS = "safety_wildguard"
# D-338 (1): axes with no Phase 1 pin, fetched at the commit the Hub resolves when they are fetched (public repos only)
RESOLVED = {"p2sst": {"repo": "nyu-mll/glue", "decision": "D-338",
                      "rule": "D-338 (1): GLUE's SST-2 (the GLUE dataset repo's sst2 configuration) at the commit the "
                              "Hub resolves when it is fetched; every file's sha256 checked against the Hub; a public "
                              "repo, so no token is read or sent"}}
# D-343 (2): the Phase 2 LoRA corpora, each at the revision its Phase 1 training record holds (public repos only)
LORA_CORPORA = {"alpaca": "lora_train__mistral7b-v01-alpaca-r16.json",
                "hh": "lora_train__mistral7b-v01-hhharmless-r16.json",
                "pku": "lora_train__mistral7b-v01-pkusafe-r16.json"}
LORA_RULE = ("D-343 (2): the corpus at the revision its Phase 1 training record holds (dataset_revision; repo "
             "config.dataset), fetched pinned with every file's sha256 checked against the Hub (Phase 1 recorded the "
             "revision but no file hashes); a public repo, so no token is read or sent")
CODE_FILES = ["src/fetch_dataset_snapshot.py", "src/phase2_config_check.py", "src/fetch_weights.py",
              "src/hf_env_guard.py", "src/provenance.py", "src/paired.py", "src/safe_outputs.py"]
HUB = CC.HUB
STORAGE_HOST_SUFFIXES = (".hf.co", ".huggingface.co")
REDIRECTS = (301, 302, 303, 307, 308)
REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")


def git_blob_sha1(data: bytes) -> str:
    """Git's object id: sha1 over 'blob <size>\\0' + content."""
    h = hashlib.sha1()
    h.update(f"blob {len(data)}\0".encode())
    h.update(data)
    return h.hexdigest()


def storage_host(url: str) -> str:
    """The host of an LFS storage redirect, or a refusal. The URL is presigned: it is never printed, not even in a
    traceback (every parser error becomes a plain refusal)."""
    if not url or any(c == "\\" or c.isspace() or ord(c) < 0x20 or ord(c) >= 0x7F for c in url):
        raise SystemExit("REFUSING: an LFS redirect whose URL is empty or holds a backslash, whitespace, control or "
                         "non-ASCII character; nothing is fetched from it")
    try:
        from urllib3.util import parse_url
        parts = urllib.parse.urlsplit(url)
        host = (parts.hostname or "").lower()
        port = parts.port
        u3_host = (parse_url(url).host or "").lower()
    except Exception:                                   # ValueError, urllib3's LocationParseError, anything else
        raise SystemExit("REFUSING: unparseable LFS redirect; nothing is fetched from it") from None
    if parts.scheme != "https" or "@" in parts.netloc or port not in (None, 443) \
            or not host.endswith(STORAGE_HOST_SUFFIXES):
        raise SystemExit(f"REFUSING: an LFS redirect to {parts.scheme or '?'}://{host or '?'}, which is not the Hub's "
                         "https storage; nothing is fetched from it")
    if u3_host != host:
        raise SystemExit(f"REFUSING: an LFS redirect whose host parses differently in urllib3 ({host}); nothing is "
                         "fetched from it")
    return host


def storage_get(url: str) -> tuple[bytes, str]:
    """The one redirect hop of an LFS object: a SEPARATE request on a fresh session (trust_env False), no token,
    redirects refused."""
    host = storage_host(url)
    h = dict(CC.UA)
    if "Authorization" in h:
        raise SystemExit("REFUSING: a storage request would carry a token")
    sess = requests.Session()
    sess.trust_env = False                              # no credential from the environment or ~/.netrc
    last = None
    for attempt in range(5):
        try:
            r = sess.get(url, headers=h, timeout=120, allow_redirects=False)
        except requests.RequestException as e:
            last = type(e).__name__
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            last = f"HTTP {r.status_code}"
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code != 200:
            raise SystemExit(f"REFUSING: the storage host {host} answered HTTP {r.status_code} (no further hop)")
        return r.content, host
    raise SystemExit(f"REFUSING: the storage host {host} kept failing ({last})")


def safe_name(name: str) -> str:
    """A listed file name that stays inside the snapshot directory."""
    if (not isinstance(name, str) or not name or name.startswith("/") or "\\" in name
            or any(ord(c) < 0x20 or ord(c) == 0x7F for c in name)
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise SystemExit(f"REFUSING: the listing names a file outside a plain relative path: {name!r}")
    return name


def lfs_object(base: str, rev: str, q: str, gated: bool) -> tuple[bytes, dict[str, Any]]:
    """An LFS object: /resolve on huggingface.co by commit (the token only for a gated repo, redirects refused); a
    redirect to the Hub's storage is followed by storage_get, with no token."""
    r = CC.get(f"{base}/resolve/{rev}/{q}", use_token=gated)
    info: dict[str, Any] = {"token_sent": bool(getattr(r, "token_sent", False)), "object_host": "huggingface.co",
                            "storage_request_token_sent": None}
    if r.status_code == 200:
        return r.content, info
    if r.status_code not in REDIRECTS:
        raise SystemExit(f"REFUSING: /resolve answered HTTP {r.status_code}")
    data, host = storage_get(r.headers.get("Location") or "")
    info.update(object_host=host, storage_request_token_sent=False)
    return data, info


def fetch_file(repo: str, rev: str, name: str, sib: dict[str, Any], gated: bool) -> tuple[bytes, dict[str, Any]]:
    """One file by commit, checked against the hash the Hub publishes; its bytes and its record."""
    base, q = f"{HUB}/datasets/{repo}", urllib.parse.quote(name, safe="/")
    lfs = sib.get("lfs") or {}
    rec: dict[str, Any] = {"name": name, "lfs": bool(lfs)}
    if lfs:
        want, size = lfs.get("sha256") or lfs.get("oid"), lfs.get("size", sib.get("size"))
        if not want:
            raise SystemExit(f"REFUSING: {name} is an LFS file with no published sha256")
        data, info = lfs_object(base, rev, q, gated)
        rec.update(info, hub_check="sha256 the Hub publishes for the LFS object")
    else:
        blob, size = sib.get("blobId") or sib.get("oid"), sib.get("size")
        if not blob:
            raise SystemExit(f"REFUSING: {name} has no published git blob id")
        r = CC.get(f"{base}/raw/{rev}/{q}", use_token=gated)
        if r.status_code != 200:
            raise SystemExit(f"REFUSING: {name}: /raw answered HTTP {r.status_code}")
        data = r.content
        if git_blob_sha1(data) != blob or (size is not None and len(data) != int(size)):
            raise SystemExit(f"HASH MISMATCH: {name}: {len(data)} bytes, blob sha1 {git_blob_sha1(data)[:16]} != "
                             f"{str(blob)[:16]}")
        rec.update(blob_id=blob, token_sent=bool(getattr(r, "token_sent", False)), object_host="huggingface.co",
                   storage_request_token_sent=None, hub_check="git blob sha1 the Hub publishes for the file")
        want = None
        pointer = FW.lfs_pointer(data) if len(data) <= FW.POINTER_MAX else None
        if pointer:                       # F-068: an LFS pointer committed as a plain blob; fetch its object
            want, size = pointer
            data, info = lfs_object(base, rev, q, gated)
            rec.update(object_host=info["object_host"], storage_request_token_sent=info["storage_request_token_sent"],
                       token_sent=rec["token_sent"] or info["token_sent"], f068_pointer=True,
                       hub_check="sha256 named by the committed LFS pointer (F-068)")
    got = sha256_bytes(data)
    if (want and got != want) or (size is not None and len(data) != int(size)):
        raise SystemExit(f"HASH MISMATCH: {name}: {len(data)} bytes, sha256 {got[:16]} != {str(want)[:16]}")
    rec.update(sha256=got, size=len(data))
    return data, rec


def fetch_snapshot(repo: str, rev: str, staging: Path, public_only: bool = False) -> dict[str, Any]:
    """Every file of repo@rev into `staging` (which must exist and be empty), each verified; the record of it.
    public_only (--resolve, D-338): a gated repo refuses here, before any request that could carry the token."""
    m = CC.get(f"{HUB}/api/datasets/{repo}", use_token=False)
    if m.status_code != 200:
        raise SystemExit(f"REFUSING: the public metadata of {repo} answered HTTP {m.status_code}")
    gated_value = m.json().get("gated")
    gated = bool(gated_value)
    if public_only and gated:
        raise SystemExit(f"REFUSING: {repo} is gated ({gated_value!r}); this axis is fetched from a public repo with no "
                         "token (D-338 (1))")
    r = CC.get(f"{HUB}/api/datasets/{repo}/revision/{rev}?blobs=true", use_token=gated)
    if r.status_code != 200:
        raise SystemExit(f"REFUSING: the file list of {repo}@{rev[:12]} answered HTTP {r.status_code}")
    body = r.json()
    if body.get("sha") != rev:
        raise SystemExit(f"REFUSING: {repo}: asked for commit {rev[:12]}, the Hub listed {str(body.get('sha'))[:12]}")
    listed = [safe_name(s.get("rfilename")) for s in body.get("siblings") or []]
    if len(set(listed)) != len(listed):
        raise SystemExit(f"REFUSING: the listing of {repo}@{rev[:12]} names a file more than once")
    sibs = {s["rfilename"]: s for s in body.get("siblings") or []}
    if not sibs:
        raise SystemExit(f"REFUSING: {repo}@{rev[:12]} lists no files")
    files = []
    for name in sorted(sibs):
        data, rec = fetch_file(repo, rev, name, sibs[name], gated)
        path = staging / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "xb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        files.append(rec)
    return {"gated": gated_value, "metadata_token_sent": bool(getattr(m, "token_sent", False)),
            "listing_token_sent": bool(getattr(r, "token_sent", False)), "files": files}


def resolve_revision(repo: str) -> str:
    """D-338 (1): the commit the Hub resolves for `repo` now, from its public metadata: no token, no redirect followed
    (phase2_config_check.get). It must name this repo, a 40-hex sha and no gating."""
    if not REPO_RE.fullmatch(repo):
        raise SystemExit(f"REFUSING: {repo!r} is not a repo id")
    m = CC.get(f"{HUB}/api/datasets/{repo}", use_token=False)
    if m.status_code != 200:
        raise SystemExit(f"REFUSING: the public metadata of {repo} answered HTTP {m.status_code} (no redirect is "
                         "followed)")
    meta = m.json()
    sha = meta.get("sha")
    if meta.get("id") != repo or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise SystemExit(f"REFUSING: the public metadata of {repo} names {meta.get('id')!r}@{sha!r}, not this repo at a "
                         "40-hex commit")
    if meta.get("gated"):
        raise SystemExit(f"REFUSING: {repo} is gated ({meta.get('gated')!r}); this axis is fetched from a public repo "
                         "with no token (D-338 (1))")
    return sha


def public_or_refuse(repo: str) -> None:
    """--lora-corpus (D-343 (2)): the repo's public metadata (no token, no redirect followed) must name this repo and no
    gating, checked before any directory is made; fetch_snapshot(public_only=True) checks the gating again."""
    m = CC.get(f"{HUB}/api/datasets/{repo}", use_token=False)
    if m.status_code != 200:
        raise SystemExit(f"REFUSING: the public metadata of {repo} answered HTTP {m.status_code} (no redirect is "
                         "followed)")
    meta = m.json()
    if meta.get("id") != repo:
        raise SystemExit(f"REFUSING: the public metadata of {repo} names {meta.get('id')!r}")
    if meta.get("gated"):
        raise SystemExit(f"REFUSING: {repo} is gated ({meta.get('gated')!r}); the LoRA corpora are fetched from public "
                         "repos with no token (D-343 (2))")


def on_nvme_or_refuse(path: Path, what: str) -> tuple[str, str]:
    try:
        mnt, src = HG.mount_of(path)
    except OSError as e:
        raise SystemExit(f"REFUSING: cannot tell which device holds the {what} {path} ({type(e).__name__})")
    if not src.startswith(HG.NVME_SOURCE_PREFIX):
        raise SystemExit(f"REFUSING: the {what} {path} is on {src or 'an unknown device'} (mounted at {mnt or '?'}), "
                         "not local NVMe")
    return mnt, src


def files_match(d: Path, files: list[dict[str, Any]]) -> bool:
    """Whether directory d holds exactly these files with these sha256."""
    on_disk = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
    return on_disk == sorted(f["name"] for f in files) and all(sha256_file(d / f["name"]) == f["sha256"] for f in files)


def staging_left(staging: Path) -> str:
    n = [p for p in staging.rglob("*") if p.is_file()] if staging.is_dir() else []
    return f"{staging} (the staging directory, never given the snapshot's name; {len(n)} files, " \
           f"{sum(p.stat().st_size for p in n)} bytes; left in place)"


def main() -> int:
    ensure_utf8_stdout()
    HG.assert_hf_env("fetch_dataset_snapshot.py")
    ap = argparse.ArgumentParser(description="Fetch a pinned dataset snapshot to NVMe as a hub cache (D-303)")
    ap.add_argument("--hub-dir", required=True, help="an NVMe directory; the snapshot goes under it")
    ap.add_argument("--pin-manifest", default=None, help=f"default {PIN_MANIFEST.relative_to(ROOT).as_posix()}")
    ap.add_argument("--axis", default=None, help=f"the pin manifest's axis (default {PIN_AXIS})")
    ap.add_argument("--resolve", default=None, choices=sorted(RESOLVED),
                    help="D-338 (1): pin the commit the Hub resolves now for this axis's registered repo (no manifest)")
    ap.add_argument("--lora-corpus", default=None, choices=sorted(LORA_CORPORA),
                    help="D-343 (2): a Phase 2 LoRA corpus at the revision its Phase 1 training record holds")
    ap.add_argument("--pin-record", default=None,
                    help="TEST ONLY (with --out): the Phase 1 training record read for --lora-corpus")
    ap.add_argument("--out", default=None, help="TEST ONLY: a record path outside the repository")
    args = ap.parse_args()

    if args.pin_record is not None and not args.lora_corpus:
        raise SystemExit("REFUSING: --pin-record is for --lora-corpus only")
    pin_of_record = PIN_MANIFEST
    if args.resolve and args.lora_corpus:
        raise SystemExit("REFUSING: --resolve and --lora-corpus are two different pins; give one")
    if args.lora_corpus:                                       # D-343 (2): the revision Phase 1's record holds
        if args.pin_manifest is not None or args.axis is not None:
            raise SystemExit("REFUSING: --lora-corpus takes no --pin-manifest or --axis; its pin is the Phase 1 "
                             "training record")
        pin_of_record = RESULTS / LORA_CORPORA[args.lora_corpus]
        pin_path = Path(args.pin_record) if args.pin_record is not None else pin_of_record
        doc = read_result(pin_path)                            # registered: listed in the record's inputs
        prel = _repo_rel(pin_path)
        if not isinstance(doc, dict) or doc.get("run_kind") != "lora_train":
            raise SystemExit(f"REFUSING: {prel} is not a Phase 1 LoRA training record (run_kind lora_train)")
        repo, rev = (doc.get("config") or {}).get("dataset"), doc.get("dataset_revision")
        if not isinstance(repo, str) or not REPO_RE.fullmatch(repo) or not isinstance(rev, str) \
                or not re.fullmatch(r"[0-9a-f]{40}", rev):
            raise SystemExit(f"REFUSING: {prel} names {repo!r}@{rev!r}, not a repo id and a 40-hex commit")
        pin = {"file": prel, "sha256": _RESULT_READS[prel], "lora_corpus": args.lora_corpus,
               "repo_field": "config.dataset", "revision_field": "dataset_revision", "rule": LORA_RULE,
               "public_only": True}
        inputs = {prel: _RESULT_READS[prel]}
        decision = "D-343"
    elif args.resolve:                                         # D-338 (1): the commit the Hub resolves now
        if args.pin_manifest is not None or args.axis is not None:
            raise SystemExit("REFUSING: --resolve takes no --pin-manifest or --axis; its revision is the Hub's now")
        res = RESOLVED[args.resolve]
        repo = res["repo"]
        rev = resolve_revision(repo)
        pin_path = None
        pin: dict[str, Any] = {"resolved_at_fetch": True, "axis": args.resolve, "rule": res["rule"],
                               "metadata_url": f"{HUB}/api/datasets/{repo}", "metadata_sha": rev,
                               "metadata_token_sent": False, "public_only": True}
        inputs: dict[str, str] = {}
        decision = res["decision"]
    else:
        pin_path = Path(args.pin_manifest if args.pin_manifest is not None else PIN_MANIFEST)
        doc = read_result(pin_path)                            # registered: listed in the record's inputs
        prel = _repo_rel(pin_path)
        axis = args.axis if args.axis is not None else PIN_AXIS
        entry = doc["axes"][axis]
        repo, rev = entry["source"], entry["source_revision_sha"]
        if not REPO_RE.fullmatch(repo) or not re.fullmatch(r"[0-9a-f]{40}", rev):
            raise SystemExit(f"REFUSING: {prel} pins {repo!r}@{rev!r}, not a repo id and a 40-hex commit")
        pin = {"file": prel, "sha256": _RESULT_READS[prel], "axis": axis}
        inputs = {prel: _RESULT_READS[prel]}
        decision = "D-303"
    org, name = repo.split("/")
    final_out = (RESULTS / "phase2" / f"fetch_dataset__{org}__{name}@{rev[:12]}.json").resolve()
    out = Path(args.out).resolve() if args.out else final_out
    final = out == final_out
    if final and pin_path is not None and pin_path.resolve() != pin_of_record.resolve():
        raise SystemExit("REFUSING the final name: it needs the pin of record")
    if not final and out.is_relative_to(ROOT.resolve()):
        raise SystemExit("REFUSING: a test run (--out) writes outside the repository")
    if out.exists():
        raise SystemExit(f"REFUSING: {out} exists; it is never overwritten")
    hub = Path(args.hub_dir).resolve()
    mnt, src = on_nvme_or_refuse(hub, "hub directory")
    snap = hub / f"datasets--{org}--{name}" / "snapshots" / rev
    if os.path.lexists(snap):
        raise SystemExit(f"REFUSING: {snap} exists; a snapshot is never overwritten")
    code0 = P.code_block(CODE_FILES)
    if args.lora_corpus:
        public_or_refuse(repo)                                 # before any directory: a gated repo creates nothing

    staging = snap.with_name(f"{rev}.partial.{os.getpid()}.{os.urandom(4).hex()}")
    staging.mkdir(parents=True)
    try:
        if not staging.resolve().is_relative_to(hub):
            raise SystemExit(f"REFUSING: the staging directory {staging} resolves outside {hub}")
        on_nvme_or_refuse(staging, "staging directory")
        t0 = time.monotonic()
        got = fetch_snapshot(repo, rev, staging, public_only=bool(args.resolve or args.lora_corpus))
        secs = time.monotonic() - t0
        if not files_match(staging, got["files"]):
            raise SystemExit("REFUSING: the staging directory does not hold exactly the bytes verified")
        total = sum(f["size"] for f in got["files"])
        if P.code_block(CODE_FILES)["sha256"] != code0["sha256"]:
            raise SystemExit("REFUSING to write: the code changed during the run")
        token_sent = got["metadata_token_sent"] or got["listing_token_sent"] or any(f["token_sent"] for f in got["files"])
        payload = {**provenance_block("fetch_dataset_snapshot"), "decision": decision, "code": code0,
                   "repo": repo, "repo_type": "dataset", "revision": rev, "gated": got["gated"],
                   "pin": pin,
                   "inputs": inputs,
                   "target": {"hub_dir": str(hub), "snapshot": f"datasets--{org}--{name}/snapshots/{rev}",
                              "mount_point": mnt, "mount_source": src},
                   "token_sent": token_sent, "metadata_token_sent": got["metadata_token_sent"],
                   "listing_token_sent": got["listing_token_sent"],
                   "token_use": "fetch_weights.load_token (.env), attached by fetch_weights.auth_headers to "
                                "https://huggingface.co only, only for a gated repo; redirects refused; the storage hop "
                                "is a fresh session with no token; never printed or written",
                   "storage_hosts": sorted({f["object_host"] for f in got["files"] if f["object_host"] != "huggingface.co"}),
                   "n_files": len(got["files"]), "total_bytes": total, "seconds": round(secs, 3),
                   "mb_per_s": round(total / 1e6 / secs, 3) if secs > 0 else None, "files": got["files"]}
        problems = input_identity_problems(payload)
        if problems:
            raise SystemExit(f"REFUSING to write {out.name} (D-297 (1) results-writer rule): {problems}")
        if os.path.lexists(out):
            raise SystemExit(f"REFUSING: {out} appeared during the run")
        try:
            SO.noreplace_rename(staging, snap)
        except SO.MovedUnverified:
            pass               # it moved: the files are verified under the final name below, and renamed aside if not
        except OSError as e:
            raise SystemExit(f"REFUSING: the snapshot name {snap} appeared during the run ({type(e).__name__})")
    except BaseException as e:
        msg = e.code if isinstance(e, SystemExit) else f"{type(e).__name__}: {e}"
        raise SO.failure(str(msg), [staging_left(staging)]) from None

    # the snapshot is in place under its final name: from here a failure renames it aside, never removes it
    try:
        if not files_match(snap, got["files"]):
            raise SystemExit(f"REFUSING: {snap} does not hold exactly the bytes verified (the rename was not ours?)")
        digest = write_results(out, payload, exclusive=True)
    except BaseException as e:
        msg = e.code if isinstance(e, SystemExit) else f"{type(e).__name__}: {e}"
        left = [SO.rename_aside(snap, None, matches=lambda d: files_match(d, got["files"]))]
        if os.path.lexists(out):
            left.append(f"{out} (the record; it exists: written by this call, or by another)")
        raise SO.failure(str(msg), left) from None
    print(f"{repo}@{rev[:12]}: {len(got['files'])} files, {total} bytes in {secs:.1f} s; token sent {token_sent}")
    print(f"Snapshot: {snap}\nWritten: {_repo_rel(out)}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
