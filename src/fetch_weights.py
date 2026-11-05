"""Fetch model weights to LOCAL NVMe, safetensors only, with the SHAs verified.

Constraints this exists to enforce, none of them optional:
  * weights never touch the NAS -- the download target is under ~/.cache, which is
    /dev/nvme1n1p1, not the quota-bound NFS home;
  * safetensors only -- .bin is a pickle. The one exception is src/bin_weights.py's BIN_ALLOWED (two repos, each at
    one revision; D-318 (7) item 8): their .bin shards are fetched and verified here like any file, then converted to
    safetensors on NVMe (torch.load(weights_only=True) of the bytes hashed, compared bit for bit), so nothing
    downstream loads a pickle;
  * the resolved commit must equal the pinned revision, so a moved branch cannot
    silently swap the weights;
  * every downloaded file's hash is checked against what the Hub says it should be.
    For LFS files the Hub publishes the sha256 of the content; for small plain files
    it publishes a git blob oid, which is a sha1 over a different preimage, so the
    two cases are checked differently rather than pretending one rule covers both.

Deletion is a separate call (--purge), so weights are removed as soon as the extraction
that needed them has finished. It is confined to this repo's directory in the NVMe HF
cache and refuses any other path (D-153); it lists the files before and after. Everywhere
else, files over 1 MB are removed only through scripts/safe_rm.sh, one per call (F-054).

Environment (2026-09-24): src/hf_env_guard.py refuses to run unless HF_HOME is on NVMe,
HF_ENDPOINT is unset or https, and no token is exported. The campus mirror is reached
only through --mirror, never through HF_ENDPOINT. The token is read from .env by
load_token() below and nowhere else, and is sent only to https://huggingface.co.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_env_guard import assert_hf_env  # noqa: E402
from provenance import ensure_utf8_stdout  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
API = "https://huggingface.co/api"
ALLOW = ["*.safetensors", "*.safetensors.index.json", "config.json",
         "generation_config.json", "tokenizer*", "*.model", "special_tokens_map.json"]
DENY = ["*.bin", "*.pth", "*.h5", "*.msgpack", "*.onnx", "*.gguf",
        "consolidated*.safetensors"]   # a second, unsharded copy of the same weights (mistral-inference); transformers
                                       # loads the sharded files through model.safetensors.index.json (Phase 2 review)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(p: Path) -> str:
    """Git's object id: sha1 over 'blob <size>\\0' + content."""
    data = p.read_bytes()
    h = hashlib.sha1()
    h.update(f"blob {len(data)}\0".encode())
    h.update(data)
    return h.hexdigest()


CANONICAL_HOST = "huggingface.co"


def load_token() -> str | None:
    """Read the HF token from .env ONLY. Never printed, never logged, never exported.

    The environment is deliberately not consulted: a token in the environment is
    inherited by every child process and is a policy violation that hf_env_guard
    refuses before this is ever called. The value stays a local of this process.
    """
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                return v.strip().strip("'\"")
    return None


def auth_headers(url: str, token: str | None) -> dict[str, str]:
    """Attach the token ONLY to https://huggingface.co.

    This is a hard guard on the URL rather than a promise about control flow. The
    campus mirror is plain HTTP and is not ours; sending a Hub credential to it would
    disclose the token to a third party and put it on the wire in clear text. Any
    other host, any other scheme, gets no Authorization header -- so a future edit
    that reroutes a download cannot leak the token by accident.
    """
    if not token:
        return {}
    parts = urllib.parse.urlsplit(url)
    if parts.scheme == "https" and parts.hostname == CANONICAL_HOST:
        return {"Authorization": f"Bearer {token}"}
    return {}


# F-069: four immediate tries could not outlast a gateway timeout of the Hub's CDN (HTTP 504 on one range,
# 2026-09-27T04:54Z, 1.95 GB into a 9.94 GB shard), and the purge that follows a failed fetch threw away
# about 2 h of verified download. Eight tries with exponential backoff (5 s doubling to 300 s, about 10 min
# in all); received bytes are kept across tries, as before.
RANGE_TRIES = 8
RANGE_BACKOFF_S = 5
RANGE_BACKOFF_MAX_S = 300
RETRY_HTTP = {408, 429, 500, 502, 503, 504}     # other HTTP statuses cannot become right on a retry


class RangeProtocolError(OSError):
    """An answer that does not fit the range asked for (Content-Range wrong or missing): never retried."""


def ranged_get(url: str, size: int, dest: Path, chunk: int = 64 << 20,
               token: str | None = None) -> None:
    """Download `url` into `dest` using HTTP Range requests.

    F-051, second act: the campus mirror advertises the correct Content-Length but
    returns a TRUNCATED body for a whole-file GET, stopping at one of two fixed
    offsets. The same mirror serves byte RANGES correctly and fast -- a 300 MB range
    came back complete at 23.7 MB/s, against a direct Hub transfer that could not
    finish 300 MB in five minutes. So the file is assembled from ranges instead.

    This is only safe because the caller verifies the finished file's sha256 against
    huggingface.co. Range assembly makes a fast but unreliable source usable; the hash
    is what makes it trustworthy.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    have = tmp.stat().st_size if tmp.exists() else 0
    with open(tmp, "ab") as fh:
        while have < size:
            end = min(have + chunk, size) - 1
            start = have
            for attempt in range(RANGE_TRIES):
                req = urllib.request.Request(url, headers={"Range": f"bytes={have}-{end}"})
                # The token is attached as an UNREDIRECTED header: urllib copies every ordinary
                # header onto a redirected request, and the Hub answers LFS resolves with a 302 to
                # its CDN, so a plain header would carry the token to a host that is not
                # huggingface.co (found by review, 2026-09-26; F-066). The CDN URL is presigned.
                for k, v in auth_headers(url, token).items():
                    req.add_unredirected_header(k, v)
                got = 0
                try:
                    with urllib.request.urlopen(req, timeout=180) as r:
                        # A partial response is recognised by its Content-Range, whatever the status:
                        # the Hub's CDN answers 206, the campus mirror answers 200 with a correct
                        # Content-Range (probed 2026-09-26, F-066 addendum). A server that ignores Range
                        # sends the file from byte 0 with no Content-Range: never append that.
                        # (The Hub's CDN was probed tokenless the same day: 206 with a Content-Range, so
                        # every partial answer, 206 or 200, must carry one.)
                        cr = r.headers.get("Content-Range")
                        if cr is not None:
                            m = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", cr.strip(), flags=re.IGNORECASE)
                            if not m or int(m.group(1)) != have or int(m.group(2)) > end or \
                                    m.group(3) not in ("*", str(size)):
                                raise RangeProtocolError(f"Content-Range {cr!r} does not answer bytes={have}-{end} of {size}")
                        elif not (r.status == 200 and have == 0 and end == size - 1):
                            raise RangeProtocolError(f"HTTP {r.status} without Content-Range to a range request")
                        while have + got <= end:
                            buf = r.read(min(1 << 20, end + 1 - have - got))
                            if not buf:
                                break
                            fh.write(buf)
                            got += len(buf)
                        if got == 0:
                            raise OSError("empty body")        # retried with backoff like any transient failure
                    fh.flush()
                    have += got
                    break
                except Exception as exc:  # noqa: BLE001
                    # Bytes received before the failure are a valid prefix of the range: they are
                    # KEPT and the next attempt asks for the rest. Nothing is truncated inside this
                    # retry loop (owner's rule: no destructive step in a retry); the size check below
                    # and the caller's sha256 against huggingface.co still decide.
                    fh.flush()
                    have += got
                    if fh.tell() != have:
                        raise                  # a local write error left the file out of step: never retried
                    if have > end:
                        break                  # the range is complete, whatever the close raised
                    code = getattr(exc, "code", None)
                    if isinstance(exc, RangeProtocolError) or (isinstance(exc, urllib.error.HTTPError)
                                                               and code not in RETRY_HTTP):
                        raise
                    if attempt == RANGE_TRIES - 1:
                        raise
                    wait = min(RANGE_BACKOFF_MAX_S, RANGE_BACKOFF_S * 2 ** attempt)
                    print(f"      range {start}-{end} attempt {attempt + 1} failed "
                          f"({type(exc).__name__}{f' {code}' if code else ''}); continuing from {have} in {wait} s",
                          flush=True)
                    time.sleep(wait)
            if fh.tell() != have:
                raise OSError(f"file position {fh.tell()} disagrees with bytes received {have}")
            if have == start:
                raise OSError(f"no progress at offset {have} for {url}")
            pct = 100.0 * have / size
            print(f"      {have / 1e9:.2f}/{size / 1e9:.2f} GB ({pct:.0f}%)", flush=True)
    if tmp.stat().st_size != size:
        raise OSError(f"assembled {tmp.stat().st_size} bytes, expected {size}")
    os.replace(tmp, dest)


# F-068: the model-info listing can describe a file as a plain git blob when the committed blob is an
# LFS POINTER (mistralai/Mistral-7B-Instruct-v0.1 @ ec5deb64, tokenizer.model.v1: "blobId" e523e8e1,
# "size" 130, no "lfs"), while /resolve serves the 493443-byte object it points to. Only the minimal
# pointer form (git-lfs spec v1) is recognised, with the final newline optional: that committed pointer
# has none (130 bytes; the Hub still resolves it). Anything else stays a plain file, whose bytes must then
# match its git blob id.
POINTER_MAX = 1024
LFS_POINTER = re.compile(rb"version https://git-lfs\.github\.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n?")


def lfs_pointer(data: bytes) -> tuple[str, int] | None:
    """(sha256, size) named by an LFS pointer, or None if `data` is not one."""
    m = LFS_POINTER.fullmatch(data)
    return (m.group(1).decode("ascii"), int(m.group(2))) if m else None


def committed_blob(repo: str, revision: str, name: str, blob_sha1: str, size: int, token: str | None) -> bytes:
    """The committed git blob of a small plain entry, from https://huggingface.co/<repo>/raw/<revision>/<name>
    (the git blob itself; /resolve would serve an LFS object for a pointer), checked against its git blob id."""
    url = f"https://{CANONICAL_HOST}/{repo}/raw/{revision}/{name}"
    data = b""
    for attempt in range(4):              # transport and HTTP status errors; a hash mismatch below is never retried
        req = urllib.request.Request(url, headers={"User-Agent": "guardrail-ccf-fetch"})
        for k, v in auth_headers(url, token).items():
            req.add_unredirected_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read(size + 1)
            break
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            if attempt == 3:
                raise
            print(f"      /raw {name} attempt {attempt + 1} failed ({type(exc).__name__}); retrying")
            time.sleep(2 ** attempt)
    h = hashlib.sha1()
    h.update(f"blob {len(data)}\0".encode())
    h.update(data)
    if len(data) != size or h.hexdigest() != blob_sha1:
        raise SystemExit(f"HASH MISMATCH for {name} (committed blob via /raw): {len(data)} bytes, blob-sha1 "
                         f"{h.hexdigest()[:16]} != {blob_sha1[:16]}")
    return data


def quarantine(blob: Path, got: str) -> Path:
    """Set a blob that failed its hash aside as <blob>.bad-<sha16>: quarantined, not deleted (review
    2026-09-26); the per-model purge (D-153 (d)), the approved deletion path for weights, removes it
    with the model's directory. An earlier copy of the same bad bytes is overwritten, not kept twice."""
    bad = blob.with_name(f"{blob.name}.bad-{got[:16]}")
    blob.replace(bad)
    return bad


# F-073: the campus mirror answered some 64 MiB range requests for Mistral-7B-Instruct-v0.1's shards with a body whose
# first 12-14 MiB were right and the rest ZEROS, under a correct Content-Range and length (every differing byte of the
# quarantined copy was 0; the bad sha256s match no file at any commit of the repo). A copy damaged that way is repaired
# by fetching from huggingface.co only the windows that hold a 1 MiB-aligned all-zero block; the whole file's sha256
# against huggingface.co still decides, and a repair that cannot finish or does not hash right falls back to the full
# re-fetch (D-281). Only a leftover of an interrupted repair, or an existing blob, refuses.
REPAIR_WINDOW = 64 << 20           # ranged_get's default range size: the unit the mirror damaged
ZERO_BLOCK = 1 << 20
REPAIR_MAX_FRACTION = 0.25         # beyond this the whole-file fetch is no slower


def zero_windows(path: Path, size: int, window: int = REPAIR_WINDOW, block: int = ZERO_BLOCK) -> list[tuple[int, int]]:
    """The windows [start, end] (inclusive; start a multiple of `window`) of `path` holding at least one
    block-aligned run of `block` zero bytes."""
    zero = bytes(block)
    out: list[tuple[int, int]] = []
    with open(path, "rb") as fh:
        for start in range(0, size, window):
            end = min(start + window, size) - 1
            fh.seek(start)
            off = start
            while off <= end:
                b = fh.read(min(block, end + 1 - off))
                if not b:
                    break
                if len(b) == block and b == zero:
                    out.append((start, end))
                    break
                off += len(b)
    return out


def fetch_range(url: str, start: int, end: int, size: int, token: str | None = None) -> bytes:
    """Bytes [start, end] of a `size`-byte file, by Range requests validated as ranged_get validates them (every answer
    must carry a Content-Range that starts where it was asked, ends no later, and names the size), the token only to
    huggingface.co and never across a redirect; transient failures are retried with backoff, keeping what arrived."""
    out = bytearray()
    want = end + 1 - start
    for attempt in range(RANGE_TRIES):
        have = start + len(out)
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-{end}"})
        for k, v in auth_headers(url, token).items():
            req.add_unredirected_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                cr = r.headers.get("Content-Range")
                m = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", (cr or "").strip(), flags=re.IGNORECASE)
                if not m or int(m.group(1)) != have or int(m.group(2)) > end or m.group(3) not in ("*", str(size)):
                    raise RangeProtocolError(f"Content-Range {cr!r} does not answer bytes={have}-{end} of {size}")
                while len(out) < want:
                    buf = r.read(min(1 << 20, want - len(out)))
                    if not buf:
                        break
                    out += buf
            if len(out) == want:
                return bytes(out)
            raise OSError("short body")
        except Exception as exc:  # noqa: BLE001
            if len(out) == want:
                return bytes(out)              # the range arrived complete, whatever the close raised
            code = getattr(exc, "code", None)
            if isinstance(exc, RangeProtocolError) or (isinstance(exc, urllib.error.HTTPError)
                                                       and code not in RETRY_HTTP):
                raise
            if attempt == RANGE_TRIES - 1:
                raise
            wait = min(RANGE_BACKOFF_MAX_S, RANGE_BACKOFF_S * 2 ** attempt)
            print(f"      repair range {start}-{end} attempt {attempt + 1} failed ({type(exc).__name__}"
                  f"{f' {code}' if code else ''}); continuing from {start + len(out)} in {wait} s", flush=True)
            time.sleep(wait)
    raise OSError(f"no complete range {start}-{end} from {url}")


def repair_leftover(blob: Path) -> None:
    """Refuse when an interrupted F-073 repair left <blob>.repair.part (the process was killed between the copy and the
    rename): it was never linked, and it is neither reused nor overwritten. Checked before any download of the file."""
    rep = blob.with_name(f"{blob.name}.repair.part")
    if os.path.lexists(rep):
        raise SystemExit(f"REFUSING: {rep} ({rep.lstat().st_size} bytes) was left by an interrupted F-073 repair and was "
                         "never linked; remove it (scripts/safe_rm.sh with its size and sha256, or src/fetch_weights.py "
                         "--purge for the whole model) and run again")


def repair_from_hub(bad: Path, blob: Path, url: str, size: int, want: str, token: str | None = None,
                    window: int = REPAIR_WINDOW, block: int = ZERO_BLOCK) -> bool:
    """F-073: a NEW copy of the quarantined `bad` file with every zero window replaced by huggingface.co's bytes
    (`url`); it becomes `blob` only if its sha256 is `want`. False (nothing linked; the caller then fetches the whole
    file, D-281) when there is no zero window, the windows exceed REPAIR_MAX_FRACTION of the file, a window cannot be
    fetched (network, HTTP or Range-protocol failure: the partial copy is renamed aside as <blob>.repairerr-<pid>), or
    the repaired copy's sha256 is still wrong (renamed aside as <blob>.repairbad-<sha16>, which replaces an earlier copy
    of the same bad bytes, as quarantine() does). Nothing is ever linked unhashed, and nothing is deleted; the purge
    removes the set-aside copies with the model. An existing blob or .repair.part refuses (repair_leftover)."""
    spans = zero_windows(bad, size, window, block)
    n = sum(e + 1 - s for s, e in spans)
    if not spans or n > REPAIR_MAX_FRACTION * size:
        print(f"    repair: {len(spans)} zero window(s), {n / 1e6:.0f} MB: not repaired by range")
        return False
    rep = blob.with_name(f"{blob.name}.repair.part")
    repair_leftover(blob)
    if os.path.lexists(blob):
        raise SystemExit(f"REFUSING: {blob} exists; the repair never overwrites a blob")
    src = f"{urllib.parse.urlsplit(url).scheme}://{urllib.parse.urlsplit(url).hostname}"
    print(f"    repair: {len(spans)} zero window(s), {n / 1e6:.0f} MB, from {src} "
          f"(token sent: {'YES' if auth_headers(url, token) else 'no'})", flush=True)
    shutil.copyfile(bad, rep)
    try:
        with open(rep, "r+b") as fh:
            for s, e in spans:
                data = fetch_range(url, s, e, size, token)
                fh.seek(s)
                fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    except (OSError, http.client.HTTPException) as exc:        # URLError, HTTPError and RangeProtocolError included
        aside = rep.with_name(f"{blob.name}.repairerr-{os.getpid()}")
        rep.replace(aside)
        print(f"    repair: a window could not be fetched ({type(exc).__name__}"
              f"{f' {exc.code}' if getattr(exc, 'code', None) else ''}); the partial copy set aside as {aside.name}")
        return False
    if rep.stat().st_size != size:
        raise OSError(f"repaired copy is {rep.stat().st_size} bytes, expected {size}")
    got = sha256_file(rep)
    if got != want:
        q = rep.with_name(f"{blob.name}.repairbad-{got[:16]}")
        rep.replace(q)
        print(f"    repair: sha256 {got[:16]} != {want[:16]}; set aside as {q.name}")
        return False
    os.replace(rep, blob)
    print(f"    repair: sha256 OK {got[:16]} ({len(spans)} window(s) from {src})")
    return True


def ranged_snapshot(repo: str, revision: str, meta: dict, endpoint: str,
                    hub: Path, wanted: list[str], token: str | None = None,
                    pointers: dict[str, tuple[str, int]] | None = None) -> Path:
    """Assemble a snapshot by ranged download, in the layout huggingface_hub expects.

    Files are stored once under blobs/<etag> and the snapshot directory holds symlinks
    to them, which is exactly what snapshot_download produces -- so transformers'
    from_pretrained(repo, revision=...) finds it with HF_HUB_OFFLINE=1 and the pinned
    revision still appears in provenance, rather than a local path standing in for it.
    """
    pointers = {} if pointers is None else pointers      # name -> (sha256, size) of F-068 pointer entries
    slug = "models--" + repo.replace("/", "--")
    blobs = hub / slug / "blobs"
    snap = hub / slug / "snapshots" / revision
    blobs.mkdir(parents=True, exist_ok=True)
    snap.mkdir(parents=True, exist_ok=True)

    # smallest first: every small file and every F-068 pointer is fetched and verified before the shards,
    # so a failure on one of them costs seconds, not the shard download that a purge then throws away
    for name in sorted(wanted, key=lambda n: int(meta[n].get("size") or (meta[n].get("lfs") or {}).get("size") or 0)):
        info = meta[name]
        lfs = info.get("lfs") or {}
        # The API names an LFS file's sha256 "sha256" (model info) or "oid" (tree listing); read it one
        # way everywhere, so routing, the blob name and the check can never disagree (review 2026-09-26).
        lfs_sha = lfs.get("sha256") or lfs.get("oid")
        if lfs and not lfs_sha:
            raise SystemExit(f"REFUSING: {name} is an LFS file with no published sha256")
        blob_sha1 = None if lfs else (info.get("blobId") or info.get("oid"))
        if not lfs and not blob_sha1:
            raise SystemExit(f"REFUSING: {name} has no published git blob id")
        size = info.get("size") or lfs.get("size")
        pointer = None
        if not lfs and size is not None and int(size) <= POINTER_MAX:
            # F-068: a pointer committed as a plain blob. The blob is verified against its git id first,
            # then its object against the sha256 the pointer names; the object comes from huggingface.co.
            pointer = lfs_pointer(committed_blob(repo, revision, name, blob_sha1, int(size), token))
            if pointer:
                print(f"  {name}: an LFS pointer committed as a plain blob (blob-sha1 {blob_sha1[:16]} OK); "
                      f"its object: sha256 {pointer[0][:16]}, {pointer[1]} bytes (F-068)")
                lfs_sha, size, blob_sha1 = pointer[0], pointer[1], None
                pointers[name] = pointer
        etag = lfs_sha or blob_sha1
        blob = blobs / etag
        link = snap / name
        link.parent.mkdir(parents=True, exist_ok=True)

        have = blob.exists() and size and blob.stat().st_size == size
        src = None                                           # where this run fetched the file from (None: had it)
        if have and blob_sha1:
            # a small file is cheap to hash, and a same-size stale version (the mirror served other
            # versions of three small files, F-066 third addendum) must not be reused on its size alone
            got = git_blob_sha1(blob)
            if got != blob_sha1:
                bad = quarantine(blob, got)
                print(f"  stale {name}: blob-sha1 {got[:16]} != {blob_sha1[:16]}; set aside as "
                      f"{bad.name}, fetching again")
                have = False
        if have:
            print(f"  have {name}")
        else:
            # Small non-LFS files (configs, tokenizer files) always come from huggingface.co, the hash
            # authority: the campus mirror served other versions of three of them for the pinned
            # revision of Mistral-7B-Instruct-v0.1 (blob sha1 mismatches, 2026-09-26, F-066 third
            # addendum). They are kilobytes; the LFS shards keep the fast mirror and their sha256. A
            # non-gated repo has no token loaded, so nothing is sent; a gated repo's endpoint is
            # huggingface.co already.
            src = endpoint if (lfs_sha and not pointer) else f"https://{CANONICAL_HOST}"
            url = f"{src.rstrip('/')}/{repo}/resolve/{revision}/{name}"
            authed = bool(auth_headers(url, token))
            if lfs_sha:
                repair_leftover(blob)                        # F-073: before the download, not after it
            print(f"  fetching {name}  ({(size or 0) / 1e9:.2f} GB) by range")
            print(f"    endpoint {urllib.parse.urlsplit(url).scheme}://"
                  f"{urllib.parse.urlsplit(url).hostname}  token sent: "
                  f"{'YES' if authed else 'no'}")
            ranged_get(url, int(size), blob, token=token)

        # Verify before linking: a blob that fails its hash must never become visible
        # under the snapshot name, or the next run will treat it as already present.
        if lfs_sha:
            got = sha256_file(blob)
            if got != lfs_sha and src is not None and urllib.parse.urlsplit(src).hostname != CANONICAL_HOST:
                # D-281: a file the mirror served with the wrong sha256 is fetched ONCE more, from huggingface.co
                # (the mirror is a caching proxy and has served wrong bytes before, F-051, F-066). The bad copy is
                # quarantined, never linked; a second mismatch stops below.
                bad = quarantine(blob, got)
                url = f"https://{CANONICAL_HOST}/{repo}/resolve/{revision}/{name}"
                print(f"  mirror served {name} with sha256 {got[:16]} != {lfs_sha[:16]}; set aside as {bad.name}")
                # F-073: first only the zero-filled windows from huggingface.co; else the whole file (D-281)
                if not repair_from_hub(bad, blob, url, int(size), lfs_sha, token=token):
                    print(f"  fetching {name} again from https://{CANONICAL_HOST} (D-281)  token sent: "
                          f"{'YES' if auth_headers(url, token) else 'no'}")
                    ranged_get(url, int(size), blob, token=token)
                got = sha256_file(blob)
            if got != lfs_sha:
                bad = quarantine(blob, got)
                raise SystemExit(f"HASH MISMATCH for {name}: {got[:16]} != "
                                 f"{lfs_sha[:16]} (blob quarantined as {bad.name}, never linked)")
            print(f"    sha256 OK {got[:16]}")
        else:
            got = git_blob_sha1(blob)
            if got != blob_sha1:
                bad = quarantine(blob, got)
                raise SystemExit(f"HASH MISMATCH for {name}: blob-sha1 {got[:16]} != "
                                 f"{blob_sha1[:16]} (blob quarantined as {bad.name}, never linked)")
            print(f"    blob-sha1 OK {got[:16]}")
        if link.is_symlink() or link.exists():
            link.unlink()
        try:
            link.symlink_to(os.path.relpath(blob, link.parent))
        except OSError:
            shutil.copy2(blob, link)   # filesystems without symlinks
    return snap


def _listing(root: Path, when: str) -> None:
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink())
    links = [p for p in root.rglob("*") if p.is_symlink()]
    total = sum(p.stat().st_size for p in files)
    print(f"  {when}: {len(files)} file(s), {total / 1e9:.3f} GB, {len(links)} link(s) under {root}")
    for p in files:
        print(f"    {p.stat().st_size:>14}  {p.relative_to(root)}")


def purge(repo: str, hf_home: Path) -> int:
    """Delete ONE repo's weights from the NVMe Hugging Face cache, and nothing else (D-153).

    Weights are deleted after extraction (the standing storage rule), and this is the one
    sanctioned exception to "files over 1 MB go through safe_rm.sh": it is confined to
    <HF_HOME>/hub/models--<org>--<name> on an NVMe mount. Every path is checked BEFORE any
    deletion; a single file or link that resolves outside that directory, or an HF_HOME
    that is not on NVMe, refuses the whole purge. The listing is printed before and after.
    """
    from hf_env_guard import NVME_SOURCE_PREFIX, mount_of

    hub = (hf_home / "hub").resolve()
    mnt, src = mount_of(hub)
    if not src.startswith(NVME_SOURCE_PREFIX):
        raise SystemExit(f"REFUSING purge: {hub} is on {src} (mounted at {mnt}), not NVMe")
    root = hub / ("models--" + repo.replace("/", "--"))
    if not root.exists():
        print(f"nothing cached for {repo} under {hub}")
        return 0
    if root.is_symlink() or root.resolve().parent != hub:
        raise SystemExit(f"REFUSING purge: {root} does not resolve to a directory directly under {hub}")
    root = root.resolve()
    entries = sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True)
    for p in entries:  # pre-check everything before touching anything
        target = Path(os.path.realpath(p))
        if root not in target.parents and target != root:
            raise SystemExit(f"REFUSING purge: {p} resolves to {target}, outside {root}")
    _listing(root, "BEFORE")
    for p in entries:  # deepest first: files and links, then their directories
        if p.is_symlink() or p.is_file():
            p.unlink()
        elif p.is_dir():
            p.rmdir()
    root.rmdir()
    print(f"  AFTER : {root} {'still exists' if root.exists() else 'removed'}")
    if root.exists():
        _listing(root, "AFTER")
        return 1
    return 0


def hub_files(repo: str, revision: str) -> dict[str, dict]:
    url = f"{API}/models/{repo}/revision/{revision}?blobs=true"
    req = urllib.request.Request(url, headers={"User-Agent": "guardrail-ccf-fetch"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = json.loads(r.read().decode("utf-8"))
    got = body.get("sha", "")
    if got != revision:
        raise SystemExit(f"REVISION MISMATCH: asked {revision}, Hub resolved {got}")
    return {f["rfilename"]: f for f in body.get("siblings", [])}


def main() -> int:
    # D-318 (7) item 8, the one .bin exception (BIN_ALLOWED); imported here, so the modules that import this file for
    # its helpers (fetch_dataset_snapshot, phase2_config_check) do not load it
    import bin_weights as BW

    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Fetch safetensors weights to NVMe, verified")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--hf-home", default=None, help="defaults to $HF_HOME or ~/.cache/hf")
    ap.add_argument("--mirror", default=None,
                    help="optional download source for UNGATED repos (the campus mirror, "
                         "GCCF_HF_MIRROR in the private site config, D-172); never sent a "
                         "credential, and hashes always come from huggingface.co")
    ap.add_argument("--purge", action="store_true",
                    help="delete this repo's weights from the NVMe HF cache only "
                         "(refuses any other location); lists before and after")
    args = ap.parse_args()

    import os
    hf_home = Path(args.hf_home or os.environ.get("HF_HOME") or (Path.home() / ".cache/hf"))
    os.environ["HF_HOME"] = str(hf_home)
    assert_hf_env("fetch_weights.py")

    # NOTE: huggingface_hub reads HF_ENDPOINT into module constants AT IMPORT, so the
    # mirror pre-flight below must run before the import for its fallback to take
    # effect. The import therefore happens after that check, not here.
    if args.purge:
        return purge(args.repo, hf_home)

    print(f"repo      : {args.repo}")
    print(f"revision  : {args.revision}")
    print(f"HF_HOME   : {hf_home}   (must be on local NVMe, not the NAS)")

    meta = hub_files(args.repo, args.revision)
    print(f"revision resolves and matches exactly; {len(meta)} files in the repo")
    # D-318 (7) item 8: .bin shards only for a BIN_ALLOWED repo at its one revision; everything else safetensors only,
    # and a revision with .bin weights only refuses here unless allowlisted
    allow, deny = BW.patterns(args.repo, args.revision, ALLOW, DENY)
    keep = BW.keep_files(list(meta), args.repo, args.revision, ALLOW, DENY)
    if BW.bin_allowed(args.repo, args.revision):
        snap0 = BW.snapshot_dir(hf_home / "hub", args.repo, args.revision)
        if BW.conversion_output(snap0):
            # a re-run after a failed sitting (scripts/p2_sitting.sh keeps the NVMe weights): the existing conversion is
            # accepted, nothing fetched, only if it is verifiably this revision's (bin_weights.verify_converted); any
            # other leftover refuses, and the purge comes first
            BW.verify_converted(snap0, args.repo, args.revision, meta, keep)
            print(f"snapshot  : {snap0}")
            return 0
        print(f"bin       : {args.repo} @ {args.revision[:12]} is in BIN_ALLOWED: its .bin shards are fetched, "
              "verified against huggingface.co, then converted to safetensors (src/bin_weights.py)")

    # The DOWNLOAD may come from a fast local mirror (--mirror); the EXPECTED
    # HASHES always come from huggingface.co, above. That split is deliberate: it lets
    # an untrusted or unverified mirror supply the bytes while the authority on what
    # those bytes should be stays the canonical Hub. A mirror that served the wrong
    # content would fail the check below. The mirror is an argument, not HF_ENDPOINT,
    # so nothing else in the process (huggingface_hub, transformers) is routed to it.
    endpoint = args.mirror

    # Gated repos: the mirror returns 404 for them (it carries no gated content) and
    # the canonical Hub returns 401 without a token. Such a repo MUST be fetched from
    # https://huggingface.co with the token, so any mirror setting is overridden here
    # rather than left to fail downstream. The token is loaded only in this branch.
    token = None
    gated = False
    probe = f"https://{CANONICAL_HOST}/{args.repo}/resolve/{args.revision}/config.json"
    try:
        urllib.request.urlopen(urllib.request.Request(probe, method="HEAD"), timeout=30)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            gated = True
    except Exception:  # noqa: BLE001 - inconclusive, treat as not gated
        pass

    if gated:
        token = load_token()
        if not token:
            raise SystemExit(f"{args.repo} is gated and no token is configured; "
                             "cannot fetch")
        if endpoint:
            print(f"  GATED repo: ignoring --mirror ({endpoint}) -- a credential "
                  f"must never go to a mirror")
        endpoint = f"https://{CANONICAL_HOST}"
        os.environ["HF_ENDPOINT"] = endpoint
        print(f"  gated: yes. token loaded (length {len(token)}, value never printed)")

    print(f"LFS shards via    : {endpoint or 'https://huggingface.co (default)'}")
    print(f"small files via   : https://{CANONICAL_HOST}  (non-LFS files, F-066 third addendum)")
    print(f"hashes from       : https://huggingface.co  (authority, regardless of source)")
    if token:
        print(f"token will be sent ONLY to https://{CANONICAL_HOST}; any other host gets "
              f"no Authorization header")
    else:
        print("token             : none loaded (gated probe: not gated, or inconclusive); no Authorization header is sent")

    # F-051: the campus mirror is a caching proxy. While it was still pulling a shard
    # from upstream it advertised the correct Content-Length but delivered a truncated
    # body, and did so inconsistently across retries. Catching that after a multi-GB
    # download wastes an hour per attempt, so check every file's length up front and
    # fall back to the canonical Hub rather than retrying against a bad source.
    # Only a mirror needs vetting. When the endpoint IS huggingface.co there is
    # nothing to cross-check it against -- it is the authority the sizes came from --
    # and an unauthenticated HEAD against a gated repo just 401s eleven times.
    if endpoint and urllib.parse.urlsplit(endpoint).hostname != CANONICAL_HOST:
        # LFS shards only: the small files never come from the mirror (F-066 third addendum)
        wanted = {n: (i.get("size") or (i.get("lfs") or {}).get("size"))
                  for n, i in meta.items()
                  if i.get("lfs") and n in keep}                     # only files that will be downloaded
        # Only a CONFIRMED short length rejects the mirror. A HEAD that times out is
        # not evidence of truncation, and treating it as such once threw away the fast
        # source over a 100-byte generation_config.json and fell back to an 8-hour
        # download. The post-download hash check is the actual safety net; this
        # pre-flight exists only to avoid wasting an hour on a known-bad source, so it
        # should be lenient about not knowing and strict only about knowing.
        short = []
        unknown = 0
        for name, size in wanted.items():
            if not size:
                continue
            url = f"{endpoint.rstrip('/')}/{args.repo}/resolve/{args.revision}/{name}"
            got = None
            for attempt in range(3):
                try:
                    req = urllib.request.Request(url, method="HEAD")
                    with urllib.request.urlopen(req, timeout=45) as r:
                        got = int(r.headers.get("Content-Length") or 0)
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt == 2:
                        print(f"  mirror HEAD inconclusive for {name} after 3 tries "
                              f"({type(exc).__name__}); not treating as short")
            if got is None:
                unknown += 1
                continue
            if got != size:
                print(f"  mirror SHORT on {name}: {got} vs {size} expected")
                short.append(name)
        checked = len(wanted) - unknown
        if short:
            print(f"  mirror is incomplete for {len(short)} file(s); "
                  f"falling back to https://huggingface.co")
            os.environ.pop("HF_ENDPOINT", None)
            endpoint = None
        elif checked == 0:
            # Do not report a clean bill of health from zero measurements. The first
            # version of this printed "advertises the correct size for all 11 wanted
            # files" for a gated repo where every single HEAD had failed, which reads
            # as confirmation when nothing was confirmed.
            print(f"  mirror size CHECK INCONCLUSIVE: 0 of {len(wanted)} files could be "
                  f"queried; proceeding, and the sha256 check after download is the "
                  f"only guarantee")
        else:
            print(f"  mirror advertises the correct size for {checked} of "
                  f"{len(wanted)} LFS shards ({unknown} inconclusive)")

    pointers: dict[str, tuple[str, int]] = {}             # F-068

    if endpoint:
        # Ranged assembly from the mirror. Chosen because the mirror truncates
        # whole-file GETs while serving ranges correctly at ~23.7 MB/s, and the direct
        # Hub could not move 300 MB in five minutes from this node.
        path = ranged_snapshot(args.repo, args.revision, meta, endpoint,
                               hf_home / "hub", keep, token=token, pointers=pointers)
    else:
        from huggingface_hub import snapshot_download

        path = None
        for attempt in range(1, 6):
            try:
                path = Path(snapshot_download(
                    repo_id=args.repo, revision=args.revision,
                    allow_patterns=allow, ignore_patterns=deny,
                    cache_dir=str(hf_home / "hub"), max_workers=4,
                    # False, not None: None lets huggingface_hub pick up a token file on its own
                    token=token or False,
                ))
                break
            except Exception as exc:  # noqa: BLE001 - network flakiness expected
                print(f"  attempt {attempt}/5 failed: {type(exc).__name__}: "
                      f"{str(exc)[:120]}")
                if attempt == 5:
                    raise
                print("  retrying; snapshot_download resumes from the partial files")
    print(f"snapshot  : {path}")

    files = sorted(p for p in path.rglob("*") if p.is_file())
    banned = BW.banned([str(p.relative_to(path)).replace("\\", "/") for p in files], args.repo, args.revision)
    if banned:
        raise SystemExit(f"REFUSING: non-safetensors weight files present: {banned}")

    ok = bad = unverified = 0
    total = 0
    for p in files:
        rel = str(p.relative_to(path)).replace("\\", "/")
        total += p.stat().st_size
        info = meta.get(rel)
        if not info:
            print(f"  ?  {rel}  (not listed in the revision)")
            unverified += 1
            continue
        lfs = info.get("lfs") or {}
        want = lfs.get("sha256") or lfs.get("oid") or (pointers[rel][0] if rel in pointers else None)
        if want:
            got = sha256_file(p)
            kind = "sha256" if rel not in pointers else "sha256 (object of a committed LFS pointer, F-068)"
        else:
            want, got, kind = info.get("blobId") or info.get("oid"), git_blob_sha1(p), "blob-sha1"
            size = info.get("size")
            if want and got != want and size is not None and int(size) <= POINTER_MAX:
                # snapshot_download stores a pointer entry's OBJECT; judge it by the sha256 its verified
                # committed pointer names, never by the pointer's own blob id (F-068)
                ptr = lfs_pointer(committed_blob(args.repo, args.revision, rel, want, int(size), token))
                if ptr:
                    want, got, kind = ptr[0], sha256_file(p), "sha256 (object of a committed LFS pointer, F-068)"
        if not want:
            print(f"  ?  {rel}  (no hash published)")
            unverified += 1
        elif got == want:
            print(f"  OK {rel}  {kind} {got[:16]}")
            ok += 1
        else:
            print(f"  BAD {rel}  {kind} got {got[:16]} want {str(want)[:16]}")
            bad += 1

    print(f"\n  verified {ok}, mismatched {bad}, unverified {unverified}, "
          f"of {len(files)} files, {total / 1e9:.2f} GB")
    print(f"  snapshot path: {path}")
    if bad:
        raise SystemExit("HASH MISMATCH -- refusing to proceed")
    if unverified:
        # a file with no published hash, or one the revision does not list, is not verified, so the
        # snapshot is not either (review of 898c705)
        raise SystemExit(f"UNVERIFIED: {unverified} file(s) could not be checked -- refusing to proceed")
    if BW.bin_allowed(args.repo, args.revision):
        # D-318 (7) item 8: only now, every file verified against huggingface.co, are the .bin shards loaded
        # (torch.load(weights_only=True) of bytes hashed again as they are read) and converted to safetensors
        BW.convert_snapshot(path, args.repo, args.revision, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
