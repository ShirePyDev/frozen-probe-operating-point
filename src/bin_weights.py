"""The one .bin exception (D-318 (7) item 8): allenai's OLMo-2 SFT and DPO checkpoints ship weights only as PyTorch
.bin pickles at their pinned revisions. Every other repo, and these two at any other revision, stay safetensors only.

For exactly the repo@revision pairs in BIN_ALLOWED:
  * src/fetch_weights.py fetches pytorch_model*.bin and pytorch_model.bin.index.json with the other files, through its
    own path (mirror first, F-073 window repair, D-281 fall-back) and verifies each against huggingface.co like any
    file: a shard's LFS sha256 at the pinned revision, the index's git blob id;
  * convert_snapshot(), called by fetch_weights.py only after that verification passed, converts the NVMe snapshot to
    safetensors: each shard's bytes are read ONCE, their sha256 must equal huggingface.co's, and those same bytes are
    loaded with torch.load(..., map_location='cpu', weights_only=True), so the bytes loaded are the bytes hashed;
    safetensors.torch.save_file writes the tensors with every dtype kept; the written file is loaded back and every
    tensor compared with the torch.load result bit for bit (names, dtype, shape and raw bytes); then
    model.safetensors.index.json is written, the .bin names are removed from the snapshot (names only, each a link
    into blobs/: the bytes stay until the per-model purge, D-153 (d)), and CONVERSION_RECORD records the .bin and the
    safetensors sha256s, the comparison and the torch and safetensors versions;
  * src/extract_p2.py loads the converted safetensors through its one loading path (it refuses any .bin in a
    snapshot), hashes every converted file against the conversion record (check_converted) and embeds the record and
    its sha256 in the extraction record.
A RE-RUN (a failed sitting keeps its NVMe weights): fetch_weights.py accepts a snapshot that is already converted, with
nothing downloaded, only through verify_converted (the conversion record's .bin hashes equal huggingface.co's listing,
every converted file rehashed, every other file the Hub's); any other leftover (output without its record, a
temporary, a blob the listing does not name) refuses and the purge comes first.
Not used: the unmerged SFconvertbot safetensors pull requests on the Hub (D-312 (8)); the author chose allenai's files.
Nothing is deleted on a failure: partial files are left in blobs/ and named, and the purge removes them with the model.
"""

from __future__ import annotations

import collections
import hashlib
import io
import json
import os
import platform
import re
import time
from pathlib import Path
from typing import Any

# repo -> the ONE revision whose .bin files may be fetched and converted: the config check's resolved commit
# (results/phase2_config_check.json models[repo]['sha']; tests/test_bin_weights.py checks that they agree)
BIN_ALLOWED = {
    "allenai/OLMo-2-1124-7B-SFT": "1de02c0175118a9de5854aec80a1f970e701e928",
    "allenai/OLMo-2-1124-7B-DPO": "e34ea60adff2e575f4fe7569eaffd1b28509b6fd",
}
BIN_INDEX = "pytorch_model.bin.index.json"
BIN_PATTERNS = ["pytorch_model*.bin", BIN_INDEX]
ST_INDEX = "model.safetensors.index.json"
CONVERSION_RECORD = "bin_conversion.json"
SHARD_RE = re.compile(r"pytorch_model-(\d{5})-of-(\d{5})\.bin")
PICKLE_SUFFIXES = (".bin", ".pth", ".h5", ".msgpack")


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def bin_allowed(repo: str, revision: str) -> bool:
    return isinstance(revision, str) and BIN_ALLOWED.get(repo) == revision


def patterns(repo: str, revision: str, allow: list[str], deny: list[str]) -> tuple[list[str], list[str]]:
    """fetch_weights' (ALLOW, DENY) for this repo@revision: for an allowlisted pair only, the .bin shards and their
    index are added and '*.bin' is lifted from DENY (another .bin, e.g. training_args.bin, matches no ALLOW pattern)."""
    if not bin_allowed(repo, revision):
        return list(allow), list(deny)
    return list(allow) + BIN_PATTERNS, [p for p in deny if p != "*.bin"]


def is_allowed_bin(name: str, repo: str, revision: str) -> bool:
    return bin_allowed(repo, revision) and "/" not in name and Path(name).match("pytorch_model*.bin")


def keep_files(names: list[str], repo: str, revision: str, allow: list[str], deny: list[str]) -> list[str]:
    """The files fetch_weights downloads. Refuses a revision whose weights are .bin only unless the pair is allowlisted,
    so such a repo, or an allowlisted repo at another revision, stops here instead of fetching configs alone."""
    a, d = patterns(repo, revision, allow, deny)
    keep = [n for n in names if any(Path(n).match(p) for p in a) and not any(Path(n).match(p) for p in d)]
    if not any(n.endswith(".safetensors") or is_allowed_bin(n, repo, revision) for n in keep):
        bins = sorted(n for n in names if Path(n).suffix in PICKLE_SUFFIXES)
        if bins:
            pinned = f"; BIN_ALLOWED names {repo} at {BIN_ALLOWED[repo]} only" if repo in BIN_ALLOWED else ""
            refuse(f"{repo} @ {revision[:12]} has no safetensors weights, only {bins}; a .bin is fetched only for a "
                   f"repo at the exact revision src/bin_weights.py BIN_ALLOWED names (D-318 (7) item 8){pinned}")
    return keep


def banned(rels: list[str], repo: str, revision: str) -> list[str]:
    """Pickle-format weight files a snapshot must not hold: all of them, except an allowlisted pair's .bin shards."""
    return [r for r in rels if Path(r).suffix in PICKLE_SUFFIXES and not is_allowed_bin(r, repo, revision)]


def snapshot_dir(hub: Path, repo: str, revision: str) -> Path:
    return Path(hub) / ("models--" + repo.replace("/", "--")) / "snapshots" / revision


def conversion_output(snap: Path) -> list[str]:
    """The conversion's files present in a snapshot (safetensors shards, their index, the conversion record)."""
    snap = Path(snap)
    if not snap.is_dir():
        return []
    return sorted(p.name for p in snap.iterdir()
                  if p.suffix == ".safetensors" or p.name in (ST_INDEX, CONVERSION_RECORD))


def refuse_if_converted(snap: Path) -> None:
    """At the start of a conversion: a snapshot that already holds conversion output (an earlier run) refuses, so no
    conversion ever mixes with another or overwrites a file. The purge removes the whole model."""
    found = conversion_output(snap)
    if found:
        refuse(f"{snap} already holds conversion output {found[:6]} (an earlier run); purge the model "
               "(src/fetch_weights.py --purge) and fetch again")


# --- conversion ---------------------------------------------------------------------------------------------------

def _raw(t: Any) -> Any:
    """A tensor's bytes as a 1-D uint8 tensor (bit for bit: -0.0 and 0.0, or two NaN payloads, differ)."""
    import torch

    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def differing(a: dict, b: dict) -> list[str]:
    """Names whose tensors differ in presence, dtype, shape or raw bytes; empty when a and b are equal bit for bit."""
    import torch

    bad = sorted(set(a) ^ set(b))
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if x.dtype != y.dtype or tuple(x.shape) != tuple(y.shape) or not torch.equal(_raw(x), _raw(y)):
            bad.append(k)
    return bad


def _for_save(sd: dict) -> tuple[dict, int]:
    """Tensors safetensors can write: contiguous, and none sharing storage with another (such a tensor is cloned, which
    is bit-identical; the comparison after the reload is against the torch.load result itself, not this copy)."""
    import torch

    ptrs = collections.Counter(t.untyped_storage().data_ptr() for t in sd.values() if t.numel())
    out, n = {}, 0
    for k, t in sd.items():
        if not t.is_contiguous() or (t.numel() and ptrs[t.untyped_storage().data_ptr()] > 1):
            t, n = t.clone(memory_format=torch.contiguous_format), n + 1
        out[k] = t
    return out, n


def convert_shard(source: bytes | Path, want_sha256: str, out_path: Path, expected_keys: set[str] | None = None,
                  label: str = "shard") -> tuple[dict[str, Any], list[str]]:
    """One .bin shard (its bytes, or a path whose bytes are read ONCE here) -> a safetensors file at out_path (which
    must not exist). The bytes' sha256 must equal want_sha256 (huggingface.co's) BEFORE torch.load sees them, and
    torch.load reads those same bytes; the file written is loaded back and must equal the torch.load result bit for
    bit, or it refuses (the file is left in place, never linked). Returns (entry, names)."""
    import torch
    from safetensors.torch import load_file, save_file

    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else source
    got, n_bytes = hashlib.sha256(data).hexdigest(), len(data)
    if got != want_sha256:
        refuse(f"{label}: sha256 {got[:16]} is not huggingface.co's {str(want_sha256)[:16]}; never loaded")
    sd = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    del data                                    # read from a path, the bytes are freed before the save (memory)
    if not isinstance(sd, dict) or not sd or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                                      for k, v in sd.items()):
        refuse(f"{label}: torch.load(weights_only=True) did not give a non-empty dict of named tensors")
    if expected_keys is not None and set(sd) != set(expected_keys):
        refuse(f"{label}: tensor names differ from the index's for this shard (missing "
               f"{sorted(set(expected_keys) - set(sd))[:5]}, extra {sorted(set(sd) - set(expected_keys))[:5]})")
    out_path = Path(out_path)
    if os.path.lexists(out_path):
        refuse(f"{out_path} exists; a conversion never overwrites a file")
    prepared, n_copied = _for_save(sd)
    print(f"    writing {out_path}", flush=True)       # named before it exists: a failure below leaves it, named
    save_file(prepared, str(out_path), metadata={"format": "pt"})
    del prepared
    back = load_file(str(out_path), device="cpu")
    bad = differing(sd, back)
    if bad:
        refuse(f"{label}: the converted file {out_path} differs from the torch.load result in {len(bad)} tensor(s), "
               f"e.g. {bad[:5]}; it is left in place and never linked")
    entry = {"bin_bytes": n_bytes, "bin_sha256": got, "bin_sha256_hub": want_sha256, "bin_sha256_equal": True,
             "n_tensors": len(sd),
             "dtypes": dict(sorted(collections.Counter(str(v.dtype) for v in sd.values()).items())),
             "tensor_bytes": int(sum(v.nbytes for v in sd.values())), "n_cloned_before_save": n_copied,
             "safetensors_bytes": out_path.stat().st_size, "safetensors_sha256": sha256_file(out_path),
             "n_tensors_compared": len(sd), "bit_for_bit_equal": True}
    return entry, sorted(sd)


def _removable(name: Path, blob: Path) -> None:
    """Removing `name` from the snapshot must delete no bytes: it is a symlink to `blob`, or a hard link of it."""
    if name.is_symlink():
        if os.path.realpath(name) != os.path.realpath(blob):
            refuse(f"{name} links to {os.path.realpath(name)}, not its blob {blob}")
    elif not (name.is_file() and blob.is_file() and os.path.samefile(name, blob)):
        refuse(f"{name} is neither a link to nor a hard link of its blob {blob}; removing it would delete bytes, which "
               "only scripts/safe_rm.sh or the purge may do")


def _link(name: Path, blob: Path) -> None:
    if os.path.lexists(name):
        refuse(f"{name} exists; a conversion never replaces a snapshot entry")
    try:
        name.symlink_to(os.path.relpath(blob, name.parent))
    except FileExistsError:
        raise                                   # appeared meanwhile: never replaced, never retried as a hard link
    except OSError:
        os.link(blob, name)                     # filesystems without symlinks: a hard link, never a copy


QUARANTINE = re.compile(r"([0-9a-f]{40}|[0-9a-f]{64})\.(bad-[0-9a-f]{16}|repairbad-[0-9a-f]{16}|repairerr-[0-9]+)")


def foreign_blobs(blobs: Path, meta: dict[str, dict]) -> list[str]:
    """Entries of blobs/ that are neither a blob huggingface.co's listing names (an LFS sha256 or a git blob id) nor a
    copy fetch_weights quarantined from one (<etag>.bad-<sha16>, .repairbad-<sha16>, .repairerr-<pid>; F-072, F-073):
    a conversion temporary (*.converting.*), a converted blob or anything else an earlier run left."""
    hub: set[str] = set()
    for info in meta.values():
        lfs = info.get("lfs") or {}
        hub |= {str(e) for e in (lfs.get("sha256"), lfs.get("oid"), info.get("blobId"), info.get("oid")) if e}
    out = []
    for p in sorted(Path(blobs).iterdir()):
        m = QUARANTINE.fullmatch(p.name)
        if p.name not in hub and not (m and m.group(1) in hub):
            out.append(p.name)
    return out


def _exclusive_blob(blobs: Path, data: bytes) -> Path:
    blob = blobs / hashlib.sha256(data).hexdigest()
    with open(blob, "xb") as fh:                # FileExistsError if it exists: never overwritten
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    return blob


def convert_snapshot(snap: Path, repo: str, revision: str, meta: dict[str, dict]) -> dict[str, Any]:
    """Convert an allowlisted pair's verified snapshot (huggingface_hub layout: snapshots/<revision> linking into
    blobs/) to safetensors, and write CONVERSION_RECORD there last. `meta` is fetch_weights.hub_files' listing of the
    revision, from huggingface.co: the hash authority. Returns {file, sha256, record}."""
    import safetensors
    import torch

    if not bin_allowed(repo, revision):
        refuse(f"{repo} @ {str(revision)[:12]} is not in BIN_ALLOWED; nothing is converted")
    snap = Path(snap)
    if snap.name != revision:
        refuse(f"the snapshot directory is {snap.name}, not the revision {revision}")
    blobs = snap.parent.parent / "blobs"
    if not blobs.is_dir():
        refuse(f"{blobs} is not a directory (huggingface_hub layout expected)")
    refuse_if_converted(snap)
    foreign = foreign_blobs(blobs, meta)
    if foreign:
        refuse(f"{blobs} holds {foreign[:6]}, which huggingface.co's listing does not name (a conversion temporary or "
               "the output of an earlier run); purge the model (src/fetch_weights.py --purge) and fetch again")
    t0 = time.time()

    info = meta.get(BIN_INDEX) or {}
    idx_blob_id = info.get("blobId") or info.get("oid")
    if info.get("lfs") or not re.fullmatch(r"[0-9a-f]{40}", str(idx_blob_id or "")):
        refuse(f"huggingface.co lists no plain git blob id for {BIN_INDEX}")
    idx_bytes = (snap / BIN_INDEX).read_bytes()
    if git_blob_sha1(idx_bytes) != idx_blob_id:              # the bytes parsed are the bytes verified
        refuse(f"{BIN_INDEX}: git blob id {git_blob_sha1(idx_bytes)[:16]} is not huggingface.co's {idx_blob_id[:16]}")
    index = json.loads(idx_bytes.decode("utf-8"))            # not a result (D-297): a verified Hub file
    wmap = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(wmap, dict) or not wmap or not all(isinstance(k, str) and isinstance(v, str)
                                                         for k, v in wmap.items()):
        refuse(f"{BIN_INDEX} has no weight_map of tensor name -> shard file")
    shards = sorted(set(wmap.values()))
    n = len(shards)
    for i, s in enumerate(shards, 1):
        m = SHARD_RE.fullmatch(s)
        if not m or int(m.group(1)) != i or int(m.group(2)) != n:
            refuse(f"{BIN_INDEX} names shard {s!r}; expected pytorch_model-{i:05d}-of-{n:05d}.bin")
    present = sorted(p.name for p in snap.iterdir() if p.suffix in PICKLE_SUFFIXES)
    if present != shards:
        refuse(f"the snapshot's pickle files {present} are not the index's shards {shards}")
    want: dict[str, str] = {}
    for s in shards:
        lfs = (meta.get(s) or {}).get("lfs") or {}
        want[s] = lfs.get("sha256") or lfs.get("oid")
        if not re.fullmatch(r"[0-9a-f]{64}", str(want[s] or "")):
            refuse(f"huggingface.co lists no LFS sha256 for {s}")
    for s in shards:                                          # before any work: removing the names deletes no bytes
        _removable(snap / s, blobs / want[s])
    _removable(snap / BIN_INDEX, blobs / idx_blob_id)

    entries, staged, st_of = [], [], {}
    for i, s in enumerate(shards, 1):
        st_name = f"model-{i:05d}-of-{n:05d}.safetensors"
        st_of[s] = st_name
        tmp = blobs / f"{st_name}.converting.{os.getpid()}.{os.urandom(4).hex()}"
        print(f"  converting {s} -> {st_name}", flush=True)
        entry, _names = convert_shard(snap / s, want[s], tmp, {k for k, v in wmap.items() if v == s}, label=s)
        print(f"    bin sha256 {entry['bin_sha256'][:16]} = huggingface.co's; {entry['n_tensors']} tensors "
              f"{entry['dtypes']}; safetensors sha256 {entry['safetensors_sha256'][:16]}; bit for bit equal",
              flush=True)
        staged.append((tmp, entry["safetensors_sha256"]))
        entries.append({"bin_file": s, "safetensors_file": st_name, **entry})

    total = sum(e["tensor_bytes"] for e in entries)
    stated = (index.get("metadata") or {}).get("total_size") if isinstance(index.get("metadata"), dict) else None
    if stated is not None and stated != total:
        refuse(f"the converted tensors hold {total} bytes; {BIN_INDEX} states total_size {stated}; nothing linked "
               f"(this run's converted files are left as {[str(t) for t, _s in staged]})")
    clash = [str(blobs / sha) for _tmp, sha in staged if os.path.lexists(blobs / sha)]
    if clash:
        refuse(f"converted blob(s) already exist: {clash}; purge the model (src/fetch_weights.py --purge) and fetch "
               f"again (this run's converted files are left as {[t.name for t, _s in staged]})")
    for tmp, sha in staged:                                   # content-addressed blobs, never overwriting
        os.link(tmp, blobs / sha)                             # FileExistsError if it exists
        os.unlink(tmp)                                        # the temporary NAME only: bytes stay in blobs/<sha>
    st_index = {"metadata": {"total_size": total}, "weight_map": {k: st_of[wmap[k]] for k in sorted(wmap)}}
    st_index_bytes = (json.dumps(st_index, indent=2, sort_keys=True) + "\n").encode("utf-8")
    idx_blob = _exclusive_blob(blobs, st_index_bytes)
    for e in entries:
        _link(snap / e["safetensors_file"], blobs / e["safetensors_sha256"])
    _link(snap / ST_INDEX, idx_blob)
    removed = []
    for name in shards + [BIN_INDEX]:                         # checked by _removable above: names only, no bytes
        (snap / name).unlink()
        removed.append(name)

    record = {
        "what": "allenai's .bin shards converted to safetensors on NVMe (D-318 (7) item 8; docs/phase2_design.md "
                "section 3 (2)); the unmerged SFconvertbot pull request on the Hub is not used",
        "repo": repo, "revision": revision,
        "allowlist": "src/bin_weights.py BIN_ALLOWED (repo -> revision, the config check's commit)",
        "hash_authority": "huggingface.co: each shard's lfs.sha256 and the index's git blob id at the pinned revision "
                          "(src/fetch_weights.py hub_files), verified by fetch_weights before this conversion",
        "load": "torch.load(io.BytesIO(<the shard's bytes, read once>), map_location='cpu', weights_only=True), only "
                "after those bytes' sha256 equalled huggingface.co's",
        "save": "safetensors.torch.save_file(tensors, file, metadata={'format': 'pt'}); dtypes kept; a tensor that is "
                "not contiguous or shares storage with another is cloned first (n_cloned_before_save)",
        "comparison": "each converted file loaded back with safetensors.torch.load_file and compared with the "
                      "torch.load result: the same tensor names, and per tensor the same dtype, the same shape and "
                      "torch.equal of the raw bytes (each tensor viewed as uint8, so -0.0 vs 0.0 and NaN payloads "
                      "count)",
        "bin_index": {"file": BIN_INDEX, "bytes": len(idx_bytes), "sha256": hashlib.sha256(idx_bytes).hexdigest(),
                      "git_blob_sha1": idx_blob_id, "git_blob_sha1_equal_to_hub": True,
                      "metadata": index.get("metadata")},
        "safetensors_index": {"file": ST_INDEX, "bytes": len(st_index_bytes),
                              "sha256": hashlib.sha256(st_index_bytes).hexdigest(), "total_size": total,
                              "n_tensors": len(wmap)},
        "shards": entries, "n_shards": n, "n_tensors": sum(e["n_tensors"] for e in entries),
        "all_bit_for_bit_equal": all(e["bit_for_bit_equal"] for e in entries),
        "removed_from_snapshot": removed,
        "removal": "snapshot names only (links into blobs/); the .bin blobs stay until src/fetch_weights.py --purge "
                   "(D-153 (d)), so the snapshot holds only what transformers loads",
        "versions": {"torch": torch.__version__, "safetensors": safetensors.__version__,
                     "python": platform.python_version()},
        "code_sha256": {"src/bin_weights.py": sha256_file(Path(__file__)),
                        "src/fetch_weights.py": sha256_file(Path(__file__).with_name("fetch_weights.py"))},
        "bin_index_total_size_checked": stated is not None,      # equal when stated (else refused above)
        "convert_s": round(time.time() - t0, 3),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    rec_bytes = (json.dumps(record, indent=1, sort_keys=True) + "\n").encode("utf-8")
    with open(snap / CONVERSION_RECORD, "xb") as fh:          # written last: its presence means the conversion finished
        fh.write(rec_bytes)
        fh.flush()
        os.fsync(fh.fileno())
    sha = hashlib.sha256(rec_bytes).hexdigest()
    print(f"  converted {n} shard(s), {record['n_tensors']} tensors, all bit for bit equal; {CONVERSION_RECORD} "
          f"sha256 {sha[:16]}; removed from the snapshot (names only): {removed}", flush=True)
    return {"file": CONVERSION_RECORD, "sha256": sha, "record": record}


# --- the extraction side ------------------------------------------------------------------------------------------

def check_converted(snap: Path, repo: str, revision: str) -> dict[str, Any]:
    """src/extract_p2.py, for an allowlisted pair: the snapshot's weight files are exactly the conversion record's
    safetensors shards and index, each hashed now and equal to the record, and the record says every tensor was equal.
    Returns the block the extraction record embeds: the record's bytes, sha256 and content."""
    p = Path(snap) / CONVERSION_RECORD
    if not p.is_file():
        refuse(f"{repo} @ {revision[:12]} ships only .bin weights and its snapshot holds no {CONVERSION_RECORD}; "
               "src/fetch_weights.py converts it (src/bin_weights.py)")
    raw = p.read_bytes()
    try:
        rec = json.loads(raw.decode("utf-8"))                 # listed (D-297): bytes, sha256, content in the record
    except (UnicodeDecodeError, ValueError) as e:
        refuse(f"{CONVERSION_RECORD} is not UTF-8 JSON ({type(e).__name__})")
    if not isinstance(rec, dict) or rec.get("repo") != repo or rec.get("revision") != revision:
        refuse(f"{CONVERSION_RECORD} is not for {repo} @ {revision[:12]}")
    shards = rec.get("shards")
    if rec.get("all_bit_for_bit_equal") is not True or not isinstance(shards, list) or not shards or not all(
            isinstance(e, dict) and e.get("bit_for_bit_equal") is True and e.get("bin_sha256_equal") is True
            and re.fullmatch(r"[0-9a-f]{64}", str(e.get("bin_sha256")))
            and e.get("bin_sha256") == e.get("bin_sha256_hub") and isinstance(e.get("safetensors_file"), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(e.get("safetensors_sha256"))) for e in shards):
        refuse(f"{CONVERSION_RECORD} does not record every shard hash-checked and equal bit for bit")
    sti = rec.get("safetensors_index")
    if not isinstance(sti, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(sti.get("sha256"))):
        refuse(f"{CONVERSION_RECORD} records no sha256 for {ST_INDEX}")
    want = {e["safetensors_file"]: e["safetensors_sha256"] for e in shards}
    want[ST_INDEX] = sti["sha256"]
    have = sorted(q.name for q in Path(snap).iterdir()
                  if q.suffix in (".safetensors",) + PICKLE_SUFFIXES or q.name.endswith(".index.json"))
    if have != sorted(want):
        refuse(f"the snapshot's weight files {have} are not the conversion record's {sorted(want)}")
    for name, sha in want.items():
        got = sha256_file(Path(snap) / name)
        if got != sha:
            refuse(f"{name}: sha256 {got[:16]} is not the conversion record's {str(sha)[:16]}")
    return {"file": CONVERSION_RECORD, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "files_rehashed": sorted(want),
            "rehash": "every safetensors shard and the index hashed by extract_p2 before loading, equal to the record",
            "record": rec}


def verify_converted(snap: Path, repo: str, revision: str, meta: dict[str, dict], keep: list[str]) -> dict[str, Any]:
    """fetch_weights.py on a re-run of an allowlisted pair whose snapshot already holds conversion output (a failed
    sitting keeps its NVMe weights). Accepted, with nothing downloaded or converted, only if: the conversion record is
    there and check_converted passes (every converted file rehashed); its .bin files are exactly the revision's
    pytorch_model*.bin on huggingface.co, each with the LFS sha256 huggingface.co lists, and its index's git blob id is
    the Hub's; every other snapshot file is a keep-list file of the revision whose hash is the one huggingface.co
    publishes, and every keep-list file is there; blobs/ holds nothing foreign. Anything else refuses (purge first)."""
    snap = Path(snap)
    purge = "purge the model (src/fetch_weights.py --purge) and fetch again"
    if not (snap / CONVERSION_RECORD).is_file():
        refuse(f"{snap} holds conversion output {conversion_output(snap)[:6]} without its {CONVERSION_RECORD} (an "
               f"interrupted conversion); {purge}")
    block = check_converted(snap, repo, revision)
    rec = block["record"]
    code = rec.get("code_sha256")
    if not isinstance(code, dict) or not all(re.fullmatch(r"[0-9a-f]{64}", str(code.get(f"src/{n}")))
                                             for n in ("bin_weights.py", "fetch_weights.py")) \
            or not isinstance(rec.get("bin_index_total_size_checked"), bool):
        refuse(f"the conversion record names no code_sha256 or total_size check (written by code before 58ee75e); "
               f"{purge}")
    hub_bins = sorted(n for n in meta if is_allowed_bin(n, repo, revision))
    if sorted(str(e.get("bin_file")) for e in rec["shards"]) != hub_bins:
        refuse(f"the conversion record's .bin files are not huggingface.co's {hub_bins}; {purge}")
    for e in rec["shards"]:
        lfs = meta[e["bin_file"]].get("lfs") or {}
        if e["bin_sha256_hub"] != (lfs.get("sha256") or lfs.get("oid")):
            refuse(f"{e['bin_file']}: the conversion record's sha256 is not huggingface.co's; {purge}")
    idx = meta.get(BIN_INDEX) or {}
    if (rec.get("bin_index") or {}).get("git_blob_sha1") != (idx.get("blobId") or idx.get("oid")):
        refuse(f"the conversion record's {BIN_INDEX} is not huggingface.co's; {purge}")
    conv = set(block["files_rehashed"]) | {CONVERSION_RECORD}
    others = sorted(p.relative_to(snap).as_posix() for p in snap.rglob("*") if p.is_file()
                    and p.relative_to(snap).as_posix() not in conv)
    want_others = sorted(n for n in keep if not is_allowed_bin(n, repo, revision) and n != BIN_INDEX)
    if others != want_others:
        refuse(f"the converted snapshot's other files {others} are not the revision's {want_others}; {purge}")
    for n in others:
        info = meta[n]
        lfs = info.get("lfs") or {}
        want = lfs.get("sha256") or lfs.get("oid")
        got = sha256_file(snap / n) if want else git_blob_sha1((snap / n).read_bytes())
        if got != (want or info.get("blobId") or info.get("oid")):
            refuse(f"{n}: its hash is not the one huggingface.co publishes; {purge}")
    foreign = foreign_blobs(snap.parent.parent / "blobs", meta)
    unexpected = sorted(set(foreign) - {e["safetensors_sha256"] for e in rec["shards"]} - {
        rec["safetensors_index"]["sha256"]})
    if unexpected:
        refuse(f"blobs/ holds {unexpected[:6]}, which neither huggingface.co's listing nor the conversion record "
               f"names; {purge}")
    print(f"  already converted: {CONVERSION_RECORD} sha256 {block['sha256'][:16]} names huggingface.co's .bin "
          f"sha256s; {len(block['files_rehashed'])} converted files and {len(others)} other files verified; nothing "
          "fetched", flush=True)
    return block
