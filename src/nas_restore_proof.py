"""The NAS restore proof (D-324 (6), N5; D-307 (3)): the proof that a gated model's NVMe snapshot is the verified NAS
keep copy goes INTO the snapshot, and src/extract_p2.py embeds it in the extraction record.

  write  scripts/p2_restore_from_nas.sh calls it after every file of the '<revision>.restoring' directory was copied and
         checked, before the directory is renamed to the snapshot path. It reads the keep manifest
         (logs/weights_keep_<org>__<model>@<rev12>.sha256, the committed list of '<sha256>  <keep dir>/<file>' lines),
         requires the directory to hold exactly the manifest's files, hashes every one of them itself, and writes
         PROOF (nas_restore_proof.json, exclusive) with: the repo and revision, the keep directory, the manifest's
         path, bytes and sha256, every file's path, bytes and sha256 (each equal to the manifest's), the file count and
         total bytes, the restore script's and this module's sha256, and the UTC time.
  check  src/extract_p2.py's snapshot() calls it for every snapshot that must come from the NAS (requires_proof: a
         gated meta-llama/* repo, which scripts/p2_sitting.sh always restores from the NAS) and for any snapshot that
         holds a proof: the proof must name this repo and revision and the expected manifest; the committed manifest
         must still have the sha256 the proof records and list exactly the proof's files; the snapshot must hold exactly
         those files (and the proof); every file is hashed again and must equal. A gated snapshot without a proof
         refuses. Returns the block the extraction record embeds: the proof's bytes, sha256 and content.
Paths in the proof are repository-relative (no host or account path, D-172). Nothing is deleted here.

usage (repo root):
  python3 src/nas_restore_proof.py write --repo R --revision V --dir <snapshot>.restoring --manifest M --keep K \\
      --script scripts/p2_restore_from_nas.sh
  python3 src/nas_restore_proof.py check --repo R --revision V --snapshot <snapshot dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PROOF = "nas_restore_proof.json"
GATED_PREFIX = "meta-llama/"        # scripts/p2_sitting.sh: meta-llama/*) GATED=1, restored from the NAS
RESTORE_SCRIPT = "scripts/p2_restore_from_nas.sh"
WRITER = "src/nas_restore_proof.py"
MANIFEST_LINE = re.compile(r"([0-9a-f]{64}) [ *](.+)")


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slug(repo: str, revision: str) -> str:
    return f"{repo.replace('/', '__')}@{revision[:12]}"


def manifest_name(repo: str, revision: str) -> str:
    return f"logs/weights_keep_{slug(repo, revision)}.sha256"


def keep_name(repo: str, revision: str) -> str:
    return f"data/weights_keep/{slug(repo, revision)}"


def requires_proof(repo: str) -> bool:
    return repo.startswith(GATED_PREFIX)


def parse_manifest(data: bytes, keep: str) -> dict[str, str]:
    """{path relative to the keep directory: sha256} from sha256sum lines '<sha256>  <keep>/<path>'."""
    if b"\r" in data:
        refuse("the manifest has CR characters (sha256sum writes LF lines)")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        refuse("the manifest is not UTF-8")
    out: dict[str, str] = {}
    for line in text.split("\n"):
        if not line:
            continue
        m = MANIFEST_LINE.fullmatch(line)
        if not m or not m.group(2).startswith(keep + "/"):
            refuse(f"manifest line {line[:100]!r} is not '<sha256>  {keep}/<file>'")
        rel = m.group(2)[len(keep) + 1:]
        parts = rel.split("/")
        if not rel or "" in parts or ".." in parts or "." in parts or rel == PROOF or rel in out:
            refuse(f"manifest path {m.group(2)!r} is empty, not clean, repeated or the proof's name")
        out[rel] = m.group(1)
    if not out:
        refuse("the manifest lists no file")
    return out


def _files(d: Path) -> dict[str, Path]:
    """The regular files under d (relative POSIX paths), the proof excepted; a symlink or other entry refuses."""
    out: dict[str, Path] = {}
    for p in sorted(Path(d).rglob("*")):
        rel = p.relative_to(d).as_posix()
        if p.is_symlink() or not (p.is_file() or p.is_dir()):
            refuse(f"{rel} in {d} is not a regular file or directory")
        if p.is_file() and rel != PROOF:
            out[rel] = p
    return out


def _rel(path: str | Path, root: Path) -> str:
    """A repository-relative path, refused if it lies outside the repository."""
    p = Path(path)
    p = p if p.is_absolute() else Path(root) / p
    rel = os.path.relpath(os.path.realpath(p), os.path.realpath(root)).replace("\\", "/")
    if rel.startswith("../") or rel == ".." or os.path.isabs(rel):
        refuse(f"{path} is outside the repository")
    return rel


def write_proof(work: Path, repo: str, revision: str, manifest: str, keep: str, script: str = RESTORE_SCRIPT,
                root: Path = ROOT) -> dict[str, Any]:
    """Hash every restored file in `work` against the keep manifest and write PROOF there (exclusive)."""
    if not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
        refuse(f"revision {revision!r} is not a 40-hex commit")
    if not requires_proof(repo):
        refuse(f"{repo} is not a gated model restored from the NAS")
    man_rel, keep_rel = _rel(manifest, root), _rel(keep, root)
    if man_rel != manifest_name(repo, revision) or keep_rel != keep_name(repo, revision):
        refuse(f"manifest {man_rel} and keep directory {keep_rel} are not {manifest_name(repo, revision)} and "
               f"{keep_name(repo, revision)}")
    work = Path(work)
    if os.path.lexists(work / PROOF):
        refuse(f"{work / PROOF} exists; a proof is never overwritten")
    man_bytes = (Path(root) / man_rel).read_bytes()
    want = parse_manifest(man_bytes, keep_rel)
    have = _files(work)
    if sorted(have) != sorted(want):
        refuse(f"the restored files {sorted(have)[:8]} are not the manifest's {sorted(want)[:8]}")
    files = []
    for rel in sorted(want):
        got = sha256_file(have[rel])
        if got != want[rel]:
            refuse(f"{rel}: restored sha256 {got[:16]} is not the manifest's {want[rel][:16]}")
        files.append({"path": rel, "bytes": have[rel].stat().st_size, "sha256": got})
    script_rel = _rel(script, root)
    proof = {
        "what": "the NVMe snapshot is the NAS keep copy, restored and verified file by file (D-307 (3); D-324 (6), N5)",
        "repo": repo, "revision": revision, "keep_dir": keep_rel,
        "manifest": {"file": man_rel, "bytes": len(man_bytes), "sha256": hashlib.sha256(man_bytes).hexdigest(),
                     "n_files": len(want)},
        "files": files, "n_files": len(files), "total_bytes": sum(f["bytes"] for f in files),
        "verified": "every restored file hashed by src/nas_restore_proof.py after the copy and equal to the manifest; "
                    "before the copy the restore script checked the NAS copy with sha256sum --strict -c of the "
                    "manifest and each copied file's sha256",
        "restore_script": {"file": script_rel, "sha256": sha256_file(Path(root) / script_rel)},
        "writer": {"file": WRITER, "sha256": sha256_file(Path(__file__))},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    data = (json.dumps(proof, indent=1, sort_keys=True) + "\n").encode("utf-8")
    with open(work / PROOF, "xb") as fh:                      # FileExistsError if it exists: never overwritten
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    print(f"restore proof: {len(files)} files, each sha256 equal to {man_rel}; {PROOF} sha256 "
          f"{hashlib.sha256(data).hexdigest()}", flush=True)
    return proof


def _attested(rec: dict[str, Any]) -> None:
    """The proof names the restore script and the writer, each with a sha256, and a UTC time it was written."""
    rs, wr = rec.get("restore_script"), rec.get("writer")
    hexed = re.compile(r"[0-9a-f]{64}")
    ok = (isinstance(rs, dict) and rs.get("file") == RESTORE_SCRIPT and hexed.fullmatch(str(rs.get("sha256")))
          and isinstance(wr, dict) and wr.get("file") == WRITER and hexed.fullmatch(str(wr.get("sha256"))))
    try:
        time.strptime(str(rec.get("created_utc")), "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        ok = False
    if not ok:
        refuse(f"{PROOF} does not name {RESTORE_SCRIPT} and {WRITER} with their sha256 and a UTC time")


def check_proof(snap: Path, repo: str, revision: str, root: Path = ROOT) -> dict[str, Any]:
    """The extraction side: the snapshot is exactly the proof's files, each hashed now, and the proof agrees with the
    committed keep manifest. Returns {file, bytes, sha256, files_rehashed, manifest_sha256_now, record}."""
    p = Path(snap) / PROOF
    if not p.is_file():
        refuse(f"{repo} @ {revision[:12]} is restored from the NAS keep copy (D-307 (3)) and its snapshot holds no "
               f"{PROOF}: restore it with {RESTORE_SCRIPT}, which writes the proof (D-324 (6), N5)")
    raw = p.read_bytes()
    try:
        rec = json.loads(raw.decode("utf-8"))                 # listed (D-297): bytes, sha256, content in the record
    except (UnicodeDecodeError, ValueError) as e:
        refuse(f"{PROOF} is not UTF-8 JSON ({type(e).__name__})")
    if not isinstance(rec, dict) or rec.get("repo") != repo or rec.get("revision") != revision:
        refuse(f"{PROOF} is not for {repo} @ {revision[:12]}")
    man = rec.get("manifest")
    if not isinstance(man, dict):
        refuse(f"{PROOF} has no manifest block")
    _attested(rec)
    if man.get("file") != manifest_name(repo, revision) or rec.get("keep_dir") != keep_name(repo, revision):
        refuse(f"{PROOF} names manifest {man.get('file')!r} and keep directory {rec.get('keep_dir')!r}, not "
               f"{manifest_name(repo, revision)} and {keep_name(repo, revision)}")
    mpath = Path(root) / man["file"]
    if not mpath.is_file():
        refuse(f"the keep manifest {man['file']} is not on disk; the proof cannot be checked against it")
    man_bytes = mpath.read_bytes()
    man_sha = hashlib.sha256(man_bytes).hexdigest()
    if man_sha != man.get("sha256"):
        refuse(f"{man['file']} has sha256 {man_sha[:16]} now; the proof records {str(man.get('sha256'))[:16]}")
    want = parse_manifest(man_bytes, rec["keep_dir"])
    files = rec.get("files")
    if not isinstance(files, list) or {f.get("path"): f.get("sha256") for f in files if isinstance(f, dict)} != want \
            or rec.get("n_files") != len(want) or len(files) != len(want):
        refuse(f"{PROOF}'s files are not exactly {man['file']}'s")
    have = _files(Path(snap))
    if sorted(have) != sorted(want):
        refuse(f"the snapshot's files {sorted(have)[:8]} are not the proof's {sorted(want)[:8]}")
    for rel in sorted(want):
        got = sha256_file(have[rel])
        if got != want[rel]:
            refuse(f"{rel}: sha256 {got[:16]} is not the proof's and the manifest's {want[rel][:16]}")
    return {"file": PROOF, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "files_rehashed": len(want), "manifest_sha256_now": man_sha,
            "rehash": "every snapshot file hashed by extract_p2 before loading, equal to the proof and to the "
                      "committed keep manifest",
            "record": rec}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="write or check the NAS restore proof (D-324 (6), N5)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    c = sub.add_parser("check")
    for s in (w, c):
        s.add_argument("--repo", required=True)
        s.add_argument("--revision", required=True)
    w.add_argument("--dir", required=True)
    w.add_argument("--manifest", required=True)
    w.add_argument("--keep", required=True)
    w.add_argument("--script", default=RESTORE_SCRIPT)
    c.add_argument("--snapshot", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "write":
        write_proof(Path(a.dir), a.repo, a.revision, a.manifest, a.keep, a.script, Path.cwd())
    else:
        out = check_proof(Path(a.snapshot), a.repo, a.revision, Path.cwd())
        print(f"restore proof OK: {out['files_rehashed']} files rehashed; {PROOF} sha256 {out['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
