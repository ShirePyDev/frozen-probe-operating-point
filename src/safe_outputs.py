"""Exclusive outputs that are never deleted (D-303; the author's standing rules after F-054).

Used by src/phase2_pool_clusters.py, src/phase2_freeze_splits.py and src/fetch_dataset_snapshot.py, which run as
background queue jobs. The rules:
  - A file that has been given a final name is never removed, and neither is any file of DELETE_LIMIT bytes or more.
    The limit is 10**6, the smaller reading of "1 MB". Such files are deleted only by a human's scripts/safe_rm.sh
    call, one file per call, never inside a background task.
  - On failure every file stays. A final-named output of this call whose sha256 is what this call wrote is renamed
    aside to '<name>.failed.<pid>' by a no-replace rename. Anything else is left where it is. The run then exits
    nonzero, naming each file left and saying that removal is a human's scripts/safe_rm.sh call.
  - Unlinking a temporary NAME after a successful hard link is not a deletion: the bytes stay under the final name.
    A temporary under DELETE_LIMIT that never got a final name may be unlinked.
"""

from __future__ import annotations

import errno
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import sha256_bytes, sha256_file  # noqa: E402

DELETE_LIMIT = 1_000_000
REMOVAL_NOTE = "removal is a human's scripts/safe_rm.sh call, one file per call"
_AT_FDCWD, _RENAME_NOREPLACE = -100, 1


class MovedUnverified(OSError):
    """The rename HAPPENED (src no longer holds the entry), but the re-check after it could not confirm that what is
    now at dst is what was at src. Callers report the true state: moved, unverified; never 'left in place'."""


def fallback_noreplace(src: Path, dst: Path) -> None:
    """The no-replace rename where renameat2 is missing (EINVAL, ENOSYS: e.g. NFS; other POSIX systems).
    FILE (or symlink): os.link(src, dst, follow_symlinks=False) gives the entry a second name and fails if dst exists,
    so nothing is ever replaced. The first name is dropped only if os.lstat shows the two names are the same
    (st_ino, st_dev); otherwise both names are left and the error says so.
    DIRECTORY: os.rename moves a directory onto nothing, or onto an EMPTY directory, which it then replaces; a non-empty
    directory at dst makes it fail (ENOTEMPTY or EEXIST) and both stay. dst is checked first, so the one case it can
    replace is an EMPTY directory created at dst between that check and the rename: no bytes are at risk, since it held
    none. The re-check after the rename confirms that the inode now at dst is the one that was at src and that src is
    gone. It cannot detect that such an empty directory was replaced. If it fails, the rename has already happened, and
    MovedUnverified reports that true state."""
    src, dst = Path(src), Path(dst)
    if not (src.is_dir() and not src.is_symlink()):
        os.link(src, dst, follow_symlinks=False)         # FileExistsError if dst exists: nothing is replaced
        a, b = os.lstat(src), os.lstat(dst)
        if (a.st_ino, a.st_dev) != (b.st_ino, b.st_dev):
            raise OSError(errno.EEXIST, f"the new name is not a second name of {src}; both names are left", str(dst))
        # re-check immediately before the unlink (a swap of src in between would need a rename-over writer, which the
        # rules forbid; the remaining window is the few instructions between this lstat and the unlink)
        c = os.lstat(src)
        if (c.st_ino, c.st_dev) != (b.st_ino, b.st_dev):
            raise OSError(errno.EEXIST, f"{src} changed before its first name was dropped; both names are left", str(dst))
        os.unlink(src)                                  # the first name only; the same bytes stay under dst
        return
    if os.path.lexists(dst):
        raise FileExistsError(errno.EEXIST, "the target exists; nothing is replaced", str(dst))
    before = os.stat(src)
    os.rename(src, dst)
    after = os.stat(dst)
    if (after.st_ino, after.st_dev) != (before.st_ino, before.st_dev) or os.path.lexists(src):
        raise MovedUnverified(errno.EEXIST, f"moved {src} to the target, but the directory there is not verifiably "
                                            "the one moved", str(dst))


def noreplace_rename(src: Path, dst: Path) -> None:
    """Rename src to dst without ever replacing anything at dst (FileExistsError, or another OSError, if it cannot).
    Linux: renameat2(RENAME_NOREPLACE); where the kernel or filesystem lacks it (EINVAL, ENOSYS), and on other POSIX
    systems, fallback_noreplace. Windows: os.rename never replaces."""
    src, dst = Path(src), Path(dst)
    if os.name == "nt":
        os.rename(src, dst)
        return
    if sys.platform.startswith("linux"):
        fn = None
        try:
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            fn = libc.renameat2
            fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            fn.restype = ctypes.c_int
        except (OSError, AttributeError):
            fn = None
        if fn is not None:
            if fn(_AT_FDCWD, os.fsencode(str(src)), _AT_FDCWD, os.fsencode(str(dst)), _RENAME_NOREPLACE) == 0:
                return
            err = ctypes.get_errno()
            if err not in (errno.EINVAL, errno.ENOSYS):
                raise OSError(err, os.strerror(err), str(dst))
    fallback_noreplace(src, dst)


def aside_name(path: Path) -> Path:
    return Path(path).with_name(f"{Path(path).name}.failed.{os.getpid()}")


def rename_aside(path: Path, want_sha: str | None, matches=None) -> str:
    """Rename one final-named output of this call aside to '<name>.failed.<pid>', and say where its bytes are.
    A file is renamed only if its sha256 is want_sha (what this call wrote). A directory (want_sha None) is renamed only
    if matches(path) says its contents are this call's. Otherwise, or if the rename is not possible, it is left."""
    path = Path(path)
    aside = aside_name(path)
    if not os.path.lexists(path):
        return f"{path} (absent)"
    try:
        if want_sha is not None and sha256_file(path) != want_sha:
            return f"{path} (left in place: its sha256 is not what this call wrote)"
        if want_sha is None and (matches is None or not matches(path)):
            return f"{path} (left in place: its contents are not verifiably this call's)"
        noreplace_rename(path, aside)
    except MovedUnverified:
        return f"{aside} (moved there from {path.name}; the re-check after the rename could not confirm it)"
    except OSError as e:
        return f"{path} (left in place: {type(e).__name__})"
    if want_sha is not None and sha256_file(aside) != want_sha:
        return f"{aside} (renamed aside from {path.name}; its sha256 now differs from what was written)"
    return f"{aside} (renamed aside from {path.name})"


def keep_or_unlink_temporary(tmp: Path) -> str | None:
    """A temporary that never got a final name: unlinked if under DELETE_LIMIT, else left (and named)."""
    tmp = Path(tmp)
    if not os.path.lexists(tmp):
        return None
    size = tmp.stat().st_size
    if size < DELETE_LIMIT:
        tmp.unlink()
        return None
    return f"{tmp} (a temporary of {size} bytes, never given a final name; left in place)"


def failure(what: str, left: list[str]) -> SystemExit:
    """The exit of a failed run: what failed, each file left, and who removes them."""
    lines = [x for x in left if x]
    tail = ("\n  files left:\n    " + "\n    ".join(lines) + f"\n  {REMOVAL_NOTE}") if lines else \
        "\n  no file of this call is left"
    return SystemExit(f"FAILED (nothing deleted): {what}{tail}")


def write_all(outputs: list[tuple[Path, bytes]]) -> dict[Path, str]:
    """Stage every output (exclusive create, fsync, size and sha256 verified), then commit every one (a hard link to
    the final name, which fails if the name exists, then a read-back check). On any failure: committed outputs whose
    bytes are this call's are renamed aside, temporaries are unlinked only under DELETE_LIMIT, and a SystemExit names
    every file left. Returns {path: sha256}."""
    temps: list[Path] = []
    staged: list[tuple[Path, Path, str]] = []
    committed: list[tuple[Path, str]] = []
    try:
        for p, b in outputs:
            p = Path(p)
            p.parent.mkdir(parents=True, exist_ok=True)
            want = sha256_bytes(b)
            tmp = p.with_name(f"{p.name}.partial.{os.getpid()}.{os.urandom(4).hex()}")
            with open(tmp, "xb") as fh:
                temps.append(tmp)                            # only once THIS call created it (a name collision is not ours)
                fh.write(b)
                fh.flush()
                os.fsync(fh.fileno())
            if tmp.stat().st_size != len(b) or sha256_file(tmp) != want:
                raise OSError(f"stage verify: {tmp.name} does not hold the bytes written")
            staged.append((p, tmp, want))
        for p, tmp, want in staged:
            os.link(tmp, p)                                  # never overwrites: FileExistsError if p exists
            committed.append((p, want))
            os.unlink(tmp)                                   # the temporary NAME only; the bytes stay under p
            temps.remove(tmp)
            if sha256_file(p) != want:
                raise OSError(f"commit verify: {p.name} does not hold the bytes written")
    except BaseException as e:
        left = [rename_aside(p, want) for p, want in committed]
        left += [keep_or_unlink_temporary(t) for t in temps]
        raise failure(f"{type(e).__name__}: {e}", left) from None
    return {p: want for p, _, want in staged}
