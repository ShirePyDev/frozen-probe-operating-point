"""Write a draw manifest (D-163) that also admits tablev_inf_d437's draws_npz block (D-446 (4)).

src/write_draw_manifest.py (unchanged, F-092) refuses on results/tablev_inf_d437.json: its 'draws_npz' block names
the file without the results/ prefix and records no size. manifests/draw_arrays_20261011.sha256 already lists that
npz with the record's sha256 (verified by hand against the record in D-438 (R)). This module applies the same checks
as write_draw_manifest.py with ONE deviation: a block whose 'file' is a bare '<name>_draws.npz' is read as
results/<name>_draws.npz, and a block with no size is admitted only if its sha256 matches the bytes on disk AND an
earlier draw manifest already lists that path with that hash. Everything else (every JSON parses, one claimant per
npz, no unclaimed npz, every earlier manifest entry kept, staged write, never overwrite) is as before.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import ensure_utf8_stdout, sha256_bytes, sha256_file, stage_bytes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DEVIATION_RECORDS = {"tablev_inf_d437.json"}                     # the only records the deviation admits


def earlier_entries(root: Path) -> tuple[list[Path], dict[str, set[str]]]:
    mans = sorted((root / "manifests").glob("draw_arrays*.sha256"))
    listed: dict[str, set[str]] = {}
    for man in mans:
        for line in man.read_text(encoding="utf-8").splitlines():
            h, f = line.split(None, 1)
            listed.setdefault(f.strip(), set()).add(h)
    return mans, listed


def claims(root: Path, listed: dict[str, set[str]]) -> tuple[dict[str, tuple[str, str]], list[str]]:
    claimed: dict[str, tuple[str, str]] = {}
    problems: list[str] = []
    for p in sorted((root / "results").glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            problems.append(f"results/{p.name}: does not parse ({e})")
            continue
        blocks = [k for k in ("draw_arrays", "draws_npz") if isinstance(d, dict) and isinstance(d.get(k), dict)]
        if not blocks:
            continue
        if len(blocks) > 1:
            problems.append(f"results/{p.name}: carries both draw_arrays and draws_npz")
            continue
        rec = d[blocks[0]]
        size = rec.get("bytes", rec.get("size"))
        if not isinstance(rec.get("file"), str) or not isinstance(rec.get("sha256"), str):
            problems.append(f"results/{p.name}: {blocks[0]} lacks file or sha256")
            continue
        f = rec["file"].replace("\\", "/")
        if "/" not in f and p.name in DEVIATION_RECORDS:
            f = "results/" + f                                   # the deviation's first half (bare file name)
        if not f.startswith("results/") or "/" in f[len("results/"):] or not f.endswith("_draws.npz"):
            problems.append(f"results/{p.name}: {blocks[0]}.file {f!r} is not results/<name>_draws.npz")
            continue
        if f in claimed:
            problems.append(f"{f}: claimed by {claimed[f][1]} and {p.name}")
            continue
        q = root / f
        if not q.exists():
            problems.append(f"{f}: named by {p.name} but missing")
            continue
        if sha256_file(q) != rec["sha256"]:
            problems.append(f"{f}: not the bytes {p.name} recorded")
            continue
        if not isinstance(size, int):                            # the deviation's second half (no size)
            if p.name not in DEVIATION_RECORDS:
                problems.append(f"results/{p.name}: {blocks[0]} records no size")
                continue
            if rec["sha256"] not in listed.get(f, set()):
                problems.append(f"{f}: {p.name} records no size and no earlier manifest lists this hash")
                continue
        elif q.stat().st_size != size:
            problems.append(f"{f}: size differs from {p.name}")
            continue
        claimed[f] = (rec["sha256"], p.name)
    on_disk = {f"results/{p.name}" for p in (root / "results").glob("*_draws.npz")}
    problems += [f"{f}: on disk but claimed by no results JSON" for f in sorted(on_disk - set(claimed))]
    for f, hs in sorted(listed.items()):
        if (root / f).exists() and (f not in claimed or claimed[f][0] not in hs or len(hs) > 1):
            problems.append(f"{f}: an earlier manifest lists it with a hash the new manifest would not keep")
    return claimed, problems


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Write a new draw manifest (D-163, D-446 (4)); never rewrites one")
    ap.add_argument("--out", default=None, help="default: manifests/draw_arrays_<UTC date>.sha256")
    args = ap.parse_args()
    out = Path(args.out).resolve() if args.out else \
        ROOT / "manifests" / f"draw_arrays_{datetime.now(timezone.utc):%Y%m%d}.sha256"
    if out.parent != (ROOT / "manifests").resolve() or not (out.name.startswith("draw_arrays") and out.name.endswith(".sha256")):
        raise SystemExit("REFUSING: a draw manifest is manifests/draw_arrays*.sha256")
    if out.exists():
        raise SystemExit(f"REFUSING: {out.name} exists; manifests are never rewritten (records store their hashes)")
    mans, listed = earlier_entries(ROOT)
    claimed, problems = claims(ROOT, listed)
    if problems:
        raise SystemExit("REFUSING:\n  " + "\n  ".join(problems))
    data = "".join(f"{h}  {f}\n" for f, (h, _) in sorted(claimed.items())).encode("utf-8")
    tmp, want = stage_bytes(out, data)
    try:
        os.link(tmp, out)
        if sha256_file(out) != want or sha256_bytes(out.read_bytes()) != want:
            raise SystemExit(f"REFUSING: {out.name} differs from what was written")
    finally:
        if tmp.exists():
            tmp.unlink()
    new = sorted(set(claimed) - set(listed))
    print(f"{out.relative_to(ROOT).as_posix()}: {len(claimed)} files ({len(new)} new), each verified against its "
          f"results JSON; every entry of {len(mans)} earlier manifests kept (sha256 {want})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
