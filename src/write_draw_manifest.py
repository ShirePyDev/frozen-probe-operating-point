"""Write a draw manifest: the sha256 of every per-draw array file (D-163).

Each results/*_draws.npz is the per-draw record behind one results JSON (its 'draw_arrays'
block, or healthcheck_d156's 'draws_npz' block, names the file, its sha256 and size). This script refuses
unless every results JSON parses, every such npz lies in results/, exists and matches its JSON's record, and every
*_draws.npz in results/ is claimed by exactly one JSON. It then writes the manifest in sha256sum format (paths
relative to the repository root; check with `sha256sum -c` from the root), staged and verified before it takes its
name (F-027, F-032).

Manifests are NEVER rewritten. Results records store a manifest's own sha256 as an input (depth_d156_twox and the
d168_a100_* records store the first one's), so rewriting it would leave their recorded hash unresolvable. The first
manifest is manifests/draw_arrays.sha256; each later one is a new file, manifests/draw_arrays_<YYYYMMDD>[x].sha256,
listing every array on disk. Every entry of every earlier manifest whose file is still on disk must reappear with
the same hash. make_public_release.py checks the union of the manifests against the arrays in both directions and
ships them all.
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
MANIFEST = ROOT / "manifests" / "draw_arrays.sha256"          # the first manifest (D-163); never rewritten


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Write a new draw manifest (D-163); never rewrites one")
    ap.add_argument("--out", default=None, help="default: manifests/draw_arrays_<UTC date>.sha256")
    args = ap.parse_args()
    out = Path(args.out).resolve() if args.out else \
        ROOT / "manifests" / f"draw_arrays_{datetime.now(timezone.utc):%Y%m%d}.sha256"
    if out.parent != (ROOT / "manifests").resolve() or not (out.name.startswith("draw_arrays") and out.name.endswith(".sha256")):
        raise SystemExit("REFUSING: a draw manifest is manifests/draw_arrays*.sha256, which the release and the next "
                         "manifest read")
    if out.exists():
        raise SystemExit(f"REFUSING: {out.name} exists; manifests are never rewritten (records store their hashes)")
    claimed: dict[str, tuple[str, int, str]] = {}
    problems: list[str] = []
    for p in sorted(RESULTS.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            problems.append(f"results/{p.name}: does not parse ({e})")
            continue
        # 'draw_arrays' (4b and its successors) or 'draws_npz' (healthcheck_d156, D-163 (6); it names the size 'size')
        blocks = [k for k in ("draw_arrays", "draws_npz") if isinstance(d, dict) and isinstance(d.get(k), dict)]
        if not blocks:
            continue
        if len(blocks) > 1:
            problems.append(f"results/{p.name}: carries both draw_arrays and draws_npz")
            continue
        rec = d[blocks[0]]
        size = rec.get("bytes", rec.get("size"))
        if not isinstance(rec.get("file"), str) or not isinstance(rec.get("sha256"), str) or not isinstance(size, int):
            problems.append(f"results/{p.name}: {blocks[0]} lacks file, sha256 or size")
            continue
        f = rec["file"].replace("\\", "/")
        if not f.startswith("results/") or "/" in f[len("results/"):] or not f.endswith("_draws.npz"):
            problems.append(f"results/{p.name}: {blocks[0]}.file {f!r} is not results/<name>_draws.npz")
            continue
        if f in claimed:
            problems.append(f"{f}: claimed by {claimed[f][2]} and {p.name}")
            continue
        claimed[f] = (rec["sha256"], size, p.name)
    on_disk = {f"results/{p.name}" for p in RESULTS.glob("*_draws.npz")}
    problems += [f"{f}: on disk but claimed by no results JSON" for f in sorted(on_disk - set(claimed))]
    for f, (h, n, owner) in sorted(claimed.items()):
        q = ROOT / f
        if not q.exists():
            problems.append(f"{f}: named by {owner} but missing")
        elif q.stat().st_size != n or sha256_file(q) != h:
            problems.append(f"{f}: not the bytes {owner} recorded")
    earlier = sorted((ROOT / "manifests").glob("draw_arrays*.sha256"))
    for man in earlier:
        for line in man.read_text(encoding="utf-8").splitlines():
            h, f = line.split(None, 1)
            f = f.strip()
            if (ROOT / f).exists() and (f not in claimed or claimed[f][0] != h):
                problems.append(f"{f}: listed in {man.name} with {h[:16]}, which the new manifest would not keep")
    if problems:
        raise SystemExit("REFUSING:\n  " + "\n  ".join(problems))

    data = "".join(f"{h}  {f}\n" for f, (h, _, _) in sorted(claimed.items())).encode("utf-8")
    tmp, want = stage_bytes(out, data)
    try:
        os.link(tmp, out)              # fails if the name appeared meanwhile: never overwrite
        if sha256_file(out) != want or sha256_bytes(out.read_bytes()) != want:
            raise SystemExit(f"REFUSING: {out.name} differs from what was written")
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"{out.relative_to(ROOT)}: {len(claimed)} files, each verified against its results JSON; every entry of "
          f"{[m.name for m in earlier]} kept (sha256 {want})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
