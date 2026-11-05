"""Does every git commit recorded in a results file still resolve? (D-150, D-153)

master was rewritten on 2026-09-24 to remove data/ from history (D-150), which changed
every commit hash on it. A results file's git.commit is part of its provenance, so each
recorded hash must still lead somewhere. It resolves if ANY of these holds:

  current    it is an ancestor of master as rewritten;
  via_map    docs/history_rewrite_*.tsv maps it to a new commit that is on master;
  preserved  it is reachable from a ref under refs/preserve/ (local-only; mirrored to C:
             as a git bundle, D-153), which also covers commits that were never on the
             workstation's master, such as the server's aaf5265.

Commits recorded as null (every node-side run since the node lost its .git, D-148) are
counted separately: they carry no git provenance to resolve. Exits 1 if any recorded
commit resolves by none of the three routes.

  python src/validate_git_commits.py [--out results/git_commit_resolution.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import ensure_utf8_stdout, provenance_block, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def is_ancestor(commit: str, ref: str) -> bool:
    return git("merge-base", "--is-ancestor", commit, ref).returncode == 0


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Resolve every recorded git commit")
    ap.add_argument("--out", default=str(ROOT / "results" / "git_commit_resolution.json"))
    args = ap.parse_args()

    mapping: dict[str, str] = {}
    for f in sorted((ROOT / "docs").glob("history_rewrite_*.tsv")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                old, new = line.split("\t")[:2]
                mapping[old] = new
    preserve = [r for r in git("for-each-ref", "--format=%(refname)", "refs/preserve/").stdout.split() if r]

    recorded: dict[str, list[str]] = defaultdict(list)
    n_null = n_nogit = 0
    for p in sorted((ROOT / "results").glob("*.json")):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:  # noqa: BLE001 - unreadable files are verify_provenance's job
            continue
        g = d.get("git") if isinstance(d, dict) else None
        if not isinstance(g, dict):
            n_nogit += 1
            continue
        if not g.get("commit"):
            n_null += 1
            continue
        recorded[g["commit"]].append(p.name)

    rows = []
    for c, files in sorted(recorded.items()):
        exists = git("cat-file", "-e", f"{c}^{{commit}}").returncode == 0
        routes = []
        if exists and is_ancestor(c, "master"):
            routes.append("current")
        if c in mapping and is_ancestor(mapping[c], "master"):
            routes.append("via_map")
        if exists:
            routes += [f"preserved:{r}" for r in preserve if is_ancestor(c, r)]
        rows.append({"commit": c, "n_results": len(files), "routes": routes,
                     "new_commit": mapping.get(c), "object_present": exists,
                     "resolves": bool(routes), "example_result": files[0]})
    unresolved = [r for r in rows if not r["resolves"]]
    counts = defaultdict(int)
    for r in rows:
        for route in r["routes"]:
            counts[route.split(":")[0]] += 1

    print(f"results with git.commit: {sum(r['n_results'] for r in rows)} "
          f"({len(rows)} distinct commits); git.commit null: {n_null}; no git block: {n_nogit}")
    print(f"resolving routes (a commit may have several): {dict(counts)}")
    print(f"UNRESOLVED: {len(unresolved)}")
    for r in unresolved:
        print(f"  {r['commit']}  ({r['n_results']} results, e.g. {r['example_result']})")

    payload = {**provenance_block("validate_git_commits"),
               "summary": {"distinct_commits": len(rows),
                           "results_with_commit": sum(r["n_results"] for r in rows),
                           "results_commit_null": n_null, "results_without_git_block": n_nogit,
                           "routes": dict(counts), "unresolved": len(unresolved),
                           "map_files": [f.name for f in sorted((ROOT / "docs").glob("history_rewrite_*.tsv"))],
                           "preserve_refs": preserve},
               "commits": rows}
    digest = write_results(args.out, payload)
    print(f"Written: {args.out}\nsha256:  {digest}")
    return 1 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
