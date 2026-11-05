"""Stale-code audit: does the code that produced each results file match the commit it names?

The author's rule (2026-09-25): for every results file, compare the recorded script
sha256 with the hash of that script at the recorded git commit; any mismatch is marked
superseded in populations.py with the reason "produced by an uncommitted code copy".

Only results written with paired.code_block record a per-file sha256. For those the rule
is applied exactly: every recorded file is compared with its blob at the recorded commit.
A file absent at that commit counts as a mismatch. A pre-rewrite commit (D-150) is used
as-is when its object is still present (refs/preserve/*), otherwise translated through
docs/history_rewrite_20260924.tsv; which one happened is recorded.

Every other results file records at most a commit and git's porcelain status at run time
(provenance.git_state keeps the first 50 lines). There is no hash, so the rule cannot be
applied. What the record does show is reported and NOT acted on:

  script_dirty    the producing script is in the recorded status (modified, added or
                  untracked), so the run used code that differs from the recorded commit
  script_clean    the status list is complete and the script is not in it
  undetermined    the list was cut at 50 lines and the script is not in the kept part

Files with commit null (run on the node's non-git copy before SYNC_COMMIT existed) and
files with no git block cannot be audited against git at all; they are counted.

Reads results/ and git only; changes nothing. Writes results/stale_code_audit.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
from provenance import ensure_utf8_stdout, provenance_block, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SRC = ROOT / "src"
REWRITE_MAP = ROOT / "docs" / "history_rewrite_20260924.tsv"
REASON = "produced by an uncommitted code copy"


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True)


def run_kind_scripts() -> dict[str, str]:
    """run_kind -> the one src file that writes it; refuses a run_kind claimed twice."""
    pat = re.compile(r'(?:provenance_block\(\s*|"run_kind":\s*)"([a-z0-9_]+)"')
    out: dict[str, str] = {}
    for f in sorted(SRC.glob("*.py")):
        if f.name == Path(__file__).name:
            continue
        for rk in set(pat.findall(f.read_text(encoding="utf-8"))):
            if rk in out and out[rk] != f"src/{f.name}":
                raise SystemExit(f"run_kind {rk} is written by both {out[rk]} and src/{f.name}")
            out[rk] = f"src/{f.name}"
    return out


def load_rewrite_map() -> dict[str, str]:
    m = {}
    for line in REWRITE_MAP.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        old, new = line.split("\t")[:2]
        m[old] = new
    return m


def resolve_commit(commit: str, rewrite: dict[str, str]) -> tuple[str | None, str]:
    if git("cat-file", "-e", f"{commit}^{{commit}}").returncode == 0:
        return commit, "object present"
    if commit in rewrite and git("cat-file", "-e", f"{rewrite[commit]}^{{commit}}").returncode == 0:
        return rewrite[commit], "translated through the D-150 rewrite map"
    return None, "commit object not found and not in the rewrite map"


def blob_sha256(commit: str, path: str) -> str | None:
    r = git("cat-file", "blob", f"{commit}:{path}")
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None


def status_paths(lines: list[str]) -> list[tuple[str, str]]:
    """(XY code, path) from `git status --porcelain` lines; a rename keeps the new path."""
    out = []
    for ln in lines:
        code, p = ln[:2], ln[3:]
        if " -> " in p:
            p = p.split(" -> ", 1)[1]
        out.append((code, p.strip().strip('"')))
    return out


def in_status(script: str, entries: list[tuple[str, str]]) -> str | None:
    for code, p in entries:
        if p == script or (p.endswith("/") and script.startswith(p)):
            return f"{code} {p}"
    return None


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Stale-code audit of results/*.json")
    ap.add_argument("--out", default=str(RESULTS / "stale_code_audit.json"))
    args = ap.parse_args()

    code = P.code_block([Path(__file__), SRC / "paired.py", SRC / "populations.py",
                         SRC / "provenance.py", SRC / "superseded_provenance.py"])
    scripts = run_kind_scripts()
    rewrite = load_rewrite_map()
    primary = set(POP.primary_result_stems())
    superseded = set(POP.SUPERSEDED) | set(POP.SUPERSEDED_BY_ESTIMATOR)
    out_name = Path(args.out).name

    rows, cat = [], Counter()
    for f in sorted(RESULTS.glob("*.json")):
        if f.name == out_name:
            continue
        stem = f.stem
        row: dict = {"file": f"results/{f.name}", "primary": stem in primary,
                     "already_superseded": stem in superseded}
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            row.update(category="unreadable", detail=str(e))
            rows.append(row); cat[row["category"]] += 1
            continue
        if not isinstance(d, dict):
            row.update(category="not a JSON object")
            rows.append(row); cat[row["category"]] += 1
            continue
        rk = d.get("run_kind")
        row["run_kind"] = rk
        row["script"] = scripts.get(rk)

        if isinstance(d.get("code"), dict) and d["code"].get("sha256"):
            c = d["code"]
            resolved, how = resolve_commit(c["commit"], rewrite) if c.get("commit") else (None, "no commit")
            files = []
            for path, rec in sorted(c["sha256"].items()):
                at = blob_sha256(resolved, path) if resolved else None
                files.append({"path": path, "recorded": rec, "at_commit": at,
                              "status": "match" if at == rec else
                                        ("absent at commit" if resolved and at is None else
                                         ("commit unresolved" if not resolved else "MISMATCH"))})
            bad = [x for x in files if x["status"] != "match"]
            row.update(category="A: code block, hash compared", commit=c.get("commit"),
                       commit_source=c.get("commit_source"), commit_resolution=how,
                       files=files, verdict="MISMATCH" if bad else "match",
                       mismatched_files=[x["path"] for x in bad])
        elif isinstance(d.get("git"), dict) and d["git"].get("commit"):
            g = d["git"]
            resolved, how = resolve_commit(g["commit"], rewrite)
            lines = g.get("dirty_files") or []
            n_dirty = g.get("dirty_file_count")
            complete = n_dirty is not None and len(lines) >= n_dirty
            entries = status_paths(lines)
            hit = in_status(row["script"], entries) if row["script"] else None
            at = blob_sha256(resolved, row["script"]) if resolved and row["script"] else None
            if not row["script"]:
                ev = "no producing script identified for this run_kind"
            elif hit:
                ev = "script_dirty"
            elif complete:
                ev = "script_clean"
            else:
                ev = "undetermined"
            row.update(category="B: commit only, no script hash", commit=g["commit"],
                       commit_resolution=how, dirty=g.get("dirty"), dirty_file_count=n_dirty,
                       status_list_complete=complete, script_status_line=hit,
                       script_exists_at_commit=at is not None, evidence=ev,
                       dirty_src_files=[p for _, p in entries if p.startswith("src/")])
        elif isinstance(d.get("git"), dict):
            row.update(category="C: git block, commit null (no commit identity)")
        else:
            row.update(category="D: no git block")
        rows.append(row)
        cat[row["category"]] += 1

    A = [r for r in rows if r["category"].startswith("A")]
    B = [r for r in rows if r["category"].startswith("B")]
    C = [r for r in rows if r["category"].startswith("C")]
    mism = [r for r in A if r["verdict"] == "MISMATCH"]
    ev = Counter(r["evidence"] for r in B)
    summary = {
        "results_files": len(rows),
        "by_category": dict(cat),
        "A_compared": len(A),
        "A_mismatches": len(mism),
        "A_mismatch_files": [r["file"] for r in mism],
        "B_evidence": dict(ev),
        "B_script_dirty_primary": sorted(r["file"] for r in B
                                         if r["evidence"] == "script_dirty" and r["primary"]),
        "B_script_dirty_not_primary": sorted(r["file"] for r in B
                                             if r["evidence"] == "script_dirty" and not r["primary"]),
        "B_undetermined_primary": sorted(r["file"] for r in B
                                         if r["evidence"] == "undetermined" and r["primary"]),
        "C_primary": sorted(r["file"] for r in C if r["primary"]),
    }

    print(f"results files audited: {len(rows)}")
    for k, v in sorted(cat.items()):
        print(f"  {k:<52}{v:>5}")
    print(f"\nA (rule applied): {len(A)} compared, {len(mism)} mismatched")
    for r in A:
        print(f"  {r['file']:<44} {r['verdict']:<9} commit {str(r['commit'])[:12]} "
              f"({r['commit_resolution']}) {', '.join(r['mismatched_files'])}")
    print("\nB (no script hash; evidence from the recorded git status, not acted on):")
    for k, v in sorted(ev.items()):
        np_ = sum(1 for r in B if r["evidence"] == k and r["primary"])
        print(f"  {k:<52}{v:>5}   of which primary: {np_}")
    print(f"\nC (commit null): {len(C)}, of which primary: {len(summary['C_primary'])}")

    sha = write_results(Path(args.out), {
        **provenance_block("stale_code_audit"),
        "code": code,
        "rule": "for every results file, compare the recorded script sha256 with the hash of "
                "that script at the recorded git commit; any mismatch is marked superseded in "
                f"populations.py with the reason '{REASON}'",
        "summary": summary, "rows": rows,
    }, exclusive=True)
    print(f"\nWritten: {args.out}  sha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
