"""Provenance audit, part (D-256 (9), D-252): the extraction records whose git block is empty.

The node has had no .git (D-148), so its extraction records name no commit; the commit, if anywhere, is only in the
run log. scripts/collect_extraction_log_evidence.sh, run read-only on the node from its repo root, lists for each such
record every log that names it (path, UTC mtime, bytes, sha256, commit lines) with its own sha256 as run and the
collection time. That TSV is committed; it holds no host or account path (D-172).

The collector matches by name, so a record is also named by logs that did not produce it (later reruns, cache moves,
audits, the log of an earlier run it overwrote). The PRODUCING LOG is therefore the log with the earliest mtime at or
after the record's utc (a record is written before its run's log is last written), and only its commit lines decide:
  RECOVERED      the producing log names exactly one commit; src/extract_hidden.py at that commit is hashed, so
                 the code can be read back from git;
  NOT RECOVERED  it names none. The record then gets the git bound only: the latest commit touching
                 src/extract_hidden.py at or before the record's utc. The node ran at most that version IF it ran a
                 committed state (not checkable; in the D-028 era the node was a working tree). Any earlier version
                 is possible, and there is no lower bound. The minutes from that commit to the record are given,
                 because a bound commit made during the run is a weak bound;
  CONFLICT       the producing log names more than one commit;
  NO LOG         no log is written at or after the record.
Commits named by the other logs are listed beside, never used. Each log that is also tracked in this repository is
compared with its tracked copy by sha256. The script checks that the evidence covers exactly the commit-less records on
disk. It writes results/extraction_commit_audit.json, never overwriting it; a test run writes outside results/.
--r2 (D-308) audits every commit-less record on disk now (D-286's plus the D-168 A100 re-run records) from newer evidence
and writes results/extraction_commit_audit_r2.json; every entry D-286 decided must come out identical.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import (_RESULT_READS, ensure_utf8_stdout, provenance_block, read_result, sha256_bytes,  # noqa: E402
                        sha256_file, write_results)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
EVIDENCE = ROOT / "manifests" / "extraction_log_evidence_20260930.tsv"
# D-308: evidence for every commit-less record on disk now, and the first audit (D-286) the --r2 entries must reproduce
EVIDENCE_R2 = ROOT / "manifests" / "extraction_log_evidence_20261001.tsv"
FIRST_AUDIT = ("results/extraction_commit_audit.json", "27fbe0141f0ec9fb66eb8f71585ccece5fe4f59a1b5965a0f9b1fd5f5560a87a")
COLLECTOR = ROOT / "scripts" / "collect_extraction_log_evidence.sh"
CODE = "src/extract_hidden.py"
HEX = re.compile(r"[0-9a-f]{7,40}", re.IGNORECASE)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def utc_of(s: str) -> datetime:
    """An ISO time with an offset, or the collector's 'YYYY-MM-DD HH:MM:SS' in UTC."""
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Commit evidence for the extraction records with an empty git block")
    ap.add_argument("--evidence", default=None, help=f"default {EVIDENCE.relative_to(ROOT)} (with --r2: {EVIDENCE_R2.relative_to(ROOT)})")
    ap.add_argument("--r2", action="store_true",
                    help="D-308: the audit over every commit-less record on disk now (D-286's 24 plus the D-168 A100 "
                         "re-run records), written as extraction_commit_audit_r2.json; every record the first audit "
                         "covered must get the identical entry")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.evidence is None:
        args.evidence = str(EVIDENCE_R2 if args.r2 else EVIDENCE)
    final_path = (RESULTS / f"extraction_commit_audit{'_r2' if args.r2 else ''}.json").resolve()
    out_path = Path(args.out).resolve() if args.out else final_path
    if out_path.exists():
        raise SystemExit(f"REFUSING: {out_path} exists; results are never overwritten")
    if out_path != final_path and out_path.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test run (--out) must not write under results/")
    ev_path = Path(args.evidence).resolve()
    ev_bytes = ev_path.read_bytes()

    records = {}
    for f in sorted(RESULTS.glob("extract__*.json")):
        d = json.loads(f.read_bytes().decode("utf-8"))
        if not (d.get("git") or {}).get("commit"):
            records[f.name] = {"sha256": sha256_file(f), "utc": d["utc"]}
    header: dict[str, str] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    tracked = set(git("ls-tree", "-r", "--name-only", "HEAD", "logs").splitlines())

    def head_sha(path: str) -> str:
        return sha256_bytes(subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, check=True, capture_output=True).stdout)
    for line in ev_bytes.decode("utf-8").splitlines():
        if line.startswith("#"):
            k, _, v = line[1:].partition("\t")
            header[k] = v
            continue
        if not line:
            continue
        rec, log, mtime, size, sha, commits = line.split("\t")
        rows.setdefault(rec, [])
        if log != "-":
            local = ("absent from this repository" if log not in tracked else
                     "equal" if head_sha(log) == sha else "DIFFERS")
            rows[rec].append({"log": log, "mtime_utc": mtime, "bytes": int(size), "sha256": sha, "tracked_copy": local,
                              "commits_named": sorted({m.group(0).lower() for m in HEX.finditer(commits)})
                              if commits != "-" else []})
    if set(rows) != set(records):
        raise SystemExit(f"REFUSING: the evidence covers {sorted(set(rows) ^ set(records))} differently from the "
                         f"commit-less records on disk")
    if header.get("collector_sha256") != sha256_file(COLLECTOR):
        raise SystemExit("REFUSING: the evidence was not collected by the committed collector")

    history = [(c, datetime.fromtimestamp(int(t), tz=timezone.utc)) for c, t in
               (ln.split() for ln in git("log", "--format=%H %ct", "--", CODE).splitlines())]
    out: dict[str, Any] = {}
    counts = {"RECOVERED": 0, "NOT RECOVERED": 0, "CONFLICT": 0, "NO LOG": 0}
    for name, r in records.items():
        utc = utc_of(r["utc"])
        later = [h for h in rows[name] if utc_of(h["mtime_utc"]) >= utc]
        first = min(utc_of(h["mtime_utc"]) for h in later) if later else None
        tied = [h for h in later if utc_of(h["mtime_utc"]) == first]          # every log with the earliest mtime decides
        producing = {"log": " + ".join(h["log"] for h in tied), "sha256": ",".join(h["sha256"][:12] for h in tied),
                     "tracked_copy": ", ".join(h["tracked_copy"] for h in tied),
                     "commits_named": sorted({c for h in tied for c in h["commits_named"]})} if tied else None
        before = [(c, t) for c, t in history if t <= utc]
        after = [(c, t) for c, t in history if t > utc]
        bound = max(before, key=lambda x: x[1]) if before else None
        nxt = min(after, key=lambda x: x[1]) if after else None
        entry: dict[str, Any] = {
            "record_sha256": r["sha256"], "record_utc": r["utc"], "rerun": "__rerun_" in name,
            "producing_log": producing["log"] if producing else None,
            "commits_named_elsewhere": sorted({c for h in rows[name] if not any(h is t for t in tied)
                                               for c in h["commits_named"]}),
            "logs": rows[name],
            "git_bound": {"file": CODE,
                          "at_most": {"commit": bound[0], "commit_utc": bound[1].isoformat(),
                                      "minutes_to_record": round((utc - bound[1]).total_seconds() / 60, 1)} if bound else None,
                          "next_commit_after_record": {"commit": nxt[0], "commit_utc": nxt[1].isoformat()} if nxt else None,
                          "reading": "the node ran at most this version IF it ran a committed state (not checkable); any "
                                     "earlier version is possible; there is no lower bound"}}
        named = sorted({c for c in producing["commits_named"]}) if producing else []
        full = sorted({git("rev-parse", "--verify", f"{c}^{{commit}}").strip() for c in named})
        if producing is None:
            entry.update({"verdict": "NO LOG", "commit": None})
        elif len(full) == 1:
            code = subprocess.run(["git", "show", f"{full[0]}:{CODE}"], cwd=ROOT, check=True, capture_output=True).stdout
            entry.update({"verdict": "RECOVERED", "commit": full[0], "code_sha256_at_commit": sha256_bytes(code),
                          "rests_on": f"{producing['log']} ({producing['sha256']}...), tracked copy: "
                                      f"{producing['tracked_copy']}"})
        elif not full:
            entry.update({"verdict": "NOT RECOVERED", "commit": None})
        else:
            entry.update({"verdict": "CONFLICT", "commits_named": full})
        counts[entry["verdict"]] += 1
        out[name] = entry
        print(f"{entry['verdict']:>13}  {name}  <- {entry['producing_log']}")
    n_logs = [h for v in rows.values() for h in v]
    first_check = None
    if args.r2:
        # every record D-286 covered must get the identical entry (same bytes, same evidence reading); only new records
        # (the D-168 A100 re-runs, D-308) may be added
        r1 = read_result(ROOT / FIRST_AUDIT[0])
        if _RESULT_READS[FIRST_AUDIT[0]] != FIRST_AUDIT[1]:
            raise SystemExit(f"REFUSING: {FIRST_AUDIT[0]} is not D-286's record ({FIRST_AUDIT[1][:8]})")
        lost = sorted(set(r1["records"]) - set(out))
        if lost:
            raise SystemExit(f"REFUSING: records the first audit covered are no longer commit-less on disk: {lost}")
        # entries compared on what decides them, including git_bound.at_most (the answer of a NOT RECOVERED entry);
        # git_bound.next_commit_after_record may legitimately change after a later commit, so it is not compared
        keys = ("record_sha256", "record_utc", "rerun", "producing_log", "verdict", "commit", "code_sha256_at_commit",
                "commits_named_elsewhere", "rests_on")

        def decides(e: dict[str, Any]) -> list[Any]:
            return [e.get(k) for k in keys] + [(e.get("git_bound") or {}).get("at_most")]
        differ = sorted(n for n in r1["records"]
                        if json.dumps(decides(r1["records"][n]), sort_keys=True) != json.dumps(decides(out[n]), sort_keys=True))
        if differ:
            raise SystemExit(f"REFUSING to write: entries differ from {FIRST_AUDIT[0]}: {differ}")
        # 'logs' may gain rows (later logs naming a record), but every row of the first audit must still be there
        lost_rows = sorted(n for n in r1["records"]
                           if any(json.dumps(h, sort_keys=True) not in {json.dumps(x, sort_keys=True) for x in out[n]["logs"]}
                                  for h in r1["records"][n]["logs"]))
        if lost_rows:
            raise SystemExit(f"REFUSING to write: evidence rows of the first audit are missing for {lost_rows}")
        keys = (*keys, "git_bound.at_most", "logs (a superset of the first audit's rows)")
        first_check = {"file": FIRST_AUDIT[0], "sha256": FIRST_AUDIT[1], "keys_compared": list(keys),
                       "n_records_equal": len(r1["records"]), "added": sorted(set(out) - set(r1["records"]))}
    payload = {**provenance_block("extraction_commit_audit"), "preregistration": "D-256 (9), D-252 (the audit item)"
               + ("; D-308 (the D-168 A100 re-run records added)" if args.r2 else ""),
               **({"extends": first_check} if first_check else {}),
               "evidence": {"file": str(ev_path.relative_to(ROOT)).replace("\\", "/") if ev_path.is_relative_to(ROOT) else ev_path.name,
                            "sha256": sha256_bytes(ev_bytes), "collected_utc": header.get("collected_utc"),
                            "collector": {"file": "scripts/collect_extraction_log_evidence.sh",
                                          "sha256_as_run_on_the_node": header.get("collector_sha256"),
                                          "sha256_committed": sha256_file(COLLECTOR)},
                            "log_rows": len(n_logs),
                            "tracked_copies": {k: sum(h["tracked_copy"] == k for h in n_logs)
                                               for k in ("equal", "DIFFERS", "absent from this repository")}},
               "producing_log_rule": "the log with the earliest mtime at or after the record's utc; only its commit lines decide",
               "code_file": CODE, "n_records": len(records), "counts": counts, "records": out}
    digest = write_results(out_path, payload, exclusive=True)
    print(f"{counts}\nWritten: {out_path}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
