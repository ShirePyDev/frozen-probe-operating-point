"""Check every SHA-256 that this repository's records and registry state against the files that are here.

    python release/verify_inputs.py            # from the repository root; Python 3.10 or later, standard library only

Every result record (results/**/*.json, docs/**/*.json) names the SHA-256 of the code that wrote it and of the
files it read; docs/registered_predictions.md names the first 16 hex digits of each result file it cites. For each
(file, SHA-256) a record lists (as src/provenance.py's _listed_files lists them) this prints one of:

  PASS         the file is here and its SHA-256 is the one stated.
  SUBSTITUTED  the file is here, its bytes are listed in release/substitutions.json, the stated SHA-256 is that
               entry's sha256_original and the file's is its sha256_public. The original hash is the authors'
               statement: the original bytes are not in this repository, so this check cannot confirm it.
  OTHER VERSION  the file is code or a document (.py, .sh, .md) that is here at another version: the record ran
               with an earlier version of a module that later changed, or read an earlier state of the registry.
               This repository ships one version of each; the record's code block names the commit it ran at, in
               the authors' working repository, which is not public. Counted; not a failure, and not a check.
  FAIL         the file is a record, manifest or other data file, is here, and its SHA-256 is neither stated
               value. Records and manifests are written once and never overwritten, so this should not occur.
  ABSENT       the file is not in this repository (hidden-state caches, probe-score arrays, records not released,
               model configurations, logs), or the record names it relative to another root (a dataset snapshot's
               own README.md, for example: a path with no directory). Counted by kind; not a failure.

It exits 1 if any FAIL is found, or if a substitutions.json entry does not match the file here; 0 otherwise.
It reads files only; it writes nothing and opens no network connection.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from provenance import _listed_files  # noqa: E402  (the same rule the records were written with)

SUBS = json.loads((ROOT / "release" / "substitutions.json").read_text(encoding="utf-8"))["files"]
SUB = {e["path"]: e for e in SUBS}
_cache: dict[str, str] = {}


def sha256(rel: str) -> str:
    if rel not in _cache:
        _cache[rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    return _cache[rel]


def norm(path: str) -> str:
    """A listed path as a path relative to the repository root (records list them relative to their own root)."""
    p = path.replace("\\", "/")
    for marker in ("guardrail-ccf/", "<REPO>/"):
        if marker in p:
            p = p.split(marker, 1)[1]
    return p.lstrip("./") if p.startswith("./") else p


VERSIONED = (".py", ".sh", ".md")


def classify(rel: str, stated: str) -> str:
    if "/" not in rel or not (ROOT / rel).is_file():
        return "ABSENT"
    here = sha256(rel)
    if here.startswith(stated.lower()) and len(stated) >= 16:
        return "PASS"
    e = SUB.get(rel)
    if e and e["sha256_original"].startswith(stated.lower()) and here == e["sha256_public"]:
        return "SUBSTITUTED"
    if rel.endswith(VERSIONED):
        return "OTHER VERSION"
    return "FAIL"


def absent_kind(rel: str) -> str:
    if "/" not in rel:
        return "a path with no directory (another root)"
    if rel.endswith((".npz", ".npy")):
        return "arrays (.npz/.npy)"
    top = rel.split("/")[0]
    return top + "/" if (ROOT / top).is_dir() or top in ("data", "logs", "queues") else "not a repository path"


def main() -> int:
    bad_table = [e["path"] for e in SUBS if not (ROOT / e["path"]).is_file() or sha256(e["path"]) != e["sha256_public"]]
    records = sorted(p.relative_to(ROOT).as_posix() for d in ("results", "docs")
                     for p in (ROOT / d).rglob("*.json"))
    counts, absent, fails, subs = Counter(), Counter(), [], Counter()
    for rec in records:
        try:
            doc = json.loads((ROOT / rec).read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            counts["unreadable record"] += 1
            continue
        for f, h in sorted(_listed_files(doc, set())):
            rel = norm(f)
            if rel == rec:
                continue
            st = classify(rel, h)
            counts[st] += 1
            if st == "ABSENT":
                absent[absent_kind(rel)] += 1
            elif st == "FAIL":
                fails.append(f"{rec}: lists {rel} at {h[:16]}, the file here is {sha256(rel)[:16]}")
            elif st == "SUBSTITUTED":
                subs[rel] += 1
    reg = 0
    for ln in (ROOT / "docs" / "registered_predictions.md").read_text(encoding="utf-8").split("\n"):
        for rel, h in re.findall(r"`(results/[^`]+\.json)` sha256 `([0-9a-f]{16})`", ln):
            st = classify(rel, h)
            reg += 1
            counts["registry " + st] += 1
            if st == "FAIL":
                fails.append(f"docs/registered_predictions.md: {rel} at {h}, the file here is {sha256(rel)[:16]}")
            elif st == "SUBSTITUTED":
                subs[rel] += 1
    print(f"records read: {len(records)}; listed (file, SHA-256) pairs: "
          f"{sum(v for k, v in counts.items() if not k.startswith('registry'))}; registry hash citations: {reg}")
    for k in ("PASS", "SUBSTITUTED", "OTHER VERSION", "FAIL", "ABSENT", "registry PASS", "registry SUBSTITUTED",
              "registry OTHER VERSION", "registry FAIL", "registry ABSENT", "unreadable record"):
        if counts[k]:
            print(f"  {k:22s} {counts[k]}")
    if absent:
        print("  ABSENT by top-level directory:", dict(absent.most_common()))
    for rel, n in sorted(subs.items()):
        e = SUB[rel]
        print(f"  SUBSTITUTED {rel}: cited {n} time(s); class {e['class']}; original {e['sha256_original'][:16]} "
              f"(authors' statement), here {e['sha256_public'][:16]}")
    for line in fails:
        print("  FAIL", line)
    if bad_table:
        print("  release/substitutions.json does not match the file here for:", bad_table)
    ok = not fails and not bad_table
    print("RESULT:", "no failure" if ok else f"{len(fails)} failure(s)" + (", substitutions table mismatch" if bad_table else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
