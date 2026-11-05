"""D-439 (2): D-423's summary (P-RA.1-4 per unit) with the author's acceptance of lane C's old LoRA records.

The author's item 2 of 2026-10-12: D-423's code at the old runs (src/realign_d423.py a90c398b, commit 045cb8e) and now
(9171c130, baaef80, run at 8f8df99) differ only by the SVD fallback branch (CHOICES (j)); every array of every re-run
unit with an old record compared number by number (scripts/compare_d423_reproduction.py; docs/reports_d439/
d423_reproduction.json) is identical, so the old LoRA records moved aside under F-089 are ACCEPTED with a fallback count
of 0 (the old code has no fallback branch: a record it completed met no non-converging SVD) and lane C's re-run stops.

src/realign_d423.py is NOT changed: its summary licenses records only through a predictions record written by its own
code block, so any edit to it would unlicense every new record. This module is realign_d423.run_summary with one
change, license_record(): a record is scored if
  NEW: it names the new predictions record (results/realign_d423_predictions.json, which passes
       realign_d423.passing_predictions against realign_d423's code block); its n_svd_fallbacks and svd_fallbacks are
       listed from the record; or
  OLD (D-439): it is a lora_ridge or lora_procrustes record that names the OLD predictions record (OLD_PRED_SHA, the
       file at OLD_PRED_REL, whose registered text must be realign_d423's), whose code block equals the new
       predictions record's in every file but src/realign_d423.py, which must be OLD_D423_SHA; it has no fallback
       field, and its count is listed as 0 with the reason.
Anything else refuses. Every other check of run_summary is kept (each record and its npz the file the manifest names,
the unit and map of each record, every record present). Modes:
  check     license every record present (missing ones listed, nothing scored, nothing written); exit 1 if any refuses
  summary   results/realign_d423_summary.json, as realign_d423's summary mode, with the D-439 block
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import realign_d423 as R  # noqa: E402  (imported, not changed)
from provenance import _RESULT_READS, _repo_rel, ensure_utf8_stdout, sha256_file  # noqa: E402

OWN = "src/realign_d423_summary_d439.py"
ASIDE = "logs/aside_20261011T120430Z_d423_f089"
OLD_PRED_REL = f"{ASIDE}/results/realign_d423_predictions.json"
OLD_PRED_SHA = "4f88bcdb778ac7828f1127a7fae5354897eb118c2ce2b14ca8294d08dc756442"
OLD_D423_SHA = "a90c398bcbb2fd3cac858c7cd7ef6aa09985633769a56d161ad6ecbf648bc81c"   # src/realign_d423.py at 045cb8e
OLD_UNITS = ("lora_ridge", "lora_procrustes")
RESTORE_MANIFEST = "logs/d439_restore_20261012.sha256"   # the 30 files moved back from ASIDE at 10:11Z (D-439 (2))
D439 = ("D-439 (2), the author's item 2 of 2026-10-12: the only code difference between D-423's old runs and now is "
        "the SVD fallback branch, and every compared array is identical; lane C's old LoRA records are accepted with "
        "a fallback count of 0 and lane C's re-run stops")


def d423_code(block: dict[str, Any]) -> dict[str, Any]:
    """realign_d423's own code block (its CODE_FILES) out of this module's: paired.code_block refuses a block that
    leaves out a loaded src module, so the block is taken with OWN and OWN dropped for the predictions record's check."""
    sha = {k: v for k, v in block["sha256"].items() if k != OWN}
    if set(sha) != set(R.CODE_FILES):
        R.refuse(f"the code block's files are not realign_d423's CODE_FILES: {sorted(set(sha) ^ set(R.CODE_FILES))}")
    return {**block, "sha256": sha}


def old_predictions() -> dict[str, Any]:
    """The old predictions record: at OLD_PRED_REL with OLD_PRED_SHA, of D-423's predictions kind and registered text."""
    p = R.ROOT / OLD_PRED_REL
    if p.is_symlink() or not p.is_file():
        R.refuse(f"no old predictions record at {OLD_PRED_REL}")
    doc = R.read_result(p)
    if _RESULT_READS[_repo_rel(p)] != OLD_PRED_SHA:
        R.refuse(f"{OLD_PRED_REL} is not the old predictions record D-439 accepted ({OLD_PRED_SHA[:12]})")
    reg = R.registered_text()
    if doc.get("run_kind") != R.RUN_KIND["predictions"] or R.canon(doc.get("predictions")) != R.canon(reg["predictions"]) \
            or R.canon(doc.get("amendment")) != R.canon(reg["amendment"]):
        R.refuse(f"{OLD_PRED_REL} is not a D-423 predictions record with this module's registered text")
    code = (doc.get("code") or {}).get("sha256")
    if not isinstance(code, dict):
        R.refuse(f"{OLD_PRED_REL} has no code block")
    return {"file": OLD_PRED_REL, "sha256": OLD_PRED_SHA, "code": code}


def restored_records() -> dict[str, str]:
    """The record files D-439 moved back from ASIDE, from RESTORE_MANIFEST (sha256sum lines): repo path -> sha256."""
    p = R.ROOT / RESTORE_MANIFEST
    if not p.is_file():
        R.refuse(f"no {RESTORE_MANIFEST}")
    out = {}
    for line in p.read_text().splitlines():
        h, _, f = line.partition("  ")
        if len(h) != 64 or not f:
            R.refuse(f"{RESTORE_MANIFEST}: a malformed line")
        out[f] = h
    return out


def license_record(unit: str, r_: str, doc: dict[str, Any], pred_new: dict[str, Any], new_code: dict[str, str],
                   pred_old: dict[str, Any] | None, sha: str | None = None,
                   restored: dict[str, str] | None = None) -> dict[str, Any]:
    """The record's licence and its SVD fallbacks, or a refusal (see the module docstring); an old record must also be
    the file RESTORE_MANIFEST names (sha, the record's sha256), when restored is given."""
    named = (doc.get("predictions_record") or {}).get("sha256")
    if named == pred_new["sha256"]:
        n = doc.get("n_svd_fallbacks")
        if not isinstance(n, int) or isinstance(n, bool) or not isinstance(doc.get("svd_fallbacks"), list) \
                or n != len(doc["svd_fallbacks"]):
            R.refuse(f"{r_}: a new record without a consistent n_svd_fallbacks / svd_fallbacks")
        return {"licence": "new", "predictions_record": pred_new["sha256"],
                "n_svd_fallbacks": doc["n_svd_fallbacks"], "svd_fallbacks": doc["svd_fallbacks"]}
    if named == OLD_PRED_SHA and unit in OLD_UNITS and pred_old is not None:
        code = (doc.get("code") or {}).get("sha256") or {}
        want = {**new_code, R.OWN: OLD_D423_SHA}
        if code != want or pred_old["code"] != want:
            R.refuse(f"{r_}: an old record whose code block is not the new one with src/realign_d423.py at "
                     f"{OLD_D423_SHA[:12]}")
        if "n_svd_fallbacks" in doc or "svd_fallbacks" in doc:
            R.refuse(f"{r_}: an old record with a fallback field (the old code had none)")
        if restored is not None and restored.get(r_) != sha:
            R.refuse(f"{r_}: an old record that is not the file {RESTORE_MANIFEST} names (D-439 restored 15 units)")
        return {"licence": "old, accepted (D-439)", "predictions_record": OLD_PRED_SHA, "n_svd_fallbacks": 0,
                "svd_fallbacks": [], "note": "no fallback field: src/realign_d423.py a90c398b has no fallback branch, "
                                             "so a record it completed met no non-converging SVD (D-439)"}
    R.refuse(f"{r_} was licensed by another predictions record ({str(named)[:12]})")
    raise AssertionError


def records(rd: Path, need_all: bool):
    """(unit, pop, form, stem, repo-relative record path, record path) for every record of D-423's 36 units."""
    for unit, (pop, form, pairs) in R.UNITS.items():
        for pair in pairs:
            s = R.stem_of(unit, pair)
            r_ = f"results/realign_d423_{unit}_{s}.json"
            p = rd / Path(r_).name
            if not p.is_file():
                if need_all:
                    R.refuse(f"{r_} is missing: every record is needed (a missing record is not a skip)")
                yield unit, pop, form, s, r_, None
                continue
            yield unit, pop, form, s, r_, p


def run_check(rd: Path) -> int:
    code0 = d423_code(R.P.code_block(list(dict.fromkeys([OWN, *R.CODE_FILES])), allow_no_commit=True))
    pred_new = R.passing_predictions(code0, rd)
    pred_old = old_predictions()
    restored = restored_records()
    out: dict[str, list[str]] = {"new": [], "old, accepted (D-439)": [], "missing": []}
    for unit, _pop, form, _s, r_, p in records(rd, need_all=False):
        if p is None:
            out["missing"].append(r_)
            continue
        doc = R.read_result(p)
        if (doc.get("config") or {}).get("map") != form or (doc.get("config") or {}).get("unit") != unit:
            R.refuse(f"{r_} is not a {unit} record")
        lic = license_record(unit, r_, doc, pred_new, code0["sha256"], pred_old, _RESULT_READS[_repo_rel(p)], restored)
        out[lic["licence"]].append(f"{r_} (fallbacks {lic['n_svd_fallbacks']})")
    print(json.dumps({k: v for k, v in out.items()}, indent=1))
    print(f"check: {len(out['new'])} new, {len(out['old, accepted (D-439)'])} old accepted, {len(out['missing'])} missing")
    return 0


def run_summary(out: str | Path | None = None, records_dir: str | Path | None = None) -> Path:
    """realign_d423.run_summary with license_record() in place of its one-predictions-record check (D-439 (2))."""
    rd = Path(records_dir).resolve() if records_dir else R.RESULTS
    final_path = (R.RESULTS / "realign_d423_summary.json").resolve()
    out_path = Path(out).resolve() if out else final_path
    if out_path != final_path and out_path.is_relative_to(R.RESULTS.resolve()):
        R.refuse("a test run (--out) must not write under results/")
    if out_path.exists():
        R.refuse(f"{out_path} exists; results are never overwritten")
    code_own = R.P.code_block(list(dict.fromkeys([OWN, *R.CODE_FILES])))
    code0 = d423_code(code_own)
    pred = R.passing_predictions(code0, rd)
    pred_old = old_predictions()
    restored = restored_records()
    listed, man_sha = R.TP2.read_manifest((R.LOGS if records_dir is None else rd) / Path(R.MANIFEST_OWN).name)
    scored, bound, fallbacks = {}, {}, {}
    for unit, (pop, form, pairs) in R.UNITS.items():
        recs = {}
        for _u, _pop, _form, s, r_, p in records(rd, need_all=True):
            if _u != unit:
                continue
            doc = R.read_result(p)
            h = _RESULT_READS[_repo_rel(p)]
            nz = doc.get("draws_npz") or {}
            if listed.get(r_) != h or listed.get(str(nz.get("file"))) != nz.get("sha256"):
                R.refuse(f"{r_} or its npz is not the file {R.MANIFEST_OWN} names")
            lic = license_record(unit, r_, doc, pred, code0["sha256"], pred_old, h, restored)
            if (doc.get("config") or {}).get("map") != form or (doc.get("config") or {}).get("unit") != unit:
                R.refuse(f"{r_} is not a {unit} record")
            recs[s] = doc
            bound[r_] = {"file": r_, "sha256": h, "licence": lic["licence"]}
            fallbacks[r_] = {k: v for k, v in lic.items() if k != "licence"}
        scored[unit] = {"population": pop, "map": form, "n_reps": R.N_REPS[form], **R.RA.score_population(recs)}
    payload = {**R.provenance_block(R.RUN_KIND["summary"]), "preregistration": R.PREREG, "code": code_own,
               "scorer": OWN, "d439": D439, "predictions_record": pred,
               "old_predictions_record": {k: v for k, v in pred_old.items() if k != "code"},
               "manifest": {"file": R.MANIFEST_OWN, "sha256": man_sha}, "records": bound,
               "svd_fallbacks_by_record": fallbacks,
               "n_svd_fallbacks_total": sum(v["n_svd_fallbacks"] for v in fallbacks.values()),
               **R.registered_text(),
               "verdicts": {u: {k: v["verdict"] for k, v in sc_.items() if k.startswith("P-RA")}
                            for u, sc_ in scored.items()},
               "units": scored, "implementation_choices": R.CHOICES, "inputs": dict(_RESULT_READS)}
    problems = R.input_identity_problems(payload)
    if problems:
        R.refuse(f"to write (D-297 (1)): {problems}")
    R.TP2.write_outputs([(out_path, R.RA.json_bytes_of(payload))])
    print(json.dumps(payload["verdicts"], indent=1))
    return out_path


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("check")
    c.add_argument("--records-dir", default=None)
    s = sub.add_parser("summary")
    s.add_argument("--out", default=None)
    s.add_argument("--records-dir", default=None)
    a = ap.parse_args(argv)
    if a.mode == "check":
        return run_check(Path(a.records_dir).resolve() if a.records_dir else R.RESULTS)
    run_summary(a.out, a.records_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
