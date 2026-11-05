"""D-440 (3): A2's scorer with D-423's code-hash licence, and the A2 predictions and category records re-issued under
the code that ran.

The author's item 3 of 2026-10-13: the A2 predictions record (results/phase3_a2_predictions.json, written 2026-10-09 at
291145f) names src/a2_p3.py ad98298; every A2 run record (6 lambda, 30 train, features, kl, arc, score) names ONE code
block, the current one (a2_p3.py 82d9d54). The five a2_p3.py commits between (045de0e, 8c5b23a: F-086, the code-block
list; 0d56959, 3a94a0e: D-431, lambda against the PRIMARY stealth bound, the author's option (A) of D-428 (4);
b3f7bd1: F-087 / D-434, clean_seed a parser default) change no registered prediction (PREDICTIONS unchanged), metric or
population (C_GRID, ATTACK_SEEDS, the category rule, score_predictions unchanged); run_predictions and run_categories
are byte-identical in their bodies. src/a2_p3.py is NOT changed (every record names its bytes); this module:

  reissue   writes results/phase3_a2_categories_d440.json and results/phase3_a2_predictions_d440.json (+ its
            _arrays.npz) with a2_p3's own run_categories / run_predictions under the current code (this module added to
            the code block, which paired.code_block requires of a loaded src module), then compares each with the record
            of 2026-10-09 leaf by leaf and array by array (numbers exactly, NaN = NaN; text leaves reported apart) and
            writes results/phase3_a2_reissue_d440.json. Refuses if any number differs.
  verdicts  results/phase3_a2_verdicts_d440.json: a2_p3.score_predictions and a1m_p3.health_check_tally (unchanged)
            over the 30 registered cells, licensed as D-423 licenses (realign_d423.passing_predictions):
              - the re-issued predictions record: A2's run_kind, its PREDICTIONS text equal to a2_p3.PREDICTIONS, its
                code block equal to THIS process's (every file, this module included);
              - the re-issue record: written by this code block, every number of both re-issued records identical to
                the 2026-10-09 records, and those records still the files it compared (sha256);
              - the 2026-10-09 predictions record's code block equal to this one in every file but src/a2_p3.py, which
                must be OLD_A2_SHA (ad98298), the only file D-438 (3) found different;
              - every score record: its code block equal to this one minus this module, in every file; F-087: its
                config.clean_seed equal to a2_p3.CLEAN_SEED and its clean_repo / clean_rev the attacked model's clean
                reference (a1m_p3.clean_identity of the clean corpus, that seed and step); the train, features, kl and
                arc records of the cell the same clean_seed; its category_record_sha256 the category record's; its ARC
                accuracy its ARC record's (D-432 (2)).
            Anything else refuses. Beside (descriptive, D-432 (2)'s method for A1m): the 30 cells split by FULL stealth
            (KL within the PRIMARY bound AND ARC within the clean arm's range, recomputed from the record and checked
            against its own flags), score_predictions applied unchanged to each subset; and the lane-written
            results/phase3_a2_verdicts.json (the unchecked scorer, lane B's last job) compared with this record's
            scored block.
Run on the node from the repo root: python3 -u src/a2_verdicts_d440.py reissue | verdicts
THIS MODULE IS IN THE CODE BLOCK of the re-issued records: any edit to it between `reissue` and `verdicts` ends the
licence (verdicts refuses: "written by other code"); run reissue first in a node scratch tree (F-081).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a2_p3 as A2  # noqa: E402  (imported, not changed)
from provenance import _RESULT_READS, canonical_hash, ensure_utf8_stdout, provenance_block, sha256_file  # noqa: E402

A = A2.A
P = A2.P
OWN = "src/a2_verdicts_d440.py"
OLD_A2_SHA = "ad98298"            # src/a2_p3.py named by the 2026-10-09 predictions record (35aaa9b), sha256 prefix
OLD = {"categories": A2.CAT_RECORD, "predictions": f"{A2.PRED_NAME}.json"}
NEW = {"categories": "phase3_a2_categories_d440.json", "predictions": f"{A2.PRED_NAME}_d440.json"}
REISSUE = "phase3_a2_reissue_d440.json"
VERDICTS = "phase3_a2_verdicts_d440.json"
LANE_VERDICTS = "phase3_a2_verdicts.json"
SKIP = {"utc", "elapsed_s", "timing", "git", "env", "code", "input_files_sha256", "inputs"}   # TOP-LEVEL keys only
D440 = ("D-440 (3), the author's item 3 of 2026-10-13: the predictions record re-issued under the code that ran; "
        "the A2 scorer checks code hashes as D-423's does; clean_seed checked in every score record (F-087)")
refuse = A2.refuse


def code_files() -> list[str]:
    return sorted(set(A2.CODE_FILES) | {OWN})


def code_now(allow_no_commit: bool = False) -> dict[str, Any]:
    return P.code_block(code_files(), allow_no_commit=allow_no_commit)


def leaves(x: Any, path: str = ""):
    if isinstance(x, dict):
        if not x:
            yield path, "{}"
        for k in sorted(x):
            if path or k not in SKIP:              # SKIP applies at the top level only (review of c8306d2)
                yield from leaves(x[k], f"{path}/{k}")
    elif isinstance(x, list):
        if not x:
            yield path, "[]"
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, x


def same_number(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return type(a) is type(b) and a == b


def compare(old_p: Path, new_p: Path, npz_key: str | None) -> dict[str, Any]:
    """Leaf by leaf (numbers exactly; strings apart) and, when the record names an npz, array by array."""
    jo, jn = json.loads(old_p.read_text(encoding="utf-8")), json.loads(new_p.read_text(encoding="utf-8"))
    lo, ln = dict(leaves(jo)), dict(leaves(jn))
    num = [k for k in sorted(set(lo) & set(ln)) if not isinstance(lo[k], str) or not isinstance(ln[k], str)]
    txt = [k for k in sorted(set(lo) & set(ln)) if isinstance(lo[k], str) and isinstance(ln[k], str)]
    out: dict[str, Any] = {
        "old": {"file": A2.rel(old_p), "sha256": sha256_file(old_p)},
        "new": {"file": A2.rel(new_p), "sha256": sha256_file(new_p)},
        "paths_only_old": sorted(set(lo) - set(ln)), "paths_only_new": sorted(set(ln) - set(lo)),
        "n_numeric_leaves": len(num), "numeric_differ": [k for k in num if not same_number(lo[k], ln[k])],
        "n_text_leaves": len(txt), "text_differ": [k for k in txt if lo[k] != ln[k]]}
    if npz_key:
        zo = np.load(old_p.parent / Path(jo[npz_key]["file"]).name, allow_pickle=False)
        zn = np.load(new_p.parent / Path(jn[npz_key]["file"]).name, allow_pickle=False)
        ko, kn = set(zo.files), set(zn.files)
        diff = [k for k in sorted(ko & kn) if zo[k].shape != zn[k].shape or zo[k].dtype != zn[k].dtype
                or not np.array_equal(zo[k], zn[k], equal_nan=zo[k].dtype.kind in "fc")]
        out["npz"] = {"arrays": len(ko & kn), "values": int(sum(zo[k].size for k in ko & kn)),
                      "only_old": sorted(ko - kn), "only_new": sorted(kn - ko), "differ": diff}
    # path sets: the re-issued record may name inputs the old did not (e.g. its own name); those are text or paths
    out["identical_numbers"] = (not out["numeric_differ"] and not out["paths_only_old"]
                                and all(not isinstance(ln[k], (int, float)) for k in out["paths_only_new"])
                                and (not npz_key or (not out["npz"]["differ"] and not out["npz"]["only_old"]
                                                     and not out["npz"]["only_new"])))
    return out


def run_reissue(*, root: Path = A2.ROOT, results_dir: Path = A2.RESULTS, allow_no_commit: bool = False) -> Path:
    results_dir = Path(results_dir)
    out_p = results_dir / REISSUE
    # every file the re-issue writes, the predictions npz included (review of 991f73d): a partial failure leaves files
    # that make a rerun refuse here; they are moved aside by hand (never deleted) before a rerun
    npz = f"{Path(NEW['predictions']).stem}{A.ARRAYS_SUFFIX}.npz"
    for f in (*NEW.values(), npz, REISSUE):
        if os.path.lexists(results_dir / f):
            refuse(f"{f} exists; the re-issue is written once")
    code0 = code_now(allow_no_commit)
    # a2_p3's writers hash a2_p3.CODE_FILES (and a1m_p3's helpers a1m_p3.CODE_FILES, F-086) at call time; this module
    # is loaded, so both lists gain it for the re-issue (paired.code_block refuses a loaded src module left out)
    saved = (A2.CODE_FILES, A.CODE_FILES, A2.CAT_RECORD, A2.PRED_NAME)
    try:
        A2.CODE_FILES, A.CODE_FILES = code_files(), code_files()
        A2.CAT_RECORD = NEW["categories"]
        A2.run_categories(root=root, results_dir=results_dir, allow_no_commit=allow_no_commit)
        A2.CAT_RECORD = OLD["categories"]                  # the predictions read the category record of record
        A2.PRED_NAME = Path(NEW["predictions"]).stem
        A2.run_predictions(root=root, results_dir=results_dir, allow_no_commit=allow_no_commit)
    finally:
        A2.CODE_FILES, A.CODE_FILES, A2.CAT_RECORD, A2.PRED_NAME = saved
    cmp = {"categories": compare(results_dir / OLD["categories"], results_dir / NEW["categories"], None),
           "predictions": compare(results_dir / OLD["predictions"], results_dir / NEW["predictions"], "npz")}
    if code_now(allow_no_commit)["sha256"] != code0["sha256"]:
        refuse("the code changed during the re-issue; nothing more is written")
    payload = {**provenance_block("phase3_a2_reissue_d440"), "decision": D440, "code": code0,
               "a2_p3_changes_since_the_predictions_record": CHANGES, "comparisons": cmp,
               "identical_numbers": all(v["identical_numbers"] for v in cmp.values())}
    payload["input_files_sha256"] = dict(_RESULT_READS)
    A._write(out_p, payload)
    for k, v in cmp.items():
        print(f"  {k}: numbers {v['n_numeric_leaves']} (differ {len(v['numeric_differ'])}), text differ "
              f"{v['text_differ']}, npz {v.get('npz', {}).get('values')} values (differ "
              f"{len(v.get('npz', {}).get('differ', []))})", flush=True)
    if not payload["identical_numbers"]:
        refuse("a re-issued record differs from the 2026-10-09 record in a number; see " + REISSUE)
    return out_p


CHANGES = [
    {"commit": "045de0e", "date": "2026-10-10", "entry": "F-086 (D-428)",
     "change": "a1m_p3.CODE_FILES set to A2's list at import (A2's records name A2's code)", "touches": "code block"},
    {"commit": "8c5b23a", "date": "2026-10-10", "entry": "F-086 review", "change": "comments, a test path",
     "touches": "nothing scored"},
    {"commit": "0d56959", "date": "2026-10-10", "entry": "D-431 (the author's option (A) of D-428 (4))",
     "change": "lambda selected against the PRIMARY stealth bound (LAMBDA_RULE, lambda_bound, run_lambda); the "
               "sandbox job; nas-check; PREREG names D-431", "touches": "the attacker's lambda (the author's decision)"},
    {"commit": "3a94a0e", "date": "2026-10-10", "entry": "D-431 review", "change": "choice (t) wording; boundary test",
     "touches": "implementation_choices text"},
    {"commit": "b3f7bd1", "date": "2026-10-10", "entry": "F-087 / D-434",
     "change": "clean_seed = CLEAN_SEED as a parser default", "touches": "argument parsing (value 0, as D-389 (2))"},
]


def registered_text_ok(preds: dict[str, Any]) -> bool:
    """a2_p3.run_predictions stores {name: {"text": PREDICTIONS[name], "verdict": "NOT YET RUN ..."}}."""
    got = preds.get("predictions")
    if not isinstance(got, dict) or set(got) != set(A2.PREDICTIONS):
        return False
    return all(isinstance(v, dict) and v.get("text") == A2.PREDICTIONS[k]
               and str(v.get("verdict", "")).startswith("NOT YET RUN") for k, v in got.items())


def _code_equal_but(a: dict[str, str], b: dict[str, str], but: set[str]) -> list[str]:
    return sorted(f for f in set(a) | set(b) if f not in but and a.get(f) != b.get(f))


def run_verdicts(*, root: Path = A2.ROOT, results_dir: Path = A2.RESULTS, allow_no_commit: bool = False) -> Path:
    results_dir = Path(results_dir)
    out_p = results_dir / VERDICTS
    if os.path.lexists(out_p):
        refuse(f"{VERDICTS} exists; results are never overwritten")
    code0 = code_now(allow_no_commit)
    now = code0["sha256"]
    run_code = {k: v for k, v in now.items() if k != OWN}
    # (1) the re-issue record and the two comparisons, re-done here
    ri, ri_block = A.read_reg(results_dir / REISSUE)
    if ri.get("run_kind") != "phase3_a2_reissue_d440" or (ri.get("code") or {}).get("sha256") != now:
        refuse(f"{REISSUE} is not a re-issue record written by this code block")
    cmp = {"categories": compare(results_dir / OLD["categories"], results_dir / NEW["categories"], None),
           "predictions": compare(results_dir / OLD["predictions"], results_dir / NEW["predictions"], "npz")}
    for k, v in cmp.items():
        if not v["identical_numbers"]:
            refuse(f"the re-issued {k} record differs from the 2026-10-09 record in a number")
        if v["old"]["sha256"] != ri["comparisons"][k]["old"]["sha256"] or \
                v["new"]["sha256"] != ri["comparisons"][k]["new"]["sha256"]:
            refuse(f"the {k} records are not the files {REISSUE} compared")
    # (2) the re-issued predictions record passes (D-423's passing_predictions)
    preds, preds_block = A.read_reg(results_dir / NEW["predictions"])
    problems = []
    if preds.get("run_kind") != "phase3_a2_predictions":
        problems.append("not an A2 predictions record")
    if not registered_text_ok(preds):
        problems.append("its registered text is not a2_p3.PREDICTIONS (or a verdict is filled in)")
    if (preds.get("code") or {}).get("sha256") != now:
        problems.append("it was written by other code")
    if problems:
        refuse(f"{NEW['predictions']} cannot license the verdicts: {problems}")
    old_preds, old_block = A.read_reg(results_dir / OLD["predictions"])
    oc = (old_preds.get("code") or {}).get("sha256") or {}
    bad = _code_equal_but(oc, run_code, {"src/a2_p3.py"})
    if bad or not str(oc.get("src/a2_p3.py", "")).startswith(OLD_A2_SHA):
        refuse(f"the 2026-10-09 predictions record's code differs beyond src/a2_p3.py {OLD_A2_SHA}: {bad}")
    # (3) categories (a2_p3's own re-derivation) and the 30 cells
    cats, cat_block = A2.categories_of(results_dir)
    cells = A2.grid([c["category"] for c in cats])
    clean = A.clean_identity(A.CLEAN_CORPUS, A2.CLEAN_SEED, A.CLEAN_STEP, A2.ROOT)
    recs, blocks, seed_checks, arcs = [], [], [], []
    for cell in cells:
        rec, block = A.read_reg(results_dir / f"phase3_{A2.LABEL}_score__{cell['tag']}.json")
        cfg = rec.get("config") or {}
        diff = _code_equal_but((rec.get("code") or {}).get("sha256") or {}, run_code, set())
        if diff:
            refuse(f"score record {cell['tag']}: its code block differs from the code that runs: {diff}")
        side, side_recs = {}, {}
        for kind in ("train", "features", "kl", "arc"):
            r2, _ = A.read_reg(results_dir / f"phase3_{A2.LABEL}_{kind}__{cell['tag']}.json")
            side[kind], side_recs[kind] = (r2.get("config") or {}).get("clean_seed"), r2
        ok = {"clean_seed": cfg.get("clean_seed") == A2.CLEAN_SEED,
              "clean_repo": cfg.get("clean_repo") == clean["repo"],
              "clean_rev": cfg.get("clean_rev") == clean["revision"],
              **{f"{k}_clean_seed": side[k] == A2.CLEAN_SEED for k in side},
              "category_record": cfg.get("category_record_sha256") == cat_block["sha256"]}
        if not all(ok.values()):
            refuse(f"score record {cell['tag']}: F-087 / category check failed: {ok}")
        seed_checks.append({"tag": cell["tag"], "clean_seed": cfg.get("clean_seed"), **ok})
        recs.append(rec)
        blocks.append(block)
        arcs.append(side_recs["arc"])
    scored = A2.score_predictions(cells, recs, preds, cats)
    health = A.health_check_tally(cells, recs, tpr_of=A2.category_tpr)
    # (4) beside: FULL stealth (D-432 (2)'s method), recomputed and checked against each record's own flags
    stealth = []
    for cell, rec, arc_rec in zip(cells, recs, arcs):
        st = rec["stealth"]
        pr = st["primary"]
        if A.arc_accuracy(arc_rec) != pr["arc_accuracy"]:
            refuse(f"{cell['tag']}: the score record's ARC accuracy is not its ARC record's (D-432 (2)'s check)")
        lo, hi = pr["arc_range_clean_arm"]
        kl_w, arc_w = bool(pr["kl_attacked_clean"] <= pr["bound"]), bool(lo <= pr["arc_accuracy"] <= hi)
        if (kl_w, arc_w, kl_w and arc_w, kl_w and arc_w) != (pr["kl_within"], pr["arc_within"], pr["stealthy"],
                                                              st["stealthy"]):
            refuse(f"{cell['tag']}: recomputed stealth differs from the record's flags")
        stealth.append({"tag": cell["tag"], "category": cell["category"], "c": cell["c"],
                        "attack_seed": cell["attack_seed"], "kl_attacked_clean": pr["kl_attacked_clean"],
                        "kl_bound": pr["bound"], "kl_within": kl_w, "arc_accuracy": pr["arc_accuracy"],
                        "arc_range_clean_arm": [lo, hi], "arc_within": arc_w, "stealthy": kl_w and arc_w})
    split: dict[str, Any] = {"note": ("DESCRIPTIVE. A subset may hold no c = SUCCESS_C cell of a category; "
                                      "score_predictions then reads P-A2.3 'falls in every cell' as False for that "
                                      "category (bool(ok) and all(ok)): an artefact of the subset, not a result")}
    for name, want in (("full_stealth", True), ("not_full_stealth", False)):
        idx = [i for i, s in enumerate(stealth) if s["stealthy"] is want]
        sub = A2.score_predictions([cells[i] for i in idx], [recs[i] for i in idx], preds, cats) if idx else None
        split[name] = {"n_cells": len(idx), "tags": [cells[i]["tag"] for i in idx],
                       "by_prediction": None if sub is None else {
                           k: {"verdict": sub[k]["verdict"], **{f: sub[k].get(f) for f in (
                               "share_within", "share_firing", "accuracy_line", "per_category") if f in sub[k]}}
                           for k in A2.PREDICTIONS},
                       "a2_overall": None if sub is None else sub["a2_overall"]["per_category_at_c"]}
    # (5) the lane-written verdicts record (the unchecked scorer) beside
    lane = results_dir / LANE_VERDICTS
    lane_cmp = None
    if lane.is_file():
        lv, lb = A.read_reg(lane)
        lo_, ln_ = dict(leaves(lv.get("scored"))), dict(leaves(scored))
        lane_cmp = {"file": lb["file"], "sha256": lb["sha256"],
                    "scored_paths_differ": sorted(k for k in set(lo_) | set(ln_) if not same_number(lo_.get(k),
                                                                                                    ln_.get(k))),
                    "health_equal": canonical_hash(lv.get("health_checks_d417a")) == canonical_hash(health)}
    cfg = {"n_cells": len(cells), "categories": [c["category"] for c in cats], "clean": clean,
           "clean_seed": A2.CLEAN_SEED}
    payload = {**provenance_block("phase3_a2_verdicts_d440"), "decision": D440, "preregistration": A2.PREREG,
               "code": code0, "config": cfg, "config_hash": canonical_hash(cfg),
               "licence": {"rule": ("this module's docstring, mode 'verdicts': D-423's passing_predictions, the "
                                    "re-issue comparison, every code block, F-087's clean_seed"),
                           "reissue_record": ri_block, "comparisons": cmp,
                           "old_predictions_record": old_block, "old_a2_p3_sha256": oc.get("src/a2_p3.py"),
                           "a2_p3_changes_since_the_predictions_record": CHANGES},
               "category_record": cat_block, "predictions_record": preds_block, "score_records": blocks,
               "clean_seed_checks_f087": seed_checks, "scored": scored, "health_checks_d417a": health,
               "full_stealth_split_beside": {"method": "D-432 (2) for A1m: KL(attacked||clean) <= the PRIMARY bound "
                                                       "AND ARC within the clean arm's seed range; score_predictions "
                                                       "unchanged on each subset; DESCRIPTIVE",
                                             "cells": stealth, **split},
               "lane_verdicts_record_beside": lane_cmp}
    payload["input_files_sha256"] = {k: v for k, v in _RESULT_READS.items()}
    if code_now(allow_no_commit)["sha256"] != now:
        refuse("the code changed during the run; nothing is written")
    A._write(out_p, payload)
    for k in A2.PREDICTIONS:
        print(f"  {k}: {scored[k]['verdict']}", flush=True)
    print(f"  {scored['a2_overall']['verdict']}; full stealth {split['full_stealth']['n_cells']} of {len(cells)}; "
          f"lane record scored paths differing: "
          f"{None if lane_cmp is None else len(lane_cmp['scored_paths_differ'])}", flush=True)
    return out_p


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["reissue", "verdicts"])
    args = ap.parse_args(argv)
    p = run_reissue() if args.cmd == "reissue" else run_verdicts()
    print(f"Written: {A2.rel(p)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
