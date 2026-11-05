"""Phase 3 attacker A1 (D-327): the POOLED scoring of P-A1.3 and P-A1.4 over the nine per-pair records, as DECISIONS.md
D-346 (4) pre-registers it (quoted whole in PREREG_TEXT and in the record). No model, cache or prediction is computed.

INPUTS (36 files, each hashed and listed in input_files_sha256, D-297 (1)): for each of the nine released pairs of
D-327 (1) (populations.UPDATE_PAIRS_REPAIR), results/phase3_a1_<pair>.json and its _draws.npz (src/phase3_a1.py), and
the predictions record that record names, results/phase3_a1_predictions_<pair>.json and its _draws.npz
(src/phase3_a1_predictions.py). REFUSES unless all nine exist and no other pair's record is there; each npz is the bytes
(sha256, size, keys) its record names, read once as bytes, hashed and loaded from those bytes; each predictions record
is the file and the bytes (JSON and npz sha256) its A1 record names, read as a registered input (D-297), the
registered predictions run of the same pair, draws, seeds, target, c grid and variants, made by the shared
predictions code; each A1 record is a registered run of its pair that passed its own self-anchor (anchor checked and
identical, no mismatch, D-327 (6)) and protocol stops (P-A1.1, P-A1.5, P-A1.6 and the linear-identity check hold,
D-327 (5)) and stores its D-170 flag; the nine share one code block for the A1 scripts and so name one
predictions-code sha256; src/phase3_a1.py and src/phase3_a1_predictions.py are the reviewed blobs (REVIEWED_SHA256: the
files at 6647ac2, unchanged at fa1b4a0) in the records and in this tree; the variants are white-box everywhere and
transfer exactly on D-346 (2)(d)'s five pairs; the c grid, draws, seeds and target agree.

P-A1.3 (D-327 (4), D-346 (4)): one cell per pair x c != 0, per variant: white-box over the nine pairs, transfer over its
  five pairs as a separate line. REALISED: the D-156 mean over the draws of the frozen arm's per-draw FPR/target (the seed
  mean of the floored per-seed FPRs, over the target), recomputed from the A1 npz and required equal to the record's
  cell. PREDICTED: re-derived from the predictions npz exactly as phase3_a1.score derives it, the D-156 mean over draws
  1..200 of the per-draw seed-mean predicted FPR/target, unfloored (primary) and floored per seed at 1/|E| (beside,
  |E| = 1000), and required equal to the A1 record's per-cell rows and cells and to the predictions record's own
  'predicted' block. D-177's accuracy line by phase3_a1.accuracy: the share of cells within x1.5 and the median
  |log(realised/predicted)|, n cells stated, a zero prediction a miss and counted (D-322 (4)); D-327's expectation: at
  least 80% of cells within x1.5. THE VERDICT LINE (P-A1.3.scored) is white-box, unfloored (D-318 (9), D-346 (2)(e));
  the other five lines (white-box floored; transfer unfloored and floored; both variants together unfloored and
  floored) are labelled 'beside'.
P-A1.4: per draw, the 4c Gaussian health flag at 2 / 0.5 against the realised class (realised paired ratio >= 2 or
  <= 0.5) over each pair's attacked cells (phase3_a1.confusion on the npz arrays, required equal to the record's own
  P-A1.4 counts), pooled over the nine pairs, white-box; sensitivity and specificity by D-255's estimator
  (healthcheck_summary_d156.pooled_counts and pair_cluster_index: the per-pair D-156 mean counts with the pair-cluster
  bootstrap, B = 5000, seed 520260925; the per-draw D-156 mean and band beside); thresholds sensitivity >= 0.90 and
  specificity >= 0.85 (D-327); holds is None, with the reason, if a pooled rate has no cell of its class. Transfer
  beside over its five pairs, not scored.
D-170: every pooled line carries the per-pair D-170 flag the A1 records store (input_verification.flag) and its number
  of flagged cells (D-331: D-168 is final and its pinned record clears the flag of the judged pin-only inputs).
RELAUNCHING THE A1 LIST: this scorer requires the nine records to share one code block for the A1 scripts (choice (f),
  strict, kept). So queues/laneC_20261003a_a1.tsv may be relaunched (for the pairs not yet written) only at a
  SYNC_COMMIT whose tree manifest gives fa1b4a0's sha256 for every file of the A1 code block (phase3_a1.CODE_FILES):
  src/phase3_a1.py, src/healthcheck_d156.py, src/phase3_a1_predictions.py, src/poolstudy_d156.py,
  src/pool_propositions.py, src/depth_d156.py, src/input_registry.py, src/paired.py, src/populations.py,
  src/provenance.py, src/superseded_provenance.py, src/metrics.py, src/repdist.py (and with the same library
  versions); records made under any other block make this scorer refuse.
CHOICES BY THE CLOSEST D-ENTRY (D-181): CHOICES below, written into the record.
Writes results/phase3_a1_pooled.json once (exclusive), then prints the pooled lines. A test reads --records-dir and
writes --out, outside results/.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import healthcheck_summary_d156 as HS  # noqa: E402
import paired as P  # noqa: E402
import phase3_a1 as A1  # noqa: E402
import phase3_a1_predictions as A1P  # noqa: E402
import populations as POP  # noqa: E402
from provenance import (  # noqa: E402
    canonical_hash, ensure_utf8_stdout, provenance_block, read_result, sha256_bytes, sha256_file, write_results,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FINAL_OUT = RESULTS / "phase3_a1_pooled.json"
THIS_FILE = "src/phase3_a1_pooled.py"
CODE_FILES = list(dict.fromkeys([THIS_FILE, "src/healthcheck_summary_d156.py", *A1.CODE_FILES]))
PAIRS = [A1P.core_of(p) for p in POP.UPDATE_PAIRS_REPAIR]                     # D-327 (1): the nine released pairs
TRANSFER_PAIRS = ("mistral_base2instruct", "llama3_base2instruct", "llama3_to_31", "WG_mistral_base2instruct",
                  "WG_llama3_base2instruct")                                   # D-346 (2)(d)
PRIMARY = "white_box"                                                          # D-318 (9), D-346 (2)(g)
# the reviewed blobs whose functions this scorer calls (sha256 of the file bytes): the files at 6647ac2 (the A1 merge),
# unchanged at fa1b4a0, and the ones the pulled A1 records name
REVIEWED_SHA256 = {"src/phase3_a1.py": "8531cfb84a851b316ebd5b5b80af92d3b8f5ef10973020ef2f5ee05af557591f",
                   "src/phase3_a1_predictions.py": "44de1ef70014ebbfb3b580971ed0d90590d7c3d7e793671328a4c1aaba5a15ab"}
REUSED = tuple(REVIEWED_SHA256)
PROTOCOL = ("P-A1.1", "P-A1.5", "P-A1.6", "linear_identity")                   # D-327 (5) and the D-181 identity check
EXPECTED_SHARE = 0.80                                                          # D-327 (4) P-A1.3
SENS_MIN, SPEC_MIN = 0.90, 0.85                                                # D-327 (4) P-A1.4
# |E|, whose 1/|E| floors the beside prediction per seed (phase3_a1.score's n_e): D-327 (1) "E the 1000 evaluation
# negatives outside R_fresh"; phase3_a1_predictions.build refuses an evaluation part that is not 2000 negatives
# (R_FRESH 1000). A wrong value cannot pass: the floored values must equal both records'.
N_E = 1000
DETECTORS = HS.DETECTORS                                                       # gaussian, direct, auc_rule
FLAG_KEYS = ("realised_class_2x", "health|gaussian_flag", "health|direct_flag", "auc_rule|flag")
NZ = [A1P.ckey(c) for c in A1P.C_GRID if c != 0]
PRED_SAME = ("pair", "n_reps", "seeds", "target_fpr", "c_grid", "variants", "draws_index_sha256")
LINES13 = (("primary_unfloored", "predicted_ratio_to_target"), ("beside_floored_1_over_E", "predicted_ratio_to_target_floored"))
VERDICT13 = "scored: the pre-registered verdict line (white-box, unfloored; D-318 (9), D-346 (2)(e))"
BESIDE13 = ["lines.white_box.beside_floored_1_over_E", "lines.transfer.primary_unfloored",
            "lines.transfer.beside_floored_1_over_E", "beside_both_variants_together.primary_unfloored",
            "beside_both_variants_together.beside_floored_1_over_E"]
CHOICE_C = ("choice (c): 'every c' (D-346 (4)) is read as every attacked c, c != 0 (D-327 (4)), with c = 0 beside; the "
            "per-draw unit is the seed-mean flag (4c's per-draw statistics, D-241 (2)), as the records hold no per-seed flag")
D170_RULE = ("D-170/D-172: a result with a flag enters no paper table until D-168 has passed; D-331: D-168 is final "
             "(results/d168_final_verdicts.json, pinned in src/input_registry.py), so the flag clears for the pin-only "
             "inputs among the 17 caches it judged")
PREREG = "D-346 (4) (the pooled scoring of D-327 (4) P-A1.3 and P-A1.4); D-327; D-177; D-241 (6); D-255; D-322 (4)"
PREREG_TEXT = (
    "(4) POOLED SCORING, PRE-REGISTERED NOW (a separate script, three reviewers, written before the pooled numbers are "
    "read; the per-pair lane logs print per-pair checks only): P-A1.3: one cell per pair x c != 0, white-box over the "
    "nine pairs and transfer over its five pairs as a separate line; per cell, realised = the D-156 mean over draws and "
    "seeds of the per-draw FPR/target, predicted = the D-156 mean of the per-draw predicted FPR/target (unfloored "
    "primary, floored beside); D-177's accuracy line: share of cells within x1.5 and median |log(realised/predicted)|, "
    "n cells stated, zero predictions as misses; expectation as D-327: at least 80% within x1.5. P-A1.4: per draw and "
    "seed, the 4c Gaussian health flag at 2 / 0.5 against the realised class (realised paired ratio >= 2 or <= 0.5), "
    "pooled over the nine pairs and every c (white-box), sensitivity and specificity with the pair-cluster bootstrap of "
    "D-241 (6) (B = 5000, seed 520260925), by the estimator code of the D-255 pooled health-check summary applied "
    "unchanged; thresholds as D-327 (sensitivity >= 0.90, specificity >= 0.85).")
CONFIG = {
    "pairs": PAIRS, "transfer_pairs": list(TRANSFER_PAIRS), "c_grid": list(A1P.C_GRID), "c_scored": NZ,
    "variants": {"P-A1.3_verdict": PRIMARY, "P-A1.3_beside": "transfer", "P-A1.4_scored": PRIMARY,
                 "P-A1.4_beside": "transfer"},
    "P-A1.3": {"expectation_share_within_factor": EXPECTED_SHARE, "factor": A1.FACTOR, "verdict_prediction": "unfloored",
               "beside_floor": "per seed at 1/|E|", "n_e": N_E},
    "P-A1.4": {"thresholds": {"sensitivity": SENS_MIN, "specificity": SPEC_MIN}, "applied_to": "the pooled point estimates",
               "detector": "gaussian", "detectors_beside": [d for d in DETECTORS if d != "gaussian"],
               "class_and_flag_cut": list(A1.CUT)},
    "bootstrap": {"B": HS.B, "seed": HS.SEED, "min_pairs": HS.MIN_PAIRS,
                  "kind": "pair-cluster on the per-pair D-156 mean counts (D-241 (6), D-255)"},
}
CHOICES = [
    "(a) P-A1.3's verdict line is white-box, unfloored (P-A1.3.scored: white-box primary, D-318 (9); the unfloored "
    "prediction primary, D-346 (2)(e)). The expectation (at least 80% within x1.5) is also tested, beside, on the "
    "white-box floored line, on the transfer line (D-346 (4)'s separate line, unfloored and floored) and on D-327 (4)'s "
    "literal cell set (pair x c != 0 x variant, both variants together, unfloored and floored); each line names its role.",
    "(b) P-A1.3's predicted values are re-derived from each pair's predictions npz (the predictions record its A1 record "
    "names, by file and sha256, the JSON read as a registered input, D-297) exactly as phase3_a1.score derives them: per "
    "draw the seed mean of the per-seed predicted FPR over the target, unfloored, and floored per seed at 1/|E| (|E| = "
    "1000, D-327 (1)), then the D-156 mean over draws 1..200. They must equal the A1 record's per-cell rows and cells and "
    "the predictions record's own 'predicted' block, else the scorer refuses. The realised values are recomputed from "
    "each A1 npz and must equal the record's.",
    "(c) 'Every c' in D-346 (4) is read as every attacked c, c != 0 (D-327 (4): 'over (pair, c != 0, draw)'); the "
    "white-box line with c = 0 included is beside, not scored. P-A1.4's unit is (pair, c != 0, draw): 'per draw and "
    "seed' is read as 4c's per-draw statistics, each the mean over seeds of per-seed ratios (D-241 (2)), so the per-draw "
    "unit is the seed-mean flag; the records hold no per-seed flag.",
    "(d) The thresholds apply to the pooled point estimates from the per-pair D-156 mean counts (the values D-255 "
    "quotes, 0.918 and 0.898); the pair-cluster interval is beside. A pooled rate with no cell of its class is "
    "undefined: holds is then None with the reason, not False.",
    "(e) The direct reference ratio and the AUC rule are pooled beside by the same estimator, not scored (D-255 reports "
    "all three).",
    "(f) 'One code block': the nine records' code sha256 maps and library versions must be identical (strict); their "
    "commits may differ (a push between two jobs moves SYNC_COMMIT without changing a hashed file) and are listed. The "
    "scorer refuses unless src/phase3_a1.py and src/phase3_a1_predictions.py in the records and in its own tree are the "
    "reviewed blobs (REVIEWED_SHA256: the files at 6647ac2, unchanged at fa1b4a0; it calls their accuracy line, "
    "confusion counts and grid); any other A1 code file that differs from this tree is listed. A relaunch of the A1 list "
    "must therefore keep fa1b4a0's code block (module docstring).",
    "(g) The transfer line's pairs must be D-346 (2)(d)'s five; the c grid, draws, seeds and target must agree across "
    "the nine.",
    "(h) D-170: each pooled line carries the per-pair flag the A1 records store (input_verification.flag) and its "
    "number of flagged cells; D-331 made D-168 final and its pinned record clears the flag of the judged pin-only "
    "inputs. No flagged or unflagged sub-line is formed (none is pre-registered).",
]


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


# ---- the inputs and the refusals ---------------------------------------------------------------------------------

def discover(rdir: Path) -> dict[str, Path]:
    """The per-pair A1 records in rdir: exactly the nine pairs (a missing or an extra pair refuses)."""
    found: dict[str, Path] = {}
    for p in sorted(rdir.glob("phase3_a1_*.json")):
        core = p.stem[len("phase3_a1_"):]
        if not core.startswith(("predictions_", "pooled")):
            found[core] = p
    missing, extra = [c for c in PAIRS if c not in found], sorted(set(found) - set(PAIRS))
    if missing or extra:
        refuse(f"the A1 records in {rdir} are not the nine pairs of D-327 (1): missing {missing}, extra {extra}")
    return {c: found[c] for c in PAIRS}


def read_npz(npz_path: Path, block: dict[str, Any], owner: str) -> tuple[str, int, dict[str, np.ndarray]]:
    """The npz `block` names ({file, sha256, size, keys}): read once as bytes, hashed, loaded from those bytes."""
    if not npz_path.is_file():
        refuse(f"{npz_path.name} is missing")
    nb = npz_path.read_bytes()
    h = sha256_bytes(nb)
    if h != block.get("sha256") or len(nb) != block.get("size"):
        refuse(f"{npz_path.name} is not the bytes {owner} names")
    with np.load(io.BytesIO(nb), allow_pickle=False) as zf:
        z = {k: zf[k] for k in zf.files}
    if sorted(z) != sorted(block.get("keys") or []):
        refuse(f"{npz_path.name} does not hold the keys {owner} lists")
    return h, len(nb), z


def load(core: str, path: Path, files: dict[str, tuple[Path, str]]) -> dict[str, Any]:
    """One per-pair record (a registered read, D-297 (1)) and its npz, loaded from the very bytes whose sha256 is
    checked against the record; then the predictions record it names (load_predictions)."""
    doc = read_result(path)
    sha = sha256_file(path)
    npz_path = path.with_name(path.stem + "_draws.npz")
    dz = doc.get("draws_npz") or {}
    if dz.get("file") != A1P.rel(npz_path):
        refuse(f"{path.name} names the npz {dz.get('file')!r}, not {A1P.rel(npz_path)!r}")
    nsha, nsize, z = read_npz(npz_path, dz, path.name)
    files[A1P.rel(path)] = (path, sha)
    files[A1P.rel(npz_path)] = (npz_path, nsha)
    return {"core": core, "doc": doc, "z": z, "file": A1P.rel(path), "sha256": sha,
            "draws_npz": {"file": A1P.rel(npz_path), "sha256": nsha, "size": nsize},
            "pred": load_predictions(core, path, doc, files)}


def load_predictions(core: str, path: Path, doc: dict[str, Any], files: dict[str, tuple[Path, str]]) -> dict[str, Any]:
    """The predictions record the A1 record at `path` names (beside it in the records directory): its JSON (a
    registered read, D-297) with the sha256 the A1 record gives, and its npz with the sha256 both records give."""
    ppath = path.with_name(f"phase3_a1_predictions_{core}.json")
    pnpz = ppath.with_name(ppath.stem + "_draws.npz")
    prec = doc.get("predictions_record") or {}
    prz = prec.get("draws_npz") or {}
    if prec.get("file") != A1P.rel(ppath) or prz.get("file") != A1P.rel(pnpz):
        refuse(f"{path.name} names the predictions record {prec.get('file')!r} and npz {prz.get('file')!r}, not "
               f"{A1P.rel(ppath)!r} and {A1P.rel(pnpz)!r}")
    if not ppath.is_file():
        refuse(f"{ppath.name}, the predictions record {path.name} names, is missing")
    pdoc = read_result(ppath)
    psha = sha256_file(ppath)
    if psha != prec.get("sha256"):
        refuse(f"{ppath.name} is not the bytes {path.name} names")
    pdz = pdoc.get("draws_npz") or {}
    if pdz.get("file") != A1P.rel(pnpz):
        refuse(f"{ppath.name} names the npz {pdz.get('file')!r}, not {A1P.rel(pnpz)!r}")
    nsha, nsize, pz = read_npz(pnpz, pdz, ppath.name)
    if nsha != prz.get("sha256"):
        refuse(f"{pnpz.name} is not the bytes {path.name} names")
    files[A1P.rel(ppath)] = (ppath, psha)
    files[A1P.rel(pnpz)] = (pnpz, nsha)
    return {"doc": pdoc, "z": pz, "file": A1P.rel(ppath), "sha256": psha,
            "draws_npz": {"file": A1P.rel(pnpz), "sha256": nsha, "size": nsize}}


def record_problems(core: str, doc: dict[str, Any]) -> list[str]:
    """Why one record cannot enter the pooled scoring (empty when it can)."""
    cfg = doc.get("config") or {}
    bad = []
    if doc.get("run_kind") != "phase3_a1" or cfg.get("pair") != f"repair_{core}":
        bad.append(f"not the A1 record of repair_{core} (run_kind {doc.get('run_kind')!r}, pair {cfg.get('pair')!r})")
    if doc.get("registered_parameters") is not True or cfg.get("anchor_report_only") is not False:
        bad.append("not a registered run, or written with --anchor-report-only")
    an = doc.get("anchor") or {}
    n_cmp = an.get("n_values_compared")
    if not (an.get("checked") is True and an.get("identical") is True and an.get("mismatches") == []
            and isinstance(n_cmp, int) and n_cmp > 0 and an.get("n_values_identical") == n_cmp):
        bad.append("did not pass its self-anchor (D-327 (6))")
    sc = doc.get("scoring") or {}
    failed = [k for k in PROTOCOL if (sc.get(k) or {}).get("holds") is not True]
    if failed:
        bad.append(f"did not pass its protocol stops: {failed}")
    want = ["white_box", "transfer"] if core in TRANSFER_PAIRS else ["white_box"]
    if cfg.get("variants") != want:
        bad.append(f"variants {cfg.get('variants')}, not {want} (D-346 (2)(d))")
    if cfg.get("c_grid") != list(A1P.C_GRID):
        bad.append(f"c grid {cfg.get('c_grid')}")
    if "flag" not in (doc.get("input_verification") or {}):
        bad.append("stores no D-170 flag (input_verification.flag)")
    return bad


def shared_code(prs: dict[str, dict[str, Any]], scorer_sha: dict[str, str]) -> dict[str, Any]:
    """The nine records' one code block for the A1 scripts (choice (f)), its predictions-code sha256, and this
    scorer's agreement with the files whose functions it calls, which must be the reviewed blobs."""
    codes = {c: pr["doc"].get("code") or {} for c, pr in prs.items()}
    ref = codes[PAIRS[0]]
    off = [c for c, cb in codes.items() if cb.get("sha256") != ref.get("sha256") or cb.get("libraries") != ref.get("libraries")]
    if off or not isinstance(ref.get("sha256"), dict):
        refuse(f"the nine records do not share one code block for the A1 scripts: {off} differ from {PAIRS[0]}")
    sha = ref["sha256"]
    pred = sha.get(A1P.THIS_FILE)
    if not pred or any(cb["sha256"].get(A1P.THIS_FILE) != pred for cb in codes.values()):
        refuse("the nine records do not name one predictions-code sha256")
    mine = [f for f in REUSED if not sha.get(f) or scorer_sha.get(f) != sha.get(f)]
    if mine:
        refuse(f"this tree's {mine} are not the records' (the scorer calls their functions)")
    unreviewed = [f for f in REUSED if sha.get(f) != REVIEWED_SHA256[f]]
    if unreviewed:
        refuse(f"the records' and this tree's {unreviewed} are not the reviewed blobs (REVIEWED_SHA256, the files at 6647ac2)")
    return {"sha256": sha, "libraries": ref.get("libraries"), "predictions_code_sha256": pred,
            "reviewed_sha256": REVIEWED_SHA256, "commits": sorted({str(cb.get("commit")) for cb in codes.values()}),
            "a1_files_differing_from_this_tree": sorted(f for f in sha if scorer_sha.get(f) != sha[f])}


def prediction_problems(core: str, pr: dict[str, Any], pred_sha: str) -> list[str]:
    """Why the predictions record does not belong to its A1 record (empty when it does): the registered predictions
    run of the same pair and parameters, made by the shared predictions code (its code block and its prediction
    function)."""
    pdoc, cfg = pr["pred"]["doc"], pr["doc"]["config"]
    pcfg = pdoc.get("config") or {}
    bad = []
    if pdoc.get("run_kind") != "phase3_a1_predictions" or pdoc.get("registered_parameters") is not True:
        bad.append(f"not a registered A1 predictions run (run_kind {pdoc.get('run_kind')!r})")
    off = [k for k in PRED_SAME if pcfg.get(k) != cfg.get(k)]
    if off:
        bad.append(f"other {off} than its A1 record")
    code_sha = ((pdoc.get("code") or {}).get("sha256") or {}).get(A1P.THIS_FILE)
    if code_sha != pred_sha or (pdoc.get("prediction_function") or {}).get("sha256") != pred_sha:
        bad.append(f"its code sha256 of {A1P.THIS_FILE} or prediction function is not the shared predictions code")
    return bad


# ---- P-A1.3 --------------------------------------------------------------------------------------------------------

def cell_rows(pr: dict[str, Any]) -> list[dict[str, Any]]:
    """The pair's P-A1.3 cells (variant x c != 0): realised recomputed from the npz and checked against the record;
    predicted (unfloored, floored) re-derived from the predictions npz as phase3_a1.score derives it, and checked
    against the record's per-cell rows and cells and the predictions record's own block."""
    doc, z, core = pr["doc"], pr["z"], pr["core"]
    pz, pblock = pr["pred"]["z"], pr["pred"]["doc"].get("predicted") or {}
    target = doc["config"]["target_fpr"]
    shape = (doc["config"]["n_reps"] + 1, len(doc["config"]["seeds"]))
    per = {(r.get("variant"), A1P.ckey(float(r.get("c")))): r for r in doc["scoring"]["P-A1.3"]["per_cell"]}
    want = [(v, kc) for v in doc["config"]["variants"] for kc in NZ]
    if sorted(per) != sorted(want) or len(per) != len(doc["scoring"]["P-A1.3"]["per_cell"]):
        refuse(f"{core}: the P-A1.3 per-cell rows are not variant x c != 0")
    rows = []
    for v, kc in want:
        r = per[(v, kc)]
        a = z.get(f"{v}|{kc}|frozen|ratio_to_target")
        if a is None:
            refuse(f"{core}: the npz has no {v}|{kc}|frozen|ratio_to_target")
        realised = P.band_summary(a[1:], a[0])["mean"]                 # as phase3_a1.summaries computes the cell
        cell = doc["cells"][v][kc]
        if not realised == r["realised_ratio_to_target"] == cell["arms"]["frozen"]["ratio_to_target"]["mean"]:
            refuse(f"{core} {v} {kc}: the realised FPR/target of the npz is not the record's")
        if not (realised is not None and np.isfinite(realised) and realised > 0):     # floored at 1/|E|: unreachable
            refuse(f"{core} {v} {kc}: the realised FPR/target {realised} is not a positive finite number")
        pfa = pz.get(f"{v}|{kc}|predicted_fpr")
        if pfa is None or pfa.shape != shape:
            refuse(f"{core}: the predictions npz has no {v}|{kc}|predicted_fpr of shape {shape}")
        raw, fl = pfa.mean(axis=1) / target, np.maximum(pfa, 1.0 / N_E).mean(axis=1) / target   # as phase3_a1.score
        pc = cell.get("prediction") or {}
        pb = (pblock.get(v) or {}).get(kc) or {}
        row = {"pair": core, "variant": v, "c": float(r["c"]), "realised_ratio_to_target": realised}
        for key, tag, per_draw, pkey in (
                ("predicted_ratio_to_target", "unfloored", raw, "predicted_ratio_to_target"),
                ("predicted_ratio_to_target_floored", "floored", fl, "predicted_ratio_to_target_floored_1_over_E")):
            if pc.get(key) != r[key]:
                refuse(f"{core} {v} {kc}: the record's per-cell {key} differs from its cell's")
            if float(per_draw[1:].mean()) != r[key]:
                refuse(f"{core} {v} {kc}: the predictions npz does not give the record's {key} (the D-156 mean over "
                       f"the draws of the per-draw seed-mean predicted FPR/target)")
            if P.band_summary(per_draw[1:], per_draw[0])["mean"] != (pb.get(pkey) or {}).get("mean"):
                refuse(f"{core} {v} {kc}: the predictions record's {pkey} is not its npz's")
            p = r[key]
            e = float(abs(np.log(realised / p))) if p > 0 else None     # None: a zero prediction, a miss (D-322 (4))
            row[key] = p
            row[f"abs_log_realised_over_predicted_{tag}"] = e
            row[f"within_factor_1_5_{tag}"] = bool(e is not None and e <= np.log(A1.FACTOR))
        rows.append(row)
    return rows


def d170_status(prs: dict[str, dict[str, Any]], names: list[str], n_cells: dict[str, int], unit: str) -> dict[str, Any]:
    """The line's D-170 status: the per-pair flag the A1 records store and the line's cells of flagged pairs."""
    iv = {n: prs[n]["doc"].get("input_verification") or {} for n in names}
    flags = {n: iv[n].get("flag") for n in names}
    flagged = [n for n in names if flags[n] is not None]
    nf = int(sum(n_cells[n] for n in flagged))
    return {"per_pair_d170_flag": flags,
            "pin_flag_cleared_by": {n: (iv[n].get("pin_flag_cleared_by") or {}).get("decision") for n in names
                                    if iv[n].get("pin_flag_cleared_by")},
            "unit": unit, "n_cells": int(sum(n_cells[n] for n in names)), "n_flagged_pairs": len(flagged),
            "n_flagged_cells": nf,
            "status": ("every stored flag is cleared (None in every record of this line): D-331 made D-168 final and "
                       "its pinned record clears the D-170 flag of the judged pin-only inputs" if not flagged else
                       f"{len(flagged)} pair(s) carry the D-170 flag ({nf} of this line's {unit}): those cells enter "
                       "no paper table until the flag clears (D-170/D-172)"),
            "rule": D170_RULE}


def accuracy_line(rows: list[dict[str, Any]], pairs: list[str], d170: dict[str, Any], verdict: bool = False) -> dict[str, Any]:
    """D-177's accuracy line (phase3_a1.accuracy) over the cells, unfloored and floored, with D-327's expectation
    tested on each and each line's role (the verdict line, or beside)."""
    out: dict[str, Any] = {"pairs": pairs, "n_pairs": len(pairs), "d170": d170}
    for name, key in LINES13:
        acc = A1.accuracy(rows, key)
        share = acc["share_within_factor_1_5"]
        out[name] = {"role": VERDICT13 if verdict and name == "primary_unfloored" else "beside", **acc,
                     "expectation": f"at least {EXPECTED_SHARE:.0%} of cells within x{A1.FACTOR:g} (D-327 (4))",
                     "expectation_met": bool(share is not None and share >= EXPECTED_SHARE)}
    return out


def scored13(lines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """P-A1.3's pre-registered verdict line: white-box, unfloored; the other five lines are beside."""
    wb = lines[PRIMARY]
    return {"line": f"lines.{PRIMARY}.primary_unfloored", "variant": PRIMARY, "prediction": "unfloored",
            "why": "white-box primary (D-318 (9)); the unfloored prediction primary (D-346 (2)(e))",
            "pairs": wb["pairs"], **{k: v for k, v in wb["primary_unfloored"].items() if k != "role"},
            "d170": wb["d170"], "beside": BESIDE13}


# ---- P-A1.4 --------------------------------------------------------------------------------------------------------

def flag_arrays(z: dict[str, np.ndarray], variant: str, cs: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Per c, the per-draw flags phase3_a1.confusion reads (c = 0 is shared by the variants: 'c=0|...')."""
    out = {}
    for kc in cs:
        pre = "c=0|" if kc == "c=0" else f"{variant}|{kc}|"
        miss = [pre + k for k in FLAG_KEYS if pre + k not in z]
        if miss:
            refuse(f"the npz lacks {miss}")
        out[kc] = {k: z[pre + k] for k in FLAG_KEYS}
    return out


def detection_line(prs: dict[str, dict[str, Any]], names: list[str], variant: str, with_c0: bool = False) -> dict[str, Any]:
    """Per pair, the per-draw confusion counts over its cells (phase3_a1.confusion; over c != 0 they must equal the
    record's own P-A1.4 arrays); pooled over the pairs by D-255's estimator, every detector."""
    cs = [A1P.ckey(c) for c in A1P.C_GRID if with_c0 or c != 0]
    conf = {n: A1.confusion(flag_arrays(prs[n]["z"], variant, cs), cs) for n in names}
    if not with_c0:
        for n in names:
            for det in DETECTORS:
                for q in HS.COUNTS:
                    key = f"P-A1.4|{variant}|{det}|{q}"
                    if key not in prs[n]["z"] or not np.array_equal(conf[n][det]["_arrays"][q], prs[n]["z"][key]):
                        refuse(f"{n}: the confusion counts of the npz flags are not the record's {key}")
    idx = HS.pair_cluster_index(len(names))
    out: dict[str, Any] = {
        "variant": variant, "pairs": names, "n_pairs": len(names), "c": cs, "n_cells_per_draw": len(names) * len(cs),
        "class": "realised paired ratio >= 2 or <= 0.5 (D-241 (3)); flag: the ratio at or beyond 2 / 0.5",
        "estimator": "healthcheck_summary_d156.pooled_counts (D-255's pooled summary, D-241 (6))",
        "interval": (f"pair-cluster bootstrap B={HS.B}, seed {HS.SEED}" if idx is not None
                     else f"point only (fewer than {HS.MIN_PAIRS} pairs)"),
        "d170": d170_status(prs, names, {n: len(cs) for n in names}, "cells per draw")}
    for det in DETECTORS:
        out[det] = HS.pooled_counts({q: [conf[n][det]["_arrays"][q] for n in names] for q in HS.COUNTS}, idx)
    out["per_pair_d156_mean_counts_gaussian"] = {n: {q: float(conf[n]["gaussian"]["_arrays"][q][1:].mean())
                                                     for q in HS.COUNTS} for n in names}
    return out


def scored(line: dict[str, Any]) -> dict[str, Any]:
    """P-A1.4's outcome: the Gaussian flag's pooled sensitivity and specificity against D-327's thresholds. A pooled
    rate with no cell of its class is undefined: its _met and holds are then None, with the reason (choice (d))."""
    pooled = line["gaussian"]["pooled_from_mean_counts"]
    s, sp = pooled["sensitivity"], pooled["specificity"]
    s_ok = None if s is None else bool(s >= SENS_MIN)
    sp_ok = None if sp is None else bool(sp >= SPEC_MIN)
    why = [w for x, w in ((s, "no positive-class cell"), (sp, "no negative-class cell")) if x is None]
    return {"variant": line["variant"], "detector": "gaussian", "pairs": line["pairs"], "c": line["c"],
            "reading": CHOICE_C,
            "sensitivity": s, "sensitivity_ci": pooled.get("sensitivity_ci"),
            "specificity": sp, "specificity_ci": pooled.get("specificity_ci"),
            "thresholds": {"sensitivity": SENS_MIN, "specificity": SPEC_MIN, "applied_to": "the pooled point estimates"},
            "sensitivity_met": s_ok, "specificity_met": sp_ok,
            "holds": None if why else bool(s_ok and sp_ok),
            "undefined_reason": "undefined: " + " and ".join(why) if why else None,
            "d170": line.get("d170")}


# ---- the scoring -----------------------------------------------------------------------------------------------------

def pool(rdir: Path, scorer_sha: dict[str, str]) -> tuple[dict[str, Any], dict[str, tuple[Path, str]]]:
    """Every refusal, then P-A1.3 and P-A1.4. Returns the record's body and the input files (name -> path, sha256)."""
    paths = discover(rdir)
    files: dict[str, tuple[Path, str]] = {}
    prs = {c: load(c, p, files) for c, p in paths.items()}
    bad = {c: b for c, b in ((c, record_problems(c, pr["doc"])) for c, pr in prs.items()) if b}
    if bad:
        refuse(f"not every record can enter the pooled scoring: {bad}")
    params = {c: {k: pr["doc"]["config"].get(k) for k in ("n_reps", "seeds", "target_fpr")} for c, pr in prs.items()}
    if any(v != params[PAIRS[0]] for v in params.values()):
        refuse(f"the nine records were run at different draws, seeds or target: {params}")
    code = shared_code(prs, scorer_sha)
    pbad = {c: b for c, b in ((c, prediction_problems(c, pr, code["predictions_code_sha256"])) for c, pr in prs.items()) if b}
    if pbad:
        refuse(f"not every predictions record is its A1 record's: {pbad}")
    rows = [r for c in PAIRS for r in cell_rows(prs[c])]
    tr_pairs = [c for c in PAIRS if c in TRANSFER_PAIRS]

    def d170_13(names: list[str], rs: list[dict[str, Any]]) -> dict[str, Any]:
        return d170_status(prs, names, {n: sum(r["pair"] == n for r in rs) for n in names}, "cells")

    wb_rows = [r for r in rows if r["variant"] == "white_box"]
    tr_rows = [r for r in rows if r["variant"] == "transfer"]
    lines = {"white_box": accuracy_line(wb_rows, PAIRS, d170_13(PAIRS, wb_rows), verdict=True),
             "transfer": accuracy_line(tr_rows, tr_pairs, d170_13(tr_pairs, tr_rows))}
    p13 = {"cells": "pair x c != 0, per variant; realised: the D-156 mean over the draws of the frozen arm's FPR/target "
                    "(the seed mean of floored per-seed FPRs); predicted: the D-156 mean of the per-draw predicted "
                    "FPR/target, unfloored (primary) and floored per seed at 1/|E| (beside), re-derived from the "
                    "predictions npz",
           "accuracy_line": "D-177 (phase3_a1.accuracy): share within x1.5, median |log(realised/predicted)|, n cells; "
                            "a zero prediction is a miss and counted (D-322 (4))",
           "scored": scored13(lines), "lines": lines,
           "beside_both_variants_together": accuracy_line(rows, PAIRS, d170_13(PAIRS, rows)),
           "rows": rows}
    wb = detection_line(prs, PAIRS, PRIMARY)
    p14 = {"scored": scored(wb), "white_box": wb,
           "transfer_beside_not_scored": detection_line(prs, tr_pairs, "transfer"),
           "white_box_with_c0_beside_not_scored": detection_line(prs, PAIRS, PRIMARY, with_c0=True)}
    records = {c: {"file": pr["file"], "sha256": pr["sha256"], "draws_npz": pr["draws_npz"],
                   "variants": pr["doc"]["config"]["variants"], "commit": (pr["doc"].get("code") or {}).get("commit"),
                   "d170_flag": (pr["doc"].get("input_verification") or {}).get("flag"),
                   "anchor": {k: pr["doc"]["anchor"].get(k) for k in ("checked", "identical", "n_values_identical",
                                                                     "n_values_compared")},
                   "protocol_stops": {k: pr["doc"]["scoring"][k]["holds"] for k in PROTOCOL},
                   "predictions_record": {"file": pr["pred"]["file"], "sha256": pr["pred"]["sha256"],
                                          "draws_npz": pr["pred"]["draws_npz"]}}
               for c, pr in prs.items()}
    return {"parameters": params[PAIRS[0]], "a1_code": code, "records": records, "P-A1.3": p13, "P-A1.4": p14}, files


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Phase 3 A1: the pooled P-A1.3 / P-A1.4 scoring (D-346 (4))")
    ap.add_argument("--records-dir", default=None, help="TESTING: a directory of per-pair records (default results/)")
    ap.add_argument("--out", default=None, help="TESTING: a JSON path outside results/")
    args = ap.parse_args()
    rdir = Path(args.records_dir).resolve() if args.records_dir else RESULTS.resolve()
    out = Path(args.out).resolve() if args.out else FINAL_OUT.resolve()
    final = out == FINAL_OUT.resolve()
    if out.exists():
        refuse(f"{out} exists; results are never overwritten")
    if final and rdir != RESULTS.resolve():
        refuse("the final name reads only the records in results/")
    if not final and out.is_relative_to(RESULTS.resolve()):
        refuse("a test output (--out) must not be under results/")
    code0 = P.code_block(CODE_FILES)
    body, files = pool(rdir, code0["sha256"])
    code1 = P.code_block(CODE_FILES)
    changed = [n for n, (p, h) in files.items() if sha256_file(p) != h]
    if code1["sha256"] != code0["sha256"] or changed:
        refuse(f"to write: the code or an input changed during the run: {changed}")
    payload = {**provenance_block("phase3_a1_pooled"), "preregistration": PREREG,
               "preregistration_text": {"entry": "DECISIONS.md D-346 (4)", "text": PREREG_TEXT},
               "code": code0, "config": CONFIG, "config_hash": canonical_hash(CONFIG),
               "input_files_sha256": {n: h for n, (_, h) in files.items()}, "choices": CHOICES, **body}
    digest = write_results(out, payload, exclusive=True)
    for v, line in body["P-A1.3"]["lines"].items():                 # printed only once the record is written
        a = line["primary_unfloored"]
        print(f"P-A1.3 {v:9s} ({line['n_pairs']} pairs): {a['n_cells']} cells, {a['n_cells_predicted_zero']} predicted "
              f"zero; within x1.5 {a['share_within_factor_1_5']}; median |log| {a['median_abs_log_realised_over_predicted']}; "
              f"expectation met {a['expectation_met']} ({a['role']})", flush=True)
    s = body["P-A1.4"]["scored"]
    print(f"P-A1.4 white-box gaussian: sensitivity {s['sensitivity']} {s['sensitivity_ci']}, specificity "
          f"{s['specificity']} {s['specificity_ci']}; holds {s['holds']}"
          + (f" ({s['undefined_reason']})" if s["undefined_reason"] else ""), flush=True)
    print(f"Written: {out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
