"""Phase 3 attacker A3 (D-332): the POOLED scoring of P-A3.1 to P-A3.4 over the nine per-pair records, as DECISIONS.md
D-352 (3) pre-registers it (quoted whole in PREREG_TEXT and in the record). No model, cache or prediction is computed.

INPUTS (36 files, each hashed and listed in input_files_sha256, D-297 (1)): for each of the nine released pairs of
D-332 (1) (populations.UPDATE_PAIRS_REPAIR), results/phase3_a3_<pair>.json and its _draws.npz (src/phase3_a3.py), and
the predictions record that record names, results/phase3_a3_predictions_<pair>.json and its _draws.npz
(src/phase3_a3_predictions.py). REFUSES (exit 1, a message, nothing written) unless all nine exist and no other pair's
record is there (discover(): only the exact names phase3_a3_<core>.json of the nine are read; every other file, the
sensitivities' phase3_a3_sens_* records of D-376 among them, is ignored; a stem shaped like another pair refuses); every JSON is read once as a registered input (D-297) and must be a JSON object; each npz is the bytes
(sha256, size, keys) its record names, read once as bytes, hashed and loaded from those bytes, one row per draw (the
frozen split first); each predictions record is the file and the bytes (JSON and npz sha256) its A3 record names, the
registered predictions run of the same pair, draws, seeds, target, rho grid, pool size, attackers and injections, at
the 6-thread pin, with its own config_hash, the A3 record's inputs and D-170 verification, made by the shared
prediction function; each A3 record is a registered run of its pair at the 6-thread pin, not --anchor-only, with
D-332's rho grid, attackers, repairs, band, cut and loss, its own config_hash and a stored D-170 flag, that passed its
own SELF-ANCHOR (D-332 (6), per pair: checked and identical, no mismatch, all 148 compared values identical and the four
minimum rhos equal, against the 4d B record of record) and its implementation checks (threshold identity, band); the
nine A3 records share one code block (sha256 map and library versions; their commits are listed and may differ, as
D-352 (6) allows) and each predictions record gives that block on its own files; src/phase3_a3.py and
src/phase3_a3_predictions.py are the reviewed blobs (REVIEWED_SHA256: the files at ee5ad0c) in the records and in this
tree; draws, seeds and target agree across the nine. Any other malformed field (a missing key, a wrong type or shape)
refuses too (pool()).
BOUND AND RE-DERIVED (choice (b)): each pair's 48 P-A3.1 cells first, one by one: the realised TPR from the A3 npz (the
D-156 mean over draws of the seed-mean TPR of the pool's score repair, as phase3_a3.summaries forms the cell) and the
predicted TPR from the predictions npz (the D-156 mean over draws of the seed-mean F1bar(t'), exactly as
phase3_a3.score derives it), each required equal to the A3 record's per-cell row. Then, from each A3 npz by phase3_a3's
own functions, the record's cells (summaries), minimum-rho table (min_rho_table), A3-band outcomes (band_outcomes),
health flag shares and scoring (score, with the predictions npz and the record's threshold-identity counts), each
required equal to the record's (the D-282 interval blocks aside: they need the caches' positives); from each
predictions npz, its 'predicted' block (phase3_a3_predictions.predicted_cells), required equal to the predictions
record's; the anchor's A3-top minimum rhos must equal the re-derived ones. The pooled lines use the re-derived values.

P-A3.1 (D-352 (3)): cells = pair x pool (R_same, R_fresh, 1% traffic) x attacker (A3-top, A3-benign) x rho (the 8 grid
  values) = 48 per pair, 432 over the nine; D-177's accuracy line by phase3_a3.accuracy (share within x1.5, median
  |log(realised/predicted)|, n cells; a zero prediction is a miss and counted, D-322 (4)). THE VERDICT LINE
  (P-A3.1.scored) is all 432 cells against D-332's expectation, at least 80% within x1.5; beside: the 144
  quantile-transfer cells alone and the 288 threshold re-set cells alone (agreement there is mechanical), each with no
  expectation tested (not a verdict).
P-A3.2: A3-benign's minimum rho >= A3-top's (phase3_a3.at_least; None reads '> 0.25', D-244 (6)) per (pair, pool,
  repair), 36 cells; a cell is decided unless both are '> 0.25'. The pooled verdict 'in every pair and pool': HOLDS if
  every decided cell holds and at least one is decided, FAILS if any decided cell fails, UNDEFINED if none is decided
  (phase3_a3.decided_verdict, D-351 (3)'s reading); counts of decided, holding, failing and undecided cells.
P-A3.3: per pair by that rule over its four cells (a cell whose band is infeasible, A3-band's m 0 in every draw, is
  undefined with its reason, D-352 (2)); pooled 'in every pair' by the same rule over the nine pair verdicts; beside,
  the number of infeasible cells with their reasons, each pair's clean-pool Gaussian flag share per pool, and D-352
  (2)'s finding (the infeasible cells whose clean pool flags in every draw).
P-A3.4: restandardisation's minimum rho >= 10 x A3-top's of each score repair (phase3_a3.at_least, factor 10, relative
  tolerance 1e-9), 27 cells (pair x score repair), the same pooling rule; no new measurement (A3-top equals 4d B's A1
  bit for bit, D-322).
  The verdict of the rule over cells equals its verdict over pair verdicts (choice (c)); each line reports both.
D-170: every pooled line carries the per-pair D-170 flag the A3 records store (input_verification.flag) and its number
  of flagged cells (D-331: D-168 is final and its pinned record clears the flag of the judged pin-only inputs).
RELAUNCHING AN A3 LIST: this scorer requires the nine A3 records to share one code block (choice (e), strict). So
  queues/laneB_20261003f_a3.tsv or queues/laneD_20261003f_a3.tsv may be relaunched, or a later half launched, only at a
  SYNC_COMMIT whose tree manifest gives, for every file of phase3_a3.CODE_FILES, the sha256 the first records give, with
  the same library versions (D-352 (6)); the commit itself may differ (listed). Records made under any other block make
  this scorer refuse.
CHOICES BY THE CLOSEST D-ENTRY (D-181): CHOICES below, written into the record.
Writes results/phase3_a3_pooled.json once (exclusive), then prints the pooled lines. A test reads --records-dir and
writes --out, outside results/.
"""

from __future__ import annotations

import argparse
import io
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import input_registry as REG  # noqa: E402
import paired as P  # noqa: E402
import phase3_a1_predictions as A1P  # noqa: E402
import phase3_a3 as A3  # noqa: E402
import phase3_a3_predictions as A3P  # noqa: E402
import poolstudy_d156 as PS  # noqa: E402
import populations as POP  # noqa: E402
from provenance import (  # noqa: E402
    canonical_hash, ensure_utf8_stdout, provenance_block, read_result, sha256_bytes, sha256_file, write_results,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FINAL_OUT = RESULTS / "phase3_a3_pooled.json"
THIS_FILE = "src/phase3_a3_pooled.py"
CODE_FILES = list(dict.fromkeys([THIS_FILE, *A3.CODE_FILES]))
PAIRS = [A1P.core_of(p) for p in POP.UPDATE_PAIRS_REPAIR]                     # D-332 (1): the nine released pairs
# discover(): a stem shaped like a pair (the nine's naming: <model>_base2instruct, WG_<model>_base2instruct,
# <model>_to_<model>) or the core of a declared repair run that is not one of the nine is an extra pair; any other
# phase3_a3_*.json (predictions, pooled, sens_<pair>, sens_pooled, ...) is ignored
PAIR_SHAPE = re.compile(r"(WG_)?[A-Za-z0-9]+_(base2instruct|to_[A-Za-z0-9]+)")
DECLARED_CORES = frozenset(k[len("repair_"):] for k in REG.ALL_RUNS if k.startswith("repair_"))
POOLS, RHOS, PREDICTED, SCORE_REPAIR = A3.POOLS, A3.RHOS, A3.PREDICTED, A3.SCORE_REPAIR
# the reviewed blobs whose functions this scorer calls (sha256 of the file bytes): the files at ee5ad0c (the A3 fix round
# after the three reviews of c220781, D-352 (2)), which the A3 lane lists run; a later change of either file breaks them
REVIEWED_SHA256 = {"src/phase3_a3.py": "9d327d245c9a7abaa2d7c9e170dc188037722e8bb15a2ea6146cf8d7409a0493",
                   "src/phase3_a3_predictions.py": "7e22451a5f25fbd2ded673911d413ea24d421222c01fa91fbae7c63dbd179ccc"}
REUSED = tuple(REVIEWED_SHA256)
EXPECTED_SHARE = 0.80                                                          # D-332 (5) P-A3.1, D-352 (3)
THREADS = "6"                                                                  # the final names' BLAS pin (both scripts)
IMPL_CHECKS = ("threshold_identity", "band")                                   # phase3_a3's implementation checks
# phase3_a3.anchor_check's count of compared D-156 summaries: the four quantities of the frozen arm, and per repair of
# the unpoisoned arm and of A3-top at every rho: 4 x (1 + 4 x (1 + 8))
ANCHOR_N = 148
# D-352 (3)'s cell counts, as written there: 48 P-A3.1 cells per pair (3 pools x 2 attackers x 8 rho), 432 over the
# nine, 144 of them quantile transfer (traffic) and 288 threshold re-set (R_same, R_fresh)
CELLS_PER_PAIR_31 = 48
N31 = {"all": 432, "quantile_transfer": 144, "threshold_reset": 288}
N32 = 36                                       # (pair, pool, repair): 9 x 4, restandardisation included (A3 choice (n))
N33 = 36                                       # (pair, pool, repair): 9 x 4 (A3 choice (m))
N34 = 27                                       # (pair, score repair): 9 x 3 (A3 choice (o))
# the D-282 interval blocks phase3_a3.main() adds after its cells, minimum-rho rows and A3-band outcomes are formed:
# not re-derivable without the caches (the eval positives' scores), so left out of the comparison
D282_KEYS = ("tpr_d282_combined", "tpr_loss_at_min_rho_d282_combined", "tpr_loss_at_largest_m_d282_combined",
             "a3_top_tpr_loss_at_same_m_d282_combined")
PRED_SAME = ("pair", "n_reps", "seeds", "target_fpr", "rhos", "predicted_attackers", "pool_n", "draws_index_sha256",
             "injections_sha256")
CONFIG_SAME = {"rhos": list(RHOS), "pool_n": PS.DEFAULT_POOL, "predicted_attackers": list(PREDICTED),
               "attackers": list(A3.ATTACKERS), "repairs": {pool: list(A3.REPAIRS_OF[pool]) for pool in POOLS},
               "band": list(A3.BAND), "health_cut": list(A3.CUT), "tpr_loss": PS.TPR_LOSS}
WORDS = {True: "HOLDS", False: "FAILS", None: "UNDEFINED"}
POOL_RULE = ("D-352 (3): HOLDS if every decided unit holds and at least one is decided, FAILS if any decided unit "
             "fails, UNDEFINED if none is decided (phase3_a3.decided_verdict; D-351 (3)'s reading)")
VERDICT31 = "scored: the pre-registered verdict line (all 432 cells, D-352 (3))"
EXPECTATION = f"at least {EXPECTED_SHARE:.0%} of cells within x{A3.FACTOR:g} (D-332 (5))"
NO_EXPECTATION = ("none: beside, not a verdict (D-352 (3) registers D-332's 80% expectation for the 432-cell verdict "
                  "line only)")
D170_RULE = ("D-170/D-172: a result with a flag enters no paper table until D-168 has passed; D-331: D-168 is final "
             "(results/d168_final_verdicts.json, pinned in src/input_registry.py), so the flag clears for the pin-only "
             "inputs among the 17 caches it judged")
PREREG = ("D-352 (3) (the pooled scoring of D-332 (5) P-A3.1 to P-A3.4); D-352 (2) (an infeasible band undefined); D-332; "
          "D-177; D-244 (6); D-322 (4); D-351 (3)")
PREREG_TEXT = (
    "(3) POOLED A3 SCORING, PRE-REGISTERED NOW (a separate script with three reviewers, written before any per-pair A3 "
    "record is read; the A3 lane logs print check and verdict words only): P-A3.1: cells = pair x pool (R_same, "
    "R_fresh, 1% traffic) x attacker (A3-top, A3-benign) x rho (the 8 grid values) = 48 per pair, 432 over the nine "
    "pairs; per cell realised = the D-156 mean over draws of the seed-mean TPR, predicted = the same of the predicted "
    "TPR; D-177's accuracy line (share within x1.5, median |log(realised/predicted)|, n cells; zero predictions are "
    "misses, D-322 (4)); the VERDICT line is all 432 cells against D-332's expectation of 80%; beside: the "
    "quantile-transfer cells alone (144) and the threshold re-set cells alone (288; agreement there is mechanical). "
    "P-A3.2 (A3-benign's minimum rho >= A3-top's): a (pair, pool, repair) cell is decided unless both minimum rhos are "
    "'> 0.25' (D-244 (6)); the pooled verdict 'in every pair and pool' HOLDS if every decided cell holds and at least "
    "one is decided, FAILS if any decided cell fails, is UNDEFINED if none is decided; counts of decided, holding, "
    "failing and undecided cells reported. P-A3.3: per pair as (2) (a pair fails if any decided cell fails, holds if "
    "every decided cell holds and one is decided, else undefined); pooled 'in every pair' by the same rule over pairs, "
    "with the number of infeasible cells and each pair's clean-pool flag share beside. P-A3.4 (restandardisation's "
    "minimum rho >= 10 x A3-top's of each score repair): the same pooling rule; it carries no new measurement (A3-top "
    "equals 4d B's A1 bit for bit, D-322). The self-anchor stays per pair (D-332 (6)).")
CONFIG = {
    "pairs": PAIRS, "pools": list(POOLS), "rhos": list(RHOS), "attackers_P-A3.1": list(PREDICTED),
    "score_repair": dict(SCORE_REPAIR), "repairs": [f"{p}|{r}" for p, r in A3.REPAIRS],
    "P-A3.1": {"expectation_share_within_factor": EXPECTED_SHARE, "factor": A3.FACTOR,
               "n_cells_per_pair": CELLS_PER_PAIR_31, "n_cells": N31,
               "verdict_line": "all cells", "beside": ["quantile_transfer cells only", "threshold_reset cells only"],
               "expectation_tested_on": "the verdict line only (D-352 (3))",
               "miss": "a zero predicted (or realised) TPR: an infinite error, outside x1.5, inside the median "
                       "(phase3_a3.accuracy; D-322 (4))"},
    "P-A3.2": {"rule": "A3-benign's minimum rho >= A3-top's (phase3_a3.at_least)", "beyond_grid": "None reads '> 0.25' "
               "(D-244 (6))", "unit": "(pair, pool, repair)", "n_cells": N32},
    "P-A3.3": {"rule": "A3-band's TPR loss at its largest m < A3-top's at the same m (phase3_a3.p33_cell)",
               "unit": "pair, over its (pool, repair) cells", "n_pairs": len(PAIRS), "n_cells": N33,
               "infeasible": "undefined, with its reason (D-352 (2))"},
    "P-A3.4": {"rule": "restandardisation's minimum rho >= factor x A3-top's of each score repair (phase3_a3.at_least)",
               "factor": A3.P34_FACTOR, "rho_tol": A3.RHO_TOL, "unit": "(pair, score repair)", "n_cells": N34},
    "pooling_rule": POOL_RULE, "tpr_loss": PS.TPR_LOSS,
    "self_anchor": {"per_pair": True, "n_values_compared": ANCHOR_N},
    "code": {"one_code_block": "the nine A3 records' code sha256 maps and library versions identical; commits listed",
             "reviewed_sha256": REVIEWED_SHA256},
}
CHOICES = [
    "(a) P-A3.1's cells are phase3_a3.score's per-cell rows: the pool's score repair (threshold re-set on R_same and "
    "R_fresh, quantile transfer on traffic; phase3_a3_predictions choice (b); restandardisation is P-A3.4's), realised "
    "and predicted as cell D-156 means (D-332 (5)). The accuracy line is phase3_a3.accuracy unchanged: a zero predicted "
    "TPR is a miss (D-322 (4)), and so is a zero realised TPR (its |log| is infinite; D-346 (2) as phase3_a3 applies it); "
    "both counts are stated. The expectation (at least 80% within x1.5) is tested on the verdict line only, as D-352 (3) "
    "registers it for the 432 cells; the two beside lines report their accuracy lines with no expectation_met.",
    "(b) Binding and re-derivation (D-351 (1)'s binding, extended): each pair's 48 P-A3.1 cells are checked one by one, "
    "the realised TPR from the A3 npz and the predicted TPR from the predictions npz, exactly as phase3_a3.score derives "
    "them, against the A3 record's per-cell rows; then each A3 record's cells, minimum-rho table, A3-band outcomes, "
    "health flag shares and scoring are recomputed from its npz by phase3_a3.summaries, min_rho_table, band_outcomes and "
    "score (with the predictions npz and the record's threshold-identity counts), and each predictions record's "
    "'predicted' block by phase3_a3_predictions.predicted_cells from its npz; each must equal the record's, the D-282 "
    "interval blocks aside (they need the eval positives' scores from the caches). The pooled lines use the re-derived "
    "values.",
    "(c) D-352 (3) pools P-A3.2 over (pair, pool, repair) cells, P-A3.3 over pairs and P-A3.4 by 'the same pooling "
    "rule'. For this rule the verdict over cells equals the verdict over the pairs' own verdicts (a decided cell fails "
    "exactly when a pair fails; a decided cell exists exactly when a pair is decided); a test proves it exhaustively on "
    "small grids. P-A3.4's verdict is computed over its 27 (pair, score repair) cells, P-A3.3's over its nine pairs (each "
    "pair's verdict by the same rule over its four cells); every line reports the other reading beside with its counts.",
    "(d) P-A3.2's cells include restandardisation on traffic (four per pair, 36; phase3_a3 choice (n)); P-A3.4's are "
    "the three score repairs per pair (27; phase3_a3 choice (o)); P-A3.3's the four (pool, repair) cells per pair "
    "(phase3_a3 choice (m)).",
    "(e) One code block, commits listed (A1's choice (f), D-351 (1); D-352 (6)): the nine A3 records' code sha256 maps "
    "and library versions are identical, and each predictions record's code block names src/phase3_a3_predictions.py "
    "and gives the A3 block's sha256 on each of its files, with the same libraries. The 18 records' commits may differ "
    "and are listed: D-352 (6) allows a relaunch, or a later half, at another SYNC_COMMIT that keeps every "
    "phase3_a3.CODE_FILES sha256. src/phase3_a3.py and src/phase3_a3_predictions.py in the records and in this tree are "
    "the reviewed blobs (REVIEWED_SHA256: the files at ee5ad0c; this scorer calls their functions); any other code file "
    "that differs from this tree is listed.",
    "(f) The self-anchor stays per pair (D-332 (6)): each A3 record's anchor must be checked and identical, with no "
    "minimum-rho or value mismatch, all ANCHOR_N (148, phase3_a3.anchor_check's count) values identical, the four "
    "minimum-rho comparisons equal, against the 4d B record of record (populations.POOLSTUDY_D156), and its A3-top "
    "minimum rhos equal to the re-derived ones; its implementation checks (threshold identity, band) held. A record "
    "that fails any refuses the whole scoring.",
    "(g) D-170: each pooled line carries the per-pair flag the A3 records store (input_verification.flag) and its "
    "number of flagged cells; D-331 made D-168 final and its pinned record clears the flag of the judged pin-only "
    "inputs. No flagged or unflagged sub-line is formed (none is pre-registered).",
    "(h) Beside P-A3.3 (D-352 (2), (3)): the infeasible cells with their reasons, each pair's clean-pool Gaussian flag "
    "share per pool (phase3_a3.band_outcomes), D-352 (2)'s finding cells, and each cell's loss comparison over only the "
    "draws with m > 0, all from the re-derived per-pair blocks. Beside P-A3.2: each pair's benign_fit_row_share_per_seed "
    "(the confound with a direction, D-352 (2)).",
    "(i) The pooled lines are printed only after the record is written (D-351 (1)); D-352 (3)'s 'check and verdict "
    "words only' governs the per-pair A3 lane logs, before any pooled scoring exists.",
]
# what a malformed record raises where a field is missing or of another type or shape: refused, never scored
MALFORMED = (KeyError, TypeError, AttributeError, ValueError, IndexError, ZeroDivisionError)


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def obj(x: Any) -> dict[str, Any]:
    """x when it is a JSON object, else an empty one (a malformed block then fails its check)."""
    return x if isinstance(x, dict) else {}


def same(mine: Any, theirs: Any) -> bool:
    """Equal as written: the canonical JSON of both (a record's floats survive the JSON round trip exactly)."""
    return canonical_hash(mine) == canonical_hash(theirs)


def strip_d282(o: Any) -> Any:
    """`o` without the D-282 interval blocks phase3_a3.main() adds (D282_KEYS), at any depth."""
    if isinstance(o, dict):
        return {k: strip_d282(v) for k, v in o.items() if k not in D282_KEYS}
    if isinstance(o, list):
        return [strip_d282(v) for v in o]
    return o


# ---- the inputs and the refusals ---------------------------------------------------------------------------------

def discover(rdir: Path) -> dict[str, Path]:
    """The nine per-pair A3 records in rdir, by their exact names phase3_a3_<core>.json for the nine cores of D-332 (1).
    Every other file is ignored, never read: the predictions records (read later by the name their A3 record gives),
    this scorer's own record, the sensitivities' phase3_a3_sens_<pair>.json and phase3_a3_sens_pooled.json (D-376),
    anything else. Refuses when a pair's record is missing, or when a stem is shaped like a pair that is not one of the
    nine (an extra pair: PAIR_SHAPE, or the core of a declared repair run)."""
    found = {c: rdir / f"phase3_a3_{c}.json" for c in PAIRS if (rdir / f"phase3_a3_{c}.json").is_file()}
    stems = {p.stem[len("phase3_a3_"):] for p in rdir.glob("phase3_a3_*.json")}
    extra = sorted(s for s in stems if s not in PAIRS and (PAIR_SHAPE.fullmatch(s) or s in DECLARED_CORES))
    missing = [c for c in PAIRS if c not in found]
    if missing or extra:
        refuse(f"the A3 records in {rdir} are not the nine pairs of D-332 (1): missing {missing}, extra {extra}")
    return found


def read_json(path: Path, what: str) -> dict[str, Any]:
    """A results JSON read as a registered input (D-297 (1)); refuses one that cannot be read or is not an object."""
    try:
        doc = read_result(path)
    except (OSError, UnicodeDecodeError, ValueError) as e:            # json.JSONDecodeError is a ValueError
        refuse(f"{path.name} ({what}) cannot be read as JSON: {type(e).__name__}")
    if not isinstance(doc, dict):
        refuse(f"{path.name} ({what}) is not a JSON object")
    return doc


def read_npz(npz_path: Path, block: Any, owner: str) -> tuple[str, int, dict[str, np.ndarray]]:
    """The npz `block` names ({file, sha256, size, keys}): read once as bytes, hashed, loaded from those bytes."""
    block = obj(block)
    if not npz_path.is_file():
        refuse(f"{npz_path.name} is missing")
    nb = npz_path.read_bytes()
    h = sha256_bytes(nb)
    if h != block.get("sha256") or len(nb) != block.get("size"):
        refuse(f"{npz_path.name} is not the bytes {owner} names")
    try:
        with np.load(io.BytesIO(nb), allow_pickle=False) as zf:
            z = {k: zf[k] for k in zf.files}
    except Exception as e:                                          # noqa: BLE001 (any unreadable npz refuses)
        refuse(f"{npz_path.name} is not a readable npz: {type(e).__name__}")
    keys = block.get("keys")
    if not isinstance(keys, list) or sorted(z) != sorted(str(k) for k in keys):
        refuse(f"{npz_path.name} does not hold the keys {owner} lists")
    return h, len(nb), z


def load(core: str, path: Path, files: dict[str, tuple[Path, str]]) -> dict[str, Any]:
    """One per-pair A3 record (a registered read, D-297 (1)) and its npz, loaded from the very bytes whose sha256 is
    checked against the record; then the predictions record it names (load_predictions)."""
    doc = read_json(path, "an A3 record")
    sha = sha256_file(path)
    npz_path = path.with_name(path.stem + "_draws.npz")
    dz = obj(doc.get("draws_npz"))
    if dz.get("file") != A1P.rel(npz_path):
        refuse(f"{path.name} names the npz {dz.get('file')!r}, not {A1P.rel(npz_path)!r}")
    nsha, nsize, z = read_npz(npz_path, dz, path.name)
    files[A1P.rel(path)] = (path, sha)
    files[A1P.rel(npz_path)] = (npz_path, nsha)
    return {"core": core, "doc": doc, "z": z, "file": A1P.rel(path), "sha256": sha,
            "draws_npz": {"file": A1P.rel(npz_path), "sha256": nsha, "size": nsize},
            "pred": load_predictions(core, path, doc, files)}


def load_predictions(core: str, path: Path, doc: dict[str, Any], files: dict[str, tuple[Path, str]]) -> dict[str, Any]:
    """The predictions record the A3 record at `path` names (beside it in the records directory): its JSON (a
    registered read, D-297) with the sha256 the A3 record gives, and its npz with the sha256 both records give."""
    ppath = path.with_name(f"phase3_a3_predictions_{core}.json")
    pnpz = ppath.with_name(ppath.stem + "_draws.npz")
    prec = obj(doc.get("predictions_record"))
    prz = obj(prec.get("draws_npz"))
    if prec.get("file") != A1P.rel(ppath) or prz.get("file") != A1P.rel(pnpz):
        refuse(f"{path.name} names the predictions record {prec.get('file')!r} and npz {prz.get('file')!r}, not "
               f"{A1P.rel(ppath)!r} and {A1P.rel(pnpz)!r}")
    if not ppath.is_file():
        refuse(f"{ppath.name}, the predictions record {path.name} names, is missing")
    pdoc = read_json(ppath, "a predictions record")
    psha = sha256_file(ppath)
    if psha != prec.get("sha256"):
        refuse(f"{ppath.name} is not the bytes {path.name} names")
    pdz = obj(pdoc.get("draws_npz"))
    if pdz.get("file") != A1P.rel(pnpz):
        refuse(f"{ppath.name} names the npz {pdz.get('file')!r}, not {A1P.rel(pnpz)!r}")
    nsha, nsize, pz = read_npz(pnpz, pdz, ppath.name)
    if nsha != prz.get("sha256"):
        refuse(f"{pnpz.name} is not the bytes {path.name} names")
    files[A1P.rel(ppath)] = (ppath, psha)
    files[A1P.rel(pnpz)] = (pnpz, nsha)
    return {"doc": pdoc, "z": pz, "file": A1P.rel(ppath), "sha256": psha,
            "draws_npz": {"file": A1P.rel(pnpz), "sha256": nsha, "size": nsize}}


def anchor_problems(core: str, an: dict[str, Any]) -> list[str]:
    """Why the record's self-anchor (D-332 (6)) did not pass (empty when it passed): checked and identical, no
    mismatch, every compared value identical (ANCHOR_N), the four minimum-rho comparisons equal, against the 4d B
    record of record."""
    n_cmp = an.get("n_values_compared")
    mr = obj(an.get("min_rho"))
    want_keys = sorted(f"{p}|{r}" for p, r in A3.REPAIRS)
    ok = (an.get("checked") is True and an.get("identical") is True and an.get("min_rho_mismatches") == []
          and an.get("value_mismatches") == [] and n_cmp == ANCHOR_N and an.get("n_values_identical") == n_cmp
          and sorted(mr) == want_keys
          and all(obj(v).get("equal") is True and "4dB_min_rho" in obj(v)
                  and obj(v).get("4dB_min_rho") == obj(v).get("a3_top_min_rho") for v in mr.values()))
    bad = [] if ok else ["did not pass its self-anchor (D-332 (6))"]
    rec = f"results/{A1P.record_of_record_stem(core)}.json"
    if an.get("file") != rec:
        bad.append(f"its self-anchor is against {an.get('file')!r}, not the 4d B record of record {rec!r}")
    return bad


def record_problems(core: str, doc: dict[str, Any]) -> list[str]:
    """Why one A3 record cannot enter the pooled scoring (empty when it can)."""
    cfg = obj(doc.get("config"))
    bad = []
    if doc.get("run_kind") != "phase3_a3" or cfg.get("pair") != f"repair_{core}":
        bad.append(f"not the A3 record of repair_{core} (run_kind {doc.get('run_kind')!r}, pair {cfg.get('pair')!r})")
    if doc.get("registered_parameters") is not True or cfg.get("anchor_only") is not False:
        bad.append("not a registered run, or written with --anchor-only")
    th = obj(doc.get("blas_threads"))
    if any(th.get(v) != THREADS for v in A3.THREAD_VARS):
        bad.append(f"not made at the 6-thread pin: {th}")
    bad += anchor_problems(core, obj(doc.get("anchor")))
    ic = obj(doc.get("implementation_checks"))
    failed = [k for k in IMPL_CHECKS if obj(ic.get(k)).get("holds") is not True]
    if failed:
        bad.append(f"did not pass its implementation checks: {failed}")
    off = sorted(k for k, v in CONFIG_SAME.items() if cfg.get(k) != v)
    if off:
        bad.append(f"other {off} than D-332's")
    if doc.get("config_hash") != canonical_hash(cfg):
        bad.append("its config_hash is not its config's")
    if "flag" not in obj(doc.get("input_verification")):
        bad.append("stores no D-170 flag (input_verification.flag)")
    return bad


def shared_code(prs: dict[str, dict[str, Any]], scorer_sha: dict[str, str]) -> dict[str, Any]:
    """The nine A3 records' one code block (choice (e)), each predictions record's agreement with it on its own files,
    the commits (listed), and this scorer's agreement with the files whose functions it calls, which must be the
    reviewed blobs."""
    codes = {c: obj(pr["doc"].get("code")) for c, pr in prs.items()}
    pcodes = {c: obj(pr["pred"]["doc"].get("code")) for c, pr in prs.items()}
    ref = codes[PAIRS[0]]
    sha = ref.get("sha256")
    off = [c for c, cb in codes.items() if cb.get("sha256") != sha or cb.get("libraries") != ref.get("libraries")]
    if off or not isinstance(sha, dict) or not sha:
        refuse(f"the nine A3 records do not share one code block: {off} differ from {PAIRS[0]}")
    poff = [c for c, cb in pcodes.items() if not isinstance(cb.get("sha256"), dict) or A3P.THIS_FILE not in cb["sha256"]
            or any(sha.get(f) != h for f, h in cb["sha256"].items()) or cb.get("libraries") != ref.get("libraries")]
    if poff:
        refuse(f"the predictions records {poff} do not give the A3 records' code block on their files")
    mine = [f for f in REUSED if not sha.get(f) or scorer_sha.get(f) != sha.get(f)]
    if mine:
        refuse(f"this tree's {mine} are not the records' (the scorer calls their functions)")
    unreviewed = [f for f in REUSED if sha.get(f) != REVIEWED_SHA256[f]]
    if unreviewed:
        refuse(f"the records' and this tree's {unreviewed} are not the reviewed blobs (REVIEWED_SHA256, the files at "
               f"ee5ad0c)")
    return {"sha256": sha, "libraries": ref.get("libraries"), "predictions_code_sha256": sha[A3P.THIS_FILE],
            "reviewed_sha256": REVIEWED_SHA256,
            "commits": {"a3_records": {c: cb.get("commit") for c, cb in codes.items()},
                        "predictions_records": {c: cb.get("commit") for c, cb in pcodes.items()},
                        "distinct": sorted({str(cb.get("commit")) for cb in (*codes.values(), *pcodes.values())})},
            "a3_files_differing_from_this_tree": sorted(f for f in sha if scorer_sha.get(f) != sha[f])}


def prediction_problems(core: str, pr: dict[str, Any], pred_sha: str) -> list[str]:
    """Why the predictions record does not belong to its A3 record (empty when it does): the registered predictions
    run of the same pair and parameters, at the pin, made by the shared prediction function, on the same inputs."""
    pdoc, doc = pr["pred"]["doc"], pr["doc"]
    pcfg, cfg = obj(pdoc.get("config")), obj(doc.get("config"))
    bad = []
    if pdoc.get("run_kind") != "phase3_a3_predictions" or pdoc.get("registered_parameters") is not True:
        bad.append(f"not a registered A3 predictions run (run_kind {pdoc.get('run_kind')!r})")
    th = obj(pdoc.get("blas_threads"))
    if any(th.get(v) != THREADS for v in A3.THREAD_VARS):
        bad.append(f"not made at the 6-thread pin: {th}")
    off = [k for k in PRED_SAME if pcfg.get(k) != cfg.get(k)]
    if off:
        bad.append(f"other {off} than its A3 record")
    if pdoc.get("config_hash") != canonical_hash(pcfg):
        bad.append("its config_hash is not its config's")
    pf = obj(pdoc.get("prediction_function"))
    if pf.get("module") != A3P.THIS_FILE or pf.get("function") != "predicted_tpr" or pf.get("sha256") != pred_sha:
        bad.append(f"its prediction function is not the shared {A3P.THIS_FILE} predicted_tpr")
    if pdoc.get("input_cache_sha256") != doc.get("input_cache_sha256") or \
            pdoc.get("input_verification") != doc.get("input_verification"):
        bad.append("other inputs or D-170 verification than its A3 record")
    return bad


# ---- the per-pair blocks, bound and re-derived from the npz ------------------------------------------------------

def health_shares(cells: dict[str, Any]) -> dict[str, Any]:
    """outcomes.health_flag_share as phase3_a3.main() forms it from the cells."""
    return {pool: {"clean": cells["clean"][pool]["health"]["share_of_draws_gaussian_flag_2x"],
                   **{att: {f"rho={rho}": cells[att][pool][f"rho={rho}"]["health"]["share_of_draws_gaussian_flag_2x"]
                            for rho in RHOS} for att in A3.ATTACKERS}}
            for pool in POOLS}


def cell_values(z: dict[str, np.ndarray], pz: dict[str, np.ndarray], pool: str, att: str, rho: float) -> tuple[Any, Any]:
    """One P-A3.1 cell's realised and predicted TPR from the two npz, exactly as phase3_a3 forms them: realised, the
    D-156 mean of the per-draw seed-mean TPR of the pool's score repair (summaries' bs); predicted, the D-156 mean of
    the per-draw seed mean of the predicted TPR (score's band_summary of parr[...].mean(axis=1))."""
    realised = A3.bs(z[f"{att}|{pool}|rho={rho}|{SCORE_REPAIR[pool]}|tpr"])["mean"]
    raw = pz[f"{A3P.ckey(pool, att, rho)}|predicted_tpr"].mean(axis=1)
    return realised, P.band_summary(raw[1:], raw[0])["mean"]


def bind_p31(core: str, doc: dict[str, Any], z: dict[str, np.ndarray], pz: dict[str, np.ndarray]) -> None:
    """The record's 48 P-A3.1 per-cell rows, one per pool x attacker x rho with the pool's score repair, each with the
    realised TPR its npz gives and the predicted TPR its predictions npz gives (cell_values)."""
    per = doc["scoring"]["P-A3.1"]["per_cell"]
    want = [(pool, att, rho) for pool in POOLS for att in PREDICTED for rho in RHOS]
    keyed = {(r["pool"], r["attacker"], r["rho"]): r for r in per}
    if len(want) != CELLS_PER_PAIR_31 or len(per) != CELLS_PER_PAIR_31 or sorted(keyed) != sorted(want):
        refuse(f"{core}: the record's P-A3.1 cells are not the {CELLS_PER_PAIR_31} of pool x attacker x rho")
    for pool, att, rho in want:
        r = keyed[(pool, att, rho)]
        if r["repair"] != SCORE_REPAIR[pool]:
            refuse(f"{core} {pool} {att} rho={rho}: the cell's repair {r['repair']!r} is not the pool's score repair")
        realised, predicted = cell_values(z, pz, pool, att, rho)
        if realised is None or predicted is None or not (math.isfinite(realised) and math.isfinite(predicted)):
            refuse(f"{core} {pool} {att} rho={rho}: no finite realised or predicted TPR in the npz")
        if realised != r["realised_tpr"]:
            refuse(f"{core} {pool} {att} rho={rho}: the A3 npz does not give the record's realised TPR")
        if predicted != r["predicted_tpr"]:
            refuse(f"{core} {pool} {att} rho={rho}: the predictions npz does not give the record's predicted TPR (the "
                   f"D-156 mean over the draws of the per-draw seed-mean predicted TPR)")


def rederive(pr: dict[str, Any]) -> dict[str, Any]:
    """The pair's P-A3.1 cells bound one by one (bind_p31); its cells, minimum-rho table, A3-band outcomes and scoring
    recomputed from its npz by phase3_a3's own functions (and the predictions npz), each required equal to the
    record's, D-282 blocks aside (choice (b)); the predictions record's 'predicted' block recomputed from its npz; the
    anchor's A3-top minimum rhos against the re-derived ones."""
    doc, z, pz, core = pr["doc"], pr["z"], pr["pred"]["z"], pr["core"]
    cfg = doc["config"]
    n_rows, target = cfg["n_reps"] + 1, cfg["target_fpr"]
    short = sorted(k for k, v in z.items() if v.ndim == 0 or v.shape[0] != n_rows)
    if short:
        refuse(f"{core}: npz arrays without one row per draw ({n_rows}): {short[:6]}")
    shape = (n_rows, len(cfg["seeds"]))
    for pool in POOLS:
        for att in PREDICTED:
            for rho in RHOS:
                k = f"{A3P.ckey(pool, att, rho)}|predicted_tpr"
                if k not in pz or pz[k].shape != shape:
                    refuse(f"{core}: the predictions npz has no {k} of shape {shape}")
    if not same(A3P.predicted_cells({"target": target}, pz), pr["pred"]["doc"].get("predicted")):
        refuse(f"{core}: the predictions record's 'predicted' block is not phase3_a3_predictions.predicted_cells of "
               f"its npz")
    try:
        bind_p31(core, doc, z, pz)
        cells = A3.summaries(z, target)
        minrho = A3.min_rho_table(cells)
        bando = A3.band_outcomes(z)
        scoring = A3.score(cells, minrho, bando, pz, doc["implementation_checks"]["threshold_identity"])
    except MALFORMED as e:
        refuse(f"{core}: phase3_a3 cannot re-derive the record from its npz: {type(e).__name__} {e}")
    for blk in bando.values():                                   # as phase3_a3.main() removes them before writing
        for r in blk["repairs"].values():
            for k in ("_band_loss_mean", "_top_loss_mean", "_infeasible"):
                r.pop(k)
    rec_out = obj(doc.get("outcomes"))
    checks = (("cells", cells, doc.get("cells")),
              ("outcomes.min_rho_10pt_tpr_loss", minrho, rec_out.get("min_rho_10pt_tpr_loss")),
              ("outcomes.a3_band", bando, rec_out.get("a3_band")),
              ("outcomes.health_flag_share", health_shares(cells), rec_out.get("health_flag_share")),
              ("scoring", scoring, doc.get("scoring")))
    off = [name for name, mine, theirs in checks if not same(mine, strip_d282(theirs))]
    if off:
        refuse(f"{core}: the record's {off} are not phase3_a3's of its npz (D-282 blocks aside)")
    amr = doc["anchor"]["min_rho"]
    aoff = [k for k, v in amr.items() if v["a3_top_min_rho"] != minrho[k]["A3-top"]["min_rho"]]
    if aoff:
        refuse(f"{core}: the self-anchor's A3-top minimum rho is not the re-derived one: {aoff}")
    return {"cells": cells, "minrho": minrho, "bando": bando, "scoring": scoring}


# ---- the pooled lines ----------------------------------------------------------------------------------------------

def counts(values: list[bool | None], noun: str) -> dict[str, int]:
    """The counts D-352 (3) reports: all, decided, holding, failing, undecided."""
    return {f"n_{noun}": len(values), f"n_{noun}_decided": sum(x is not None for x in values),
            f"n_{noun}_holding": sum(x is True for x in values), f"n_{noun}_failing": sum(x is False for x in values),
            f"n_{noun}_undecided": sum(x is None for x in values)}


def pooled_verdict(values: list[bool | None], noun: str) -> dict[str, Any]:
    """D-352 (3)'s pooling rule over `values` (phase3_a3.decided_verdict): the verdict word, holds and the counts."""
    holds = A3.decided_verdict(list(values))
    return {"verdict": WORDS[holds], "holds": holds, "over": noun, **counts(list(values), noun), "rule": POOL_RULE}


def d170_status(prs: dict[str, dict[str, Any]], n_cells: dict[str, int], unit: str) -> dict[str, Any]:
    """The line's D-170 status: the per-pair flag the A3 records store and the line's cells of flagged pairs."""
    names = list(n_cells)
    iv = {n: obj(prs[n]["doc"].get("input_verification")) for n in names}
    flags = {n: iv[n].get("flag") for n in names}
    flagged = [n for n in names if flags[n] is not None]
    nf = int(sum(n_cells[n] for n in flagged))
    return {"per_pair_d170_flag": flags,
            "pin_flag_cleared_by": {n: obj(iv[n].get("pin_flag_cleared_by")).get("decision") for n in names
                                    if iv[n].get("pin_flag_cleared_by")},
            "unit": unit, "n_cells": int(sum(n_cells.values())), "n_flagged_pairs": len(flagged),
            "n_flagged_cells": nf,
            "status": ("every stored flag is cleared (None in every record of this line): D-331 made D-168 final and "
                       "its pinned record clears the D-170 flag of the judged pin-only inputs" if not flagged else
                       f"{len(flagged)} pair(s) carry the D-170 flag ({nf} of this line's {unit}): those cells enter "
                       "no paper table until the flag clears (D-170/D-172)"),
            "rule": D170_RULE}


def d170_of(prs: dict[str, dict[str, Any]], cells: list[dict[str, Any]]) -> dict[str, Any]:
    """d170_status over a line's cells (each with its 'pair')."""
    return d170_status(prs, {n: sum(x["pair"] == n for x in cells) for n in PAIRS}, "cells")


def expectation_met(share: float | None) -> bool | None:
    """D-332 (5)'s expectation on a line's share within x1.5 (None when the line has no cell)."""
    return None if share is None else bool(share >= EXPECTED_SHARE)


def cell_error(r: dict[str, Any]) -> float | None:
    """|log(realised / predicted)| of one P-A3.1 cell; None for a miss (a zero predicted or realised TPR), as
    phase3_a3.accuracy counts it."""
    re_, pr_ = r["realised_tpr"], r["predicted_tpr"]
    return float(abs(math.log(re_ / pr_))) if pr_ > 0 and re_ > 0 else None


def accuracy_line(rows: list[dict[str, Any]], role: str, label: str, d170: dict[str, Any] | None,
                  verdict: bool = False) -> dict[str, Any]:
    """D-177's accuracy line (phase3_a3.accuracy) over the rows; D-332's expectation is tested on the verdict line
    only (D-352 (3) registers it for the 432 cells), a beside line carries no expectation_met."""
    acc = A3.accuracy(rows)
    if verdict:
        return {"role": role, "label": label, **acc, "expectation": EXPECTATION,
                "expectation_met": expectation_met(acc["share_within_factor_1_5"]), "d170": d170}
    return {"role": role, "label": label, **acc, "expectation": NO_EXPECTATION, "d170": d170}


def p31(prs: dict[str, dict[str, Any]], der: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """P-A3.1: the 432 cells, the verdict line over all of them, and the two beside lines."""
    rows = []
    for c in PAIRS:
        per = der[c]["scoring"]["P-A3.1"]["per_cell"]
        if len(per) != CELLS_PER_PAIR_31:
            refuse(f"P-A3.1 has {len(per)} cells for {c}, not {CELLS_PER_PAIR_31}")
        for r in per:
            e = cell_error(r)
            rows.append({"pair": c, **r, "abs_log_realised_over_predicted": e,
                         "within_factor_1_5": bool(e is not None and e <= math.log(A3.FACTOR))})
    sets = {"all": rows, **{rep: [r for r in rows if r["repair"] == rep] for rep in ("quantile_transfer", "threshold_reset")}}
    got = {k: len(v) for k, v in sets.items()}
    if got != N31:
        refuse(f"P-A3.1 has {got} cells, not {N31}")
    return {"cells": "pair x pool x attacker (A3-top, A3-benign) x rho, the pool's score repair (phase3_a3.score's "
                     "per-cell rows, re-derived): realised = the D-156 mean over the draws of the seed-mean TPR, "
                     "predicted = the D-156 mean over the draws of the seed-mean predicted TPR F1bar(t')",
            "accuracy_line": "D-177 (phase3_a3.accuracy): share within x1.5, median |log(realised/predicted)|, n cells; "
                             "a zero predicted (or realised) TPR is a miss and counted (D-322 (4))",
            "scored": accuracy_line(sets["all"], VERDICT31, "every cell of the nine pairs", d170_of(prs, sets["all"]),
                                    verdict=True),
            "beside_quantile_transfer_cells_only": accuracy_line(
                sets["quantile_transfer"], "beside",
                "the quantile-transfer cells only (traffic): the cells where the prediction is not the repair's own rule",
                d170_of(prs, sets["quantile_transfer"])),
            "beside_threshold_reset_cells_only": accuracy_line(
                sets["threshold_reset"], "beside",
                "the threshold re-set cells only (R_same, R_fresh): agreement there is mechanical (the prediction is "
                "the repair's own rule, up to ties at the k-th place)", d170_of(prs, sets["threshold_reset"])),
            "rows": rows}


def split_key(key: str) -> dict[str, str]:
    pool, rep = key.split("|")
    return {"pool": pool, "repair": rep}


def pair_lines(cells: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each pair's verdict by D-352 (3)'s rule over its own cells, with the counts."""
    return {c: pooled_verdict([x["holds"] for x in cells if x["pair"] == c], "cells") for c in PAIRS}


def p32(prs: dict[str, dict[str, Any]], der: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """P-A3.2 over the 36 (pair, pool, repair) cells (D-352 (3)); the pairs' verdicts beside."""
    cells = [{"pair": c, **split_key(k), **v} for c in PAIRS for k, v in der[c]["scoring"]["P-A3.2"]["cells"].items()]
    if len(cells) != N32:
        refuse(f"P-A3.2 has {len(cells)} cells, not {N32}")
    per_pair = pair_lines(cells)
    d170 = d170_of(prs, cells)
    return {"rule": "A3-benign's minimum rho >= A3-top's, per (pair, pool, repair), restandardisation on traffic "
                    "included; a minimum rho of None reads '> 0.25' (D-244 (6)); a cell is decided unless both are "
                    "'> 0.25'",
            "scored": {"line": "in every pair and pool", **pooled_verdict([x["holds"] for x in cells], "cells"),
                       "d170": d170},
            "beside_over_pair_verdicts": {**pooled_verdict([v["holds"] for v in per_pair.values()], "pairs"),
                                          "note": "equals the verdict over cells (choice (c))", "d170": d170},
            "per_pair": per_pair,
            "beside_confound_benign_fit_row_share_per_seed": {
                "per_pair": {c: prs[c]["doc"].get("benign_fit_row_share_per_seed") for c in PAIRS},
                "label": "D-352 (2): about 80% of A3-benign's candidates are each seed's own D-117 fit rows, which "
                         "score low under the probe fitted on them; this weakens A3-benign and biases P-A3.2 toward "
                         "holding (rule 5)"},
            "cells": cells}


P33_KEYS = ("holds", "why_undecided", "band_infeasible_m0_every_draw", "a3_band_tpr_loss_at_largest_m",
            "a3_top_tpr_loss_at_same_m", "clean_pool_gaussian_flag_share", "comparison_over_draws_with_m_gt_0_beside")


def p33(prs: dict[str, dict[str, Any]], der: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """P-A3.3 per pair over its cells (an infeasible band undefined, with its reason), pooled over the nine pair
    verdicts (D-352 (3)); beside, the infeasible cells, the clean-pool flag shares and D-352 (2)'s finding."""
    cells = [{"pair": c, **split_key(k), **{q: v[q] for q in P33_KEYS}}
             for c in PAIRS for k, v in der[c]["scoring"]["P-A3.3"]["cells"].items()]
    if len(cells) != N33:
        refuse(f"P-A3.3 has {len(cells)} cells, not {N33}")
    per_pair = pair_lines(cells)
    infeasible = [{q: x[q] for q in ("pair", "pool", "repair", "why_undecided")} for x in cells
                  if x["band_infeasible_m0_every_draw"] is True]
    finding = [{"pair": c, **split_key(k)} for c in PAIRS
               for k in der[c]["scoring"]["P-A3.3"]["finding_band_admits_no_attacker_where_4c_fires_on_the_clean_pool"]["cells"]]
    d170 = d170_of(prs, cells)
    for c in PAIRS:
        per_pair[c]["n_cells_band_infeasible"] = sum(x["pair"] == c for x in infeasible)
    return {"rule": "per pair (phase3_a3 choice (m)): A3-band's mean TPR loss at its largest m < A3-top's at the same "
                    "per-draw m, per (pool, repair); a cell whose band is infeasible (A3-band's m 0 in every draw) is "
                    "undefined with its reason, not failed (D-352 (2)); the pair verdict by D-352 (3)'s rule over its "
                    "cells; pooled 'in every pair' by the same rule over the pairs",
            "scored": {"line": "in every pair", **pooled_verdict([v["holds"] for v in per_pair.values()], "pairs"),
                       "d170": d170},
            "beside_over_cells": {**pooled_verdict([x["holds"] for x in cells], "cells"),
                                  "note": "equals the verdict over pairs (choice (c))", "d170": d170},
            "per_pair": per_pair,
            "beside": {"n_cells_band_infeasible": len(infeasible), "infeasible_cells": infeasible,
                       "clean_pool_gaussian_flag_share": {c: {pool: der[c]["bando"][pool]["clean_pool_gaussian_flag_share"]
                                                              for pool in POOLS} for c in PAIRS},
                       "finding_band_admits_no_attacker_where_4c_fires_on_the_clean_pool": {
                           "cells": finding, "n_cells": len(finding),
                           "label": "a finding (D-352 (2)), not a verdict: the infeasible cells whose clean pool "
                                    "already crosses the 4c cut in every draw"},
                       "d170": d170},
            "cells": cells}


def p34(prs: dict[str, dict[str, Any]], der: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """P-A3.4 over the 27 (pair, score repair) cells by the same pooling rule; the pairs' verdicts beside."""
    cells = [{"pair": c, **split_key(k), **v} for c in PAIRS for k, v in der[c]["scoring"]["P-A3.4"]["cells"].items()]
    if len(cells) != N34:
        refuse(f"P-A3.4 has {len(cells)} cells, not {N34}")
    per_pair = pair_lines(cells)
    d170 = d170_of(prs, cells)
    return {"rule": f"A3-top's minimum rho on restandardisation (traffic) >= {A3.P34_FACTOR:g} x A3-top's of each score "
                    f"repair (relative tolerance {A3.RHO_TOL:g}); None reads '> 0.25'; a cell the grid cannot decide "
                    "is undecided; no new measurement (A3-top equals 4d B's A1 bit for bit, D-322)",
            "scored": {"line": "every pair and score repair", **pooled_verdict([x["holds"] for x in cells], "cells"),
                       "d170": d170},
            "beside_over_pair_verdicts": {**pooled_verdict([v["holds"] for v in per_pair.values()], "pairs"),
                                          "note": "equals the verdict over cells (choice (c))", "d170": d170},
            "per_pair": per_pair,
            "cells": cells}


# ---- the scoring -----------------------------------------------------------------------------------------------------

def pool(rdir: Path, scorer_sha: dict[str, str]) -> tuple[dict[str, Any], dict[str, tuple[Path, str]]]:
    """Every refusal, the binding and re-derivation, then P-A3.1 to P-A3.4. Returns the record's body and the input
    files (name -> path, sha256). A malformed field anywhere refuses (MALFORMED), never a partial score."""
    try:
        return _pool(rdir, scorer_sha)
    except MALFORMED as e:
        refuse(f"a malformed input record (no pooled line is formed): {type(e).__name__}: {e}")
        raise                                                        # unreachable: refuse() raises


def _pool(rdir: Path, scorer_sha: dict[str, str]) -> tuple[dict[str, Any], dict[str, tuple[Path, str]]]:
    paths = discover(rdir)
    files: dict[str, tuple[Path, str]] = {}
    prs = {c: load(c, p, files) for c, p in paths.items()}
    bad = {c: b for c, b in ((c, record_problems(c, pr["doc"])) for c, pr in prs.items()) if b}
    if bad:
        refuse(f"not every A3 record can enter the pooled scoring: {bad}")
    params = {c: {k: pr["doc"]["config"].get(k) for k in ("n_reps", "seeds", "target_fpr")} for c, pr in prs.items()}
    if any(v != params[PAIRS[0]] for v in params.values()):
        refuse(f"the nine records were run at different draws, seeds or target: {params}")
    code = shared_code(prs, scorer_sha)
    pbad = {c: b for c, b in ((c, prediction_problems(c, pr, code["predictions_code_sha256"])) for c, pr in prs.items())
            if b}
    if pbad:
        refuse(f"not every predictions record is its A3 record's: {pbad}")
    der = {c: rederive(prs[c]) for c in PAIRS}
    records = {c: {"file": pr["file"], "sha256": pr["sha256"], "draws_npz": pr["draws_npz"],
                   "commit": obj(pr["doc"].get("code")).get("commit"),
                   "d170_flag": obj(pr["doc"].get("input_verification")).get("flag"),
                   "implementation_checks": {k: pr["doc"]["implementation_checks"][k]["holds"] for k in IMPL_CHECKS},
                   "predictions_record": {"file": pr["pred"]["file"], "sha256": pr["pred"]["sha256"],
                                          "draws_npz": pr["pred"]["draws_npz"],
                                          "commit": obj(pr["pred"]["doc"].get("code")).get("commit")}}
               for c, pr in prs.items()}
    anchor = {"rule": "per pair (D-332 (6)); every pair's anchor passed, else the scorer refuses (choice (f))",
              "per_pair": {c: {**{k: pr["doc"]["anchor"].get(k) for k in ("checked", "identical", "n_values_identical",
                                                                          "n_values_compared", "min_rho")},
                               "fourd_b_record": {"file": pr["doc"]["anchor"].get("file"),
                                                  "sha256": pr["doc"]["anchor"].get("sha256")}}
                           for c, pr in prs.items()}}
    return {"parameters": params[PAIRS[0]], "a3_code": code, "records": records, "self_anchor": anchor,
            "P-A3.1": p31(prs, der), "P-A3.2": p32(prs, der), "P-A3.3": p33(prs, der), "P-A3.4": p34(prs, der)}, files


def verdict_lines(body: dict[str, Any]) -> list[str]:
    """The pooled lines the job prints once the record is written (each with its D-170 status)."""
    out = []
    for name in ("scored", "beside_quantile_transfer_cells_only", "beside_threshold_reset_cells_only"):
        a = body["P-A3.1"][name]
        exp = (f"expectation met {a['expectation_met']}" if "expectation_met" in a else "no expectation: beside, not a verdict")
        out.append("P-A3.1 {}: {} cells, {} predicted zero, {} realised zero (predicted positive); within x1.5 {}; "
                   "median |log| {}; {} ({}); D-170 flagged pairs {}".format(
                       name, a["n_cells"], a["n_cells_predicted_zero"], a["n_cells_realised_zero_predicted_positive"],
                       a["share_within_factor_1_5"], a["median_abs_log_realised_over_predicted"], exp, a["role"],
                       a["d170"]["n_flagged_pairs"]))
    for p, noun in (("P-A3.2", "cells"), ("P-A3.3", "pairs"), ("P-A3.4", "cells")):
        s = body[p]["scored"]
        out.append("{} {}: {} ({} {} decided: {} holding, {} failing; {} undecided); D-170 flagged pairs {}".format(
            p, s["line"], s["verdict"], s[f"n_{noun}_decided"], noun, s[f"n_{noun}_holding"], s[f"n_{noun}_failing"],
            s[f"n_{noun}_undecided"], s["d170"]["n_flagged_pairs"]))
    out.append("P-A3.3 beside: {} band-infeasible cells of {}".format(
        body["P-A3.3"]["beside"]["n_cells_band_infeasible"], N33))
    return out


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Phase 3 A3: the pooled P-A3.1 to P-A3.4 scoring (D-352 (3))")
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
    payload = {**provenance_block("phase3_a3_pooled"), "preregistration": PREREG,
               "preregistration_text": {"entry": "DECISIONS.md D-352 (3)", "text": PREREG_TEXT},
               "code": code0, "config": CONFIG, "config_hash": canonical_hash(CONFIG),
               "input_files_sha256": {n: h for n, (_, h) in files.items()}, "choices": CHOICES, **body}
    digest = write_results(out, payload, exclusive=True)
    for line in verdict_lines(body):                                 # printed only once the record is written
        print(line, flush=True)
    print(f"Written: {out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
