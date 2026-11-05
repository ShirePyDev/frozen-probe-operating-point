"""Phase 3 A3's two sensitivities (D-352 (4), approved by D-369): SA, the clean-relative band, and SB, the held-out
benign attacker, one pair per run, with A3-top computed alongside as their comparator. Reported beside the A3 records,
never primary (D-369 (2)). Reuses src/phase3_a3.py's and src/phase3_a3_predictions.py's functions unchanged (their
sha256 are pinned by the pooled scorer) and src/poolstudy_d156.py's machinery through them.

SPECIFICATION (D-352 (4), unchanged by D-369 (1); written before any A3 number existed):
  SA  R_fresh and 1% traffic only (on R_same the defender holds no clean reference): A3-band's greedy, order, rho grid
      (0.001 .. 0.25) and outcomes unchanged (phase3_a3 choices (k)-(m)), but an item is added only while the Gaussian
      health ratio of the poisoned pool computed against the CLEAN pool's post moments (in place of cal_d's pre
      moments) at T_d stays strictly inside (0.5, 2): phase3_a3.band_greedy itself, with the clean pool's post scores
      passed as its reference (its pre_cal argument), so the screen and every decision are healthcheck_d156.health
      with that reference (choice (r)).
  SB  A3-benign with, per seed, the candidates restricted to that seed's D-117 held-out (ho20) probe_train negatives
      (paired.d117_split's ho: the 20% the seed's probe was not fitted on), ranked by that seed's post-update score,
      injected into that seed's pools (R_same, R_fresh, traffic); outcomes as A3-benign's (choice (s)).
COMPARATOR: A3-top (4d B's A1) by phase3_a3.measure itself, self-anchored exactly as phase3_a3.main() does
  (phase3_a3.anchor_check against 4d B's record of record: the four repairs' minimum rho and every D-156 summary of the
  frozen arm, the unpoisoned repairs and A3-top at every rho; a mismatch refuses, nothing written), computed before any
  SA or SB number (D-332 (6)). This script reads no A3 record and no A3 predictions record (D-352 (3), D-369 (2)):
  ENFORCED just before each write, which refuses unless the results read (provenance's registered reads) are exactly
  the expected ones (per pair: the selection, depth and 4d B records; pooled: the nine sensitivity records) and none
  is an A3 record (a name matching phase3_a3_ not followed by sens_).
SCORING per pair (D-369 (2): D-352 (3)'s rules):
  P-A3.2 for SB against A3-top: SB's minimum rho (10-point TPR loss, 4d B's rule) >= A3-top's, per pool and repair,
         restandardisation included (phase3_a3 choice (n)): phase3_a3.at_least ('> 0.25' reading, D-244 (6)) and
         decided_verdict (D-352 (3): False if a decided cell fails, True if every decided cell holds and one is
         decided, None if none is decided).
  P-A3.3 for SA against A3-top: SA's mean TPR loss at its largest m < A3-top's at the same per-draw m, per pool and
         repair (R_fresh threshold re-set; traffic quantile transfer and restandardisation): phase3_a3.p33_cell (a cell
         where SA's m is 0 in every draw is UNDEFINED with its reason, not failed, D-352 (2)) and decided_verdict.
  D-282: the combined interval on every threshold-level arm's TPR and on every TPR loss reported on one (A3-top's,
         SB's and SA's at their minimum rho; SA's and A3-top's at SA's largest m), paired within item and draw, by
         phase3_a3.d282_replicates / d282_loss (phase3_a3 choice (p)).
POOLED (--pool; D-369 (2): 'the same pooled rules as D-352 (3)', defined here before any per-pair sensitivity record
  exists): reads the nine results/phase3_a3_sens_<pair>.json and their npz, each bound by sha256 (the npz loaded from
  the very bytes hashed; the records sharing one code block, equal to this tree's, at one set of draws, seeds and
  target; each a registered run that passed its self-anchor and SA band check), re-derives every per-pair P-A3.2 and
  P-A3.3 cell from its npz with this file's functions (derive, score) and refuses unless the record's cells and pair
  verdicts equal them; then
  P-A3.2 pooled over the 36 (pair, pool, repair) cells (4 per pair): True if every decided cell holds and at least one
         is decided, False if any decided cell fails, None if none is decided (D-352 (3)); decided, holding, failing
         and undecided counts; each pair's verdict beside.
  P-A3.3 per pair as D-352 (2)-(3) (phase3_a3.p33_cell, decided_verdict), then 'in every pair' by the same rule over the
         nine pair verdicts (a pair with no decided cell is undecided); the infeasible-cell count, and each pair's
         clean-pool 4c and clean-relative flag shares, beside.
  Writes results/phase3_a3_sens_pooled.json (exclusive; provenance.write_results). Its lane log prints check and
  verdict words only.
IMPLEMENTATION CHECK (D-181, D-346's pattern; exit 3, nothing written): every SA pool with m > 0 is strictly inside
  (0.5, 2) by health() against the clean pool's post scores, and the greedy's screen agreed with health() on every
  item it added (phase3_a3.band_checks' rule).
REFUSALS (nothing written): an existing output; --out under results/; the final name without the registered draws,
  seeds and target at the 6-thread BLAS pin, or with --anchor-only; a 4d B record that is not the run of these inputs;
  A3-top not reproducing 4d B's A1; a seed whose held-out rows are not the complement of its fit rows, an SB candidate
  that is a fit row of its seed, or a seed with fewer SB candidates than rho 0.25's budget; inputs or code changed
  during the run; an A3 record read, or results read other than the expected ones (choice (x)). Pooled: a missing or
  extra pair, a record or npz that is not the bytes named, a record that is not a registered, self-anchored run with
  its SA band check passed and no A3 record read (and, for the final name, at the 6-thread pin), records with another
  code block (or not this tree's) or other parameters, per-pair cells that are not their npz's.
CHOICES BY THE CLOSEST D-ENTRY (D-181; also in the record's implementation_choices, beside (a)-(q) of the A3 scripts):
  (r) SA's reference is the clean pool's post scores (the pool before any injection: R_fresh's 1000, or the 1% traffic
      pool with its 10 harmful items) at T_d (cal_d's pre 1% threshold); phase3_a3.band_greedy and health() unchanged.
      The clean pool against itself has ratio 1, so SA starts inside its band in every draw.
  (s) SB per seed: the seed's ho20 rows are paired.d117_split(n_probe_train, seed)[0]; the candidates are the probe_train
      negatives among them, ranked by np.argsort(-x) on that seed's post score (phase3_a3_predictions choice (e)'s
      expression, per seed); seed s's pool at rho is the clean pool plus its own first m; every repair runs per seed on
      that seed's pool (the threshold re-set and quantile transfer per seed as in phase3_a3.repair_flags; the
      restandardisation's feature statistics from that seed's pool); FPR, the paired ratio and TPR are the seed means
      of phase3_a1.arm_numbers, the health ratio the mean over seeds of per-seed ratios (phase3_a3 choice (j)). SB has
      no fit-row confound: no candidate is a fit row of its seed (checked; phase3_a3_predictions choice (f) does not
      arise). A confound with a direction (rule 5): the ho20 negatives are D-117's selection data (C and layer were
      chosen on mean ho20 AUROC on the pre model), so they are mildly selected to score low, which weakens SB slightly
      and biases P-A3.2 toward holding: the same direction as A3-benign's fit-row confound, and smaller. A seed with
      fewer than round(0.25 n) candidates refuses (as A3-benign's build refuses).
  (t) SA's outcomes are A3-band's (D-332 (4), D-352 (4) 'outcomes unchanged'): realised m per rho, TPR and FPR/target,
      the minimum rho for a 10-point TPR loss (phase3_a3.min_rho_table's rule, on SA's three repairs), the largest m
      and the loss there with A3-top's at the same per-draw m; the health ratio and flag reported at each rho are 4c's
      registered check (cal_d's pre moments, phase3_a3 choice (j)); SA's own clean-relative ratio and flag stand beside
      (health_clean_ref), with the clean pool's (1 by construction).
  (u) P-A3.3's undefined reason for SA comes from the clean pool's clean-relative flag share (0: SA's band can only be
      infeasible because no harmful item keeps the pool inside); the clean pool's 4c flag share stands beside each
      cell (D-352 (3) 'each pair's clean-pool flag share beside'). phase3_a3's finding 'the band admits no attacker
      where 4c already fires on the clean pool' does not apply to SA (its reference is the clean pool itself).
  (v) SB's minimum rho is phase3_a3.min_rho_table itself with SB's cells in A3-benign's place (SB is A3-benign with
      other candidates; 'outcomes as A3-benign's'), relabelled; SB's D-156 cell summaries likewise phase3_a3.summaries.
  (w) the per-pair lane log prints checks, counts and verdict-free words only, no SA, SB or P-A3.x number (D-346 (4),
      D-352 (3)); the pooled run prints check and verdict words only.
Writes results/phase3_a3_sens_<pair>.json and results/phase3_a3_sens_<pair>_draws.npz (tracked, D-163 (6)), once
(exclusive), with its config's hash (config_hash) and the sandbox disclosure (sandbox_disclosure). Only registered
parameters at the 6-thread BLAS pin write the final names.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_d156 as D  # noqa: E402
import healthcheck_d156 as HC  # noqa: E402
import paired as P  # noqa: E402
import phase3_a1 as A1  # noqa: E402
import phase3_a1_predictions as A1P  # noqa: E402
import phase3_a3 as A3  # noqa: E402
import phase3_a3_predictions as A3P  # noqa: E402
import populations as POP  # noqa: E402
import poolstudy_d156 as PS  # noqa: E402
import provenance as PV  # noqa: E402  (PV._RESULT_READS: the results this process read, D-297 (1))
from provenance import (  # noqa: E402
    canonical_hash, ensure_utf8_stdout, provenance_block, read_result, sha256_bytes, sha256_file, write_results,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
THIS_FILE = "src/phase3_a3_sens.py"
CODE_FILES = [THIS_FILE, *A3.CODE_FILES]
PAIRS = tuple(A1P.core_of(p) for p in POP.UPDATE_PAIRS_REPAIR)      # the nine released pairs (D-332 (1))
A3_RECORD = re.compile(r"phase3_a3_(?!sens_)")                      # an A3 record or A3 predictions record
POOLED_NAME = "phase3_a3_sens_pooled.json"
P32, P33 = "P-A3.2 (SB against A3-top)", "P-A3.3 (SA against A3-top)"
POOLS, RHOS = A3P.POOLS, A3P.RHOS
VARIANTS = ("SB", "SA")
SB_POOLS = POOLS                                                     # D-352 (4): A3-benign's pools
SA_POOLS = ("R_fresh", "traffic_1pct")                               # D-352 (4): R_fresh and 1% traffic only
SB_REPAIRS = A3.REPAIRS                                              # P-A3.2's cells (phase3_a3 choice (n))
SA_REPAIRS = tuple(r for r in A3.REPAIRS if r[0] in SA_POOLS)        # P-A3.3's cells
SA_CMP = "A3-top@SA_m"                                               # A3-top at SA's per-draw largest m
CLEAN_REF = "clean_ref"                                              # SA's ratio: against the clean pool's post moments
PREREG = ("D-352 (4) (the two sensitivities SA and SB, specified completely before any A3 number existed); D-369 (the "
          "author's approval, D-357 (B)(6): run exactly as D-352 (4) specified, before any A3 number of record is read, "
          "reported beside the A3 records and never primary, scored by D-352 (3)'s rules: P-A3.2 for SB against A3-top, "
          "P-A3.3 for SA against A3-top); D-352 (2)-(3) (an infeasible band undefined, not failed; the decided/undecided "
          "reading); D-332 (the Phase 3 A3 pre-registration); D-318 (9); machinery of D-244/D-298 (4d B), D-241/D-264 "
          "(4c), D-117 (ho20), D-282 (TPR interval)")
SANDBOX = ("D-352 (5)'s pattern (D-346 (3)): at 4df53d8 (branch a3sens; src unchanged in 4b3b334 apart from this "
           "disclosure), before review, a node sandbox ran this script on llama31: the A3-top self-anchor at 200 draws "
           "(--anchor-only: 4d B's A1 values, reproduced) and the full run at 5 draws, to check that it runs; nobody read "
           "its SA, SB or P-A3.2 / P-A3.3 fields; every record of the lane is written fresh by the reviewed code, and no "
           "sandbox figure is quoted")
PREREG_POOLED = ("D-369 (2) (SA and SB scored by the same pooled rules as D-352 (3): P-A3.2 for SB against A3-top, "
                 "P-A3.3 for SA against A3-top); D-352 (3) (the pooled reading: True if every decided cell holds and at "
                 "least one is decided, False if any decided cell fails, None if none is decided; P-A3.3 per pair, then "
                 "'in every pair' by the same rule over pairs, the infeasible cells and each pair's clean-pool flag "
                 "share beside); D-352 (2) (an infeasible band undefined, not failed); D-244 (6) ('> 0.25'); this "
                 "pooled mode was written before any results/phase3_a3_sens_<pair>.json existed")
CHOICES = {
    "r": "SA's reference is the clean pool's post scores (the pool before any injection) at T_d: phase3_a3.band_greedy "
         "and healthcheck_d156.health unchanged, the clean pool's post scores in place of cal_d's pre scores (D-352 "
         "(4)); the clean pool against itself has ratio 1, so SA starts inside its band in every draw",
    "s": "SB per seed: ho20 = paired.d117_split(n_probe_train, seed)[0]; candidates the probe_train negatives among "
         "them, ranked by np.argsort(-x) on that seed's post score (choice (e)'s expression, per seed); seed s's pool is "
         "the clean pool plus its own first m; every repair per seed on that seed's pool (restandardisation's feature "
         "statistics from that seed's pool); FPR, paired ratio and TPR the seed means of phase3_a1.arm_numbers, the "
         "health ratio the mean of per-seed ratios (choice (j)); no candidate is a fit row of its seed (checked), so "
         "choice (f)'s confound does not arise; a confound with a direction (rule 5): the ho20 negatives are D-117's "
         "selection data (C and layer chosen on mean ho20 AUROC on the pre model), so they are mildly selected to score "
         "low, which weakens SB slightly and biases P-A3.2 toward holding, the same direction as A3-benign's confound "
         "and smaller; fewer than round(0.25 n) candidates in a seed refuses",
    "t": "SA's outcomes are A3-band's (D-332 (4); D-352 (4) 'outcomes unchanged'): realised m per rho, TPR and "
         "FPR/target, the loss at the largest m with A3-top's at the same per-draw m; the health ratio and flag reported "
         "at each rho are 4c's registered check (cal_d's pre moments, choice (j)); SA's clean-relative ratio and flag "
         "stand beside (health_clean_ref), with the clean pool's",
    "u": "P-A3.3 for SA: phase3_a3.p33_cell with the clean pool's clean-relative flag share choosing the undefined "
         "reason (0 by construction: SA is infeasible only when no harmful item keeps the pool inside); the clean pool's "
         "4c flag share beside each cell (D-352 (3)); phase3_a3's finding on cells where 4c fires on the clean pool does "
         "not apply to SA",
    "v": "SB's minimum rho and D-156 cell summaries are phase3_a3.min_rho_table and summaries themselves with SB's "
         "arrays in A3-benign's place (SB is A3-benign with other candidates, 'outcomes as A3-benign's'), relabelled",
    "w": "the per-pair lane log prints checks, counts and words only, no SA, SB or P-A3.x number (D-346 (4); D-352 "
         "(3)); the pooled run prints check and verdict words only",
    "x": "the 'reads no A3 record' rule is enforced before each write: the registered results reads must be exactly the "
         "expected files (per pair the selection, depth and 4d B records; pooled the nine sensitivity records) and none "
         "may match phase3_a3_ not followed by sens_ (D-352 (3), D-369 (2))",
    "y": "pooled (D-369 (2), D-352 (3)): every per-pair P-A3.2 and P-A3.3 cell and pair verdict re-derived from the "
         "record's npz (bound by sha256) with derive and score and required equal to the record's; P-A3.2 pooled over "
         "the 36 (pair, pool, repair) cells by decided_verdict; P-A3.3 per pair, then over the nine pair verdicts by "
         "decided_verdict; a pair whose P-A3.3 is None counts as undecided",
}


# ---- SB's candidates ------------------------------------------------------------------------------------------------

def sb_candidates(S: dict[str, Any]) -> dict[str, Any]:
    """SB's injection order per seed (D-352 (4); choice (s)), as score-table indices (A3-benign's layout: the i-th
    probe_train negative at n_pool + n_h + i): the probe_train negatives in the seed's D-117 held-out 20%, highest post
    score of THAT seed first. Refuses if the seed's split is not a partition of probe_train into held-out and fit rows,
    if its fit rows are not the rows its probe was fitted on (so a candidate could be a fit row), if a candidate is not
    a negative, or if a seed has fewer candidates than rho 0.25's budget."""
    n_tr = len(S["y_tr"])
    ptr_neg = np.asarray(S["ptr_neg"])
    b0 = S["n_pool"] + S["n_h_src"]
    m_need = max(S["m_max"].values())
    orders, n_cand = [], []
    h = hashlib.sha256()
    for j, seed in enumerate(S["seeds"]):
        ho, fit = P.d117_split(n_tr, seed)
        if not np.array_equal(np.sort(np.concatenate([ho, fit])), np.arange(n_tr)):
            raise SystemExit(f"REFUSING: seed {seed}'s D-117 split is not a partition of probe_train (held-out, fit)")
        if not np.array_equal(np.sort(fit), np.sort(np.asarray(S["rows_of"][j]))):
            raise SystemExit(f"REFUSING: seed {seed}'s D-117 fit rows are not the rows its probe was fitted on")
        cand = np.flatnonzero(np.isin(ptr_neg, ho))                 # positions among the probe_train negatives
        if np.any(S["y_tr"][ptr_neg[cand]]):
            raise SystemExit(f"REFUSING: an SB candidate of seed {seed} is not a probe_train negative")
        order = b0 + cand[np.argsort(-S["SC"]["post"]["B"][cand, j])]
        if len(order) < m_need:
            raise SystemExit(f"REFUSING: seed {seed} has {len(order)} held-out probe_train negatives; rho 0.25 needs "
                             f"{m_need}")
        orders.append(order)
        n_cand.append(int(len(order)))
        h.update(np.asarray(order, dtype="<i8").tobytes())
    return {"orders": orders, "n_candidates_per_seed": n_cand, "sha256": h.hexdigest()}


# ---- the measurement --------------------------------------------------------------------------------------------------

def repair_flags_seeds(S: dict[str, Any], X: dict[str, Any], rep: str, T: np.ndarray, E: np.ndarray, XE: np.ndarray,
                       tidx: np.ndarray, idxs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """phase3_a3.repair_flags with seed s's own pool idxs[s] (SB injects per seed, choice (s)): column s is
    phase3_a3.repair_flags(..., idxs[s])'s column s (the same expressions, one seed at a time)."""
    SC, target = S["SC"], S["target"]
    nS = len(S["seeds"])
    if len(idxs) != nS:
        raise ValueError("repair_flags_seeds: one pool per seed")
    if rep == "threshold_reset":
        t = np.array([P.thr_at_fpr(S["post_table"][idxs[s], s], target) for s in range(nS)])
    elif rep == "quantile_transfer":
        t = np.array([P.quantile_transfer(S["pre_table"][tidx, s], S["post_table"][idxs[s], s], T[s]) for s in range(nS)])
    elif rep == "restandardise":
        FE, QP = [], []
        for s, q in enumerate(S["probes"]):
            mu, sd = PS.feature_stats(X["ALL_post"][idxs[s]])
            FE.append(PS.restandardised_scores(q, mu, sd, XE) >= T[s])
            QP.append(PS.restandardised_scores(q, mu, sd, X["XP"]) >= T[s])
        return np.column_stack(FE), np.column_stack(QP), None
    else:
        raise ValueError(f"unknown repair {rep!r}")
    return SC["post"]["pool"][E] >= t, SC["post"]["pos"] >= t, t


def pool_scores(pt: np.ndarray, idxs: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """The pool's post scores (n, S): one pool for every seed, or seed s's column from its own pool idxs[s]."""
    if isinstance(idxs, np.ndarray):
        return pt[idxs]
    return np.column_stack([pt[idx, s] for s, idx in enumerate(idxs)])


def measure_sens(S: dict[str, Any], X: dict[str, Any], which: tuple[str, ...], sb: dict[str, Any] | None = None,
                 log: Callable[[str], None] = print) -> dict[str, np.ndarray]:
    """Per draw (row 0 the frozen split), for `which` (from 'SB', 'SA'): every per-draw array, in phase3_a3.measure's
    layout. Keys: SB|<pool>|rho=<rho>|<repair or health>|<q>; SA|<pool>|rho=<rho>|m_realised, SA|<pool>|rho=<rho>|
    <repair, health (4c) or clean_ref|health>|<q>, SA|<pool>|greedy|<stat>, clean|<pool>|clean_ref|health|<q> (the
    clean pool against itself), A3-top@SA_m|<pool>|<repair>|<q> and A3-top@SA_m|<pool>|m."""
    target, SC, pt = S["target"], S["SC"], S["post_table"]
    if "SB" in which and sb is None:
        raise ValueError("measure_sens: SB needs sb_candidates(S)")
    acc: dict[str, list[Any]] = {}
    put = lambda k, x: acc.setdefault(k, []).append(x)  # noqa: E731
    t0 = time.time()
    for d, dr in enumerate(S["draws"]):
        E, T = dr["E"], S["T"][d]
        F_pre = SC["pre"]["pool"][E] >= T
        pre_cal = SC["pre"]["pool"][dr["cal"]]
        XE = X["PN_post"][E]

        def arms(prefix: str, pool: str, tidx: np.ndarray, idxs: np.ndarray | list[np.ndarray]) -> None:
            for rep in A3.REPAIRS_OF[pool]:
                if isinstance(idxs, np.ndarray):                     # one pool for every seed: phase3_a3's own arm
                    F, Q, t = A3.repair_flags(S, X, rep, T, E, XE, tidx, idxs)
                else:
                    F, Q, t = repair_flags_seeds(S, X, rep, T, E, XE, tidx, idxs)
                r = A1.arm_numbers(F_pre, F, Q, target)
                for qn in A3.QUANT:
                    put(f"{prefix}|{rep}|{qn}", r[qn])
                if t is not None:
                    put(f"{prefix}|{rep}|threshold_per_seed", t)

        def health(prefix: str, ref: np.ndarray, idxs: np.ndarray | list[np.ndarray]) -> None:
            h = HC.health(ref, pool_scores(pt, idxs), T)
            for qn in A3.HEALTH:
                put(f"{prefix}|health|{qn}", h[qn])

        if "SB" in which:
            for pool in SB_POOLS:
                tidx = dr[A3P.POOL_OF_DRAW[pool]]
                for rho in RHOS:
                    m = A3P.m_of(rho, len(tidx))
                    idxs = [np.concatenate([tidx, o[:m]]) for o in sb["orders"]]
                    arms(f"SB|{pool}|rho={rho}", pool, tidx, idxs)
                    health(f"SB|{pool}|rho={rho}", pre_cal, idxs)
        if "SA" in which:
            for pool in SA_POOLS:
                tidx = dr[A3P.POOL_OF_DRAW[pool]]
                ref = pt[tidx]                                       # choice (r): the clean pool's post scores
                order = S["top_order"][pool][d]                      # A3-band's candidates: A3-top's order
                g = A3.band_greedy(ref, pt[tidx], pt[order], T, S["m_max"][pool])
                seq = order[g["chosen"]]
                put(f"SA|{pool}|greedy|length", float(len(seq)))
                put(f"SA|{pool}|greedy|stopped_no_fit", float(g["stopped_no_fit"]))
                put(f"SA|{pool}|greedy|n_health_decisions", float(g["n_direct"]))
                put(f"SA|{pool}|greedy|screen_max_rel_err", g["screen_max_rel_err"])
                put(f"SA|{pool}|greedy|screen_disagreements", float(g["screen_disagreements"]))
                health(f"clean|{pool}|{CLEAN_REF}", ref, tidx)
                for rho in RHOS:
                    md = min(A3P.m_of(rho, len(tidx)), len(seq))
                    idx = np.concatenate([tidx, seq[:md]])
                    put(f"SA|{pool}|rho={rho}|m_realised", float(md))
                    arms(f"SA|{pool}|rho={rho}", pool, tidx, idx)
                    health(f"SA|{pool}|rho={rho}", pre_cal, idx)                    # 4c's registered check
                    health(f"SA|{pool}|rho={rho}|{CLEAN_REF}", ref, idx)            # SA's own constraint
                put(f"{SA_CMP}|{pool}|m", float(len(seq)))                           # P-A3.3 (choice (m))
                arms(f"{SA_CMP}|{pool}", pool, tidx, np.concatenate([tidx, order[:len(seq)]]))
        if d % 50 == 0:
            log(f"  {'+'.join(which)}: draw {d} of {len(S['draws']) - 1} ({time.time() - t0:.0f} s)")
    out = {k: np.asarray(v, dtype="<f8") for k, v in acc.items()}
    for k in [k for k in out if k.endswith("|health|gaussian_ratio")]:
        out[k[:-len("gaussian_ratio")] + "gaussian_flag"] = HC.crosses(out[k], A3.CUT).astype("<f8")
    return out


# ---- summaries, outcomes, checks ------------------------------------------------------------------------------------

def sb_summaries(a: dict[str, np.ndarray], target: float = P.TARGET_FPR) -> dict[str, Any]:
    """SB's D-156 cell blocks: phase3_a3.summaries on SB's arrays in A3-benign's place (choice (v))."""
    alias = {"A3-benign" + k[len("SB"):]: v for k, v in a.items() if k.startswith("SB|")}
    return A3.summaries(alias, target)["A3-benign"]


def sa_summaries(a: dict[str, np.ndarray]) -> dict[str, Any]:
    """SA's D-156 cell blocks per pool and rho, as phase3_a3.summaries' A3-band blocks (m budget and realised, repairs,
    4c health), with SA's clean-relative ratio beside (health_clean_ref; choice (t))."""
    blk: dict[str, Any] = {}
    for pool in SA_POOLS:
        for rho in RHOS:
            p = f"SA|{pool}|rho={rho}"
            mr = a[f"{p}|m_realised"]
            blk.setdefault(pool, {})[f"rho={rho}"] = {
                "m": {"budget": A3P.m_of(rho, PS.DEFAULT_POOL), "realised": A3.bs(mr),
                      "realised_min_over_draws": int(mr[1:].min()), "realised_max_over_draws": int(mr[1:].max())},
                "repairs": {rep: {qn: A3.bs(a[f"{p}|{rep}|{qn}"]) for qn in A3.QUANT} for rep in A3.REPAIRS_OF[pool]},
                "health": A3.health_block(a, p),
                "health_clean_ref": A3.health_block(a, f"{p}|{CLEAN_REF}")}
    return blk


def sb_min_rho(cells: dict[str, Any], sb_cells: dict[str, Any]) -> dict[str, Any]:
    """D-332 (4)'s minimum rho for a 10-point TPR loss, per repair, for A3-top and SB: phase3_a3.min_rho_table itself
    with SB's cells in A3-benign's place, relabelled (choice (v))."""
    t = A3.min_rho_table({"clean": cells["clean"], "A3-top": cells["A3-top"], "A3-benign": sb_cells})
    return {k: {"A3-top": v["A3-top"], "SB": v["A3-benign"]} for k, v in t.items()}


def sa_min_rho(cells: dict[str, Any], sa_cells: dict[str, Any]) -> dict[str, Any]:
    """D-332 (4)'s minimum rho for SA per repair (choice (t)): phase3_a3.min_rho_table's rule (4d B's: the smallest rho
    whose D-156 mean TPR is at least 10 points below the same repair's unpoisoned one), line for line, on SA's repairs
    (min_rho_table itself loops over R_same, which SA has not)."""
    out: dict[str, Any] = {}
    for pool, rep in SA_REPAIRS:
        t0 = cells["clean"][pool]["repairs"][rep]["tpr"]["mean"]
        hit = None
        for rho in RHOS:
            if t0 - sa_cells[pool][f"rho={rho}"]["repairs"][rep]["tpr"]["mean"] >= PS.TPR_LOSS:
                hit = rho
                break
        c = sa_cells[pool][f"rho={hit}"] if hit is not None else None
        out[f"{pool}|{rep}"] = {
            "min_rho": hit, "searched_up_to": RHOS[-1], "clean_tpr_mean": t0,
            "m": c["m"] if c else None,
            "tpr_mean_at_min_rho": c["repairs"][rep]["tpr"]["mean"] if c else None,
            "gaussian_flag_share_at_min_rho": c["health"]["share_of_draws_gaussian_flag_2x"] if c else None}
    return out


def sa_band_checks(a: dict[str, np.ndarray]) -> dict[str, Any]:
    """Implementation check (phase3_a3.band_checks' rule for SA's band): every SA pool with m > 0 is strictly inside
    (0.5, 2) by health() against the clean pool's post scores, and the greedy's screen agreed with health() on every item
    it added."""
    outside, worst, disagree = [], 0.0, 0
    for pool in SA_POOLS:
        for rho in RHOS:
            p = f"SA|{pool}|rho={rho}"
            bad = np.flatnonzero((a[f"{p}|m_realised"] > 0) & (a[f"{p}|{CLEAN_REF}|health|gaussian_flag"] > 0))
            outside += [f"{pool} rho={rho} draw {int(d)}" for d in bad]
        worst = max(worst, float(a[f"SA|{pool}|greedy|screen_max_rel_err"].max()))
        disagree += int(a[f"SA|{pool}|greedy|screen_disagreements"].sum())
    ok = not outside and worst <= A3.SCREEN_AGREE and disagree == 0
    return {"holds": ok, "pools_outside_band_with_m_gt_0": outside[:20], "n_outside": len(outside),
            "screen_max_rel_err": worst, "screen_bound": A3.SCREEN_AGREE, "screen_disagreements": disagree}


def sa_outcomes(a: dict[str, np.ndarray], att: str = "SA", cmp: str = SA_CMP) -> dict[str, Any]:
    """D-332 (4)'s band outcomes for SA per pool (phase3_a3.band_outcomes' block, line for line, on SA's pools; choice
    (t)): its largest m, the share of draws whose greedy stopped for lack of a fitting item, the clean pool's 4c flag
    share and its clean-relative one (beside), and per repair the TPR loss at its largest m with A3-top's at the same
    per-draw m; beside, the same comparison over only the draws with m > 0. Keys starting with '_' feed score() and are
    removed before the record is written."""
    out: dict[str, Any] = {}
    top = f"rho={RHOS[-1]}"
    for pool in SA_POOLS:
        mr = a[f"{att}|{pool}|{top}|m_realised"]
        pos_m = mr[1:] > 0
        blk: dict[str, Any] = {
            "largest_m": A3.bs(mr), "largest_m_max_over_draws": int(mr[1:].max()),
            "share_of_draws_largest_m_0": float((mr[1:] == 0).mean()),
            "share_of_draws_stopped_no_fit": float(a[f"{att}|{pool}|greedy|stopped_no_fit"][1:].mean()),
            "clean_pool_gaussian_flag_share": float(a[f"clean|{pool}|health|gaussian_flag"][1:].mean()),
            "clean_pool_clean_ref_flag_share": float(a[f"clean|{pool}|{CLEAN_REF}|health|gaussian_flag"][1:].mean()),
            "repairs": {}}
        for rep in A3.REPAIRS_OF[pool]:
            clean = a[f"clean|{pool}|{rep}|tpr"]
            lb = clean - a[f"{att}|{pool}|{top}|{rep}|tpr"]
            lt = clean - a[f"{cmp}|{pool}|{rep}|tpr"]
            n_pos = int(pos_m.sum())
            beside = {"n_draws_with_m_gt_0": n_pos, "n_draws": int(pos_m.size),
                      "sa_tpr_loss_mean": float(lb[1:][pos_m].mean()) if n_pos else None,
                      "a3_top_tpr_loss_at_same_m_mean": float(lt[1:][pos_m].mean()) if n_pos else None,
                      "sa_loss_smaller": bool(lb[1:][pos_m].mean() < lt[1:][pos_m].mean()) if n_pos else None,
                      "label": "beside, not the verdict: the mean over only the draws where SA injected (m > 0)"}
            blk["repairs"][rep] = {"tpr_loss_at_largest_m": A3.bs(lb), "a3_top_tpr_loss_at_same_m": A3.bs(lt),
                                   "comparison_over_draws_with_m_gt_0_beside": beside,
                                   "_band_loss_mean": float(lb[1:].mean()), "_top_loss_mean": float(lt[1:].mean()),
                                   "_infeasible": bool((mr[1:] == 0).all())}
        out[pool] = blk
    return out


def add_d282(S: dict[str, Any], a: dict[str, np.ndarray], cells: dict[str, Any], sens: dict[str, Any],
             counts: np.ndarray) -> dict[str, np.ndarray]:
    """D-282's interval on every threshold-level arm's TPR (phase3_a3 choice (p)): frozen, the clean repairs, A3-top, SB
    and SA at every rho; returns each arm's replicates (for the losses)."""
    reps: dict[str, np.ndarray] = {}

    def one(p: str) -> dict[str, Any]:
        reps[p] = A3.d282_replicates(S, a[f"{p}|threshold_per_seed"], a[f"{p}|tpr"], counts, p)
        return {"tpr_post_combined": P.ci(reps[p])}

    cells["frozen"]["tpr_d282_combined"] = one("frozen")
    for pool, rep in A3.THRESHOLD_LEVEL:
        cells["clean"][pool]["repairs"][rep]["tpr_d282_combined"] = one(f"clean|{pool}|{rep}")
        for rho in RHOS:
            r = f"rho={rho}"
            cells["A3-top"][pool][r]["repairs"][rep]["tpr_d282_combined"] = one(f"A3-top|{pool}|{r}|{rep}")
            sens["SB"][pool][r]["repairs"][rep]["tpr_d282_combined"] = one(f"SB|{pool}|{r}|{rep}")
            if pool in SA_POOLS:
                sens["SA"][pool][r]["repairs"][rep]["tpr_d282_combined"] = one(f"SA|{pool}|{r}|{rep}")
    return reps


def add_d282_losses(S: dict[str, Any], a: dict[str, np.ndarray], reps: dict[str, np.ndarray], minrho: dict[str, Any],
                    sa_out: dict[str, Any], counts: np.ndarray) -> None:
    """D-282 (1) for every TPR loss reported on a threshold-level arm (phase3_a3 choice (p)): A3-top's, SB's and SA's at
    their minimum rho, SA's and A3-top's at SA's largest m, each the clean arm's replicates minus the poisoned arm's,
    paired within item and draw."""
    top = f"rho={RHOS[-1]}"
    for pool, rep in A3.THRESHOLD_LEVEL:
        clean = reps[f"clean|{pool}|{rep}"]
        for att, row in minrho[f"{pool}|{rep}"].items():
            row["tpr_loss_at_min_rho_d282_combined"] = (
                A3.d282_loss(clean, reps[f"{att}|{pool}|rho={row['min_rho']}|{rep}"]) if row["min_rho"] is not None
                else None)
        if pool in SA_POOLS:
            p = f"{SA_CMP}|{pool}|{rep}"
            top_reps = A3.d282_replicates(S, a[f"{p}|threshold_per_seed"], a[f"{p}|tpr"], counts, p)
            r = sa_out[pool]["repairs"][rep]
            r["tpr_loss_at_largest_m_d282_combined"] = A3.d282_loss(clean, reps[f"SA|{pool}|{top}|{rep}"])
            r["a3_top_tpr_loss_at_same_m_d282_combined"] = A3.d282_loss(clean, top_reps)


# ---- scoring ----------------------------------------------------------------------------------------------------------

P33_NAMES = {"a3_band_tpr_loss_at_largest_m": "sa_tpr_loss_at_largest_m",
             "band_infeasible_m0_every_draw": "sa_infeasible_m0_every_draw",
             "clean_pool_gaussian_flag_share": "clean_pool_clean_ref_flag_share",
             "tpr_loss_at_largest_m_d282_combined": "sa_tpr_loss_at_largest_m_d282_combined"}


def p33_sa_cell(r: dict[str, Any], ref_flag_share: float, flag_share_4c: float) -> dict[str, Any]:
    """One P-A3.3 cell for SA (choices (m), (u)): phase3_a3.p33_cell with the clean pool's clean-relative flag share
    choosing the undefined reason, its A3-band field names given SA's, and the clean pool's 4c flag share beside."""
    c = A3.p33_cell(r, ref_flag_share)
    out = {P33_NAMES.get(k, k): v for k, v in c.items()}
    out["clean_pool_gaussian_flag_share_4c_beside"] = flag_share_4c
    return out


def score(minrho: dict[str, Any], sa_out: dict[str, Any]) -> dict[str, Any]:
    """D-369 (2) per pair: P-A3.2 for SB against A3-top and P-A3.3 for SA against A3-top, by D-352 (2)-(3)'s rules."""
    p32 = {}
    for pool, rep in SB_REPAIRS:
        t, b = minrho[f"{pool}|{rep}"]["A3-top"]["min_rho"], minrho[f"{pool}|{rep}"]["SB"]["min_rho"]
        p32[f"{pool}|{rep}"] = {"a3_top_min_rho": t, "sb_min_rho": b, "holds": A3.at_least(b, t)}
    v32 = [c["holds"] for c in p32.values()]
    p33 = {f"{pool}|{rep}": p33_sa_cell(sa_out[pool]["repairs"][rep], sa_out[pool]["clean_pool_clean_ref_flag_share"],
                                        sa_out[pool]["clean_pool_gaussian_flag_share"]) for pool, rep in SA_REPAIRS}
    v33 = [c["holds"] for c in p33.values()]
    counted = lambda v: {"n_cells_decided": sum(x is not None for x in v), "n_cells_holding": sum(x is True for x in v),  # noqa: E731
                         "n_cells_failing": sum(x is False for x in v), "n_cells_undecided": sum(x is None for x in v)}
    reading = ("the pair verdict is D-352 (3)'s reading (D-351 (3)'s): False if any decided cell fails, True if every "
               "decided cell holds and at least one is decided, None if none is decided")
    return {
        P32: {
            "cells": p32, "holds_in_this_pair": A3.decided_verdict(v32), **counted(v32),
            "rule": "SB's minimum rho >= A3-top's per pool and repair, restandardisation on traffic included (phase3_a3 "
                    "choice (n)); None reads '> 0.25' (D-244 (6)); a cell where both are '> 0.25' is undecided (None); "
                    + reading + "; a sensitivity of P-A3.2 (D-352 (4), D-369): reported beside, never primary"},
        P33: {
            "cells": p33, "holds_in_this_pair": A3.decided_verdict(v33), **counted(v33),
            "n_cells_band_infeasible": sum(x is None for x in v33),
            "rule": "SA's mean TPR loss at its largest m < A3-top's at the same per-draw m (phase3_a3 choice (m)), per "
                    "pool and repair on R_fresh and traffic; a cell where SA's m is 0 in every draw is None (band "
                    "infeasible, why_undecided), not False (D-352 (2)); " + reading + "; the clean pool's 4c flag share "
                    "and the comparison over the draws with m > 0 are beside, not the verdict; a sensitivity of P-A3.3 "
                    "(D-352 (4), D-369): reported beside, never primary"},
    }


def derive(a: dict[str, np.ndarray], target: float) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """From the per-draw arrays only: the D-156 cells (frozen, clean, A3-top), SB's and SA's cells, the minimum-rho
    table (A3-top and SB per repair, SA on its repairs) and SA's outcomes. The per-pair run and the pooled re-derivation
    take this one path (choice (y))."""
    cells = A3.summaries(a, target)
    sens = {"SB": sb_summaries(a, target), "SA": sa_summaries(a)}
    mr = sb_min_rho(cells, sens["SB"])
    for k, row in sa_min_rho(cells, sens["SA"]).items():
        mr[k]["SA"] = row
    return cells, sens, mr, sa_outcomes(a)


def reads_problems(reads: dict[str, str], expected: dict[str, str]) -> list[str]:
    """Choice (x): why the results this process read (provenance's registered reads, name -> sha256) include an A3
    record or are not exactly the expected ones (empty when neither)."""
    bad = []
    a3 = sorted(k for k in reads if A3_RECORD.search(k))
    if a3:
        bad.append(f"an A3 record was read (D-352 (3), D-369 (2)): {a3}")
    if reads != expected:
        bad.append(f"the results read are not exactly the expected ones: read {sorted(reads)}, expected "
                   f"{sorted(expected)}" + (" (same names, other sha256)" if set(reads) == set(expected) else ""))
    return bad


# ---- pooled (D-369 (2), D-352 (3)) ----------------------------------------------------------------------------------

def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def pool_discover(rdir: Path) -> dict[str, Path]:
    """The per-pair sensitivity records in rdir: exactly the nine pairs (a missing or an extra pair refuses)."""
    found: dict[str, Path] = {}
    for p in sorted(rdir.glob("phase3_a3_sens_*.json")):
        core = p.stem[len("phase3_a3_sens_"):]
        if core != "pooled":
            found[core] = p
    missing, extra = [c for c in PAIRS if c not in found], sorted(set(found) - set(PAIRS))
    if missing or extra:
        refuse(f"the sensitivity records in {rdir} are not the nine pairs: missing {missing}, extra {extra}")
    return {c: found[c] for c in PAIRS}


def pool_read_npz(npz_path: Path, block: dict[str, Any], owner: str) -> tuple[str, int, dict[str, np.ndarray]]:
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


def pool_load(core: str, path: Path, files: dict[str, tuple[Path, str]]) -> dict[str, Any]:
    """One per-pair record (a registered read, D-297 (1)) and its npz, bound by sha256."""
    doc = read_result(path)
    sha = sha256_file(path)
    npz_path = path.with_name(path.stem + "_draws.npz")
    dz = doc.get("draws_npz") or {}
    if dz.get("file") != A1P.rel(npz_path):
        refuse(f"{path.name} names the npz {dz.get('file')!r}, not {A1P.rel(npz_path)!r}")
    nsha, nsize, z = pool_read_npz(npz_path, dz, path.name)
    files[A1P.rel(path)] = (path, sha)
    files[A1P.rel(npz_path)] = (npz_path, nsha)
    return {"core": core, "doc": doc, "z": z, "file": A1P.rel(path), "sha256": sha,
            "draws_npz": {"file": A1P.rel(npz_path), "sha256": nsha, "size": nsize}}


def pool_record_problems(core: str, doc: dict[str, Any], final: bool) -> list[str]:
    """Why one per-pair record cannot enter the pooled scoring (empty when it can)."""
    cfg = doc.get("config") or {}
    bad = []
    if doc.get("run_kind") != "phase3_a3_sens" or cfg.get("pair") != f"repair_{core}":
        bad.append(f"not the sensitivity record of repair_{core} (run_kind {doc.get('run_kind')!r}, pair {cfg.get('pair')!r})")
    if doc.get("registered_parameters") is not True or cfg.get("anchor_only") is not False:
        bad.append("not a registered run, or written with --anchor-only")
    an = doc.get("anchor") or {}
    n_cmp = an.get("n_values_compared")
    if not (an.get("checked") is True and an.get("identical") is True and isinstance(n_cmp, int) and n_cmp > 0
            and an.get("n_values_identical") == n_cmp):
        bad.append("did not pass its self-anchor (D-332 (6))")
    if ((doc.get("implementation_checks") or {}).get("sa_band") or {}).get("holds") is not True:
        bad.append("did not pass its SA band check")
    if doc.get("a3_records_read") != []:
        bad.append(f"does not record that it read no A3 record: {doc.get('a3_records_read')!r}")
    if final and any((doc.get("blas_threads") or {}).get(v) != "6" for v in A3.THREAD_VARS):
        bad.append(f"not written at the 6-thread pin: {doc.get('blas_threads')}")
    return bad


def pool_shared_code(prs: dict[str, dict[str, Any]], mine: dict[str, Any]) -> dict[str, Any]:
    """The nine records' one code block, which must be this tree's (the re-derivation calls its functions)."""
    codes = {c: pr["doc"].get("code") or {} for c, pr in prs.items()}
    ref = codes[PAIRS[0]]
    off = [c for c, cb in codes.items() if cb.get("sha256") != ref.get("sha256") or cb.get("libraries") != ref.get("libraries")]
    if off or not isinstance(ref.get("sha256"), dict):
        refuse(f"the nine records do not share one code block: {off} differ from {PAIRS[0]}")
    diff = sorted(f for f in set(ref["sha256"]) | set(mine["sha256"]) if ref["sha256"].get(f) != mine["sha256"].get(f))
    if diff or ref.get("libraries") != mine.get("libraries"):
        refuse(f"this tree's code is not the records' (the pooled re-derivation calls its functions): files {diff}, "
               f"libraries {ref.get('libraries')} vs {mine.get('libraries')}")
    return {"sha256": ref["sha256"], "libraries": ref.get("libraries"),
            "commits": sorted({str(cb.get("commit")) for cb in codes.values()})}


def pool_view(sc: dict[str, Any]) -> dict[str, Any]:
    """The fields of a per-pair scoring the pooled verdicts rest on (the D-282 intervals aside)."""
    p32, p33 = sc.get(P32) or {}, sc.get(P33) or {}
    k32 = ("a3_top_min_rho", "sb_min_rho", "holds")
    k33 = ("holds", "why_undecided", "sa_infeasible_m0_every_draw", "sa_tpr_loss_at_largest_m",
           "a3_top_tpr_loss_at_same_m", "clean_pool_clean_ref_flag_share", "clean_pool_gaussian_flag_share_4c_beside")
    cnt = ("holds_in_this_pair", "n_cells_decided", "n_cells_holding", "n_cells_failing", "n_cells_undecided")
    return {P32: {"cells": {k: {f: (c or {}).get(f) for f in k32} for k, c in (p32.get("cells") or {}).items()},
                  **{f: p32.get(f) for f in cnt}},
            P33: {"cells": {k: {f: (c or {}).get(f) for f in k33} for k, c in (p33.get("cells") or {}).items()},
                  **{f: p33.get(f) for f in (*cnt, "n_cells_band_infeasible")}}}


def pool_rederive(pr: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Choice (y): the per-pair scoring re-derived from the record's npz (derive, score), and whether the record's
    verdict-bearing fields equal it exactly."""
    _, _, mr, sa_out = derive(pr["z"], pr["doc"]["config"]["target_fpr"])
    sc = score(mr, sa_out)
    mine = json.dumps(json.loads(json.dumps(pool_view(sc), default=str)), sort_keys=True)  # not a result (D-297): own round trip
    return sc, sa_out, mine == json.dumps(pool_view(pr["doc"].get("scoring") or {}), sort_keys=True)


def pool_score(derived: dict[str, tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    """D-352 (3) over the nine pairs: P-A3.2 pooled over the 36 cells; P-A3.3 'in every pair' over the pair verdicts."""
    cells32, pair32, pairs33, n_inf = {}, {}, {}, 0
    for c in PAIRS:
        sc, sa_out = derived[c]
        for k, cell in sc[P32]["cells"].items():
            cells32[f"{c}|{k}"] = {"a3_top_min_rho": cell["a3_top_min_rho"], "sb_min_rho": cell["sb_min_rho"],
                                   "holds": cell["holds"]}
        pair32[c] = sc[P32]["holds_in_this_pair"]
        p = sc[P33]
        n_inf += p["n_cells_band_infeasible"]
        pairs33[c] = {"holds": p["holds_in_this_pair"],
                      **{k: p[k] for k in ("n_cells_decided", "n_cells_holding", "n_cells_failing", "n_cells_band_infeasible")},
                      "cells": {k: {"holds": v["holds"], "why_undecided": v["why_undecided"]} for k, v in p["cells"].items()},
                      "clean_pool_gaussian_flag_share_4c_beside": {pool: sa_out[pool]["clean_pool_gaussian_flag_share"]
                                                                  for pool in SA_POOLS},
                      "clean_pool_clean_ref_flag_share_beside": {pool: sa_out[pool]["clean_pool_clean_ref_flag_share"]
                                                                for pool in SA_POOLS}}
    v32 = [x["holds"] for x in cells32.values()]
    v33 = [x["holds"] for x in pairs33.values()]
    reading = ("D-352 (3)'s reading: True if every decided {u} holds and at least one is decided, False if any decided "
               "{u} fails, None (undefined) if none is decided")
    return {
        P32: {"cells": cells32, "n_cells": len(v32), "holds_pooled": A3.decided_verdict(v32),
              "n_cells_decided": sum(x is not None for x in v32), "n_cells_holding": sum(x is True for x in v32),
              "n_cells_failing": sum(x is False for x in v32), "n_cells_undecided": sum(x is None for x in v32),
              "per_pair_verdict_beside": pair32,
              "rule": "SB's minimum rho >= A3-top's in every (pair, pool, repair) cell, 4 per pair, restandardisation "
                      "included; '> 0.25' as D-244 (6); " + reading.format(u="cell") + "; a sensitivity, never primary "
                      "(D-369 (2))"},
        P33: {"pairs": pairs33, "n_pairs": len(v33), "holds_in_every_pair": A3.decided_verdict(v33),
              "n_pairs_decided": sum(x is not None for x in v33), "n_pairs_holding": sum(x is True for x in v33),
              "n_pairs_failing": sum(x is False for x in v33), "n_pairs_undecided": sum(x is None for x in v33),
              "n_cells": sum(len(p["cells"]) for p in pairs33.values()), "n_cells_band_infeasible": n_inf,
              "rule": "per pair as D-352 (2)-(3) (an SA cell with m 0 in every draw undefined; the pair False if a decided "
                      "cell fails, True if every decided cell holds and one is decided, else None), then 'in every pair' "
                      "by " + reading.format(u="pair") + "; the infeasible cells and each pair's clean-pool 4c and "
                      "clean-relative flag shares beside; a sensitivity, never primary (D-369 (2))"}}


def pool(rdir: Path, mine: dict[str, Any], final: bool) -> tuple[dict[str, Any], dict[str, tuple[Path, str]]]:
    """Every refusal, then the pooled P-A3.2 and P-A3.3. Returns the record's body and the input files."""
    paths = pool_discover(rdir)
    files: dict[str, tuple[Path, str]] = {}
    prs = {c: pool_load(c, p, files) for c, p in paths.items()}
    bad = {c: b for c, b in ((c, pool_record_problems(c, prs[c]["doc"], final)) for c in PAIRS) if b}
    if bad:
        refuse(f"not every record can enter the pooled scoring: {bad}")
    params = {c: {k: prs[c]["doc"]["config"].get(k) for k in ("n_reps", "seeds", "target_fpr")} for c in PAIRS}
    if any(v != params[PAIRS[0]] for v in params.values()):
        refuse(f"the nine records were run at different draws, seeds or target: {params}")
    code = pool_shared_code(prs, mine)
    derived, off = {}, []
    for c in PAIRS:
        sc, sa_out, same = pool_rederive(prs[c])
        derived[c] = (sc, sa_out)
        if not same:
            off.append(c)
    if off:
        refuse(f"the P-A3.2 / P-A3.3 cells of {off} are not their npz's (re-derived by derive and score)")
    records = {c: {"file": prs[c]["file"], "sha256": prs[c]["sha256"], "draws_npz": prs[c]["draws_npz"],
                   "commit": (prs[c]["doc"].get("code") or {}).get("commit"),
                   "anchor": {k: prs[c]["doc"]["anchor"].get(k) for k in ("checked", "identical", "n_values_identical",
                                                                         "n_values_compared")}}
               for c in PAIRS}
    return {"parameters": params[PAIRS[0]], "sens_code": code, "records": records, **pool_score(derived)}, files


def pooled_main(args: argparse.Namespace) -> int:
    """--pool: results/phase3_a3_sens_pooled.json from the nine per-pair records (D-369 (2), D-352 (3))."""
    rdir = Path(args.records_dir).resolve() if args.records_dir else RESULTS.resolve()
    final_out = (RESULTS / POOLED_NAME).resolve()
    out = Path(args.out).resolve() if args.out else final_out
    final = out == final_out
    if out.exists():
        refuse(f"{out} exists; results are never overwritten")
    if not final and out.is_relative_to(RESULTS.resolve()):
        refuse("a test output (--out) must not be under results/")
    threads = {v: os.environ.get(v) for v in A3.THREAD_VARS}
    if final and (rdir != RESULTS.resolve() or any(t != "6" for t in threads.values())):
        refuse(f"the final name reads only the records in results/ at the 6-thread pin: records {rdir}, threads {threads}")
    code0 = P.code_block(CODE_FILES)
    body, files = pool(rdir, code0, final)
    code1 = P.code_block(CODE_FILES)
    changed = [n for n, (p, h) in files.items() if sha256_file(p) != h]
    if code1["sha256"] != code0["sha256"] or changed:
        refuse(f"to write: the code or an input changed during the run: {changed}")
    reads = dict(PV._RESULT_READS)                    # choice (x): exactly the nine records, no A3 record
    bad = reads_problems(reads, {n: h for n, (_, h) in files.items() if n.endswith(".json")})
    if bad:
        refuse(f"to write: {bad}")
    config = {"pairs": list(PAIRS), "records": "phase3_a3_sens_<pair>.json and _draws.npz, one per pair",
              "p32_cells": "pair x (pool, repair): 4 per pair, 36", "p33_cells": "pair x (pool, repair) on R_fresh and "
              "traffic: 3 per pair, 27; pooled over the 9 pair verdicts", "registered_parameters": body["parameters"]}
    payload = {**provenance_block("phase3_a3_sens_pooled"), "preregistration": PREREG_POOLED, "code": code0,
               "blas_threads": threads, "config": config, "config_hash": canonical_hash(config),
               "sandbox_disclosure": SANDBOX,
               "role": "pooled sensitivity of the A3 scoring (D-352 (4), D-369 (2)): reported beside, never primary",
               "a3_records_read": sorted(k for k in reads if A3_RECORD.search(k)),
               "results_read": {"files": reads, "rule": "choice (x): exactly the nine sensitivity records"},
               "input_files_sha256": {n: h for n, (_, h) in files.items()},
               "implementation_choices": {k: CHOICES[k] for k in ("x", "y")}, **body}
    digest = write_results(out, payload, exclusive=True)
    print(f"pooled: {len(PAIRS)} records and their npz bound by sha256, every per-pair cell re-derived from its npz; "
          f"P-A3.2 (SB) pooled holds {body[P32]['holds_pooled']}; P-A3.3 (SA) in every pair "
          f"{body[P33]['holds_in_every_pair']}", flush=True)
    print(f"Written: {out}\nsha256:  {digest}")
    return 0


# ---- main -------------------------------------------------------------------------------------------------------------

def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Phase 3 A3 sensitivities SA and SB (D-352 (4), D-369)")
    ap.add_argument("--pair", default=None, choices=POP.UPDATE_PAIRS_REPAIR, help="one pair (or --pool)")
    ap.add_argument("--pool", action="store_true",
                    help="the pooled scoring over the nine per-pair records (D-369 (2), D-352 (3))")
    ap.add_argument("--records-dir", default=None, help="--pool, TESTING: a directory of per-pair records (default results/)")
    ap.add_argument("--n-reps", type=int, default=A3P.REGISTERED["n_reps"])
    ap.add_argument("--target-fpr", type=float, default=A3P.REGISTERED["target_fpr"])
    ap.add_argument("--seeds", type=int, nargs="*", default=A3P.REGISTERED["seeds"])
    ap.add_argument("--out", default=None, help="TESTING: a JSON path outside results/")
    ap.add_argument("--anchor-only", action="store_true",
                    help="TESTING ONLY: stop after the self-anchor (clean and A3-top only), write nothing")
    args = ap.parse_args()
    if args.pool:
        if args.pair is not None or args.anchor_only:
            raise SystemExit("REFUSING: --pool takes no --pair and no --anchor-only")
        return pooled_main(args)
    if args.pair is None or args.records_dir is not None:
        raise SystemExit("REFUSING: a per-pair run needs --pair and takes no --records-dir (or use --pool)")
    t0 = time.time()
    core = A1P.core_of(args.pair)
    registered = (args.n_reps == A3P.REGISTERED["n_reps"] and args.target_fpr == A3P.REGISTERED["target_fpr"]
                  and list(args.seeds) == A3P.REGISTERED["seeds"])
    final_path = (RESULTS / f"phase3_a3_sens_{core}.json").resolve()
    out_path = Path(args.out).resolve() if args.out else final_path
    npz_path = out_path.with_name(out_path.stem + "_draws.npz")
    final = out_path == final_path
    for p in (out_path, npz_path):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists; results are never overwritten")
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test run (--out) must not write under results/")
    threads = {v: os.environ.get(v) for v in A3.THREAD_VARS}
    if final and (not registered or any(t != "6" for t in threads.values()) or args.anchor_only):
        raise SystemExit(f"REFUSING the final name: registered parameters {registered}, threads {threads}, anchor-only "
                         f"{args.anchor_only}")
    code0 = P.code_block(CODE_FILES)
    S = A3P.setup(args.pair, args.n_reps, list(args.seeds), args.target_fpr)
    sb = sb_candidates(S)                             # SB's candidates checked before any computation
    rec_rel = f"results/{A1P.record_of_record_stem(core)}.json"
    rec_sha = sha256_file(ROOT / rec_rel)
    rec = read_result(ROOT / rec_rel)                 # registered read: listed with its sha256 (D-297 (1))
    bad_rec = A1.check_record_of_record(rec, S, registered)
    if bad_rec:
        raise SystemExit(f"REFUSING: {rec_rel} is not the 4d B record of these inputs, probe and draws: {bad_rec}")
    print(f"A3 sensitivities {args.pair} [{S['axis']}] L{S['L_sel']} C {S['C_sel']:g}: {len(S['draws']) - 1} draws + "
          f"frozen; 4d B record {rec_rel}; registered {registered}", flush=True)
    X = A3.precompute(S)
    log = lambda m: print(m, flush=True)  # noqa: E731

    # the clean arms and the comparator A3-top first: the self-anchor before any SA or SB number (D-332 (6))
    a = A3.measure(S, X, ("clean", "A3-top"), log)
    cells = A3.summaries(a, S["target"])
    minrho = A3.min_rho_table(cells)
    if registered:
        anc = A3.anchor_check(cells, minrho, rec)
        anchor = {"checked": True, **anc}
        print(f"  self-anchor: min rho {sum(v['equal'] for v in anc['min_rho'].values())} of {len(anc['min_rho'])} "
              f"equal to 4d B's A1; {anc['n_values_identical']} of {anc['n_values_compared']} 4d B summaries identical; "
              f"{'PASS' if anc['identical'] else 'FAIL'}", flush=True)
        if not anc["identical"]:
            raise SystemExit("REFUSING: A3-top does not reproduce 4d B's A1 (D-332 (6)); nothing written:\n  "
                             + "\n  ".join((anc["min_rho_mismatches"] + anc["value_mismatches"])[:12]))
    else:
        anchor = {"checked": False, "why": "non-registered draws, seeds or target (a test): the records hold the registered run"}
    anchor.update({"file": A1P.rel(ROOT / rec_rel), "sha256": rec_sha})
    if args.anchor_only:
        print("  --anchor-only: stopping after the self-anchor; nothing written", flush=True)
        return 0
    a.update(measure_sens(S, X, VARIANTS, sb, log))
    cells, sens, mr, sa_out = derive(a, S["target"])          # the pooled mode re-derives the scoring by this path
    bandc = sa_band_checks(a)
    print(f"  SA band check {'holds' if bandc['holds'] else 'FAILS'}", flush=True)
    if not bandc["holds"]:
        print(f"IMPLEMENTATION CHECK FAILED (D-181; D-346's pattern): nothing written.\n  band {bandc}", flush=True)
        return 3
    counts = P.counts_matrix(S["n_pos"], D.REGISTERED["n_boot"], D.ITEM_BOOT_SEED["eval_positives"])
    reps = add_d282(S, a, cells, sens, counts)
    add_d282_losses(S, a, reps, mr, sa_out, counts)
    scoring = score(mr, sa_out)
    for blk in sa_out.values():
        for r in blk["repairs"].values():
            for k in ("_band_loss_mean", "_top_loss_mean", "_infeasible"):
                r.pop(k)
    print("  scored: P-A3.2 (SB against A3-top) and P-A3.3 (SA against A3-top) per pair (verdicts in the record)",
          flush=True)

    npz = dict(a)
    nb = A1P.npz_bytes(npz)
    changed = A3P.changed_inputs(S, code0, CODE_FILES, [(rec_rel, sha256_file(ROOT / rec_rel), rec_sha)])
    if changed:
        raise SystemExit(f"REFUSING to write: changed during the run: {changed}")
    reads = dict(PV._RESULT_READS)                    # choice (x): exactly these reads, no A3 record
    expected = {A1P.rel(S["sel_path"]): S["sel_sha"], A1P.rel(A3P.ROOT / S["depth_rel"]): S["depth_sha"],
                A1P.rel(ROOT / rec_rel): rec_sha}
    bad_reads = reads_problems(reads, expected)
    if bad_reads:
        raise SystemExit(f"REFUSING to write: {bad_reads}")
    block = {k: v for k, v in A3P.input_block(S).items() if k not in ("predicted_attackers", "attacker_sources")}
    config = {**block, "variants": list(VARIANTS), "comparator": "A3-top", "sb_pools": list(SB_POOLS),
              "sa_pools": list(SA_POOLS), "repairs": {pool: list(A3.REPAIRS_OF[pool]) for pool in POOLS},
              "attacker_sources": {
                  "A3-top": "threshold_calib harmful H not in the pool, 4d B's A1 order",
                  "SB": "per seed, the probe_train negatives in that seed's D-117 held-out 20% (ho20), that seed's post "
                        "score order",
                  "SA": "threshold_calib harmful H not in the pool, in A3-top's order (A3-band's candidates)"},
              "sb_candidates_per_seed": sb["n_candidates_per_seed"], "sb_injections_sha256": sb["sha256"],
              "d117_holdout": P.HOLDOUT, "band": list(A3.BAND), "sa_band_reference": "the clean pool's post scores",
              "health_cut": list(A3.CUT), "tpr_loss": PS.TPR_LOSS, "screen_tol": A3.SCREEN_TOL, "anchor_only": False}
    payload = {**provenance_block("phase3_a3_sens"), "preregistration": PREREG, "code": code0,
               "registered_parameters": registered, "blas_threads": threads, "input_verification": S["verification"],
               "config": config, "config_hash": canonical_hash(config), "sandbox_disclosure": SANDBOX,
               "role": "sensitivity of the A3 records (D-352 (4), D-369): reported beside them, never primary",
               "a3_records_read": sorted(k for k in reads if A3_RECORD.search(k)),
               "results_read": {"files": reads, "rule": "choice (x): exactly the selection, depth and 4d B records, and "
                                                        "no A3 record or A3 predictions record (D-352 (3), D-369 (2)); "
                                                        "A3-top is recomputed here and self-anchored to 4d B"},
               "implementation_choices": {**A3P.CHOICES, **A3.CHOICES, **CHOICES},
               "anchor": anchor, "depth_record": {"file": S["depth_rel"], "sha256": S["depth_sha"]},
               "implementation_checks": {"sa_band": bandc,
                                         "sb_candidates": "per seed exactly the probe_train negatives in the seed's "
                                                          "D-117 held-out 20%, none a fit row, at least rho 0.25's budget",
                                         "d282_reproduction": "every combined-interval arm reproduced its TPR draw by draw"},
               "d282_tpr": {"where": "cells.<frozen, clean, A3-top>... and sensitivity_cells.<SB, SA>.<pool>.<rho>.repairs."
                                     "<threshold-level repair>.tpr_d282_combined",
                            "losses_where": "outcomes.min_rho_10pt_tpr_loss.<threshold-level arm>.<A3-top, SB, SA>."
                                            "tpr_loss_at_min_rho_d282_combined; outcomes.sa.<pool>.repairs.<threshold-"
                                            "level repair>.tpr_loss_at_largest_m_d282_combined and "
                                            "a3_top_tpr_loss_at_same_m_d282_combined (and P-A3.3's cells)",
                            "losses_paired": "within item and draw: replicate r is the clean arm's TPR minus the "
                                             "poisoned arm's on the same counts row and draw",
                            "arms": [f"{p}|{r}" for p, r in A3.THRESHOLD_LEVEL],
                            "not": "restandardise: the positives' scores move per draw (D-156 band only), for its "
                                   "TPR and its losses",
                            "n_boot": int(counts.shape[0]), "counts_seed": D.ITEM_BOOT_SEED["eval_positives"],
                            "draw_of_replicate": "D-282: replicate r uses draw 1 + ((r - 1) mod n_draws)"},
               "probe_fit": [q.diag for q in S["probes"]],
               "input_cache_sha256": S["input_sha"],
               "input_cache_source": {kk: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"]}
                                      for kk, r in S["input_recs"].items()},
               "draws_npz": {"file": A1P.rel(npz_path), "sha256": sha256_bytes(nb), "size": len(nb), "keys": sorted(npz),
                             "rows": "draw index (0 the frozen split); per-seed arrays have the seeds as columns"},
               "outcomes": {"min_rho_10pt_tpr_loss": mr, "sa": sa_out,
                            "health_flag_share": {
                                pool: {"clean": cells["clean"][pool]["health"]["share_of_draws_gaussian_flag_2x"],
                                       "A3-top": {f"rho={rho}": cells["A3-top"][pool][f"rho={rho}"]["health"]["share_of_draws_gaussian_flag_2x"]
                                                  for rho in RHOS},
                                       "SB": {f"rho={rho}": sens["SB"][pool][f"rho={rho}"]["health"]["share_of_draws_gaussian_flag_2x"]
                                              for rho in RHOS},
                                       **({"SA": {f"rho={rho}": sens["SA"][pool][f"rho={rho}"]["health"]["share_of_draws_gaussian_flag_2x"]
                                                  for rho in RHOS},
                                           "SA_clean_ref": {f"rho={rho}": sens["SA"][pool][f"rho={rho}"]["health_clean_ref"]["share_of_draws_gaussian_flag_2x"]
                                                            for rho in RHOS}} if pool in SA_POOLS else {})}
                                for pool in POOLS}},
               "scoring": scoring, "cells": cells, "sensitivity_cells": sens,
               "runtime_seconds": round(time.time() - t0, 1)}
    for p, hh in A1P.write_pair(out_path, payload, npz_path, nb):
        print(f"Written: {p}\nsha256:  {hh}")
    print(f"runtime {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
