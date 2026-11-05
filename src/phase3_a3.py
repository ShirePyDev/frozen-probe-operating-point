"""Phase 3 attacker A3, the reference-set poisoner (D-332), part 2 of 2: the SIMULATION, one pair per run.

REFUSES unless the predictions record (src/phase3_a3_predictions.py) exists, names this pair, these draws (the hash of
every draw's index arrays), these inputs, these injections (their hash) and parameters, and the current sha256 of the
prediction function's file, and unless its arrays equal this run's own recomputation bit for bit (D-332 (5), (7)).

THE ATTACK (D-332 (2), (3)): m = round(rho * n) items join the pool a repair recalibrates on (n = 1000; rho on 4d's grid):
  POOLS     R_same (cal_d's negatives: the stored calibration set, as if contaminated at collection; here this models
            contamination of the post-update re-scoring only, since T_d and the band's pre moments come from the clean
            cal_d), R_fresh (the draw's 1000 fresh benign), traffic_1pct (4d B's 1% traffic pool as the post window;
            the pre window stays the clean pool).
  REPAIRS   threshold re-set (the 1% threshold on the pool's post scores, paired.thr_at_fpr) on R_same and R_fresh; on
            traffic, quantile transfer (paired.quantile_transfer: T_d's rank among the clean pre window's pre scores
            carried to the poisoned post window, F-070) and restandardisation (poolstudy_d156.feature_stats of the
            poisoned pool's post features, restandardised_scores with the probe weights and T_d fixed): poolstudy_d156
            main()'s recal, quantile and feature, line for line, so A3-top is 4d B's A1 bit for bit.
  ATTACKERS A3-top and A3-benign as the predictions module states them; A3-band (choice (k)): from the harmful items
            not in the pool, in A3-top's order, at each step the highest-ranked remaining item whose addition keeps the
            pool's 4c Gaussian health ratio strictly inside (0.5, 2); it stops at the budget or when no item fits.
  HEALTH    the 4c Gaussian health ratio on a pool (D-241 (2), D-332 (3)): healthcheck_d156.health with cal_d's PRE
            scores (the stored calibration set's pre moments, which set T_d) against the pool's POST scores, at T_d per
            seed, the mean over seeds of per-seed ratios (choice (j)); its flag at 2 / 0.5 (D-241 (3), D-264).
EVALUATED as 4d B's put() (phase3_a1.arm_numbers): FPR on E (1000 negatives; seed mean of floored per-seed FPRs), the
paired ratio against FPR_pre at T_d, FPR/target, TPR on the eval positives; the frozen arm beside. D-156 mean and band.
OUTCOMES (D-332 (4)) per pool, repair and attacker: TPR and FPR/target at each rho; the minimum rho (and m) for a
10-point TPR loss against the same repair's unpoisoned TPR (4d B's rule: D-156 mean TPR clean minus poisoned >= 0.10);
the Gaussian flag's share of draws at that rho and at each rho; for A3-band the realised m per rho, its largest m (the
realised m at rho 0.25) and the TPR loss there, with A3-top's loss at the same per-draw m (P-A3.3). Every TPR loss
reported on a threshold-level arm (at the minimum rho; A3-band's and A3-top's at the largest m) carries D-282's
combined interval, paired within item (choice (p)).
SELF-ANCHOR (D-332 (6)): computed first, before any A3-benign or A3-band number: A3-top's minimum rho for each of the
four repairs must equal the 4d B record of record's selected A1 minimum rho (populations.POOLSTUDY_D156; option B for
Aegis, D-298), and (an implementation check in D-346's pattern) every D-156 summary of the frozen arm, the four
unpoisoned repairs and A3-top at every rho must equal 4d B's exactly. A mismatch refuses: nothing is written. The
anchor needs the registered draws, seeds and target (a test with fewer draws records it as not checked).
IMPLEMENTATION CHECKS (D-181, D-346's linear-identity pattern; exit 3, nothing written):
  threshold identity  on R_same and R_fresh, A3-top and A3-benign, every rho, draw and seed: the realised threshold
                      equals the predictions record's t' where the k-th largest is not tied, else the smallest pool
                      score above t' (paired.thr_at_fpr's rule), so the record's injections are the simulation's.
  band                every A3-band pool with m > 0 is inside (0.5, 2) by health() itself; the greedy's screen agrees
                      with health() to SCREEN_AGREE (relative) on every item it added.
  D-282               each combined-interval arm's inputs reproduce its TPR draw by draw (else a refusal, as A1).
SCORING (D-332 (5); per pair; the pooled lines over the nine pairs are a separate step, not here):
  P-A3.1 D-177's accuracy line on TPR over the 48 cells pool x {A3-top, A3-benign} x rho (choice (b) of the
         predictions module: the pool's score repair), realised and predicted as cell D-156 means; a cell whose
         predicted or realised TPR is 0 is a miss (D-322 (4), D-346 (2)); by pool and attacker beside. The 32 threshold
         re-set cells (about two thirds) agree by construction (the prediction is the repair's own rule, up to ties at
         the k-th place): their count and the tie count are recorded, and named lines over the 16 quantile-transfer
         cells only and over the 32 threshold re-set cells only stand beside the registered line (as D-352 (3) pools
         them).
  P-A3.2 A3-benign's minimum rho >= A3-top's, per pool and repair, restandardisation included (choice (n)).
  P-A3.3 A3-band's TPR loss at its largest m < A3-top's at the same m, per pool and repair; a cell where the band is
         infeasible (A3-band's m is 0 in every draw) is undecided, not failed (choice (m)).
  P-A3.4 restandardisation on traffic: A3-top's minimum rho >= 10 x A3-top's of each score repair (choice (o)); fully
         determined by A3-top, which equals 4d B's A1 bit for bit (the self-anchor): no new measurement.
CHOICES BY THE CLOSEST D-ENTRY (D-181; besides (a)-(h) in src/phase3_a3_predictions.py; also in the record):
  (i) the four repairs are 4d B's code paths (D-332 (1): "src/poolstudy_d156.py's partition and pools"; D-244 (5));
      A3-top is 4d B's A1 expression, so the self-anchor can hold bit for bit (D-332 (6)).
  (j) the health ratio's pre moments are cal_d's pre scores for every pool: D-332 (3) "the pool's post moments against
      the pre moments", D-241 (1)-(2) (the reference set's pre-model moments, the scores that set T_d); the defender
      holds no pre model for fresh items (D-318 (9)). healthcheck_d156.health unchanged (one definition, D-152 P3);
      its direct ratio is not reported (its 1/n floor uses the pre set's n, not the pool's).
  (k) A3-band: candidates are A3-top's order for the pool and draw; a candidate that does not fit stays available for
      later steps; an item fits when the pool WITH it has a Gaussian ratio strictly inside (0.5, 2) (the complement of
      4c's flag, healthcheck_d156.crosses at 2 / 0.5), also when the clean pool's ratio was outside the band; the
      greedy runs once per draw and pool to the budget of rho 0.25 and each rho takes its first min(m, length) items
      (nested); a one-step update from the pool's exact moments screens the candidates and any candidate within
      SCREEN_TOL (relative) of a band edge, and every item added, is decided by health() itself. The clean pool's
      ratio and flag, the greedy's length and whether it stopped for lack of a fitting item are recorded per draw.
  (l) A3-band's "largest m" is its realised m at the grid's largest rho (0.25; the budget caps it at 250).
  (m) P-A3.3: per pool and repair, A3-top at "the same m" injects, in each draw, the top m_d of its order, m_d =
      A3-band's largest m in that draw; each loss is the D-156 mean of the per-draw loss against the same repair's
      unpoisoned TPR; holds when A3-band's is strictly smaller. A cell with m_d = 0 in every draw is UNDECIDED (None),
      not failed: the band is infeasible there (the reason recorded: 'band infeasible: the clean pool already crosses
      the 4c cut' when the clean pool flags in every draw, else that no harmful item keeps the pool inside), so A3-band
      injects nothing and both losses are 0. The pair verdict, D-351 (3)'s reading: False if any decided cell fails,
      True if every decided cell holds and at least one is decided, None if none is decided. Beside each cell, not the
      verdict: the clean pool's Gaussian flag share and the same loss comparison over only the draws with m_d > 0.
      The infeasible cells whose clean pool flags in every draw are listed as a finding (D-352 (2): the band admits
      no attacker where 4c already fires on the clean pool).
  (n) P-A3.2 per pool and repair (four cells: restandardisation on traffic is scored here too, while P-A3.1 takes
      only the pool's score repair); a minimum rho of None (no 10-point loss up to 0.25) reads as '> 0.25' (D-244 (6)):
      benign '> 0.25' holds against any top value, benign finite against top '> 0.25' fails, both '> 0.25' is
      undecided (None). The pair verdict is D-352 (3)'s reading (D-351 (3)'s): False if any decided cell fails, True
      if every decided cell holds and at least one is decided, None if none is decided. Choice (f) of the predictions
      module (most of A3-benign's candidates are each seed's own fit rows, which score low) biases P-A3.2 toward
      holding (rule 5).
  (o) P-A3.4 per pair against each score repair (R_same, R_fresh, quantile transfer), with the same '> 0.25' reading
      and a relative tolerance of 1e-9 (10 x a grid value is not exact in binary); a cell the grid cannot decide is
      None, and the pair verdict is the reading of (n). Its inputs are A3-top's minimum rhos only, which equal 4d B's
      A1 values bit for bit (D-322 reports them): P-A3.4 is fully determined by 4d B and adds no new measurement.
  (p) D-282's combined TPR interval for every threshold-level arm (frozen, threshold re-set, quantile transfer) clean
      and at every attacker and rho (the positives' scores are fixed; only the threshold moves), each reproducing its
      TPR draw by draw; not for restandardisation (the positives' scores move per draw), as D-346 (2)(j). Every TPR
      LOSS reported on a threshold-level arm (each attacker's at its minimum rho; A3-band's and A3-top's at A3-band's
      largest m) has the same interval, paired within item and draw: replicate r is the clean arm's minus the
      poisoned arm's TPR on the same counts row and draw.
  (q) the lane log prints checks and verdict words only, no number (D-346 (4)).
Writes results/phase3_a3_<pair>.json and results/phase3_a3_<pair>_draws.npz (tracked, D-163 (6)), once (exclusive),
with its config's hash (config_hash) and D-352 (5)'s sandbox disclosure (sandbox_disclosure).
Only registered parameters at the 6-thread BLAS pin, reading the final predictions record, write the final names.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_d156 as D  # noqa: E402
import healthcheck_d156 as HC  # noqa: E402
import paired as P  # noqa: E402
import phase3_a1 as A1  # noqa: E402
import phase3_a1_predictions as A1P  # noqa: E402
import phase3_a3_predictions as A3P  # noqa: E402
import populations as POP  # noqa: E402
import poolstudy_d156 as PS  # noqa: E402
from provenance import (  # noqa: E402
    canonical_hash, ensure_utf8_stdout, provenance_block, read_result, sha256_bytes, sha256_file,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
THIS_FILE = "src/phase3_a3.py"
CODE_FILES = [THIS_FILE, A3P.THIS_FILE, *A1.CODE_FILES]
POOLS, RHOS, PREDICTED = A3P.POOLS, A3P.RHOS, A3P.PREDICTED
ATTACKERS = ("A3-top", "A3-benign", "A3-band")
REPAIRS_OF = {"R_same": ("threshold_reset",), "R_fresh": ("threshold_reset",),
              "traffic_1pct": ("quantile_transfer", "restandardise")}
REPAIRS = tuple((pool, rep) for pool in POOLS for rep in REPAIRS_OF[pool])
SCORE_REPAIR = {"R_same": "threshold_reset", "R_fresh": "threshold_reset", "traffic_1pct": "quantile_transfer"}
THRESHOLD_LEVEL = tuple(r for r in REPAIRS if r[1] != "restandardise")        # D-282 supported (choice (p))
# 4d B's names for each repair: (arm, the unpoisoned configuration, the poisoned configuration's prefix); its A1 is A3-top
FOURD = {("R_same", "threshold_reset"): ("score", "score|R_same|n=1000"),
         ("R_fresh", "threshold_reset"): ("score", "score|R_fresh|n=1000"),
         ("traffic_1pct", "quantile_transfer"): ("score", f"score|{A3P.TRAFFIC}"),
         ("traffic_1pct", "restandardise"): ("feature", f"feature|{A3P.TRAFFIC}")}
QUANT = A1.QUANT                                       # fpr, paired_ratio, ratio_to_target, tpr (4d B's put())
HEALTH = ("z_mean", "log_sd_ratio", "gaussian_ratio", "log_gaussian_ratio")
BAND = (0.5, 2.0)                                      # D-332 (3): inside (0.5, 2), the complement of 4c's flag
CUT = (2.0, 0.5)                                       # D-241 (3): the flag, ratio >= 2 or <= 0.5
SCREEN_TOL = 1e-6                                      # choice (k): screened ratios this close to an edge go to health()
SCREEN_AGREE = 1e-9                                    # implementation check: screen vs health() on every item added
FACTOR, P34_FACTOR, RHO_TOL = 1.5, 10.0, 1e-9
THREAD_VARS = A1.THREAD_VARS
PREREG = ("D-332 (the Phase 3 A3 pre-registration: (1)-(7)); D-352 (the review outcome: an infeasible band scored "
          "undefined, the band kept as registered; the pooled scoring's per-cell reading); D-318 (9); "
          "docs/phase3_design.md revision 2; machinery of D-244/D-298 (4d B), D-241/D-264 (4c), D-221/D-242 (P4), "
          "D-282 (TPR interval), D-177 (accuracy line)")
SANDBOX = A3P.SANDBOX                                  # D-352 (5)'s disclosure, one text for both records
CHOICES = {
    "i": "the four repairs are poolstudy_d156.main()'s recal, quantile and feature code paths and A3-top its A1 "
         "expression, so the self-anchor can hold bit for bit (D-332 (1), (6); D-244 (5))",
    "j": "health ratio: healthcheck_d156.health with cal_d's pre scores against the pool's post scores at T_d for every "
         "pool (D-332 (3) 'the pool's post moments against the pre moments'; D-241 (1)-(2); the defender holds no pre "
         "model for fresh items, D-318 (9)); the direct ratio is not reported",
    "k": "A3-band: candidates in A3-top's order; at each step the highest-ranked remaining candidate whose addition "
         "leaves the pool's Gaussian ratio strictly inside (0.5, 2) (also when the clean pool's ratio was outside); a "
         "non-fitting candidate stays available; one greedy per draw and pool to rho 0.25's budget, nested over rho; "
         "a one-step moment update screens, health() decides near the edges and every item added",
    "l": "A3-band's largest m = its realised m at rho 0.25 (the budget caps it at 250)",
    "m": "P-A3.3 per pool and repair: A3-top at A3-band's per-draw largest m; D-156 mean of per-draw TPR losses; "
         "strict '<'; a cell with A3-band's m = 0 in every draw is None, not False (band infeasible: the clean pool "
         "already crosses the 4c cut, or no harmful item keeps the pool inside; the reason is recorded); the pair "
         "verdict (D-351 (3)'s reading) is False if any decided cell fails, True if every decided cell holds and at "
         "least one is decided, None if none is decided; beside each cell, not the verdict, the clean pool's Gaussian "
         "flag share and the loss comparison over only the draws with m > 0; the infeasible cells whose clean pool "
         "flags in every draw are listed as a finding (D-352 (2): the band admits no attacker where 4c already fires "
         "on the clean pool)",
    "n": "P-A3.2 per pool and repair, restandardisation included (P-A3.1 takes only the score repair); None reads "
         "'> 0.25' (D-244 (6)); a cell where both are '> 0.25' is undecided (None); the pair verdict is D-352 (3)'s "
         "reading (False if any decided cell fails, True if every decided cell holds and one is decided, else None); "
         "choice (f)'s fit-row confound biases it toward holding",
    "o": "P-A3.4 per pair against each score repair (R_same, R_fresh, quantile transfer); '> 0.25' as in (n); "
         "relative tolerance 1e-9; a cell the grid cannot decide is None; the pair verdict is D-352 (3)'s reading as "
         "in (n); fully determined by A3-top, which equals 4d B's A1 values bit for bit (D-322 reports them): no new "
         "measurement",
    "p": "D-282 combined TPR interval for every threshold-level arm clean and at every attacker and rho; not for "
         "restandardisation (D-346 (2)(j)); the same interval for every TPR loss reported on a threshold-level arm "
         "(each attacker's at its minimum rho, A3-band's and A3-top's at A3-band's largest m), paired within item "
         "and draw (the clean and the poisoned arm's flags on the same counts row)",
    "q": "the lane log prints checks and verdict words only (D-346 (4))",
}


def bs(a: np.ndarray) -> dict[str, Any]:
    """The D-156 summary of one per-draw array (row 0 the frozen split)."""
    a = np.asarray(a, dtype=np.float64)
    return P.band_summary(a[1:], a[0])


def precompute(S: dict[str, Any]) -> dict[str, Any]:
    """The float64 post-feature table (4d B's ALL_post, then the A3-benign candidates' rows) and the positives'."""
    f64 = S["f64"]
    ALL_post = np.vstack([f64(S["RAW"]["post"]["pool"]), f64(S["RAW"]["post"]["H"]), f64(S["RAW"]["post"]["B"])])
    return {"ALL_post": ALL_post, "PN_post": ALL_post[:S["n_pool"]], "XP": f64(S["RAW"]["post"]["pos"])}


def repair_flags(S: dict[str, Any], X: dict[str, Any], rep: str, T: np.ndarray, E: np.ndarray, XE: np.ndarray,
                 tidx: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """One repair on the pool idx (the clean pool tidx, then the injected items) in one draw: flags on E, flags on the
    positives and the per-seed threshold (None for restandardisation). poolstudy_d156.main()'s recal, quantile and
    feature exactly (choice (i))."""
    SC, nS, target = S["SC"], len(S["seeds"]), S["target"]
    if rep == "threshold_reset":
        t = np.array([P.thr_at_fpr(S["post_table"][idx, s], target) for s in range(nS)])
    elif rep == "quantile_transfer":
        t = np.array([P.quantile_transfer(S["pre_table"][tidx, s], S["post_table"][idx, s], T[s]) for s in range(nS)])
    elif rep == "restandardise":
        mu, sd = PS.feature_stats(X["ALL_post"][idx])
        FE, QP = [], []
        for s, q in enumerate(S["probes"]):
            FE.append(PS.restandardised_scores(q, mu, sd, XE) >= T[s])
            QP.append(PS.restandardised_scores(q, mu, sd, X["XP"]) >= T[s])
        return np.column_stack(FE), np.column_stack(QP), None
    else:
        raise ValueError(f"unknown repair {rep!r}")
    return SC["post"]["pool"][E] >= t, SC["post"]["pos"] >= t, t


def band_greedy(pre_cal: np.ndarray, pool_post: np.ndarray, cand_post: np.ndarray, T: np.ndarray, max_m: int,
                band: tuple[float, float] = BAND, tol: float = SCREEN_TOL) -> dict[str, Any]:
    """A3-band for one draw and pool (D-332 (3); choice (k)). pool_post (n, S): the clean pool's post scores;
    cand_post (C, S): the candidates' post scores in A3-top's order. At each step the first remaining candidate (in that
    order) whose addition leaves healthcheck_d156.health's Gaussian ratio (cal_d's pre scores against the pool's post
    scores at T) strictly inside the band is added; a candidate that does not fit stays for later steps. A one-step
    update of the pool's exact moments screens every remaining candidate at once; health() itself decides every
    candidate the screen puts within tol (relative) of an edge, and confirms every candidate added. Returns the
    positions added (into cand_post, in order), whether it stopped for lack of a fitting candidate, the number of
    health() decisions, the largest relative screen error on an added item and the count of screen/health() disagreements
    on an added item (0 unless the screen is off by more than tol)."""
    lo, hi = band
    lg_pre = norm.logsf((T - pre_cal.mean(axis=0)) / pre_cal.std(axis=0))       # health()'s g_pre, per seed
    cur = np.asarray(pool_post, dtype=np.float64)
    avail = np.ones(cand_post.shape[0], dtype=bool)
    chosen: list[int] = []
    n_direct, worst, disagree, stopped = 0, 0.0, 0, False
    while len(chosen) < max_m:
        pa = np.flatnonzero(avail)
        if pa.size == 0:
            stopped = True
            break
        x = cand_post[pa]
        n1 = cur.shape[0] + 1
        mu = cur.mean(axis=0)
        m2 = ((cur - mu) ** 2).sum(axis=0)
        mu1 = mu + (x - mu) / n1
        s1 = np.sqrt((m2 + (x - mu) * (x - mu1)) / n1)
        r = np.exp(norm.logsf((T - mu1) / s1) - lg_pre).mean(axis=1)
        near = (np.abs(r - lo) <= tol * lo) | (np.abs(r - hi) <= tol * hi)
        pick = None
        for i in np.flatnonzero(((r > lo) & (r < hi)) | near):
            trial = np.vstack([cur, cand_post[pa[i]][None, :]])
            rd = HC.health(pre_cal, trial, T)["gaussian_ratio"]
            n_direct += 1
            if not lo < rd < hi:
                if not near[i]:
                    disagree += 1                     # the screen said clearly inside; health() decides (outside)
                continue
            worst = max(worst, abs(float(r[i]) - rd) / rd)
            pick, cur = int(pa[i]), trial
            break
        if pick is None:
            stopped = True
            break
        chosen.append(pick)
        avail[pick] = False
    return {"chosen": np.asarray(chosen, dtype=np.int64), "stopped_no_fit": stopped, "n_direct": n_direct,
            "screen_max_rel_err": worst, "screen_disagreements": disagree}


def measure(S: dict[str, Any], X: dict[str, Any], which: tuple[str, ...],
            log: Callable[[str], None] = print) -> dict[str, np.ndarray]:
    """Per draw (row 0 the frozen split), for `which` (from 'clean', 'A3-top', 'A3-benign', 'A3-band'): every per-draw
    array. 'clean' is the frozen arm, the four unpoisoned repairs and each clean pool's health. Keys: frozen|<q>,
    clean|<pool>|<repair or health>|<q>, <attacker>|<pool>|rho=<rho>|<repair or health>|<q>, A3-band|<pool>|rho=<rho>|
    m_realised, A3-band|<pool>|greedy|<stat>, A3-top@band_m|<pool>|<repair>|<q> and A3-top@band_m|<pool>|m."""
    target, SC, pt = S["target"], S["SC"], S["post_table"]
    acc: dict[str, list[Any]] = {}
    put = lambda k, x: acc.setdefault(k, []).append(x)  # noqa: E731
    t0 = time.time()
    for d, dr in enumerate(S["draws"]):
        E, T = dr["E"], S["T"][d]
        F_pre = SC["pre"]["pool"][E] >= T
        pre_cal = SC["pre"]["pool"][dr["cal"]]
        XE = X["PN_post"][E]

        def arms(prefix: str, pool: str, tidx: np.ndarray, idx: np.ndarray) -> None:
            for rep in REPAIRS_OF[pool]:
                F, Q, t = repair_flags(S, X, rep, T, E, XE, tidx, idx)
                r = A1.arm_numbers(F_pre, F, Q, target)
                for qn in QUANT:
                    put(f"{prefix}|{rep}|{qn}", r[qn])
                if t is not None:
                    put(f"{prefix}|{rep}|threshold_per_seed", t)

        def health(prefix: str, idx: np.ndarray) -> None:
            h = HC.health(pre_cal, pt[idx], T)
            for qn in HEALTH:
                put(f"{prefix}|health|{qn}", h[qn])

        if "clean" in which:
            r = A1.arm_numbers(F_pre, SC["post"]["pool"][E] >= T, SC["post"]["pos"] >= T, target)
            for qn in QUANT:
                put(f"frozen|{qn}", r[qn])
            put("frozen|threshold_per_seed", T)
            put("frozen|fpr_pre_per_seed", P.floored_fpr(F_pre))          # the D-257 floor's FPR_pre, per seed
            for pool in POOLS:
                tidx = dr[A3P.POOL_OF_DRAW[pool]]
                arms(f"clean|{pool}", pool, tidx, tidx)
                health(f"clean|{pool}", tidx)
        for att in PREDICTED:
            if att not in which:
                continue
            for pool in POOLS:
                tidx = dr[A3P.POOL_OF_DRAW[pool]]
                inj = A3P.injected(S, att, pool, d)
                for rho in RHOS:
                    idx = np.concatenate([tidx, inj[:A3P.m_of(rho, len(tidx))]])
                    arms(f"{att}|{pool}|rho={rho}", pool, tidx, idx)
                    health(f"{att}|{pool}|rho={rho}", idx)
        if "A3-band" in which:
            for pool in POOLS:
                tidx = dr[A3P.POOL_OF_DRAW[pool]]
                order = S["top_order"][pool][d]
                g = band_greedy(pre_cal, pt[tidx], pt[order], T, S["m_max"][pool])
                seq = order[g["chosen"]]
                put(f"A3-band|{pool}|greedy|length", float(len(seq)))
                put(f"A3-band|{pool}|greedy|stopped_no_fit", float(g["stopped_no_fit"]))
                put(f"A3-band|{pool}|greedy|n_health_decisions", float(g["n_direct"]))
                put(f"A3-band|{pool}|greedy|screen_max_rel_err", g["screen_max_rel_err"])
                put(f"A3-band|{pool}|greedy|screen_disagreements", float(g["screen_disagreements"]))
                for rho in RHOS:
                    md = min(A3P.m_of(rho, len(tidx)), len(seq))
                    idx = np.concatenate([tidx, seq[:md]])
                    put(f"A3-band|{pool}|rho={rho}|m_realised", float(md))
                    arms(f"A3-band|{pool}|rho={rho}", pool, tidx, idx)
                    health(f"A3-band|{pool}|rho={rho}", idx)
                # P-A3.3 (choice (m)): A3-top at A3-band's largest m in this draw
                put(f"A3-top@band_m|{pool}|m", float(len(seq)))
                arms(f"A3-top@band_m|{pool}", pool, tidx, np.concatenate([tidx, order[:len(seq)]]))
        if d % 50 == 0:
            log(f"  {'+'.join(which)}: draw {d} of {len(S['draws']) - 1} ({time.time() - t0:.0f} s)")
    out = {k: np.asarray(v, dtype="<f8") for k, v in acc.items()}
    for k in [k for k in out if k.endswith("|health|gaussian_ratio")]:
        out[k[:-len("gaussian_ratio")] + "gaussian_flag"] = HC.crosses(out[k], CUT).astype("<f8")
    return out


# ---- summaries, outcomes and the anchor ---------------------------------------------------------------------------

def health_block(a: dict[str, np.ndarray], prefix: str) -> dict[str, Any]:
    f = a[f"{prefix}|health|gaussian_flag"]
    return {**{qn: bs(a[f"{prefix}|health|{qn}"]) for qn in HEALTH},
            "share_of_draws_gaussian_flag_2x": float(f[1:].mean()), "frozen_split_gaussian_flag": bool(f[0])}


def summaries(a: dict[str, np.ndarray], target: float = P.TARGET_FPR) -> dict[str, Any]:
    """The D-156 cell blocks of whatever `a` holds: frozen, clean (per pool: repairs and health) and per attacker, pool
    and rho (m, repairs, health)."""
    out: dict[str, Any] = {}
    if "frozen|tpr" in a:
        out["frozen"] = {qn: bs(a[f"frozen|{qn}"]) for qn in QUANT}
        out["frozen"]["floor_fpr_pre_ratio_to_target"] = bs(a["frozen|fpr_pre_per_seed"].mean(axis=1) / target)
    if "clean|R_same|threshold_reset|tpr" in a:
        out["clean"] = {pool: {"repairs": {rep: {qn: bs(a[f"clean|{pool}|{rep}|{qn}"]) for qn in QUANT}
                                           for rep in REPAIRS_OF[pool]},
                               "health": health_block(a, f"clean|{pool}")} for pool in POOLS}
    for att in ATTACKERS:
        if f"{att}|R_same|rho={RHOS[0]}|threshold_reset|tpr" not in a:
            continue
        blk: dict[str, Any] = {}
        for pool in POOLS:
            for rho in RHOS:
                p = f"{att}|{pool}|rho={rho}"
                m_budget = A3P.m_of(rho, PS.DEFAULT_POOL)
                if att == "A3-band":
                    mr = a[f"{p}|m_realised"]
                    m = {"budget": m_budget, "realised": bs(mr), "realised_min_over_draws": int(mr[1:].min()),
                         "realised_max_over_draws": int(mr[1:].max())}
                else:
                    m = m_budget
                blk.setdefault(pool, {})[f"rho={rho}"] = {
                    "m": m, "repairs": {rep: {qn: bs(a[f"{p}|{rep}|{qn}"]) for qn in QUANT} for rep in REPAIRS_OF[pool]},
                    "health": health_block(a, p)}
        out[att] = blk
    return out


def min_rho_table(cells: dict[str, Any]) -> dict[str, Any]:
    """D-332 (4): per repair and attacker, the smallest rho whose D-156 mean TPR is at least 10 points below the same
    repair's unpoisoned D-156 mean TPR (4d B's rule, poolstudy_d156.main()), its m and the pool's Gaussian flag share
    of draws there; None when no rho up to 0.25 reaches it."""
    out: dict[str, Any] = {}
    for pool, rep in REPAIRS:
        t0 = cells["clean"][pool]["repairs"][rep]["tpr"]["mean"]
        for att in ATTACKERS:
            if att not in cells:
                continue
            hit = None
            for rho in RHOS:
                if t0 - cells[att][pool][f"rho={rho}"]["repairs"][rep]["tpr"]["mean"] >= PS.TPR_LOSS:
                    hit = rho
                    break
            c = cells[att][pool][f"rho={hit}"] if hit is not None else None
            out.setdefault(f"{pool}|{rep}", {})[att] = {
                "min_rho": hit, "searched_up_to": RHOS[-1], "clean_tpr_mean": t0,
                "m": c["m"] if c else None,
                "tpr_mean_at_min_rho": c["repairs"][rep]["tpr"]["mean"] if c else None,
                "gaussian_flag_share_at_min_rho": c["health"]["share_of_draws_gaussian_flag_2x"] if c else None}
    return out


def anchor_check(cells: dict[str, Any], minrho: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
    """D-332 (6): A3-top's minimum rho against 4d B's selected A1 minimum rho for each of the four repairs (the stop);
    and every D-156 summary of the frozen arm, the four unpoisoned repairs and A3-top at every rho against 4d B's
    configurations exactly (the implementation check, D-346)."""
    cfgs = rec["configurations"][A3P.PSET]
    same, bad = 0, []

    def cmp(label: str, mine: Any, theirs: Any) -> None:
        nonlocal same
        a = json.dumps(json.loads(json.dumps(mine, default=str)), sort_keys=True)  # not a result (D-297): own round trip
        b = json.dumps(theirs, sort_keys=True)
        if a == b:
            same += 1
        else:
            bad.append(f"{label}: {a} != {b}")

    for qn in QUANT:
        cmp(f"frozen {qn}", cells["frozen"][qn], (cfgs.get("frozen") or {}).get(qn))
    rho_cmp, rho_bad = {}, []
    for (pool, rep), (arm, base) in FOURD.items():
        for qn in QUANT:
            cmp(f"clean {pool} {rep} ({base}) {qn}", cells["clean"][pool]["repairs"][rep][qn], (cfgs.get(base) or {}).get(qn))
        for rho in RHOS:
            name = f"{arm}|poison|{pool}|A1|rho={rho}|m={A3P.m_of(rho, PS.DEFAULT_POOL)}"
            for qn in QUANT:
                cmp(f"A3-top {pool} {rep} ({name}) {qn}", cells["A3-top"][pool][f"rho={rho}"]["repairs"][rep][qn],
                    (cfgs.get(name) or {}).get(qn))
        key = f"{A3P.PSET}|{arm}|{pool}|A1"
        theirs = (rec.get("min_rho_for_10_point_tpr_loss") or {}).get(key)
        mine = minrho[f"{pool}|{rep}"]["A3-top"]["min_rho"]
        ok = isinstance(theirs, dict) and "min_rho" in theirs and theirs["min_rho"] == mine
        rho_cmp[f"{pool}|{rep}"] = {"4dB_key": key, "4dB_min_rho": (theirs or {}).get("min_rho"), "a3_top_min_rho": mine,
                                    "equal": ok}
        if not ok:
            rho_bad.append(f"min rho {pool} {rep}: A3-top {mine} != 4d B {key} {theirs}")
    n_cmp = len(QUANT) * (1 + len(FOURD) * (1 + len(RHOS)))
    return {"min_rho": rho_cmp, "min_rho_mismatches": rho_bad, "n_values_identical": same, "n_values_compared": n_cmp,
            "value_mismatches": bad, "identical": not rho_bad and not bad}


def threshold_identity(S: dict[str, Any], a: dict[str, np.ndarray], parr: dict[str, np.ndarray]) -> dict[str, Any]:
    """Implementation check: on R_same and R_fresh, A3-top and A3-benign, every rho, draw and seed, the threshold re-set's
    realised threshold equals the predictions record's t' when the k-th largest score is not tied (at most k scores
    >= t'), else the smallest pool score above t' (+inf if none): paired.thr_at_fpr's rule on the same pool."""
    n, n_tied, bad = 0, 0, []
    for pool in ("R_same", "R_fresh"):
        for att in PREDICTED:
            for rho in RHOS:
                tr = a[f"{att}|{pool}|rho={rho}|threshold_reset|threshold_per_seed"]
                tp = parr[f"{A3P.ckey(pool, att, rho)}|t_prime"]
                for d, dr in enumerate(S["draws"]):
                    tidx = dr[A3P.POOL_OF_DRAW[pool]]
                    m = A3P.m_of(rho, len(tidx))
                    sc = S["post_table"][np.concatenate([tidx, A3P.injected(S, att, pool, d)[:m]])]
                    k = A3P.whole_item_k(S["target"], len(tidx) + m)
                    for s in range(sc.shape[1]):
                        v, col = tp[d, s], sc[:, s]
                        if int((col >= v).sum()) <= k:
                            want = v
                        else:
                            n_tied += 1
                            above = col[col > v]
                            want = float(above.min()) if above.size else math.inf
                        n += 1
                        if tr[d, s] != want:
                            bad.append(f"{att} {pool} rho={rho} draw {d} seed {s}")
    return {"holds": not bad, "n_checked": n, "n_tied_at_k": n_tied, "failures": bad[:20], "n_failures": len(bad)}


def band_checks(a: dict[str, np.ndarray]) -> dict[str, Any]:
    """Implementation check: every A3-band pool with m > 0 is inside the band (health() on the pool itself), and the
    greedy's screen agreed with health() on every item it added."""
    outside, worst, disagree = [], 0.0, 0
    for pool in POOLS:
        for rho in RHOS:
            p = f"A3-band|{pool}|rho={rho}"
            bad = np.flatnonzero((a[f"{p}|m_realised"] > 0) & (a[f"{p}|health|gaussian_flag"] > 0))
            outside += [f"{pool} rho={rho} draw {int(d)}" for d in bad]
        worst = max(worst, float(a[f"A3-band|{pool}|greedy|screen_max_rel_err"].max()))
        disagree += int(a[f"A3-band|{pool}|greedy|screen_disagreements"].sum())
    ok = not outside and worst <= SCREEN_AGREE and disagree == 0
    return {"holds": ok, "pools_outside_band_with_m_gt_0": outside[:20], "n_outside": len(outside),
            "screen_max_rel_err": worst, "screen_bound": SCREEN_AGREE, "screen_disagreements": disagree}


def d282_replicates(S: dict[str, Any], thr: np.ndarray, tpr: np.ndarray, counts: np.ndarray, label: str) -> np.ndarray:
    """D-282's combined replicates of one threshold-level arm's TPR (choice (p)): the fixed eval positives against the
    arm's per-draw, per-seed threshold; its inputs must reproduce the arm's TPR draw by draw, else a refusal (as A1).
    Replicate r uses draw 1 + ((r - 1) mod n_draws) and counts row r - 1 (paired.combined_tpr_replicates)."""
    pos = S["SC"]["post"]["pos"]
    again = np.array([float(P.rate(pos >= thr[d]).mean()) for d in range(thr.shape[0])])
    if not np.array_equal(again, tpr):
        raise SystemExit(f"REFUSING: {label}: the D-282 inputs do not reproduce its TPR draw by draw")
    return P.combined_tpr_replicates(thr[1:], S["SC"]["pre"]["pos"], pos, counts)["tpr_post"]


def d282_cell(S: dict[str, Any], thr: np.ndarray, tpr: np.ndarray, counts: np.ndarray, label: str) -> dict[str, Any]:
    """D-282's combined TPR interval for one threshold-level arm (choice (p))."""
    return {"tpr_post_combined": P.ci(d282_replicates(S, thr, tpr, counts, label))}


def d282_loss(clean_reps: np.ndarray, poisoned_reps: np.ndarray) -> dict[str, Any]:
    """D-282's combined interval of a TPR LOSS (clean minus poisoned), paired within item and draw: both replicate
    arrays come from the same counts matrix and draw mapping, so replicate r of each uses the same counts row and the
    same draw (choice (p))."""
    if clean_reps.shape != poisoned_reps.shape:
        raise ValueError("d282_loss: the two arms' replicates are not paired")
    return {"tpr_loss_combined": P.ci(clean_reps - poisoned_reps),
            "paired": "within item and draw (the same counts row and draw for the clean and the poisoned arm)"}


def add_d282(S: dict[str, Any], a: dict[str, np.ndarray], cells: dict[str, Any], counts: np.ndarray) -> dict[str, np.ndarray]:
    """D-282's interval on every threshold-level arm's TPR (choice (p)); returns each arm's replicates (for the
    losses)."""
    reps: dict[str, np.ndarray] = {}

    def one(p: str) -> dict[str, Any]:
        reps[p] = d282_replicates(S, a[f"{p}|threshold_per_seed"], a[f"{p}|tpr"], counts, p)
        return {"tpr_post_combined": P.ci(reps[p])}

    cells["frozen"]["tpr_d282_combined"] = one("frozen")
    for pool, rep in THRESHOLD_LEVEL:
        cells["clean"][pool]["repairs"][rep]["tpr_d282_combined"] = one(f"clean|{pool}|{rep}")
        for att in ATTACKERS:
            for rho in RHOS:
                cells[att][pool][f"rho={rho}"]["repairs"][rep]["tpr_d282_combined"] = one(f"{att}|{pool}|rho={rho}|{rep}")
    return reps


def add_d282_losses(S: dict[str, Any], a: dict[str, np.ndarray], reps: dict[str, np.ndarray], minrho: dict[str, Any],
                    bando: dict[str, Any], counts: np.ndarray) -> None:
    """D-282 (1) for every TPR loss reported on a threshold-level arm (choice (p)): each attacker's loss at its minimum
    rho (outcomes.min_rho_10pt_tpr_loss) and A3-band's and A3-top's at A3-band's largest m (outcomes.a3_band), each the
    clean arm's replicates minus the poisoned arm's, paired within item and draw."""
    top = f"rho={RHOS[-1]}"
    for pool, rep in THRESHOLD_LEVEL:
        clean = reps[f"clean|{pool}|{rep}"]
        for att, row in minrho[f"{pool}|{rep}"].items():
            row["tpr_loss_at_min_rho_d282_combined"] = (
                d282_loss(clean, reps[f"{att}|{pool}|rho={row['min_rho']}|{rep}"]) if row["min_rho"] is not None else None)
        p = f"A3-top@band_m|{pool}|{rep}"
        top_reps = d282_replicates(S, a[f"{p}|threshold_per_seed"], a[f"{p}|tpr"], counts, p)
        r = bando[pool]["repairs"][rep]
        r["tpr_loss_at_largest_m_d282_combined"] = d282_loss(clean, reps[f"A3-band|{pool}|{top}|{rep}"])
        r["a3_top_tpr_loss_at_same_m_d282_combined"] = d282_loss(clean, top_reps)


def band_outcomes(a: dict[str, np.ndarray]) -> dict[str, Any]:
    """D-332 (4) for A3-band per pool: its largest m (choice (l)), the share of draws whose greedy stopped for lack of a
    fitting item, the clean pool's flag share (the band's starting point), and per repair the TPR loss at its largest m
    with A3-top's at the same per-draw m (choice (m)); beside, the same comparison over only the draws with m > 0.
    Keys starting with '_' feed score() and are removed before the record is written."""
    out: dict[str, Any] = {}
    top = f"rho={RHOS[-1]}"
    for pool in POOLS:
        mr = a[f"A3-band|{pool}|{top}|m_realised"]
        pos_m = mr[1:] > 0
        blk: dict[str, Any] = {
            "largest_m": bs(mr), "largest_m_max_over_draws": int(mr[1:].max()),
            "share_of_draws_largest_m_0": float((mr[1:] == 0).mean()),
            "share_of_draws_stopped_no_fit": float(a[f"A3-band|{pool}|greedy|stopped_no_fit"][1:].mean()),
            "clean_pool_gaussian_flag_share": float(a[f"clean|{pool}|health|gaussian_flag"][1:].mean()),
            "repairs": {}}
        for rep in REPAIRS_OF[pool]:
            clean = a[f"clean|{pool}|{rep}|tpr"]
            lb = clean - a[f"A3-band|{pool}|{top}|{rep}|tpr"]
            lt = clean - a[f"A3-top@band_m|{pool}|{rep}|tpr"]
            n_pos = int(pos_m.sum())
            beside = {"n_draws_with_m_gt_0": n_pos, "n_draws": int(pos_m.size),
                      "a3_band_tpr_loss_mean": float(lb[1:][pos_m].mean()) if n_pos else None,
                      "a3_top_tpr_loss_at_same_m_mean": float(lt[1:][pos_m].mean()) if n_pos else None,
                      "band_loss_smaller": bool(lb[1:][pos_m].mean() < lt[1:][pos_m].mean()) if n_pos else None,
                      "label": "beside, not the verdict: the mean over only the draws where A3-band injected (m > 0)"}
            blk["repairs"][rep] = {"tpr_loss_at_largest_m": bs(lb), "a3_top_tpr_loss_at_same_m": bs(lt),
                                   "comparison_over_draws_with_m_gt_0_beside": beside,
                                   "_band_loss_mean": float(lb[1:].mean()), "_top_loss_mean": float(lt[1:].mean()),
                                   "_infeasible": bool((mr[1:] == 0).all())}
        out[pool] = blk
    return out


# ---- scoring -----------------------------------------------------------------------------------------------------

def accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """D-177's accuracy line on TPR: the share of cells within x1.5 and the median |log(realised / predicted)| over ALL
    cells. A cell whose predicted or realised TPR is 0 is a miss (D-322 (4)): an infinite error, outside x1.5, inside
    the median (None, with median_is_a_miss, when the median falls on a miss). Beside, the same over the cells with both
    positive."""
    e = np.array([abs(math.log(r["realised_tpr"] / r["predicted_tpr"])) if r["predicted_tpr"] > 0 and r["realised_tpr"] > 0
                  else math.inf for r in rows], dtype=np.float64)
    ok = e[np.isfinite(e)]
    med = float(np.median(e)) if e.size else None
    return {"n_cells": int(e.size),
            "n_cells_predicted_zero": int(sum(r["predicted_tpr"] <= 0 for r in rows)),
            "n_cells_realised_zero_predicted_positive": int(sum(r["realised_tpr"] <= 0 < r["predicted_tpr"] for r in rows)),
            "share_within_factor_1_5": float((e <= math.log(FACTOR)).mean()) if e.size else None,
            "median_abs_log_realised_over_predicted": med if med is not None and math.isfinite(med) else None,
            "median_is_a_miss": bool(med is not None and not math.isfinite(med)),
            "beside_positive_cells_only": {
                "n_cells": int(ok.size), "share_within_factor_1_5": float((ok <= math.log(FACTOR)).mean()) if ok.size else None,
                "median_abs_log_realised_over_predicted": float(np.median(ok)) if ok.size else None}}


def at_least(a: float | None, b: float | None, factor: float = 1.0) -> bool | None:
    """Whether minimum rho a >= factor * b, a None reading '> 0.25' (beyond the grid, D-244 (6)); None when the grid
    cannot decide (choices (n), (o)). factor >= 1."""
    top = RHOS[-1]
    if a is None and b is None:
        return None
    if a is None:
        return True if factor * b <= top * (1 + RHO_TOL) else None
    if b is None:
        return False
    return bool(a >= factor * b * (1 - RHO_TOL))


INFEASIBLE_CLEAN = "band infeasible: the clean pool already crosses the 4c cut"
INFEASIBLE_NO_FIT = "band infeasible: no harmful item keeps the pool strictly inside (0.5, 2) in any draw"
INFEASIBLE_MIXED = ("band infeasible: in every draw the clean pool already crosses the 4c cut or no harmful item keeps "
                    "the pool strictly inside (0.5, 2)")


def decided_verdict(v: list[bool | None]) -> bool | None:
    """The pair verdict of P-A3.2, P-A3.3 and P-A3.4 over their cells (choices (m)-(o); D-352 (3)'s reading, which is
    D-351 (3)'s): False if any decided cell fails, True if every decided cell holds and at least one is decided, None if
    none is decided."""
    d = [x for x in v if x is not None]
    if any(x is False for x in d):
        return False
    return True if d else None


def p33_cell(r: dict[str, Any], clean_flag_share: float) -> dict[str, Any]:
    """One P-A3.3 cell (choice (m)): None with the reason where A3-band's m is 0 in every draw (the band is
    infeasible), else whether A3-band's mean TPR loss at its largest m is strictly below A3-top's at the same m."""
    if r["_infeasible"]:
        holds = None
        why = (INFEASIBLE_CLEAN if clean_flag_share == 1.0 else INFEASIBLE_NO_FIT if clean_flag_share == 0.0
               else INFEASIBLE_MIXED)
    else:
        holds, why = bool(r["_band_loss_mean"] < r["_top_loss_mean"]), None
    return {"a3_band_tpr_loss_at_largest_m": r["_band_loss_mean"], "a3_top_tpr_loss_at_same_m": r["_top_loss_mean"],
            "band_infeasible_m0_every_draw": r["_infeasible"], "holds": holds, "why_undecided": why,
            "clean_pool_gaussian_flag_share": clean_flag_share,
            "comparison_over_draws_with_m_gt_0_beside": r["comparison_over_draws_with_m_gt_0_beside"],
            **{k: r[k] for k in ("tpr_loss_at_largest_m_d282_combined", "a3_top_tpr_loss_at_same_m_d282_combined")
               if k in r}}


def score(cells: dict[str, Any], minrho: dict[str, Any], bando: dict[str, Any],
          parr: dict[str, np.ndarray], ident: dict[str, Any]) -> dict[str, Any]:
    """D-332 (5) per pair: P-A3.1's accuracy line, P-A3.2, P-A3.3 and P-A3.4 (choices (m)-(o)); ident is
    threshold_identity's result (the tie count of the threshold re-set cells)."""
    rows = []
    for pool in POOLS:
        rep = SCORE_REPAIR[pool]
        for att in PREDICTED:
            for rho in RHOS:
                raw = parr[f"{A3P.ckey(pool, att, rho)}|predicted_tpr"].mean(axis=1)
                rows.append({"pool": pool, "repair": rep, "attacker": att, "rho": rho, "m": A3P.m_of(rho, PS.DEFAULT_POOL),
                             "realised_tpr": cells[att][pool][f"rho={rho}"]["repairs"][rep]["tpr"]["mean"],
                             "predicted_tpr": P.band_summary(raw[1:], raw[0])["mean"]})
    ident_rows = [r for r in rows if r["repair"] == "threshold_reset"]
    qt_rows = [r for r in rows if r["repair"] == "quantile_transfer"]
    p31 = {"cells": "pool x attacker (A3-top, A3-benign) x rho, the pool's score repair; realised and predicted TPR as "
                    "cell D-156 means (the predictions record)",
           "accuracy": accuracy(rows),
           "beside_quantile_transfer_cells_only": {
               "label": "beside the registered line (accuracy, every cell): the quantile-transfer cells only, the cells "
                        "where the prediction is not the repair's own rule",
               **accuracy(qt_rows)},
           "beside_threshold_reset_cells_only": {
               "label": "beside the registered line (accuracy, every cell): the threshold re-set cells only, where "
                        "agreement is mechanical (the prediction is the repair's own rule, up to ties at the k-th place)",
               **accuracy(ident_rows)},
           "n_identity_cells": len(ident_rows),
           "n_identity_cells_realised_equal_predicted": int(sum(r["realised_tpr"] == r["predicted_tpr"] for r in ident_rows)),
           "n_tied_at_k": int(ident["n_tied_at_k"]),
           "n_thresholds_in_identity_cells": int(ident["n_checked"]),
           "by_pool": {pool: accuracy([r for r in rows if r["pool"] == pool]) for pool in POOLS},
           "by_attacker": {att: accuracy([r for r in rows if r["attacker"] == att]) for att in PREDICTED},
           "expectation": "at least 80% of cells within x1.5 (D-332 (5)), over all pairs (pooled; not computed here). "
                          "About two thirds of the cells (the threshold re-set cells, n_identity_cells of n_cells) agree "
                          "mechanically: there the prediction is the repair's own rule on the same pool, equal up to ties "
                          "at the k-th place (n_tied_at_k thresholds); the quantile-transfer cells, where D-332's 1% rule "
                          "is not the repair's rule and the cells are expected to be under-predicted (choice (b)), are "
                          "the line beside",
           "note": "on the threshold re-set the prediction is the repair's own rule applied to the attacker's items, so "
                   "those cells agree up to ties at the k-th place (the threshold identity check)",
           "per_cell": rows}
    p32 = {}
    for pool, rep in REPAIRS:
        t, b = minrho[f"{pool}|{rep}"]["A3-top"]["min_rho"], minrho[f"{pool}|{rep}"]["A3-benign"]["min_rho"]
        p32[f"{pool}|{rep}"] = {"a3_top_min_rho": t, "a3_benign_min_rho": b, "holds": at_least(b, t)}
    v32 = [c["holds"] for c in p32.values()]
    p33 = {f"{pool}|{rep}": p33_cell(bando[pool]["repairs"][rep], bando[pool]["clean_pool_gaussian_flag_share"])
           for pool, rep in REPAIRS}
    v33 = [c["holds"] for c in p33.values()]
    fires = sorted(k for k, c in p33.items() if c["band_infeasible_m0_every_draw"] and c["why_undecided"] == INFEASIBLE_CLEAN)
    rs = minrho["traffic_1pct|restandardise"]["A3-top"]["min_rho"]
    p34 = {}
    for pool in POOLS:
        sr = minrho[f"{pool}|{SCORE_REPAIR[pool]}"]["A3-top"]["min_rho"]
        p34[f"{pool}|{SCORE_REPAIR[pool]}"] = {"score_repair_min_rho": sr, "restandardise_min_rho": rs,
                                               "holds": at_least(rs, sr, P34_FACTOR)}
    v34 = [c["holds"] for c in p34.values()]
    counted = lambda v: {"n_cells_decided": sum(x is not None for x in v), "n_cells_holding": sum(x is True for x in v),  # noqa: E731
                         "n_cells_failing": sum(x is False for x in v), "n_cells_undecided": sum(x is None for x in v)}
    reading = ("the pair verdict is D-352 (3)'s reading (D-351 (3)'s): False if any decided cell fails, True if every "
               "decided cell holds and at least one is decided, None if none is decided")
    return {
        "P-A3.1": p31,
        "P-A3.2": {"cells": p32, "holds_in_this_pair": decided_verdict(v32), **counted(v32),
                   "rule": "A3-benign's minimum rho >= A3-top's per pool and repair, restandardisation on traffic "
                           "included (P-A3.1 takes only the score repair); None reads '> 0.25' (D-244 (6)); a cell where "
                           "both are '> 0.25' is undecided (None); " + reading + "; a pair where it is lower is reported "
                           "as a finding (D-332 (5)). A confound with a direction (choice (f), D-352 (2), rule 5): about "
                           "80% of A3-benign's candidates are each seed's own D-117 fit rows (the share among the injected "
                           "items is benign_fit_row_share_per_seed), which score low under the probe fitted on them "
                           "(F-045 / F-060 keep fit rows out of 4d's pools); this weakens A3-benign and biases P-A3.2 "
                           "toward holding"},
        "P-A3.3": {"cells": p33, "holds_in_this_pair": decided_verdict(v33),
                   "n_cells_decided": sum(x is not None for x in v33), "n_cells_band_infeasible": sum(x is None for x in v33),
                   "finding_band_admits_no_attacker_where_4c_fires_on_the_clean_pool": {
                       "cells": fires, "n_cells": len(fires),
                       "label": "a finding (D-352 (2)), not a verdict: the infeasible cells whose clean pool already "
                                "crosses the 4c cut in every draw, so A3-band can inject nothing"},
                   "rule": "A3-band's mean TPR loss at its largest m < A3-top's at the same per-draw m (choice (m)); a "
                           "cell where A3-band's m is 0 in every draw is None (band infeasible, why_undecided), not "
                           "False (D-352 (2)); " + reading + "; the clean pool's flag share and the comparison over the "
                           "draws with m > 0 are beside, not the verdict; whether A3-band reaches a 10-point loss is "
                           "outcomes.min_rho (no direction pre-registered)"},
        "P-A3.4": {"cells": p34, "holds_in_this_pair": decided_verdict(v34), **counted(v34),
                   "rule": "A3-top's minimum rho on restandardisation >= 10 x each score repair's (choice (o)); None "
                           "where the grid cannot decide; " + reading + "; fully determined by A3-top, which equals 4d "
                           "B's A1 values bit for bit (the self-anchor; D-322 reports them): no new measurement"},
    }


# ---- the refusals (each a function, so the tests exercise them) -------------------------------------------------------

def check_predictions(pdoc: dict[str, Any], pnpz_sha: str | None, fn_sha: str, pair: str, n_reps: int, seeds: list[int],
                      target: float, final: bool) -> None:
    """D-332 (5), (7), before any computation: the record is the A3 predictions record of this pair, made by this
    prediction function (the sha256 of its file now), at these parameters; its npz is the bytes it records."""
    pf = pdoc.get("prediction_function") or {}
    if pdoc.get("run_kind") != "phase3_a3_predictions" or (pdoc.get("config") or {}).get("pair") != pair:
        raise SystemExit(f"REFUSING: not the A3 predictions record of {pair}")
    if pf.get("module") != A3P.THIS_FILE or pf.get("function") != "predicted_tpr" or pf.get("sha256") != fn_sha:
        raise SystemExit(f"REFUSING: the predictions record was made by another prediction function "
                         f"({str(pf.get('sha256'))[:12]} vs {fn_sha[:12]})")
    want = {"n_reps": n_reps, "seeds": list(seeds), "target_fpr": target, "rhos": list(RHOS),
            "predicted_attackers": list(PREDICTED), "pool_n": PS.DEFAULT_POOL}
    got = {k: pdoc["config"].get(k) for k in want}
    if got != want:
        raise SystemExit(f"REFUSING: the predictions record has other parameters: {got}")
    if final:
        if pdoc.get("registered_parameters") is not True:
            raise SystemExit("REFUSING the final name: the predictions record was not a registered run")
        th = pdoc.get("blas_threads") or {}
        if any(th.get(v) != "6" for v in THREAD_VARS):
            raise SystemExit(f"REFUSING the final name: the predictions record was not made at the 6-thread pin: {th}")
        want_npz = f"results/phase3_a3_predictions_{A1P.core_of(pair)}_draws.npz"
        if (pdoc.get("draws_npz") or {}).get("file") != want_npz:
            raise SystemExit(f"REFUSING the final name: the predictions record's npz is not {want_npz}")
    if pnpz_sha is None or pnpz_sha != (pdoc.get("draws_npz") or {}).get("sha256"):
        raise SystemExit("REFUSING: the predictions' npz is missing or not the bytes the record names")


def check_predictions_against(S: dict[str, Any], pdoc: dict[str, Any], parr: dict[str, np.ndarray]) -> None:
    """After setup: the same draws (index hash), inputs and injections (hash), every array equal to this code's
    recomputation from the clean pools and the injected items' scores, bit for bit, and the JSON's 'predicted' block
    equal to the JSON round trip of predicted_cells on those arrays."""
    if S["draws_sha256"] != pdoc["config"].get("draws_index_sha256"):
        raise SystemExit("REFUSING: the predictions record names other draws")
    if pdoc.get("input_cache_sha256") != S["input_sha"]:
        raise SystemExit("REFUSING: the predictions record names other inputs")
    if S["injections_sha256"] != pdoc["config"].get("injections_sha256"):
        raise SystemExit("REFUSING: the predictions record names other injections")
    mine = A3P.prediction_arrays(S)
    off = sorted(set(mine) ^ set(parr)) + [k for k in mine if k in parr and not (
        mine[k].dtype == parr[k].dtype and mine[k].shape == parr[k].shape and np.array_equal(mine[k], parr[k]))]
    if off:
        raise SystemExit(f"REFUSING: the predictions record's arrays are not this code's on these inputs: {off[:6]}")
    mine_cells = json.dumps(json.loads(json.dumps(A3P.predicted_cells(S, parr), default=str)),  # not a result (D-297): own round trip
                            sort_keys=True)
    if mine_cells != json.dumps(pdoc.get("predicted"), sort_keys=True):
        raise SystemExit("REFUSING: the predictions record's 'predicted' block is not predicted_cells of its own arrays")


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Phase 3 A3 simulation (D-332)")
    ap.add_argument("--pair", required=True, choices=POP.UPDATE_PAIRS_REPAIR)
    ap.add_argument("--n-reps", type=int, default=A3P.REGISTERED["n_reps"])
    ap.add_argument("--target-fpr", type=float, default=A3P.REGISTERED["target_fpr"])
    ap.add_argument("--seeds", type=int, nargs="*", default=A3P.REGISTERED["seeds"])
    ap.add_argument("--predictions", default=None, help="the predictions record (default: its final name)")
    ap.add_argument("--out", default=None, help="TESTING: a JSON path outside results/")
    ap.add_argument("--anchor-only", action="store_true",
                    help="TESTING ONLY: stop after the self-anchor (clean and A3-top only), write nothing")
    args = ap.parse_args()
    t0 = time.time()
    core = A1P.core_of(args.pair)
    registered = (args.n_reps == A3P.REGISTERED["n_reps"] and args.target_fpr == A3P.REGISTERED["target_fpr"]
                  and list(args.seeds) == A3P.REGISTERED["seeds"])
    final_path = (RESULTS / f"phase3_a3_{core}.json").resolve()
    out_path = Path(args.out).resolve() if args.out else final_path
    npz_path = out_path.with_name(out_path.stem + "_draws.npz")
    final = out_path == final_path
    pred_final = (RESULTS / f"phase3_a3_predictions_{core}.json").resolve()
    pred_path = Path(args.predictions).resolve() if args.predictions else pred_final
    for p in (out_path, npz_path):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists; results are never overwritten")
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test run (--out) must not write under results/")
    threads = {v: os.environ.get(v) for v in THREAD_VARS}
    if final and (not registered or any(t != "6" for t in threads.values()) or pred_path != pred_final
                  or args.anchor_only):
        raise SystemExit(f"REFUSING the final name: registered parameters {registered}, threads {threads}, predictions "
                         f"{pred_path.name}, anchor-only {args.anchor_only}")
    if not pred_path.is_file():                       # D-332 (5), (7): the predictions first
        raise SystemExit(f"REFUSING: no predictions record {pred_path}; run src/phase3_a3_predictions.py first (D-332 (7))")
    code0 = P.code_block(CODE_FILES)
    pred_sha = sha256_file(pred_path)
    pdoc = read_result(pred_path)                     # registered read: listed with its sha256 (D-297 (1))
    pnpz_path = pred_path.with_name(pred_path.stem + "_draws.npz")
    pnpz_b = pnpz_path.read_bytes() if pnpz_path.is_file() else None
    pnpz_sha = sha256_bytes(pnpz_b) if pnpz_b is not None else None
    check_predictions(pdoc, pnpz_sha, code0["sha256"][A3P.THIS_FILE], args.pair, args.n_reps, list(args.seeds),
                      args.target_fpr, final)
    with np.load(io.BytesIO(pnpz_b), allow_pickle=False) as z:          # the very bytes whose sha256 was checked
        parr = {k: z[k] for k in z.files}
    S = A3P.setup(args.pair, args.n_reps, list(args.seeds), args.target_fpr)
    check_predictions_against(S, pdoc, parr)
    rec_rel = f"results/{A1P.record_of_record_stem(core)}.json"
    rec_sha = sha256_file(ROOT / rec_rel)
    rec = read_result(ROOT / rec_rel)
    bad_rec = A1.check_record_of_record(rec, S, registered)
    if bad_rec:
        raise SystemExit(f"REFUSING: {rec_rel} is not the 4d B record of these inputs, probe and draws: {bad_rec}")
    print(f"A3 {args.pair} [{S['axis']}] L{S['L_sel']} C {S['C_sel']:g}: {len(S['draws']) - 1} draws + frozen; "
          f"predictions {pred_sha[:12]}; 4d B record {rec_rel}; registered {registered}", flush=True)
    X = precompute(S)
    log = lambda m: print(m, flush=True)  # noqa: E731

    # the clean arms and A3-top first: the self-anchor before any A3-benign or A3-band number (D-332 (6))
    a = measure(S, X, ("clean", "A3-top"), log)
    cells = summaries(a, S["target"])
    minrho = min_rho_table(cells)
    if registered:
        anc = anchor_check(cells, minrho, rec)
        anchor = {"checked": True, **anc}
        print(f"  self-anchor: min rho {sum(v['equal'] for v in anc['min_rho'].values())} of {len(anc['min_rho'])} "
              f"equal to 4d B's A1; {anc['n_values_identical']} of {anc['n_values_compared']} 4d B summaries identical; "
              f"{'PASS' if anc['identical'] else 'FAIL'}", flush=True)
        if not anc["identical"]:
            raise SystemExit("REFUSING: A3-top does not reproduce 4d B's A1 (D-332 (6)); nothing written:\n  "
                             + "\n  ".join((anc["min_rho_mismatches"] + anc["value_mismatches"])[:12]))
    else:
        anchor = {"checked": False, "why": "non-registered draws, seeds or target (a test): the records hold the registered run"}
    anchor.update({"file": A1P.rel(ROOT / rec_rel), "sha256": rec_sha})     # rec_rel itself when ROOT is the repo
    if args.anchor_only:
        print("  --anchor-only: stopping after the self-anchor; nothing written", flush=True)
        return 0
    a.update(measure(S, X, ("A3-benign", "A3-band"), log))
    cells = summaries(a, S["target"])
    minrho = min_rho_table(cells)
    ident = threshold_identity(S, a, parr)
    bandc = band_checks(a)
    print(f"  threshold identity {'holds' if ident['holds'] else 'FAILS'} ({ident['n_checked']} thresholds); band check "
          f"{'holds' if bandc['holds'] else 'FAILS'}", flush=True)
    if not (ident["holds"] and bandc["holds"]):
        print("IMPLEMENTATION CHECK FAILED (D-181; D-346's pattern): nothing written."
              f"\n  threshold identity {ident}\n  band {bandc}", flush=True)
        return 3
    counts = P.counts_matrix(S["n_pos"], D.REGISTERED["n_boot"], D.ITEM_BOOT_SEED["eval_positives"])
    reps = add_d282(S, a, cells, counts)
    bando = band_outcomes(a)
    add_d282_losses(S, a, reps, minrho, bando, counts)
    scoring = score(cells, minrho, bando, parr, ident)
    for blk in bando.values():
        for r in blk["repairs"].values():
            for k in ("_band_loss_mean", "_top_loss_mean", "_infeasible"):
                r.pop(k)
    print(f"  scored: P-A3.1 over {scoring['P-A3.1']['accuracy']['n_cells']} cells; P-A3.2, P-A3.3, P-A3.4 per pair "
          f"(verdicts in the record)", flush=True)

    npz = dict(a)
    nb = A1P.npz_bytes(npz)
    changed = A3P.changed_inputs(S, code0, CODE_FILES, [(rec_rel, sha256_file(ROOT / rec_rel), rec_sha),
                                                        ("predictions", sha256_file(pred_path), pred_sha),
                                                        ("predictions npz", sha256_file(pnpz_path), pnpz_sha)])
    if changed:
        raise SystemExit(f"REFUSING to write: changed during the run: {changed}")
    config = {**A3P.input_block(S), "attackers": list(ATTACKERS), "repairs": {pool: list(REPAIRS_OF[pool]) for pool in POOLS},
              "band": list(BAND), "health_cut": list(CUT), "tpr_loss": PS.TPR_LOSS, "screen_tol": SCREEN_TOL,
              "anchor_only": False}
    payload = {**provenance_block("phase3_a3"), "preregistration": PREREG, "code": code0,
               "registered_parameters": registered, "blas_threads": threads, "input_verification": S["verification"],
               "config": config, "config_hash": canonical_hash(config), "sandbox_disclosure": SANDBOX,
               "predictions_record": {"file": A1P.rel(pred_path), "sha256": pred_sha,
                                      "draws_npz": {"file": A1P.rel(pnpz_path), "sha256": pdoc["draws_npz"]["sha256"]},
                                      "checks": "same pair, draws (index hash), inputs, injections (hash), parameters "
                                                "and prediction-function sha256; every array equal to this run's "
                                                "recomputation bit for bit"},
               "implementation_choices": {**A3P.CHOICES, **CHOICES},
               "anchor": anchor, "depth_record": {"file": S["depth_rel"], "sha256": S["depth_sha"]},
               "implementation_checks": {"threshold_identity": ident, "band": bandc,
                                         "d282_reproduction": "every combined-interval arm reproduced its TPR draw by draw"},
               "d282_tpr": {"where": "cells.<...>.repairs.<threshold-level repair>.tpr_d282_combined; cells.frozen",
                            "losses_where": "outcomes.min_rho_10pt_tpr_loss.<threshold-level arm>.<attacker>."
                                            "tpr_loss_at_min_rho_d282_combined; outcomes.a3_band.<pool>.repairs."
                                            "<threshold-level repair>.tpr_loss_at_largest_m_d282_combined and "
                                            "a3_top_tpr_loss_at_same_m_d282_combined (and P-A3.3's cells)",
                            "losses_paired": "within item and draw: replicate r is the clean arm's TPR minus the "
                                             "poisoned arm's on the same counts row and draw",
                            "arms": [f"{p}|{r}" for p, r in THRESHOLD_LEVEL],
                            "not": "restandardise: the positives' scores move per draw (D-156 band only), for its "
                                   "TPR and its losses",
                            "n_boot": int(counts.shape[0]), "counts_seed": D.ITEM_BOOT_SEED["eval_positives"],
                            "draw_of_replicate": "D-282: replicate r uses draw 1 + ((r - 1) mod n_draws)"},
               "benign_fit_row_share_per_seed": S["benign_fit_row_share_per_seed"],
               "probe_fit": [q.diag for q in S["probes"]],
               "input_cache_sha256": S["input_sha"],
               "input_cache_source": {kk: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"]}
                                      for kk, r in S["input_recs"].items()},
               "draws_npz": {"file": A1P.rel(npz_path), "sha256": sha256_bytes(nb), "size": len(nb), "keys": sorted(npz),
                             "rows": "draw index (0 the frozen split); per-seed arrays have the seeds as columns"},
               "outcomes": {"min_rho_10pt_tpr_loss": minrho, "a3_band": bando,
                            "health_flag_share": {pool: {"clean": cells["clean"][pool]["health"]["share_of_draws_gaussian_flag_2x"],
                                                         **{att: {f"rho={rho}": cells[att][pool][f"rho={rho}"]["health"]["share_of_draws_gaussian_flag_2x"]
                                                                  for rho in RHOS} for att in ATTACKERS}}
                                                  for pool in POOLS}},
               "scoring": scoring, "cells": cells, "runtime_seconds": round(time.time() - t0, 1)}
    for p, hh in A1P.write_pair(out_path, payload, npz_path, nb):
        print(f"Written: {p}\nsha256:  {hh}")
    print(f"runtime {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
