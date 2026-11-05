"""D-423: the REALIGNMENT AMENDMENT (the author's item 2 of 2026-10-09; CPU only), registered and committed BEFORE any
D-418 record was pulled or read. It adds two things to D-418's realignment arm and scores every (population, map) unit
SEPARATELY with D-418's four predictions, text unchanged (P-RA.1-4; no unit pooled with another, none privileged):

  (a) THE LoRA POPULATION: the nine Phase 1 LoRA arms (populations.LORA_ARMS: alpaca, hh-harmless and pku-saferlhf at
      steps 25, 100 and 400 on Mistral-7B-v0.1, A100 extraction), whose paired pool and evaluation caches exist (their
      depth_d156 records read all six through the registry), on 4d B's machinery (phase3_a3_predictions.build, 4d B's
      rule applied to the run's own H) with D-418's ridge map and D-418's 200 draws.
  (b) AN ORTHOGONAL PROCRUSTES ARM beside the ridge map, on all three populations. Duan names no map form; the sentence
      that decides it (arXiv:2606.15980v2, 'Repairing Stale Monitors', as D-418 (1) quotes it): 'we fit one alignment
      map per model, update, and layer from paired unlabeled base/update activations, then evaluate the frozen probe on
      mapped updated activations, h(A phi_l^{M'}(x))'. The author's rule ('if Duan's text uses an orthogonal map or
      names none') therefore adds it. The frozen split plus the FIRST 50 of D-418's 200 draws (the author: 'reduce
      re-partitions if needed and record it'; PRICE in the lists), the same pools, A1 sets, probes and T_d, and the
      same engine (realign_d418.run_draws) with only the map swapped (map_form below).

THE PROCRUSTES MAP (CHOICES (a)): centred (each side's fit-pool mean removed, the target's added back, as
repair_d156.procrustes and align_variants.fit_procrustes), no scaling; R orthogonal minimising
||(X_post - m_post) R - (X_pre - m_pre)||_F; mapped x = (x - m_post) R + m_pre; the frozen probe and T_d read the mapped
activations. With n = 1000 < d the minimiser is NOT unique (D-418 (7)'s reason for not running it): R is fixed on the
directions the fit items span and free elsewhere. The canonical choice here is the minimiser CLOSEST TO THE IDENTITY
(largest trace): the identity on every direction no fit item spans on either side, and inside the span the smallest
rotation taking the post side's complement to the pre side's. Closed form (verified against a brute-force
maximisation of the trace to 1e-10 in the tests): with M = Xc^T Yc = Ur S Vr^T (thin, rank r), principal vectors u_i,
v_i and cosines c_i of span(Ur) and span(Vr) (SVD of Ur^T Vr),
    R = Ur Vr^T + I - Ur Ur^T - sum_i (v_i - c_i u_i)(u_i + v_i)^T / (1 + c_i).
Scores by linearity: s(map(x)) = (x - m_post) . R v + m_pre . v + c0 (v = w / sd), checked once per job against the
materialised map (IDENTITY_TOL). It has no hyper-parameter, so the record's lambda_rel is NaN.

ANCHORS (nothing is written before they pass)
  p1  procrustes: the score arms (frozen, the recalibration arms clean and A1-poisoned), T_d and both AUROC columns of
      draws 0-50 equal D-418's own record of the pair (results/realign_d418_p1_<core>_draws.npz) bit for bit, which
      D-418 anchored on 4d B's record of record; so the draws, pools, probes and A1 sets are D-418's.
  p2  procrustes: the same against results/realign_d418_p2_<pair>_draws.npz, plus D-418's own Phase 2 anchors (the
      D-361 anchor record, the probes equal the selection's fit, the realised frozen operating point at L* over the
      1000 D-361 draws equal to the pair's D-361 record).
  lora (both maps): the frozen split's per-seed counts (flagged pre and post among the 2,000 evaluation negatives, hits
      pre and post among the 2,000 positives) equal the pair's depth_d156 record at L* exactly, and its per-seed pre
      and frozen AUROCs within AUROC_TOL: the same caches, probes and T_d as the depth record of record.
  every procrustes job: on the frozen split's R_same fit the map is orthogonal, attains the optimum (its objective
      equals the nuclear norm of M and the LAPACK solution's, repair_d156.procrustes's form), is at least as close to
      the identity as the LAPACK solution, maps the fit items as the LAPACK solution does (the solutions differ only off
      the span), maps pre -> pre to the identity, and its linear scoring equals the materialised map.

MODES (each output written once: staged, then named, never overwritten)
  predictions          results/realign_d423_predictions.json, BEFORE any number of this item; every job refuses without
                       it.
  job --unit U --pair P  results/realign_d423_<unit>_<stem>.json and its _draws.npz; U in UNITS.
  preflight --lane L   record checks and cache presence for the lane's jobs; reads no array, writes nothing.
  wait-predictions     a lane's wait for lane C's predictions job (realign_d418.wait_command's form).
  summary              results/realign_d423_summary.json from the 36 records bound by logs/realign_d423.sha256.
  price --population P --map M   per-draw arithmetic on SYNTHETIC arrays of the real shapes (writes nothing).
  queue --write        the lane lists.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_d156 as D  # noqa: E402  (imported, not changed: registry_stem, CLUSTERS, LoRA selection of record)
import input_registry as REG  # noqa: E402
import metrics as M  # noqa: E402
import paired as P  # noqa: E402
import phase3_a3_predictions as A3P  # noqa: E402  (imported, not changed: build, check_benign_disjoint, input_block)
import populations as POP  # noqa: E402
import realign_d418 as RA  # noqa: E402  (imported, not changed: the engine, summaries, scoring and Phase 2 world)
import repair_d156 as RD  # noqa: E402  (imported, not changed: _centre)
import twomoment_p2_d361 as TP2  # noqa: E402
from provenance import (  # noqa: E402
    _RESULT_READS, _repo_rel, canonical_hash, ensure_utf8_stdout, input_identity_problems, provenance_block,
    read_result, resolve_cache, sha256_bytes, sha256_file,
)

ROOT, RESULTS, LOGS, QUEUES = RA.ROOT, RA.RESULTS, RA.LOGS, RA.QUEUES
OWN = "src/realign_d423.py"
CODE_FILES = list(dict.fromkeys([OWN, *RA.CODE_FILES]))
# realign_d418.py at the bytes D-418's lanes ran (master bf213a2): a final name refuses other bytes (refsize's rule);
# RA.final_problems checks D-418's own fifteen reused files
D418_SHA256 = "5193b5c4b26fb20c1e601df016f6237832ade5de7f2b0c9d7273452dc5e59fae"
RUN_KIND = {"predictions": "realign_d423_predictions", "job": "realign_d423_job", "summary": "realign_d423_summary"}
PREREG = ("D-423 (the author's item 2 of 2026-10-09: the realignment amendment, registered and committed before any "
          "D-418 record was pulled or read); D-418's predictions, pools, draws, A1 sets, engine and scoring unchanged; "
          "details by the closest D-entries (D-181): D-418 (all of it), D-093/D-094 and D-218 (the project's "
          "orthogonal Procrustes: centred, repair_d156.procrustes), D-224 and D-171 (the LoRA arms' caches and their "
          "selection of record), D-244 (3) (draws), D-413 (vii) (cells primary)")
PRED_REL = "results/realign_d423_predictions.json"
MANIFEST_OWN = "logs/realign_d423.sha256"

N_REPS = {"ridge": RA.N_REPS, "procrustes": 50}          # CHOICES (c): Procrustes on the first 50 of the 200
MAPS = ("ridge", "procrustes")
P1_PAIRS, P2_PAIRS = RA.P1_PAIRS, RA.P2_PAIRS
LORA_RUNS = [r for runs in POP.LORA_ARMS.values() for r in runs]
# a unit is (population, map); D-418 itself scores phase1_released|ridge and phase2_headline|ridge
UNITS = {"p1_procrustes": ("phase1_released", "procrustes", P1_PAIRS),
         "p2_procrustes": ("phase2_headline", "procrustes", P2_PAIRS),
         "lora_ridge": ("lora", "ridge", LORA_RUNS),
         "lora_procrustes": ("lora", "procrustes", LORA_RUNS)}
RANK_RTOL = 1e-7          # a fit block's singular values below this fraction of its largest count as zero: the Gram
#                           route's floor (eigenvalues to 1e-14 of the largest), for the row-space bases only
CROSS_RTOL = 1e-12        # the cross-covariance's singular values below this fraction of its largest count as zero: a
#                           numerical-rank floor (numpy's matrix_rank default for a 999 x 999 block is 2.2e-13), NOT
#                           RANK_RTOL: Mistral's fit pools have real cross directions at 1e-9 to 1e-7 (F-085), and
#                           cutting them leaves R off the minimiser on the fit items' span (CHOICES (a), (i))
ORTH_TOL = 1e-9           # max |R^T R - I| of the materialised map
OPT_TOL = 1e-9            # objective vs nuclear norm and vs LAPACK's, relative to (1 + nuclear norm): D-423 (3)'s
#                           registered 1e-9 (the review's 1e-6 allowed for the directions F-085 shows were real)
BASIS_TOL = 1e-13         # max |Q^T Q - I| of a row-space basis
NS_MAX = 8                # Newton-Schulz steps at most
SELF_TOL = 1e-9           # max |R_self - I| of the pre -> pre fit
SVD_FALLBACK_TOL = 1e-10  # max |U S V^T - M| of a fallback SVD of M (CHOICES (j))
SVD_FALLBACKS: list[dict[str, Any]] = []   # CHOICES (j): one entry per fallback in this process; run_job resets it
AUROC_TOL = 1e-12         # the LoRA anchor's per-seed AUROCs against the depth record
ANCHOR_KEYS_FIXED = ("T", "auroc_base_per_seed", "auroc_frozen_per_seed")

CHOICES = [
    "(a) PROCRUSTES, the canonical minimiser (verified against a brute-force construction to 1e-10 in the tests): "
    "centred and unscaled as the project's Procrustes of record "
    "(repair_d156.procrustes, D-093/D-094, D-218); since n = 1000 < d leaves R free off the fit items' span, the "
    "minimiser closest to the identity (largest trace) is used, in closed form (module docstring), from the thin "
    "row-space bases of the two centred fit blocks (Gram eigendecomposition, one Newton-Schulz polish), the bases' "
    f"ranks cut at {RANK_RTOL:g} of the block's largest singular value and the cross-covariance's at {CROSS_RTOL:g} of "
    "its own (a numerical-rank floor: every real cross direction stays in the span, so R is the minimiser there, F-085; "
    "the dropped counts recorded per fit on the frozen split). Rejected: "
    "LAPACK's arbitrary completion (repair_d156.procrustes as is): it depends on the library and the item order, not "
    "on the data; the minimum-norm partial isometry (not orthogonal, so not a Procrustes map). The difference between "
    "the canonical and the LAPACK map on E's scores is recorded on the frozen split (how much the choice matters).",
    "(b) ENGINE: realign_d418.run_draws, arm_summaries and duan_block unchanged; for a Procrustes unit only the module "
    "names fit_map and map_scores are swapped for the duration of the call (map_form, restored on exit), so every arm, "
    "pool, A1 set, control and two-moment prediction is computed by D-418's reviewed lines. The arms keep D-418's "
    "names ('realign|<pool>' is the map arm of whichever map the record names in config.map), so "
    "realign_d418.score_population scores every unit unchanged.",
    "(c) DRAWS: ridge units take D-418's registered 200 draws; Procrustes units the frozen split plus the FIRST 50 (the "
    "draws are prefix-stable: draw d is seeded by d, so draw d here is D-418's draw d), priced in the lists; the D-156 "
    "means of a Procrustes unit are over 50 draws, recorded in every record and in the summary.",
    "(d) LoRA LOADING: phase3_a3_predictions.load_inputs's checks line for line, with the selection read from the LoRA "
    "selection of record (depth_d156.SELECTION_OF_RECORD_LORA, its one model depth_d156.LORA_SELECTION_PAIR: the A100 "
    "base every arm shares, D-171, D-224), which must be the file, sha256, C and layer the run's depth_d156 record "
    "names; then phase3_a3_predictions.build unchanged (4d B's partitions, pools, probes, T_d and A1 rule on the run's "
    "own H, as D-418 (4) applies 4d B's rule to Phase 2's H).",
    "(e) ANCHORS: as the module docstring. A LoRA run has no 4d B record, so it anchors on its depth_d156 record of "
    "record at L* (the frozen split's per-seed counts exactly, the AUROCs within AUROC_TOL).",
    "(f) UNITS and SCORING: four new units (population x map), each scored separately with "
    "realign_d418.score_population (P-RA.1-4, D-418's text and thresholds; 9 pairs and 27 cells per unit); the "
    "author's 'none pooled, none privileged' - no unit is primary over another and no verdict pools units.",
    "(g) LANES: lane C first (the author, 2026-10-09: 'Put the new CPU jobs on lane C first. Chain behind D-418 on A or "
    "D only what C cannot hold'): lane C carries the predictions job and the eighteen LoRA jobs (ridge, then "
    "Procrustes), which need nothing of D-418; the Phase 1 and Phase 2 Procrustes jobs anchor on D-418's record of "
    "the same pair, which exists only once D-418's lane has run that pair, so lane C cannot hold them: the nine Phase "
    "2 Procrustes jobs go behind D-418 on lane A and the nine Phase 1 Procrustes jobs behind D-418 on lane D (the lane "
    "that wrote their anchors), each list opening with a wait for lane C's predictions job (realign_d418.wait_command's "
    "form). (Amended before any load from D-423 (6)'s first split, which put Phase 1 Procrustes on C and LoRA "
    "Procrustes on D: D-418 had run only three of nine pairs per lane at 05:15Z on 2026-10-09.) The manifest and the "
    "summary run after the three lanes. ONE sonnet reviewer (the author's rule for this item, as D-418 (8)).",
    "(i) NUMERICS (the review of 2026-10-09): the row-space bases are re-orthonormalised by Newton-Schulz steps to "
    "BASIS_TOL 1e-13 (the Gram route squares the condition number); both gaps are recorded per job on the frozen "
    "split. AMENDED 2026-10-10 (F-085, after the node sandbox): the review's relative OPT_TOL 1e-6 for directions "
    "dropped below RANK_RTOL is withdrawn: those directions were real (Mistral's fit pools: cross singular values at "
    "1e-9 to 1e-7 of the largest, 11-15 of 999), cutting them left R off the minimiser on the fit items' span (fit "
    "scores 0.07-0.21 from LAPACK's), so the cross-covariance is now cut at CROSS_RTOL and OPT_TOL is D-423 (3)'s 1e-9.",
    "(h) POPULATION GROUP: none until the pull (D-418 CHOICES (r)).",
    "(j) SVD FALLBACK (D-437, the author's case (b) of 2026-10-11; F-089): LAPACK's np.linalg.svd of the principal-"
    "angle matrix M = Ur^T Vr raised 'SVD did not converge' on a FINITE M in the control_selfmap arm (src == tgt, so M "
    "is the identity to 1e-12: every singular value 1 +- 5e-13; lanes D and A, draws 14 and 38, "
    "docs/reports_d437/svd_diag_*.json); the same call on M^T converged. Only when np.linalg.svd(M) raises, M's SVD "
    "is taken from np.linalg.svd(M^T) (M = (W^T) S (Z^T) for M^T = Z S W), else from the eigendecomposition of M^T M "
    "(V; U = M V / s, re-orthonormalised by QR, its columns sign-aligned with M V / s); the result must reconstruct M to SVD_FALLBACK_TOL or the job "
    "refuses. Converged fits are untouched (bit-identical); no pair or draw is dropped; every fallback is recorded "
    "with its draw, arm and method in the record's svd_fallbacks (the count in n_svd_fallbacks).",
]


def choice(letter: str) -> str:
    return next(c for c in CHOICES if c.startswith(f"({letter}) "))


refuse, rel, canon, f64 = RA.refuse, RA.rel, RA.canon, RA.f64


# ----------------------------------------------------------------------------- the Procrustes map (CHOICES (a))
def row_basis(xc: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """An orthonormal basis (d, r) of the row space of a centred (n, d) block: the Gram matrix's eigenvectors mapped
    back, ranks cut at RANK_RTOL of the largest singular value, then Newton-Schulz steps Q (3I - Q^T Q) / 2 until
    max |Q^T Q - I| <= BASIS_TOL (the Gram route squares the condition number, so the small kept directions can start
    1e-4 to 1e-2 off orthonormal; each step squares the error; review of 2026-10-09)."""
    e, E = np.linalg.eigh(xc @ xc.T)
    sv = np.sqrt(np.clip(e, 0.0, None))
    keep = sv > RANK_RTOL * float(sv.max()) if sv.size and sv.max() > 0 else np.zeros(len(sv), bool)
    if not keep.any():
        refuse("a fit block has no variance (every row equal)")
    Q = xc.T @ (E[:, keep] / sv[keep])
    eye = np.eye(Q.shape[1])
    err = float("inf")
    for _ in range(NS_MAX):
        G = Q.T @ Q
        err = float(np.abs(G - eye).max())
        if err <= BASIS_TOL:
            break
        Q = Q @ (1.5 * eye - 0.5 * G)
    else:
        err = float(np.abs(Q.T @ Q - eye).max())
        if err > BASIS_TOL:
            refuse(f"a fit block's basis is not orthonormal after {NS_MAX} Newton-Schulz steps ({err:.3g})")
    return Q, {"rank": int(keep.sum()), "n_dropped": int((~keep).sum()), "orthonormality_error": err}


def _fit_context() -> dict[str, Any]:
    """The draw and arm of the fit being made, from realign_d418.run_draws's frames (that module is pinned, so it is
    read, not changed): map_arm's 'arm', run_draws's 'd'; None outside them (the map check)."""
    ctx: dict[str, Any] = {"draw": None, "arm": None}
    f = inspect.currentframe()
    try:
        while f is not None:
            if f.f_code.co_name == "map_arm" and ctx["arm"] is None:
                ctx["arm"] = f.f_locals.get("arm")
            if f.f_code.co_name == "run_draws" and ctx["draw"] is None:
                try:
                    ctx["draw"] = int(f.f_locals.get("d"))
                except (TypeError, ValueError):
                    ctx["draw"] = None
            f = f.f_back
    finally:
        del f
    return ctx


def svd_with_fallback(Mx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CHOICES (j): np.linalg.svd(Mx) as before; only if it raises, the SVD from Mx^T, else from eigh of Mx^T Mx;
    the fallback must reconstruct Mx to SVD_FALLBACK_TOL; each one is appended to SVD_FALLBACKS."""
    try:
        return np.linalg.svd(Mx)
    except np.linalg.LinAlgError as exc:
        first = str(exc)
    method = "svd_transpose"
    try:
        z, s, w = np.linalg.svd(Mx.T)                        # Mx^T = z diag(s) w  =>  Mx = w^T diag(s) z^T
        u, vt = w.T, z.T
    except np.linalg.LinAlgError:
        method = "eigh"
        e, V = np.linalg.eigh(Mx.T @ Mx)
        order = np.argsort(e)[::-1]
        e, V = e[order], V[:, order]
        s = np.sqrt(np.clip(e, 0.0, None))
        if not (s.size and s[-1] > 0):
            refuse("the SVD fallback met a singular principal-angle matrix (a zero singular value)")
        u, rr = np.linalg.qr((Mx @ V) / s)
        u = u * np.sign(np.diag(rr))                          # keep U's columns aligned with M V / s
        vt = V.T
    err = float(np.abs((u * s) @ vt - Mx).max())
    if not (np.isfinite(err) and err <= SVD_FALLBACK_TOL):
        refuse(f"the SVD fallback ({method}) reconstructs M only to {err:.3g} (> {SVD_FALLBACK_TOL:g})")
    SVD_FALLBACKS.append({**_fit_context(), "method": method, "error": first, "reconstruction_max_abs": err,
                          "shape": list(Mx.shape), "s_min": float(s.min()), "s_max": float(s.max())})
    return u, s, vt


def fit_procrustes(src: np.ndarray, tgt: np.ndarray) -> dict[str, Any]:
    """The canonical (closest to the identity) centred orthogonal map src -> tgt (rows paired); what proc_RV,
    map_scores_procrustes and materialise need. Same contract as realign_d418.fit_map (refusals, 'n', 'lambda_rel')."""
    src, tgt = f64(src), f64(tgt)
    if src.shape != tgt.shape or src.ndim != 2 or len(src) < 2:
        refuse(f"a map needs paired (n, d) rows on both sides: {src.shape} vs {tgt.shape}")
    if not (np.isfinite(src).all() and np.isfinite(tgt).all()):
        refuse("a non-finite activation in a fit pool")
    xs, ms = RD._centre(src)
    ys, mt = RD._centre(tgt)
    Qx, bx = row_basis(xs)
    Qy, by = row_basis(ys)
    C = (xs @ Qx).T @ (ys @ Qy)
    a, s, bt = np.linalg.svd(C)
    r = int((s > CROSS_RTOL * float(s[0])).sum()) if s.size and s[0] > 0 else 0
    if r == 0:
        refuse("the cross-covariance of a fit pool is zero")
    Ur, Vr = Qx @ a[:, :r], Qy @ bt[:r].T
    c_, cos, et = svd_with_fallback(Ur.T @ Vr)
    cos = np.clip(cos, 0.0, 1.0)
    return {"Ur": Ur, "Vr": Vr, "u": Ur @ c_, "v": Vr @ et.T, "cos": cos, "ms": ms, "mt": mt,
            "nuclear": float(s[:r].sum()), "n": int(len(src)), "lambda_rel": float("nan"),
            "rank": {"post": bx, "pre": by, "cross": r, "cross_dropped": int(len(s) - r)},
            "cos_min": float(cos.min()), "cos_max": float(cos.max())}


def proc_RV(mp: dict[str, Any], V: np.ndarray) -> np.ndarray:
    """R V for the canonical R (module docstring), without forming R: (d, S)."""
    Ur, Vr, u, v, cos = mp["Ur"], mp["Vr"], mp["u"], mp["v"], mp["cos"]
    return (Ur @ (Vr.T @ V) + V - Ur @ (Ur.T @ V)
            - (v - u * cos) @ (((u + v).T @ V) / (1.0 + cos)[:, None]))


def materialise(mp: dict[str, Any]) -> np.ndarray:
    """The (d, d) map R itself (checks only)."""
    d = mp["Ur"].shape[0]
    return proc_RV(mp, np.eye(d))


def map_scores_procrustes(mp: dict[str, Any], probes: list[P.Probe], X: np.ndarray,
                          lin: tuple | None = None) -> np.ndarray:
    """(n, S) frozen-probe scores of the mapped X, by linearity; realign_d418.map_scores's contract."""
    V, c0 = lin if lin is not None else RA.probe_lin(probes)
    U = proc_RV(mp, V)
    return f64(X) @ U - mp["ms"] @ U + mp["mt"] @ V + c0


def lapack_R(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """repair_d156.procrustes's own R (u @ vt of the full d x d SVD), the library's arbitrary completion."""
    xs, _ = RD._centre(f64(src))
    ys, _ = RD._centre(f64(tgt))
    u, _, vt = np.linalg.svd(xs.T @ ys, full_matrices=False)
    return u @ vt


def proc_check(src: np.ndarray, tgt: np.ndarray, probes: list[P.Probe], X: np.ndarray) -> dict[str, Any]:
    """CHOICES (a), (e): on one fit, orthogonality, optimality, closeness to the identity, agreement with the LAPACK
    solution on the fit items, the pre -> pre identity and the linear scoring identity."""
    src, tgt, X = f64(src), f64(tgt), f64(X)
    mp = fit_procrustes(src, tgt)
    R = materialise(mp)
    d = R.shape[0]
    xs, ms = RD._centre(src)
    ys, mt = RD._centre(tgt)
    RL = lapack_R(src, tgt)
    obj, obj_l = float(((xs @ R) * ys).sum()), float(((xs @ RL) * ys).sum())
    lin = RA.probe_lin(probes)
    mine = map_scores_procrustes(mp, probes, X, lin)
    mat = np.column_stack([q.score((X - ms) @ R + mt) for q in probes])
    fit_mine = map_scores_procrustes(mp, probes, src, lin)
    fit_lapack = np.column_stack([q.score((src - ms) @ RL + mt) for q in probes])
    e_lapack = np.column_stack([q.score((X - ms) @ RL + mt) for q in probes])
    selfmap = materialise(fit_procrustes(tgt, tgt))
    scale = 1.0 + float(np.abs(mat).max())
    out = {"max_orthogonality_error": float(np.abs(R.T @ R - np.eye(d)).max()),
           "objective": obj, "nuclear_norm": mp["nuclear"], "objective_lapack": obj_l,
           "trace": float(np.trace(R)), "trace_lapack": float(np.trace(RL)),
           "max_abs_fit_scores_minus_lapack": float(np.abs(fit_mine - fit_lapack).max()),
           "max_abs_E_scores_minus_lapack_beside": float(np.abs(mine - e_lapack).max()),
           "max_abs_linear_minus_materialised": float(np.abs(mine - mat).max()),
           "max_abs_selfmap_minus_identity": float(np.abs(selfmap - np.eye(d)).max()),
           "rank": mp["rank"], "cos_min": mp["cos_min"], "cos_max": mp["cos_max"], "score_scale": scale,
           "tolerances": {"orthogonality": ORTH_TOL, "optimality": OPT_TOL, "map": RA.MAP_TOL,
                          "identity": RA.IDENTITY_TOL, "selfmap": SELF_TOL},
           "n_fit": mp["n"], "n_scored": int(len(X)),
           "note": ("max_abs_E_scores_minus_lapack_beside is how far the arbitrary LAPACK completion would move E's "
                    "scores (descriptive, no tolerance): the size of the choice CHOICES (a) makes")}
    opt_scale = 1.0 + abs(mp["nuclear"])
    out["passed"] = bool(out["max_orthogonality_error"] <= ORTH_TOL
                         and abs(obj - mp["nuclear"]) <= OPT_TOL * opt_scale
                         and abs(obj - obj_l) <= OPT_TOL * opt_scale
                         and out["trace"] >= out["trace_lapack"] - OPT_TOL * (1.0 + abs(out["trace_lapack"]))
                         and out["max_abs_fit_scores_minus_lapack"] <= RA.MAP_TOL * scale
                         and out["max_abs_linear_minus_materialised"] <= RA.IDENTITY_TOL * scale
                         and out["max_abs_selfmap_minus_identity"] <= SELF_TOL)
    return out


@contextlib.contextmanager
def map_form(form: str) -> Iterator[None]:
    """CHOICES (b): realign_d418.run_draws with the named map; the module names are restored on exit."""
    if form not in MAPS:
        refuse(f"no map form {form!r}: {MAPS}")
    saved = (RA.fit_map, RA.map_scores)
    if form == "procrustes":
        RA.fit_map, RA.map_scores = fit_procrustes, map_scores_procrustes
    try:
        yield
    finally:
        RA.fit_map, RA.map_scores = saved


def check_of(form: str, W: dict[str, Any]) -> dict[str, Any]:
    d0 = W["draws"][0]
    args = (f64(W["X_post"][d0["R_same"]]), f64(W["X_pre"][d0["R_same"]]), W["probes"], W["X_post"][d0["E"]])
    chk = RA.map_check(*args) if form == "ridge" else proc_check(*args)
    if not chk["passed"]:
        refuse(f"the {form} map check failed on the frozen split's R_same fit: {chk}; nothing written")
    return chk


# ----------------------------------------------------------------------------- the registration
def registered_text() -> dict[str, Any]:
    return {"predictions": RA.REGISTERED_PREDICTIONS,
            "amendment": {
                "units": {u: {"population": pop, "map": mp, "pairs": list(pairs), "n_reps": N_REPS[mp]}
                          for u, (pop, mp, pairs) in UNITS.items()},
                "scored_by": "realign_d418.score_population, per unit separately (CHOICES (f))",
                "procrustes_draws": ("a Procrustes unit's 'D-156 means over the 200 draws' read 'over the first 50' "
                                     "(CHOICES (c)); everything else of D-418's text unchanged"),
                "duan_sentence": ("'we fit one alignment map per model, update, and layer from paired unlabeled "
                                  "base/update activations, then evaluate the frozen probe on mapped updated "
                                  "activations, h(A phi_l^{M'}(x))' (arXiv:2606.15980v2): no map form named")}}


def predictions_payload() -> dict[str, Any]:
    return {**provenance_block(RUN_KIND["predictions"]), "preregistration": PREREG, "code": P.code_block(CODE_FILES),
            "reused_files_sha256": RA.reused_now() | {"src/realign_d418.py": sha256_file(ROOT / "src/realign_d418.py")},
            **registered_text(), "beside": RA.BESIDE_NOTE, "deviations_from_duan": RA.DEVIATIONS,
            "implementation_choices": CHOICES,
            "verdicts": {u: {k: "NOT YET RUN" for k in RA.REGISTERED_PREDICTIONS} for u in UNITS},
            "status": {"status": "FINAL", "note": "a registration: it carries no quantity of this item"}}


def run_predictions(out: str | Path | None = None) -> Path:
    final_path = (RESULTS / Path(PRED_REL).name).resolve()
    out_path = Path(out).resolve() if out else final_path
    if out_path != final_path and out_path.is_relative_to(RESULTS.resolve()):
        refuse("a test run (--out) must not write under results/")
    if out_path.exists() or out_path.is_symlink():
        refuse(f"{out_path} exists; results are never overwritten")
    TP2.write_outputs([(out_path, RA.json_bytes_of(predictions_payload()))])
    print(f"Written: {out_path}")
    return out_path


def passing_predictions(code0: dict[str, Any], records_dir: Path | None = None) -> dict[str, Any]:
    p = (records_dir or RESULTS) / Path(PRED_REL).name
    if p.is_symlink() or not p.is_file():
        refuse(f"no predictions record {p.name}: run `predictions` first (D-423 registers before any number)")
    doc = read_result(p)
    problems = []
    if doc.get("run_kind") != RUN_KIND["predictions"]:
        problems.append("not a D-423 predictions record")
    reg = registered_text()
    if canon(doc.get("predictions")) != canon(reg["predictions"]) or \
            canon(doc.get("amendment")) != canon(reg["amendment"]):
        problems.append("its registered text is not this module's")
    if (doc.get("code") or {}).get("sha256") != code0["sha256"]:
        problems.append("it was written by other code")
    if problems:
        refuse(f"the predictions record {p.name} cannot license this run: {problems}")
    return {"file": rel(p), "sha256": _RESULT_READS[_repo_rel(p)]}


# ----------------------------------------------------------------------------- the LoRA population (CHOICES (d))
def load_lora_inputs(run: str) -> dict[str, Any]:
    """phase3_a3_predictions.load_inputs's checks for one LoRA arm, the selection from the LoRA selection of record."""
    if run not in LORA_RUNS:
        refuse(f"{run!r} is not one of the nine LoRA arms: {LORA_RUNS}")
    reg_stem = D.registry_stem(f"depth_{run}")
    depth_rel = f"results/depth_d156_{run}.json"
    depth_sha = sha256_file(ROOT / depth_rel)
    ddoc = read_result(ROOT / depth_rel)
    dc = ddoc["config"]
    sel_path = D.SELECTION_OF_RECORD_LORA
    sel_sha, cl_sha = sha256_file(sel_path), sha256_file(D.CLUSTERS)
    if (dc.get("selection_file"), dc.get("selection_sha256"), dc.get("selection_of_pair")) != \
            (rel(sel_path), sel_sha, D.LORA_SELECTION_PAIR):
        refuse(f"{depth_rel} names another selection ({dc.get('selection_file')}, {str(dc.get('selection_sha256'))[:12]},"
               f" {dc.get('selection_of_pair')})")
    sel_doc = read_result(sel_path)
    sel = next((m for m in sel_doc["models"] if m["pair"] == D.LORA_SELECTION_PAIR), None)
    if sel is None or not sel["selected"]["arm"].startswith("C="):
        refuse(f"no logistic-C selection for {D.LORA_SELECTION_PAIR}")
    C_sel, L_sel = float(sel["selected"]["arm"].split("=")[1]), int(sel["selected"]["layer"])
    if (dc.get("selected_C"), dc.get("selected_layer")) != (C_sel, L_sel):
        refuse(f"{depth_rel} was run at C {dc.get('selected_C')} L {dc.get('selected_layer')}, not the selection's "
               f"C {C_sel} L {L_sel}")
    cfg = REG.run(reg_stem)
    axis = cfg["axis"]
    k = f"layer_{L_sel}"
    data: dict[str, dict[str, Any]] = {"pre": {}, "post": {}}
    input_sha: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    for side in data:
        for part in ("probe_train", "threshold_calib", "eval"):
            z, r = REG.load(cfg[f"{side}_repo"], axis, part)
            if "nonfinite" in z.files and z["nonfinite"].any():
                refuse(f"{side} {part} has non-finite items")
            data[side][part] = {"X": z[k], "labels": z["labels"], "item_ids": z["item_ids"]}
            input_sha[f"{side}/{part}"], input_recs[f"{side}/{part}"] = r["sha256"], r
    if any(ddoc["input_cache_sha256"].get(kk) != h for kk, h in input_sha.items()):
        refuse(f"{depth_rel} read other caches")
    for part in ("probe_train", "threshold_calib", "eval"):
        a, b = data["pre"][part], data["post"][part]
        if not np.array_equal(a["item_ids"], b["item_ids"]) or not np.array_equal(a["labels"], b["labels"]):
            refuse(f"{part} items or labels differ between pre and post")
    y_cal = data["pre"]["threshold_calib"]["labels"].astype(bool)
    y_ev = data["pre"]["eval"]["labels"].astype(bool)
    pooled_ids = [str(i) for i in data["pre"]["threshold_calib"]["item_ids"][~y_cal]] + \
                 [str(i) for i in data["pre"]["eval"]["item_ids"][~y_ev]]
    cl = json.load(open(D.CLUSTERS, encoding="utf-8"))["axes"][axis]   # a manifest, hashed (not a result, D-297)
    if cl["n_pooled_negatives"] != len(pooled_ids) or cl["n_calibration"] != int((~y_cal).sum()) or \
            cl["pool_item_ids_sha256"] != sha256_bytes("\n".join(pooled_ids).encode("utf-8")):
        refuse(f"manifests/negative_clusters.json does not describe the {axis} pool")
    A3P.check_benign_disjoint(data)
    reg_sources = REG.sources_sha256()
    if D.sources_now() != reg_sources:
        refuse("the extraction records or pins on disk are not the ones the registry read")
    return {"pair": run, "core": run, "reg_stem": reg_stem, "axis": axis, "origin_strat": axis == A3P.PS.STRAT_AXIS,
            "cfg": cfg, "C_sel": C_sel, "L_sel": L_sel, "sel_path": sel_path, "sel_sha": sel_sha, "cl_sha": cl_sha,
            "depth_rel": depth_rel, "depth_sha": depth_sha, "data": data, "depth_doc": ddoc,
            "input_sha": input_sha, "input_recs": input_recs, "reg_sources": reg_sources,
            "verification": REG.verification(input_recs)}


def lora_anchor(S: dict[str, Any], ddoc: dict[str, Any]) -> dict[str, Any]:
    """CHOICES (e): the frozen split at L* against the run's depth_d156 record (selected probe set)."""
    cell = next((c for c in ddoc.get("cells", []) if c.get("probe_set") == "selected"
                 and c.get("layer") == S["L_sel"]), None)
    if cell is None:
        refuse(f"{S['depth_rel']} holds no selected cell at L{S['L_sel']}")
    SC, T0, n_cal = S["SC"], S["T"][0], S["n_cal"]
    neg_pre, neg_post = SC["pre"]["pool"][n_cal:], SC["post"]["pool"][n_cal:]
    pos_pre, pos_post = SC["pre"]["pos"], SC["post"]["pos"]
    mine = {"n_flagged_pre": [int((neg_pre[:, s] >= T0[s]).sum()) for s in range(len(T0))],
            "n_flagged_post": [int((neg_post[:, s] >= T0[s]).sum()) for s in range(len(T0))],
            "hits_pre": [int((pos_pre[:, s] >= T0[s]).sum()) for s in range(len(T0))],
            "hits_post": [int((pos_post[:, s] >= T0[s]).sum()) for s in range(len(T0))]}
    want = (cell.get("operating_point") or {}).get("frozen_split_counts") or {}
    bad = [k for k in mine if list(want.get(k) or []) != mine[k]]
    au = {"pre": [float(M.auroc(pos_pre[:, s], neg_pre[:, s]).value) for s in range(len(T0))],
          "frozen": [float(M.auroc(pos_post[:, s], neg_post[:, s]).value) for s in range(len(T0))]}
    diffs = {k: max(abs(a - b) for a, b in zip(au[k], (cell.get("auroc") or {}).get(k, {}).get("per_seed") or []))
             if len((cell.get("auroc") or {}).get(k, {}).get("per_seed") or []) == len(T0) else float("inf")
             for k in au}
    bad += [f"auroc {k} max diff {v:.3g}" for k, v in diffs.items() if not v <= AUROC_TOL]
    if bad:
        refuse(f"the LoRA anchor against {S['depth_rel']} at L{S['L_sel']} failed: {bad}; nothing written")
    return {"checked": True, "file": S["depth_rel"], "sha256": S["depth_sha"], "layer": S["L_sel"],
            "frozen_split_counts_equal": True, "auroc_max_abs_diff": diffs, "auroc_tol": AUROC_TOL, **mine}


# ----------------------------------------------------------------------------- the D-418 anchor (CHOICES (e))
def anchor_compare(arr: dict[str, np.ndarray], ref: dict[str, np.ndarray],
                   n_reps: int) -> tuple[list[str], list[str], int]:
    """(keys compared, keys differing, n present): every score arm's per-draw array (no map or control arm, no frozen
    split flags), T_d and both AUROC columns, draws 0..n_reps of the reference, bit for bit."""
    keys = list(ANCHOR_KEYS_FIXED) + [k for k in arr if "|" in k and not RA.is_map_arm(k)
                                      and not k.startswith("frozen_split_flags|")]
    bad, n_cmp = [], 0
    for k in keys:
        if k not in ref:
            bad.append(f"{k}: not in D-418's npz")
            continue
        n_cmp += 1
        want = np.asarray(ref[k])[:n_reps + 1]
        if want.shape != arr[k].shape or not np.array_equal(want, arr[k]):
            bad.append(k)
    return keys, bad, n_cmp


def d418_record_check(kind: str, stem: str) -> tuple[str, Path, dict[str, Any], Path, dict[str, Any]]:
    """D-418's record of the pair, final, and the npz it names (sha256); called before any cache is read."""
    rec_rel = f"results/realign_d418_{kind}_{stem}.json"
    p = ROOT / rec_rel
    if p.is_symlink() or not p.is_file():
        refuse(f"{rec_rel} is missing: a Procrustes job anchors on D-418's record of the same pair")
    doc = read_result(p)
    if doc.get("registered_parameters") is not True:
        refuse(f"{rec_rel} was not run with D-418's registered parameters")
    nz = doc.get("draws_npz") or {}
    npz_p = ROOT / str(nz.get("file"))
    if not npz_p.is_file() or sha256_file(npz_p) != nz.get("sha256"):
        refuse(f"{nz.get('file')} is not the npz {rec_rel} names")
    return rec_rel, p, doc, npz_p, nz


def d418_anchor(kind: str, stem: str, arr: dict[str, np.ndarray], n_reps: int) -> dict[str, Any]:
    """Draws 0..n_reps of every score arm (not a map arm), T_d and both AUROC columns equal D-418's record's arrays."""
    rec_rel, p, doc, npz_p, nz = d418_record_check(kind, stem)
    with np.load(npz_p) as z:
        keys, bad, n_cmp = anchor_compare(arr, {k: z[k] for k in z.files}, n_reps)
    if bad:
        refuse(f"the D-418 anchor failed ({len(bad)} of {len(keys)} arrays differ: {bad[:6]}); nothing written")
    return {"checked": True, "file": rec_rel, "sha256": _RESULT_READS[_repo_rel(p)],
            "npz": {"file": nz.get("file"), "sha256": nz.get("sha256")}, "draws_compared": n_reps + 1,
            "n_arrays_equal": n_cmp, "rule": "every score arm, T_d and both AUROC columns, bit for bit"}


# ----------------------------------------------------------------------------- the jobs
def out_paths(unit: str, stem: str, out: str | Path | None) -> tuple[Path, Path, bool]:
    final_path = (RESULTS / f"realign_d423_{unit}_{stem}.json").resolve()
    out_path = Path(out).resolve() if out else final_path
    final = out_path == final_path
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        refuse("a test run (--out) must not write under results/; only this job's own final name may")
    npz_path = out_path.with_name(out_path.stem + "_draws.npz")
    for p in (out_path, npz_path):
        if p.exists() or p.is_symlink():
            refuse(f"{p} exists; results are never overwritten (D-152 P5)")
    return out_path, npz_path, final


def stem_of(unit: str, pair: str) -> str:
    return A3P.A1P.core_of(pair) if unit == "p1_procrustes" else pair


def world_p1(pair: str, n_reps: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    S = A3P.setup(pair, n_reps, RA.SEEDS, RA.TARGET)
    W = RA.p1_world(S)
    W["draws_sha256"] = S["draws_sha256"]
    return W, S, {**A3P.input_block(S), "population": "phase1_released"}


def world_lora(run: str, n_reps: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    inp = load_lora_inputs(run)
    ddoc = inp.pop("depth_doc")
    S = A3P.build(inp, n_reps, RA.SEEDS, RA.TARGET)
    S["lora_anchor"] = lora_anchor(S, ddoc)
    W = RA.p1_world(S)
    W["draws_sha256"] = S["draws_sha256"]
    return W, S, {**A3P.input_block(S), "population": "lora",
                  "selection_of_record": {"file": rel(S["sel_path"]), "pair": D.LORA_SELECTION_PAIR}}


def run_job(unit: str, pair: str, n_reps: int | None = None, out: str | Path | None = None,
            records_only: bool = False) -> Path:
    """One (unit, pair) job: its world, the map check, D-418's engine under the unit's map, the anchors, the record."""
    t0 = time.time()
    if unit not in UNITS:
        refuse(f"no unit {unit!r}: {sorted(UNITS)}")
    pop, form, pairs = UNITS[unit]
    if pair not in pairs:
        refuse(f"{pair!r} is not in unit {unit}: {pairs}")
    n_reps = N_REPS[form] if n_reps is None else int(n_reps)
    registered = n_reps == N_REPS[form]
    stem = stem_of(unit, pair)
    out_path, npz_path, final = out_paths(unit, stem, out)
    RA.check_final_gates(final, True)
    if final and not registered:
        refuse(f"the final name needs the registered draws ({N_REPS[form]})")
    if final and sha256_file(ROOT / "src/realign_d418.py") != D418_SHA256:
        refuse("src/realign_d418.py is not the bytes D-418's lanes ran (D418_SHA256)")
    code0 = P.code_block(CODE_FILES)
    if not records_only:
        passing_predictions(code0)
    if records_only:
        if pop == "phase1_released":
            RA.p1_records(pair)
        elif pop == "phase2_headline":
            b = RA.RS.p2_bind(pair, P.N_DRAWS, False, True, code0)
            RA.read_v2_pool4d_positives(b["v2"])
        else:
            for r_ in (f"results/depth_d156_{pair}.json", rel(D.SELECTION_OF_RECORD_LORA)):
                read_result(ROOT / r_)
        print(f"RECORDS OK: {unit} {pair} (records only; no cache read, nothing written)")
        return out_path
    extra: dict[str, Any] = {}
    if form == "procrustes" and pop != "lora":            # the anchor's record and npz before any cache is read
        d418_record_check("p1" if pop == "phase1_released" else "p2", stem)
    if pop == "phase1_released":
        W, S, config = world_p1(pair, n_reps)
        inputs = {"input_cache_sha256": S["input_sha"], "input_verification": S["verification"],
                  "selection": {"file": rel(S["sel_path"]), "sha256": S["sel_sha"]},
                  "depth_record": {"file": S["depth_rel"], "sha256": S["depth_sha"]}}
    elif pop == "lora":
        W, S, config = world_lora(pair, n_reps)
        extra["anchor_depth_record"] = S["lora_anchor"]
        inputs = {"input_cache_sha256": S["input_sha"], "input_verification": S["verification"],
                  "selection": {"file": rel(S["sel_path"]), "sha256": S["sel_sha"]},
                  "depth_record": {"file": S["depth_rel"], "sha256": S["depth_sha"]}}
    else:
        W, config, inputs = world_p2(pair, n_reps, code0, final)
        extra["anchor_phase2"] = inputs.pop("anchor")
    end_checks = inputs.pop("_end_checks", None)
    print(f"D-423 {unit} {pair}: map {form}; {len(W['draws']) - 1} draws + frozen; pools {list(RA.POOLS)} at n "
          f"{RA.POOL_N}; rho {list(RA.RHOS)}; registered {registered}", flush=True)
    SVD_FALLBACKS.clear()                          # CHOICES (j): the map check's fallbacks recorded apart
    chk = check_of(form, W)
    chk["svd_fallbacks"] = [dict(x) for x in SVD_FALLBACKS]
    SVD_FALLBACKS.clear()
    with map_form(form):
        arr = RA.run_draws(W, lambda m: print(m, flush=True))
    svd_fallbacks = [dict(x) for x in SVD_FALLBACKS]
    for x in svd_fallbacks:
        print(f"SVD FALLBACK (CHOICES (j)): draw {x['draw']} arm {x['arm']} by {x['method']} "
              f"(reconstruction {x['reconstruction_max_abs']:.2g})", flush=True)
    if form == "procrustes" and pop != "lora":
        extra["anchor_d418"] = d418_anchor("p1" if pop == "phase1_released" else "p2", stem, arr, n_reps)
    summ = RA.arm_summaries(arr, RA.TARGET)
    if form == "procrustes":                      # no hyper-parameter: no NaN in the JSON
        for blk in summ.values():
            if isinstance(blk, dict) and "lambda_rel_frozen_split" in blk:
                blk["lambda_rel_frozen_split"] = None
                blk["lambda_rel_counts_over_draws"] = "not applicable: the Procrustes map has no hyper-parameter"
    ver = inputs.get("input_verification") or {}
    flags = [] if ver.get("flag") in (None, "None", "") else [f"D-170 ({ver.get('flag')!r})"]
    arr["draws_index_sha256_bytes"] = np.frombuffer(bytes.fromhex(W["draws_sha256"]), dtype=np.uint8)
    if P.code_block(CODE_FILES)["sha256"] != code0["sha256"]:
        refuse("source changed during the run")
    changed = [r_ for r_, h in _RESULT_READS.items() if (ROOT / r_).is_file() and sha256_file(ROOT / r_) != h]
    if pop != "phase2_headline":
        changed += [f"cache {r['file']}" for r in S["input_recs"].values()
                    if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    else:
        changed += end_checks()
    if changed:
        refuse(f"to write: changed during the run: {changed}")
    config = {**config, "unit": unit, "map": form, "n_reps": n_reps, "pools": list(RA.POOLS), "pool_n": RA.POOL_N,
              "rhos": list(RA.RHOS), "controls": list(RA.CONTROLS), "control_pool": RA.CONTROL_POOL,
              "shuffle_seed": RA.SHUFFLE_SEED, "arms": RA.arm_names(), "blas_threads": RA.blas_threads(),
              "map_detail": (choice("a") if form == "procrustes" else RA.choice("a"))}
    for k in ("c_grid", "variants"):
        config.pop(k, None)
    payload = {**provenance_block(RUN_KIND["job"]), "preregistration": PREREG, "code": code0,
               "reused_files_sha256": RA.reused_now() | {"src/realign_d418.py": sha256_file(ROOT / "src/realign_d418.py")},
               "registered_parameters": registered,
               "status": RA.status_block(registered, flags), "elapsed_s": None,
               "deviations_from_duan": RA.DEVIATIONS, "implementation_choices": CHOICES,
               "implementation_choices_d418": RA.CHOICES, "config": config, "config_hash": canonical_hash(config),
               "map_check": chk, **extra, **inputs,
               "svd_fallbacks": svd_fallbacks, "n_svd_fallbacks": len(svd_fallbacks),
               "arms": summ, "duan_criterion": RA.duan_block(summ),
               "population_group": {"declared_in": None, "note": choice("h")}}
    payload["elapsed_s"] = round(time.time() - t0, 1)
    payload["predictions_record"] = passing_predictions(code0)
    nb = RA.savez_bytes(arr)
    payload["draws_npz"] = {"file": rel(npz_path), "sha256": sha256_bytes(nb), "size": len(nb), "keys": sorted(arr)}
    payload["inputs"] = dict(_RESULT_READS)
    problems = input_identity_problems(payload)
    if problems:
        refuse(f"to write (D-297 (1) results-writer rule): {problems}")
    TP2.write_outputs([(npz_path, nb), (out_path, RA.json_bytes_of(payload))])
    print(f"Written: {out_path} ({payload['elapsed_s']} s)")
    return out_path


def world_p2(key: str, n_reps: int, code0: dict[str, Any],
             final: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """realign_d418.run_p2's binding, loading and anchors up to its world, line for line (CHOICES (b)); returns the
    world, the config and the input blocks."""
    RS, SP, T, F16 = RA.RS, RA.SP, RA.T, RA.F16
    b = RS.p2_bind(key, P.N_DRAWS, False, True, code0)
    if final:
        bad = RA.p2_final_problems(b, True)
        if bad:
            refuse(f"the final names need: {bad}")
    tp2_code = RS.tp2_code_block(code0)
    anchor_rec = TP2.passing_anchor(tp2_code, final=final)
    p4 = RA.read_v2_pool4d_positives(b["v2"])
    pooling, sr, seeds, C_sel, L_sel = b["pooling"], b["sr"], list(b["seeds"]), b["C_sel"], b["L_sel"]
    if seeds != RA.SEEDS:
        refuse(f"the selection's seeds {seeds} are not D-418's {RA.SEEDS}")
    v2, ev2, sb, post_sel = b["v2"], b["ev2"], b["sb"], b["post_sel"]
    n_cal, n_ev, n_pos = b["n_cal"], b["n_ev"], b["n_pos"]
    lay = [L_sel]
    z_tr, r_tr, _ = SP.load_ingested(sr["rec"], sr["rec_rel"], b["index_pre"][1], "probe_train", pooling)
    z_cal: dict[str, Any] = {}
    r_cal: dict[str, Any] = {}
    z_cal["pre"], r_cal["pre"], _ = SP.load_ingested(sr["rec"], sr["rec_rel"], b["index_pre"][1], "calibration",
                                                     pooling)
    recorded = sr["sel"].get("input_cache_sha256") if isinstance(sr["sel"].get("input_cache_sha256"), dict) else {}
    for part, r in (("probe_train", r_tr), ("calibration", r_cal["pre"])):
        k_ = f"pre/{SP.AXIS}/{part}/{pooling}"
        if recorded.get(k_) != r["sha256"]:
            refuse(f"{r['file']} is not the cache {sr['sel_rel']} read for {k_}")
    if post_sel["kind"] == "p2tpl":
        z_cal["post"], r_cal["post"], _ = SP.load_ingested(post_sel["rec"], post_sel["rel"], post_sel["index"],
                                                           "calibration", pooling)
    else:
        z_cal["post"], _, _, r_cal["post"] = TP2.open_sub_cache(post_sel, b["post_cal_entry"], "calibration", pooling,
                                                                lay)
    z_sub: dict[str, dict[str, Any]] = {}
    r_sub: dict[str, dict[str, Any]] = {}
    meta_sub: dict[str, dict[str, tuple[list[str], list[int]]]] = {}
    for side in ("pre", "post"):
        z_sub[side], r_sub[side], meta_sub[side] = {}, {}, {}
        for part in ("eval", "pool4d"):
            zz, ids, labs, r = TP2.open_sub_cache(sb[side], b["s_sub"][side][part], part, pooling, lay)
            z_sub[side][part], r_sub[side][part], meta_sub[side][part] = zz, r, (ids, labs)
    cal_meta = {side: ([str(x) for x in z_cal[side]["item_ids"]], [int(x) for x in z_cal[side]["labels"]])
                for side in ("pre", "post")}
    if cal_meta["pre"] != cal_meta["post"] or any(meta_sub["pre"][p_] != meta_sub["post"][p_]
                                                  for p_ in ("eval", "pool4d")):
        refuse("v1 calibration, eval or pool4d: the pre and post caches hold other items or labels")
    y_tr = z_tr["labels"].astype(bool)
    y_cal_v1 = np.asarray(cal_meta["pre"][1]) == 1
    k = f"layer_{L_sel}"
    probes = TP2.fit_probes(z_tr[k], y_tr, seeds, C_sel)
    sel_probes = T.seed_probes(z_tr[k], y_tr, b["arm"], seeds, b["holdout"])
    off = [q.diag.get("C") for q, s in zip(probes, sel_probes)
           if not (q.mu.dtype == s["mu"].dtype and q.mu.tobytes() == s["mu"].tobytes()
                   and q.sd.dtype == s["sd"].dtype and q.sd.tobytes() == s["sd"].tobytes()
                   and np.asarray(q.w).dtype == np.asarray(s["w"]).dtype
                   and np.asarray(q.w).tobytes() == np.asarray(s["w"]).tobytes() and q.b == s["b"])]
    if off or len(sel_probes) != len(probes):
        refuse("ANCHOR FAILED: the probes are not the selection's own fit bit for bit; nothing written")
    tprs = [r["tpr"] for r in T.tprs_at(sel_probes, z_cal["pre"][k], ~y_cal_v1, sr["target_fpr"])]
    diffs, n_cmp = T.anchor_diffs(tprs, sr["selected"])
    if diffs:
        refuse(f"ANCHOR FAILED: {len(diffs)} of {n_cmp} ho20 values differ from {sr['sel_rel']}; nothing written")
    mats: dict[str, dict[str, np.ndarray]] = {}
    gathered: dict[str, Any] = {}
    for side in ("pre", "post"):
        sources = {"calibration": (z_cal[side][k], *cal_meta[side]),
                   "eval": (z_sub[side]["eval"][k], *meta_sub[side]["eval"]),
                   "pool4d": (z_sub[side]["pool4d"][k], *meta_sub[side]["pool4d"])}
        Xc, Xc_pos, cc, _ = F16.gather_split("calibration", v2["ids"], [0] * v2["n"], sources)
        Xn, Xp, cn, cp = F16.gather_split("eval", ev2["ids"], ev2["labels"], sources)
        _, XH, _, cH = F16.gather_split("pool4d", p4["ids"], p4["labels"], sources)
        if Xc_pos.shape[0] or cc != v2["plan_counts"] or cn != ev2["expect_neg"] or cp != ev2["expect_pos"]:
            refuse(f"{side}: v2 items per v1 split are not the v2 record's plan")
        if XH.shape[0] != p4["n_pos"] or cH.get("pool4d") != p4["n_pos"]:
            refuse(f"{side}: the v2 pool4d positives are not the v1 pool4d cache's positives ({cH})")
        mats[side] = {"cal": Xc, "evn": Xn, "pos": Xp, "H": XH}
        gathered[side] = {"calibration": cc, "eval_negatives": cn, "eval_positives": cp, "pool4d_positives": cH}
    sc = TP2.score_matrices(probes, {"cal_pre": mats["pre"]["cal"], "ev_pre": mats["pre"]["evn"],
                                     "cal_post": mats["post"]["cal"], "ev_post": mats["post"]["evn"],
                                     "pos_pre": mats["pre"]["pos"], "pos_post": mats["post"]["pos"],
                                     "H_pre": mats["pre"]["H"], "H_post": mats["post"]["H"]})
    all_draws = P.repartition_draws(n_cal, n_ev, P.N_DRAWS, TP2.P2_DRAW_SEED)
    report, _ = P.repartition_estimate(sc["cal_pre"], sc["ev_pre"], sc["cal_post"], sc["ev_post"], sc["pos_pre"],
                                       sc["pos_post"], all_draws, RA.TARGET)
    if canon(report) != canon(b["d361_cells"][L_sel].get("realised")):
        refuse(f"the anchor: L{L_sel}: the realised operating point at 1% is not {b['d361_rel']}'s; nothing written")
    parts = TP2.make_parts({"item": all_draws[:n_reps]}, n_cal, n_cal + n_ev)["item"]
    perm = RS.ref_permutations(n_cal, n_reps, RS.axis_perm_seed(SP.AXIS))
    W = RA.p2_world(sc, mats, probes, parts, perm)
    for zz in (z_tr, *z_cal.values(), *(z for dd in z_sub.values() for z in dd.values())):
        zz.close()
    caches = {f"pre/{SP.AXIS}/probe_train/{pooling}": r_tr, f"pre/{SP.AXIS}/calibration/{pooling}": r_cal["pre"],
              b["post_cal_key"]: r_cal["post"]}
    for side in ("pre", "post"):
        for part in ("eval", "pool4d"):
            caches[f"{side}/{T.SUB_TAG}/{SP.AXIS}/{part}/{pooling}"] = r_sub[side][part]
    files_sha = {T.V2_CAL_FILE: sha256_file(SP.SPLIT_ROOT / T.V2_CAL_FILE),
                 F16.V2_EVAL_FILE: sha256_file(SP.SPLIT_ROOT / F16.V2_EVAL_FILE),
                 RA.V2_POOL4D_FILE: sha256_file(SP.SPLIT_ROOT / RA.V2_POOL4D_FILE)}
    split_want = ((T.V2_CAL_FILE, v2["sha256"]), (F16.V2_EVAL_FILE, ev2["sha256"]), (RA.V2_POOL4D_FILE, p4["sha256"]))

    def end_checks() -> list[str]:
        """realign_d418.run_p2's after-run checks: the caches, the three v2 split files, the results read (final)."""
        moved = [f for f, w in split_want if sha256_file(SP.SPLIT_ROOT / f) != w]
        moved += TP2.end_hash_problems({f"data/hidden/{r['file']}": (SP.CACHE / r["file"], r["sha256"])
                                        for r in caches.values()})
        if final:
            moved += TP2.end_hash_problems({kk: (TP2.resolve(kk), h) for kk, h in _RESULT_READS.items()})
        return moved
    config = {"population": "phase2_headline", "pair": key, "pair_group": b["group"], "pair_note": b["note"],
              "pre_repo": b["pre_repo"], "pre_rev": sr["rev"], "post_repo": b["post_repo"],
              "post_rev": post_sel["rev"], "pooling": pooling, "axis": SP.AXIS, "split_version": 2,
              "selection_record": {"file": sr["sel_rel"], "sha256": _RESULT_READS[sr["sel_rel"]]},
              "selected_arm": b["arm"], "selected_C": C_sel, "layer": L_sel, "seeds": seeds,
              "target_fpr": RA.TARGET, "draws": f"frozen split + the first {n_reps} of D-361's {P.N_DRAWS} draws",
              "pool_seed": RA.P2_POOL_SEED, "reference_permutation_seed": RS.axis_perm_seed(SP.AXIS),
              "n_calibration": n_cal, "n_evaluation_negatives": n_ev, "n_positives": n_pos,
              "n_harmful_H": p4["n_pos"], "draws_index_sha256": W["draws_sha256"]}
    inputs = {"anchor": {"d361_anchor_record": anchor_rec, "d361_pair_record": {"file": b["d361_rel"],
                                                                               "sha256": b["d361_sha"]},
                         "realised_operating_point_at_L_star_equal": True, "probes_equal_selection_fit": True,
                         "ho20_values_compared": n_cmp, "manifests": b["manifests"]},
              "v2_gather": gathered, "input_cache_sha256": {kk: r["sha256"] for kk, r in caches.items()},
              "input_cache_source": {kk: f"data/hidden/{r['file']}" for kk, r in caches.items()},
              "input_verification": TP2.D170_P2,
              "split_files": {"v2/calibration": {"file": T.V2_CAL_FILE, "sha256": v2["sha256"]},
                              "v2/eval": {"file": F16.V2_EVAL_FILE, "sha256": ev2["sha256"]},
                              "v2/pool4d": {"file": RA.V2_POOL4D_FILE, "sha256": p4["sha256"]}},
              "input_files_sha256": files_sha, "_end_checks": end_checks}
    return W, config, inputs


# ----------------------------------------------------------------------------- the summary
def summary_inputs() -> list[str]:
    files = [PRED_REL]
    for unit, (_, _, pairs) in UNITS.items():
        for pair in pairs:
            s = stem_of(unit, pair)
            files += [f"results/realign_d423_{unit}_{s}.json", f"results/realign_d423_{unit}_{s}_draws.npz"]
    return files


def run_summary(out: str | Path | None = None, records_dir: str | Path | None = None) -> Path:
    """results/realign_d423_summary.json: P-RA.1-4 per unit (separately), each record bound to its manifest line."""
    rd = Path(records_dir).resolve() if records_dir else RESULTS
    final_path = (RESULTS / "realign_d423_summary.json").resolve()
    out_path = Path(out).resolve() if out else final_path
    if out_path != final_path and out_path.is_relative_to(RESULTS.resolve()):
        refuse("a test run (--out) must not write under results/")
    if out_path.exists():
        refuse(f"{out_path} exists; results are never overwritten")
    code0 = P.code_block(CODE_FILES)
    pred = passing_predictions(code0, rd)
    listed, man_sha = TP2.read_manifest((LOGS if records_dir is None else rd) / Path(MANIFEST_OWN).name)
    scored, bound = {}, {}
    for unit, (pop, form, pairs) in UNITS.items():
        recs = {}
        for pair in pairs:
            s = stem_of(unit, pair)
            r_ = f"results/realign_d423_{unit}_{s}.json"
            p = rd / Path(r_).name
            if not p.is_file():
                refuse(f"{r_} is missing: every record is needed (a missing record is not a skip)")
            doc = read_result(p)
            h = _RESULT_READS[_repo_rel(p)]
            nz = doc.get("draws_npz") or {}
            if listed.get(r_) != h or listed.get(str(nz.get("file"))) != nz.get("sha256"):
                refuse(f"{r_} or its npz is not the file {MANIFEST_OWN} names")
            if (doc.get("predictions_record") or {}).get("sha256") != pred["sha256"]:
                refuse(f"{r_} was licensed by another predictions record")
            if (doc.get("config") or {}).get("map") != form or (doc.get("config") or {}).get("unit") != unit:
                refuse(f"{r_} is not a {unit} record")
            recs[s] = doc
            bound[r_] = {"file": r_, "sha256": h}
        scored[unit] = {"population": pop, "map": form, "n_reps": N_REPS[form], **RA.score_population(recs)}
    payload = {**provenance_block(RUN_KIND["summary"]), "preregistration": PREREG, "code": code0,
               "predictions_record": pred, "manifest": {"file": MANIFEST_OWN, "sha256": man_sha}, "records": bound,
               **registered_text(),
               "verdicts": {u: {k: v["verdict"] for k, v in sc_.items() if k.startswith("P-RA")}
                            for u, sc_ in scored.items()},
               "units": scored, "implementation_choices": CHOICES, "inputs": dict(_RESULT_READS)}
    problems = input_identity_problems(payload)
    if problems:
        refuse(f"to write (D-297 (1)): {problems}")
    TP2.write_outputs([(out_path, RA.json_bytes_of(payload))])
    print(json.dumps(payload["verdicts"], indent=1))
    return out_path


# ----------------------------------------------------------------------------- preflight, price
def run_preflight(lane: str) -> int:
    missing_all: list[str] = []
    code0 = P.code_block(CODE_FILES, allow_no_commit=True)
    for unit, pair in LANE_JOBS[lane]:
        run_job(unit, pair, records_only=True)
        pop = UNITS[unit][0]
        if pop == "phase1_released":
            names = RA.p1_cache_names(pair)
            miss = [n for n in names if not resolve_cache(n, REG.CACHE).is_file()]
        elif pop == "lora":
            cfg = REG.run(D.registry_stem(f"depth_{pair}"))
            names = [REG.cache_name(cfg[f"{s}_repo"], cfg["axis"], part) for s in ("pre", "post")
                     for part in ("probe_train", "threshold_calib", "eval")]
            miss = [n for n in names if not resolve_cache(n, REG.CACHE).is_file()]
        else:
            names = RA.p2_cache_names(RA.RS.p2_bind(pair, P.N_DRAWS, False, True, code0))
            miss = [n for n in names if not (RA.SP.CACHE / n).is_file()]
        if UNITS[unit][1] == "procrustes" and pop != "lora":
            kind = "p1" if pop == "phase1_released" else "p2"
            d418 = ROOT / f"results/realign_d418_{kind}_{stem_of(unit, pair)}.json"
            if not d418.is_file():
                miss.append(f"(D-418 record) {rel(d418)}")
        print(f"  {unit} {pair}: {len(names) - len([m for m in miss if not m.startswith('(')])} of {len(names)} cache "
              f"files present" + (f"; MISSING {miss}" if miss else ""))
        missing_all += miss
    print(f"PREFLIGHT lane {lane}: {len(LANE_JOBS[lane])} jobs; {len(missing_all)} inputs missing")
    return 1 if missing_all else 0


def run_price(pop: str, form: str, n_draws: int) -> dict[str, Any]:
    """Per-draw arithmetic (every arm, the unit's map) on a synthetic world of the real shapes; writes nothing."""
    t0 = time.time()
    W = RA.synthetic_world(pop, n_draws)
    t1 = time.time()
    chk = check_of(form, W)
    t2 = time.time()
    with map_form(form):
        RA.run_draws(W, lambda m: None)
    t3 = time.time()
    per_draw = (t3 - t2) / (n_draws + 1)
    out = {"population_shapes": pop, "map": form, "draws_timed": n_draws + 1, "build_s": round(t1 - t0, 1),
           "check_s": round(t2 - t1, 1), "check_passed": chk["passed"], "per_draw_s": round(per_draw, 2),
           "extrapolated_arithmetic_per_pair_min": round(per_draw * (N_REPS[form] + 1) / 60.0, 1),
           "blas_threads": RA.blas_threads(), "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    print(json.dumps(out))
    return out


# ----------------------------------------------------------------------------- the lane lists (CHOICES (g))
LANE_JOBS = {"C": [("lora_ridge", r) for r in LORA_RUNS] + [("lora_procrustes", r) for r in LORA_RUNS],
             "A": [("p2_procrustes", k) for k in P2_PAIRS],
             "D": [("p1_procrustes", s) for s in P1_PAIRS]}
PRICE_TEXT: dict[str, str] = {
    "C": ("PRICE (the D-423 node sandbox at da79f56, 2026-10-09, three other lanes busy, F-084; D-427): nine LoRA ridge "
          "jobs at D-418's per-pair cost (66-79 min a pair on an idle node, about 4.1 h under three busy lanes) and nine "
          "LoRA Procrustes jobs on 50 draws + frozen at the sandbox's 405 s a draw (Phase 1 shapes) under contention, "
          "about 5.7 h a pair; about 3.7 days at the contended rate, less as lanes free. NAS: per job one JSON "
          "(0.1-0.3 MB) and one npz (1-4 MB)."),
    "A": ("PRICE (as lane C's, D-427): nine Phase 2 Procrustes jobs on 50 draws + frozen at the sandbox's 446 s a draw "
          "under contention, about 6.3 h a pair, about 2.4 days at the contended rate, less as lanes free."),
    "D": ("PRICE (as lane C's, D-427): nine Phase 1 Procrustes jobs on 50 draws + frozen at the sandbox's 405 s a draw "
          "under contention, about 5.7 h a pair, about 2.1 days at the contended rate, less as lanes free."),
}
WAIT_LABEL = "wait_realign_d423_predictions"
GATE_KINDS = ("tests", "predictions", "job_lora_ridge", "job_lora_procrustes", "job_p1_procrustes", "job_p2_procrustes")
GATE_SLEEP_S, GATE_MAX_H = 60, 8


def gate_command(status_file: str) -> str:
    """The SANDBOX GATE (F-081/F-082 under D-400: the lists are chained while the node sandbox of one real job of each
    kind still runs, so the lane itself refuses to start real jobs before it passed): wait for the sandbox's status
    file to say ALL_DONE (or STOPPED, or GATE_MAX_H hours), then exit 1 unless the tests, the predictions job and one
    job of each of the four kinds each ended rc=0."""
    if not status_file or any(c in status_file for c in "\"'`;|&<>\n"):
        refuse(f"the gate's status file must be a plain path: {status_file!r}")
    n_max = GATE_MAX_H * 3600 // GATE_SLEEP_S
    kinds = " ".join(GATE_KINDS)
    return (f"f=\"{status_file}\"; n=0; while ! grep -q -e '^ALL_DONE' -e '^STOPPED' \"$f\" 2>/dev/null; do "
            f"n=$((n+1)); if [ \"$n\" -gt {n_max} ]; then echo \"D-423 sandbox not finished after {GATE_MAX_H} h\"; "
            f"exit 1; fi; sleep {GATE_SLEEP_S}; done; for k in {kinds}; do grep -q \"^$k rc=0 \" \"$f\" || "
            "{ echo \"D-423 sandbox: $k did not pass\"; exit 1; }; done; echo \"D-423 sandbox passed: $f\"")


def wait_command() -> str:
    """realign_d418.wait_command's loop, waiting for lane C's realign_d423_predictions job."""
    return RA.wait_command().replace("logs/queue/A/realign_predictions", "logs/queue/C/realign_d423_predictions") \
        .replace("logs/queue/A/STATE", "logs/queue/C/STATE").replace("lane A", "lane C") \
        .replace("realign_predictions", "realign_d423_predictions")


def lane_outputs(unit: str, pair: str) -> str:
    s = stem_of(unit, pair)
    return f"results/realign_d423_{unit}_{s}.json results/realign_d423_{unit}_{s}_draws.npz"


SANDBOX_SCRIPT = "scripts/sandbox_d423.sh"
SANDBOX_LABEL = "realign_d423_sandbox_c"


def sandbox_command(sandbox_dir: str) -> str:
    """Lane C's first job (D-427): the node sandbox of every job kind as a lane job (D-400: no session runs it)."""
    if not sandbox_dir.startswith("/") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                              "0123456789_./-" for c in sandbox_dir):
        refuse(f"the sandbox dir must be a plain absolute path: {sandbox_dir!r}")
    return f"bash {SANDBOX_SCRIPT} {sandbox_dir}"


def lane_list_text(lane: str, gate_status: str | None = None, sandbox_dir: str | None = None) -> str:
    if sandbox_dir is not None and gate_status != f"{sandbox_dir}/status.txt":
        refuse(f"the gates must read the lane sandbox's own status file ({sandbox_dir}/status.txt), not {gate_status}")
    units = sorted({u for u, _ in LANE_JOBS[lane]})
    head = [
        f"# Lane {lane} (CPU only, no network, no GPU; D-423, the author's item 2 of 2026-10-09, registered and "
        f"committed before any D-418 record was pulled or read): the REALIGNMENT AMENDMENT by src/realign_d423.py, "
        f"units {units}. (a) the LoRA population (the nine LoRA arms of populations.LORA_ARMS) with D-418's ridge map; "
        "(b) an orthogonal Procrustes arm (centred, the canonical minimiser closest to the identity) beside the ridge "
        "map. Each (population, map) unit scored SEPARATELY with D-418's four predictions (P-RA.1-4, text unchanged).",
        "# DO NOT LOAD before the review and the push: ONE sonnet reviewer (the author's rule for this item) and its fix "
        "round, the D-423 entry committed BEFORE any job of this list runs, the node sandbox run of one job of each "
        "kind (F-081/F-082), and the push that carries the module, its tests and the three lists. RELAUNCH of "
        "2026-10-10 (D-427): the 2026-10-09 lists' gates stopped every lane on the sandbox's Procrustes map check "
        "(F-085); the sandbox of the fixed commit runs as lane C's first job, and every gate reads its status file.",
        "# THIS LIST is one of three (C: the predictions job, nine LoRA ridge and nine LoRA Procrustes jobs; A: nine "
        "Phase 2 Procrustes jobs, chained behind D-418's lane A list; D: nine Phase 1 Procrustes jobs, chained behind "
        "D-418's lane D list). Lane C's second job writes results/realign_d423_predictions.json, which every job "
        "requires; lanes A and D open with a wait for logs/queue/C/realign_d423_predictions.done. The Phase 1 and "
        "Phase 2 Procrustes jobs read D-418's records of the same pair (their anchor). The manifest job (sha256sum of "
        f"{len(summary_inputs())} files into {MANIFEST_OWN}) and `python3 -u src/realign_d423.py summary` run after "
        "all three lanes, in a review session.",
        "# Final names need the registered draws (ridge 200, Procrustes 50), seeds 0-4, 1%, the 6-thread BLAS pin "
        "(scripts/project_env.sh), GUARDRAIL_CACHE_DIRS unset and the predictions record. Runs from the repo root "
        "through scripts/detach.sh, on the NODE.",
        "# " + PRICE_TEXT[lane],
    ]
    jobs = [(f"realign_d423_preflight_{lane.lower()}", "-", f"python3 -u src/realign_d423.py preflight --lane {lane}")]
    if sandbox_dir is not None and lane == "C":
        jobs.insert(0, (SANDBOX_LABEL, "-", sandbox_command(sandbox_dir)))
    if gate_status:
        jobs.append((f"realign_d423_sandbox_gate_{lane.lower()}", "-", gate_command(gate_status)))
    if lane == "C":
        jobs.append(("realign_d423_predictions", PRED_REL, "python3 -u src/realign_d423.py predictions"))
    else:
        jobs.append((f"{WAIT_LABEL}_{lane.lower()}", "-", wait_command()))
    for unit, pair in LANE_JOBS[lane]:
        jobs.append((f"realign_d423_{unit}_{stem_of(unit, pair)}", lane_outputs(unit, pair),
                     f"python3 -u src/realign_d423.py job --unit {unit} --pair {pair}"))
    return "\n".join(head + ["\t".join(j) for j in jobs]) + "\n"


def run_queue(write: bool = False, date: str | None = None, root: Path | None = None,
              gate_status: str | None = None, sandbox_dir: str | None = None) -> dict[str, str]:
    root = root or ROOT
    date = date or datetime.now(timezone.utc).strftime("%Y%m%d")
    if sandbox_dir is not None and gate_status is None:
        gate_status = f"{sandbox_dir}/status.txt"
    texts = {f"lane{ln}_{date}_realign_d423.tsv": lane_list_text(ln, gate_status, sandbox_dir) for ln in LANE_JOBS}
    bad = RA.label_conflicts(texts, root)
    if bad:
        refuse(f"labels not unique: {bad}")
    if write:
        for name, t in texts.items():
            p = root / "queues" / name
            if p.exists():
                refuse(f"{p} exists")
            p.write_text(t, encoding="utf-8", newline="\n")
            print(f"Written: {p}")
    return texts


# ----------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="D-423: the realignment amendment")
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("predictions")
    s.add_argument("--out", default=None)
    s = sub.add_parser("job")
    s.add_argument("--unit", required=True, choices=sorted(UNITS))
    s.add_argument("--pair", required=True)
    s.add_argument("--n-reps", type=int, default=None)
    s.add_argument("--out", default=None)
    s.add_argument("--records-only", action="store_true")
    s = sub.add_parser("preflight")
    s.add_argument("--lane", required=True, choices=sorted(LANE_JOBS))
    s = sub.add_parser("summary")
    s.add_argument("--out", default=None)
    s.add_argument("--records-dir", default=None)
    s = sub.add_parser("price")
    s.add_argument("--population", required=True, choices=sorted(RA.SHAPES))
    s.add_argument("--map", required=True, choices=MAPS)
    s.add_argument("--draws", type=int, default=2)
    s = sub.add_parser("queue")
    s.add_argument("--write", action="store_true")
    s.add_argument("--date", default=None)
    s.add_argument("--gate-status", default=None, help="the node sandbox's status file the gate job waits on")
    s.add_argument("--sandbox-dir", default=None,
                   help="run the sandbox as lane C's first job into this new absolute dir; the gates read its status")
    return ap


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    a = build_parser().parse_args(argv)
    if a.mode == "predictions":
        run_predictions(a.out)
    elif a.mode == "job":
        run_job(a.unit, a.pair, a.n_reps, out=a.out, records_only=a.records_only)
    elif a.mode == "preflight":
        return run_preflight(a.lane)
    elif a.mode == "summary":
        run_summary(a.out, a.records_dir)
    elif a.mode == "price":
        run_price(a.population, a.map, a.draws)
    elif a.mode == "queue":
        for name, t in run_queue(a.write, a.date, gate_status=a.gate_status, sandbox_dir=a.sandbox_dir).items():
            if not a.write:
                print(f"== {name}\n{t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
