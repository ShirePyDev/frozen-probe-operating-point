"""The one definition of the paired-damage machinery, shared by every script (D-152 P3).

Until 2026-09-24 each analysis script carried its own copy of the probe fit, the
threshold rule and the paired ratio, and two of them disagreed on the same C=1 cell
(F-059) because they fitted the same rows in different orders. Everything a paired
result depends on is therefore defined HERE once, and the point estimate and its
bootstrap replicates go through the SAME functions (counts=None for the point, a
resample-counts matrix for the replicates), so a CI can never use a different
convention from the number it is reported with:

  d117_split          the D-117 split: perm = default_rng(seed).permutation(n); hold out
                      the first 20%, fit on the rest
  fit_probe           logistic probe standardised on the fit rows, fitted on the rows IN
                      THE ORDER GIVEN, labelled by name: "d117_permutation" (the selection
                      fit, P1), "file_order" or "reversed_file_order" (the C=1 check, P2);
                      lbfgs iterations, convergence flag and gradient norm recorded
  thr_at_fpr          the frozen threshold
  floored_fpr / rate  per-seed flag rate (FPR floored at 1/n_neg, F-034), point or replicate
  ratio_from_flags    the paired ratio: mean over seeds of floored post/pre FPR
  delta_tpr_from_flags  mean over seeds of TPR_post - TPR_pre at the frozen threshold
  paired_fpr, tpr_pair  the full per-cell record, with counts, ranges and discordance
  mcnemar_exact       two-sided exact McNemar, in exact integer arithmetic (no overflow)
  counts_matrix, ci   the shared item bootstrap and its percentile interval
  code_block          sha256 of every source file a run used, the synced commit, library
                      versions; refuses to run with no commit identity at all

Nothing here reads or writes a file except code_block (which hashes source files).
"""

from __future__ import annotations

import hashlib
import subprocess
import warnings
from dataclasses import dataclass, field
from math import comb
from pathlib import Path
from typing import Any

import numpy as np

HOLDOUT = 0.20
TARGET_FPR = 0.01
MAX_ITER = 2000
ROOT = Path(__file__).resolve().parent.parent
ROW_ORDERS = ("d117_permutation", "file_order", "reversed_file_order")


def d117_split(n: int, seed: int, holdout: float = HOLDOUT) -> tuple[np.ndarray, np.ndarray]:
    """(ho, fit) exactly as src/regularisation_sweep.py draws them for the selection;
    `fit` is in PERMUTATION order, which is the order the selection fitted in."""
    perm = np.random.default_rng(seed).permutation(n)
    n_ho = int(holdout * n)
    return perm[:n_ho], perm[n_ho:]


@dataclass
class Probe:
    w: np.ndarray
    b: float
    mu: np.ndarray
    sd: np.ndarray
    diag: dict[str, Any] = field(default_factory=dict)

    def z(self, x: np.ndarray) -> np.ndarray:
        """Features standardised with the probe's (pre-update, fit-set) statistics."""
        return (x - self.mu) / self.sd

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.z(x) @ self.w + self.b


def fit_probe(x_fit: np.ndarray, y_fit: np.ndarray, seed: int, C: float,
              row_order: str, tol: float | None = None) -> Probe:
    """Fit on the rows IN THE ORDER GIVEN; `row_order` names that order (ROW_ORDERS).

    Standardisation uses the fit rows' own mean and sd, as the D-117 selection did.
    Recorded for every fit: lbfgs iterations, whether it stopped before MAX_ITER without
    a ConvergenceWarning, and the norm of the objective's gradient at the solution
    (objective = 0.5 ||w||^2 + C * sum(logloss); the intercept is not penalised).
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    if row_order not in ROW_ORDERS:
        raise ValueError(f"unknown row order {row_order!r}; one of {ROW_ORDERS}")
    mu, sd = x_fit.mean(0), x_fit.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    xs = (x_fit - mu) / sd
    ys = np.asarray(y_fit).astype(int)
    kw = {} if tol is None else {"tol": tol}     # None = the library default (1e-4)
    clf = LogisticRegression(max_iter=MAX_ITER, C=C, random_state=seed, **kw)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(xs, ys)
    w, b = clf.coef_.ravel().astype(np.float64), float(clf.intercept_[0])
    p = 1.0 / (1.0 + np.exp(-(xs @ w + b)))
    g_w = w + C * (xs.T @ (p - ys))
    g_b = C * float(np.sum(p - ys))
    n_iter = int(np.max(clf.n_iter_))
    warned = any(issubclass(c.category, ConvergenceWarning) for c in caught)
    return Probe(w, b, mu, sd, {
        "C": C, "row_order": row_order, "tol": float(clf.tol), "n_fit": int(len(ys)), "n_iter": n_iter,
        "max_iter": MAX_ITER, "converged": bool(n_iter < MAX_ITER and not warned),
        "convergence_warning": warned,
        "grad_norm": float(np.sqrt(np.sum(g_w ** 2) + g_b ** 2)),
        "grad_max_abs": float(max(np.max(np.abs(g_w)), abs(g_b))),
        "w_norm": float(np.linalg.norm(w)),
    })


def _thr_at_fpr_loop(neg_scores: np.ndarray, target: float = TARGET_FPR) -> float:
    """The project's original definition (13 scripts carry it): the smallest distinct
    score t whose exceedance (neg >= t).mean() is <= target, else +inf. Kept as the
    reference the vectorised form below is tested against."""
    c = np.concatenate([np.unique(neg_scores), [np.inf]])
    for t in c:
        if (neg_scores >= t).mean() <= target:
            return float(t)
    return float(np.inf)


def thr_at_fpr(neg_scores: np.ndarray, target: float = TARGET_FPR) -> float:
    """The frozen threshold, same definition as _thr_at_fpr_loop, in O(n log n).

    count_ge(u) = n - searchsorted(sorted, u, 'left') for each distinct u; the answer is
    the first distinct value whose exceedance fraction is <= target. Needed for the
    no-update nulls (tens of thousands of thresholds). Equality with the loop form is
    tested on random and tie-heavy inputs before use.
    """
    a = np.sort(np.asarray(neg_scores).ravel())
    u = np.unique(a)
    frac = (a.size - np.searchsorted(a, u, side="left")) / a.size
    ok = np.flatnonzero(frac <= target)
    return float(u[ok[0]]) if ok.size else float(np.inf)


def quantile_transfer(pre_ref: np.ndarray, post_ref: np.ndarray, t: float) -> float:
    """The quantile-transfer threshold (F-070): t's fractional rank among the sorted PRE reference scores, carried to
    the sorted POST reference scores. With the same items and no update (post == pre) it returns t exactly, and with a
    pure translation by c it returns t + c. When the two windows differ in size (a pre window against a larger post
    window, D-244) the rank is carried by its position scaled by n_post/n_pre, which keeps the upper-tail fraction. The
    earlier form, np.quantile(post, share of pre below t) with linear interpolation, lands about one order statistic
    BELOW t with no update (fresh-item FPR x1.10 at n = 1000, x1.49 at n = 200). A t above every pre score transfers
    to +inf (nothing flagged)."""
    s0 = np.sort(np.asarray(pre_ref, dtype=np.float64).ravel())
    s1 = np.sort(np.asarray(post_ref, dtype=np.float64).ravel())
    if s0.size == 0 or s1.size == 0:
        raise ValueError("quantile_transfer needs non-empty reference windows")
    if t > s0[-1]:
        return float(np.inf)
    pos = float(np.interp(t, s0, np.arange(s0.size, dtype=np.float64)))
    pos1 = pos * s1.size / s0.size if s1.size != s0.size else pos
    return float(np.interp(pos1, np.arange(s1.size, dtype=np.float64), s1))


# --- rates: ONE definition for the point estimate and for every bootstrap replicate ---

def rate(F: np.ndarray, counts: np.ndarray | None = None) -> np.ndarray:
    """Flag rate per seed. F: (n, S) booleans. Point: (S,). With counts (B, n): (B, S)."""
    n = F.shape[0]
    if counts is None:
        return F.mean(axis=0)
    return counts @ F.astype(np.float32) / n


def floored_fpr(F: np.ndarray, counts: np.ndarray | None = None) -> np.ndarray:
    """FPR per seed, floored at the resolution 1/n_neg (F-034)."""
    return np.maximum(rate(F, counts), 1.0 / F.shape[0])


def per_seed_ratio(F_pre: np.ndarray, F_post: np.ndarray,
                   counts: np.ndarray | None = None) -> np.ndarray:
    """THE paired-ratio formula (D-152 P3): floored post FPR / floored pre FPR, per seed.
    F_*: (n_neg, S) flags of the SAME negatives. Point: (S,). Replicates: (B, S).
    Every other statement of the ratio in this repo calls this function."""
    return floored_fpr(F_post, counts) / floored_fpr(F_pre, counts)


def ratio_from_flags(F_pre: np.ndarray, F_post: np.ndarray,
                     counts: np.ndarray | None = None) -> np.ndarray | float:
    """The reported paired ratio: MEAN over seeds of per_seed_ratio (the paired_damage.py
    convention since D-119). Point: float. Replicates: (B,)."""
    r = per_seed_ratio(F_pre, F_post, counts).mean(axis=-1)
    return float(r) if counts is None else r


def delta_tpr_from_flags(P_pre: np.ndarray, P_post: np.ndarray,
                         counts: np.ndarray | None = None) -> np.ndarray | float:
    """Mean over seeds of TPR_post - TPR_pre at the frozen threshold (no floor)."""
    d = (rate(P_post, counts) - rate(P_pre, counts)).mean(axis=-1)
    return float(d) if counts is None else d


def _rng(v: np.ndarray) -> list[float]:
    return [float(np.min(v)), float(np.max(v))]


def paired_fpr(PRE: np.ndarray, POST: np.ndarray, T: np.ndarray) -> dict[str, Any]:
    """PRE/POST: (n_neg, S) scores of the SAME negatives; T: (S,) per-seed thresholds."""
    F_pre, F_post = PRE >= T, POST >= T
    f_pre, f_post = floored_fpr(F_pre), floored_fpr(F_post)
    per_seed = per_seed_ratio(F_pre, F_post)
    return {"n_neg": int(PRE.shape[0]), "floor": 1.0 / PRE.shape[0],
            "fpr_pre_per_seed": f_pre.tolist(), "fpr_post_per_seed": f_post.tolist(),
            "fpr_pre_mean": float(f_pre.mean()), "fpr_pre_range": _rng(f_pre),
            "fpr_post_mean": float(f_post.mean()), "fpr_post_range": _rng(f_post),
            "ratio_per_seed": per_seed.tolist(), "ratio_mean": ratio_from_flags(F_pre, F_post),
            "ratio_range": _rng(per_seed),
            "n_flagged_pre_per_seed": F_pre.sum(axis=0).astype(int).tolist(),
            "n_flagged_post_per_seed": F_post.sum(axis=0).astype(int).tolist(),
            "n_censored_pre": int((rate(F_pre) == 0).sum()),
            "n_censored_post": int((rate(F_post) == 0).sum())}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p on b + c discordant pairs (Binomial(b + c, 1/2)).

    Exact integer arithmetic: tail * 2 / 2**n is a Python int/int true division, which
    is correctly rounded and cannot overflow (the float version raised OverflowError
    once b + c passed about 1024, which positives at a 5% threshold can reach)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1))
    return float(min(1.0, (2 * tail) / (1 << n)))


def discordance(PRE: np.ndarray, POST: np.ndarray, T: np.ndarray) -> dict[str, Any]:
    fl_pre, fl_post = PRE >= T, POST >= T
    b = (fl_pre & ~fl_post).sum(axis=0)
    c = (~fl_pre & fl_post).sum(axis=0)
    p = [mcnemar_exact(int(x), int(y)) for x, y in zip(b, c)]
    return {"n_both_per_seed": (fl_pre & fl_post).sum(axis=0).astype(int).tolist(),
            "n_pre_only_per_seed": b.astype(int).tolist(),
            "n_post_only_per_seed": c.astype(int).tolist(),
            "mcnemar_exact_p_per_seed": p, "mcnemar_exact_p_max": float(max(p))}


def tpr_pair(PRE_pos: np.ndarray, POST_pos: np.ndarray, T: np.ndarray) -> dict[str, Any]:
    """TPR at the FROZEN threshold, pre and post, on the SAME positives, per seed."""
    P_pre, P_post = PRE_pos >= T, POST_pos >= T
    t_pre, t_post = rate(P_pre), rate(P_post)
    return {"n_pos": int(PRE_pos.shape[0]),
            "tpr_pre_per_seed": t_pre.tolist(), "tpr_post_per_seed": t_post.tolist(),
            "tpr_pre_mean": float(t_pre.mean()), "tpr_pre_range": _rng(t_pre),
            "tpr_post_mean": float(t_post.mean()), "tpr_post_range": _rng(t_post),
            "delta_tpr_mean": delta_tpr_from_flags(P_pre, P_post),
            "delta_tpr_range": _rng(t_post - t_pre),
            "hits_pre_per_seed": P_pre.sum(axis=0).astype(int).tolist(),
            "hits_post_per_seed": P_post.sum(axis=0).astype(int).tolist(),
            **{f"pos_{k}": v for k, v in discordance(PRE_pos, POST_pos, T).items()}}


# --- the shared item bootstrap ------------------------------------------------------

def counts_matrix(n: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) float32 resample counts; row r is replicate r's draw of n items with
    replacement. The SAME matrix is applied to every run that shares the items, so
    replicate r means the same resample everywhere (between-run CIs are paired)."""
    rng = np.random.default_rng(seed)
    cnt = np.zeros((n_boot, n), dtype=np.float32)
    for r in range(n_boot):
        cnt[r] = np.bincount(rng.integers(0, n, size=n), minlength=n)
    return cnt


def ci(reps: np.ndarray) -> dict[str, Any]:
    """Percentile 95% interval of bootstrap replicates, with the number of non-finite
    replicates COUNTED rather than silently dropped (None when none is finite)."""
    v = np.asarray(reps, dtype=float).ravel()
    ok = v[np.isfinite(v)]
    return {"ci95": [float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))] if len(ok) else None,
            "n_replicates": int(len(v)), "n_nonfinite": int(len(v) - len(ok))}


# --- D-282: the item bootstrap over positives combined with the re-partition draws ----

def draw_of_replicate(n_boot: int, n_draws: int) -> np.ndarray:
    """D-282: replicate r = 1..n_boot is evaluated at re-partition draw d_r = 1 + ((r - 1) mod n_draws)."""
    return 1 + np.arange(n_boot) % n_draws


def combined_tpr_replicates(T_draws: np.ndarray, pos_pre: np.ndarray, pos_post: np.ndarray,
                            counts: np.ndarray) -> dict[str, np.ndarray]:
    """D-282: the combined replicates of TPR pre, TPR post and delta TPR (each the mean over seeds).

    T_draws (n_draws, S): the per-seed thresholds of draws 1..n_draws, row j being draw j + 1 (the frozen split is
    not a draw). pos_pre, pos_post (n, S): the positives' scores. counts (B, n): item-bootstrap counts over the
    positives. Replicate r uses draw d_r's thresholds and row r - 1 of counts. The same row weights the pre and the
    post flags and every seed, so the resample is PAIRED WITHIN ITEM. Rates use rate()'s weighted form on float64
    counts, so a row of ones returns the unweighted value bit for bit."""
    if pos_pre.shape != pos_post.shape or counts.shape[1] != pos_pre.shape[0] or T_draws.shape[1] != pos_pre.shape[1]:
        raise ValueError("combined_tpr_replicates: the shapes of the thresholds, scores and counts do not agree")
    d_of_r = draw_of_replicate(counts.shape[0], T_draws.shape[0])
    out = {q: np.full(counts.shape[0], np.nan) for q in ("tpr_pre", "tpr_post", "delta_tpr")}
    for j in range(T_draws.shape[0]):
        r = np.flatnonzero(d_of_r == j + 1)
        if r.size == 0:
            continue
        C = counts[r].astype(np.float64)
        Q_pre, Q_post = pos_pre >= T_draws[j], pos_post >= T_draws[j]
        out["tpr_pre"][r] = rate(Q_pre, C).mean(axis=-1)
        out["tpr_post"][r] = rate(Q_post, C).mean(axis=-1)
        out["delta_tpr"][r] = delta_tpr_from_flags(Q_pre, Q_post, C)
    return out


# --- code identity ------------------------------------------------------------------

def code_block(files: list[str | Path], allow_no_commit: bool = False) -> dict[str, Any]:
    """sha256 of each source file used, the commit the tree was synced from, and the
    numeric library versions. Call it at START-UP, before any computation, and compare at
    the end, so the record names the code that actually ran.

    Commit identity: SYNC_COMMIT at the repo root (written by scripts/receive_push.py at
    each push to the node, which has no .git, D-148); on a tree with .git, HEAD and a
    dirty flag. With neither, refuse unless allow_no_commit is set, and say so in the record.
    """
    from importlib import metadata

    out: dict[str, Any] = {}
    for f in files:
        p = Path(f)
        p = p if p.is_absolute() else ROOT / p
        out[str(p.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(p.read_bytes()).hexdigest()
    # D-152 P5 names "each script and imported src module": refuse if a loaded src module is
    # missing from the list (found 2026-09-25: populations.py began importing
    # superseded_provenance.py and no CODE_FILES list followed). Called at start AND end, so a
    # module imported lazily mid-run is caught too.
    import sys
    src_dir = (ROOT / "src").resolve()
    loaded = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f and Path(f).suffix == ".py" and Path(f).resolve().parent == src_dir:
            loaded.add(str(Path(f).resolve().relative_to(ROOT.resolve())).replace("\\", "/"))
    unhashed = sorted(loaded - set(out))
    if unhashed:
        raise SystemExit(f"REFUSING: src modules loaded but not in the code block: {unhashed}")
    sc = ROOT / "SYNC_COMMIT"
    commit, source = None, None
    if sc.exists():
        commit, source = sc.read_text().strip(), "SYNC_COMMIT"
    elif (ROOT / ".git").exists():
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True)
        commit = head.stdout.strip() or None
        source = "git HEAD" + (" (dirty tree)" if dirty.stdout.strip() else "")
    if commit is None and not allow_no_commit:
        raise SystemExit("REFUSING: no commit identity (no SYNC_COMMIT, no .git); push the tree "
                         "with scripts/push_to_node.sh or pass the explicit override")
    libs = {}
    for dist in ("numpy", "scipy", "scikit-learn"):
        try:
            libs[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            libs[dist] = None
    return {"sha256": out, "commit": commit, "commit_source": source or "NONE (override)",
            "libraries": libs}


# --- D-156: the re-partition estimator for operating-point metrics -------------------

def min_mcnemar_p(n_discordant: int) -> float:
    """Smallest two-sided exact McNemar p attainable with n discordant pairs: every one
    of them going the same way, p = 2 * 0.5**n = 2**(1-n) (1.0 for n <= 1)."""
    return 1.0 if n_discordant <= 1 else float(min(1.0, 2.0 ** (1 - n_discordant)))


# D-156: one fixed partition seed per axis, so draw r is the same item split for every cell
# on that axis in every script (first used by src/damage_d156.py, results/damage_d156.json).
AXIS_DRAW_SEED = {"safety": 20260925, "safety_wildguard": 20260926}
N_DRAWS = 1000


def band_summary(per_draw: np.ndarray, frozen: float | None) -> dict[str, Any]:
    """D-156 report of one operating-point quantity: the mean over the draws, the 2.5-97.5%
    band, the frozen split's value and its mid-rank percentile among the draws. Non-finite
    draws (an undefined prediction) are COUNTED and left out, never silently dropped."""
    a = np.asarray(per_draw, dtype=float).ravel()
    ok = a[np.isfinite(a)]
    fz = float(frozen) if frozen is not None and np.isfinite(frozen) else None
    return {"mean": float(ok.mean()) if ok.size else None,
            "band95": [float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))] if ok.size else None,
            "frozen": fz,
            "frozen_percentile": (float(100.0 * (np.sum(ok < fz) + 0.5 * np.sum(ok == fz)) / ok.size)
                                  if ok.size and fz is not None else None),
            "n_nonfinite_draws": int(a.size - ok.size),
            # spread of the draws, so the Monte Carlo error of the mean (sd / sqrt(n)) and the
            # noise between two draw sets (e.g. item vs cluster level) can be read off the file
            "sd_over_draws": float(ok.std(ddof=1)) if ok.size > 1 else None}


def repartition_draws(n_cal: int, n_ev: int, n_draws: int, seed: int) -> np.ndarray:
    """(n_draws, n_cal + n_ev) permutations of the pooled negatives. Row r's first n_cal
    indices are draw r's calibration part, the rest its evaluation part. Pool order is
    calibration negatives then eval negatives, each in cache order, so index i is the
    same item for every cell on the same axis (D-156: one partition per draw everywhere)."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(n_cal + n_ev) for _ in range(n_draws)])


# D-164: the cluster-level sensitivity draws its partitions from its own per-axis seeds.
AXIS_CLUSTER_DRAW_SEED = {"safety": 20260927, "safety_wildguard": 20260928}


def cluster_units(pool_ids: list[str], clusters: list[list[str]]) -> np.ndarray:
    """Unit label per pooled item: items of one near-duplicate cluster share a label, every
    other item is its own unit. Refuses a cluster member that is not in the pool."""
    index = {iid: i for i, iid in enumerate(pool_ids)}
    unit = np.arange(len(pool_ids))
    seen: set[str] = set()
    for g in clusters:
        missing = [iid for iid in g if iid not in index]
        if missing:
            raise ValueError(f"cluster members not in the pooled negatives: {missing[:3]}")
        twice = [iid for iid in g if iid in seen]
        if twice:   # an item in two clusters would split the first one into straddling units
            raise ValueError(f"items listed in more than one cluster: {twice[:3]}")
        seen.update(g)
        first = index[g[0]]
        for iid in g:
            unit[index[iid]] = first
    return unit


def cluster_repartition_draws(unit: np.ndarray, n_cal: int, n_draws: int, seed: int) -> np.ndarray:
    """D-164: (n_draws, n_pool) partitions like repartition_draws (first n_cal indices are the
    calibration part) in which no unit straddles calibration and evaluation. Per draw the units
    are permuted and the calibration part is filled unit by unit in that order; a unit that would
    overflow n_cal goes to evaluation. The calibration part ends at exactly n_cal because the
    singletons (thousands, against clusters of at most 14 items) fill any remainder; if it did not,
    this refuses rather than split a cluster. (A second pass cannot help - the fill only grows -
    so there is none; review 2026-09-25 measured no fill bias by unit size.)"""
    rng = np.random.default_rng(seed)
    labels, inverse = np.unique(unit, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(labels))]
    sizes = np.array([len(m) for m in members])
    out = []
    for _ in range(n_draws):
        order = rng.permutation(len(members))
        to_cal = np.zeros(len(members), bool)
        filled = 0
        for k in order:
            if filled + sizes[k] <= n_cal:
                to_cal[k] = True
                filled += sizes[k]
        if filled != n_cal:
            raise ValueError(f"could not fill the calibration part to {n_cal} without splitting a cluster")
        cal = np.concatenate([members[k] for k in order if to_cal[k]])
        ev = np.concatenate([members[k] for k in order if not to_cal[k]])
        out.append(np.concatenate([cal, ev]))
    return np.stack(out)


def operating_point(pool_pre: np.ndarray, pool_post: np.ndarray, cal: np.ndarray, ev: np.ndarray,
                    pos_pre: np.ndarray, pos_post: np.ndarray, target: float = TARGET_FPR) -> dict[str, Any]:
    """Every operating-point metric for ONE partition: thresholds per seed at `target` on
    the PRE scores of the calibration part; FPR on the evaluation part (pre and post, same
    items); TPR on the fixed positives. All arrays (items, seeds)."""
    T = np.array([thr_at_fpr(pool_pre[cal, s], target) for s in range(pool_pre.shape[1])])
    F_pre, F_post = pool_pre[ev] >= T, pool_post[ev] >= T
    Q_pre, Q_post = pos_pre >= T, pos_post >= T
    f_pre, f_post = floored_fpr(F_pre), floored_fpr(F_post)
    b = (F_pre & ~F_post).sum(axis=0)
    c = (~F_pre & F_post).sum(axis=0)
    p = [mcnemar_exact(int(x), int(y)) for x, y in zip(b, c)]
    bp = (Q_pre & ~Q_post).sum(axis=0)
    cp = (~Q_pre & Q_post).sum(axis=0)
    pp = [mcnemar_exact(int(x), int(y)) for x, y in zip(bp, cp)]
    return {"fpr_pre": float(f_pre.mean()), "fpr_post": float(f_post.mean()),
            "paired_ratio": ratio_from_flags(F_pre, F_post),
            "ratio_to_target": float(f_post.mean() / target),
            "floor_ratio_to_target": float(f_pre.mean() / target),
            "floor_abs_log": float(abs(np.log(f_pre.mean() / target))),
            "tpr_pre": float(rate(Q_pre).mean()), "tpr_post": float(rate(Q_post).mean()),
            "delta_tpr": delta_tpr_from_flags(Q_pre, Q_post),
            "mcnemar_neg_p_max": float(max(p)),
            "mcnemar_neg_min_achievable_p": float(max(min_mcnemar_p(int(x + y)) for x, y in zip(b, c))),
            "mcnemar_pos_p_max": float(max(pp)),
            "mcnemar_pos_min_achievable_p": float(max(min_mcnemar_p(int(x + y)) for x, y in zip(bp, cp))),
            "_counts": {"n_flagged_pre": F_pre.sum(axis=0).tolist(), "n_flagged_post": F_post.sum(axis=0).tolist(),
                        "neg_discordant_b_c": [[int(x), int(y)] for x, y in zip(b, c)],
                        "hits_pre": Q_pre.sum(axis=0).tolist(), "hits_post": Q_post.sum(axis=0).tolist(),
                        "pos_discordant_b_c": [[int(x), int(y)] for x, y in zip(bp, cp)],
                        "n_eval_negatives": int(len(ev)), "n_positives": int(pos_pre.shape[0])}}


def abs_log(r: Any) -> np.ndarray:
    """|log r| elementwise in float64: the ONE definition of the frozen curve |log(FPR_post / target)|
    per draw (D-180), shared by depth_d156 and 4d so their values can match bit for bit."""
    return np.abs(np.log(np.asarray(r, dtype=np.float64)))


D156_METRICS = ("fpr_pre", "fpr_post", "paired_ratio", "ratio_to_target", "floor_ratio_to_target",
                "floor_abs_log", "tpr_pre", "tpr_post", "delta_tpr")


def repartition_estimate(cal_pre: np.ndarray, ev_pre: np.ndarray, cal_post: np.ndarray, ev_post: np.ndarray,
                         pos_pre: np.ndarray, pos_post: np.ndarray, draws: np.ndarray,
                         target: float = TARGET_FPR) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """D-156 for one cell. Returns (report, per-draw arrays). The frozen split is draw
    'frozen' (calibration part = the recorded threshold_calib negatives), evaluated by
    the same function as every random draw, and placed by its mid-rank percentile."""
    n_cal = cal_pre.shape[0]
    pool_pre, pool_post = np.vstack([cal_pre, ev_pre]), np.vstack([cal_post, ev_post])
    if draws.shape[1] != pool_pre.shape[0]:
        raise ValueError("draws do not match the pooled negatives")
    frozen = operating_point(pool_pre, pool_post, np.arange(n_cal), np.arange(n_cal, pool_pre.shape[0]),
                             pos_pre, pos_post, target)
    per = [operating_point(pool_pre, pool_post, d[:n_cal], d[n_cal:], pos_pre, pos_post, target) for d in draws]
    arrays = {m: np.array([x[m] for x in per]) for m in D156_METRICS + ("mcnemar_neg_p_max", "mcnemar_pos_p_max")}
    report: dict[str, Any] = {"n_draws": int(len(draws))}
    for m in D156_METRICS:
        report[m] = band_summary(arrays[m], frozen[m])
    report["frozen_split_significance"] = {k: frozen[k] for k in ("mcnemar_neg_p_max", "mcnemar_neg_min_achievable_p",
                                                                  "mcnemar_pos_p_max", "mcnemar_pos_min_achievable_p")}
    report["share_of_draws_mcnemar_neg_p_max_below_0_05"] = float((arrays["mcnemar_neg_p_max"] < 0.05).mean())
    report["share_of_draws_mcnemar_pos_p_max_below_0_05"] = float((arrays["mcnemar_pos_p_max"] < 0.05).mean())
    report["frozen_split_counts"] = frozen["_counts"]
    return report, arrays
