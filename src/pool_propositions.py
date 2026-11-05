"""The closed-form propositions of the 4d pool study (D-221), as functions, so that 4d can print each prediction
beside what it measures and tests/test_pool_propositions_d221.py can check each one by simulation first.

Notation (D-221): pi is the harmful prevalence of a stream; alpha the target FPR; F0bar and F1bar the benign and
harmful score survival functions at a threshold t; m0, m1 the class-mean scores and d = m1 - m0.
  P1 flag rate            f = pi*TPR + (1-pi)*FPR;  df = (TPR - FPR)*dpi + pi*dTPR + (1-pi)*dFPR (to first order)
  P2 pooled-mean shift    delta = (pi_ref - pi_pre)*d; benign FPR at the frozen threshold F0bar(t + delta)
  P3 quantile drift       realised FPR of the k-th largest of n benign scores, k = ceil(n*alpha), is Beta(k, n-k+1);
                          with harmful contamination eps: benign FPR = (alpha - eps*F1bar(t))/(1 - eps) <= alpha;
                          a later benign location drift by c: FPR = F0bar(t - c)
  P4 poisoning budget     m items scoring above s raise the quantile threshold to at least s once
                          m > n_b*(alpha - F0bar(s))/(1 - alpha); silencing needs about n_b*alpha/(1 - alpha);
                          for the pooled-mean repair, m items offset by c shift the mean by m*c/(n + m)
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import norm


# P1
def flag_rate(pi: float, tpr: float, fpr: float) -> float:
    return pi * tpr + (1.0 - pi) * fpr


def flag_rate_change(pi: float, tpr: float, fpr: float, dpi: float, dtpr: float, dfpr: float) -> float:
    """First-order change of the flag rate (exact when only one of the three moves)."""
    return (tpr - fpr) * dpi + pi * dtpr + (1.0 - pi) * dfpr


# P2
def pooled_mean_shift(pi_ref: float, pi_pre: float, d: float) -> float:
    """The mis-centring, in probe score units, of a restandardisation whose mean is estimated at pi_ref while the
    pre-update statistics came from pi_pre (sign: positive when the reference pool holds more harmful items)."""
    return (pi_ref - pi_pre) * d


def gaussian_fpr_after_shift(t: float, delta: float, m0: float, s0: float) -> float:
    return float(norm.sf((t + delta - m0) / s0))


# P3
def quantile_k(n: int, alpha: float) -> int:
    return int(math.ceil(n * alpha))


def quantile_fpr_moments(n: int, alpha: float) -> tuple[float, float]:
    """Mean and sd of the realised FPR of the k-th largest of n clean benign scores (continuous scores)."""
    k = quantile_k(n, alpha)
    b = beta_dist(k, n - k + 1)
    return float(b.mean()), float(b.std())


def contaminated_benign_fpr(alpha: float, eps: float, f1bar_t: float) -> float:
    return (alpha - eps * f1bar_t) / (1.0 - eps)


def gaussian_contaminated_threshold(alpha: float, eps: float, m0: float, s0: float, m1: float, s1: float) -> float:
    """The t with (1-eps)*F0bar(t) + eps*F1bar(t) = alpha for Gaussian classes (bisection)."""
    lo, hi = m0 - 20 * s0, max(m0, m1) + 20 * max(s0, s1)
    f = lambda t: (1 - eps) * norm.sf((t - m0) / s0) + eps * norm.sf((t - m1) / s1) - alpha  # noqa: E731
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


# P4
def poisoning_min_items(n_b: int, alpha: float, f0bar_s: float) -> float:
    """The number of injected items above s beyond which the quantile threshold is at least s."""
    return n_b * (alpha - f0bar_s) / (1.0 - alpha)


def silencing_items(n_b: int, alpha: float) -> float:
    return n_b * alpha / (1.0 - alpha)


def pooled_mean_offset(n: int, m: int, c: np.ndarray | float) -> np.ndarray | float:
    return m * np.asarray(c) / (n + m)
