"""Two alignment repairs the ridge map is not: norm-preserving, and full-covariance (D-093).

R10 in the ledger said a variance-preserving map had not been tested, which is the
obvious objection to D-078/D-082 ("ridge alignment calibrates worse than doing
nothing"): least-squares shrinks, so maybe only OUR map fails. This script answers it.

  procrustes_paired   orthogonal Procrustes fitted post -> pre on PAIRED unlabelled
                      activations. An orthogonal matrix preserves norms, so this map
                      cannot shrink. If shrinkage is the cause, this should calibrate
                      better than ridge.
  coral_unpaired      whiten with the post covariance, colour with the pre covariance
                      (Ledoit-Wolf shrinkage, since n < d), then add the pre mean.
                      Needs NO paired items - only stored pre-side statistics - and
                      matches first AND second moments including off-diagonals. It is
                      the strict generalisation of restandardise, so it tests whether
                      the off-diagonal part of the covariance carries any damage.

Baselines recomputed here so every arm meets the same probe, threshold and splits:
frozen, restandardise, paired_align (ridge, as in moment_decomposition), refit_oracle.

The maps are fitted ONCE per layer on unlabelled probe_train activations, never on
the probe's seed subsample and never on labels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
from provenance import (  # noqa: E402
    canonical_hash,
    ensure_utf8_stdout,
    provenance_block,
    resolve_cache,
    sha256_file,
    write_results,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "hidden"
PARTS = ("probe_train", "threshold_calib", "eval")
ARMS = ("frozen", "restandardise", "paired_align", "procrustes_paired", "coral_unpaired",
        "refit_oracle")


def load(repo: str, rev: str, axis: str, part: str):
    p = resolve_cache(f"{repo.replace('/', '__')}@{rev[:12]}__{axis}__{part}__mean.npz", CACHE)
    if not p.exists():
        raise FileNotFoundError(f"missing cache {p.name}")
    return np.load(p, allow_pickle=False), p


def fit_probe(x, y, seed):
    from sklearn.linear_model import LogisticRegression

    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit((x - mu) / sd, y)
    return clf.coef_.ravel(), float(clf.intercept_[0]), mu, sd


# The frozen-threshold rule is defined once, in src/paired.py (2026-09-25). This file's
# local copy was verified to give identical thresholds on 12,000 random and tie-heavy
# cases before it was replaced.
from paired import thr_at_fpr  # noqa: E402


def fit_ridge(x_src, x_tgt, xv_src, xv_tgt):
    """Ridge affine map, identical in form to moment_decomposition.fit_alignment."""
    ms, mt = x_src.mean(0), x_tgt.mean(0)
    xs, xt = x_src - ms, x_tgt - mt
    gram, cross = xs.T @ xs, xs.T @ xt
    scale = float(np.trace(gram)) / gram.shape[0]
    best = None
    for lam_rel in (1e-3, 1e-2, 1e-1, 1.0):
        a = np.linalg.solve(gram + lam_rel * scale * np.eye(gram.shape[0]), cross)
        mse = float((((xv_src - ms) @ a + mt - xv_tgt) ** 2).mean())
        if best is None or mse < best[0]:
            best = (mse, lam_rel, a)
    mse, lam_rel, a = best
    return (lambda x: (x - ms) @ a + mt), {"lambda_rel_chosen": lam_rel, "val_mse": mse}


def fit_procrustes(x_src, x_tgt):
    """Orthogonal Procrustes: argmin over orthogonal R of ||(src-ms) R - (tgt-mt)||_F.

    R = U V^T from the SVD of (src-ms)^T (tgt-mt). Orthogonal, so it preserves the
    norm of every centred vector and cannot shrink the representation.
    """
    ms, mt = x_src.mean(0), x_tgt.mean(0)
    u, _, vt = np.linalg.svd((x_src - ms).T @ (x_tgt - mt), full_matrices=False)
    r = u @ vt
    orth_err = float(np.abs(r @ r.T - np.eye(r.shape[0])).max())
    return (lambda x: (x - ms) @ r + mt), {"max_orthogonality_error": orth_err}


def _shrunk_cov(x, eps=1e-6):
    """Ledoit-Wolf shrinkage towards a scaled identity; n < d here, so the empirical
    covariance is singular and cannot be inverted. sklearn's implementation is used
    because the closed form needs the full fourth-moment term, which is far too slow
    written as a Python loop at d = 4096."""
    from sklearn.covariance import ledoit_wolf

    cov, lam = ledoit_wolf(x, assume_centered=False)
    return cov + eps * np.eye(cov.shape[0]), float(lam)


def _inv_sqrt(cov):
    w, v = np.linalg.eigh(cov)
    w = np.maximum(w, 1e-10)
    return (v / np.sqrt(w)) @ v.T


def _sqrt(cov):
    w, v = np.linalg.eigh(cov)
    w = np.maximum(w, 1e-10)
    return (v * np.sqrt(w)) @ v.T


def fit_coral(x_post, x_pre):
    """Whiten with the post covariance, colour with the pre covariance. Unpaired."""
    cp, lam_p = _shrunk_cov(x_post)
    cq, lam_q = _shrunk_cov(x_pre)
    a = _inv_sqrt(cp) @ _sqrt(cq)
    mpost, mpre = x_post.mean(0), x_pre.mean(0)
    return (lambda x: (x - mpost) @ a + mpre), {"shrinkage_post": lam_p, "shrinkage_pre": lam_q}


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Procrustes and CORAL alignment repairs (D-093)")
    ap.add_argument("--pre-repo", required=True)
    ap.add_argument("--pre-rev", required=True)
    ap.add_argument("--post-repo", required=True)
    ap.add_argument("--post-rev", required=True)
    ap.add_argument("--update-kind", required=True)
    ap.add_argument("--axis", default="safety")
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hashes, data = {}, {}
    for side, repo, rev in (("pre", args.pre_repo, args.pre_rev), ("post", args.post_repo, args.post_rev)):
        data[side] = {}
        for part in PARTS:
            z, p = load(repo, rev, args.axis, part)
            data[side][part] = {k: z[k] for k in z.files}
            hashes[f"{side}/{part}"] = sha256_file(p)
    for part in PARTS:
        if not np.array_equal(data["pre"][part]["item_ids"], data["post"][part]["item_ids"]):
            raise RuntimeError(f"item_ids differ for {part}")
        bad = data["pre"][part]["nonfinite"].astype(bool) | data["post"][part]["nonfinite"].astype(bool)
        if bad.any():
            print(f"dropping {int(bad.sum())} non-finite from {part}")
            for side in ("pre", "post"):
                for k, arr in list(data[side][part].items()):
                    if hasattr(arr, "shape") and arr.shape and arr.shape[0] == bad.shape[0]:
                        data[side][part][k] = arr[~bad]

    P, Q = data["pre"], data["post"]
    common = sorted([k for k in P["probe_train"] if k.startswith("layer_") and k in Q["probe_train"]],
                    key=lambda s: int(s.split("_")[1]))
    if args.layers:
        want = {f"layer_{i}" for i in args.layers}
        common = [k for k in common if k in want]
    y_tr = P["probe_train"]["labels"].astype(bool)
    y_cal = P["threshold_calib"]["labels"].astype(bool)
    y_ev = P["eval"]["labels"].astype(bool)
    n_tr, tgt = len(y_tr), args.target_fpr

    rows: list[dict[str, Any]] = []
    for k in common:
        li = int(k.split("_")[1])
        if P["probe_train"][k].shape[1] != Q["probe_train"][k].shape[1]:
            print(f"  layer {li}: width mismatch, skipped")
            continue
        f64 = lambda a: a.astype(np.float64)  # noqa: E731
        ridge, ridge_info = fit_ridge(f64(Q["probe_train"][k]), f64(P["probe_train"][k]),
                                      f64(Q["threshold_calib"][k]), f64(P["threshold_calib"][k]))
        proc, proc_info = fit_procrustes(f64(Q["probe_train"][k]), f64(P["probe_train"][k]))
        coral, coral_info = fit_coral(f64(Q["probe_train"][k]), f64(P["probe_train"][k]))
        mapped = {"paired_align": ridge, "procrustes_paired": proc, "coral_unpaired": coral}
        MQ = {name: {part: fn(f64(Q[part][k])) for part in ("threshold_calib", "eval")}
              for name, fn in mapped.items()}

        acc = {a: {"fpr": [], "auroc": [], "tpr": []} for a in ARMS}
        std_ratio = {a: [] for a in mapped}
        auroc_pre = []
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n_tr, size=int(args.subsample * n_tr), replace=False)
            w, b, mu_a, sd_a = fit_probe(P["probe_train"][k][idx], y_tr[idx], seed)
            sc = lambda x, mu, sd: (x - mu) / sd @ w + b  # noqa: E731
            cal_pre = sc(P["threshold_calib"][k], mu_a, sd_a)
            t = thr_at_fpr(cal_pre[~y_cal], tgt)
            ev_pre = sc(P["eval"][k], mu_a, sd_a)
            auroc_pre.append(M.auroc(ev_pre[y_ev], ev_pre[~y_ev]).value)

            xb_tr = Q["probe_train"][k][idx]
            mu_b, sd_b = xb_tr.mean(0), np.where(xb_tr.std(0) < 1e-8, 1.0, xb_tr.std(0))
            post = {"frozen": sc(Q["eval"][k], mu_a, sd_a),
                    "restandardise": sc(Q["eval"][k], mu_b, sd_b)}
            for name in mapped:
                s_ev = sc(MQ[name]["eval"], mu_a, sd_a)
                post[name] = s_ev
                std_ratio[name].append(float(s_ev.std() / (ev_pre.std() or 1.0)))
            for arm, s in post.items():
                acc[arm]["fpr"].append(float((s[~y_ev] >= t).mean()))
                acc[arm]["tpr"].append(float((s[y_ev] >= t).mean()))
                acc[arm]["auroc"].append(M.auroc(s[y_ev], s[~y_ev]).value)

            w2, b2, mu2, sd2 = fit_probe(xb_tr, y_tr[idx], seed)
            s2c = (Q["threshold_calib"][k] - mu2) / sd2 @ w2 + b2
            t2 = thr_at_fpr(s2c[~y_cal], tgt)
            s2 = (Q["eval"][k] - mu2) / sd2 @ w2 + b2
            acc["refit_oracle"]["fpr"].append(float((s2[~y_ev] >= t2).mean()))
            acc["refit_oracle"]["tpr"].append(float((s2[y_ev] >= t2).mean()))
            acc["refit_oracle"]["auroc"].append(M.auroc(s2[y_ev], s2[~y_ev]).value)

        n_neg_ev = int((~y_ev).sum())
        row: dict[str, Any] = {"layer": li, "auroc_pre": float(np.mean(auroc_pre)),
                               "n_neg_eval": n_neg_ev, "ridge": ridge_info,
                               "procrustes": proc_info, "coral": coral_info,
                               "score_std_ratio": {a: float(np.mean(v)) for a, v in std_ratio.items()},
                               "arms": {}}
        for arm in ARMS:
            f = np.asarray(acc[arm]["fpr"])
            m = float(f.mean())
            # F-034: an FPR of 0 is CENSORED at the eval resolution, not 1e-9.
            row["arms"][arm] = {
                "fpr_mean": m, "fpr_sd": float(f.std(ddof=1)) if f.size > 1 else 0.0,
                "censored_zero_fpr": bool(m == 0),
                "log_fpr_err": float(abs(np.log(max(m, 1.0 / n_neg_ev) / tgt))),
                "auroc_mean": float(np.mean(acc[arm]["auroc"])),
                "tpr_mean": float(np.mean(acc[arm]["tpr"]))}
        rows.append(row)
        print(f"  layer {li:>3}  preAUROC {row['auroc_pre']:.4f}  score-std ratio "
              + "  ".join(f"{a} {v:.3f}" for a, v in row["score_std_ratio"].items()), flush=True)
        for arm in ARMS:
            r = row["arms"][arm]
            print(f"      {arm:<18} FPR {r['fpr_mean']:.4f}  err {r['log_fpr_err']:.3f}  "
                  f"AUROC {r['auroc_mean']:.4f}" + ("  [CENSORED]" if r["censored_zero_fpr"] else ""))

    summary = {a: {"mean_log_fpr_err": float(np.mean([r["arms"][a]["log_fpr_err"] for r in rows])),
                   "mean_auroc_drop_vs_pre": float(np.mean([r["auroc_pre"] - r["arms"][a]["auroc_mean"]
                                                            for r in rows])),
                   "n_censored_cells": int(sum(r["arms"][a]["censored_zero_fpr"] for r in rows))}
               for a in ARMS}
    print("\n  mean |log(FPR/target)| (resolution-floored), AUROC drop, censored cells:")
    for a in ARMS:
        s = summary[a]
        print(f"    {a:<18} {s['mean_log_fpr_err']:.4f}   {s['mean_auroc_drop_vs_pre']:+.4f}   "
              f"{s['n_censored_cells']}")

    payload = {**provenance_block("align_variants"), "preregistration": "D-093",
               "config": vars(args), "config_hash": canonical_hash(vars(args)),
               "update_kind": args.update_kind, "input_cache_sha256": hashes,
               "summary": summary, "rows": rows}
    digest = write_results(args.out, payload)
    print(f"\nWritten: {args.out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
