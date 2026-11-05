"""E14 - does update robustness depend on probe DEPTH?

Motivation
----------
Existing work selects a probe layer for ACCURACY, and reports that linear
separability grows with depth (arXiv:2604.13386 and the layer-probing
literature). Separately, arXiv:2606.15980 shows fine-tuning-style updates stale
frozen probes. Nobody has joined these: **if the accuracy-optimal depth is also
the least update-robust depth, then standard practice selects the most fragile
layer available.**

Our own data hinted at it. Reading results/repdist_qwen.json, CKA across
Qwen2.5 -> Qwen3 is 0.963 / 0.966 / 0.965 at 25 / 50 / 75 percent depth but
0.482 at the final layer - drift is ~15x larger at the output end. This script
tests that properly, on a dense layer grid, with a functional measure rather
than CKA alone.

What is measured, per layer
---------------------------
  auroc_pre_refit      probe fit on A's probe_train, scored on A's eval
                       -> accuracy at that depth (the usual selection criterion)
  auroc_post_refit     probe refit on B's probe_train, scored on B's eval
                       -> accuracy after the update, diligent-operator condition
  auroc_frozen         probe fit on A, applied unchanged to B's eval
                       -> no-maintenance condition
  drop_frozen          auroc_pre_refit - auroc_frozen
                       -> FUNCTIONAL update fragility at that depth
  cka                  representational distance at that depth
  frozen_threshold_*   the D-039 phenomenon as a function of depth: take the
                       threshold calibrated at target FPR on A, apply it to B,
                       and record what FPR and flag rate it actually delivers

Stochasticity (CLAUDE.md rule 6, and the lesson of D-036)
---------------------------------------------------------
A logistic probe fit by lbfgs is deterministic given its data, so "five seeds"
would be five identical numbers. The declared stochasticity source here is a
per-seed SUBSAMPLE of probe_train. Between-seed spread is reported as the
headline uncertainty, because D-036 recorded a case where a per-seed bootstrap
CI excluded zero while the sign flipped across seeds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
from repdist import linear_cka  # noqa: E402
from provenance import (  # noqa: E402
    resolve_cache,
    canonical_hash,
    ensure_utf8_stdout,
    provenance_block,
    sha256_file,
    write_results,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "hidden"


def load(repo: str, rev: str, axis: str, part: str, chat: bool = False):
    """Load one cached split. `chat` selects the chat-template arm of the same
    checkpoint, which is a SEPARATE measurement stored under a __chat suffix."""
    suffix = "mean__chat" if chat else "mean"
    p = resolve_cache(f"{repo.replace('/', '__')}@{rev[:12]}__{axis}__{part}__{suffix}.npz", CACHE)
    if not p.exists():
        raise FileNotFoundError(f"missing cache {p.name}")
    return np.load(p, allow_pickle=False), p


def fit_probe(x, y, seed):
    from sklearn.linear_model import LogisticRegression

    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit((x - mu) / sd, y)
    return clf, mu, sd


def score(clf, mu, sd, x):
    return clf.decision_function((x - mu) / sd)


# The frozen-threshold rule is defined once, in src/paired.py (2026-09-25). This file's
# local copy was verified to give identical thresholds on 12,000 random and tie-heavy
# cases before it was replaced.
from paired import thr_at_fpr  # noqa: E402


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="E14 depth vs update-robustness curve")
    ap.add_argument("--pre-repo", required=True)
    ap.add_argument("--pre-rev", required=True)
    ap.add_argument("--post-repo", required=True)
    ap.add_argument("--post-rev", required=True)
    ap.add_argument("--post-chat-template", action="store_true",
                    help="read the post side's CHAT-TEMPLATE cache arm instead of the raw one "
                         "(confound control: is the shift instruction tuning or input format?)")
    ap.add_argument("--update-kind", required=True,
                    help="label for this perturbation, e.g. version_increment or instruction_tuning")
    ap.add_argument("--axis", default="safety")
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--subsample", type=float, default=0.8,
                    help="fraction of probe_train used per seed (the stochasticity source)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hashes: dict[str, str] = {}
    data: dict[str, dict[str, Any]] = {}
    for side, repo, rev in (("pre", args.pre_repo, args.pre_rev),
                            ("post", args.post_repo, args.post_rev)):
        data[side] = {}
        for part in ("probe_train", "threshold_calib", "eval"):
            z, p = load(repo, rev, args.axis, part,
                        chat=(side == "post" and args.post_chat_template))
            data[side][part] = {k: z[k] for k in z.files}
            hashes[f"{side}/{part}"] = sha256_file(p)

    # Lesson F-012 / rule 2: verify alignment rather than assuming it.
    for part in ("probe_train", "threshold_calib", "eval"):
        if not np.array_equal(data["pre"][part]["item_ids"], data["post"][part]["item_ids"]):
            raise RuntimeError(f"item_ids differ between sides for {part}")

    # Lesson from Qwen3: non-finite states are zero-filled by the extractor; a
    # zero vector is not a measurement, so those items are dropped and counted.
    dropped: dict[str, int] = {}
    for part in ("probe_train", "threshold_calib", "eval"):
        bad = (data["pre"][part]["nonfinite"].astype(bool)
               | data["post"][part]["nonfinite"].astype(bool))
        dropped[part] = int(bad.sum())
        if bad.any():
            keep = ~bad
            for side in ("pre", "post"):
                for k, arr in list(data[side][part].items()):
                    if hasattr(arr, "shape") and arr.shape and arr.shape[0] == bad.shape[0]:
                        data[side][part][k] = arr[keep]
    if any(dropped.values()):
        print(f"dropped non-finite items: {dropped}")

    layer_keys = sorted(
        [k for k in data["pre"]["probe_train"] if k.startswith("layer_")],
        key=lambda s: int(s.split("_")[1]),
    )
    common = [k for k in layer_keys if k in data["post"]["probe_train"]]
    n_layers_pre = max(int(k.split("_")[1]) for k in common)
    print(f"layers compared: {len(common)}  (max index {n_layers_pre})")

    y_tr = data["pre"]["probe_train"]["labels"].astype(bool)
    y_cal = data["pre"]["threshold_calib"]["labels"].astype(bool)
    y_ev = data["pre"]["eval"]["labels"].astype(bool)
    n_tr = len(y_tr)

    rows: list[dict[str, Any]] = []
    for k in common:
        li = int(k.split("_")[1])
        per_seed: dict[str, list[float]] = {
            "auroc_pre_refit": [], "auroc_post_refit": [], "auroc_frozen": [],
            "drop_frozen": [], "post_fpr_at_frozen_thr": [],
            "post_flagrate_harmful": [], "pre_flagrate_harmful": [],
        }
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n_tr, size=int(args.subsample * n_tr), replace=False)

            xa_tr = data["pre"]["probe_train"][k][idx]
            xb_tr = data["post"]["probe_train"][k][idx]
            ya = y_tr[idx]

            clf_a, mu_a, sd_a = fit_probe(xa_tr, ya, seed)
            clf_b, mu_b, sd_b = fit_probe(xb_tr, ya, seed)

            sa_ev = score(clf_a, mu_a, sd_a, data["pre"]["eval"][k])
            sb_ev_refit = score(clf_b, mu_b, sd_b, data["post"]["eval"][k])
            sb_ev_frozen = score(clf_a, mu_a, sd_a, data["post"]["eval"][k])

            au_a = M.auroc(sa_ev[y_ev], sa_ev[~y_ev]).value
            au_b_refit = M.auroc(sb_ev_refit[y_ev], sb_ev_refit[~y_ev]).value
            au_b_frozen = M.auroc(sb_ev_frozen[y_ev], sb_ev_frozen[~y_ev]).value

            # Threshold transfer (the D-039 phenomenon) as a function of depth.
            sa_cal = score(clf_a, mu_a, sd_a, data["pre"]["threshold_calib"][k])
            sb_cal_frozen = score(clf_a, mu_a, sd_a, data["post"]["threshold_calib"][k])
            t = thr_at_fpr(sa_cal[~y_cal], args.target_fpr)

            per_seed["auroc_pre_refit"].append(au_a)
            per_seed["auroc_post_refit"].append(au_b_refit)
            per_seed["auroc_frozen"].append(au_b_frozen)
            per_seed["drop_frozen"].append(au_a - au_b_frozen)
            per_seed["post_fpr_at_frozen_thr"].append(float((sb_cal_frozen[~y_cal] >= t).mean()))
            per_seed["pre_flagrate_harmful"].append(float((sa_ev[y_ev] >= t).mean()))
            per_seed["post_flagrate_harmful"].append(float((sb_ev_frozen[y_ev] >= t).mean()))

        row: dict[str, Any] = {
            "layer": li,
            "relative_depth": li / n_layers_pre if n_layers_pre else None,
            "cka": linear_cka(
                data["pre"]["eval"][k].astype(np.float64),
                data["post"]["eval"][k].astype(np.float64),
            ),
        }
        for name, vals in per_seed.items():
            arr = np.asarray([v for v in vals if v is not None], dtype=float)
            row[f"{name}_mean"] = float(arr.mean()) if arr.size else None
            row[f"{name}_sd"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        rows.append(row)
        print(f"  layer {li:>3} (d={row['relative_depth']:.2f})  "
              f"CKA={row['cka']:.5f}  AUROC_pre={row['auroc_pre_refit_mean']:.4f}  "
              f"frozen={row['auroc_frozen_mean']:.4f}  drop={row['drop_frozen_mean']:+.4f}"
              f" (sd {row['drop_frozen_sd']:.4f})  postFPR@frozenThr="
              f"{row['post_fpr_at_frozen_thr_mean']:.4f}")

    payload = {
        **provenance_block("depth_curve"),
        "config": vars(args),
        "config_hash": canonical_hash(vars(args)),
        "update_kind": args.update_kind,
        "input_cache_sha256": hashes,
        "dropped_nonfinite": dropped,
        "n_eval": int(len(y_ev)),
        "stochasticity_source": f"probe_train subsample at {args.subsample}, per seed",
        "rows": rows,
    }
    digest = write_results(args.out, payload)

    ok = [r for r in rows if r["auroc_pre_refit_mean"] is not None]
    if ok:
        best_acc = max(ok, key=lambda r: r["auroc_pre_refit_mean"])
        best_rob = min(ok, key=lambda r: abs(r["drop_frozen_mean"]))
        print("\n" + "=" * 70)
        print(f"most ACCURATE layer : {best_acc['layer']} (d={best_acc['relative_depth']:.2f})  "
              f"AUROC={best_acc['auroc_pre_refit_mean']:.4f}  drop={best_acc['drop_frozen_mean']:+.4f}")
        print(f"most ROBUST layer   : {best_rob['layer']} (d={best_rob['relative_depth']:.2f})  "
              f"AUROC={best_rob['auroc_pre_refit_mean']:.4f}  drop={best_rob['drop_frozen_mean']:+.4f}")
        print(f"same layer? {best_acc['layer'] == best_rob['layer']}")
        print("=" * 70)
    print(f"\nWritten: {args.out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
