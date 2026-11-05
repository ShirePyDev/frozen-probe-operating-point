"""Does post-update calibration collapse hit ADVERSARIAL prompts harder?

Why this matters
----------------
Everything measured so far pools all prompts together. But a guardrail exists to
stop adversarial input, so the security-relevant question is not "how much does
calibration drift on average" but "does it drift MORE on the prompts an attacker
would actually send". If adversarial prompts degrade disproportionately, then a
routine model update quietly weakens the defence precisely where it is load
bearing, and an aggregate measurement would hide that.

WildGuardMix is the only axis here that can answer it: its rows carry an
`adversarial` flag, which freeze_splits.py copies verbatim into the frozen
records (D-051), so the split can be made without re-deriving anything.

Design decision that matters
----------------------------
The decision threshold is calibrated ONCE on the full pre-update calibration set
and is GLOBAL, exactly as deployed - an operator does not run a separate
threshold per prompt type, and usually cannot tell the types apart at inference.
The stratification is applied only to the MEASUREMENT. Calibrating per stratum
would answer a different and much less useful question.
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
    read_jsonl,
    sha256_file,
    write_results,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "hidden"
FROZEN = ROOT / "data" / "frozen"


def load_cache(repo: str, rev: str, axis: str, part: str):
    p = CACHE / f"{repo.replace('/', '__')}@{rev[:12]}__{axis}__{part}__mean.npz"
    if not p.exists():
        raise FileNotFoundError(f"missing cache {p.name}")
    return np.load(p, allow_pickle=False), p


def flag_map(axis: str, part: str, field: str) -> dict[str, str]:
    rows = read_jsonl(FROZEN / f"{axis}__{part}.jsonl")
    missing = [r for r in rows if field not in r]
    if missing:
        raise RuntimeError(
            f"{len(missing)} frozen records lack '{field}'; re-freeze with extra_fields")
    return {r["item_id"]: str(r[field]) for r in rows}


def fit_probe(x, y, seed):
    from sklearn.linear_model import LogisticRegression

    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit((x - mu) / sd, y)
    return clf, mu, sd


# The frozen-threshold rule is defined once, in src/paired.py (2026-09-25). This file's
# local copy was verified to give identical thresholds on 12,000 random and tie-heavy
# cases before it was replaced.
from paired import thr_at_fpr  # noqa: E402


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Adversarial vs benign stratified calibration drift")
    ap.add_argument("--pre-repo", required=True)
    ap.add_argument("--pre-rev", required=True)
    ap.add_argument("--post-repo", required=True)
    ap.add_argument("--post-rev", required=True)
    ap.add_argument("--update-kind", required=True)
    ap.add_argument("--axis", default="safety_wildguard")
    ap.add_argument("--field", default="adversarial")
    ap.add_argument("--layers", nargs="*", type=int, default=[8, 16, 24, 32])
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hashes, data = {}, {}
    for side, repo, rev in (("pre", args.pre_repo, args.pre_rev),
                            ("post", args.post_repo, args.post_rev)):
        data[side] = {}
        for part in ("probe_train", "threshold_calib", "eval"):
            z, p = load_cache(repo, rev, args.axis, part)
            data[side][part] = {k: z[k] for k in z.files}
            hashes[f"{side}/{part}"] = sha256_file(p)
    for part in ("probe_train", "threshold_calib", "eval"):
        if not np.array_equal(data["pre"][part]["item_ids"], data["post"][part]["item_ids"]):
            raise RuntimeError(f"item_ids differ for {part}")
        bad = (data["pre"][part]["nonfinite"].astype(bool)
               | data["post"][part]["nonfinite"].astype(bool))
        if bad.any():
            print(f"dropping {int(bad.sum())} non-finite from {part}")
            for side in ("pre", "post"):
                for k, arr in list(data[side][part].items()):
                    if hasattr(arr, "shape") and arr.shape and arr.shape[0] == bad.shape[0]:
                        data[side][part][k] = arr[~bad]

    fmap = flag_map(args.axis, "eval", args.field)
    ev_ids = [i.decode() if isinstance(i, bytes) else str(i)
              for i in data["pre"]["eval"]["item_ids"]]
    strata_vals = np.array([fmap.get(i, "UNKNOWN") for i in ev_ids])
    y_ev = data["pre"]["eval"]["labels"].astype(bool)
    y_tr = data["pre"]["probe_train"]["labels"].astype(bool)
    y_cal = data["pre"]["threshold_calib"]["labels"].astype(bool)
    n_tr = len(y_tr)

    levels = sorted(set(strata_vals.tolist()))
    print(f"strata on '{args.field}': " + ", ".join(
        f"{lv}={int((strata_vals == lv).sum())}" for lv in levels))
    if "UNKNOWN" in levels:
        print("  WARNING: some eval items had no flag in the frozen split")

    rows: list[dict[str, Any]] = []
    for li in args.layers:
        k = f"layer_{li}"
        if k not in data["pre"]["probe_train"] or k not in data["post"]["probe_train"]:
            print(f"  layer {li}: not in cache, skipped")
            continue
        per: dict[str, dict[str, list[float]]] = {
            lv: {"fpr_pre": [], "fpr_post": [], "tpr_pre": [], "tpr_post": [],
                 "auroc_post": []} for lv in levels}
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n_tr, size=int(args.subsample * n_tr), replace=False)
            clf, mu, sd = fit_probe(data["pre"]["probe_train"][k][idx], y_tr[idx], seed)
            w, b = clf.coef_.ravel(), float(clf.intercept_[0])
            s = lambda x: (x - mu) / sd @ w + b  # noqa: E731

            # ONE global threshold, calibrated on the full pre-update calib set.
            t = thr_at_fpr(s(data["pre"]["threshold_calib"][k])[~y_cal], args.target_fpr)
            ev_pre, ev_post = s(data["pre"]["eval"][k]), s(data["post"]["eval"][k])

            for lv in levels:
                m = strata_vals == lv
                neg, pos = m & ~y_ev, m & y_ev
                if neg.sum():
                    per[lv]["fpr_pre"].append(float((ev_pre[neg] >= t).mean()))
                    per[lv]["fpr_post"].append(float((ev_post[neg] >= t).mean()))
                if pos.sum():
                    per[lv]["tpr_pre"].append(float((ev_pre[pos] >= t).mean()))
                    per[lv]["tpr_post"].append(float((ev_post[pos] >= t).mean()))
                if neg.sum() and pos.sum():
                    a = M.auroc(ev_post[pos], ev_post[neg]).value
                    if a is not None:
                        per[lv]["auroc_post"].append(a)

        row: dict[str, Any] = {"layer": li, "strata": {}}
        for lv in levels:
            d = per[lv]
            row["strata"][lv] = {
                "n_total": int((strata_vals == lv).sum()),
                "n_pos": int(((strata_vals == lv) & y_ev).sum()),
                "n_neg": int(((strata_vals == lv) & ~y_ev).sum()),
                **{f"{name}_mean": (float(np.mean(v)) if v else None) for name, v in d.items()},
                **{f"{name}_sd": (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
                   for name, v in d.items()},
            }
        rows.append(row)

    print()
    print(f"  {'layer':>5} {'stratum':>10} {'n_neg':>7} {'n_pos':>6} | "
          f"{'FPR pre':>8} {'FPR post':>9} {'x target':>9} | {'TPR pre':>8} {'TPR post':>9}")
    print("  " + "-" * 86)
    for r in rows:
        for lv in levels:
            st = r["strata"][lv]
            fpo = st["fpr_post_mean"]
            ratio = (fpo / args.target_fpr) if fpo is not None else float("nan")
            print(f"  {r['layer']:>5} {lv:>10} {st['n_neg']:>7} {st['n_pos']:>6} | "
                  f"{st['fpr_pre_mean']:>8.4f} {fpo:>9.4f} {ratio:>8.1f}x | "
                  f"{st['tpr_pre_mean']:>8.4f} {st['tpr_post_mean']:>9.4f}")

    payload = {
        **provenance_block("adversarial_strat"),
        "config": vars(args), "config_hash": canonical_hash(vars(args)),
        "update_kind": args.update_kind, "input_cache_sha256": hashes,
        "frozen_split_sha256": sha256_file(FROZEN / f"{args.axis}__eval.jsonl"),
        "levels": levels, "rows": rows,
    }
    digest = write_results(args.out, payload)
    print(f"\nWritten: {args.out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
