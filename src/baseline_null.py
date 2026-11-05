"""Is the pre-update baseline FPR at the selected probe what the split sampling predicts?

Check 1 on 4a (author, 2026-09-25): the frozen threshold is set at 1% on the 1000
threshold_calib negatives, yet on the 2000 eval negatives the pre-update model realises
6.6/2000 = 0.33% in the LoRA cell. A realised rate below target is expected from finite
calibration sets, but how far below is informative. This places each cell's observed
baseline in its own no-update null, the D-083 resplit construction:

  pool the calibration and eval negatives' scores (1000 + 2000), re-partition at the
  original sizes 1000 times - ONE partition per draw, shared by all five seeds, as the
  frozen split is - set each seed's 1% threshold on the calibration part, measure the
  realised FPR on the eval part, floor at 1/n_eval (F-034), mean over seeds.

The observed value is the same statistic on the frozen partition. Reported: the observed
value, the null mean and 2.5/50/97.5 percentiles, the observed value's mid-rank
percentile, and whether it lies outside the central 95% band.

Cells: the 4a LoRA cell (A100 Mistral base, its own D-117 selection) and the nine released
update pairs of UPDATE_PAIRS_REPAIR at their D-117 selected (C, layer). Probe exactly as
D-152 P1: five seeds, D-117 80% split in permutation order, per-seed threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
from provenance import ensure_utf8_stdout, provenance_block, resolve_cache, sha256_file, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "hidden"
RESULTS = ROOT / "results"
CODE_FILES = ["src/baseline_null.py", "src/paired.py", "src/populations.py", "src/provenance.py",
              "src/superseded_provenance.py"]


def load(repo: str, rev: str, axis: str, part: str):
    p = resolve_cache(f"{repo.replace('/', '__')}@{rev[:12]}__{axis}__{part}__mean.npz", CACHE)
    if not p.exists():
        raise FileNotFoundError(f"missing cache {p.name}")
    return np.load(p, allow_pickle=False), p


def selections(path: Path) -> dict[str, dict[str, Any]]:
    d = json.load(open(path, encoding="utf-8"))
    out = {}
    for m in d["models"]:
        arm = m["selected"]["arm"]
        if not arm.startswith("C="):
            raise SystemExit(f"REFUSING: {m['pair']} selection is {arm!r}, not a logistic C")
        out[m["pair"]] = {"C": float(arm.split("=")[1]), "layer": int(m["selected"]["layer"]),
                          "input_sha": {k.split(":", 1)[1]: v for k, v in d["input_cache_sha256"].items()
                                        if k.startswith(m["pair"] + ":")},
                          "seeds": d["config"]["seeds"], "holdout": d["config"]["holdout"]}
    return out


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="No-update null of the pre-update baseline FPR")
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--null-seed", type=int, default=20260925)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--target-fpr", type=float, default=P.TARGET_FPR)
    ap.add_argument("--out", default=str(RESULTS / "baseline_null_selected.json"))
    args = ap.parse_args()
    out_path = Path(args.out).resolve()
    if out_path.exists():
        raise SystemExit(f"REFUSING: {out_path} exists")
    code0 = P.code_block(CODE_FILES)

    cells = [("4a LoRA base (A100)", "repair_loraA100_mistral_s25", RESULTS / "regularisation_sweep_lora_a100.json")]
    cells += [(stem, stem, RESULTS / "regularisation_sweep.json") for stem in POP.UPDATE_PAIRS_REPAIR]
    sel_cache: dict[Path, dict] = {}
    rows, input_sha = [], {}
    for label, stem, sel_file in cells:
        if sel_file not in sel_cache:
            sel_cache[sel_file] = selections(sel_file)
        sel = sel_cache[sel_file][stem]
        if list(sel["seeds"]) != list(args.seeds) or float(sel["holdout"]) != P.HOLDOUT:
            raise SystemExit(f"REFUSING: {stem} selection used seeds/holdout {sel['seeds']}/{sel['holdout']}")
        cfg = json.load(open(RESULTS / f"{stem}.json", encoding="utf-8"))["config"]
        axis = cfg.get("axis", "safety")
        z, paths = {}, {}
        for part in ("probe_train", "threshold_calib", "eval"):
            z[part], paths[part] = load(cfg["pre_repo"], cfg["pre_rev"], axis, part)
            input_sha[f"{stem}:pre/{part}"] = sha256_file(paths[part])
        for part in ("probe_train", "threshold_calib"):
            if input_sha[f"{stem}:pre/{part}"] != sel["input_sha"].get(f"pre/{part}"):
                raise SystemExit(f"REFUSING: {stem} pre/{part} is not the bytes the selection used")
        k = f"layer_{sel['layer']}"
        y_tr = z["probe_train"]["labels"].astype(bool)
        y_cal = z["threshold_calib"]["labels"].astype(bool)
        y_ev = z["eval"]["labels"].astype(bool)
        for part in z:
            if "nonfinite" in z[part].files and z[part]["nonfinite"].any():
                raise SystemExit(f"REFUSING: {stem} pre/{part} has non-finite items")
        X_tr, X_cal, X_ev = z["probe_train"][k], z["threshold_calib"][k], z["eval"][k]
        cal_s, ev_s = [], []
        for seed in args.seeds:
            _, fit = P.d117_split(len(y_tr), seed)
            pr = P.fit_probe(X_tr[fit], y_tr[fit], seed, sel["C"], "d117_permutation")
            cal_s.append(pr.score(X_cal[~y_cal]))
            ev_s.append(pr.score(X_ev[~y_ev]))
        CAL, EV = np.column_stack(cal_s), np.column_stack(ev_s)          # (1000, S), (2000, S)
        n_cal, n_ev = CAL.shape[0], EV.shape[0]

        def realised(cal: np.ndarray, ev: np.ndarray) -> float:
            T = np.array([P.thr_at_fpr(cal[:, s], args.target_fpr) for s in range(cal.shape[1])])
            return float(P.floored_fpr(ev >= T).mean())

        observed = realised(CAL, EV)
        pool = np.vstack([CAL, EV])
        rng = np.random.default_rng(args.null_seed)
        null = np.empty(args.n_null)
        for r in range(args.n_null):
            perm = rng.permutation(n_cal + n_ev)
            null[r] = realised(pool[perm[:n_cal]], pool[perm[n_cal:]])
        pct = 100.0 * (np.sum(null < observed) + 0.5 * np.sum(null == observed)) / args.n_null
        q = np.percentile(null, [2.5, 50, 97.5])
        outside = bool(observed < q[0] or observed > q[2])
        rows.append({"cell": label, "pair": stem, "axis": axis, "selected_C": sel["C"], "layer": sel["layer"],
                     "n_calib_negatives": n_cal, "n_eval_negatives": n_ev,
                     "observed_fpr_pre_mean_over_seeds": observed,
                     "observed_flagged_mean": observed * n_ev,
                     "null_mean": float(null.mean()), "null_q2_5": float(q[0]), "null_median": float(q[1]),
                     "null_q97_5": float(q[2]), "observed_percentile_midrank": float(pct),
                     "outside_central_95": outside})
        print(f"{label:<36} C={sel['C']:<6g} L{sel['layer']:<3} observed {observed:.5f} ({observed * n_ev:.1f}/{n_ev})  "
              f"null mean {null.mean():.5f} [{q[0]:.5f}, {q[2]:.5f}]  percentile {pct:5.1f}  "
              f"{'OUTSIDE' if outside else 'inside'}", flush=True)

    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"]:
        raise SystemExit("REFUSING to write: source changed during the run")
    n_out = sum(r["outside_central_95"] for r in rows)
    payload = {**provenance_block("baseline_null"), "code": code0,
               "config": {"n_null": args.n_null, "null_seed": args.null_seed, "seeds": args.seeds,
                          "split": f"D-117 (holdout {P.HOLDOUT}), permutation order",
                          "target_fpr": args.target_fpr,
                          "null": "calib+eval negatives pooled and re-partitioned at the original sizes; one partition per draw shared by all seeds (D-083)",
                          "population": "4a LoRA base + UPDATE_PAIRS_REPAIR (src/populations.py)"},
               "input_cache_sha256": input_sha,
               "summary": {"cells": len(rows), "outside_central_95": n_out},
               "cells": rows}
    digest = write_results(out_path, payload, exclusive=True)
    print(f"\n{n_out} of {len(rows)} cells outside their null's central 95%\nWritten: {out_path}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
