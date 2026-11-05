"""Apply the D-079 pre-registered conditions to the LoRA dose series. Mechanically.

D-079 fixed the predictions and refutation conditions BEFORE any LoRA result
existed. This script encodes those conditions exactly as written and reads the
results from disk, so the verdict cannot be adjusted after seeing the numbers.
Any change to a threshold here would have to be recorded as a deviation from the
pre-registration, not made silently.

Conditions, verbatim in substance from D-079
--------------------------------------------
H-LoRA-1  frozen mean |log(realised FPR / target)| is monotone non-decreasing over
          steps 25 <= 100 <= 400.
H-LoRA-2  the MEAN share of the restandardise repair FALLS with dose AND the gap
          (restandardise error - refit oracle error) WIDENS with dose.
          REFUTED if at step 400: mean share >= 0.70 AND that gap <= 0.10.
          If the step-400 mean share < 0.40: direction/covariance change dominates.
H-LoRA-3  frozen-probe AUROC drop grows with dose.

Where a quantity is numerically unstable it is reported as such rather than forced
into a verdict: the mean share is a ratio whose denominator (frozen - restandardise
error) can be near zero at a small dose, and it is marked UNSTABLE below 0.05.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import ensure_utf8_stdout, provenance_block, sha256_file, write_results  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
STEPS = (25, 100, 400)


def load(p: Path) -> dict[str, Any] | None:
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def nondecreasing(xs: list[float]) -> bool:
    return all(a <= b for a, b in zip(xs, xs[1:]))


def strictly_decreasing(xs: list[float]) -> bool:
    return all(a > b for a, b in zip(xs, xs[1:]))


def strictly_increasing(xs: list[float]) -> bool:
    return all(a < b for a, b in zip(xs, xs[1:]))


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Evaluate D-079 pre-registered LoRA hypotheses")
    ap.add_argument("--tag", default="lora", help="results prefix, e.g. lora or lorasafe")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tag = args.tag

    per: dict[int, dict[str, Any]] = {}
    inputs: dict[str, str] = {}
    for s in STEPS:
        files = {k: RESULTS / f"{k}_{tag}_mistral_s{s}.json" for k in ("repair", "moment", "depth")}
        d = {k: load(p) for k, p in files.items()}
        for k, p in files.items():
            if p.exists():
                inputs[p.name] = sha256_file(p)
        if any(v is None for v in d.values()):
            missing = [k for k, v in d.items() if v is None]
            print(f"step {s}: missing {missing} - cannot evaluate")
            return 2
        m = d["moment"]["summary"]
        e_frozen = m["frozen"]["mean_log_fpr_err"]
        e_restd = m["restandardise"]["mean_log_fpr_err"]
        denom = e_frozen - e_restd
        rows = d["depth"]["rows"]
        per[s] = {
            "frozen_err_17layers": d["repair"]["mean_abs_log_fpr_error"]["frozen_all"],
            "restd_err_17layers": d["repair"]["mean_abs_log_fpr_error"]["restandardise"],
            "oracle_err_17layers": d["repair"]["mean_abs_log_fpr_error"]["refit_probe"],
            "frozen_err_5layers": e_frozen,
            "mean_share": d["moment"]["share_of_repair_from_mean_alone"],
            "mean_share_unstable": abs(denom) < 0.05,
            "gap_restd_minus_oracle_5layers": e_restd - m["refit_oracle"]["mean_log_fpr_err"],
            "frozen_auroc_drop_5layers": m["frozen"]["mean_auroc_drop_vs_pre"],
            "frozen_auroc_drop_mean_17layers": sum(r["drop_frozen_mean"] for r in rows) / len(rows),
            "frozen_auroc_drop_max_17layers": max(r["drop_frozen_mean"] for r in rows),
            "worst_fpr_ratio_17layers": max(r["post_fpr_at_frozen_thr_mean"] for r in rows)
                                         / d["depth"]["config"]["target_fpr"],
        }

    v: dict[str, Any] = {}
    fe = [per[s]["frozen_err_17layers"] for s in STEPS]
    v["H-LoRA-1"] = {"frozen_err_by_step": fe,
                     "verdict": "SUPPORTED" if nondecreasing(fe) else "NOT SUPPORTED"}

    shares = [per[s]["mean_share"] for s in STEPS]
    gaps = [per[s]["gap_restd_minus_oracle_5layers"] for s in STEPS]
    unstable = [per[s]["mean_share_unstable"] for s in STEPS]
    s400, g400 = per[400]["mean_share"], per[400]["gap_restd_minus_oracle_5layers"]
    if per[400]["mean_share_unstable"] or s400 is None:
        h2 = "INCONCLUSIVE (step-400 mean share numerically unstable)"
    elif s400 >= 0.70 and g400 <= 0.10:
        h2 = "REFUTED (pre-registered condition met: share >= 0.70 and gap <= 0.10 at step 400)"
    elif (None not in shares and not any(unstable)
          and strictly_decreasing(shares) and strictly_increasing(gaps)):
        h2 = "SUPPORTED (share falls and gap widens monotonically with dose)"
    else:
        h2 = "INCONCLUSIVE (neither the support nor the refutation condition is met)"
    v["H-LoRA-2"] = {"mean_share_by_step": shares, "share_unstable_by_step": unstable,
                     "gap_restd_minus_oracle_by_step": gaps,
                     "direction_dominates_at_400": (s400 is not None and s400 < 0.40
                                                    and not per[400]["mean_share_unstable"]),
                     "verdict": h2}

    dr = [per[s]["frozen_auroc_drop_mean_17layers"] for s in STEPS]
    v["H-LoRA-3"] = {"frozen_auroc_drop_mean_by_step": dr,
                     "verdict": "SUPPORTED" if strictly_increasing(dr) else "NOT SUPPORTED"}

    print("=" * 88)
    print(f"D-079 PRE-REGISTERED HYPOTHESES - tag '{tag}'")
    print("=" * 88)
    print(f"  {'step':>5} {'frozenErr17':>12} {'restd17':>9} {'oracle17':>9} {'meanShare':>10} "
          f"{'gap5':>7} {'AUROCdrop17':>12} {'worstFPRx':>10}")
    for s in STEPS:
        p = per[s]
        sh = "unstable" if p["mean_share_unstable"] else (f"{p['mean_share']:.1%}" if p["mean_share"] is not None else "None")
        print(f"  {s:>5} {p['frozen_err_17layers']:>12.4f} {p['restd_err_17layers']:>9.4f} "
              f"{p['oracle_err_17layers']:>9.4f} {sh:>10} {p['gap_restd_minus_oracle_5layers']:>7.3f} "
              f"{p['frozen_auroc_drop_mean_17layers']:>+12.4f} {p['worst_fpr_ratio_17layers']:>9.1f}x")
    for h, r in v.items():
        print(f"  {h}: {r['verdict']}")

    out = Path(args.out) if args.out else RESULTS / f"preregistration_verdict_{tag}.json"
    digest = write_results(out, {**provenance_block("evaluate_lora_preregistration"),
                                 "preregistration": "D-079", "tag": tag,
                                 "input_sha256": inputs, "per_step": per, "verdicts": v})
    print(f"\nWritten: {out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
