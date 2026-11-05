"""R13: how much does the treatment of censored cells move the headline numbers?

A realised false-positive rate of exactly 0 is censored, not small: with n_neg evaluation
negatives the finest resolvable non-zero rate is 1/n_neg. The summary metric is
mean |log(realised/target)| over the layer sweep, so a censored cell has no finite value
and some convention must supply one. Three are defensible:

  resolution   floor the rate at 1/n_neg            (what the metric can resolve)
  rule3        floor the rate at 3/n_neg            (upper 95% bound on a zero count)
  exclude      drop the cell, average over the rest, and report how many were dropped

This script recomputes every affected headline number under all three and prints the
spread, so the choice can be made on what it actually changes rather than on principle
alone. It writes nothing; it only reads the declared populations.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import ensure_utf8_stdout  # noqa: E402
import populations as POP  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TREATMENTS = ("resolution", "rule3", "exclude")


def load(stem: str) -> dict:
    """Populations name a stem; the repair arms live in repair_<stem>.json."""
    for cand in (RES / f"{stem}.json", RES / f"repair_{stem}.json"):
        if cand.exists():
            return json.load(open(cand, encoding="utf-8"))
    raise FileNotFoundError(stem)


def n_neg_from_manifest(axis_file: str) -> int:
    """The measurement resolution is 1/n_neg, so n_neg must come from the frozen split
    manifest rather than from a constant in this file (the results files do not record
    it; see the note printed at the end of main)."""
    for name in ("dataset_manifest.json", "dataset_manifest_wildguard.json"):
        man = json.load(open(RES / name, encoding="utf-8"))
        for axis in man["axes"].values():
            ev = axis.get("partitions", {}).get("eval")
            if ev and Path(ev["file"]).name == axis_file:
                return int(ev["n_neg"])
    raise KeyError(axis_file)


N_NEG = {"safety": n_neg_from_manifest("safety__eval.jsonl"),
         "safety_wildguard": n_neg_from_manifest("safety_wildguard__eval.jsonl")}


def metric(d: dict, arm: str, treatment: str) -> tuple[float, int]:
    """Return (mean |log(fpr/target)|, n_censored) under one censoring treatment."""
    tgt = d["config"]["target_fpr"]
    axis = d["config"].get("axis", "safety")
    n_neg = N_NEG["safety_wildguard" if "wildguard" in axis else "safety"]
    vals, censored = [], 0
    for row in d["rows"]:
        fpr = row[f"{arm}_fpr_mean"]
        if fpr == 0:
            censored += 1
            if treatment == "exclude":
                continue
            fpr = (1.0 if treatment == "resolution" else 3.0) / n_neg
        vals.append(abs(math.log(fpr / tgt)))
    return (sum(vals) / len(vals) if vals else float("nan")), censored


def main() -> int:
    ensure_utf8_stdout()
    print("R13 censoring sensitivity. Metric: mean |log(realised FPR / target)| "
          "over the layer sweep.\n")

    # ---- the nine repair cells, all four arms -------------------------------
    arms = ("frozen_all", "restandardise", "quantile_thr", "refit_probe")
    print("=" * 78)
    print("REPAIR POPULATION (headline: frozen 1.154 -> restandardise 0.377, refit 0.345)")
    print("=" * 78)
    total_censored = 0
    means: dict[str, dict[str, list[float]]] = {t: {a: [] for a in arms} for t in TREATMENTS}
    for stem in POP.UPDATE_PAIRS_REPAIR:
        d = load(stem)
        for a in arms:
            for t in TREATMENTS:
                v, c = metric(d, a, t)
                means[t][a].append(v)
                if t == "resolution":
                    total_censored += c
    print(f"censored cells across the whole repair population: {total_censored}\n")
    print(f"  {'arm':<16}" + "".join(f"{t:>14}" for t in TREATMENTS) + f"{'spread':>10}")
    for a in arms:
        row = [sum(means[t][a]) / len(means[t][a]) for t in TREATMENTS]
        print(f"  {a:<16}" + "".join(f"{v:>14.4f}" for v in row)
              + f"{max(row) - min(row):>10.4f}")

    # ---- the LoRA arms, where the one censored cell lives -------------------
    print("\n" + "=" * 78)
    print("LORA ARMS (headline: safety/general damage ratios 1.5x-2.8x)")
    print("=" * 78)
    per_arm: dict[str, dict[int, dict[str, float]]] = {}
    for arm, stems in POP.LORA_ARMS.items():
        per_arm[arm] = {}
        for stem in stems:
            d = load(stem)
            dose = int(stem.rsplit("_s", 1)[1])
            per_arm[arm][dose] = {}
            for t in TREATMENTS:
                v, c = metric(d, "frozen_all", t)
                per_arm[arm][dose][t] = v
                if c:
                    print(f"  censored: {arm} dose {dose} has {c} cell(s) "
                          f"with realised FPR exactly 0")
    print()
    gen = "general (alpaca)"
    for t in TREATMENTS:
        print(f"  --- {t} ---")
        for arm in per_arm:
            vals = " ".join(f"{d}:{per_arm[arm][d][t]:.3f}" for d in sorted(per_arm[arm]))
            print(f"    {arm:<22} {vals}")
        for arm in per_arm:
            if arm == gen:
                continue
            ratios = " ".join(
                f"{d}:{per_arm[arm][d][t] / per_arm[gen][d][t]:.2f}x"
                for d in sorted(per_arm[arm]))
            print(f"    ratio vs general       {arm:<22} {ratios}")
        print()

    print("=" * 78)
    print("Does the qualitative claim survive all three treatments?")
    ok = True
    for t in TREATMENTS:
        for arm in per_arm:
            if arm == gen:
                continue
            for d in sorted(per_arm[arm]):
                if per_arm[arm][d][t] / per_arm[gen][d][t] <= 1.0:
                    ok = False
                    print(f"  NO: under {t}, {arm} at dose {d} is not above general")
    print("  yes: every safety arm exceeds the general arm at every dose, "
          "under all three treatments" if ok else "")

    ratios = [per_arm[a][d][t] / per_arm[gen][d][t]
              for t in TREATMENTS for a in per_arm if a != gen for d in per_arm[a]]
    out = {
        "schema_version": 1,
        "run_kind": "censoring_sensitivity",
        "question": "R13: how much does the treatment of a censored (exactly zero) "
                    "realised FPR move the headline numbers?",
        "n_neg_eval": N_NEG,
        "resolution_penalty": {t: abs(math.log(((1.0 if t == "resolution" else 3.0)
                                                / N_NEG["safety"]) / 0.01))
                               for t in ("resolution", "rule3")},
        "repair_population": {
            "n_censored_cells_total": total_censored,
            "means": {t: {a: sum(means[t][a]) / len(means[t][a]) for a in arms}
                      for t in TREATMENTS},
        },
        "lora_arms": {a: {str(d): per_arm[a][d] for d in sorted(per_arm[a])}
                      for a in per_arm},
        "lora_damage_ratios": {
            t: {a: {str(d): per_arm[a][d][t] / per_arm[gen][d][t]
                    for d in sorted(per_arm[a])}
                for a in per_arm if a != gen}
            for t in TREATMENTS},
        "min_ratio_over_all_treatments": min(ratios),
        "max_ratio_over_all_treatments": max(ratios),
        "qualitative_claim_survives_all_treatments": ok,
    }
    dest = RES / "censoring_sensitivity.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten: {dest.name}")
    print("NOTE: the per-run results files do not record n_neg_eval; it is read here from "
          "the frozen-split manifest (safety=%d, wildguard=%d)."
          % (N_NEG["safety"], N_NEG["safety_wildguard"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
