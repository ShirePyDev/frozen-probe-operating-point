"""Would an AUC-based maintenance pipeline certify our miscalibrated cells as healthy? (D-085)

arXiv:2606.15980v2 fixes, before its full run, these criteria (verbatim in substance):
  stale            big_drop = 1[dAUC <= -0.05]  or  operational failure (post AUC < 0.70)
  repair succeeds  "recovers at least half the lost AUC or restores AUC above 0.70"

This script applies exactly those criteria to results already on disk and counts
how many cells whose frozen threshold is badly miscalibrated they would call fine.
No new data; every input is hashed into the output. Pure Python (no numpy), so it
runs anywhere the results files are.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import ensure_utf8_stdout, provenance_block, sha256_file, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# Fixed in D-085 before any number was computed.
DEPTH_FILES = ["gemma2_base2instruct", "llama3_base2instruct", "llama31_base2instruct", "llama3_to_31",
               "mistral_base2instruct", "mistral_v01_v02_dense", "olmo_base2instruct",
               "qwen_base2instruct", "WG_llama3_base2instruct", "WG_mistral_base2instruct"]
MOMENT_FILES = ["gemma2_base2instruct", "llama3_base2instruct", "llama31_base2instruct", "llama3_to_31",
                "mistral_base2instruct_v2", "olmo_base2instruct", "qwen_base2instruct"]
BIG_DROP, OP_FAIL, ELIGIBLE = 0.05, 0.70, 0.70
CAL_FACTORS = {"primary_2x": 2.0, "sensitivity_5x": 5.0}


def stale_by_auc(auc_base: float, auc_post: float) -> bool:
    return (auc_post - auc_base) <= -BIG_DROP or auc_post < OP_FAIL


def repair_passes_auc(auc_base: float, auc_frozen: float, auc_arm: float) -> bool:
    lost = auc_base - auc_frozen
    return (lost > 0 and (auc_arm - auc_frozen) >= 0.5 * lost) or auc_arm > OP_FAIL


# D-102: the LoRA population, complete, evaluated under the SAME criteria.
LORA_FILES = ["loraA100_mistral_s25", "loraA100_mistral_s100", "loraA100_mistral_s400",
              "lorasafe_mistral_s25", "lorasafe_mistral_s100", "lorasafe_mistral_s400",
              "hw_v100_a100"]


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Apply the prior work's AUC staleness rule to our cells")
    ap.add_argument("--population", choices=["main", "lora"], default="main",
                    help="main = D-085's 10 update pairs; lora = D-102's LoRA arms plus the control")
    args = ap.parse_args()
    global DEPTH_FILES, MOMENT_FILES
    if args.population == "lora":
        DEPTH_FILES = LORA_FILES
        # the decomposition exists for the LoRA arms but not for the GPU-change control
        MOMENT_FILES = [f for f in LORA_FILES if f != "hw_v100_a100"]
    inputs: dict[str, str] = {}
    out: dict[str, Any] = {"criteria": {"big_drop": -BIG_DROP, "operational_failure_auc": OP_FAIL,
                                        "eligibility_base_auc_ASSUMED": ELIGIBLE,
                                        "calibration_failure_factors": CAL_FACTORS}}

    # ---- primary: depth_* cells ----
    cells = []
    for name in DEPTH_FILES:
        p = RES / f"depth_{name}.json"
        inputs[p.name] = sha256_file(p)
        d = json.load(open(p, encoding="utf-8"))
        tgt = d["config"]["target_fpr"]
        for r in d["rows"]:
            if r.get("auroc_pre_refit_mean") is None:
                continue
            cells.append({"pair": name, "layer": r["layer"], "auc_base": r["auroc_pre_refit_mean"],
                          "auc_frozen": r["auroc_frozen_mean"],
                          "fpr_ratio": r["post_fpr_at_frozen_thr_mean"] / tgt})

    prim: dict[str, Any] = {}
    for scope in ("eligible", "all"):
        cs = [c for c in cells if scope == "all" or c["auc_base"] >= ELIGIBLE]
        entry: dict[str, Any] = {"n_cells": len(cs), "n_pairs": len({c["pair"] for c in cs}),
                                 "n_stale_by_auc": sum(stale_by_auc(c["auc_base"], c["auc_frozen"]) for c in cs)}
        for tag, f in CAL_FACTORS.items():
            fail = [c for c in cs if c["fpr_ratio"] >= f or c["fpr_ratio"] <= 1 / f]
            invisible = [c for c in fail if not stale_by_auc(c["auc_base"], c["auc_frozen"])]
            per_pair = {}
            for c in fail:
                pp = per_pair.setdefault(c["pair"], {"cal_fail": 0, "invisible": 0})
                pp["cal_fail"] += 1
                pp["invisible"] += not stale_by_auc(c["auc_base"], c["auc_frozen"])
            entry[tag] = {"n_calibration_failures": len(fail), "n_invisible_to_auc_criterion": len(invisible),
                          "fraction_invisible": (len(invisible) / len(fail)) if fail else None,
                          "n_pairs_with_invisible": len({c["pair"] for c in invisible}),
                          "fpr_ratio_range_invisible": ([min(c["fpr_ratio"] for c in invisible),
                                                         max(c["fpr_ratio"] for c in invisible)]
                                                        if invisible else None),
                          "per_pair": per_pair}
        prim[scope] = entry
    out["primary_depth_cells"] = prim

    # ---- secondary: moment_* repair arms judged by their AUC repair rule ----
    sec: dict[str, Any] = {}
    for arm in ("paired_align", "restandardise"):
        n = n_pass = n_pass_worse = n_stale = n_stale_pass = n_stale_pass_worse = 0
        for name in MOMENT_FILES:
            p = RES / f"moment_{name}.json"
            inputs[p.name] = sha256_file(p)
            d = json.load(open(p, encoding="utf-8"))
            for r in d["rows"]:
                a = r["arms"]
                if r["auroc_pre"] < ELIGIBLE:
                    continue
                n += 1
                passes = repair_passes_auc(r["auroc_pre"], a["frozen"]["auroc_mean"], a[arm]["auroc_mean"])
                worse = a[arm]["log_fpr_err"] > a["frozen"]["log_fpr_err"]
                n_pass += passes
                n_pass_worse += passes and worse
                if stale_by_auc(r["auroc_pre"], a["frozen"]["auroc_mean"]):
                    n_stale += 1
                    n_stale_pass += passes
                    n_stale_pass_worse += passes and worse
        sec[arm] = {"n_eligible_cells": n, "n_pass_auc_repair_rule": n_pass,
                    "n_pass_but_calibration_worse_than_frozen": n_pass_worse,
                    "n_stale_by_auc": n_stale, "n_stale_and_pass": n_stale_pass,
                    "n_stale_pass_but_calibration_worse_than_frozen": n_stale_pass_worse}
    out["secondary_moment_arms"] = sec

    print(json.dumps({"primary": {s: {k: (v if not isinstance(v, dict) else
                                          {kk: vv for kk, vv in v.items() if kk != "per_pair"})
                                      for k, v in e.items()} for s, e in prim.items()},
                      "secondary": sec}, indent=1))
    dest = RES / (f"auc_criterion_audit_{args.population}.json" if args.population != "main"
                  else "auc_criterion_audit.json")
    digest = write_results(dest, {**provenance_block("auc_criterion_audit"), "preregistration": "D-085",
                                  "input_sha256": inputs, **out})
    print(f"\nWritten: {dest}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
