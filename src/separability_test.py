"""Did the updated representation become less SEPARABLE, or did it just MOVE? (D-099)

Three published accounts explain why a frozen probe breaks after a backbone update:

  direction misalignment  arXiv:2606.15980v2 - a paired map back to the old space
                          should repair it (tested in D-094: it does not)
  separability loss       arXiv:2603.01297 - alignment reduces class separability, so
                          a probe REFIT on the updated model should also be worse
  moment shift            ours (D-082) - the representation moves, so a refit probe
                          recovers discrimination while the FROZEN threshold breaks

Only the last two are distinguished here, and they make opposite predictions about one
number we already measured for every pair: the AUROC of a probe refit on the POST model
against the AUROC of the same probe protocol on the PRE model.

Reads results/depth_*.json only. No caches, no GPU, no new fits.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import ensure_utf8_stdout, provenance_block, sha256_file, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# Fixed in D-099: every pair we have, none excluded. Label says what the update IS,
# because the separability account is specifically about alignment/instruction tuning.
PAIRS = [
    ("depth_mistral_base2instruct", "Mistral v0.1 -> Instruct", "instruct"),
    ("depth_gemma2_base2instruct", "Gemma-2 9b -> it", "instruct"),
    ("depth_llama3_base2instruct", "Llama-3 -> Instruct", "instruct"),
    ("depth_llama31_base2instruct", "Llama-3.1 -> Instruct", "instruct"),
    ("depth_qwen_base2instruct", "Qwen2.5 -> Instruct", "instruct"),
    ("depth_olmo_base2instruct", "OLMo-2 -> Instruct", "instruct"),
    ("depth_WG_mistral_base2instruct", "Mistral -> Instruct (WildGuard)", "instruct"),
    ("depth_WG_llama3_base2instruct", "Llama-3 -> Instruct (WildGuard)", "instruct"),
    ("depth_mistral_v01_v02_dense", "Mistral v0.1 -> v0.2", "version"),
    ("depth_llama3_to_31", "Llama-3 -> Llama-3.1", "version"),
    ("depth_loraA100_mistral_s25", "LoRA general, 25 steps", "lora_general"),
    ("depth_loraA100_mistral_s100", "LoRA general, 100 steps", "lora_general"),
    ("depth_loraA100_mistral_s400", "LoRA general, 400 steps", "lora_general"),
    ("depth_lorasafe_mistral_s25", "LoRA safety, 25 steps", "lora_safety"),
    ("depth_lorasafe_mistral_s100", "LoRA safety, 100 steps", "lora_safety"),
    ("depth_lorasafe_mistral_s400", "LoRA safety, 400 steps", "lora_safety"),
    ("depth_hw_v100_a100", "GPU change only (control)", "control"),
]


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Separability vs moment shift (D-099)")
    ap.add_argument("--out", default=str(RES / "separability_test.json"))
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    inputs: dict[str, str] = {}
    for stem, label, kind in PAIRS:
        p = RES / f"{stem}.json"
        if not p.exists():
            print(f"  MISSING (not yet run): {stem}")
            continue
        inputs[p.name] = sha256_file(p)
        d = json.load(open(p, encoding="utf-8"))
        tgt = d["config"]["target_fpr"]
        ok = [r for r in d["rows"] if r.get("auroc_pre_refit_mean") is not None
              and r.get("auroc_post_refit_mean") is not None]
        if not ok:
            print(f"  NO USABLE LAYERS: {stem}")
            continue
        # Separability change per layer, then the median over layers of that pair.
        deltas = [r["auroc_post_refit_mean"] - r["auroc_pre_refit_mean"] for r in ok]
        drops = [r["auroc_pre_refit_mean"] - r["auroc_frozen_mean"] for r in ok]
        ratios = [r["post_fpr_at_frozen_thr_mean"] / tgt for r in ok]
        # seed spread of the refit AUROCs, to judge whether a small delta is meaningful
        sd_pre = st.median([r.get("auroc_pre_refit_sd", 0.0) for r in ok])
        sd_post = st.median([r.get("auroc_post_refit_sd", 0.0) for r in ok])
        rows.append({
            "pair": stem, "label": label, "kind": kind, "n_layers": len(ok),
            "median_separability_change": st.median(deltas),
            "min_separability_change": min(deltas), "max_separability_change": max(deltas),
            "median_auroc_pre_refit": st.median([r["auroc_pre_refit_mean"] for r in ok]),
            "median_auroc_post_refit": st.median([r["auroc_post_refit_mean"] for r in ok]),
            "median_frozen_auroc_drop": st.median(drops),
            "median_seed_sd_refit": max(sd_pre, sd_post),
            "worst_fpr_ratio": max(ratios), "min_fpr_ratio": min(ratios),
        })

    print(f"{'update':<34}{'kind':<14}{'sep change':>12}{'(range)':>20}"
          f"{'seed sd':>9}{'frozen dAUROC':>15}{'worst FPR':>11}")
    for r in rows:
        rng = f"[{r['min_separability_change']:+.3f},{r['max_separability_change']:+.3f}]"
        print(f"{r['label']:<34}{r['kind']:<14}{r['median_separability_change']:>+12.4f}{rng:>20}"
              f"{r['median_seed_sd_refit']:>9.4f}{r['median_frozen_auroc_drop']:>+15.4f}"
              f"{r['worst_fpr_ratio']:>10.1f}x")

    real = [r for r in rows if r["kind"] != "control"]
    med = st.median([r["median_separability_change"] for r in real]) if real else float("nan")
    n_neg = sum(1 for r in real if r["median_separability_change"] < 0)
    # D-099 rules, applied mechanically.
    if med <= -0.02 and n_neg > len(real) / 2:
        verdict = "SEPARABILITY LOSS IS REAL in our data (rule 1)"
    elif abs(med) < 0.01:
        verdict = "SEPARABILITY INTACT: damage is calibration-only in our setting (rule 2)"
    else:
        verdict = "INCONCLUSIVE (rule 3)"
    by_kind = {k: st.median([r["median_separability_change"] for r in rows if r["kind"] == k])
               for k in sorted({r["kind"] for r in rows})}
    print(f"\n  median separability change over {len(real)} real updates: {med:+.4f} "
          f"({n_neg} of {len(real)} negative)")
    print(f"  by update kind: " + "  ".join(f"{k} {v:+.4f}" for k, v in by_kind.items()))
    print(f"  VERDICT (D-099): {verdict}")

    payload = {**provenance_block("separability_test"), "preregistration": "D-099",
               "input_sha256": inputs, "rows": rows,
               "median_separability_change_real_updates": med,
               "n_negative_of_n": [n_neg, len(real)], "median_by_kind": by_kind,
               "verdict": verdict}
    digest = write_results(args.out, payload)
    print(f"\nWritten: {args.out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
