"""Score the D-104 predictions from the per-pair shift_alignment results.

Applies the rules exactly as pre-registered:
  1 alignment effect   |cos| larger for safety at ALL matched doses AND ||d|| within 1.5x
  2 bigger update      ||d|| larger for safety by more than 1.5x with comparable |cos|
  3 both               report the decomposition of the log ratio, name no single cause
  4 not supported      |cos| not consistently ordered across doses
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
DOSES = (25, 100, 400)


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Score D-104")
    ap.add_argument("--out", default=str(RES / "shift_alignment_summary.json"))
    args = ap.parse_args()

    rows, inputs = [], {}
    for p in sorted(RES.glob("shiftalign_*.json")):
        d = json.load(open(p, encoding="utf-8"))
        inputs[p.name] = sha256_file(p)
        s = d["summary"]
        rows.append({"tag": p.stem.replace("shiftalign_", ""), "label": d["label"],
                     "kind": d["update_kind"], "shift_norm": s["shift_norm"],
                     "cos": s["cos"], "cos_abs": s["cos_abs"],
                     "score_shift_abs": abs(s["score_shift"]),
                     "random_cos": s["random_cos_reference"], "n_layers": s["n_layers"]})
    by_tag = {r["tag"]: r for r in rows}

    print(f"{'update':<38}{'||d||':>9}{'|cos|':>9}{'cos':>9}{'|score shift|':>14}{'vs random cos':>14}")
    for r in sorted(rows, key=lambda x: (x["kind"], x["tag"])):
        print(f"{r['label']:<38}{r['shift_norm']:>9.2f}{r['cos_abs']:>9.4f}{r['cos']:>+9.4f}"
              f"{r['score_shift_abs']:>14.2f}{r['cos_abs'] / r['random_cos']:>13.1f}x")

    # the matched A/B, dose by dose, for EVERY safety corpus against the general arm.
    # D-116: the log decomposition is stored, not just printed, so it is reproducible
    # from a results file rather than from console output.
    comp: dict[str, Any] = {}
    for arm_tag, arm_name in (("lora_safety", "hh-harmless"), ("lora_pku", "pku-saferlhf")):
        for s in DOSES:
            g, sf = by_tag.get(f"lora_general_s{s}"), by_tag.get(f"{arm_tag}_s{s}")
            if not (g and sf):
                continue
            lr = math.log(sf["score_shift_abs"] / g["score_shift_abs"]) if g["score_shift_abs"] else None
            ln = math.log(sf["shift_norm"] / g["shift_norm"]) if g["shift_norm"] else None
            lc = math.log(sf["cos_abs"] / g["cos_abs"]) if g["cos_abs"] else None
            comp[f"{arm_name}_s{s}"] = {
                "corpus": arm_name, "dose": s,
                "cos_abs_general": g["cos_abs"], "cos_abs_safety": sf["cos_abs"],
                "cos_ratio": sf["cos_abs"] / g["cos_abs"] if g["cos_abs"] else None,
                "shift_norm_general": g["shift_norm"], "shift_norm_safety": sf["shift_norm"],
                "shift_ratio": sf["shift_norm"] / g["shift_norm"] if g["shift_norm"] else None,
                "log_score_shift_ratio": lr, "log_size_term": ln, "log_alignment_term": lc,
                "residual": (lr - ln - lc) if None not in (lr, ln, lc) else None,
                "alignment_share_of_explained": (lc / (lc + ln)) if None not in (lc, ln) and (lc + ln) else None,
            }
    legacy: dict[str, Any] = {}
    for s in DOSES:
        g, sf = by_tag.get(f"lora_general_s{s}"), by_tag.get(f"lora_safety_s{s}")
        if not (g and sf):
            continue
        legacy[str(s)] = {
            "cos_abs_general": g["cos_abs"], "cos_abs_safety": sf["cos_abs"],
            "cos_ratio_safety_over_general": sf["cos_abs"] / g["cos_abs"] if g["cos_abs"] else None,
            "shift_norm_general": g["shift_norm"], "shift_norm_safety": sf["shift_norm"],
            "shift_ratio_safety_over_general": sf["shift_norm"] / g["shift_norm"] if g["shift_norm"] else None,
            "score_shift_general": g["score_shift_abs"], "score_shift_safety": sf["score_shift_abs"],
        }
    verdict = "NOT EVALUATED (matched doses missing)"
    if len(legacy) == len(DOSES):
        cos_bigger = all(c["cos_ratio_safety_over_general"] > 1.0 for c in legacy.values())
        norm_close = all(1 / 1.5 <= c["shift_ratio_safety_over_general"] <= 1.5 for c in legacy.values())
        norm_bigger = all(c["shift_ratio_safety_over_general"] > 1.5 for c in legacy.values())
        cos_close = all(1 / 1.5 <= c["cos_ratio_safety_over_general"] <= 1.5 for c in legacy.values())
        if cos_bigger and norm_close:
            verdict = "H-ALIGN SUPPORTED (rule 1): alignment effect, shift sizes comparable"
        elif norm_bigger and cos_close:
            verdict = "H-ALIGN REFUTED (rule 2): the safety update is simply larger"
        elif cos_bigger and norm_bigger:
            verdict = "BOTH GROW (rule 3): report the decomposition, name no single cause"
        else:
            verdict = "H-ALIGN NOT SUPPORTED (rule 4): |cos| not consistently ordered"
        print("\n  matched-dose A/B (same base, recipe and dose; only the corpus differs):")
        for s, c in legacy.items():
            print(f"    step {s:>3}: |cos| general {c['cos_abs_general']:.4f} vs safety "
                  f"{c['cos_abs_safety']:.4f} ({c['cos_ratio_safety_over_general']:.2f}x)   "
                  f"||d|| general {c['shift_norm_general']:.2f} vs safety {c['shift_norm_safety']:.2f} "
                  f"({c['shift_ratio_safety_over_general']:.2f}x)")
            # log-decomposition of the score-shift ratio into its two factors
            if c["score_shift_general"]:
                lr = math.log(c["score_shift_safety"] / c["score_shift_general"])
                ln = math.log(c["shift_ratio_safety_over_general"])
                lc = math.log(c["cos_ratio_safety_over_general"])
                print(f"             log score-shift ratio {lr:+.3f} = size {ln:+.3f} + alignment {lc:+.3f}"
                      f"  (residual {lr - ln - lc:+.3f})")
    print(f"\n  VERDICT (D-104): {verdict}")

    payload = {**provenance_block("shift_alignment_summary"), "preregistration": "D-104",
               "input_sha256": inputs, "rows": rows, "matched_dose_comparison": comp,
               "matched_dose_hh_only_legacy": legacy,
               "verdict": verdict}
    digest = write_results(args.out, payload)
    print(f"\nWritten: {args.out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
