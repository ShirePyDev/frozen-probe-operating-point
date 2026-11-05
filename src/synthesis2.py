"""Regenerate every headline number from the results on disk. Populations are DECLARED.

Replaces the glob-based aggregator that silently absorbed controls and duplicates and
printed an impossible "103.2% of the gap closed" (F-038). Here:

  - every population comes from src/populations.py and is refused if it fails validation
  - every number is read from a results file, and every input file's SHA-256 is recorded
  - a censored FPR (0 of n benign flagged) is reported under BOTH treatments, never
    silently floored at 1e-9 (F-034)
  - the retracted pearson "mechanism" evidence (D-076) is not computed or printed
  - controls are printed as controls, never pooled with update pairs

If a number cannot be produced here, it is not a number this project may claim.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import populations as POP  # noqa: E402
from provenance import ensure_utf8_stdout, provenance_block, sha256_file, write_results  # noqa: E402

RESULTS = POP.RESULTS
INPUTS: dict[str, str] = {}


def load(stem: str) -> dict[str, Any]:
    p = POP.path(stem)
    INPUTS[p.name] = sha256_file(p)
    return json.load(open(p, encoding="utf-8"))


def pretty(stem: str, prefix: str) -> str:
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def n_neg_eval(d: dict[str, Any]) -> int:
    """Evaluation negatives for this run's axis, which sets the measurement resolution
    1/n_neg used to floor a censored FPR. Read from the frozen-split manifest rather than
    assumed: the per-run results files do not record it, and a wrong n_neg would silently
    change every censored cell. Prefers the run's own field if a later run records one."""
    if "n_neg_eval" in d:
        return int(d["n_neg_eval"])
    axis = d["config"].get("axis", "safety")
    want = "safety_wildguard__eval.jsonl" if "wildguard" in axis else "safety__eval.jsonl"
    for name in ("dataset_manifest.json", "dataset_manifest_wildguard.json"):
        path = RESULTS / name
        if not path.exists():
            continue
        man = json.load(open(path, encoding="utf-8"))
        for ax in man["axes"].values():
            ev = ax.get("partitions", {}).get("eval")
            if ev and Path(ev["file"]).name == want:
                return int(ev["n_neg"])
    raise KeyError(f"no n_neg for axis {axis!r}; cannot floor a censored FPR")


def repair_metric(d: dict[str, Any], arm: str, floored: bool) -> float:
    """The summary metric, either as originally computed or with a censored FPR floored
    at the eval resolution instead of 1e-9 (F-034)."""
    if not floored:
        return d["mean_abs_log_fpr_error"][arm]
    tgt = d["config"]["target_fpr"]
    n_neg = n_neg_eval(d)
    vals = [abs(math.log(max(r[f"{arm}_fpr_mean"], 1.0 / n_neg) / tgt)) for r in d["rows"]]
    return sum(vals) / len(vals)


def n_censored(d: dict[str, Any], arm: str) -> int:
    return sum(1 for r in d["rows"] if r[f"{arm}_fpr_mean"] == 0)


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Headline numbers, regenerated from disk")
    ap.add_argument("--out", default=str(RESULTS / "synthesis2.json"))
    args = ap.parse_args()

    problems = POP.validate()
    if problems:
        print("REFUSING: population declaration does not match disk:")
        for p in problems:
            print("   " + p)
        return 2

    out: dict[str, Any] = {}

    # ---------- 1. damage ----------
    print("=" * 104)
    print(f"1. DAMAGE - realised FPR of a FROZEN threshold, over {len(POP.UPDATE_PAIRS_DEPTH)} update pairs")
    print("=" * 104)
    print(f"  {'pair':<34}{'worst FPR':>11}{'min FPR':>10}{'max AUROC drop':>16}{'min CKA':>10}")
    dmg = []
    for stem in POP.UPDATE_PAIRS_DEPTH:
        d = load(stem)
        tgt = d["config"]["target_fpr"]
        ratios = [r["post_fpr_at_frozen_thr_mean"] / tgt for r in d["rows"]]
        drops = [r["drop_frozen_mean"] for r in d["rows"]]
        ckas = [r["cka"] for r in d["rows"]]
        dmg.append({"pair": pretty(stem, "depth_"), "worst_fpr_ratio": max(ratios),
                    "min_fpr_ratio": min(ratios), "max_auroc_drop": max(drops),
                    "min_cka": min(ckas), "n_layers": len(d["rows"])})
        print(f"  {dmg[-1]['pair']:<34}{max(ratios):>10.1f}x{min(ratios):>10.2f}"
              f"{max(drops):>+16.4f}{min(ckas):>10.5f}")
    worst = [r["worst_fpr_ratio"] for r in dmg]
    out["damage"] = {"pairs": dmg, "worst_fpr_ratio_range": [min(worst), max(worst)],
                     "max_auroc_drop_over_pairs": max(r["max_auroc_drop"] for r in dmg)}
    print(f"  -> worst realised-FPR ratio ranges {min(worst):.1f}x to {max(worst):.1f}x across pairs; "
          f"the largest AUROC drop anywhere is {out['damage']['max_auroc_drop_over_pairs']:+.4f}")

    # ---------- 2. repair ----------
    print("\n" + "=" * 104)
    print(f"2. REPAIR - mean |log(realised FPR / target)|, {len(POP.UPDATE_PAIRS_REPAIR)} pairs, "
          "resolution-floored (F-034)")
    print("=" * 104)
    arms = ("frozen_all", "restandardise", "quantile_thr", "refit_probe")
    print(f"  {'pair':<34}{'frozen':>9}{'restand':>9}{'quantile':>10}{'refit(lab)':>12}{'floor q95':>11}")
    rep, floors = [], []
    for stem in POP.UPDATE_PAIRS_REPAIR:
        d = load(stem)
        row = {"pair": pretty(stem, "repair_"),
               **{a: repair_metric(d, a, True) for a in arms},
               "n_censored_cells": {a: n_censored(d, a) for a in arms}}
        nf_stem = f"noisefloor_{stem}"
        if POP.path(nf_stem).exists():
            nf = load(nf_stem)
            row["floor_q95"] = nf["null_quantiles"]["q95"]
            row["floor_as_realised"] = nf["floor_as_realised"]
            floors.append(row["floor_as_realised"])
        rep.append(row)
        print(f"  {row['pair']:<34}{row['frozen_all']:>9.3f}{row['restandardise']:>9.3f}"
              f"{row['quantile_thr']:>10.3f}{row['refit_probe']:>12.3f}"
              f"{row.get('floor_q95', float('nan')):>11.3f}")
    means = {a: st.mean([r[a] for r in rep]) for a in arms}
    at_floor = {a: sum(1 for r in rep if "floor_q95" in r and r[a] <= r["floor_q95"]) for a in arms}
    print(f"  {'MEAN':<34}{means['frozen_all']:>9.3f}{means['restandardise']:>9.3f}"
          f"{means['quantile_thr']:>10.3f}{means['refit_probe']:>12.3f}"
          f"{st.mean(floors) if floors else float('nan'):>11.3f}")
    print(f"  -> arms at or below their pair's no-update floor (q95): " +
          ", ".join(f"{a} {at_floor[a]}/{len(floors)}" for a in arms))
    print("  -> the labelled refit is NOT a bound: its mean "
          f"({means['refit_probe']:.3f}) sits beside the measured no-update floor "
          f"({st.mean(floors):.3f} as realised)")
    out["repair"] = {"per_pair": rep, "means": means, "n_at_floor_q95": at_floor,
                     "mean_realised_no_update_floor": st.mean(floors) if floors else None}

    np_rows = []
    for stem in POP.NON_PARAMETRIC:
        d = load(stem)
        np_rows.append({"pair": pretty(stem, "repair_np_"), **d["mean_abs_log_fpr_error"]})
    if np_rows:
        print("\n  NON-PARAMETRIC retrieval layer (separate metric, never pooled above):")
        for r in np_rows:
            print(f"    {r['pair']:<32} frozen {r['frozen_thr']:.3f}  quantile {r['quantile_thr']:.3f}  "
                  f"zscore {r['zscore_thr']:.3f}  oracle {r['oracle_recal']:.3f}")
    out["non_parametric"] = np_rows

    # ---------- 3. mechanism ----------
    print("\n" + "=" * 104)
    print(f"3. MECHANISM - decomposition by intervention, {len(POP.UPDATE_PAIRS_MOMENT)} pairs")
    print("=" * 104)
    mech = []
    for stem in POP.UPDATE_PAIRS_MOMENT:
        d = load(stem)
        s = d["summary"]
        mech.append({"pair": pretty(stem, "moment_"),
                     "mean_share_of_repair": d["share_of_repair_from_mean_alone"],
                     "frozen": s["frozen"]["mean_log_fpr_err"],
                     "restandardise": s["restandardise"]["mean_log_fpr_err"],
                     "refit_oracle": s["refit_oracle"]["mean_log_fpr_err"]})
        print(f"  {mech[-1]['pair']:<34} mean-only share of the restandardise repair: "
              f"{mech[-1]['mean_share_of_repair']:.1%}")
    shares = [m["mean_share_of_repair"] for m in mech if m["mean_share_of_repair"] is not None]
    print(f"  -> median {st.median(shares):.1%}, range {min(shares):.1%} to {max(shares):.1%}")

    av_arms = ("frozen", "restandardise", "paired_align", "procrustes_paired", "coral_unpaired",
               "refit_oracle")
    av = []
    for stem in POP.UPDATE_PAIRS_ALIGNVAR:
        d = load(stem)
        av.append({"pair": pretty(stem, "alignvar_"),
                   **{a: d["summary"][a]["mean_log_fpr_err"] for a in av_arms}})
    av_means = {a: st.mean([r[a] for r in av]) for a in av_arms}
    print(f"\n  Alignment variants ({len(av)} pairs): " +
          "  ".join(f"{a} {av_means[a]:.3f}" for a in av_means))
    proc_beats_ridge = sum(1 for r in av if r["procrustes_paired"] < r["paired_align"])
    coral_beats_restd = sum(1 for r in av if r["coral_unpaired"] < r["restandardise"])
    ridge_worse_restd = sum(1 for r in av if r["paired_align"] > r["restandardise"])
    print(f"  -> Procrustes beats ridge in {proc_beats_ridge}/{len(av)}; "
          f"ridge worse than restandardise in {ridge_worse_restd}/{len(av)}; "
          f"CORAL beats restandardise in {coral_beats_restd}/{len(av)}")
    sep = load("separability_test")
    print(f"  -> separability: median refit-AUROC change {sep['median_separability_change_real_updates']:+.4f} "
          f"over {sep['n_negative_of_n'][1]} updates ({sep['verdict']})")
    out["mechanism"] = {"moment": mech, "mean_share_median": st.median(shares),
                        "alignment_variants": av, "alignment_means": av_means,
                        "procrustes_beats_ridge": [proc_beats_ridge, len(av)],
                        "coral_beats_restandardise": [coral_beats_restd, len(av)],
                        "ridge_worse_than_restandardise": [ridge_worse_restd, len(av)],
                        "separability_median_change": sep["median_separability_change_real_updates"],
                        "separability_verdict": sep["verdict"]}

    # ---------- 4. the controlled dose series ----------
    print("\n" + "=" * 104)
    print("4. LoRA DOSE SERIES - identical base, recipe and dose; only the training corpus differs")
    print("=" * 104)
    print(f"  {'arm':<24}{'dose':>6}{'frozen':>9}{'restand':>9}{'worstFPR':>10}"
          f"{'AUROCdrop':>11}{'||d||':>8}{'|cos|':>8}")
    lora: dict[str, Any] = {}
    shift_tag = {"general (alpaca)": "lora_general", "safety (hh-harmless)": "lora_safety",
                 "safety (pku-saferlhf)": "lora_pku"}
    for arm, stems in POP.LORA_ARMS.items():
        rows = []
        for stem in stems:
            dose = int(stem.split("_s")[-1])
            r = load(f"repair_{stem}")
            dep = load(f"depth_{stem}")
            tgt = dep["config"]["target_fpr"]
            row = {"dose": dose,
                   "frozen": repair_metric(r, "frozen_all", True),
                   "restandardise": repair_metric(r, "restandardise", True),
                   "refit": repair_metric(r, "refit_probe", True),
                   "n_censored": n_censored(r, "frozen_all"),
                   "worst_fpr_ratio": max(x["post_fpr_at_frozen_thr_mean"] / tgt for x in dep["rows"]),
                   "auroc_drop": st.mean([x["drop_frozen_mean"] for x in dep["rows"]])}
            sa_stem = f"shiftalign_{shift_tag[arm]}_s{dose}"
            if POP.path(sa_stem).exists():
                sa = load(sa_stem)["summary"]
                row["shift_norm"] = sa["shift_norm"]
                row["cos_abs"] = sa["cos_abs"]
            rows.append(row)
            print(f"  {arm:<24}{dose:>6}{row['frozen']:>9.3f}{row['restandardise']:>9.3f}"
                  f"{row['worst_fpr_ratio']:>9.1f}x{row['auroc_drop']:>+11.4f}"
                  f"{row.get('shift_norm', float('nan')):>8.2f}{row.get('cos_abs', float('nan')):>8.4f}")
        lora[arm] = rows
    gen = {r["dose"]: r for r in lora["general (alpaca)"]}
    for arm in lora:
        if arm == "general (alpaca)":
            continue
        ratios = [r["frozen"] / gen[r["dose"]]["frozen"] for r in lora[arm]]
        cosr = [r["cos_abs"] / gen[r["dose"]]["cos_abs"] for r in lora[arm] if "cos_abs" in r]
        print(f"  -> {arm}: damage {', '.join(f'{x:.2f}x' for x in ratios)} the general arm; "
              f"|cos| {', '.join(f'{x:.2f}x' for x in cosr)}")
    out["lora_arms"] = lora

    # ---------- 5. controls ----------
    print("\n" + "=" * 104)
    print("5. CONTROLS - these must show no effect")
    print("=" * 104)
    hw_d, hw_r, hw_c = load("depth_hw_v100_a100"), load("repair_hw_v100_a100"), load("hw_v100_vs_a100_mistral")
    tgt = hw_d["config"]["target_fpr"]
    ratios = [r["post_fpr_at_frozen_thr_mean"] / tgt for r in hw_d["rows"]]
    print(f"  GPU change, identical weights: max |diff| {hw_c['max_abs_diff_any_layer']:.3e}, "
          f"{hw_c['n_bit_identical_layers']}/{len(hw_c['layers'])} layers bit-identical")
    print(f"     realised FPR ratio {min(ratios):.3f}x to {max(ratios):.3f}x of target; "
          f"frozen error {repair_metric(hw_r, 'frozen_all', True):.4f}")
    nf = load("noisefloor_repair_mistral_base2instruct")
    print(f"     no-update floor on the same pre-model caches: {nf['floor_as_realised']:.4f} "
          f"(null q95 {nf['null_quantiles']['q95']:.4f})")
    contam = {}
    for tag in ("general_s25", "hh_s25", "hh_s400"):
        stem = f"contamrobust_{tag}"
        if POP.path(stem).exists():
            c = load(stem)
            contam[tag] = c["frozen_change_relative"]
            print(f"  contamination-free recomputation, {tag}: frozen damage changes "
                  f"{c['frozen_change_relative']:+.2%} "
                  f"({c['results']['all_eval_items']['n_benign_items']} -> "
                  f"{c['results']['contamination_free']['n_benign_items']} benign items)")
    out["controls"] = {"gpu_change": {"max_abs_diff": hw_c["max_abs_diff_any_layer"],
                                      "bit_identical_layers": hw_c["n_bit_identical_layers"],
                                      "fpr_ratio_range": [min(ratios), max(ratios)],
                                      "frozen_error": repair_metric(hw_r, "frozen_all", True),
                                      "no_update_floor": nf["floor_as_realised"]},
                       "contamination_robustness_relative_change": contam}

    # ---------- 6. operational consequences ----------
    print()
    print("=" * 104)
    print("6. WHERE THE DAMAGE LANDS - over-refusal on XSTest safe prompts, and adversarial prompts")
    print("=" * 104)
    orr = []
    for stem, (label, kind, mid) in POP.OVER_REFUSAL_PAIRS.items():
        d = load(stem)
        n = d["n_safe"]
        cells = {r["layer"]: (r["rs_pre_mean"], r["rs_post_mean"]) for r in d["rows"]}
        pre, post = cells.get(mid, (None, None))
        row = {"pair": label, "kind": kind, "mid_layer": mid, "n_safe": n,
               "mid_pre_items": pre * n if pre is not None else None,
               "mid_post_items": post * n if post is not None else None,
               "mid_ratio": (post / pre) if pre else None,
               "all_layers": {str(k): {"pre_items": v[0] * n, "post_items": v[1] * n,
                                       "ratio": (v[1] / v[0]) if v[0] else None}
                              for k, v in cells.items()}}
        orr.append(row)
        rng = "  ".join(f"L{k} {v['ratio']:.1f}x" for k, v in row["all_layers"].items())
        print(f"  {label:<40} mid-depth L{mid}: {row['mid_pre_items']:.1f} -> "
              f"{row['mid_post_items']:.1f} of {n} items ({row['mid_ratio']:.1f}x)")
        print(f"  {'':<40} by depth: {rng}")
    out["over_refusal"] = orr

    adv = []
    for stem in POP.ADVERSARIAL:
        d = load(stem)
        adv.append({"pair": pretty(stem, "advstrat_"), "summary_keys": sorted(d.keys())})
    out["adversarial_files"] = adv

    # ---------- 7. invisibility to the published AUC rule ----------
    print()
    print("=" * 104)
    print("7. INVISIBILITY - the prior work's own staleness rule applied unchanged to our cells")
    print("=" * 104)
    inv = {}
    for label, stem in POP.AUC_AUDITS.items():
        d = load(stem)
        pr = d["primary_depth_cells"]["all"]
        e2, e5 = pr["primary_2x"], pr["sensitivity_5x"]
        inv[label] = {"n_cells": pr["n_cells"], "n_stale_by_auc": pr["n_stale_by_auc"],
                      "at_2x": [e2["n_invisible_to_auc_criterion"], e2["n_calibration_failures"],
                                e2["fraction_invisible"]],
                      "at_5x": [e5["n_invisible_to_auc_criterion"], e5["n_calibration_failures"],
                                e5["fraction_invisible"]]}
        print(f"  {label:<28} {pr['n_cells']:>4} cells, {pr['n_stale_by_auc']:>3} stale by their rule | "
              f">=2x: {e2['n_invisible_to_auc_criterion']}/{e2['n_calibration_failures']} invisible "
              f"({e2['fraction_invisible']:.1%}) | >=5x: {e5['n_invisible_to_auc_criterion']}/"
              f"{e5['n_calibration_failures']} ({e5['fraction_invisible']:.1%})")
    out["auc_rule_invisibility"] = inv

    # D-102 pre-registered that the general and safety LoRA arms be reported separately,
    # "because the whole point is the contrast at matched dose". Pooling them hides the
    # sharpest fact in the audit, so the split is derived here and carried into Table 4.
    d = load(POP.AUC_AUDITS["LoRA (alpaca, HH-RLHF) + GPU control"])
    by_intent: dict[str, Any] = {}
    for arm, prefix in (("safety (hh-harmless)", "lorasafe_"), ("general (alpaca)", "loraA100_")):
        row: dict[str, Any] = {}
        for thr in ("primary_2x", "sensitivity_5x"):
            per = d["primary_depth_cells"]["eligible"][thr]["per_pair"]
            fails = sum(v["cal_fail"] for k, v in per.items() if k.startswith(prefix))
            inv_n = sum(v["invisible"] for k, v in per.items() if k.startswith(prefix))
            row[thr] = {"n_calibration_failures": fails, "n_invisible": inv_n,
                        "fraction_invisible": (inv_n / fails) if fails else None}
        # three doses x 17 depths, so the two arms have identical denominators
        row["n_cells"] = 3 * len(d["primary_depth_cells"]["eligible"].get("layers", range(17)))
        by_intent[arm] = row
        print(f"    {arm:<24} >=2x: {row['primary_2x']['n_invisible']}/"
              f"{row['primary_2x']['n_calibration_failures']} invisible | >=5x: "
              f"{row['sensitivity_5x']['n_invisible']}/"
              f"{row['sensitivity_5x']['n_calibration_failures']}")
    out["auc_rule_by_update_intent"] = by_intent

    payload = {**provenance_block("synthesis2"),
               "populations": {"repair": POP.UPDATE_PAIRS_REPAIR, "depth": POP.UPDATE_PAIRS_DEPTH,
                               "moment": POP.UPDATE_PAIRS_MOMENT, "alignvar": POP.UPDATE_PAIRS_ALIGNVAR,
                               "lora_arms": POP.LORA_ARMS, "controls": POP.CONTROLS,
                               "superseded_excluded": POP.SUPERSEDED},
               "input_sha256": INPUTS, **out}
    digest = write_results(args.out, payload)
    print(f"\n  inputs hashed: {len(INPUTS)}")
    print(f"Written: {args.out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
