"""Do the two pilots whose inputs were overwritten (F-024) still reproduce?

verify_provenance.py found pilot_e6.json and pilot_e6_qwen.json were computed from
caches later replaced by denser re-extractions of the same model. This compares
each original record against a re-run on the caches now on disk.

  pilot_e6        re-run with IDENTICAL arguments. Layers 16 and 24 exist in the
                  dense grid, so if extraction is deterministic the numbers must
                  match exactly. Any difference is reported, not rounded away.

  pilot_e6_qwen   CANNOT be re-run exactly: it used Qwen2.5 layer 21 for the
                  similarity layer, and the dense grid holds even layers only.
                  Re-run at the neighbours, 20 and 22. The test is qualitative:
                  do the phenomena D-037/D-038/D-039 were drawn from - beta pinned
                  at its bound or undefined, a consistently signed coupling term,
                  and the similarity layer's flag rate collapsing - persist?

Reads results only.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import ensure_utf8_stdout, provenance_block, write_results  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def seed_view(d: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for k in sorted(d["seeds"], key=int):
        s = d["seeds"][k]
        dc = s["decomposition"]
        post = s["layers"]["post"]
        out[k] = {
            "beta_pre": dc["beta_pre"]["value"],
            "beta_post": dc["beta_post"]["value"],
            "delta_beta": dc["delta_beta"],
            "coupling_term": dc["coupling_term"],
            "p_all_pre": dc["p_all_pre"],
            "p_all_post": dc["p_all_post"],
            "L1_auroc_post": post["L1_probe"]["auroc"],
            "L3_auroc_post": post["L3_sim"]["auroc"],
            "L3_flag_pos_post": post["L3_sim"]["eval_flag_rate_pos"],
            "L1_calib_fpr_post": post["L1_probe"]["calib_fpr"],
        }
    return out


def maxdiff(a: dict, b: dict) -> tuple[float, int, int]:
    worst, n, n_diff = 0.0, 0, 0
    for seed in a:
        for key, va in a[seed].items():
            vb = b.get(seed, {}).get(key)
            n += 1
            if va is None or vb is None:
                if (va is None) != (vb is None):
                    n_diff += 1
                    worst = math.inf
                continue
            d = abs(va - vb)
            if d != 0.0:
                n_diff += 1
            worst = max(worst, d)
    return worst, n, n_diff


def main() -> int:
    ensure_utf8_stdout()
    report: dict[str, Any] = {}

    print("=" * 84)
    print("pilot_e6: original vs exact re-run on current caches")
    print("=" * 84)
    orig = json.load(open(RESULTS / "pilot_e6.json", encoding="utf-8"))
    rer = json.load(open(RESULTS / "pilot_e6_RERUN.json", encoding="utf-8"))
    a, b = seed_view(orig), seed_view(rer)
    worst, n, nd = maxdiff(a, b)
    verdict = "BIT-EXACT" if nd == 0 else ("NUMERICALLY EQUAL (<1e-9)" if worst < 1e-9 else "DIFFERS")
    print(f"  compared {n} quantities across {len(a)} seeds: {nd} differ, max |diff| = {worst:.3e}")
    print(f"  VERDICT: {verdict}")
    report["pilot_e6"] = {"n_compared": n, "n_differ": nd, "max_abs_diff": worst, "verdict": verdict}
    if nd:
        for seed in a:
            for key in a[seed]:
                if a[seed][key] != b[seed][key]:
                    print(f"    seed {seed} {key}: {a[seed][key]} -> {b[seed][key]}")

    print()
    print("=" * 84)
    print("pilot_e6_qwen: original (sim layer 21) vs re-runs at neighbouring layers 20 and 22")
    print("=" * 84)
    q = {"21 (orig)": seed_view(json.load(open(RESULTS / "pilot_e6_qwen.json", encoding="utf-8")))}
    for sl in (20, 22):
        p = RESULTS / f"pilot_e6_qwen_RERUN_sim{sl}.json"
        if p.exists():
            q[f"{sl} (rerun)"] = seed_view(json.load(open(p, encoding="utf-8")))
    fmt = lambda x: "None" if x is None else f"{x:+.4f}"  # noqa: E731
    phen: dict[str, Any] = {}
    for label, view in q.items():
        bps = [v["beta_post"] for v in view.values()]
        cts = [v["coupling_term"] for v in view.values()]
        fl = [v["L3_flag_pos_post"] for v in view.values()]
        cf = [v["L1_calib_fpr_post"] for v in view.values()]
        pinned = sum(1 for x in bps if x is None or abs(abs(x) - 1.0) < 1e-9)
        phen[label] = {
            "beta_post_by_seed": bps,
            "seeds_beta_at_bound_or_undefined": pinned,
            "coupling_term_signs": sorted({("+" if c > 0 else "-" if c < 0 else "0") for c in cts}),
            "L3_flag_rate_harmful_post_max": max(fl),
            "L1_calib_fpr_post_mean": sum(cf) / len(cf),
        }
        print(f"  sim layer {label:<11} beta_post by seed: {[fmt(x) for x in bps]}")
        print(f"  {'':<21} beta at bound or undefined: {pinned}/{len(bps)} seeds   "
              f"coupling signs: {phen[label]['coupling_term_signs']}")
        print(f"  {'':<21} L3 harmful flag rate post (max over seeds): "
              f"{phen[label]['L3_flag_rate_harmful_post_max']:.4f}   "
              f"L1 calib FPR post: {phen[label]['L1_calib_fpr_post_mean']:.4f}")
    report["pilot_e6_qwen"] = phen

    digest = write_results(RESULTS / "compare_reruns.json",
                           {**provenance_block("compare_reruns"), **report})
    print(f"\nWritten: results/compare_reruns.json\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
