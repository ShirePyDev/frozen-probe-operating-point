"""Item 8 additions (D-205, pre-registered): stratum sizes and floored draws, and the draft's adversarial
fail-open claim tested at the selected C. Reads only results files; no new draws.

  (1) Per WildGuardMix pair and stratum (adversarial, vanilla): benign items (pooled negatives), harmful
      items (evaluation positives), and the share of draws floored at the deployer's layer, selected C,
      item and cluster level, from results/advstrat_d156_<pair>.json.
  (2) FAIL-OPEN CELLS: every layer cell of results/depth_d156_WG_<pair>_base2instruct.json at the selected C
      with D-156 mean paired ratio <= 0.5 AND a 95% band whose upper end is below 1 (item level; the cluster
      level beside). For each, the stratification at that layer from the advstrat file (it holds every
      layer at selected C). If no cell qualifies, the record says the draft's adversarial fail-open claim
      has no cell at the selected C.
Writes results/advstrat_failopen_d205.json (never overwritten).
--r2 (D-281 (1), D-282): reads the re-issued results/advstrat_d156_<pair>_r2.json instead. Their numbers equal the
first records' (each r2 file's reissue_of check). Beside each fail-open cell's item-level stratification it adds the
D-282 combined interval for per-stratum TPR and the delta-TPR contrast. Writes results/advstrat_failopen_d205_r2.json,
which supersedes the first record (never deleted).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as P  # noqa: E402
from provenance import ensure_utf8_stdout, provenance_block, sha256_bytes, sha256_file, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/advstrat_failopen.py", "src/paired.py", "src/provenance.py"]
LEVELS = ("item_level_D156", "cluster_level_D164")
STRATA = ("adversarial", "vanilla")
PAIRS = {"mistral": "WG_mistral_base2instruct", "llama3": "WG_llama3_base2instruct"}
CUT = 0.5                                                            # D-205 (2)
MANIFESTS = {"depth": ROOT / "logs" / "depth_d156_20260926.sha256", "advstrat": ROOT / "logs" / "advstrat_d156_20260926.sha256"}
R2_MANIFEST = ROOT / "logs" / "advstrat_d156_r2_20260930.sha256"       # the 1% re-issues (D-282 (3); queues/laneB_20260930d)
D282_KEY = "item_bootstrap_D282"


def manifest(p: Path) -> dict[str, str]:
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, f = line.split(maxsplit=1)
            out[f.strip().lstrip("*")] = h
    return out


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Item 8 additions (D-205)")
    ap.add_argument("--r2", action="store_true", help="read the D-282 re-issues and add their combined TPR intervals")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    final_path = (RESULTS / f"advstrat_failopen_d205{'_r2' if args.r2 else ''}.json").resolve()
    out = Path(args.out) if args.out else final_path
    if out.exists():
        raise SystemExit(f"REFUSING: {out} exists; results are never overwritten")
    final = out.resolve() == final_path
    if not final and out.resolve().is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test output must not be under results/")
    code0 = P.code_block(CODE_FILES)
    man_paths = {**MANIFESTS, **({"advstrat": R2_MANIFEST, "advstrat_first": MANIFESTS["advstrat"]} if args.r2 else {})}
    missing = [str(p.relative_to(ROOT)).replace("\\", "/") for p in man_paths.values() if not p.is_file()]
    if missing:
        raise SystemExit(f"REFUSING: {missing} not on disk (for --r2: the 1% re-issues have not been pulled yet)")
    mans = {k: manifest(v) for k, v in man_paths.items()}
    inputs: dict[str, str] = {str(v.relative_to(ROOT)).replace("\\", "/"): sha256_file(v) for v in man_paths.values()}
    sizes: dict[str, Any] = {}
    cells_out: dict[str, Any] = {}
    for stem, wg in PAIRS.items():
        a_rel, d_rel = f"results/advstrat_d156_{stem}{'_r2' if args.r2 else ''}.json", f"results/depth_d156_{wg}.json"
        docs = {}
        for rel, kind in ((a_rel, "advstrat"), (d_rel, "depth")):
            b = (ROOT / rel).read_bytes()
            h = sha256_bytes(b)
            if mans[kind].get(rel) != h:
                raise SystemExit(f"REFUSING: {rel} is not the file {man_paths[kind].name} names")
            inputs[rel] = h
            docs[kind] = json.loads(b.decode("utf-8"))
        adv, dep = docs["advstrat"], docs["depth"]
        if args.r2:
            ro, first = adv.get("reissue_of") or {}, f"results/advstrat_d156_{stem}.json"
            if ro.get("compared") is not True or ro.get("file") != first or ro.get("sha256") != mans["advstrat_first"].get(first):
                raise SystemExit(f"REFUSING: {a_rel} carries no passed re-issue check of {first} as "
                                 f"{MANIFESTS['advstrat'].name} lists it (D-282 (3))")
            if any(D282_KEY not in c["item_level_D156"] for c in adv["cells"]):
                raise SystemExit(f"REFUSING: {a_rel} has a cell without its D-282 block")
        if adv["config"]["registry_pair"] != f"repair_{wg}" or dep["config"]["registry_pair"] != f"repair_{wg}":
            raise SystemExit(f"REFUSING: {stem}: the advstrat and depth files are not the same pair")
        C_sel, L_sel = dep["config"]["selected_C"], dep["config"]["selected_layer"]
        if (adv["config"]["selected_C"], adv["config"]["selected_layer"]) != (C_sel, L_sel):
            raise SystemExit(f"REFUSING: {stem}: the two files carry different selections")
        # (1) sizes and floored shares at the deployer's layer, selected C
        h = adv["headline_selected_layer"]["selected"]
        sizes[stem] = {"layer": L_sel, "selected_C": C_sel,
                       "strata_sizes": {s: {"benign_pooled_negatives": adv["config"]["strata_sizes"][s]["pooled_negatives"],
                                            "harmful_eval_positives": adv["config"]["strata_sizes"][s]["eval_positives"]} for s in STRATA},
                       "share_of_draws_floored": {lvl: {s: h[lvl][s]["share_of_draws_floored"] for s in STRATA} for lvl in LEVELS},
                       "frozen_split_floored": {lvl: {s: h[lvl][s]["frozen_split_floored"] for s in STRATA} for lvl in LEVELS}}
        # (2) the fail-open cells, then the stratification there
        a_cells = {(c["layer"], c["probe_set"]): c for c in adv["cells"]}
        # the two files must describe the same grid at the same selected C (review)
        d_layers = sorted(c["layer"] for c in dep["cells"] if c["probe_set"] == "selected")
        a_layers = sorted(li for li, ps in a_cells if ps == "selected")
        if d_layers != a_layers or d_layers != sorted(adv["config"]["layers"]):
            raise SystemExit(f"REFUSING: {stem}: the depth and advstrat files hold different layer grids")
        if any(c["probe_C"] != C_sel for c in dep["cells"] + adv["cells"] if c["probe_set"] == "selected"):
            raise SystemExit(f"REFUSING: {stem}: a selected-C cell carries another C than {C_sel}")
        hits = []
        for c in dep["cells"]:
            if c["probe_set"] != "selected":
                continue
            pr = c["operating_point"]["paired_ratio"]
            prc = c["cluster_level_D164"]["operating_point"]["paired_ratio"]
            if pr["mean"] <= CUT and pr["band95"][1] < 1.0:
                ac = a_cells[(c["layer"], "selected")]
                hits.append({"layer": c["layer"], "paired_ratio_item_level": pr, "paired_ratio_cluster_level": prc,
                             "cluster_level_also_meets_criterion": prc["mean"] <= CUT and prc["band95"][1] < 1.0,
                             "stratification": {lvl: {**{s: {q: ac[lvl][s][q] for q in ("paired_ratio", "ratio_to_target", "tpr_pre",
                                                                                          "tpr_post", "delta_tpr", "share_of_draws_floored")}
                                                          for s in STRATA},
                                                      "contrast_adversarial_minus_vanilla": ac[lvl]["contrast_adversarial_minus_vanilla"],
                                                      # D-282: the item level only; the cluster level keeps its band
                                                      **({"combined_interval_D282": ac[lvl][D282_KEY]}
                                                         if args.r2 and lvl == LEVELS[0] else {})}
                                                for lvl in LEVELS}})
        n_sel = sum(c["probe_set"] == "selected" for c in dep["cells"])
        cells_out[stem] = {"selected_C": C_sel, "n_layer_cells_examined": n_sel, "fail_open_cells": hits,
                           "finding": f"{len(hits)} cell(s) meet the criterion" if hits else "no cell in this pair",
                           "description_not_preregistered": {
                               "lowest_paired_ratio_cell": min(({"layer": c["layer"], "paired_ratio": c["operating_point"]["paired_ratio"]}
                                                                for c in dep["cells"] if c["probe_set"] == "selected"),
                                                               key=lambda r: r["paired_ratio"]["mean"])}}
        low = cells_out[stem]["description_not_preregistered"]["lowest_paired_ratio_cell"]
        print(f"{stem}: {cells_out[stem]['finding']} ({n_sel} layer cells at C={C_sel}); [description] lowest mean paired "
              f"ratio {low['paired_ratio']['mean']:.3f} at L{low['layer']}", flush=True)
    any_hit = any(v["fail_open_cells"] for v in cells_out.values())
    hit_list = ", ".join("{} L{}".format(s, h["layer"]) for s, v in cells_out.items() for h in v["fail_open_cells"])
    overall = ("fail-open cells: " + hit_list) if any_hit else "the draft's adversarial fail-open claim has no cell at the selected C"
    print(f"FINDING (D-205 (2)): {overall}", flush=True)
    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"] or any(sha256_file(ROOT / f) != h for f, h in inputs.items()):
        raise SystemExit("REFUSING to write: code or an input changed during the run")
    digest = write_results(out, {**provenance_block("advstrat_failopen"),
                                 "preregistration": "D-205" + ("; D-281 (1), D-282 (the combined TPR interval)" if args.r2 else ""),
                                 **({"supersedes": ["results/advstrat_failopen_d205.json (D-282: the r2 inputs, whose "
                                                    "numbers are the first records', plus the combined TPR intervals)"]}
                                    if args.r2 else {}),
                                 "code": code0,
                                 "criterion": {"paired_ratio_mean_at_most": CUT, "band95_upper_below": 1.0, "level": LEVELS[0],
                                               "probe_set": "selected C (D-117)"},
                                 "input_files_sha256": inputs, "strata_sizes_and_floored_draws": sizes,
                                 "finding": overall, "fail_open": cells_out}, exclusive=True)
    print(f"Written: {out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
