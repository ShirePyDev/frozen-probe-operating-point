"""Item 9 at the deployer's layer (D-206 (2)): the over-refusal table with the D-117 selected layer as PRIMARY,
read from the item-9 records; the declared mid layer (the table reported so far) as the secondary.

For each of the seven over-refusal pairs the D-117 cell comes from results/overrefusal_d156_<pair>.json when
that file's layers hold it (five pairs), else from the tagged full-grid re-run
results/overrefusal_d156_<pair>_xsfull.json (Qwen2.5, Mistral v0.1 -> v0.2). CHECK, not a criterion: at every
layer a re-run shares with the original file (selected C, both levels), the safe prompts flagged pre and post
must agree within the D-202/D-204 XSTest bound (0.001 of the 250 safe prompts): the new A100 full-grid cache
against the pinned original. Writes results/overrefusal_d117_table.json (never overwritten).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d168_rules as R  # noqa: E402
import paired as P  # noqa: E402
from provenance import ensure_utf8_stdout, provenance_block, sha256_bytes, sha256_file, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/overrefusal_d117_table.py", "src/d168_rules.py", "src/paired.py", "src/provenance.py"]
XS_TAG = "d168a100_20260926b"
SELECTION = RESULTS / "selection_sweep_d165.json"
MANIFEST = ROOT / "logs" / "overrefusal_d156_20260926.sha256"
XS_MANIFEST = ROOT / "logs" / "overrefusal_d156_xsfull.sha256"
STEMS = ("mistral", "gemma2", "llama3", "llama31", "qwen", "mistral_v01_v02", "llama3_to_31")
LEVELS = ("item_level_D156", "cluster_level_D164")
QUANT = ("safe_flagged_pre_mean", "safe_flagged_post_mean", "refusal_change", "paired_refusal_ratio", "contrast_flag_change")


def manifest(p: Path) -> dict[str, str]:
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                h, f = line.split(maxsplit=1)
                out[f.strip().lstrip("*")] = h
    return out


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Item 9 at the D-117 layer (D-206 (2))")
    ap.add_argument("--out", default=str(RESULTS / "overrefusal_d117_table.json"))
    ap.add_argument("--xs-bounds", required=True,
                    help="the XSTest control-shift record the re-run check uses, regenerated under the current "
                         "src/overrefusal_d156.py (D-217 (f)), e.g. results/d168_a100_xstest_control_shift_<tag>_<rev>.json")
    args = ap.parse_args()
    out = Path(args.out)
    XS_CONTROL = Path(args.xs_bounds)
    if not R.is_control_record(XS_CONTROL, RESULTS, "xstest", XS_TAG):
        raise SystemExit(f"REFUSING: {XS_CONTROL} is not an XSTest control-shift record of tag {XS_TAG}")
    if out.exists():
        raise SystemExit(f"REFUSING: {out} exists")
    if out.resolve() != (RESULTS / "overrefusal_d117_table.json").resolve() and out.resolve().is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test output must not be under results/")
    code0 = P.code_block(CODE_FILES)
    man, xman = manifest(MANIFEST), manifest(XS_MANIFEST)
    inputs = {"results/selection_sweep_d165.json": sha256_file(SELECTION)}
    # the re-run check's bound, D-202 (3) / D-204 (iii): max(2 x the XSTest control shift, 0.001) (src/d168_rules.py)
    ctrl = json.loads(XS_CONTROL.read_text(encoding="utf-8"))
    if ctrl.get("run_kind") != "d168_xstest_check" or ctrl.get("mode") != "control":
        raise SystemExit(f"REFUSING: {XS_CONTROL} is not an XSTest control-shift record")
    # D-217 (f): the bound must come from the CURRENT over-refusal code
    if ctrl["code"]["sha256"].get("src/overrefusal_d156.py") != sha256_file(ROOT / "src" / "overrefusal_d156.py"):
        raise SystemExit(f"REFUSING: {XS_CONTROL} was measured under another src/overrefusal_d156.py; regenerate it (D-217 (f))")
    inputs[str(XS_CONTROL.relative_to(ROOT)).replace("\\", "/")] = sha256_file(XS_CONTROL)
    xs_bound = max(R.bound(m, v["control_shift"]) for m, v in ctrl["bounds"].items())
    sel = {m["pair"]: m["selected"] for m in json.loads(SELECTION.read_text(encoding="utf-8"))["models"]}
    rows, checks = {}, {}
    for stem in STEMS:
        rel = f"results/overrefusal_d156_{stem}.json"
        b = (ROOT / rel).read_bytes()
        if man.get(rel) != sha256_bytes(b):
            raise SystemExit(f"REFUSING: {rel} is not the file {MANIFEST.name} names")
        inputs[rel] = sha256_bytes(b)
        old = json.loads(b.decode("utf-8"))
        cfg = old["config"]
        arm, L = sel[cfg["selection_of_pair"]]["arm"], sel[cfg["selection_of_pair"]]["layer"]
        if float(arm.split("=")[1]) != cfg["selected_C"]:
            raise SystemExit(f"REFUSING: {stem}: the item-9 file's selected C is not the selection's")
        src, doc = rel, old
        if L not in cfg["layers"]:
            xrel = f"results/overrefusal_d156_{stem}_xsfull.json"
            if not (ROOT / xrel).exists():
                raise SystemExit(f"REFUSING: {stem}: L{L} is not in {rel}, and {xrel} does not exist yet (D-206 (1) first)")
            xb = (ROOT / xrel).read_bytes()
            if xman.get(xrel) != sha256_bytes(xb):
                raise SystemExit(f"REFUSING: {xrel} is not the file {XS_MANIFEST.name} names")
            inputs[xrel] = sha256_bytes(xb)
            doc, src = json.loads(xb.decode("utf-8")), xrel
            if doc.get("xstest_cache_tag") != "xsfull" or L not in doc["config"]["layers"]:
                raise SystemExit(f"REFUSING: {xrel} is not a tagged full-grid re-run holding L{L}")
            if (doc["config"]["selected_C"], doc["config"]["selection_sha256"]) != (cfg["selected_C"], cfg["selection_sha256"]):
                raise SystemExit(f"REFUSING: {xrel} used another selection than {rel}")
            # the check: the re-run against the original at their shared layers
            oc = {(c["layer"], c["probe_set"]): c for c in old["cells"]}
            n = doc["config"]["n_safe"]
            diffs = []
            for c in doc["cells"]:
                if c["probe_set"] != "selected" or (c["layer"], "selected") not in oc:
                    continue
                for lvl in LEVELS:
                    for q in ("safe_flagged_pre_mean", "safe_flagged_post_mean"):
                        diffs.append({"layer": c["layer"], "level": lvl, "number": q,
                                      "abs_diff_share": abs(c[lvl][q]["mean"] - oc[(c["layer"], "selected")][lvl][q]["mean"]) / n})
            checks[stem] = {"shared_layers": sorted({d["layer"] for d in diffs}), "bound_share": xs_bound,
                            "max_abs_diff_share": max((d["abs_diff_share"] for d in diffs), default=None),
                            "all_within": all(d["abs_diff_share"] <= xs_bound for d in diffs), "cells": diffs}
            print(f"  check {stem}: re-run vs original at shared layers {checks[stem]['shared_layers']}: max |d share| "
                  f"{checks[stem]['max_abs_diff_share']} vs bound {xs_bound}; all within: {checks[stem]['all_within']}", flush=True)
        cell = {ps: next(c for c in doc["cells"] if c["layer"] == L and c["probe_set"] == ps) for ps in ("selected", "C=1")}
        rows[stem] = {"source": src, "d117_layer": L, "selected_C": cfg["selected_C"],
                      "flag": doc["input_verification"]["flag"],
                      "primary": {ps: {lvl: {q: cell[ps][lvl][q] for q in QUANT} for lvl in LEVELS} | {"xstest_auroc_change": cell[ps]["xstest_auroc"]["change"]}
                                  for ps in cell},
                      "secondary_mid_layer": {"layer": old["headline_mid_layer"]["selected"]["layer"],
                                              "selected": {q: old["headline_mid_layer"]["selected"][LEVELS[0]][q] for q in QUANT}}}
        h = rows[stem]["primary"]["selected"][LEVELS[0]]
        print(f"{stem:17s} L{L:>2}: safe flagged {h['safe_flagged_pre_mean']['mean']:.1f} -> {h['safe_flagged_post_mean']['mean']:.1f}; "
              f"refusal change {h['refusal_change']['mean']:+.3f}; paired ratio {h['paired_refusal_ratio']['mean']:.2f}  [{src}]", flush=True)
    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"] or any(sha256_file(ROOT / f) != h for f, h in inputs.items()):
        raise SystemExit("REFUSING to write: code or an input changed during the run")
    digest = write_results(out, {**provenance_block("overrefusal_d117_table"), "preregistration": "D-206 (2)", "code": code0,
                                 "input_files_sha256": inputs, "primary": "the D-117 selected layer (selected C; C=1 beside)",
                                 "secondary": "the declared mid layer (the table reported before D-206)",
                                 "rows": rows, "rerun_checks": checks}, exclusive=True)
    print(f"Written: {out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
