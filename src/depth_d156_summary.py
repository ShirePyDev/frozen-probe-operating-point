"""Depth under D-156, cross-pair summary (D-180): Table 1 per pair, and the rows over all ten pairs
and over the five unflagged pairs, from the ten results/depth_d156_<pair>.json files.

Pre-registered in D-180 before any depth_d156 number existed:
  TABLE 1     per pair, probe set (selected = primary, C=1 = continuity) and level (item; cluster
              beside): the headline row (D-117 selected layer: paired ratio, ratio to target, TPR),
              the sweep maximum for both quantities with its no-update null, the lowest and the
              two-sided rows, the maximum and mean AUROC drop over layers with their item-bootstrap
              intervals, the minimum CKA, and the pair's D-170 flag.
  ACROSS      over all ten pairs (flagged, D-170) and, beside, over the five unflagged pairs (no
  PAIRS       contrast between the two sets, D-177 (1)): median and range of the headline and the
              sweep-maximum paired ratio (D-156 means) and of the maximum AUROC drop; the number of
              pairs whose headline band lies above 1; Spearman between the sweep-maximum paired
              ratio and the maximum AUROC drop.
  CKA         (author 7) each file's per-layer CKA compared with the superseded depth_<pair>.json's on
              the same input bytes: a DESCRIPTION only, never an input to any number here.
Writes results/depth_d156_summary.json (never overwritten).
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
from provenance import (ensure_utf8_stdout, provenance_block, reissue_payload, sha256_bytes, sha256_file,  # noqa: E402
                        write_results)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/depth_d156_summary.py", "src/paired.py", "src/populations.py", "src/provenance.py",
              "src/superseded_provenance.py"]
LEVELS = ("item_level_D156", "cluster_level_D164")
PROBE_SETS = ("selected", "C=1")
PSET_ROLE = {"selected": "PRIMARY: the D-117 selected C (D-180)", "C=1": "continuity row: C=1, file order (D-180)"}
LEVEL_ROLE = {"item_level_D156": "the estimator (D-156)",
              "cluster_level_D164": "cluster-level sensitivity, beside the item level, never instead (D-164)"}
TWOMOMENT_FIT_SHA256 = "515faab949465d0852cd61dbc1c7805cf3d800c5e2222029e4622261727a3b96"   # D-174
# D-300: the record --r2 re-issues (it names no repo ID or revision for its cache inputs, D-297 (1))
FIRST_RECORD = ("results/depth_d156_summary.json", "efabb2181946617d944c10e9deec3251073da28b84476b98df3c7a4bfb3de3b7")
ROPE_NOTE = ("not a pure weight update: rope_theta 10000.0 -> 1000000.0 and sliding_window 4096 -> None (Phase 0 "
             "audit); the paper's pair list must state the RoPE-base change (D-180 author 4)")
# D-180: the five pairs whose inputs carry no D-170 flag (checked against the files' own flags).
UNFLAGGED = ["depth_gemma2_base2instruct", "depth_llama31_base2instruct", "depth_olmo_base2instruct",
             "depth_WG_mistral_base2instruct", "depth_WG_llama3_base2instruct"]


def out_name_of(depth_stem: str) -> str:
    """The depth_d156 result stem of a UPDATE_PAIRS_DEPTH pair (its config names the revisions)."""
    return f"depth_d156_{depth_stem[len('depth_'):]}"


def spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman's rho (average ranks for ties); None below 3 pairs or with no spread."""
    if len(x) < 3:
        return None
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Depth under D-156: cross-pair summary (D-180)")
    ap.add_argument("--results-dir", default=str(RESULTS), help="where the ten depth_d156 files are")
    ap.add_argument("--out", default=None, help="default results/depth_d156_summary.json (_r2 with --r2)")
    ap.add_argument("--r2", action="store_true",
                    help="re-issue naming each cache input's repo ID and revision, checked equal to the first record (D-300)")
    ap.add_argument("--test-unregistered", action="store_true",
                    help="TESTING ONLY: accept depth_d156 files run with non-registered parameters; refused "
                         "unless --out is outside results/")
    args = ap.parse_args()
    out = Path(args.out) if args.out else RESULTS / f"depth_d156_summary{'_r2' if args.r2 else ''}.json"
    if out.exists():
        raise SystemExit(f"REFUSING: {out} exists; results are never overwritten")
    if args.r2 and out.resolve().is_relative_to(RESULTS.resolve()) and out.name != "depth_d156_summary_r2.json":
        raise SystemExit("REFUSING: under results/, --r2 writes only depth_d156_summary_r2.json")
    if args.test_unregistered and out.resolve().is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: --test-unregistered may not write under results/")
    rdir = Path(args.results_dir).resolve()
    if rdir != RESULTS.resolve() and not args.test_unregistered:
        raise SystemExit("REFUSING: the summary of record reads the ten files in results/ only (--results-dir is for tests)")
    code0 = P.code_block(CODE_FILES)

    files = {s: rdir / f"{out_name_of(s)}.json" for s in POP.UPDATE_PAIRS_DEPTH}
    missing = [str(f) for f in files.values() if not f.exists()]
    if missing:
        raise SystemExit(f"REFUSING: depth_d156 files missing: {missing}")
    raw = {s: f.read_bytes() for s, f in files.items()}              # hash and parse ONE read
    sha = {s: sha256_bytes(b) for s, b in raw.items()}
    docs = {s: json.loads(b.decode("utf-8")) for s, b in raw.items()}
    flagged = {s: d["input_verification"]["flag"] is not None for s, d in docs.items()}
    unflagged_ok = sorted(s for s, f in flagged.items() if not f) == sorted(UNFLAGGED)
    if not unflagged_ok and not args.test_unregistered:
        raise SystemExit(f"REFUSING: the unflagged pairs are not D-180's: {sorted(s for s, f in flagged.items() if not f)}")
    problems = []
    for s, d in docs.items():
        if d.get("run_kind") != "depth_d156" or d["config"]["pair"] != s:
            problems.append(f"{files[s].name}: not the depth_d156 run of {s}")
        if not args.test_unregistered:       # the final-name conditions of depth_d156, per file
            if not d["registered_parameters"]:
                problems.append(f"{s}: non-registered parameters")
            if d["config"]["selection_file"] != "results/selection_sweep_d165.json":
                problems.append(f"{s}: not the selection of record")
            if d["twomoment_check"].get("twomoment_fit_sha256") != TWOMOMENT_FIT_SHA256:
                problems.append(f"{s}: not checked against the D-174 fit")
            if any(v != "6" for k, v in d["env"]["blas_threads"].items() if k != "NUMEXPR_NUM_THREADS"):
                problems.append(f"{s}: no 6-thread BLAS pin")
    # one code, one environment, one selection across the ten files (the D-174 precedent)
    ref = docs[POP.UPDATE_PAIRS_DEPTH[0]]
    for key, get in (("code.sha256", lambda d: d["code"]["sha256"]), ("env.libraries", lambda d: d["env"]["libraries"]),
                     ("env.blas_threads", lambda d: d["env"]["blas_threads"]),
                     ("config.selection_sha256", lambda d: d["config"]["selection_sha256"])):
        odd = [s for s, d in docs.items() if get(d) != get(ref)]
        if odd:
            problems.append(f"{key} differs across the files: {odd}")
    if problems and not args.test_unregistered:
        raise SystemExit("REFUSING:\n  " + "\n  ".join(problems))
    flag_text = next((d["input_verification"]["flag"] for d in docs.values() if d["input_verification"]["flag"]), None)

    def row_flag(pairs: list[str]) -> dict[str, Any]:
        n = sum(flagged[s] for s in pairs)
        return {"n_pairs_flagged": n, "flag": flag_text if n else None}

    table: dict[str, Any] = {}
    for s, d in docs.items():
        ps = d["pair_summary"]
        table[s] = {"input_verification": row_flag([s]), "axis": d["config"]["axis"],
                    "selected_layer": d["config"]["selected_layer"], "selected_C": d["config"]["selected_C"],
                    "selection_of_pair": d["config"]["selection_of_pair"],
                    **({"note": ROPE_NOTE} if s == "depth_mistral_v01_v02_dense" else {}),
                    **{pset: {"role": PSET_ROLE[pset],
                              **{lvl: ps[pset][lvl] for lvl in LEVELS},          # every D-180 row, whole
                              "auroc_drop_frozen": ps[pset]["auroc_drop_frozen"],
                              "min_cka_over_layers": ps[pset]["min_cka_over_layers"]} for pset in PROBE_SETS}}

    def across(pairs: list[str], pset: str, lvl: str) -> dict[str, Any]:
        head = [table[s][pset][lvl]["headline_selected_layer"]["paired_ratio"] for s in pairs]
        sweep = [table[s][pset][lvl]["sweep_maximum"]["paired_ratio"]["paired_ratio"]["mean"] for s in pairs]
        drop = [table[s][pset]["auroc_drop_frozen"]["max_over_layers"]["point"] for s in pairs]
        stat = lambda v: {"median": float(np.median(v)), "range": [float(min(v)), float(max(v))]}  # noqa: E731
        return {"n_pairs": len(pairs), "pairs": pairs, "input_verification": row_flag(pairs),
                "probe_set_role": PSET_ROLE[pset], "level_role": LEVEL_ROLE[lvl],
                "headline_paired_ratio_mean": stat([h["mean"] for h in head]),
                "sweep_maximum_paired_ratio_mean": stat(sweep),
                "max_auroc_drop": stat(drop),
                "n_pairs_headline_band_above_1": int(sum(h["band95"][0] > 1 for h in head)),
                "spearman_sweep_maximum_paired_ratio_vs_max_auroc_drop": spearman(sweep, drop)}

    all_pairs = list(POP.UPDATE_PAIRS_DEPTH)
    rows = {pset: {lvl: {"all_ten_pairs": across(all_pairs, pset, lvl),
                         "unflagged_five_pairs": {**across(UNFLAGGED, pset, lvl),
                                                  "note": "beside the all-pairs row; no contrast with the flagged pairs (D-177 (1))"}}
                   for lvl in LEVELS} for pset in PROBE_SETS}

    # author 7: CKA against the superseded curves, same input bytes - a description only
    cka_cmp: dict[str, Any] = {}
    old_sha: dict[str, str] = {}
    for s, d in docs.items():
        old = RESULTS / f"{s}.json"
        if not old.exists():
            cka_cmp[s] = "superseded file not on disk"
            continue
        ob = old.read_bytes()                                         # hash and parse ONE read
        old_sha[s] = sha256_bytes(ob)
        od = json.loads(ob.decode("utf-8"))
        old_cka = {r["layer"]: r["cka"] for r in od["rows"]}
        new_cka = {c["layer"]: c["cka_frozen_eval_split"] for c in d["cells"] if c["probe_set"] == "selected"}
        common = sorted(set(old_cka) & set(new_cka))
        diffs = [abs(new_cka[li] - old_cka[li]) for li in common]
        cka_cmp[s] = {"superseded_file": old.name, "sha256": old_sha[s],
                      # CKA reads only the two eval caches; equality as recorded by the superseded file
                      "same_eval_cache_bytes_as_recorded": {k: (od.get("input_cache_sha256") or {}).get(k) == d["input_cache_sha256"][k]
                                                            for k in ("pre/eval", "post/eval")},
                      "n_layers_old": len(old_cka), "n_layers_new": len(new_cka),
                      "n_layers_compared": len(common), "n_layers_equal": int(sum(x == 0 for x in diffs)),
                      "max_abs_diff": float(max(diffs)) if diffs else None}

    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"] or any(sha256_file(f) != sha[s] for s, f in files.items()) or \
            any(sha256_file(RESULTS / f"{s}.json") != h for s, h in old_sha.items()):
        raise SystemExit("REFUSING to write: code, a depth_d156 file or a compared superseded file changed during the run")
    rel = lambda p: str(p.relative_to(ROOT)).replace("\\", "/") if p.is_relative_to(ROOT) else str(p)  # noqa: E731
    payload = {**provenance_block("depth_d156_summary"), "preregistration": "D-180", "code": code0,
               "test_unregistered": args.test_unregistered, "unflagged_set_matches_d180": unflagged_ok,
               "test_problems_not_refused": problems if args.test_unregistered else [],
               "input_verification": {**row_flag(all_pairs), "n_pairs": len(flagged)},
               "per_pair_code_commits": {s: d["code"]["commit"] for s, d in docs.items()},
               "input_files_sha256": {rel(files[s]): h for s, h in sha.items()},
               "input_cache_sha256": {f"{out_name_of(s)}:{k}": v for s, d in docs.items() for k, v in d["input_cache_sha256"].items()},
               "table_1": table, "across_pairs": rows,
               "cka_vs_superseded_description_only": cka_cmp}
    if args.r2:
        payload = reissue_payload(payload, *FIRST_RECORD)
    digest = write_results(out, payload, exclusive=True)
    print(f"Written: {out}\nsha256:  {digest}")
    for pset in PROBE_SETS:
        for key, r in rows[pset][LEVELS[0]].items():
            print(f"[{pset} item level] {key}: headline paired ratio median {r['headline_paired_ratio_mean']['median']:.3f} "
                  f"range {[round(v, 3) for v in r['headline_paired_ratio_mean']['range']]}; bands above 1: "
                  f"{r['n_pairs_headline_band_above_1']}/{r['n_pairs']}; sweep max median "
                  f"{r['sweep_maximum_paired_ratio_mean']['median']:.3f}; max AUROC drop median {r['max_auroc_drop']['median']:.4f}; "
                  f"Spearman {r['spearman_sweep_maximum_paired_ratio_vs_max_auroc_drop']}")
    print("CKA vs superseded curves:", {s: (v if isinstance(v, str) else f"{v['n_layers_equal']}/{v['n_layers_compared']} equal, "
                                          f"max diff {v['max_abs_diff']}") for s, v in cka_cmp.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
