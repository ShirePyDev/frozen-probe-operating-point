"""Score files for the author's gates (D-213, D-214): for one unflagged cell of each remaining primary table, the
per-seed frozen thresholds and the scores they are applied to, so that scores -> threshold -> counts can be
re-derived independently of this code.

Every table below shares one code path, re-done here exactly as its script does it: the D-117 selected probes
(paired.d117_split; paired.fit_probe on the pre model's probe_train at the selected C, rows in d117_permutation
order), each seed's FROZEN threshold = paired.thr_at_fpr on its scores of the pre model's threshold_calib
negatives (the frozen split = draw 0 of D-156; target 1% FPR), and flags = score >= threshold on the eval
negatives and eval positives, pre and post. The subsets are scored separately, as the table scripts score them.
  depth      results/depth_d156_llama31_base2instruct.json    L14, selected C   (Table 1, headline selected layer)
  intent     results/damage_d156_r2.json, 4a lorasafe_mistral_s25  L12, selected C   (4a)
  mechanism  results/twomoment_repair_olmo_base2instruct.json  L14, selected C   (4b realised counts)
  failopen   results/advstrat_d156_mistral.json (WildGuardMix) L32, selected C   (item 8, per stratum)
ANCHORS (refuses to write otherwise): the table file is the committed one (sha256 below); every input cache is
the one it read, re-hashed at the end; the probe-fit diagnostics equal the stored ones exactly; the per-seed
frozen-split counts (flagged and hit, pre and post, and their discordant pairs; per stratum for failopen) equal
the stored ones. Writes results/scorefile_<table>_<cell>_L<layer>.npz and .json, staged and linked together;
never overwritten. results/ only at the 6-thread BLAS pin the tables ran at.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import advstrat_d156 as AS  # noqa: E402
import input_registry as REG  # noqa: E402
import paired as P  # noqa: E402
from provenance import (  # noqa: E402
    commit_staged, ensure_utf8_stdout, provenance_block, resolve_cache, sha256_bytes, sha256_file, stage_bytes,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
# advstrat_d156 (for the stratum labels) loads every other module used here; paired.code_block refuses an unlisted one
CODE_FILES = ["src/score_file_export.py", *AS.CODE_FILES]
COUNT_KEYS = ("n_flagged_pre", "n_flagged_post", "neg_discordant_b_c", "hits_pre", "hits_post", "pos_discordant_b_c")


def sel_cell(layer: int) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda d: next(c for c in d["cells"] if c["layer"] == layer and c["probe_set"] == "selected")


# D-214: the cells are fixed here, not chosen at run time
TABLES: dict[str, dict[str, Any]] = {
    "depth": {"file": "results/depth_d156_llama31_base2instruct.json",
              "sha256": "321faa00b11395a7ce27a2028ad433fcd048cddebe2183d29bdeb64ca86457a2",
              "name": "llama31_base2instruct", "layer": 14, "cell": sel_cell(14),
              "counts": lambda c: c["operating_point"]["frozen_split_counts"],
              "registry": lambda d: d["config"]["registry_pair"], "key_prefix": ""},
    "intent": {"file": "results/damage_d156_r2.json",
               "sha256": "9c6cd0d6ee2700f60626adca034eae45ace6c18f336fcd2cffc306b59e3919f6",
               "name": "lorasafe_mistral_s25", "layer": 12,
               "cell": lambda d: next(c for c in d["cells"] if c["scope"] == "4a" and c["cell"] == "lorasafe_mistral_s25"
                                      and c["probe_set"] == "selected"),
               "counts": lambda c: c["frozen_split_counts"],
               "registry": lambda d: "repair_lorasafe_mistral_s25", "key_prefix": "lorasafe_mistral_s25:"},
    "mechanism": {"file": "results/twomoment_repair_olmo_base2instruct.json",
                  "sha256": "e106ac2597c0fb1cbc393822c47dea206c30c205e829d322d947bb31181f59b7",
                  "name": "olmo_base2instruct", "layer": 14, "cell": sel_cell(14),
                  "counts": lambda c: c["realised"]["frozen_split_counts"],
                  "registry": lambda d: d["config"]["pair"], "key_prefix": ""},
    "failopen": {"file": "results/advstrat_d156_mistral.json",
                 "sha256": "707867278116536d62d1e32878f75b8590bc3553f3dc596438c63dc6a236effe",
                 "name": "WG_mistral", "layer": 32, "cell": sel_cell(32),
                 "counts": lambda c: c["item_level_D156"]["pooled_operating_point"]["frozen_split_counts"],
                 "registry": lambda d: d["config"]["registry_pair"], "key_prefix": ""},
}


def rel(p: Path) -> str:
    return str(p.resolve().relative_to(ROOT.resolve())).replace("\\", "/") if p.resolve().is_relative_to(ROOT.resolve()) else str(p)


def counts(F_pre: np.ndarray, F_post: np.ndarray, Q_pre: np.ndarray, Q_post: np.ndarray) -> dict[str, Any]:
    """paired.operating_point's frozen-split counts, from the flag matrices (items, seeds)."""
    return {"n_flagged_pre": F_pre.sum(axis=0).tolist(), "n_flagged_post": F_post.sum(axis=0).tolist(),
            "neg_discordant_b_c": [[int(x), int(y)] for x, y in zip((F_pre & ~F_post).sum(axis=0), (~F_pre & F_post).sum(axis=0))],
            "hits_pre": Q_pre.sum(axis=0).tolist(), "hits_post": Q_post.sum(axis=0).tolist(),
            "pos_discordant_b_c": [[int(x), int(y)] for x, y in zip((Q_pre & ~Q_post).sum(axis=0), (~Q_pre & Q_post).sum(axis=0))]}


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Score files for the author's gates (D-213, D-214)")
    ap.add_argument("--table", required=True, choices=list(TABLES))
    ap.add_argument("--out-dir", default=str(RESULTS))
    args = ap.parse_args()
    spec = TABLES[args.table]
    threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    out_dir = Path(args.out_dir).resolve()
    final = out_dir == RESULTS.resolve()
    if final and not all(t == "6" for t in threads.values()):
        raise SystemExit(f"REFUSING: results/ only at the 6-thread BLAS pin the tables ran at: {threads}")
    if not final and out_dir.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test output must not be under results/")
    stem = f"scorefile_{args.table}_{spec['name']}_L{spec['layer']}"
    npz_path, js_path = out_dir / f"{stem}.npz", out_dir / f"{stem}.json"
    for p in (npz_path, js_path):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists; never overwritten")
    code0 = P.code_block(CODE_FILES)

    tb = (ROOT / spec["file"]).read_bytes()
    if sha256_bytes(tb) != spec["sha256"]:
        raise SystemExit(f"REFUSING: {spec['file']} is not the committed file D-214 names")
    doc = json.loads(tb.decode("utf-8"))
    if final and code0["libraries"] != doc["code"]["libraries"]:
        raise SystemExit(f"REFUSING: libraries {code0['libraries']} are not the table's {doc['code']['libraries']}")
    cell = spec["cell"](doc)
    layer, C = int(cell["layer"]), float(cell["probe_C"])
    if layer != spec["layer"] or cell["probe_set"] != "selected" or cell.get("fit_row_order", "d117_permutation") != "d117_permutation":
        raise SystemExit("REFUSING: the stored cell is not the D-214 cell")
    if (cell["input_verification"] or {}).get("flag") is not None:
        raise SystemExit(f"REFUSING: the D-214 cell carries the D-170 flag: {cell['input_verification']}")
    stored = spec["counts"](cell)
    seeds = [0, 1, 2, 3, 4]
    target = P.TARGET_FPR
    if list(doc["config"]["seeds"]) != seeds or float(doc["config"]["target_fpr"]) != target:
        raise SystemExit(f"REFUSING: the table ran seeds {doc['config']['seeds']} at target {doc['config']['target_fpr']}, "
                         f"not {seeds} at {target}")
    if len(cell["probe_fit_diagnostics"]) != len(seeds):
        raise SystemExit("REFUSING: the stored cell does not hold five seed probes")

    reg_stem = spec["registry"](doc)
    cfg = REG.run(reg_stem)
    axis = cfg["axis"]
    for x in ("pre_repo", "pre_rev", "post_repo", "post_rev"):
        if cfg[x] != doc["config"].get(spec["key_prefix"] + x):
            raise SystemExit(f"REFUSING: the registry's {reg_stem} {x} is not the one {spec['file']} ran")
    data: dict[str, Any] = {}
    input_sha: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    for side, part in (("pre", "probe_train"), ("pre", "threshold_calib"), ("pre", "eval"), ("post", "eval")):
        z, r = REG.load(cfg[f"{side}_repo"], axis, part)
        key = f"{side}/{part}"
        if r["sha256"] != doc["input_cache_sha256"].get(spec["key_prefix"] + key):
            raise SystemExit(f"REFUSING: {key} is not the cache {spec['file']} read")
        data[key], input_sha[key], input_recs[key] = z, r["sha256"], r
    reg_sources = REG.sources_sha256()
    if AS.sources_now() != reg_sources:
        raise SystemExit("REFUSING: the extraction records or pins on disk are not the ones the registry read")
    verification = REG.verification(input_recs)
    if verification["flag"] is not None:
        raise SystemExit(f"REFUSING: the inputs are pin-only: {verification}")
    a, b = data["pre/eval"], data["post/eval"]
    if not np.array_equal(a["item_ids"], b["item_ids"]) or not np.array_equal(a["labels"], b["labels"]):
        raise SystemExit("REFUSING: eval items or labels differ between pre and post")
    k = f"layer_{layer}"
    tr, cal = data["pre/probe_train"], data["pre/threshold_calib"]
    y_tr = tr["labels"].astype(bool)
    ncal = ~cal["labels"].astype(bool)
    y_ev = a["labels"].astype(bool)
    neg, pos = ~y_ev, y_ev

    X_tr = tr[k]
    probes = []
    for seed in seeds:
        _, fit = P.d117_split(len(y_tr), seed)
        probes.append(P.fit_probe(X_tr[fit], y_tr[fit], seed, C, "d117_permutation"))
    diag_equal = [q.diag == s for q, s in zip(probes, cell["probe_fit_diagnostics"])]
    if not all(diag_equal):
        raise SystemExit(f"REFUSING: the re-fitted probes' diagnostics are not the table's (per seed: {diag_equal})")
    Xc, Xa, Xb = cal[k], a[k], b[k]
    S = lambda x: np.column_stack([q.score(x) for q in probes])  # noqa: E731
    sc = {"cal_neg_pre": S(Xc[ncal]), "eval_neg_pre": S(Xa[neg]), "eval_neg_post": S(Xb[neg]),
          "eval_pos_pre": S(Xa[pos]), "eval_pos_post": S(Xb[pos])}
    T = np.array([P.thr_at_fpr(sc["cal_neg_pre"][:, s], target) for s in range(len(seeds))])
    got = counts(sc["eval_neg_pre"] >= T, sc["eval_neg_post"] >= T, sc["eval_pos_pre"] >= T, sc["eval_pos_post"] >= T)
    cal_counts = (sc["cal_neg_pre"] >= T).sum(axis=0).tolist()
    anchor = {q: {"rederived": got[q], "stored": stored[q], "equal": got[q] == stored[q]} for q in COUNT_KEYS}
    anchor["n_eval_negatives"] = {"rederived": int(neg.sum()), "stored": stored["n_eval_negatives"], "equal": int(neg.sum()) == stored["n_eval_negatives"]}
    anchor["n_positives"] = {"rederived": int(pos.sum()), "stored": stored["n_positives"], "equal": int(pos.sum()) == stored["n_positives"]}
    arrays: dict[str, np.ndarray] = {"thresholds": T, "seeds": np.array(seeds), **sc,
                                     "cal_neg_item_ids": np.array([str(i) for i in cal["item_ids"][ncal]]),
                                     "eval_neg_item_ids": np.array([str(i) for i in a["item_ids"][neg]]),
                                     "eval_pos_item_ids": np.array([str(i) for i in a["item_ids"][pos]])}
    strata: dict[str, Any] = {}
    if args.table == "failopen":
        lab_ev, sha_ev = AS.stratum_labels("eval", a["item_ids"])
        split_rel = "data/frozen/safety_wildguard__eval.jsonl"
        if sha_ev != doc["config"]["frozen_splits_sha256"][split_rel]:
            raise SystemExit(f"REFUSING: {split_rel} is not the split {spec['file']} read its strata from")
        arrays["eval_neg_stratum"], arrays["eval_pos_stratum"] = lab_ev[neg], lab_ev[pos]
        for st in AS.STRATA.values():
            e, q = lab_ev[neg] == st, lab_ev[pos] == st
            g = counts(sc["eval_neg_pre"][e] >= T, sc["eval_neg_post"][e] >= T, sc["eval_pos_pre"][q] >= T, sc["eval_pos_post"][q] >= T)
            g["n_neg"], g["n_pos"] = int(e.sum()), int(q.sum())
            want = cell["item_level_D156"][st]["frozen_split_counts"]
            strata[st] = {kk: {"rederived": g[kk], "stored": want[kk], "equal": g[kk] == want[kk]} for kk in (*COUNT_KEYS, "n_neg", "n_pos")}
        strata["eval_split_sha256"] = sha_ev
    for s, seed in enumerate(seeds):
        print(f"seed {seed}: frozen threshold {float(T[s])!r}  calibration negatives >= T {cal_counts[s]}/{int(ncal.sum())}  "
              f"eval negatives flagged pre {got['n_flagged_pre'][s]} post {got['n_flagged_post'][s]} of {int(neg.sum())}  "
              f"positives hit pre {got['hits_pre'][s]} post {got['hits_post'][s]} of {int(pos.sum())}", flush=True)
    bad = [q for q, v in anchor.items() if not v["equal"]] + \
          [f"{st}.{kk}" for st, blk in strata.items() if isinstance(blk, dict) for kk, v in blk.items() if not v["equal"]]
    if bad:
        raise SystemExit(f"REFUSING to write: re-derived counts differ from the stored ones in {bad}")

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    npz_bytes = buf.getvalue()
    code1 = P.code_block(CODE_FILES)
    changed = [n for n, x, y in (("code", code1["sha256"], code0["sha256"]), ("table file", sha256_file(ROOT / spec["file"]), spec["sha256"]),
                                 ("registry sources (re-read from disk)", AS.sources_now(), reg_sources)) if x != y]
    changed += [f"cache {r['file']}" for r in input_recs.values() if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    if strata and sha256_file(ROOT / "data" / "frozen" / "safety_wildguard__eval.jsonl") != strata["eval_split_sha256"]:
        changed.append("data/frozen/safety_wildguard__eval.jsonl")
    if changed:
        raise SystemExit(f"REFUSING to write: changed during the run: {changed}")
    payload = {**provenance_block("score_file_export"), "purpose": "the author's score-file gate (D-213, D-214)",
               "code": code0, "blas_threads": threads, "input_verification": verification,
               "table": args.table, "table_file": {"file": spec["file"], "sha256": spec["sha256"], "code": doc["code"],
                                                   "env": doc.get("env"),
                                                   "code_files_differing": sorted(f for f, h in doc["code"]["sha256"].items()
                                                                                  if f in code0["sha256"] and code0["sha256"][f] != h),
                                                   "code_files_not_loaded_here": sorted(f for f in doc["code"]["sha256"]
                                                                                        if f not in code0["sha256"])},
               "cell": {"name": spec["name"], "registry_pair": reg_stem, "axis": axis, "layer": layer, "probe_C": C,
                        "cell_input_verification": cell["input_verification"]},
               "seeds": seeds, "target_fpr": target,
               "split": "frozen (draw 0 of D-156): thresholds on the pre model's threshold_calib negatives",
               "flag_rule": "score >= threshold; score = the probe's logit on its fit-set-standardised features",
               "frozen_thresholds_per_seed": T.tolist(), "calibration_negatives_at_or_above_threshold": cal_counts,
               "n_calibration_negatives": int(ncal.sum()), "counts": got, "anchor": anchor, "anchor_strata": strata or None,
               "probe_fit_diagnostics_identical": diag_equal, "probe_fit_diagnostics": [q.diag for q in probes],
               "input_cache_sha256": input_sha,
               "input_cache_source": {kk: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"]}
                                      for kk, r in input_recs.items()},
               "input_registry_sources_sha256": reg_sources,
               "score_file": {"file": rel(npz_path), "sha256": sha256_bytes(npz_bytes), "bytes": len(npz_bytes), "keys": sorted(arrays),
                              "rows": "each *_pre/*_post array: items in cache order of that subset; columns: seeds"}}
    json_bytes = json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8")
    json.loads(json_bytes.decode("utf-8"))
    staged: list[Path] = []
    try:
        t_npz, h_npz = stage_bytes(npz_path, npz_bytes)
        staged.append(t_npz)
        t_json, h_json = stage_bytes(js_path, json_bytes)
        staged.append(t_json)
        commit_staged(t_npz, npz_path, h_npz)
        commit_staged(t_json, js_path, h_json)
    finally:
        for t in staged:
            if t.exists():
                t.unlink()
    print(f"Written: {rel(npz_path)}\nsha256:  {h_npz}\nWritten: {rel(js_path)}\nsha256:  {h_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
