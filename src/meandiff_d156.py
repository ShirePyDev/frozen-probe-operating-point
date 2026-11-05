"""Item 19 (D-220; D-246): the mean-difference probe as a SECONDARY standardisation check, under D-156.

Per released repair pair: the mean-difference probe w = mean(z | harmful) - mean(z | benign) on the fit rows' own
standardised features (paired.fit_probe's standardisation; regularisation_sweep.fit_meandiff), on the D-117 80% fit
split, seeds 0-4. Its layer is the D-117 criterion restricted to the mean_diff arm, read from the selection of record
(results/selection_sweep_d165.json, grid 'mean_diff'): the highest mean holdout AUROC, the first such layer on a tie
(as the sweep's strict '>' picks). At that layer:
  OPERATING POINT  paired.repartition_estimate at the item level (D-156) and the D-164 cluster level beside: FPR pre and
                   post, paired ratio, ratio to target, the floor, TPR pre and post, delta TPR, with McNemar.
  AUROC            pre and post on the frozen eval split, with the D-152 P5 item bootstrap (as depth_d156).
  BESIDE           the logistic selected-C rows, read from the pair's depth_d156 file (not recomputed) at the logistic
                   selected layer and at the mean_diff layer.
  CHECK            the mean train and holdout AUROCs of the mean_diff probe at its layer equal the selection sweep's
                   grid values within 1e-6.
A secondary population (populations.MEANDIFF_D156); it enters no primary table. Writes
results/meandiff_d156_<pair>.json (never overwritten); a test writes outside results/.
--r2 (D-296, D-297 (1)): the re-issue with explicit repo IDs and revisions. config gains pre_repo, pre_rev, post_repo
and post_rev from the input registry. A registered run checks every other number against the first record (hash-pinned
by logs/meandiff_d156_20260927.sha256) and refuses on any difference. It writes results/meandiff_d156_<pair>_r2.json;
the first record is superseded, never deleted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_d156 as D  # noqa: E402
import input_registry as REG  # noqa: E402
import metrics as M  # noqa: E402
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
from provenance import (  # noqa: E402
    ensure_utf8_stdout, provenance_block, read_result, resolve_cache, sha256_bytes, sha256_file, write_results,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/meandiff_d156.py", *D.CODE_FILES]
LEVELS, REGISTERED = D.LEVELS, D.REGISTERED
FIRST_MANIFEST = ROOT / "logs" / "meandiff_d156_20260927.sha256"           # the first records (D-251)
IDENTITY_KEYS = ("pre_repo", "pre_rev", "post_repo", "post_rev")            # what --r2 adds to config (D-297 (1))
REISSUE_MAY_DIFFER = ("utc", "git", "env", "code", "preregistration")       # preregistration: extended, checked below
REG_SOURCES_KEY = "input_registry_sources_sha256"


def reissue_check(payload: dict[str, Any], r1b: bytes, r1_rel: str, r1_sha: str) -> dict[str, Any]:
    """D-297 (1): the re-issue equals the first record in every key but the run's header and the identity keys it
    adds; config's registry snapshot may have grown, keeping every first-record source with its hash (as D-284)."""
    if sha256_bytes(r1b) != r1_sha:
        raise SystemExit(f"REFUSING to write: {r1_rel} is not the first record ({r1_sha[:8]})")
    r1 = json.loads(r1b.decode("utf-8"))                     # listed (D-297): hashed above, in reissue_of
    mine = json.loads(json.dumps(payload, default=str))      # not a result (D-297): the payload's own round trip
    added = {"supersedes", "reissue_of"}
    if set(mine) - added != set(r1):
        raise SystemExit(f"REFUSING to write: top-level keys differ from {r1_rel}: {sorted((set(mine) - added) ^ set(r1))}")
    if any(k in r1["config"] for k in IDENTITY_KEYS):
        raise SystemExit(f"REFUSING: {r1_rel} already names its repos; nothing to re-issue")
    if not str(mine.get("preregistration", "")).startswith(str(r1.get("preregistration", ""))):
        raise SystemExit(f"REFUSING to write: the preregistration does not extend {r1_rel}'s")
    old_src, new_src = r1["config"][REG_SOURCES_KEY], mine["config"][REG_SOURCES_KEY]
    lost = sorted(k for k, h in old_src.items() if new_src.get(k) != h)
    if lost:
        raise SystemExit(f"REFUSING to write: registry sources of {r1_rel} missing or changed: {lost}")
    norm = lambda k, o: {x: y for x, y in o.items() if x not in (REG_SOURCES_KEY, *IDENTITY_KEYS)} if k == "config" else o  # noqa: E731
    compared = [k for k in r1 if k not in REISSUE_MAY_DIFFER]
    differ = [k for k in compared if json.dumps(norm(k, mine[k]), sort_keys=True) != json.dumps(norm(k, r1[k]), sort_keys=True)]
    if differ:
        raise SystemExit(f"REFUSING to write: {differ} differ from {r1_rel} (D-297 (1))")
    return {"file": r1_rel, "sha256": r1_sha, "compared": True, "keys_compared": compared,
            "keys_allowed_to_differ": list(REISSUE_MAY_DIFFER), "config_keys_added": list(IDENTITY_KEYS),
            "method": "the payload serialised as written and parsed back; each compared key gives the same "
                      "json.dumps(sort_keys=True) text as the first record's, config without the added identity keys "
                      "and its registry snapshot",
            "config_registry_sources": {"n_first_record": len(old_src), "n_now": len(new_src),
                                        "added_since": sorted(set(new_src) - set(old_src))},
            "result": "every compared key equals the first record exactly"}


def meandiff_probe(x_fit: np.ndarray, y_fit: np.ndarray) -> P.Probe:
    """regularisation_sweep.fit_meandiff on paired.fit_probe's standardisation (the fit rows' mean and sd)."""
    x = np.asarray(x_fit)
    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    z = (x - mu) / sd
    w = z[y_fit].mean(0) - z[~y_fit].mean(0)
    return P.Probe(w, -float(z.mean(0) @ w), mu, sd, {"kind": "mean_diff"})


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Item 19: mean_diff under D-156 (D-220, D-246)")
    ap.add_argument("--pair", required=True, choices=POP.UPDATE_PAIRS_REPAIR)
    ap.add_argument("--n-draws", type=int, default=REGISTERED["n_draws"])
    ap.add_argument("--target-fpr", type=float, default=REGISTERED["target_fpr"])
    ap.add_argument("--seeds", type=int, nargs="*", default=REGISTERED["seeds"])
    ap.add_argument("--n-boot", type=int, default=REGISTERED["n_boot"])
    ap.add_argument("--r2", action="store_true", help="the re-issue with explicit repo IDs and revisions (D-297 (1))")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    core = args.pair[len("repair_"):]
    reg_stem = D.registry_stem(f"depth_{core}")
    registered = (args.n_draws == REGISTERED["n_draws"] and args.target_fpr == REGISTERED["target_fpr"]
                  and list(args.seeds) == REGISTERED["seeds"] and args.n_boot == REGISTERED["n_boot"])
    final_path = (RESULTS / f"meandiff_d156_{core}{'_r2' if args.r2 else ''}.json").resolve()
    out_path = Path(args.out).resolve() if args.out else final_path
    final = out_path == final_path
    if out_path.exists():
        raise SystemExit(f"REFUSING: {out_path} exists; results are never overwritten")
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test run (--out) must not write under results/")
    threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    if final and (not registered or any(t != "6" for t in threads.values())):
        raise SystemExit(f"REFUSING the final name: registered parameters {registered}, threads {threads}")
    code0 = P.code_block(CODE_FILES)
    sel_path = D.SELECTION_OF_RECORD
    sel_sha, cl_sha = sha256_file(sel_path), sha256_file(D.CLUSTERS)
    sel_doc = read_result(sel_path)                        # registered: the record must list it (D-297 (1))
    if list(sel_doc["config"]["seeds"]) != list(args.seeds) or float(sel_doc["config"]["holdout"]) != P.HOLDOUT:
        raise SystemExit("REFUSING: selection seeds/holdout differ from this run's")
    sel = next((m for m in sel_doc["models"] if m["pair"] == reg_stem), None)
    if sel is None or "mean_diff" not in sel["grid"]:
        raise SystemExit(f"REFUSING: the selection of record holds no mean_diff arm for {reg_stem}")
    grid = sel["grid"]["mean_diff"]
    L_md, best = None, None
    for lk in sorted(grid, key=int):                      # the sweep's strict '>' keeps the first maximum
        if best is None or grid[lk]["auroc_ho20"] > best:
            L_md, best = int(lk), grid[lk]["auroc_ho20"]
    C_sel, L_log = float(sel["selected"]["arm"].split("=")[1]), int(sel["selected"]["layer"])
    depth_rel = f"results/depth_d156_{core}.json"
    depth_sha = sha256_file(ROOT / depth_rel)
    ddoc = read_result(ROOT / depth_rel)
    if (ddoc["config"]["selected_C"], ddoc["config"]["selected_layer"], ddoc["config"]["selection_sha256"]) != (C_sel, L_log, sel_sha):
        raise SystemExit(f"REFUSING: {depth_rel} was run on another selection")
    for k_, want_ in (("registry_pair", reg_stem), ("selection_of_pair", reg_stem),
                      ("selection_file", str(sel_path.relative_to(ROOT)).replace("\\", "/"))):
        if k_ in ddoc["config"] and ddoc["config"][k_] != want_:
            raise SystemExit(f"REFUSING: {depth_rel} records {k_} {ddoc['config'][k_]!r}, this run resolved {want_!r}")
    dcells = {(c["layer"], c["probe_set"]): c for c in ddoc["cells"]}

    cfg = REG.run(reg_stem)
    axis = cfg["axis"]
    data: dict[str, dict[str, Any]] = {"pre": {}, "post": {}}
    input_sha: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    for side in data:
        for part in ("probe_train", "threshold_calib", "eval"):
            z, r = REG.load(cfg[f"{side}_repo"], axis, part)
            data[side][part] = z
            input_sha[f"{side}/{part}"], input_recs[f"{side}/{part}"] = r["sha256"], r
    reg_sources = REG.sources_sha256()
    if D.sources_now() != reg_sources:
        raise SystemExit("REFUSING: the extraction records or pins on disk are not the ones the registry read")
    if any(ddoc["input_cache_sha256"].get(k) != h for k, h in input_sha.items()):
        raise SystemExit(f"REFUSING: {depth_rel} read other caches")
    verification = REG.verification(input_recs)
    for part in ("probe_train", "threshold_calib", "eval"):
        a, b = data["pre"][part], data["post"][part]
        if not np.array_equal(a["item_ids"], b["item_ids"]) or not np.array_equal(a["labels"], b["labels"]):
            raise SystemExit(f"REFUSING: {part} items or labels differ between pre and post")

    y_tr = data["pre"]["probe_train"]["labels"].astype(bool)
    y_cal = data["pre"]["threshold_calib"]["labels"].astype(bool)
    y_ev = data["pre"]["eval"]["labels"].astype(bool)
    ncal, neg, pos = ~y_cal, ~y_ev, y_ev
    pooled_ids = [str(i) for i in data["pre"]["threshold_calib"]["item_ids"][ncal]] + \
                 [str(i) for i in data["pre"]["eval"]["item_ids"][neg]]
    n_cal, n_pool = int(ncal.sum()), len(pooled_ids)
    cl = json.load(open(D.CLUSTERS, encoding="utf-8"))["axes"][axis]   # not a result (D-297): a manifest, hashed
    if cl["n_pooled_negatives"] != n_pool or cl["n_calibration"] != n_cal or \
            cl["pool_item_ids_sha256"] != sha256_bytes("\n".join(pooled_ids).encode("utf-8")):
        raise SystemExit(f"REFUSING: manifests/negative_clusters.json does not describe the {axis} pool")
    draws = {LEVELS[0]: P.repartition_draws(n_cal, n_pool - n_cal, args.n_draws, P.AXIS_DRAW_SEED[axis]),
             LEVELS[1]: P.cluster_repartition_draws(P.cluster_units(pooled_ids, cl["clusters"]), n_cal,
                                                    args.n_draws, P.AXIS_CLUSTER_DRAW_SEED[axis])}
    k = f"layer_{L_md}"
    X = {s: {p: data[s][p][k] for p in ("probe_train", "threshold_calib", "eval")} for s in data}
    print(f"mean_diff {args.pair} [{axis}]: mean_diff layer {L_md} (ho20 AUROC {best:.4f}); logistic C {C_sel:g} at layer "
          f"{L_log}; D-170 flag {verification['flag']}; registered {registered}", flush=True)

    probes, fit_au, ho_au = [], [], []
    for seed in args.seeds:
        ho, fit = P.d117_split(len(y_tr), seed)
        q = meandiff_probe(X["pre"]["probe_train"][fit], y_tr[fit])
        probes.append(q)
        s_fit, s_ho = q.score(X["pre"]["probe_train"][fit]), q.score(X["pre"]["probe_train"][ho])
        fit_au.append(M.auroc(s_fit[y_tr[fit]], s_fit[~y_tr[fit]]).value)
        ho_au.append(M.auroc(s_ho[y_tr[ho]], s_ho[~y_tr[ho]]).value)
    check = {"layer": L_md, "selection_auroc_fit": grid[str(L_md)]["auroc_fit"], "selection_auroc_ho20": grid[str(L_md)]["auroc_ho20"],
             "auroc_fit_here": float(np.mean(fit_au)), "auroc_ho20_here": float(np.mean(ho_au)), "tolerance": 1e-6}
    check["abs_diff_max"] = max(abs(check["auroc_fit_here"] - check["selection_auroc_fit"]),
                                abs(check["auroc_ho20_here"] - check["selection_auroc_ho20"]))
    check["enforced"] = True                               # always: the seeds are the selection's (checked above)
    if check["abs_diff_max"] > 1e-6:
        raise SystemExit(f"REFUSING: the mean_diff probe does not reproduce the selection sweep at layer {L_md}: {check}")
    S = lambda x: np.column_stack([q.score(x) for q in probes])  # noqa: E731
    pre_pool = np.vstack([S(X["pre"]["threshold_calib"][ncal]), S(X["pre"]["eval"][neg])])
    post_pool = np.vstack([S(X["post"]["threshold_calib"][ncal]), S(X["post"]["eval"][neg])])
    pos_pre, pos_post = S(X["pre"]["eval"][pos]), S(X["post"]["eval"][pos])
    op = {}
    for lvl in LEVELS:
        rep, _ = P.repartition_estimate(pre_pool[:n_cal], pre_pool[n_cal:], post_pool[:n_cal], post_pool[n_cal:],
                                        pos_pre, pos_post, draws[lvl], args.target_fpr)
        op[lvl] = rep
    n_neg, n_pos = int(neg.sum()), int(pos.sum())
    CNT_N = P.counts_matrix(n_neg, args.n_boot, D.ITEM_BOOT_SEED["eval_negatives"])
    CNT_P = P.counts_matrix(n_pos, args.n_boot, D.ITEM_BOOT_SEED["eval_positives"])
    ONE_N, ONE_P = np.ones((1, n_neg)), np.ones((1, n_pos))
    au: dict[str, Any] = {"note": D.AUROC_NOTE}
    reps: dict[str, np.ndarray] = {}
    points: dict[str, np.ndarray] = {}
    for name, (sp, sn) in {"pre": (pos_pre, pre_pool[n_cal:]), "frozen": (pos_post, post_pool[n_cal:])}.items():
        pt = np.array([M.auroc(sp[:, s], sn[:, s]).value for s in range(len(args.seeds))])
        unit = np.array([D.weighted_auroc(sp[:, s], sn[:, s], ONE_P, ONE_N)[0] for s in range(len(args.seeds))])
        if not np.array_equal(pt, unit):                  # as depth_d156: unit weights must be metrics.auroc exactly
            raise SystemExit(f"REFUSING: {name}: the weighted AUROC is not metrics.auroc bit for bit")
        points[name] = pt
        reps[name] = np.mean([D.weighted_auroc(sp[:, s], sn[:, s], CNT_P, CNT_N) for s in range(len(args.seeds))], axis=0)
    points["drop_frozen"] = points["pre"] - points["frozen"]
    reps["drop_frozen"] = reps["pre"] - reps["frozen"]
    for name in ("pre", "frozen", "drop_frozen"):            # depth_d156's AUROC schema
        au[name] = {"mean_over_seeds": float(points[name].mean()), "per_seed": points[name].tolist(),
                    "range_over_seeds": [float(points[name].min()), float(points[name].max())], **P.ci(reps[name])}
    beside = {f"logistic_selected_C_at_layer_{li}": {"layer": li, "probe_C": C_sel,
                                                     "operating_point": {LEVELS[0]: dcells[(li, "selected")]["operating_point"],
                                                                         LEVELS[1]: dcells[(li, "selected")][LEVELS[1]]["operating_point"]},
                                                     "auroc": dcells[(li, "selected")]["auroc"], "source": depth_rel}
              for li in sorted({L_log, L_md})}
    code1 = P.code_block(CODE_FILES)
    changed = [n for n, a, b in (("code", code1["sha256"], code0["sha256"]), ("selection", sha256_file(sel_path), sel_sha),
                                 ("cluster manifest", sha256_file(D.CLUSTERS), cl_sha), (depth_rel, sha256_file(ROOT / depth_rel), depth_sha),
                                 ("registry sources", D.sources_now(), reg_sources)) if a != b]
    changed += [f"cache {r['file']}" for r in input_recs.values() if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    if changed:
        raise SystemExit(f"REFUSING to write: changed during the run: {changed}")
    payload = {**provenance_block("meandiff_d156"), "preregistration": "D-220, D-246", "code": code0,
               "registered_parameters": registered, "blas_threads": threads, "input_verification": verification,
               "config": {"pair": args.pair, "registry_pair": reg_stem, "axis": axis, "meandiff_layer": L_md,
                          "meandiff_layer_rule": "argmax of the mean_diff arm's mean holdout AUROC in the selection of "
                                                 "record, the first on a tie (the D-117 criterion restricted to the arm)",
                          "logistic_selected_C": C_sel, "logistic_selected_layer": L_log,
                          "selection_file": str(sel_path.relative_to(ROOT)).replace("\\", "/"), "selection_sha256": sel_sha,
                          "seeds": args.seeds, "target_fpr": args.target_fpr, "n_draws": args.n_draws, "n_boot": args.n_boot,
                          "draw_seed": P.AXIS_DRAW_SEED[axis], "cluster_draw_seed": P.AXIS_CLUSTER_DRAW_SEED[axis],
                          "cluster_manifest": {"file": "manifests/negative_clusters.json", "sha256": cl_sha},
                          "item_bootstrap": {"n_boot": args.n_boot, "seeds": D.ITEM_BOOT_SEED},
                          "input_registry_sources_sha256": reg_sources,
                          **({k: cfg[k] for k in IDENTITY_KEYS} if args.r2 else {})},
               "depth_file": {"file": depth_rel, "sha256": depth_sha, "role": "the logistic rows beside, read not recomputed"},
               "population": {"declared_in": "src/populations.py", "group": "MEANDIFF_D156", "role": "secondary (D-220)"},
               "selection_check": check, "input_cache_sha256": input_sha,
               "meandiff": {"layer": L_md, "operating_point": op, "auroc": au},
               "logistic_beside": beside}
    if args.r2:
        r1_rel = f"results/meandiff_d156_{core}.json"
        first = {ln.split()[1]: ln.split()[0] for ln in FIRST_MANIFEST.read_text(encoding="utf-8").splitlines() if ln.strip()}
        if r1_rel not in first:
            raise SystemExit(f"REFUSING: {FIRST_MANIFEST.name} does not list {r1_rel}")
        payload["preregistration"] += "; D-296, D-297 (1) (re-issue with explicit repo IDs and revisions)"
        payload["supersedes"] = [f"{r1_rel} (D-297 (1): the same numbers, with the model repos and revisions named)"]
        if registered:
            payload["reissue_of"] = reissue_check(payload, (ROOT / r1_rel).read_bytes(), r1_rel, first[r1_rel])
        else:
            payload["reissue_of"] = {"file": r1_rel, "sha256": first[r1_rel], "compared": False,
                                     "why": "non-registered parameters (a test run)"}
    digest = write_results(out_path, payload, exclusive=True)
    r = op[LEVELS[0]]["paired_ratio"]
    print(f"mean_diff L{L_md}: paired ratio {r['mean']:.3f} [{r['band95'][0]:.3f}, {r['band95'][1]:.3f}]; "
          f"AUROC pre {au['pre']['mean_over_seeds']:.4f} post {au['frozen']['mean_over_seeds']:.4f}")
    print(f"Written: {out_path}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
