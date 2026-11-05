"""The source test before 4d part B (the author's D-271 (5); pre-registered in D-274, before this code existed).

4d B draws its pools from named sources (D-244 (2), (4)): benign traffic from cal_d, the draw's 1000 calibration
negatives, whose items come from two ORIGINS (the 1000 threshold_calib negatives and the 2000 eval negatives of the
pooled 3000); harmful traffic and injections from H, the 1000 threshold_calib harmful items; R_FRESH from the
evaluation negatives. Arms are scored on the evaluation negatives and the 2000 eval positives. Given D-250 (the
probe_train negatives were not exchangeable with the calibration/eval negatives), each source is tested here UNDER
NO UPDATE, for each of the nine 4d B pairs, with the PRE model at its D-117 layer L* (selected C primary, C=1
beside; seeds 0-4, fitted as depth_d156 fits them), at T_d, each seed's 1% threshold on the pre scores of cal_d, over
the item-level D-156 draws (the frozen split at index 0):
  (a) BENIGN ORIGIN: in the draw's evaluation part, the raw FPR at T_d of the items of each origin, minus the null:
      the FPR of a random subset of the same evaluation part, of the same size (D-250's method; per draw and seed).
  (b) HARMFUL SOURCE (amended before any run, D-276): rate(H) - rate(eval positives) at T_d, minus the same contrast
      for one random split of the pooled 3000 positives into 1000 + 2000, drawn per draw and shared by all seeds.
      (D-274's first form, a random 1000 of the eval positives per seed, rejected about half the time under exact
      exchangeability, because H is one fixed set; the pre-run review's simulation.)
  NOT EXCHANGEABLE (judged at selected C): the band of the seed-mean difference per draw lies wholly above or wholly
  below 0. D-274 (3) fixes what follows from each outcome; this script reports the outcome and the rule per pair.
ANCHOR: the seed mean of floored FPRs of the whole evaluation part at T_d equals the pair's depth_d156 fpr_pre at L*
(item level, both probe sets), exactly.
Writes results/source_test_4db.json (never overwritten); a test writes outside results/.
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
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
from provenance import (  # noqa: E402
    ensure_utf8_stdout, provenance_block, resolve_cache, sha256_bytes, sha256_file, write_results,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FINAL_OUT = RESULTS / "source_test_4db.json"
CODE_FILES = ["src/source_test_4db.py", *D.CODE_FILES]
PROBE_SETS, REGISTERED = D.PROBE_SETS, D.REGISTERED
LEVEL = D.LEVELS[0]                                   # 4d B is item level only (D-244 (3))
DEPTH_MANIFEST = ROOT / "logs" / "depth_d156_20260926.sha256"
SOURCE_NULL_SEED = 20260962                           # this test's null subsets (no other stream uses it; recorded in the result entry)
ORIGINS = ("threshold_calib", "eval")
N_H, N_POS = 1000, 2000                               # D-244 (4): H; the eval positives


def manifest_sha(man: Path, rel: str) -> str:
    rows = [ln.split(maxsplit=1) for ln in man.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if any(len(r) != 2 for r in rows):
        raise SystemExit(f"REFUSING: {man.name} has a malformed line")
    hits = [r[0] for r in rows if r[1].strip().lstrip("*") == rel]
    if len(hits) != 1:
        raise SystemExit(f"REFUSING: {man.name} names {rel} {len(hits)} times")
    return hits[0]


def thresholds(S: np.ndarray, target: float) -> np.ndarray:
    return np.array([P.thr_at_fpr(S[:, s], target) for s in range(S.shape[1])])


def arr_block(v: list[float]) -> dict[str, Any]:
    a = np.array(v, dtype="<f8")
    return {"band": P.band_summary(a[1:], a[0]), "per_draw": [float(x) if np.isfinite(x) else None for x in a],
            "per_draw_sha256": sha256_bytes(a.tobytes())}


def verdict(band: dict[str, Any]) -> str:
    lo, hi = band["band95"]
    return "not exchangeable (above)" if lo > 0 else "not exchangeable (below)" if hi < 0 else "exchangeable"


def run_pair(stem: str, args: argparse.Namespace, sel_doc: dict[str, Any], sel_sha: str,
             seen: dict[str, str]) -> dict[str, Any]:
    core = stem[len("repair_"):]
    sel = next((m for m in sel_doc["models"] if m["pair"] == stem), None)
    if sel is None or not sel["selected"]["arm"].startswith("C="):
        raise SystemExit(f"REFUSING: no logistic-C selection of record for {stem}")
    C_sel, L_sel = float(sel["selected"]["arm"].split("=")[1]), int(sel["selected"]["layer"])
    depth_rel = f"results/depth_d156_{core}.json"
    db = (ROOT / depth_rel).read_bytes()
    depth_sha = sha256_bytes(db)
    if manifest_sha(DEPTH_MANIFEST, depth_rel) != depth_sha:
        raise SystemExit(f"REFUSING: {depth_rel} is not the file {DEPTH_MANIFEST.name} names")
    seen[depth_rel] = depth_sha
    ddoc = json.loads(db.decode("utf-8"))
    if (ddoc["config"]["selected_C"], ddoc["config"]["selected_layer"], ddoc["config"]["selection_sha256"]) != (C_sel, L_sel, sel_sha):
        raise SystemExit(f"REFUSING: {depth_rel} was run on another selection")
    dcells = {(c["layer"], c["probe_set"]): c for c in ddoc["cells"]}
    cfg = REG.run(stem)
    axis = cfg["axis"]
    data: dict[str, Any] = {}
    input_sha: dict[str, str] = {}
    recs: dict[str, dict[str, Any]] = {}
    for part in ("probe_train", "threshold_calib", "eval"):
        z, r = REG.load(cfg["pre_repo"], axis, part)
        data[part], input_sha[f"pre/{part}"], recs[f"pre/{part}"] = z, r["sha256"], r
    if any(ddoc["input_cache_sha256"].get(k) != h for k, h in input_sha.items()):
        raise SystemExit(f"REFUSING: {depth_rel} read other pre caches")
    for z in data.values():
        if "nonfinite" in z.files and z["nonfinite"].any():
            raise SystemExit(f"REFUSING: {stem} has non-finite items")
        if f"layer_{L_sel}" not in z.files:
            raise SystemExit(f"REFUSING: {stem}: a cache has no layer {L_sel}")
    verification = REG.verification(recs)
    y_tr = data["probe_train"]["labels"].astype(bool)
    y_cal = data["threshold_calib"]["labels"].astype(bool)
    y_ev = data["eval"]["labels"].astype(bool)
    ncal, hcal, neg, pos = ~y_cal, y_cal, ~y_ev, y_ev
    if int(hcal.sum()) != N_H or int(pos.sum()) != N_POS:
        raise SystemExit(f"REFUSING: {stem}: H holds {int(hcal.sum())} and the eval positives {int(pos.sum())} (D-244 (4))")
    pooled_ids = [str(i) for i in data["threshold_calib"]["item_ids"][ncal]] + [str(i) for i in data["eval"]["item_ids"][neg]]
    n_cal, n_pool = int(ncal.sum()), len(pooled_ids)
    if (n_cal, n_pool - n_cal) != (1000, 2000):
        raise SystemExit(f"REFUSING: {stem}: the pool is {n_cal} + {n_pool - n_cal}, not D-274's 1000 + 2000")
    cl = json.load(open(D.CLUSTERS, encoding="utf-8"))["axes"][axis]
    if cl["n_pooled_negatives"] != n_pool or cl["n_calibration"] != n_cal or \
            cl["pool_item_ids_sha256"] != sha256_bytes("\n".join(pooled_ids).encode("utf-8")):
        raise SystemExit(f"REFUSING: manifests/negative_clusters.json does not describe the {axis} pool")
    origin_of = np.array([0] * n_cal + [1] * (n_pool - n_cal))                  # 0 threshold_calib, 1 eval (pool order)
    draws = P.repartition_draws(n_cal, n_pool - n_cal, args.n_draws, P.AXIS_DRAW_SEED[axis])
    parts = [(np.arange(n_cal), np.arange(n_cal, n_pool))] + [(d[:n_cal], d[n_cal:]) for d in draws]
    k_ = f"layer_{L_sel}"
    Xtr, Xcal, Xev = data["probe_train"][k_], data["threshold_calib"][k_], data["eval"][k_]
    subs: dict[tuple[int, int, int], np.ndarray] = {}            # (draw, seed, source) -> the null subset, both probe sets
    out: dict[str, Any] = {"pair": stem, "axis": axis, "selected_C": C_sel, "selected_layer": L_sel,
                           "draw_seed": P.AXIS_DRAW_SEED[axis], "pre_repo": cfg["pre_repo"], "pre_rev": cfg["pre_rev"],
                           "depth_anchor": {"file": depth_rel, "sha256": depth_sha}, "input_cache_sha256": input_sha,
                           "input_verification": verification, "probe_sets": {}}
    n_anchor = 0
    for pset in PROBE_SETS:
        probes = []
        for seed in args.seeds:
            _, fit = P.d117_split(len(y_tr), seed)
            rows, C, order = (fit, C_sel, "d117_permutation") if pset == "selected" else (np.sort(fit), 1.0, "file_order")
            probes.append(P.fit_probe(Xtr[rows], y_tr[rows], seed, C, order))
        S = lambda x: np.column_stack([q.score(x) for q in probes])  # noqa: E731
        pool = np.vstack([S(Xcal[ncal]), S(Xev[neg])])
        sH, sP = S(Xcal[hcal]), S(Xev[pos])
        acc = {f"benign_origin_{o}": {"source": [], "null": [], "diff": [], "n_items": []} for o in ORIGINS}
        acc["harmful_H"] = {"source": [], "null": [], "diff": [], "n_items": []}
        fpr_pre = []
        for d_i, (c_, e_) in enumerate(parts):
            T = thresholds(pool[c_], args.target_fpr)
            fpr_pre.append(float(P.floored_fpr(pool[e_] >= T).mean()))
            for oi, o in enumerate(ORIGINS):
                items = e_[origin_of[e_] == oi]
                if d_i == 0:
                    # the frozen evaluation part is the 2000 eval-origin negatives only, so (a) is undefined there: no
                    # threshold_calib-origin item, and the eval-origin null would be the whole part (review of D-274)
                    a = acc[f"benign_origin_{o}"]
                    for kk in ("source", "null", "diff"):
                        a[kk].append(float("nan"))
                    a["n_items"].append(int(len(items)))
                    continue
                src, nul = [], []
                for s_i, seed in enumerate(args.seeds):
                    key = (d_i, seed, oi)
                    if key not in subs:
                        subs[key] = np.random.default_rng([SOURCE_NULL_SEED, d_i, seed, oi]).choice(len(e_), size=len(items), replace=False)
                    src.append(float((pool[items, s_i] >= T[s_i]).mean()))
                    nul.append(float((pool[e_[subs[key]], s_i] >= T[s_i]).mean()))
                a = acc[f"benign_origin_{o}"]
                a["source"].append(float(np.mean(src)))
                a["null"].append(float(np.mean(nul)))
                a["diff"].append(float(np.mean(np.array(src) - np.array(nul))))
                a["n_items"].append(int(len(items)))
            # (b), amended before any run (D-276): H is ONE fixed set, so its null is a permutation of the pooled 3000
            # positives (H + the eval positives), ONE per draw and shared by every seed and both probe sets: the statistic
            # is rate(H) - rate(eval positives); its null, rate(pseudo-H) - rate(the other 2000)
            key = (d_i, -1, len(ORIGINS))
            if key not in subs:
                subs[key] = np.random.default_rng([SOURCE_NULL_SEED, d_i, len(ORIGINS)]).permutation(N_H + N_POS)
            ph, pr = subs[key][:N_H], subs[key][N_H:]
            src, nul = [], []
            for s_i in range(len(args.seeds)):
                allp = np.concatenate([sH[:, s_i], sP[:, s_i]])
                src.append(float((sH[:, s_i] >= T[s_i]).mean() - (sP[:, s_i] >= T[s_i]).mean()))
                nul.append(float((allp[ph] >= T[s_i]).mean() - (allp[pr] >= T[s_i]).mean()))
            a = acc["harmful_H"]
            a["source"].append(float(np.mean(src)))
            a["null"].append(float(np.mean(nul)))
            a["diff"].append(float(np.mean(np.array(src) - np.array(nul))))
            a["n_items"].append(N_H)
        # the anchor: the whole evaluation part's floored FPR at T_d is the depth file's fpr_pre, exactly
        if args.registered:
            want = dcells[(L_sel, pset)]["operating_point"]["fpr_pre"]
            if json.loads(json.dumps(P.band_summary(np.array(fpr_pre[1:]), fpr_pre[0]))) != want:
                raise SystemExit(f"REFUSING: {stem} {pset}: the evaluation FPR at T_d is not {depth_rel}'s fpr_pre")
            n_anchor += 1
        blk: dict[str, Any] = {}
        for name, a in acc.items():
            diff = arr_block(a["diff"])
            blk[name] = {"source_rate": arr_block(a["source"])["band"], "null_rate": arr_block(a["null"])["band"],
                         "source_minus_null": diff,
                         "n_items": {"frozen_split": a["n_items"][0], "min_over_draws": min(a["n_items"][1:]),
                                     "max_over_draws": max(a["n_items"][1:])},
                         ("verdict" if pset == "selected" else "verdict_not_judged"): verdict(diff["band"])}
        out["probe_sets"][pset] = blk
    if args.registered and n_anchor != len(PROBE_SETS):
        raise SystemExit(f"REFUSING: {stem}: {n_anchor} anchor blocks")
    sel_blk = out["probe_sets"]["selected"]
    benign_bad = [o for o in ORIGINS if sel_blk[f"benign_origin_{o}"]["verdict"] != "exchangeable"]
    h_bad = sel_blk["harmful_H"]["verdict"] != "exchangeable"
    out["d274_rule"] = {"benign_origins_not_exchangeable": benign_bad, "harmful_H_not_exchangeable": h_bad,
                        "applies": ("run as registered; the four D-272 held records are verified and kept"
                                    if not benign_bad and not h_bad else
                                    "; ".join((["stratify benign traffic, drift windows and R_FRESH by origin, in each draw's "
                                                "evaluation-part proportions, with FPR per origin beside"] if benign_bad else []) +
                                              (["exclude H from score-level pools; harmful source = a seeded half of the eval "
                                                "positives per draw, TPR on the other half in every cell"] if h_bad else []) +
                                              ["the changed configurations supersede the D-272 held records (never deleted)"]))}
    out["n_anchor_blocks"] = n_anchor
    out["_recs"] = recs
    out["_run"] = cfg
    return out


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="The source test before 4d part B (D-271 (5), D-274)")
    ap.add_argument("--pairs", nargs="+", default=list(POP.UPDATE_PAIRS_REPAIR), choices=list(POP.UPDATE_PAIRS_REPAIR))
    ap.add_argument("--n-draws", type=int, default=REGISTERED["n_draws"])
    ap.add_argument("--target-fpr", type=float, default=REGISTERED["target_fpr"])
    ap.add_argument("--seeds", type=int, nargs="*", default=REGISTERED["seeds"])
    ap.add_argument("--out", default=str(FINAL_OUT))
    args = ap.parse_args()
    out_path = Path(args.out).resolve()
    final = out_path == FINAL_OUT.resolve()
    if out_path.exists():
        raise SystemExit(f"REFUSING: {out_path} exists; never overwritten")
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test output must not be under results/")
    # the anchors hold for any subset of pairs at the registered parameters; "registered" (the record) needs all nine
    args.registered = (args.n_draws == REGISTERED["n_draws"] and args.target_fpr == REGISTERED["target_fpr"]
                       and list(args.seeds) == REGISTERED["seeds"])
    all_nine = sorted(args.pairs) == sorted(POP.UPDATE_PAIRS_REPAIR) and len(args.pairs) == len(POP.UPDATE_PAIRS_REPAIR)
    threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    if final and (not args.registered or any(t != "6" for t in threads.values()) or not all_nine):
        raise SystemExit(f"REFUSING the final name: registered {args.registered}, threads {threads}, all nine pairs {all_nine}")
    code0 = P.code_block(CODE_FILES)
    sel_path = D.SELECTION_OF_RECORD
    sel_sha, cl_sha = sha256_file(sel_path), sha256_file(D.CLUSTERS)
    sel_doc = json.load(open(sel_path, encoding="utf-8"))
    if list(sel_doc["config"]["seeds"]) != list(args.seeds) or float(sel_doc["config"]["holdout"]) != P.HOLDOUT:
        raise SystemExit("REFUSING: selection seeds/holdout differ from this run's")
    seen: dict[str, str] = {}
    man_sha = sha256_file(DEPTH_MANIFEST)
    # the registry's sources, before any pair loads, compared again at the end (as overfit_d156, review of D-274)
    reg_sources = REG.sources_sha256()
    if D.sources_now() != reg_sources:
        raise SystemExit("REFUSING: the extraction records or pins on disk are not the ones the registry read")
    pairs_out: dict[str, Any] = {}
    all_recs: dict[str, dict[str, Any]] = {}
    runs: dict[str, Any] = {}
    for stem in args.pairs:
        r = run_pair(stem, args, sel_doc, sel_sha, seen)
        for k, v in r.pop("_recs").items():
            all_recs[f"{stem}:{k}"] = v
        runs[stem] = r.pop("_run")
        pairs_out[stem] = r
        print(f"{stem}: L{r['selected_layer']}; " + "; ".join(
            f"{name} {r['probe_sets']['selected'][name]['verdict']} "
            f"({r['probe_sets']['selected'][name]['source_minus_null']['band']['mean']:+.4f})"
            for name in r["probe_sets"]["selected"]) + f"; D-274 (3): {r['d274_rule']['applies']}", flush=True)
    code1 = P.code_block(CODE_FILES)
    changed = [n for n, a, b in (("code", code1["sha256"], code0["sha256"]), ("selection", sha256_file(sel_path), sel_sha),
                                 ("cluster manifest", sha256_file(D.CLUSTERS), cl_sha),
                                 (DEPTH_MANIFEST.name, sha256_file(DEPTH_MANIFEST), man_sha),
                                 ("registry sources", D.sources_now(), reg_sources)) if a != b]
    changed += [f for f, h in seen.items() if sha256_file(ROOT / f) != h]
    changed += [f"cache {r['file']}" for r in all_recs.values() if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    if changed:
        raise SystemExit(f"REFUSING to write: changed during the run: {changed}")
    payload = {**provenance_block("source_test_4db"), "code": code0, "preregistration": "D-271 (5) (author), D-274",
               "registered_parameters": bool(args.registered and all_nine), "blas_threads": threads,
               "config": {"pairs": args.pairs, "level": LEVEL, "seeds": args.seeds, "target_fpr": args.target_fpr,
                          "n_draws": args.n_draws, "null_seed": SOURCE_NULL_SEED, "origins": list(ORIGINS),
                          "n_H": N_H, "n_eval_positives": N_POS,
                          "selection_file": str(sel_path.relative_to(ROOT)).replace("\\", "/"), "selection_sha256": sel_sha,
                          "cluster_manifest": {"file": "manifests/negative_clusters.json", "sha256": cl_sha},
                          "depth_manifest": {"file": str(DEPTH_MANIFEST.relative_to(ROOT)).replace("\\", "/"), "sha256": man_sha},
                          "all_nine_pairs": all_nine, "runs": runs, "inputs_from": "src/input_registry.py",
                          "input_registry_sources_sha256": reg_sources,
                          # D-271 (5), D-274 (1): the named sources of every 4d B pool
                          "sources": {"benign_traffic_drift_windows_pool_size_sweep": "cal_d, the draw's 1000 calibration "
                                      "negatives (origins: the 1000 threshold_calib and 2000 eval negatives, re-partitioned)",
                                      "R_SAME": "cal_d re-run through the post model",
                                      "R_FRESH": "1000 of the draw's 2000 evaluation negatives",
                                      "harmful_traffic_and_A0_A1_injections": "H, the 1000 threshold_calib harmful items",
                                      "evaluation": "the other 1000 evaluation negatives (FPR) and the 2000 eval positives (TPR)",
                                      "reported_not_tested": "traffic drawn from cal_d reuses the items that set T_d (D-244)"}},
               "definitions": {
                   "benign_origin_<o>": "raw FPR at T_d of the draw's evaluation-part items of origin o, minus the FPR of a "
                                        "same-size random subset of that evaluation part (per draw and seed; seed mean); "
                                        "undefined at the frozen split, whose evaluation part is eval-origin only. The two "
                                        "origins partition the evaluation part, so their differences are near mirror images "
                                        "(about 2/3 and -1/3 of one gap), not two independent tests",
                   "harmful_H": "rate(H) - rate(eval positives) at T_d (source), minus the same contrast for one random "
                                "1000 + 2000 split of the pooled 3000 positives per draw, shared by all seeds (null; D-276). "
                                "Replaces D-274's first form: the flag rate at T_d of the 1000 threshold_calib harmful items, minus that of a random 1000 "
                                "of the 2000 eval positives (per draw and seed; seed mean)",
                   "verdict": "judged at selected C: not exchangeable when the band of source - null lies wholly above or "
                              "below 0 (D-274 (2)). 'exchangeable' means no difference was detected at this test's power, "
                              "which is limited for small rate ratios (D-271 (4)); the test compares out-of-sample items "
                              "only, so it cannot see 4d B's reuse of the cal_d items that set T_d (D-276)"},
               "summary": {s: {"d274_rule": v["d274_rule"],
                               "verdicts": {n: b["verdict"] for n, b in v["probe_sets"]["selected"].items()}}
                           for s, v in pairs_out.items()},
               # read by src/verify_provenance.py (D-217 (d), F-035, D-170): keys "<pair>:<side>/<part>", runs in config.runs
               "input_verification": REG.verification(all_recs),
               "input_cache_sha256": {k: r["sha256"] for k, r in all_recs.items()},
               "input_cache_source": {k: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"]}
                                      for k, r in all_recs.items()},
               "pairs": pairs_out}
    digest = write_results(out_path, payload, exclusive=True)
    print(f"Written: {out_path}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
