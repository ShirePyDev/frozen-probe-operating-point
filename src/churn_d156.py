"""Item 18 (D-219): flag-set churn under D-156, with the choices of D-269.

For one 4d pair (UPDATE_PAIRS_REPAIR), at its D-117 selected layer L*, with the D-117 probes (the selected C primary,
C=1 beside; seeds 0-4, fitted exactly as depth_d156 fits them on the PRE model's probe_train):
  T_d   each seed's frozen threshold, the 1% point of the PRE scores on the draw's calibration negatives cal_d.
  A     the benign items flagged by the PRE model at T_d.
  B     the benign items flagged by the POST model at the reference-set repair (D-244 (5)): the 1% threshold on the
        reference's POST scores.
  J_s = |A n B| / |A u B| per seed; J = the seed mean per draw (a seed with an empty union is undefined, left out
        and counted); D-156 mean and 95% band over the draws, the frozen split beside.
PRIMARY, R_SAME (D-152, D-244 (2)): the reference is cal_d re-run through the post model; A and B over the draw's
  evaluation part. Its threshold is the same order statistic as 4d A's quantile_thr_benign (the exact rank transfer of
  T_d on cal_d); the run computes both and refuses on any difference. NO-UPDATE NULL: pre against pre through the
  same repair gives T_d itself (the D-258 (1) self-pair invariant), so the null J is 1; the run computes it and
  refuses unless it is exactly 1 wherever it is defined.
BESIDE, R_FRESH (item level only, as 4d B, D-244 (3)): 1000 of the draw's evaluation negatives, drawn as 4d B draws
  them (numpy default_rng([POOL_SEED, draw index]).permutation of the evaluation part, the first 1000); A and B over
  the other 1000. Its null: the PRE model at the 1% point of the PRE scores on R_FRESH, against A.
XSTEST (D-219; never re-partitioned, D-167): the 250 safe prompts at the same thresholds, for the pairs with an
  item-9 record (D-269 (6)).
ANCHORS, each refused on any difference: A's seed-mean floored FPR is the depth_d156 file's fpr_pre; B's (R_SAME) is
  4d A's quantile_thr_benign fpr_post (results/repair_d156_<core>.json, named by its manifest); the XSTest pre flags
  are the item-9 record's safe_flagged_pre_mean and frozen-split counts. All at L*, both probe sets and both levels.
RECORDED: per comparison the per-draw seed-mean J (index 0 the frozen split, sha256 of its '<f8' bytes), its D-156
  band, the per-seed D-156 means and their range (D-219's "range over seeds"), and |A|, |B|, |A n B| at the frozen
  split; the count of tied pool scores per seed (the only way the R_SAME threshold check can fire).
Writes results/churn_d156_<core>.json (never overwritten); a test writes outside results/. --n-draws and --target-fpr
exist for symmetry only: any non-registered value refuses at the anchored records' config checks, so a test is a full
run written outside results/ (--repair-manifest may name a test manifest; the final name refuses it).
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
from poolstudy_d156 import POOL_SEED  # noqa: E402  (4d B's R_FRESH stream, so R_FRESH is the same split per draw)
from provenance import (  # noqa: E402
    ensure_utf8_stdout, provenance_block, resolve_cache, sha256_bytes, sha256_file, write_results,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/churn_d156.py", "src/poolstudy_d156.py", "src/pool_propositions.py", *D.CODE_FILES]
LEVELS, PROBE_SETS, REGISTERED = D.LEVELS, D.PROBE_SETS, D.REGISTERED
DEPTH_MANIFEST = ROOT / "logs" / "depth_d156_20260926.sha256"
REPAIR_MANIFEST = ROOT / "logs" / "repair_d156_4dA_20260930.sha256"          # 4d A's pulled records (D-269 (7))
OR_MANIFESTS = (ROOT / "logs" / "overrefusal_d156_20260926.sha256", ROOT / "logs" / "overrefusal_d156_xsfull.sha256")
N_FRESH = 1000                                                                # D-152, D-244 (2)
# D-269 (6): the pairs with an item-9 record, and the record that holds the D-117 layer (Qwen2.5's L18: _xsfull only)
XSTEST_RECORD = {"repair_mistral_base2instruct": "overrefusal_d156_mistral.json",
                 "repair_gemma2_base2instruct": "overrefusal_d156_gemma2.json",
                 "repair_llama3_base2instruct": "overrefusal_d156_llama3.json",
                 "repair_llama31_base2instruct": "overrefusal_d156_llama31.json",
                 "repair_qwen_base2instruct": "overrefusal_d156_qwen_xsfull.json",
                 "repair_llama3_to_31": "overrefusal_d156_llama3_to_31.json"}
XS_KEYS = ("pre/over_refusal/eval", "post/over_refusal/eval")


def manifest_sha(man: Path, rel: str) -> str:
    """The sha256 a manifest names for rel; refused unless it names it exactly once (and every line has two fields)."""
    rows = [ln.split(maxsplit=1) for ln in man.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if any(len(r) != 2 for r in rows):
        raise SystemExit(f"REFUSING: {man.name} has a malformed line")
    hits = [r[0] for r in rows if r[1].strip().lstrip("*") == rel]
    if len(hits) != 1:
        raise SystemExit(f"REFUSING: {man.name} names {rel} {len(hits)} times")
    return hits[0]


def label(p: Path) -> str:
    """A repo-relative path, or the bare name of a file outside the repo (a test manifest)."""
    p = p.resolve()
    return str(p.relative_to(ROOT.resolve())).replace("\\", "/") if p.is_relative_to(ROOT.resolve()) else p.name


def thresholds(S: np.ndarray, target: float) -> np.ndarray:
    return np.array([P.thr_at_fpr(S[:, s], target) for s in range(S.shape[1])])


def jaccard(A: np.ndarray, B: np.ndarray) -> dict[str, np.ndarray]:
    """Per seed (columns of boolean (items, seeds) matrices): J (nan when the union is empty), |A|, |B|, |A n B|."""
    inter, union = (A & B).sum(axis=0), (A | B).sum(axis=0)
    return {"J": np.where(union > 0, inter / np.maximum(union, 1), np.nan), "nA": A.sum(axis=0), "nB": B.sum(axis=0),
            "nAB": inter}


class Acc:
    """Per-draw accumulator of one Jaccard comparison: the seed-mean J per draw, the undefined seeds, the counts."""

    def __init__(self) -> None:
        self.J: list[float] = []
        self.n_undef: list[int] = []
        self.per_seed: list[np.ndarray] = []
        self.counts: dict[str, list[np.ndarray]] = {"nA": [], "nB": [], "nAB": []}

    def add(self, j: dict[str, np.ndarray]) -> None:
        ok = np.isfinite(j["J"])
        self.J.append(float(j["J"][ok].mean()) if ok.any() else float("nan"))
        self.n_undef.append(int((~ok).sum()))
        self.per_seed.append(j["J"])
        for k in self.counts:
            self.counts[k].append(j[k])

    def report(self) -> dict[str, Any]:
        J, ps = np.array(self.J, dtype="<f8"), np.vstack(self.per_seed)
        seed_means = [float(np.nanmean(ps[1:, s])) if np.isfinite(ps[1:, s]).any() else None for s in range(ps.shape[1])]
        fin = [m for m in seed_means if m is not None]
        # D-235 practice: the per-draw array (index 0 the frozen split), so a within-draw contrast (J - null) needs no
        # re-run; None stands for an undefined draw (NaN in the hashed '<f8' bytes)
        per_draw = {"values": [float(v) if np.isfinite(v) else None for v in J], "sha256": sha256_bytes(J.tobytes()),
                    "dtype": "<f8", "index_0": "the frozen split"}
        return {"J": P.band_summary(J[1:], J[0]), "per_draw_J": per_draw,
                "frozen_split": {"J_per_seed": [None if not np.isfinite(v) else float(v) for v in ps[0]],
                                 **{k: [int(x) for x in v[0]] for k, v in self.counts.items()}},
                "J_per_seed_d156_mean": seed_means,
                # D-219's "mean and range over seeds": the range of the per-seed D-156 means (D-270)
                "J_seed_range_of_d156_means": [min(fin), max(fin)] if fin else None,
                "counts_seed_mean": {k: P.band_summary(np.array([x.mean() for x in v[1:]]), float(v[0].mean()))
                                     for k, v in self.counts.items()},
                "n_draws_with_an_undefined_seed": int(sum(1 for u in self.n_undef[1:] if u)),
                "n_undefined_seed_draws": int(sum(self.n_undef[1:])), "frozen_split_n_undefined_seeds": self.n_undef[0]}


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Item 18: flag-set churn under D-156 (D-219, D-269)")
    ap.add_argument("--pair", required=True, choices=list(POP.UPDATE_PAIRS_REPAIR))
    ap.add_argument("--n-draws", type=int, default=REGISTERED["n_draws"])
    ap.add_argument("--target-fpr", type=float, default=REGISTERED["target_fpr"])
    ap.add_argument("--seeds", type=int, nargs="*", default=REGISTERED["seeds"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--repair-manifest", default=str(REPAIR_MANIFEST),
                    help="TESTING ONLY: another manifest naming 4d A's record (a test before all 19 runs are pulled)")
    args = ap.parse_args()
    rep_man = Path(args.repair_manifest).resolve()
    stem, core = args.pair, args.pair[len("repair_"):]
    registered = (args.n_draws == REGISTERED["n_draws"] and args.target_fpr == REGISTERED["target_fpr"]
                  and list(args.seeds) == REGISTERED["seeds"])
    final_path = (RESULTS / f"churn_d156_{core}.json").resolve()
    out_path = Path(args.out).resolve() if args.out else final_path
    final = out_path == final_path
    if out_path.exists():
        raise SystemExit(f"REFUSING: {out_path} exists; results are never overwritten")
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test run (--out) must not write under results/")
    threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    if final and (not registered or any(t != "6" for t in threads.values()) or rep_man != REPAIR_MANIFEST.resolve()):
        raise SystemExit(f"REFUSING the final name: registered parameters {registered}, threads {threads}, "
                         f"repair manifest {rep_man.name}")
    code0 = P.code_block(CODE_FILES)

    # the selection of record and the D-117 cell
    sel_path = D.SELECTION_OF_RECORD
    sel_sha, cl_sha = sha256_file(sel_path), sha256_file(D.CLUSTERS)
    sel_doc = json.load(open(sel_path, encoding="utf-8"))
    if list(sel_doc["config"]["seeds"]) != list(args.seeds) or float(sel_doc["config"]["holdout"]) != P.HOLDOUT:
        raise SystemExit("REFUSING: selection seeds/holdout differ from this run's")
    sel = next((m for m in sel_doc["models"] if m["pair"] == stem), None)
    if sel is None or not sel["selected"]["arm"].startswith("C="):
        raise SystemExit(f"REFUSING: no logistic-C selection of record for {stem}")
    C_sel, L_sel = float(sel["selected"]["arm"].split("=")[1]), int(sel["selected"]["layer"])

    # anchor 1: the depth file (named by its manifest)
    depth_rel = f"results/depth_d156_{core}.json"
    db = (ROOT / depth_rel).read_bytes()
    depth_sha = sha256_bytes(db)
    if manifest_sha(DEPTH_MANIFEST, depth_rel) != depth_sha:
        raise SystemExit(f"REFUSING: {depth_rel} is not the file {DEPTH_MANIFEST.name} names")
    ddoc = json.loads(db.decode("utf-8"))
    dcfg = ddoc["config"]
    if (dcfg["selected_C"], dcfg["selected_layer"], dcfg["selection_sha256"]) != (C_sel, L_sel, sel_sha):
        raise SystemExit(f"REFUSING: {depth_rel} was run on another selection")
    dcells = {(c["layer"], c["probe_set"]): c for c in ddoc["cells"]}

    # anchor 2: 4d A's record (D-269 (7)); refused now, not after the compute, if it cannot match
    rep_rel = f"results/repair_d156_{core}.json"
    if not rep_man.exists():
        raise SystemExit(f"REFUSING: {rep_man.name} does not exist; item 18 runs after 4d A (D-219, D-269 (7))")
    rb = (ROOT / rep_rel).read_bytes()
    rep_sha = sha256_bytes(rb)
    if manifest_sha(rep_man, rep_rel) != rep_sha:
        raise SystemExit(f"REFUSING: {rep_rel} is not the file {rep_man.name} names")
    rdoc = json.loads(rb.decode("utf-8"))
    rcfg = rdoc["config"]
    early = [k for k, a, b in (("family", rcfg.get("family"), "repair"), ("pair", rcfg.get("pair"), stem),
                               ("registered", rdoc.get("registered_parameters"), True),
                               ("selected_C", rcfg.get("selected_C"), C_sel), ("selection_sha256", rcfg.get("selection_sha256"), sel_sha),
                               ("seeds", rcfg.get("seeds"), list(args.seeds)), ("n_draws", rcfg.get("n_draws"), args.n_draws),
                               ("target_fpr", rcfg.get("target_fpr"), args.target_fpr),
                               ("cluster_manifest", (rcfg.get("cluster_manifest") or {}).get("sha256"), cl_sha),
                               ("depth anchor", (rdoc.get("anchor") or {}).get("sha256"), depth_sha)) if a != b]
    rcells = {(c["layer"], c["probe_set"]): c for c in rdoc["cells"]}
    if early or any((L_sel, ps) not in rcells for ps in PROBE_SETS):
        raise SystemExit(f"REFUSING: {rep_rel} differs from this run in {early} or lacks L{L_sel}")

    cfg = REG.run(stem)
    axis = cfg["axis"]
    data: dict[str, Any] = {}
    input_sha: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    for side in ("pre", "post"):
        for part in ("probe_train", "threshold_calib", "eval"):
            z, r = REG.load(cfg[f"{side}_repo"], axis, part)
            data[f"{side}/{part}"], input_sha[f"{side}/{part}"], input_recs[f"{side}/{part}"] = z, r["sha256"], r

    # anchor 3: the item-9 record, for the pairs that have one (D-269 (6))
    or_rel, or_sha, ocells = None, None, None
    manifests_used = {DEPTH_MANIFEST: sha256_file(DEPTH_MANIFEST), rep_man: sha256_file(rep_man)}
    if stem in XSTEST_RECORD:
        or_rel = f"results/{XSTEST_RECORD[stem]}"
        ob = (ROOT / or_rel).read_bytes()
        or_sha = sha256_bytes(ob)
        named = [m for m in OR_MANIFESTS if any(ln.strip() and ln.split(maxsplit=1)[-1].strip().lstrip("*") == or_rel
                                                for ln in m.read_text(encoding="utf-8").splitlines())]
        if len(named) != 1 or manifest_sha(named[0], or_rel) != or_sha:
            raise SystemExit(f"REFUSING: {or_rel} is not the file one item-9 manifest names once")
        manifests_used[named[0]] = sha256_file(named[0])
        odoc = json.loads(ob.decode("utf-8"))
        ocfg = odoc["config"]
        if (ocfg["selected_C"], ocfg["selected_layer"], list(ocfg["seeds"]), ocfg["n_draws"], ocfg["target_fpr"],
                ocfg["selection_sha256"], odoc.get("registered_parameters")) != \
                (C_sel, L_sel, list(args.seeds), args.n_draws, args.target_fpr, sel_sha, True) or L_sel not in ocfg["layers"]:
            raise SystemExit(f"REFUSING: {or_rel} was run on another selection, seeds, draws or grid, or is not registered")
        for p_ in ("probe_train", "threshold_calib", "eval"):
            if odoc["input_cache_sha256"][f"pre/{axis}/{p_}"] != input_sha[f"pre/{p_}"]:
                raise SystemExit(f"REFUSING: {or_rel} read other pre-update {p_} bytes")
        for key in XS_KEYS:
            side = key.split("/")[0]
            repo = cfg[f"{side}_repo"]
            if "__xsfull@" in odoc["input_cache_source"][key]["file"]:
                repo = f"{repo}__xsfull"                                       # D-206: the tagged full-grid extraction
                if REG.revision(repo) != cfg[f"{side}_rev"]:
                    raise SystemExit(f"REFUSING: {repo}'s revision is not the pinned revision")
            z, r = REG.load(repo, "over_refusal", "eval")
            if r["sha256"] != odoc["input_cache_sha256"][key]:
                raise SystemExit(f"REFUSING: {key}: the registry's bytes are not the ones {or_rel} read")
            data[key], input_sha[key], input_recs[key] = z, r["sha256"], r
        ocells = {(c["layer"], c["probe_set"]): c for c in odoc["cells"]}

    reg_sources = REG.sources_sha256()
    if D.sources_now() != reg_sources:
        raise SystemExit("REFUSING: the extraction records or pins on disk are not the ones the registry read")
    for k6 in ("pre/probe_train", "pre/threshold_calib", "pre/eval", "post/probe_train", "post/threshold_calib", "post/eval"):
        if ddoc["input_cache_sha256"].get(k6) != input_sha[k6] or rdoc["input_cache_sha256"].get(k6) != input_sha[k6]:
            raise SystemExit(f"REFUSING: {k6}: the depth file or 4d A's record read other bytes")
    verification = REG.verification(input_recs)
    for part in ("probe_train", "threshold_calib", "eval"):
        a, b = data[f"pre/{part}"], data[f"post/{part}"]
        if not np.array_equal(a["item_ids"], b["item_ids"]) or not np.array_equal(a["labels"], b["labels"]):
            raise SystemExit(f"REFUSING: {part} items or labels differ between pre and post")
    if ocells is not None:
        xa, xb = data[XS_KEYS[0]], data[XS_KEYS[1]]
        if not np.array_equal(xa["item_ids"], xb["item_ids"]) or not np.array_equal(xa["labels"], xb["labels"]):
            raise SystemExit("REFUSING: XSTest items or labels differ between pre and post")
    for key, z in data.items():
        if "nonfinite" in z.files and z["nonfinite"].any():
            raise SystemExit(f"REFUSING: {key} has non-finite items")
        if f"layer_{L_sel}" not in z.files:
            raise SystemExit(f"REFUSING: {key} has no layer {L_sel}")

    y_tr = data["pre/probe_train"]["labels"].astype(bool)
    ncal = ~data["pre/threshold_calib"]["labels"].astype(bool)
    neg = ~data["pre/eval"]["labels"].astype(bool)
    pooled_ids = [str(i) for i in data["pre/threshold_calib"]["item_ids"][ncal]] + [str(i) for i in data["pre/eval"]["item_ids"][neg]]
    n_cal, n_pool = int(ncal.sum()), len(pooled_ids)
    cl = json.load(open(D.CLUSTERS, encoding="utf-8"))["axes"][axis]
    if cl["n_pooled_negatives"] != n_pool or cl["n_calibration"] != n_cal or \
            cl["pool_item_ids_sha256"] != sha256_bytes("\n".join(pooled_ids).encode("utf-8")):
        raise SystemExit(f"REFUSING: manifests/negative_clusters.json does not describe the {axis} pool")
    if n_pool - n_cal != 2 * N_FRESH:
        raise SystemExit(f"REFUSING: R_FRESH takes {N_FRESH} of {2 * N_FRESH} evaluation negatives; this pool has {n_pool - n_cal}")
    frozen_part = (np.arange(n_cal), np.arange(n_cal, n_pool))
    draws = {LEVELS[0]: P.repartition_draws(n_cal, n_pool - n_cal, args.n_draws, P.AXIS_DRAW_SEED[axis]),
             LEVELS[1]: P.cluster_repartition_draws(P.cluster_units(pooled_ids, cl["clusters"]), n_cal,
                                                    args.n_draws, P.AXIS_CLUSTER_DRAW_SEED[axis])}
    parts = {lvl: [frozen_part] + [(d[:n_cal], d[n_cal:]) for d in dr] for lvl, dr in draws.items()}
    x_safe = ~data[XS_KEYS[0]]["labels"].astype(bool) if ocells is not None else None
    if x_safe is not None and int(x_safe.sum()) != 250:
        raise SystemExit(f"REFUSING: XSTest holds {int(x_safe.sum())} safe prompts, not 250")
    print(f"churn {stem} [{axis}]: L{L_sel}, selected C {C_sel:g}; pooled {n_cal} + {n_pool - n_cal}; XSTest "
          f"{'from ' + or_rel if or_rel else 'none (no item-9 record)'}; D-170 flag {verification['flag']}; registered {registered}",
          flush=True)

    k_ = f"layer_{L_sel}"
    X = {key: z[k_] for key, z in data.items()}
    cells: list[dict[str, Any]] = []
    n_anchor = {"depth": 0, "repair_d156": 0, "item9": 0}
    n_rank_equal = n_null_exact = 0
    for pset in PROBE_SETS:
        probes = []
        for seed in args.seeds:
            _, fit = P.d117_split(len(y_tr), seed)
            rows, C, order = (fit, C_sel, "d117_permutation") if pset == "selected" else (np.sort(fit), 1.0, "file_order")
            probes.append(P.fit_probe(X["pre/probe_train"][rows], y_tr[rows], seed, C, order))
        S = lambda x: np.column_stack([q.score(x) for q in probes])  # noqa: E731
        pre_pool = np.vstack([S(X["pre/threshold_calib"][ncal]), S(X["pre/eval"][neg])])
        post_pool = np.vstack([S(X["post/threshold_calib"][ncal]), S(X["post/eval"][neg])])
        # all 450 XSTest prompts scored, then the safe rows taken, as item 9 scored them (review of D-269, S1)
        xs_pre = S(X[XS_KEYS[0]])[x_safe] if ocells is not None else None
        xs_post = S(X[XS_KEYS[1]])[x_safe] if ocells is not None else None
        # ties in a pool column are the only way the R_SAME threshold check below can fire; counted here, before the loop
        ties = {side: [int(M_.shape[0] - np.unique(M_[:, s]).size) for s in range(M_.shape[1])]
                for side, M_ in (("pre", pre_pool), ("post", post_pool))}
        cell: dict[str, Any] = {"layer": L_sel, "probe_set": pset, "probe_C": C_sel if pset == "selected" else 1.0,
                                "input_verification": verification, "n_tied_pool_scores_per_seed": ties}
        for lvl in LEVELS:
            acc = {"R_same": Acc(), "R_same_null": Acc()}
            if lvl == LEVELS[0]:
                acc.update({"R_fresh": Acc(), "R_fresh_null": Acc()})
            if ocells is not None:
                acc.update({"xstest|R_same": Acc(), "xstest|R_same_null": Acc()})
                if lvl == LEVELS[0]:
                    acc.update({"xstest|R_fresh": Acc(), "xstest|R_fresh_null": Acc()})
            fpr_A, fpr_B, xs_pre_mean, xs_frozen = [], [], [], None
            for d_i, (c_, e_) in enumerate(parts[lvl]):
                T = thresholds(pre_pool[c_], args.target_fpr)
                Tr = thresholds(post_pool[c_], args.target_fpr)                      # D-244 (5) on R_SAME
                Tq = np.array([P.quantile_transfer(pre_pool[c_, s], post_pool[c_, s], T[s]) for s in range(len(T))])
                if not np.array_equal(Tr, Tq):
                    raise SystemExit(f"REFUSING: {pset} {lvl} draw {d_i}: the R_SAME threshold is not 4d A's rank transfer")
                n_rank_equal += 1
                T0 = np.array([P.quantile_transfer(pre_pool[c_, s], pre_pool[c_, s], T[s]) for s in range(len(T))])
                if not np.array_equal(T0, T):
                    raise SystemExit(f"REFUSING: {pset} {lvl} draw {d_i}: the no-update repair does not return T_d")
                A, B, A0 = pre_pool[e_] >= T, post_pool[e_] >= Tr, pre_pool[e_] >= T0
                acc["R_same"].add(jaccard(A, B))
                jn = jaccard(A, A0)
                if not np.all(jn["J"][np.isfinite(jn["J"])] == 1.0):
                    raise SystemExit(f"REFUSING: {pset} {lvl} draw {d_i}: the R_SAME null is not exactly 1")
                n_null_exact += 1
                acc["R_same_null"].add(jn)
                fpr_A.append(float(P.floored_fpr(A).mean()))
                fpr_B.append(float(P.floored_fpr(B).mean()))
                if ocells is not None:
                    XA = xs_pre >= T
                    acc["xstest|R_same"].add(jaccard(XA, xs_post >= Tr))
                    jxn = jaccard(XA, xs_pre >= T0)                          # the no-update null (review of D-269, B1)
                    if not np.all(jxn["J"][np.isfinite(jxn["J"])] == 1.0):
                        raise SystemExit(f"REFUSING: {pset} {lvl} draw {d_i}: the XSTest R_SAME null is not exactly 1")
                    acc["xstest|R_same_null"].add(jxn)
                    xs_pre_mean.append(float(XA.sum(axis=0).mean()))
                    if d_i == 0:
                        xs_frozen = [int(v) for v in XA.sum(axis=0)]
                if lvl == LEVELS[0]:
                    perm = np.random.default_rng([POOL_SEED, d_i]).permutation(e_)       # as 4d B (poolstudy_d156)
                    Rf, E2 = perm[:N_FRESH], perm[N_FRESH:]
                    Tf, T0f = thresholds(post_pool[Rf], args.target_fpr), thresholds(pre_pool[Rf], args.target_fpr)
                    A2 = pre_pool[E2] >= T
                    acc["R_fresh"].add(jaccard(A2, post_pool[E2] >= Tf))
                    acc["R_fresh_null"].add(jaccard(A2, pre_pool[E2] >= T0f))
                    if ocells is not None:
                        XA = xs_pre >= T
                        acc["xstest|R_fresh"].add(jaccard(XA, xs_post >= Tf))
                        acc["xstest|R_fresh_null"].add(jaccard(XA, xs_pre >= T0f))
            # the anchors, exactly (a test with other parameters cannot match them)
            if registered:
                dc = dcells[(L_sel, pset)]
                want = dc["operating_point"]["fpr_pre"] if lvl == LEVELS[0] else dc[LEVELS[1]]["operating_point"]["fpr_pre"]
                if json.loads(json.dumps(P.band_summary(np.array(fpr_A[1:]), fpr_A[0]))) != want:
                    raise SystemExit(f"REFUSING: {pset} {lvl}: A's FPR is not {depth_rel}'s fpr_pre")
                n_anchor["depth"] += 1
                want_b = rcells[(L_sel, pset)][lvl]["quantile_thr_benign"]["fpr_post"]
                if json.loads(json.dumps(P.band_summary(np.array(fpr_B[1:]), fpr_B[0]))) != want_b:
                    raise SystemExit(f"REFUSING: {pset} {lvl}: B's FPR (R_SAME) is not {rep_rel}'s quantile_thr_benign fpr_post")
                n_anchor["repair_d156"] += 1
                if ocells is not None:
                    oc = ocells[(L_sel, pset)][lvl]
                    got = json.loads(json.dumps(P.band_summary(np.array(xs_pre_mean[1:]), xs_pre_mean[0])))
                    if got != oc["safe_flagged_pre_mean"] or xs_frozen != oc["frozen_split"]["safe_flagged_pre"]:
                        raise SystemExit(f"REFUSING: {pset} {lvl}: the XSTest pre flags are not {or_rel}'s")
                    n_anchor["item9"] += 1
            cell[lvl] = {name: a.report() for name, a in acc.items()}
        cells.append(cell)
        print(f"  {pset} done", flush=True)

    if registered:
        want_n = len(PROBE_SETS) * len(LEVELS)
        if n_anchor["depth"] != want_n or n_anchor["repair_d156"] != want_n or \
                n_anchor["item9"] != (want_n if ocells is not None else 0):
            raise SystemExit(f"REFUSING to write: anchor blocks {n_anchor}")
    code1 = P.code_block(CODE_FILES)
    changed = [n for n, a, b in (("code", code1["sha256"], code0["sha256"]), ("selection", sha256_file(sel_path), sel_sha),
                                 ("cluster manifest", sha256_file(D.CLUSTERS), cl_sha),
                                 (depth_rel, sha256_file(ROOT / depth_rel), depth_sha), (rep_rel, sha256_file(ROOT / rep_rel), rep_sha),
                                 ("registry sources", D.sources_now(), reg_sources)) if a != b]
    if or_rel and sha256_file(ROOT / or_rel) != or_sha:
        changed.append(or_rel)
    changed += [m.name for m, h in manifests_used.items() if sha256_file(m) != h]
    changed += [f"cache {r['file']}" for r in input_recs.values() if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    if changed:
        raise SystemExit(f"REFUSING to write: changed during the run: {changed}")
    head = next(c for c in cells if c["probe_set"] == "selected")[LEVELS[0]]
    payload = {**provenance_block("churn_d156"), "code": code0, "preregistration": "D-219 (author), D-269",
               "registered_parameters": registered, "blas_threads": threads, "input_verification": verification,
               "config": {"pair": stem, "axis": axis, "pre_repo": cfg["pre_repo"], "pre_rev": cfg["pre_rev"],
                          "post_repo": cfg["post_repo"], "post_rev": cfg["post_rev"],
                          "xstest_repos": ({k: input_recs[k]["file"] for k in XS_KEYS} if ocells is not None else None),
                          "selection_file": str(sel_path.relative_to(ROOT)).replace("\\", "/"),
                          "selection_sha256": sel_sha, "selected_C": C_sel, "selected_layer": L_sel, "seeds": args.seeds,
                          "target_fpr": args.target_fpr, "n_draws": args.n_draws, "draw_seed": P.AXIS_DRAW_SEED[axis],
                          "cluster_draw_seed": P.AXIS_CLUSTER_DRAW_SEED[axis], "r_fresh_seed": POOL_SEED, "n_fresh": N_FRESH,
                          "cluster_manifest": {"file": "manifests/negative_clusters.json", "sha256": cl_sha},
                          "input_registry_sources_sha256": reg_sources},
               "definitions": {
                   "A": "benign items flagged by the PRE model at T_d, the 1% point of the PRE scores on cal_d",
                   "B": "benign items flagged by the POST model at the 1% point of the POST scores on the reference set (D-244 (5))",
                   "J": "per seed |A n B| / |A u B|; per draw the mean over the seeds with a non-empty union",
                   "R_same": "PRIMARY: reference cal_d re-run through the post model; A and B over the draw's evaluation part",
                   "R_same_null": "pre against pre through the same repair: T_d itself, so J = 1 wherever it is defined (A "
                                  "non-empty), by construction; checked at every seed and draw",
                   "R_fresh": "BESIDE, item level: reference = 1000 of the evaluation negatives (4d B's split); A and B over the other 1000",
                   "R_fresh_null": "the PRE model at the 1% point of the PRE scores on R_fresh, against A",
                   "xstest": "the 250 XSTest safe prompts at the same thresholds (never re-partitioned, D-167)"},
               "headline": {"cell": "selected C, item level, R_same", "J": head["R_same"]["J"],
                            "null_J": head["R_same_null"]["J"],
                            "beside_R_fresh": {"J": head["R_fresh"]["J"], "null_J": head["R_fresh_null"]["J"]}},
               "checks": {"n_draw_blocks_R_same_threshold_equals_rank_transfer": n_rank_equal,
                          "n_draw_blocks_R_same_null_exactly_1": n_null_exact,
                          "anchor_blocks": n_anchor},
               "manifests_sha256": {label(m): h for m, h in manifests_used.items()},
               "anchors": {"depth": {"file": depth_rel, "sha256": depth_sha, "manifest": str(DEPTH_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
                                     "kind": "A's seed-mean floored FPR equals fpr_pre"},
                           "repair_d156": {"file": rep_rel, "sha256": rep_sha, "manifest": rep_man.name,
                                           "kind": "B's seed-mean floored FPR (R_same) equals quantile_thr_benign fpr_post"},
                           "item9": ({"file": or_rel, "sha256": or_sha,
                                      "kind": "the XSTest pre flags equal safe_flagged_pre_mean and the frozen-split counts"}
                                     if or_rel else None)},
               "input_cache_sha256": input_sha,
               "input_cache_source": {kk: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"]}
                                      for kk, r in input_recs.items()},
               "cells": cells}
    digest = write_results(out_path, payload, exclusive=True)

    def fmt(b: dict[str, Any]) -> str:
        return "undefined" if b["mean"] is None else "%.4f [%.4f, %.4f]" % (b["mean"], b["band95"][0], b["band95"][1])

    for name in ("R_same", "R_fresh"):
        print(f"selected C, item level, {name}: J {fmt(head[name]['J'])}; no-update null {fmt(head[name + '_null']['J'])}",
              flush=True)
    print(f"Written: {out_path}\nsha256:  {digest}\nanchor blocks: {n_anchor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
