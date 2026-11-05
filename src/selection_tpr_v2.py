"""Phase 2: the selection records' ho20 TPR line recomputed under split version 2's calibration (the author's D-324 (7);
D-320 (4), (5)(ii); D-326). NOTHING IS RE-SELECTED: each results/phase2/selection_p2__<model>@<rev12>__<pooling>.json
chose its arm and layer by ho20 AUROC on probe_train alone, and then reported, for the selected cell only, the ho20 TPR
at the target-FPR threshold set on v1 calibration's negatives (8,167 items). Under v2 (src/phase2_resplit_v2.py)
calibration holds 5,000 negatives drawn from v1 calibration, eval and pool4d, so that one line is recomputed here, with
the selection record's own probes. src/selection_p2.py is imported and not changed (its sha256 binds the selection
anchor); its reporting block is repeated below and proven equal to it on every run (step 3).

For one model and one pooling (both read from the command line; nothing else is an option), in this order; steps 1
and 2 read records only, so a missing v2 record or subset refuses in seconds, before any cache is read:
  1. SELECTION  results/phase2/selection_p2__<org>__<model>@<rev12>__<pooling>.json (read_result), where <rev12> is the
     revision of the model's select record (selection_p2.select_record). It must have run_kind selection_p2 and name
     this model, revision, pooling and axis; name the select record by file and by its sha256 as it is now; name this
     run's BLAS threads (selection_p2.check_threads: all three "6") and this run's numpy, scipy and scikit-learn; and
     name src/selection_p2.py, src/regularisation_sweep.py and src/paired.py at their present sha256 (BOUND_FILES). Its
     selected arm, layer L*, seeds, holdout and target FPR are the ones used. This run's host facts
     (selection_p2.HOST_KEYS) are recorded beside the selection record's (the anchor, not the host, is the test).
  2. RECORDS FOR v2 (no cache is read)
     v2 record  results/phase2/dataset_manifest_p2_v2.json (read_result): run_kind phase2_resplit_v2, split version 2,
                written by src/phase2_resplit_v2.py as it is in this tree (its code.sha256 entry equals the file's,
                which is in this run's code block), config_hash the hash of its config, config.excluded_items present
                (copied to the output). The v2 calibration sha256 is the record's (no pin: D-329 may still change v2).
                Its calibration partition names the literal path data/frozen_p2_v2/p2wg_v2__calibration.jsonl; the
                file must be those bytes, n rows, unique item ids, every label 0, no excluded item, no positive in the
                record. Its probe_train partition must be the select record's probe_train split (file and sha256), and
                its v1 calibration, eval and pool4d files (v1.partitions) the files the three caches below index.
     subset     results/phase2/extract__<org>__<model>__p2sub@<rev12>__rest.json (src/subset_p2.py; exactly one; run_kind
                subset_p2, cache_tag p2sub, cache_repo <repo>__p2sub, this revision, set rest, ingest yes). Its
                source_record must be the model's rest extraction record (extract__<org>__<model>__p2tpl@<rev12>__rest
                .json, read_result) as it is now. For p2wg/eval and p2wg/pool4d at the pooling: a well-formed cache name
                and sha256; no non-finite or truncated item counted; L* in layers_resolved; the source cache and the
                split file the rest record's; the cache in the subset's ingest index manifests/ingest_p2/<subset record
                stem>.sha256 with the record's sha256. Its layer plan for the pooling must name, in every entry where
                this model is the pair's pre side (role 'pre'), this selection record, its sha256 and L* (a FULL_GRID
                plan names none; recorded as such).
  3. ANCHOR  the select record's p2wg/probe_train and p2wg/calibration caches for the pooling, read by
     selection_p2.load_ingested exactly as the selection read them (data/hidden only, regular file, mode 444, in the
     ingest index with the record's sha256, hashing to it, item for item the frozen split), whose sha256 must be the
     selection record's input_cache_sha256. Per seed, selection_p2.sweep's reporting block: a seeded permutation splits
     probe_train into ho20 and the fitting 80%; standardise on the fitting part; fit the selected arm; threshold
     paired.thr_at_fpr on the calibration negatives' scores; TPR on the ho20 positives. Every per-seed TPR, their mean
     and their range must EQUAL the record's ho20_tpr_per_seed, ho20_tpr_at_1pct and ho20_tpr_range bit for bit;
     otherwise every difference is printed and nothing is written ("ANCHOR FAILED").
  4. v2 CALIBRATION  the subset's p2wg/eval and p2wg/pool4d caches, each read from data/hidden only (D-307 (1)): a
     regular file, mode 444, read ONCE (the bytes hashed are the bytes loaded, as src/subset_p2.py reads its sources),
     sha256 the record's, exactly the layers of layers_resolved, n items, no flagged item, item for item the v1 split
     file (hashed against the record). Each v2 calibration item's L* row is gathered BY item_id under the v2 record's
     JOIN RULE, from the cache of the split v1 put it in: v1 calibration from the select record's p2wg/calibration cache
     (step 3's), v1 eval and pool4d from the subset's. Refuses if an item is in none of the three caches or in two, if
     its label there is not 0, if the caches disagree on dtype or width, or if the items per v1 split differ from the v2
     record's plan.items_from_v1_to_v2 ('<v1 split>->calibration').
  5. v2 TPR  step 3's per-seed probes (fitted once; not refitted), thresholded on the v2 calibration negatives' scores;
     TPR on the same ho20 positives.

OUTPUT results/phase2/selection_tpr_v2__<org>__<model>@<rev12>__<pooling>.json, exclusive (never overwritten): the
inputs (every read result with its sha256, the caches' sha256, the split files, the ingest indexes), the v2 record's
code sha256 and excluded items, the subset's layer-plan check, the host facts, the v1 anchor values and their check,
the v2 per-seed TPRs with thresholds and counts, their mean and range, and the number of v2 calibration negatives with
how many came from each v1 split. A one-line summary is printed.

  python3 -u src/selection_tpr_v2.py --repo <repo> --pooling {usermean,lastgen}     (through scripts/project_env.sh)

CPU only.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import (  # noqa: E402
    _RESULT_READS,
    canonical_hash,
    ensure_utf8_stdout,
    provenance_block,
    read_result,
    sha256_bytes,
    sha256_file,
    write_results,
)
import paired as P  # noqa: E402
import provenance as PV  # noqa: E402
import regularisation_sweep as RS  # noqa: E402  (the Phase 1 selection's functions, as selection_p2 uses them)
import selection_p2 as SP  # noqa: E402  (imported, not changed: its paths, checks and loaders are used as they are)

V2_CODE_FILE = "src/phase2_resplit_v2.py"            # hashed, not imported: the v2 record must name this file's sha256
CODE_FILES = ["src/selection_tpr_v2.py", V2_CODE_FILE, *SP.CODE_FILES]
BOUND_FILES = ("src/selection_p2.py", "src/regularisation_sweep.py", "src/paired.py")   # the code that wrote the selection
AXIS = SP.AXIS
POOLINGS = SP.POOLINGS
SUB_TAG = "p2sub"                                     # subset_p2.TAG; not imported (this module reads, never cuts)
V1_SPLITS = ("calibration", "eval", "pool4d")         # where a v2 calibration item can come from (the join rule)
V2_RECORD_NAME = "dataset_manifest_p2_v2.json"
V2_CAL_FILE = "data/frozen_p2_v2/p2wg_v2__calibration.jsonl"

refuse = SP.refuse
rel = SP.rel


def arms() -> list[str]:
    """The arm names selection_p2.sweep can select."""
    return [f"C={C:g}" for C in RS.C_GRID] + ["mean_diff"]


# --------------------------------------------------------------------------- selection_p2.sweep's reporting block

def seed_probes(X: Any, y_all: np.ndarray, arm: str, seeds: list[int], holdout: float) -> list[dict[str, Any]]:
    """Per seed, the selected cell's probe exactly as selection_p2.sweep fits it for the reported TPR (its lines from
    'REPORTING ONLY'): the same permutation, split, standardisation and fit, in the same expressions. Returns per seed
    the standardisation, the weights and the ho20 positives' scores."""
    n = len(y_all)
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_ho = int(holdout * n)
        ho, fit = perm[:n_ho], perm[n_ho:]
        xf = X[fit]
        mu, sd = xf.mean(0), xf.std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        xf_s = (xf - mu) / sd
        if arm == "mean_diff":
            w, b = RS.fit_meandiff(xf_s, y_all[fit])
        else:
            w, b = RS.fit_logistic(xf_s, y_all[fit], seed, float(arm.split("=")[1]))
        s_ho = (X[ho] - mu) / sd @ w + b
        out.append({"seed": seed, "mu": mu, "sd": sd, "w": w, "b": b, "s_pos": s_ho[y_all[ho]]})
    return out


def tprs_at(probes: list[dict[str, Any]], X_thr: Any, neg: np.ndarray, target_fpr: float) -> list[dict[str, Any]]:
    """Per seed: the threshold at target_fpr on the negatives' scores (paired.thr_at_fpr, through regularisation_sweep as
    selection_p2 calls it) and the ho20 positives' TPR at it."""
    rows = []
    for p in probes:
        t = RS.thr_at_fpr(((X_thr - p["mu"]) / p["sd"] @ p["w"] + p["b"])[neg], target_fpr)
        hit = p["s_pos"] >= t
        rows.append({"seed": p["seed"], "threshold": float(t), "n_ho20_positive": int(len(hit)),
                     "n_ho20_positive_at_or_above": int(hit.sum()), "tpr": float(hit.mean())})
    return rows


def anchor_diffs(tprs: list[float], s: dict[str, Any]) -> tuple[list[str], int]:
    """(differences, values compared) between the recomputed per-seed TPRs and the selection record's selected cell:
    every per-seed value, the mean and the range must be EQUAL (exact after the JSON round trip)."""
    rec_rng = s.get("ho20_tpr_range") if isinstance(s.get("ho20_tpr_range"), list) else []
    rec_rng = (rec_rng + [None, None])[:2]
    pairs = [(f"ho20_tpr_per_seed[{i}]", a, b) for i, (a, b) in enumerate(zip(tprs, s["ho20_tpr_per_seed"]))]
    pairs += [("ho20_tpr_at_1pct", float(np.mean(tprs)), s.get("ho20_tpr_at_1pct")),
              ("ho20_tpr_range[0]", float(np.min(tprs)), rec_rng[0]),
              ("ho20_tpr_range[1]", float(np.max(tprs)), rec_rng[1])]
    diffs = [f"{lab}: recomputed {a!r}, recorded {b!r}" for lab, a, b in pairs
             if not (isinstance(b, float) and a == b)]
    return diffs, len(pairs)


# --------------------------------------------------------------------------- step 1: the selection record

def read_selection(repo: str, pooling: str, threads: dict[str, Any], code0: dict[str, Any]) -> dict[str, Any]:
    rec_path = SP.select_record(repo)
    rec_rel = rel(rec_path)
    rec = read_result(rec_path)
    if rec.get("repo") != repo or rec.get("set_name") != "select":
        refuse(f"{rec_rel} names repo {rec.get('repo')!r}, set {rec.get('set_name')!r}")
    if rec.get("ingest") != "yes":
        refuse(f"{rec_rel} says ingest: {rec.get('ingest')!r}; a result reads only ingested caches (D-307 (1))")
    rev = rec.get("revision")
    if not isinstance(rev, str) or len(rev) != 40 or rec.get("cache_revision") != rev \
            or rec.get("cache_repo") != f"{repo}__{SP.TAG}":
        refuse(f"{rec_rel}: revision, cache_revision or cache_repo is not the documented-tag form")
    sel_path = SP.P2 / f"selection_p2__{repo.replace('/', '__')}@{rev[:12]}__{pooling}.json"
    if sel_path.is_symlink() or not sel_path.is_file():
        refuse(f"no selection record {rel(sel_path)} (or not a regular file): run src/selection_p2.py first")
    sel_rel = rel(sel_path)
    sel = read_result(sel_path)
    cfg = sel.get("config") if isinstance(sel.get("config"), dict) else {}
    model = sel.get("model") if isinstance(sel.get("model"), dict) else {}
    if sel.get("run_kind") != "selection_p2":
        refuse(f"{sel_rel} has run_kind {sel.get('run_kind')!r}, not selection_p2")
    if (cfg.get("pre_repo"), cfg.get("pre_rev"), cfg.get("pooling"), cfg.get("axis")) != (repo, rev, pooling, AXIS) \
            or (model.get("repo"), model.get("revision"), model.get("pooling")) != (repo, rev, pooling):
        refuse(f"{sel_rel} does not name {repo}@{rev[:12]}, pooling {pooling}, axis {AXIS} (config and model)")
    er = sel.get("extraction_record") if isinstance(sel.get("extraction_record"), dict) else {}
    if er.get("file") != rec_rel or er.get("sha256") != _RESULT_READS[rec_rel]:
        refuse(f"{sel_rel} names the extraction record {er.get('file')!r} ({str(er.get('sha256'))[:16]}), not "
               f"{rec_rel} as it is now ({_RESULT_READS[rec_rel][:16]})")
    if cfg.get("blas_threads") != threads:
        refuse(f"{sel_rel} ran with BLAS threads {cfg.get('blas_threads')}, this run with {threads}")
    code = sel.get("code") if isinstance(sel.get("code"), dict) else {}
    want_libs = {k: (code.get("libraries") or {}).get(k) for k in SP.LIBS}
    have_libs = {k: code0["libraries"].get(k) for k in SP.LIBS}
    if have_libs != want_libs:
        refuse(f"libraries {have_libs} are not the ones {sel_rel} ran with {want_libs}")
    for f in BOUND_FILES:
        was, now = (code.get("sha256") or {}).get(f), code0["sha256"].get(f)
        if not isinstance(was, str) or was != now:
            refuse(f"{f} is not the code that wrote {sel_rel} (recorded {str(was)[:16]}, now {str(now)[:16]})")
    s = model.get("selected") if isinstance(model.get("selected"), dict) else {}
    seeds, holdout, target = cfg.get("seeds"), cfg.get("holdout"), cfg.get("target_fpr")
    arm, layer = s.get("arm"), s.get("layer")
    if arm not in arms():
        refuse(f"{sel_rel}: selected arm {arm!r} is not one of {arms()}")
    if not isinstance(layer, int) or isinstance(layer, bool) or layer not in (rec.get("layers_resolved") or []):
        refuse(f"{sel_rel}: selected layer {layer!r} is not a layer of {rec_rel}")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(x, int) and not isinstance(x, bool) for x in seeds) \
            or len(set(seeds)) != len(seeds):
        refuse(f"{sel_rel}: seeds {seeds!r} are not a list of distinct integers")
    if not isinstance(holdout, float) or not 0.0 < holdout < 1.0 or model.get("holdout_fraction") != holdout:
        refuse(f"{sel_rel}: holdout {holdout!r} (model holdout_fraction {model.get('holdout_fraction')!r})")
    if not isinstance(target, float) or not 0.0 < target < 1.0 or s.get("ho20_tpr_target_fpr") != target:
        refuse(f"{sel_rel}: target FPR {target!r} (selected cell's ho20_tpr_target_fpr {s.get('ho20_tpr_target_fpr')!r})")
    per = s.get("ho20_tpr_per_seed")
    if not isinstance(per, list) or len(per) != len(seeds):
        refuse(f"{sel_rel}: ho20_tpr_per_seed {per!r} does not hold one value per seed ({len(seeds)})")
    return {"rec_path": rec_path, "rec_rel": rec_rel, "rec": rec, "rev": rev, "sel_path": sel_path, "sel_rel": sel_rel,
            "sel": sel, "selected": s, "arm": arm, "layer": layer, "seeds": seeds, "holdout": holdout,
            "target_fpr": target}


# --------------------------------------------------------------------------- step 2: the records for v2 (no cache read)

def read_v2(code0: dict[str, Any]) -> dict[str, Any]:
    """The v2 record, bound to src/phase2_resplit_v2.py as it is in this tree, and its calibration file, checked
    (module docstring, step 2). The calibration file's sha256 is the record's: no pin, since the author's answer to
    D-329 may still change every v2 hash."""
    p = SP.P2 / V2_RECORD_NAME
    if p.is_symlink() or not p.is_file():
        refuse(f"no split v2 record {rel(p)} (or not a regular file): run src/phase2_resplit_v2.py first (D-320 (7), "
               "D-329)")
    v2_rel = rel(p)
    v2 = read_result(p)
    if v2.get("run_kind") != "phase2_resplit_v2" or v2.get("split_version") != 2:
        refuse(f"{v2_rel} has run_kind {v2.get('run_kind')!r}, split_version {v2.get('split_version')!r}")
    was = (((v2.get("code") or {}).get("sha256")) or {}).get(V2_CODE_FILE)
    now = code0["sha256"].get(V2_CODE_FILE)
    if not isinstance(was, str) or was != now:
        refuse(f"{v2_rel} was not written by {V2_CODE_FILE} as it is in this tree (recorded {str(was)[:16]}, now "
               f"{str(now)[:16]})")
    cfg = v2.get("config")
    if not isinstance(cfg, dict) or v2.get("config_hash") != canonical_hash(cfg):
        refuse(f"{v2_rel}: config_hash {str(v2.get('config_hash'))[:16]} is not the hash of its config")
    excluded = cfg.get("excluded_items")
    if not isinstance(excluded, dict):
        refuse(f"{v2_rel} lists no config.excluded_items")
    parts = (((v2.get("axes") or {}).get(AXIS) or {}).get("partitions")) or {}
    part = parts.get("calibration") or {}
    if part.get("file") != V2_CAL_FILE:
        refuse(f"{v2_rel} names {part.get('file')!r} for v2 calibration, not {V2_CAL_FILE}")
    if not isinstance(part.get("sha256"), str) or len(part["sha256"]) != 64:
        refuse(f"{v2_rel}: malformed v2 calibration sha256 {part.get('sha256')!r}")
    n = part.get("n")
    if not isinstance(n, int) or n < 1 or part.get("n_pos") != 0 or part.get("n_neg") != n:
        refuse(f"{v2_rel}: v2 calibration n {n!r}, n_pos {part.get('n_pos')!r}, n_neg {part.get('n_neg')!r} (negatives only)")
    f = SP.SPLIT_ROOT / V2_CAL_FILE
    if f.is_symlink() or not f.is_file():
        refuse(f"no v2 calibration file {V2_CAL_FILE} (or not a regular file)")
    b = f.read_bytes()
    if sha256_bytes(b) != part["sha256"]:
        refuse(f"{V2_CAL_FILE} is not the bytes {v2_rel} names ({sha256_bytes(b)[:16]} vs {part['sha256'][:16]})")
    rows = [json.loads(line) for line in b.decode("utf-8").split("\n") if line.strip()]  # not a result (D-297)
    if len(rows) != n:
        refuse(f"{V2_CAL_FILE} holds {len(rows)} rows, {v2_rel} says {n}")
    ids = [r.get("item_id") for r in rows]
    if not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids):
        refuse(f"{V2_CAL_FILE}: an item_id is missing or repeats")
    bad = [r.get("item_id") for r in rows if r.get("label") != 0 or isinstance(r.get("label"), bool)]
    if bad:
        refuse(f"{V2_CAL_FILE}: {len(bad)} rows whose label is not 0 (first {bad[0]})")
    held = sorted(set(ids) & set(excluded))
    if held:
        refuse(f"{V2_CAL_FILE} holds excluded items {held} (config.excluded_items of {v2_rel})")
    v1 = (v2.get("v1") or {}).get("partitions") or {}
    v1_files = {sp: {"file": (v1.get(sp) or {}).get("file"), "sha256": (v1.get(sp) or {}).get("sha256")}
                for sp in V1_SPLITS}
    tr = parts.get("probe_train") or {}
    plan = (v2.get("plan") or {}).get("items_from_v1_to_v2")
    if not isinstance(plan, dict):
        refuse(f"{v2_rel} has no plan.items_from_v1_to_v2")
    return {"path": p, "rel": v2_rel, "rec": v2, "rows": rows, "ids": ids, "n": n, "sha256": part["sha256"],
            "v1_files": v1_files, "probe_train": {"file": tr.get("file"), "sha256": tr.get("sha256")},
            "plan_counts": {sp: plan.get(f"{sp}->calibration", 0) for sp in V1_SPLITS},
            "join_rule": cfg.get("join_rule"), "excluded_items": excluded, "code_sha256": was}


def read_subset(repo: str, rev: str) -> dict[str, Any]:
    """The model's durable rest subset record, its ingest index and the rest extraction record it was cut from."""
    org_model = repo.replace("/", "__")
    found = sorted(SP.P2.glob(f"extract__{org_model}__{SUB_TAG}@*__rest.json"))
    if len(found) != 1:
        refuse(f"{len(found)} rest subset records for {repo} under {rel(SP.P2)} (want exactly one): "
               f"{[p.name for p in found]}; cut and ingest them with src/subset_p2.py first (D-324 (1))")
    p = found[0]
    sub_rel = rel(p)
    sub = read_result(p)
    if p.name != f"extract__{org_model}__{SUB_TAG}@{rev[:12]}__rest.json":
        refuse(f"{sub_rel} is not at the select record's revision {rev[:12]}")
    want = {"run_kind": "subset_p2", "cache_tag": SUB_TAG, "cache_repo": f"{repo}__{SUB_TAG}", "repo": repo,
            "revision": rev, "cache_revision": rev, "set_name": "rest", "ingest": "yes"}
    got = {k: sub.get(k) for k in want}
    if got != want:
        refuse(f"{sub_rel}: {[f'{k}={got[k]!r}' for k in want if got[k] != want[k]]}, expected {want}")
    src = sub.get("source_record") if isinstance(sub.get("source_record"), dict) else {}
    rest_path = SP.P2 / f"extract__{org_model}__{SP.TAG}@{rev[:12]}__rest.json"
    rest_rel = rel(rest_path)
    if src.get("file") != rest_rel:
        refuse(f"{sub_rel} names the source record {src.get('file')!r}, not {rest_rel}")
    if rest_path.is_symlink() or not rest_path.is_file():
        refuse(f"no rest extraction record {rest_rel} (or not a regular file)")
    rest = read_result(rest_path)
    if src.get("sha256") != _RESULT_READS[rest_rel]:
        refuse(f"{rest_rel} is not the bytes {sub_rel} was cut from ({_RESULT_READS[rest_rel][:16]} vs "
               f"{str(src.get('sha256'))[:16]})")
    if (rest.get("run_kind"), rest.get("repo"), rest.get("revision"), rest.get("set_name")) \
            != ("extract_p2", repo, rev, "rest"):
        refuse(f"{rest_rel} names run_kind {rest.get('run_kind')!r}, repo {rest.get('repo')!r}, revision "
               f"{str(rest.get('revision'))[:12]}, set {rest.get('set_name')!r}")
    index_path, index = SP.ingest_index(p)
    return {"path": p, "rel": sub_rel, "rec": sub, "rest_path": rest_path, "rest_rel": rest_rel, "rest": rest,
            "index_path": index_path, "index": index}


def check_l_star(sb: dict[str, Any], sr: dict[str, Any], pooling: str) -> dict[str, Any]:
    """The subset's layer plan for the pooling names, in every entry where this model is the pair's pre side, this
    selection record, its sha256 and L* (a FULL_GRID plan keeps every layer and names no L*; recorded as such)."""
    pp = ((sb["rec"].get("layer_plan") or {}).get("per_pooling") or {}).get(pooling)
    if not isinstance(pp, dict):
        refuse(f"{sb['rel']} has no layer_plan for {pooling}")
    if pp.get("full_grid_reason"):
        return {"full_grid_reason": pp["full_grid_reason"], "pre_entries": 0, "pairs": []}
    pre = [e for e in (pp.get("l_star") or []) if isinstance(e, dict) and e.get("role") == "pre"]
    want = (sr["sel_rel"], _RESULT_READS[sr["sel_rel"]], sr["layer"])
    got = [(e.get("selection_record"), e.get("selection_record_sha256"), e.get("layer")) for e in pre]
    if not pre or any(g != want for g in got):
        refuse(f"{sb['rel']}'s layer plan for {pooling} does not name {sr['sel_rel']} ({want[1][:16]}) at layer "
               f"{sr['layer']} in every entry where {sr['rec'].get('repo')} is the pre side: {got}")
    return {"full_grid_reason": None, "pre_entries": len(pre), "pairs": [e.get("pair") for e in pre]}


def subset_entry(sb: dict[str, Any], part: str, pooling: str, layer: int) -> dict[str, Any]:
    """One subset split's entry, checked on the records alone (module docstring, step 2): no cache is read."""
    key = f"{AXIS}/{part}/{pooling}"
    sub_rel = sb["rel"]
    s = (sb["rec"].get("splits") or {}).get(key)
    if not isinstance(s, dict):
        refuse(f"{sub_rel} has no split {key}")
    name, want = s.get("cache_file"), s.get("cache_sha256")
    if not isinstance(name, str) or "/" in name or "\\" in name or not isinstance(want, str) or len(want) != 64:
        refuse(f"{sub_rel} split {key}: malformed cache_file or cache_sha256")
    if s.get("n_nonfinite") != 0 or s.get("n_truncated") != 0:
        refuse(f"{name}: the record counts {s.get('n_nonfinite')} non-finite and {s.get('n_truncated')} truncated items")
    layers = s.get("layers_resolved")
    if not isinstance(layers, list) or layer not in layers:
        refuse(f"L* {layer} is not in {name}'s layers_resolved {layers} ({sub_rel})")
    src = (sb["rest"].get("splits") or {}).get(key) or {}
    if not isinstance(src.get("cache_sha256"), str) \
            or (s.get("source_cache_file"), s.get("source_cache_sha256")) != (src.get("cache_file"), src.get("cache_sha256")):
        refuse(f"{name}: its source cache {s.get('source_cache_file')!r} ({str(s.get('source_cache_sha256'))[:16]}) is not "
               f"the one {sb['rest_rel']} names for {key} ({src.get('cache_file')!r}, {str(src.get('cache_sha256'))[:16]})")
    if (s.get("split_file"), s.get("split_sha256")) != (src.get("split_file"), src.get("split_sha256")):
        refuse(f"{name}: its split file is not the one {sb['rest_rel']} names for {key}")
    if sb["index"].get(name) != want:
        refuse(f"{name} is not in the ingest index {rel(sb['index_path'])} with the record's sha256 "
               f"(index {str(sb['index'].get(name))[:16]})")
    return s


def load_subset_cache(sb: dict[str, Any], s: dict[str, Any], part: str, pooling: str,
                      layer: int) -> tuple[np.ndarray, list[str], list[int], dict[str, Any]]:
    """One durable subset cache at layer L*, from data/hidden only, read ONCE: the bytes hashed are the bytes loaded
    (module docstring, step 4). `s` is its entry, checked by subset_entry."""
    key = f"{AXIS}/{part}/{pooling}"
    sub_rel = sb["rel"]
    name, want, layers = s["cache_file"], s["cache_sha256"], s["layers_resolved"]
    p = SP.CACHE / name
    if p.is_symlink() or not p.is_file():
        refuse(f"{name} is not a regular file in {rel(SP.CACHE)}: ingest the subset record first (D-307 (1)); NVMe "
               "scratch is never read for a result")
    if (p.stat().st_mode & 0o777) != 0o444:
        refuse(f"{name} is mode {oct(p.stat().st_mode & 0o777)}, not 0o444 (an ingested cache is read-only)")
    raw = p.read_bytes()
    have = sha256_bytes(raw)
    if have != want:
        refuse(f"{name} is not the bytes {sub_rel} names ({have[:16]} vs {want[:16]})")
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        in_file = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("layer_"))
        if in_file != sorted(layers):
            refuse(f"{name} holds layers {in_file}, not the record's layers_resolved {layers}")
        if "nonfinite" not in z.files or bool(np.asarray(z["nonfinite"]).any()):
            refuse(f"{name} has no nonfinite flag, or flags a non-finite item")
        ids = [str(x) for x in z["item_ids"]]
        labels = [int(x) for x in z["labels"]]
        X = z[f"layer_{layer}"]
    del raw
    if len(ids) != s.get("n") or len(labels) != len(ids):
        refuse(f"{name} holds {len(ids)} items, the record says {s.get('n')}")
    if X.ndim != 2 or X.shape[0] != len(ids):
        refuse(f"{name} layer_{layer} has shape {X.shape}, not (n_items={len(ids)}, hidden)")
    rows = SP.split_rows(s, key)
    if [r["item_id"] for r in rows] != ids or [int(r["label"]) for r in rows] != labels:
        refuse(f"{name}: item ids or labels are not the frozen split's, item for item")
    return X, ids, labels, {"file": name, "sha256": have, "expected_from": sub_rel, "cache_repo": sb["rec"]["cache_repo"],
                            "layers_resolved": layers,
                            "split": {"file": s["split_file"], "sha256": s["split_sha256"]},
                            "source_cache": {"file": s["source_cache_file"], "sha256": s["source_cache_sha256"]}}


def gather(v2: dict[str, Any], sources: dict[str, tuple[np.ndarray, list[str], list[int]]]) -> tuple[np.ndarray,
                                                                                                   dict[str, int]]:
    """The v2 calibration matrix at L*, in the v2 file's order: each item's row BY item_id from the one v1 cache that
    holds it, copied bit for bit. Refuses a missing item, an item in two caches, a label that is not 0, or caches that
    disagree on dtype or width."""
    where: dict[str, tuple[str, int]] = {}
    for sname, (_, ids, _) in sources.items():
        for i, iid in enumerate(ids):
            if iid in where:
                refuse(f"item {iid} is in two v1 caches or twice in one ({where[iid][0]}, {sname})")
            where[iid] = (sname, i)
    shapes = {sname: (str(X.dtype), X.shape[1]) for sname, (X, _, _) in sources.items()}
    if len(set(shapes.values())) != 1:
        refuse(f"the v1 caches differ in dtype or width: {shapes}")
    first = next(iter(sources.values()))[0]
    out = np.empty((v2["n"], first.shape[1]), dtype=first.dtype)
    pos: dict[str, list[int]] = {sname: [] for sname in sources}
    src_rows: dict[str, list[int]] = {sname: [] for sname in sources}
    for j, iid in enumerate(v2["ids"]):
        hit = where.get(iid)
        if hit is None:
            refuse(f"v2 calibration item {iid} is in none of the v1 caches ({', '.join(sources)})")
        sname, i = hit
        if sources[sname][2][i] != 0:
            refuse(f"v2 calibration item {iid} has label {sources[sname][2][i]} in the v1 {sname} cache, not 0")
        pos[sname].append(j)
        src_rows[sname].append(i)
    for sname, (X, _, _) in sources.items():
        if pos[sname]:
            out[np.asarray(pos[sname])] = X[np.asarray(src_rows[sname])]
    return out, {sname: len(pos[sname]) for sname in sources}


# --------------------------------------------------------------------------- the run

def run(repo: str, pooling: str) -> Path:
    if pooling not in POOLINGS:
        refuse(f"pooling must be one of {POOLINGS}")
    threads = SP.check_threads()
    code0 = P.code_block(CODE_FILES)
    sr = read_selection(repo, pooling, threads, code0)
    rev, arm, layer = sr["rev"], sr["arm"], sr["layer"]
    seeds, holdout, target = sr["seeds"], sr["holdout"], sr["target_fpr"]
    out = SP.P2 / f"selection_tpr_v2__{repo.replace('/', '__')}@{rev[:12]}__{pooling}.json"
    if out.exists() or out.is_symlink():
        refuse(f"{rel(out)} exists; results are never overwritten (D-152 P5)")
    host = {"this_run": {k: PV.env_state().get(k) for k in SP.HOST_KEYS},
            "selection_record": {k: (sr["sel"].get("env") or {}).get(k) for k in SP.HOST_KEYS}}
    host["equal"] = host["this_run"] == host["selection_record"]

    # step 2: the records for v2, before any cache is read
    v2 = read_v2(code0)
    sb = read_subset(repo, rev)
    s_ev = subset_entry(sb, "eval", pooling, layer)
    s_p4 = subset_entry(sb, "pool4d", pooling, layer)
    l_star = check_l_star(sb, sr, pooling)
    sel_splits = sr["rec"].get("splits") or {}

    def split_of(e: dict[str, Any]) -> dict[str, Any]:
        return {"file": e.get("split_file"), "sha256": e.get("split_sha256")}
    mine = {"calibration": split_of(sel_splits.get(f"{AXIS}/calibration/{pooling}") or {}),
            "eval": split_of(s_ev), "pool4d": split_of(s_p4)}
    for sp in V1_SPLITS:
        if v2["v1_files"][sp] != mine[sp]:
            refuse(f"{v2['rel']} names v1 {sp} {v2['v1_files'][sp]}, but the cache for it indexes {mine[sp]}")
    tr_split = split_of(sel_splits.get(f"{AXIS}/probe_train/{pooling}") or {})
    if v2["probe_train"] != tr_split:
        refuse(f"{v2['rel']} names probe_train {v2['probe_train']}, not the select record's split {tr_split}")

    # step 3: the anchor, on the caches the selection read
    index_path, index = SP.ingest_index(sr["rec_path"])
    z_tr, r_tr, _ = SP.load_ingested(sr["rec"], sr["rec_rel"], index, "probe_train", pooling)
    z_cal, r_cal, _ = SP.load_ingested(sr["rec"], sr["rec_rel"], index, "calibration", pooling)
    recorded = sr["sel"].get("input_cache_sha256") if isinstance(sr["sel"].get("input_cache_sha256"), dict) else {}
    for part, r in (("probe_train", r_tr), ("calibration", r_cal)):
        k = f"pre/{AXIS}/{part}/{pooling}"
        if recorded.get(k) != r["sha256"]:
            refuse(f"{r['file']} ({r['sha256'][:16]}) is not the cache {sr['sel_rel']} read for {k} "
                   f"({str(recorded.get(k))[:16]})")
    y_all = z_tr["labels"].astype(bool)
    if sr["sel"]["model"].get("n_probe_train") != len(y_all):
        refuse(f"probe_train holds {len(y_all)} items, {sr['sel_rel']} says {sr['sel']['model'].get('n_probe_train')}")
    k = f"layer_{layer}"
    X_tr = z_tr[k]
    X_cal = z_cal[k]
    y_cal = z_cal["labels"].astype(bool)
    if y_cal.all():
        refuse("v1 calibration holds no negative")
    print(f"{repo}@{rev[:12]} [{AXIS}, {pooling}]: {arm} at layer {layer} (from {sr['sel_rel']}); seeds {seeds}, holdout "
          f"{holdout}, target FPR {target}; anchor on v1 calibration ({int((~y_cal).sum())} negatives)", flush=True)
    probes = seed_probes(X_tr, y_all, arm, seeds, holdout)
    v1_rows = tprs_at(probes, X_cal, ~y_cal, target)
    tprs_v1 = [r["tpr"] for r in v1_rows]
    diffs, n_cmp = anchor_diffs(tprs_v1, sr["selected"])
    if diffs:
        for d in diffs:
            print(f"  DIFFERENT {d}")
        raise SystemExit(f"ANCHOR FAILED: {len(diffs)} of {n_cmp} values are not equal to {sr['sel_rel']}; nothing "
                         "written")
    print(f"  ANCHOR OK: {n_cmp} of {n_cmp} values equal, bit for bit", flush=True)

    # step 4: v2 calibration, gathered by item_id from the three v1 caches
    X_ev, ids_ev, lab_ev, r_ev = load_subset_cache(sb, s_ev, "eval", pooling, layer)
    X_p4, ids_p4, lab_p4, r_p4 = load_subset_cache(sb, s_p4, "pool4d", pooling, layer)
    cal_split = r_cal["split"]
    sources = {"calibration": (X_cal, [str(x) for x in z_cal["item_ids"]], [int(x) for x in z_cal["labels"]]),
               "eval": (X_ev, ids_ev, lab_ev), "pool4d": (X_p4, ids_p4, lab_p4)}
    X_v2, counts = gather(v2, sources)
    if counts != v2["plan_counts"]:
        refuse(f"v2 calibration items per v1 split {counts} are not the v2 record's plan {v2['plan_counts']}")

    # step 5: the same probes, thresholded on the v2 calibration negatives
    v2_rows = tprs_at(probes, X_v2, np.ones(v2["n"], dtype=bool), target)
    tprs_v2 = [r["tpr"] for r in v2_rows]

    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"]:
        refuse("source changed during the run")
    caches = {f"pre/{AXIS}/probe_train/{pooling}": r_tr, f"pre/{AXIS}/calibration/{pooling}": r_cal,
              f"pre/{SUB_TAG}/{AXIS}/eval/{pooling}": r_ev, f"pre/{SUB_TAG}/{AXIS}/pool4d/{pooling}": r_p4}
    m1, m2 = float(np.mean(tprs_v1)), float(np.mean(tprs_v2))
    digest = write_results(out, {
        **provenance_block("selection_tpr_v2"),
        "decision": "D-324 (7) (the author: split v2 approved; the selection records' ho20 TPR line recomputed under v2 "
                    "calibration, selections unchanged); D-320 (4) (the join rule), (5)(ii); D-326 and D-329 (the v2 "
                    "split's exclusions, read from the v2 record)",
        "code": code0,
        "config": {"pre_repo": repo, "pre_rev": rev, "axis": AXIS, "pooling": pooling, "arm": arm, "layer": layer,
                   "seeds": seeds, "holdout": holdout, "target_fpr": target, "blas_threads": threads,
                   "settings_from": "the selection record: arm, layer, seeds, holdout and target FPR (nothing is "
                                    "re-selected and none is a command-line option)",
                   "probes": "per seed, selection_p2.sweep's reporting block, fitted once on the seed's fitting 80% "
                             "of probe_train and used for both thresholds",
                   "threshold_population_v1": "v1 calibration's negatives (the selection record's anchor)",
                   "threshold_population_v2": "v2 calibration's negatives, each L* row gathered by item_id from the "
                                              "cache of the split v1 put it in",
                   "join_rule": v2["join_rule"],
                   "inputs_from": "the selection record; the select record's ingested caches; the v2 record and "
                                  "calibration file; the durable rest subset's ingested caches"},
        "inputs": dict(_RESULT_READS),
        "selection_record": {"file": sr["sel_rel"], "sha256": _RESULT_READS[sr["sel_rel"]]},
        "extraction_record": {"file": sr["rec_rel"], "sha256": _RESULT_READS[sr["rec_rel"]]},
        "subset_record": {"file": sb["rel"], "sha256": _RESULT_READS[sb["rel"]]},
        "rest_extraction_record": {"file": sb["rest_rel"], "sha256": _RESULT_READS[sb["rest_rel"]]},
        "v2_record": {"file": v2["rel"], "sha256": _RESULT_READS[v2["rel"]],
                      "written_by": {V2_CODE_FILE: v2["code_sha256"], "equal_to_this_tree": True},
                      "excluded_items": v2["excluded_items"]},
        "subset_layer_plan": l_star,
        "host": host,
        "ingest_indexes": {"select": {"file": rel(index_path), "sha256": sha256_file(index_path)},
                           "subset": {"file": rel(sb["index_path"]), "sha256": sha256_file(sb["index_path"])}},
        "input_cache_sha256": {key: r["sha256"] for key, r in caches.items()},
        "input_cache_source": {key: f"{r['file']} (expected bytes from {r['expected_from']})"
                               for key, r in caches.items()},
        "split_files": {"v1/probe_train": r_tr["split"], "v1/calibration": cal_split, "v1/eval": r_ev["split"],
                        "v1/pool4d": r_p4["split"], "v2/calibration": {"file": V2_CAL_FILE, "sha256": v2["sha256"]}},
        "subset_caches": {"eval": {k2: r_ev[k2] for k2 in ("layers_resolved", "source_cache")},
                          "pool4d": {k2: r_p4[k2] for k2 in ("layers_resolved", "source_cache")}},
        "code_binding": {"files": {f: code0["sha256"][f] for f in BOUND_FILES},
                         "equal_to_the_selection_record": True},
        "selected": {"arm": arm, "layer": layer, "unchanged": True},
        "v1_anchor": {"ho20_tpr_per_seed_recorded": sr["selected"]["ho20_tpr_per_seed"],
                      "ho20_tpr_per_seed_recomputed": tprs_v1, "ho20_tpr_at_1pct": m1,
                      "ho20_tpr_range": [float(np.min(tprs_v1)), float(np.max(tprs_v1))],
                      "values_compared": n_cmp, "values_differing": 0, "exact_equal": True,
                      "n_calibration_negative": int((~y_cal).sum()), "per_seed": v1_rows},
        "v2": {"ho20_tpr_per_seed": tprs_v2, "ho20_tpr_mean": m2,
               "ho20_tpr_range": [float(np.min(tprs_v2)), float(np.max(tprs_v2))], "target_fpr": target,
               "n_calibration_negative": v2["n"], "from_v1_split": counts,
               "v2_record_plan_counts": v2["plan_counts"], "per_seed": v2_rows},
    }, exclusive=True)
    print(f"{repo}@{rev[:12]} [{pooling}] {arm} layer {layer}: ho20 TPR at FPR {target:g}, v1 calibration {m1:.4f} "
          f"(anchor exact) -> v2 calibration {m2:.4f} (seed range {min(tprs_v2):.4f}-{max(tprs_v2):.4f}); v2 negatives "
          f"{v2['n']} (from v1 calibration {counts['calibration']}, eval {counts['eval']}, pool4d {counts['pool4d']})")
    print(f"Written: {rel(out)}\nsha256:  {digest}")
    return out


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="the selection records' ho20 TPR line under split v2's calibration "
                                             "(D-324 (7)); one model, one pooling; nothing is re-selected")
    ap.add_argument("--repo", required=True, help="the selection's model (a pair's pre side)")
    ap.add_argument("--pooling", required=True, choices=POOLINGS)
    args = ap.parse_args()
    run(args.repo, args.pooling)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
