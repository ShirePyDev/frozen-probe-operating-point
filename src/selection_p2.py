"""Phase 2 D-117 selection: C and the deployer's layer for one model and one pooling (D-303 (1); docs/phase2_design.md
revision 3), chosen on the seed's held-out 20% of probe_train exactly as src/regularisation_sweep.py chose them for
Phase 1: the same C grid, arms (logistic at each C, and the mean-difference floor), split, standardisation, fits,
argmax over (arm, layer) on ho20 only, and reporting TPR at the target FPR. Its functions (C_GRID, auroc, fit_logistic,
fit_meandiff, thr_at_fpr) are imported, not copied, and sweep() below repeats its per-pair block computation for
computation; --anchor-phase1 proves that on a Phase 1 pair, BIT FOR BIT (the D-280 standard: D-280 reproduced two
selection grids bit for bit on the node).

Both modes refuse unless OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS are all "6", the pin the Phase 1
sweep ran under (D-152 P6; F-037: the thread count alone moves results). Run through scripts/project_env.sh.

ANCHOR MODE (--anchor-phase1 <stem>): sweep() on a Phase 1 pair's pre-side caches, read through src/input_registry.py
as the Phase 1 sweep read them (the same cache sha256s as results/selection_sweep_d165.json names), with that record's
seeds, holdout and target FPR, and numpy, scipy and scikit-learn at that record's versions. Every value of the pair's
entry must be EQUAL: the selected arm and layer, every grid value (auroc_fit, auroc_ho20, gap per arm and layer, in the
record's arm and layer order), the ho20 TPR and its seed range, n_probe_train and the holdout fraction. Writes
results/phase2/selection_p2_anchor__<stem>.json only then; otherwise it prints every difference and writes nothing.

PHASE 2 MODE (--repo, --pooling):
  anchor   refuses unless an anchor record in results/phase2/ was written by this exact code (every CODE_FILES sha256)
           with the same numpy, scipy and scikit-learn and the same BLAS threads. Its seeds, holdout and target FPR
           (the Phase 1 record's: D-117's 0-4, 0.20, 0.01) are the ones used; none is a command-line option, so no
           other setting can take the exclusive output name.
  inputs   the model's select record, results/phase2/extract__<org>__<model>__p2tpl@<rev12>__select.json (exactly
           one). It must say ingest: yes (D-307 (1)), hold the full every-second-layer grid, and name the model at its
           D-303 (8) revision (the Phase 1 pin where the Phase 1 extraction records name the model, else the config
           check's commit; the model must be in the config check). Its p2wg/probe_train and p2wg/calibration caches for
           the pooling are read from data/hidden only: a regular file (no symlink), mode 444, listed with the record's
           sha256 in the ingest index manifests/ingest_p2/<record stem>.sha256 (scripts/p2_ingest_record.sh), hashing
           to that sha256, holding the record's layers as (n, hidden) arrays, n items, no non-finite item, and item
           for item the frozen split's items and labels (the split file hashed against the record).
  selection  on probe_train's ho20 per seed. calibration (all negatives) sets the target-FPR threshold behind the
           ho20 TPR, which is REPORTING ONLY, as in Phase 1 (D-117); it is the TPR of per-seed probes fitted on 80% of
           probe_train, at a threshold from the frozen calibration split, not a deployed probe's operating TPR. eval
           is not read. The runner-up cell and its margin are recorded beside the selection.
  record   also the probe_train composition (per label: items, adversarial items; items in multi-item near-duplicate
           clusters), so the population the selection ran on is on the record (rule 5).
  output   results/phase2/selection_p2__<org>__<model>@<rev12>__<pooling>.json, exclusive (never overwritten). One
           selection per pooling: D-303 (d) records both poolings and names no primary one (an author question).

CPU only.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import (  # noqa: E402
    ensure_utf8_stdout,
    provenance_block,
    read_result,
    sha256_bytes,
    sha256_file,
    write_results,
)
import input_registry as REG  # noqa: E402
import paired as P  # noqa: E402
import provenance as PV  # noqa: E402
import regularisation_sweep as RS  # noqa: E402  (the Phase 1 selection's functions, imported, not copied)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
P2 = RESULTS / "phase2"
CACHE = ROOT / "data" / "hidden"
INGEST_INDEX = ROOT / "manifests" / "ingest_p2"
SPLIT_ROOT = ROOT                 # split files are named repo-relative in the extraction record
CONFIG_CHECK = RESULTS / "phase2_config_check.json"
PHASE1_SELECTION = RESULTS / "selection_sweep_d165.json"
CODE_FILES = ["src/selection_p2.py", "src/regularisation_sweep.py", "src/input_registry.py", "src/paired.py",
              "src/populations.py", "src/provenance.py", "src/superseded_provenance.py"]
TAG = "p2tpl"                     # extract_p2.TAG; not imported, so this module loads no torch-side code
POOLINGS = ("usermean", "lastgen")
AXIS = "p2wg"
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
THREADS = "6"
LIBS = ("numpy", "scipy", "scikit-learn")


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def rel(p: Path) -> str:
    """The name a read is registered and listed under (provenance._repo_rel, the form read_result uses)."""
    return PV._repo_rel(Path(p))


def check_threads() -> dict[str, str | None]:
    got = {v: os.environ.get(v) for v in THREAD_VARS}
    if any(x != THREADS for x in got.values()):
        refuse(f"BLAS threads {got}; every one must be {THREADS!r}, the Phase 1 sweep's pin (D-152 P6, F-037); "
               "run through scripts/project_env.sh")
    return got


# --------------------------------------------------------------------------- the D-117 selection (Phase 1's block)

def sweep(z_tr: Any, z_cal: Any, seeds: list[int], holdout: float, target_fpr: float) -> dict[str, Any]:
    """regularisation_sweep.main's per-pair computation, unchanged: per layer and seed, a seeded permutation splits
    probe_train into ho20 and the fitting 80%; features are standardised on the fitting part; each arm is fitted and
    scored by AUROC on both parts; means over seeds; the argmax over (arm, layer) on ho20 (the first maximum in arm
    then layer order is kept); then, for the selected cell only, the ho20 TPR at the threshold set on z_cal's
    negatives. z_tr and z_cal are npz-like: z[key] gives an array, z.files lists the keys. The runner-up and its
    margin are added beside (not part of the Phase 1 block, and not compared by the anchor)."""
    y_all = z_tr["labels"].astype(bool)
    n = len(y_all)
    keys = sorted((k for k in z_tr.files if k.startswith("layer_")), key=lambda s: int(s.split("_")[1]))
    grid: dict[str, dict[int, list[tuple[float, float]]]] = {}
    for k in keys:
        li = int(k.split("_")[1])
        X = z_tr[k]
        if X.ndim != 2 or X.shape[0] != n:
            refuse(f"{k} has shape {X.shape}, not (n_items={n}, hidden)")
        for seed in seeds:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n)
            n_ho = int(holdout * n)
            ho, fit = perm[:n_ho], perm[n_ho:]
            xf, yf = X[fit], y_all[fit]
            mu, sd = xf.mean(0), xf.std(0)
            sd = np.where(sd < 1e-8, 1.0, sd)
            xf_s = (xf - mu) / sd
            xh_s = (X[ho] - mu) / sd
            yh = y_all[ho]
            for C in RS.C_GRID:
                w, b = RS.fit_logistic(xf_s, yf, seed, C)
                grid.setdefault(f"C={C:g}", {}).setdefault(li, []).append(
                    (RS.auroc(xf_s @ w + b, yf), RS.auroc(xh_s @ w + b, yh)))
            w, b = RS.fit_meandiff(xf_s, yf)
            grid.setdefault("mean_diff", {}).setdefault(li, []).append(
                (RS.auroc(xf_s @ w + b, yf), RS.auroc(xh_s @ w + b, yh)))

    table: dict[str, dict[int, dict[str, float]]] = {}
    best: dict[str, Any] | None = None
    for arm, per_layer in grid.items():
        table[arm] = {}
        for li, vals in per_layer.items():
            f = float(np.mean([v[0] for v in vals]))
            h = float(np.mean([v[1] for v in vals]))
            table[arm][li] = {"auroc_fit": f, "auroc_ho20": h, "gap": f - h}
            if best is None or h > best["auroc_ho20"]:
                best = {"arm": arm, "layer": li, "auroc_fit": f, "auroc_ho20": h, "gap": f - h}
    if best is None:
        refuse("no layer_* array in the probe_train cache")

    # REPORTING ONLY (D-117): the selected probe's ho20 TPR at the threshold set on the calibration negatives
    k = f"layer_{best['layer']}"
    y_cal = z_cal["labels"].astype(bool)
    tprs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_ho = int(holdout * n)
        ho, fit = perm[:n_ho], perm[n_ho:]
        xf = z_tr[k][fit]
        mu, sd = xf.mean(0), xf.std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        xf_s = (xf - mu) / sd
        if best["arm"] == "mean_diff":
            w, b = RS.fit_meandiff(xf_s, y_all[fit])
        else:
            w, b = RS.fit_logistic(xf_s, y_all[fit], seed, float(best["arm"].split("=")[1]))
        t = RS.thr_at_fpr(((z_cal[k] - mu) / sd @ w + b)[~y_cal], target_fpr)
        s_ho = (z_tr[k][ho] - mu) / sd @ w + b
        tprs.append(float((s_ho[y_all[ho]] >= t).mean()))
    best["ho20_tpr_at_1pct"] = float(np.mean(tprs))       # Phase 1's key name, kept so the anchor compares like keys
    best["ho20_tpr_range"] = [float(np.min(tprs)), float(np.max(tprs))]
    best["ho20_tpr_per_seed"] = tprs
    best["ho20_tpr_target_fpr"] = target_fpr               # the FPR the key's '1pct' stands for in this record
    others = [(c["auroc_ho20"], a, li) for a, d in table.items() for li, c in d.items()
              if (a, li) != (best["arm"], best["layer"])]
    if others:
        h2, a2, l2 = max(others, key=lambda t: t[0])
        best["runner_up"] = {"arm": a2, "layer": l2, "auroc_ho20": h2, "margin": best["auroc_ho20"] - h2}
    return {"n_probe_train": n, "holdout_fraction": holdout, "selected": best,
            "grid": {a: {str(li): v for li, v in d.items()} for a, d in table.items()}}


def print_summary(label: str, m: dict[str, Any]) -> None:
    s = m["selected"]
    print(f"  {label}: SELECTED on ho20 {s['arm']} at layer {s['layer']}, ho20 AUROC {s['auroc_ho20']:.4f} "
          f"(fit {s['auroc_fit']:.4f}, gap {s['gap']:.4f}); ho20 TPR at the target FPR {s['ho20_tpr_at_1pct']:.4f} "
          f"(seed range {s['ho20_tpr_range'][0]:.4f}-{s['ho20_tpr_range'][1]:.4f})", flush=True)


# --------------------------------------------------------------------------- anchor on a Phase 1 pair

def compare(new: dict[str, Any], old: dict[str, Any]) -> tuple[list[str], int]:
    """(differences, values compared) between a sweep() result and a Phase 1 record's model entry: every value must
    be EQUAL (bit for bit after the JSON round trip, which is exact for floats), in the record's order."""
    diffs: list[str] = []
    n = 0

    def same(label: str, a: Any, b: Any) -> None:
        nonlocal n
        n += 1
        if a != b or (isinstance(a, float) and a != a):          # NaN never agrees
            diffs.append(f"{label}: {a!r} vs {b!r}")

    sn, so = new["selected"], old["selected"]
    for key in ("arm", "layer", "auroc_fit", "auroc_ho20", "gap", "ho20_tpr_at_1pct"):
        same(f"selected {key}", sn[key], so[key])
    for i in (0, 1):
        same(f"selected ho20_tpr_range[{i}]", sn["ho20_tpr_range"][i], so["ho20_tpr_range"][i])
    same("n_probe_train", new["n_probe_train"], old["n_probe_train"])
    same("holdout_fraction", new["holdout_fraction"], old["holdout_fraction"])
    same("arm order", list(new["grid"]), list(old["grid"]))
    for arm in [a for a in old["grid"] if a in new["grid"]]:
        same(f"{arm} layer order", list(new["grid"][arm]), list(old["grid"][arm]))
        for li in [x for x in old["grid"][arm] if x in new["grid"][arm]]:
            for k in ("auroc_fit", "auroc_ho20", "gap"):
                same(f"{arm} layer {li} {k}", new["grid"][arm][li][k], old["grid"][arm][li][k])
    return diffs, n


def run_anchor(stem: str) -> Path:
    threads = check_threads()
    code0 = P.code_block(CODE_FILES)
    out = P2 / f"selection_p2_anchor__{stem}.json"
    if out.exists():
        refuse(f"{rel(out)} exists; results are never overwritten (D-152 P5)")
    sel = read_result(PHASE1_SELECTION)
    entry = [m for m in sel.get("models", []) if m.get("pair") == stem]
    if len(entry) != 1:
        refuse(f"{rel(PHASE1_SELECTION)} holds {len(entry)} entries for {stem}")
    want_libs = {k: (sel.get("env") or {}).get("libraries", {}).get(k) for k in LIBS}
    have_libs = {k: code0["libraries"].get(k) for k in LIBS}
    if have_libs != want_libs:
        refuse(f"libraries {have_libs} are not the Phase 1 sweep's {want_libs}")
    p1_threads = {v: (sel.get("env") or {}).get("blas_threads", {}).get(v) for v in THREAD_VARS}
    if any(x != THREADS for x in p1_threads.values()):
        refuse(f"the Phase 1 record names BLAS threads {p1_threads}, not {THREADS} for each")
    c = sel["config"]
    seeds, holdout, target = c["seeds"], c["holdout"], c["target_fpr"]
    if c.get("C_grid") != RS.C_GRID:
        refuse(f"the Phase 1 record's C grid {c.get('C_grid')} is not regularisation_sweep.C_GRID {RS.C_GRID}")
    cfg = REG.run(stem)                                  # refuses an undeclared run, as the Phase 1 sweep did
    if c.get("runs", {}).get(stem) != cfg:
        refuse(f"the registry's run {cfg} is not the one the Phase 1 record names {c.get('runs', {}).get(stem)}")
    z_tr, r_tr = REG.load(cfg["pre_repo"], cfg["axis"], "probe_train")
    z_cal, r_cal = REG.load(cfg["pre_repo"], cfg["axis"], "threshold_calib")
    recs = {f"{stem}:pre/probe_train": r_tr, f"{stem}:pre/threshold_calib": r_cal}
    for k, r in recs.items():
        if sel["input_cache_sha256"].get(k) != r["sha256"]:
            refuse(f"{k}: the cache read now ({r['sha256'][:16]}) is not the one the Phase 1 record read "
                   f"({str(sel['input_cache_sha256'].get(k))[:16]})")
    print(f"ANCHOR {stem}: sweep() on the Phase 1 pre-side caches against {rel(PHASE1_SELECTION)}", flush=True)
    m = sweep(z_tr, z_cal, seeds, holdout, target)
    print_summary(stem, m)
    diffs, n = compare(m, entry[0])
    if diffs:
        for d in diffs:
            print(f"  DIFFERENT {d}")
        raise SystemExit(f"ANCHOR FAILED: {len(diffs)} of {n} values are not equal; nothing written")
    print(f"  ANCHOR OK: {n} of {n} values equal, bit for bit; selected {m['selected']['arm']} at layer "
          f"{m['selected']['layer']} as recorded", flush=True)
    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"]:
        refuse("source changed during the run")
    digest = write_results(out, {
        **provenance_block("selection_p2_anchor"),
        "preregistration": "D-280 standard: a re-computation reproduces the record of record bit for bit before the "
                           "module is used; this anchors src/selection_p2.sweep against the D-165 selection sweep",
        "code": code0,
        "config": {"runs": {stem: cfg}, "seeds": seeds, "holdout": holdout, "target_fpr": target,
                   "C_grid": RS.C_GRID, "equality": "exact", "blas_threads": threads,
                   "inputs_from": "src/input_registry.py"},
        "input_verification": REG.verification(recs),
        "input_cache_sha256": {k: r["sha256"] for k, r in recs.items()},
        "input_cache_source": {k: f"{r['file']} (expected bytes from {r['expected_from']})" for k, r in recs.items()},
        "registry_sources_sha256": REG.sources_sha256(),
        "anchor": {"file": rel(PHASE1_SELECTION), "sha256": sha256_file(PHASE1_SELECTION), "pair": stem,
                   "values_compared": n, "values_differing": 0, "exact_equal": True,
                   "selected": {k: m["selected"][k] for k in ("arm", "layer")}},
        "model": {"pair": stem, **m},
    }, exclusive=True)
    print(f"Written: {rel(out)}\nsha256:  {digest}")
    return out


# --------------------------------------------------------------------------- Phase 2 inputs

HOST_KEYS = ("python", "platform", "machine", "cpu_count")   # OpenBLAS picks its kernels by CPU: the same host class


def passing_anchor(code0: dict[str, Any], threads: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """The anchor record written by this exact code, libraries, threads and host facts; every anchor record read is
    listed."""
    found = sorted(P2.glob("selection_p2_anchor__*.json"))
    if not found:
        refuse(f"no anchor record in {rel(P2)}: run --anchor-phase1 <stem> with this code first")
    host = {k: PV.env_state().get(k) for k in HOST_KEYS}
    read, match = [], None
    for p in found:
        a = read_result(p)
        read.append({"file": rel(p), "sha256": sha256_file(p)})
        ok = (a.get("code", {}).get("sha256") == code0["sha256"]
              and {k: a.get("code", {}).get("libraries", {}).get(k) for k in LIBS}
              == {k: code0["libraries"].get(k) for k in LIBS}
              and {k: (a.get("env") or {}).get(k) for k in HOST_KEYS} == host
              and a.get("config", {}).get("blas_threads") == threads
              and a.get("anchor", {}).get("exact_equal") is True and a.get("anchor", {}).get("values_differing") == 0)
        if ok and match is None:
            match = {"file": rel(p), "sha256": sha256_file(p), "config": a["config"]}
    if match is None:
        refuse(f"none of the {len(found)} anchor records was written by this code with these libraries, threads and "
               "host facts; run --anchor-phase1 <stem> with this code first")
    return match, read


def select_record(repo: str) -> Path:
    org_model = repo.replace("/", "__")
    found = sorted(P2.glob(f"extract__{org_model}__{TAG}@*__select.json"))
    if len(found) != 1:
        refuse(f"{len(found)} select records for {repo} under {rel(P2)} (want exactly one): {[p.name for p in found]}")
    return found[0]


def check_revision(repo: str, rev: str, cc: dict[str, Any]) -> dict[str, Any]:
    """D-303 (8): where the Phase 1 extraction records name the model, its one pinned revision; else the config check's
    commit. The model must be in the config check either way. extract_p2 decides 'Phase 1 model' by its PHASE1_MODELS
    list instead; for all 17 models of the config check the two rules give the same revision (D-312 review), and a
    select record exists only after extract_p2 accepted its revision."""
    m = (cc.get("models") or {}).get(repo)
    if not isinstance(m, dict):
        refuse(f"the config check names no model {repo}")
    cc_sha = m.get("sha")
    pinned = sorted({r["revision"] for r in REG._load_records() if r["repo"] == repo})
    if pinned:
        if pinned != [rev]:
            refuse(f"the select record names {repo} at {rev[:12]}, but the Phase 1 records pin {[v[:12] for v in pinned]}")
        source = "phase1_pin"
    else:
        if cc_sha != rev:
            refuse(f"the select record names {repo} at {rev[:12]}, not the config check's {str(cc_sha)[:12]}")
        source = "config_check"
    return {"revision_source": source, "config_check_sha": cc_sha, "phase1_pins": pinned,
            "registry_sources_sha256": REG.sources_sha256()}


def ingest_index(rec_path: Path) -> tuple[Path, dict[str, str]]:
    """manifests/ingest_p2/<record stem>.sha256 (written by scripts/p2_ingest_record.sh): {cache file: sha256}."""
    p = INGEST_INDEX / f"{rec_path.stem}.sha256"
    if p.is_symlink() or not p.is_file():
        refuse(f"no ingest index {rel(p)} (or not a regular file): ingest the select record with "
               "scripts/p2_ingest_record.sh first (D-307 (1))")
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").split("\n")[:-1]:
        h, _, path = line.partition("  ")
        if not path.startswith("data/hidden/") or len(h) != 64:
            refuse(f"{rel(p)}: malformed line {line!r}")
        name = path[len("data/hidden/"):]
        if name in out:
            refuse(f"{rel(p)} names {name} twice")
        out[name] = h
    return p, out


def split_rows(s: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """The frozen split the cache was extracted from, hashed against the extraction record."""
    f = s.get("split_file")
    if not isinstance(f, str) or Path(f).is_absolute() or ".." in Path(f).parts:
        refuse(f"split {key}: split_file {f!r} is not a repo-relative path")
    b = (SPLIT_ROOT / f).read_bytes()
    if sha256_bytes(b) != s.get("split_sha256"):
        refuse(f"{f} is not the bytes the extraction record names for {key}")
    return [json.loads(line) for line in b.decode("utf-8").split("\n") if line.strip()]  # not a result (D-297)


def load_ingested(rec: dict[str, Any], rec_rel: str, index: dict[str, str], part: str,
                  pooling: str) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    """One cache of the select record, from data/hidden only, checked before use (module docstring)."""
    key = f"{AXIS}/{part}/{pooling}"
    s = (rec.get("splits") or {}).get(key)
    if not isinstance(s, dict):
        refuse(f"{rec_rel} has no split {key}")
    name, want = s.get("cache_file"), s.get("cache_sha256")
    if not isinstance(name, str) or "/" in name or "\\" in name or not isinstance(want, str) or len(want) != 64:
        refuse(f"{rec_rel} split {key}: malformed cache_file or cache_sha256")
    if s.get("n_nonfinite") != 0:
        refuse(f"{name}: the record counts {s.get('n_nonfinite')} non-finite items (primary inputs must be finite)")
    if index.get(name) != want:
        refuse(f"{name} is not in the ingest index with the record's sha256 (index {str(index.get(name))[:16]})")
    p = CACHE / name
    if p.is_symlink() or not p.is_file():
        refuse(f"{name} is not a regular file in {rel(CACHE)}: ingest the select record first (D-307 (1)); NVMe "
               "scratch is never read for a result")
    if (p.stat().st_mode & 0o777) != 0o444:
        refuse(f"{name} is mode {oct(p.stat().st_mode & 0o777)}, not 0o444 (an ingested cache is read-only)")
    have = sha256_file(p)
    if have != want:
        refuse(f"{name} is not the bytes {rec_rel} names ({have[:16]} vs {want[:16]})")
    z = np.load(p, allow_pickle=False)
    layers = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("layer_"))
    if layers != list(rec.get("layers_resolved") or []):
        refuse(f"{name} holds layers {layers}, not the record's {rec.get('layers_resolved')}")
    if "nonfinite" not in z.files or bool(np.asarray(z["nonfinite"]).any()):
        refuse(f"{name} has no nonfinite flag, or flags a non-finite item")
    n = len(z["labels"])
    if n != s.get("n"):
        refuse(f"{name} holds {n} items, the record says {s.get('n')}")
    rows = split_rows(s, key)
    if [r["item_id"] for r in rows] != [str(x) for x in z["item_ids"]] \
            or [int(r["label"]) for r in rows] != [int(x) for x in z["labels"]]:
        refuse(f"{name}: item ids or labels are not the frozen split's, item for item")
    return z, {"file": name, "sha256": have, "expected_from": rec_rel,
               "split": {"file": s["split_file"], "sha256": s["split_sha256"]}}, rows


def composition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per label: items and adversarial items; items in multi-item near-duplicate clusters (rule 5, the population)."""
    cl = collections.Counter(r.get("cluster") for r in rows)
    multi = {c: k for c, k in cl.items() if c is not None and k > 1}
    out: dict[str, Any] = {}
    for lab in (1, 0):
        rs = [r for r in rows if int(r["label"]) == lab]
        out[f"label_{lab}"] = {"n": len(rs), "adversarial": sum(str(r.get("adversarial")) == "True" for r in rs)}
    out["multi_item_clusters"] = len(multi)
    out["items_in_multi_item_clusters"] = int(sum(multi.values()))
    out["largest_cluster"] = int(max(multi.values())) if multi else 1
    out["mixed_label_cluster_items"] = sum(bool(r.get("mixed_label_cluster")) for r in rows)
    return out


def run_phase2(repo: str, pooling: str) -> Path:
    if pooling not in POOLINGS:
        refuse(f"pooling must be one of {POOLINGS}")
    threads = check_threads()
    code0 = P.code_block(CODE_FILES)
    anchor, anchors_read = passing_anchor(code0, threads)
    seeds, holdout, target_fpr = anchor["config"]["seeds"], anchor["config"]["holdout"], anchor["config"]["target_fpr"]
    rec_path = select_record(repo)
    rec_rel = rel(rec_path)
    rec = read_result(rec_path)
    if rec.get("repo") != repo or rec.get("set_name") != "select":
        refuse(f"{rec_rel} names repo {rec.get('repo')!r}, set {rec.get('set_name')!r}")
    if rec.get("ingest") != "yes":
        refuse(f"{rec_rel} says ingest: {rec.get('ingest')!r}; a result reads only ingested caches (D-307 (1))")
    if rec.get("layers_every_second_grid") is not True:
        refuse(f"{rec_rel} does not hold the full every-second-layer grid the selection runs over (D-303 (1))")
    rev = rec.get("revision")
    if not isinstance(rev, str) or len(rev) != 40 or rec.get("cache_revision") != rev \
            or rec.get("cache_repo") != f"{repo}__{TAG}":
        refuse(f"{rec_rel}: revision, cache_revision or cache_repo is not the documented-tag form")
    cc = read_result(CONFIG_CHECK)
    rev_block = check_revision(repo, rev, cc)
    out = P2 / f"selection_p2__{repo.replace('/', '__')}@{rev[:12]}__{pooling}.json"
    if out.exists():
        refuse(f"{rel(out)} exists; results are never overwritten (D-152 P5)")
    index_path, index = ingest_index(rec_path)

    z_tr, r_tr, rows_tr = load_ingested(rec, rec_rel, index, "probe_train", pooling)
    z_cal, r_cal, _ = load_ingested(rec, rec_rel, index, "calibration", pooling)
    y_tr, y_cal = z_tr["labels"].astype(bool), z_cal["labels"].astype(bool)
    if y_tr.all() or not y_tr.any():
        refuse("probe_train holds one class only")
    if y_cal.all():
        refuse("calibration holds no negative")
    comp = composition(rows_tr)
    print(f"{repo}@{rev[:12]} [{AXIS}, {pooling}]: probe_train {len(y_tr)} ({int(y_tr.sum())} positive), calibration "
          f"{len(y_cal)} ({int((~y_cal).sum())} negative); selection on ho20 only, eval not read; probe_train "
          f"composition {comp}", flush=True)
    m = sweep(z_tr, z_cal, seeds, holdout, target_fpr)
    print_summary(pooling, m)

    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"]:
        refuse("source changed during the run")
    inputs = {f"pre/{AXIS}/probe_train/{pooling}": r_tr, f"pre/{AXIS}/calibration/{pooling}": r_cal}
    digest = write_results(out, {
        **provenance_block("selection_p2"),
        "preregistration": "D-117 (the protocol, unchanged: src/regularisation_sweep.py's functions, anchored bit for "
                           "bit); D-303 (1) (the selection runs on each pair's pre side over the full grid); D-303 (d) "
                           "(both poolings are recorded: one selection per pooling, no primary pooling named)",
        "code": code0,
        "config": {"pre_repo": repo, "pre_rev": rev, "axis": AXIS, "pooling": pooling, "C_grid": RS.C_GRID,
                   "seeds": seeds, "holdout": holdout, "target_fpr": target_fpr, "blas_threads": threads,
                   "settings_from": "the anchor record's config (the Phase 1 selection record's)",
                   "selection_population": "held-out 20% of probe_train, per seed (item-level permutation, as Phase 1)",
                   "eval_used_for_selection": False,
                   "threshold_population": "the calibration split's negatives, for the reported ho20 TPR only",
                   "inputs_from": "the model's Phase 2 select record; caches read from data/hidden (ingested, indexed)"},
        "anchor": anchor, "anchors_read": anchors_read,
        "revision_check": rev_block,
        "extraction_record": {"file": rec_rel, "sha256": sha256_file(rec_path)},
        "config_check": {"file": rel(CONFIG_CHECK), "sha256": sha256_file(CONFIG_CHECK)},
        "ingest_index": {"file": rel(index_path), "sha256": sha256_file(index_path)},
        "input_cache_sha256": {k: r["sha256"] for k, r in inputs.items()},
        "input_cache_source": {k: f"{r['file']} (expected bytes from {r['expected_from']})" for k, r in inputs.items()},
        "split_files": {k: r["split"] for k, r in inputs.items()},   # hashed above against the extraction record
        "n_calibration": int(len(y_cal)), "n_calibration_negative": int((~y_cal).sum()),
        "n_probe_train_positive": int(y_tr.sum()),
        "probe_train_composition": comp,
        "model": {"repo": repo, "revision": rev, "pooling": pooling, **m},
    }, exclusive=True)
    print(f"Written: {rel(out)}\nsha256:  {digest}")
    return out


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Phase 2 D-117 selection (one model, one pooling), or its Phase 1 anchor; "
                                             "seeds, holdout and target FPR come from the Phase 1 record")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--repo", help="the pre-side model (Phase 2 mode)")
    g.add_argument("--anchor-phase1", metavar="STEM", help="a Phase 1 pair in results/selection_sweep_d165.json")
    ap.add_argument("--pooling", choices=POOLINGS)
    args = ap.parse_args()
    if args.anchor_phase1:
        if args.pooling:
            refuse("the anchor takes no --pooling")
        run_anchor(args.anchor_phase1)
    else:
        if not args.pooling:
            refuse("--pooling is required with --repo")
        run_phase2(args.repo, args.pooling)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
