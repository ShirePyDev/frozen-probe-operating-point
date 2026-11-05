"""Do the three pre-unification thr_at_fpr variants move any threshold or floor on real scores?

Commit 5e55307 replaced 16 local copies of the frozen-threshold rule with one import from
src/paired.py. The copies were three textual variants of one loop (the smallest distinct
score t with (neg >= t).mean() <= target, else +inf), reproduced verbatim below from the
blobs at 5e55307^:
  A  14 files, e.g. src/depth_curve.py         blob dadeb98b87310bb448d8981d41f3eb711e28e54f
  B  src/pilot_e6.py (threshold_at_fpr)        blob 82419b077e03e4ff3c9a7d2ca4fbd47ad40649d4
  C  src/repair_nonparametric.py               blob 4bdc5df48ccf37c8b19498b953ac72e90fee095b
paired.thr_at_fpr is the vectorised form of the same definition.

On the nine released pairs at their D-117 selected layer, selected C and C=1, five seeds
(D-152 P1/P2 probes, exactly as src/damage_d156.py fits them), every threshold of the
frozen split and of the 1000 D-156 re-partitions is computed by all four and compared
exactly, and the floor (realised pre-update FPR on the evaluation part over the target,
mean over seeds) is compared per partition. Reads frozen caches; CPU only.
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
from provenance import ensure_utf8_stdout, provenance_block, resolve_cache, sha256_file, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "hidden"
RESULTS = ROOT / "results"
CODE_FILES = ["src/thr_variants_check.py", "src/paired.py", "src/populations.py", "src/provenance.py",
              "src/superseded_provenance.py"]


# --- the three variants, verbatim from 5e55307^ ------------------------------------------
def variant_a(neg, target):
    c = np.concatenate([np.unique(neg), [np.inf]])
    for t in c:
        if (neg >= t).mean() <= target:
            return float(t)
    return float(np.inf)


def variant_b(neg_scores: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold whose FPR on negatives is <= target. FLAG when >= thr."""
    cands = np.unique(neg_scores)
    cands = np.concatenate([cands, [np.inf]])
    for t in cands:
        if (neg_scores >= t).mean() <= target_fpr:
            return float(t)
    return float(np.inf)


def variant_c(neg: np.ndarray, target: float) -> float:
    c = np.concatenate([np.unique(neg), [np.inf]])
    for t in c:
        if (neg >= t).mean() <= target:
            return float(t)
    return float(np.inf)


VARIANTS = {"A_14_files": variant_a, "B_pilot_e6": variant_b, "C_repair_nonparametric": variant_c}


def load(repo: str, rev: str, axis: str, part: str):
    p = resolve_cache(f"{repo.replace('/', '__')}@{rev[:12]}__{axis}__{part}__mean.npz", CACHE)
    if not p.exists():
        raise FileNotFoundError(f"missing cache {p.name}")
    return np.load(p, allow_pickle=False), p


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Real-data check of the three thr_at_fpr variants")
    ap.add_argument("--n-draws", type=int, default=P.N_DRAWS)
    ap.add_argument("--out", default=str(RESULTS / "thr_variants_check.json"))
    args = ap.parse_args()
    out_path = Path(args.out).resolve()
    if out_path.exists():
        raise SystemExit(f"REFUSING: {out_path} exists")
    code0 = P.code_block(CODE_FILES)
    sel_path = RESULTS / "regularisation_sweep.json"
    sel_doc = json.load(open(sel_path, encoding="utf-8"))
    target = P.TARGET_FPR
    rows: list[dict[str, Any]] = []
    input_sha: dict[str, str] = {}
    totals = {v: {"thresholds": 0, "differing": 0, "floors": 0, "floors_differing": 0} for v in VARIANTS}
    for stem in POP.UPDATE_PAIRS_REPAIR:
        m = next(x for x in sel_doc["models"] if x["pair"] == stem)
        C, L = float(m["selected"]["arm"].split("=")[1]), int(m["selected"]["layer"])
        cfg = json.load(open(RESULTS / f"{stem}.json", encoding="utf-8"))["config"]
        axis, k = cfg.get("axis", "safety"), f"layer_{L}"
        z = {}
        for part in ("probe_train", "threshold_calib", "eval"):
            z[part], p = load(cfg["pre_repo"], cfg["pre_rev"], axis, part)
            input_sha[f"{stem}:pre/{part}"] = sha256_file(p)
            if sel_doc["input_cache_sha256"].get(f"{stem}:pre/{part}", input_sha[f"{stem}:pre/{part}"]) != input_sha[f"{stem}:pre/{part}"]:
                raise SystemExit(f"REFUSING: {stem} pre/{part} is not the bytes the selection used")
        y_tr = z["probe_train"]["labels"].astype(bool)
        ncal = ~z["threshold_calib"]["labels"].astype(bool)
        nev = ~z["eval"]["labels"].astype(bool)
        n_cal = int(ncal.sum())
        n_pool = n_cal + int(nev.sum())
        draws = P.repartition_draws(n_cal, n_pool - n_cal, args.n_draws, P.AXIS_DRAW_SEED[axis])
        parts = [(np.arange(n_cal), np.arange(n_cal, n_pool))] + [(d[:n_cal], d[n_cal:]) for d in draws]
        X_tr = z["probe_train"][k]
        for pset in ("selected", "C=1"):
            probes = []
            for seed in (0, 1, 2, 3, 4):
                _, fit = P.d117_split(len(y_tr), seed)
                if pset == "selected":
                    probes.append(P.fit_probe(X_tr[fit], y_tr[fit], seed, C, "d117_permutation"))
                else:
                    fo = np.sort(fit)
                    probes.append(P.fit_probe(X_tr[fo], y_tr[fo], seed, 1.0, "file_order"))
            pool = np.vstack([np.column_stack([q.score(z["threshold_calib"][k][ncal]) for q in probes]),
                              np.column_stack([q.score(z["eval"][k][nev]) for q in probes])])
            row = {"pair": stem, "axis": axis, "layer": L, "probe_set": pset, "n_partitions": len(parts)}
            for vname, fn in VARIANTS.items():
                n_diff, max_abs, n_fdiff, max_fdiff = 0, 0.0, 0, 0.0
                for cal, ev in parts:
                    t_ref = np.array([P.thr_at_fpr(pool[cal, s], target) for s in range(pool.shape[1])])
                    t_var = np.array([fn(pool[cal, s], target) for s in range(pool.shape[1])])
                    d = t_ref != t_var
                    n_diff += int(d.sum())
                    if d.any():
                        fin = d & np.isfinite(t_ref) & np.isfinite(t_var)
                        if fin.any():
                            max_abs = max(max_abs, float(np.abs(t_ref[fin] - t_var[fin]).max()))
                    f_ref = float(P.floored_fpr(pool[ev] >= t_ref).mean() / target)
                    f_var = float(P.floored_fpr(pool[ev] >= t_var).mean() / target)
                    if f_ref != f_var:
                        n_fdiff += 1
                        max_fdiff = max(max_fdiff, abs(f_ref - f_var))
                row[vname] = {"thresholds_compared": len(parts) * pool.shape[1], "thresholds_differing": n_diff,
                              "max_abs_threshold_difference": max_abs, "floors_compared": len(parts),
                              "floors_differing": n_fdiff, "max_abs_floor_difference": max_fdiff}
                t = totals[vname]
                t["thresholds"] += len(parts) * pool.shape[1]
                t["differing"] += n_diff
                t["floors"] += len(parts)
                t["floors_differing"] += n_fdiff
            rows.append(row)
            print(f"  {stem:<34} {pset:<8} " + "  ".join(
                f"{v}: {row[v]['thresholds_differing']}/{row[v]['thresholds_compared']} thr, "
                f"{row[v]['floors_differing']}/{row[v]['floors_compared']} floors" for v in VARIANTS), flush=True)

    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"]:
        raise SystemExit("REFUSING to write: source changed during the run")
    digest = write_results(out_path, {
        **provenance_block("thr_variants_check"), "code": code0,
        "config": {"n_draws": args.n_draws, "target_fpr": target, "axis_draw_seeds": P.AXIS_DRAW_SEED,
                   "selection_file": "results/regularisation_sweep.json", "selection_sha256": sha256_file(sel_path),
                   "variant_sources_at_5e55307^": {"A_14_files": "src/depth_curve.py blob dadeb98b87310bb448d8981d41f3eb711e28e54f",
                                                   "B_pilot_e6": "src/pilot_e6.py blob 82419b077e03e4ff3c9a7d2ca4fbd47ad40649d4",
                                                   "C_repair_nonparametric": "src/repair_nonparametric.py blob 4bdc5df48ccf37c8b19498b953ac72e90fee095b"},
                   "reference": "paired.thr_at_fpr (vectorised)"},
        "population": {"declared_in": "src/populations.py", "group": "UPDATE_PAIRS_REPAIR"},
        "input_cache_sha256": input_sha, "totals": totals, "rows": rows}, exclusive=True)
    print(f"\nTOTALS: {json.dumps(totals)}\nWritten: {out_path}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
