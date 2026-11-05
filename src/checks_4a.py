"""Checks 2 and 5 on 4a (author, 2026-09-25), on the 4a cell: A100 Mistral base, the
LoRA arms' post-update eval caches, the selected probe (D-117: C=0.001, layer 12).

CHECK 2 - random-direction null for |cos(w, d)|. For each seed's ACTUAL selected-C w,
10,000 directions u ~ N(0, I) in the standardised feature space; |cos(w, u)| gives the
null. Reported: null mean and central 95% band per seed and pooled, each arm/dose's
observed |cos(w, d)| (d on eval negatives, as 4a) and its percentile in that seed's null,
and ||d||. The observed values are recomputed and must equal results/intent_selected.json.

CHECK 5 - F-059 diagnostic. The C=1 probe refitted with tolerance tightened 100x
(sklearn default 1e-4 -> 1e-6), same max_iter, in file order and reversed file order:
iterations, convergence, gradient norm, cos(w_file, w_reversed) per seed, and each run's
paired ratio under both orders and their difference; alongside, cos between the default-
tolerance and tight-tolerance fits. The library-default arm itself is unchanged (default
tolerance, named row order); this is a diagnostic of WHY the orders differ.
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
CODE_FILES = ["src/checks_4a.py", "src/paired.py", "src/populations.py", "src/provenance.py",
              "src/superseded_provenance.py"]
DOSES = (25, 100, 400)
TIGHT_TOL = 1e-6   # sklearn default 1e-4, tightened 100x


def load(repo: str, rev: str, axis: str, part: str):
    p = resolve_cache(f"{repo.replace('/', '__')}@{rev[:12]}__{axis}__{part}__mean.npz", CACHE)
    if not p.exists():
        raise FileNotFoundError(f"missing cache {p.name}")
    return np.load(p, allow_pickle=False), p


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="4a checks 2 (random-direction null) and 5 (tolerance)")
    ap.add_argument("--intent", default=str(RESULTS / "intent_selected.json"))
    ap.add_argument("--n-dirs", type=int, default=10000)
    ap.add_argument("--dir-seed", type=int, default=20260925)
    ap.add_argument("--out", default=str(RESULTS / "checks_4a.json"))
    args = ap.parse_args()
    out_path = Path(args.out).resolve()
    if out_path.exists():
        raise SystemExit(f"REFUSING: {out_path} exists")
    code0 = P.code_block(CODE_FILES)
    intent = json.load(open(args.intent, encoding="utf-8"))
    intent_sha = sha256_file(Path(args.intent))
    cfg = intent["config"]
    C_sel, L, seeds = cfg["selected_C"], cfg["selected_layer"], cfg["seeds"]
    pre_repo, pre_rev, axis = cfg["pre_repo"], cfg["pre_rev"], cfg["axis"]

    runs = [(arm, stem, dose) for arm, stems in POP.LORA_ARMS.items() for stem, dose in zip(stems, DOSES)]
    pre, input_sha = {}, {}
    for part in ("probe_train", "threshold_calib", "eval"):
        pre[part], p = load(pre_repo, pre_rev, axis, part)
        input_sha[f"pre/{part}"] = sha256_file(p)
    post = {}
    for _, stem, _ in runs:
        post[stem], p = load(cfg[f"{stem}:post_repo"], cfg[f"{stem}:post_rev"], axis, "eval")
        input_sha[f"{stem}:post/eval"] = sha256_file(p)
    diff = [k for k, h in input_sha.items() if intent["input_cache_sha256"].get(k) != h]
    if diff:
        raise SystemExit(f"REFUSING: caches differ from those 4a read: {diff}")

    k = f"layer_{L}"
    y_tr = pre["probe_train"]["labels"].astype(bool)
    y_cal = pre["threshold_calib"]["labels"].astype(bool)
    neg = ~pre["eval"]["labels"].astype(bool)
    X_tr, X_cal, X_ev = pre["probe_train"][k], pre["threshold_calib"][k], pre["eval"][k]

    # ---- CHECK 2 --------------------------------------------------------------------
    probes = []
    for seed in seeds:
        _, fit = P.d117_split(len(y_tr), seed)
        probes.append(P.fit_probe(X_tr[fit], y_tr[fit], seed, C_sel, "d117_permutation"))
    rng = np.random.default_rng(args.dir_seed)
    U = rng.standard_normal((args.n_dirs, X_tr.shape[1]))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    nulls = [np.abs(U @ (pr.w / np.linalg.norm(pr.w))) for pr in probes]      # per seed (n_dirs,)
    pooled = np.concatenate(nulls)
    null_rows = [{"seed": s, "mean": float(n.mean()), "q2_5": float(np.percentile(n, 2.5)),
                  "q97_5": float(np.percentile(n, 97.5))} for s, n in zip(seeds, nulls)]
    stored = {(c["run"]): c for c in intent["cells"] if c["is_selected_layer"]}
    cos_rows = []
    for arm, stem, dose in runs:
        dx = post[stem][k][neg].mean(0) - X_ev[neg].mean(0)
        obs, pct, dn = [], [], []
        for pr, n in zip(probes, nulls):
            d = dx / pr.sd
            c = abs(float(pr.w @ d / (np.linalg.norm(pr.w) * np.linalg.norm(d))))
            obs.append(c)
            pct.append(float(100.0 * (np.sum(n < c) + 0.5 * np.sum(n == c)) / n.size))
            dn.append(float(np.linalg.norm(d)))
        want = stored[stem]["geometry"]["neg"]["abs_cos"]["per_seed"]
        if not np.allclose(obs, want, rtol=1e-6, atol=1e-9):
            raise SystemExit(f"REFUSING: recomputed |cos| for {stem} {obs} != 4a's {want}")
        cos_rows.append({"arm": arm, "run": stem, "dose": dose,
                         "abs_cos_per_seed": obs, "abs_cos_mean": float(np.mean(obs)),
                         "percentile_in_own_seed_null_per_seed": pct,
                         "percentile_range": [float(min(pct)), float(max(pct))],
                         "d_norm_per_seed": dn, "d_norm_mean": float(np.mean(dn)),
                         "d_norm_range": [float(min(dn)), float(max(dn))]})
        print(f"  {stem:<24} |cos| {np.mean(obs):.4f}  percentile {min(pct):5.1f}-{max(pct):5.1f}  ||d|| {np.mean(dn):.2f}")
    print(f"random-direction null (pooled over seeds, {args.n_dirs} x {len(seeds)}): mean {pooled.mean():.5f}, "
          f"95% band [{np.percentile(pooled, 2.5):.5f}, {np.percentile(pooled, 97.5):.5f}]; "
          f"sqrt(2/(pi d)) = {np.sqrt(2 / (np.pi * X_tr.shape[1])):.5f}", flush=True)

    # ---- CHECK 5 --------------------------------------------------------------------
    T_of = lambda prs: np.array([P.thr_at_fpr(q.score(X_cal)[~y_cal]) for q in prs])  # noqa: E731
    fits: dict[str, list[P.Probe]] = {"default_file": [], "tight_file": [], "tight_reversed": []}
    for seed in seeds:
        _, fit = P.d117_split(len(y_tr), seed)
        fo = np.sort(fit)
        fits["default_file"].append(P.fit_probe(X_tr[fo], y_tr[fo], seed, 1.0, "file_order"))
        fits["tight_file"].append(P.fit_probe(X_tr[fo], y_tr[fo], seed, 1.0, "file_order", tol=TIGHT_TOL))
        fits["tight_reversed"].append(P.fit_probe(X_tr[fo[::-1]], y_tr[fo[::-1]], seed, 1.0,
                                                  "reversed_file_order", tol=TIGHT_TOL))
    cosw = lambda a, b: [float(x.w @ y.w / (np.linalg.norm(x.w) * np.linalg.norm(y.w))) for x, y in zip(fits[a], fits[b])]  # noqa: E731
    thr = {name: T_of(prs) for name, prs in fits.items()}
    pre_s = {name: np.column_stack([q.score(X_ev[neg]) for q in prs]) for name, prs in fits.items()}
    ratio_rows = []
    for arm, stem, dose in runs:
        r = {}
        for name, prs in fits.items():
            post_s = np.column_stack([q.score(post[stem][k][neg]) for q in prs])
            r[name] = P.ratio_from_flags(pre_s[name] >= thr[name], post_s >= thr[name])
        ratio_rows.append({"arm": arm, "run": stem, "dose": dose, "ratio": r,
                           "tight_file_minus_tight_reversed": r["tight_file"] - r["tight_reversed"],
                           "default_file_minus_tight_file": r["default_file"] - r["tight_file"]})
        print(f"  {stem:<24} C=1 ratio default {r['default_file']:.3f}  tight file {r['tight_file']:.3f}  "
              f"tight reversed {r['tight_reversed']:.3f}  (orders differ by {r['tight_file'] - r['tight_reversed']:+.4f})")
    diag = {name: [q.diag for q in prs] for name, prs in fits.items()}
    tol_summary = {"cos_tight_file_vs_tight_reversed_per_seed": cosw("tight_file", "tight_reversed"),
                   "cos_default_file_vs_tight_file_per_seed": cosw("default_file", "tight_file"),
                   "max_abs_ratio_diff_between_orders_tight": float(max(abs(r["tight_file_minus_tight_reversed"]) for r in ratio_rows)),
                   "max_abs_ratio_diff_default_vs_tight": float(max(abs(r["default_file_minus_tight_file"]) for r in ratio_rows))}
    for name, ds in diag.items():
        print(f"  {name:<15} iters {[x['n_iter'] for x in ds]} converged {[x['converged'] for x in ds]} "
              f"grad_norm {[round(x['grad_norm'], 4) for x in ds]}")
    print(f"  cos(tight file, tight reversed) {[round(c, 7) for c in tol_summary['cos_tight_file_vs_tight_reversed_per_seed']]}")
    print(f"  cos(default file, tight file)   {[round(c, 7) for c in tol_summary['cos_default_file_vs_tight_file_per_seed']]}")

    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"]:
        raise SystemExit("REFUSING to write: source changed during the run")
    payload = {**provenance_block("checks_4a"), "code": code0,
               "config": {"intent_file": "results/intent_selected.json", "intent_sha256": intent_sha,
                          "selected_C": C_sel, "layer": L, "seeds": seeds, "n_dirs": args.n_dirs,
                          "dir_seed": args.dir_seed, "tight_tol": TIGHT_TOL, "max_iter": P.MAX_ITER,
                          "pre_repo": pre_repo, "pre_rev": pre_rev, "axis": axis,
                          **{f"{s}:post_repo": cfg[f"{s}:post_repo"] for _, s, _ in runs},
                          **{f"{s}:post_rev": cfg[f"{s}:post_rev"] for _, s, _ in runs}},
               "population": {"declared_in": "src/populations.py", "group": "LORA_ARMS"},
               "input_cache_sha256": input_sha,
               "check2_random_direction_null": {"per_seed": null_rows, "pooled_mean": float(pooled.mean()),
                                                "pooled_q2_5": float(np.percentile(pooled, 2.5)),
                                                "pooled_q97_5": float(np.percentile(pooled, 97.5)),
                                                "analytic_mean_sqrt_2_over_pi_d": float(np.sqrt(2 / (np.pi * X_tr.shape[1]))),
                                                "arms": cos_rows},
               "check5_tolerance": {"summary": tol_summary, "fit_diagnostics": diag, "runs": ratio_rows}}
    digest = write_results(out_path, payload, exclusive=True)
    print(f"Written: {out_path}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
