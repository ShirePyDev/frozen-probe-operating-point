"""4c pooled across pairs (D-152 4c; D-169 item 12; D-241 (6)), from the ten results/healthcheck_d156_<pair>.json
and their _draws.npz only (no model is read). Replaces AUC_AUDITS.

Per probe set (selected C primary, C=1 the sensitivity) and level (item level primary, the D-164 cluster level
beside), for every detector (Gaussian health ratio, direct reference ratio, the AUC rule) against every class
(realised paired ratio >= 2 or <= 0.5 primary; >= 5 or <= 0.2; >= 2 only):
  PER DRAW   the confusion counts summed over the pairs' cells; sensitivity TP/(TP+FN) and specificity TN/(TN+FP):
             D-156 mean and 95% band with the frozen split (a draw with no positive has no sensitivity; counted).
  POOLED     sensitivity and specificity from the per-pair D-156 mean counts, with a pair-cluster bootstrap 95%
             interval (B = 5000, seed 520260925, pairs resampled; D-177 (b); rows of at least 3 pairs).
             The estimator is pooled_counts (with pair_cluster_index); src/phase3_a1_pooled.py applies it unchanged
             (D-346 (4)).
  ROC AUC    per draw, across the row's cells, of |log Gaussian ratio|, |log direct ratio| and |z_mean| for the class:
             D-156 mean and band (threshold-free detection power).
Rows: the nine released pairs; the pairs whose records carry no D-170 flag. The GPU control (no update) is its own
row: its false-alarm rate per detector and the share of draws in which any of its cells falls in a class. Llama-3 ->
3.1 is listed at every layer for both C. Per-pair bands cover partitions only, pooled intervals pairs only (D-166).
Every input must be named by logs/healthcheck_d156.sha256 (JSON and npz). Writes
results/healthcheck_d156_summary.json (never overwritten); a test writes outside results/.
--fpr5 (4e, D-248) pools the ten 5% records; --lora (D-278) pools the nine LoRA runs (rows all nine, general 3,
safety 6; the D-156 replacement of the AUC_AUDITS LoRA part) from logs/healthcheck_d156_lora.sha256.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
from provenance import ensure_utf8_stdout, provenance_block, sha256_bytes, sha256_file, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FINAL_OUT = RESULTS / "healthcheck_d156_summary.json"
MANIFEST = ROOT / "logs" / "healthcheck_d156.sha256"
CODE_FILES = ["src/healthcheck_summary_d156.py", "src/metrics.py", "src/paired.py", "src/populations.py",
              "src/provenance.py", "src/superseded_provenance.py"]
LEVELS = ("item_level_D156", "cluster_level_D164")
PROBE_SETS = ("selected", "C=1")
CLASSES = {"two_sided_2x": (2.0, 0.5), "two_sided_5x": (5.0, 0.2), "high_side_2x": (2.0, None)}
DETECTORS = ("gaussian", "direct", "auc_rule")
CONTROL = "hw_v100_a100"
LISTED = "llama3_to_31"                                 # D-152 4c: shown at every layer for both C
B, SEED, MIN_PAIRS = 5000, 520260925, 3
COUNTS = ("tp", "fn", "fp", "tn")


def core_of(stem: str) -> str:
    return stem[len("repair_"):]


def crosses(x: np.ndarray, cut: tuple[float, float | None]) -> np.ndarray:
    hi, lo = cut
    return (x >= hi) | (x <= lo) if lo is not None else (x >= hi)


def rate(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def load(core: str, man: dict[str, str], inputs: dict[str, str], sfx: str = "", man_name: str = MANIFEST.name) -> dict[str, Any]:
    rel = f"results/healthcheck_d156_{core}{sfx}.json"
    b = (ROOT / rel).read_bytes()
    if man.get(rel) != sha256_bytes(b):
        raise SystemExit(f"REFUSING: {rel} is not the file {man_name} names")
    if sfx and not json.loads(b.decode("utf-8")).get("fpr5_self_anchor"):
        raise SystemExit(f"REFUSING: {rel} carries no 1% self-anchor (D-248)")
    d = json.loads(b.decode("utf-8"))
    if d.get("registered_parameters") is not True:
        raise SystemExit(f"REFUSING: {rel} is not a registered run")
    nrel = d["draws_npz"]["file"]
    nb = (ROOT / nrel).read_bytes()
    if man.get(nrel) != sha256_bytes(nb) or sha256_bytes(nb) != d["draws_npz"]["sha256"]:
        raise SystemExit(f"REFUSING: {nrel} is not the npz {rel} and {man_name} name")
    inputs[rel], inputs[nrel] = sha256_bytes(b), sha256_bytes(nb)
    z = np.load(ROOT / nrel, allow_pickle=False)
    return {"doc": d, "z": {k: z[k] for k in z.files}, "flag": d["input_verification"]["flag"]}


def pair_cluster_index(k: int) -> np.ndarray | None:
    """D-177 (b), D-241 (6): the pairs each replicate of the pair-cluster bootstrap resamples (B x k indices, seed
    SEED), or None for a row of fewer than MIN_PAIRS pairs (point only)."""
    return np.random.default_rng(SEED).integers(0, k, size=(B, k)) if k >= MIN_PAIRS else None


def pooled_counts(draw_counts: dict[str, list[np.ndarray]], idx: np.ndarray | None) -> dict[str, Any]:
    """The D-255 estimator for one detector and class (D-241 (6)). draw_counts[q], q in COUNTS: one per-draw count
    array per pair (row 0 the frozen split, then the draws). PER DRAW, the counts summed over the pairs give sensitivity
    and specificity (D-156 mean and band); POOLED, the per-pair D-156 mean counts give them, with the pair-cluster
    bootstrap interval over idx (pair_cluster_index). Shared with src/phase3_a1_pooled.py (D-346 (4))."""
    cnt = {q: np.sum(draw_counts[q], axis=0) for q in COUNTS}
    sens, spec = rate(cnt["tp"], cnt["tp"] + cnt["fn"]), rate(cnt["tn"], cnt["tn"] + cnt["fp"])
    mean_c = {q: np.array([a[1:].mean() for a in draw_counts[q]]) for q in COUNTS}
    pooled = {"sensitivity": float(mean_c["tp"].sum() / (mean_c["tp"].sum() + mean_c["fn"].sum())) if (mean_c["tp"].sum() + mean_c["fn"].sum()) > 0 else None,
              "specificity": float(mean_c["tn"].sum() / (mean_c["tn"].sum() + mean_c["fp"].sum())) if (mean_c["tn"].sum() + mean_c["fp"].sum()) > 0 else None}
    if idx is not None:
        s_tp, s_fn, s_fp, s_tn = (mean_c[q][idx].sum(axis=1) for q in COUNTS)
        pooled["sensitivity_ci"] = P.ci(rate(s_tp, s_tp + s_fn))
        pooled["specificity_ci"] = P.ci(rate(s_tn, s_tn + s_fp))
    return {"per_draw_sensitivity": P.band_summary(sens[1:], sens[0]),
            "per_draw_specificity": P.band_summary(spec[1:], spec[0]),
            "mean_counts_over_draws": {q: float(v.sum()) for q, v in mean_c.items()},
            "pooled_from_mean_counts": pooled}


def scoring(prs: dict[str, dict[str, Any]], names: list[str], ps: str, lvl: str) -> dict[str, Any]:
    k = len(names)
    out: dict[str, Any] = {"pairs": names, "n_pairs": k}
    if k == 0:
        return {**out, "note": "no pair in this row"}
    idx = pair_cluster_index(k)
    out["interval"] = f"pair-cluster bootstrap B={B}, seed {SEED}" if idx is not None else f"point only (fewer than {MIN_PAIRS} pairs)"
    for c in CLASSES:
        for dname in DETECTORS:
            out[f"{dname}|{c}"] = pooled_counts({q: [prs[n]["z"][f"counts|{ps}|{lvl}|{dname}|{c}|{q}"] for n in names]
                                                 for q in COUNTS}, idx)
    # threshold-free detection power: ROC AUC per draw across the row's cells
    keys = [(n, kk.rsplit("|", 1)[0]) for n in names for kk in prs[n]["z"] if kk.startswith("L") and
            kk.endswith("|realised_paired_ratio") and f"|{ps}|{lvl}|" in kk]
    rr = np.vstack([prs[n]["z"][f"{base}|realised_paired_ratio"] for n, base in keys])
    stats = {"abs_log_gaussian_ratio": np.abs(np.vstack([prs[n]["z"][f"{base}|log_gaussian_ratio"] for n, base in keys])),
             "abs_log_direct_ratio": np.abs(np.log(np.vstack([prs[n]["z"][f"{base}|direct_ratio"] for n, base in keys]))),
             "abs_z_mean": np.abs(np.vstack([prs[n]["z"][f"{base}|z_mean"] for n, base in keys]))}
    out["n_cells"] = len(keys)
    for c, cut in CLASSES.items():
        y = crosses(rr, cut)
        for sname, sv in stats.items():
            a = np.array([M.auroc(sv[y[:, d], d], sv[~y[:, d], d]).value if 0 < y[:, d].sum() < y.shape[0] else np.nan
                          for d in range(rr.shape[1])])
            out[f"roc_auc|{sname}|{c}"] = P.band_summary(a[1:], a[0])
    return out


def lora_summary(args: argparse.Namespace, man: dict[str, str], manifest: Path, mb: bytes, inputs: dict[str, str],
                 out: Path, code0: dict[str, Any]) -> int:
    """D-278: the nine LoRA runs pooled as the released pairs are (the same scoring), rows all nine, general, safety.
    The nine share one pre model, so a run-cluster interval covers variation between runs only."""
    groups = {"general": list(POP.LORA_ARMS["general (alpaca)"]),
              "safety": [*POP.LORA_ARMS["safety (hh-harmless)"], *POP.LORA_ARMS["safety (pku-saferlhf)"]]}
    runs = [r for g in groups.values() for r in g]
    if len(runs) != 9 or len(set(runs)) != 9:
        raise SystemExit(f"REFUSING: the LoRA runs are not nine distinct runs: {runs}")
    prs = {r: load(r, man, inputs, "", manifest.name) for r in runs}
    bad = [r for r in runs if prs[r]["doc"]["config"]["control"]
           or prs[r]["doc"]["config"]["selection_file"] != "results/selection_sweep_lora_a100_d165.json"
           or prs[r]["doc"]["config"]["target_fpr"] != P.TARGET_FPR]
    if bad:
        raise SystemExit(f"REFUSING: not D-278's LoRA health-check records (control, LoRA A100 selection, 1%): {bad}")
    rows: dict[str, Any] = {}
    for ps in PROBE_SETS:
        for lvl in LEVELS:
            rows[f"{ps}|{lvl}"] = {"lora_all": scoring(prs, runs, ps, lvl),
                                   **{f"lora_{g}": scoring(prs, rr, ps, lvl) for g, rr in groups.items()}}
    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"] or any(sha256_file(ROOT / f) != h for f, h in inputs.items()):
        raise SystemExit("REFUSING to write: code or an input changed during the run")
    prim = rows["selected|item_level_D156"]
    for row in ("lora_all", "lora_general", "lora_safety"):
        s = prim[row]["gaussian|two_sided_2x"]
        print(f"{row:12s} gaussian two-sided 2x: pooled {s['pooled_from_mean_counts']}", flush=True)
    payload = {**provenance_block("healthcheck_d156_summary"), "code": code0, "preregistration": "D-278 (D-271 (3) item 3)",
               "primary": {"probe_set": "selected", "level": "item_level_D156", "class": "two_sided_2x", "detector": "gaussian"},
               "band_note": "per-draw bands cover partitions only; pooled intervals cover runs only (D-166). The nine runs "
                            "share one pre model (the A100 Mistral-7B-v0.1 base), so they are not independent update pairs",
               "groups": groups, "input_files_sha256": inputs, "rows": rows,
               "replaces": ["auc_criterion_audit_lora (its LoRA part; its GPU-control part is in healthcheck_d156_summary)"]}
    digest = write_results(out, payload, exclusive=True)
    print(f"Written: {out}\nsha256:  {digest}")
    return 0


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="4c pooled across pairs (D-241 (6))")
    ap.add_argument("--fpr5", action="store_true",
                    help="4e (D-248): pool the ten 5%% per-pair records (healthcheck_d156_<pair>_fpr5.json) into "
                         "healthcheck_d156_summary_fpr5.json, from logs/healthcheck_d156_fpr5.sha256")
    ap.add_argument("--lora", action="store_true",
                    help="D-278: pool the nine LoRA runs (healthcheck_d156_<run>.json, 1%%) into "
                         "healthcheck_d156_summary_lora.json, from logs/healthcheck_d156_lora.sha256; rows all nine, "
                         "general (3), safety (6); the D-156 replacement of the AUC_AUDITS LoRA part")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.lora and args.fpr5:
        raise SystemExit("REFUSING: 4e has no 5% arm for the LoRA runs (D-278)")
    sfx = "_fpr5" if args.fpr5 else "_lora" if args.lora else ""
    final_out = RESULTS / f"healthcheck_d156_summary{sfx}.json"
    manifest = ROOT / "logs" / f"healthcheck_d156{sfx}.sha256"
    out = Path(args.out).resolve() if args.out else final_out.resolve()
    final = out == final_out.resolve()
    if out.exists():
        raise SystemExit(f"REFUSING: {out} exists; never overwritten")
    if not final and out.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test output must not be under results/")
    code0 = P.code_block(CODE_FILES)
    if not manifest.is_file():
        raise SystemExit(f"REFUSING: no manifest {manifest} (written after the {'nine LoRA' if args.lora else 'ten per-pair'} "
                         f"runs are verified)")
    mb = manifest.read_bytes()
    man = {}
    for line in mb.decode("utf-8").splitlines():
        if line.strip():
            h, f = line.split(maxsplit=1)
            man[f.strip().lstrip("*")] = h
    inputs = {str(manifest.relative_to(ROOT)).replace("\\", "/"): sha256_bytes(mb)}
    if args.lora:
        return lora_summary(args, man, manifest, mb, inputs, out, code0)
    released = [core_of(s) for s in POP.UPDATE_PAIRS_REPAIR]
    prs = {c: load(c, man, inputs, sfx, manifest.name) for c in [*released, CONTROL]}
    if not prs[CONTROL]["doc"]["config"]["control"] or any(prs[c]["doc"]["config"]["control"] for c in released):
        raise SystemExit("REFUSING: the control flag of the records is not as expected")
    unflagged = [c for c in released if prs[c]["flag"] is None]
    rows: dict[str, Any] = {}
    for ps in PROBE_SETS:
        for lvl in LEVELS:
            key = f"{ps}|{lvl}"
            rows[key] = {"released_pairs": scoring(prs, released, ps, lvl),
                         "unflagged_pairs": scoring(prs, unflagged, ps, lvl)}
            cz = prs[CONTROL]["z"]
            fa = {}
            for c in CLASSES:
                for dname in DETECTORS:
                    fp, tn = cz[f"counts|{ps}|{lvl}|{dname}|{c}|fp"], cz[f"counts|{ps}|{lvl}|{dname}|{c}|tn"]
                    far = rate(fp, fp + tn)
                    fa[f"{dname}|{c}"] = {"false_alarm_rate": P.band_summary(far[1:], far[0])}
                pos = cz[f"counts|{ps}|{lvl}|gaussian|{c}|tp"] + cz[f"counts|{ps}|{lvl}|gaussian|{c}|fn"]
                fa[f"share_of_draws_with_a_control_cell_in_class|{c}"] = float((pos[1:] > 0).mean())
            rows[key]["gpu_control"] = fa
    listed = [{"layer": c["layer"], "probe_set": c["probe_set"], "auc_rule_flag": c["auc_rule"]["flag"],
               **{q: c[LEVELS[0]][q]["mean"] for q in ("gaussian_ratio", "direct_ratio", "realised_paired_ratio", "z_mean")}}
              for c in prs[LISTED]["doc"]["cells"]]
    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"] or any(sha256_file(ROOT / f) != h for f, h in inputs.items()):
        raise SystemExit("REFUSING to write: code or an input changed during the run")
    prim = rows["selected|item_level_D156"]["released_pairs"]
    for dname in DETECTORS:
        s = prim[f"{dname}|two_sided_2x"]
        print(f"{dname:9s} two-sided 2x: per-draw sensitivity {s['per_draw_sensitivity']['mean']:.3f} "
              f"specificity {s['per_draw_specificity']['mean']:.3f}; pooled {s['pooled_from_mean_counts']}", flush=True)
    payload = {**provenance_block("healthcheck_d156_summary"), "code": code0,
               "preregistration": "D-152 4c, D-241" + ("; 4e: D-152, D-248 (the 5% arm)" if args.fpr5 else ""),
               "primary": {"probe_set": "selected", "level": "item_level_D156", "class": "two_sided_2x", "detector": "gaussian"},
               "band_note": "per-draw bands cover partitions only; pooled intervals cover pairs only (D-166)",
               "input_files_sha256": inputs, "rows": rows, "listed_llama3_to_31_every_layer": listed,
               "replaces": [] if args.fpr5 else list(POP.AUC_AUDITS.values())}
    digest = write_results(out, payload, exclusive=True)
    print(f"Written: {out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
