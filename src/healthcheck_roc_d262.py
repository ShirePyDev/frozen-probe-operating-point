"""4c: ROC AUC over cells with pair-cluster intervals (the author's request of 2026-09-28; pre-registered in D-262).

From the ten per-pair records named by logs/healthcheck_d156.sha256 only (the per-cell D-156 means; no model, no npz):
  UNITS   every layer cell of the nine released pairs (row 'all') and of the five unflagged pairs ('unflagged');
          selected C and the item level primary, C=1 and the D-164 cluster level beside.
  CLASS   per cell, from its D-156 mean realised paired ratio: >= 2 or <= 0.5 (two-sided 2x); 5x and the high side
          (>= 2) beside.
  SCORES  Gaussian |mean log_gaussian_ratio|; direct |log(mean direct_ratio)|; |mean z_mean| beside.
  AUC     metrics.auroc over the row's cells (ties 1/2); interval: pair-cluster bootstrap, B = 5000, seed 520260925,
          pairs resampled with replacement and all their cells pooled; a replicate lacking a class is undefined and
          counted. The GPU-change control's cells get the same class and flags as a check.
Beside, read from results/healthcheck_d156_summary.json and not recomputed (D-255): sensitivity and specificity at the
same cut (false-alarm rate = 1 - specificity) with their pair-cluster intervals, the control's false-alarm rates, and the
Llama-3 -> 3.1 listing. Writes results/healthcheck_d156_roc_pairs.json (never overwritten); a test writes elsewhere.
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
FINAL_OUT = RESULTS / "healthcheck_d156_roc_pairs.json"
MANIFEST = ROOT / "logs" / "healthcheck_d156.sha256"
SUMMARY = "results/healthcheck_d156_summary.json"
SUMMARY_SHA256 = "8ef049f304b97ea9cf30a151ac90956d6097cb492c51b1d7f03b27b43df6047e"      # D-255
CODE_FILES = ["src/healthcheck_roc_d262.py", "src/metrics.py", "src/paired.py", "src/populations.py",
              "src/provenance.py", "src/superseded_provenance.py"]
LEVELS = ("item_level_D156", "cluster_level_D164")
PROBE_SETS = ("selected", "C=1")
PRIMARY = ("selected", "item_level_D156")
CLASSES = {"two_sided_2x": (2.0, 0.5), "two_sided_5x": (5.0, 0.2), "high_side_2x": (2.0, None)}
SCORES = ("gaussian", "direct", "z_mean")
CONTROL = "hw_v100_a100"
B, SEED = 5000, 520260925                             # D-177 (b)


def in_class(r: float, cut: tuple[float, float | None]) -> bool:
    hi, lo = cut
    return r >= hi or (lo is not None and r <= lo)


def cells_of(doc: dict[str, Any], ps: str, lvl: str) -> list[dict[str, float]]:
    out = []
    for c in doc["cells"]:
        if c["probe_set"] != ps:
            continue
        s = c[lvl]
        out.append({"layer": c["layer"], "realised": s["realised_paired_ratio"]["mean"],
                    "gaussian": abs(s["log_gaussian_ratio"]["mean"]), "direct": abs(float(np.log(s["direct_ratio"]["mean"]))),
                    "z_mean": abs(s["z_mean"]["mean"]), "log_gaussian_signed": s["log_gaussian_ratio"]["mean"]})
    return out


def auc(cells: list[dict[str, float]], cls: str, score: str) -> float | None:
    y = np.array([in_class(c["realised"], CLASSES[cls]) for c in cells])
    s = np.array([c[score] for c in cells])
    return M.auroc(s[y], s[~y]).value


def row(pairs: dict[str, list[dict[str, float]]], stems: list[str]) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(stems), size=(B, len(stems)))
    pooled = [c for s in stems for c in pairs[s]]
    out: dict[str, Any] = {"pairs": stems, "n_pairs": len(stems), "n_cells": len(pooled)}
    for cls in CLASSES:
        y = [in_class(c["realised"], CLASSES[cls]) for c in pooled]
        blk: dict[str, Any] = {"n_cells_in_class": int(sum(y)), "n_cells_not_in_class": int(len(y) - sum(y))}
        for sc in SCORES:
            reps = []
            for b in range(B):
                v = auc([c for j in idx[b] for c in pairs[stems[j]]], cls, sc)
                reps.append(np.nan if v is None else v)
            reps = np.array(reps)
            ok = reps[np.isfinite(reps)]
            blk[f"roc_auc|{sc}"] = {"point": auc(pooled, cls, sc),
                                    "ci95": [float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))] if ok.size else None,
                                    "n_replicates": B, "n_undefined": int(B - ok.size)}
        out[cls] = blk
    return out


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="D-262: 4c ROC AUC over cells with pair-cluster intervals")
    ap.add_argument("--out", default=str(FINAL_OUT))
    args = ap.parse_args()
    out = Path(args.out).resolve()
    final = out == FINAL_OUT.resolve()
    if out.exists():
        raise SystemExit(f"REFUSING: {out} exists; never overwritten")
    if not final and out.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test output must not be under results/")
    code0 = P.code_block(CODE_FILES)
    mb = MANIFEST.read_bytes()
    man: dict[str, str] = {}
    for line in mb.decode("utf-8").splitlines():
        if line.strip():
            h, f = line.split(maxsplit=1)
            f = f.strip().lstrip("*")
            if f in man:
                raise SystemExit(f"REFUSING: {MANIFEST.name} names {f} twice")
            man[f] = h
    inputs = {str(MANIFEST.relative_to(ROOT)).replace("\\", "/"): sha256_bytes(mb)}
    docs: dict[str, dict[str, Any]] = {}
    for core in [*(s[len("repair_"):] for s in POP.UPDATE_PAIRS_REPAIR), CONTROL]:
        rel = f"results/healthcheck_d156_{core}.json"
        b = (ROOT / rel).read_bytes()
        if man.get(rel) != sha256_bytes(b):
            raise SystemExit(f"REFUSING: {rel} is not the file {MANIFEST.name} names")
        inputs[rel] = sha256_bytes(b)
        docs[core] = json.loads(b.decode("utf-8"))
    sb = (ROOT / SUMMARY).read_bytes()
    if sha256_bytes(sb) != SUMMARY_SHA256:
        raise SystemExit(f"REFUSING: {SUMMARY} is not the D-255 summary")
    inputs[SUMMARY] = SUMMARY_SHA256
    summ = json.loads(sb.decode("utf-8"))
    released = [s[len("repair_"):] for s in POP.UPDATE_PAIRS_REPAIR]
    unflagged = [c for c in released if docs[c]["input_verification"]["flag"] is None]
    if unflagged != [s for s in summ["rows"]["selected|item_level_D156"]["unflagged_pairs"]["pairs"]]:
        raise SystemExit("REFUSING: the unflagged pairs differ from the D-255 summary's")
    res: dict[str, Any] = {}
    for ps in PROBE_SETS:
        for lvl in LEVELS:
            pairs = {c: cells_of(docs[c], ps, lvl) for c in released}
            ctrl = cells_of(docs[CONTROL], ps, lvl)
            ctrl_chk = {cls: {"n_cells": len(ctrl),
                              "n_cells_in_class": int(sum(in_class(c["realised"], CLASSES[cls]) for c in ctrl)),
                              "n_cells_gaussian_flag": int(sum(in_class(float(np.exp(c["log_gaussian_signed"])), CLASSES[cls])
                                                               for c in ctrl))}
                        for cls in CLASSES}
            res[f"{ps}|{lvl}"] = {"judged": (ps, lvl) == PRIMARY, "all": row(pairs, released),
                                  "unflagged": row(pairs, unflagged), "gpu_control_per_cell_check": ctrl_chk}
            p = res[f"{ps}|{lvl}"]
            print(f"{ps}|{lvl}: ROC AUC (2x) Gaussian {p['all']['two_sided_2x']['roc_auc|gaussian']['point']:.3f} "
                  f"{p['all']['two_sided_2x']['roc_auc|gaussian']['ci95']}, direct {p['all']['two_sided_2x']['roc_auc|direct']['point']:.3f} "
                  f"{p['all']['two_sided_2x']['roc_auc|direct']['ci95']}; unflagged Gaussian "
                  f"{p['unflagged']['two_sided_2x']['roc_auc|gaussian']['point']:.3f}", flush=True)
    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"] or any(sha256_file(ROOT / f) != h for f, h in inputs.items()):
        raise SystemExit("REFUSING to write: code or an input changed during the run")
    prim = summ["rows"]["selected|item_level_D156"]
    beside = {"source": SUMMARY, "note": "read, not recomputed (D-255): per-draw confusion counts pooled over pairs",
              "at_the_same_cut": {row_: {f"{det}|{cls}": prim[row_][f"{det}|{cls}"]["pooled_from_mean_counts"]
                                         for det in ("gaussian", "direct", "auc_rule") for cls in CLASSES}
                                  for row_ in ("released_pairs", "unflagged_pairs")},
              "gpu_control": prim["gpu_control"], "llama3_to_31_every_layer": summ["listed_llama3_to_31_every_layer"]}
    payload = {**provenance_block("healthcheck_d156_roc_pairs"), "code": code0, "preregistration": "D-262 (author's request); D-241, D-177 (b)",
               "primary": {"probe_set": PRIMARY[0], "level": PRIMARY[1], "class": "two_sided_2x"},
               "per_cell": "D-156 means: class from the mean realised paired ratio; scores |mean log_gaussian_ratio|, "
                           "|log(mean direct_ratio)|, |mean z_mean|",
               "interval": f"pair-cluster bootstrap B={B}, seed {SEED}, pairs resampled, their cells pooled",
               "input_files_sha256": inputs, "rows": res, "beside_from_d255": beside}
    digest = write_results(out, payload, exclusive=True)
    print(f"Written: {out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
