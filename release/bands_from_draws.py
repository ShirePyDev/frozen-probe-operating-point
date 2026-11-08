"""Recompute, from the stored per-draw values, every band the paper prints in its headline table (tab:headline: the
paired ratio r of each Phase 1 cell at L*, at the 1% and the 5% target) and in its LoRA table (tab:lora: the frozen
error of each run), and check each against the record the tables are written from.

    python release/bands_from_draws.py

Needs Python 3.10 or later and numpy; reads files only and writes nothing. Exits 1 on any FAIL.

What a band is (the paper's Section 4.1, our registered estimator): the 1000 re-partitions of the pooled calibration
and evaluation negatives give 1000 values of the statistic; the band is their 2.5th and 97.5th percentiles
(numpy.percentile, linear) and the printed value their mean. Index 0 of every stored array is the fixed split, kept
beside. The computation is src/paired.py's band_summary, imported unchanged.

Sources. Phase 1 r: results/healthcheck_d156_<pair>[_fpr5]_draws.npz, key 'L<L*>|selected|item_level_D156|
realised_paired_ratio'; the array's SHA-256 is checked against the 'draws_npz' block of
results/healthcheck_d156_<pair>[_fpr5].json, and the band against operating_point.paired_ratio of the selected cell
of results/depth_d156_<pair>[_fpr5].json. LoRA frozen error: the 1001 values that results/repair_d156_<run>.json
stores under pair_per_draw_mean_abs_log_over_layers (their SHA-256, over the little-endian float64 bytes, is in the
same block), against pair_summary_mean_abs_log_over_layers.

Not recomputable here, because no per-draw values of them were stored: the band of each LoRA run's worst FPR/target
(results/depth_d156_<run>.json holds its mean and band only) and the bands of the Phase 1 XSTest ratios
(results/overrefusal_d156_*.json hold their means and bands only). The script lists them.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from paired import band_summary  # noqa: E402
import pairs_table_d468 as PT  # noqa: E402
import populations as POP  # noqa: E402

KEY = "L{L}|selected|item_level_D156|realised_paired_ratio"


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)


def check(name: str, got: dict, want: dict, fails: list[str], d: int) -> None:
    ok = same(got["mean"], want["mean"]) and all(same(x, y) for x, y in zip(got["band95"], want["band95"]))
    lo, hi = got["band95"]
    print(f"{'PASS' if ok else 'FAIL'}  {name}: mean {got['mean']:.{d}f} band [{lo:.{d}f}, {hi:.{d}f}]")
    if not ok:
        fails.append(name)


def main() -> int:
    fails: list[str] = []
    rows, _ = PT.rows()
    head = [r for r in rows if r["headline"] and r["phase"] == 1]
    for r in head:
        for suffix, target in (("", "1%"), ("_fpr5", "5%")):
            depth = load(f"results/depth_d156_{r['record']}{suffix}.json")
            cell = [c for c in depth["cells"] if c.get("is_selected_layer") and c.get("probe_set") == "selected"
                    and c["layer"] == r["L"] and math.isclose(c["probe_C"], r["C"])]
            if len(cell) != 1:
                fails.append(f"{r['id']} {r['dataset']} {target}: no single selected cell")
                continue
            want = cell[0]["operating_point"]["paired_ratio"]
            pin = load(f"results/healthcheck_d156_{r['record']}{suffix}.json")["draws_npz"]
            npz = ROOT / pin["file"]
            if sha(npz) != pin["sha256"]:
                fails.append(f"{pin['file']}: SHA-256 differs from its record")
                print(f"FAIL  {pin['file']}: SHA-256 differs from its record")
                continue
            a = np.load(npz)[KEY.format(L=r["L"])]
            check(f"{r['id']} {r['dataset']} L{r['L']} r at {target}", band_summary(a[1:], a[0]), want, fails, 2)
    for runs in POP.LORA_ARMS.values():
        for run in runs:
            rep = load(f"results/repair_d156_{run}.json")
            blk = rep["pair_per_draw_mean_abs_log_over_layers"]["selected"]["item_level_D156"]["frozen"]
            v = np.asarray(blk["values"], dtype="<f8")
            if hashlib.sha256(v.tobytes()).hexdigest() != blk["sha256"]:
                fails.append(f"{run}: stored values differ from their SHA-256")
                print(f"FAIL  {run}: stored values differ from their SHA-256")
                continue
            want = rep["pair_summary_mean_abs_log_over_layers"]["selected"]["item_level_D156"]["frozen"]
            check(f"{run} frozen error", band_summary(v[1:], v[0]), want, fails, 3)
    print("not recomputable here (no per-draw values stored): the 9 LoRA runs' worst FPR/target bands; the 7 Phase 1 "
          "XSTest bands")
    print(f"{'FAIL' if fails else 'PASS'}: {len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
