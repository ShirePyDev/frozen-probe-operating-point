"""One 4b cell's realised against predicted paired damage over the D-156 draws (the author's question of D-300).

For one update pair and layer (selected C, item level) it copies, from the pair's twomoment record, the D-156 summaries
of the realised paired ratio and of the PRIMARY prediction (moment_full_in_sample, D-163), and gives:
  - the cell's D-156 offset, log(mean realised / mean predicted) over the common defined draws (twomoment_fit's
    d156_estimates; the per-cell quantity D-177's accuracy line and D-178's pair offset are built from), with D-177's
    single-cell test (within a factor 1.5);
  - the distribution over the re-partition draws of e = log(realised / predicted), each side being the log of the mean
    over seeds of per-seed floored ratios: median, 5th and 95th percentiles (asked), the 2.5-97.5 band, the frozen
    split's e and its mid-rank percentile among the draws (paired.band_summary's rule); and, as a partition
    sensitivity (D-166), not D-177's cell-level line, the share of draws within a factor 1.5 and the median |e|;
  - the share of defined draws where either side was floored (twomoment_cells' fraction_of_draws_floored, checked
    equal to the record's), flagged above twomoment_fit's sensitivity threshold.
Beside it, the pair's D-178 offset is copied from the 4b supplement of record (a mean over the pair's selected-C cells,
not this cell's).

Inputs (each hashed once and listed with that hash): results/twomoment_<pair>.json, its per-draw arrays
results/twomoment_<pair>_draws.npz (the record's own draw_arrays entry and manifests/draw_arrays.sha256) and the
supplement of record (pinned). No cache is read. Writes results/cell_offset_<pair without 'repair_'>_L<layer>_d300.json;
never overwrites.

    python src/cell_offset_d300.py --pair repair_olmo_base2instruct --layer 14
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as P  # noqa: E402
import twomoment_fit as TF  # noqa: E402
from provenance import (_RESULT_READS, _repo_rel, ensure_utf8_stdout, provenance_block, read_result,  # noqa: E402
                        sha256_bytes, write_results)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/cell_offset_d300.py", "src/twomoment_fit.py", "src/paired.py", "src/provenance.py",
              "src/populations.py", "src/superseded_provenance.py"]                # the last two via twomoment_fit
PRIMARY = "moment_full_in_sample"                     # D-163: the 4b primary predictor
# the D-178 offsets of record (D-300 re-issues it as _r4 with every compared key equal, so the numbers are the same)
SUPPLEMENT = ("results/twomoment_supplement_r3.json", "b7ecaf97d9caf3dd004aaee860ca8008a38d7600e441ae850a781076c9b14d74")
DRAW_MANIFEST = ROOT / "manifests" / "draw_arrays.sha256"
FACTOR = 1.5                                           # D-177's accuracy line: within a factor 1.5


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def manifest_sha(rel: str) -> str | None:
    for line in DRAW_MANIFEST.read_text(encoding="utf-8").splitlines():
        h, _, name = line.strip().partition("  ")
        if name == rel:
            return h
    return None


def get(d: dict[str, Any], *path: str) -> Any:
    """d[path[0]][path[1]]...; a clean refusal, not a traceback, when a key is missing."""
    o: Any = d
    for k in path:
        if not isinstance(o, dict) or k not in o:
            refuse(f"missing key {'.'.join(path)}")
        o = o[k]
    return o


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="one 4b cell: realised vs predicted over the D-156 draws (D-300)")
    ap.add_argument("--pair", required=True, help="e.g. repair_olmo_base2instruct")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else RESULTS / f"cell_offset_{args.pair.removeprefix('repair_')}_L{args.layer}_d300.json"
    if out.exists():
        refuse(f"{out} exists; results are never overwritten")
    code0 = P.code_block(CODE_FILES)

    rec_path = RESULTS / f"twomoment_{args.pair}.json"
    if not rec_path.is_file():
        refuse(f"{rec_path.name} is not on disk")
    rec = read_result(rec_path)
    if rec.get("run_kind") != "twomoment_cells" or get(rec, "config", "pair") != args.pair:
        refuse(f"{rec_path.name} is not the twomoment_cells record of {args.pair}")
    idx = [i for i, c in enumerate(rec["cells"]) if c.get("layer") == args.layer and c.get("probe_set") == "selected"]
    if len(idx) != 1:
        refuse(f"{len(idx)} selected-C cells at layer {args.layer} in {rec_path.name}")
    i = idx[0]
    cell = rec["cells"][i]
    flag = get(cell, "input_verification", "flag")      # required: a missing key must not read as unflagged

    # the per-draw arrays: hashed and loaded from ONE read; bound to the record and to the draw manifest
    npz_rel = get(rec, "draw_arrays", "file")
    npz_bytes = (ROOT / npz_rel).read_bytes()
    npz_sha = sha256_bytes(npz_bytes)
    if npz_sha != get(rec, "draw_arrays", "sha256") or manifest_sha(npz_rel) != npz_sha:
        refuse(f"{npz_rel} is not the record's draw_arrays or does not match {DRAW_MANIFEST.relative_to(ROOT)}")
    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as z:
        real, pred = z["log_real_ratio"], z[f"log_pred_ratio__{PRIMARY}"]
        rc, pc = z["real_censored"], z[f"pred_censored__{PRIMARY}"]
        row_layer, row_pset = int(z["cell_layer"][i]), str(z["cell_probe_set"][i])
    n_draws = get(rec, "config", "n_draws")
    if real.shape != pred.shape or real.shape != (len(rec["cells"]), n_draws + 1):
        refuse(f"{npz_rel} arrays {real.shape}/{pred.shape} do not match the record's cells and draws")
    if (row_layer, row_pset) != (args.layer, "selected"):
        refuse(f"row {i} of {npz_rel} is layer {row_layer} {row_pset}, not the cell")
    fr_real = get(cell, "realised", "paired_ratio", "frozen")
    fr_pred = get(cell, "predicted", PRIMARY, "ratio", "frozen")
    if fr_real is None or fr_pred is None or not np.isclose(np.exp(real[i, 0]), fr_real, rtol=1e-9) \
            or not np.isclose(np.exp(pred[i, 0]), fr_pred, rtol=1e-9):
        refuse("the arrays' frozen column does not equal the record's frozen ratios")

    e = real[i] - pred[i]                              # column 0: the frozen split; 1..n: the draws
    draws, frozen = e[1:], float(e[0])
    defined = np.isfinite(real[i, 1:]) & np.isfinite(pred[i, 1:])
    ok = draws[defined]
    if not ok.size:
        refuse("no draw has both realised and predicted defined")
    x, y, n_common = TF.d156_estimates(pred[i:i + 1], real[i:i + 1])
    cell_offset = float(y[0] - x[0])                   # log(mean realised / mean predicted) on the common draws
    floored = float((rc[i, 1:] | pc[i, 1:])[defined].mean())
    stored_floored = get(cell, "predicted", PRIMARY, "fraction_of_draws_floored")
    if stored_floored is None or not np.isclose(floored, stored_floored, rtol=0, atol=1e-12):
        refuse(f"the floored share {floored} is not the record's {stored_floored}")
    band = P.band_summary(draws, frozen)

    sup = read_result(ROOT / SUPPLEMENT[0])
    if _RESULT_READS[SUPPLEMENT[0]] != SUPPLEMENT[1]:
        refuse(f"{SUPPLEMENT[0]} is not the supplement of record ({SUPPLEMENT[1][:8]})")
    off = get(sup, "results", "item_level_D156", "pair_offsets", "per_pair", args.pair)
    rec_rel = _repo_rel(rec_path)
    payload = {**provenance_block("cell_offset_d300"),
               "preregistration": "D-300 (the author's question of 2026-10-01); D-156, D-163 (primary predictor), "
                                  "D-166 (partition sensitivity), D-177 (accuracy line), D-178 (offset)",
               "code": code0,
               "input_verification": {"flag": flag, "rule": "the cell's own flag in the twomoment record (D-170/D-172)"},
               "inputs": {"twomoment_record": {"file": rec_rel, "sha256": _RESULT_READS[rec_rel]},
                          "draw_arrays": {"file": npz_rel, "sha256": npz_sha,
                                          "manifest": str(DRAW_MANIFEST.relative_to(ROOT)).replace("\\", "/")},
                          "supplement": {"file": SUPPLEMENT[0], "sha256": SUPPLEMENT[1]}},
               "cell": {"pair": args.pair, "layer": args.layer, "probe_set": "selected", "probe_C": cell.get("probe_C"),
                        "is_selected_layer": cell.get("is_selected_layer"), "cell_index": i, "level": "item_level_D156"},
               "realised_paired_ratio_D156": cell["realised"]["paired_ratio"],
               "predicted_paired_ratio_D156": {"predictor": PRIMARY, **cell["predicted"][PRIMARY]["ratio"]},
               "cell_offset_D156": {
                   "definition": "log(mean realised / mean predicted paired ratio) over the draws where both are defined "
                                 "(twomoment_fit.d156_estimates): the per-cell quantity of D-177's line and D-178's offset",
                   "value": cell_offset, "n_common_draws": int(n_common[0]),
                   "within_factor_1_5_D177": bool(abs(cell_offset) <= np.log(FACTOR))},
               "log_realised_over_predicted_per_draw": {
                   "definition": "e = log realised - log predicted paired ratio per draw, each the log of the mean over "
                                 "seeds of per-seed floored ratios (twomoment_cells)",
                   "n_draws": int(draws.size), "n_undefined_draws": int(draws.size - ok.size),
                   "median": float(np.median(ok)), "p5": float(np.percentile(ok, 5)), "p95": float(np.percentile(ok, 95)),
                   "band95": band["band95"], "mean_of_per_draw_e": band["mean"],
                   "frozen": frozen, "frozen_percentile": band["frozen_percentile"],
                   "percentile_rule": "paired.band_summary: mid-rank among the defined draws",
                   "partition_sensitivity_D166": {
                       "note": "over this cell's draws; NOT D-177's accuracy line, which is over cells",
                       "share_of_draws_within_factor_1_5": float(np.mean(np.abs(ok) <= np.log(FACTOR))),
                       "median_abs_e": float(np.median(np.abs(ok)))},
                   "floored": {"share_of_defined_draws_realised_or_predicted_floored": floored,
                               "equals_record_fraction_of_draws_floored": True,
                               "above_sensitivity_threshold": floored > TF.SENSITIVITY_FLOORED_SHARE,
                               "threshold": TF.SENSITIVITY_FLOORED_SHARE,
                               "frozen_realised_floored": bool(rc[i, 0]), "frozen_predicted_floored": bool(pc[i, 0])}},
               "pair_offset_D178": {"note": "the mean over the pair's selected-C cells of each cell's D-156 offset "
                                            "(item level, D-178), copied from the supplement of record; not this cell's",
                                    **off}}
    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"]:
        refuse("code changed during the run")
    digest = write_results(out, payload, exclusive=True)
    print(f"Written: {out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
