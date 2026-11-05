"""C=1 quantities of Results V-C recomputed on current D-156 records (D-645, the author's items A1 and A2 of 2026-11-02).
Reads committed records only: no cache, no GPU, no new fit.

  (A1) SEPARABILITY. results.tex's C2.4 lines print five values whose source, results/separability_test.json, is
       superseded (D-161: code provenance unverifiable). D-099's quantity on the same 17 sweeps, read from the current
       results/depth_d156_<sweep>.json at probe_set 'C=1': per layer auroc.post_refit minus auroc.pre (each the mean over
       seeds, fixed evaluation split), the median over the sweep's layers, then the median over the 16 sweeps that are
       not the hardware control, the count of those medians below 0, the control's value, the median over the eight
       base-to-instruct sweeps, and D-611's two-sided exact sign test over their distinct backbones, a backbone being
       the (pre repo@revision, post repo@revision) pair in the record's config. The three depth_d156_lorapku_* sweeps
       stay out: D-099's list holds no PKU sweep and the printed sentence says 16.
  (A2) ALIGNMENT. results.tex prints '6 of 7' for row #15 of the ledger from the seven superseded alignvar_<pair> records
       (D-094: Procrustes under the ridge map on the mean over five layers, C=1, fixed split). The same count on the
       current results/alignvar_d156_<pair>.json at probe_set 'C=1', four ways: over the record's whole layer grid or
       over the old record's five layers (read from that record, for the layer list only), and on the fixed split
       ('frozen') or the mean over the 1000 re-partitions ('mean'). The maps differ from the old run's (D-229 fits
       them on probe_train plus the threshold_calib harmful items), so no view equals the old run's quantity.

Every value is compared with what commit 2ea2abc prints, at the printed precision, and the comparison is stored.
Writes results/c1_recompute_d645.json (never overwritten); --out elsewhere for tests.
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
from provenance import ensure_utf8_stdout, provenance_block, read_result, sha256_file, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "c1_recompute_d645.json"
CODE_FILES = ["src/c1_recompute_d645.py", "src/paired.py", "src/populations.py", "src/provenance.py",
              "src/superseded_provenance.py"]
PROBE_SET = "C=1"
LEVEL = "item_level_D156"
ARMS = ("ridge", "procrustes")
# D-099's list (src/separability_test.py PAIRS), each sweep under its D-156 name.
SWEEPS = [("mistral_base2instruct", "instruct"), ("gemma2_base2instruct", "instruct"),
          ("llama3_base2instruct", "instruct"), ("llama31_base2instruct", "instruct"),
          ("qwen_base2instruct", "instruct"), ("olmo_base2instruct", "instruct"),
          ("WG_mistral_base2instruct", "instruct"), ("WG_llama3_base2instruct", "instruct"),
          ("mistral_v01_v02_dense", "version"), ("llama3_to_31", "version"),
          ("loraA100_mistral_s25", "lora_general"), ("loraA100_mistral_s100", "lora_general"),
          ("loraA100_mistral_s400", "lora_general"), ("lorasafe_mistral_s25", "lora_safety"),
          ("lorasafe_mistral_s100", "lora_safety"), ("lorasafe_mistral_s400", "lora_safety"),
          ("hw_v100_a100", "control")]
CONTROL = "control"
INSTRUCT = "instruct"
PRINTED = {"commit": "2ea2abc", "where": "paper/sections/results.tex, the two C2.4 lines and the row #15 line",
           "median_16_sweeps": "+0.0017", "n_negative": "4 of 16", "control": "+0.0001", "median_8_instruct": "+0.0062",
           "p_backbones": "0.031", "n_backbones": 6, "procrustes_under_ridge": "6 of 7"}


def exact_two_sided(k: int, n: int) -> Fraction:
    """D-611's p: 2 x min(P[X <= k], P[X >= k]) under Binomial(n, 1/2), at most 1."""
    lo = sum(Fraction(comb(n, i), 2 ** n) for i in range(0, k + 1))
    hi = sum(Fraction(comb(n, i), 2 ** n) for i in range(k, n + 1))
    return min(Fraction(1), 2 * min(lo, hi))


def sweep_row(name: str, kind: str, rec: dict[str, Any]) -> dict[str, Any]:
    """D-099's per-sweep quantity at C=1 from one depth_d156 record."""
    cells = [c for c in rec["cells"] if c["probe_set"] == PROBE_SET]
    if not cells or any(float(c["probe_C"]) != 1.0 for c in cells):
        raise SystemExit(f"REFUSING: {name}: no {PROBE_SET} cells, or one whose probe_C is not 1")
    deltas = {str(c["layer"]): c["auroc"]["post_refit"]["mean_over_seeds"] - c["auroc"]["pre"]["mean_over_seeds"]
              for c in cells}
    cfg = rec["config"]
    return {"sweep": name, "kind": kind, "n_layers": len(cells), "median_change": st.median(deltas.values()),
            "change_per_layer": deltas, "pre": f"{cfg['pre_repo']}@{cfg['pre_rev']}",
            "post": f"{cfg['post_repo']}@{cfg['post_rev']}"}


def sign_test(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """D-611's backbone test: stops if a backbone's rows disagree in sign, as no rule for that is registered."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["pre"], r["post"]), []).append(r)
    backbones = []
    for (pre, post), g in groups.items():
        if any(r["median_change"] == 0 for r in g) or len({r["median_change"] > 0 for r in g}) != 1:
            raise SystemExit(f"REFUSING: backbone {pre} -> {post}: a zero change or rows that disagree in sign")
        backbones.append({"pre": pre, "post": post, "sweeps": [r["sweep"] for r in g],
                          "sign": "+" if g[0]["median_change"] > 0 else "-"})
    n, k = len(backbones), sum(b["sign"] == "+" for b in backbones)
    p = exact_two_sided(k, n)
    return {"n_backbones": n, "n_positive": k, "p_exact": f"{p.numerator}/{p.denominator}", "p": float(p),
            "backbones": backbones}


def separability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    real = [r for r in rows if r["kind"] != CONTROL]
    control = [r for r in rows if r["kind"] == CONTROL]
    ins = [r for r in rows if r["kind"] == INSTRUCT]
    if len(real) != 16 or len(control) != 1 or len(ins) != 8:
        raise SystemExit(f"REFUSING: {len(real)} sweeps, {len(control)} controls, {len(ins)} base-to-instruct")
    med = st.median(r["median_change"] for r in real)
    n_neg = sum(r["median_change"] < 0 for r in real)
    ctl = control[0]["median_change"]
    med_ins = st.median(r["median_change"] for r in ins)
    test = sign_test(ins)
    got = {"median_16_sweeps": f"{med:+.4f}", "n_negative": f"{n_neg} of {len(real)}", "control": f"{ctl:+.4f}",
           "median_8_instruct": f"{med_ins:+.4f}", "p_backbones": f"{test['p']:.3f}"}
    return {"median_16_sweeps": med, "n_negative": n_neg, "n_sweeps": len(real), "control": ctl,
            "median_8_instruct": med_ins, "n_instruct_positive": sum(r["median_change"] > 0 for r in ins),
            "sign_test": test, "rows": rows, "at_printed_precision": got,
            "matches_printed": {k: got[k] == PRINTED[k] for k in got}}


def mean_abs_log(rec: dict[str, Any], arm: str, field: str, layers: list[int] | None) -> float:
    """Mean over `layers` (None: the record's whole grid) of the arm's |log(FPR/target)| at C=1, item level."""
    cells = [c for c in rec["cells"] if c["probe_set"] == PROBE_SET and (layers is None or c["layer"] in layers)]
    if layers is not None and sorted(c["layer"] for c in cells) != sorted(layers):
        raise SystemExit(f"REFUSING: {rec['config']['pair']}: the grid lacks a layer of {layers}")
    return st.fmean(c[LEVEL][arm]["abs_log_ratio_to_target"][field] for c in cells)


def alignvar_views(recs: dict[str, dict[str, Any]], five: dict[str, list[int]]) -> dict[str, Any]:
    views: dict[str, Any] = {}
    for layers_name in ("grid", "five_layers"):
        for field, field_name in (("frozen", "fixed_split"), ("mean", "repartition_mean")):
            rows = {}
            for pair, rec in recs.items():
                vals = {a: mean_abs_log(rec, a, field, None if layers_name == "grid" else five[pair]) for a in ARMS}
                rows[pair] = {**vals, "procrustes_under_ridge": vals["procrustes"] < vals["ridge"]}
            n = sum(r["procrustes_under_ridge"] for r in rows.values())
            views[f"{layers_name}_{field_name}"] = {"n": n, "of": len(rows), "at_printed_precision": f"{n} of {len(rows)}",
                                                    "matches_printed": f"{n} of {len(rows)}" == PRINTED["procrustes_under_ridge"],
                                                    "rows": rows}
    return views


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"REFUSING: {args.out} exists; results are never overwritten")
    code0 = P.code_block(CODE_FILES)
    listed = set(POP.SUPERSEDED) | set(POP.SUPERSEDED_BY_ESTIMATOR)

    rows, read = [], []
    for name, kind in SWEEPS:
        stem = f"depth_d156_{name}"
        if stem in listed:
            raise SystemExit(f"REFUSING: {stem} is a superseded record")
        rows.append(sweep_row(name, kind, read_result(RESULTS / f"{stem}.json")))
        read.append(f"results/{stem}.json")
    sep = separability(rows)

    recs, five = {}, {}
    for new, old in zip(POP.ALIGNVAR_D156, POP.UPDATE_PAIRS_ALIGNVAR):
        if new in listed:
            raise SystemExit(f"REFUSING: {new} is a superseded record")
        pair = new[len("alignvar_d156_"):]
        recs[pair] = read_result(RESULTS / f"{new}.json")
        five[pair] = [r["layer"] for r in read_result(RESULTS / f"{old}.json")["rows"]]    # the layer list, nothing else
        read += [f"results/{new}.json", f"results/{old}.json"]
    align = {"views": alignvar_views(recs, five), "old_records_read_for_the_layer_list_only": True,
             "five_layers": five}

    code1 = P.code_block(CODE_FILES)
    hashes = {f: sha256_file(ROOT / f) for f in read}
    if code0["sha256"] != code1["sha256"]:
        raise SystemExit("REFUSING to write: code changed during the run")
    digest = write_results(args.out, {
        **provenance_block("c1_recompute_d645"),
        "preregistration": "none: a recomputation by the author's items A1 and A2 of 2026-11-02 (D-645); no verdict",
        "code": code0, "input_files_sha256": hashes,
        "config": {"probe_set": PROBE_SET, "level": LEVEL, "sweeps": [s for s, _ in SWEEPS], "printed": PRINTED},
        "separability": sep, "alignvar": align}, exclusive=True)
    for k, v in sep["at_printed_precision"].items():
        print(f"{k}: recomputed {v}, printed {PRINTED[k]}, match {sep['matches_printed'][k]}")
    for k, v in align["views"].items():
        print(f"{k}: Procrustes under ridge in {v['at_printed_precision']}")
    print(f"Written: {args.out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
