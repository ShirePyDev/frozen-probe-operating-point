"""The Results figures (the author's items 1-3 of 2026-10-22; pre-registered D-570; Fig. 1 of D-570 dropped by the
author's item 1 of 2026-10-24, second message, D-585).

    python src/results_figures_d570.py           # writes paper/figures_results.tex and paper/figures_supplement.tex
    python src/results_figures_d570.py --print   # also prints the sign check and the counts behind each caption

A new module (F-092). Two pgfplots figures, so no image file is written: every coordinate is read from a committed
record through read_result (no hand edits). Each file's header lists the records ITS figures read, with their sha256
(not provenance's process-global list, which D-513 found can hold other modules' reads), so src/check_paper.py's
'generated' step can rebuild paper/figures_results.tex, which paper/main.tex \\inputs, and compare it byte for byte.
paper/figures_supplement.tex, which paper/supplement.tex \\inputs, is outside that step (as D-540's table is); its
rebuild is compared by tests/test_results_figures_d570.py.

D-585 (the author's item 1 of 2026-10-24, second message): D-570's fig:auroc-ratio (the 18 headline cells' paired
    ratio against their AUROC change, with its sign check) is dropped: every point was a tab:headline row, and the
    text states its message. fig:fourc-scatter, D-570's Fig. 2, is now Fig. 1; Fig. S-1 keeps its number.
fig:fourc-scatter (Fig. 1, one column) - the 156 layer cells of the nine Phase 1 headline cells
    (results/healthcheck_d156_<rec>.json, the records C5.1 names; selected probe set, item level). x: the 4c ratio
    G_d; y: the realised paired ratio; each on the fixed split (the 'frozen' field, the author's item 3 of 2026-10-22,
    second FIX message, D-575; before D-575 each point was the D-156 mean over the draws), x to 4 decimals and y to 3
    (D-576), both axes log, with the cuts at 0.5 and 2 on both. Squares mark the cells the AUC rule flags
    (auc_rule.flag, read on the fixed split). The caption gives two labelled sets of quadrant counts: first the fixed
    split (the plotted points per quadrant, which must equal each pair's frozen_split confusion counts, summed
    97/13/4/42, or the script refuses), then the records' pooled confusion counts per draw (gaussian|two_sided_2x,
    d156_mean), summed over the nine pairs, at 1 decimal: the precision at which they also equal the sums of
    tab:fourc's printed rows (two of the 2-decimal sums differ by 0.01 between the two ways of adding them).
fig:layer-sweep (Fig. S-1, full width) - per Phase 1 headline cell, results/depth_d156_<rec>.json's paired ratio at
    each layer (operating_point.paired_ratio.mean, the selected probe set at the selected C), 3 decimals, against the
    layer index, on a log axis with the cuts at 0.5 and 2. A ring marks L*; an open marker marks a layer where some
    draw floored a false-positive rate (realised_censoring.share_of_draws_floored > 0: a probe seed's realised FPR,
    before or after the update, was 0 and was floored at 1/n_neg, D-163 (2)). The nine lines differ by dash pattern
    and marker shape; every stroke is black and every fill grey or white, so nothing rests on colour. D-644 (23), the
    author's B23: nine lines between layers 2 and 16 could not be told apart in greyscale, so the figure is three panels
    on identical axes, three cells each, in tab:headline's order (no grouping chosen from the outcomes). Every line
    keeps its style, ring and open markers, and each panel has its own legend; only the bottom panel has x labels.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import pairs_table_d468 as PT  # noqa: E402
import results_tables_d504 as RT  # noqa: E402
from provenance import _RESULT_READS, _repo_rel, read_result  # noqa: E402

PAPER_DIR = ROOT / os.environ.get("GUARDRAIL_PAPER_DIR", "paper")  # where the .tex files are written (README, config/example.env)
OUT_MAIN = PAPER_DIR / "figures_results.tex"
OUT_SUPP = PAPER_DIR / "figures_supplement.tex"
DATA = RT.DATA
LO, HI = 0.5, 2.0                        # D-241 (3): damaged (and, for G_d, flagged) at <= 0.5 or >= 2
REF = "black!45, line width=0.5pt"       # the cut lines
FILL = "black!55"                        # a marker's fill when its layer has no floored draw
# Fig. S-1: one (dash pattern, marker, rotation) per Phase 1 cell, in tab:headline's order. One dotted pattern only
# (the review: three dotted variants did not separate in print), and each triangle on a different dash family.
SWEEP_STYLES = [("solid", "*", 0), ("dashed", "square*", 0), ("dotted", "triangle*", 0),
                ("dashdotted", "diamond*", 0), ("densely dashed", "pentagon*", 0),
                ("dash pattern=on 6pt off 2pt", "triangle*", 180), ("dashdotdotted", "oplus*", 0),
                ("loosely dashed", "otimes*", 0), ("densely dashdotted", "triangle*", 90)]


def fin(x) -> float:
    if not (isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)):
        raise SystemExit(f"REFUSING: a value is not a finite number ({x!r})")
    return float(x)


def pos(x: float, d: int) -> str:
    """A value for a log axis, with d decimals; refuses one that rounds to 0."""
    s = f"{fin(x):.{d}f}"
    if not float(s) > 0:
        raise SystemExit(f"REFUSING: {x} is not positive at {d} decimals (log axis)")
    return s


class Reads:
    """read_result, with the records one output file's figures read kept apart for its header (D-513)."""

    def __init__(self) -> None:
        self.rels: set[str] = set()

    def __call__(self, rel: str) -> dict:
        doc = read_result(ROOT / rel)
        self.rels.add(_repo_rel(ROOT / rel))
        return doc

    def header(self) -> list[str]:
        return (["% GENERATED by src/results_figures_d570.py (D-570): do not edit; rerun the script.",
                 "% Sources (sha256):"] + [f"%   {rel} {_RESULT_READS[rel]}" for rel in sorted(self.rels)])


# -------------------------------------------------------------------------------------------- fig:fourc-scatter
def fourc_points(load: Reads) -> tuple[list[dict], dict[str, float], dict[str, int]]:
    """Fig. 1's points, each a layer cell's fixed-split values (D-575), with the records' pooled confusion counts per
    draw (d156_mean, summed) and on the fixed split (frozen_split, summed). Refuses unless each pair's points by
    quadrant equal its record's frozen_split counts."""
    rows, _ = PT.rows()
    head = [r for r in rows if r["headline"] and r["phase"] == 1]
    if len(head) != 9:
        raise SystemExit(f"REFUSING: {len(head)} Phase 1 headline cells, C5.1 counts nine")
    keys = ("tp", "fn", "fp", "tn")
    pts, pooled, frozen = [], {k: 0.0 for k in keys}, {k: 0 for k in keys}
    for r in head:
        rel = f"results/healthcheck_d156_{r['record']}.json"
        doc = load(rel)
        cells = [c for c in doc["cells"] if c["probe_set"] == "selected"]
        if any(c["probe_C"] != r["C"] for c in cells) or sum(c["is_selected_layer"] for c in cells) != 1:
            raise SystemExit(f"REFUSING: {rel}'s selected cells are not at C={r['C']} with one L*")
        mine = []
        for c in cells:
            it = c["item_level_D156"]
            mine.append({"id": r["id"], "dataset": r["dataset"], "layer": c["layer"], "lstar": c["is_selected_layer"],
                         "g": fin(it["gaussian_ratio"]["frozen"]), "r": fin(it["realised_paired_ratio"]["frozen"]),
                         "auc_flag": c["auc_rule"]["flag"] is True})
        cc = doc["confusion_counts_pooled_over_cells"]["selected"]["item_level_D156"]
        rec = {k: cc[f"gaussian|two_sided_2x|{k}"]["frozen_split"] for k in keys}
        got = {k: 0 for k in keys}
        for p in mine:
            got[quadrant(p["g"], p["r"])] += 1
        if got != rec:
            raise SystemExit(f"REFUSING: {rel}: the fixed-split points by quadrant {got} are not the record's "
                             f"frozen_split counts {rec}")
        for k in keys:
            pooled[k] += fin(cc[f"gaussian|two_sided_2x|{k}"]["d156_mean"])
            frozen[k] += int(rec[k])
        pts += mine
    if len(pts) != 156:
        raise SystemExit(f"REFUSING: {len(pts)} layer cells, C5.1 counts 156")
    for p in pts:
        # x to 4 decimals (D-576): the smallest fixed-split 4c ratio, 0.000511, would plot at 0.001 at 3, 0.29 decades
        # off on the log axis; at 4 every x is within 0.01 decades of its value
        p["x"], p["y"] = pos(p["g"], 4), pos(p["r"], 3)
    return pts, pooled, frozen


def quadrant(g: float, r: float) -> str:
    """A cell's class: damaged when r is at least 2 or at most 0.5, flagged when g is (D-241 (3), inclusive cuts)."""
    dam, fl = not LO < r < HI, not LO < g < HI
    return ("tp" if fl else "fn") if dam else ("fp" if fl else "tn")


def point_quadrants(pts: list[dict]) -> dict[str, int]:
    """Fig. 1's plotted points per quadrant. Since D-575 a point is a cell's fixed-split values, so these are the fixed
    split's counts. Counted on the plotted 3-decimal values, which the reader sees; refuses if a point's unrounded
    values put it in another quadrant."""
    out = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for p in pts:
        k = quadrant(float(p["x"]), float(p["y"]))
        if k != quadrant(p["g"], p["r"]):
            raise SystemExit(f"REFUSING: {p['id']} {DATA[p['dataset']]} L{p['layer']} is in another quadrant at 3 "
                             "decimals")
        out[k] += 1
    return out


def fig_fourc(pts: list[dict], pooled: dict[str, float], frozen: dict[str, int]) -> list[str]:
    flags = [p for p in pts if p["auc_flag"]]
    if len(flags) != 2:
        raise SystemExit(f"REFUSING: the AUC rule flags {len(flags)} layer cells, the author's item 1 names two")
    n = point_quadrants(pts)
    if n != frozen:
        raise SystemExit(f"REFUSING: the plotted points by quadrant {n} are not the records' frozen_split sums "
                         f"{frozen}")
    q = {k: f"{v:.1f}" for k, v in pooled.items()}
    return (["\\begin{figure}[t]",
             "\\centering",
             "\\begin{tikzpicture}",
             "\\begin{axis}[width=0.97\\columnwidth, height=0.72\\columnwidth, xmode=log, ymode=log,",
             "  log ticks with fixed point, xtick={0.001,0.01,0.1,0.5,2,10}, ytick={0.1,0.5,1,2,10,100},",
             "  extra x ticks={0.5,2}, extra y ticks={0.5,2}, extra x tick labels={}, extra y tick labels={},",
             "  extra tick style={grid=major, grid style={" + REF + "}},",
             "  xlabel={Estimated rate ratio $G_d$}, ylabel={Realised paired ratio $r$},",
             "  tick label style={font=\\scriptsize}, label style={font=\\footnotesize},",
             "  legend style={font=\\scriptsize, draw=none, fill=none}, legend pos=north west, legend cell align=left]",
             "\\addplot[only marks, mark=o, mark size=1.2pt, black] coordinates {"]
            + [f"  ({p['x']},{p['y']}) % {p['id']} {DATA[p['dataset']]} L{p['layer']}" for p in pts]
            + ["};", "\\addlegendentry{layer cell}",
               "\\addplot[only marks, mark=square, mark size=3pt, thick, black] coordinates {"]
            + [f"  ({p['x']},{p['y']}) % {p['id']} {DATA[p['dataset']]} L{p['layer']}" for p in flags]
            + ["};", "\\addlegendentry{flagged by the AUC rule}",
               "\\end{axis}",
               "\\end{tikzpicture}",
               "\\caption{Realised paired ratio against the estimated rate ratio $G_d$ in the 156 Phase~1 layer "
               "cells, on the fixed split. Grey lines: the cuts at 0.5 and 2 on both axes. Cells damaged and flagged / "
               "damaged and not "
               f"flagged / undamaged and flagged / undamaged and not flagged: fixed split, {n['tp']}~/~{n['fn']}~/~"
               f"{n['fp']}~/~{n['tn']}; per draw (the mean over the 1000 re-partitions, to one decimal), "
               f"{q['tp']}~/~{q['fn']}~/~{q['fp']}~/~{q['tn']}. Squares: the two cells the AUC rule flags, on the "
               "fixed split.}",
               "\\label{fig:fourc-scatter}",
               "\\end{figure}"])


# ---------------------------------------------------------------------------------------------- fig:layer-sweep
def sweep_lines(load: Reads) -> list[dict]:
    rows, _ = PT.rows()
    head = [r for r in rows if r["headline"] and r["phase"] == 1]
    if len(head) != 9:
        raise SystemExit(f"REFUSING: {len(head)} Phase 1 headline cells, C5.1 counts nine")
    out = []
    for r in head:
        rel = f"results/depth_d156_{r['record']}.json"
        doc = load(rel)
        sel = PT.selected_cell(doc, rel)
        cells = sorted((c for c in doc["cells"] if c["probe_set"] == "selected"), key=lambda c: c["layer"])
        if any(c["probe_C"] != sel["probe_C"] for c in cells):
            raise SystemExit(f"REFUSING: {rel}'s selected probe set is not at one C")
        pts = [{"layer": c["layer"], "r": fin(c["operating_point"]["paired_ratio"]["mean"]),
                "floored": fin(c["realised_censoring"]["share_of_draws_floored"]) > 0} for c in cells]
        for p in pts:
            p["y"] = pos(p["r"], 3)
        out.append({"id": r["id"], "dataset": r["dataset"], "L": sel["layer"], "points": pts})
    return out


def sweep_series(ln: dict, style: tuple[str, str, int]) -> list[str]:
    """One cell's line, its open markers and its ring."""
    dash, mark, rot = style
    opts = f"solid, fill={FILL}" + (f", rotate={rot}" if rot else "")
    name = f"{ln['id']} {DATA[ln['dataset']]}"
    body = ([f"\\addplot[black, {dash}, line width=0.6pt, mark={mark}, mark size=2pt, mark options={{{opts}}}] "
             "coordinates {"]
            + [f"  ({p['layer']},{p['y']})" for p in ln["points"]]
            + ["};", f"\\addlegendentry{{{name}}}"])
    fl = [p for p in ln["points"] if p["floored"]]
    if fl:
        body += ([f"\\addplot[black, only marks, mark={mark}, mark size=2pt, "
                  f"mark options={{{opts.replace(FILL, 'white')}}}, forget plot] coordinates {{"]
                 + [f"  ({p['layer']},{p['y']}) % {name}: floored draws" for p in fl] + ["};"])
    ls = [p for p in ln["points"] if p["layer"] == ln["L"]]
    if len(ls) != 1:
        raise SystemExit(f"REFUSING: {name} has no single point at L*={ln['L']}")
    body += [f"\\addplot[black, only marks, mark=o, mark size=3.6pt, line width=0.7pt, forget plot] "
             f"coordinates {{({ln['L']},{ls[0]['y']})}}; % {name}: L*"]
    return body


def sweep_axis(i: int, last: bool) -> list[str]:
    """Panel i of three: the same axes in each, so a height reads the same across panels."""
    place = "name=s1p0" if i == 0 else f"name=s1p{i}, at={{(s1p{i - 1}.south west)}}, anchor=north west, yshift=-0.35cm"
    return [f"\\begin{{axis}}[{place}, width=\\textwidth, height=0.27\\textwidth, ymode=log, log ticks with fixed point,",
            "  ymin=0.05, ymax=80, ytick={0.1,0.5,1,2,10}, xtick distance=2, xmin=-1, xmax=43,",
            "  extra y ticks={0.5,2}, extra y tick labels={}, extra y tick style={grid=major, grid style={" + REF + "}},",
            ("  xlabel={Layer}, " if last else "  xticklabels={}, ") + "ylabel={Paired ratio $r$},",
            "  tick label style={font=\\scriptsize}, label style={font=\\footnotesize},",
            "  legend style={font=\\scriptsize, draw=none, fill=none, at={(0.99,0.97)}, anchor=north east,",
            "    legend columns=3, /tikz/every even column/.append style={column sep=8pt}}, legend cell align=left]"]


def fig_sweep(lines: list[dict]) -> list[str]:
    if len(lines) != 9:
        raise SystemExit(f"REFUSING: {len(lines)} lines, three panels of three need nine")
    body = []
    for i in range(3):
        body += sweep_axis(i, i == 2)
        for ln, style in list(zip(lines, SWEEP_STYLES))[3 * i:3 * i + 3]:
            body += sweep_series(ln, style)
        body += ["\\end{axis}"]
    return (["\\begin{figure*}[t]",
             "\\centering",
             "\\begin{tikzpicture}"]
            + body +
            ["\\end{tikzpicture}",
             "\\caption{Paired ratio at the 1\\% target against layer in the nine Phase~1 headline cells, each a mean "
             "over the 1000 re-partitions, in three panels on the same axes. Grey lines: the cuts at 0.5 and 2. A ring "
             "marks the cell's $L^*$. An open "
             "marker marks a layer where on some draw a probe seed's false-positive rate, before or after the "
             "update, was 0 and was floored at $1/n_{\\mathrm{neg}}$. Aegis: Aegis~2.0; WGM: WildGuardMix.}",
             "\\label{fig:layer-sweep}",
             "\\end{figure*}"])


# --------------------------------------------------------------------------------------------------------- files
def texts() -> tuple[str, str]:
    main, supp = Reads(), Reads()
    one = fig_fourc(*fourc_points(main))
    s1 = fig_sweep(sweep_lines(supp))
    a = "\n".join(main.header() + one) + "\n"
    b = "\n".join(supp.header() + s1) + "\n"
    for t in (a, b):
        if "$$" in t or "\r" in t or not t.isascii():
            raise SystemExit("REFUSING: '$$', a carriage return or a non-ASCII character in the output")
    return a, b


def main(argv: list[str]) -> int:
    a, b = texts()
    for path, t in ((OUT_MAIN, a), (OUT_SUPP, b)):
        data = t.encode("utf-8")
        # Written only when it differs (the review): src/check_paper.py runs this script and puts back only
        # paper/figures_results.tex, so an unchanged supplement file is never touched, and a changed one is named.
        same = path.is_file() and path.read_bytes() == data
        if not same:
            path.write_bytes(data)
        print(f"{'unchanged' if same else 'wrote'} {path.relative_to(ROOT).as_posix()} ({len(data)} bytes, sha256 "
              f"{hashlib.sha256(data).hexdigest()[:16]})")
    if "--print" in argv:
        load = Reads()
        pts, pooled, frozen = fourc_points(load)
        print("fig 1 pooled means per draw " + " ".join(f"{k} {v:.4f}" for k, v in pooled.items())
              + " | fixed split: points per quadrant " + " ".join(f"{k} {v}" for k, v in point_quadrants(pts).items())
              + ", records' frozen_split " + " ".join(f"{k} {v}" for k, v in frozen.items()))
        print("fig 1 AUC-rule flags " + ", ".join(f"{p['id']} {DATA[p['dataset']]} L{p['layer']}"
                                                  for p in pts if p["auc_flag"]))
        for ln in sweep_lines(load):
            print(f"S-1 {ln['id']} {DATA[ln['dataset']]:6} L*={ln['L']} layers {len(ln['points'])} floored "
                  f"{[p['layer'] for p in ln['points'] if p['floored']]} max {max(p['r'] for p in ln['points']):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
