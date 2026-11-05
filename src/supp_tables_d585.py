"""Four Supplement tables the Method promises and no section reported (the author's item 2 of 2026-10-24, second
message; pre-registered D-585).

    python src/supp_tables_d585.py           # writes paper/tables_supplement.tex (paper/supplement.tex \\inputs it)
    python src/supp_tables_d585.py --print   # also prints each table's rows

A new module (F-092): no pinned module is edited. Every value is read from a committed record (no hand edits, no
computation but sums, rounding and the D-241 (3) class); the header lists each record read with its sha256, and the
committed main-paper table that two of the tables are anchored to. All four tables are DESCRIPTIVE, as their records
say: none is scored by a registered prediction.

tab:x5 (method.tex IV-C, 'A x5 rule ... is reported beside', D-241 (3)) - per Phase 1 headline cell (tab:fourc's
    nine rows), the 4c check's D-156 mean counts per draw over the cell's layer cells under the x5 rule
    (results/healthcheck_d156_<rec>.json confusion_counts_pooled_over_cells/selected/item_level_D156,
    gaussian|two_sided_5x): damaged = tp + fn and flagged of them = tp; undamaged = fp + tn and flagged of them = fp.
    D-241 (3) registered the x5 rule to be reported beside the x2 rule, with no prediction.
tab:geomean (method.tex IV-C, 'the geometric mean over draws of the per-draw paired ratio', D-455 (d)) - the 18
    headline cells: tab:headline's r beside the geometric mean over the 1000 draws of the per-draw paired ratio, each
    with its D-241 (3) class, from docs/paper/review_20261016b_calc.json c_d_released_pairs (status DESCRIPTIVE).
    Refuses unless each row is anchored (the arithmetic mean of the same draws equals the record's D-156 mean) and its
    r, printed at 3 decimals, and L* equal paper/tables_results.tex's tab:headline row.
tab:contrast (method.tex IV-C, 'the 200 unsafe contrast prompts are reported beside', method_facts (e)) - the share of
    XSTest's 200 unsafe contrast prompts that the frozen probe flags, before and after the update (D-156 means over
    the draws, selected probe), in the cells whose safe-prompt ratio Results V-D reports: the nine Phase 2 headline
    pairs at L* (results/overrefusal_p2_d440_<rec>.json), 1a at L16 (results/overrefusal_d156_mistral.json), 1g at
    L14 (results/overrefusal_d156_llama3_to_31.json) and 1h at L2, L4, L6, L16 and L18
    (results/overrefusal_d156_mistral_v01_v02_xsfull.json, the record results/overrefusal_curves_d156_full_r2.json
    reads). Each record is bound to its cell ID by its pre and post repo@revision. DESCRIPTIVE (D-440 (8); no
    contrast prediction is registered).
tab:lastgen (method.tex IV-A, 'the last templated position is a sensitivity arm', method_facts (b), (i), (k5)) - per
    Phase 2 pair, 2a to 2j, at the last templated position: the pair's own L* and C, r at the 1% target and its class
    (results/twomoment_p2_d361_<rec>_lastgen.json), the AUROC change, frozen minus pre, mean over the seeds on the
    fixed split (results/p2auroc_d424_<rec>_lastgen.json), and the XSTest safe-prompt paired ratio
    (results/overrefusal_p2_d440_<rec>_lastgen.json): 28 records. 2j, outside the headline, has no AUROC or XSTest
    record and prints a dash there. Refuses unless the three records of a pair agree on L* and C. DESCRIPTIVE
    (D-361 (4): beside, never a verdict; D-429; D-440 (8)).

Layout as src/results_tables_d504.py (booktabs, no vertical rules, no colour, fixed decimals per column, signs in
math). No \\ref in the file: the Supplement has no link to the main paper's labels.
"""
from __future__ import annotations

import hashlib
import re
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import pairs_table_d468 as PT  # noqa: E402
import results_tables_d504 as RT  # noqa: E402
from provenance import _RESULT_READS, _repo_rel, read_result  # noqa: E402

PAPER_DIR = ROOT / os.environ.get("GUARDRAIL_PAPER_DIR", "paper")  # where the .tex files are written (README, config/example.env)
OUT = PAPER_DIR / "tables_supplement.tex"
HEADLINE_TEX = PAPER_DIR / "tables_results.tex"
CALC = "docs/paper/review_20261016b_calc.json"
EOL = RT.EOL
DATA, CLASS, fin, f, cls241 = RT.DATA, RT.CLASS, RT.fin, RT.f, RT.cls241
# tab:contrast's Phase 1 cells: (cell ID, dataset of the threshold, record stem, layers), as Results V-D reports them
XS_P1 = [("1a", "Aegis 2.0", "mistral", [16]), ("1g", "Aegis 2.0", "llama3_to_31", [14]),
         ("1h", "Aegis 2.0", "mistral_v01_v02_xsfull", [2, 4, 6, 16, 18])]
DASH = "--"


class Reads:
    """read_result, with the records this file's tables read kept apart for its header (as src/results_figures_d570)."""

    def __init__(self) -> None:
        self.rels: set[str] = set()

    def __call__(self, rel: str) -> dict:
        doc = read_result(ROOT / rel)
        self.rels.add(_repo_rel(ROOT / rel))
        return doc

    def header(self) -> list[str]:
        anchor = hashlib.sha256(HEADLINE_TEX.read_bytes()).hexdigest()
        return (["% GENERATED by src/supp_tables_d585.py (D-585): do not edit; rerun the script.",
                 "% Sources (sha256):"] + [f"%   {rel} {_RESULT_READS[rel]}" for rel in sorted(self.rels)]
                + ["% Anchor (sha256): tab:geomean's r and L* equal tab:headline in",
                   f"%   paper/tables_results.tex {anchor}"])


def rows_by(phase: int, headline_only: bool = True) -> list[dict]:
    rows, _ = PT.rows()
    return [r for r in rows if r["phase"] == phase and (r["headline"] or not headline_only)]


def bind(cfg: dict, row: dict, rel: str) -> None:
    """A record belongs to a cell when its pre and post repo@revision are the cell's."""
    if ((cfg.get("pre_repo"), cfg.get("pre_rev")), (cfg.get("post_repo"), cfg.get("post_rev"))) != (row["pre"],
                                                                                                    row["post"]):
        raise SystemExit(f"REFUSING: {rel} is not the record of {row['id']} ({row['record']})")


def selected(doc: dict, rel: str, L: int | None = None, C: float | None = None) -> dict:
    cells = [c for c in doc["cells"] if c.get("is_selected_layer") and c.get("probe_set", "selected") == "selected"]
    if len(cells) != 1 or (L is not None and (cells[0]["layer"], cells[0]["probe_C"]) != (L, C)):
        raise SystemExit(f"REFUSING: {rel} has no single selected cell" + (f" at L{L}, C={C}" if L is not None else ""))
    return cells[0]


# --------------------------------------------------------------------------------------------------------- tab:x5
def x5_cells(load: Reads) -> list[dict]:
    head = rows_by(1)
    if len(head) != 9:
        raise SystemExit(f"REFUSING: {len(head)} Phase 1 headline cells, tab:fourc has nine")
    out = []
    for r in head:
        cc = load(f"results/healthcheck_d156_{r['record']}.json")["confusion_counts_pooled_over_cells"]["selected"][
            "item_level_D156"]
        g = {k: fin(cc[f"gaussian|two_sided_5x|{k}"]["d156_mean"]) for k in ("tp", "fn", "fp", "tn")}
        out.append({"id": r["id"], "dataset": r["dataset"], "damaged": g["tp"] + g["fn"], "flag_damaged": g["tp"],
                    "undamaged": g["fp"] + g["tn"], "flag_undamaged": g["fp"]})
    return out


def table_x5(cells: list[dict]) -> list[str]:
    body = [" & ".join([c["id"], DATA[c["dataset"]], f(c["damaged"], 2), f(c["flag_damaged"], 2),
                        f(c["undamaged"], 2), f(c["flag_undamaged"], 2)]) + EOL for c in cells]
    return ["\\begin{table}[t]",
            "\\caption{The estimated-rate check under the $\\times5$ rule, per Phase~1 cell (descriptive).}",
            "\\label{tab:x5}",
            "\\centering\\scriptsize\\setlength{\\tabcolsep}{0pt}",
            "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}llrrrr@{}}",
            "\\toprule",
            " & & \\multicolumn{2}{c}{damaged at $\\times5$ (per draw)} & "
            "\\multicolumn{2}{c}{undamaged at $\\times5$ (per draw)}" + EOL,
            "\\cmidrule(lr){3-4}\\cmidrule(l){5-6}",
            "ID & Data & layer cells & flagged & layer cells & flagged" + EOL,
            "\\midrule"] + body + [
            "\\bottomrule",
            "\\end{tabular*}",
            "\\par\\smallskip\\parbox{\\columnwidth}{\\scriptsize Means over the 1000 re-partitions of the number "
            "of the cell's evaluated layers in each class on a draw, at the selected $C$. Damaged at $\\times5$: the "
            "paired ratio is at least 5 or at most 0.2 on the draw; flagged: $G_d$ is at least 5 or at "
            "most 0.2. Aegis: Aegis~2.0; WGM: WildGuardMix.}",
            "\\end{table}"]


# ---------------------------------------------------------------------------------------------------- tab:geomean
def headline_printed() -> dict[tuple[str, str], tuple[str, str]]:
    """tab:headline as paper/tables_results.tex prints it: (ID, Data) -> (L*, r at the 1% target)."""
    text = HEADLINE_TEX.read_text(encoding="utf-8")
    blk = text.split("\\label{tab:headline}", 1)[1].split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
    out = {}
    for ln in blk.splitlines():
        if ln.endswith(EOL):
            c = [x.strip() for x in ln.removesuffix(EOL).split("&")]
            out[(c[0], c[1])] = (c[2], c[3])  # ID, Data, L*, r (D-590 dropped tab:headline's Ph. column)
    if len(out) != 18:
        raise SystemExit(f"REFUSING: {HEADLINE_TEX.name}'s tab:headline has {len(out)} rows, not 18")
    return out


def geomean_cells(load: Reads) -> list[dict]:
    calc = load(CALC)
    if calc.get("status") != "DESCRIPTIVE":
        raise SystemExit(f"REFUSING: {CALC} is not marked DESCRIPTIVE")
    by = {(x["phase"], x["pair"]): x for x in calc["c_d_released_pairs"]["rows"]}
    printed, out = headline_printed(), []
    for r in rows_by(1) + rows_by(2):
        x = by.get((r["phase"], r["record"]))
        if x is None or "gm" not in x:
            raise SystemExit(f"REFUSING: {CALC} has no geometric mean for {r['id']} ({r['record']})")
        gm = x["gm"]
        if not gm.get("anchored") or gm.get("n_nonfinite") != 0 or gm.get("n_draws_used") != 1000:
            raise SystemExit(f"REFUSING: {r['id']} ({r['record']}): the geometric mean is not anchored on 1000 draws")
        L, rp = printed[(r["id"], DATA[r["dataset"]])]
        if (x["L"], x["C"]) != (r["L"], r["C"]) or str(x["L"]) != L or f(fin(x["ratio"]), 3) != rp:
            raise SystemExit(f"REFUSING: {r['id']} {DATA[r['dataset']]}: {CALC} (L{x['L']}, r {x['ratio']}) is not "
                             f"tab:headline's (L{L}, r {rp})")
        a, g = fin(x["ratio"]), fin(gm["geometric_mean"])
        if cls241(a) != x["class"] or cls241(g) != gm["class"]:
            raise SystemExit(f"REFUSING: {r['id']}: a class in {CALC} is not D-241 (3)'s")
        out.append({"id": r["id"], "phase": r["phase"], "dataset": r["dataset"], "L": r["L"], "r": a, "gm": g,
                    "class_r": cls241(a), "class_gm": cls241(g)})
    return out


def table_geomean(cells: list[dict]) -> list[str]:
    body = []
    for i, c in enumerate(cells):
        if i and c["phase"] != cells[i - 1]["phase"]:
            body.append("\\addlinespace")
        body.append(" & ".join([c["id"], str(c["phase"]), DATA[c["dataset"]], str(c["L"]), f(c["r"], 3),
                                CLASS[c["class_r"]], f(c["gm"], 3), CLASS[c["class_gm"]]]) + EOL)
    return ["\\begin{table}[t]",
            "\\caption{The paired ratio $r$ and the geometric mean of its per-draw values in the 18 headline cells, "
            "at the 1\\% target (descriptive).}",
            "\\label{tab:geomean}",
            "\\centering\\scriptsize\\setlength{\\tabcolsep}{0pt}",
            "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}llllrlrl@{}}",
            "\\toprule",
            " & & & & \\multicolumn{2}{c}{arithmetic mean} & \\multicolumn{2}{c}{geometric mean}" + EOL,
            "\\cmidrule(lr){5-6}\\cmidrule(l){7-8}",
            "ID & Ph. & Data & $L^*$ & $r$ & class & value & class" + EOL,
            "\\midrule"] + body + [
            "\\bottomrule",
            "\\end{tabular*}",
            "\\par\\smallskip\\parbox{\\columnwidth}{\\scriptsize $r$: the mean over the 1000 re-partitions of the "
            "per-draw paired ratio, as in the main paper's headline table; geometric mean: $\\exp$ of the mean of "
            "its logarithm over the same draws. Class: closed at 2 or more, open at 0.5 or less, within otherwise. "
            "Aegis: Aegis~2.0; WGM: WildGuardMix.}",
            "\\end{table}"]


# --------------------------------------------------------------------------------------------------- tab:contrast
def contrast_cells(load: Reads) -> list[dict]:
    out = []
    p1 = {(r["id"], r["dataset"]): r for r in rows_by(1, headline_only=False)}
    for cid, data, stem, layers in XS_P1:
        rel = f"results/overrefusal_d156_{stem}.json"
        doc = load(rel)
        bind(doc["config"], p1[(cid, data)], rel)
        for L in layers:
            c = [x for x in doc["cells"] if x["layer"] == L and x["probe_set"] == "selected"]
            if len(c) != 1 or c[0]["probe_C"] != doc["config"]["selected_C"]:
                raise SystemExit(f"REFUSING: {rel} has no single selected-probe cell at L{L}")
            out.append(contrast_row(cid, 1, data, L, c[0], rel))
    for r in rows_by(2):
        rel = f"results/overrefusal_p2_d440_{r['record']}.json"
        doc = load(rel)
        bind(doc["config"], r, rel)
        out.append(contrast_row(r["id"], 2, r["dataset"], r["L"], selected(doc, rel, r["L"], r["C"]), rel))
    return out


def contrast_row(cid: str, phase: int, data: str, L: int, c: dict, rel: str) -> dict:
    it = c["item_level_D156"]
    if it["frozen_split"]["n_contrast"] != 200 or it["frozen_split"]["n_safe"] != 250:
        raise SystemExit(f"REFUSING: {rel} L{L} does not count 200 contrast and 250 safe prompts")
    return {"id": cid, "phase": phase, "dataset": data, "L": L, "pre": fin(it["contrast_flag_pre"]["mean"]),
            "post": fin(it["contrast_flag_post"]["mean"]), "safe_ratio": fin(it["paired_refusal_ratio"]["mean"])}


def table_contrast(cells: list[dict]) -> list[str]:
    body = []
    for i, c in enumerate(cells):
        if i and c["phase"] != cells[i - 1]["phase"]:
            body.append("\\addlinespace")
        body.append(" & ".join([c["id"], str(c["phase"]), DATA[c["dataset"]], str(c["L"]), f(c["pre"], 3),
                                f(c["post"], 3)]) + EOL)
    return ["\\begin{table}[t]",
            "\\caption{The share of XSTest's 200 unsafe contrast prompts that the frozen probe flags, in the cells "
            "whose safe-prompt ratio the main paper reports (descriptive).}",
            "\\label{tab:contrast}",
            "\\centering\\scriptsize\\setlength{\\tabcolsep}{0pt}",
            "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}llllrr@{}}",
            "\\toprule",
            " & & & & \\multicolumn{2}{c}{contrast prompts flagged}" + EOL,
            "\\cmidrule(l){5-6}",
            "ID & Ph. & Threshold & Layer & before & after" + EOL,
            "\\midrule"] + body + [
            "\\bottomrule",
            "\\end{tabular*}",
            "\\par\\smallskip\\parbox{\\columnwidth}{\\scriptsize Share of the 200 prompts flagged at the frozen "
            "1\\% threshold, mean over the probe seeds and then over the 1000 re-partitions that set the threshold; "
            "XSTest itself is not re-partitioned. Threshold: the dataset the threshold is set on. Aegis: Aegis~2.0; "
            "WGM: WildGuardMix.}",
            "\\end{table}"]


# ---------------------------------------------------------------------------------------------------- tab:lastgen
def lastgen_cells(load: Reads) -> list[dict]:
    pairs = rows_by(2, headline_only=False)
    if len(pairs) != 10 or sum(r["headline"] for r in pairs) != 9:
        raise SystemExit(f"REFUSING: {len(pairs)} Phase 2 pairs, the 28 lastgen records cover ten")
    out, n = [], 0
    for r in pairs:
        rel = f"results/twomoment_p2_d361_{r['record']}_lastgen.json"
        doc = load(rel)
        n += 1
        bind(doc["config"], r, rel)
        if doc["config"].get("pooling") != "lastgen":
            raise SystemExit(f"REFUSING: {rel} is not the lastgen pooling")
        c = selected(doc, rel)
        L, C = c["layer"], c["probe_C"]
        e = {"id": r["id"], "headline": r["headline"], "L": L, "C": C,
             "r": fin(c["realised"]["paired_ratio"]["mean"]), "dauc": None, "xs": None}
        e["class"] = cls241(e["r"])
        for kind, key in (("p2auroc_d424", "dauc"), ("overrefusal_p2_d440", "xs")):
            krel = f"results/{kind}_{r['record']}_lastgen.json"
            if not (ROOT / krel).is_file():
                if r["headline"]:
                    raise SystemExit(f"REFUSING: {krel} is missing for a headline pair")
                continue
            kdoc = load(krel)
            n += 1
            bind(kdoc["config"], r, krel)
            if kdoc["config"].get("pooling") != "lastgen":
                raise SystemExit(f"REFUSING: {krel} is not the lastgen pooling")
            kc = selected(kdoc, krel, L, C)
            e[key] = fin(kc["auroc"]["change"]["mean_over_seeds"] if key == "dauc"
                         else kc["item_level_D156"]["paired_refusal_ratio"]["mean"])
        out.append(e)
    if n != 28:
        raise SystemExit(f"REFUSING: {n} lastgen records read, the author's item 2 (d) names 28")
    return out


def table_lastgen(cells: list[dict]) -> list[str]:
    body = []
    for c in cells:
        body.append(" & ".join([c["id"] + ("" if c["headline"] else "$^{\\dagger}$"), str(c["L"]),
                                f"{c['C']:g}", f(c["r"], 3), CLASS[c["class"]],
                                DASH if c["dauc"] is None else f(c["dauc"], 4, signed=True),
                                DASH if c["xs"] is None else f(c["xs"], 3)]) + EOL)
    return ["\\begin{table}[t]",
            "\\caption{Phase~2 at the last templated position, each pair at its own $L^*$ (descriptive).}",
            "\\label{tab:lastgen}",
            "\\centering\\scriptsize\\setlength{\\tabcolsep}{0pt}",
            "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lllrlrr@{}}",
            "\\toprule",
            "ID & $L^*$ & $C$ & $r$ & class & $\\Delta$AUROC & XSTest $r$" + EOL,
            "\\midrule"] + body + [
            "\\bottomrule",
            "\\end{tabular*}",
            "\\par\\smallskip\\parbox{\\columnwidth}{\\scriptsize Probe and threshold fit on the hidden state at "
            "the last templated position, WildGuardMix, 1\\% target. $r$: the paired ratio, mean over the 1000 "
            "re-partitions; class: closed at 2 or more, open at 0.5 or less, within otherwise; $\\Delta$AUROC: the "
            "frozen probe after the update minus the probe before it, on the fixed split, mean over the five probe "
            "seeds; XSTest $r$: the paired ratio of the share of XSTest's 250 safe prompts flagged. $^{\\dagger}$ "
            "Outside the headline cells; it has no AUROC or XSTest record.}",
            "\\end{table}"]


# --------------------------------------------------------------------------------------------------------- file
def tex() -> str:
    load = Reads()
    x5, gm, xs, lg = x5_cells(load), geomean_cells(load), contrast_cells(load), lastgen_cells(load)
    lines = table_x5(x5) + [""] + table_geomean(gm) + [""] + table_contrast(xs) + [""] + table_lastgen(lg)
    text = "\n".join(load.header() + lines) + "\n"
    if "$$" in text or "\r" in text or not text.isascii() or "\\ref" in text:
        raise SystemExit("REFUSING: '$$', a carriage return, a non-ASCII character or a \\ref in the output")
    return text


def main(argv: list[str]) -> int:
    data = tex().encode("utf-8")
    same = OUT.is_file() and OUT.read_bytes() == data
    if not same:
        OUT.write_bytes(data)
    print(f"{'unchanged' if same else 'wrote'} {OUT.relative_to(ROOT).as_posix()} ({len(data)} bytes, sha256 "
          f"{hashlib.sha256(data).hexdigest()[:16]})")
    if "--print" in argv:
        for ln in data.decode("utf-8").splitlines():
            if ln.endswith(EOL) and not re.match(r"^(ID|\s*&)", ln):
                print(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
