"""The Results tables (the author's items 4 and 13 of 2026-10-18, sixth message; pre-registered D-504).

    python src/results_tables_d504.py           # writes paper/tables_results.tex (the file paper/main.tex \\inputs)
    python src/results_tables_d504.py --print   # also prints each table's rows

A new module (F-092). Three single-column tables, every value read from a committed record (no hand edits); the
header lists each record read with its sha256, so src/check_paper.py's 'generated' step (D-438) can rebuild the file
and compare it byte for byte.

tab:headline (item 4) - the 18 headline cells at L* (src/pairs_table_d468.rows(), headline), as
    docs/paper/results_prep_d490.py's cell_p1 / cell_p2 read them (D-490 (5a)): the paired ratio's D-156 mean at the
    1% and the 5% threshold, its D-241 (3) class at each (>= 2 fail closed, <= 0.5 fail open, else within), the TPR at
    the 1% frozen threshold before and after (D-156 means), and AUROC before and after (the mean over probe seeds of
    the probe before the update and of the frozen probe after it). Phase 1: results/depth_d156_<rec>.json and
    _fpr5.json, the selected cell; Phase 2: results/twomoment_p2_d361_<rec>_usermean.json cells and cells_fpr5 at
    (L*, C), and results/p2auroc_d424_<rec>.json for AUROC.
tab:lora (V-G) - the nine LoRA runs (populations.LORA_ARMS), as src/tables_d156.table_3 reads them: the frozen error
    (mean |log(FPR/target)| over layers, D-156 mean), the worst FPR/target (the sweep maximum of the per-layer mean,
    with its layer), delta TPR at L* (src/tables_d156.delta_tpr_at_lstar, the D-282 supplement anchored on the depth
    record), and the signed cos(theta) and ||d|| at L* (seed means on the negatives, intent_selected_d165.json).
    The HH-harmless rows carry a mark; the text states that arm's exposure (D-349, D-380, D-503 (2)). The note prints
    D-155 check 2's random-direction reference for |cos(theta)| (results/checks_4a.json: mean, analytic mean, 95% band,
    and each arm's range of per-seed percentiles), which supersedes D-104's order-of-magnitude scale (D-629); it is
    printed only while check 2's nine cells are the table's (cos_reference).
tab:fourc (V-C) - per Phase 1 headline cell, the 4c check's D-156 mean counts per draw over its layer cells
    (results/healthcheck_d156_<rec>.json confusion_counts_pooled_over_cells/selected/item_level_D156,
    gaussian|two_sided_2x): damaged = tp + fn and flagged of them = tp; undamaged = fp + tn and flagged of them = fp.
    Point values, no interval (D-503 (3)).

Design (item 13): booktabs rules, no vertical lines, no colour; numbers right-aligned with a fixed number of decimals
per column, so their decimal points align; signs set as math so + and - have one width; units in the header, not in
the cells; a one-sentence caption; abbreviations in a note under the table.

D-590 (the author's item 22 of 2026-10-24, fourth message): tab:headline drops its Ph. column (the IDs encode the
phase) and its note gives the evaluation negatives per draw (2000 in Phase 1, 6000 in Phase 2: method.tex IV-B);
tab:lora's caption says the safety arms' shifts align with the probe direction and the general arm's do not;
tab:fourc's caption gives 1e's damaged and 1g's undamaged counts per draw, read from the cells it prints.

D-632 (the author's message of 2026-10-30, second, items B1 and B2): tab:lora's caption states the comparison between
the arms only (the safety arms' shifts have a larger |cos(theta)| with the probe direction than the general arm's, and
every shift is nearly orthogonal to it: D-105, D-112), and its note gains the random-direction reference (D-629).

D-635 (the author's message of 2026-10-30, third, items 1 and 2): tab:lora's caption also says the safety arms' shifts
are longer than the general arm's, and it is printed only while both of its comparisons hold at every dose
(caption_holds); the note gives ||d||'s units. intent_selected_d165.json carries no description of d_norm, so the
words are written here, from D-104 and D-105 (d = (mu_post - mu_pre)/sd_pre) and from the record's own code, which it
pins by sha256 (src/intent_selected.py's docstring; sd is the probe's standardisation vector, src/paired.py
fit_probe); they are printed only while that pin and that docstring line hold (dnorm_definition).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import pairs_table_d468 as PT  # noqa: E402
import populations as POP  # noqa: E402
import tables_d156 as T  # noqa: E402
from provenance import _RESULT_READS, read_result  # noqa: E402

PAPER_DIR = ROOT / os.environ.get("GUARDRAIL_PAPER_DIR", "paper")  # where the .tex files are written (README, config/example.env)
OUT = PAPER_DIR / "tables_results.tex"
EOL = " \\\\"
DATA = {"Aegis 2.0": "Aegis", "WildGuardMix": "WGM"}
CLASS = {"fail closed": "closed", "fail open": "open", "within": "within"}
CORPUS = {"general (alpaca)": "Alpaca", "safety (hh-harmless)": "HH-harmless", "safety (pku-saferlhf)": "PKU-SafeRLHF"}
HH_ARM = "safety (hh-harmless)"
PKU_ARM = "safety (pku-saferlhf)"
GENERAL_ARM = "general (alpaca)"
HH_MARK = "$^{\\S}$"


def load(rel: str) -> dict:
    return read_result(ROOT / rel)


def fin(x) -> float:
    if not (isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)):
        raise SystemExit(f"REFUSING: a value is not a finite number ({x!r})")
    return float(x)


def cls241(r: float) -> str:
    """D-241 (3) on the D-156 mean: >= 2 fail closed, <= 0.5 fail open, else within."""
    return "fail closed" if r >= 2 else "fail open" if r <= 0.5 else "within"


def f(x: float, d: int, signed: bool = False) -> str:
    """x with d decimals; a signed value sets its sign in math mode so that + and - have the same width."""
    s = f"{abs(x):.{d}f}"
    if not signed:
        if x < 0:
            raise SystemExit(f"REFUSING: an unsigned column got {x}")
        return s
    if float(s) == 0:
        return "\\phantom{$+$}" + s
    return ("$-$" if x < 0 else "$+$") + s


# ------------------------------------------------------------------------------------------------- tab:headline
def cell_p1(r: dict) -> dict:
    rec, L, C = r["record"], r["L"], r["C"]
    one = f"results/depth_d156_{rec}.json"
    c1 = PT.selected_cell(load(one), one)
    op = c1["operating_point"]
    five = f"results/depth_d156_{rec}_fpr5.json"
    sel = [c for c in load(five)["cells"] if c.get("is_selected_layer") and c.get("probe_set") == "selected"]
    if len(sel) != 1 or (sel[0]["layer"], sel[0]["probe_C"]) != (L, C):
        raise SystemExit(f"REFUSING: {five} has no single selected cell at L{L}, C={C}")
    return {"r1": fin(op["paired_ratio"]["mean"]), "r5": fin(sel[0]["operating_point"]["paired_ratio"]["mean"]),
            "tpr_pre": fin(op["tpr_pre"]["mean"]), "tpr_post": fin(op["tpr_post"]["mean"]),
            "auc_pre": fin(c1["auroc"]["pre"]["mean_over_seeds"]),
            "auc_post": fin(c1["auroc"]["frozen"]["mean_over_seeds"])}


def cell_p2(r: dict) -> dict:
    rec, L, C = r["record"], r["L"], r["C"]
    rel = f"results/twomoment_p2_d361_{rec}_usermean.json"
    rr = load(rel)
    one = [c for c in rr["cells"] if c.get("layer") == L and c.get("probe_C") == C]
    five = [c for c in rr["cells_fpr5"] if c.get("layer") == L and c.get("probe_C") == C]
    if len(one) != 1 or not one[0].get("is_selected_layer") or len(five) != 1:
        raise SystemExit(f"REFUSING: {rel} has no single selected cell (1% and 5%) at L{L}, C={C}")
    arel = f"results/p2auroc_d424_{rec}.json"
    ac = [x for x in load(arel)["cells"] if x.get("is_selected_layer")]
    if len(ac) != 1 or ac[0]["layer"] != L or ac[0].get("probe_C", C) != C:
        raise SystemExit(f"REFUSING: {arel} has no single selected cell at L{L}, C={C}")
    re1 = one[0]["realised"]
    return {"r1": fin(re1["paired_ratio"]["mean"]), "r5": fin(five[0]["realised"]["paired_ratio"]["mean"]),
            "tpr_pre": fin(re1["tpr_pre"]["mean"]), "tpr_post": fin(re1["tpr_post"]["mean"]),
            "auc_pre": fin(ac[0]["auroc"]["pre"]["mean_over_seeds"]),
            "auc_post": fin(ac[0]["auroc"]["frozen"]["mean_over_seeds"])}


def headline_cells() -> list[dict]:
    rows, _ = PT.rows()
    head = [r for r in rows if r["headline"]]
    if len(head) != 18:
        raise SystemExit(f"REFUSING: {len(head)} headline cells, the Method states 18 (D-468 (4))")
    out = []
    for r in head:
        e = {"id": r["id"], "phase": r["phase"], "dataset": r["dataset"], "record": r["record"], "L": r["L"],
             "C": r["C"]}
        e.update(cell_p1(r) if r["phase"] == 1 else cell_p2(r))
        e["class1"], e["class5"] = cls241(e["r1"]), cls241(e["r5"])
        out.append(e)
    return out


def table_headline(cells: list[dict]) -> list[str]:
    body = []
    for i, c in enumerate(cells):
        if i and c["phase"] != cells[i - 1]["phase"]:
            body.append("\\addlinespace")
        body.append(" & ".join([c["id"], DATA[c["dataset"]], str(c["L"]),
                                f(c["r1"], 3), CLASS[c["class1"]], f(c["r5"], 3), CLASS[c["class5"]],
                                f(c["tpr_pre"], 3), f(c["tpr_post"], 3), f(c["auc_pre"], 4), f(c["auc_post"], 4)])
                    + EOL)
    return ["\\begin{table}[t]",
            "\\caption{In the headline cells the frozen threshold's operating point moves in both directions while "
            "AUROC barely changes.}",
            "\\label{tab:headline}",
            "\\centering\\scriptsize\\setlength{\\tabcolsep}{0pt}",
            "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lllrlrlrrrr@{}}",
            "\\toprule",
            " & & & \\multicolumn{2}{c}{1\\% target} & \\multicolumn{2}{c}{5\\% target} & "
            "\\multicolumn{2}{c}{TPR at 1\\%} & \\multicolumn{2}{c}{AUROC}" + EOL,
            "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(l){10-11}",
            "ID & Data & $L^*$ & $r$ & class & $r$ & class & pre & post & pre & post" + EOL,
            "\\midrule"] + body + [
            "\\bottomrule",
            "\\end{tabular*}",
            "\\par\\smallskip\\parbox{\\columnwidth}{\\scriptsize $r$: the paired ratio "
            "FPR$_{\\mathrm{post}}$/FPR$_{\\mathrm{pre}}$ at the frozen threshold, mean over the 1000 re-partitions, "
            "each with 2000 evaluation negatives in Phase~1 (1a to 1g) and 6000 in Phase~2 (2a to 2i); "
            "class: closed if $r\\ge2$, open if $r\\le0.5$, within otherwise; TPR at the frozen 1\\% threshold, mean "
            "over the re-partitions; AUROC before the update and of the frozen probe after it, on the fixed split, "
            "mean over the five probe seeds. Aegis: Aegis~2.0; WGM: WildGuardMix.}",
            "\\end{table}"]


# ----------------------------------------------------------------------------------------------------- tab:lora
def lora_runs() -> list[dict]:
    geo = {c["run"]: c["geometry"]["neg"] for c in load("results/intent_selected_d165.json")["cells"]
           if c["is_selected_layer"]}
    t3 = T.t3_tpr()
    out = []
    for arm, runs in POP.LORA_ARMS.items():
        for run in runs:
            rep = load(f"results/repair_d156_{run}.json")["pair_summary_mean_abs_log_over_layers"]["selected"][
                "item_level_D156"]
            dep_doc = load(f"results/depth_d156_{run}.json")
            sm = dep_doc["pair_summary"]["selected"]["item_level_D156"]["sweep_maximum"]["ratio_to_target"]
            dt, L = T.delta_tpr_at_lstar(t3, run, dep_doc)
            g = geo[run]
            out.append({"arm": arm, "run": run, "steps": int(run.rsplit("_s", 1)[1]), "L": L,
                        "frozen": fin(rep["frozen"]["mean"]), "worst": fin(sm["ratio_to_target"]["mean"]),
                        "worst_layer": int(sm["layer"]), "dtpr": fin(dt["mean_D156"]),
                        "cos": fin(g["cos"]["mean"]), "abs_cos": fin(g["abs_cos"]["mean"]),
                        "dnorm": fin(g["d_norm"]["mean"])})
    if len({r["L"] for r in out}) != 1:
        raise SystemExit("REFUSING: the LoRA runs do not share one L*")
    return out


def caption_holds(runs: list[dict]) -> None:
    """D-635: the caption says that at every dose both safety arms' shifts are longer (||d||) and have a larger
    |cos(theta)| (the seed mean of |cos|, as D-155 check 2 reads it) than the general arm's; refuse otherwise."""
    by = {(r["arm"], r["steps"]): r for r in runs}
    doses = sorted({r["steps"] for r in runs})
    for dose in doses:
        g = by[(GENERAL_ARM, dose)]
        for arm in (HH_ARM, PKU_ARM):
            s = by[(arm, dose)]
            if not (s["dnorm"] > g["dnorm"] and s["abs_cos"] > g["abs_cos"]):
                raise SystemExit(f"REFUSING: tab:lora's caption does not hold for {arm} at {dose} steps")


# ||d||'s definition (D-635): the record carries none; the code it pins states it, and the note's words follow it.
DNORM_CODE = "src/intent_selected.py"
DNORM_DEF = "d = mean(z_post) - mean(z_pre), z standardised by the probe's own statistics"


def dnorm_definition() -> None:
    """Refuse unless src/intent_selected.py is the code intent_selected_d165.json pins (sha256) and its docstring
    still defines the geometry as DNORM_DEF."""
    raw = (ROOT / DNORM_CODE).read_bytes()
    pin = load("results/intent_selected_d165.json")["code"]["sha256"].get(DNORM_CODE)
    if pin != hashlib.sha256(raw).hexdigest() or DNORM_DEF not in raw.decode("utf-8"):
        raise SystemExit(f"REFUSING: {DNORM_CODE} is not the code intent_selected_d165.json pins, or no longer "
                         "defines d as the note says")


def cos_reference(runs: list[dict]) -> dict:
    """D-155 check 2 (results/checks_4a.json), the random-direction reference for |cos(theta)| (D-629). check 2 was
    computed against results/intent_selected.json (D-154), which intent_selected_d165.json, the record this table
    reads, supersedes with identical numbers (D-216 (6)); so the reference is printed only while check 2's cells are
    the table's: the same nine runs and arms in order, layer L*, C, seeds, revisions and input caches, and per-seed
    |cos(theta)| and ||d|| equal to the D-165 record's within src/checks_4a.py's own rtol 1e-6."""
    ck, it = load("results/checks_4a.json"), load("results/intent_selected_d165.json")
    c2, ckc, itc = ck["check2_random_direction_null"], ck["config"], it["config"]
    if [(a["arm"], a["run"]) for a in c2["arms"]] != [(r["arm"], r["run"]) for r in runs]:
        raise SystemExit("REFUSING: D-155 check 2's runs are not tab:lora's")
    if not (ckc["layer"] == itc["selected_layer"] == runs[0]["L"] and ckc["selected_C"] == itc["selected_C"]
            and ckc["seeds"] == itc["seeds"] and ck["input_cache_sha256"] == it["input_cache_sha256"]):
        raise SystemExit("REFUSING: D-155 check 2's layer, C, seeds or input caches are not tab:lora's")
    revs = [k for k in ckc if k.endswith(("_repo", "_rev"))]
    if len(revs) != 2 + 2 * len(runs) or any(ckc[k] != itc.get(k) for k in revs):
        raise SystemExit("REFUSING: D-155 check 2's model revisions are not tab:lora's")
    geo = {c["run"]: c["geometry"]["neg"] for c in it["cells"] if c["is_selected_layer"]}
    for a in c2["arms"]:
        g = geo[a["run"]]
        for mine, theirs in ((a["abs_cos_per_seed"], g["abs_cos"]["per_seed"]),
                             (a["d_norm_per_seed"], g["d_norm"]["per_seed"])):
            if len(mine) != len(ckc["seeds"]) or len(mine) != len(theirs) or not all(
                    math.isclose(fin(x), fin(y), rel_tol=1e-6, abs_tol=1e-9) for x, y in zip(mine, theirs)):
                raise SystemExit(f"REFUSING: D-155 check 2's {a['run']} differs from intent_selected_d165.json")
    dim = 2.0 / (math.pi * fin(c2["analytic_mean_sqrt_2_over_pi_d"]) ** 2)
    if abs(dim - round(dim)) > 1e-6:
        raise SystemExit("REFUSING: check 2's analytic mean does not give an integer dimension")
    pct: dict[str, list[float]] = {}
    for a in c2["arms"]:
        lo, hi = (fin(x) for x in a["percentile_range"])
        p = pct.setdefault(a["arm"], [lo, hi])
        p[0], p[1] = min(p[0], lo), max(p[1], hi)
    if set(pct) != {GENERAL_ARM, HH_ARM, PKU_ARM}:
        raise SystemExit("REFUSING: check 2's arms are not the general and the two safety arms")
    return {"dim": int(round(dim)), "n_dirs": int(ckc["n_dirs"]), "mean": fin(c2["pooled_mean"]),
            "analytic": fin(c2["analytic_mean_sqrt_2_over_pi_d"]), "lo": fin(c2["pooled_q2_5"]),
            "hi": fin(c2["pooled_q97_5"]), "pct": pct,
            "safety": [min(pct[HH_ARM][0], pct[PKU_ARM][0]), max(pct[HH_ARM][1], pct[PKU_ARM][1])]}


def cos_reference_note(ref: dict) -> str:
    """The note's sentence on the reference scale (D-629, D-632 B1): every number from cos_reference."""
    def rng(p: list[float]) -> str:
        return f"{p[0]:.1f} to {p[1]:.1f}"
    return (f"For a random direction in {ref['dim']:,} dimensions, $|\\cos\\theta|$ has mean {ref['mean']:.5f} "
            f"(analytically $\\sqrt{{2/({ref['dim']}\\pi)}}={ref['analytic']:.5f}$) and 95\\% band "
            f"[{ref['lo']:.5f}, {ref['hi']:.5f}], pooled over {ref['n_dirs']:,} directions for each probe seed; each "
            f"seed's observed $|\\cos\\theta|$ lies at percentiles {rng(ref['pct'][GENERAL_ARM])} of its own "
            f"random-direction distribution for Alpaca and {rng(ref['safety'])} for the safety arms (HH-harmless "
            f"{rng(ref['pct'][HH_ARM])}, PKU-SafeRLHF {rng(ref['pct'][PKU_ARM])}). ")


def table_lora(runs: list[dict], ref: dict) -> list[str]:
    body = []
    for i, r in enumerate(runs):
        if i and r["arm"] != runs[i - 1]["arm"]:
            body.append("\\addlinespace")
        first = i == 0 or r["arm"] != runs[i - 1]["arm"]
        name = (CORPUS[r["arm"]] + (HH_MARK if r["arm"] == HH_ARM else "")) if first else ""
        body.append(" & ".join([name, str(r["steps"]), f(r["frozen"], 3), f(r["worst"], 2), str(r["worst_layer"]),
                                f(r["dtpr"], 3, True), f(r["cos"], 4, True), f(r["dnorm"], 2)]) + EOL)
    L = runs[0]["L"]
    return ["\\begin{table}[t]",
            "\\caption{At matched dose the two safety-trained adapters move the frozen threshold more than the "
            "general one, and their shifts are both longer than the general arm's and have a larger "
            "$|\\cos\\theta|$ with the probe direction, though every shift is nearly orthogonal to it.}",
            "\\label{tab:lora}",
            "\\centering\\scriptsize\\setlength{\\tabcolsep}{0pt}",
            "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lrrrrrrr@{}}",
            "\\toprule",
            " & & & \\multicolumn{2}{c}{worst FPR/target} & \\multicolumn{3}{c}{at $L^*=" + str(L) + "$}" + EOL,
            "\\cmidrule(lr){4-5}\\cmidrule(l){6-8}",
            "Corpus & Steps & Frozen error & ratio & layer & $\\Delta$TPR & $\\cos\\theta$ & "
            "$\\lVert d\\rVert$" + EOL,
            "\\midrule"] + body + [
            "\\bottomrule",
            "\\end{tabular*}",
            "\\par\\smallskip\\parbox{\\columnwidth}{\\scriptsize LoRA fine-tunes of Mistral-7B v0.1 on Aegis~2.0. "
            "Frozen error: mean $|\\log(\\mathrm{FPR}/\\mathrm{target})|$ over the layer sweep; worst FPR/target: the "
            "largest per-layer mean over the sweep and its layer; $\\Delta$TPR: post minus pre at the frozen "
            "threshold; $\\cos\\theta$ and $\\lVert d\\rVert$: the signed cosine between the mean shift of the "
            "fixed evaluation split's negatives and the probe direction, and the shift's length in standard "
            "deviations, $d=(\\mu_{\\mathrm{post}}-\\mu_{\\mathrm{pre}})/\\mathrm{sd}_{\\mathrm{pre}}$: the negatives' "
            "mean hidden state after minus before the update, divided feature by feature by the standard deviation "
            "of the probe's pre-update fit prompts; all means over the re-partitions or the "
            "probe seeds. " + cos_reference_note(ref) + HH_MARK +
            " Trained on conversations that contain evaluation items (see text).}",
            "\\end{table}"]


# ---------------------------------------------------------------------------------------------------- tab:fourc
def fourc_cells() -> list[dict]:
    rows, _ = PT.rows()
    head = [r for r in rows if r["headline"] and r["phase"] == 1]
    if len(head) != 9:
        raise SystemExit(f"REFUSING: {len(head)} Phase 1 headline cells, C5.1 counts nine")
    out = []
    for r in head:
        cc = load(f"results/healthcheck_d156_{r['record']}.json")["confusion_counts_pooled_over_cells"]["selected"][
            "item_level_D156"]
        g = {k: fin(cc[f"gaussian|two_sided_2x|{k}"]["d156_mean"]) for k in ("tp", "fn", "fp", "tn")}
        out.append({"id": r["id"], "dataset": r["dataset"], "record": r["record"],
                    "damaged": g["tp"] + g["fn"], "flag_damaged": g["tp"],
                    "undamaged": g["fp"] + g["tn"], "flag_undamaged": g["fp"]})
    return out


def table_fourc(cells: list[dict]) -> list[str]:
    body = [" & ".join([c["id"], DATA[c["dataset"]], f(c["damaged"], 2), f(c["flag_damaged"], 2),
                        f(c["undamaged"], 2), f(c["flag_undamaged"], 2)]) + EOL for c in cells]
    by_id = {c["id"]: c for c in cells if c["id"] in ("1e", "1g")}  # D-590: the caption's two small counts
    if sorted(by_id) != ["1e", "1g"] or sum(c["id"] in by_id for c in cells) != 2:
        raise SystemExit("REFUSING: tab:fourc's caption needs exactly one 1e and one 1g cell")
    return ["\\begin{table}[t]",
            "\\caption{Per Phase~1 cell the estimated-rate check flags most damaged layer cells and few undamaged ones, except "
            "in 1e, where it misses most of the few damaged cells (" + f(by_id["1e"]["damaged"], 2) + " per draw), "
            "and in 1g, where it flags most of the few undamaged cells (" + f(by_id["1g"]["undamaged"], 2)
            + " per draw).}",
            "\\label{tab:fourc}",
            "\\centering\\scriptsize\\setlength{\\tabcolsep}{0pt}",
            "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}llrrrr@{}}",
            "\\toprule",
            " & & \\multicolumn{2}{c}{damaged (per draw)} & \\multicolumn{2}{c}{undamaged (per draw)}" + EOL,
            "\\cmidrule(lr){3-4}\\cmidrule(l){5-6}",
            "ID & Data & layer cells & flagged & layer cells & flagged" + EOL,
            "\\midrule"] + body + [
            "\\bottomrule",
            "\\end{tabular*}",
            "\\par\\smallskip\\parbox{\\columnwidth}{\\scriptsize Means over the 1000 re-partitions of the number "
            "of the cell's evaluated layers in each class on a draw. Aegis: Aegis~2.0; "
            "WGM: WildGuardMix.}",
            "\\end{table}"]


# --------------------------------------------------------------------------------------------------------- file
def tex() -> str:
    head, lora, four = headline_cells(), lora_runs(), fourc_cells()
    caption_holds(lora)
    dnorm_definition()
    lines =(table_headline(head) + [""] + table_lora(lora, cos_reference(lora)) + [""] + table_fourc(four))
    hdr = ["% GENERATED by src/results_tables_d504.py (D-504): do not edit; rerun the script.",
           "% Sources (sha256):"] + [f"%   {rel} {sha}" for rel, sha in sorted(_RESULT_READS.items())]
    return "\n".join(hdr + lines) + "\n"


def main(argv: list[str]) -> int:
    text = tex()
    if "$$" in text:
        raise SystemExit("REFUSING: '$$' would open display math")
    OUT.write_bytes(text.encode("utf-8"))
    print(f"wrote {OUT.relative_to(ROOT).as_posix()} ({len(text.encode('utf-8'))} bytes, sha256 "
          f"{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]})")
    if "--print" in argv:
        for ln in text.splitlines():
            if ln.endswith(EOL):
                print(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
