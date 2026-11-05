"""Paper tables rebuilt from the D-156 records whose inputs are final (D-297 (6)): Tables I to V.

Writes paper/tables_d156.tex, the file paper/main.tex \\inputs (the old synthesis2 tables, pre-D-156, are set aside as
paper/aside/tables_pre_d438.tex, D-438). Every number is COPIED from a record; none is computed here. The header names
each source file with its sha256.

  TABLE I (D-180): per update pair, the selected C, item level. The headline row at the D-117 selected layer L*: the
    paired ratio (D-156 mean [95% band]), ratio to target, and delta TPR (post - pre, D-156 mean) with the D-282
    combined 95% interval. Then the
    sweep maximum of the paired ratio (its layer, mean [band]; its null is 1) and min CKA. Rows whose inputs carry
    the D-170 flag are marked with a dagger: they enter no paper table before D-168 is final.
    Sources: depth_d156_summary_r2.json and table1_tpr_d282_r2.json, the D-305 re-issues naming every cache input's
    repo ID and revision (D-300 (5)); equal to depth_d156_summary.json (D-190) and table1_tpr_d282.json (D-287, D-295)
    in every key but utc, git, env, code and preregistration (provenance.reissue_compare).
  TABLE II (4d; layout D-318 (3), D-324 (4)): per released pair (the nine with a repair run, Table I's order) at L*,
    selected C, item level: FPR/target as D-156 mean [95% band] (set on two lines) and, in a second panel, TPR (D-156
    mean only; the D-282 combined interval is not computed for these arms), for each repair labelled by its reference
    pool and grouped by what the defender must hold: frozen; stored calibration data (threshold-only on the stored
    calibration set, R_same = 4d A's quantile_thr_benign, the exact quantile transfer of T_d on cal_d's negatives;
    restandardisation on the class-balanced reference, 4d A's restandardise); fresh benign prompts (4d B
    'score|R_fresh|n=1000'); live traffic at 1% (4d B quantile transfer 'score|traffic|pi=0.01|n=1000' and
    restandardisation 'feature|traffic|pi=0.01|n=1000'); labels (4d A's refit_oracle); the floor (FPR_pre/target and
    TPR_pre, identical in every 4d A arm block or refused). A group's columns sit together, so the order differs
    from the D-318 (3) list in that its columns 2 and 5 are adjacent. 4d A's per-draw population (calibration and
    evaluation negatives, evaluation positives) is copied from the depth record each repair record anchors to (its
    sha256 checked). Markers from the D-257 paired test (D-257 (3) wording), Aegis pairs only: '*' on the frozen
    column where the record's damage_detected is true; 'f' on the restandardise and refit_oracle columns where the
    verdict is 'at floor' (refused if any is above floor, which the caption does not word), with the largest
    'at most X above floor' of each column in the caption
    (moment_contrasts_d156_r2.json over moment_d156_<core>_d117.json; used only where that record is the file the
    contrasts read, at 4d A's layer and input caches, with its frozen, restandardise and refit_oracle blocks equal to
    4d A's exactly). R_same, the 4d B columns and the WildGuardMix pairs have no such test. LoRA runs, the GPU
    control, C=1 and the cluster level are left out.
    Sources: repair_d156_<core>.json (D-289) and the depth_d156_<core>.json each anchors to; the POP.POOLSTUDY_D156
    records (option B for the Aegis pairs, D-306; the _r2 re-issues for WildGuardMix; D-322);
    moment_contrasts_d156_r2.json and moment_d156_<core>_d117.json (D-257).
  TABLE III (4a, the LoRA arms at matched dose): per run, mean |log(FPR/target)| over the layer sweep for the frozen
    threshold and for restandardisation (4d A, repair_d156_<run>), D-156 mean [band]; the worst FPR/target (the depth
    record's sweep maximum of ratio to target, mean, with its layer); delta TPR (post - pre) at the selected layer L*,
    the D-156 mean with the D-282 combined 95% interval, as Table I's column (D-287); the mean AUROC drop over layers
    with its item bootstrap interval; and the update's geometry at the selected layer as the record holds it: ||d||
    and the signed cos(theta), seed means on the negatives. The delta TPR cell is read from table3_tpr_d282.json, used
    only where that supplement names the bytes of the depth record read here, at its L*, with the frozen value, mean
    and band of the record's headline delta TPR exactly; it must be a registered 1% record covering the nine runs, run
    under the 6-thread BLAS pin, with the bytes its line in logs/table3_tpr_d282_20261003.sha256 names (the lane
    list's manifest job). Rows whose supplement row carries under_check (D-344 (3), D-349) get the marker $^{\\S}$ on
    their steps cell and the caption a sentence saying why; the flagged rows must be exactly the HH arm's, and the
    record's status text must state that sentence. The supplement's between-arm gaps are not printed.
    D-401 (c), D-380: those three rows ENTER the paper with their D-363 sensitivity value BESIDE each number -- the same
    quantity recomputed with the D-363 removal set (826 items) applied, copied from hh_sensitivity_d363.json's
    table_iii block (BESIDE_LAYOUT: a second line under the row) and refused unless the record is the one D-380 names,
    at its config hash and removal set, with every number this table needs and with each cell's value OF RECORD equal
    to the number printed beside it. The mark and its sentence stay (D-401 (c)); the caption gains one sentence.
    Sources: repair_d156_<run>.json (D-289), depth_d156_<run>.json (D-224), intent_selected_d165.json,
    table3_tpr_d282.json (D-282 supplement, src/table3_tpr_d282.py), hh_sensitivity_d363.json (D-363, D-380).
  TABLE IV (4c): sensitivity and specificity of the primary Gaussian health ratio (D-264), the secondary direct ratio
    and the AUC staleness rule, for a two-fold change, pooled from mean counts with pair-cluster intervals. The LoRA
    rows that pool the HH runs (all nine, the six safety runs; read from each row's pairs, refused unless exactly
    those two) carry Table III's under-check marker $^{\\S}$ and its caption sentence (D-357 (B)(4), D-363), until
    D-349 is done; the numbers of record are unchanged. D-401 (c), D-380: those two rows carry their D-363 sensitivity
    value beside each number, copied from hh_sensitivity_d363.json's table_iv_lora block ('LoRA, all nine runs' and
    'LoRA, safety (six runs)') under the same refusals as Table III's, with one caption sentence.
    Sources: healthcheck_d156_summary.json (D-255), healthcheck_d156_summary_lora.json (D-295),
    hh_sensitivity_d363.json (D-363, D-380).
    LOWER PANEL (D-432 (4), moved here from src/tables.py by D-438): the published staleness rule's count per
    population of record at selected C -- cells and stale by rule, for the released pairs and the nine LoRA runs, with
    the two controls (Mistral v0.1 -> v0.2; GPU change) beside -- with src/tables.py's values and wording, copied from
    docs/reports_d419/item3_numbers.json (written by docs/reports_d419/item3.py from the results/depth_d156_<core>.json
    records, cross-checked against results/healthcheck_d156_<core>.json), read only once its bytes match
    manifests/item3_numbers_d438.sha256 (refused otherwise). Its sha256 is stamped on the header's first line.
  TABLE V (the 4d robustness table, D-324 (4)), written after Table IV so that Tables I-IV keep their numbers: per
    released pair, from the same 4d B records, in three panels. (a) Prevalence drift: quantile transfer with the pre
    window at 1% and the post window at 5% and 25% (FPR/target mean [band], TPR mean). (b) Poisoning: the stored
    minimum rho for a 10-point TPR loss for A0 and A1, the score arm into R_same (re-set at 1% on cal_d, on 4d B's
    draws), into R_fresh and into traffic_1pct, and the feature arm into traffic_1pct, with
    the number of injected items m copied from the key of the poisoned configuration at that rho; a stored null
    prints '>' and the record's searched_up_to, which must equal the last rho of the grid the caption lists. The
    records ran poisoning at one pool size only (checked: one m per rho, each m = round(rho * n) for the record's
    single target size); m at the other pool sizes is not in the records and is not computed here. (c) Pool size:
    the threshold re-set on a nested R_fresh subset, quantile transfer on 1% traffic and restandardisation on 1%
    traffic at each of config.pool_sizes (FPR/target mean [band]), under the poisoning panel's headers. The Aegis
    records' origin-stratified cal_d counts (origin_strata) are copied into the population sentence. FPR is floored at 1 of the 1000 evaluation negatives per seed, so a
    mean that prints as that floor is shown as '<= floor'. The caption's traffic counts at 1% follow the record's
    rule round(pi * n) on config.default_pool, and its sources string is checked.
    Sources: the POP.POOLSTUDY_D156 records.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import input_registry as REG  # noqa: E402
import populations as POP  # noqa: E402
from provenance import _RESULT_READS, _repo_rel, ensure_utf8_stdout, read_result  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "paper" / "tables_d156.tex"
EOL = " \\\\\n"
NICE = {"mistral_base2instruct": "Mistral-7B v0.1 $\\to$ Instruct-v0.1", "gemma2_base2instruct": "Gemma-2-9b $\\to$ 9b-it",
        "llama3_base2instruct": "Llama-3-8B $\\to$ Instruct", "llama31_base2instruct": "Llama-3.1-8B $\\to$ Instruct",
        "qwen_base2instruct": "Qwen2.5-7B $\\to$ Instruct", "olmo_base2instruct": "OLMo-2-7B $\\to$ Instruct",
        "llama3_to_31": "Llama-3-8B $\\to$ Llama-3.1-8B", "WG_mistral_base2instruct": "Mistral-7B v0.1 $\\to$ Instruct-v0.1 (WGM)",
        "WG_llama3_base2instruct": "Llama-3-8B $\\to$ Instruct (WGM)", "mistral_v01_v02_dense": "Mistral-7B v0.1 $\\to$ v0.2"}
ARMS = {"general (alpaca)": "general (alpaca)", "safety (hh-harmless)": "safety (hh-harmless)",
        "safety (pku-saferlhf)": "safety (pku-saferlhf)"}


def num(v: float, f: str) -> str:
    """v formatted, with a value that rounds to zero ('-0.0000' from e.g. -2e-05, or '+0.000') printed unsigned."""
    s = f.format(v)
    return s.lstrip("+-") if re.fullmatch(r"[-+]?0\.0*", s) else s


def mb(b: dict[str, Any], f: str = "{:.2f}") -> str:
    """'mean [lo, hi]' of a D-156 band summary."""
    return f"{num(b['mean'], f)} [{num(b['band95'][0], f)}, {num(b['band95'][1], f)}]"


def ci(v: float, c: list[float], f: str = "{:.3f}") -> str:
    return f"{num(v, f)} [{num(c[0], f)}, {num(c[1], f)}]"


DAGGER = "$^\\dagger$"
_FINAL: list[dict | None] = []


def d168_final() -> dict | None:
    """D-300 (3): the final D-168 record, once src/input_registry.py pins it (None before). When pinned it is read
    through read_result, so the header lists it with its sha256; the D-170 daggers are then removed."""
    if not _FINAL:
        f = REG.d168_final()
        if f is not None:
            read_result(REG.D168_FINAL)
            rel = _repo_rel(REG.D168_FINAL)
            if _RESULT_READS.get(rel) != f["sha256"]:          # the bytes listed are the bytes the registry checked
                raise SystemExit(f"REFUSING: {rel} changed between the registry's check and the tables' read")
        _FINAL.append(f)
    return _FINAL[0]


def flagged_pairs() -> set[str]:
    """The update pairs whose depth inputs carry the D-170 flag (input verified by pin only, pending D-168); none once
    the final D-168 record is pinned."""
    if d168_final() is not None:
        return set()
    summ = read_result(RES / "depth_d156_summary_r2.json")
    return {stem[len("depth_"):] for stem, r in summ["table_1"].items() if r["input_verification"]["flag"]}


def table_1() -> str:
    summ = read_result(RES / "depth_d156_summary_r2.json")
    t1 = read_result(RES / "table1_tpr_d282_r2.json")
    rows = []
    for stem in POP.UPDATE_PAIRS_DEPTH:
        r = summ["table_1"][stem]
        it = r["selected"]["item_level_D156"]
        h, sm = it["headline_selected_layer"], it["sweep_maximum"]["paired_ratio"]
        L = h["layer"]
        dt = t1["rows"][stem]["probe_sets"]["selected"]["layers"][str(L)]["delta_tpr"]
        flag = DAGGER if r["input_verification"]["flag"] and d168_final() is None else ""
        rows.append(f"{NICE[stem[len('depth_'):]]}{flag} & {L} & {mb(h['paired_ratio'])} & {mb(h['ratio_to_target'])} & "
                    f"{ci(dt['mean_D156'], dt['combined']['ci95'], '{:+.3f}')} & "
                    f"L{sm['layer']}: {mb(sm['paired_ratio'])} & {r['selected']['min_cka_over_layers']:.5f}" + EOL)
    return ("\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\caption{Frozen-threshold damage per update pair at the D-117 selected "
            "layer $L^*$ (selected $C$, item level): D-156 mean [95\\% band over 1000 re-partitions]; $\\Delta$TPR "
            "(post $-$ pre): D-156 mean with the combined 95\\% interval (re-partitions and an item bootstrap over "
            "positives, D-282); the sweep maximum of the paired ratio, whose no-update value is 1."
            + (" $^\\dagger$: input verified by pin only, pending D-168." if d168_final() is None else "") + "}\n"
            "\\label{tab:damage-d156}\n\\label{tab:damage}\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrlllll@{}}\n\\toprule\n"
            "update & $L^*$ & paired ratio & FPR/target & $\\Delta$TPR & sweep max (paired ratio) & min CKA" + EOL +
            "\\midrule\n" + "".join(rows) + "\\bottomrule\n\\end{tabular*}\n\\end{table*}\n")


# 4d (Table II and the robustness table): the nine released pairs that have a repair run, in Table I's order
REL_PAIRS = [s[len("depth_"):] for s in POP.UPDATE_PAIRS_DEPTH if f"repair_{s[len('depth_'):]}" in POP.UPDATE_PAIRS_REPAIR]
MARK_DAMAGE, MARK_FLOOR, AT_MOST = "$^{\\ast}$", "$^{f}$", "$\\leq$"
EVAL_4DB = "the draw's evaluation part minus R_FRESH (1000 negatives); eval positives"   # the captions say so
N_EVAL_4DB = int(re.search(r"\((\d+) negatives\)", EVAL_4DB).group(1))   # 4d B's evaluation negatives per draw
SOURCES_4DB = "benign: cal_d negatives; harmful: threshold_calib harmful H; probe_train in no pool (D-244 (4))"
FLOOR_ARMS = ("restandardise", "refit_oracle")   # D-257 (3): 'at floor' for repair arms; the frozen arm: damage detected
DESIGN_4DB = {True: "as registered (WildGuardMix", False: "option B, origin-stratified"}  # keyed by 'is a WGM pair'
POPULATIONS = ("4d B over {n_b} draws, evaluated on {n_e} negatives, each draw's evaluation part minus "
               "R$_\\mathrm{{fresh}}$, and the evaluation positives; for the Aegis pairs the draws are origin-stratified "
               "(D-298: cal$_d$ holds {tc} threshold-calibration and {ev} evaluation negatives), for WildGuardMix as "
               "registered")
TPR_NOTE = "TPR: D-156 mean only (no D-282 interval for these arms)."
WGM_NOTE = "(WGM): WildGuardMix; the other rows are Aegis pairs."


def same(vals: list[Any], what: str) -> Any:
    """The one value every record holds for `what`; refuses when they differ."""
    if any(v != vals[0] for v in vals):
        raise SystemExit(f"REFUSING: the records differ in {what}: {vals}")
    return vals[0]


def part_a(core: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """4d A (D-289): repair_d156_<core>.json and its one cell at config.selected_layer with probe set 'selected'."""
    rec = read_result(RES / f"repair_d156_{core}.json")
    L = rec["config"]["selected_layer"]
    cells = [c for c in rec["cells"] if c["layer"] == L and c["probe_set"] == "selected"]
    if len(cells) != 1 or cells[0]["probe_C"] != rec["config"]["selected_C"]:
        raise SystemExit(f"REFUSING: repair_d156_{core}.json holds no single selected-C cell at layer {L}")
    return rec, cells[0]


def population_a(rec: dict[str, Any], core: str) -> tuple[int, int, int]:
    """4d A's per-draw population (calibration negatives cal_d, evaluation negatives, evaluation positives), from the
    depth record the repair record anchors its frozen arm to (anchor.file, which must carry anchor.sha256)."""
    rel = rec["anchor"]["file"]
    dep = read_result(ROOT / rel)
    if _RESULT_READS[rel] != rec["anchor"]["sha256"] or dep["config"]["selected_layer"] != rec["config"]["selected_layer"]:
        raise SystemExit(f"REFUSING: {rel} is not the depth record repair_d156_{core}.json anchors to")
    pn = dep["config"]["pooled_negatives"]
    return pn["n_calibration"], pn["n_evaluation"], dep["config"]["n_eval_positives"]


def part_b() -> dict[str, tuple[str, dict[str, Any]]]:
    """4d B's primary record per pair, keyed by the core of its config.pair: POP.POOLSTUDY_D156 (option B for the
    Aegis pairs, D-306; the _r2 re-issues for WildGuardMix). Refuses unless they cover the nine pairs once each, and each
    evaluates on EVAL_4DB, draws its traffic from SOURCES_4DB and carries its population's design."""
    out: dict[str, tuple[str, dict[str, Any]]] = {}
    for stem in POP.POOLSTUDY_D156:
        rec = read_result(RES / f"{stem}.json")
        pair = rec["config"]["pair"]
        core = pair[len("repair_"):]
        if not pair.startswith("repair_") or core in out:
            raise SystemExit(f"REFUSING: {stem}.json names pair {pair!r}, not one new released pair")
        if (rec["config"]["evaluation"] != EVAL_4DB or rec["config"]["sources"] != SOURCES_4DB
                or not rec["config"]["design"].startswith(DESIGN_4DB[core.startswith("WG_")])):
            raise SystemExit(f"REFUSING: {stem}.json's evaluation, sources or design is not the one the captions state")
        out[core] = (stem, rec)
    if sorted(out) != sorted(REL_PAIRS):
        raise SystemExit(f"REFUSING: 4d B's records cover {sorted(out)}, not the released pairs {REL_PAIRS}")
    return out


def origin_cal(B: dict[str, tuple[str, dict[str, Any]]]) -> dict[str, int]:
    """4d B's population wording, as format fields: n_b (draws), n_e (evaluation negatives) and the (TC, EV) origin
    counts of cal_d that every stratified draw of every Aegis record realised (origin_strata, checked over all n_b
    draws); the WildGuardMix records must carry no origin strata."""
    n_b = same([B[c][1]["config"]["n_reps"] for c in REL_PAIRS], "4d B's n_reps")
    if any("origin_strata" in B[c][1] for c in REL_PAIRS if c.startswith("WG_")):
        raise SystemExit("REFUSING: a WildGuardMix 4d B record carries origin strata; the captions say it does not")
    strata = [B[c][1]["origin_strata"] for c in REL_PAIRS if not c.startswith("WG_")]
    if any(o["n_stratified_draws_checked"] != n_b for o in strata):
        raise SystemExit("REFUSING: an Aegis 4d B record did not check its origin counts in every draw")
    tc, ev = same([o["realised_counts_tc_ev"]["cal"] for o in strata], "cal_d's realised origin counts")
    return {"n_b": n_b, "n_e": N_EVAL_4DB, "tc": tc, "ev": ev}


def floor_of(cell: dict[str, Any], core: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The item-level no-update floor of a 4d A cell: FPR_pre/target and TPR_pre. Every arm block holds a copy;
    refuses unless all copies are identical."""
    it = cell["item_level_D156"]
    copies = [(it[a]["floor_ratio_to_target"], it[a]["tpr_pre"]) for a in it]
    if any(c != copies[0] for c in copies):
        raise SystemExit(f"REFUSING: the arm blocks of repair_d156_{core}.json at L* hold different floors")
    return copies[0]


def d257_verdicts(A: dict[str, tuple[dict[str, Any], dict[str, Any]]]) -> dict[tuple[str, str], tuple[str, float | None]]:
    """D-257's paired floor test at L* (moment_contrasts_d156_r2.json, by_metric d117_layer, selected C, item level;
    D_arm = |log(FPR/target)| of the arm minus the floor's, within each draw). Per (core, 'frozen'): ('damage detected'
    or 'not detected', None) from the record's damage_detected, which must agree with the band of D_frozen lying wholly
    above 0. Per (core, arm) for FLOOR_ARMS: ('at floor' or 'above floor', X), X the upper end of the band of D_arm,
    which must agree with the record's verdict and its text 'at most X above floor'. A pair is used only if the moment
    record named for it is the file the contrasts read (same sha256), at 4d A's layer and input caches, with its frozen
    and FLOOR_ARMS blocks equal to the 4d A cell's exactly; the verdicts must cover exactly the Aegis pairs. R_same
    (quantile_thr_benign), the 4d B columns and the WildGuardMix pairs have no such test in any record (D-289)."""
    mc = read_result(RES / "moment_contrasts_d156_r2.json")
    out: dict[tuple[str, str], tuple[str, float | None]] = {}
    for entry in mc["by_metric"]["d117_layer"]["per_pair"].values():
        rel = entry["record"]
        m = re.fullmatch(r"results/moment_d156_(.+)_d117\.json", rel)
        if not m or m.group(1) not in A:
            raise SystemExit(f"REFUSING: moment contrast record {rel!r} maps to no released pair")
        core = m.group(1)
        rec_a, cell_a = A[core]
        mom = read_result(ROOT / rel)
        if _RESULT_READS[rel] != mc["input_files_sha256"].get(rel):
            raise SystemExit(f"REFUSING: {rel} is not the file moment_contrasts_d156_r2.json read")
        L = rec_a["config"]["selected_layer"]
        mcell = [c for c in mom["cells"] if c["layer"] == L and c["probe_set"] == "selected"]
        if (entry["d117_layer"] != L or len(mcell) != 1 or not rec_a["input_cache_sha256"]
                or mom["input_cache_sha256"] != rec_a["input_cache_sha256"]
                or any(mcell[0]["item_level_D156"][a] != cell_a["item_level_D156"][a] for a in ("frozen", *FLOOR_ARMS))):
            raise SystemExit(f"REFUSING: {rel} at layer {L} is not 4d A's cell for {core}")
        test = entry["selected|item_level_D156"]
        if test["judged"] is not True:
            raise SystemExit(f"REFUSING: the paired floor test for {core} is not judged")
        d_lo = test["paired_floor_test"]["frozen"]["arm_minus_floor"]["band95"][0]
        if test["damage_detected"] is not (d_lo > 0):
            raise SystemExit(f"REFUSING: {core}: damage_detected {test['damage_detected']} against band low {d_lo}")
        out[(core, "frozen")] = ("damage detected" if test["damage_detected"] else "not detected", None)
        for a in FLOOR_ARMS:
            t = test["paired_floor_test"][a]
            lo, hi = t["arm_minus_floor"]["band95"]
            at = t["verdict"] == "at floor"
            if not ((at and lo <= 0 <= hi and t["text"] == f"at most {hi:.3f} above floor")
                    or (t["verdict"] == "above floor" and lo > 0)):
                raise SystemExit(f"REFUSING: {core} {a}: verdict {t['verdict']!r} against band [{lo}, {hi}]")
            out[(core, a)] = (t["verdict"], hi)
    if {c for c, _ in out} != {c for c in A if not c.startswith("WG_")}:
        raise SystemExit("REFUSING: the paired floor test does not cover exactly the Aegis pairs, as the caption says")
    return out


def two_line(b: dict[str, Any], mark: str = "", prefix: str = "") -> tuple[str, str]:
    """mb(b) set on two lines, the mean (with its prefix and mark) above its band, so that many columns fit."""
    mean, band = mb(b).split(" ", 1)
    return prefix + mean + mark, band


def table_2() -> str:
    flagged = flagged_pairs()
    A = {core: part_a(core) for core in REL_PAIRS}
    B = part_b()
    verdict = d257_verdicts(A)
    n_a = same([A[c][0]["config"]["n_draws"] for c in REL_PAIRS], "4d A's n_draws")
    n_cal, n_ev, n_pos = same([population_a(A[c][0], c) for c in REL_PAIRS], "4d A's per-draw population")
    rows_f, rows_t, no_tpr = [], [], False
    for core in REL_PAIRS:
        rec_a, cell = A[core]
        stem_b, rec_b = B[core]
        L = rec_a["config"]["selected_layer"]
        if rec_b["config"]["layer"] != L or rec_b["config"]["selected_C"] != rec_a["config"]["selected_C"]:
            raise SystemExit(f"REFUSING: {stem_b}.json is not at 4d A's layer and C for {core}")
        it, sel = cell["item_level_D156"], rec_b["configurations"]["selected"]
        floor, tpr_pre = floor_of(cell, core)
        damage = verdict.get((core, "frozen"), ("", None))[0] == "damage detected"
        at = {a: verdict.get((core, a), ("", None))[0] == "at floor" for a in FLOOR_ARMS}
        # (FPR/target summary, TPR summary or None, marker), in column order
        cols = [(it["frozen"]["ratio_to_target"], it["frozen"].get("tpr_post"), MARK_DAMAGE if damage else ""),
                (it["quantile_thr_benign"]["ratio_to_target"], it["quantile_thr_benign"].get("tpr_post"), ""),
                (it["restandardise"]["ratio_to_target"], it["restandardise"].get("tpr_post"),
                 MARK_FLOOR if at["restandardise"] else ""),
                (sel["score|R_fresh|n=1000"]["ratio_to_target"], sel["score|R_fresh|n=1000"].get("tpr"), ""),
                (sel["score|traffic|pi=0.01|n=1000"]["ratio_to_target"], sel["score|traffic|pi=0.01|n=1000"].get("tpr"), ""),
                (sel["feature|traffic|pi=0.01|n=1000"]["ratio_to_target"], sel["feature|traffic|pi=0.01|n=1000"].get("tpr"), ""),
                (it["refit_oracle"]["ratio_to_target"], it["refit_oracle"].get("tpr_post"), MARK_FLOOR if at["refit_oracle"] else ""),
                (floor, tpr_pre, "")]
        means, bands, tprs = [], [], []
        for fpr, tpr, mark in cols:
            mean, band = two_line(fpr, mark)
            means.append(mean)
            bands.append(band)
            tprs.append("--" if tpr is None else num(tpr["mean"], "{:.3f}"))
            no_tpr |= tpr is None
        name = NICE[core] + (DAGGER if core in flagged else "")
        rows_f.append(f"{name} & {L} & " + " & ".join(means) + EOL + " & & " + " & ".join(bands) + EOL)
        rows_t.append(f"{name} & {L} & " + " & ".join(tprs) + EOL)
    # every Aegis restand. and refit cell is at floor, so the caption needs no 'above floor' wording for them
    if any(v != "at floor" for (_, arm), (v, _x) in verdict.items() if arm in FLOOR_ARMS):
        raise SystemExit("REFUSING: an Aegis restandardise or refit cell is above floor; the caption has no wording for it")
    # the largest X ('at most X above floor') over the marked cells of each floor-tested column, as the record prints it
    x_max = {a: max(x for (_, arm), (v, x) in verdict.items() if arm == a and v == "at floor") for a in FLOOR_ARMS}
    return ("\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\caption{Repairs of the operating "
            "point per update pair at the D-117 selected layer $L^*$ (selected $C$, item level), each labelled by the "
            "reference pool it needs; the column groups name what the defender must hold. (a) FPR/target, D-156 mean "
            "with its 95\\% band below; FPR/target $=1$ is on target. (b) TPR, D-156 mean only (no D-282 interval for "
            "these arms). Frozen: the pre-update threshold $T$ kept. Threshold (R$_\\mathrm{same}$): re-set at the "
            "1\\% upper tail of the post-update scores of the draw's "
            f"{n_cal} stored calibration negatives (cal$_d$). Restand.\\ (class-bal.): the post features standardised "
            "with the post model's mean and SD on the probe's own class-balanced training rows. Threshold "
            "(R$_\\mathrm{fresh}$): re-set at 1\\% FPR on the post scores of 1000 benign prompts held out of "
            "evaluation. Live traffic, a simulated pool of 1000 prompts at 1\\% harmful prevalence "
            "(Table~\\ref{tab:robustness-d156}): quantile transfer of $T$ from the pool's pre to its post scores "
            "(matched prevalence); restand.\\ (traffic), standardisation with the pool's post-feature mean and SD. "
            "Both restandardise columns keep the probe's $w$, $b$ and $T$. Refit oracle: the probe refitted on the "
            "labelled post-update training rows, with its own 1\\% threshold (not available to the defender). Floor: "
            "FPR$_\\mathrm{pre}$/target, the pre model at $T$ with no update; its TPR is TPR$_\\mathrm{pre}$. "
            f"Populations: frozen, R$_\\mathrm{{same}}$, restand.\\ (class-bal.), refit and the floor are 4d A over "
            f"{n_a} re-partitions, each evaluated on {n_ev} negatives and {n_pos} positives; the floor and the D-257 "
            "test are on this population. R$_\\mathrm{fresh}$ and the traffic columns are "
            + POPULATIONS.format(**origin_cal(B)) + "; 4d B stores no floor. Markers, from the D-257 paired "
            "test at $L^*$ (the per-draw difference between a column's $|\\log(\\mathrm{FPR}/\\mathrm{target})|$ and "
            f"the floor's), Aegis pairs only: {MARK_DAMAGE} on frozen, damage detected (the band of the difference "
            f"wholly above 0); {MARK_FLOOR} on restand.\\ (class-bal.) and refit, at floor: the test does not reject "
            "the floor (the band includes 0), reported as at most $X$ above floor in $|\\log|$ units, here at most "
            f"{num(x_max['restandardise'], '{:.3f}')} (restand., class-bal.) and {num(x_max['refit_oracle'], '{:.3f}')} "
            "(refit), the largest upper band ends over the marked cells. At floor is not `on target'. An unmarked "
            "Aegis frozen cell: damage not detected. R$_\\mathrm{same}$, the 4d B columns and the WildGuardMix rows "
            "carry no test. Damage itself is "
            "Table~\\ref{tab:damage-d156}'s measure. " + WGM_NOTE
            + (" --: no TPR in the record." if no_tpr else "")
            + (" $^\\dagger$: input verified by pin only, pending D-168." if d168_final() is None else "") + "}\n"
            "\\label{tab:repair-d156}\n\\label{tab:repair}\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrllllllll@{}}\n\\toprule\n"
            " & & & \\multicolumn{2}{c}{stored calibration data} & \\multicolumn{1}{c}{fresh benign prompts} & "
            "\\multicolumn{2}{c}{live traffic (1\\%)} & \\multicolumn{1}{c}{labels} &" + EOL +
            "\\cmidrule(lr){4-5}\\cmidrule(lr){6-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-9}\n"
            "update & $L^*$ & frozen & threshold & restand. & threshold & quantile & restand. & refit & floor" + EOL +
            " & & & (R$_\\mathrm{same}$) & (class-bal.) & (R$_\\mathrm{fresh}$) & transfer & (traffic) & oracle & (pre)" + EOL +
            "\\midrule\n\\multicolumn{10}{@{}l}{(a) FPR/target: D-156 mean, its 95\\% band below}" + EOL + "".join(rows_f) +
            "\\midrule\n\\multicolumn{10}{@{}l}{(b) TPR: D-156 mean}" + EOL + "".join(rows_t) +
            "\\bottomrule\n\\end{tabular*}\n\\end{table*}\n")


T3_TPR = RES / "table3_tpr_d282.json"           # D-282 for Table III (src/table3_tpr_d282.py), the 1% supplement
# the manifest job of queues/laneA_20261004a_tpr4a.tsv writes the record's sha256 here
T3_TPR_MANIFEST = ROOT / "logs" / "table3_tpr_d282_20261003.sha256"
BLAS_PIN = {v: "6" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}    # D-152 P6
LORA_STEMS = [f"depth_{r}" for runs in POP.LORA_ARMS.values() for r in runs]
# D-344 (3), D-349: the HH arm's rows are under check; marked in Table III with this sign and the caption's sentence
UNDER_CHECK_ARM = "safety (hh-harmless)"
MARK_UNDER_CHECK = "$^{\\S}$"
UNDER_CHECK_TEXT = "the HH arm's training conversations contained frozen evaluation items in later Human turns"
# D-357 (B)(4), D-363: Table IV's LoRA rows that pool the HH runs carry the same sign and sentence until D-349 is done
UNDER_CHECK_T4_ROWS = ("LoRA, all nine runs", "LoRA, safety (six runs)")

# D-401 (c), D-380: the HH rows ENTER the paper, each number with its D-363 sensitivity value beside it -- the same
# quantity recomputed with the D-363 removal set R applied, COPIED from the record below (rule 2), never computed here.
# The marks and their caption sentences STAY (D-401 (c)): removing them is a separate author decision.
HH_SENS = RES / "hh_sensitivity_d363.json"
HH_SENS_RUN_KIND = "hh_sensitivity_d363_summary"
# the bytes D-380 names: its line in the pull manifest that entry names, whose sha256 starts with the prefix it prints
HH_SENS_PULL_MANIFEST = ROOT / "logs" / "pulls" / "pull_20261007a.sha256"
HH_SENS_SHA256_PREFIX = "030f6c4ccc639231"
HH_SENS_CONFIG_HASH = "325746e55110258b1f4473abf7606803434b4bc53c4830a62658675fcf296572"   # of the record D-380 names
HH_SENS_REMOVAL = {"probe_train": 338, "threshold_calib": 158, "eval": 330}     # D-380: R = 826 items, these sizes
HH_SENS_N_ITEMS = sum(HH_SENS_REMOVAL.values())                                 # 826, as D-380 records it
# LAYOUT -- the implementer's choice, recorded for the author (D-318 (3) and D-324 (4) are the precedents); changing
# this one constant is the whole change: 'line' sets the beside values on a SECOND LINE under the row, one per column
# in parentheses, as Table II sets each band under its mean, so no column grows and neither table widens; 'inline'
# puts ' (value)' after each number instead, which is wider.
BESIDE_LAYOUT = "line"
BESIDE_WHERE = {"line": "under", "inline": "after"}            # how the caption words the position, per layout
# the one caption sentence both tables gain (D-401 (c)); {n} is the record's own removal-set size
BESIDE_SENTENCE = ("The value in parentheses {where} each number of the {rows} is that number recomputed with the "
                   "D-363 removal set ({n} contaminated items) removed from every split of every run (the D-380 "
                   "sensitivity, D-401 (c)); the marks stay until the paper's text says what that value means.")
# Table III's columns, in printed order, as the record's table_iii cells key them, each with its printed format
T3_BESIDE = (("frozen_mean_abs_log", "{:.3f}"), ("restandardise_mean_abs_log", "{:.3f}"),
             ("worst_fpr_to_target", "{:.2f}"), ("delta_tpr_at_lstar", "{:+.3f}"),
             ("auroc_drop_mean_over_layers", "{:.4f}"), ("d_norm_at_lstar", "{:.2f}"), ("cos_at_lstar", "{:+.4f}"))
# Table IV's six printed cells, in order, as the record's table_iv_lora cells key them
# D-432 (4), D-438: Table IV's lower panel, the published staleness rule's count per population of record at selected C
# (the released pairs, the nine LoRA runs, the two controls beside), from the record D-432 (4) names. It was emitted by
# src/tables.py into paper/tables.tex, which paper/main.tex does not \input; D-438 moves it here, same values and wording.
BIGDROP = ROOT / "docs" / "reports_d419" / "item3_numbers.json"     # written by docs/reports_d419/item3.py (write_json)
BIGDROP_MANIFEST = ROOT / "manifests" / "item3_numbers_d438.sha256"  # sha256sum format; the bytes D-438 reads
# Table IV group -> (item3 auroc_share.selected key, row label printed: src/tables.py's wording)
BIGDROP_ROWS = {"released update pairs": ("phase1_released_pairs", "released update pairs"),
                "LoRA, all nine runs": ("lora_all_nine_runs", "LoRA (alpaca, HH-RLHF, PKU-SafeRLHF)")}
BIGDROP_CONTROLS = (("beside_v01_v02_dense", "Mistral v0.1 $\\to$ v0.2"), ("beside_gpu_control", "GPU change"))
BIGDROP_CAPTION = (" Lower panel (D-432 (4)): cells and the cells stale by that rule (stale if $\\Delta\\mathrm{AUC}\\le-0.05$ "
                   "or post-update AUC $<0.70$) per population at selected $C$, with the two controls beside.")
T4_BESIDE = tuple((det, q) for det in ("gaussian", "direct", "auc_rule") for q in ("sensitivity", "specificity"))
_HH_SENS: list[dict[str, Any]] = []


def hh_sensitivity() -> dict[str, Any]:
    """The D-380 record holding every HH-affected Table III and Table IV number recomputed with the D-363 removal set
    applied (D-401 (c)). Refuses unless it is on disk, is the bytes D-380 names (its line in the pull manifest that
    entry names, and the sha256 prefix it prints), is the registered D-363 summary, carries that record's config_hash
    and the removal set D-380 recorded (826 items: probe_train 338, threshold_calib 158, eval 330), and holds both
    table blocks. Read through read_result, so the header binds its sha256 as it binds every other source's."""
    if not _HH_SENS:
        if not HH_SENS.is_file():
            raise SystemExit(f"REFUSING: {HH_SENS.name} is not on disk; D-401 (c) prints its D-380 sensitivity value "
                             "beside every HH-affected number of Tables III and IV")
        rec = read_result(HH_SENS)
        got = _RESULT_READS[_repo_rel(HH_SENS)]
        listed = ({ln.split()[1]: ln.split()[0] for ln in HH_SENS_PULL_MANIFEST.read_text(encoding="utf-8").splitlines()
                   if ln.strip()} if HH_SENS_PULL_MANIFEST.is_file() else {})
        if listed.get(f"results/{HH_SENS.name}") != got or not got.startswith(HH_SENS_SHA256_PREFIX):
            raise SystemExit(f"REFUSING: {HH_SENS.name} is not the bytes D-380 names ({HH_SENS_SHA256_PREFIX}..., its "
                             f"line in {HH_SENS_PULL_MANIFEST.name})")
        if rec.get("run_kind") != HH_SENS_RUN_KIND or rec.get("registered_parameters") is not True:
            raise SystemExit(f"REFUSING: {HH_SENS.name} is not the registered {HH_SENS_RUN_KIND} record")
        if rec.get("config_hash") != HH_SENS_CONFIG_HASH:
            raise SystemExit(f"REFUSING: {HH_SENS.name}'s config_hash {rec.get('config_hash')} is not the one of the "
                             f"record D-380 names ({HH_SENS_CONFIG_HASH})")
        r = rec.get("removal_set") or {}
        if r.get("n_items") != HH_SENS_REMOVAL or r.get("n_items_total") != HH_SENS_N_ITEMS:
            raise SystemExit(f"REFUSING: {HH_SENS.name}'s removal set is {r.get('n_items')} (total "
                             f"{r.get('n_items_total')}), not the {HH_SENS_REMOVAL} (total {HH_SENS_N_ITEMS}) D-380 "
                             "records")
        for blk in ("table_iii", "table_iv_lora"):
            if not rec.get(blk):
                raise SystemExit(f"REFUSING: {HH_SENS.name} holds no {blk} block, which D-401 (c) reads")
        _HH_SENS.append(rec)
    return _HH_SENS[0]


def beside(cell: dict[str, Any] | None, printed: float, f: str, what: str) -> str:
    """One D-380 beside value, formatted as its column prints its own number: the cell's sensitivity value. Refuses if
    the record holds no such number (D-401 (c): none is silently omitted) and if the cell's value OF RECORD is not the
    number printed beside it, which would put the sensitivity value beside another quantity."""
    sens, rec = (cell or {}).get("sensitivity") or {}, (cell or {}).get("record") or {}
    if sens.get("value") is None or "value" not in rec:
        raise SystemExit(f"REFUSING: {HH_SENS.name} holds no sensitivity value for {what}; D-401 (c) prints one beside "
                         "every HH-affected number")
    if rec["value"] != printed:
        raise SystemExit(f"REFUSING: {HH_SENS.name}'s value of record for {what} is {rec['value']}, not the {printed} "
                         "the table prints; its sensitivity value would sit beside another quantity")
    return num(sens["value"], f)


def beside_t3(run: str, printed: list[float]) -> list[str]:
    """Table III's seven beside values for one HH row, in printed column order (D-401 (c), D-380)."""
    row = {r["run"]: r for r in hh_sensitivity()["table_iii"]["rows"]}.get(run) or {}
    if row.get("arm") != UNDER_CHECK_ARM or row.get("under_check_arm") is not True:
        raise SystemExit(f"REFUSING: {HH_SENS.name} holds no table_iii row for {run} in the {UNDER_CHECK_ARM} arm")
    cells = row["cells"].get("item_level_D156") or {}
    return [beside(cells.get(k), p, f, f"{run} {k}") for (k, f), p in zip(T3_BESIDE, printed)]


def beside_t4(name: str, printed: list[float]) -> list[str]:
    """Table IV's six beside values for one LoRA row that pools the HH runs (D-401 (c), D-380)."""
    cells = hh_sensitivity()["table_iv_lora"]["cells"].get(name) or {}
    if not cells:
        raise SystemExit(f"REFUSING: {HH_SENS.name}'s table_iv_lora holds no cell {name!r}, which D-401 (c) reads")
    return [beside((cells.get(det) or {}).get(q), p, "{:.3f}", f"{name} {det} {q}")
            for (det, q), p in zip(T4_BESIDE, printed)]


def with_beside(lead: str, cells: list[str], besides: list[str], n_lead: int) -> str:
    """A row and its D-380 beside values, in the BESIDE_LAYOUT presentation: 'line' sets them on a second line under
    the row, one per column in parentheses, the row's n_lead label cells left empty (Table II's two-line style,
    D-318 (3), D-324 (4)); 'inline' puts ' (value)' after each number. `lead` ends with ' & '."""
    if len(besides) != len(cells):
        raise SystemExit(f"REFUSING: {len(besides)} beside values for {len(cells)} printed cells (D-401 (c))")
    if BESIDE_LAYOUT not in BESIDE_WHERE:
        raise SystemExit(f"REFUSING: BESIDE_LAYOUT {BESIDE_LAYOUT!r} is not one of {sorted(BESIDE_WHERE)}")
    if BESIDE_LAYOUT == "inline":
        return lead + " & ".join(f"{c} ({b})" for c, b in zip(cells, besides)) + EOL
    return lead + " & ".join(cells) + EOL + " &" * n_lead + " " + " & ".join(f"({b})" for b in besides) + EOL


def t3_tpr() -> dict[str, Any]:
    """The D-282 supplement for Table III: a registered 1% record whose rows are exactly the nine LoRA runs, run under
    the 6-thread BLAS pin, and whose bytes are the ones its line in T3_TPR_MANIFEST names."""
    if not T3_TPR.is_file():
        raise SystemExit(f"REFUSING: {T3_TPR.name} is not on disk (not yet run); Table III's delta TPR column reads it")
    t3 = read_result(T3_TPR)
    if (t3.get("registered_parameters") is not True or t3["config"]["target_fpr"] != 0.01
            or t3["config"]["runs"] != LORA_STEMS or sorted(t3["rows"]) != sorted(LORA_STEMS)):
        raise SystemExit(f"REFUSING: {T3_TPR.name} is not a registered 1% supplement covering the nine LoRA runs")
    if t3.get("blas_threads") != BLAS_PIN:
        raise SystemExit(f"REFUSING: {T3_TPR.name} was not run under the 6-thread BLAS pin (D-152 P6)")
    listed = ({ln.split()[1]: ln.split()[0] for ln in T3_TPR_MANIFEST.read_text(encoding="utf-8").splitlines() if ln.strip()}
              if T3_TPR_MANIFEST.is_file() else {})
    if listed.get(f"results/{T3_TPR.name}") != _RESULT_READS.get(_repo_rel(T3_TPR)):
        raise SystemExit(f"REFUSING: {T3_TPR.name} is not the bytes its line in {T3_TPR_MANIFEST.name} names")
    return t3


def under_check_rows(t3: dict[str, Any]) -> set[str]:
    """D-344 (3), D-349: the supplement's rows that carry under_check, which must be exactly the HH arm's rows, as the
    record's own under_check list names them, each with the record's status text, which must state the caption's
    sentence (UNDER_CHECK_TEXT)."""
    hh = [f"depth_{r}" for r in POP.LORA_ARMS[UNDER_CHECK_ARM]]
    flagged = [s for s in LORA_STEMS if t3["rows"][s].get("under_check")]
    uc = t3.get("under_check") or {}
    if (flagged != hh or uc.get("runs") != hh or UNDER_CHECK_TEXT not in str(uc.get("status"))
            or any(t3["rows"][s]["under_check"].get("status") != uc.get("status") for s in flagged)):
        raise SystemExit(f"REFUSING: {T3_TPR.name} flags {flagged} under check, not the HH arm's rows {hh}, or its status "
                         f"text is not the caption's (D-344 (3), D-349)")
    return set(flagged)


def delta_tpr_at_lstar(t3: dict[str, Any], run: str, dep_doc: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Table III's delta TPR for one run: the supplement's selected-C block at the headline layer, used only if the
    supplement names the bytes of the depth record read here (anchor.depth_records_sha256), its L* is the record's
    selected layer and headline layer, and its frozen value, D-156 mean and band equal the record's pair_summary
    headline delta TPR exactly. Returns (the delta_tpr block, L*)."""
    rel = f"results/depth_d156_{run}.json"
    row = t3["rows"][f"depth_{run}"]
    head = dep_doc["pair_summary"]["selected"]["item_level_D156"]["headline_selected_layer"]
    L = dep_doc["config"]["selected_layer"]
    if (row["depth_record"] != rel or t3["anchor"]["depth_records_sha256"].get(rel) != _RESULT_READS.get(rel)
            or row["headline_layer"] != L or head["layer"] != L
            or row["probe_sets"]["selected"]["probe_C"] != dep_doc["config"]["selected_C"]):
        raise SystemExit(f"REFUSING: {T3_TPR.name}'s row for {run} is not anchored on {rel} at its L*")
    blk = row["probe_sets"]["selected"]["layers"].get(str(L))
    if blk is None or blk["is_headline"] is not True:
        raise SystemExit(f"REFUSING: {T3_TPR.name} holds no headline block for {run} at L{L}")
    dt, want = blk["delta_tpr"], head["delta_tpr"]
    if ((dt["frozen"], dt["mean_D156"], dt["band95_D156"]) != (want["frozen"], want["mean"], want["band95"])
            or dt["combined"]["ci95"] is None or dt["combined"]["n_nonfinite"] != 0):
        raise SystemExit(f"REFUSING: {T3_TPR.name}'s delta TPR for {run} is not {rel}'s headline, or has no interval")
    return dt, L


def table_3() -> str:
    geo = {c["run"]: c["geometry"]["neg"] for c in read_result(RES / "intent_selected_d165.json")["cells"]
           if c["is_selected_layer"]}
    t3 = t3_tpr()
    flagged = under_check_rows(t3)
    rows, lstar, beside_rows = [], [], []
    for arm, runs in POP.LORA_ARMS.items():
        for i, run in enumerate(runs):
            rep = read_result(RES / f"repair_d156_{run}.json")["pair_summary_mean_abs_log_over_layers"]["selected"]["item_level_D156"]
            dep_doc = read_result(RES / f"depth_d156_{run}.json")
            dep = dep_doc["pair_summary"]["selected"]
            sm = dep["item_level_D156"]["sweep_maximum"]["ratio_to_target"]
            au = dep["auroc_drop_frozen"]["mean_over_layers"]
            dt, L = delta_tpr_at_lstar(t3, run, dep_doc)
            lstar.append(L)
            marked = f"depth_{run}" in flagged
            dose = run.rsplit("_s", 1)[1] + (MARK_UNDER_CHECK if marked else "")
            g = geo[run]
            cells = [mb(rep["frozen"], "{:.3f}"), mb(rep["restandardise"], "{:.3f}"),
                     f"{mb(sm['ratio_to_target'])} (L{sm['layer']}; {sm['null_no_update']['value']:.2f})",
                     ci(dt["mean_D156"], dt["combined"]["ci95"], "{:+.3f}"), ci(au["point"], au["ci95"], "{:.4f}"),
                     f"{g['d_norm']['mean']:.2f}", num(g["cos"]["mean"], "{:+.4f}")]
            lead = f"{ARMS[arm] if i == 0 else ''} & {dose} & "
            if marked:          # D-401 (c): the marked rows are the HH arm's, each number with its D-380 value beside
                printed = [rep["frozen"]["mean"], rep["restandardise"]["mean"], sm["ratio_to_target"]["mean"],
                           dt["mean_D156"], au["point"], g["d_norm"]["mean"], g["cos"]["mean"]]
                rows.append(with_beside(lead, cells, beside_t3(run, printed), 2))
                beside_rows.append(run)
            else:
                rows.append(lead + " & ".join(cells) + EOL)
        rows.append("\\addlinespace\n")
    if sorted(beside_rows) != sorted(POP.LORA_ARMS[UNDER_CHECK_ARM]):
        raise SystemExit(f"REFUSING: Table III's rows with a D-380 value beside are {beside_rows}, not the "
                         f"{UNDER_CHECK_ARM} arm's {POP.LORA_ARMS[UNDER_CHECK_ARM]} (D-401 (c))")
    L = same(lstar, "Table III's selected layer L*")
    n_items = hh_sensitivity()["removal_set"]["n_items_total"]
    return ("\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\caption{Update intent at matched dose (4a), selected $C$, item level. "
            "frozen and restand.: mean $|\\log(\\mathrm{FPR}/\\mathrm{target})|$ over the layer sweep, D-156 mean [95\\% band]; "
            "worst FPR/target: the sweep maximum of the per-layer mean ratio to target, D-156 mean [95\\% band] (its layer; "
            "the no-update value, the same maximum of FPR$_\\mathrm{pre}$/target); $\\Delta$TPR (post $-$ pre) at the "
            f"frozen threshold at the selected layer $L^*={L}$: D-156 mean with the combined 95\\% interval (re-partitions "
            "and an item bootstrap over positives, D-282); AUROC drop: mean over layers with its "
            "item-bootstrap interval; $\\lVert d\\rVert$ and signed $\\cos\\theta$ at the selected layer, seed means on the "
            f"negatives. {MARK_UNDER_CHECK} Under check (D-349): {UNDER_CHECK_TEXT}. "
            + BESIDE_SENTENCE.format(where=BESIDE_WHERE[BESIDE_LAYOUT], rows="HH rows", n=n_items) + "}\n"
            "\\label{tab:intent-d156}\n\\label{tab:intent}\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrlllllrr@{}}\n\\toprule\n"
            "corpus & steps & frozen & restand. & worst FPR/target & $\\Delta$TPR ($L^*$) & mean AUROC drop & "
            "$\\lVert d\\rVert$ & $\\cos\\theta$" + EOL +
            "\\midrule\n" + "".join(rows).removesuffix("\\addlinespace\n") + "\\bottomrule\n\\end{tabular*}\n\\end{table*}\n")


def bigdrop_record() -> tuple[dict[str, Any], str]:
    """docs/reports_d419/item3_numbers.json and its sha256, read only if its bytes are the ones
    manifests/item3_numbers_d438.sha256 lists (D-438); refuses on a missing manifest, a missing line or a mismatch."""
    rel, man = BIGDROP.relative_to(ROOT).as_posix(), BIGDROP_MANIFEST.relative_to(ROOT).as_posix()
    if not BIGDROP_MANIFEST.is_file():
        raise SystemExit(f"REFUSING: {man} is missing; Table IV's lower panel reads {rel} only against it (D-438)")
    listed = dict(reversed(ln.split("  ", 1)) for ln in BIGDROP_MANIFEST.read_text(encoding="utf-8").splitlines()
                  if ln.strip())
    b = BIGDROP.read_bytes()
    got = hashlib.sha256(b).hexdigest()
    if listed.get(rel) != got:
        raise SystemExit(f"REFUSING: {rel} has sha256 {got}, but {man} lists {listed.get(rel) or 'no line for it'}; "
                         "Table IV's lower panel reads only the bytes the manifest names (D-438)")
    return json.loads(b.decode("utf-8")), got


def bigdrop_cells(sel: dict[str, Any], key: str) -> tuple[int, int]:
    """(cells, stale by rule) of one population at selected C. The column is the full staleness rule
    (beside_n_full_duan_rule); D-432 (4) records that the dAUROC <= -0.05 line gives the same count on every set, and
    this refuses if the record ever says otherwise (the column would then need the author's choice)."""
    v = sel[key]
    if v["beside_n_full_duan_rule"] != v["n_d_auroc_le_minus_0_05"]:
        raise SystemExit(f"REFUSING: {BIGDROP.name} {key}: the full rule and the -0.05 line disagree (D-432 (4))")
    return v["n_cells"], v["beside_n_full_duan_rule"]


def bigdrop_panel(groups: list[tuple[str, dict[str, Any]]]) -> str:
    """Table IV's lower panel (D-432 (4), D-438), src/tables.py's rows: cells and stale by rule for the released pairs
    and the nine LoRA runs, then the two controls beside. Each population must be the pairs of the Table IV group it
    sits under (refused otherwise). Emitted as ONE physical line, and Table IV's two labels share one line, so the
    lines of paper/tables_d156.tex cited by number in src/recalnoise_d433.py's METHOD_TEXT (:89, :179, :223, recorded
    in its results; tests/test_recalnoise_d433.py pins them) keep their content."""
    sel = bigdrop_record()[0]["auroc_share"]["selected"]
    by_name = dict(groups)
    rows = ["population & cells & stale by rule & & & &"]
    for name, (key, shown) in BIGDROP_ROWS.items():
        if name not in by_name or set(sel[key]["per_pair_n_cells"]) != set(by_name[name]["pairs"]):
            raise SystemExit(f"REFUSING: {BIGDROP.name} {key} is not the pairs of Table IV's group {name!r} (D-438)")
        cells, stale = bigdrop_cells(sel, key)
        rows.append(f"{shown} & {cells} & {stale} & & & &")
    ctl = [bigdrop_cells(sel, k) for k, _ in BIGDROP_CONTROLS]
    rows.append("\\quad controls, beside: " + "; ".join(n for _, n in BIGDROP_CONTROLS)
                + " & " + "; ".join(str(c) for c, _ in ctl) + " & " + "; ".join(str(s) for _, s in ctl) + " & & & &")
    return "\\midrule " + " ".join(r + " \\\\" for r in rows) + "\n"


def table_4() -> str:
    rel = read_result(RES / "healthcheck_d156_summary.json")["rows"]["selected|item_level_D156"]
    lora = read_result(RES / "healthcheck_d156_summary_lora.json")["rows"]["selected|item_level_D156"]
    groups = [("released update pairs", rel["released_pairs"]), ("released, unflagged pairs", rel["unflagged_pairs"]),
              ("LoRA, all nine runs", lora["lora_all"]), ("LoRA, general (alpaca)", lora["lora_general"]),
              ("LoRA, safety (six runs)", lora["lora_safety"])]
    if d168_final() is not None:                 # D-318 (7): the row is dropped once D-168 is final
        groups = [g for g in groups if g[0] != "released, unflagged pairs"]
    flagged = flagged_pairs()
    hh = set(POP.LORA_ARMS[UNDER_CHECK_ARM])
    rows, under = [], []
    for name, g in groups:
        cells, printed = [], []
        for det in ("gaussian", "direct", "auc_rule"):
            x = g[f"{det}|two_sided_2x"]["pooled_from_mean_counts"]
            cells += [ci(x["sensitivity"], x["sensitivity_ci"]["ci95"], "{:.3f}"),
                      ci(x["specificity"], x["specificity_ci"]["ci95"], "{:.3f}")]
            printed += [x["sensitivity"], x["specificity"]]
        pools_hh = bool(hh & set(g["pairs"]))              # D-357 (B)(4), D-363: the rows that pool the HH runs
        under += [name] if pools_hh else []
        mark = (DAGGER if flagged & set(g["pairs"]) else "") + ("$^\\ddagger$" if name == "LoRA, general (alpaca)" else "") \
            + (MARK_UNDER_CHECK if pools_hh else "")
        lead = f"{name}{mark} ({g['n_pairs']}) & "
        # D-401 (c): the rows that pool the HH runs carry their D-380 sensitivity value beside each number
        rows.append(with_beside(lead, cells, beside_t4(name, printed), 1) if pools_hh
                    else lead + " & ".join(cells) + EOL)
    if under != list(UNDER_CHECK_T4_ROWS):
        raise SystemExit(f"REFUSING: Table IV's rows that pool the HH runs are {under}, not {list(UNDER_CHECK_T4_ROWS)} "
                         "(D-357 (B)(4), D-363)")
    return ("\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\caption{Label-free detection of operating-point damage (4c): "
            "sensitivity and specificity for a two-fold change, selected $C$, item level, pooled from mean counts over "
            "the re-partitions with pair-cluster 95\\% intervals. Gaussian: the primary health ratio (D-264); direct: "
            "secondary; AUC rule: the published staleness rule applied unchanged." + BIGDROP_CAPTION
            + (" $^\\dagger$: the group includes pairs whose input is verified by pin only, pending D-168."
               if d168_final() is None else "") + " $^\\ddagger$: about 1.7 damaged cells per draw, so the "
            "sensitivity rests on almost nothing and is not read (D-295)."
            f" {MARK_UNDER_CHECK} Under check (D-349): {UNDER_CHECK_TEXT}. "
            + BESIDE_SENTENCE.format(where=BESIDE_WHERE[BESIDE_LAYOUT], n=hh_sensitivity()["removal_set"]["n_items_total"],
                                     rows="LoRA rows that pool the HH runs")
            + "}\n\\label{tab:invisible-d156}\\label{tab:invisible}\n"     # one line: see bigdrop_panel
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lllllll@{}}\n\\toprule\n"
            " & \\multicolumn{2}{c}{Gaussian (primary)} & \\multicolumn{2}{c}{direct} & \\multicolumn{2}{c}{AUC rule}" + EOL +
            "population (cells' pairs) & sens. & spec. & sens. & spec. & sens. & spec." + EOL + "\\midrule\n" + "".join(rows) +
            bigdrop_panel(groups) + "\\bottomrule\n\\end{tabular*}\n\\end{table*}\n")


DRIFT_POST = (("0.05", "5\\%"), ("0.25", "25\\%"))             # the post-window prevalences shown, as keyed
# poisoning columns: (arm, target, header), grouped as in Table II: stored data, fresh prompts, live traffic
POISON_COLS = (("score", "R_same", "R$_\\mathrm{same}$ re-set"), ("score", "R_fresh", "R$_\\mathrm{fresh}$ re-set"),
               ("score", "traffic_1pct", "quantile transfer, traffic"), ("feature", "traffic_1pct", "restand., traffic"))
# pool-size columns: (key template, header), with the poisoning panel's headers and order
POOL_COLS = (("score|R_fresh|n={n}", POISON_COLS[1][2]), ("score|traffic|pi=0.01|n={n}", POISON_COLS[2][2]),
             ("feature|traffic|pi=0.01|n={n}", POISON_COLS[3][2]))
ATTACKERS = ("A0", "A1")


def poison_n(stem: str, rec: dict[str, Any]) -> int:
    """The one pool size n at which a 4d B record ran poisoning, read from its configuration keys
    '<arm>|poison|<target>|<A>|rho=<rho>|m=<m>': each (arm, target, A, rho) must occur with exactly one m, and every m
    must equal round(rho * n) for the single n the record names for its targets (config.default_pool for
    traffic_1pct; predictions.P4_silencing n_b for R_same and R_fresh). Refuses otherwise."""
    seen: dict[tuple[str, ...], tuple[float, int]] = {}
    for k in rec["configurations"]["selected"]:
        p = k.split("|")
        if p[1:2] == ["poison"]:
            if len(p) != 6 or not (p[4].startswith("rho=") and p[5].startswith("m=")) or tuple(p[:5]) in seen:
                raise SystemExit(f"REFUSING: {stem}.json: poison key {k!r} is malformed or repeats its rho")
            seen[tuple(p[:5])] = (float(p[4][len("rho="):]), int(p[5][len("m="):]))
    n_of = {"traffic_1pct": rec["config"]["default_pool"],
            **{t: rec["predictions"]["P4_silencing"][t]["n_b"] for t in ("R_same", "R_fresh")}}
    if (len(set(n_of.values())) != 1 or not seen
            or any(m != int(round(rho * n_of[key[2]])) for key, (rho, m) in seen.items())):
        raise SystemExit(f"REFUSING: {stem}.json's poisoned configurations are not all at one pool size")
    return n_of["traffic_1pct"]


def min_rho_cell(stem: str, rec: dict[str, Any], arm: str, target: str, att: str, top: float) -> str:
    """'rho (m)': the stored minimum rho for a 10-point TPR loss and the m of the poisoned configuration at that rho
    (copied from its key); '>' and the searched maximum when the stored minimum is null. Refuses unless the record
    searched up to `top`, the value the caption prints."""
    mr = rec["min_rho_for_10_point_tpr_loss"][f"selected|{arm}|{target}|{att}"]
    if mr["searched_up_to"] != top:
        raise SystemExit(f"REFUSING: {stem}.json searched {arm}|{target}|{att} up to {mr['searched_up_to']}, not {top}")
    if mr["min_rho"] is None:
        return f"$>${mr['searched_up_to']:g}"
    prefix = f"{arm}|poison|{target}|{att}|rho={mr['min_rho']}|m="
    keys = [k for k in rec["configurations"]["selected"] if k.startswith(prefix)]
    if len(keys) != 1:
        raise SystemExit(f"REFUSING: {stem}.json holds {len(keys)} configurations '{prefix}*'")
    return f"{mr['min_rho']:g} ({keys[0][len(prefix):]})"


def table_robustness() -> str:
    flagged = flagged_pairs()
    B = part_b()
    cfg = [B[c][1]["config"] for c in REL_PAIRS]
    n_rf = same([B[c][1]["predictions"]["P4_silencing"]["R_fresh"]["n_b"] for c in REL_PAIRS], "R_fresh's size")
    sizes = same([c["pool_sizes"] for c in cfg], "4d B's pool_sizes")
    rhos = same([c["rhos"] for c in cfg], "4d B's rho grid")
    same([c["tpr_loss"] for c in cfg] + [0.1], "the TPR loss (the caption's 10 points)")
    target = same([c["target_fpr"] for c in cfg] + [0.01], "the target FPR")
    n_p = same([poison_n(*B[c]) for c in REL_PAIRS], "the poisoned pool size")
    n_w = same([c["default_pool"] for c in cfg], "4d B's default pool (traffic, drift windows)")
    n_h = int(round(0.01 * n_w))                 # the 1% pool's harmful count, poolstudy_d156's rule round(pi * n)
    floor = 1 / N_EVAL_4DB / target              # FPR floored at 1 of N_EVAL_4DB negatives per seed: the lowest ratio
    top = rhos[-1]

    def fpr_cell(b: dict[str, Any]) -> tuple[str, str]:
        """two_line, with AT_MOST before a mean that prints as the floor (it can be no lower)."""
        return two_line(b, prefix=AT_MOST if num(b["mean"], "{:.2f}") == num(floor, "{:.2f}") else "")

    rows_a, rows_b, rows_c = [], [], []
    for core in REL_PAIRS:
        stem, rec = B[core]
        sel = rec["configurations"]["selected"]
        name = NICE[core] + (DAGGER if core in flagged else "")
        top_a = []
        for pi, _ in DRIFT_POST:                 # few columns here, so mean and band share one line
            d = sel[f"score|drift|pre=0.01|post={pi}"]
            top_a += [" ".join(fpr_cell(d["ratio_to_target"])), num(d["tpr"]["mean"], "{:.3f}")]
        rows_a.append(f"{name} & " + " & ".join(top_a) + EOL)
        cells_b = [min_rho_cell(stem, rec, arm, tgt, att, top) for arm, tgt, _ in POISON_COLS for att in ATTACKERS]
        rows_b.append(f"{name} & " + " & ".join(cells_b) + EOL)
        top_c, low_c = [], []
        for key, _ in POOL_COLS:
            for n in sizes:
                mean, band = fpr_cell(sel[key.format(n=n)]["ratio_to_target"])
                top_c.append(mean)
                low_c.append(band)
        rows_c.append(f"{name} & " + " & ".join(top_c) + EOL + " & " + " & ".join(low_c) + EOL)
    others = [n for n in sizes if n != n_p]
    fl = num(floor, "{:.2f}")
    k_c = 1 + len(sizes) * len(POOL_COLS)
    return ("\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\caption{Exposure of the "
            "label-free repairs to prevalence drift and poisoning, and their pool-size sensitivity (selected $C$, "
            f"item level, $L^*$ as in Table~\\ref{{tab:repair-d156}}): "
            + POPULATIONS.format(**origin_cal(B)) + ". " + WGM_NOTE + " Traffic is simulated: a pool of "
            f"{n_w} prompts at harmful prevalence $\\pi$ takes its benign prompts from cal$_d$'s negatives and its "
            f"harmful ones from the threshold-calibration harmful items (at $\\pi=1\\%$: {n_w - n_h} benign, {n_h} "
            "harmful). FPR/target $=1$ is on target. Each seed's FPR is floored at 1 of the "
            f"{N_EVAL_4DB} evaluation negatives, so FPR/target cannot fall below {fl}; {AT_MOST}{fl} marks a mean "
            "that prints as the floor. " + TPR_NOTE + " (a) Drift: quantile transfer with the pre window at 1\\% "
            f"and the post window at 5\\% or 25\\% ({n_w} prompts each); FPR/target, D-156 mean [95\\% band], and "
            "TPR. (b) Poisoning: the smallest $\\rho$ of the grid $\\{"
            + ", ".join(f"{r:g}" for r in rhos) + "\\}$ at which the mean TPR falls 10 points below the same arm's "
            "unpoisoned TPR, with $m$ in parentheses: $m$ harmful items added to an $n$-item pool ($m=\\rho n$, "
            "rounded); A0 adds random harmful items, A1 those with the highest post-update probe score (mean over "
            "seeds), none already in the pool. R$_\\mathrm{same}$ re-set (on 4d B's draws): the threshold re-set at "
            f"1\\% FPR on the post scores of cal$_d$'s {n_p} negatives plus the injected items; R$_\\mathrm{{fresh}}$ "
            f"re-set: the same on the {n_rf} fresh benign prompts; quantile transfer, traffic: from the clean pre "
            "window to the poisoned 1\\% pool; restand., traffic: restandardisation on the poisoned pool. "
            f"$>${top:g}: no such loss up to {top:g}. Poisoning was run at $n={n_p}$ only; $m$ at $n="
            + "$ and $n=".join(str(n) for n in others) + "$ is not in the records. (c) Pool size: FPR/target with a "
            f"reference or 1\\% traffic pool of $n$ prompts (R$_\\mathrm{{fresh}}$: a nested subset of $n$ of its "
            f"{n_rf}), D-156 mean with its 95\\% band below."
            + (" $^\\dagger$: input verified by pin only, pending D-168." if d168_final() is None else "") + "}\n"
            "\\label{tab:robustness-d156}\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lllll@{}}\n\\toprule\n"
            "(a) & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{drift, post window {lab}}}" for _, lab in DRIFT_POST) + EOL +
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            "update & FPR/target & TPR & FPR/target & TPR" + EOL + "\\midrule\n" + "".join(rows_a) +
            "\\bottomrule\n\\end{tabular*}\n\n\\medskip\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}" + "l" * (1 + 2 * len(POISON_COLS)) + "@{}}\n"
            "\\toprule\n(b) & \\multicolumn{" + str(2 * len(POISON_COLS)) + "}{c}{poisoning: minimum $\\rho$ ($m$) "
            "for a 10-point TPR loss}" + EOL + "\\cmidrule(lr){2-" + str(1 + 2 * len(POISON_COLS)) + "}\n"
            " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for _, _, lab in POISON_COLS) + EOL +
            "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(POISON_COLS))) + "\n"
            "update & " + " & ".join(ATTACKERS * len(POISON_COLS)) + EOL + "\\midrule\n" + "".join(rows_b) +
            "\\bottomrule\n\\end{tabular*}\n\n\\medskip\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}" + "l" * k_c + "@{}}\n\\toprule\n(c) & "
            + " & ".join(f"\\multicolumn{{{len(sizes)}}}{{c}}{{{lab}}}" for _, lab in POOL_COLS) + EOL +
            "".join(f"\\cmidrule(lr){{{2 + len(sizes) * i}-{1 + len(sizes) * (i + 1)}}}" for i in range(len(POOL_COLS)))
            + "\nupdate & " + " & ".join(f"$n={n}$" for n in sizes * len(POOL_COLS)) + EOL + "\\midrule\n"
            + "".join(rows_c) + "\\bottomrule\n\\end{tabular*}\n\\end{table*}\n")


def main() -> int:
    ensure_utf8_stdout()
    body = table_1() + "\n" + table_2() + "\n" + table_3() + "\n" + table_4() + "\n" + table_robustness()
    f = d168_final()
    # D-438: the lower panel's source on the first line, not as a '%   ' source line, so no later line moves (see
    # bigdrop_panel); its bytes are checked against the manifest before they are read
    head = ("% GENERATED by src/tables_d156.py from the D-156 records below and, for Table IV's lower panel, "
            f"{BIGDROP.relative_to(ROOT).as_posix()} sha256 {bigdrop_record()[1]} "
            f"({BIGDROP_MANIFEST.relative_to(ROOT).as_posix()}; D-432 (4), D-438). Do not edit by hand.\n"
            + ("% DRAFT: rows and groups marked with a dagger rest on inputs verified by pin only and\n"
               "% enter no paper table before D-168's final verdict (D-170); whether they are then kept or omitted is the\n"
               "% author's decision.\n" if f is None else
               f"% D-168 is final ({f['decision']}; {f['file']} {f['sha256']}): the D-170 daggers are removed and the rows\n"
               "% are kept (D-300 (3)).\n")
            + "".join(f"%   {rel}  {h}\n" for rel, h in sorted(_RESULT_READS.items())) + "\n")
    # D-357 (B)(2): the generated file is replaced by writing <name>.part and renaming it over the old one, so no rm is
    # needed (the old version stays recoverable from git); a leftover .part from an interrupted run refuses
    part = OUT.with_name(OUT.name + ".part")
    if part.exists():
        raise SystemExit(f"REFUSING: {part.relative_to(ROOT)} exists (left by an interrupted run); inspect it first")
    part.write_text(head + body, encoding="utf-8", newline="\n")
    os.replace(part, OUT)
    print(f"Written: {OUT.relative_to(ROOT)} ({len(_RESULT_READS)} source files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
