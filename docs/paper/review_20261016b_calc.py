"""REVIEW 2026-10-16b: the descriptive computations pre-registered in D-455 (2) and D-456 (read only; no verdict).

Reads committed records and frozen data under the repo root; writes docs/paper/review_20261016b_calc.json only.
Prints counts, ids, cat-k IDs and numbers; never a prompt text or a category label (CLAUDE.md rule 10): category
names are loaded into variables and compared, never printed.

    python -I docs/paper/review_20261016b_calc.py

Parts (D-455 (2)): (a) v1 vs v2 pool4d positives; (b) A2 training rows and their overlaps; (c) the 19 released pairs;
(d) geometric-mean paired ratios; (e) split overlaps per population; (f) realignment cells; (g) the ledger table.
D-456: the Duan cross-tab, inside (c).
"""
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for sub in ("src", "scripts", "docs/reports_d425"):
    sys.path.insert(0, str(ROOT / sub))

import a2_p3 as A2  # noqa: E402  (norm_cat: the category rule of the training mask, unchanged)
import capcost_p3 as CC  # noqa: E402  (split_rows: the ATTACK / MEASURE rows, unchanged)
import tally_ledger_verdicts_d449 as TL  # noqa: E402  (verdict(): the ledger's verdict word, unchanged)

OUT = ROOT / "docs/paper/review_20261016b_calc.json"
READ: dict[str, str] = {}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rel(p: Path | str) -> str:
    return str(p).replace(os.sep, "/")


def rjson(r: str):
    b = (ROOT / r).read_bytes()
    READ[r] = sha(b)
    return json.loads(b.decode("utf-8"))


def rjsonl(r: str) -> list[dict]:
    b = (ROOT / r).read_bytes()
    READ[r] = sha(b)
    return [json.loads(x) for x in b.decode("utf-8").split("\n") if x.strip()]


def rnpz(r: str):
    READ[r] = sha((ROOT / r).read_bytes())
    return np.load(ROOT / r, allow_pickle=False)


def tsha(text: str) -> str:
    return sha(text.encode("utf-8"))


def nsha(text: str) -> str:
    return sha(" ".join(text.casefold().split()).encode("utf-8"))


def keys_of(rows: list[dict]) -> dict[str, set]:
    return {"item_id": {str(r["item_id"]) for r in rows}, "text_sha256": {tsha(r["text"]) for r in rows},
            "normalised_text_sha256": {nsha(r["text"]) for r in rows}}


def overlap(a: list[dict], b: list[dict]) -> dict[str, int]:
    ka, kb = keys_of(a), keys_of(b)
    return {k: len(ka[k] & kb[k]) for k in ka}


def pos(rows: list[dict]) -> list[dict]:
    return [r for r in rows if str(r.get("label")) == "1"]


# ------------------------------------------------------------------------------------------------ (a) pool4d v1 / v2
def part_a(canary_ids: list[str], h_ids: list[str]) -> dict:
    v1 = pos(rjsonl("data/frozen_p2/p2wg__pool4d.jsonl"))
    v2 = pos(rjsonl("data/frozen_p2_v2/p2wg_v2__pool4d.jsonl"))
    b1, b2 = {str(r["item_id"]): r for r in v1}, {str(r["item_id"]): r for r in v2}
    common = sorted(set(b1) & set(b2))
    diff = Counter()
    for i in common:
        for f in ("text", "label", "subcategory", "meta_type", "adversarial"):
            if str(b1[i].get(f)) != str(b2[i].get(f)):
                diff[f] += 1
    canary_diff = Counter()
    for i in canary_ids:
        if i not in b2:
            canary_diff["missing_in_v2"] += 1
        elif any(str(b1[i].get(f)) != str(b2[i].get(f)) for f in ("text", "label", "subcategory")):
            canary_diff["differ"] += 1
    return {"n_pos_v1": len(v1), "n_pos_v2": len(v2), "n_common_ids": len(common),
            "only_v1": len(set(b1) - set(b2)), "only_v2": len(set(b2) - set(b1)),
            "fields_differing_on_common_ids": dict(diff), "id_sets_equal": set(b1) == set(b2),
            "h_ids_equal_v1_positive_ids": sorted(h_ids) == sorted(b1), "h_ids_equal_v2_positive_ids": sorted(h_ids) == sorted(b2),
            "h_ids_in_v1_file_order": h_ids == [str(r["item_id"]) for r in v1],
            "canaries_identical_in_v2": dict(canary_diff) == {}, "canary_differences": dict(canary_diff)}


# --------------------------------------------------------------------------------------------- (b) A2 training rows
def part_b(canary_rows: list[dict]) -> dict:
    cats = rjson("results/phase3_a2_categories.json")
    chosen = [c["category"] for c in cats["chosen"]]
    attack, _ = CC.split_rows(CC.ATTACK, root=ROOT)
    measure, _ = CC.split_rows(CC.MEASURE, root=ROOT)
    READ[CC.SPLIT_FILE] = sha((ROOT / CC.SPLIT_FILE).read_bytes())
    ev2 = pos(rjsonl("data/frozen_p2_v2/p2wg_v2__eval.jsonl"))
    out = {"n_attack": len(attack), "n_attack_harmful": sum(str(r["label"]) == "1" for r in attack),
           "n_attack_unharmful": sum(str(r["label"]) == "0" for r in attack), "n_measure": len(measure),
           "n_eval_positives_v2": len(ev2), "n_canaries": len(canary_rows),
           "category_value_is_normalised": all(A2.norm_cat(c) == c for c in chosen), "per_category": {},
           "attack_overlap": {"canaries": overlap(attack, canary_rows), "eval_positives_v2": overlap(attack, ev2),
                              "measure": overlap(attack, measure)}}
    for k, cat in enumerate(chosen):
        m = [r for r in attack if A2.norm_cat(r.get("subcategory")) == cat]
        out["per_category"][f"cat-{k + 1}"] = {
            "n_masked": len(m), "n_masked_harmful": sum(str(r["label"]) == "1" for r in m),
            "n_masked_unharmful": sum(str(r["label"]) == "0" for r in m),
            "overlap": {"canaries": overlap(m, canary_rows), "eval_positives_v2": overlap(m, ev2),
                        "measure": overlap(m, measure)}}
    return out


# ------------------------------------------------------------------------------------- (c), (d) the 19 released pairs
P1 = ["mistral_base2instruct", "gemma2_base2instruct", "llama3_base2instruct", "llama31_base2instruct",
      "qwen_base2instruct", "olmo_base2instruct", "llama3_to_31", "WG_mistral_base2instruct", "WG_llama3_base2instruct"]
P1_TYPE = {p: ("base -> instruct (Aegis)" if "base2instruct" in p and not p.startswith("WG_") else
               "base -> instruct (WildGuardMix axis)" if p.startswith("WG_") else "base -> base version increment")
           for p in P1}
P2 = ["mistral_v01_v02", "llama3_to_31", "qwen2_to_25", "phi3_128k_to_35", "olmo2_base_sft", "olmo2_sft_dpo",
      "olmo2_dpo_instruct", "tulu3_sft_dpo", "tulu3_dpo_final", "mistral_v02_v03"]


def klass(r: float) -> str:
    return "fail closed" if r >= 2 else "fail open" if r <= 0.5 else "within"


def gm_cell(npz_rel: str, json_rel: str, layer: int) -> dict:
    R = rjson(json_rel)
    z = rnpz(npz_rel)
    lay = [int(v) for v in z["cell_layer"]]
    ps = [str(v) for v in z["cell_probe_set"]]
    rows = [i for i in range(len(lay)) if lay[i] == layer and ps[i] == "selected"]
    if len(rows) != 1:
        raise SystemExit(f"{npz_rel}: {len(rows)} selected rows at layer {layer}")
    x = z["log_real_ratio"][rows[0], 1:]
    fin = np.isfinite(x)
    jc = [c for c in R["cells"] if c["layer"] == layer and c["probe_set"] == "selected"][0]
    jm = jc["realised"]["paired_ratio"]["mean"]
    am = float(np.exp(x[fin]).mean())
    return {"geometric_mean": float(math.exp(x[fin].mean())), "arithmetic_from_draws": am, "json_mean": jm,
            "anchored": abs(am - jm) <= 1e-9 * max(1.0, abs(jm)), "n_draws_used": int(fin.sum()),
            "n_nonfinite": int((~fin).sum())}


def part_c(big: float, oper: float, elig: float) -> dict:
    rows = []
    for p in P1 + ["mistral_v01_v02_dense"]:
        r = rjson(f"results/depth_d156_{p}.json")
        c = [c for c in r["cells"] if c["is_selected_layer"] and c["probe_set"] == "selected"]
        assert len(c) == 1, p
        c = c[0]
        op, au = c["operating_point"], c["auroc"]
        cfg = r["config"]
        row = {"phase": 1, "pair": p, "type": P1_TYPE.get(p, "base -> base (the v0.1 -> v0.2 control)"),
               "pre": f"{cfg['pre_repo']}@{cfg['pre_rev']}", "post": f"{cfg['post_repo']}@{cfg['post_rev']}",
               "L": c["layer"], "C": c["probe_C"], "ratio": op["paired_ratio"]["mean"],
               "ratio_band": op["paired_ratio"]["band95"], "dtpr": op["delta_tpr"]["mean"],
               "dtpr_band": op["delta_tpr"]["band95"], "auc_pre": au["pre"]["mean_over_seeds"],
               "auc_post": au["frozen"]["mean_over_seeds"], "dauc": -au["drop_frozen"]["mean_over_seeds"],
               "dauc_ci": [-au["drop_frozen"]["ci95"][1], -au["drop_frozen"]["ci95"][0]], "control": p.endswith("_dense")}
        if not row["control"]:
            row["gm"] = gm_cell(f"results/twomoment_repair_{p}_draws.npz", f"results/twomoment_repair_{p}.json", c["layer"])
        rows.append(row)
    for p in P2:
        r = rjson(f"results/twomoment_p2_d361_{p}_usermean.json")
        c = [c for c in r["cells"] if c["is_selected_layer"] and c["probe_set"] == "selected"]
        assert len(c) == 1, p
        c = c[0]
        re_ = c["realised"]
        cfg = r["config"]
        from twomoment_p2_d361 import PAIRS  # noqa: E402
        row = {"phase": 2, "pair": p, "type": PAIRS[p][2], "pre": f"{cfg['pre_repo']}@{cfg['pre_rev']}",
               "post": f"{cfg['post_repo']}@{cfg['post_rev']}", "L": c["layer"], "C": c["probe_C"],
               "ratio": re_["paired_ratio"]["mean"], "ratio_band": re_["paired_ratio"]["band95"],
               "dtpr": re_["delta_tpr"]["mean"], "dtpr_band": re_["delta_tpr"]["band95"], "control": False,
               "gm": gm_cell(f"results/twomoment_p2_d361_{p}_usermean_draws.npz",
                             f"results/twomoment_p2_d361_{p}_usermean.json", c["layer"])}
        ap = ROOT / f"results/p2auroc_d424_{p}.json"
        if ap.exists():
            a = rjson(f"results/p2auroc_d424_{p}.json")
            ac = [x for x in a["cells"] if x["is_selected_layer"]]
            assert len(ac) == 1 and ac[0]["layer"] == c["layer"], p
            au = ac[0]["auroc"]
            row.update(auc_pre=au["pre"]["mean_over_seeds"], auc_post=au["frozen"]["mean_over_seeds"],
                       dauc=au["change"]["mean_over_seeds"], dauc_ci=au["change"]["ci95"],
                       record_duan_flags=ac[0]["duan_flags"])
        else:
            row.update(auc_pre=None, auc_post=None, dauc=None, dauc_ci=None, auroc_record="NOT FOUND")
        rows.append(row)
    for row in rows:
        row["class"] = klass(row["ratio"])
        if row.get("gm"):
            row["gm"]["class"] = klass(row["gm"]["geometric_mean"])
        if row["auc_pre"] is None:
            row["duan"] = {"defined": False, "fires": None}
        else:
            e = row["auc_pre"] >= elig
            b = row["dauc"] <= big
            o = row["auc_post"] < oper
            row["duan"] = {"defined": True, "eligible": e, "big_drop": b, "operational_failure": o,
                           "fires": bool(e and (b or o))}
            f = row.get("record_duan_flags")
            if f is not None:
                row["duan"]["matches_record_flags"] = (f["eligible"] == e and f["big_drop"] == b
                                                       and f["operational_failure"] == o)
    main = [r for r in rows if not r["control"]]
    distinct = sorted({(r["pre"], r["post"]) for r in main})
    xt = defaultdict(lambda: defaultdict(list))
    for r in main:
        col = "undefined" if not r["duan"]["defined"] else "fires" if r["duan"]["fires"] else "silent"
        xt[r["class"]][col].append(f"P{r['phase']} {r['pair']}")
    changes = [f"P{r['phase']} {r['pair']}: {r['class']} -> {r['gm']['class']}" for r in main
               if r.get("gm") and r["gm"]["class"] != r["class"]]
    return {"rows": rows, "n_pairs": len(main), "n_distinct_model_pairs": len(distinct),
            "class_counts": dict(Counter(r["class"] for r in main)),
            "gm_class_counts": dict(Counter(r["gm"]["class"] for r in main if r.get("gm"))),
            "gm_class_changes": changes, "gm_unanchored": [r["pair"] for r in main if r.get("gm") and not r["gm"]["anchored"]],
            "duan_constants": {"big_drop": big, "operational": oper, "eligible": elig, "source": "src/realign_d418.py:172"},
            "duan_crosstab": {k: {c: v for c, v in d.items()} for k, d in xt.items()}}


# --------------------------------------------------------------------------------------- (e) split overlaps
POPS = {
    "Aegis Phase 1": {"probe_train": "data/frozen/safety__probe_train.jsonl",
                      "threshold_calib": "data/frozen/safety__threshold_calib.jsonl", "eval": "data/frozen/safety__eval.jsonl"},
    "WildGuardMix Phase 1": {"probe_train": "data/frozen/safety_wildguard__probe_train.jsonl",
                             "threshold_calib": "data/frozen/safety_wildguard__threshold_calib.jsonl",
                             "eval": "data/frozen/safety_wildguard__eval.jsonl"},
    "WildGuardMix v1": {"probe_train": "data/frozen_p2/p2wg__probe_train.jsonl", "calibration": "data/frozen_p2/p2wg__calibration.jsonl",
                        "eval": "data/frozen_p2/p2wg__eval.jsonl", "pool4d": "data/frozen_p2/p2wg__pool4d.jsonl"},
    "WildGuardMix v2": {"probe_train": "data/frozen_p2/p2wg__probe_train.jsonl",
                        "calibration": "data/frozen_p2_v2/p2wg_v2__calibration.jsonl",
                        "eval": "data/frozen_p2_v2/p2wg_v2__eval.jsonl", "pool4d": "data/frozen_p2_v2/p2wg_v2__pool4d.jsonl"},
}


def part_e() -> dict:
    out = {}
    for pop, files in POPS.items():
        rows = {k: rjsonl(v) for k, v in files.items()}
        names = list(rows)
        out[pop] = {"n": {k: len(v) for k, v in rows.items()}, "pairs": {}}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                out[pop]["pairs"][f"{names[i]} x {names[j]}"] = overlap(rows[names[i]], rows[names[j]])
        out[pop]["all_zero"] = all(v == 0 for d in out[pop]["pairs"].values() for v in d.values())
    return out


# ------------------------------------------------------------------------------------------ (f) realignment
def part_f() -> dict:
    cells = []
    s18 = rjson("results/realign_d418_summary.json")
    for pop, d in s18["populations"].items():
        for c in d["P-RA.1"]["cells"]:
            cells.append({"source": "D-418 ridge", "population": pop, "pair": c["pair"], "pool": c["pool"],
                          "d_auc_frozen": c["d_auc_frozen"], "d_auc_arm": c["d_auc_arm"], "passes": c["passes"]})
    s23 = rjson("results/realign_d423_summary.json")
    for unit, d in s23["units"].items():
        for c in d["P-RA.1"]["cells"]:
            cells.append({"source": f"D-423 {unit}", "population": d.get("population"), "pair": c["pair"],
                          "pool": c["pool"], "d_auc_frozen": c["d_auc_frozen"], "d_auc_arm": c["d_auc_arm"],
                          "passes": c["passes"]})
    fin = [c for c in cells if isinstance(c["d_auc_frozen"], (int, float))]
    lo = min(fin, key=lambda c: c["d_auc_frozen"])
    return {"n_cells": len(cells), "lowest_d_auc_frozen": lo,
            "n_big_drop_frozen": sum(c["d_auc_frozen"] <= -0.05 for c in fin),
            "n_d_auc_arm_le_big_drop": sum(isinstance(c["d_auc_arm"], (int, float)) and c["d_auc_arm"] <= -0.05 for c in cells),
            "cells": cells}


# -------------------------------------------------------------------------------------------- (g) the ledger
def phase_of(n: int) -> str:
    if n in range(56, 77) or n in range(90, 101) or n in range(105, 138):
        return "Phase 3"
    if n in range(77, 80) or n in range(83, 86):
        return "Phase 2"
    return "Phase 0/1"


def part_g() -> dict:
    led = "docs/registered_predictions.md"
    b = (ROOT / led).read_bytes()
    READ[led] = sha(b)
    import re
    rows = {}
    for line in b.decode("utf-8").split("\n"):
        m = re.match(r"^\|\s*(\d+)\s*\|", line)
        if m:
            if int(m.group(1)) in rows:
                raise SystemExit(f"duplicate ledger row {m.group(1)}")
            cells = line.rstrip("\r").split("|")
            rows[int(m.group(1))] = (cells[2].strip(), TL.verdict(cells[3:]))
    table = [{"row": n, "label": lab, "phase": phase_of(n), "verdict": v} for n, (lab, v) in sorted(rows.items())]
    tot = defaultdict(Counter)
    for t in table:
        tot[t["phase"]][t["verdict"]] += 1
        tot["overall"][t["verdict"]] += 1
    return {"n_rows": len(table), "totals": {k: dict(v) for k, v in tot.items()},
            "n_per_phase": {k: sum(v.values()) for k, v in tot.items()}, "table": table}


def main() -> None:
    rec = rjson("results/phase3_defence_d2e_d443.json")
    z = rnpz(rec["npz"]["file"])
    canary_ids = [str(x) for x in z["canary|ids"]]
    h_ids = [str(x) for x in z["canary|h_ids"]]
    v1 = {str(r["item_id"]): r for r in rjsonl("data/frozen_p2/p2wg__pool4d.jsonl")}
    canary_rows = [v1[i] for i in canary_ids]
    import realign_d418 as RA  # noqa: E402  (D-456 (1): the constants at src/realign_d418.py:172, imported)
    res = {"pre_registration": "D-455 (2), D-456", "status": "DESCRIPTIVE",
           "a_pool4d_v1_v2": part_a(canary_ids, h_ids), "b_a2_training_rows": part_b(canary_rows),
           "c_d_released_pairs": part_c(RA.BIG_DROP, RA.OPERATIONAL, RA.ELIGIBLE), "e_split_overlaps": part_e(),
           "f_realignment": part_f(), "g_ledger": part_g(),
           "d2e_draws_and_tau": {"n_reps": rec["config"]["n_reps"], "n_draws": rec["config"]["n_draws"],
                                 "draw_seed": rec["config"]["draw_seed"], "matched_tau": rec["matched_tau"],
                                 "clean_false_alarms": rec["clean_false_alarms"],
                                 "clean_seed_confound": rec["clean_seed_confound"]}}
    res["files_read_sha256"] = dict(sorted(READ.items()))
    OUT.write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")

    a, b, c, e, f, g = (res[k] for k in ("a_pool4d_v1_v2", "b_a2_training_rows", "c_d_released_pairs",
                                          "e_split_overlaps", "f_realignment", "g_ledger"))
    print("(a) pool4d positives v1 / v2:", {k: a[k] for k in ("n_pos_v1", "n_pos_v2", "n_common_ids", "only_v1", "only_v2",
          "fields_differing_on_common_ids", "h_ids_equal_v1_positive_ids", "h_ids_equal_v2_positive_ids",
          "canaries_identical_in_v2")})
    print("(b) A2 training:", {k: b[k] for k in ("n_attack", "n_attack_harmful", "n_attack_unharmful", "n_measure",
          "category_value_is_normalised", "attack_overlap")})
    for k, v in b["per_category"].items():
        print(f"    {k}: masked {v['n_masked']} (harmful {v['n_masked_harmful']}, unharmful {v['n_masked_unharmful']}); "
              f"overlap {v['overlap']}")
    print(f"(c) pairs {c['n_pairs']}, distinct model pairs {c['n_distinct_model_pairs']}; classes {c['class_counts']}; "
          f"GM classes {c['gm_class_counts']}; GM changes {c['gm_class_changes']}; unanchored {c['gm_unanchored']}")
    for r in c["rows"]:
        d = r["duan"]
        dd = "UNDEFINED (no AUROC record)" if not d["defined"] else (
            f"elig {d['eligible']} big {d['big_drop']} oper {d['operational_failure']} fires {d['fires']}"
            + (f" rec-match {d['matches_record_flags']}" if "matches_record_flags" in d else ""))
        gm = r.get("gm") or {}
        au = "NOT FOUND" if r["auc_pre"] is None else (f"pre {r['auc_pre']:.4f} post {r['auc_post']:.4f} "
                                                       f"d {r['dauc']:+.4f} [{r['dauc_ci'][0]:+.4f}, {r['dauc_ci'][1]:+.4f}]")
        print(f"  P{r['phase']} {r['pair']:<26} {r['type']:<34} L{r['L']} C{r['C']} ratio {r['ratio']:.3f} "
              f"[{r['ratio_band'][0]:.3f}, {r['ratio_band'][1]:.3f}] {r['class']:<11} GM "
              f"{gm.get('geometric_mean', float('nan')):.3f} ({gm.get('class', '-')}, anch {gm.get('anchored', '-')}, "
              f"nonfin {gm.get('n_nonfinite', '-')}) dTPR {r['dtpr']:+.3f} [{r['dtpr_band'][0]:+.3f}, "
              f"{r['dtpr_band'][1]:+.3f}] AUROC {au} | Duan {dd}{' CONTROL' if r['control'] else ''}")
    print("    Duan cross-tab:", json.dumps(c["duan_crosstab"]))
    for pop, d in e.items():
        print(f"(e) {pop}: n {d['n']}; all zero {d['all_zero']}; " + "; ".join(f"{k} {v}" for k, v in d["pairs"].items()
                                                                               if any(v.values())))
    lo = f["lowest_d_auc_frozen"]
    print(f"(f) realignment cells {f['n_cells']}; frozen big drops {f['n_big_drop_frozen']}; arm <= -0.05 "
          f"{f['n_d_auc_arm_le_big_drop']}; lowest frozen dAUC {lo}")
    print(f"(g) ledger rows {g['n_rows']}; per phase {g['n_per_phase']}")
    for k, v in g["totals"].items():
        print(f"    {k}: {v}")
    print(f"wrote {rel(OUT.relative_to(ROOT))} sha256 {sha(OUT.read_bytes())}")


if __name__ == "__main__":
    main()
