"""D-282 for Table III (4a, update intent at matched dose: the nine Phase 1 LoRA runs): the combined item-bootstrap x
re-partition interval (D-281 (1)) for TPR pre, TPR post and delta TPR at the frozen threshold, beside the D-156 band,
as a SUPPLEMENT record (D-287's precedent: no source record is re-issued).

Table III (src/tables_d156.py, tab:intent-d156) reads results/repair_d156_<run>.json (4d A, D-289),
results/depth_d156_<run>.json (D-224) and results/intent_selected_d165.json, and quoted no TPR before this supplement
(docs/phase1_exit.md section 3 item 3). The frozen-threshold TPR its records report is the depth record's: the operating
point of every cell, and the D-180 rows of its pair_summary (headline at the selected layer L*, sweep maximum, lowest
layer, two-sided), which have the structure of Table 1's rows. For each run, probe set and layer this script re-does
depth_d156's work at that layer only, as src/table1_tpr_d282.py does (D-287): the same caches through the input
registry, the LoRA A100 selection of record (D-171, D-224), the same D-117 fits and the same item-level D-156 draws.
Then, per block (run, probe set, layer):
  - ANCHOR: the item-level operating point equals the depth record's cell as canonical JSON text, and the per-draw
    ratio_to_target array hashes to the cell's per_draw_ratio_to_target_sha256, which pins the draw ORDER that the
    pairing of draw d_r with counts row r needs; at L* the three TPR blocks also equal the pair_summary headline's;
  - ONES CHECK: counts of all ones give back the D-156 per-draw TPR arrays bit for bit (D-282 (4));
  - SHARED PRE SIDE: the nine runs share the pre model (Mistral-7B-v0.1 on the A100, D-224), so every run's TPR-pre
    replicates must equal those of the first run with the same pre side, C, layer and probe set.
paired.combined_tpr_replicates then gives the combined interval: replicate r = 1..5000 uses draw
d_r = 1 + ((r - 1) mod 1000) and row r of paired.counts_matrix(n positives, 5000,
depth_d156.ITEM_BOOT_SEED["eval_positives"]); the same counts weight the pre and post flags and every seed column
(paired within item), rates take P.rate's weighted form, and the interval is the 2.5-97.5 percentiles. The D-156 band,
its mean and the frozen split's value are copied beside it from the depth record's cell.

CHOICES by the closest D-entry (D-181), each also in the record's implementation_choices (CHOICES below):
  (1) SUPPLEMENT (D-287 (1)): results/table3_tpr_d282.json and _fpr5; the depth, repair and intent records are
      untouched and anchored exactly.
  (2) SCOPE (D-287 (2), D-281 (1)): tpr_pre, tpr_post and delta_tpr, both probe sets (the selected C primary, C=1
      beside), at every layer where the run's depth record's pair_summary quotes TPR at the item level (the D-180
      rows), plus L*, at 1% and at 5% (D-248). Table III prints delta TPR at L*, as Table I does (D-287). No D-209
      layers: they belong to the WildGuardMix rows of Table 1.
  (3) NOT COVERED: the repair arms' TPR (Table II's precedent: no D-282 interval for these arms) and the cluster level
      (D-282 (2): it keeps its D-156 band only).
  (4) COUNTS (D-287 (3)): depth_d156's eval_positives stream, a function of n positives. Every run reads the same pre
      eval cache, so the positives are the same items in the same order (checked by an ordered item-id hash) and one
      matrix pairs by item across runs.
  (5) BINDING (D-287 (1), (4); D-224; D-289): each 1% depth record's sha256 must equal its line in
      logs/depth_d156_lora_hw_20260927.sha256 and the anchor of repair_d156_<run>.json (4d A's frozen arm, Table III's
      frozen column), whose own sha256 must equal its line in logs/repair_d156_4dA_20260930.sha256. At 5% the _fpr5
      record must equal its line in logs/fpr5_stage_depth_20260928.sha256 and carry a complete fpr5_self_anchor naming
      that 1% record and hash; its layers come from its own pair_summary (there is no 5% Table III). The final name
      also needs each manifest read to be the bytes pinned in MANIFEST_SHA256 (the provenance review of 1f328d4), and
      every depth record read must name, in its twomoment_check, the damage record read for (12) and its bytes.
  (6) SHARED PRE (D-287 (5)): every (pre repo and revision, axis, C, layer, probe set) seen again must give identical
      TPR-pre replicates.
  (7) FLAGS: the D-170 flag per row and per file (D-172), each row's equal to its depth record's (D-331 cleared the
      pin-only flag; the LoRA inputs are extraction-record verified). D-344 (3), D-349 (the HH arm's training
      conversations contained frozen evaluation items in later Human turns): the hh-harmless rows, and the between-arm
      rows of that arm, carry under_check; Table III prints those rows with a marker and a caption note
      (src/tables_d156.py); the interval is computed for every run and nothing here clears that flag.
  (8) IDENTITY (D-297 (1), D-300): config names each cache input's repo ID and revision (provenance.identity_config,
      from the depth record's config); every results file is read through read_result and listed with its sha256;
      each cache's resolved path is recorded, and the final name refuses a set GUARDRAIL_CACHE_DIRS (the caches then
      come only from the input registry's directory).
  (9) POPULATION (D-287 (6)): groups DEPTH_D156_LORA_D282 and DEPTH_D156_LORA_D282_FPR5, declared in
      src/populations.py when the records are pulled.
  (10) RELEASE (D-179; as table1_tpr_d282.py, D-287): RESEARCH.
  (11) REVIEW (D-271 (1), D-282 (4)): new analysis code, three reviewers before the run of record on a node lane.
  (12) BETWEEN-ARM GAP (D-282 (1), D-152 4a: 4a's primary contrast is a TPR quantity): the between-arm delta-TPR gap
      (safety minus general at matched dose) of results/damage_d156_r2.json's between_arms_4a (at 5%,
      damage_d156_r2_fpr5.json's), at L*, both probe sets, item level, for exactly the comparisons that record lists
      (read from it, not fixed here). Replicate r is the safety run's delta-TPR replicate r minus the general run's,
      on the same draw d_r and the same counts row r (valid because (4) checks that the runs share the positives in
      the same order). The record's frozen value, D-156 mean and band are copied, and anchored by recomputing its
      band summary from the two anchored per-draw delta-TPR arrays (canonical JSON, exactly). The record's 4a cells
      must sit at L* with the probe set's C. The cluster level keeps its D-156 band only (D-282 (2)). In this
      supplement only: Table III does not print the gap.
It writes results/table3_tpr_d282[_fpr5].json once (provenance.write_results, exclusive); only the registered
parameters, all nine runs, the 6-thread BLAS pin, the pinned manifests and an unset GUARDRAIL_CACHE_DIRS may write the
final name. A test writes outside results/.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_d156 as D  # noqa: E402
import input_registry as REG  # noqa: E402
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
import table1_tpr_d282 as T1  # noqa: E402
from provenance import (  # noqa: E402
    _RESULT_READS, ensure_utf8_stdout, identity_config, provenance_block, read_result, resolve_cache, sha256_bytes,
    sha256_file, write_results,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/table3_tpr_d282.py", "src/table1_tpr_d282.py", *D.CODE_FILES]
QUANT = T1.QUANT
RUNS = list(D.LORA_DEPTH)                                     # POP.LORA_ARMS order: alpaca, hh-harmless, pku
ARM_OF = {f"depth_{r}": arm for arm, runs in POP.LORA_ARMS.items() for r in runs}
SELECTION = D.SELECTION_OF_RECORD_LORA                        # D-171, D-224
LORA_DEPTH_MANIFEST = ROOT / "logs" / "depth_d156_lora_hw_20260927.sha256"      # the pulled D-224 depth records
REPAIR_MANIFEST = ROOT / "logs" / "repair_d156_4dA_20260930.sha256"             # 4d A (D-289), Table III's frozen column
FPR5_DEPTH_MANIFEST = ROOT / "logs" / "fpr5_stage_depth_20260928.sha256"        # D-277 (as table1_tpr_d282)
# the final name needs these manifest bytes (provenance review of 1f328d4; each value is the sha256 of the committed file)
MANIFEST_SHA256 = {"depth_d156_lora_hw_20260927.sha256": "8e45c1c5eef31b0eec7f3da030579e4a8600d7f4e7f466f02112b15e99a3d269",
                   "repair_d156_4dA_20260930.sha256": "5f41056ffe26ebb8dfcf6141d7d40be25142d680f3cdbb184ba0ce5ea1d2b757",
                   "fpr5_stage_depth_20260928.sha256": "a379f9fc41cdb9a9946c1d1f0b17cf57660fdcef8a121f513da2cdc209bf7db4"}
DAMAGE_RECORD = D.DAMAGE_OF_RECORD                     # CHOICES (12): 4a's between-arm gaps (D-173)
DAMAGE_RECORD_FPR5 = D.DAMAGE_FPR5                     # their 5% twin (D-248)
D344_ARM = "safety (hh-harmless)"
D344 = {"decision": "D-344",
        "status": "UNDER CHECK (D-344 (3); D-349: the HH arm's training conversations contained frozen evaluation items "
                  "in later Human turns). Until the author decides, Table III prints this row with the under-check "
                  "marker (a section sign) and that sentence in its caption, and no claim rests on it. The interval is "
                  "computed; nothing here clears the flag"}
# a between-arm gap with an HH side is in this supplement only (review of 411f234, minor 1): its own status says so
D344_GAP = {"decision": "D-344",
            "status": "UNDER CHECK (D-344 (3); D-349: the HH arm's training conversations contained frozen evaluation items "
                      "in later Human turns). This between-arm gap is in this supplement only, not printed in Table III, "
                      "and no claim rests on it. The interval is computed; nothing here clears the flag"}
CHOICES = (
    {"choice": "a supplement, not a re-issue: the depth, repair and intent records are untouched and anchored exactly",
     "rule": "D-287 (1)"},
    {"choice": "tpr_pre, tpr_post and delta_tpr, both probe sets, at every layer where the depth record's pair_summary "
               "quotes TPR at the item level (the D-180 rows) plus L*, at 1% and 5%; Table III prints delta TPR at L* "
               "as Table I does; no D-209 layers (WildGuardMix rows only)",
     "rule": "D-287 (2), D-281 (1), D-248"},
    {"choice": "the repair arms' TPR and the cluster level get no D-282 interval",
     "rule": "Table II's precedent (no D-282 interval for these arms); D-282 (2)"},
    {"choice": "counts: depth_d156's eval_positives stream, one matrix for every run (same positives, same order, checked)",
     "rule": "D-287 (3), D-282 (2)"},
    {"choice": "binding: the 1% depth record by logs/depth_d156_lora_hw_20260927.sha256 and by repair_d156_<run>'s "
               "anchor (itself by logs/repair_d156_4dA_20260930.sha256); at 5% by logs/fpr5_stage_depth_20260928.sha256 "
               "and a complete fpr5_self_anchor on that 1% record; 5% layers from the _fpr5 record's own pair_summary; "
               "the final name needs every manifest read to be the pinned bytes; each depth record's twomoment_check "
               "names the damage record read and its bytes",
     "rule": "D-287 (1), (4); D-224; D-289"},
    {"choice": "shared pre side: runs with the same pre repo and revision, axis, C, layer and probe set must give "
               "identical TPR-pre replicates", "rule": "D-287 (5)"},
    {"choice": "the D-170 flag per row and per file, equal to the depth record's; the hh-harmless rows and that arm's "
               "between-arm rows carry D-344's under_check (D-349's count), which is not cleared here; Table III marks "
               "them", "rule": "D-172, D-331; D-344 (3), D-349"},
    {"choice": "each cache input's repo ID and revision named in config; every results file read through read_result "
               "and listed with its sha256; each cache's resolved path recorded; the final name refuses a set "
               "GUARDRAIL_CACHE_DIRS", "rule": "D-297 (1), D-300"},
    {"choice": "population groups DEPTH_D156_LORA_D282 and DEPTH_D156_LORA_D282_FPR5, declared when the records are "
               "pulled", "rule": "D-287 (6)"},
    {"choice": "release class RESEARCH", "rule": "D-179, D-287"},
    {"choice": "new analysis code: three reviewers before the run of record", "rule": "D-271 (1), D-282 (4)"},
    {"choice": "4a's between-arm delta-TPR gap (safety minus general at matched dose, the comparisons the damage record "
               "lists), at L*, both probe sets, item level: replicate r is the safety run's delta-TPR replicate minus "
               "the general run's (the same draw d_r and counts row r); frozen, D-156 mean and band copied from the "
               "damage record and anchored by recomputing them from the two anchored per-draw arrays; the cluster "
               "level keeps its band only; in this supplement only, not in Table III",
     "rule": "D-282 (1), D-152 4a"},
)


def rel_path(p: Path) -> str:
    p = Path(p)
    return str(p.relative_to(ROOT)).replace("\\", "/") if p.is_relative_to(ROOT) else p.name


def path_text(p: Path) -> str:
    """A path as recorded: repo-relative when under the repository, else absolute."""
    p = Path(p)
    return str(p.relative_to(ROOT)).replace("\\", "/") if p.is_relative_to(ROOT) else str(p)


def manifest(path: Path) -> tuple[dict[str, str], str]:
    """({repo-relative path: sha256} of a sha256sum manifest, the sha256 of the bytes parsed); refuses a missing one.
    One read: the hash is of the bytes the mapping comes from."""
    if not path.is_file():
        raise SystemExit(f"REFUSING: {rel_path(path)} is not on disk")
    b = path.read_bytes()
    return {ln.split()[1]: ln.split()[0] for ln in b.decode("utf-8").splitlines() if ln.strip()}, sha256_bytes(b)


def read_listed(rel: str) -> dict[str, Any]:
    """A results file read through read_result (D-297 (1)); refuses unless the read is registered under `rel`, the
    repo-relative name under which the record lists it."""
    doc = read_result(ROOT / rel)
    if rel not in _RESULT_READS:
        raise SystemExit(f"REFUSING: {rel} was not registered under its repo-relative name")
    return doc


def under_check(stem: str) -> dict[str, str] | None:
    """D-344 (3): the hh-harmless rows of 4a are under check."""
    return dict(D344) if ARM_OF.get(stem) == D344_ARM else None


def quoted_layers(pair_summary: dict[str, Any], pset: str, L_sel: int) -> list[int]:
    """Every layer at which the depth record's D-180 rows quote TPR at the item level (table1_tpr_d282's rule), and L*."""
    return sorted(set(T1.table1_layers(pair_summary, pset)) | {L_sel})


def bind_depth(stem: str, target: float, lora_man: dict[str, str], repair_man: dict[str, str],
               fpr5_man: dict[str, str], sel_sha: str, C_sel: float, L_sel: int, dmg_rel: str,
               dmg_sha: str) -> dict[str, Any]:
    """CHOICES (5): read and bind the run's depth record (and at 5% its 1% twin) and its repair record; the depth
    record read must name the damage record `dmg_rel` with the bytes `dmg_sha` in its twomoment_check (CHOICES (12)).
    Returns {doc, rel, sha, rel1, sha1, repair_rel, repair_sha}; refuses on any mismatch."""
    name = D.out_name(stem)
    rel1, repair_rel = f"results/{name}.json", f"results/repair_d156_{stem[len('depth_'):]}.json"
    doc1 = read_listed(rel1)
    sha1 = _RESULT_READS[rel1]
    rep = read_listed(repair_rel)
    repair_sha = _RESULT_READS[repair_rel]
    if lora_man.get(rel1) != sha1:
        raise SystemExit(f"REFUSING: {rel1} is not the depth record {rel_path(LORA_DEPTH_MANIFEST)} pins (D-224)")
    if repair_man.get(repair_rel) != repair_sha:
        raise SystemExit(f"REFUSING: {repair_rel} is not the 4d A record {rel_path(REPAIR_MANIFEST)} pins (D-289)")
    anc = rep.get("anchor") or {}
    if anc.get("file") != rel1 or anc.get("sha256") != sha1:
        raise SystemExit(f"REFUSING: {repair_rel} (Table III's frozen column) anchors to other bytes than {rel1}")
    rel, doc, sha = rel1, doc1, sha1
    if target != D.REGISTERED["target_fpr"]:
        rel = f"results/{name}_fpr5.json"
        doc = read_listed(rel)
        sha = _RESULT_READS[rel]
        sa = doc.get("fpr5_self_anchor") or {}
        if fpr5_man.get(rel) != sha or sa.get("file") != rel1 or sa.get("sha256") != sha1 or \
                sa.get("n_cells_identical_at_1pct") != len(doc["cells"]):
            raise SystemExit(f"REFUSING: {rel} is not a verified 5% depth record self-anchored on {rel1} (D-248, D-277)")
    cfg = doc["config"]
    if (cfg["selected_C"], cfg["selected_layer"], cfg["selection_sha256"], cfg["target_fpr"]) != (C_sel, L_sel, sel_sha, target):
        raise SystemExit(f"REFUSING: {rel} was run on another selection or target")
    if (cfg.get("registry_pair"), cfg.get("selection_of_pair")) != (D.registry_stem(stem), D.LORA_SELECTION_PAIR):
        raise SystemExit(f"REFUSING: {rel} names another registry run or selection pair")
    tc = doc.get("twomoment_check") or {}
    if tc.get("file") != dmg_rel or tc.get("sha256") != dmg_sha:
        raise SystemExit(f"REFUSING: {rel} was checked against other bytes than {dmg_rel}, the damage record read here (D-224)")
    return {"doc": doc, "rel": rel, "sha": sha, "rel1": rel1, "sha1": sha1, "repair_rel": repair_rel,
            "repair_sha": repair_sha}


def probe_scores(X: dict[tuple[str, str], np.ndarray], y_tr: np.ndarray, masks: dict[str, np.ndarray],
                 seeds: list[int], C: float, pset: str) -> dict[str, np.ndarray]:
    """depth_d156's probes and scores at one layer: per seed the D-117 80% fit split, the selected C in permutation
    order ('selected') or C=1 on the sorted rows in file order ('C=1'); scores (items, seeds) of the pre and post
    threshold_calib negatives, eval negatives and eval positives."""
    rows_of, order = (lambda f: f, "d117_permutation") if pset == "selected" else (np.sort, "file_order")
    probes = []
    for seed in seeds:
        _, fit = P.d117_split(len(y_tr), seed)
        idx = rows_of(fit)
        probes.append(P.fit_probe(X[("pre", "probe_train")][idx], y_tr[idx], seed, C, order))
    S = lambda side, part, m: np.column_stack([q.score(X[(side, part)][m]) for q in probes])  # noqa: E731
    return {"cal_pre": S("pre", "threshold_calib", masks["ncal"]), "ev_pre": S("pre", "eval", masks["neg"]),
            "cal_post": S("post", "threshold_calib", masks["ncal"]), "ev_post": S("post", "eval", masks["neg"]),
            "pos_pre": S("pre", "eval", masks["pos"]), "pos_post": S("post", "eval", masks["pos"])}


def anchored_replicates(sc: dict[str, np.ndarray], draws: np.ndarray, target: float, cell: dict[str, Any] | None,
                        counts: np.ndarray, where: str
                        ) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """One block: the D-156 report anchored on the depth record's cell (canonical JSON and the per-draw
    ratio_to_target hash), the thresholds T_d of every draw, the ones check, and the combined replicates (D-282 (2)).
    Returns (the recomputed item-level report, its per-draw arrays, {quantity: replicates})."""
    n_cal = sc["cal_pre"].shape[0]
    report, arr = P.repartition_estimate(sc["cal_pre"], sc["ev_pre"], sc["cal_post"], sc["ev_post"],
                                         sc["pos_pre"], sc["pos_post"], draws, target)
    rt = np.concatenate([[report["ratio_to_target"]["frozen"]], arr["ratio_to_target"]]).astype("<f8")
    if cell is None or T1.canon(report) != T1.canon(cell["operating_point"]) or \
            sha256_bytes(rt.tobytes()) != cell["frozen_curve_abs_log_ratio_to_target"]["per_draw_ratio_to_target_sha256"]:
        raise SystemExit(f"REFUSING: {where}: the operating point or its per-draw order is not the depth record's")
    pool_pre = np.vstack([sc["cal_pre"], sc["ev_pre"]])
    n_seeds = pool_pre.shape[1]
    T_dr = np.array([[P.thr_at_fpr(pool_pre[d[:n_cal], s], target) for s in range(n_seeds)] for d in draws])
    n_pos = sc["pos_pre"].shape[0]
    ones = P.combined_tpr_replicates(T_dr, sc["pos_pre"], sc["pos_post"], np.ones((len(draws), n_pos), dtype=np.float32))
    if any(not np.array_equal(ones[q], arr[q]) for q in QUANT):
        raise SystemExit(f"REFUSING: {where}: counts of ones do not give back the D-156 draws (D-282 (4))")
    return report, arr, P.combined_tpr_replicates(T_dr, sc["pos_pre"], sc["pos_post"], counts)


def check_shared_pre(shared: dict[tuple, tuple[str, str]], key: tuple, stem: str, h: str, where: str) -> bool:
    """CHOICES (6): True when `key` was seen before (and its TPR-pre replicates hash equals the first run's); refuses
    a difference; records the first run otherwise."""
    if key in shared:
        if shared[key][1] != h:
            raise SystemExit(f"REFUSING: {where}: TPR pre differs from {shared[key][0]}'s, which shares its pre side")
        return True
    shared[key] = (stem, h)
    return False


def between_arm_gaps(dmg: dict[str, Any], rel: str, at_lstar: dict[tuple[str, str], tuple[np.ndarray, float, np.ndarray]],
                     run_rows: list[str], C_sel: float, L_sel: int) -> tuple[list[dict[str, Any]], int]:
    """CHOICES (12): 4a's between-arm delta-TPR gap at L*, both probe sets, item level, for every comparison the damage
    record `dmg` (read as `rel`) lists in between_arms_4a: a safety run minus the general run at the same dose
    (D-152 4a). at_lstar[(stem, probe set)] = (the anchored per-draw delta-TPR array at L*, its frozen value, its
    combined replicates). The record's band summary of the gap must equal the one recomputed from the two anchored
    per-draw arrays as canonical JSON; replicate r of the gap is the safety run's replicate r minus the general run's
    (the same draw d_r and counts row r). Returns (the gap rows, the number of the record's comparisons left out
    because a run is not in this run's rows); refuses any comparison that is not safety minus general at matched dose,
    a repeated one, 4a cells off L* or the probe set's C, and any mismatch."""
    cells = {(c["cell"], c["probe_set"]): c for c in dmg["cells"] if c.get("scope") == "4a"}
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    n_left_out = 0
    for row in dmg["between_arms_4a"]:
        pset, s_run, g_run, dose = row["probe_set"], row["safety_run"], row["general_run"], row["dose"]
        s, g = f"depth_{s_run}", f"depth_{g_run}"
        where = f"{rel} between_arms_4a {pset} {s_run} - {g_run}"
        if (pset not in D.PROBE_SETS or (pset, s) in seen or s not in ARM_OF or g not in ARM_OF
                or ARM_OF[s] != row["safety_arm"] or ARM_OF[s].startswith("general") or not ARM_OF[g].startswith("general")
                or any(r.rsplit("_s", 1)[-1] != str(dose) for r in (s_run, g_run))):
            raise SystemExit(f"REFUSING: {where} is not one safety-minus-general comparison at matched dose (D-152 4a)")
        seen.add((pset, s))
        if s not in run_rows or g not in run_rows:
            n_left_out += 1
            continue
        C = C_sel if pset == "selected" else 1.0
        if any((cells.get((r, pset)) or {}).get("layer") != L_sel or cells[(r, pset)]["probe_C"] != C for r in (s_run, g_run)):
            raise SystemExit(f"REFUSING: {where}: the record's 4a cells are not at L{L_sel} with C {C:g}")
        (a_s, f_s, r_s), (a_g, f_g, r_g) = at_lstar[(s, pset)], at_lstar[(g, pset)]
        per_draw = a_s - a_g                                  # damage_d156's between-arm gap, formed within each draw
        mine = {**P.band_summary(per_draw, float(np.array(f_s) - np.array(f_g))),
                "share_of_draws_above_0": float((per_draw > 0).mean())}
        want = row["delta_tpr_gap"]
        if T1.canon(mine) != T1.canon(want):
            raise SystemExit(f"REFUSING: {where}: its delta TPR gap is not the one the two anchored per-draw arrays give")
        gap = r_s - r_g
        out.append({"probe_set": pset, "probe_C": C, "layer": L_sel, "safety_arm": row["safety_arm"],
                    "safety_run": s_run, "general_run": g_run, "dose": dose,
                    "input_verification_damage_record": row["input_verification"],
                    "under_check": dict(D344_GAP) if (under_check(s) or under_check(g)) else None,
                    "delta_tpr_gap": {"frozen": want["frozen"], "mean_D156": want["mean"], "band95_D156": want["band95"],
                                      "share_of_draws_above_0_D156": want["share_of_draws_above_0"],
                                      "combined": P.ci(gap), "replicates_sha256": sha256_bytes(gap.astype("<f8").tobytes())}})
    return out, n_left_out


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Table III's TPR column with D-282's combined interval (the LoRA arms)")
    ap.add_argument("--fpr5", action="store_true", help="the 5%% supplement (D-248), against the depth _fpr5 records")
    ap.add_argument("--runs", nargs="+", default=None, choices=RUNS, help="a subset of the runs (a test)")
    ap.add_argument("--n-draws", type=int, default=D.REGISTERED["n_draws"])
    ap.add_argument("--n-boot", type=int, default=D.REGISTERED["n_boot"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    target = D.TARGET_5 if args.fpr5 else D.REGISTERED["target_fpr"]
    seeds = D.REGISTERED["seeds"]
    run_rows = list(RUNS) if args.runs is None else list(args.runs)
    if len(set(run_rows)) != len(run_rows):
        raise SystemExit(f"REFUSING: {run_rows} are not distinct runs")
    registered = (args.n_draws == D.REGISTERED["n_draws"] and args.n_boot == D.REGISTERED["n_boot"] and run_rows == RUNS)
    final_path = (RESULTS / f"table3_tpr_d282{'_fpr5' if args.fpr5 else ''}.json").resolve()
    out_path = Path(args.out).resolve() if args.out else final_path
    final = out_path == final_path
    if out_path.exists():
        raise SystemExit(f"REFUSING: {rel_path(out_path)} exists; results are never overwritten")
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test run (--out) must not write under results/")
    threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    cache_dirs = os.environ.get("GUARDRAIL_CACHE_DIRS")
    (lora_man, lora_h), (repair_man, repair_h) = manifest(LORA_DEPTH_MANIFEST), manifest(REPAIR_MANIFEST)
    fpr5_man, fpr5_h = manifest(FPR5_DEPTH_MANIFEST) if args.fpr5 else ({}, "")
    man_sha = {LORA_DEPTH_MANIFEST: lora_h, REPAIR_MANIFEST: repair_h, **({FPR5_DEPTH_MANIFEST: fpr5_h} if args.fpr5 else {})}
    if final:
        problems = []
        if not registered:
            problems.append(f"registered {registered} (the registered n_draws and n_boot and all nine runs)")
        if any(t != "6" for t in threads.values()):
            problems.append(f"threads {threads} (the 6-thread BLAS pin)")
        if cache_dirs:
            problems.append(f"GUARDRAIL_CACHE_DIRS is set ({cache_dirs!r}); the caches must come from the registry's directory")
        unpinned = [rel_path(p) for p, h in man_sha.items() if MANIFEST_SHA256.get(p.name) != h]
        if unpinned:
            problems.append(f"manifests {unpinned} are not the pinned bytes")
        if problems:
            raise SystemExit(f"REFUSING the final name: {problems}")
    code0 = P.code_block(CODE_FILES)
    sel_rel = rel_path(SELECTION)
    sel_doc = read_listed(sel_rel)
    sel_sha = _RESULT_READS[sel_rel]
    if list(sel_doc["config"]["seeds"]) != seeds or float(sel_doc["config"]["holdout"]) != P.HOLDOUT:
        raise SystemExit("REFUSING: the selection's seeds or holdout differ from the registered ones")
    sel = next((m for m in sel_doc["models"] if m["pair"] == D.LORA_SELECTION_PAIR), None)
    if sel is None or not sel["selected"]["arm"].startswith("C="):
        raise SystemExit(f"REFUSING: no logistic-C selection for {D.LORA_SELECTION_PAIR} in {sel_rel}")
    C_sel, L_sel = float(sel["selected"]["arm"].split("=")[1]), int(sel["selected"]["layer"])
    reg_sources = REG.sources_sha256()
    if D.sources_now() != reg_sources:
        raise SystemExit("REFUSING: the extraction records or pins on disk are not the ones the registry read")
    dmg_rel = rel_path(DAMAGE_RECORD_FPR5 if args.fpr5 else DAMAGE_RECORD)        # CHOICES (12)
    dmg_doc = read_listed(dmg_rel)
    dmg_sha = _RESULT_READS[dmg_rel]
    if dmg_doc["config"]["target_fpr"] != target:
        raise SystemExit(f"REFUSING: {dmg_rel} is not at the target {target}")

    counts: dict[int, np.ndarray] = {}
    positives_id: dict[str, str] = {}
    shared_pre: dict[tuple, tuple[str, str]] = {}
    bound: dict[str, dict[str, Any]] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    resolved: dict[str, Path] = {}                            # input key -> the cache file read (CHOICES (8))
    at_lstar: dict[tuple[str, str], tuple[np.ndarray, float, np.ndarray]] = {}     # CHOICES (12)
    out_rows: dict[str, Any] = {}
    n_anchor = n_ones = n_shared = n_headline = 0
    want_blocks = 0                                           # from the depth records' cells, not from this loop
    for stem in run_rows:
        b = bind_depth(stem, target, lora_man, repair_man, fpr5_man, sel_sha, C_sel, L_sel, dmg_rel, dmg_sha)
        bound[stem] = b
        ddoc, rel = b["doc"], b["rel"]
        dcells = {(c["layer"], c["probe_set"]): c for c in ddoc["cells"]}
        reg_stem = D.registry_stem(stem)
        cfg = REG.run(reg_stem)
        axis = cfg["axis"]
        if any(cfg[k] != ddoc["config"].get(k) for k in ("pre_repo", "pre_rev", "post_repo", "post_rev")):
            raise SystemExit(f"REFUSING: {rel} names other repos or revisions than the input registry's {reg_stem}")
        z: dict[tuple[str, str], Any] = {}
        row_recs: dict[str, dict[str, Any]] = {}
        for side in ("pre", "post"):
            for part in ("probe_train", "threshold_calib", "eval"):
                z[(side, part)], r = REG.load(cfg[f"{side}_repo"], axis, part)
                row_recs[f"{side}/{part}"] = r
                input_recs[f"{D.out_name(stem)}:{side}/{part}"] = r
                resolved[f"{D.out_name(stem)}:{side}/{part}"] = resolve_cache(r["file"], REG.CACHE)
                if ddoc["input_cache_sha256"].get(f"{side}/{part}") != r["sha256"]:
                    raise SystemExit(f"REFUSING: {rel} read other bytes for {side}/{part}")
        for part in ("threshold_calib", "eval"):              # the same items, labelled alike, through both models
            z_pre, z_post = z[("pre", part)], z[("post", part)]
            if not np.array_equal(z_pre["item_ids"], z_post["item_ids"]) or not np.array_equal(z_pre["labels"], z_post["labels"]):
                raise SystemExit(f"REFUSING: {stem} {part}: item ids or labels differ pre/post")
        verification = REG.verification(row_recs)
        if verification["flag"] != ddoc["input_verification"]["flag"]:
            raise SystemExit(f"REFUSING: {stem}'s D-170 flag differs from {rel}'s")
        y_tr = z[("pre", "probe_train")]["labels"].astype(bool)
        ncal = ~z[("pre", "threshold_calib")]["labels"].astype(bool)
        y_ev = z[("pre", "eval")]["labels"].astype(bool)
        masks = {"ncal": ncal, "neg": ~y_ev, "pos": y_ev}
        n_cal, n_ev, n_pos = int(ncal.sum()), int((~y_ev).sum()), int(y_ev.sum())
        pooled = [str(i) for i in z[("pre", "threshold_calib")]["item_ids"][ncal]] + \
                 [str(i) for i in z[("pre", "eval")]["item_ids"][~y_ev]]
        if hashlib.sha256("\n".join(pooled).encode()).hexdigest() != ddoc["config"]["pooled_negatives"]["item_ids_sha256"] \
                or n_pos != ddoc["config"]["n_eval_positives"]:
            raise SystemExit(f"REFUSING: {stem}'s pooled negatives or eval positives are not {rel}'s")
        pid = sha256_bytes("\n".join(str(i) for i in z[("pre", "eval")]["item_ids"][y_ev]).encode("utf-8"))
        if positives_id.setdefault(axis, pid) != pid:
            raise SystemExit(f"REFUSING: {stem}'s eval positives are not the other {axis} runs' (the counts would not pair by item)")
        if n_pos not in counts:
            counts[n_pos] = P.counts_matrix(n_pos, args.n_boot, D.ITEM_BOOT_SEED["eval_positives"])
        CNT = counts[n_pos]
        draws = P.repartition_draws(n_cal, n_ev, args.n_draws, P.AXIS_DRAW_SEED[axis])
        layer_sets = {pset: quoted_layers(ddoc["pair_summary"], pset, L_sel) for pset in D.PROBE_SETS}
        want_blocks += sum(c["layer"] in layer_sets.get(c["probe_set"], ()) for c in ddoc["cells"])
        out_rows[stem] = {"registry_pair": reg_stem, "selection_of_pair": D.LORA_SELECTION_PAIR, "axis": axis,
                          "arm": ARM_OF[stem], "headline_layer": L_sel, "selected_C": C_sel, "depth_record": rel,
                          "depth_record_1pct": b["rel1"], "repair_record": b["repair_rel"], "n_positives": n_pos,
                          "eval_positives_item_ids_sha256": pid,
                          **{f"{side}_{kk}": cfg[f"{side}_{kk}"] for side in ("pre", "post") for kk in ("repo", "rev")},
                          "input_verification": verification, "under_check": under_check(stem),
                          "input_cache_source": {k: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"],
                                                     "resolved_path": path_text(resolved[f"{D.out_name(stem)}:{k}"])}
                                                 for k, r in row_recs.items()},
                          "probe_sets": {}}
        head = {pset: ddoc["pair_summary"][pset]["item_level_D156"]["headline_selected_layer"] for pset in D.PROBE_SETS}
        if any(h["layer"] != L_sel for h in head.values()):
            raise SystemExit(f"REFUSING: {rel}'s headline layer is not the selection's L{L_sel}")
        for li in sorted(set().union(*layer_sets.values())):
            k = f"layer_{li}"
            X = {(s, p): z[(s, p)][k] for s in ("pre", "post") for p in ("probe_train", "threshold_calib", "eval")}
            for pset in D.PROBE_SETS:
                if li not in layer_sets[pset]:
                    continue
                C = C_sel if pset == "selected" else 1.0
                where = f"{stem} {pset} L{li}"
                cell = dcells.get((li, pset))
                sc = probe_scores(X, y_tr, masks, seeds, C, pset)
                report, arr, reps = anchored_replicates(sc, draws, target, cell, CNT, where)
                n_anchor += 1
                n_ones += 1
                if li == L_sel:
                    if any(T1.canon(report[q]) != T1.canon(head[pset][q]) for q in QUANT):
                        raise SystemExit(f"REFUSING: {where}: the TPR blocks are not {rel}'s pair_summary headline")
                    n_headline += 1
                    at_lstar[(stem, pset)] = (arr["delta_tpr"], report["delta_tpr"]["frozen"], reps["delta_tpr"])
                hashes = {q: sha256_bytes(reps[q].astype("<f8").tobytes()) for q in QUANT}
                n_shared += check_shared_pre(shared_pre, (cfg["pre_repo"], cfg["pre_rev"], axis, C, li, pset), stem,
                                             hashes["tpr_pre"], where)
                op = cell["operating_point"]
                out_rows[stem]["probe_sets"].setdefault(pset, {"probe_C": C, "layers": {}})["layers"][str(li)] = {
                    "is_headline": li == L_sel,
                    **{q: {"frozen": op[q]["frozen"], "mean_D156": op[q]["mean"], "band95_D156": op[q]["band95"],
                           "combined": P.ci(reps[q]), "replicates_sha256": hashes[q]} for q in QUANT}}
                bq = out_rows[stem]["probe_sets"][pset]["layers"][str(li)]["delta_tpr"]
                print(f"  {stem:<30} {pset:<8} L{li:<3} delta TPR frozen {bq['frozen']:+.4f}; D-156 band "
                      f"[{bq['band95_D156'][0]:+.4f}, {bq['band95_D156'][1]:+.4f}]; combined "
                      f"[{bq['combined']['ci95'][0]:+.4f}, {bq['combined']['ci95'][1]:+.4f}]", flush=True)

    blocks = [blk for r in out_rows.values() for ps in r["probe_sets"].values() for blk in ps["layers"].values()]
    n_stored, n_head_stored = len(blocks), sum(blk["is_headline"] is True for blk in blocks)
    if not (n_anchor == n_ones == n_stored == want_blocks) or not (n_headline == n_head_stored == len(run_rows) * len(D.PROBE_SETS)):
        raise SystemExit(f"REFUSING to write: {n_anchor} anchored, {n_ones} ones-checked, {n_stored} stored and "
                         f"{n_headline} headline blocks ({n_head_stored} stored); the depth records hold {want_blocks} "
                         f"cells at the quoted layers")
    if len(run_rows) > 1 and n_shared == 0:
        raise SystemExit("REFUSING to write: no shared-pre check ran, though every run shares the pre side (D-224)")
    gaps, n_gaps_left_out = between_arm_gaps(dmg_doc, dmg_rel, at_lstar, run_rows, C_sel, L_sel)
    if registered and (n_gaps_left_out or not gaps):
        raise SystemExit(f"REFUSING to write: {len(gaps)} between-arm gaps, {n_gaps_left_out} of {dmg_rel}'s left out")
    for gr in gaps:
        bq = gr["delta_tpr_gap"]
        print(f"  gap {gr['safety_run']:<22} - {gr['general_run']:<22} {gr['probe_set']:<8} L{L_sel}: frozen "
              f"{bq['frozen']:+.4f}; D-156 band [{bq['band95_D156'][0]:+.4f}, {bq['band95_D156'][1]:+.4f}]; combined "
              f"[{bq['combined']['ci95'][0]:+.4f}, {bq['combined']['ci95'][1]:+.4f}]", flush=True)
    input_sha = {k: r["sha256"] for k, r in input_recs.items()}
    idc, id_sources = identity_config(list(input_sha))      # re-reads the 1% depth records: a change refuses below
    read_now = {rel: h for rel, h in _RESULT_READS.items()}
    first_read = {sel_rel: sel_sha, dmg_rel: dmg_sha}
    for b in bound.values():
        first_read.update({b["rel"]: b["sha"], b["rel1"]: b["sha1"], b["repair_rel"]: b["repair_sha"]})
    changed = [n for n, a, b in (("code", P.code_block(CODE_FILES)["sha256"], code0["sha256"]),
                                 ("registry sources", D.sources_now(), reg_sources)) if a != b]
    changed += [rel for rel, h in first_read.items() if read_now.get(rel) != h]
    changed += [rel for rel, h in read_now.items() if sha256_file(ROOT / rel) != h]
    changed += [rel_path(p) for p, h in man_sha.items() if sha256_file(p) != h]
    caches = {r["file"]: (resolved[k], r["sha256"]) for k, r in input_recs.items()}
    changed += [f"cache {f}" for f, (p, h) in caches.items() if resolve_cache(f, REG.CACHE) != p or sha256_file(p) != h]
    if changed:
        raise SystemExit(f"REFUSING to write: changed during the run: {changed}")
    payload ={**provenance_block("table3_tpr_d282"),
               "preregistration": "D-281 (1), D-282 (3) (the other TPR columns when their tables are rebuilt); D-287 "
                                  "(the supplement); D-224 (the LoRA depth records), D-289 (4d A)" +
                                  ("; D-248 (5%)" if args.fpr5 else ""),
               "code": code0, "registered_parameters": registered, "blas_threads": threads, "note": T1.NOTE,
               "implementation_choices": list(CHOICES),
               "config": {"target_fpr": target, "seeds": seeds, "n_draws": args.n_draws, "n_boot": args.n_boot,
                          "runs": run_rows, "arms": {k: v for k, v in POP.LORA_ARMS.items()},
                          "level": "item_level_D156 (the cluster level keeps its D-156 band only, D-282 (2))",
                          "layers": "every layer where the run's depth record's D-180 rows quote TPR (its pair_summary, "
                                    "item level) and the headline layer L*",
                          "table_3": {"builder": "src/tables_d156.py table_3 (tab:intent-d156)",
                                      "column": "delta TPR at L*, selected C, item level: the D-156 mean with this "
                                                "record's combined interval (as Table I's column, D-287)",
                                      "under_check": "rows with under_check carry a marker and a caption note (D-349)",
                                      "between_arms": "not printed (supplement only)"},
                          "between_arms": {"file": dmg_rel, "sha256": dmg_sha, "key": "between_arms_4a",
                                           "quantity": "delta_tpr_gap (safety minus general at matched dose), item level",
                                           "replicate": "the safety run's delta-TPR replicate r minus the general run's "
                                                        "(the same draw d_r and counts row r)"},
                          "cache_dirs": {"GUARDRAIL_CACHE_DIRS": cache_dirs, "registry_cache": path_text(REG.CACHE)},
                          "draw_seeds": P.AXIS_DRAW_SEED, "draw_of_replicate": "d_r = 1 + ((r - 1) mod n_draws)",
                          "counts": "paired.counts_matrix(n positives, n_boot, depth_d156.ITEM_BOOT_SEED['eval_positives']) "
                                    "(depth_d156's CNT_P, a function of n positives); every run has the same eval "
                                    "positives in the same order, so the matrix pairs by item across runs",
                          "counts_seed": D.ITEM_BOOT_SEED["eval_positives"],
                          "eval_positives_item_ids_sha256_per_axis": positives_id,
                          "replicates_sha256_of": "float64 little-endian bytes, r = 1..n_boot in order",
                          "selection_file": sel_rel, "selection_sha256": sel_sha, "selection_of_pair": D.LORA_SELECTION_PAIR,
                          "selected_C": C_sel, "selected_layer": L_sel,
                          "manifests": [{"file": rel_path(p), "sha256": h} for p, h in man_sha.items()],
                          "input_registry_sources_sha256": reg_sources,
                          "identity_sources_sha256": id_sources, **idc},
               "anchor": {"depth_records_sha256": {b["rel"]: b["sha"] for b in bound.values()},
                          "depth_records_1pct_sha256": {b["rel1"]: b["sha1"] for b in bound.values()},
                          "repair_records_sha256": {b["repair_rel"]: b["repair_sha"] for b in bound.values()},
                          "damage_record": {"file": dmg_rel, "sha256": dmg_sha, "n_between_arm_gaps_identical": len(gaps),
                                            "n_between_arm_gaps_left_out": n_gaps_left_out,
                                            "kind": "each between-arm delta-TPR gap's band summary recomputed from the two "
                                                    "anchored per-draw arrays equals the record's as canonical JSON; every "
                                                    "depth record read names this record and its bytes (twomoment_check)"},
                          "n_blocks_identical": n_anchor, "n_headline_blocks_identical": n_headline,
                          "kind": "each block's item-level operating point equals the depth record's cell as canonical "
                                  "JSON text, and its per-draw ratio_to_target array hashes to the cell's "
                                  "per_draw_ratio_to_target_sha256 (the draw order); at L* the TPR blocks equal the "
                                  "pair_summary headline's; the 1% depth record is the one the repair record (Table "
                                  "III's frozen column) anchors to" +
                                  ("; the 5% record is self-anchored on it (fpr5_self_anchor)" if args.fpr5 else "")},
               "ones_check": {"n_blocks": n_ones, "kind": "counts of all ones give back the D-156 per-draw TPR arrays bit for bit"},
               "shared_pre_check": {"n_blocks": n_shared, "kind": "runs sharing the pre model and revision, axis, C, "
                                                                  "layer and probe set give identical TPR-pre replicates"},
               "under_check": {"decision": "D-344", "runs": [s for s in run_rows if under_check(s)],
                               "between_arm_gaps": [f"{g['probe_set']} {g['safety_run']} - {g['general_run']}"
                                                    for g in gaps if g["under_check"]],
                               "status": D344["status"]},
               "population": {"declared_in": "src/populations.py",
                              "group": "DEPTH_D156_LORA_D282_FPR5" if args.fpr5 else "DEPTH_D156_LORA_D282",
                              "replaces": None},
               "input_verification": {**REG.verification(input_recs),
                                      "n_rows_flagged": sum(r["input_verification"]["flag"] is not None for r in out_rows.values())},
               "input_cache_sha256": input_sha, "rows": out_rows, "between_arms_4a_delta_tpr_gap": gaps}
    digest = write_results(out_path, payload, exclusive=True)
    print(f"Written: {rel_path(out_path)}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
