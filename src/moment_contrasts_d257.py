"""The moment contrasts re-issued under the author's corrected floor rule (D-256 (2)) with the D-117 layer as the
primary repair metric (D-256 (3), pre-registered in D-257), and the paired floor test of the LoRA runs (D-256 (5)).

INPUTS, each named by a committed manifest and checked as in D-235:
  PRIMARY     results/moment_d156_<core>_d117.json (the seven released pairs and the nine LoRA runs; repair_d156.py
              --d117-layer): per draw, m_arm(d) = |log(FPR_arm,d(L*)/target)| at the pair's D-117 selected layer L*, and
              the floor m_floor(d) = |log(FPR_pre,d(L*)/target)|. Named by logs/moment_d156_d117.sha256.
  CONTINUITY  the per-draw mean over layers of the same metric: the seven results/moment_d156_<core>_r2.json
              (logs/moment_d156_r2.sha256) and the nine results/moment_d156_<run>.json (logs/moment_ext_d156_20260928.sha256).
Index 0 of every array is the frozen split, 1..n the D-156 (item level) or D-164 (cluster level) draws.

PER PAIR (or LoRA run), probe set and level, on each metric:
  CONTRASTS within draws, D-156 mean and band (D-235): restandardise - frozen, restandardise - paired_align,
    restandardise - refit_oracle, paired_align - frozen; mean_only's share of restandardise's improvement (a ratio
    beside its denominator, UNDEFINED unless the denominator's band lies wholly above 0; D-236).
  PAIRED FLOOR TEST (D-256 (2), D-257 (3)): D_arm(d) = m_arm(d) - m_floor(d) on the same partition. "Damage detected"
    (frozen): the band of D_frozen lies wholly above 0. An arm is "at floor" when the band of D_arm includes 0,
    reported "at most X above floor" (X the band's upper end); a band wholly above 0 is "above floor by [lo, hi]",
    wholly below 0 "below floor by [lo, hi]".
  THE DRAFT'S CLAIM (D-237 (1)) on each metric: paired_align's mean error exceeds frozen's AND the band of
    paired_align - frozen lies wholly above 0; reproduced overall when at least 4 of 7 pairs are.
POOLED (released pairs only; rows all 7, unflagged, unseen; D-237 (2)): the equal-weight mean of per-pair D-156 means
  with a pair-cluster bootstrap 95% interval (B = 5000, seed 520260925; point only below 3 pairs; D-177 (b)); the
  pooled D_arm per arm; the counts of pairs with damage detected and of pairs at floor per arm.
LoRA RUNS (D-256 (5), D-257 (8)): the same per-run tests as their own rows, general (alpaca) and safety (HH-harmless,
  PKU-SafeRLHF) groups, never pooled with the released pairs and not bootstrapped (the runs share one pre model).
JUDGED only at selected C and the item level; C=1 and the D-164 cluster level beside ('..._not_judged').
CROSS-CHECK: at L*, each arm's per-draw array must reproduce the record of record's per-layer band summary of
abs_log_ratio_to_target exactly (the d117 record's cells equal that record's cells, D-257 (10)).
Writes results/moment_contrasts_d156_r2.json (never overwritten; 3cc370d3... stays as written); a test writes outside
results/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moment_contrasts_d156 as MC  # noqa: E402
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
from provenance import ensure_utf8_stdout, provenance_block, sha256_bytes, sha256_file, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FINAL_OUT = RESULTS / "moment_contrasts_d156_r2.json"
MAN_D117 = ROOT / "logs" / "moment_d156_d117.sha256"
MAN_R2 = ROOT / "logs" / "moment_d156_r2.sha256"
MAN_EXT = ROOT / "logs" / "moment_ext_d156_20260928.sha256"
# D-271 (4), D-273: the 5% arm (the seven released pairs only; 4e has no LoRA moment records at 5%), a sensitivity arm
FINAL_OUT_5 = RESULTS / "moment_contrasts_d156_r2_fpr5.json"
MAN_D117_5 = ROOT / "logs" / "moment_d156_d117_fpr5.sha256"
MAN_R2_5 = ROOT / "logs" / "moment_d156_r2_fpr5.sha256"
TARGET_5 = 0.05
CODE_FILES = ["src/moment_contrasts_d257.py", "src/moment_contrasts_d156.py", "src/paired.py", "src/populations.py",
              "src/provenance.py", "src/superseded_provenance.py"]
LEVELS, PROBE_SETS, PRIMARY, FLOOR, ARMS = MC.LEVELS, MC.PROBE_SETS, MC.PRIMARY, MC.FLOOR, MC.ARMS
DIFFS, SHARE, EARLIER, SAME_CODE = MC.DIFFS, MC.SHARE, MC.EARLIER, MC.SAME_CODE
B, SEED, MIN_PAIRS, CLAIM_MIN = MC.B, MC.SEED, MC.MIN_PAIRS, MC.CLAIM_MIN
METRICS = ("d117_layer", "mean_over_layers")          # primary, continuity (D-257 (1), (7))
LORA_GROUPS = {"general": list(POP.LORA_ARMS["general (alpaca)"]),
               "safety": [*POP.LORA_ARMS["safety (hh-harmless)"], *POP.LORA_ARMS["safety (pku-saferlhf)"]]}


# the D-235 contrasts record this one re-issues under the corrected rule (committed at 9c46d04, D-247); never touched
PRIOR_CONTRASTS = ("results/moment_contrasts_d156.json", "3cc370d325e4f5ef6f6895cd6479b4141f7603ec74abed77c1d2d9966ba4d6e5")


def manifest(path: Path) -> tuple[dict[str, str], str]:
    """A sha256sum manifest as {path: sha256}; refuses a path named twice (MC.manifest keeps the last)."""
    if not path.is_file():
        raise SystemExit(f"REFUSING: no manifest {path}")
    b = path.read_bytes()
    man: dict[str, str] = {}
    for line in b.decode("utf-8").splitlines():
        if line.strip():
            h, f = line.split(maxsplit=1)
            f = f.strip().lstrip("*")
            if f in man:
                raise SystemExit(f"REFUSING: {path.name} names {f} twice")
            man[f] = h
    return man, sha256_bytes(b)


def arrays(blocks: dict[str, Any], rel: str, n: int, summaries: dict[str, Any]) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Per (probe set, level), every arm's and the floor's per-draw array, each hashing to its recorded sha256 and
    reproducing the record's own summary of it (as D-235's loader requires)."""
    m: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for ps in PROBE_SETS:
        for lvl in LEVELS:
            blk = blocks[ps][lvl]
            if set(blk) != {*ARMS, FLOOR}:
                raise SystemExit(f"REFUSING: {rel} {ps} {lvl}: arms {sorted(blk)}")
            m[(ps, lvl)] = {}
            for name, v in blk.items():
                a = np.asarray(v["values"], dtype="<f8")
                if a.shape != (n,) or sha256_bytes(a.tobytes()) != v["sha256"]:
                    raise SystemExit(f"REFUSING: {rel} {ps} {lvl} {name}: not n_draws + 1 values hashing to their sha256")
                if json.dumps(MC.summary(a), sort_keys=True) != json.dumps(summaries[ps][lvl][name], sort_keys=True):
                    raise SystemExit(f"REFUSING: {rel} {ps} {lvl} {name}: the per-draw array does not reproduce the record's summary")
                m[(ps, lvl)][name] = a
    return m


def load(rel: str, man: dict[str, str], man_name: str, inputs: dict[str, str]) -> tuple[dict[str, Any], str]:
    b = (ROOT / rel).read_bytes()
    if man.get(rel) != sha256_bytes(b):
        raise SystemExit(f"REFUSING: {rel} is not the file {man_name} names")
    inputs[rel] = sha256_bytes(b)
    return json.loads(b.decode("utf-8")), sha256_bytes(b)


def load_unit(stem: str, family: str, core: str, mans: dict[str, tuple[dict[str, str], str]],
              inputs: dict[str, str], sfx: str = "", target: float = 0.01) -> dict[str, Any]:
    """One released pair (family moment) or LoRA run (moment_ext): both metrics, checked. sfx '_fpr5' reads the 5% arm
    (D-273): the _r2_fpr5 record and its D-117-layer run, both at target 0.05."""
    cont_rel = f"results/moment_d156_{core}_r2{sfx}.json" if family == "moment" else f"results/moment_d156_{core}.json"
    cont_man = "r2" if family == "moment" else "ext"
    d117_rel = f"results/moment_d156_{core}_d117{sfx}.json"
    dc, dc_sha = load(cont_rel, mans[cont_man][0], mans[cont_man][1], inputs)
    d1, _ = load(d117_rel, mans["d117"][0], mans["d117"][1], inputs)
    for rel, d in ((cont_rel, dc), (d117_rel, d1)):
        if (d["config"]["family"], d["config"]["pair"], d.get("registered_parameters"), d["config"].get("target_fpr")) != \
                (family, stem, True, target):
            raise SystemExit(f"REFUSING: {rel} is not the registered {family} run of {stem} at target {target}")
    chk = d1.get("d117_check") or {}
    L = d1["config"]["selected_layer"]
    if (chk.get("record_of_record"), chk.get("sha256"), chk.get("n_cells_identical"), chk.get("layer"), d1["config"]["layers"]) != \
            (cont_rel, dc_sha, len(PROBE_SETS), L, [L]) or dc["config"]["selected_layer"] != L:
        raise SystemExit(f"REFUSING: {d117_rel} is not the D-117-layer run of {cont_rel} (D-257 (10))")
    if "pair_per_draw_mean_abs_log_over_layers" not in dc or "per_draw_at_d117_layer" not in d1:
        raise SystemExit(f"REFUSING: {cont_rel} or its D-117-layer record has no per-draw arrays")
    n = int(dc["config"]["n_draws"]) + 1
    m_cont = arrays(dc["pair_per_draw_mean_abs_log_over_layers"], cont_rel, n, dc["pair_summary_mean_abs_log_over_layers"])
    m_d117 = arrays(d1["per_draw_at_d117_layer"], d117_rel, n, d1["summary_at_d117_layer"])
    # the D-117-layer record's cells equal the record of record's cells at L*, re-checked here (not taken on d117_check)
    canon = lambda x: json.dumps(json.loads(json.dumps(x)), sort_keys=True)  # noqa: E731
    if canon(d1["cells"]) != canon([c for c in dc["cells"] if c["layer"] == L]):
        raise SystemExit(f"REFUSING: {d117_rel}'s cells are not {cont_rel}'s at L{L}")
    # the floor at L* against the depth file both records are anchored to (its operating point's floor_abs_log)
    dep_rel, dep_sha = d1["anchor"]["file"], d1["anchor"]["sha256"]
    dep_b = (ROOT / dep_rel).read_bytes()
    if sha256_bytes(dep_b) != dep_sha or dc["anchor"]["sha256"] != dep_sha:
        raise SystemExit(f"REFUSING: {dep_rel} is not the depth file both records of {core} are anchored to")
    inputs[dep_rel] = dep_sha
    dcell = {(c["layer"], c["probe_set"]): c for c in json.loads(dep_b.decode("utf-8"))["cells"]}
    # cross-check: at L*, each arm's per-draw array reproduces the record of record's per-layer band summary exactly
    cells = {(c["layer"], c["probe_set"]): c for c in dc["cells"]}
    for (ps, lvl), blk in m_d117.items():
        for arm in ARMS:
            want = cells[(L, ps)][lvl][arm]["abs_log_ratio_to_target"]
            if json.dumps(MC.summary(blk[arm]), sort_keys=True) != json.dumps(want, sort_keys=True):
                raise SystemExit(f"REFUSING: {core} L{L} {ps} {lvl} {arm}: the per-draw array is not {cont_rel}'s cell")
        dop = dcell[(L, ps)]["operating_point"] if lvl == LEVELS[0] else dcell[(L, ps)][LEVELS[1]]["operating_point"]
        if json.dumps(MC.summary(blk[FLOOR]), sort_keys=True) != json.dumps(dop["floor_abs_log"], sort_keys=True):
            raise SystemExit(f"REFUSING: {core} L{L} {ps} {lvl}: the floor array is not {dep_rel}'s floor_abs_log")
    return {"stem": stem, "core": core, "family": family, "records": {"d117_layer": d117_rel,
            "mean_over_layers": cont_rel}, "flag": dc["input_verification"]["flag"], "d117_layer": L,
            "n_layers": len(dc["config"]["layers"]), "m": {"d117_layer": m_d117, "mean_over_layers": m_cont},
            "code": {f: d1["code"]["sha256"].get(f) for f in SAME_CODE}}


def floor_test(d: dict[str, Any]) -> dict[str, Any]:
    lo, hi = d["band95"]
    if lo > 0:
        return {"verdict": "above floor", "by": [lo, hi], "text": f"above floor by [{lo:.3f}, {hi:.3f}]"}
    if hi < 0:
        return {"verdict": "below floor", "by": [lo, hi], "text": f"below floor by [{lo:.3f}, {hi:.3f}]"}
    return {"verdict": "at floor", "at_most_above_floor": hi, "text": f"at most {hi:.3f} above floor"}


def unit_block(m: dict[str, np.ndarray], judged: bool) -> dict[str, Any]:
    nj = "" if judged else "_not_judged"
    out: dict[str, Any] = {"judged": judged}
    for k, (a, b) in DIFFS.items():
        s = MC.summary(m[a] - m[b])
        out[k] = {**s, f"band_excludes_0{nj}": MC.excludes_zero(s)}
    num, den = m["frozen"] - m["mean_only"], m["frozen"] - m["restandardise"]
    den_s = MC.summary(den)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = num / den
    fin = share[1:][np.isfinite(share[1:])]
    out[SHARE] = {**MC.summary(share), "median": float(np.median(fin)) if fin.size else None,
                  "denominator_restandardise_improvement": den_s,
                  "numerator_mean_only_improvement": MC.summary(num), "undefined": MC.sign_of(den_s) != "above 0",
                  "note": "a ratio beside its denominator, never a percentage; UNDEFINED unless the denominator's band "
                          "lies wholly above 0 (D-236)"}
    out["floor"] = MC.summary(m[FLOOR])
    out[f"paired_floor_test{nj}"] = {}
    for arm in ARMS:
        s = MC.summary(m[arm] - m[FLOOR])
        out[f"paired_floor_test{nj}"][arm] = {"arm_minus_floor": s, **floor_test(s)}
    out[f"damage_detected{nj}"] = out[f"paired_floor_test{nj}"]["frozen"]["verdict"] == "above floor"
    if judged:
        pa, fz = MC.summary(m["paired_align"])["mean"], MC.summary(m["frozen"])["mean"]
        above = bool(out["paired_align_minus_frozen"]["band95"][0] > 0)
        out.update({"paired_align_error_mean": pa, "frozen_error_mean": fz, "paired_align_error_exceeds_frozen": bool(pa > fz),
                    "paired_align_minus_frozen_band_wholly_above_0": above,
                    "worse_than_no_repair_paired_align": bool(pa > fz) and above})
    return out


def pooled(rows: dict[str, dict[str, Any]], stems: list[str], judged: bool) -> dict[str, Any]:
    k = len(stems)
    if k == 0:
        return {"pairs": [], "n_pairs": 0}
    nj = "" if judged else "_not_judged"
    idx = np.random.default_rng(SEED).integers(0, k, size=(B, k)) if k >= MIN_PAIRS else None
    ci = lambda v: P.ci(v[idx].mean(axis=1)) if idx is not None else {"ci95": None}  # noqa: E731
    res: dict[str, Any] = {"pairs": stems, "n_pairs": k, "judged": judged,
                           "interval": f"pair-cluster bootstrap B={B}, seed {SEED}" if idx is not None else
                                       f"point only (fewer than {MIN_PAIRS} pairs, D-177 (b))"}
    for key in DIFFS:
        v = np.array([rows[s][key]["mean"] for s in stems])
        res[key] = {"mean": float(v.mean()), **ci(v)}
    num = np.array([rows[s][SHARE]["numerator_mean_only_improvement"]["mean"] for s in stems])
    den = np.array([rows[s][SHARE]["denominator_restandardise_improvement"]["mean"] for s in stems])
    den_ci = ci(den)
    den_ok = den_ci["ci95"][0] > 0 if den_ci["ci95"] is not None else float(den.mean()) > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        res[SHARE] = {"ratio_of_pooled_means": float(num.mean() / den.mean()), "pooled_denominator": {"mean": float(den.mean()), **den_ci},
                      **(P.ci(num[idx].mean(axis=1) / den[idx].mean(axis=1)) if idx is not None else {"ci95": None}),
                      "undefined": not den_ok}
    res[f"pooled_arm_minus_floor{nj}"] = {}
    for arm in ARMS:
        v = np.array([rows[s][f"paired_floor_test{nj}"][arm]["arm_minus_floor"]["mean"] for s in stems])
        c_ = ci(v)
        # D-257 (5): the pooled paired floor test, judged on the pair-cluster interval (none below 3 pairs)
        res[f"pooled_arm_minus_floor{nj}"][arm] = {"mean": float(v.mean()), **c_,
                                                    **(floor_test({"band95": c_["ci95"]}) if c_["ci95"] is not None else {"verdict": None})}
    res[f"pooled_damage_detected{nj}"] = res[f"pooled_arm_minus_floor{nj}"]["frozen"]["verdict"] == "above floor"
    res[f"n_pairs_damage_detected{nj}"] = int(sum(rows[s][f"damage_detected{nj}"] for s in stems))
    res[f"n_pairs_at_floor{nj}"] = {arm: int(sum(rows[s][f"paired_floor_test{nj}"][arm]["verdict"] == "at floor" for s in stems))
                                   for arm in ARMS}
    return res


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="D-257: moment contrasts at the D-117 layer under the paired floor rule")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fpr5", action="store_true",
                    help="D-271 (4), D-273: the 5%% arm (seven released pairs, no LoRA rows), a sensitivity arm; writes "
                         "moment_contrasts_d156_r2_fpr5.json")
    args = ap.parse_args()
    final_out = FINAL_OUT_5 if args.fpr5 else FINAL_OUT
    sfx, target = ("_fpr5", TARGET_5) if args.fpr5 else ("", 0.01)
    out = Path(args.out).resolve() if args.out else final_out.resolve()
    final = out == final_out.resolve()
    if out.exists():
        raise SystemExit(f"REFUSING: {out} exists; never overwritten")
    if not final and out.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test output must not be under results/")
    code0 = P.code_block(CODE_FILES)
    inputs: dict[str, str] = {}
    mans: dict[str, tuple[dict[str, str], str]] = {}
    man_list = (("d117", MAN_D117_5), ("r2", MAN_R2_5)) if args.fpr5 else (("d117", MAN_D117), ("r2", MAN_R2), ("ext", MAN_EXT))
    for key, path in man_list:
        man, msha = manifest(path)
        inputs[str(path.relative_to(ROOT)).replace("\\", "/")] = msha
        mans[key] = (man, path.name)
    if not args.fpr5:
        if sha256_file(ROOT / PRIOR_CONTRASTS[0]) != PRIOR_CONTRASTS[1]:
            raise SystemExit(f"REFUSING: {PRIOR_CONTRASTS[0]} is not the D-235 record this one re-issues")
        inputs[PRIOR_CONTRASTS[0]] = PRIOR_CONTRASTS[1]
    stems = list(POP.UPDATE_PAIRS_MOMENT)
    units = {s: load_unit(s, "moment", MC.core_of(s), mans, inputs, sfx, target) for s in stems}
    # D-273 (2): no LoRA rows at 5% (4e ran no LoRA moment record at 5%)
    lora = {} if args.fpr5 else {r: load_unit(f"moment_{r}", "moment_ext", r, mans, inputs)
                                 for g in LORA_GROUPS.values() for r in g}
    codes = {json.dumps(u["code"], sort_keys=True) for u in [*units.values(), *lora.values()]}
    if len(codes) != 1 or any(v is None for v in units[stems[0]]["code"].values()):
        raise SystemExit(f"REFUSING: the D-117-layer records do not share one {SAME_CODE}: {sorted(codes)}")
    unflagged = [s for s in stems if units[s]["flag"] is None]
    unseen = [s for s in stems if MC.core_of(s) not in EARLIER]
    result: dict[str, Any] = {}
    for metric in METRICS:
        per_pair = {s: {"record": u["records"][metric], "flag": u["flag"], "d117_layer": u["d117_layer"],
                        **{f"{ps}|{lvl}": unit_block(u["m"][metric][(ps, lvl)], (ps, lvl) == PRIMARY)
                           for ps in PROBE_SETS for lvl in LEVELS}} for s, u in units.items()}
        rows = {}
        for ps in PROBE_SETS:
            for lvl in LEVELS:
                c, j = f"{ps}|{lvl}", (ps, lvl) == PRIMARY
                rows[c] = {"all_pairs": pooled({s: per_pair[s][c] for s in stems}, stems, j),
                           "unflagged_pairs": pooled({s: per_pair[s][c] for s in unflagged}, unflagged, j),
                           "unseen_pairs": pooled({s: per_pair[s][c] for s in unseen}, unseen, j)}
        key = "|".join(PRIMARY)
        worse = [s for s in stems if per_pair[s][key]["worse_than_no_repair_paired_align"]]
        damaged = [s for s in stems if per_pair[s][key]["damage_detected"]]
        split = lambda xs: {"n_among_unflagged": len([s for s in xs if s in unflagged]), "n_unflagged": len(unflagged),  # noqa: E731
                            "n_among_unseen": len([s for s in xs if s in unseen]), "n_unseen": len(unseen)}
        at_floor = {arm: [s for s in stems if per_pair[s][key]["paired_floor_test"][arm]["verdict"] == "at floor"] for arm in ARMS}
        n_w, sw = len(worse), split(worse)
        statements = {
            "damage_detected": {"pairs": damaged, "n": len(damaged), "of": len(stems), **split(damaged),
                                "text": f"damage detected (frozen - floor band wholly above 0) in {len(damaged)} of {len(stems)} pairs"},
            "at_floor_per_arm": {arm: {"pairs": ps_, "n": len(ps_), "of": len(stems), **split(ps_)} for arm, ps_ in at_floor.items()},
            "draft_claim": {"test": "D-237 (1) on this metric; reproduced overall when at least 4 of 7 pairs are",
                            "pairs_worse": worse, "n_worse": n_w, "of": len(stems), **sw, "reproduced": n_w >= CLAIM_MIN,
                            "consequence": (f"5% sensitivity arm, beside the 1% record only (D-248, D-273): {n_w} of {len(stems)} "
                                            f"pairs worse ({sw['n_among_unflagged']} of {sw['n_unflagged']} unflagged, "
                                            f"{sw['n_among_unseen']} of {sw['n_unseen']} unseen)") if args.fpr5 else
                                           (f"reproduced: {n_w} of {len(stems)} pairs worse (the draft states 4 of 7); "
                                            f"{sw['n_among_unflagged']} of {sw['n_unflagged']} unflagged, "
                                            f"{sw['n_among_unseen']} of {sw['n_unseen']} unseen pairs worse")
                                           if n_w >= CLAIM_MIN else
                                           ((f"WITHDRAWN (D-235 (7), D-247): " if metric == "mean_over_layers" else
                                             "not reproduced at the D-117 layer: ") + f"{n_w} of {len(stems)} pairs worse, fewer than 4 "
                                            f"({sw['n_among_unflagged']} of {sw['n_unflagged']} unflagged, "
                                            f"{sw['n_among_unseen']} of {sw['n_unseen']} unseen pairs worse)")}}
        lora_rows = {} if args.fpr5 else {
            g: {r: {"record": lora[r]["records"][metric], "flag": lora[r]["flag"], "d117_layer": lora[r]["d117_layer"],
                    **{f"{ps}|{lvl}": unit_block(lora[r]["m"][metric][(ps, lvl)], (ps, lvl) == PRIMARY)
                       for ps in PROBE_SETS for lvl in LEVELS}} for r in runs} for g, runs in LORA_GROUPS.items()}
        lora_counts = {g: {"restandardise_above_floor": sum(v[key]["paired_floor_test"]["restandardise"]["verdict"] == "above floor"
                                                            for v in rr.values()),
                           "damage_detected": sum(v[key]["damage_detected"] for v in rr.values()), "n_runs": len(rr)}
                       for g, rr in lora_rows.items()}
        result[metric] = {"per_pair": per_pair, "pooled": rows, "statements": statements, "lora_runs": lora_rows,
                          "lora_counts": lora_counts}
        print(f"== {metric}: {statements['damage_detected']['text']}; draft claim {len(worse)} of {len(stems)} worse; "
              f"LoRA {lora_counts}", flush=True)
        for s in stems:
            r = per_pair[s][key]
            where = f"L{units[s]['d117_layer']:>2}" if metric == "d117_layer" else "all"
            print(f"  {MC.core_of(s):22s} {where:>3} frozen: {r['paired_floor_test']['frozen']['text']}; "
                  f"restandardise: {r['paired_floor_test']['restandardise']['text']}; "
                  f"restd-frozen {r['restandardise_minus_frozen']['mean']:+.3f}", flush=True)
    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"] or any(sha256_file(ROOT / f) != h for f, h in inputs.items()):
        raise SystemExit("REFUSING to write: code or an input changed during the run")
    payload = {**provenance_block("moment_contrasts_d156_r2" + sfx), "code": code0,
               "preregistration": "D-256 (2), (3), (5); D-257 (the D-117-layer primary, the paired floor test); D-235, D-236, D-237"
                                  + ("; D-271 (4), D-273 (the 5% arm)" if args.fpr5 else ""),
               "target_fpr": target,
               **({"sensitivity_arm": "4e at 5%: never replaces a 1% number (D-248); the draft-claim test is reported beside only; "
                                      "no LoRA rows (D-273 (2))"} if args.fpr5 else {}),
               "supersedes": None if args.fpr5 else
                             {"file": PRIOR_CONTRASTS[0], "sha256": PRIOR_CONTRASTS[1],
                              "what": "its floor rule (band overlap, D-236) and its metric as primary (the mean over layers is "
                                      "now the continuity row); its numbers stay as written (D-256 (2), (3))"},
               "primary": {"metric": "d117_layer", "probe_set": PRIMARY[0], "level": PRIMARY[1]},
               "continuity": "mean_over_layers",
               "floor_rule": "D_arm(d) = m_arm(d) - m_floor(d) within each draw; damage detected: the band of D_frozen "
                             "wholly above 0; at floor: the band of D_arm includes 0, reported 'at most X above floor' (D-256 (2))",
               "band_note": "per-pair bands cover partition variation only; pooled intervals cover between-pair variation only (D-166)",
               "input_files_sha256": inputs, "by_metric": result}
    digest = write_results(out, payload, exclusive=True)
    print(f"Written: {out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
