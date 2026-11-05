"""The moment family's pre-registered contrasts (D-235), from the seven results/moment_d156_<pair>_r2.json only.

Per pair, probe set and level, from each record's per-draw arrays m_arm(d) = the mean over the pair's layers of
|log(FPR_post/target)| (0 = on target; LOWER is better; index 0 the frozen split, 1..n the draws) and the no-update
floor row m_floor(d) (the same metric with FPR_pre, the pre model's eval FPR at the draw's own frozen threshold):
  CONTRASTS, paired within each draw (the same draw for both arms), D-156 mean and 95% band, frozen split beside:
    restandardise - frozen;  restandardise - paired_align;  restandardise - refit_oracle;
    paired_align - frozen (the draft's claim, D-235 (7)). Negative = the first arm is closer to target.
    mean_only's share of restandardise's improvement, per draw (m_frozen - m_mean_only) / (m_frozen - m_restandardise),
    as a ratio beside its denominator, never a percentage (D-216, D-089 answer). UNDEFINED unless the denominator's
    band lies wholly above 0 (D-235 (3)(d); a band wholly below 0 means there is no improvement to share, D-236).
  REACHES THE FLOOR, every arm: its band of m_arm overlaps the band of m_floor (D-235 (4)).
  POOLED: the equal-weight mean over the row's pairs of each pair's D-156 mean, with a pair-cluster bootstrap 95%
    interval (B = 5000, seed 520260925, pairs resampled; D-177 (b)) for rows of at least 3 pairs, the point only
    below; the pooled share is the ratio of the pooled numerator to the pooled denominator under the same resamples,
    UNDEFINED unless the pooled denominator lies above 0 (its interval, or its point without one; D-236).
    Rows: all seven pairs; the pairs whose record carries no D-170 flag; the four pairs unseen before D-235 was
    registered (Mistral, Llama-3 and Gemma-2 had been read; D-237 (2)).
  JUDGED only at the primary cell, selected C and item level. C=1 and the D-164 cluster level are reported beside
    with the same numbers and descriptive fields named '..._not_judged' (D-235 (6)).
  THE DRAFT'S CLAIM: a pair is WORSE when paired_align's error (the D-156 mean of m) exceeds the frozen arm's AND the
    band of the within-draw difference paired_align - frozen lies wholly above 0 (selected C, item level; D-237 (1));
    the draft's "worse than no repair at all in 4 of 7 pairs" is reproduced only if at least 4 of 7 are.
Each record must be named by logs/moment_d156_r2.sha256, be the registered final moment run of its pair, carry
per-draw arrays of length n_draws + 1 that hash to their sha256 and reproduce its own pair summary, and all seven
must come from the same src/repair_d156.py and src/paired.py. The three pairs with an earlier record must name it as
superseded. Writes results/moment_contrasts_d156.json (never overwritten); a test writes outside results/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
from provenance import ensure_utf8_stdout, provenance_block, sha256_bytes, sha256_file, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FINAL_OUT = RESULTS / "moment_contrasts_d156.json"
MANIFEST = ROOT / "logs" / "moment_d156_r2.sha256"
CODE_FILES = ["src/moment_contrasts_d156.py", "src/paired.py", "src/populations.py", "src/provenance.py",
              "src/superseded_provenance.py"]
LEVELS = ("item_level_D156", "cluster_level_D164")
PROBE_SETS = ("selected", "C=1")
PRIMARY = ("selected", "item_level_D156")
FLOOR = "no_update_floor"
ARMS = ("frozen", "mean_only", "restandardise", "score_affine", "score_affine_benign", "paired_align", "align_restd",
        "align_affine", "self_align_PRE", "refit_oracle")
DIFFS = {"restandardise_minus_frozen": ("restandardise", "frozen"),
         "restandardise_minus_paired_align": ("restandardise", "paired_align"),
         "restandardise_minus_refit_oracle": ("restandardise", "refit_oracle"),
         "paired_align_minus_frozen": ("paired_align", "frozen")}
SHARE = "mean_only_share_of_restandardise_improvement"
EARLIER = ("mistral_base2instruct", "llama3_base2instruct", "gemma2_base2instruct")     # D-235 (0), (8); D-237 (2)
SAME_CODE = ("src/repair_d156.py", "src/paired.py")
B, SEED, MIN_PAIRS = 5000, 520260925, 3                # D-177 (b)
CLAIM_MIN = 4                                          # "in 4 of 7 pairs"


def core_of(stem: str) -> str:
    c = stem[len("moment_"):]
    return c[:-3] if c.endswith("_v2") else c


def manifest(path: Path) -> tuple[dict[str, str], str]:
    if not path.is_file():
        raise SystemExit(f"REFUSING: no manifest {path} (written after the seven _r2 records are pulled)")
    b = path.read_bytes()
    man = {}
    for line in b.decode("utf-8").splitlines():
        if line.strip():
            h, f = line.split(maxsplit=1)
            man[f.strip().lstrip("*")] = h
    return man, sha256_bytes(b)


def summary(x: np.ndarray) -> dict[str, Any]:
    return P.band_summary(x[1:], x[0])


def excludes_zero(s: dict[str, Any]) -> bool:
    return s["band95"] is not None and (s["band95"][0] > 0 or s["band95"][1] < 0)


def overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a["band95"][0] <= b["band95"][1] and b["band95"][0] <= a["band95"][1]


def load_pair(stem: str, man: dict[str, str], inputs: dict[str, str]) -> dict[str, Any]:
    core = core_of(stem)
    rel = f"results/moment_d156_{core}_r2.json"
    b = (ROOT / rel).read_bytes()
    if man.get(rel) != sha256_bytes(b):
        raise SystemExit(f"REFUSING: {rel} is not the file {MANIFEST.name} names")
    inputs[rel] = sha256_bytes(b)
    d = json.loads(b.decode("utf-8"))
    if d["config"]["family"] != "moment" or d["config"]["pair"] != stem or d.get("registered_parameters") is not True:
        raise SystemExit(f"REFUSING: {rel} is not the registered moment run of {stem}")
    sup = d.get("supersedes")
    if core in EARLIER:
        sup_ok = bool(sup and sup.get("cells_identical") is True and sup.get("file") == f"results/moment_d156_{core}.json")
    else:
        sup_ok = sup is None
    if not sup_ok:
        raise SystemExit(f"REFUSING: {rel}: 'supersedes' {sup!r} does not match D-235 (8)")
    if "pair_per_draw_mean_abs_log_over_layers" not in d:
        raise SystemExit(f"REFUSING: {rel} has no per-draw arrays (D-235 (8))")
    n = int(d["config"]["n_draws"]) + 1
    m: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for ps in PROBE_SETS:
        for lvl in LEVELS:
            blk = d["pair_per_draw_mean_abs_log_over_layers"][ps][lvl]
            if set(blk) != {*ARMS, FLOOR}:
                raise SystemExit(f"REFUSING: {rel} {ps} {lvl}: arms {sorted(blk)}")
            m[(ps, lvl)] = {}
            for name, v in blk.items():
                a = np.asarray(v["values"], dtype="<f8")
                if a.shape != (n,):
                    raise SystemExit(f"REFUSING: {rel} {ps} {lvl} {name}: {a.shape[0]} values, not n_draws + 1 = {n}")
                if sha256_bytes(a.tobytes()) != v["sha256"]:
                    raise SystemExit(f"REFUSING: {rel} {ps} {lvl} {name}: the values do not hash to their sha256")
                if json.dumps(summary(a), sort_keys=True) != json.dumps(d["pair_summary_mean_abs_log_over_layers"][ps][lvl][name], sort_keys=True):
                    raise SystemExit(f"REFUSING: {rel} {ps} {lvl} {name}: the per-draw array does not reproduce the pair summary")
                m[(ps, lvl)][name] = a
    return {"rel": rel, "flag": d["input_verification"]["flag"], "n_layers": len(d["config"]["layers"]), "m": m,
            "code": {f: d["code"]["sha256"].get(f) for f in SAME_CODE}}


def sign_of(s: dict[str, Any]) -> str:
    lo, hi = s["band95"]
    return "above 0" if lo > 0 else "below 0" if hi < 0 else "includes 0"


def contrasts(m: dict[str, np.ndarray], judged: bool) -> dict[str, Any]:
    nj = "" if judged else "_not_judged"
    out: dict[str, Any] = {"judged": judged}
    for k, (a, b) in DIFFS.items():
        s = summary(m[a] - m[b])
        out[k] = {**s, f"band_excludes_0{nj}": excludes_zero(s)}
    num, den = m["frozen"] - m["mean_only"], m["frozen"] - m["restandardise"]
    den_s = summary(den)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = num / den
    ssum = summary(share)
    out[SHARE] = {**ssum, "median": float(np.median(share[1:][np.isfinite(share[1:])])) if np.isfinite(share[1:]).any() else None,
                  "denominator_restandardise_improvement": den_s, "denominator_band": sign_of(den_s),
                  "numerator_mean_only_improvement": summary(num),
                  "undefined": sign_of(den_s) != "above 0",
                  "note": "a ratio beside its denominator, never a percentage (D-216, D-089 answer); UNDEFINED unless "
                          "the denominator's band lies wholly above 0 (D-235 (3)(d), D-236); the median is descriptive"}
    fl = summary(m[FLOOR])
    out["floor"] = fl
    out[f"reaches_the_floor{nj}"] = {arm: overlaps(summary(m[arm]), fl) for arm in ARMS}
    if judged:
        # D-237 (1): a pair reproduces the claim when paired_align's error exceeds the frozen arm's (D-156 means of m)
        # AND the band of the within-draw difference lies wholly above 0
        pa_mean, fz_mean = summary(m["paired_align"])["mean"], summary(m["frozen"])["mean"]
        exceeds = bool(pa_mean > fz_mean)
        above = bool(out["paired_align_minus_frozen"]["band95"][0] > 0)
        out["paired_align_error_mean"], out["frozen_error_mean"] = pa_mean, fz_mean
        out["paired_align_error_exceeds_frozen"] = exceeds
        out["paired_align_minus_frozen_band_wholly_above_0"] = above
        out["worse_than_no_repair_paired_align"] = exceeds and above
    return out


def pooled(rows: dict[str, dict[str, Any]], stems: list[str], judged: bool) -> dict[str, Any]:
    k = len(stems)
    if k == 0:
        return {"pairs": [], "n_pairs": 0, "note": "no pair in this row"}
    nj = "" if judged else "_not_judged"
    idx = np.random.default_rng(SEED).integers(0, k, size=(B, k)) if k >= MIN_PAIRS else None
    res: dict[str, Any] = {"pairs": stems, "n_pairs": k, "judged": judged,
                           "interval": f"pair-cluster bootstrap B={B}, seed {SEED}" if idx is not None else
                                       f"point only (fewer than {MIN_PAIRS} pairs, D-177 (b))"}
    for key in DIFFS:
        v = np.array([rows[s][key]["mean"] for s in stems])
        res[key] = {"mean": float(v.mean()), **(P.ci(v[idx].mean(axis=1)) if idx is not None else {"ci95": None})}
    num = np.array([rows[s][SHARE]["numerator_mean_only_improvement"]["mean"] for s in stems])
    den = np.array([rows[s][SHARE]["denominator_restandardise_improvement"]["mean"] for s in stems])
    den_ci = P.ci(den[idx].mean(axis=1)) if idx is not None else {"ci95": None}
    den_ok = den_ci["ci95"][0] > 0 if den_ci["ci95"] is not None else float(den.mean()) > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        res[SHARE] = {"ratio_of_pooled_means": float(num.mean() / den.mean()),
                      "pooled_denominator": {"mean": float(den.mean()), **den_ci},
                      **(P.ci(num[idx].mean(axis=1) / den[idx].mean(axis=1)) if idx is not None else {"ci95": None}),
                      "undefined": not den_ok,
                      "note": "a ratio beside its denominator, never a percentage; UNDEFINED unless the pooled "
                              "denominator lies above 0 (D-236)"}
    res[f"n_pairs_reaching_the_floor{nj}"] = {arm: int(sum(rows[s][f"reaches_the_floor{nj}"][arm] for s in stems)) for arm in ARMS}
    return res


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="D-235: the moment family's pre-registered contrasts")
    ap.add_argument("--out", default=str(FINAL_OUT))
    args = ap.parse_args()
    out = Path(args.out).resolve()
    final = out == FINAL_OUT.resolve()
    if out.exists():
        raise SystemExit(f"REFUSING: {out} exists; never overwritten")
    if not final and out.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test output must not be under results/")
    code0 = P.code_block(CODE_FILES)
    man, man_sha = manifest(MANIFEST)
    inputs = {str(MANIFEST.relative_to(ROOT)).replace("\\", "/"): man_sha}
    stems = list(POP.UPDATE_PAIRS_MOMENT)
    pairs = {s: load_pair(s, man, inputs) for s in stems}
    codes = {json.dumps(pr["code"], sort_keys=True) for pr in pairs.values()}
    if len(codes) != 1 or any(v is None for v in pairs[stems[0]]["code"].values()):
        raise SystemExit(f"REFUSING: the seven records do not share one {SAME_CODE}: {sorted(codes)}")
    per_pair: dict[str, Any] = {}
    for s, pr in pairs.items():
        per_pair[s] = {"record": pr["rel"], "flag": pr["flag"], "n_layers": pr["n_layers"],
                       **{f"{ps}|{lvl}": contrasts(pr["m"][(ps, lvl)], (ps, lvl) == PRIMARY) for ps in PROBE_SETS for lvl in LEVELS}}
    unflagged = [s for s in stems if pairs[s]["flag"] is None]
    unseen = [s for s in stems if core_of(s) not in EARLIER]        # D-237 (2): not read before D-235 was registered
    pooled_rows = {}
    for ps in PROBE_SETS:
        for lvl in LEVELS:
            c, j = f"{ps}|{lvl}", (ps, lvl) == PRIMARY
            pooled_rows[c] = {"all_pairs": pooled({s: per_pair[s][c] for s in stems}, stems, j),
                              "unflagged_pairs": pooled({s: per_pair[s][c] for s in unflagged}, unflagged, j),
                              "unseen_pairs": pooled({s: per_pair[s][c] for s in unseen}, unseen, j)}
    key = "|".join(PRIMARY)
    worse = [s for s in stems if per_pair[s][key]["worse_than_no_repair_paired_align"]]
    n_w, n_wu = len(worse), len([s for s in worse if pairs[s]["flag"] is None])
    n_wn = len([s for s in worse if s in unseen])
    claim = {"claim": "a paired alignment map calibrates worse than no repair at all in 4 of 7 pairs "
                      "(paper/main.tex:60, introduction.tex:68, discussion.tex:50, conclusion.tex:18)",
             "test": "the claim is reproduced at a pair when paired_align's error (the D-156 mean of m) exceeds the frozen "
                     "arm's AND the band95 of the within-draw difference paired_align - frozen lies wholly above 0; selected "
                     "C, item level (D-237 (1)); reproduced overall when at least 4 of 7 pairs are (D-235 (7))",
             "pairs_worse": worse, "n_worse": n_w, "n_pairs": len(stems),
             "n_worse_among_unflagged": n_wu, "n_unflagged": len(unflagged),
             "n_worse_among_unseen": n_wn, "n_unseen": len(unseen),
             "seen_before_registration": [s for s in stems if s not in unseen],
             "seen_note": "before D-235 was registered, the Llama-3 and Gemma-2 summaries had been read, and Mistral's frozen "
                          "and self_align_PRE lines quoted (D-235 (0)); the unseen pairs are reported as their own row (D-237 (2))",
             "pooled_paired_align_minus_frozen_unseen": pooled_rows[key]["unseen_pairs"]["paired_align_minus_frozen"],
             "reproduced": n_w >= CLAIM_MIN,
             "consequence": (f"reproduced: {n_w} of {len(stems)} pairs worse (the draft states 4 of 7); the count includes "
                             f"D-170-flagged pairs and the three pairs seen before D-235 ({n_wu} of {len(unflagged)} unflagged, "
                             f"{n_wn} of {len(unseen)} unseen pairs worse)")
                            if n_w >= CLAIM_MIN else
                            (f"WITHDRAWN from the draft (D-235 (7)): {n_w} of {len(stems)} pairs worse, fewer than 4 "
                             f"({n_wu} of {len(unflagged)} unflagged, {n_wn} of {len(unseen)} unseen pairs worse)"),
             "pooled_paired_align_minus_frozen": pooled_rows[key]["all_pairs"]["paired_align_minus_frozen"]}

    def fmt(x: dict[str, Any]) -> str:
        return f"{x['mean']:+.3f} [{x['band95'][0]:+.3f}, {x['band95'][1]:+.3f}]"

    for s in stems:
        r = per_pair[s][key]
        sh = r[SHARE]
        shown = "UNDEFINED" if sh["undefined"] else format(sh["mean"], ".3f")
        print(f"{core_of(s):22s} flag {'yes' if pairs[s]['flag'] else 'no ':3s} restd-frozen {fmt(r['restandardise_minus_frozen'])}  "
              f"restd-align {fmt(r['restandardise_minus_paired_align'])}  restd-oracle {fmt(r['restandardise_minus_refit_oracle'])}  "
              f"align-frozen {fmt(r['paired_align_minus_frozen'])}  share {shown} (denominator "
              f"{fmt(sh['denominator_restandardise_improvement'])})  floor reached: "
              f"{[a for a, v in r['reaches_the_floor'].items() if v]}", flush=True)
    for row in ("all_pairs", "unflagged_pairs", "unseen_pairs"):
        pr_ = pooled_rows[key][row]
        if pr_["n_pairs"]:
            ci_ = lambda x: "point only" if x.get("ci95") is None else f"[{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"  # noqa: E731
            print(f"pooled {row} ({pr_['n_pairs']}): " + "  ".join(f"{kk} {pr_[kk]['mean']:+.3f} {ci_(pr_[kk])}" for kk in DIFFS),
                  flush=True)
    print(f"draft claim: {claim['consequence']}")

    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"] or any(sha256_file(ROOT / f) != h for f, h in inputs.items()):
        raise SystemExit("REFUSING to write: code or an input changed during the run")
    payload = {**provenance_block("moment_contrasts_d156"), "code": code0, "preregistration": "D-235 (choices: D-236)",
               "primary": {"probe_set": PRIMARY[0], "level": PRIMARY[1]},
               "beside": "C=1 and the D-164 cluster level: the same numbers, never judged (fields '..._not_judged', D-235 (6))",
               "metric": "per draw, the mean over the pair's layers of |log(FPR/target)| (0 = on target, lower is better); "
                         "index 0 the frozen split. Contrasts are arm minus arm within each draw: negative = the first arm "
                         "is closer to target",
               "floor_row": "no_update_floor: the same metric with FPR_pre, the pre model's eval FPR at each draw's own "
                            "frozen threshold (no update)",
               "band_note": "per-pair bands cover partition variation only; pooled intervals cover between-pair "
                            "variation only (D-166)",
               "input_files_sha256": inputs, "per_pair": per_pair, "pooled": pooled_rows, "draft_claim": claim}
    digest = write_results(out, payload, exclusive=True)
    print(f"Written: {out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
