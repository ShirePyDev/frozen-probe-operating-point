"""Read results from disk and print them. Never computes anything fresh.

CLAUDE.md rule 2: every number in any summary must be read back from a results
file produced by a run in this repo. This module is the only sanctioned way to
produce a summary, and it deliberately has no access to models, datasets or
metrics - it can only open JSON and format it. If a number cannot be printed
from here, it is not a number this project is allowed to claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import ensure_utf8_stdout, sha256_file  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load(name: str) -> tuple[dict[str, Any] | None, str | None]:
    path = RESULTS / name
    if not path.exists():
        return None, None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh), sha256_file(path)


def _prov(d: dict[str, Any]) -> str:
    g = d.get("git", {})
    commit = str(g.get("commit"))[:12] if g.get("commit") else "UNCOMMITTED"
    dirty = g.get("dirty")
    flag = "" if dirty is False else ("  [DIRTY TREE]" if dirty else "  [dirty unknown]")
    return f"run {d.get('utc')}  commit {commit}{flag}"


def checkpoints() -> None:
    d, digest = load("phase0_checkpoint_audit.json")
    if d is None:
        print("phase0_checkpoint_audit.json: NOT YET RUN")
        return
    print("=" * 78)
    print("PHASE 0 - CHECKPOINT AUDIT")
    print(f"{_prov(d)}   token_present={d.get('hf_token_present')}")
    print(f"file sha256 {digest}")
    print("=" * 78)
    s = d["summary"]
    for pair in d["pairs"]:
        blocked = f"  blocked_on={pair['blocked_on']}" if pair.get("blocked_on") else ""
        print(f"  {pair['classification']:<9} {pair['family']:<12} "
              f"{pair['from_version']:>5} -> {pair['to_version']:<7}{blocked}")
        if "params" in pair:
            print(f"            params {pair['params']['from']} -> {pair['params']['to']}"
                  f"   delta {pair['params_rel_delta']:.4%}")
        for note in pair.get("notes", []):
            print(f"            note: {note}")
    print("-" * 78)
    print(f"  CLEAN {s['clean_pair_count']}   CAVEATED {s['caveated_pair_count']}   "
          f"UNUSABLE {s['unusable_pair_count']} "
          f"(of which {s['unusable_blocked_on_token']} blocked on token only)")


def datasets() -> None:
    d, digest = load("phase0_dataset_audit.json")
    if d is None:
        print("\nphase0_dataset_audit.json: NOT YET RUN")
        return
    print("\n" + "=" * 78)
    print("PHASE 0 - DATASET AUDIT")
    print(f"{_prov(d)}   token_present={d.get('hf_token_present')}")
    print(f"file sha256 {digest}")
    print("=" * 78)
    for e in d["datasets"]:
        r = e.get("resolved")
        if not r:
            print(f"  {e['key']:<14} phase{e['needed_by_phase']}  UNRESOLVED")
            continue
        print(f"  {e['key']:<14} phase{e['needed_by_phase']}  {r['dataset_id']}")
        print(f"       readable={r['access_verified']}  gated={r['gated']}  "
              f"licence={r['license']}  rows={r['total_rows']}")
        for k, v in sorted((r.get("row_counts") or {}).items()):
            print(f"         {k}: {v}")
    s = d["summary"]
    print("-" * 78)
    print(f"  readable {s['readable_count']}/{s['readable_count'] + s['blocked_count'] + s['missing_count']}"
          f"   Phase 1 readable {s['phase1_datasets_readable']}/{s['phase1_datasets_total']}")


def citations() -> None:
    d, digest = load("phase0_citation_audit.json")
    if d is None:
        print("\nphase0_citation_audit.json: NOT YET RUN")
        return
    print("\n" + "=" * 78)
    print("PHASE 0 - CITATION AUDIT")
    print(f"{_prov(d)}")
    print(f"file sha256 {digest}")
    print("=" * 78)
    for c in d["citations"]:
        state = {True: "RESOLVED", False: "NOT FOUND", None: "UNDETERMINED"}[c["resolved"]]
        print(f"  [{state}] arXiv:{c['arxiv_id']}  via {c.get('sources_positive')}")
        print(f"      {c['title']}")
        n = len(c.get("authors") or [])
        print(f"      {n} author(s), {c.get('published')}")
    s = d["summary"]
    print("-" * 78)
    print(f"  resolved {s['resolved']}/{s['total']}   not_found {s['not_found']}   "
          f"undetermined {s['undetermined']}")


def power() -> None:
    d, digest = load("phase0_power_analysis.json")
    if d is None:
        print("\nphase0_power_analysis.json: NOT YET RUN")
        return
    print("\n" + "=" * 78)
    print("PHASE 0.5 - POWER ANALYSIS")
    print(f"{_prov(d)}")
    print(f"file sha256 {digest}")
    print("=" * 78)
    cfg = d["config"]
    print(f"  baseline beta {cfg['baseline_beta']}   replicates/cell {cfg['n_rep']}   "
          f"seed {cfg['seed']}")
    print(f"\n  estimator bias (mean beta-hat minus analytic truth):")
    print(f"  {'L':>3} {'p':>6} {'n_items':>8} {'mean_bhat':>10} {'bias':>9} {'SE':>9}")
    print("  " + "-" * 50)
    for c in d["cells"]:
        if c["n_items"] in (500, 2000, 20000) and c.get("se"):
            print(f"  {c['n_layers']:>3} {c['p']:>6.2f} {c['n_items']:>8} "
                  f"{c['mean_beta_hat']:>10.4f} {c['bias']:>+9.4f} {c['se']:>9.4f}")
    print(f"\n  smallest n_items to detect delta-beta="
          f"{cfg['target_delta_for_recommendation']} at 80% power:")
    for r in d["recommendations"]:
        need = r["n_items_required"]
        label = str(need) if need else "NOT ACHIEVABLE on tested grid"
        print(f"    n_layers={r['n_layers']}  p={r['p']:<5} -> {label}")


def pilot(name: str = "pilot_e6.json") -> None:
    d, digest = load(name)
    if d is None:
        print(f"\n{name}: NOT YET RUN")
        return
    print("\n" + "=" * 78)
    print(f"E6 PILOT - {name}")
    print(f"{_prov(d)}")
    print(f"file sha256 {digest}")
    print("=" * 78)
    cfg = d["config"]
    print(f"  pre : {cfg['pre_repo']} @ {cfg['pre_rev'][:12]}")
    print(f"  post: {cfg['post_repo']} @ {cfg['post_rev'][:12]}")
    print(f"  probe_layer={cfg['probe_layer']}  sim_layer={cfg['sim_layer']}  "
          f"het={cfg.get('het_repo')}  target_fpr={cfg['target_fpr']}")
    h = d["health"]
    print(f"  health: pre nonfinite={h['pre']['n_nonfinite']} trunc={h['pre']['n_truncated']}  "
          f"post nonfinite={h['post']['n_nonfinite']} trunc={h['post']['n_truncated']}")

    first = d["seeds"][sorted(d["seeds"])[0]]
    print(f"\n  frozen thresholds (calibrated on PRE only): {first['layers']['thresholds']}")
    print(f"\n  {'side':<5} {'layer':<10} {'AUROC':>7} {'miss@thr':>9} {'flag_pos':>9} "
          f"{'flag_neg':>9} {'calibFPR':>9}")
    print("  " + "-" * 62)
    for side in ("pre", "post"):
        for ln, v in first["layers"][side].items():
            au = v["auroc"]
            print(f"  {side:<5} {ln:<10} {au:>7.4f} {v['miss_rate_on_pos']:>9.4f} "
                  f"{v['eval_flag_rate_pos']:>9.4f} {v['eval_flag_rate_neg']:>9.4f} "
                  f"{v['calib_fpr']:>9.4f}")

    print(f"\n  {'seed':>4} {'beta_pre':>9} {'beta_post':>10} {'d_beta':>9} "
          f"{'coupling':>10} {'P_all_pre':>10} {'P_all_post':>11} {'phi_pre':>8} {'phi_post':>9}")
    print("  " + "-" * 88)
    dbs, cts = [], []
    for k in sorted(d["seeds"], key=int):
        s = d["seeds"][k]
        dc = s["decomposition"]
        bp = dc["beta_pre"]["value"]
        bq = dc["beta_post"]["value"]
        db = dc["delta_beta"]
        if db is not None:
            dbs.append(db)
        cts.append(dc["coupling_term"])
        fmt = lambda x: f"{x:>9.4f}" if x is not None else f"{'None':>9}"  # noqa: E731
        print(f"  {k:>4} {fmt(bp)} {fmt(bq)[1:]:>10} {fmt(db)} {dc['coupling_term']:>10.5f} "
              f"{dc['p_all_pre']:>10.4f} {dc['p_all_post']:>11.4f} "
              f"{s['phi_pre']:>8.4f} {s['phi_post']:>9.4f}")
    import statistics as st
    if dbs:
        print("  " + "-" * 88)
        sd = st.stdev(dbs) if len(dbs) > 1 else float("nan")
        print(f"  delta_beta across seeds: mean={st.mean(dbs):+.4f}  sd={sd:.4f}  "
              f"min={min(dbs):+.4f}  max={max(dbs):+.4f}  "
              f"signs={'+' if all(x > 0 for x in dbs) else 'MIXED' if any(x > 0 for x in dbs) else '-'}")
        print(f"  coupling_term across seeds: mean={st.mean(cts):+.6f}  "
              f"min={min(cts):+.6f}  max={max(cts):+.6f}")
    b = first.get("bootstrap_delta_beta") or {}
    if b.get("ci_low") is not None:
        print(f"  seed0 delta_beta 95% CI: [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]  "
              f"bias={b['bias_estimate']:+.4f}  excludes_zero="
              f"{not (b['ci_low'] <= 0 <= b['ci_high'])}")
    c = first.get("bootstrap_coupling_term") or {}
    if c.get("ci_low") is not None:
        print(f"  seed0 coupling_term 95% CI: [{c['ci_low']:+.6f}, {c['ci_high']:+.6f}]  "
              f"excludes_zero={not (c['ci_low'] <= 0 <= c['ci_high'])}")


def depth(name: str) -> None:
    """Depth curve: accuracy vs update-robustness vs calibration transfer."""
    d, digest = load(name)
    if d is None:
        print(f"\n{name}: NOT YET RUN")
        return
    print("\n" + "=" * 90)
    print(f"DEPTH CURVE - {name}   [update_kind: {d.get('update_kind')}]")
    print(f"{_prov(d)}")
    print(f"file sha256 {digest}")
    print(f"target_fpr={d['config']['target_fpr']}  seeds={d['config']['seeds']}  "
          f"stochasticity={d.get('stochasticity_source')}")
    print("=" * 90)
    rows = d["rows"]
    tgt = d["config"]["target_fpr"]
    print(f"  {'layer':>5} {'depth':>6} {'CKA':>8} {'AUROC':>7} {'frozen':>7} "
          f"{'drop':>8} {'+-sd':>7} {'postFPR':>8} {'FPRratio':>9}")
    print("  " + "-" * 76)
    for r in rows:
        fpr = r["post_fpr_at_frozen_thr_mean"]
        ratio = (fpr / tgt) if (fpr is not None and tgt) else None
        print(f"  {r['layer']:>5} {r['relative_depth']:>6.2f} {r['cka']:>8.5f} "
              f"{r['auroc_pre_refit_mean']:>7.4f} {r['auroc_frozen_mean']:>7.4f} "
              f"{r['drop_frozen_mean']:>+8.4f} {r['drop_frozen_sd']:>7.4f} "
              f"{fpr:>8.4f} {ratio:>8.1f}x")

    import math
    def corr(xs, ys):
        xs = [x for x, y in zip(xs, ys) if x is not None and y is not None]
        ys2 = [y for x, y in zip(xs, ys) if x is not None and y is not None]
        n = len(xs)
        if n < 3:
            return None
        mx, my = sum(xs) / n, sum(ys2) / n
        sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        sy = math.sqrt(sum((y - my) ** 2 for y in ys2))
        if sx == 0 or sy == 0:
            return None
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys2)) / (sx * sy)

    cka = [r["cka"] for r in rows]
    drop = [r["drop_frozen_mean"] for r in rows]
    fprs = [r["post_fpr_at_frozen_thr_mean"] for r in rows]
    absdev = [abs(math.log((f or 1e-9) / tgt)) for f in fprs]

    print("  " + "-" * 76)
    print("  DOES REPRESENTATIONAL SIMILARITY PREDICT FUNCTIONAL DAMAGE?")
    c1 = corr(cka, drop)
    c2 = corr(cka, absdev)
    print(f"    pearson(CKA, frozen AUROC drop)          = "
          f"{c1:+.3f}" if c1 is not None else "    n/a")
    print(f"    pearson(CKA, |log FPR ratio|)            = "
          f"{c2:+.3f}" if c2 is not None else "    n/a")
    print("    (a strongly negative value would mean lower CKA predicts more damage,")
    print("     i.e. CKA is a usable drift proxy; near zero or positive means it is not)")

    ok = [r for r in rows if r["auroc_pre_refit_mean"] is not None]
    best_acc = max(ok, key=lambda r: r["auroc_pre_refit_mean"])
    best_rob = min(ok, key=lambda r: abs(r["drop_frozen_mean"]))
    best_cal = min(ok, key=lambda r: abs(math.log((r["post_fpr_at_frozen_thr_mean"] or 1e-9) / tgt)))
    worst_cal = max(ok, key=lambda r: abs(math.log((r["post_fpr_at_frozen_thr_mean"] or 1e-9) / tgt)))
    print("  " + "-" * 76)
    print(f"    most accurate layer      : {best_acc['layer']:>3} (d={best_acc['relative_depth']:.2f})  "
          f"AUROC={best_acc['auroc_pre_refit_mean']:.4f}")
    print(f"    most AUROC-robust layer  : {best_rob['layer']:>3} (d={best_rob['relative_depth']:.2f})  "
          f"drop={best_rob['drop_frozen_mean']:+.4f}")
    print(f"    best calibration transfer: {best_cal['layer']:>3} (d={best_cal['relative_depth']:.2f})  "
          f"postFPR={best_cal['post_fpr_at_frozen_thr_mean']:.4f}")
    print(f"    WORST calibration        : {worst_cal['layer']:>3} (d={worst_cal['relative_depth']:.2f})  "
          f"postFPR={worst_cal['post_fpr_at_frozen_thr_mean']:.4f}  "
          f"(CKA there = {worst_cal['cka']:.5f})")


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Print results read from disk")
    ap.add_argument("--only", choices=["checkpoints", "datasets", "citations", "power", "pilot", "depth"])
    ap.add_argument("--file", default="pilot_e6.json", help="which pilot results file")
    args = ap.parse_args()
    if args.only == "depth":
        depth(args.file)
        return 0
    if args.only == "pilot":
        pilot(args.file)
        return 0
    fns = {"checkpoints": checkpoints, "datasets": datasets,
           "citations": citations, "power": power}
    if args.only:
        fns[args.only]()
    else:
        for fn in (checkpoints, datasets, citations, power):
            fn()
        pilot(args.file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
