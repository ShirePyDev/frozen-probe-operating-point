"""4e: D-083's four rules applied unchanged at the 5% operating point (D-152 4e; D-248), from the seven 5% moment
records only (results/moment_d156_<pair>_r2_fpr5.json, named by logs/moment_d156_r2_fpr5.sha256). The same rules
on the 1% records (results/moment_d156_<pair>_r2.json, logs/moment_d156_r2.sha256) are reported beside, never judged.

The metric is each record's per-draw mean over layers of |log(FPR/target)|, selected C, item level (index 0 the frozen
split). The NO-UPDATE NULL is the record's floor row: the same metric with FPR_pre at each draw's own frozen threshold.
  (1) an arm is AT THE FLOOR for a pair when its D-156 mean error is at or below the 95th percentile of that pair's
      null (over the draws); the share of draws in which the arm is at or below it is reported beside.
  (2) if restandardise AND refit_oracle are both at the floor for Gemma-2 or OLMo-2, D-082's "label-free beats
      labelled" becomes "indistinguishable" for that pair.
  (3) if the frozen arm is at the floor for a pair, that pair's damage claim is withdrawn at 5%.
  (4) if refit_oracle is at the floor in most pairs (more than half), any headline share of the repair must be restated
      relative to the floor.
Writes results/fpr5_rules_d156.json (never overwritten); a test writes outside results/.
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
FINAL_OUT = RESULTS / "fpr5_rules_d156.json"
CODE_FILES = ["src/fpr5_rules_d156.py", "src/paired.py", "src/populations.py", "src/provenance.py",
              "src/superseded_provenance.py"]
MANIFESTS = {"5pct": ROOT / "logs" / "moment_d156_r2_fpr5.sha256", "1pct": ROOT / "logs" / "moment_d156_r2.sha256"}
CELL = ("selected", "item_level_D156")
FLOOR = "no_update_floor"
RULE2_PAIRS = ("gemma2_base2instruct", "olmo_base2instruct")


def core_of(stem: str) -> str:
    c = stem[len("moment_"):]
    return c[:-3] if c.endswith("_v2") else c


def manifest(path: Path) -> tuple[dict[str, str], str]:
    if not path.is_file():
        raise SystemExit(f"REFUSING: no manifest {path}")
    b = path.read_bytes()
    man = {}
    for line in b.decode("utf-8").splitlines():
        if line.strip():
            h, f = line.split(maxsplit=1)
            man[f.strip().lstrip("*")] = h
    return man, sha256_bytes(b)


def rules(per_draw: dict[str, np.ndarray]) -> dict[str, Any]:
    null = per_draw[FLOOR][1:]
    p95 = float(np.percentile(null, 95))
    out: dict[str, Any] = {"null_p95": p95, "null": P.band_summary(null, per_draw[FLOOR][0]), "arms": {}}
    for arm, v in per_draw.items():
        if arm == FLOOR:
            continue
        out["arms"][arm] = {"mean": float(v[1:].mean()), "at_the_floor": bool(v[1:].mean() <= p95),
                            "share_of_draws_at_or_below_null_p95": float((v[1:] <= p95).mean())}
    return out


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="4e: D-083's rules at 5% (D-248)")
    ap.add_argument("--out", default=str(FINAL_OUT))
    args = ap.parse_args()
    out = Path(args.out).resolve()
    final = out == FINAL_OUT.resolve()
    if out.exists():
        raise SystemExit(f"REFUSING: {out} exists; never overwritten")
    if not final and out.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test output must not be under results/")
    code0 = P.code_block(CODE_FILES)
    inputs: dict[str, str] = {}
    by_target: dict[str, dict[str, Any]] = {}
    for tgt, mpath in MANIFESTS.items():
        man, msha = manifest(mpath)
        inputs[str(mpath.relative_to(ROOT)).replace("\\", "/")] = msha
        by_target[tgt] = {}
        for stem in POP.UPDATE_PAIRS_MOMENT:
            core = core_of(stem)
            rel = f"results/moment_d156_{core}_r2{'_fpr5' if tgt == '5pct' else ''}.json"
            b = (ROOT / rel).read_bytes()
            if man.get(rel) != sha256_bytes(b):
                raise SystemExit(f"REFUSING: {rel} is not the file {mpath.name} names")
            inputs[rel] = sha256_bytes(b)
            d = json.loads(b.decode("utf-8"))
            want_t = 0.05 if tgt == "5pct" else 0.01
            if d["config"]["target_fpr"] != want_t or d.get("registered_parameters") is not True:
                raise SystemExit(f"REFUSING: {rel} is not the registered run at {want_t}")
            if tgt == "5pct" and not (d.get("fpr5_self_anchor") or {}).get("n_level_blocks_identical_at_1pct"):
                raise SystemExit(f"REFUSING: {rel} carries no 1% self-anchor (D-248)")
            blk = d["pair_per_draw_mean_abs_log_over_layers"][CELL[0]][CELL[1]]
            per = {arm: np.asarray(v["values"], dtype="<f8") for arm, v in blk.items()}
            for arm, v in blk.items():
                if sha256_bytes(per[arm].tobytes()) != v["sha256"]:
                    raise SystemExit(f"REFUSING: {rel} {arm}: values do not hash to their sha256")
            by_target[tgt][core] = {"record": rel, "flag": d["input_verification"]["flag"], **rules(per)}
    judged = by_target["5pct"]
    rule2 = {p: {"restandardise_at_floor": judged[p]["arms"]["restandardise"]["at_the_floor"],
                 "refit_oracle_at_floor": judged[p]["arms"]["refit_oracle"]["at_the_floor"],
                 "label_free_beats_labelled_downgraded_to_indistinguishable":
                     judged[p]["arms"]["restandardise"]["at_the_floor"] and judged[p]["arms"]["refit_oracle"]["at_the_floor"]}
             for p in RULE2_PAIRS}
    rule3 = {p: {"frozen_at_floor": v["arms"]["frozen"]["at_the_floor"],
                 "damage_claim_withdrawn_at_5pct": v["arms"]["frozen"]["at_the_floor"]} for p, v in judged.items()}
    n_oracle = sum(v["arms"]["refit_oracle"]["at_the_floor"] for v in judged.values())
    rule4 = {"n_pairs_oracle_at_floor": n_oracle, "n_pairs": len(judged),
             "headline_share_restated_relative_to_the_floor": n_oracle > len(judged) / 2}
    for p, v in judged.items():
        print(f"{p:24s} null p95 {v['null_p95']:.3f}; at the floor: {[a for a, x in v['arms'].items() if x['at_the_floor']]}")
    print(f"rule 3 (damage withdrawn at 5%): {[p for p, v in rule3.items() if v['damage_claim_withdrawn_at_5pct']]}; "
          f"rule 4: oracle at the floor in {n_oracle} of {len(judged)}")
    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"] or any(sha256_file(ROOT / f) != h for f, h in inputs.items()):
        raise SystemExit("REFUSING to write: code or an input changed during the run")
    payload = {**provenance_block("fpr5_rules_d156"), "code": code0, "preregistration": "D-083 rules (1)-(4), D-152 4e, D-248",
               "cell": {"probe_set": CELL[0], "level": CELL[1]},
               "metric": "per-draw mean over layers of |log(FPR/target)|; the null is the floor row (no update)",
               "judged": "5pct", "beside": "1pct, never judged",
               "input_files_sha256": inputs, "per_pair": by_target,
               "rule2": rule2, "rule3": rule3, "rule4": rule4}
    digest = write_results(out, payload, exclusive=True)
    print(f"Written: {out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
