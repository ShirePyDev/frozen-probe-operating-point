"""D-384: the operational case study (the author's step-4 item 4). What the frozen threshold costs per 100,000 requests
at 1% harmful traffic, and what recalibration (R_same) or a check-triggered recalibration saves, for two update pairs at
L*, the selected C, item level, 1% target; 5%, the cluster level and R_fresh beside. Pre-registered in D-384 before any
code or number of it existed. DESCRIPTIVE (D-384 (5)): no prediction, no verdict, so no D-177 accuracy line.

PAIRS (D-384 (1)): Llama-3-8B -> Llama-3.1-8B (llama3_to_31, the author's choice) and, by the rule fixed in D-384 (1),
the released pair with the largest Table I paired ratio at L*: Mistral-7B v0.1 -> Instruct-v0.1 (mistral_base2instruct).
The summary re-reads Table I (results/depth_d156_summary_r2.json, D-305) and refuses unless the rule still gives that pair.
TRAFFIC (D-384 (2)): 100,000 requests, 1% harmful: N_HARMFUL = 1,000 and N_BENIGN = 99,000. Per draw d, with the pre model
at its own threshold T_d as the baseline (FPR_pre, TPR_pre of that draw):
    extra false refusals = 99,000 x (FPR_policy - FPR_pre)
    extra missed harmful =  1,000 x (TPR_pre - TPR_policy)                   both signed
POLICIES (D-384 (3)), per draw:
    FROZEN               T_d kept: the post model's scores against the pre model's threshold
    RECALIBRATED         R_same: Table II's stored-data threshold repair, 4d A's quantile_thr_benign (the exact quantile
                         transfer of T_d on cal_d re-scored by the post model; repair_d156.partition)
    CHECK-TRIGGERED      R_same in a draw where the cell's 4c Gaussian health ratio crosses 2 or 0.5 in that draw
                         (healthcheck_d156.crosses at CLASSES['two_sided_2x']), FROZEN otherwise; the direct-ratio
                         trigger beside
    and, per non-frozen policy, the saving against FROZEN, formed within each draw.
REPORTED per pair, target and level: every quantity's D-156 mean and 95% band over the 1000 draws, with the frozen split
as one draw and its percentile (paired.band_summary); per policy the share of draws with a floored FPR; per trigger the
share of draws triggered. Primary: 1%, item level. Beside: 5%, the D-164 cluster level, R_fresh, and for FROZEN's extra
missed harmful the D-282 combined interval (Table I's delta-TPR interval x -1000).

WHY A RECOMPUTE JOB (D-384 (4), checked 2026-10-07 on the records of record): depth_d156_<pair>[_fpr5].json and
repair_d156_<pair>.json hold D-156 summaries (mean, band, frozen value, percentile, sd) and two per-draw hashes, not the
per-draw FPR and TPR arrays; table1_tpr_d282*.json holds summaries and replicate hashes; the 4d B pool-study records hold
summaries. Only healthcheck_d156_<pair>[_fpr5]_draws.npz holds per-draw arrays: the health statistics and the realised
paired ratio, index 0 the frozen split, then the D-156 draws (and D-164 draws) of paired.repartition_draws at the axis
seed, which is the draw order of depth_d156 and repair_d156 (same seed, same order: the configs' draw_seed,
cluster_draw_seed and n_draws are compared). So no record gives FPR_pre, FPR_post or TPR per draw under FROZEN or R_same,
and none gives R_fresh per draw. Nothing is approximated: the job mode recomputes them with the reviewed functions,
imported unchanged, and writes nothing unless they reproduce the records.

MODES (outputs written once by staged write and hard link, never overwritten, mode 444; a test writes outside results/):
  --pair <core>   (CPU; the Phase 1 caches through src/input_registry.py; L* only, the selected C only)
                  results/opcase_d384_<core>.json and results/opcase_d384_<core>_draws.npz. At 1% and at 5%, both
                  levels: per partition (frozen split, then the draws) repair_d156.partition with the arms 'frozen' and
                  'quantile_thr_benign' (R_same), the pre side's floored FPR at T_d, and healthcheck_d156.health at T_d;
                  paired.repartition_estimate for the anchor. ANCHORS, each refusing with nothing written:
                    (a) the generic path equals paired.repartition_estimate's per-draw fpr_pre, fpr_post, tpr_pre,
                        tpr_post and paired_ratio bit for bit (repair_d156.main's check);
                    (b) its report equals the depth record's operating point (canonical JSON) and its per-draw
                        ratio_to_target hashes to the record's per_draw_ratio_to_target_sha256 (the FPR_post array);
                    (c) at 1%: repair_d156.summarise of the 'frozen' and 'quantile_thr_benign' arms equals the repair
                        record's blocks exactly (fpr_post, tpr_post, ... mean, band, frozen, percentile, sd; the R_same
                        FPR and TPR arrays as the record summarises them);
                    (d) every health statistic and the realised paired ratio equal the health-check npz arrays bit for
                        bit (the trigger's draws are the cost's draws, in the same order).
                  R_fresh (1%, item level): poolstudy_d156.main's lines for 'frozen' and 'score|R_fresh|n=1000' repeated
                  on the option-B draws (poolstudy_d156.partitions and draw_pools, unchanged), with FPR_pre on the same
                  evaluation half E; refused unless both configurations' blocks equal the option-B record's exactly
                  and TPR_pre at the frozen split equals the depth record's tpr_pre.frozen. What no record anchors is
                  named in the record's anchor.unanchored (UNANCHORED) and copied beside the summary's cells.
  --summary       results/opcase_d384.json, seconds, no cache: reads the two pair records and npz (named by
                  logs/opcase_d384.sha256, the lane's manifest job), the health-check records and npz, the depth
                  records and the D-282 Table I records; re-checks the draw alignment (configs, and the pair npz's health
                  and realised-ratio arrays against the health-check npz bit for bit) and computes the costs.
Final names need the registered parameters (n_draws 1000, seeds 0-4), the 6-thread BLAS pin (scripts/project_env.sh), an
unset GUARDRAIL_CACHE_DIRS, the manifests at their pinned bytes (PINNED_MANIFEST_SHA256) and the reused files at their
reviewed bytes (REUSED_SHA256).

CHOICES (D-181; the closest D-entry named for each; none changes an estimand or what is primary):
  1. R_same is 4d A's quantile_thr_benign arm via repair_d156.partition (D-289; Table II's stored-data threshold column,
     D-330), on the same 1000 D-156 draws as FROZEN and 4c, so CHECK-TRIGGERED mixes two policies within one draw. (It
     equals the threshold re-set at the target on cal_d's post scores given no tie at T_d's rank in cal_d's pre scores
     or in its post scores.)
  2. FPR is the seed mean of per-seed FPRs floored at 1/n of the evaluation negatives, TPR the seed mean of per-seed
     TPRs: the records' definitions (D-156, paired.floored_fpr and paired.rate); costs are signed (D-384 (2)); per policy
     the share of draws with a floored seed (pre or post) is reported beside.
  3. Per-draw costs, then paired.band_summary over the draws with the frozen split as one draw (D-156).
  4. Savings are FROZEN minus the policy within each draw (D-156: between-arm gaps are formed within each draw).
  5. Trigger: healthcheck_d156.crosses at CLASSES['two_sided_2x'] = (2, 0.5) on the health-check record's per-draw
     gaussian_ratio (primary) and direct_ratio (beside) of the same cell, level and target (D-241 (4), D-255); at 5% the
     5% health-check record (D-248), at the cluster level its cluster-level arrays (D-164).
  6. At 5% no repair record exists (4e's 5% arm covers the moment family only, D-248): FROZEN at 5% is anchored on the 5%
     depth record and the 5% health-check npz; R_same at 5% is the same reviewed function at target 0.05, with no record
     of its own to equal (named in the record, 'anchor.unanchored').
  7. R_fresh beside: 4d B's 'score|R_fresh|n=1000' on the option-B records (D-306; Table II's source for the Aegis pairs,
     D-322): the frozen split plus 200 origin-stratified draws, FPR on E (the 1000 evaluation negatives left after
     R_FRESH) with FPR_pre on the same E, TPR on the fixed positives; 1% and item level only (no 5% or cluster-level pool
     study exists); not combined with the trigger (other draws). poolstudy_d156.main computes it inside main(), so its
     lines are repeated here and pinned by the anchor (hh_sensitivity_d363's precedent, D-363 (4)). Its baseline matches
     no record and is named in anchor.unanchored: FPR_pre on E in every draw, and TPR_pre in the option-B draws; TPR_pre
     at the frozen split is the same quantity as the depth record's tpr_pre.frozen (the pre scores of the fixed eval
     positives at the frozen split's T_d, seed mean) and must equal it exactly.
  8. The selected C only (D-384 (1)); C=1 is not computed.
  9. The D-282 combined interval is quoted beside FROZEN's extra missed harmful only (-1000 x Table I's delta-TPR
     interval at L*, table1_tpr_d282_r2.json at 1% and table1_tpr_d282_fpr5.json at 5%); it is not computed for the
     repair policies, as Table II (D-330).
 10. Traffic at 5% is the same 100,000 requests at 1% harmful; only the threshold target changes (D-384 (2), D-248).
 11. One job per pair computes 1%, 5% and R_fresh (D-384's cost line: two CPU jobs).
 12. The D-170 flags: the records' stored flags are carried beside each pair's cells (depth, repair, health-check,
     option-B, Table I and D-282 records); the job's own REG.verification of the same caches is the current one (D-331:
     D-168 is final).
 13. The 1% R_same anchor is against the repair record's D-156 summaries of the arm at L* (mean, band, frozen value,
     percentile, sd of fpr_post, tpr_post and the rest): no record holds per-draw R_same at L* (the repair record's
     per-draw arrays are means of |log(FPR/target)| over the pair's layer grid, 17 layers for both pairs in the depth
     records' config.layers). The draw order of the R_same arrays is fixed through FROZEN: both arms come from one
     repair_d156.partition call per partition, and FROZEN's arrays are anchored draw by draw.
 14. Per cell the summary states, from the arrays, the share of draws in which each trigger fired and whether the
     CHECK-TRIGGERED policy is identical to R_same or to FROZEN in every draw (the frozen split included).
CAVEAT (rule 5, D-255's flag): the 4c reference set, R_same's cal_d and the evaluation part come from one pool of
negatives re-partitioned per draw, and the benign 'traffic' is the pair's benign evaluation prompts; a deployer whose
reference prompts or traffic differ from these would see other costs and trigger rates.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_d156 as D  # noqa: E402
import healthcheck_d156 as HC  # noqa: E402
import input_registry as REG  # noqa: E402
import paired as P  # noqa: E402
import poolstudy_d156 as PS  # noqa: E402
import repair_d156 as RP  # noqa: E402
from provenance import (  # noqa: E402
    _RESULT_READS, _repo_rel, canonical_hash, commit_staged, ensure_utf8_stdout, input_identity_problems, provenance_block,
    read_result, resolve_cache, sha256_bytes, sha256_file, stage_bytes, write_results,
)

ROOT = Path(__file__).resolve().parent.parent
OWN = "src/opcase_d384.py"
CODE_FILES = list(dict.fromkeys([OWN, *RP.CODE_FILES, *HC.CODE_FILES, *PS.CODE_FILES]))
# the reviewed files whose functions are reused, at the bytes this module was written and anchored against (master
# fb329e0); a final name refuses other bytes
REUSED_SHA256 = {
    "src/depth_d156.py": "a1f7b50ce70395f290b681f336e5a5e95e1428a2ca7c5915818644ad70f26734",
    "src/repair_d156.py": "01211e1ec9b69adf2bcc7b738e80a06f07c229c283f105ced11f05083cbe358e",
    "src/healthcheck_d156.py": "0ec2687ecdd887a28d4b8fb7a4db4b9081b69ea3fcf2b175c345279b4ed74de7",
    "src/poolstudy_d156.py": "e39614ed8f7e86f9f5251aaad25f44329c56f4fa3b70c3ffb77f8ce2ffb64908",
    "src/pool_propositions.py": "0adb4937c212ee46235c7398d9800b8dc6126ea5a331bfde663870278c80458d",
    "src/paired.py": "80429fa57f0267236a2af1670c74ef327802598257ab75834fddf64a70d97195",
}
PREREG = ("D-384 (the author's step-4 item 4, an operational case study); details by the closest D-entries (D-181): "
          "D-156, D-164, D-241, D-248, D-255, D-282, D-289, D-306, D-322, D-330, D-363 (4)")
LEVELS = D.LEVELS                                      # item level (primary), cluster level (beside)
PSET = "selected"                                      # D-384 (1): the selected C only
REGISTERED = {"n_draws": P.N_DRAWS, "seeds": [0, 1, 2, 3, 4]}
TARGETS = {"fpr1": P.TARGET_FPR, "fpr5": D.TARGET_5}   # 1% primary, 5% beside (D-248)
PRIMARY = ("fpr1", LEVELS[0])
PAIRS = ("llama3_to_31", "mistral_base2instruct")
PAIR_ROLE = {"llama3_to_31": "the author's choice (D-384 (1))",
             "mistral_base2instruct": "the released pair with the largest Table I paired ratio at L* (D-384 (1)'s rule)"}
RULE_ROW = "depth_mistral_base2instruct"
# D-384 (2): 100,000 requests at 1% harmful
N_REQUESTS = 100_000
N_HARMFUL = N_REQUESTS // 100
N_BENIGN = N_REQUESTS - N_HARMFUL
assert (N_HARMFUL, N_BENIGN) == (1_000, 99_000)
R_SAME_ARM = "quantile_thr_benign"                     # Table II's stored-data threshold repair (4d A, D-289, D-330)
TRIGGER_CLASS = "two_sided_2x"
TRIGGER_CUT = HC.CLASSES[TRIGGER_CLASS]                # (2.0, 0.5), 4c's registered cut (D-241 (3)-(4))
TRIGGERS = {"gaussian": "gaussian_ratio", "direct": "direct_ratio"}      # primary, beside
POLICIES = ("FROZEN", "RECALIBRATED_R_same", "CHECK_TRIGGERED_gaussian", "CHECK_TRIGGERED_direct")
COSTS = ("extra_false_refusals", "extra_missed_harmful")
RF_CONFIG = f"score|R_fresh|n={PS.DEFAULT_POOL}"       # 4d B's R_fresh threshold re-set at the default size
RF_POOL = PS.DEFAULT_POOL
LEVEL_KEYS = ("fpr_pre", "tpr_pre", "fpr_post|frozen", "tpr_post|frozen", "floored|frozen",
              "fpr_post|R_same", "tpr_post|R_same", "floored|R_same", *HC.STATS)
RF_KEYS = ("fpr_pre", "tpr_pre", "fpr_post|frozen", "tpr_post|frozen", "floored|frozen",
           "fpr_post|R_fresh", "tpr_post|R_fresh", "floored|R_fresh")
NPZ_KEYS = ("<fpr1|fpr5>|<level>|<name> for name in LEVEL_KEYS (index 0 the frozen split, then the D-156 / D-164 draws in "
            "paired.repartition_draws / cluster_repartition_draws order); rfresh|<name> for name in RF_KEYS (index 0 the "
            "frozen split, then the 200 option-B draws, D-298)")
# records of record and the manifests that name them (repo-relative)
JOB_MANIFESTS = {"depth": "logs/depth_d156_20260926.sha256", "depth5": "logs/fpr5_stage_depth_20260928.sha256",
                 "repair": "logs/repair_d156_4dA_20260930.sha256", "hc": "logs/healthcheck_d156.sha256",
                 "hc5": "logs/healthcheck_d156_fpr5.sha256", "pool": "logs/poolstudy_4dB_B_20261001.sha256"}
SUMMARY_MANIFESTS = {"depth": JOB_MANIFESTS["depth"], "depth5": JOB_MANIFESTS["depth5"], "hc": JOB_MANIFESTS["hc"],
                     "hc5": JOB_MANIFESTS["hc5"], "reissue": "logs/reissue_d305_20261001.sha256",
                     "t1_fpr5": "logs/table1_tpr_d282_20260930.sha256"}
PINNED_MANIFEST_SHA256 = {
    "logs/depth_d156_20260926.sha256": "2d447887c5cdb152de719f0b87a1c93adc48955c8e1078814bb54a4aa7743f4c",
    "logs/fpr5_stage_depth_20260928.sha256": "a379f9fc41cdb9a9946c1d1f0b17cf57660fdcef8a121f513da2cdc209bf7db4",
    "logs/repair_d156_4dA_20260930.sha256": "5f41056ffe26ebb8dfcf6141d7d40be25142d680f3cdbb184ba0ce5ea1d2b757",
    "logs/healthcheck_d156.sha256": "fe1e2bb3350a68b492bf9ca9ca6519db124196535a6f7885ffc942136b522612",
    "logs/healthcheck_d156_fpr5.sha256": "92cab86acf4a2729d014853a9670fe2ebf33dedac83820df804c84316ba02cbd",
    "logs/poolstudy_4dB_B_20261001.sha256": "460609b6e0bbf632eb08d252339322b17331110dae923c7cd4c678de37afb317",
    "logs/reissue_d305_20261001.sha256": "10f8e9d40e640584c19a34063f041f3c3895c7d36f3025dd1ba9b0ccdab702c6",
    "logs/table1_tpr_d282_20260930.sha256": "6a03e9064e785c05864f1256d2e609d8eda4d7e0e3b2a5f4b198f4c7ed03c060",
}
RUN_MANIFEST = "logs/opcase_d384.sha256"               # the lane's manifest job; the summary reads it
UNANCHORED = {
    "R_same at 5%": ("no repair record exists at 5% (4e's 5% arm covers the moment family, D-248): the same reviewed "
                     "function (repair_d156.partition) at target 0.05, beside only; FROZEN at 5% and the 5% health arrays "
                     "are anchored"),
    "CHECK_TRIGGERED at 5%": "R_same at 5% (above) in the triggered draws; the 5% triggers are the 5% health-check npz's",
    "R_fresh FPR_pre on E": ("every draw: the pre model's FPR on E, the evaluation half left after R_FRESH; no record "
                             "holds it (the option-B record holds post FPRs and paired ratios only)"),
    "R_fresh TPR_pre": ("the 200 option-B draws: no record holds TPR_pre on them; the frozen split's equals the depth "
                        "record's tpr_pre.frozen, checked exactly"),
}
PAIR_CONFIG_KEYS = ("selected_C", "selected_layer", "seeds", "n_draws", "draw_seed", "cluster_draw_seed",
                    "selection_sha256", "axis", "pre_repo", "pre_rev", "post_repo", "post_rev", "rfresh")
TABLE_I = "results/depth_d156_summary_r2.json"         # Table I's record (D-305; tables_d156)
T1_TPR = {"fpr1": "results/table1_tpr_d282_r2.json", "fpr5": "results/table1_tpr_d282_fpr5.json"}
T1_MANIFEST = {"fpr1": "reissue", "fpr5": "t1_fpr5"}
CAVEAT = ("D-255's flag (rule 5): the 4c reference set, R_same's cal_d and the evaluation part come from one pool of "
          "negatives re-partitioned per draw, and the benign traffic is the pair's benign evaluation prompts; a deployer "
          "whose reference prompts or traffic differ would see other costs and trigger rates")


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def record_paths(core: str) -> dict[str, str]:
    """The records of record of one pair (repo-relative)."""
    return {"depth": f"results/depth_d156_{core}.json", "depth5": f"results/depth_d156_{core}_fpr5.json",
            "repair": f"results/repair_d156_{core}.json", "hc": f"results/healthcheck_d156_{core}.json",
            "hc5": f"results/healthcheck_d156_{core}_fpr5.json", "pool": f"results/poolstudy_d156_{core}_originstrat.json"}


def rel(p: Path, root: Path = ROOT) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(root.resolve())).replace("\\", "/") if p.is_relative_to(root.resolve()) else str(p)


def blas_threads() -> dict[str, str | None]:
    return {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}


def final_problems(threads: dict[str, str | None], root: Path = ROOT, manifests: tuple[str, ...] = ()) -> list[str]:
    """Why a final name may not be written (empty when it may)."""
    out = []
    if any(t != "6" for t in threads.values()):
        out.append(f"no 6-thread BLAS pin (D-152 P6): {threads}")
    if os.environ.get("GUARDRAIL_CACHE_DIRS"):
        out.append("GUARDRAIL_CACHE_DIRS is set; the caches must come from the input registry's directory")
    for m in manifests:
        have = sha256_file(root / m) if (root / m).is_file() else None
        if have != PINNED_MANIFEST_SHA256[m]:
            out.append(f"{m} is not the pinned bytes ({(have or 'missing')[:12]})")
    for f, want in REUSED_SHA256.items():
        have = sha256_file(ROOT / f) if (ROOT / f).is_file() else None
        if have != want:
            out.append(f"{f} is not the reviewed bytes this module reuses ({(have or 'missing')[:12]} vs {want[:12]})")
    return out


def manifest(root: Path, rel_path: str) -> tuple[dict[str, str], str]:
    """({repo-relative path: sha256} of a sha256sum manifest, the sha256 of the bytes parsed); one read."""
    p = root / rel_path
    if not p.is_file():
        refuse(f"{rel_path} is not on disk")
    b = p.read_bytes()
    return {ln.split()[1].lstrip("*"): ln.split()[0] for ln in b.decode("utf-8").splitlines() if ln.strip()}, sha256_bytes(b)


def read_bound(root: Path, rel_path: str, listed: dict[str, str] | None) -> tuple[dict[str, Any], str]:
    """A results JSON read once through read_result (D-297 (1)); refuses unless `listed` (a manifest's mapping) names it
    with these bytes, when given."""
    doc = read_result(root / rel_path)
    h = _RESULT_READS[_repo_rel(root / rel_path)]
    if listed is not None and listed.get(rel_path) != h:
        refuse(f"{rel_path} is not the file its manifest names ({h[:12]})")
    if not isinstance(doc, dict):
        refuse(f"{rel_path} is not a JSON object")
    return doc, h


def npz_bound(root: Path, rel_path: str, listed: dict[str, str], want: str) -> tuple[dict[str, np.ndarray], str]:
    """A per-draw npz read once from its bytes, which must be the manifest's and its record's (want)."""
    b = (root / rel_path).read_bytes()
    h = sha256_bytes(b)
    if listed.get(rel_path) != h or want != h:
        refuse(f"{rel_path} is not the npz its manifest and its record name ({h[:12]})")
    with np.load(io.BytesIO(b), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}, h


def same_bits(a: Any, b: Any) -> bool:
    """Bit-for-bit equality of two per-draw arrays as little-endian float64 (NaN-safe), with equal shapes."""
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.astype("<f8").tobytes() == b.astype("<f8").tobytes()


def cell_of(doc: dict[str, Any], layer: int) -> dict[str, Any]:
    """The record's cell at (layer, selected); refuses none or several."""
    hits = [c for c in doc["cells"] if c["layer"] == layer and c["probe_set"] == PSET]
    if len(hits) != 1:
        refuse(f"{len(hits)} cells at L{layer} {PSET}")
    return hits[0]


def level_block(cell: dict[str, Any], lvl: str) -> dict[str, Any]:
    """A depth cell's block at one level: the item level sits at the cell's top, the cluster level under its key."""
    return cell if lvl == LEVELS[0] else cell[LEVELS[1]]


# ------------------------------------------------------------------------------------------------ the arithmetic
def costs(fpr_pre: np.ndarray, tpr_pre: np.ndarray, fpr_pol: np.ndarray, tpr_pol: np.ndarray) -> dict[str, np.ndarray]:
    """D-384 (2), per draw and signed: extra false refusals = N_BENIGN x (FPR_policy - FPR_pre), extra missed harmful =
    N_HARMFUL x (TPR_pre - TPR_policy), per 100,000 requests at 1% harmful."""
    return {"extra_false_refusals": N_BENIGN * (np.asarray(fpr_pol, "<f8") - np.asarray(fpr_pre, "<f8")),
            "extra_missed_harmful": N_HARMFUL * (np.asarray(tpr_pre, "<f8") - np.asarray(tpr_pol, "<f8"))}


def triggered(ratio: np.ndarray) -> np.ndarray:
    """4c's flag per draw: the health ratio crosses 2 or 0.5 (healthcheck_d156.crosses at CLASSES['two_sided_2x']).
    NaN never triggers; +inf triggers (it crosses 2), as in 4c."""
    return np.asarray(HC.crosses(np.asarray(ratio, dtype=float), TRIGGER_CUT), dtype=bool)


def policies(a: dict[str, np.ndarray], flags: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Per policy its per-draw FPR, TPR and floored indicator; CHECK-TRIGGERED takes R_same where its trigger flags the
    draw and FROZEN elsewhere, draw by draw."""
    out = {"FROZEN": {"fpr": a["fpr_post|frozen"], "tpr": a["tpr_post|frozen"], "floored": a["floored|frozen"]},
           "RECALIBRATED_R_same": {"fpr": a["fpr_post|R_same"], "tpr": a["tpr_post|R_same"], "floored": a["floored|R_same"]}}
    n = len(a["fpr_pre"])
    for t, f in flags.items():
        f = np.asarray(f, dtype=bool)
        if f.shape != (n,):
            refuse(f"the {t} trigger has {f.shape} draws, the costs {n}")
        out[f"CHECK_TRIGGERED_{t}"] = {k: np.where(f, out["RECALIBRATED_R_same"][k], out["FROZEN"][k])
                                       for k in ("fpr", "tpr", "floored")}
    return out


def cost_cell(a: dict[str, np.ndarray], flags: dict[str, np.ndarray]) -> dict[str, Any]:
    """One cell (pair, target, level): per policy each cost's D-156 summary (index 0 the frozen split), the share of
    draws with a floored seed, and per non-frozen policy the within-draw saving against FROZEN."""
    n = len(a["fpr_pre"])
    if any(len(np.asarray(v)) != n for v in a.values()):
        refuse("the per-draw arrays of one cell differ in length")
    pol = policies(a, flags)
    per = {name: costs(a["fpr_pre"], a["tpr_pre"], p["fpr"], p["tpr"]) for name, p in pol.items()}
    out: dict[str, Any] = {"n_draws": n - 1}
    for name, p in pol.items():
        blk: dict[str, Any] = {q: P.band_summary(v[1:], v[0]) for q, v in per[name].items()}
        fl = np.asarray(p["floored"], dtype=bool)
        blk["share_of_draws_floored"] = float(fl[1:].mean())
        blk["frozen_split_floored"] = bool(fl[0])
        if name != "FROZEN":
            blk["saving_vs_frozen"] = {q: P.band_summary((per["FROZEN"][q] - v)[1:], (per["FROZEN"][q] - v)[0])
                                       for q, v in per[name].items()}
        if name.startswith("CHECK_TRIGGERED_"):
            f = np.asarray(flags[name[len("CHECK_TRIGGERED_"):]], dtype=bool)
            blk["share_of_draws_triggered"] = float(f[1:].mean())
            blk["frozen_split_triggered"] = bool(f[0])
            blk["identical_to"] = ("RECALIBRATED_R_same" if f.all() else "FROZEN" if not f.any() else None)
            blk["identical_to_note"] = ("the policy in the frozen split and every draw, read from the trigger array; None "
                                        "when it mixes the two")
        out[name] = blk
    out["trigger"] = {t: {"share_of_draws_triggered": float(np.asarray(f, bool)[1:].mean()),
                          "frozen_split_triggered": bool(np.asarray(f, bool)[0])} for t, f in flags.items()}
    return out


def rfresh_cell(a: dict[str, np.ndarray]) -> dict[str, Any]:
    """R_fresh beside (1%, item level, the option-B draws): FROZEN and R_FRESH on the evaluation half E, the saving
    against FROZEN within each draw."""
    per = {"FROZEN": costs(a["fpr_pre"], a["tpr_pre"], a["fpr_post|frozen"], a["tpr_post|frozen"]),
           "RECALIBRATED_R_fresh": costs(a["fpr_pre"], a["tpr_pre"], a["fpr_post|R_fresh"], a["tpr_post|R_fresh"])}
    out: dict[str, Any] = {"n_draws": len(a["fpr_pre"]) - 1}
    for name, fl_key in (("FROZEN", "floored|frozen"), ("RECALIBRATED_R_fresh", "floored|R_fresh")):
        blk: dict[str, Any] = {q: P.band_summary(v[1:], v[0]) for q, v in per[name].items()}
        fl = np.asarray(a[fl_key], dtype=bool)
        blk["share_of_draws_floored"] = float(fl[1:].mean())
        blk["frozen_split_floored"] = bool(fl[0])
        if name != "FROZEN":
            blk["saving_vs_frozen"] = {q: P.band_summary((per["FROZEN"][q] - v)[1:], (per["FROZEN"][q] - v)[0])
                                       for q, v in per[name].items()}
        out[name] = blk
    return out


# ------------------------------------------------------------------------------------------------ the job's computation
def lstar_scores(X: dict[str, dict[str, np.ndarray]], y_tr: np.ndarray, ncal: np.ndarray, neg: np.ndarray,
                 pos: np.ndarray, hcal: np.ndarray, C_sel: float, seeds: list[int]
                 ) -> tuple[list[P.Probe], np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """repair_d156.main's probes and scores at one layer for the selected probe set (its lines, repeated): per seed the
    D-117 fit rows, the selected C in permutation order (poolstudy_d156.probe_spec), scored per part then stacked."""
    probes = []
    for seed in seeds:
        rows, C, order = PS.probe_spec(len(y_tr), seed, PSET, C_sel)
        probes.append(P.fit_probe(X["pre"]["probe_train"][rows], y_tr[rows], seed, C, order))
    S = lambda x: np.column_stack([q.score(x) for q in probes])  # noqa: E731
    cal_pre_x, ev_pre_x = X["pre"]["threshold_calib"], X["pre"]["eval"]
    cal_post_x, ev_post_x = X["post"]["threshold_calib"], X["post"]["eval"]
    pre_pool = np.vstack([S(cal_pre_x[ncal]), S(ev_pre_x[neg])])
    pre_pos, pre_H = S(ev_pre_x[pos]), S(cal_pre_x[hcal])
    post = {"pool": np.vstack([S(cal_post_x[ncal]), S(ev_post_x[neg])]), "pos": S(ev_post_x[pos]), "H": S(cal_post_x[hcal])}
    return probes, pre_pool, pre_pos, pre_H, post


def load_caches(reg_stem: str) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, str], dict[str, dict[str, Any]]]:
    """The pair's six caches through the input registry (repair_d156.main's loading): the declared run, the caches, the
    sha256 of each input and the registry's record of each."""
    cfg = REG.run(reg_stem)
    data: dict[str, dict[str, Any]] = {"pre": {}, "post": {}}
    input_sha: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    for side in data:
        for part in ("probe_train", "threshold_calib", "eval"):
            z, r = REG.load(cfg[f"{side}_repo"], cfg["axis"], part)
            data[side][part] = z
            input_sha[f"{side}/{part}"], input_recs[f"{side}/{part}"] = r["sha256"], r
    return cfg, data, input_sha, input_recs


def lstar_inputs(data: dict[str, dict[str, Any]], L_sel: int, C_sel: float, seeds: list[int]) -> dict[str, Any]:
    """From the caches to every score at L* (repair_d156.main's and poolstudy_d156.main's wiring, their lines repeated):
    the checks on items, labels, grids and splits (D-228), the masks and the pooled ids (threshold_calib negatives, then
    eval negatives, cache order), the probes and both scorings: repair_d156's (each part scored, then stacked: FROZEN,
    R_same and 4c) and poolstudy_d156's (raw cache rows stacked, then scored: R_fresh)."""
    for part in ("probe_train", "threshold_calib", "eval"):
        a, b = data["pre"][part], data["post"][part]
        if not np.array_equal(a["item_ids"], b["item_ids"]) or not np.array_equal(a["labels"], b["labels"]):
            refuse(f"{part} items or labels differ between pre and post")
        for z in (a, b):
            if "nonfinite" in z.files and z["nonfinite"].any():
                refuse(f"{part} has non-finite items")
    grids = [sorted(int(f.split("_")[1]) for f in z.files if f.startswith("layer_")) for s in data for z in data[s].values()]
    if any(g != grids[0] for g in grids) or L_sel not in grids[0]:
        refuse(f"the caches' layer grids differ or hold no L{L_sel}")
    y_tr = data["pre"]["probe_train"]["labels"].astype(bool)
    y_cal = data["pre"]["threshold_calib"]["labels"].astype(bool)
    y_ev = data["pre"]["eval"]["labels"].astype(bool)
    ncal, hcal, neg, pos = ~y_cal, y_cal, ~y_ev, y_ev
    pooled_ids = [str(i) for i in data["pre"]["threshold_calib"]["item_ids"][ncal]] + \
                 [str(i) for i in data["pre"]["eval"]["item_ids"][neg]]
    H_ids = {str(i) for i in data["pre"]["threshold_calib"]["item_ids"][hcal]}
    ev_ids = {str(i) for i in data["pre"]["eval"]["item_ids"]}
    tr_ids = {str(i) for i in data["pre"]["probe_train"]["item_ids"]}
    if H_ids & set(pooled_ids) or H_ids & ev_ids or tr_ids & set(pooled_ids) or tr_ids & ev_ids:
        refuse("a threshold_calib harmful or probe_train item is in the pool or in eval (D-228)")
    lay = f"layer_{L_sel}"
    X = {s: {p: data[s][p][lay] for p in ("probe_train", "threshold_calib", "eval")} for s in data}
    probes, pre_pool, pre_pos, pre_H, post = lstar_scores(X, y_tr, ncal, neg, pos, hcal, C_sel, seeds)
    RAW = {s: {"pool": np.vstack([data[s]["threshold_calib"][lay][ncal], data[s]["eval"][lay][neg]]),
               "H": data[s]["threshold_calib"][lay][hcal], "pos": data[s]["eval"][lay][pos]} for s in data}
    SC = {side: {part: np.column_stack([q.score(RAW[side][part]) for q in probes]) for part in ("pool", "H", "pos")}
          for side in ("pre", "post")}
    return {"pooled_ids": pooled_ids, "n_cal": int(ncal.sum()), "n_pool": len(pooled_ids), "n_h": int(hcal.sum()),
            "probes": probes, "pre_pool": pre_pool, "pre_pos": pre_pos, "pre_H": pre_H, "post": post, "SC": SC}


def level_arrays(parts: list[tuple[np.ndarray, np.ndarray]], pre_pool: np.ndarray, pre_pos: np.ndarray,
                 pre_H: np.ndarray, post: dict[str, np.ndarray], target: float
                 ) -> tuple[list[dict[str, dict[str, float]]], dict[str, np.ndarray]]:
    """Every partition (frozen split first): repair_d156.partition with the FROZEN and R_same arms, the pre side's
    floored FPR at T_d, and healthcheck_d156.health at T_d. Returns partition()'s outputs (for repair_d156.summarise) and
    the per-draw arrays of LEVEL_KEYS."""
    arms = {"frozen": post, R_SAME_ARM: post}
    per: list[dict[str, dict[str, float]]] = []
    rows: dict[str, list[Any]] = {k: [] for k in LEVEL_KEYS}
    for c, e in parts:
        p = RP.partition(c, e, pre_pool, pre_pos, pre_H, arms, target)
        T = HC.thresholds(pre_pool[c], target)
        f_pre = float(P.floored_fpr(pre_pool[e] >= T).mean())
        # the pre side is partition()'s: the same T_d, the same FPR (point()'s floor_ratio_to_target) and TPR in both arms
        if f_pre / target != p["frozen"]["floor_ratio_to_target"] or p["frozen"]["tpr_pre"] != p[R_SAME_ARM]["tpr_pre"] \
                or p["frozen"]["floor_ratio_to_target"] != p[R_SAME_ARM]["floor_ratio_to_target"]:
            refuse("the pre side differs from repair_d156.partition's")
        h = HC.health(pre_pool[c], post["pool"][c], T)
        per.append(p)
        rows["fpr_pre"].append(f_pre)
        rows["tpr_pre"].append(p["frozen"]["tpr_pre"])
        for arm, nm in (("frozen", "frozen"), (R_SAME_ARM, "R_same")):
            rows[f"fpr_post|{nm}"].append(p[arm]["fpr_post"])
            rows[f"tpr_post|{nm}"].append(p[arm]["tpr_post"])
            rows[f"floored|{nm}"].append(p[arm]["floored"])
        for q in HC.STATS:
            rows[q].append(p["frozen"]["paired_ratio"] if q == "realised_paired_ratio" else h[q])
    out = {k: np.asarray(v, dtype=bool) if k.startswith("floored|") else np.asarray(v, dtype="<f8") for k, v in rows.items()}
    return per, out


def anchor_level(label: str, arrays: dict[str, np.ndarray], per: list[dict[str, dict[str, float]]],
                 rep: dict[str, Any], arr: dict[str, np.ndarray], depth_blk: dict[str, Any],
                 repair_blk: dict[str, Any] | None, hc_z: dict[str, np.ndarray], hc_prefix: str) -> list[str]:
    """The anchors of one (target, level) block; refuses at the first failure, else returns the names of the checks."""
    done = []
    # (a) the generic path is paired.repartition_estimate's, draw by draw (repair_d156.main's check)
    for mine, theirs in (("fpr_pre", "fpr_pre"), ("tpr_pre", "tpr_pre"), ("fpr_post|frozen", "fpr_post"),
                         ("tpr_post|frozen", "tpr_post"), ("realised_paired_ratio", "paired_ratio")):
        want = np.concatenate([[rep[theirs]["frozen"]], arr[theirs]]).astype("<f8")
        if not same_bits(arrays[mine], want):
            refuse(f"{label}: {mine} is not paired.repartition_estimate's {theirs}, draw by draw")
    done.append("generic path = paired.repartition_estimate (5 arrays, bit for bit)")
    # (b) the depth record's operating point and its per-draw FPR_post / target array
    if RP.canonical(rep) != RP.canonical(depth_blk["operating_point"]):
        refuse(f"{label}: the operating point is not the depth record's")
    rt = np.concatenate([[rep["ratio_to_target"]["frozen"]], arr["ratio_to_target"]]).astype("<f8")
    if sha256_bytes(rt.tobytes()) != depth_blk["frozen_curve_abs_log_ratio_to_target"]["per_draw_ratio_to_target_sha256"]:
        refuse(f"{label}: the per-draw FPR_post / target array does not hash to the depth record's")
    done.append("depth operating point (canonical JSON) and per_draw_ratio_to_target_sha256")
    # (c) the repair record's FROZEN and R_same blocks
    if repair_blk is not None:
        for arm in ("frozen", R_SAME_ARM):
            if RP.canonical(RP.summarise(per, arm)) != RP.canonical(repair_blk[arm]):
                refuse(f"{label}: repair_d156.summarise of '{arm}' is not the repair record's block")
        done.append(f"repair record's 'frozen' and '{R_SAME_ARM}' blocks (canonical JSON)")
    # (d) the health-check npz: every statistic and the realised paired ratio, bit for bit (the trigger's draws)
    for q in HC.STATS:
        k = f"{hc_prefix}|{q}"
        if k not in hc_z or not same_bits(arrays[q], hc_z[k]):
            refuse(f"{label}: {q} is not the health-check npz's {k}, draw by draw")
    done.append(f"health-check npz {hc_prefix}|<6 statistics> (bit for bit)")
    return done


def rfresh_arrays(SC: dict[str, dict[str, np.ndarray]], n_cal: int, n_pool: int, n_h_src: int, axis: str,
                  target: float, n_reps: int = PS.N_REPS) -> tuple[dict[str, np.ndarray], dict[str, dict[str, list[float]]]]:
    """poolstudy_d156.main's lines for the configurations 'frozen' and 'score|R_fresh|n=1000' (selected C, option B),
    repeated: the option-B partitions and each repetition's pools (poolstudy_d156.partitions and draw_pools, unchanged);
    T_d each seed's threshold on cal_d's pre scores; FPR on E, the evaluation half left after R_FRESH; R_FRESH's
    threshold re-set at the target on its post scores. Returns the per-draw arrays of RF_KEYS (with FPR_pre on E) and
    put()'s per-draw quantities, for the anchor on the record."""
    S = SC
    all_post = np.vstack([S["post"]["pool"], S["post"]["H"]])
    parts = PS.partitions(n_cal, n_pool, n_reps, axis, True)
    rows: dict[str, list[Any]] = {k: [] for k in RF_KEYS}
    acc: dict[str, dict[str, list[float]]] = {}
    for di, (cal, ev) in enumerate(parts):
        dp = PS.draw_pools(di, cal, ev, n_cal, n_pool, n_h_src, True)
        E, rf = dp["E"], dp["R_fresh_subsets"][RF_POOL]
        e_tc = E < n_cal
        ns = S["pre"]["pool"].shape[1]
        T = np.array([P.thr_at_fpr(S["pre"]["pool"][cal, s], target) for s in range(ns)])
        F_pre = S["pre"]["pool"][E] >= T
        tr = np.array([P.thr_at_fpr(all_post[rf, s], target) for s in range(ns)])
        arms = {"frozen": (S["post"]["pool"][E] >= T, S["post"]["pos"] >= T),
                RF_CONFIG: (S["post"]["pool"][E] >= tr, S["post"]["pos"] >= tr)}
        rows["fpr_pre"].append(float(P.floored_fpr(F_pre).mean()))
        rows["tpr_pre"].append(float(P.rate(S["pre"]["pos"] >= T).mean()))
        for name, (F, Q) in arms.items():
            # poolstudy_d156.main's put(), option B
            a = acc.setdefault(name, {"fpr": [], "paired_ratio": [], "ratio_to_target": [], "tpr": [],
                                      "fpr_origin_tc": [], "fpr_origin_ev": []})
            f = float(P.floored_fpr(F).mean())
            a["fpr"].append(f)
            a["paired_ratio"].append(P.ratio_from_flags(F_pre, F))
            a["ratio_to_target"].append(f / target)
            a["tpr"].append(float(P.rate(Q).mean()))
            a["fpr_origin_tc"].append(float(P.floored_fpr(F[e_tc]).mean()) if e_tc.any() else float("nan"))
            a["fpr_origin_ev"].append(float(P.floored_fpr(F[~e_tc]).mean()) if (~e_tc).any() else float("nan"))
            nm = "frozen" if name == "frozen" else "R_fresh"
            rows[f"fpr_post|{nm}"].append(f)
            rows[f"tpr_post|{nm}"].append(float(P.rate(Q).mean()))
            rows[f"floored|{nm}"].append(bool(((F_pre.sum(axis=0) == 0) | (F.sum(axis=0) == 0)).any()))
    out = {k: np.asarray(v, dtype=bool) if k.startswith("floored|") else np.asarray(v, dtype="<f8") for k, v in rows.items()}
    return out, acc


def anchor_rfresh(acc: dict[str, dict[str, list[float]]], pool_sel: dict[str, Any], tpr_pre: np.ndarray,
                  depth_op: dict[str, Any]) -> list[str]:
    """Both configurations' blocks equal the option-B record's exactly (poolstudy_d156.main's band_summary per quantity),
    and TPR_pre at the frozen split equals the depth record's tpr_pre.frozen (the same quantity); the rest of the
    baseline is in UNANCHORED."""
    for name in ("frozen", RF_CONFIG):
        mine = {q: P.band_summary(np.array(v[1:]), v[0]) for q, v in acc[name].items()}
        if name not in pool_sel or RP.canonical(mine) != RP.canonical(pool_sel[name]):
            refuse(f"R_fresh: the '{name}' configuration is not the option-B record's")
    if float(tpr_pre[0]) != depth_op["tpr_pre"]["frozen"]:
        refuse("R_fresh: TPR_pre at the frozen split is not the depth record's tpr_pre.frozen")
    return [f"option-B record's 'frozen' and '{RF_CONFIG}' configurations (canonical JSON)",
            "TPR_pre at the frozen split = the depth record's tpr_pre.frozen"]


# ------------------------------------------------------------------------------------------------ the job
def run_pair(args: argparse.Namespace) -> int:
    t_start = time.time()
    core = args.pair
    results = ROOT / "results"
    final_path = (results / f"opcase_d384_{core}.json").resolve()
    out_path = Path(args.out).resolve() if args.out else final_path
    npz_path = out_path.with_name(out_path.stem + "_draws.npz")
    final = out_path == final_path
    for p in (out_path, npz_path):
        if p.exists():
            refuse(f"{p} exists; results are never overwritten (D-152 P5)")
    if not final and out_path.is_relative_to(results.resolve()):
        refuse("a test run (--out) must not write under results/; only the run's own final name may")
    threads = blas_threads()
    if final:
        problems = final_problems(threads, ROOT, tuple(JOB_MANIFESTS.values()))
        if problems:
            refuse(f"the final name: {problems}; use --out outside results/ for a test")
    seeds, n_draws = REGISTERED["seeds"], REGISTERED["n_draws"]
    code0 = P.code_block(CODE_FILES)

    # --- the records of record, each bound to its manifest ----------------------------------------------------------
    man = {k: manifest(ROOT, m) for k, m in JOB_MANIFESTS.items()}
    if final and any(man[k][1] != PINNED_MANIFEST_SHA256[m] for k, m in JOB_MANIFESTS.items()):
        refuse("a manifest read is not its pinned bytes (PINNED_MANIFEST_SHA256)")
    rp = record_paths(core)
    docs: dict[str, dict[str, Any]] = {}
    shas: dict[str, str] = {}
    for k, r in rp.items():
        docs[k], shas[k] = read_bound(ROOT, r, man[k][0])
    hc_z: dict[str, dict[str, np.ndarray]] = {}
    hc_npz: dict[str, dict[str, str]] = {}
    for k in ("hc", "hc5"):
        f = docs[k]["draws_npz"]["file"]
        if f != rp[k][:-len(".json")] + "_draws.npz":
            refuse(f"{rp[k]} names the npz {f}")
        hc_z[k], h = npz_bound(ROOT, f, man[k][0], docs[k]["draws_npz"]["sha256"])
        hc_npz[k] = {"file": f, "sha256": h}
    reg_stem = D.registry_stem(f"depth_{core}")
    sel_stem = D.SELECTION_STEM.get(f"depth_{core}", reg_stem)
    sel_doc, sel_sha = read_bound(ROOT, rel(D.SELECTION_OF_RECORD), None)
    sel = next((m for m in sel_doc["models"] if m["pair"] == sel_stem), None)
    if sel is None or not sel["selected"]["arm"].startswith("C="):
        refuse(f"no logistic-C selection for {sel_stem}")
    C_sel, L_sel = float(sel["selected"]["arm"].split("=")[1]), int(sel["selected"]["layer"])
    cl_bytes = D.CLUSTERS.read_bytes()
    cl_sha = sha256_bytes(cl_bytes)
    cl_doc = json.loads(cl_bytes.decode("utf-8"))  # not a result (D-297): manifests/negative_clusters.json, hashed and listed
    dc, d5, rc, hc, h5, pc = (docs[k]["config"] for k in ("depth", "depth5", "repair", "hc", "hc5", "pool"))
    sel_of = lambda c, lk="selected_layer": (c["selected_C"], c[lk], c["selection_sha256"])  # noqa: E731
    draws_of = lambda c: (list(c["seeds"]), c["n_draws"], c["draw_seed"], c["cluster_draw_seed"])  # noqa: E731
    checks = [("selection", [sel_of(dc), sel_of(d5), sel_of(rc), sel_of(hc), sel_of(h5), sel_of(pc, "layer")],
               [(C_sel, L_sel, sel_sha)] * 6),
              ("registry run", [dc.get("registry_pair"), d5.get("registry_pair"), rc.get("registry_pair"), hc.get("registry_pair"),
                                h5.get("registry_pair"), pc.get("registry_pair")], [reg_stem] * 6),
              ("draws (seeds, n_draws, draw_seed, cluster_draw_seed)", [draws_of(dc), draws_of(d5), draws_of(rc), draws_of(hc),
                                                                        draws_of(h5)], [draws_of(dc)] * 5),
              ("registered draws", draws_of(dc), (seeds, n_draws, P.AXIS_DRAW_SEED[dc["axis"]], P.AXIS_CLUSTER_DRAW_SEED[dc["axis"]])),
              ("targets", [dc["target_fpr"], d5["target_fpr"], rc["target_fpr"], hc["target_fpr"], h5["target_fpr"], pc["target_fpr"]],
               [TARGETS["fpr1"], TARGETS["fpr5"], TARGETS["fpr1"], TARGETS["fpr1"], TARGETS["fpr5"], TARGETS["fpr1"]]),
              ("cluster manifest", [dc["cluster_manifest"]["sha256"], d5["cluster_manifest"]["sha256"], rc["cluster_manifest"]["sha256"],
                                    hc["cluster_manifest"]["sha256"], h5["cluster_manifest"]["sha256"], pc["cluster_manifest"]["sha256"]],
               [cl_sha] * 6),
              ("anchors name the depth records", [docs["repair"]["anchor"]["sha256"], docs["hc"]["anchor"]["sha256"],
                                                  docs["hc5"]["anchor"]["sha256"], docs["depth5"]["fpr5_self_anchor"]["sha256"],
                                                  docs["pool"]["anchor"]["sha256"]],
               [shas["depth"], shas["depth"], shas["depth5"], shas["depth"], shas["depth"]]),
              ("repair family", (rc["family"], rc["pair"]), ("repair", f"repair_{core}")),
              ("option-B pool study", (pc["seeds"], pc["n_reps"], pc["pool_seed"], pc["strat_draw_seed"], pc["default_pool"],
                                       str(pc["design"]).startswith("option B")),
               (seeds, PS.N_REPS, PS.POOL_SEED, PS.STRAT_DRAW_SEED, PS.DEFAULT_POOL, True)),
              ("registered records", [docs[k]["registered_parameters"] for k in rp], [True] * len(rp))]
    bad = [w for w, a, b in checks if RP.canonical(a) != RP.canonical(b)]
    if bad:
        refuse(f"the records of record disagree with this run or with each other: {bad}")

    # --- the caches through the input registry, the bytes every record read (repair_d156.main's loading) -----------
    run_decl = REG.run(reg_stem)
    if run_decl["axis"] != dc["axis"] or any(run_decl[f] != dc.get(f) for f in ("pre_repo", "pre_rev", "post_repo", "post_rev")):
        refuse(f"the input registry's {reg_stem} is not the depth record's run")
    cfg, data, input_sha, input_recs = load_caches(reg_stem)
    axis = cfg["axis"]
    reg_sources = REG.sources_sha256()
    if D.sources_now() != reg_sources:
        refuse("the extraction records or pins on disk are not the ones the registry read")
    other = [k for k in rp if any(docs[k]["input_cache_sha256"].get(kk) != h for kk, h in input_sha.items())]
    if other:
        refuse(f"records of record read other caches: {other}")
    verification = REG.verification(input_recs)
    s = lstar_inputs(data, L_sel, C_sel, seeds)
    pooled_ids, n_cal, n_pool, n_h = s["pooled_ids"], s["n_cal"], s["n_pool"], s["n_h"]
    pre_pool, pre_pos, pre_H, post = s["pre_pool"], s["pre_pos"], s["pre_H"], s["post"]
    cl = cl_doc["axes"][axis]
    if cl["n_pooled_negatives"] != n_pool or cl["n_calibration"] != n_cal or \
            cl["pool_item_ids_sha256"] != sha256_bytes("\n".join(pooled_ids).encode("utf-8")):
        refuse(f"manifests/negative_clusters.json does not describe the {axis} pool")
    frozen_part = (np.arange(n_cal), np.arange(n_cal, n_pool))
    draws = {LEVELS[0]: P.repartition_draws(n_cal, n_pool - n_cal, n_draws, P.AXIS_DRAW_SEED[axis]),
             LEVELS[1]: P.cluster_repartition_draws(P.cluster_units(pooled_ids, cl["clusters"]), n_cal, n_draws,
                                                    P.AXIS_CLUSTER_DRAW_SEED[axis])}
    parts = {lvl: [frozen_part] + [(d[:n_cal], d[n_cal:]) for d in dr] for lvl, dr in draws.items()}
    print(f"D-384 {core} via {reg_stem} [{axis}]: L*={L_sel}, C={C_sel:g}; pooled {n_cal} + {n_pool - n_cal}; "
          f"H {n_h}; D-170 flag {verification['flag']}; final {final}", flush=True)

    arrays: dict[str, np.ndarray] = {}
    anchors: dict[str, list[str]] = {}
    for tgt, target in TARGETS.items():
        dcell = cell_of(docs["depth" if tgt == "fpr1" else "depth5"], L_sel)
        repcell = cell_of(docs["repair"], L_sel) if tgt == "fpr1" else None
        for lvl in LEVELS:
            t0 = time.time()
            per, a = level_arrays(parts[lvl], pre_pool, pre_pos, pre_H, post, target)
            rep, arr = P.repartition_estimate(pre_pool[:n_cal], pre_pool[n_cal:], post["pool"][:n_cal], post["pool"][n_cal:],
                                              pre_pos, post["pos"], draws[lvl], target)
            label = f"{tgt} {lvl}"
            anchors[f"{tgt}|{lvl}"] = anchor_level(label, a, per, rep, arr, level_block(dcell, lvl),
                                                   repcell[lvl] if repcell is not None else None,
                                                   hc_z["hc" if tgt == "fpr1" else "hc5"], f"L{L_sel}|{PSET}|{lvl}")
            arrays.update({f"{tgt}|{lvl}|{kk}": v for kk, v in a.items()})
            print(f"  {label}: {len(anchors[f'{tgt}|{lvl}'])} anchor groups PASS ({time.time() - t0:.1f} s)", flush=True)
    # R_fresh beside, 1%, item level, option B: poolstudy_d156.main's scoring (raw cache rows stacked, then scored)
    t0 = time.time()
    rf, acc = rfresh_arrays(s["SC"], n_cal, n_pool, n_h, axis, TARGETS["fpr1"])
    anchors["rfresh"] = anchor_rfresh(acc, docs["pool"]["configurations"][PSET], rf["tpr_pre"],
                                      cell_of(docs["depth"], L_sel)["operating_point"])
    arrays.update({f"rfresh|{kk}": v for kk, v in rf.items()})
    print(f"  R_fresh (option B, {len(rf['fpr_pre']) - 1} draws): PASS ({time.time() - t0:.1f} s)", flush=True)

    # --- nothing read may have changed while the run went ------------------------------------------------------------
    code1 = P.code_block(CODE_FILES)
    changed = [w for w, a, b in (("code", code1["sha256"], code0["sha256"]), ("registry sources", D.sources_now(), reg_sources),
                                 ("cluster manifest", sha256_file(D.CLUSTERS), cl_sha)) if a != b]
    gone_or = lambda r_, h: not (ROOT / r_).is_file() or sha256_file(ROOT / r_) != h  # noqa: E731
    changed += [r_ for r_, h in _RESULT_READS.items() if gone_or(r_, h)]
    changed += [v["file"] for v in hc_npz.values() if gone_or(v["file"], v["sha256"])]
    changed += [m for kk, m in JOB_MANIFESTS.items() if gone_or(m, man[kk][1])]
    changed += [f"cache {r['file']}" for r in input_recs.values() if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    if changed:
        refuse(f"changed during the run: {changed}")

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    npz_bytes = buf.getvalue()
    config = {"pair": core, "role": PAIR_ROLE[core], "registry_pair": reg_stem, "selection_of_pair": sel_stem, "axis": axis,
              "selection_file": rel(D.SELECTION_OF_RECORD), "selection_sha256": sel_sha, "selected_C": C_sel,
              "selected_layer": L_sel, "probe_set": f"{PSET} (the D-117 selected C, permutation order; C=1 not computed)",
              "seeds": seeds, "targets": TARGETS, "n_draws": n_draws, "draw_seed": P.AXIS_DRAW_SEED[axis],
              "cluster_draw_seed": P.AXIS_CLUSTER_DRAW_SEED[axis], "levels": list(LEVELS),
              "cluster_manifest": {"file": rel(D.CLUSTERS), "sha256": cl_sha},
              "r_same_arm": f"repair_d156.partition '{R_SAME_ARM}' (Table II's stored-data threshold repair)",
              "rfresh": {"configuration": RF_CONFIG, "design": "option B (D-298, D-306)", "n_reps": PS.N_REPS,
                         "pool_seed": PS.POOL_SEED, "strat_draw_seed": PS.STRAT_DRAW_SEED, "target_fpr": TARGETS["fpr1"],
                         "evaluation": pc["evaluation"]},
              "pre_repo": cfg["pre_repo"], "pre_rev": cfg["pre_rev"], "post_repo": cfg["post_repo"], "post_rev": cfg["post_rev"],
              "inputs_from": "src/input_registry.py (D-165)", "input_registry_sources_sha256": reg_sources}
    records = {k: {"file": rp[k], "sha256": shas[k]} for k in rp}
    records.update({f"{k}_npz": v for k, v in hc_npz.items()})
    records["selection"] = {"file": rel(D.SELECTION_OF_RECORD), "sha256": sel_sha}
    payload: dict[str, Any] = {
        **provenance_block("opcase_d384_pair"), "preregistration": PREREG, "code": code0,
        "reused_files_sha256": {f: sha256_file(ROOT / f) for f in REUSED_SHA256},
        "registered_parameters": True, "blas_threads": threads, "elapsed_s": round(time.time() - t_start, 1),
        "input_verification": verification,
        "stored_input_verification": {k: docs[k].get("input_verification") for k in rp},
        "config": config, "config_hash": canonical_hash(config),
        "anchor": {"passed": anchors, "n_groups": sum(len(v) for v in anchors.values()),
                   "unanchored": UNANCHORED},
        "records_of_record": records,
        "manifests": {JOB_MANIFESTS[kk]: man[kk][1] for kk in JOB_MANIFESTS},
        "input_files_sha256": {**dict(_RESULT_READS), **{v["file"]: v["sha256"] for v in hc_npz.values()},
                               rel(D.CLUSTERS): cl_sha, **{JOB_MANIFESTS[kk]: man[kk][1] for kk in JOB_MANIFESTS}},
        "input_cache_sha256": input_sha,
        "input_cache_source": {kk: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"]}
                               for kk, r in input_recs.items()},
        "draws_npz": {"file": rel(npz_path), "sha256": sha256_bytes(npz_bytes), "size": len(npz_bytes), "keys": NPZ_KEYS,
                      "n_arrays": len(arrays)},
        "note": "per-draw arrays only; the costs are computed by --summary (results/opcase_d384.json)"}
    write_outputs([(npz_path, npz_bytes), (out_path, json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8"))],
                  payload)
    print(f"elapsed {time.time() - t_start:.1f} s")
    return 0


def write_outputs(items: list[tuple[Path, bytes]], payload: dict[str, Any]) -> None:
    """Stage every output, then name them (hard link, never overwrite, D-152 P5), then mode 444; refuses an existing
    name or a payload that breaks the D-297 (1) results-writer rule, with nothing written."""
    for p, _ in items:
        if Path(p).exists():
            refuse(f"{p} exists; results are never overwritten (D-152 P5)")
    problems = input_identity_problems(payload)
    if problems:
        refuse(f"to write (D-297 (1) results-writer rule): {problems}")
    staged: list[tuple[Path, Path, str]] = []
    committed: list[Path] = []
    try:
        for p, b in items:
            t, h = stage_bytes(p, b)
            staged.append((t, Path(p), h))
        for t, p, h in staged:
            try:
                commit_staged(t, p, h)
            except BaseException as e:
                for q in committed:
                    os.chmod(q, 0o444)
                refuse(f"naming {p} failed ({type(e).__name__}: {e}); already named, now mode 444: "
                       f"{', '.join(str(q) for q in committed) or 'none'}")
            committed.append(p)
    finally:
        for t, _, _ in staged:
            if t.exists():
                t.unlink()
    for _, p, h in staged:
        os.chmod(p, 0o444)
        print(f"Written: {p}\nsha256:  {h}")


# ------------------------------------------------------------------------------------------------ the summary
def pair_rule(t1: dict[str, Any]) -> dict[str, Any]:
    """D-384 (1)'s rule on Table I: the row with the largest headline paired ratio (selected C, item level, D-156 mean)
    must be the Mistral base -> instruct pair; refuses otherwise."""
    means = {k: v["selected"]["item_level_D156"]["headline_selected_layer"]["paired_ratio"]["mean"]
             for k, v in t1["table_1"].items()}
    top = max(means, key=lambda r: means[r])
    if top != RULE_ROW or sum(m == means[top] for m in means.values()) != 1:
        refuse(f"Table I's largest headline paired ratio is {top}'s, not {RULE_ROW}'s (D-384 (1))")
    return {"rule": "the released pair with the largest Table I paired ratio at L* (selected C, item level, D-156 mean)",
            "rows_compared": sorted(means), "largest": top, "paired_ratio_mean": means[top],
            "runner_up": max((r for r in means if r != top), key=lambda r: means[r])}


def alignment(core: str, rec: dict[str, Any], z_op: dict[str, np.ndarray], docs: dict[str, dict[str, Any]],
              shas: dict[str, str], hc_z: dict[str, dict[str, np.ndarray]], hc_npz_sha: dict[str, str]) -> dict[str, Any]:
    """The draw alignment of a pair record with the health-check and depth records: the same selection, seeds, draw
    seeds, draw count and cluster manifest; the records it was anchored on are the ones read here; and per target and
    level the pair npz's health statistics and realised paired ratio equal the health-check npz's bit for bit (the same
    draws in the same order). Refuses at the first failure."""
    c = rec["config"]
    L = c["selected_layer"]
    key = lambda x: (x["selected_C"], x["selected_layer"], x["selection_sha256"], list(x["seeds"]), x["n_draws"],  # noqa: E731
                     x["draw_seed"], x["cluster_draw_seed"], x["cluster_manifest"]["sha256"])
    want = key(c)
    for k in ("depth", "depth5", "hc", "hc5"):
        if RP.canonical(key(docs[k]["config"])) != RP.canonical(want):
            refuse(f"{core}: {k}'s selection, seeds, draws or cluster manifest differ from the pair record's")
        if docs[k]["input_cache_sha256"] != rec["input_cache_sha256"]:
            refuse(f"{core}: {k} read other caches than the pair record")
    named = rec["records_of_record"]
    for k in ("depth", "depth5", "hc", "hc5"):
        if named.get(k, {}).get("sha256") != shas[k]:
            refuse(f"{core}: the pair record was anchored on another {k} record")
    for k in ("hc", "hc5"):
        if named.get(f"{k}_npz", {}).get("sha256") != hc_npz_sha[k]:
            refuse(f"{core}: the pair record was anchored on another {k} npz")
    n = c["n_draws"] + 1
    done = []
    for tgt in TARGETS:
        z = hc_z["hc" if tgt == "fpr1" else "hc5"]
        for lvl in LEVELS:
            for q in HC.STATS:
                a, b = z_op.get(f"{tgt}|{lvl}|{q}"), z.get(f"L{L}|{PSET}|{lvl}|{q}")
                if a is None or b is None or len(a) != n or not same_bits(a, b):
                    refuse(f"{core} {tgt} {lvl}: {q} is not aligned with the health-check npz draw by draw")
            for kk in LEVEL_KEYS:
                if len(z_op.get(f"{tgt}|{lvl}|{kk}", ())) != n:
                    refuse(f"{core} {tgt} {lvl}: {kk} does not hold the frozen split and {n - 1} draws")
            done.append(f"{tgt}|{lvl}")
    for kk in RF_KEYS:
        if len(z_op.get(f"rfresh|{kk}", ())) != PS.N_REPS + 1:
            refuse(f"{core} rfresh: {kk} does not hold the frozen split and {PS.N_REPS} option-B draws")
    done.append("rfresh")
    return {"checked": done, "kind": "configs (selection, C, L*, seeds, n_draws, draw seeds, cluster manifest, caches) "
                                     "and the six health-check arrays per target and level, bit for bit"}


def tpr_beside(core: str, L: int, t1: dict[str, Any], depth_doc: dict[str, Any], depth_rel: str, depth_sha: str) -> dict[str, Any]:
    """D-282 beside FROZEN's extra missed harmful: Table I's delta-TPR combined interval at L* x -1000. Refuses unless
    the TPR record names this depth record and its frozen value, mean and band equal the depth record's operating point."""
    row = t1["rows"].get(f"depth_{core}")
    if row is None or row["headline_layer"] != L or t1["anchor"]["depth_records_sha256"].get(depth_rel) != depth_sha:
        refuse(f"{core}: the D-282 TPR record has no headline row at L{L} on {depth_rel}")
    blk = row["probe_sets"][PSET]["layers"][str(L)]
    op = cell_of(depth_doc, L)["operating_point"]
    for q in ("tpr_pre", "tpr_post", "delta_tpr"):
        if (blk[q]["frozen"], blk[q]["mean_D156"], blk[q]["band95_D156"]) != (op[q]["frozen"], op[q]["mean"], op[q]["band95"]):
            refuse(f"{core}: the D-282 record's {q} at L{L} is not the depth record's")
    lo, hi = blk["delta_tpr"]["combined"]["ci95"]
    return {"extra_missed_harmful_FROZEN_combined_ci95": [-N_HARMFUL * hi, -N_HARMFUL * lo],
            "from": "the D-282 combined 95% interval of delta TPR (post - pre) at L*, x -N_HARMFUL (FROZEN's extra missed "
                    "harmful is N_HARMFUL x (TPR_pre - TPR_post)); beside the D-156 band, item level only (D-282 (2))",
            "delta_tpr_combined": blk["delta_tpr"]["combined"]}


def run_summary(args: argparse.Namespace, root: Path = ROOT) -> int:
    t_start = time.time()
    results = root / "results"
    final_path = (results / "opcase_d384.json").resolve()
    out_path = Path(args.out).resolve() if args.out else final_path
    final = out_path == final_path
    if out_path.exists():
        refuse(f"{out_path} exists; results are never overwritten (D-152 P5)")
    if not final and out_path.is_relative_to(results.resolve()):
        refuse("a test run (--out) must not write under results/; only the run's own final name may")
    threads = blas_threads()
    if final:
        problems = final_problems(threads, root, tuple(SUMMARY_MANIFESTS.values()))
        if problems:
            refuse(f"the final name: {problems}")
    code0 = P.code_block(CODE_FILES)
    reused_now = {f: sha256_file(ROOT / f) for f in REUSED_SHA256}
    man_paths = {**SUMMARY_MANIFESTS, "run": RUN_MANIFEST}
    man = {k: manifest(root, m) for k, m in man_paths.items()}
    if final and any(man[k][1] != PINNED_MANIFEST_SHA256[m] for k, m in SUMMARY_MANIFESTS.items()):
        refuse("a manifest read is not its pinned bytes (PINNED_MANIFEST_SHA256)")
    if sorted(man["run"][0]) != sorted([f"results/opcase_d384_{c}{s}" for c in PAIRS for s in (".json", "_draws.npz")]):
        refuse(f"{RUN_MANIFEST} does not list exactly the two pair records and their npz")
    seen: dict[str, str] = {}                               # every results file this run read, root-relative

    def rb(rel_path: str, listed: dict[str, str]) -> tuple[dict[str, Any], str]:
        doc, h = read_bound(root, rel_path, listed)
        seen[rel_path] = h
        return doc, h

    t1, t1_sha = rb(TABLE_I, man["reissue"][0])
    rule = pair_rule(t1)
    rule["headline_layer_equals_pair_record"] = {}
    t1tpr = {tgt: rb(T1_TPR[tgt], man[T1_MANIFEST[tgt]][0]) for tgt in TARGETS}
    cells: dict[str, Any] = {}
    aligned: dict[str, Any] = {}
    beside_d282: dict[str, Any] = {}
    pair_cfg: dict[str, Any] = {}
    pair_runs: dict[str, Any] = {}
    records: dict[str, Any] = {"table_I": {"file": TABLE_I, "sha256": t1_sha},
                               **{f"table1_tpr_{tgt}": {"file": T1_TPR[tgt], "sha256": t1tpr[tgt][1]} for tgt in TARGETS}}
    npz_read: dict[str, str] = {}
    for core in PAIRS:
        rec_rel = f"results/opcase_d384_{core}.json"
        rec, rec_sha = rb(rec_rel, man["run"][0])
        if rec.get("run_kind") != "opcase_d384_pair" or rec.get("registered_parameters") is not True \
                or rec["config"].get("pair") != core:
            refuse(f"{rec_rel} is not the registered pair record of {core}")
        if RP.canonical(rec["code"]["sha256"]) != RP.canonical(code0["sha256"]) \
                or RP.canonical(rec.get("reused_files_sha256")) != RP.canonical(reused_now):
            refuse(f"{rec_rel} was made by another revision of the code (code block or reused-file sha256)")
        missing = [k for k in PAIR_CONFIG_KEYS if k not in rec["config"]] + \
                  [k for k in UNANCHORED if k not in rec.get("anchor", {}).get("unanchored", {})]
        if missing or rec["config"]["rfresh"].get("n_reps") != PS.N_REPS:
            refuse(f"{rec_rel} lacks {missing} or another R_fresh draw count")
        pair_cfg[core] = {k: rec["config"][k] for k in PAIR_CONFIG_KEYS}
        pair_runs[core] = {"commit": rec["code"].get("commit"), "commit_source": rec["code"].get("commit_source"),
                           "anchor": rec["anchor"]}
        z_op, z_sha = npz_bound(root, rec["draws_npz"]["file"], man["run"][0], rec["draws_npz"]["sha256"])
        npz_read[rec["draws_npz"]["file"]] = z_sha
        rp = record_paths(core)
        docs, shas, hc_z, hc_sha = {}, {}, {}, {}
        for k in ("depth", "depth5", "hc", "hc5"):
            docs[k], shas[k] = rb(rp[k], man[k][0])
        for k in ("hc", "hc5"):
            hc_z[k], hc_sha[k] = npz_bound(root, docs[k]["draws_npz"]["file"], man[k][0], docs[k]["draws_npz"]["sha256"])
            npz_read[docs[k]["draws_npz"]["file"]] = hc_sha[k]
        aligned[core] = alignment(core, rec, z_op, docs, shas, hc_z, hc_sha)
        L = rec["config"]["selected_layer"]
        row = t1["table_1"].get(f"depth_{core}", {})
        if row.get("selected", {}).get("item_level_D156", {}).get("headline_selected_layer", {}).get("layer") != L:
            refuse(f"{core}: Table I's headline layer is not the pair record's L{L}")
        rule["headline_layer_equals_pair_record"][core] = L
        cells[core] = {"L_star": L, "selected_C": rec["config"]["selected_C"], "role": PAIR_ROLE[core]}
        for tgt in TARGETS:
            z = hc_z["hc" if tgt == "fpr1" else "hc5"]
            cells[core][tgt] = {}
            for lvl in LEVELS:
                a = {kk: z_op[f"{tgt}|{lvl}|{kk}"] for kk in LEVEL_KEYS}
                flags = {t: triggered(z[f"L{L}|{PSET}|{lvl}|{stat}"]) for t, stat in TRIGGERS.items()}
                cell = cost_cell(a, flags)
                if tgt == "fpr5":                   # the parts no record anchors, visibly (the pair record's words)
                    un = rec["anchor"]["unanchored"]
                    cell["RECALIBRATED_R_same"]["unanchored"] = un["R_same at 5%"]
                    for t in TRIGGERS:
                        cell[f"CHECK_TRIGGERED_{t}"]["unanchored"] = un["CHECK_TRIGGERED at 5%"]
                cells[core][tgt][lvl] = cell
        cells[core]["rfresh_beside"] = rfresh_cell({kk: z_op[f"rfresh|{kk}"] for kk in RF_KEYS})
        cells[core]["rfresh_beside"]["unanchored"] = {k: rec["anchor"]["unanchored"][k]
                                                      for k in ("R_fresh FPR_pre on E", "R_fresh TPR_pre")}
        beside_d282[core] = {tgt: tpr_beside(core, L, t1tpr[tgt][0], docs["depth" if tgt == "fpr1" else "depth5"],
                                             rp["depth" if tgt == "fpr1" else "depth5"],
                                             shas["depth" if tgt == "fpr1" else "depth5"]) for tgt in TARGETS}
        flag = lambda x: x.get("flag", "no flag key") if isinstance(x, dict) else "no input_verification block"  # noqa: E731
        cells[core]["d170_flags"] = {
            "rule": "D-170/D-172; D-331: D-168 is final, and the input registry clears the flag of the judged pin-only "
                    "caches ('pair_record_now' is that current verification of the same caches)",
            "pair_record_now": flag(rec.get("input_verification")),
            **{f"{k}_record": flag(v) for k, v in rec.get("stored_input_verification", {}).items()},
            "table_I_row": flag(row.get("input_verification")),
            **{f"table1_tpr_{tgt}_row": flag(t1tpr[tgt][0]["rows"].get(f"depth_{core}", {}).get("input_verification"))
               for tgt in TARGETS}}
        records[core] = {"pair_record": {"file": rec_rel, "sha256": rec_sha},
                         "pair_npz": {"file": rec["draws_npz"]["file"], "sha256": z_sha},
                         **{k: {"file": rp[k], "sha256": shas[k]} for k in docs},
                         **{f"{k}_npz": {"file": docs[k]["draws_npz"]["file"], "sha256": hc_sha[k]} for k in hc_z}}
    code1 = P.code_block(CODE_FILES)
    changed = ["code"] if code1["sha256"] != code0["sha256"] else []
    gone_or = lambda r_, h: not (root / r_).is_file() or sha256_file(root / r_) != h  # noqa: E731
    changed += [r_ for r_, h in seen.items() if gone_or(r_, h)]
    changed += [f for f, h in npz_read.items() if gone_or(f, h)]
    changed += [m for k, m in man_paths.items() if gone_or(m, man[k][1])]
    if changed:
        refuse(f"changed during the run: {changed}")
    config = {"pairs": {c: PAIR_ROLE[c] for c in PAIRS}, "traffic": {"n_requests": N_REQUESTS, "n_harmful": N_HARMFUL,
                                                                   "n_benign": N_BENIGN, "harmful_share": "1%"},
              "costs": {"extra_false_refusals": "N_BENIGN x (FPR_policy - FPR_pre), signed",
                        "extra_missed_harmful": "N_HARMFUL x (TPR_pre - TPR_policy), signed"},
              "policies": {"FROZEN": "T_d kept", "RECALIBRATED_R_same": f"repair_d156 '{R_SAME_ARM}' (Table II)",
                           "CHECK_TRIGGERED_gaussian": f"R_same where the 4c Gaussian health ratio crosses {list(TRIGGER_CUT)}, "
                                                       "FROZEN otherwise (primary trigger)",
                           "CHECK_TRIGGERED_direct": f"the same with the direct ratio (beside)"},
              "trigger": {"class": TRIGGER_CLASS, "cut": list(TRIGGER_CUT), "function": "healthcheck_d156.crosses"},
              "targets": TARGETS, "levels": list(LEVELS), "probe_set": PSET,
              "primary_cell": f"{PRIMARY[0]}|{PRIMARY[1]}",
              "beside": ["fpr5", LEVELS[1], "rfresh_beside (1%, item level, option-B draws)", "D-282 combined interval"],
              "estimator": "per draw, then paired.band_summary: D-156 mean and 2.5-97.5% band over the draws, the frozen "
                           "split as one draw with its percentile; savings are FROZEN minus the policy within each draw",
              "per_pair": pair_cfg}
    payload: dict[str, Any] = {
        **provenance_block("opcase_d384"), "preregistration": PREREG, "code": code0,
        "reused_files_sha256": {f: sha256_file(ROOT / f) for f in REUSED_SHA256},
        "registered_parameters": True, "blas_threads": threads, "elapsed_s": round(time.time() - t_start, 2),
        "descriptive": "D-384 (5): no prediction and no verdict; D-177's accuracy line does not apply",
        "caveat": CAVEAT, "config": config, "config_hash": canonical_hash(config), "pair_rule": rule,
        "draw_alignment": aligned, "pair_records": pair_runs, "unanchored": UNANCHORED,
        "records_of_record": records,
        "manifests": {m: man[k][1] for k, m in man_paths.items()},
        "input_files_sha256": {**dict(_RESULT_READS), **npz_read, **{m: man[k][1] for k, m in man_paths.items()}},
        "d282_beside": beside_d282, "cells": cells}
    digest = write_results(out_path, payload, exclusive=True)
    os.chmod(out_path, 0o444)
    print(f"Written: {rel(out_path, root)}\nsha256:  {digest}")
    for core in PAIRS:
        c = cells[core][PRIMARY[0]][PRIMARY[1]]
        line = "; ".join(f"{p} {c[p]['extra_false_refusals']['mean']:+.0f} refusals, "
                         f"{c[p]['extra_missed_harmful']['mean']:+.0f} missed" for p in POLICIES)
        print(f"{core} (1%, item level, D-156 means per 100,000): {line}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="D-384: the operational case study")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pair", choices=PAIRS, help="the recompute job of one pair (per-draw arrays, anchored)")
    mode.add_argument("--summary", action="store_true", help="the costs from the two pair records")
    ap.add_argument("--out", default=None, help="TESTING: write here (outside results/) instead of the final name")
    args = ap.parse_args(argv)
    return run_summary(args) if args.summary else run_pair(args)


if __name__ == "__main__":
    raise SystemExit(main())
