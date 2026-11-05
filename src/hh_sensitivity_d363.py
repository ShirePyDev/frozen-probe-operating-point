"""D-363: the HH contamination sensitivity of Table III and of Table IV's LoRA rows (D-349 option (a), D-357 (B)(4)).

Pre-registered in D-363 before any of this code or its numbers existed. For all nine Phase 1 LoRA runs (alpaca, HH, PKU x
steps 25, 100, 400) and their shared pre side (Mistral-7B-v0.1 on the A100), one fixed removal set R is removed from
every split (probe_train, threshold_calib, evaluation negatives and positives; pre and post) and every Table III cell,
4a's between-arm comparisons and Table IV's LoRA rows are recomputed, each beside its value of record. No reviewed module
is edited (the records bind their sha256): their functions are called, and where a quantity is computed inside a
reviewed main() the lines are repeated here and pinned by an anchor that reproduces the record exactly.

R (D-363 (1)): the distinct item_ids of EVERY entry of results/lora_contamination_p1.json's counts.hh.matches (any role,
exact or not, checked prompt or not), restricted to the three Aegis frozen files 4a reads (AEGIS_FILES). Every entry must
lie inside the HH run's consumed prefix at step 400 (position < 6400, the record's dose rule) and the per-file sizes must
be 338, 158 and 330 (EXPECTED_R), or the run refuses. One set for every run, dose and arm. Each item of R must be in its
split's cache exactly once, and in no other split's.

WHERE EACH CELL COMES FROM (record of record: the reviewed code that made it; what this module calls):
  Table III, per run (selected C, item level; the D-164 cluster level beside):
    frozen, restand.   repair_d156_<run>.json pair_summary_mean_abs_log_over_layers: repair_d156.partition and
                       repair_d156.summarise per layer and level (arms 'frozen' and 'restandardise'), the per-draw mean
                       over layers, paired.band_summary. The same functions; the restandardise arm is built as
                       repair_d156.main builds it (the post model's probe_train fit rows of each seed).
    worst FPR/target   depth_d156_<run>.json pair_summary.<C>.<level>.sweep_maximum.ratio_to_target: per layer
                       paired.repartition_estimate, then depth_d156.main's argmax of the per-layer mean (TIE_RULE) and
                       its no-update null. The same calls; the main() lines repeated (pick, layer_row).
    delta TPR at L*    table3_tpr_d282.json rows.depth_<run>.probe_sets.selected.layers.<L*>.delta_tpr:
                       table3_tpr_d282.anchored_replicates (paired.repartition_estimate, paired.combined_tpr_replicates)
                       with paired.counts_matrix(n positives, n_boot, depth_d156.ITEM_BOOT_SEED['eval_positives']). The
                       same function (anchored on the depth record's cell when R is empty, on this run's own cell else).
    AUROC drop         depth_d156_<run>.json pair_summary.<C>.auroc_drop_frozen.mean_over_layers: metrics.auroc and
                       depth_d156.weighted_auroc (the D-152 P5 item bootstrap, depth_d156.ITEM_BOOT_SEED) per layer,
                       then the mean over layers. The same functions.
    ||d||, cos         intent_selected_d165.json cells at L* (geometry.neg; C=1 in c1_sensitivity): intent_selected.
                       main's measure(), a closure, whose 'neg' geometry lines are repeated here (geometry_neg) with
                       intent_selected.mean_rng and paired.counts_matrix at the record's boot seed.
  Between arms         damage_d156_r2.json between_arms_4a (and _cluster_level_D164): damage_d156.main's within-draw gap
                       (log paired ratio, delta TPR; safety minus general at matched dose; paired.band_summary), its lines
                       repeated on the per-draw arrays at L*. Beside it the delta-TPR gap's combined interval (D-356
                       (2)(a); table3_tpr_d282.json between_arms_4a_delta_tpr_gap).
  Table IV LoRA rows   healthcheck_d156_summary_lora.json rows: healthcheck_d156.thresholds, health and crosses per
                       partition, layer, probe set and level, confusion counts summed over cells (healthcheck_d156.main's
                       lines repeated), pooled by healthcheck_summary_d156.scoring (pooled_counts, pair_cluster_index:
                       D-255's estimator unchanged, B = 5000, seed 520260925) over all nine, the six safety and the three
                       general runs. The same functions.
PROBES, as the records fit them: per seed 0-4, paired.d117_split on the REDUCED probe_train (the D-117 80% rule applied to
the reduced set) and paired.fit_probe with the D-117 selected C of record (results/selection_sweep_lora_a100_d165.json,
checked against the C every depth-record cell carries) in permutation order, and C=1 in file order; the depth records'
layers. Layers and C are NOT re-selected. ESTIMATOR: D-156 over n_draws re-partitions of the reduced pooled negatives at
the reduced frozen sizes (paired.repartition_draws, the axis seed), the frozen split as one more draw, the positives fixed
(evaluation positives minus R); D-164 beside (the cluster manifest's clusters restricted to the reduced pool).

ANCHOR (D-363 (4)): every per-run job first recomputes the run at L* with R EMPTY and refuses unless it reproduces, as
canonical JSON text or bit for bit, the depth record's cells (operating point, frozen curve and its hashes, censoring,
AUROC blocks, probe fit diagnostics; both levels), the damage record's 4a cells, the repair record's frozen and
restandardise blocks, the health-check record's cells and per-draw arrays, the intent record's geometry and the TPR
supplement's blocks. --anchor-only does that alone; with every layer (the lane's first job) it also reproduces the
pair-level values (sweep maximum, AUROC drop over layers, the repair means over layers and their per-draw hashes, every
health-check array and confusion count). The summary anchors its between-arm gaps on the damage record and its TPR gaps
on the supplement, both from the per-run R-empty arrays at L*; its Table IV scoring by scoring the nine health-check
records' own npz arrays and comparing with healthcheck_d156_summary_lora.json; and it refuses records made by another
revision of the code (code block or reused-file sha256).

MODES and outputs (written once; only registered parameters, every layer, the 6-thread BLAS pin, an unset
GUARDRAIL_CACHE_DIRS, the pinned records (PINNED) and the reused files at their pinned bytes (REUSED_SHA256) may write a
final name; a test writes outside results/):
  --run <run>                  results/hh_sensitivity_d363_<run>.json and _draws.npz (D-163 (6))
  --run <run> --anchor-only    R empty; results/hh_sensitivity_d363_anchor_<run>.json (every layer)
  --summary                    results/hh_sensitivity_d363.json from the nine per-run records and the anchor record named
                               by logs/hh_sensitivity_d363.sha256 (the lane's manifest job)
READING (D-363 (3), fixed before the numbers): at a dose the HH rows SURVIVE if their sensitivity frozen mean |log| exceeds
alpaca's with the D-156 band of the HH-minus-alpaca per-draw difference wholly above 0; the change of every cell
(sensitivity minus record) is reported; PKU beside (a removal-only effect on an uncontaminated arm). C3 rests on PKU
whatever this shows; the HH rows enter the paper only as the author decides; the under-check marks stay until then.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import damage_d156 as DM  # noqa: E402
import depth_d156 as D  # noqa: E402
import healthcheck_d156 as HC  # noqa: E402
import healthcheck_summary_d156 as HS  # noqa: E402
import input_registry as REG  # noqa: E402
import intent_selected as IS  # noqa: E402
import metrics as M  # noqa: E402
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
import repair_d156 as RP  # noqa: E402
import table3_tpr_d282 as T3  # noqa: E402
from provenance import (  # noqa: E402
    _RESULT_READS, _repo_rel, canonical_hash, commit_staged, ensure_utf8_stdout, input_identity_problems, provenance_block,
    read_result, resolve_cache, sha256_bytes, sha256_file, stage_bytes, write_results,
)

ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = Path(__file__).resolve().parent.parent      # where the source files are hashed (ROOT may be a test tree)
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
OWN = "src/hh_sensitivity_d363.py"
CODE_FILES = list(dict.fromkeys([OWN, *D.CODE_FILES, *RP.CODE_FILES, *DM.CODE_FILES, *HC.CODE_FILES, *HS.CODE_FILES,
                                 *IS.CODE_FILES, *T3.CODE_FILES]))
# the reviewed files whose functions are reused, at the bytes this module was written and anchored against (master
# b6b8db2); a final name refuses other bytes, so an edit of a reused file cannot pass unnoticed
REUSED_SHA256 = {
    "src/depth_d156.py": "a1f7b50ce70395f290b681f336e5a5e95e1428a2ca7c5915818644ad70f26734",
    "src/repair_d156.py": "01211e1ec9b69adf2bcc7b738e80a06f07c229c283f105ced11f05083cbe358e",
    "src/damage_d156.py": "88bca913717188bfc897343d8eb81831596d85c5da6bcf38798f59dd04d76c49",
    "src/healthcheck_d156.py": "0ec2687ecdd887a28d4b8fb7a4db4b9081b69ea3fcf2b175c345279b4ed74de7",
    "src/healthcheck_summary_d156.py": "b149b66181fe1c2d953d255355e44e2d15651bff9ccf8a456819092db526ccf5",
    "src/intent_selected.py": "fa206d9312c7c2bc059d378e50eb763e7e1bceeb693105275243aa0c395f1e27",
    "src/table3_tpr_d282.py": "22cab06f8ecef3d778429a3789d75473ea30f21b39ef58f573170ffcb713dbdc",
    "src/table1_tpr_d282.py": "820778858ad01780d77bcfd7899ff3f8b816a35f3a8df0b8d05087bdee4825bb",
}
LEVELS, PROBE_SETS = D.LEVELS, D.PROBE_SETS
REGISTERED = {"n_draws": P.N_DRAWS, "n_boot": 5000, "seeds": [0, 1, 2, 3, 4], "target_fpr": P.TARGET_FPR}
PARTS = ("probe_train", "threshold_calib", "eval")
AEGIS_FILES = {"probe_train": "data/frozen/safety__probe_train.jsonl",
               "threshold_calib": "data/frozen/safety__threshold_calib.jsonl",
               "eval": "data/frozen/safety__eval.jsonl"}
EXPECTED_R = {"probe_train": 338, "threshold_calib": 158, "eval": 330}           # D-363 (1)
EXPECTED_R_EXACT = {"probe_train": 286, "threshold_calib": 128, "eval": 286}     # its exact-headline subset (reported)
DOSE_LIMIT = 6400                       # step 400's consumed prefix: 16 x 400 examples (the record's dose rule)
GENERAL, HH, PKU = "general (alpaca)", "safety (hh-harmless)", "safety (pku-saferlhf)"
ARM_OF = {r: arm for arm, runs in POP.LORA_ARMS.items() for r in runs}
RUNS = list(ARM_OF)                     # POP.LORA_ARMS order: alpaca, HH, PKU; doses 25, 100, 400
DOSES = DM.DOSES
ANCHOR_RUN = "lorasafe_mistral_s400"    # the lane's every-layer anchor: the HH run at the largest dose
REPAIR_ARMS = ("frozen", "restandardise")                  # Table III's two repair columns
NULL_RULE = ("the maximum over layers of the per-layer D-156 mean of FPR_pre / target (D-156 floor, D-180)")

# records of record and the manifests that name them; a final name needs each at its pinned bytes
CONTAMINATION = RESULTS / "lora_contamination_p1.json"
SELECTION = D.SELECTION_OF_RECORD_LORA
CLUSTERS = D.CLUSTERS
DAMAGE = D.DAMAGE_OF_RECORD
INTENT = RESULTS / "intent_selected_d165.json"
T3TPR = RESULTS / "table3_tpr_d282.json"
HC_SUMMARY = RESULTS / "healthcheck_d156_summary_lora.json"
DEPTH_MANIFEST = T3.LORA_DEPTH_MANIFEST
REPAIR_MANIFEST = T3.REPAIR_MANIFEST
HC_MANIFEST = LOGS / "healthcheck_d156_lora.sha256"
T3TPR_MANIFEST = LOGS / "table3_tpr_d282_20261003.sha256"
RUN_MANIFEST = LOGS / "hh_sensitivity_d363.sha256"         # written by the lane's manifest job; the summary reads it
PINNED = {"CONTAMINATION": "c39a2d78125d9c4b5215090a2d8e05e25159659d8482a201d49bb8b1ff5ed6f1",
          "SELECTION": "a4e46a24cbc6b7ef3f1314afc104bed50e10740634f417fcd31bbbb26558fea9",
          "CLUSTERS": "0b30195cc1005ae590093963b35ed56b55f0f8cfd36eeecc9458157ce53de7c2",
          "DAMAGE": D.DAMAGE_OF_RECORD_SHA256,
          "INTENT": "8ca4b69e61ceb8cf4f1c036e0cc1fc76bda50964c413d89534b59233595fc3c9",
          "HC_SUMMARY": "f697569d57c82d07678d8c8e79c0a953f9b6b7ffff8e4d6dffb802f7b16fe500",
          "DEPTH_MANIFEST": "8e45c1c5eef31b0eec7f3da030579e4a8600d7f4e7f466f02112b15e99a3d269",
          "REPAIR_MANIFEST": "5f41056ffe26ebb8dfcf6141d7d40be25142d680f3cdbb184ba0ce5ea1d2b757",
          "HC_MANIFEST": "f91f72ecbc60f730a8c9891f6b3303108439b4c825fadc1a9becc3c9c71c5f0d",
          "T3TPR_MANIFEST": "00b36aef6503c1c2d011c6338c74ab9ca3fd8da9b67d035ff5cd1910899aa53e"}
UNDER_CHECK = ("D-363 (3), D-357 (B)(4), (B)(11): the HH rows enter the paper only as the author decides from this "
               "record; Table III's section mark and Table IV's LoRA-row marks stay until then; C3 rests on "
               "PKU-SafeRLHF, the clean safety arm, whatever this sensitivity shows")
READING_RULE = ("D-363 (3): at a dose the HH rows SURVIVE if their sensitivity frozen mean |log(FPR/target)| over the "
                "layer sweep (selected C, item level, D-156 mean) exceeds alpaca's at that dose with the D-156 band of "
                "the per-draw HH-minus-alpaca difference wholly above 0, as the rows of record are read; PKU beside "
                "(a removal-only effect on an uncontaminated arm); the cluster level beside")
NPZ_KEYS = ("L<layer>|<probe set>|<level>|<statistic> and counts|<probe set>|<level>|<detector>|<class>|tp/fn/fp/tn "
            "(healthcheck_d156's keys, R removed); lstar|<probe set>|<level>|paired_ratio and |delta_tpr at L*; "
            "lstar|<probe set>|delta_tpr_combined (the D-282 replicates at L*); repair|<probe set>|<level>|<arm> (the "
            "per-draw mean over layers of |log(FPR/target)|); anchor|... the same at L* with R empty; index 0 the frozen "
            "split, then the draws")


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def rel(p: Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(ROOT.resolve())).replace("\\", "/") if p.is_relative_to(ROOT.resolve()) else str(p)


def blas_threads() -> dict[str, str | None]:
    return {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}


def final_problems(registered: bool, threads: dict[str, str | None]) -> list[str]:
    """Why a final name may not be written (empty when it may)."""
    out = []
    if not registered:
        out.append("non-registered parameters (n_draws 1000, n_boot 5000, every layer)")
    if any(t != "6" for t in threads.values()):
        out.append(f"no 6-thread BLAS pin (D-152 P6): {threads}")
    if os.environ.get("GUARDRAIL_CACHE_DIRS"):
        out.append("GUARDRAIL_CACHE_DIRS is set; the caches must come from the input registry's directory")
    for name, want in PINNED.items():
        p = globals()[name]
        have = sha256_file(p) if Path(p).is_file() else None
        if have != want:
            out.append(f"{rel(p)} is not the pinned bytes ({(have or 'missing')[:12]} vs {want[:12]})")
    for f, want in REUSED_SHA256.items():
        have = sha256_file(CODE_ROOT / f) if (CODE_ROOT / f).is_file() else None
        if have != want:
            out.append(f"{f} is not the reviewed bytes this module reuses ({(have or 'missing')[:12]} vs {want[:12]})")
    return out


def read_bound(path: Path, listed: dict[str, str] | None = None) -> tuple[dict[str, Any], str]:
    """A results JSON read through read_result (D-297 (1)); refuses unless `listed` (a manifest's mapping) names it with
    these bytes, when given."""
    doc = read_result(path)
    r = _repo_rel(Path(path))
    h = _RESULT_READS[r]
    if listed is not None and listed.get(r) != h:
        refuse(f"{r} is not the file its manifest names")
    return doc, h


def npz_bound(rel_path: str, listed: dict[str, str], want: str) -> tuple[dict[str, np.ndarray], str]:
    """A per-draw npz read once from its bytes, which must be the manifest's and its record's (want)."""
    b = (ROOT / rel_path).read_bytes()
    h = sha256_bytes(b)
    if listed.get(rel_path) != h or want != h:
        refuse(f"{rel_path} is not the npz its manifest and its record name")
    with np.load(io.BytesIO(b), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}, h


# ------------------------------------------------------------------------------------------------ D-363 (1): the set R
def removal_set(doc: dict[str, Any]) -> tuple[dict[str, frozenset[str]], dict[str, Any]]:
    """R per split (D-363 (1)) from the contamination record's counts.hh.matches; refuses a listing that is not
    complete, an entry outside step 400's consumed prefix, other per-file sizes, or an item in two files."""
    hh = doc["counts"]["hh"]
    matches = hh["matches"]
    if len(matches) != hh["n_matches_listed"]:
        refuse(f"counts.hh lists {len(matches)} matches, not its n_matches_listed {hh['n_matches_listed']}")
    late = [m for m in matches if not 0 <= int(m["position"]) < DOSE_LIMIT]
    if late:
        refuse(f"{len(late)} HH matches lie outside step 400's consumed prefix (position >= {DOSE_LIMIT})")
    R: dict[str, set[str]] = {s: set() for s in AEGIS_FILES}
    exact: dict[str, set[str]] = {s: set() for s in AEGIS_FILES}
    n_entries = 0
    for m in matches:
        hit = False
        for split, f in AEGIS_FILES.items():
            if f in m["files"]:
                hit = True
                R[split].add(str(m["item_id"]))
                if m["exact"] is True and m["role"] == "user" and m["is_checked_prompt"] is False:
                    exact[split].add(str(m["item_id"]))
        n_entries += hit
    sizes = {s: len(v) for s, v in R.items()}
    if sizes != EXPECTED_R:
        refuse(f"R's per-file sizes are {sizes}, not D-363 (1)'s {EXPECTED_R}")
    splits = list(R)
    twice = [(a, b) for i, a in enumerate(splits) for b in splits[i + 1:] if R[a] & R[b]]
    if twice:
        refuse(f"an item of R is in two frozen files: {twice}")
    info = {"rule": "D-363 (1): the distinct item_ids of every counts.hh.matches entry (any role; exact, normalised or "
                    "contained; checked prompt or not) in the three Aegis frozen files 4a reads; every entry inside step "
                    f"400's consumed prefix (position < {DOSE_LIMIT}); one set for every run, dose and arm",
            "files": dict(AEGIS_FILES), "n_matches_listed": len(matches), "n_entries_in_these_files": n_entries,
            "n_items": sizes, "n_items_total": sum(sizes.values()),
            "n_items_exact_headline_subset": {s: len(v) for s, v in exact.items()},
            "exact_headline_subset_rule": "exact, role user, not the checked prompt (reported, not used)",
            "item_ids_sha256": {s: sha256_bytes("\n".join(sorted(v)).encode("utf-8")) for s, v in R.items()}}
    if info["n_items_exact_headline_subset"] != EXPECTED_R_EXACT:
        refuse(f"R's exact-headline subset is {info['n_items_exact_headline_subset']}, not D-363 (1)'s {EXPECTED_R_EXACT}")
    return {s: frozenset(v) for s, v in R.items()}, info


# ------------------------------------------------------------------------------------------------ the computation
def pick(m: np.ndarray, how: str, layers: list[int]) -> tuple[int, int]:
    """depth_d156.main's pick: (layer, number tied) for the argmax/argmin of per-layer means (TIE_RULE)."""
    target = m.max() if how == "max" else m.min()
    tied = np.isclose(m, target, rtol=1e-12, atol=0.0)
    return int(layers[int(np.argmax(tied))]), int(tied.sum())


def geometry_neg(probes: list[P.Probe], A_pre: np.ndarray, A_post: np.ndarray, cnt: np.ndarray) -> dict[str, Any]:
    """intent_selected.main's measure(), its 'neg' geometry block (the closure's lines repeated): d = mean(z_post) -
    mean(z_pre) on the evaluation negatives, ||d||, cos(w, d), |cos|, w.d with their item-bootstrap intervals."""
    W = np.column_stack([q.w for q in probes])
    SD = np.column_stack([q.sd for q in probes])
    wn = np.linalg.norm(W, axis=0)
    n = A_pre.shape[0]
    Dm = (A_post.mean(0) - A_pre.mean(0))[:, None] / SD
    dn, sh = np.linalg.norm(Dm, axis=0), (W * Dm).sum(axis=0)
    cs = sh / (wn * dn)
    DXr = cnt @ (A_post - A_pre).astype(np.float32) / n
    dn_r = np.stack([np.linalg.norm(DXr / SD[:, s], axis=1) for s in range(SD.shape[1])], 1)
    sh_r = np.stack([(DXr / SD[:, s]) @ W[:, s] for s in range(SD.shape[1])], 1)
    cs_r = sh_r / (wn[None, :] * dn_r)
    flip = (np.sign(sh_r) != np.sign(sh)[None, :]).mean(axis=0)
    return {"d_norm": {**IS.mean_rng(dn), **P.ci(dn_r.mean(axis=1))},
            "cos": {**IS.mean_rng(cs), **P.ci(cs_r.mean(axis=1))},
            "abs_cos": {**IS.mean_rng(np.abs(cs)), **P.ci(np.abs(cs_r).mean(axis=1))},
            "score_shift": {**IS.mean_rng(sh), **P.ci(sh_r.mean(axis=1))},
            "w_norm": IS.mean_rng(wn),
            "shift_sign_flip_fraction_per_seed": flip.tolist()}


def restandardise_arm(probes: list[P.Probe], rows_of: list[np.ndarray], X_post_tr: np.ndarray, cal_post_x: np.ndarray,
                      ev_post_x: np.ndarray, ncal: np.ndarray, neg: np.ndarray, pos: np.ndarray) -> dict[str, np.ndarray]:
    """repair_d156.main's 'restandardise' arm (lines repeated): the post features standardised with the post model's
    mean and sd on each seed's probe_train fit rows, the probe's w and b kept."""
    cols_pool, cols_pos = [], []
    for q, rows in zip(probes, rows_of):
        xb = X_post_tr[rows]
        mu_b = xb.mean(0)
        sd_b = np.where(xb.std(0) < 1e-8, 1.0, xb.std(0))
        f = lambda x, mu=mu_b, sd=sd_b, q=q: ((x - mu) / sd) @ q.w + q.b  # noqa: E731
        cols_pool.append(np.concatenate([f(cal_post_x[ncal]), f(ev_post_x[neg])]))
        cols_pos.append(f(ev_post_x[pos]))
    return {"pool": np.column_stack(cols_pool), "pos": np.column_stack(cols_pos)}


def compute(data: dict[str, dict[str, Any]], keep: dict[str, np.ndarray], layers: list[int], L_sel: int, C_sel: float,
            seeds: list[int], n_draws: int, n_boot: int, target: float, axis: str, cl_axis: dict[str, Any],
            verification: dict[str, Any], intent_seed: int, tpr_cells: dict[tuple[int, str], dict[str, Any]] | None,
            source_note: str, label: str) -> dict[str, Any]:
    """Every quantity this record needs, for the items `keep` holds (True = kept; all True = the records' population),
    at `layers`. tpr_cells: the cells table3_tpr_d282.anchored_replicates anchors on at L* (the depth record's when R
    is empty); None anchors on this computation's own cell."""
    pre = data["pre"]
    y_tr = pre["probe_train"]["labels"].astype(bool)
    tr_idx = np.flatnonzero(keep["probe_train"])          # rows of the full cache kept, ascending (file order)
    y_cal = pre["threshold_calib"]["labels"].astype(bool)
    y_ev = pre["eval"]["labels"].astype(bool)
    ncal, hcal = ~y_cal & keep["threshold_calib"], y_cal & keep["threshold_calib"]
    neg, pos = ~y_ev & keep["eval"], y_ev & keep["eval"]
    pooled_ids = [str(i) for i in pre["threshold_calib"]["item_ids"][ncal]] + [str(i) for i in pre["eval"]["item_ids"][neg]]
    n_cal, n_pool, n_neg, n_pos = int(ncal.sum()), len(pooled_ids), int(neg.sum()), int(pos.sum())
    in_pool = set(pooled_ids)
    clusters = [g for g in ([i for i in c if i in in_pool] for c in cl_axis["clusters"]) if g]
    draws = {LEVELS[0]: P.repartition_draws(n_cal, n_pool - n_cal, n_draws, P.AXIS_DRAW_SEED[axis]),
             LEVELS[1]: P.cluster_repartition_draws(P.cluster_units(pooled_ids, clusters), n_cal, n_draws,
                                                    P.AXIS_CLUSTER_DRAW_SEED[axis])}
    frozen_part = (np.arange(n_cal), np.arange(n_cal, n_pool))
    parts = {lvl: [frozen_part] + [(d[:n_cal], d[n_cal:]) for d in dr] for lvl, dr in draws.items()}
    CNT_N = P.counts_matrix(n_neg, n_boot, D.ITEM_BOOT_SEED["eval_negatives"])
    CNT_P = P.counts_matrix(n_pos, n_boot, D.ITEM_BOOT_SEED["eval_positives"])
    ONE_N, ONE_P = np.ones((1, n_neg)), np.ones((1, n_pos))
    n_seeds = len(seeds)
    population = {"n_probe_train": int(len(tr_idx)), "n_probe_train_positives": int(y_tr[tr_idx].sum()),
                  "n_fit_per_seed": int(len(tr_idx)) - int(P.HOLDOUT * len(tr_idx)),
                  "n_calibration_negatives": n_cal, "n_calibration_harmful_H": int(hcal.sum()),
                  "n_evaluation_negatives": n_neg, "n_evaluation_positives": n_pos, "n_pooled_negatives": n_pool,
                  "pooled_item_ids_sha256": sha256_bytes("\n".join(pooled_ids).encode("utf-8")),
                  "eval_positives_item_ids_sha256": sha256_bytes("\n".join(str(i) for i in pre["eval"]["item_ids"][pos]).encode("utf-8")),
                  # near-duplicate clusters left in the pool: groups of two or more (a member alone is a singleton unit)
                  "n_clusters_in_pool": sum(len(g) >= 2 for g in clusters),
                  "n_items_in_clusters": int(sum(len(g) for g in clusters if len(g) >= 2))}
    out: dict[str, Any] = {"population": population, "depth": {}, "repair": {}, "health": {}, "lstar": {},
                           "hc_arrays": {}, "confusion": {}}
    drop_reps: dict[str, dict[int, np.ndarray]] = {s: {} for s in PROBE_SETS}
    abslog: dict[tuple[str, str, str], list[np.ndarray]] = {}
    confusion: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    CNT_IN = None
    for li in layers:
        k = f"layer_{li}"
        X = {s: {p: data[s][p][k] for p in PARTS} for s in data}
        fits: dict[str, list[P.Probe]] = {s: [] for s in PROBE_SETS}
        rows_of: dict[str, list[np.ndarray]] = {s: [] for s in PROBE_SETS}
        for seed in seeds:
            _, fit = P.d117_split(len(tr_idx), seed)
            if len(fit) != population["n_fit_per_seed"]:
                refuse(f"{label}: the D-117 fit split holds {len(fit)} rows, not the {population['n_fit_per_seed']} recorded")
            for pset, rows, C, order in (("selected", tr_idx[fit], C_sel, "d117_permutation"),
                                         ("C=1", tr_idx[np.sort(fit)], 1.0, "file_order")):
                fits[pset].append(P.fit_probe(X["pre"]["probe_train"][rows], y_tr[rows], seed, C, order))
                rows_of[pset].append(rows)
        for pset in PROBE_SETS:
            probes = fits[pset]
            C = C_sel if pset == "selected" else 1.0
            S = lambda side, part, m: np.column_stack([q.score(X[side][part][m]) for q in probes])  # noqa: E731
            sc = {"cal_pre": S("pre", "threshold_calib", ncal), "ev_pre": S("pre", "eval", neg),
                  "cal_post": S("post", "threshold_calib", ncal), "ev_post": S("post", "eval", neg),
                  "pos_pre": S("pre", "eval", pos), "pos_post": S("post", "eval", pos)}
            H_pre, H_post = S("pre", "threshold_calib", hcal), S("post", "threshold_calib", hcal)
            pool_pre, pool_post = np.vstack([sc["cal_pre"], sc["ev_pre"]]), np.vstack([sc["cal_post"], sc["ev_post"]])
            reports = {lvl: P.repartition_estimate(sc["cal_pre"], sc["ev_pre"], sc["cal_post"], sc["ev_post"],
                                                   sc["pos_pre"], sc["pos_post"], draws[lvl], target) for lvl in LEVELS}
            floored = {lvl: D.floored_draws(pool_pre, pool_post, parts[lvl], target) for lvl in LEVELS}
            where = f"{label} L{li} {pset}"

            # --- depth_d156: AUROC on the evaluation split with the D-152 P5 item bootstrap (pre and frozen) ---
            au: dict[str, Any] = {"note": D.AUROC_NOTE}
            reps: dict[str, np.ndarray] = {}
            points: dict[str, np.ndarray] = {}
            for name, (pk, nk) in {"pre": ("pos_pre", "ev_pre"), "frozen": ("pos_post", "ev_post")}.items():
                pt = np.array([M.auroc(sc[pk][:, s], sc[nk][:, s]).value for s in range(n_seeds)])
                unit = np.array([D.weighted_auroc(sc[pk][:, s], sc[nk][:, s], ONE_P, ONE_N)[0] for s in range(n_seeds)])
                if not np.array_equal(pt, unit):
                    refuse(f"{where} {name}: the weighted AUROC is not metrics.auroc bit for bit")
                points[name] = pt
                reps[name] = np.mean([D.weighted_auroc(sc[pk][:, s], sc[nk][:, s], CNT_P, CNT_N) for s in range(n_seeds)],
                                     axis=0)
            points["drop_frozen"] = points["pre"] - points["frozen"]
            reps["drop_frozen"] = reps["pre"] - reps["frozen"]
            for name in ("pre", "frozen", "drop_frozen"):
                au[name] = {"mean_over_seeds": float(points[name].mean()), "per_seed": points[name].tolist(),
                            "range_over_seeds": [float(points[name].min()), float(points[name].max())],
                            **P.ci(reps[name])}
            drop_reps[pset][li] = np.concatenate([[points["drop_frozen"].mean()], reps["drop_frozen"]])

            # --- depth_d156: the operating point, the frozen curve and the censoring, both levels ---
            dcell: dict[str, Any] = {"layer": li, "probe_set": pset, "probe_C": C,
                                     "fit_row_order": "d117_permutation" if pset == "selected" else "file_order",
                                     "is_selected_layer": li == L_sel, "auroc": au}
            for lvl in LEVELS:
                report, arr = reports[lvl]
                rt = np.concatenate([[report["ratio_to_target"]["frozen"]], arr["ratio_to_target"]]).astype("<f8")
                fc = P.abs_log(rt).astype("<f8")
                block = {"operating_point": report,
                         "frozen_curve_abs_log_ratio_to_target": {
                             **P.band_summary(fc[1:], fc[0]), "note": D.FROZEN_CURVE_NOTE,
                             "per_draw_sha256": sha256_bytes(fc.tobytes()),
                             "per_draw_ratio_to_target_sha256": sha256_bytes(rt.tobytes())},
                         "realised_censoring": {"share_of_draws_floored": float(floored[lvl][1:].mean()),
                                                "frozen_split_floored": bool(floored[lvl][0])}}
                if lvl == LEVELS[0]:
                    dcell.update(block)
                else:
                    dcell[lvl] = {"note": D.CLUSTER_NOTE, **block}
            dcell["probe_fit_diagnostics"] = [q.diag for q in probes]
            out["depth"][(li, pset)] = dcell

            # --- repair_d156 (family 'repair'): the frozen and restandardise arms per partition, both levels ---
            arms = {"frozen": {"pool": pool_post, "pos": sc["pos_post"], "H": H_post},
                    "restandardise": restandardise_arm(probes, rows_of[pset], X["post"]["probe_train"],
                                                       X["post"]["threshold_calib"], X["post"]["eval"], ncal, neg, pos)}
            rblk: dict[str, Any] = {}
            for lvl in LEVELS:
                per = [RP.partition(c_, e_, pool_pre, sc["pos_pre"], H_pre, arms, target) for c_, e_ in parts[lvl]]
                rblk[lvl] = {name: RP.summarise(per, name) for name in arms}
                rep, arr = reports[lvl]
                for q in ("fpr_post", "paired_ratio", "ratio_to_target", "tpr_post", "delta_tpr", "floor_ratio_to_target",
                          "tpr_pre"):
                    mine = np.array([p["frozen"][q] for p in per[1:]])
                    if not np.array_equal(mine, arr[q]) or per[0]["frozen"][q] != rep[q]["frozen"]:
                        refuse(f"{where} {lvl}: the repair path is not repartition_estimate ({q})")
                for name in arms:
                    abslog.setdefault((pset, lvl, name), []).append(np.array([p[name]["abs_log_ratio_to_target"] for p in per]))
            out["repair"][(li, pset)] = rblk

            # --- healthcheck_d156: the reference-set statistics per partition, both levels ---
            d_auc = au["frozen"]["mean_over_seeds"] - au["pre"]["mean_over_seeds"]
            auc_flag = bool(d_auc <= -0.05 or au["frozen"]["mean_over_seeds"] < 0.70)
            cblk: dict[str, Any] = {"layer": li, "probe_set": pset, "probe_C": C, "is_selected_layer": li == L_sel,
                                    "input_verification": verification,
                                    "auc_rule": {"auroc_pre": au["pre"]["mean_over_seeds"],
                                                 "auroc_post_frozen": au["frozen"]["mean_over_seeds"], "d_auroc": d_auc,
                                                 "flag": auc_flag, "source": source_note}}
            for lvl in LEVELS:
                rep, arr = reports[lvl]
                hper: dict[str, list[float]] = {q: [] for q in HC.STATS if q != "realised_paired_ratio"}
                mine_r = []
                for c_, e_ in parts[lvl]:
                    T = HC.thresholds(pool_pre[c_], target)
                    h = HC.health(pool_pre[c_], pool_post[c_], T)
                    for q in hper:
                        hper[q].append(h[q])
                    mine_r.append(P.ratio_from_flags(pool_pre[e_] >= T, pool_post[e_] >= T))
                v = {q: np.asarray(x, dtype="<f8") for q, x in hper.items()}
                v["realised_paired_ratio"] = np.concatenate([[rep["paired_ratio"]["frozen"]], arr["paired_ratio"]]).astype("<f8")
                if not np.array_equal(np.asarray(mine_r, dtype="<f8"), v["realised_paired_ratio"]):
                    refuse(f"{where} {lvl}: the health partitions are not repartition_estimate's")
                blk: dict[str, Any] = {q: P.band_summary(x[1:], x[0]) for q, x in v.items()}
                rr = v["realised_paired_ratio"]
                flags = {"gaussian": v["gaussian_ratio"], "direct": v["direct_ratio"]}
                blk["share_of_draws_in_class"] = {c: float(HC.crosses(rr[1:], cut).mean()) for c, cut in HC.CLASSES.items()}
                blk["share_of_draws_flagged"] = {f"{dn}|{c}": float(HC.crosses(x[1:], cut).mean()) for dn, x in flags.items()
                                                 for c, cut in HC.CLASSES.items()}
                blk["share_of_draws_flagged"].update({f"auc_rule|{c}": float(auc_flag) for c in HC.CLASSES})
                cblk[lvl] = blk
                for q, x in v.items():
                    out["hc_arrays"][f"L{li}|{pset}|{lvl}|{q}"] = x
                for c, cut in HC.CLASSES.items():
                    y = HC.crosses(rr, cut)
                    for dname in HC.DETECTORS:
                        f = np.full(rr.shape, auc_flag) if dname == "auc_rule" else HC.crosses(flags[dname], cut)
                        cc = confusion.setdefault((pset, lvl), {})
                        for key, val in (("tp", y & f), ("fn", y & ~f), ("fp", ~y & f), ("tn", ~y & ~f)):
                            cc[f"{dname}|{c}|{key}"] = cc.get(f"{dname}|{c}|{key}", np.zeros(rr.shape, dtype=np.int64)) + val
            out["health"][(li, pset)] = cblk

            # --- at L*: the D-282 combined TPR interval (table3_tpr_d282) and the geometry (intent_selected) ---
            if li == L_sel:
                rt = np.concatenate([[reports[LEVELS[0]][0]["ratio_to_target"]["frozen"]],
                                     reports[LEVELS[0]][1]["ratio_to_target"]]).astype("<f8")
                own = {"operating_point": reports[LEVELS[0]][0],
                       "frozen_curve_abs_log_ratio_to_target": {"per_draw_ratio_to_target_sha256": sha256_bytes(rt.tobytes())}}
                anchor_cell = own if tpr_cells is None else tpr_cells.get((li, pset))
                report_t, _, reps_t = T3.anchored_replicates(sc, draws[LEVELS[0]], target, anchor_cell, CNT_P, where)
                tpr = {q: {"frozen": report_t[q]["frozen"], "mean_D156": report_t[q]["mean"],
                           "band95_D156": report_t[q]["band95"], "combined": P.ci(reps_t[q]),
                           "replicates_sha256": sha256_bytes(reps_t[q].astype("<f8").tobytes())} for q in T3.QUANT}
                if CNT_IN is None:
                    CNT_IN = P.counts_matrix(n_neg, n_boot, intent_seed)
                geo = geometry_neg(probes, X["pre"]["eval"][neg], X["post"]["eval"][neg], CNT_IN)
                out["lstar"][pset] = {
                    "operating_point": {lvl: reports[lvl][0] for lvl in LEVELS}, "tpr": tpr, "geometry_neg": geo,
                    "arrays": {**{f"lstar|{pset}|{lvl}|{q}": np.concatenate([[reports[lvl][0][q]["frozen"]],
                                                                             reports[lvl][1][q]]).astype("<f8")
                                  for lvl in LEVELS for q in ("paired_ratio", "delta_tpr")},
                               f"lstar|{pset}|delta_tpr_combined": reps_t["delta_tpr"].astype("<f8")}}
        print(f"  [{label}] layer {li:>2} done", flush=True)

    # --- pair level: depth_d156's sweep maximum of the ratio to target and AUROC drop over layers; repair_d156's
    #     per-draw mean over layers; healthcheck_d156's confusion counts ---
    def at(li: int, pset: str, lvl: str) -> dict[str, Any]:
        c = out["depth"][(li, pset)]
        return c if lvl == LEVELS[0] else c[lvl]

    def layer_row(pset: str, lvl: str, li: int) -> dict[str, Any]:
        op = at(li, pset, lvl)["operating_point"]
        return {"layer": li, **{q: op[q] for q in ("paired_ratio", "ratio_to_target", "tpr_pre", "tpr_post", "delta_tpr",
                                                   "floor_ratio_to_target", "floor_abs_log")},
                "share_of_draws_floored": at(li, pset, lvl)["realised_censoring"]["share_of_draws_floored"]}

    pair: dict[str, Any] = {"n_layers": len(layers), "sweep_maximum_ratio_to_target": {}, "auroc_drop_frozen": {},
                            "repair_mean_abs_log_over_layers": {}, "repair_per_draw_sha256": {}}
    rep_arrays: dict[str, np.ndarray] = {}
    for pset in PROBE_SETS:
        for lvl in LEVELS:
            mrt = np.array([at(li, pset, lvl)["operating_point"]["ratio_to_target"]["mean"] for li in layers])
            mfl = np.array([at(li, pset, lvl)["operating_point"]["floor_ratio_to_target"]["mean"] for li in layers])
            (l_rt, t_rt), (l_fl, t_fl) = pick(mrt, "max", layers), pick(mfl, "max", layers)
            pair["sweep_maximum_ratio_to_target"].setdefault(pset, {})[lvl] = {
                **layer_row(pset, lvl, l_rt), "n_layers_tied": t_rt,
                "null_no_update": {"value": float(mfl[layers.index(l_fl)]), "layer": l_fl, "n_layers_tied": t_fl,
                                   "rule": NULL_RULE}}
            for name in REPAIR_ARMS:
                m = np.mean(np.vstack(abslog[(pset, lvl, name)]), axis=0).astype("<f8")
                pair["repair_mean_abs_log_over_layers"].setdefault(pset, {}).setdefault(lvl, {})[name] = P.band_summary(m[1:], m[0])
                pair["repair_per_draw_sha256"].setdefault(pset, {}).setdefault(lvl, {})[name] = sha256_bytes(m.tobytes())
                rep_arrays[f"repair|{pset}|{lvl}|{name}"] = m
        Dr = np.stack([drop_reps[pset][li] for li in layers])
        pair["auroc_drop_frozen"][pset] = {
            "note": D.AUROC_NOTE,
            "mean_over_layers": {"point": float(Dr[:, 0].mean()), **P.ci(Dr[:, 1:].mean(axis=0))},
            "max_over_layers": {"point": float(Dr[:, 0].max()), "layer": int(layers[int(np.argmax(Dr[:, 0]))]),
                                **P.ci(Dr[:, 1:].max(axis=0))}}
    counts: dict[str, Any] = {}
    for (pset, lvl), cc in confusion.items():
        blk = counts.setdefault(pset, {}).setdefault(lvl, {})
        for key, x in cc.items():
            out["hc_arrays"][f"counts|{pset}|{lvl}|{key}"] = x
            blk[key] = {"d156_mean": float(x[1:].mean()), "frozen_split": int(x[0])}
    out["confusion"] = counts
    out["pair"] = pair
    out["repair_arrays"] = rep_arrays
    return out


# ------------------------------------------------------------------------------------------------ the anchor
def anchor_compare(comp: dict[str, Any], core: str, L_sel: int, full: bool, depth_doc: dict[str, Any],
                   repair_doc: dict[str, Any], hc_doc: dict[str, Any], hc_z: dict[str, np.ndarray],
                   intent_doc: dict[str, Any], t3_doc: dict[str, Any], dmg_doc: dict[str, Any]) -> dict[str, Any]:
    """D-363 (4): the R-empty computation against the records of record, as canonical JSON text or bit for bit; refuses
    on any difference. full: also the pair-level values, which need every layer."""
    bad: list[str] = []
    n: Counter[str] = Counter()

    def same(kind: str, where: str, a: Any, b: Any) -> None:
        n[kind] += 1
        if RP.canonical(a) != RP.canonical(b):
            bad.append(f"{kind} {where}")

    dcells = {(c["layer"], c["probe_set"]): c for c in depth_doc["cells"]}
    rcells = {(c["layer"], c["probe_set"]): c for c in repair_doc["cells"]}
    hcells = {(c["layer"], c["probe_set"]): c for c in hc_doc["cells"]}
    for (li, pset), mine in comp["depth"].items():
        w = f"L{li} {pset}"
        rec, rrec, hrec = dcells.get((li, pset)), rcells.get((li, pset)), hcells.get((li, pset))
        if rec is None or rrec is None or hrec is None:
            bad.append(f"no record cell {w}")
            continue
        for key in ("probe_C", "fit_row_order", "is_selected_layer", "operating_point",
                    "frozen_curve_abs_log_ratio_to_target", "realised_censoring", "probe_fit_diagnostics"):
            same("depth", f"{w} {key}", mine[key], rec[key])
        for key in ("note", "operating_point", "frozen_curve_abs_log_ratio_to_target", "realised_censoring"):
            same("depth", f"{w} {LEVELS[1]} {key}", mine[LEVELS[1]][key], rec[LEVELS[1]][key])
        for name in ("pre", "frozen", "drop_frozen"):
            same("depth", f"{w} auroc {name}", mine["auroc"][name], rec["auroc"][name])
        for lvl in LEVELS:
            for arm in REPAIR_ARMS:
                same("repair", f"{w} {lvl} {arm}", comp["repair"][(li, pset)][lvl][arm], rrec[lvl][arm])
        hm = comp["health"][(li, pset)]
        strip = lambda c: {k: ({kk: vv for kk, vv in v.items() if kk != "source"} if k == "auc_rule" else v)  # noqa: E731
                           for k, v in c.items()}
        same("healthcheck", w, strip(hm), strip(hrec))
        for key, x in comp["hc_arrays"].items():
            if key.startswith(f"L{li}|{pset}|"):
                n["healthcheck_array"] += 1
                y = hc_z.get(key)
                if y is None or y.dtype != x.dtype or not np.array_equal(x, y):
                    bad.append(f"healthcheck_array {key}")
    dmg = {(c["cell"], c["probe_set"]): c for c in dmg_doc["cells"] if c.get("scope") == "4a"}
    icell = next((c for c in intent_doc["cells"] if c["run"] == core and c["is_selected_layer"]), None)
    irow = next((c for c in intent_doc["c1_sensitivity"] if c["run"] == core), None)
    t3row = (t3_doc["rows"].get(f"depth_{core}") or {}).get("probe_sets") or {}
    for pset, ls in comp["lstar"].items():
        w = f"L{L_sel} {pset}"
        dc = dmg.get((core, pset))
        if dc is None or dc["layer"] != L_sel:
            bad.append(f"damage {w}: no 4a cell at L*")
        else:
            mine = ls["operating_point"][LEVELS[0]]
            same("damage", w, mine, {k: dc.get(k) for k in mine})
            same("damage", f"{w} {LEVELS[1]}", ls["operating_point"][LEVELS[1]], dc[LEVELS[1]])
        ic = icell if pset == "selected" else irow
        if ic is None or ic["layer"] != L_sel:
            bad.append(f"intent {w}: no cell at L*")
        else:
            same("intent", w, ls["geometry_neg"], ic["geometry"]["neg"])
        tb = ((t3row.get(pset) or {}).get("layers") or {}).get(str(L_sel))
        if tb is None:
            bad.append(f"table3_tpr {w}: no block")
        else:
            for q in T3.QUANT:
                same("table3_tpr", f"{w} {q}", ls["tpr"][q], tb[q])
    if full:
        ps_rec = depth_doc["pair_summary"]
        for pset in PROBE_SETS:
            for lvl in LEVELS:
                same("depth_pair", f"{pset} {lvl} sweep_maximum.ratio_to_target",
                     comp["pair"]["sweep_maximum_ratio_to_target"][pset][lvl], ps_rec[pset][lvl]["sweep_maximum"]["ratio_to_target"])
                for arm in REPAIR_ARMS:
                    same("repair_pair", f"{pset} {lvl} {arm}", comp["pair"]["repair_mean_abs_log_over_layers"][pset][lvl][arm],
                         repair_doc["pair_summary_mean_abs_log_over_layers"][pset][lvl][arm])
                    same("repair_pair", f"{pset} {lvl} {arm} per-draw sha256", comp["pair"]["repair_per_draw_sha256"][pset][lvl][arm],
                         repair_doc["pair_per_draw_mean_abs_log_over_layers"][pset][lvl][arm]["sha256"])
            same("depth_pair", f"{pset} auroc_drop_frozen", comp["pair"]["auroc_drop_frozen"][pset], ps_rec[pset]["auroc_drop_frozen"])
        same("healthcheck_pair", "confusion_counts_pooled_over_cells", comp["confusion"], hc_doc["confusion_counts_pooled_over_cells"])
        n["healthcheck_pair"] += 1
        if list(comp["hc_arrays"]) != list(hc_z) or any(not np.array_equal(comp["hc_arrays"][k], hc_z[k])
                                                         or comp["hc_arrays"][k].dtype != hc_z[k].dtype for k in hc_z):
            bad.append("healthcheck_pair: the npz arrays (keys, order or values)")
    if bad:
        refuse(f"the anchor (R empty) does not reproduce the records of record in {len(bad)} of {sum(n.values())} "
               f"comparisons: {bad[:10]}")
    return {"identical": True, "n_comparisons": sum(n.values()), "by_kind": dict(n), "layers": sorted({li for li, _ in comp["depth"]}),
            "pair_level": full,
            "kind": "every comparison as canonical JSON text (RP.canonical) or, for arrays, bit for bit with the same "
                    "dtype; the health check's auc_rule.source string names the depth file and is left out"}


# ------------------------------------------------------------------------------------------------ Table III row
def table_iii_cells(comp: dict[str, Any]) -> dict[str, Any]:
    """The Table III cells of one run from a computation, each with its printed value."""
    sel = "selected"
    ls = comp["lstar"][sel]
    rm = comp["pair"]["repair_mean_abs_log_over_layers"][sel]
    sm = comp["pair"]["sweep_maximum_ratio_to_target"][sel]
    out: dict[str, Any] = {}
    for lvl in LEVELS:
        out[lvl] = {"frozen_mean_abs_log": {**rm[lvl]["frozen"], "value": rm[lvl]["frozen"]["mean"]},
                    "restandardise_mean_abs_log": {**rm[lvl]["restandardise"], "value": rm[lvl]["restandardise"]["mean"]},
                    "worst_fpr_to_target": {"layer": sm[lvl]["layer"], "ratio_to_target": sm[lvl]["ratio_to_target"],
                                            "null_no_update": sm[lvl]["null_no_update"]["value"],
                                            "value": sm[lvl]["ratio_to_target"]["mean"]}}
    dt = ls["tpr"]["delta_tpr"]
    out[LEVELS[0]]["delta_tpr_at_lstar"] = {**dt, "value": dt["mean_D156"]}
    dc = ls["operating_point"][LEVELS[1]]["delta_tpr"]
    out[LEVELS[1]]["delta_tpr_at_lstar"] = {"frozen": dc["frozen"], "mean_D156": dc["mean"], "band95_D156": dc["band95"],
                                            "value": dc["mean"], "note": "D-156 band only at the cluster level (D-282 (2))"}
    au = comp["pair"]["auroc_drop_frozen"][sel]["mean_over_layers"]
    out[LEVELS[0]]["auroc_drop_mean_over_layers"] = {**au, "value": au["point"]}
    g = ls["geometry_neg"]
    out[LEVELS[0]]["d_norm_at_lstar"] = {"mean": g["d_norm"]["mean"], "ci95": g["d_norm"]["ci95"], "value": g["d_norm"]["mean"]}
    out[LEVELS[0]]["cos_at_lstar"] = {"mean": g["cos"]["mean"], "ci95": g["cos"]["ci95"], "value": g["cos"]["mean"]}
    return out


def table_iii_record(core: str, L_sel: int, depth_doc: dict[str, Any], repair_doc: dict[str, Any],
                     t3_doc: dict[str, Any], intent_doc: dict[str, Any]) -> dict[str, Any]:
    """The same cells read from the records of record (Table III's sources, src/tables_d156.py table_3)."""
    sel = "selected"
    rm = repair_doc["pair_summary_mean_abs_log_over_layers"][sel]
    ps = depth_doc["pair_summary"][sel]
    out: dict[str, Any] = {}
    for lvl in LEVELS:
        sm = ps[lvl]["sweep_maximum"]["ratio_to_target"]
        out[lvl] = {"frozen_mean_abs_log": {**rm[lvl]["frozen"], "value": rm[lvl]["frozen"]["mean"]},
                    "restandardise_mean_abs_log": {**rm[lvl]["restandardise"], "value": rm[lvl]["restandardise"]["mean"]},
                    "worst_fpr_to_target": {"layer": sm["layer"], "ratio_to_target": sm["ratio_to_target"],
                                            "null_no_update": sm["null_no_update"]["value"],
                                            "value": sm["ratio_to_target"]["mean"]}}
    dt = t3_doc["rows"][f"depth_{core}"]["probe_sets"][sel]["layers"][str(L_sel)]["delta_tpr"]
    out[LEVELS[0]]["delta_tpr_at_lstar"] = {**dt, "value": dt["mean_D156"]}
    cell = next(c for c in depth_doc["cells"] if c["layer"] == L_sel and c["probe_set"] == sel)
    dc = cell[LEVELS[1]]["operating_point"]["delta_tpr"]
    out[LEVELS[1]]["delta_tpr_at_lstar"] = {"frozen": dc["frozen"], "mean_D156": dc["mean"], "band95_D156": dc["band95"],
                                            "value": dc["mean"], "note": "D-156 band only at the cluster level (D-282 (2))"}
    au = ps["auroc_drop_frozen"]["mean_over_layers"]
    out[LEVELS[0]]["auroc_drop_mean_over_layers"] = {**au, "value": au["point"]}
    g = next(c for c in intent_doc["cells"] if c["run"] == core and c["is_selected_layer"])["geometry"]["neg"]
    out[LEVELS[0]]["d_norm_at_lstar"] = {"mean": g["d_norm"]["mean"], "ci95": g["d_norm"]["ci95"], "value": g["d_norm"]["mean"]}
    out[LEVELS[0]]["cos_at_lstar"] = {"mean": g["cos"]["mean"], "ci95": g["cos"]["ci95"], "value": g["cos"]["mean"]}
    return out


def beside(sens: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
    """Every cell's sensitivity value beside its value of record, and the change (sensitivity minus record)."""
    return {lvl: {cell: {"sensitivity": sens[lvl][cell], "record": rec[lvl][cell],
                         "change": change(sens[lvl][cell]["value"], rec[lvl][cell]["value"])}
                  for cell in sens[lvl]} for lvl in sens}


# ------------------------------------------------------------------------------------------------ per-run job
def run_job(args: argparse.Namespace) -> int:
    t_start = time.time()
    core = args.run
    stem = f"depth_{core}"
    seeds, target = REGISTERED["seeds"], REGISTERED["target_fpr"]
    registered = args.n_draws == REGISTERED["n_draws"] and args.n_boot == REGISTERED["n_boot"] and args.layers is None
    name = f"hh_sensitivity_d363_anchor_{core}" if args.anchor_only else f"hh_sensitivity_d363_{core}"
    final_path = (RESULTS / f"{name}.json").resolve()
    out_path = Path(args.out).resolve() if args.out else final_path
    npz_path = out_path.with_name(out_path.stem + "_draws.npz")
    final = out_path == final_path
    for p in [out_path] + ([] if args.anchor_only else [npz_path]):
        if p.exists():
            refuse(f"{p} exists; results are never overwritten (D-152 P5)")
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        refuse("a test run (--out) must not write under results/; only the run's own final name may")
    threads = blas_threads()
    if final:
        problems = final_problems(registered, threads)
        if args.anchor_only and core != ANCHOR_RUN:
            problems.append(f"the every-layer anchor of record is {ANCHOR_RUN}'s")
        if problems:
            refuse(f"the final name: {problems}; use --out outside results/ for a test")
    code0 = P.code_block(CODE_FILES)

    # --- the records of record, each bound to its manifest or its record --------------------------------------------
    contam, contam_sha = read_bound(CONTAMINATION)
    R, r_info = removal_set(contam)
    sel_doc, sel_sha = read_bound(SELECTION)
    if list(sel_doc["config"]["seeds"]) != seeds or float(sel_doc["config"]["holdout"]) != P.HOLDOUT:
        refuse("the selection's seeds or holdout differ from the registered ones")
    sel = next((m for m in sel_doc["models"] if m["pair"] == D.LORA_SELECTION_PAIR), None)
    if sel is None or not sel["selected"]["arm"].startswith("C="):
        refuse(f"no logistic-C selection for {D.LORA_SELECTION_PAIR} in {rel(SELECTION)}")
    C_sel, L_sel = float(sel["selected"]["arm"].split("=")[1]), int(sel["selected"]["layer"])
    man_paths = {"depth": DEPTH_MANIFEST, "repair": REPAIR_MANIFEST, "healthcheck": HC_MANIFEST, "table3_tpr": T3TPR_MANIFEST}
    man = {k: T3.manifest(p) for k, p in man_paths.items()}
    depth_rel = f"results/depth_d156_{core}.json"
    depth_doc, depth_sha = read_bound(ROOT / depth_rel, man["depth"][0])
    repair_doc, repair_sha = read_bound(RESULTS / f"repair_d156_{core}.json", man["repair"][0])
    hc_doc, hc_sha = read_bound(RESULTS / f"healthcheck_d156_{core}.json", man["healthcheck"][0])
    hc_npz_rel = hc_doc["draws_npz"]["file"]
    hc_z, hc_npz_sha = npz_bound(hc_npz_rel, man["healthcheck"][0], hc_doc["draws_npz"]["sha256"])
    intent_doc, intent_sha = read_bound(INTENT)
    t3_doc, t3_sha = read_bound(T3TPR, man["table3_tpr"][0])
    dmg_doc, dmg_sha = read_bound(DAMAGE)
    dmg_rel = _repo_rel(DAMAGE)
    cl_bytes = Path(CLUSTERS).read_bytes()
    cl_sha = sha256_bytes(cl_bytes)
    cl_doc = json.loads(cl_bytes.decode("utf-8"))  # not a result (D-297): manifests/negative_clusters.json, hashed and listed
    # the records were made on this selection and these parameters, and each names the others' bytes
    dc = depth_doc["config"]
    checks = [("depth selection", (dc["selected_C"], dc["selected_layer"], dc["selection_sha256"]), (C_sel, L_sel, sel_sha)),
              ("depth run", (dc.get("registry_pair"), dc.get("selection_of_pair")), (D.registry_stem(stem), D.LORA_SELECTION_PAIR)),
              ("depth parameters", (list(dc["seeds"]), dc["n_draws"], dc["target_fpr"]), (seeds, args.n_draws, target)),
              ("depth cluster manifest", dc["cluster_manifest"]["sha256"], cl_sha),
              ("depth's damage record", ((depth_doc.get("twomoment_check") or {}).get("file"),
                                         (depth_doc.get("twomoment_check") or {}).get("sha256")), (dmg_rel, dmg_sha)),
              ("repair family and anchor", (repair_doc["config"]["family"], repair_doc["anchor"]["file"],
                                            repair_doc["anchor"]["sha256"], repair_doc["config"]["n_draws"]),
               ("repair", depth_rel, depth_sha, args.n_draws)),
              ("healthcheck anchor", (hc_doc["anchor"]["file"], hc_doc["anchor"]["sha256"], hc_doc["config"]["n_draws"]),
               (depth_rel, depth_sha, args.n_draws)),
              ("table3_tpr anchor", (t3_doc["anchor"]["depth_records_sha256"].get(depth_rel), t3_doc["config"]["n_draws"],
                                     t3_doc["config"]["n_boot"]), (depth_sha, args.n_draws, args.n_boot)),
              ("intent selection", (intent_doc["config"]["selection_sha256"], intent_doc["config"]["selected_C"],
                                    intent_doc["config"]["selected_layer"], intent_doc["config"]["n_boot"]),
               (sel_sha, C_sel, L_sel, args.n_boot)),
              ("damage parameters", (dmg_doc["config"]["n_draws"], dmg_doc["config"]["target_fpr"]), (args.n_draws, target))]
    bad = [w for w, a, b in checks if RP.canonical(a) != RP.canonical(b)]
    if bad:
        refuse(f"the records of record disagree with this run or with each other: {bad}")
    layers_all = list(dc["layers"])
    # the C of record per layer: every depth cell carries the selected C (selected) or 1 (C=1)
    wrong_c = [(c["layer"], c["probe_set"]) for c in depth_doc["cells"]
               if c["probe_C"] != (C_sel if c["probe_set"] == "selected" else 1.0)]
    if wrong_c or L_sel not in layers_all:
        refuse(f"the depth record's cells carry another C {wrong_c[:3]} or no L{L_sel}")
    if args.layers is None:
        layers = layers_all
    else:
        unknown = sorted(set(args.layers) - set(layers_all))
        if unknown:
            refuse(f"layers {unknown} are not in the depth record's grid {layers_all}")
        layers = sorted(set(args.layers) | {L_sel})
    intent_seed = int(intent_doc["config"]["boot_seeds"]["eval_negatives"])

    # --- the caches through the input registry, the bytes every record read --------------------------------------
    reg_stem = D.registry_stem(stem)
    cfg = REG.run(reg_stem)
    axis = cfg["axis"]
    if axis not in P.AXIS_DRAW_SEED or any(cfg[k] != dc.get(k) for k in ("pre_repo", "pre_rev", "post_repo", "post_rev")):
        refuse(f"the input registry's {reg_stem} is not the depth record's run")
    data: dict[str, dict[str, Any]] = {"pre": {}, "post": {}}
    input_sha: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    resolved: dict[str, str] = {}
    for side in data:
        for part in PARTS:
            z, r = REG.load(cfg[f"{side}_repo"], axis, part)
            data[side][part] = z
            input_sha[f"{side}/{part}"], input_recs[f"{side}/{part}"] = r["sha256"], r
            resolved[f"{side}/{part}"] = str(resolve_cache(r["file"], REG.CACHE))
    reg_sources = REG.sources_sha256()
    if D.sources_now() != reg_sources:
        refuse("the extraction records or pins on disk are not the ones the registry read")
    read_by = {"depth": depth_doc["input_cache_sha256"], "repair": repair_doc["input_cache_sha256"],
               "healthcheck": hc_doc["input_cache_sha256"],
               "table3_tpr": {k: t3_doc["input_cache_sha256"].get(f"{D.out_name(stem)}:{k}") for k in input_sha},
               "damage": {k: dmg_doc["input_cache_sha256"].get(f"{core}:{k}") for k in input_sha if k != "post/probe_train"},
               "intent": {**{f"pre/{p}": intent_doc["input_cache_sha256"].get(f"pre/{p}") for p in PARTS},
                          "post/eval": intent_doc["input_cache_sha256"].get(f"{core}:post/eval")},
               "selection": {k: sel_doc["input_cache_sha256"].get(f"{D.LORA_SELECTION_PAIR}:{k}")
                             for k in ("pre/probe_train", "pre/threshold_calib")}}
    other = [f"{who}:{k}" for who, m in read_by.items() for k, h in m.items() if input_sha.get(k) != h]
    if other:
        refuse(f"a record of record read other cache bytes: {other}")
    verification = REG.verification(input_recs)
    for part in PARTS:
        a, b = data["pre"][part], data["post"][part]
        if not np.array_equal(a["item_ids"], b["item_ids"]) or not np.array_equal(a["labels"], b["labels"]):
            refuse(f"{part} items or labels differ between pre and post")
        for zz in (a, b):
            if "nonfinite" in zz.files and zz["nonfinite"].any():
                refuse(f"{part} has non-finite items")
    grids = [sorted(int(kk.split("_")[1]) for kk in zz.files if kk.startswith("layer_")) for s in data for zz in data[s].values()]
    if any(g != grids[0] for g in grids) or grids[0] != layers_all:
        refuse(f"the caches' layer grids differ from each other or from the depth record's {layers_all}")
    cl_axis = cl_doc["axes"][axis]
    y_cal0 = data["pre"]["threshold_calib"]["labels"].astype(bool)
    y_ev0 = data["pre"]["eval"]["labels"].astype(bool)
    full_pool = [str(i) for i in data["pre"]["threshold_calib"]["item_ids"][~y_cal0]] + \
                [str(i) for i in data["pre"]["eval"]["item_ids"][~y_ev0]]
    if cl_axis["n_pooled_negatives"] != len(full_pool) or cl_axis["n_calibration"] != int((~y_cal0).sum()) or \
            cl_axis["pool_item_ids_sha256"] != sha256_bytes("\n".join(full_pool).encode("utf-8")):
        refuse(f"manifests/negative_clusters.json does not describe the {axis} pool")

    # --- R in the caches: each item of R in its own split's cache exactly once, in no other ------------------------
    all_R = frozenset().union(*R.values())
    keep: dict[str, np.ndarray] = {}
    removed: dict[str, dict[str, int]] = {}
    for part in PARTS:
        ids = np.array([str(i) for i in data["pre"][part]["item_ids"]])
        if len(set(ids.tolist())) != len(ids):
            refuse(f"{part}: the cache holds an item id twice")
        inR = np.isin(ids, sorted(all_R))
        hit = set(ids[inR].tolist())
        if hit != set(R[part]):
            refuse(f"{part}: the cache holds {len(hit & R[part])} of R's {len(R[part])} items of its file and "
                   f"{len(hit - R[part])} of R's other files'")
        keep[part] = ~inR
        lab = data["pre"][part]["labels"].astype(bool)
        removed[part] = {"negatives": int((inR & ~lab).sum()), "positives": int((inR & lab).sum())}
    keep_all = {part: np.ones(len(keep[part]), bool) for part in PARTS}

    print(f"D-363 {core} [{axis}]: C={C_sel:g}, L*={L_sel}; layers {layers}; R removed {removed}; n_draws {args.n_draws}, "
          f"n_boot {args.n_boot}; registered {registered}; anchor-only {args.anchor_only}", flush=True)
    rec_note = f"{depth_rel} (frozen eval split)"
    dcells = {(c["layer"], c["probe_set"]): c for c in depth_doc["cells"]}
    common = dict(L_sel=L_sel, C_sel=C_sel, seeds=seeds, n_draws=args.n_draws, n_boot=args.n_boot, target=target,
                  axis=axis, cl_axis=cl_axis, verification=verification, intent_seed=intent_seed)
    # --- ANCHOR first (R empty): at L*, or at every layer with --anchor-only --------------------------------------
    anchor_layers = layers if args.anchor_only else [L_sel]
    comp0 = compute(data, keep_all, anchor_layers, tpr_cells=dcells, source_note=rec_note, label="anchor, R empty", **common)
    full = args.anchor_only and anchor_layers == layers_all
    anchor = anchor_compare(comp0, core, L_sel, full, depth_doc, repair_doc, hc_doc, hc_z, intent_doc, t3_doc, dmg_doc)
    print(f"anchor: {anchor['n_comparisons']} comparisons identical {anchor['by_kind']}", flush=True)
    comp = None
    if not args.anchor_only:
        comp = compute(data, keep, layers, tpr_cells=None,
                       source_note="this record's AUROC blocks (reduced evaluation split)", label="R removed", **common)

    # --- nothing read may have changed while the run went ----------------------------------------------------------
    code1 = P.code_block(CODE_FILES)
    changed = [w for w, a, b in (("code", code1["sha256"], code0["sha256"]), ("registry sources", D.sources_now(), reg_sources),
                                 ("cluster manifest", sha256_file(CLUSTERS), cl_sha)) if a != b]
    changed += [r_ for r_, h in _RESULT_READS.items() if sha256_file(ROOT / r_) != h]
    changed += [hc_npz_rel] if sha256_file(ROOT / hc_npz_rel) != hc_npz_sha else []
    changed += [rel(p) for k, p in man_paths.items() if sha256_file(p) != man[k][1]]
    changed += [f"cache {r['file']}" for r in input_recs.values() if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    if changed:
        refuse(f"changed during the run: {changed}")

    config = {"run": core, "depth_stem": stem, "registry_pair": reg_stem, "arm": ARM_OF.get(core), "axis": axis,
              "dose": int(core.rsplit("_s", 1)[1]),
              "selection_file": rel(SELECTION), "selection_sha256": sel_sha, "selection_of_pair": D.LORA_SELECTION_PAIR,
              "selected_C": C_sel, "selected_layer": L_sel, "layers": layers, "anchor_layers": anchor_layers,
              "probe_sets": {"selected": f"C={C_sel:g} (the D-117 selected C of record, every depth-record cell), "
                                         "D-117 permutation order",
                             "C=1": "C=1, file order (the sensitivity row)"},
              "split": f"D-117 (holdout {P.HOLDOUT}) applied to the reduced probe_train", "seeds": seeds,
              "target_fpr": target, "n_draws": args.n_draws, "n_boot": args.n_boot,
              "draw_seed": P.AXIS_DRAW_SEED[axis], "cluster_draw_seed": P.AXIS_CLUSTER_DRAW_SEED[axis],
              "item_bootstrap_seeds": D.ITEM_BOOT_SEED, "intent_boot_seed_eval_negatives": intent_seed,
              "estimator": "D-156 over n_draws re-partitions of the reduced pooled negatives (threshold_calib minus R, "
                           "then eval minus R, cache order) at the reduced sizes, the frozen split as one more draw, the "
                           "positives (eval positives minus R) fixed; D-164 beside, the cluster manifest's clusters "
                           "restricted to the reduced pool",
              "cluster_manifest": {"file": rel(CLUSTERS), "sha256": cl_sha},
              "pre_repo": cfg["pre_repo"], "pre_rev": cfg["pre_rev"], "post_repo": cfg["post_repo"], "post_rev": cfg["post_rev"],
              "inputs_from": "src/input_registry.py (D-165)", "input_registry_sources_sha256": reg_sources}
    records = {"depth": {"file": depth_rel, "sha256": depth_sha},
               "repair": {"file": f"results/repair_d156_{core}.json", "sha256": repair_sha},
               "healthcheck": {"file": f"results/healthcheck_d156_{core}.json", "sha256": hc_sha},
               "healthcheck_npz": {"file": hc_npz_rel, "sha256": hc_npz_sha},
               "intent": {"file": rel(INTENT), "sha256": intent_sha}, "table3_tpr": {"file": rel(T3TPR), "sha256": t3_sha},
               "damage": {"file": dmg_rel, "sha256": dmg_sha}, "contamination": {"file": rel(CONTAMINATION), "sha256": contam_sha},
               "manifests": {rel(p): man[k][1] for k, p in man_paths.items()}}
    payload: dict[str, Any] = {
        **provenance_block("hh_sensitivity_d363_anchor" if args.anchor_only else "hh_sensitivity_d363"),
        "preregistration": "D-363 (the author's D-357 (B)(4), option (a) of D-349); details by the closest D-entries "
                           "(D-181): D-344 (3), D-349, D-152 4a and 4c, D-156, D-164, D-117, D-255, D-282, D-337, D-356",
        "under_check": UNDER_CHECK, "code": code0,
        "reused_files_sha256": {f: sha256_file(CODE_ROOT / f) for f in REUSED_SHA256},
        "registered_parameters": registered, "blas_threads": threads, "elapsed_s": round(time.time() - t_start, 1),
        "input_verification": verification, "config": config, "config_hash": canonical_hash(config),
        "removal_set": {**r_info, "removed_from_this_run": removed},
        "anchor": {**anchor, "population": comp0["population"]},
        "records_of_record": records,
        "input_files_sha256": {**dict(_RESULT_READS), hc_npz_rel: hc_npz_sha, rel(CLUSTERS): cl_sha,
                               **{rel(p): man[k][1] for k, p in man_paths.items()}},
        "input_cache_sha256": input_sha,
        "input_cache_source": {k: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"],
                                   "resolved_path": resolved[k]} for k, r in input_recs.items()}}
    if comp is None:
        digest = write_results(out_path, payload, exclusive=True)
        print(f"Written: {rel(out_path)}\nsha256:  {digest}")
        return 0
    arrays = dict(comp["hc_arrays"])
    for pset in PROBE_SETS:
        arrays.update(comp["lstar"][pset]["arrays"])
    arrays.update(comp["repair_arrays"])
    for pset in PROBE_SETS:
        arrays.update({f"anchor|{k}": v for k, v in comp0["lstar"][pset]["arrays"].items()})
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    npz_bytes = buf.getvalue()
    sens = table_iii_cells(comp)
    rec = table_iii_record(core, L_sel, depth_doc, repair_doc, t3_doc, intent_doc)
    cells = [{"layer": li, "probe_set": pset, "probe_C": comp["depth"][(li, pset)]["probe_C"],
              "is_selected_layer": li == L_sel, "depth": comp["depth"][(li, pset)],
              "repair": comp["repair"][(li, pset)], "healthcheck": comp["health"][(li, pset)]}
             for li in layers for pset in PROBE_SETS]
    payload.update({
        "population": comp["population"],
        "table_iii_row": {"arm": ARM_OF.get(core), "dose": config["dose"], "under_check_arm": ARM_OF.get(core) == HH,
                          "cells": beside(sens, rec)},
        "lstar": {pset: {"operating_point": comp["lstar"][pset]["operating_point"], "tpr_D282": comp["lstar"][pset]["tpr"],
                         "geometry_neg": comp["lstar"][pset]["geometry_neg"]} for pset in PROBE_SETS},
        "pair_summary": comp["pair"], "confusion_counts_pooled_over_cells": comp["confusion"],
        "draws_npz": {"file": rel(npz_path), "sha256": sha256_bytes(npz_bytes), "size": len(npz_bytes), "keys": NPZ_KEYS},
        "cells": cells})
    json_bytes = json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8")
    problems = input_identity_problems(payload)
    if problems:
        refuse(f"to write (D-297 (1) results-writer rule): {problems}")
    staged: list[tuple[Path, Path, str]] = []
    try:
        for p, b in ((npz_path, npz_bytes), (out_path, json_bytes)):
            t, h = stage_bytes(p, b)
            staged.append((t, p, h))
        for t, p, h in staged:
            commit_staged(t, p, h)
    finally:
        for t, _, _ in staged:
            if t.exists():
                t.unlink()
    for _, p, h in staged:
        print(f"Written: {rel(p)}\nsha256:  {h}")
    c = sens[LEVELS[0]]
    print(f"{core}: frozen mean |log| {c['frozen_mean_abs_log']['mean']:.3f} (record {rec[LEVELS[0]]['frozen_mean_abs_log']['mean']:.3f}); "
          f"restand. {c['restandardise_mean_abs_log']['mean']:.3f}; delta TPR at L* {c['delta_tpr_at_lstar']['mean_D156']:+.4f}")
    return 0


# ------------------------------------------------------------------------------------------------ the summary
def between_arms(z_of: dict[str, dict[str, np.ndarray]], prefix: str) -> dict[str, list[dict[str, Any]]]:
    """damage_d156.main's 4a between-arm gaps (lines repeated): safety minus general at matched dose, formed within each
    draw at L*, both probe sets and levels, from the per-draw arrays (index 0 the frozen split)."""
    out: dict[str, list[dict[str, Any]]] = {lvl: [] for lvl in LEVELS}
    for lvl in LEVELS:
        for pset in PROBE_SETS:
            for arm, stems in POP.LORA_ARMS.items():
                if arm.startswith("general"):
                    continue
                for run, dose in zip(stems, DOSES):
                    g = POP.LORA_ARMS[GENERAL][DOSES.index(dose)]
                    row: dict[str, Any] = {"probe_set": pset, "safety_arm": arm, "safety_run": run, "general_run": g, "dose": dose}
                    for key, q, f in (("log_damage_gap", "paired_ratio", np.log), ("delta_tpr_gap", "delta_tpr", lambda v: v)):
                        a_s, a_g = z_of[run][f"{prefix}lstar|{pset}|{lvl}|{q}"], z_of[g][f"{prefix}lstar|{pset}|{lvl}|{q}"]
                        per_draw = f(a_s[1:]) - f(a_g[1:])
                        fz = f(np.array(a_s[0])) - f(np.array(a_g[0]))
                        row[key] = {**P.band_summary(per_draw, float(fz)), "share_of_draws_above_0": float((per_draw > 0).mean())}
                    out[lvl].append(row)
    return out


def tpr_gaps(z_of: dict[str, dict[str, np.ndarray]], prefix: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """D-356 (2)(a): the between-arm delta-TPR gap's combined interval at L* (item level): replicate r is the safety
    run's D-282 replicate r minus the general run's; the D-156 parts from the item-level gap rows."""
    out = []
    for row in rows:
        pset, s, g = row["probe_set"], row["safety_run"], row["general_run"]
        gap = z_of[s][f"{prefix}lstar|{pset}|delta_tpr_combined"] - z_of[g][f"{prefix}lstar|{pset}|delta_tpr_combined"]
        b = row["delta_tpr_gap"]
        out.append({"probe_set": pset, "safety_arm": row["safety_arm"], "safety_run": s, "general_run": g, "dose": row["dose"],
                    "delta_tpr_gap": {"frozen": b["frozen"], "mean_D156": b["mean"], "band95_D156": b["band95"],
                                      "share_of_draws_above_0_D156": b["share_of_draws_above_0"], "combined": P.ci(gap),
                                      "replicates_sha256": sha256_bytes(gap.astype("<f8").tobytes())}})
    return out


def table_iv_rows(z_of: dict[str, dict[str, np.ndarray]]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """healthcheck_summary_d156.lora_summary's rows (D-278): scoring over all nine, the three general and the six safety
    runs, every probe set and level, with D-255's estimator (pooled_counts, pair_cluster_index) unchanged."""
    groups = {"general": list(POP.LORA_ARMS[GENERAL]), "safety": [*POP.LORA_ARMS[HH], *POP.LORA_ARMS[PKU]]}
    runs = [r for g in groups.values() for r in g]
    prs = {r: {"z": {k: v for k, v in z_of[r].items() if k.startswith("L") or k.startswith("counts|")}} for r in runs}
    rows: dict[str, Any] = {}
    for ps in PROBE_SETS:
        for lvl in LEVELS:
            rows[f"{ps}|{lvl}"] = {"lora_all": HS.scoring(prs, runs, ps, lvl),
                                   **{f"lora_{g}": HS.scoring(prs, rr, ps, lvl) for g, rr in groups.items()}}
    return groups, rows


TABLE_IV_ROWS = (("LoRA, all nine runs", "lora_all"), ("LoRA, general (alpaca)", "lora_general"),
                 ("LoRA, safety (six runs)", "lora_safety"))


def change(a: Any, b: Any) -> float | None:
    """a - b, or None when either side has no value (e.g. a pooled sensitivity with no positive cell)."""
    return None if a is None or b is None else a - b


def table_iv_cells(rows: dict[str, Any]) -> dict[str, Any]:
    """Table IV's printed LoRA cells (src/tables_d156.py table_4): selected C, item level, two-fold change, pooled from
    mean counts with the pair-cluster interval, per detector (a value or interval the pooling left undefined is None)."""
    prim = rows["selected|item_level_D156"]
    return {name: {det: {q: {"value": x["pooled_from_mean_counts"].get(q),
                             "ci95": (x["pooled_from_mean_counts"].get(f"{q}_ci") or {}).get("ci95")}
                         for q in ("sensitivity", "specificity")}
                   for det in HC.DETECTORS for x in (prim[key][f"{det}|two_sided_2x"],)}
            for name, key in TABLE_IV_ROWS}


def table_iv_beside(sens_rows: dict[str, Any], rec_rows: dict[str, Any]) -> dict[str, Any]:
    """Table IV's printed LoRA cells, the sensitivity beside the record, with the change (None when a side is None)."""
    s, r = table_iv_cells(sens_rows), table_iv_cells(rec_rows)
    return {name: {det: {q: {"sensitivity": s[name][det][q], "record": r[name][det][q],
                             "change": change(s[name][det][q]["value"], r[name][det][q]["value"])}
                         for q in ("sensitivity", "specificity")} for det in HC.DETECTORS} for name in s}


def reading(sens: dict[str, np.ndarray], rec: dict[str, np.ndarray]) -> dict[str, Any]:
    """D-363 (3) on the frozen arm's per-draw mean over layers of |log(FPR/target)| (selected C), safety minus alpaca at
    each dose, within each draw; sens / rec: {f'{run}|{level}': per-draw array, index 0 the frozen split}. Survival is
    defined for the HH rows only; PKU's rows carry the same bands and the removal-only change (sensitivity minus record),
    with no verdict."""
    out: dict[str, Any] = {"rule": READING_RULE}
    for lvl in LEVELS:
        blk: dict[str, Any] = {}
        for dose in DOSES:
            alp = POP.LORA_ARMS[GENERAL][DOSES.index(dose)]
            d: dict[str, Any] = {}
            for arm, tag in ((HH, "hh_minus_alpaca"), (PKU, "pku_minus_alpaca")):
                run = POP.LORA_ARMS[arm][DOSES.index(dose)]
                d[tag] = {}
                for which, src in (("sensitivity", sens), ("record", rec)):
                    s, a = src[f"{run}|{lvl}"], src[f"{alp}|{lvl}"]
                    diff = s[1:] - a[1:]
                    band = P.band_summary(diff, float(s[0] - a[0]))
                    ms, ma = P.band_summary(s[1:], s[0])["mean"], P.band_summary(a[1:], a[0])["mean"]
                    blk_w: dict[str, Any] = {"safety_run": run, "general_run": alp, "safety_mean": ms, "alpaca_mean": ma,
                                             "difference_D156": band}
                    if arm == HH:
                        exceeds, above = bool(ms > ma), bool(band["band95"][0] > 0)
                        blk_w.update({"safety_exceeds_alpaca": exceeds, "band_wholly_above_0": above,
                                      "survives": exceeds and above})
                    d[tag][which] = blk_w
                d[tag]["change"] = {"safety_mean": change(d[tag]["sensitivity"]["safety_mean"], d[tag]["record"]["safety_mean"]),
                                    "difference_mean": change(d[tag]["sensitivity"]["difference_D156"]["mean"],
                                                              d[tag]["record"]["difference_D156"]["mean"])}
                if arm == PKU:
                    d[tag]["note"] = ("a removal-only effect on an uncontaminated arm (D-363 (3)): reported beside, with "
                                      "no survival verdict")
            blk[str(dose)] = d
        out[lvl] = blk
    out["verdict_hh_rows_item_level"] = {dose: ("SURVIVE" if out[LEVELS[0]][dose]["hh_minus_alpaca"]["sensitivity"]["survives"]
                                                else "DO NOT SURVIVE") for dose in out[LEVELS[0]]}
    out["note"] = UNDER_CHECK
    return out


def summary_job(args: argparse.Namespace) -> int:
    t_start = time.time()
    final_path = (RESULTS / "hh_sensitivity_d363.json").resolve()
    out_path = Path(args.out).resolve() if args.out else final_path
    final = out_path == final_path
    if out_path.exists():
        refuse(f"{out_path} exists; results are never overwritten")
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        refuse("a test output must not be under results/")
    threads = blas_threads()
    code0 = P.code_block(CODE_FILES)
    man_runs, man_runs_sha = T3.manifest(RUN_MANIFEST)
    docs, zs, inputs = {}, {}, {}
    for core in RUNS:
        d, h = read_bound(RESULTS / f"hh_sensitivity_d363_{core}.json", man_runs)
        z, hz = npz_bound(d["draws_npz"]["file"], man_runs, d["draws_npz"]["sha256"])
        docs[core], zs[core] = d, z
        inputs[d["draws_npz"]["file"]] = hz
    anchor_rel = f"results/hh_sensitivity_d363_anchor_{ANCHOR_RUN}.json"
    adoc, _ = read_bound(ROOT / anchor_rel, man_runs)
    registered = all(d.get("registered_parameters") is True for d in [*docs.values(), adoc])
    if final:
        problems = final_problems(registered, threads)
        if problems:
            refuse(f"the final name: {problems}")
    if adoc["anchor"].get("identical") is not True or adoc["anchor"].get("pair_level") is not True or \
            adoc["config"]["run"] != ANCHOR_RUN:
        refuse(f"{anchor_rel} is not an every-layer anchor of {ANCHOR_RUN}")
    # the arrays bound here were made by this revision: every record's code block and reused-file hashes are this run's
    reused_now = {f: sha256_file(CODE_ROOT / f) for f in REUSED_SHA256}
    other_code = [n for n, d in [*docs.items(), (anchor_rel, adoc)]
                  if RP.canonical(d["code"]["sha256"]) != RP.canonical(code0["sha256"])
                  or RP.canonical(d.get("reused_files_sha256")) != RP.canonical(reused_now)]
    if other_code:
        refuse(f"records made by another revision of the code (code or reused-file sha256): {other_code}")
    # one design for all nine: the same R, parameters, selection, layers and reduced population
    same_keys = ("selected_C", "selected_layer", "layers", "seeds", "target_fpr", "n_draws", "n_boot", "draw_seed",
                 "cluster_draw_seed", "selection_sha256", "cluster_manifest", "pre_repo", "pre_rev", "axis")
    ref = docs[RUNS[0]]
    diff = [(c, k) for c, d in docs.items() for k in same_keys if RP.canonical(d["config"][k]) != RP.canonical(ref["config"][k])]
    diff += [(c, "removal_set") for c, d in docs.items() if RP.canonical({k: v for k, v in d["removal_set"].items() if k != "removed_from_this_run"})
             != RP.canonical({k: v for k, v in ref["removal_set"].items() if k != "removed_from_this_run"})]
    diff += [(c, "population") for c, d in docs.items() if RP.canonical(d["population"]) != RP.canonical(ref["population"])]
    diff += [(c, "run") for c, d in docs.items() if d["config"]["run"] != c or d.get("anchor", {}).get("identical") is not True]
    if diff:
        refuse(f"the nine per-run records are not one design: {diff[:6]}")
    L_sel = ref["config"]["selected_layer"]
    # records of record for the values beside, and the R-empty anchors of the summary's own steps
    dmg_doc, dmg_sha = read_bound(DAMAGE)
    man_t3, _ = T3.manifest(T3TPR_MANIFEST)
    t3_doc, _ = read_bound(T3TPR, man_t3)
    hcs_doc, _ = read_bound(HC_SUMMARY)
    man_hc, _ = T3.manifest(HC_MANIFEST)
    man_rep, _ = T3.manifest(REPAIR_MANIFEST)
    rec_z, rec_rep = {}, {}
    for core in RUNS:
        hd, _ = read_bound(RESULTS / f"healthcheck_d156_{core}.json", man_hc)
        rec_z[core], hz = npz_bound(hd["draws_npz"]["file"], man_hc, hd["draws_npz"]["sha256"])
        inputs[hd["draws_npz"]["file"]] = hz
        rd, _ = read_bound(RESULTS / f"repair_d156_{core}.json", man_rep)
        rec_rep[core] = rd
    # the per-run records read these same records of record
    for core, d in docs.items():
        for key in ("damage", "table3_tpr", "repair", "healthcheck", "healthcheck_npz"):
            r_ = d["records_of_record"][key]
            if sha256_file(ROOT / r_["file"]) != r_["sha256"]:
                refuse(f"{core}'s record of record {r_['file']} has changed since its run")
    bad: list[str] = []
    # (a) between arms: the R-empty arrays reproduce damage_d156_r2's rows exactly
    anc = between_arms(zs, "anchor|")
    for lvl, key in ((LEVELS[0], "between_arms_4a"), (LEVELS[1], "between_arms_4a_cluster_level_D164")):
        want = [{k: r[k] for k in ("probe_set", "safety_arm", "safety_run", "general_run", "dose", "log_damage_gap", "delta_tpr_gap")}
                for r in dmg_doc[key]]
        if RP.canonical(anc[lvl]) != RP.canonical(want):
            bad.append(f"between-arm gaps ({lvl}) are not {rel(DAMAGE)}'s")
    # (b) the delta-TPR gap's combined interval: the R-empty replicates reproduce the supplement's
    anc_t = tpr_gaps(zs, "anchor|", anc[LEVELS[0]])
    want_t = {(r["probe_set"], r["safety_run"]): r["delta_tpr_gap"] for r in t3_doc["between_arms_4a_delta_tpr_gap"]}
    for r in anc_t:
        w = want_t.get((r["probe_set"], r["safety_run"]))
        if w is None or RP.canonical(r["delta_tpr_gap"]) != RP.canonical(w):
            bad.append(f"TPR gap {r['probe_set']} {r['safety_run']} is not {rel(T3TPR)}'s")
    # (c) Table IV: the scoring of the records' arrays reproduces healthcheck_d156_summary_lora.json
    groups, rec_rows = table_iv_rows(rec_z)
    if RP.canonical(rec_rows) != RP.canonical(hcs_doc["rows"]) or RP.canonical(groups) != RP.canonical(hcs_doc["groups"]):
        bad.append(f"the Table IV scoring of the records' arrays is not {rel(HC_SUMMARY)}'s")
    # (d) the reading's record side: the repair records' per-draw arrays hash and summarise to their records
    rec_arr: dict[str, np.ndarray] = {}
    for core, rd in rec_rep.items():
        for lvl in LEVELS:
            pdv = rd["pair_per_draw_mean_abs_log_over_layers"]["selected"][lvl]["frozen"]
            m = np.asarray(pdv["values"], dtype="<f8")
            if sha256_bytes(m.tobytes()) != pdv["sha256"] or RP.canonical(P.band_summary(m[1:], m[0])) != \
                    RP.canonical(rd["pair_summary_mean_abs_log_over_layers"]["selected"][lvl]["frozen"]):
                bad.append(f"repair_d156_{core} per-draw frozen values ({lvl})")
            rec_arr[f"{core}|{lvl}"] = m
    if bad:
        refuse(f"the summary's anchors (R empty) do not reproduce the records of record: {bad[:8]}")
    print(f"summary anchors: {sum(len(v) for v in anc.values())} between-arm rows, {len(anc_t)} TPR gaps, Table IV "
          f"scoring and 9 repair per-draw arrays identical to the records", flush=True)

    # --- the sensitivity -------------------------------------------------------------------------------------------
    sens_ba = between_arms(zs, "")
    sens_t = tpr_gaps(zs, "", sens_ba[LEVELS[0]])
    _, sens_rows = table_iv_rows(zs)
    ba_out = {}
    for lvl, key in ((LEVELS[0], "between_arms_4a"), (LEVELS[1], "between_arms_4a_cluster_level_D164")):
        recs = {(r["probe_set"], r["safety_run"]): r for r in dmg_doc[key]}
        ba_out[lvl] = [{**{k: s[k] for k in ("probe_set", "safety_arm", "safety_run", "general_run", "dose")},
                        "under_check_arm": s["safety_arm"] == HH,
                        **{q: {"sensitivity": s[q], "record": recs[(s["probe_set"], s["safety_run"])][q],
                               "change_of_mean": change(s[q]["mean"], recs[(s["probe_set"], s["safety_run"])][q]["mean"])}
                           for q in ("log_damage_gap", "delta_tpr_gap")}}
                       for s in sens_ba[lvl]]
    t_out = [{**{k: s[k] for k in ("probe_set", "safety_arm", "safety_run", "general_run", "dose")},
              "sensitivity": s["delta_tpr_gap"], "record": want_t[(s["probe_set"], s["safety_run"])],
              "note": "D-356 (2)(a): in the supplement only, as the record's"} for s in sens_t]
    t4 = table_iv_beside(sens_rows, hcs_doc["rows"])
    sens_arr = {f"{core}|{lvl}": zs[core][f"repair|selected|{lvl}|frozen"] for core in RUNS for lvl in LEVELS}
    table_iii = [{"run": core, "arm": ARM_OF[core], "dose": docs[core]["config"]["dose"], "under_check_arm": ARM_OF[core] == HH,
                  "cells": docs[core]["table_iii_row"]["cells"]} for core in RUNS]
    inputs.update({str(rel(RUN_MANIFEST)): man_runs_sha})
    payload = {**provenance_block("hh_sensitivity_d363_summary"),
               "preregistration": ref["preregistration"], "under_check": UNDER_CHECK, "code": code0,
               "reused_files_sha256": {f: sha256_file(CODE_ROOT / f) for f in REUSED_SHA256},
               "registered_parameters": registered, "blas_threads": threads, "elapsed_s": round(time.time() - t_start, 1),
               "config": {k: ref["config"][k] for k in same_keys}, "config_hash": canonical_hash({k: ref["config"][k] for k in same_keys}),
               "removal_set": {k: v for k, v in ref["removal_set"].items() if k != "removed_from_this_run"},
               "removed_per_run": {c: d["removal_set"]["removed_from_this_run"] for c, d in docs.items()},
               "population": ref["population"],
               "anchors": {"per_run": {c: {k: d["anchor"][k] for k in ("n_comparisons", "by_kind", "layers")} for c, d in docs.items()},
                           "every_layer": {"file": anchor_rel, **{k: adoc["anchor"][k] for k in ("n_comparisons", "by_kind", "layers")}},
                           "between_arms": f"{sum(len(v) for v in anc.values())} rows equal {rel(DAMAGE)}'s",
                           "tpr_gaps": f"{len(anc_t)} rows equal {rel(T3TPR)}'s",
                           "table_iv": f"the scoring of the nine records' arrays equals {rel(HC_SUMMARY)}'s rows",
                           "reading_record_side": "the nine repair records' per-draw frozen arrays hash to their records"},
               "input_files_sha256": {**dict(_RESULT_READS), **inputs},
               "table_iii": {"note": "per run, selected C; item level primary, the D-164 cluster level beside; every cell "
                                     "with its value of record and the change (sensitivity minus record)",
                             "rows": table_iii},
               "between_arms_4a": ba_out, "between_arms_4a_delta_tpr_gap_combined": t_out,
               "table_iv_lora": {"note": "Table IV's printed LoRA cells (selected C, item level, two-fold change): pooled "
                                         "sensitivity and specificity from mean counts with the run-cluster interval",
                                 "cells": t4},
               "table_iv_lora_rows_all": {"groups": groups, "sensitivity": sens_rows, "record": hcs_doc["rows"],
                                          "note": "every probe set and level, the record's rows beside"},
               "reading": reading(sens_arr, rec_arr)}
    digest = write_results(out_path, payload, exclusive=True)
    print(f"Written: {rel(out_path)}\nsha256:  {digest}")
    for dose, v in payload["reading"]["verdict_hh_rows_item_level"].items():
        print(f"  dose {dose}: HH rows {v}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="D-363: the HH contamination sensitivity (Table III, Table IV's LoRA rows)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", choices=RUNS, help="one of the nine LoRA runs (a per-run job)")
    g.add_argument("--summary", action="store_true", help="the summary from the nine per-run records")
    ap.add_argument("--anchor-only", action="store_true", help="R empty: compare with the records of record only")
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="TESTING: a subset of layers (L* always added)")
    ap.add_argument("--n-draws", type=int, default=REGISTERED["n_draws"])
    ap.add_argument("--n-boot", type=int, default=REGISTERED["n_boot"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.summary and (args.anchor_only or args.layers is not None):
        refuse("--summary takes no --anchor-only or --layers")
    return summary_job(args) if args.summary else run_job(args)


if __name__ == "__main__":
    raise SystemExit(main())
