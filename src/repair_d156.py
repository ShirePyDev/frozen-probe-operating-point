"""Repair arms under D-156 (the author's items 6, 7, 8 = D-169 items 5-7; D-228, D-229, D-230).

One script, three families, sharing src/depth_d156.py's machinery (inputs through the registry, the D-117
probes, the D-156 draws at both levels, every layer of the pair's grid):

  moment         frozen, mean_only, restandardise, score_affine, score_affine_benign, paired_align,
                 align_restd, align_affine, self_align_PRE (a no-update control), refit_oracle
  moment_ext     the same arms on the nine LoRA runs (D-169 item 10, the repair part; D-237 (3), D-238), each with
                 the selection and registry run its depth_d156 file used; writes results/moment_d156_<run>.json
  repair         frozen, restandardise, quantile_thr (cal_d negatives plus H, the old threshold_calib meaning),
                 quantile_thr_benign, refit_oracle on the nine released pairs, the nine LoRA runs and the GPU-change
                 control (D-169 item 13, part (A); D-243); writes results/repair_d156_<core>.json
  alignvar       frozen, restandardise, ridge, procrustes, coral, refit_oracle; and the AUROC after
                 mapping on the eval split with the D-152 P5 item bootstrap
  nonparametric  the similarity layer (max cosine to 256 harmful exemplars per seed, drawn from probe_train
                 positives and re-embedded by each model): frozen_thr, quantile_thr, quantile_thr_benign,
                 zscore_thr, zscore_thr_benign, oracle_recal

RULES (D-228). FEATURE-level fits use probe_train only: restandardisation statistics come from the post
model's probe_train fit rows of each seed; the maps (post -> pre) are fitted on probe_train plus H, the 1000
threshold_calib HARMFUL items, which no D-156 draw ever evaluates on (disjoint from eval by the split
manifests). Ridge lambda_rel is chosen from LAMBDA_GRID by 5-fold cross-validated MSE inside probe_train,
then refitted on all pairs (D-229). SCORE- and THRESHOLD-level calibration uses each draw's own calibration
part cal_d: T_d is each seed's threshold at 1% on the PRE scores of cal_d (the frozen threshold); score_affine
matches post-score mean and sd to pre on cal_d negatives plus H (its original meaning), score_affine_benign
on cal_d only; refit_oracle and oracle_recal set their own threshold on their post scores of cal_d.

REPORTED per family, pair, layer, probe set (moment, alignvar) and level: per arm the D-156 band of FPR post
(floored at 1/n), paired ratio against the same draw's pre FPR, ratio to target, the floor, TPR pre and post,
delta TPR and |log(FPR_post / target)| (the Table 2 metric), with the frozen split as one draw and the share
of draws floored; per pair and arm, the per-draw mean over layers of |log| with its band.
CHECKS. The frozen arm (moment, alignvar) is computed by paired.repartition_estimate and must equal the
pair's results/depth_d156_<pair>.json operating point exactly at every layer, probe set and level, and its
per-draw |log| array must hash to that file's per_draw_sha256 (D-183 (4)). The generic per-draw code path
must reproduce repartition_estimate's arrays bit for bit on the frozen arm. self_align_PRE is reported with
whether its paired-ratio band includes 1 (a no-update control). D-089: no "share" is formed here.
D-235 (2), (8): the pair summary carries the NO-UPDATE FLOOR ROW, 'no_update_floor': the same metric with
FPR_pre, the pre model's eval FPR at each draw's own frozen threshold, i.e. |log(floor_ratio_to_target)|, checked
against paired.repartition_estimate's floor_abs_log draw by draw. Every arm's and the floor's per-draw mean over
layers (index 0 the frozen split) is written with its sha256, so contrasts can be paired within draws. The
moment family is written under _r2 (results/moment_d156_<pair>_r2.json); where the earlier moment record exists,
the re-run refuses to write unless its cells and pair summary equal that record's exactly, and it names it as
superseded.
D-256 (1): the non-parametric family is written under _r2 (results/np_d156_<pair>_r2.json) with the F-070 quantile
transfer; it refuses to write unless every arm but quantile_thr and quantile_thr_benign equals D-232's record exactly.
D-257 (10): --d117-layer (moment, moment_ext) computes the pair's D-117 selected layer only and writes
results/moment_d156_<core>_d117.json with every arm's and the floor's per-draw |log(FPR/target)| at that layer; it
refuses to write unless its cells there equal, byte for byte, the record of record's (the moment _r2 record or the
moment_ext record, as named by its committed manifest). It refuses reduced parameters, so its smoke test is a full
single-layer run written outside results/.
Only registered parameters at the 6-thread BLAS pin write the final name (results/moment_d156_<pair>_r2.json,
alignvar_d156_<pair>.json, np_d156_<pair>_r2.json; moment_ext: moment_d156_<run>.json; repair: repair_d156_<core>.json;
--d117-layer: moment_d156_<core>_d117.json); tests write outside results/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_d156 as D  # noqa: E402
import input_registry as REG  # noqa: E402
import metrics as M  # noqa: E402
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
from provenance import (  # noqa: E402
    ensure_utf8_stdout, provenance_block, resolve_cache, sha256_bytes, sha256_file, write_results,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/repair_d156.py", *D.CODE_FILES]
LEVELS, PROBE_SETS, REGISTERED = D.LEVELS, D.PROBE_SETS, D.REGISTERED
LAMBDA_GRID = (1e-3, 1e-2, 1e-1, 1.0)                  # moment_decomposition.fit_alignment's grid
CV_FOLDS, CV_SEED = 5, 20260935                       # D-229
N_EXEMPLARS = 256                                     # repair_nonparametric's layer
MOMENT_ARMS = ("frozen", "mean_only", "restandardise", "score_affine", "score_affine_benign", "paired_align",
               "align_restd", "align_affine", "self_align_PRE", "refit_oracle")
FAMILIES: dict[str, dict[str, Any]] = {
    "moment": {"pairs": POP.UPDATE_PAIRS_MOMENT, "prefix": "moment_", "out": "moment_d156_", "rev": "_r2",   # D-235 (8)
               "arms": MOMENT_ARMS},
    # D-169 item 10, the repair part (D-228; D-237 (3); D-238): the moment arms on the nine LoRA runs, each anchored
    # to its own depth_d156 file (D-224) and read with the selection that file used. (Item 11's repair part,
    # repair_hw_v100_a100, is a repair_test record and is regenerated with the repair_* runs in 4d, D-238.)
    "moment_ext": {"pairs": [f"moment_{r}" for runs in POP.LORA_ARMS.values() for r in runs],
                   "prefix": "moment_", "out": "moment_d156_", "rev": "", "arms": MOMENT_ARMS},
    # D-169 item 13, part (A) (D-238, D-243): the repair_test records regenerated under D-156 and D-228 - the nine
    # released pairs, the nine LoRA runs and the GPU-change control; quantile_thr keeps its old meaning (the
    # calibration negatives plus the harmful calibration items H), a benign-only variant beside it
    "repair": {"pairs": [*POP.UPDATE_PAIRS_REPAIR, *[f"repair_{r}" for runs in POP.LORA_ARMS.values() for r in runs],
                         "repair_hw_v100_a100"],
               "prefix": "repair_", "out": "repair_d156_", "rev": "",
               "arms": ("frozen", "restandardise", "quantile_thr", "quantile_thr_benign", "refit_oracle")},
    "alignvar": {"pairs": POP.UPDATE_PAIRS_ALIGNVAR, "prefix": "alignvar_", "out": "alignvar_d156_", "rev": "",
                 "arms": ("frozen", "restandardise", "ridge", "procrustes", "coral", "refit_oracle")},
    # D-256 (1): re-run as _r2 under the F-070 quantile transfer; every other arm must equal D-232's record exactly
    "nonparametric": {"pairs": POP.NON_PARAMETRIC, "prefix": "repair_np_", "out": "np_d156_", "rev": "_r2",
                      "arms": ("frozen_thr", "quantile_thr", "quantile_thr_benign", "zscore_thr", "zscore_thr_benign",
                               "oracle_recal")},
}
FLOOR_ROW = "no_update_floor"                         # D-235 (2)
TARGET_5 = 0.05                                       # 4e (D-152, D-248)
# D-235 (8): the moment records written before the floor row; each final _r2 run of these pairs must reproduce its
# record, pinned here, and no other pair may have one
EARLIER_MOMENT_SHA256 = {
    "mistral_base2instruct": "02c0f5f8d9b4ba9e4a9501a5260d95f6950b96575ebc9911eb1e5132e3484b96",
    "llama3_base2instruct": "8966ed0daa1c56b25b049e1ec57d20b8619ccfb81ab37647a8f649eeccb447fc",
    "gemma2_base2instruct": "bda2f7b2dba235b3dfb630b56cec9eae12ebba319c06f32dd71c51b18b52cb7e",
}
# D-256 (1): the D-232 non-parametric records the _r2 run must reproduce in every arm but the two quantile arms (F-070)
EARLIER_NP_SHA256 = {
    "mistral_base2instruct": "7342fdeed70427f9162f1d0e7b68344d7f78ccb5e8d2ca4fc757af292111b0f1",
    "qwen_base2instruct": "cbc162adc73c2b38cf60653bef525f32c9f6c279dc797372293d3874a334974b",
}
QUANTILE_ARMS = ("quantile_thr", "quantile_thr_benign")
# D-257 (10): the records of record whose cell at the D-117 layer a --d117-layer run must equal, and the manifests
# (committed) that name them
D117_RECORD_OF_RECORD = {"moment": ("results/moment_d156_{core}_r2.json", "logs/moment_d156_r2.sha256"),
                         "moment_ext": ("results/moment_d156_{core}.json", "logs/moment_ext_d156_20260928.sha256"),
                         # D-271 (4), D-273: the 5% D-117-layer mode (the moment family only, as 4e's 5% arm)
                         "moment|fpr5": ("results/moment_d156_{core}_r2_fpr5.json", "logs/moment_d156_r2_fpr5.sha256")}
OWN_THRESHOLD = {"refit_oracle", "oracle_recal"}      # threshold at 1% on the arm's own post scores of cal_d
PER_DRAW = {"score_affine", "score_affine_benign", "align_affine", "quantile_thr", "quantile_thr_benign",
            "zscore_thr", "zscore_thr_benign"}        # a transform estimated on each draw's calibration part
WITH_H = {"score_affine", "align_affine", "quantile_thr", "zscore_thr"}        # pool = cal_d negatives + H
MAPS = ("ridge", "procrustes", "coral")               # alignvar maps with an AUROC after mapping
QUANT = ("fpr_post", "paired_ratio", "ratio_to_target", "floor_ratio_to_target", "tpr_pre", "tpr_post", "delta_tpr",
         "abs_log_ratio_to_target")


def core_of(stem: str, family: str) -> str:
    c = stem[len(FAMILIES[family]["prefix"]):]
    return c[:-3] if c.endswith("_v2") else c


def canonical(x: Any) -> str:
    """The JSON text a record holds for x (write_results' encoding), keys sorted. Two blocks are equal when their
    texts are, which also holds for NaN (a NaN float never equals itself)."""
    return json.dumps(json.loads(json.dumps(x, default=str)), sort_keys=True)


def manifest_entry(man_rel: str, rel: str) -> str:
    """The sha256 a committed sha256sum manifest names for rel; refuses when it names none or several."""
    hits = []
    for line in (ROOT / man_rel).read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, f = line.split(maxsplit=1)
            if f.strip().lstrip("*") == rel:
                hits.append(h)
    if len(hits) != 1:
        raise SystemExit(f"REFUSING: {man_rel} names {rel} {len(hits)} times")
    return hits[0]


def strip_arms(x: Any, drop: set[str]) -> Any:
    """A copy of a cell list or a pair summary without the arms in drop (D-256 (1): the NP quantile arms)."""
    if not drop:
        return x
    if isinstance(x, list):
        return [{k: ({a: v for a, v in c[k].items() if a not in drop} if k in LEVELS else c[k]) for k in c} for c in x]
    return {ps: {lv: {a: v for a, v in blk.items() if a not in drop} for lv, blk in lvs.items()} for ps, lvs in x.items()}


# ---------------------------------------------------------------- maps (post -> pre), float64
def _centre(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = x.mean(0)
    return x - m, m


def ridge_cv(src_tr: np.ndarray, tgt_tr: np.ndarray, src_all: np.ndarray, tgt_all: np.ndarray
             ) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """Affine ridge map src -> tgt in moment_decomposition.fit_alignment's form (lambda relative to
    trace(gram)/d). lambda_rel by 5-fold CV MSE INSIDE probe_train (src_tr/tgt_tr); refit on all pairs."""
    n = len(src_tr)
    folds = np.array_split(np.random.default_rng(CV_SEED).permutation(n), CV_FOLDS)
    mse = {lam: 0.0 for lam in LAMBDA_GRID}
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        xs, ms = _centre(src_tr[tr])
        ys, mt = _centre(tgt_tr[tr])
        # dual form (n_fit < d): A = X^T (X X^T + mu I)^-1 Y, mu = lambda_rel * trace(X^T X) / d
        K = xs @ xs.T
        e, U = np.linalg.eigh(K)
        scale = float(np.trace(K)) / xs.shape[1]
        KvU = ((src_tr[f] - ms) @ xs.T) @ U
        UtY = U.T @ ys
        for lam in LAMBDA_GRID:
            pred = KvU @ (UtY / (e + lam * scale)[:, None]) + mt
            mse[lam] += float(((pred - tgt_tr[f]) ** 2).sum())
    lam = min(LAMBDA_GRID, key=lambda v: (mse[v], v))
    xs, ms = _centre(src_all)
    ys, mt = _centre(tgt_all)
    gram = xs.T @ xs
    scale = float(np.trace(gram)) / gram.shape[0]
    A = np.linalg.solve(gram + lam * scale * np.eye(gram.shape[0]), xs.T @ ys)
    return (lambda x: (x - ms) @ A + mt), {"lambda_rel_chosen": lam, "cv_mse_per_lambda": {str(k): v / n for k, v in mse.items()},
                                          "n_pairs_fit": int(len(src_all)), "n_pairs_cv": int(n)}


def procrustes(src: np.ndarray, tgt: np.ndarray) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """Orthogonal Procrustes src -> tgt, centred (align_variants.fit_procrustes)."""
    xs, ms = _centre(src)
    ys, mt = _centre(tgt)
    u, _, vt = np.linalg.svd(xs.T @ ys, full_matrices=False)
    r = u @ vt
    return (lambda x: (x - ms) @ r + mt), {"max_orthogonality_error": float(np.abs(r @ r.T - np.eye(r.shape[0])).max()),
                                          "n_pairs_fit": int(len(src))}


def coral(src: np.ndarray, tgt: np.ndarray) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """Whiten with the post covariance, colour with the pre covariance (Ledoit-Wolf), then the pre mean
    (align_variants.fit_coral), both estimated on the same fitting items."""
    from sklearn.covariance import ledoit_wolf

    def shrunk(x):
        c, lam = ledoit_wolf(x, assume_centered=False)
        return c + 1e-6 * np.eye(c.shape[0]), float(lam)

    def mpow(c, p):
        w, v = np.linalg.eigh(c)
        return (v * np.maximum(w, 1e-10) ** p) @ v.T

    cp, lp = shrunk(src)
    cq, lq = shrunk(tgt)
    A = mpow(cp, -0.5) @ mpow(cq, 0.5)
    mp, mq = src.mean(0), tgt.mean(0)
    return (lambda x: (x - mp) @ A + mq), {"shrinkage_post": lp, "shrinkage_pre": lq, "n_items_fit": int(len(src))}


def sim_scores(x: np.ndarray, ex: np.ndarray) -> np.ndarray:
    """Max cosine similarity to any exemplar (repair_nonparametric.sim_scores)."""
    xn = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    en = ex / np.clip(np.linalg.norm(ex, axis=1, keepdims=True), 1e-12, None)
    return (xn @ en.T).max(axis=1)


# ---------------------------------------------------------------- one partition, every arm
def thresholds(S: np.ndarray, target: float) -> np.ndarray:
    return np.array([P.thr_at_fpr(S[:, s], target) for s in range(S.shape[1])])


def point(F_pre: np.ndarray, Q_pre: np.ndarray, F: np.ndarray, Q: np.ndarray, target: float) -> dict[str, float]:
    """The operating-point numbers of paired.operating_point, from flag matrices (items, seeds)."""
    f_pre, f = P.floored_fpr(F_pre), P.floored_fpr(F)
    return {"fpr_post": float(f.mean()), "paired_ratio": P.ratio_from_flags(F_pre, F),
            "ratio_to_target": float(f.mean() / target), "floor_ratio_to_target": float(f_pre.mean() / target),
            "tpr_pre": float(P.rate(Q_pre).mean()), "tpr_post": float(P.rate(Q).mean()),
            "delta_tpr": P.delta_tpr_from_flags(Q_pre, Q),
            "abs_log_ratio_to_target": float(P.abs_log(f.mean() / target)),
            "floored": bool(((F_pre.sum(axis=0) == 0) | (F.sum(axis=0) == 0)).any())}


def affine(post: np.ndarray, post_ref: np.ndarray, pre_ref: np.ndarray) -> np.ndarray:
    """Per seed, map post scores so their mean and sd on the reference items equal the pre scores' there."""
    mp, sp = post_ref.mean(axis=0), post_ref.std(axis=0)
    mq, sq = pre_ref.mean(axis=0), pre_ref.std(axis=0)
    sp = np.where(sp > 0, sp, 1.0)
    sq = np.where(sq > 0, sq, 1.0)                         # as moment_decomposition.affine_to (std or 1.0)
    return (post - mp) / sp * sq + mq


def partition(cal: np.ndarray, ev: np.ndarray, pre_pool: np.ndarray, pre_pos: np.ndarray, pre_H: np.ndarray,
              arms: dict[str, dict[str, np.ndarray]], target: float) -> dict[str, dict[str, float]]:
    """Every arm at ONE partition. arms[name] holds its post scores: 'pool', 'pos', and 'H' for the
    per-draw transforms. T_d = each seed's threshold on the PRE scores of cal_d (the frozen threshold)."""
    T = thresholds(pre_pool[cal], target)
    F_pre, Q_pre = pre_pool[ev] >= T, pre_pos >= T
    out = {}
    for name, a in arms.items():
        if name in OWN_THRESHOLD:
            Ta = thresholds(a["pool"][cal], target)
            F, Q = a["pool"][ev] >= Ta, a["pos"] >= Ta
        elif name in PER_DRAW:
            post_ref = np.vstack([a["pool"][cal], a["H"]]) if name in WITH_H else a["pool"][cal]
            pre_ref = np.vstack([pre_pool[cal], pre_H]) if name in WITH_H else pre_pool[cal]
            if name.startswith("quantile"):
                # F-070: the exact rank transfer (returns T_d with no update); np.quantile at the share below T_d
                # landed about one order statistic below it
                tq = np.array([P.quantile_transfer(pre_ref[:, s], post_ref[:, s], T[s]) for s in range(post_ref.shape[1])])
                F, Q = a["pool"][ev] >= tq, a["pos"] >= tq
            elif name.startswith("zscore"):
                mq, sq = pre_ref.mean(axis=0), pre_ref.std(axis=0)
                mp, sp = post_ref.mean(axis=0), post_ref.std(axis=0)
                tz = (T - mq) / np.where(sq > 0, sq, 1.0)
                sp = np.where(sp > 0, sp, 1.0)
                F, Q = (a["pool"][ev] - mp) / sp >= tz, (a["pos"] - mp) / sp >= tz
            else:                                                  # score_affine family
                F = affine(a["pool"][ev], post_ref, pre_ref) >= T
                Q = affine(a["pos"], post_ref, pre_ref) >= T
        else:
            F, Q = a["pool"][ev] >= T, a["pos"] >= T
        out[name] = point(F_pre, Q_pre, F, Q, target)
    return out


def summarise(per: list[dict[str, dict[str, float]]], arm: str) -> dict[str, Any]:
    blk = {q: P.band_summary(np.array([p[arm][q] for p in per[1:]]), per[0][arm][q]) for q in QUANT}
    blk["share_of_draws_floored"] = float(np.mean([p[arm]["floored"] for p in per[1:]]))
    blk["frozen_split_floored"] = bool(per[0][arm]["floored"])
    return blk


# ---------------------------------------------------------------- main
def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Repair arms under D-156 (D-228, D-229, D-230)")
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--pair", required=True)
    ap.add_argument("--n-draws", type=int, default=REGISTERED["n_draws"])
    ap.add_argument("--target-fpr", type=float, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=REGISTERED["seeds"])
    ap.add_argument("--n-boot", type=int, default=REGISTERED["n_boot"])
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="TESTING: a subset of layers")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reproduce", default=None,
                    help="TESTING: refuse to write unless the cells and pair summary equal this earlier output's "
                         "(a final _r2 run compares with the earlier record by itself, D-235 (8))")
    ap.add_argument("--fpr5", action="store_true",
                    help="4e (D-152, D-248): the moment family's registered 5%% arm, written as <name>_fpr5; anchored to "
                         "the 5%% depth file, and the frozen arm's 1%% twin must equal the 1%% depth record exactly")
    ap.add_argument("--d117-layer", action="store_true",
                    help="D-257 (10): the registered D-117-layer mode (moment, moment_ext): the pair's selected layer only, "
                         "written as moment_d156_<core>_d117.json with every arm's and the floor's per-draw array there; "
                         "its cells must equal the record of record's cells at that layer")
    args = ap.parse_args()
    if args.d117_layer and (args.family not in D117_RECORD_OF_RECORD or (args.fpr5 and args.family != "moment")
                            or args.layers or args.reproduce):
        raise SystemExit("REFUSING: --d117-layer is for the moment and moment_ext families (at 5% the moment family only, "
                         "D-273), without --layers or --reproduce (D-257)")
    rr_key = f"{args.family}|fpr5" if args.fpr5 else args.family      # the record of record's key (D-257 (10), D-273)
    if args.fpr5 and args.family != "moment":
        raise SystemExit("REFUSING: the 5% arm covers the moment family (tab:repair and its floor, D-248)")
    if args.fpr5 and (args.target_fpr not in (None, TARGET_5) or args.reproduce):
        raise SystemExit("REFUSING: --fpr5 fixes the target at 0.05 and takes no --reproduce")
    args.target_fpr = TARGET_5 if args.fpr5 else (REGISTERED["target_fpr"] if args.target_fpr is None else args.target_fpr)
    fam = FAMILIES[args.family]
    if args.pair not in fam["pairs"]:
        raise SystemExit(f"REFUSING: {args.pair} is not in the {args.family} population {fam['pairs']}")
    core = core_of(args.pair, args.family)
    # the registry run and the selection of record, as depth_d156 reads them for this pair's depth file (D-224):
    # the GPU control has no repair run in the registry, and the LoRA arms take the LoRA A100 selection
    reg_stem = D.registry_stem(f"depth_{core}")
    sel_path = D.SELECTION_OF_RECORD_LORA if f"depth_{core}" in D.LORA_DEPTH else D.SELECTION_OF_RECORD
    sel_stem = D.SELECTION_STEM.get(f"depth_{core}", reg_stem)
    probe_based = args.family != "nonparametric"
    registered = (args.target_fpr == (TARGET_5 if args.fpr5 else REGISTERED["target_fpr"]) and args.n_draws == REGISTERED["n_draws"]
                  and list(args.seeds) == REGISTERED["seeds"] and args.n_boot == REGISTERED["n_boot"] and args.layers is None)
    final_path = (RESULTS / (f"{fam['out']}{core}_d117{'_fpr5' if args.fpr5 else ''}.json" if args.d117_layer else
                             f"{fam['out']}{core}{fam['rev']}{'_fpr5' if args.fpr5 else ''}.json")).resolve()
    out_path = Path(args.out).resolve() if args.out else final_path
    final = out_path == final_path
    if out_path.exists():
        raise SystemExit(f"REFUSING: {out_path} exists; results are never overwritten")
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test run (--out) must not write under results/")
    threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    if final and (not registered or any(t != "6" for t in threads.values())):
        raise SystemExit(f"REFUSING the final name: registered parameters {registered}, threads {threads}")
    # D-235 (8): the earlier record a final _r2 run must reproduce; a test names its own with --reproduce
    if final and args.reproduce:
        raise SystemExit("REFUSING: --reproduce is for tests; a final run compares with the earlier record by itself")
    prior_path: Path | None = None
    if args.reproduce:
        prior_path = Path(args.reproduce).resolve()
        if not prior_path.is_file():
            raise SystemExit(f"REFUSING: --reproduce {prior_path} does not exist")
    elif final and fam["rev"] and not args.fpr5 and not args.d117_layer:   # a 5% or D-117-layer run re-runs no record by name
        cand = RESULTS / f"{fam['out']}{core}.json"
        pins = {"moment": EARLIER_MOMENT_SHA256, "nonparametric": EARLIER_NP_SHA256}.get(args.family, {})
        pinned = pins.get(core)
        if pinned and not cand.is_file():
            raise SystemExit(f"REFUSING: the earlier record {cand.name} this run must reproduce is missing (D-235 (8), D-256 (1))")
        if not pinned and cand.exists():
            raise SystemExit(f"REFUSING: {cand.name} exists but is not a pinned earlier record (D-235 (8), D-256 (1))")
        prior_path = cand if pinned else None
    old: dict[str, Any] | None = None
    prior_sha = None
    if prior_path:
        pb = prior_path.read_bytes()                         # read once: the bytes hashed are the bytes compared
        prior_sha = sha256_bytes(pb)
        pins = {"moment": EARLIER_MOMENT_SHA256, "nonparametric": EARLIER_NP_SHA256}.get(args.family, {})
        if not args.reproduce and prior_sha != pins.get(core):
            raise SystemExit(f"REFUSING: {prior_path.name} is not the pinned earlier record (sha256 {prior_sha})")
        old = json.loads(pb.decode("utf-8"))
        if not all(k in old for k in ("cells", "pair_summary_mean_abs_log_over_layers", "config", "code",
                                      "input_cache_sha256", "input_verification")):
            raise SystemExit(f"REFUSING: {prior_path.name} is not a repair_d156 record")
        if (old["config"].get("family"), old["config"].get("pair")) != (args.family, args.pair) or \
                (not args.reproduce and old.get("registered_parameters") is not True):
            raise SystemExit(f"REFUSING: {prior_path.name} is not the {'test' if args.reproduce else 'registered'} "
                             f"{args.family} run of {args.pair}")
    code0 = P.code_block(CODE_FILES)
    sel_sha, cl_sha = sha256_file(sel_path), sha256_file(D.CLUSTERS)

    # the pair's depth file of record: the frozen arm's anchor, and the selection it used
    depth_rel = f"results/depth_d156_{core}{'_fpr5' if args.fpr5 else ''}.json"
    depth_sha = sha256_file(ROOT / depth_rel) if probe_based else None
    ddoc = json.load(open(ROOT / depth_rel, encoding="utf-8")) if probe_based else None
    dcells1: dict[tuple[int, str], dict[str, Any]] = {}
    depth1_rel, depth1_sha = f"results/depth_d156_{core}.json", None
    if args.fpr5:
        # D-248: the 5% depth file must itself be complete and self-anchored; the 1% depth record anchors the 1% twin
        if (ddoc.get("fpr5_self_anchor") or {}).get("n_cells_identical_at_1pct") != len(ddoc["cells"]):
            raise SystemExit(f"REFUSING: {depth_rel} is not a complete, self-anchored 5% depth file (D-248)")
        d1b = (ROOT / depth1_rel).read_bytes()
        depth1_sha = sha256_bytes(d1b)
        if depth1_sha != ddoc["fpr5_self_anchor"]["sha256"]:
            raise SystemExit(f"REFUSING: {depth1_rel} is not the 1% record {depth_rel} was anchored to")
        dcells1 = {(c["layer"], c["probe_set"]): c for c in json.loads(d1b.decode("utf-8"))["cells"]}
    n_self_anchor = 0
    if probe_based:
        sel_doc = json.load(open(sel_path, encoding="utf-8"))
        sel = next((m for m in sel_doc["models"] if m["pair"] == sel_stem), None)
        if sel is None or not sel["selected"]["arm"].startswith("C="):
            raise SystemExit(f"REFUSING: no logistic-C selection for {sel_stem} in {sel_path.name}")
        C_sel, L_sel = float(sel["selected"]["arm"].split("=")[1]), int(sel["selected"]["layer"])
        if (ddoc["config"]["selected_C"], ddoc["config"]["selected_layer"], ddoc["config"]["selection_sha256"]) != (C_sel, L_sel, sel_sha):
            raise SystemExit(f"REFUSING: {depth_rel} was run on another selection")
        # the registry run and selection this run resolved must be the ones the depth file records (D-238)
        dc_ = ddoc["config"]
        for k_, want_ in (("registry_pair", reg_stem), ("selection_of_pair", sel_stem),
                          ("selection_file", str(sel_path.relative_to(ROOT)).replace("\\", "/"))):
            if k_ in dc_ and dc_[k_] != want_:
                raise SystemExit(f"REFUSING: {depth_rel} records {k_} {dc_[k_]!r}, this run resolved {want_!r}")
        dcells = {(c["layer"], c["probe_set"]): c for c in ddoc["cells"]}
    else:
        C_sel = L_sel = None
        dcells = {}
    # D-257 (10): the record of record whose cells at L_sel this run must equal, named by its committed manifest
    rr_rel = rr_sha = None
    rr_doc: dict[str, Any] | None = None
    if args.d117_layer:
        rr_rel, rr_man = (t.format(core=core) for t in D117_RECORD_OF_RECORD[rr_key])
        rrb = (ROOT / rr_rel).read_bytes()
        rr_sha = sha256_bytes(rrb)
        if rr_sha != manifest_entry(rr_man, rr_rel):
            raise SystemExit(f"REFUSING: {rr_rel} is not the file {rr_man} names")
        rr_doc = json.loads(rrb.decode("utf-8"))
        if (rr_doc["config"].get("family"), rr_doc["config"].get("pair"), rr_doc["config"].get("selected_layer"),
                rr_doc.get("registered_parameters")) != (args.family, args.pair, L_sel, True):
            raise SystemExit(f"REFUSING: {rr_rel} is not the registered {args.family} run of {args.pair} at L{L_sel}")
        # refuse now, not after the layer's compute: the record of record must share this run's parameters and anchor
        rc_ = rr_doc["config"]
        ax_ = rc_.get("axis")
        early = [k for k, a, b in (("selected_C", rc_.get("selected_C"), C_sel), ("selection_sha256", rc_.get("selection_sha256"), sel_sha),
                                   ("seeds", rc_.get("seeds"), list(args.seeds)), ("n_draws", rc_.get("n_draws"), args.n_draws),
                                   ("target_fpr", rc_.get("target_fpr"), args.target_fpr),
                                   ("draw_seed", rc_.get("draw_seed"), P.AXIS_DRAW_SEED.get(ax_)),
                                   ("cluster_draw_seed", rc_.get("cluster_draw_seed"), P.AXIS_CLUSTER_DRAW_SEED.get(ax_)),
                                   ("cluster_manifest", (rc_.get("cluster_manifest") or {}).get("sha256"), cl_sha),
                                   ("anchor", (rr_doc.get("anchor") or {}).get("sha256"), depth_sha)) if a != b]
        n_at_L = sum(c["layer"] == L_sel for c in rr_doc["cells"])
        if early or n_at_L != len(PROBE_SETS):
            raise SystemExit(f"REFUSING: {rr_rel} differs from this run in {early} or holds {n_at_L} cells at L{L_sel} (D-257 (10))")

    cfg = REG.run(reg_stem)
    axis = cfg["axis"]
    data: dict[str, dict[str, Any]] = {"pre": {}, "post": {}}
    input_sha: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    for side in data:
        for part in ("probe_train", "threshold_calib", "eval"):
            z, r = REG.load(cfg[f"{side}_repo"], axis, part)
            data[side][part] = z
            input_sha[f"{side}/{part}"], input_recs[f"{side}/{part}"] = r["sha256"], r
    reg_sources = REG.sources_sha256()
    if D.sources_now() != reg_sources:
        raise SystemExit("REFUSING: the extraction records or pins on disk are not the ones the registry read")
    if probe_based and any(ddoc["input_cache_sha256"].get(k) != h for k, h in input_sha.items()):
        raise SystemExit(f"REFUSING: {depth_rel} read other caches")
    verification = REG.verification(input_recs)
    for part in ("probe_train", "threshold_calib", "eval"):
        a, b = data["pre"][part], data["post"][part]
        if not np.array_equal(a["item_ids"], b["item_ids"]) or not np.array_equal(a["labels"], b["labels"]):
            raise SystemExit(f"REFUSING: {part} items or labels differ between pre and post")
        for z in (a, b):
            if "nonfinite" in z.files and z["nonfinite"].any():
                raise SystemExit(f"REFUSING: {part} has non-finite items")
    grids = [sorted(int(k.split("_")[1]) for k in z.files if k.startswith("layer_")) for s in data for z in data[s].values()]
    if any(g != grids[0] for g in grids):
        raise SystemExit("REFUSING: the input caches have different layer grids")
    layers = [li for li in grids[0] if args.layers is None or li in args.layers]
    if args.d117_layer:
        if L_sel not in grids[0]:
            raise SystemExit(f"REFUSING: the D-117 layer {L_sel} is not in the caches' grid {grids[0]}")
        layers = [L_sel]
        mism = [k for k, a, b in (("input caches", rr_doc["input_cache_sha256"], input_sha),
                                  ("input_verification", rr_doc["input_verification"], verification),
                                  ("numeric libraries", rr_doc["code"].get("libraries"), code0.get("libraries")))
                if canonical(a) != canonical(b)]
        if mism:
            raise SystemExit(f"REFUSING: this run cannot equal {rr_rel} at L{L_sel}: it differs in {mism} (D-257)")
    if old is not None:
        # refuse now, not after hours: the earlier record must have read these inputs, under this flag state,
        # over these layers, with these numeric libraries
        mism = [k for k, a, b in (("input caches", old["input_cache_sha256"], input_sha),
                                  ("input_verification", old["input_verification"], verification),
                                  ("layers", old["config"].get("layers"), layers),
                                  ("numeric libraries", old["code"].get("libraries"), code0.get("libraries")))
                if canonical(a) != canonical(b)]
        if mism:
            raise SystemExit(f"REFUSING: this run cannot reproduce {prior_path.name}: it differs in {mism} (D-235 (8))")

    y_tr = data["pre"]["probe_train"]["labels"].astype(bool)
    y_cal = data["pre"]["threshold_calib"]["labels"].astype(bool)
    ncal, hcal = ~y_cal, y_cal
    y_ev = data["pre"]["eval"]["labels"].astype(bool)
    neg, pos = ~y_ev, y_ev
    pooled_ids = [str(i) for i in data["pre"]["threshold_calib"]["item_ids"][ncal]] + \
                 [str(i) for i in data["pre"]["eval"]["item_ids"][neg]]
    n_cal, n_pool = int(ncal.sum()), len(pooled_ids)
    cl = json.load(open(D.CLUSTERS, encoding="utf-8"))["axes"][axis]
    if cl["n_pooled_negatives"] != n_pool or cl["n_calibration"] != n_cal or \
            cl["pool_item_ids_sha256"] != sha256_bytes("\n".join(pooled_ids).encode("utf-8")):
        raise SystemExit(f"REFUSING: manifests/negative_clusters.json does not describe the {axis} pool")
    H_ids = {str(i) for i in data["pre"]["threshold_calib"]["item_ids"][hcal]}
    ev_ids = {str(i) for i in data["pre"]["eval"]["item_ids"]}
    if H_ids & set(pooled_ids) or H_ids & ev_ids:
        raise SystemExit("REFUSING: a threshold_calib harmful item is in the pool or in eval (D-228)")
    tr_ids = {str(i) for i in data["pre"]["probe_train"]["item_ids"]}
    if tr_ids & set(pooled_ids) or tr_ids & ev_ids:
        raise SystemExit("REFUSING: a probe_train item is in the pool or in eval (D-228: feature-level fits use probe_train only)")
    frozen_part = (np.arange(n_cal), np.arange(n_cal, n_pool))
    draws = {LEVELS[0]: P.repartition_draws(n_cal, n_pool - n_cal, args.n_draws, P.AXIS_DRAW_SEED[axis]),
             LEVELS[1]: P.cluster_repartition_draws(P.cluster_units(pooled_ids, cl["clusters"]), n_cal,
                                                    args.n_draws, P.AXIS_CLUSTER_DRAW_SEED[axis])}
    parts = {lvl: [frozen_part] + [(d[:n_cal], d[n_cal:]) for d in dr] for lvl, dr in draws.items()}
    n_neg, n_pos = int(neg.sum()), int(pos.sum())
    CNT_N = P.counts_matrix(n_neg, args.n_boot, D.ITEM_BOOT_SEED["eval_negatives"])
    CNT_P = P.counts_matrix(n_pos, args.n_boot, D.ITEM_BOOT_SEED["eval_positives"])
    print(f"{args.family} {args.pair} via {reg_stem} [{axis}]: layers {layers}; arms {fam['arms']}; "
          f"pooled {n_cal} + {n_pool - n_cal}; H {int(hcal.sum())}; D-170 flag {verification['flag']}; registered {registered}",
          flush=True)

    cells: list[dict[str, Any]] = []
    per_layer_abslog: dict[tuple[str, str, str], list[np.ndarray]] = {}    # (probe_set, level, arm) -> per layer (1+n_draws,)
    d117_curves: dict[tuple[str, str], str] = {}                           # D-257 (10): the depth file's frozen-curve sha256
    n_anchor = n_generic = 0
    for li in layers:
        k = f"layer_{li}"
        X = {s: {p: data[s][p][k] for p in ("probe_train", "threshold_calib", "eval")} for s in data}
        cal_pre_x, ev_pre_x = X["pre"]["threshold_calib"], X["pre"]["eval"]
        cal_post_x, ev_post_x = X["post"]["threshold_calib"], X["post"]["eval"]
        map_info: dict[str, Any] = {}
        maps: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
        if probe_based:
            f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
            src_tr, tgt_tr = f64(X["post"]["probe_train"]), f64(X["pre"]["probe_train"])
            src_all = np.vstack([src_tr, f64(cal_post_x[hcal])])
            tgt_all = np.vstack([tgt_tr, f64(cal_pre_x[hcal])])
            if args.family in ("moment", "moment_ext"):
                maps["paired_align"], map_info["paired_align"] = ridge_cv(src_tr, tgt_tr, src_all, tgt_all)
                pre_all = np.vstack([tgt_tr, f64(cal_pre_x[hcal])])
                maps["self_align_PRE"], map_info["self_align_PRE"] = ridge_cv(tgt_tr, tgt_tr, pre_all, pre_all)
            elif args.family == "alignvar":
                maps["ridge"], map_info["ridge"] = ridge_cv(src_tr, tgt_tr, src_all, tgt_all)
                maps["procrustes"], map_info["procrustes"] = procrustes(src_all, tgt_all)
                maps["coral"], map_info["coral"] = coral(src_all, tgt_all)
            # the family 'repair' has no map arm, so it fits no map (review of D-243)
            # each map applied ONCE per layer (it does not depend on the probe or the seed)
            MX: dict[str, dict[str, np.ndarray]] = {}
            for name, m in maps.items():
                if name == "self_align_PRE":
                    MX[name] = {"cal": m(f64(cal_pre_x)), "ev": m(f64(ev_pre_x))}
                else:
                    MX[name] = {"cal": m(f64(cal_post_x)), "ev": m(f64(ev_post_x))}
                    if name == "paired_align":                       # align_restd's statistics
                        MX[name]["tr"] = m(src_tr)
            probe_sets = PROBE_SETS
        else:
            probe_sets = ("similarity",)
        for pset in probe_sets:
            if probe_based:
                probes, refits, fit_rows = [], [], []
                for seed in args.seeds:
                    _, fit = P.d117_split(len(y_tr), seed)
                    rows, C, order = (fit, C_sel, "d117_permutation") if pset == "selected" else (np.sort(fit), 1.0, "file_order")
                    probes.append(P.fit_probe(X["pre"]["probe_train"][rows], y_tr[rows], seed, C, order))
                    refits.append(P.fit_probe(X["post"]["probe_train"][rows], y_tr[rows], seed, C, order))
                    fit_rows.append(rows)
                S = lambda f, x: np.column_stack([f(q, x) for q in probes])  # noqa: E731
                score = lambda q, x: q.score(x)  # noqa: E731
                pre_pool = np.vstack([S(score, cal_pre_x[ncal]), S(score, ev_pre_x[neg])])
                pre_pos, pre_H = S(score, ev_pre_x[pos]), S(score, cal_pre_x[hcal])
                post = {"pool": np.vstack([S(score, cal_post_x[ncal]), S(score, ev_post_x[neg])]),
                        "pos": S(score, ev_post_x[pos]), "H": S(score, cal_post_x[hcal])}
                arms: dict[str, dict[str, np.ndarray]] = {}
                for name in fam["arms"]:
                    if name == "frozen":
                        arms[name] = post
                    elif name in ("mean_only", "restandardise"):
                        cols_pool, cols_pos = [], []
                        for q, rows in zip(probes, fit_rows):
                            xb = X["post"]["probe_train"][rows]
                            mu_b = xb.mean(0)
                            sd_b = np.where(xb.std(0) < 1e-8, 1.0, xb.std(0)) if name == "restandardise" else q.sd
                            f = lambda x, mu=mu_b, sd=sd_b, q=q: ((x - mu) / sd) @ q.w + q.b  # noqa: E731
                            cols_pool.append(np.concatenate([f(cal_post_x[ncal]), f(ev_post_x[neg])]))
                            cols_pos.append(f(ev_post_x[pos]))
                        arms[name] = {"pool": np.column_stack(cols_pool), "pos": np.column_stack(cols_pos)}
                    elif name in ("score_affine", "score_affine_benign", "quantile_thr", "quantile_thr_benign"):
                        arms[name] = post                          # score- or threshold-level: per draw, in partition()
                    elif name in ("paired_align", "ridge", "procrustes", "coral", "align_affine"):
                        mx = MX["paired_align" if name == "align_affine" else name]
                        arms[name] = {"pool": np.vstack([S(score, mx["cal"][ncal]), S(score, mx["ev"][neg])]),
                                      "pos": S(score, mx["ev"][pos]), "H": S(score, mx["cal"][hcal])}
                    elif name == "align_restd":
                        mx = MX["paired_align"]
                        cols_pool, cols_pos = [], []
                        for q, rows in zip(probes, fit_rows):
                            xa = mx["tr"][rows]
                            mu_a, sd_a = xa.mean(0), np.where(xa.std(0) < 1e-8, 1.0, xa.std(0))
                            f = lambda x, mu=mu_a, sd=sd_a, q=q: ((x - mu) / sd) @ q.w + q.b  # noqa: E731
                            cols_pool.append(np.concatenate([f(mx["cal"][ncal]), f(mx["ev"][neg])]))
                            cols_pos.append(f(mx["ev"][pos]))
                        arms[name] = {"pool": np.column_stack(cols_pool), "pos": np.column_stack(cols_pos)}
                    elif name == "self_align_PRE":
                        mx = MX["self_align_PRE"]
                        arms[name] = {"pool": np.vstack([S(score, mx["cal"][ncal]), S(score, mx["ev"][neg])]),
                                      "pos": S(score, mx["ev"][pos])}
                    elif name == "refit_oracle":
                        R = lambda x: np.column_stack([q.score(x) for q in refits])  # noqa: E731
                        arms[name] = {"pool": np.vstack([R(cal_post_x[ncal]), R(ev_post_x[neg])]), "pos": R(ev_post_x[pos])}
            else:
                pos_idx = np.flatnonzero(y_tr)
                exs = [np.sort(np.random.default_rng(seed).choice(pos_idx, size=min(N_EXEMPLARS, pos_idx.size), replace=False))
                       for seed in args.seeds]
                SS = lambda side, x: np.column_stack([sim_scores(x, X[side]["probe_train"][e]) for e in exs])  # noqa: E731
                pre_pool = np.vstack([SS("pre", cal_pre_x[ncal]), SS("pre", ev_pre_x[neg])])
                pre_pos, pre_H = SS("pre", ev_pre_x[pos]), SS("pre", cal_pre_x[hcal])
                post = {"pool": np.vstack([SS("post", cal_post_x[ncal]), SS("post", ev_post_x[neg])]),
                        "pos": SS("post", ev_post_x[pos]), "H": SS("post", cal_post_x[hcal])}
                arms = {name: post for name in fam["arms"]}
            # every partition at both levels; per-draw arrays per arm
            cell: dict[str, Any] = {"layer": li, "probe_set": pset, "input_verification": verification,
                                    "probe_C": (C_sel if pset == "selected" else 1.0) if probe_based else None,
                                    "maps": map_info or None}
            for lvl in LEVELS:
                per = [partition(c, e, pre_pool, pre_pos, pre_H, arms, args.target_fpr) for c, e in parts[lvl]]
                cell[lvl] = {name: summarise(per, name) for name in arms}
                frozen_name = "frozen" if probe_based else "frozen_thr"
                # the generic path must reproduce paired.repartition_estimate on the frozen arm, draw by draw
                rep, arr = P.repartition_estimate(pre_pool[:n_cal], pre_pool[n_cal:], post["pool"][:n_cal], post["pool"][n_cal:],
                                                  pre_pos, post["pos"], draws[lvl], args.target_fpr)
                for q_rep, q_mine in (("fpr_post", "fpr_post"), ("paired_ratio", "paired_ratio"),
                                      ("ratio_to_target", "ratio_to_target"), ("tpr_post", "tpr_post"), ("delta_tpr", "delta_tpr"),
                                      ("floor_ratio_to_target", "floor_ratio_to_target"), ("tpr_pre", "tpr_pre")):
                    mine = np.array([p[frozen_name][q_mine] for p in per[1:]])
                    if not np.array_equal(mine, arr[q_rep]) or per[0][frozen_name][q_mine] != rep[q_rep]["frozen"]:
                        raise SystemExit(f"REFUSING: L{li} {pset} {lvl}: the generic path is not repartition_estimate ({q_rep})")
                n_generic += 1
                if probe_based and registered:
                    # the anchor: the pair's depth file, exactly (a test with other parameters cannot match it)
                    dc = dcells[(li, pset)]
                    want = dc["operating_point"] if lvl == LEVELS[0] else dc[LEVELS[1]]["operating_point"]
                    if json.loads(json.dumps(rep, default=str)) != want:
                        raise SystemExit(f"REFUSING: L{li} {pset} {lvl}: the frozen arm is not {depth_rel}'s operating point")
                    rt = np.concatenate([[rep["ratio_to_target"]["frozen"]], arr["ratio_to_target"]]).astype("<f8")
                    fc = P.abs_log(rt).astype("<f8")
                    want_fc = dc["frozen_curve_abs_log_ratio_to_target"] if lvl == LEVELS[0] else dc[LEVELS[1]]["frozen_curve_abs_log_ratio_to_target"]
                    if sha256_bytes(fc.tobytes()) != want_fc["per_draw_sha256"]:
                        raise SystemExit(f"REFUSING: L{li} {pset} {lvl}: the frozen curve is not {depth_rel}'s bit for bit")
                    if args.d117_layer:
                        # D-257 (10): the frozen per-draw array this mode WRITES is the depth file's curve, bit for bit
                        fz = np.array([p["frozen"]["abs_log_ratio_to_target"] for p in per]).astype("<f8")
                        if sha256_bytes(fz.tobytes()) != want_fc["per_draw_sha256"]:
                            raise SystemExit(f"REFUSING: L{li} {pset} {lvl}: the written frozen array is not {depth_rel}'s curve")
                        d117_curves[(pset, lvl)] = want_fc["per_draw_sha256"]
                    n_anchor += 1
                if args.fpr5 and probe_based and registered:
                    # D-248 self-anchor: the frozen arm at 1% equals the 1% depth record's operating point
                    rep1, _ = P.repartition_estimate(pre_pool[:n_cal], pre_pool[n_cal:], post["pool"][:n_cal], post["pool"][n_cal:],
                                                     pre_pos, post["pos"], draws[lvl], P.TARGET_FPR)
                    d1 = dcells1[(li, pset)]
                    want1 = d1["operating_point"] if lvl == LEVELS[0] else d1[LEVELS[1]]["operating_point"]
                    if json.dumps(json.loads(json.dumps(rep1, default=str)), sort_keys=True) != json.dumps(want1, sort_keys=True):
                        raise SystemExit(f"REFUSING: L{li} {pset} {lvl}: the frozen arm's 1% twin is not {depth1_rel}'s (D-248)")
                    n_self_anchor += 1
                for name in arms:
                    per_layer_abslog.setdefault((pset, lvl, name), []).append(
                        np.array([p[name]["abs_log_ratio_to_target"] for p in per]))
                # D-235 (2): the no-update floor, |log(FPR_pre / target)| at each draw's frozen threshold
                fl = np.array([float(P.abs_log(p[frozen_name]["floor_ratio_to_target"])) for p in per])
                if not np.array_equal(fl[1:], arr["floor_abs_log"]) or fl[0] != rep["floor_abs_log"]["frozen"]:
                    raise SystemExit(f"REFUSING: L{li} {pset} {lvl}: the floor row is not repartition_estimate's floor_abs_log")
                per_layer_abslog.setdefault((pset, lvl, FLOOR_ROW), []).append(fl)
                if "self_align_PRE" in arms:
                    b = cell[lvl]["self_align_PRE"]["paired_ratio"]["band95"]
                    cell[lvl]["self_align_PRE"]["no_update_control_band_includes_1"] = bool(b[0] <= 1.0 <= b[1])
            if args.family == "alignvar":
                # AUROC after mapping, eval split, D-152 P5 item bootstrap (unit weights = metrics.auroc)
                au: dict[str, Any] = {}
                srcs = {"pre": (pre_pos, pre_pool[n_cal:]), "frozen": (post["pos"], post["pool"][n_cal:]),
                        **{m: (arms[m]["pos"], arms[m]["pool"][n_cal:]) for m in MAPS},
                        "refit_oracle": (arms["refit_oracle"]["pos"], arms["refit_oracle"]["pool"][n_cal:])}
                reps: dict[str, np.ndarray] = {}
                for name, (sp_, sn_) in srcs.items():
                    pt = np.array([M.auroc(sp_[:, s], sn_[:, s]).value for s in range(sp_.shape[1])])
                    reps[name] = np.mean([D.weighted_auroc(sp_[:, s], sn_[:, s], CNT_P, CNT_N) for s in range(sp_.shape[1])], axis=0)
                    au[name] = {"mean_over_seeds": float(pt.mean()), "per_seed": pt.tolist(), **P.ci(reps[name])}
                for name in ("frozen", *MAPS, "refit_oracle"):
                    au[f"drop_{name}"] = {"point": au["pre"]["mean_over_seeds"] - au[name]["mean_over_seeds"],
                                          **P.ci(reps["pre"] - reps[name])}
                cell["auroc_eval_split"] = au
            cells.append(cell)
        print(f"  layer {li:>2} done", flush=True)

    # per pair and arm: the per-draw mean over layers of |log(FPR/target)| (Table 2's metric), with its band;
    # the floor row beside, and every per-draw array with its sha256 (D-235 (2), (8))
    pair_summary: dict[str, Any] = {}
    pair_per_draw: dict[str, Any] = {}
    for (pset, lvl, name), rows in per_layer_abslog.items():
        m = np.mean(np.vstack(rows), axis=0).astype("<f8")
        pair_summary.setdefault(pset, {}).setdefault(lvl, {})[name] = P.band_summary(m[1:], m[0])
        pair_per_draw.setdefault(pset, {}).setdefault(lvl, {})[name] = {"sha256": sha256_bytes(m.tobytes()),
                                                                        "values": [float(v) for v in m]}

    if args.d117_layer:
        # the bytes actually written: every frozen per-draw array hashes to the depth file's curve at L*
        bad = [(ps, lv) for ps in PROBE_SETS for lv in LEVELS
               if pair_per_draw[ps][lv]["frozen"]["sha256"] != d117_curves.get((ps, lv))]
        if bad:
            raise SystemExit(f"REFUSING to write: the written frozen arrays {bad} are not {depth_rel}'s curves at L{L_sel}")
    if final and probe_based and n_anchor != len(layers) * len(PROBE_SETS) * len(LEVELS):
        raise SystemExit(f"REFUSING to write: {n_anchor} anchor blocks, not {len(layers) * len(PROBE_SETS) * len(LEVELS)}")
    code1 = P.code_block(CODE_FILES)
    changed = [n for n, a, b in (("code", code1["sha256"], code0["sha256"]), ("selection", sha256_file(sel_path), sel_sha),
                                 ("cluster manifest", sha256_file(D.CLUSTERS), cl_sha),
                                 ("registry sources (re-read from disk)", D.sources_now(), reg_sources)) if a != b]
    if probe_based and sha256_file(ROOT / depth_rel) != depth_sha:
        changed.append(depth_rel)
    changed += [f"cache {r['file']}" for r in input_recs.values() if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    if args.fpr5 and final and n_self_anchor != len(layers) * len(PROBE_SETS) * len(LEVELS):
        raise SystemExit(f"REFUSING to write: {n_self_anchor} self-anchored blocks (D-248)")
    if args.fpr5 and sha256_file(ROOT / depth1_rel) != depth1_sha:
        changed.append(depth1_rel)
    if prior_path and sha256_file(prior_path) != prior_sha:
        changed.append(str(prior_path))
    if args.d117_layer and sha256_file(ROOT / rr_rel) != rr_sha:
        changed.append(rr_rel)
    if changed:
        raise SystemExit(f"REFUSING to write: changed during the run: {changed}")
    supersedes = None
    # D-256 (1): the NP _r2 run may differ from D-232 only in the two quantile arms (F-070); every other arm must equal it
    drop = set(QUANTILE_ARMS) if args.family == "nonparametric" else set()
    if prior_path and old is not None:
        new_ps = {ps: {lv: {a: v for a, v in blk.items() if a != FLOOR_ROW} for lv, blk in lvs.items()}
                  for ps, lvs in pair_summary.items()}
        same = {"cells": canonical(strip_arms(old["cells"], drop)) == canonical(strip_arms(cells, drop)),
                "pair_summary": canonical(strip_arms(old["pair_summary_mean_abs_log_over_layers"], drop))
                                == canonical(strip_arms(new_ps, drop))}
        if not all(same.values()):
            raise SystemExit(f"REFUSING to write: this run does not reproduce {prior_path.name}: {same} (D-235 (8), D-256 (1))")
        supersedes = {"file": str(prior_path.relative_to(ROOT)).replace("\\", "/") if prior_path.is_relative_to(ROOT) else str(prior_path),
                      "sha256": prior_sha, "cells_identical": not drop, "pair_summary_identical_apart_from_the_floor_row": not drop}
        if drop:
            supersedes.update({"cells_identical_apart_from_arms": sorted(drop),
                               "pair_summary_identical_apart_from_the_floor_row_and_arms": sorted(drop),
                               "why": "the quantile arms use paired.quantile_transfer, the F-070 fix (D-245, D-256 (1))"})
        print(f"reproduces {prior_path.name} exactly (cells and pair summary{', apart from ' + str(sorted(drop)) if drop else ''}; "
              f"D-235 (8), D-256 (1))", flush=True)
    # D-257 (10): the cells at the D-117 layer equal the record of record's cells there, byte for byte
    d117_check = None
    if args.d117_layer:
        want_cells = [c for c in rr_doc["cells"] if c["layer"] == L_sel]
        if canonical(want_cells) != canonical(cells) or len(cells) != len(PROBE_SETS):
            raise SystemExit(f"REFUSING to write: the cells at L{L_sel} are not {rr_rel}'s (D-257 (10))")
        d117_check = {"record_of_record": rr_rel, "sha256": rr_sha, "manifest": D117_RECORD_OF_RECORD[rr_key][1],
                      "layer": L_sel, "n_cells_identical": len(cells),
                      "kind": "every cell at the D-117 layer (both probe sets, both levels) equals the record of record's"}
        print(f"cells at L{L_sel} equal {rr_rel}'s ({len(cells)} of {len(PROBE_SETS)}; D-257 (10))", flush=True)
    payload = {**provenance_block(f"repair_d156_{args.family}"),
               "preregistration": ("D-224, D-228, D-237 (3), D-238 (D-169 item 10, the repair part: the LoRA arms)"
                                   if args.family == "moment_ext" else
                                   "D-228, D-238, D-243 (D-169 item 13, part (A): the repair_test records)"
                                   if args.family == "repair" else
                                   "D-228, D-229, D-230 (D-169 items 5-7; the author's items 6, 7, 8)") +
                                  "; D-235 (the floor row, the per-draw arrays, the moment _r2 records)" +
                                  ("; 4e: D-152, D-248 (the 5% arm)" if args.fpr5 else "") +
                                  ("; D-256 (1), D-258 (2) (the NP _r2 re-run under the F-070 quantile transfer)"
                                   if args.family == "nonparametric" else "") +
                                  ("; D-256 (3), D-257 (the D-117-layer mode)" if args.d117_layer else "") +
                                  ("; D-271 (4), D-273 (the D-117-layer mode at 5%)" if args.d117_layer and args.fpr5 else ""),
               **({"d117_check": d117_check} if args.d117_layer else {}),
               **({"fpr5_self_anchor": {"file": depth1_rel, "sha256": depth1_sha, "n_level_blocks_identical_at_1pct": n_self_anchor,
                                        "kind": "the frozen arm recomputed at 1% equals the 1% depth record's operating point "
                                                "at every layer, probe set and level (D-248)"}} if args.fpr5 else {}),
               "supersedes": supersedes,
               "code": code0, "registered_parameters": registered, "blas_threads": threads, "input_verification": verification,
               "config": {"family": args.family, "pair": args.pair, "registry_pair": reg_stem, "axis": axis, "arms": list(fam["arms"]),
                          "selection_file": str(sel_path.relative_to(ROOT)).replace("\\", "/"), "selection_pair": sel_stem,
                          "selection_sha256": sel_sha, "selected_C": C_sel, "selected_layer": L_sel, "layers": layers,
                          "seeds": args.seeds, "target_fpr": args.target_fpr, "n_draws": args.n_draws,
                          "draw_seed": P.AXIS_DRAW_SEED[axis], "cluster_draw_seed": P.AXIS_CLUSTER_DRAW_SEED[axis],
                          "cluster_manifest": {"file": "manifests/negative_clusters.json", "sha256": cl_sha},
                          "maps_fit_on": "post -> pre on probe_train plus the threshold_calib harmful items (D-229)",
                          "ridge_lambda": {"grid": list(LAMBDA_GRID), "cv_folds": CV_FOLDS, "cv_seed": CV_SEED, "inside": "probe_train"},
                          "calibration": "every score- or threshold-level calibration on each draw's calibration part (D-228 (1b))",
                          "n_exemplars": N_EXEMPLARS if not probe_based else None,
                          "item_bootstrap": {"n_boot": args.n_boot, "seeds": D.ITEM_BOOT_SEED},
                          "pre_repo": cfg["pre_repo"], "pre_rev": cfg["pre_rev"], "post_repo": cfg["post_repo"], "post_rev": cfg["post_rev"],
                          "input_registry_sources_sha256": reg_sources},
               "anchor": {"file": depth_rel, "sha256": depth_sha, "n_level_blocks_identical": n_anchor,
                          "kind": "the frozen arm's operating point and its per-draw frozen curve"} if probe_based else
                         {"kind": "none: the similarity layer has no depth file; the generic path is checked against "
                                  "paired.repartition_estimate on frozen_thr"},
               "n_level_blocks_generic_path_equals_repartition_estimate": n_generic,
               "input_cache_sha256": input_sha,
               "input_cache_source": {kk: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"]}
                                      for kk, r in input_recs.items()},
               **({"summary_at_d117_layer": pair_summary, "per_draw_at_d117_layer": pair_per_draw} if args.d117_layer else
                  {"pair_summary_mean_abs_log_over_layers": pair_summary}),
               "floor_row": (f"'{FLOOR_ROW}' (D-235 (2)): the same metric under no update, "
                             + ("at the D-117 layer, " if args.d117_layer else "the per-draw mean over layers of ") +
                             "|log(FPR_pre / target)|, FPR_pre the pre model's eval FPR at the draw's own frozen threshold "
                             "(= paired.repartition_estimate's floor_abs_log, checked draw by draw)"),
               **({} if args.d117_layer else {"pair_per_draw_mean_abs_log_over_layers": pair_per_draw}),
               "per_draw_note": "index 0 is the frozen split, 1..n_draws the D-156 (item level) or D-164 (cluster level) draws; "
                                "sha256 over the little-endian float64 bytes of 'values'",
               "cells": cells}
    digest = write_results(out_path, payload, exclusive=True)
    print(f"Written: {out_path}\nsha256:  {digest}\nanchor blocks identical: {n_anchor}; generic-path blocks: {n_generic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
