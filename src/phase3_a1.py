"""Phase 3 attacker A1, score translation in feature space (D-327), part 2 of 2: the SIMULATION, one pair per run.

REFUSES unless the predictions record (src/phase3_a1_predictions.py) exists, names this pair, these draws (the hash of
every draw's index arrays), these inputs and parameters, and the current sha256 of the prediction function's file, and
unless its arrays equal this run's own recomputation from c = 0 data and the geometry bit for bit (D-327 (6)).

THE ATTACK (D-327 (2)): every post-update feature vector the defender scores after the update gets x + delta: E, the
eval positives, cal_d re-scored (R_same and the health check), R_FRESH, the 1% traffic pool (post window), and the
post probe_train fit rows of class-balanced restandardisation (and of the refit oracle). delta per draw d and seed s:
white-box c * sigma_ds * v / (v . v), v = w / sd of the frozen probe; transfer the same norm along the surrogate's
direction (src/phase3_a1_predictions.py, attack_vector). c in {-3, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 3}.
  Frozen-probe scores of x + delta are s(x) + delta . v (the probe is linear: ((x + delta - mu) / sd) . w + b), with
  delta . v computed from the materialised delta; the identity is checked on the frozen split against the score of
  the materialised x + delta (linear_identity_check). Re-scoring the materialised x + delta instead would put float64
  scores of attacked items beside float32-path scores of the clean ones (the caches are float32, and the clean scores
  must be 4d B's exactly for the anchor); their rounding differences (about 1e-6) reorder near-tied items, which is
  not the attack. Restandardisation and the refit oracle compute their statistics
  (per-feature mean and SD) on the MATERIALISED attacked features (poolstudy_d156.feature_stats), then score with
  poolstudy_d156.restandardised_scores plus delta . (w / sd') (the same linearity).
MEASURES per variant, c and draw (the frozen split first, then 4d B's 200 draws; D-156 mean and band), D-327 (3):
  arms      frozen; R_same (1% threshold on cal_d's attacked post scores); R_fresh (n 1000); quantile transfer on 1%
            traffic (pre window unattacked, post window attacked; paired.quantile_transfer, F-070); restandardise on
            the class-balanced reference (the post probe_train fit rows of each seed, 4d A's reference, with 4d B's
            float64 statistics); restandardise on 1% traffic (4d B's feature arm); the refit oracle beside (upper
            bound): each arm's FPR (seed mean of floored per-seed FPRs on E), paired ratio against the pre FPR at T_d,
            FPR/target and TPR on the eval positives, as 4d B's put() computes them.
  AUROC     on E and the positives, with and without the attack (seed mean); on the frozen eval split for the AUC
            rule (dAUC <= -0.05 or post AUROC < 0.70, against the pre AUROC; D-241 (4)).
  health    the 4c statistics on cal_d (healthcheck_d156.health: pre scores against attacked post scores at T_d), the
            Gaussian health ratio primary (D-264) and its flag at 2 / 0.5.
  TPR       the D-282 combined interval (paired.combined_tpr_replicates): at c = 0 for every arm whose positive scores
            are fixed across draws (frozen, R_same, R_fresh, quantile transfer, class-balanced restandardisation, refit
            oracle); at every c != 0 for frozen, R_same, R_fresh and quantile transfer, as the clean positives against
            the exact shifted threshold t' = min{p : fl(p + Delta) >= t} per draw and seed (Delta the realised delta . v;
            fl(p + Delta) is non-decreasing in p, so p >= t' is fl(p + Delta) >= t bit for bit, which p >= t - Delta is
            not when a positive ties the threshold-setting score), each checked to reproduce the arm's TPR draw by
            draw. The restandardised arms and the refit oracle move their
            positives per draw under the attack: c = 0 only (their TPR equals its c = 0 value by translation invariance).
  recorded  FPR_pre per draw and seed (the D-257 floor) and the relative budget ||delta|| / mean ||z|| over cal_d's raw
            post features (design rev 2 section 2.3), per draw and seed.
SELF-ANCHOR (D-327 (6)): at c = 0, before any attacked score, every arm 4d B ran (frozen, R_same, R_fresh n 1000,
  quantile transfer and restandardisation on 1% traffic at n 1000) must give FPR, paired ratio, FPR/target and TPR
  D-156 summaries equal to the 4d B record of record's (populations.POOLSTUDY_D156) exactly, and the frozen eval
  split's per-seed AUROCs pre and post equal the depth_d156 record's (the AUC rule's source); else it refuses. The
  anchor needs the registered draws, seeds and target (a test with fewer draws records it as not checked).
SCORING (D-327 (4), (5)): P-A1.1 |AUROC(post + delta) - AUROC(post)| <= 1e-12 per variant, c, draw and seed (on E and
  on the frozen eval split); P-A1.2 the AUC rule's flag at every c equals its c = 0 flag, draw by draw; P-A1.3 D-177's
  accuracy line over this pair's cells (c != 0 x variant; realised and predicted FPR/target as per-cell D-156 means;
  unfloored primary, floored beside); P-A1.4 the per-draw confusion counts over c != 0 (white-box primary, transfer
  beside) of the Gaussian flag against the realised class (paired ratio >= 2 or <= 0.5), with their D-156 means for the
  pair-cluster pooling of D-241 (6) across pairs (not done here); P-A1.5 the FPR/target bands of R_same, R_fresh and
  quantile transfer include 1 at every c and variant; P-A1.6 both restandardisations' per-seed FPR equals its c = 0
  value in every draw and seed to a relative 1e-9. A failed P-A1.1, P-A1.5 or P-A1.6 is a protocol error (D-327 (5));
  so is, as an implementation check by D-181, a linear-identity check beyond its bound (IDENTITY_TOL): the run prints
  the scoring and exits 3 without writing.
CHOICES BY THE CLOSEST D-ENTRY (D-181; besides (a)-(e) in src/phase3_a1_predictions.py):
  (f) refit oracle: the post probe refitted once per seed on the clean post fit rows (4d A's rule: same rows, C, row
      order); under the attack its standardisation is recomputed on the attacked fit rows, which leaves the
      standardised training matrix, so the logistic fit, unchanged (exactly, up to rounding); its threshold is its own
      1% threshold on its attacked post scores of cal_d (4d A's OWN_THRESHOLD). Its statistics are float64 at every c.
  (g) P-A1.4 is scored on the white-box attacker (D-327 (4) names no variant; white-box is primary, D-318 (9)); the
      transfer variant is reported beside.
  (h) the AUC rule is evaluated per draw (delta depends on sigma_ds) on the frozen eval split, as D-241 (4) reads it.
Writes results/phase3_a1_<pair>.json and results/phase3_a1_<pair>_draws.npz (tracked, D-163 (6)), once (exclusive).
Only registered parameters at the 6-thread BLAS pin, reading the final predictions record, write the final names.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_d156 as D  # noqa: E402
import healthcheck_d156 as HC  # noqa: E402
import input_registry as REG  # noqa: E402
import metrics as M  # noqa: E402
import paired as P  # noqa: E402
import phase3_a1_predictions as A1P  # noqa: E402
import populations as POP  # noqa: E402
import poolstudy_d156 as PS  # noqa: E402
from provenance import (  # noqa: E402
    ensure_utf8_stdout, provenance_block, read_result, resolve_cache, sha256_bytes, sha256_file,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
THIS_FILE = "src/phase3_a1.py"
CODE_FILES = [THIS_FILE, "src/healthcheck_d156.py", *A1P.CODE_FILES]
ARMS = ("frozen", "R_same", "R_fresh", "quantile_transfer_traffic_1pct", "restandardise_class_balanced",
        "restandardise_traffic_1pct", "refit_oracle")
ANCHORED = {"frozen": "frozen", "R_same": "score|R_same|n=1000", "R_fresh": "score|R_fresh|n=1000",
            "quantile_transfer_traffic_1pct": f"score|{A1P.TRAFFIC}", "restandardise_traffic_1pct": f"feature|{A1P.TRAFFIC}"}
QUANT = ("fpr", "paired_ratio", "ratio_to_target", "tpr")
THRESHOLD_LEVEL = ("R_same", "R_fresh", "quantile_transfer_traffic_1pct")            # P-A1.5
RESTANDARDISED = ("restandardise_class_balanced", "restandardise_traffic_1pct")       # P-A1.6
D282_ARMS = ("frozen", "R_same", "R_fresh", "quantile_transfer_traffic_1pct", "restandardise_class_balanced",
             "refit_oracle")                                                  # D-282 at c = 0
D282_SHIFTED = ("frozen", "R_same", "R_fresh", "quantile_transfer_traffic_1pct")    # D-282 at c != 0 (exact_shifted)
# linear_identity_check bounds, score units, on the frozen split: the frozen probe compares the float64 score of the
# materialised x + delta with the float32 cache path's s(x) + delta . v (sandbox llama31: 1.7e-7); the restandardised
# probe is float64 both ways (sandbox: 8e-15). Both sides use the same delta, so the bound catches a SCORING-PATH
# mismatch (e.g. a wrong v or standardisation in the linear identity), not a wrong delta; a wrong delta is caught by
# the predictions' white-box check (delta . v against c * sigma) and the transfer geometry.
IDENTITY_TOL = {"frozen_probe_max_abs": 1e-4, "restandardised_probe_max_abs": 1e-9}
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
HEALTH = ("z_mean", "log_sd_ratio", "gaussian_ratio", "log_gaussian_ratio", "direct_ratio")
CUT = (2.0, 0.5)                                       # D-241 (3) primary class and the health flag (D-327 (3))
AUC_TOL, REL_TOL, FACTOR = 1e-12, 1e-9, 1.5
PREREG = ("D-327 (the Phase 3 A1 pre-registration: (1)-(6)); D-318 (9); docs/phase3_design.md revision 2; machinery "
          "of D-244/D-298 (4d B), D-241/D-264 (4c), D-282 (TPR interval), D-177 (accuracy line)")


def arm_numbers(F_pre: np.ndarray, F: np.ndarray, Q: np.ndarray, target: float) -> dict[str, Any]:
    """4d B's put(): FPR (seed mean of floored per-seed FPRs), the paired ratio against F_pre, FPR/target, TPR; and the
    per-seed floored FPR (P-A1.6)."""
    fs = P.floored_fpr(F)
    f = float(fs.mean())
    return {"fpr": f, "paired_ratio": P.ratio_from_flags(F_pre, F), "ratio_to_target": f / target,
            "tpr": float(P.rate(Q).mean()), "fpr_per_seed": fs}


def precompute(S: dict[str, Any]) -> dict[str, Any]:
    """The float64 feature tables (4d B's), the fit rows' post features per seed, the refit oracle's probes (choice
    (f)), the score tables and the pre AUROCs of the frozen eval split."""
    f64 = S["f64"]
    RAW, SC = S["RAW"], S["SC"]
    ALL_post = np.vstack([f64(RAW["post"]["pool"]), f64(RAW["post"]["H"])])     # 4d B's feature table: pool, then H
    refits = []
    for seed in S["seeds"]:
        rows, C, order = PS.probe_spec(len(S["y_tr"]), seed, A1P.PSET, S["C_sel"])
        refits.append(P.fit_probe(S["Xtr"]["post"][rows], S["y_tr"][rows], seed, C, order))
    n_pool = S["n_pool"]
    evneg = np.arange(S["n_cal"], n_pool)                                       # the frozen eval split's negatives
    oneP, oneF = np.ones((1, S["n_pos"])), np.ones((1, len(evneg)))
    auroc_pre = np.array([D.weighted_auroc(SC["pre"]["pos"][:, j], SC["pre"]["pool"][evneg, j], oneP, oneF)[0]
                          for j in range(len(S["seeds"]))])
    return {"ALL_post": ALL_post, "PN_post": ALL_post[:n_pool], "POS_post": f64(RAW["post"]["pos"]),
            "POS_scores": SC["post"]["pos"], "XF": [f64(S["Xtr"]["post"][rows]) for rows in S["rows_of"]],
            "refits": refits, "pre_all": np.vstack([SC["pre"]["pool"], SC["pre"]["H"]]),
            "post_all": np.vstack([SC["post"]["pool"], SC["post"]["H"]]),
            "evneg": evneg, "oneP": oneP, "oneF": oneF, "oneE": np.ones((1, len(S["draws"][0]["E"]))),
            "auroc_pre": auroc_pre, "auroc_pre_mean": float(auroc_pre.mean()),
            "z_norm_pool_post": np.linalg.norm(ALL_post[:n_pool], axis=1)}         # raw post-feature norms, pooled


def measure(S: dict[str, Any], X: dict[str, Any], variant: str, c: float) -> dict[str, np.ndarray]:
    """Every D-327 (3) measure at one (variant, c), per draw (row 0 the frozen split). Returns per-draw arrays
    '<arm>|<quantity>' and per-seed arrays (draws x seeds). At c = 0 nothing is attacked (delta is 0)."""
    target, nS, SC = S["target"], len(S["seeds"]), S["SC"]
    acc: dict[str, list[Any]] = {}
    put = lambda k, x: acc.setdefault(k, []).append(x)  # noqa: E731
    for d, dr in enumerate(S["draws"]):
        cal, E, Rf, tr = dr["cal"], dr["E"], dr["R_fresh"], dr["traffic"]
        T = S["T"][d]
        deltas = [A1P.attack_vector(S, variant, c, d, j) for j in range(nS)]
        shift = np.array([float(dv @ S["v"][j]) for j, dv in enumerate(deltas)])   # the frozen scores' realised shift
        post_all = X["post_all"] + shift if c != 0 else X["post_all"]               # every post score, pool then H
        post_pos = X["POS_scores"] + shift if c != 0 else X["POS_scores"]
        postE = post_all[E]
        F_pre = SC["pre"]["pool"][E] >= T
        put("frozen|fpr_pre_per_seed", P.floored_fpr(F_pre))           # the D-257 floor's FPR_pre, per seed
        res = {"frozen": arm_numbers(F_pre, postE >= T, post_pos >= T, target)}
        thr: dict[str, np.ndarray] = {"frozen": T}
        for name, idx in (("R_same", cal), ("R_fresh", Rf)):
            t_ = np.array([P.thr_at_fpr(post_all[idx, j], target) for j in range(nS)])
            res[name], thr[name] = arm_numbers(F_pre, postE >= t_, post_pos >= t_, target), t_
        tq = np.array([P.quantile_transfer(X["pre_all"][tr, j], post_all[tr, j], T[j]) for j in range(nS)])
        res["quantile_transfer_traffic_1pct"] = arm_numbers(F_pre, postE >= tq, post_pos >= tq, target)
        thr["quantile_transfer_traffic_1pct"] = tq
        # restandardisation and the refit oracle: statistics of the materialised attacked post features
        base_tr = X["ALL_post"][tr]
        cols: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {n: ([], []) for n in (*RESTANDARDISED, "refit_oracle")}
        t_ref = np.empty(nS)
        for j, q in enumerate(S["probes"]):
            dv = deltas[j]
            st_tr = PS.feature_stats(base_tr if c == 0 else base_tr + dv)
            st_fit = PS.feature_stats(X["XF"][j] if c == 0 else X["XF"][j] + dv)
            for name, qq, (mu, sd) in (("restandardise_traffic_1pct", q, st_tr), ("restandardise_class_balanced", q, st_fit),
                                       ("refit_oracle", X["refits"][j], st_fit)):
                eE = PS.restandardised_scores(qq, mu, sd, X["PN_post"][E])
                eP = PS.restandardised_scores(qq, mu, sd, X["POS_post"])
                eC = PS.restandardised_scores(qq, mu, sd, X["PN_post"][cal]) if name == "refit_oracle" else None
                if c != 0:
                    sh = float(dv @ (qq.w / sd))
                    eE, eP = eE + sh, eP + sh
                    eC = eC + sh if eC is not None else None
                if eC is not None:
                    t_ref[j] = P.thr_at_fpr(eC, target)              # the refit oracle's own 1% threshold on cal_d
                tt = t_ref[j] if name == "refit_oracle" else T[j]
                cols[name][0].append(eE >= tt)
                cols[name][1].append(eP >= tt)
        for name, (fe, fp) in cols.items():
            res[name] = arm_numbers(F_pre, np.column_stack(fe), np.column_stack(fp), target)
        thr["restandardise_class_balanced"], thr["refit_oracle"] = T, t_ref.copy()
        for name, r in res.items():
            for qn in QUANT:
                put(f"{name}|{qn}", r[qn])
            if name in RESTANDARDISED:
                put(f"{name}|fpr_per_seed", r["fpr_per_seed"])
        for name in D282_ARMS:
            put(f"{name}|threshold_per_seed", thr[name])
        # the 4c health check on cal_d (pre scores of the stored calibration set against attacked post scores)
        h = HC.health(SC["pre"]["pool"][cal], post_all[cal], T)
        for qn in HEALTH:
            put(f"health|{qn}", h[qn])
        # AUROC with and without the attack: E and the positives; the frozen eval split for the AUC rule
        aE = np.array([D.weighted_auroc(post_pos[:, j], postE[:, j], X["oneP"], X["oneE"])[0] for j in range(nS)])
        aF = np.array([D.weighted_auroc(post_pos[:, j], post_all[X["evneg"], j], X["oneP"], X["oneF"])[0]
                       for j in range(nS)])
        put("auroc_E|per_seed", aE)
        put("auroc_E|mean_over_seeds", float(aE.mean()))
        put("auroc_frozen_eval|per_seed", aF)
        post_mean = float(aF.mean())
        put("auc_rule|auroc_post_mean_over_seeds", post_mean)
        put("auc_rule|d_auroc", post_mean - X["auroc_pre_mean"])
        put("auc_rule|flag", float(post_mean - X["auroc_pre_mean"] <= -0.05 or post_mean < 0.70))
        put("shift|delta_dot_v_per_seed", shift)
        dnorm = np.array([float(np.linalg.norm(dv)) for dv in deltas])
        zbar = float(X["z_norm_pool_post"][cal].mean())             # mean ||z|| of cal_d's raw post features
        put("shift|delta_norm_per_seed", dnorm)
        put("shift|mean_z_norm_cal_post", zbar)
        put("shift|relative_budget_per_seed", dnorm / zbar)
    out = {k: np.asarray(v, dtype="<f8") for k, v in acc.items()}
    out["health|gaussian_flag"] = HC.crosses(out["health|gaussian_ratio"], CUT).astype("<f8")
    out["health|direct_flag"] = HC.crosses(out["health|direct_ratio"], CUT).astype("<f8")
    out["realised_class_2x"] = HC.crosses(out["frozen|paired_ratio"], CUT).astype("<f8")
    return out


def linear_identity_check(S: dict[str, Any], X: dict[str, Any], variant: str, c: float) -> dict[str, float]:
    """On the frozen split: the largest |score(x + delta) - (score(x) + delta . v)| of the frozen probe over every
    pool, H and positive item (x + delta materialised in float64; score(x) through the raw cache path, so the cache
    path's float32 rounding bounds it), and of the class-balanced restandardised probe on E (float64 both ways)."""
    f64, worst_f, worst_r = S["f64"], 0.0, 0.0
    E = S["draws"][0]["E"]
    for j, q in enumerate(S["probes"]):
        dv = A1P.attack_vector(S, variant, c, 0, j)
        sh = float(dv @ S["v"][j])
        for part in ("pool", "H", "pos"):
            direct = q.score(f64(S["RAW"]["post"][part]) + dv)
            worst_f = max(worst_f, float(np.max(np.abs(direct - (S["SC"]["post"][part][:, j] + sh)))))
        mu, sd = PS.feature_stats(X["XF"][j] + dv)
        direct = PS.restandardised_scores(q, mu, sd, X["PN_post"][E] + dv)
        lin = PS.restandardised_scores(q, mu, sd, X["PN_post"][E]) + float(dv @ (q.w / sd))
        worst_r = max(worst_r, float(np.max(np.abs(direct - lin))))
    return {"frozen_probe_max_abs": worst_f, "restandardised_probe_max_abs": worst_r}


def summaries(arr: dict[str, np.ndarray], target: float = P.TARGET_FPR) -> dict[str, Any]:
    """The D-156 cell block of one (variant, c): every per-draw array as mean and band with the frozen split."""
    bs = lambda a: P.band_summary(a[1:], a[0])  # noqa: E731
    out: dict[str, Any] = {"arms": {name: {qn: bs(arr[f"{name}|{qn}"]) for qn in QUANT} for name in ARMS}}
    out["arms"]["refit_oracle"]["role"] = "upper bound: needs labels the defender does not have (D-318 (9))"
    out["health"] = {**{qn: bs(arr[f"health|{qn}"]) for qn in HEALTH},
                     "share_of_draws_gaussian_flag_2x": float(arr["health|gaussian_flag"][1:].mean()),
                     "share_of_draws_direct_flag_2x": float(arr["health|direct_flag"][1:].mean()),
                     "share_of_draws_realised_class_2x": float(arr["realised_class_2x"][1:].mean()),
                     "frozen_split": {"gaussian_flag": bool(arr["health|gaussian_flag"][0]),
                                      "realised_class_2x": bool(arr["realised_class_2x"][0])}}
    out["auroc"] = {"E_mean_over_seeds": bs(arr["auroc_E|mean_over_seeds"]),
                    "frozen_eval_split_post_mean_over_seeds": bs(arr["auc_rule|auroc_post_mean_over_seeds"])}
    out["auc_rule"] = {"d_auroc": bs(arr["auc_rule|d_auroc"]), "share_of_draws_flagged": float(arr["auc_rule|flag"][1:].mean()),
                       "frozen_split_flag": bool(arr["auc_rule|flag"][0])}
    out["realised_shift"] = {"delta_dot_v_mean_over_seeds": bs(arr["shift|delta_dot_v_per_seed"].mean(axis=1)),
                             "delta_norm_mean_over_seeds": bs(arr["shift|delta_norm_per_seed"].mean(axis=1)),
                             "mean_z_norm_cal_post": bs(arr["shift|mean_z_norm_cal_post"]),
                             "relative_budget_mean_over_seeds": bs(arr["shift|relative_budget_per_seed"].mean(axis=1))}
    out["floor"] = {"fpr_pre_ratio_to_target": bs(arr["frozen|fpr_pre_per_seed"].mean(axis=1) / target),
                    "note": "FPR_pre (pre scores of E at T_d, floored per seed at 1/|E|): D-257's no-update floor"}
    return out


# ---- the refusals and the anchor (each a function, so the tests exercise them) ------------------------------------

def check_predictions(pdoc: dict[str, Any], pnpz_sha: str | None, fn_sha: str, pair: str, n_reps: int, seeds: list[int],
                      target: float, final: bool) -> None:
    """D-327 (6), before any computation: the record is the A1 predictions record of this pair, made by this prediction
    function (the sha256 of its file now), at these parameters; its npz is the bytes it records."""
    pf = pdoc.get("prediction_function") or {}
    if pdoc.get("run_kind") != "phase3_a1_predictions" or (pdoc.get("config") or {}).get("pair") != pair:
        raise SystemExit(f"REFUSING: not the A1 predictions record of {pair}")
    if pf.get("module") != A1P.THIS_FILE or pf.get("function") != "predicted_fpr" or pf.get("sha256") != fn_sha:
        raise SystemExit(f"REFUSING: the predictions record was made by another prediction function "
                         f"({str(pf.get('sha256'))[:12]} vs {fn_sha[:12]})")
    want = {"n_reps": n_reps, "seeds": list(seeds), "target_fpr": target, "c_grid": list(A1P.C_GRID)}
    got = {k: pdoc["config"].get(k) for k in want}
    if got != want:
        raise SystemExit(f"REFUSING: the predictions record has other parameters: {got}")
    if final:
        if pdoc.get("registered_parameters") is not True:
            raise SystemExit("REFUSING the final name: the predictions record was not a registered run")
        th = pdoc.get("blas_threads") or {}
        if any(th.get(v) != "6" for v in THREAD_VARS):
            raise SystemExit(f"REFUSING the final name: the predictions record was not made at the 6-thread pin: {th}")
        want_npz = f"results/phase3_a1_predictions_{A1P.core_of(pair)}_draws.npz"
        if (pdoc.get("draws_npz") or {}).get("file") != want_npz:
            raise SystemExit(f"REFUSING the final name: the predictions record's npz is not {want_npz}")
    if pnpz_sha is None or pnpz_sha != (pdoc.get("draws_npz") or {}).get("sha256"):
        raise SystemExit("REFUSING: the predictions' npz is missing or not the bytes the record names")


def check_predictions_against(S: dict[str, Any], pdoc: dict[str, Any], parr: dict[str, np.ndarray]) -> None:
    """After setup: the same draws (index hash), inputs and variants, and every array equal to this code's
    recomputation from c = 0 data and the geometry, bit for bit."""
    if S["draws_sha256"] != pdoc["config"].get("draws_index_sha256"):
        raise SystemExit("REFUSING: the predictions record names other draws")
    if pdoc.get("input_cache_sha256") != S["input_sha"] or pdoc["config"].get("variants") != S["variants"]:
        raise SystemExit("REFUSING: the predictions record names other inputs or variants")
    mine = A1P.prediction_arrays(S)
    off = sorted(set(mine) ^ set(parr)) + [k for k in mine if k in parr and not (
        mine[k].dtype == parr[k].dtype and mine[k].shape == parr[k].shape and np.array_equal(mine[k], parr[k]))]
    if off:
        raise SystemExit(f"REFUSING: the predictions record's arrays are not this code's on these inputs: {off[:6]}")


def check_record_of_record(rec: dict[str, Any], S: dict[str, Any], registered: bool) -> list[str]:
    """The 4d B record of record is the run of these inputs, probe, selection and draw design (and, for the anchor,
    at the registered draws, seeds and target)."""
    rc = rec.get("config") or {}
    bad = [] if rec.get("registered_parameters") is True else ["not a registered run"]
    want = {"pair": S["pair"], "layer": S["L_sel"], "selected_C": S["C_sel"], "pool_seed": PS.POOL_SEED,
            "selection_sha256": S["sel_sha"]}
    if registered:
        want.update({"n_reps": S["n_reps"], "seeds": S["seeds"], "target_fpr": S["target"]})
    bad += [f"config {k}: {rc.get(k)!r} != {v!r}" for k, v in want.items() if rc.get(k) != v]
    if ("strat_draw_seed" in rc) != S["origin_strat"] or (S["origin_strat"] and rc.get("strat_draw_seed") != PS.STRAT_DRAW_SEED):
        bad.append("another draw design")
    rin = rec.get("input_cache_sha256") or {}
    if set(rin) != {k for k in S["input_sha"] if not k.startswith("surrogate/")} or any(rin[k] != S["input_sha"][k] for k in rin):
        bad.append("other caches")
    return bad


def anchor_check(s0: dict[str, Any], a0: dict[str, np.ndarray], rec: dict[str, Any], dcell: dict[str, Any],
                 X: dict[str, Any], S: dict[str, Any]) -> tuple[int, list[str]]:
    """D-327 (6): (identical values, mismatches) of the c = 0 numbers against the 4d B record of record (FPR, paired
    ratio, FPR/target and TPR summaries of every arm it ran) and the depth record's per-seed AUROCs (the AUC rule)."""
    n, bad = 0, []
    cfgs = rec["configurations"][A1P.PSET]
    for arm, cname in ANCHORED.items():
        for qn in QUANT:
            a = json.dumps(json.loads(json.dumps(s0["arms"][arm][qn], default=str)), sort_keys=True)
            b = json.dumps((cfgs.get(cname) or {}).get(qn), sort_keys=True)
            if a == b:
                n += 1
            else:
                bad.append(f"{arm} ({cname}) {qn}: {a} != {b}")
    au = dcell["auroc"]
    pre_ps = [float(x) for x in X["auroc_pre"]]
    post_ps = [float(x) for x in a0["auroc_frozen_eval|per_seed"][0]]
    m_pre = [M.auroc(S["SC"]["pre"]["pos"][:, j], S["SC"]["pre"]["pool"][X["evneg"], j]).value for j in range(len(S["seeds"]))]
    if pre_ps != au["pre"]["per_seed"] or post_ps != au["frozen"]["per_seed"] or pre_ps != m_pre:
        bad.append(f"frozen eval split AUROC per seed: pre {pre_ps} vs {au['pre']['per_seed']} (metrics.auroc {m_pre}); "
                   f"post {post_ps} vs {au['frozen']['per_seed']}")
    if (a0["auroc_frozen_eval|per_seed"] != a0["auroc_frozen_eval|per_seed"][0]).any():
        bad.append("the c = 0 frozen eval split AUROC differs between draws")
    return n, bad


def d282_cell(S: dict[str, Any], X: dict[str, Any], a: dict[str, np.ndarray], c: float,
              counts: np.ndarray) -> dict[str, Any]:
    """D-282's combined TPR interval at one (variant, c). At c = 0, every arm whose positive scores are fixed across
    draws (D282_ARMS). At c != 0, the arms whose TPR compares the attacked positives with a per-draw threshold
    (D282_SHIFTED): the attacked flag fl(pos + Delta_ds) >= t_ds, Delta the realised delta . v (white-box and transfer
    alike), equals pos >= t'_ds with t'_ds from exact_shifted_threshold (NOT pos >= t_ds - Delta_ds, which differs in
    floating point when a positive ties the threshold-setting score), so the fixed clean positive scores go in with
    t'. Each arm's inputs must
    reproduce its measured TPR draw by draw, else it refuses. The restandardised arms and the refit oracle move their
    positives' scores per draw under the attack (statistics per draw), which the D-282 code does not take: c = 0 only."""
    if "d282_pos" not in X:
        pos = {name: X["POS_scores"] for name in D282_SHIFTED}
        for name, qs in (("restandardise_class_balanced", S["probes"]), ("refit_oracle", X["refits"])):
            pos[name] = np.column_stack([PS.restandardised_scores(q, *PS.feature_stats(X["XF"][j]), X["POS_post"])
                                         for j, q in enumerate(qs)])
        X["d282_pos"] = pos
    out: dict[str, Any] = {}
    for name in (D282_ARMS if c == 0 else D282_SHIFTED):
        thr = a[f"{name}|threshold_per_seed"]
        if c != 0:
            thr = np.array([exact_shifted_threshold(X["d282_pos"][name], a["shift|delta_dot_v_per_seed"][d], thr[d])
                            for d in range(thr.shape[0])])
        tpr_draws = np.array([float(P.rate(X["d282_pos"][name] >= thr[d]).mean()) for d in range(thr.shape[0])])
        if not np.array_equal(tpr_draws, a[f"{name}|tpr"]):
            raise SystemExit(f"REFUSING: {name} at c = {c:g}: the D-282 inputs do not reproduce its TPR draw by draw")
        reps = P.combined_tpr_replicates(thr[1:], S["SC"]["pre"]["pos"], X["d282_pos"][name], counts)
        out[name] = {"tpr_post_combined": P.ci(reps["tpr_post"]),
                     "threshold": "the arm's own per-draw threshold" + (", shifted exactly (exact_shifted_threshold)"
                                                                         if c != 0 else "")}
    return out


def exact_shifted_threshold(pos: np.ndarray, shift: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per seed j: t'_j = the smallest clean positive score p with fl(p + shift_j) >= t_j (+inf if there is none).
    fl(p + s) is non-decreasing in p, so {p : fl(p + s) >= t} = {p : p >= t'}: the clean positives against t' give the
    measured attacked flags fl(pos + shift) >= t bit for bit. (pos >= t - shift does not: t = fl(n + s) for the
    threshold-setting score n, and fl(fl(n + s) - s) != n in a few percent to a quarter of cases, which flips a tied
    positive.)"""
    out = np.full(np.shape(t), np.inf)
    for j in range(len(out)):
        hit = pos[:, j][pos[:, j] + shift[j] >= t[j]]
        if hit.size:
            out[j] = hit.min()
    return out


def run_grid(S: dict[str, Any], X: dict[str, Any], a0: dict[str, np.ndarray], s0: dict[str, Any],
             log: Callable[[str], None] = print, identity: bool = True) -> tuple[dict, dict, dict]:
    """Every variant and c != 0 (c = 0 is a0, shared by the variants)."""
    arrs: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    cells: dict[str, dict[str, Any]] = {}
    ident: dict[str, Any] = {}
    for variant in S["variants"]:
        arrs[variant], cells[variant], ident[variant] = {}, {}, {}
        for c in A1P.C_GRID:
            kc = A1P.ckey(c)
            if c == 0:
                arrs[variant][kc], cells[variant][kc] = a0, s0
                continue
            t1 = time.time()
            arrs[variant][kc] = measure(S, X, variant, c)
            cells[variant][kc] = summaries(arrs[variant][kc], S["target"])
            if identity:
                ident[variant][kc] = linear_identity_check(S, X, variant, c)
            log(f"  {variant} {kc}: frozen FPR/target {cells[variant][kc]['arms']['frozen']['ratio_to_target']['mean']:.4g}"
                f"  ({time.time() - t1:.0f} s)")
    return arrs, cells, ident


def accuracy(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """D-177's accuracy line over ALL cells: the share within x1.5 and the median |log(realised / predicted)|. A cell
    whose prediction is 0 is a miss (D-322 (4)): an infinite error, outside x1.5, inside the median (which is then None
    with median_is_a_miss when it falls on a miss). Beside, the same over the cells with a positive prediction."""
    e = np.array([abs(np.log(r["realised_ratio_to_target"] / r[key])) if r[key] > 0 else np.inf for r in rows])
    ok = e[np.isfinite(e)]
    med = float(np.median(e)) if e.size else None
    fin = lambda x: x if x is not None and np.isfinite(x) else None  # noqa: E731
    return {"n_cells": int(e.size), "n_cells_predicted_zero": int(e.size - ok.size),
            "share_within_factor_1_5": float((e <= np.log(FACTOR)).mean()) if e.size else None,
            "median_abs_log_realised_over_predicted": fin(med),
            "median_is_a_miss": bool(med is not None and not np.isfinite(med)),
            "beside_positive_predictions_only": {
                "n_cells": int(ok.size), "share_within_factor_1_5": float((ok <= np.log(FACTOR)).mean()) if ok.size else None,
                "median_abs_log_realised_over_predicted": float(np.median(ok)) if ok.size else None}}


def confusion(arrs: dict[str, dict[str, np.ndarray]], cs: list[str]) -> dict[str, Any]:
    """P-A1.4 for one variant: per draw, counts over the c != 0 cells of each detector against the realised class
    (D-241 (3), (6)); D-156 mean counts (the pooling unit across pairs) and the per-draw sensitivity and specificity."""
    out: dict[str, Any] = {}
    y = np.array([arrs[c]["realised_class_2x"] > 0 for c in cs])                   # (cells, draws)
    for det, key in (("gaussian", "health|gaussian_flag"), ("direct", "health|direct_flag"), ("auc_rule", "auc_rule|flag")):
        f = np.array([arrs[c][key] > 0 for c in cs])
        cnt = {k: v.sum(axis=0).astype(np.int64) for k, v in
               (("tp", y & f), ("fn", y & ~f), ("fp", ~y & f), ("tn", ~y & ~f))}
        with np.errstate(invalid="ignore", divide="ignore"):
            sens = cnt["tp"] / (cnt["tp"] + cnt["fn"])
            spec = cnt["tn"] / (cnt["tn"] + cnt["fp"])
        out[det] = {"counts": {k: {"d156_mean": float(v[1:].mean()), "frozen_split": int(v[0])} for k, v in cnt.items()},
                    "sensitivity": P.band_summary(sens[1:], sens[0]), "specificity": P.band_summary(spec[1:], spec[0]),
                    "_arrays": cnt}
    return out


def score(S: dict[str, Any], arrs: dict, cells: dict, parr: dict[str, np.ndarray],
          ident: dict[str, Any] | None = None) -> tuple[dict[str, Any], bool, dict]:
    """D-327 (4), (5): every prediction's outcome; the second value is False on a protocol error (P-A1.1, P-A1.5 or
    P-A1.6 failed, or a linear-identity check of `ident` beyond IDENTITY_TOL); the third the P-A1.4 per-draw arrays."""
    target = S["target"]
    nz = [c for c in A1P.C_GRID if c != 0]
    a0 = arrs[S["variants"][0]]["c=0"]
    a1_1 = {"E": 0.0, "frozen_eval_split": 0.0}
    a1_2_bad, a1_5_bad, a1_6_bad, a1_6_worst = [], [], [], 0.0
    for variant in S["variants"]:
        for c in nz:
            a = arrs[variant][A1P.ckey(c)]
            a1_1["E"] = max(a1_1["E"], float(np.max(np.abs(a["auroc_E|per_seed"] - a0["auroc_E|per_seed"]))))
            a1_1["frozen_eval_split"] = max(a1_1["frozen_eval_split"], float(np.max(np.abs(
                a["auroc_frozen_eval|per_seed"] - a0["auroc_frozen_eval|per_seed"]))))
            if not np.array_equal(a["auc_rule|flag"], a0["auc_rule|flag"]):
                a1_2_bad.append(f"{variant} {A1P.ckey(c)}")
            for name in RESTANDARDISED:
                r = np.abs(a[f"{name}|fpr_per_seed"] - a0[f"{name}|fpr_per_seed"]) / np.abs(a0[f"{name}|fpr_per_seed"])
                a1_6_worst = max(a1_6_worst, float(r.max()))
                if (r > REL_TOL).any():
                    a1_6_bad.append(f"{variant} {A1P.ckey(c)} {name}: {int((r > REL_TOL).sum())} draw-seed values")
        for c in A1P.C_GRID:
            for name in THRESHOLD_LEVEL:
                b = cells[variant][A1P.ckey(c)]["arms"][name]["ratio_to_target"]["band95"]
                if not b[0] <= 1.0 <= b[1]:
                    a1_5_bad.append(f"{variant} {A1P.ckey(c)} {name}: band {b}")
    p11 = max(a1_1.values()) <= AUC_TOL
    n_e = len(S["draws"][0]["E"])
    rows: list[dict[str, Any]] = []
    for variant in S["variants"]:
        for c in nz:
            pfa = parr[f"{variant}|{A1P.ckey(c)}|predicted_fpr"]
            pr, prf = pfa.mean(axis=1) / target, np.maximum(pfa, 1.0 / n_e).mean(axis=1) / target
            rows.append({"variant": variant, "c": c,
                         "realised_ratio_to_target": cells[variant][A1P.ckey(c)]["arms"]["frozen"]["ratio_to_target"]["mean"],
                         "predicted_ratio_to_target": float(pr[1:].mean()),
                         "predicted_ratio_to_target_floored": float(prf[1:].mean())})
    p13 = {"cells": "pair x c != 0 x variant (this pair); realised and predicted FPR/target as per-cell D-156 means",
           "primary_unfloored": accuracy(rows, "predicted_ratio_to_target"),
           "beside_floored_1_over_E": accuracy(rows, "predicted_ratio_to_target_floored"),
           "by_variant": {v: {"unfloored": accuracy([r for r in rows if r["variant"] == v], "predicted_ratio_to_target"),
                              "floored": accuracy([r for r in rows if r["variant"] == v], "predicted_ratio_to_target_floored")}
                          for v in S["variants"]},
           "expectation": "at least 80% of cells within x1.5 over all pairs (scored on the pooled cells, not here)",
           "per_cell": rows}
    conf = {v: confusion(arrs[v], [A1P.ckey(c) for c in nz]) for v in S["variants"]}
    scoring = {
        "P-A1.1": {"holds": p11, "max_abs_auroc_change": a1_1, "tolerance": AUC_TOL,
                   "over": "every variant, c != 0, draw and seed"},
        "P-A1.2": {"holds": not a1_2_bad, "cells_whose_flag_changed": a1_2_bad,
                   "flag_share_of_draws_at_c0": float(a0["auc_rule|flag"][1:].mean())},
        "P-A1.3": p13,
        "P-A1.4": {"primary": "white_box", "class": "realised paired ratio >= 2 or <= 0.5", "detector": "gaussian",
                   "per_pair": {v: {det: {kk: vv for kk, vv in blk.items() if kk != "_arrays"} for det, blk in cf.items()}
                                for v, cf in conf.items()},
                   "pooled": "the pair-cluster bootstrap of D-241 (6) over the per-pair D-156 mean counts, across the "
                             "nine pairs: not computed by this per-pair script",
                   "expectation": "sensitivity >= 0.90 and specificity >= 0.85, pooled"},
        "P-A1.5": {"holds": not a1_5_bad, "failures": a1_5_bad, "arms": list(THRESHOLD_LEVEL)},
        "P-A1.6": {"holds": not a1_6_bad, "failures": a1_6_bad, "max_relative_fpr_change": a1_6_worst,
                   "tolerance": REL_TOL, "arms": list(RESTANDARDISED)},
    }
    id_ok = True
    if ident is not None:
        worst = {k: max([float(v[k]) for per in ident.values() for v in per.values()], default=0.0) for k in IDENTITY_TOL}
        bad = {k: x for k, x in worst.items() if not x <= IDENTITY_TOL[k]}
        id_ok = not bad
        scoring["linear_identity"] = {"holds": id_ok, "max_abs": worst, "bounds": IDENTITY_TOL, "beyond_bound": bad,
                                      "over": "every variant and c != 0, on the frozen split (score units)"}
    return scoring, bool(p11 and not a1_5_bad and not a1_6_bad and id_ok), conf


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Phase 3 A1 simulation (D-327)")
    ap.add_argument("--pair", required=True, choices=POP.UPDATE_PAIRS_REPAIR)
    ap.add_argument("--n-reps", type=int, default=A1P.REGISTERED["n_reps"])
    ap.add_argument("--target-fpr", type=float, default=A1P.REGISTERED["target_fpr"])
    ap.add_argument("--seeds", type=int, nargs="*", default=A1P.REGISTERED["seeds"])
    ap.add_argument("--predictions", default=None, help="the predictions record (default: its final name)")
    ap.add_argument("--out", default=None, help="TESTING: a JSON path outside results/")
    ap.add_argument("--anchor-report-only", action="store_true",
                    help="TESTING ONLY: record a c = 0 anchor mismatch and go on (never with the final names)")
    args = ap.parse_args()
    t0 = time.time()
    core = A1P.core_of(args.pair)
    registered = (args.n_reps == A1P.REGISTERED["n_reps"] and args.target_fpr == A1P.REGISTERED["target_fpr"]
                  and list(args.seeds) == A1P.REGISTERED["seeds"])
    final_path = (RESULTS / f"phase3_a1_{core}.json").resolve()
    out_path = Path(args.out).resolve() if args.out else final_path
    npz_path = out_path.with_name(out_path.stem + "_draws.npz")
    final = out_path == final_path
    pred_final = (RESULTS / f"phase3_a1_predictions_{core}.json").resolve()
    pred_path = Path(args.predictions).resolve() if args.predictions else pred_final
    for p in (out_path, npz_path):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists; results are never overwritten")
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test run (--out) must not write under results/")
    threads = {v: os.environ.get(v) for v in THREAD_VARS}
    if final and (not registered or any(t != "6" for t in threads.values()) or pred_path != pred_final
                  or args.anchor_report_only):
        raise SystemExit(f"REFUSING the final name: registered parameters {registered}, threads {threads}, predictions "
                         f"{pred_path.name}, anchor-report-only {args.anchor_report_only}")
    if not pred_path.is_file():                       # D-327 (6): the predictions first
        raise SystemExit(f"REFUSING: no predictions record {pred_path}; run src/phase3_a1_predictions.py first (D-327 (6))")
    code0 = P.code_block(CODE_FILES)
    pred_sha = sha256_file(pred_path)
    pdoc = read_result(pred_path)                     # registered read: listed with its sha256 (D-297 (1))
    pnpz_path = pred_path.with_name(pred_path.stem + "_draws.npz")
    pnpz_b = pnpz_path.read_bytes() if pnpz_path.is_file() else None
    pnpz_sha = sha256_bytes(pnpz_b) if pnpz_b is not None else None
    check_predictions(pdoc, pnpz_sha, code0["sha256"][A1P.THIS_FILE], args.pair, args.n_reps, list(args.seeds),
                      args.target_fpr, final)
    with np.load(io.BytesIO(pnpz_b), allow_pickle=False) as z:          # the very bytes whose sha256 was checked
        parr = {k: z[k] for k in z.files}
    S = A1P.setup(args.pair, args.n_reps, list(args.seeds), args.target_fpr)
    check_predictions_against(S, pdoc, parr)
    rec_rel = f"results/{A1P.record_of_record_stem(core)}.json"
    rec_sha = sha256_file(ROOT / rec_rel)
    rec = read_result(ROOT / rec_rel)
    bad_rec = check_record_of_record(rec, S, registered)
    if bad_rec:
        raise SystemExit(f"REFUSING: {rec_rel} is not the 4d B record of these inputs, probe and draws: {bad_rec}")
    print(f"A1 {args.pair} [{S['axis']}] L{S['L_sel']} C {S['C_sel']:g}: {len(S['draws']) - 1} draws + frozen; variants "
          f"{S['variants']}; predictions {pred_sha[:12]}; 4d B record {rec_rel}; registered {registered}", flush=True)
    X = precompute(S)

    # c = 0, before any attacked score: the self-anchor (4d B) and the AUC rule's anchor (depth_d156)
    a0 = measure(S, X, S["variants"][0], 0.0)
    s0 = summaries(a0, S["target"])
    if registered:
        n_same, bad = anchor_check(s0, a0, rec, S["dcell"], X, S)
        anchor = {"checked": True, "n_values_identical": n_same, "n_values_compared": len(ANCHORED) * len(QUANT),
                  "mismatches": bad, "identical": not bad}
        print(f"  anchor at c = 0: {n_same} of {len(ANCHORED) * len(QUANT)} 4d B summaries identical; "
              f"{'PASS' if not bad else 'FAIL'}", flush=True)
        if bad and not args.anchor_report_only:
            raise SystemExit("REFUSING: the c = 0 numbers are not the 4d B / depth records' (D-327 (6)):\n  " + "\n  ".join(bad[:12]))
    else:
        anchor = {"checked": False, "why": "non-registered draws, seeds or target (a test): the records hold the registered run"}
    anchor.update({"file": rec_rel, "sha256": rec_sha, "configurations": ANCHORED, "quantities": list(QUANT),
                   "auc_rule_source": {"file": S["depth_rel"], "sha256": S["depth_sha"],
                                       "kind": "per-seed AUROC pre and frozen on the frozen eval split (D-241 (4))"}})
    counts = P.counts_matrix(S["n_pos"], D.REGISTERED["n_boot"], D.ITEM_BOOT_SEED["eval_positives"])
    s0["tpr_d282_combined"] = d282_cell(S, X, a0, 0.0, counts)
    arrs, cells, ident = run_grid(S, X, a0, s0, log=lambda m: print(m, flush=True))
    for variant in S["variants"]:
        for c in A1P.C_GRID:
            if c != 0:
                cells[variant][A1P.ckey(c)]["tpr_d282_combined"] = d282_cell(S, X, arrs[variant][A1P.ckey(c)], c, counts)
    scoring, protocol_ok, conf = score(S, arrs, cells, parr, ident)
    for row in scoring["P-A1.3"]["per_cell"]:
        cells[row["variant"]][A1P.ckey(row["c"])]["prediction"] = {**row, "source": "the predictions record"}
    p13 = scoring["P-A1.3"]
    print(f"  P-A1.1 {'holds' if scoring['P-A1.1']['holds'] else 'FAILS'} {scoring['P-A1.1']['max_abs_auroc_change']}; "
          f"P-A1.2 {'holds' if scoring['P-A1.2']['holds'] else 'FAILS'}; P-A1.3 unfloored {p13['primary_unfloored']}, "
          f"floored {p13['beside_floored_1_over_E']}; P-A1.4 white-box gaussian sensitivity "
          f"{conf['white_box']['gaussian']['sensitivity']['mean']} specificity {conf['white_box']['gaussian']['specificity']['mean']}; "
          f"P-A1.5 {'holds' if scoring['P-A1.5']['holds'] else 'FAILS'}; P-A1.6 "
          f"{'holds' if scoring['P-A1.6']['holds'] else 'FAILS'} (max rel {scoring['P-A1.6']['max_relative_fpr_change']:.3g})",
          flush=True)
    if not protocol_ok:
        print("PROTOCOL ERROR: a failed P-A1.1, P-A1.5 or P-A1.6 (D-327 (5)), or a linear-identity check beyond its "
              "bound (an implementation check, D-181), stops the run for diagnosis; nothing written."
              f"\n  P-A1.1 {scoring['P-A1.1']}\n  P-A1.5 {scoring['P-A1.5']['failures'][:12]}"
              f"\n  P-A1.6 {scoring['P-A1.6']['failures'][:12]}\n  identity {scoring.get('linear_identity')}", flush=True)
        return 3

    npz: dict[str, np.ndarray] = {f"c=0|{k}": v for k, v in a0.items()}
    for variant in S["variants"]:
        for kc, a in arrs[variant].items():
            if kc != "c=0":
                npz.update({f"{variant}|{kc}|{k}": v for k, v in a.items()})
    for v, cf in conf.items():
        for det, blk in cf.items():
            npz.update({f"P-A1.4|{v}|{det}|{k}": x for k, x in blk["_arrays"].items()})
    nb = A1P.npz_bytes(npz)
    code1 = P.code_block(CODE_FILES)
    changed = [n for n, a, b in (("code", code1["sha256"], code0["sha256"]), ("selection", sha256_file(S["sel_path"]), S["sel_sha"]),
                                 ("cluster manifest", sha256_file(D.CLUSTERS), S["cl_sha"]),
                                 (S["depth_rel"], sha256_file(ROOT / S["depth_rel"]), S["depth_sha"]),
                                 (rec_rel, sha256_file(ROOT / rec_rel), rec_sha), ("predictions", sha256_file(pred_path), pred_sha),
                                 ("predictions npz", sha256_file(pnpz_path), pnpz_sha),
                                 ("registry sources", D.sources_now(), S["reg_sources"])) if a != b]
    changed += [f"cache {r['file']}" for r in S["input_recs"].values()
                if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    if changed:
        raise SystemExit(f"REFUSING to write: changed during the run: {changed}")
    payload = {**provenance_block("phase3_a1"), "preregistration": PREREG, "code": code0,
               "registered_parameters": registered, "blas_threads": threads, "input_verification": S["verification"],
               "config": {**A1P.input_block(S), "arms": list(ARMS), "health_cut": list(CUT),
                          "anchor_report_only": bool(args.anchor_report_only)},
               "predictions_record": {"file": A1P.rel(pred_path), "sha256": pred_sha,
                                      "draws_npz": {"file": A1P.rel(pnpz_path), "sha256": pdoc["draws_npz"]["sha256"]},
                                      "checks": "same pair, draws (index hash), inputs, parameters, variants and "
                                                "prediction-function sha256; every array equal to this run's "
                                                "recomputation bit for bit"},
               "anchor": anchor, "depth_record": {"file": S["depth_rel"], "sha256": S["depth_sha"]},
               "geometry": {**S["geometry"], "surrogate": S["surrogate"]},
               "linear_identity_check": {"bounds": IDENTITY_TOL, "values": ident},
               "d282_tpr": {"where": "cells.<variant>.<c>.tpr_d282_combined", "arms_at_c0": list(D282_ARMS),
                            "arms_at_c_nonzero": list(D282_SHIFTED),
                            "at_c_nonzero": "the clean positives against the exact shifted threshold t' = min{p : "
                                            "fl(p + delta . v) >= t} per draw and seed (exact because fl(p + s) is "
                                            "non-decreasing in p), reproduced draw by draw",
                            "not_at_c_nonzero": "restandardise_class_balanced, restandardise_traffic_1pct, refit_oracle: "
                                                "their positives move per draw under the attack (the D-156 band; their "
                                                "TPR equals its c = 0 value by translation invariance)",
                            "n_boot": int(counts.shape[0]), "counts_seed": D.ITEM_BOOT_SEED["eval_positives"],
                            "draw_of_replicate": "D-282: replicate r uses draw 1 + ((r - 1) mod n_draws)"},
               "probe_fit": [q.diag for q in S["probes"]], "refit_fit": [q.diag for q in X["refits"]],
               "input_cache_sha256": S["input_sha"],
               "input_cache_source": {kk: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"]}
                                      for kk, r in S["input_recs"].items()},
               "draws_npz": {"file": A1P.rel(npz_path), "sha256": sha256_bytes(nb), "size": len(nb), "keys": sorted(npz),
                             "rows": "draw index (0 the frozen split); per-seed arrays have the seeds as columns"},
               "scoring": scoring, "cells": cells, "runtime_seconds": round(time.time() - t0, 1)}
    for p, hh in A1P.write_pair(out_path, payload, npz_path, nb):
        print(f"Written: {p}\nsha256:  {hh}")
    print(f"runtime {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
