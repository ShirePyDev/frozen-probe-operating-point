"""4a: the Section IV-F intent decomposition under the SELECTED probe, with a variance term.

Pre-registered in D-152 (written before any number here existed). The LoRA dose series of
LORA_ARMS (general alpaca, safety HH-RLHF, safety PKU-SafeRLHF; steps 25/100/400; one base,
one recipe, only the corpus differs) is re-measured with the probe the deployer would pick
(D-117 selection re-run on the A100 base these arms actually use), not the C=1 default.

Per run (9 cells) and layer (selected + [2, 8, 16, 24, 32] for continuity with Table
intent), on the frozen probe fitted per seed 0-4 on the D-117 80% split, in the D-117
permutation order (P1; the selection's own fits):

  paired FPR   post/pre realised FPR at each seed's own 1% threshold on eval NEGATIVES,
               floored at 1/n_neg, exact McNemar, bootstrap CI             (src/paired.py)
  TPR          pre and post at the same frozen threshold on eval POSITIVES, k/n, McNemar, CI
  geometry     d = mean(z_post) - mean(z_pre), z standardised by the probe's own statistics,
               on eval negatives (primary) and on all eval items (continuity); ||d||,
               cos(w, d) and the mean score shift w.d = ||w|| ||d|| cos, all with CIs
  variance     w'Sigma_post w / w'Sigma_pre w on eval negatives = var(post neg scores) /
               var(pre neg scores), its own column with a CI (ddof 0)

Every quantity is reported as the mean over seeds with its range; every CI comes from ONE
resample matrix per item set (eval negatives, eval positives, all eval items) shared by all
runs, so replicate r is the same resample everywhere and between-run CIs are paired.

Between arms at matched dose (safety vs general, 6 comparisons, both probe sets): the log
score-shift gap split as log||d|| + log|cos| in two forms - the mean over seeds of per-seed
logs (exactly additive: the probe is shared) and the log of the ratio of seed means (the
D-105 form) - beside the between-arm variance term, the damage gap log(ratio_s/ratio_g) and
the TPR-change difference. A "share" is emitted only when the two terms have the same sign.
The fraction of replicates in which each run's w.d changes sign is reported; where it is
not small, log|w.d| is not interpretable and the row says so.

C=1 at the selected layer is the sensitivity row (P2) with ALL of the above, fitted in file
order, with iterations, convergence and gradient norm, and a reversed-file-order refit
reporting cos(w_file, w_reversed) and the paired-ratio difference.

Refuses rather than guesses: output exists; selection file ambiguous or not a logistic C;
selection made on other bytes or other seeds; any input cache not the bytes the registry expects;
a required layer missing; non-finite items; code or the registry's sources changed while running.
Runs on the node, CPU, BLAS threads 6, through scripts/detach.sh.

D-216 (6), 2026-09-27: re-run under D-165. The selection of record is
results/selection_sweep_lora_a100_d165.json (D-171: bit-identical to the superseded sweep). The runs'
repos and revisions and every input cache come through src/input_registry.py, not from the superseded
repair_* records. The D-170 input_verification block is written. The output is
results/intent_selected_d165.json; results/intent_selected.json is superseded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import input_registry as REG  # noqa: E402
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
from provenance import (  # noqa: E402
    ensure_utf8_stdout,
    provenance_block,
    resolve_cache,
    sha256_file,
    write_results,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "hidden"
RESULTS = ROOT / "results"
CONTINUITY_LAYERS = [2, 8, 16, 24, 32]
DOSES = (25, 100, 400)
A100_THRESHOLD_CALIB = "abcb4c9ccc3cb6a63bf6da90763c6a0927c9527fbc5c38ccc77d2906fb83510d"  # D-152 4a
CODE_FILES = ["src/intent_selected.py", "src/input_registry.py", "src/paired.py", "src/populations.py",
              "src/provenance.py", "src/superseded_provenance.py"]
SIGN_FLIP_LIMIT = 0.05   # above this fraction of replicates, log|w.d| is flagged uninterpretable


def sources_now() -> dict[str, str]:
    """The registry's sources re-read from disk (original extraction records and committed pins), so that a
    record written during the run is caught at the end (the registry caches its hashes at first load)."""
    out = {f"results/{f.name}": sha256_file(f) for f in sorted(RESULTS.glob("extract__*.json")) if "__rerun_" not in f.name}
    for m in REG.MANIFESTS:
        if m.exists():
            out[str(m.relative_to(ROOT)).replace("\\", "/")] = sha256_file(m)
    return out


def mean_rng(v: np.ndarray) -> dict[str, Any]:
    v = np.asarray(v, dtype=float)
    return {"per_seed": v.tolist(), "mean": float(v.mean()), "range": [float(v.min()), float(v.max())]}


def centred_var(S: np.ndarray, counts: np.ndarray | None = None) -> np.ndarray:
    """Variance per seed (ddof 0), point (S,) or per replicate (B, S). Centred first, in
    float64, so E[s^2] - E[s]^2 does not cancel catastrophically."""
    Sc = (S - S.mean(axis=0)).astype(np.float64)
    if counts is None:
        return Sc.var(axis=0)
    n = S.shape[0]
    m1 = counts @ Sc / n
    return counts @ (Sc ** 2) / n - m1 ** 2


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="4a: intent decomposition on the selected probe")
    ap.add_argument("--selection", default=str(RESULTS / "selection_sweep_lora_a100_d165.json"))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--boot-seed", type=int, default=20260924)
    ap.add_argument("--target-fpr", type=float, default=P.TARGET_FPR)
    ap.add_argument("--out", default=str(RESULTS / "intent_selected_d165.json"))
    args = ap.parse_args()

    # --- identity of everything, BEFORE any computation (review finding) -------------
    out_path = Path(args.out).resolve()
    if out_path.exists():
        raise SystemExit(f"REFUSING: {out_path} exists; results are never overwritten (D-152 P5)")
    final = out_path == (RESULTS / "intent_selected_d165.json").resolve()
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test run (--out) must not write under results/")
    if final:
        import os
        threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        registered = list(args.seeds) == [0, 1, 2, 3, 4] and args.n_boot == 5000 and args.boot_seed == 20260924 \
            and args.target_fpr == P.TARGET_FPR and Path(args.selection).resolve() == (RESULTS / "selection_sweep_lora_a100_d165.json").resolve()
        if not all(t == "6" for t in threads.values()) or not registered:
            raise SystemExit(f"REFUSING the final name: 6-thread pin {threads}, registered parameters {registered} (D-152 P6)")
    sel_path = Path(args.selection).resolve()
    try:
        sel_name = str(sel_path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        sel_name = str(sel_path)
    sel_sha = sha256_file(sel_path)
    code0 = P.code_block(CODE_FILES)

    # --- population and configs (declared, never globbed: F-038) ---------------------
    runs: list[tuple[str, str, int]] = []   # (arm, run stem, dose)
    for arm, stems in POP.LORA_ARMS.items():
        for stem, dose in zip(stems, DOSES):
            if not stem.endswith(f"_s{dose}"):
                raise SystemExit(f"REFUSING: LORA_ARMS order is not 25/100/400: {stem}")
            runs.append((arm, stem, dose))
    # D-165: the declared runs' repos and revisions from the registry (extraction records), not the
    # superseded repair_* records
    cfgs = {stem: REG.run(f"repair_{stem}") for _, stem, _ in runs}
    pre_keys = {(c["pre_repo"], c["pre_rev"], c["axis"]) for c in cfgs.values()}
    if len(pre_keys) != 1:
        raise SystemExit(f"REFUSING: the LoRA runs do not share one pre-update model: {pre_keys}")
    pre_repo, pre_rev, axis = pre_keys.pop()

    # --- the selected probe for THIS base (D-152 4a: D-117 re-run on the A100 base) -----
    sel_doc = json.load(open(sel_path, encoding="utf-8"))
    if len(sel_doc.get("models", [])) != 1:
        raise SystemExit("REFUSING: the selection file must hold exactly one model")
    sel_model = sel_doc["models"][0]
    sel = sel_model["selected"]
    if not sel["arm"].startswith("C="):
        raise SystemExit(f"REFUSING: selection is {sel['arm']!r}, not a logistic C (D-152)")
    C_sel, L_sel = float(sel["arm"].split("=")[1]), int(sel["layer"])
    sel_cfg = REG.run(sel_model["pair"])
    if (sel_cfg["pre_repo"], sel_cfg["pre_rev"]) != (pre_repo, pre_rev):
        raise SystemExit(f"REFUSING: selection made on {sel_cfg['pre_repo']}@{sel_cfg['pre_rev'][:12]}, "
                         f"not {pre_repo}@{pre_rev[:12]}")
    if list(sel_doc["config"]["seeds"]) != list(args.seeds) or float(sel_doc["config"]["holdout"]) != P.HOLDOUT:
        raise SystemExit(f"REFUSING: selection seeds/holdout {sel_doc['config']['seeds']}/"
                         f"{sel_doc['config']['holdout']} differ from {args.seeds}/{P.HOLDOUT}")
    print(f"pre-update model {pre_repo}@{pre_rev[:12]} [{axis}]; selected probe C={C_sel:g}, "
          f"layer {L_sel} (from {sel_name}, sha256 {sel_sha[:16]})")

    # --- caches, and the bytes they must be ---------------------------------------------
    # D-165: every cache through the registry, which checks its bytes against the extraction records or
    # the committed pins; the D-170 flag follows those bytes
    input_sha: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    pre = {}
    for part in ("probe_train", "threshold_calib", "eval"):
        z, r = REG.load(pre_repo, axis, part)
        pre[part] = z
        input_sha[f"pre/{part}"], input_recs[f"pre/{part}"] = r["sha256"], r
    post = {}
    for _, stem, _ in runs:
        z, r = REG.load(cfgs[stem]["post_repo"], axis, "eval")
        post[stem] = z
        input_sha[f"{stem}:post/eval"], input_recs[f"{stem}:post/eval"] = r["sha256"], r
    reg_sources = REG.sources_sha256()
    if sources_now() != reg_sources:
        raise SystemExit("REFUSING: the extraction records or pins on disk are not the ones the registry read")
    verification = REG.verification(input_recs)
    sel_in = sel_doc.get("input_cache_sha256", {})
    checks = [("selection pre/probe_train", input_sha["pre/probe_train"], sel_in.get(f"{sel_model['pair']}:pre/probe_train")),
              ("selection pre/threshold_calib", input_sha["pre/threshold_calib"], sel_in.get(f"{sel_model['pair']}:pre/threshold_calib")),
              ("A100 threshold_calib (D-152)", input_sha["pre/threshold_calib"], A100_THRESHOLD_CALIB)]
    bad = [f"{what}: have {have[:16]}, expected {(want or 'NOT RECORDED')[:16]}" for what, have, want in checks if have != want]
    if bad:
        raise SystemExit("REFUSING: input caches are not the recorded bytes:\n  " + "\n  ".join(bad))
    for part, z in pre.items():
        if "nonfinite" in z.files and z["nonfinite"].any():
            raise SystemExit(f"REFUSING: pre/{part} has non-finite items")
    for stem, z in post.items():
        if not np.array_equal(z["item_ids"], pre["eval"]["item_ids"]):
            raise SystemExit(f"REFUSING: {stem} eval item_ids differ from the pre-update eval")
        if not np.array_equal(z["labels"], pre["eval"]["labels"]):
            raise SystemExit(f"REFUSING: {stem} eval labels differ")
        if "nonfinite" in z.files and z["nonfinite"].any():
            raise SystemExit(f"REFUSING: {stem} post/eval has non-finite items")
    need = sorted({L_sel, *CONTINUITY_LAYERS})
    for name, z in [*((f"pre/{k}", v) for k, v in pre.items()), *((f"{s}:post/eval", v) for s, v in post.items())]:
        have = {int(k.split("_")[1]) for k in z.files if k.startswith("layer_")}
        if set(need) - have:
            raise SystemExit(f"REFUSING: {name} lacks layers {sorted(set(need) - have)} (D-152 requires {need})")
    print(f"{len(checks)} input-identity checks passed; layers {need}")

    y_tr = pre["probe_train"]["labels"].astype(bool)
    y_cal = pre["threshold_calib"]["labels"].astype(bool)
    y_ev = pre["eval"]["labels"].astype(bool)
    neg, pos = ~y_ev, y_ev
    n_neg, n_pos, n_all = int(neg.sum()), int(pos.sum()), int(len(y_ev))
    print(f"eval negatives {n_neg}, positives {n_pos}; calib negatives {int((~y_cal).sum())}; "
          f"seeds {args.seeds}; bootstrap B={args.n_boot}")
    CNT_N = P.counts_matrix(n_neg, args.n_boot, args.boot_seed)
    CNT_P = P.counts_matrix(n_pos, args.n_boot, args.boot_seed + 1)
    CNT_A = P.counts_matrix(n_all, args.n_boot, args.boot_seed + 2)

    def pre_side(probes: list[P.Probe], X_cal: np.ndarray, X_ev: np.ndarray) -> dict[str, Any]:
        return {"T": np.array([P.thr_at_fpr(q.score(X_cal)[~y_cal], args.target_fpr) for q in probes]),
                "S_n": np.column_stack([q.score(X_ev[neg]) for q in probes]),
                "S_p": np.column_stack([q.score(X_ev[pos]) for q in probes]),
                "W": np.column_stack([q.w for q in probes]),
                "SD": np.column_stack([q.sd for q in probes]),
                "diag": [q.diag for q in probes]}

    def measure(ps: dict[str, Any], probes: list[P.Probe], X_ev: np.ndarray, X_post: np.ndarray):
        """One full cell for one probe set on one run at one layer; returns (record, reps)."""
        T, W, SD = ps["T"], ps["W"], ps["SD"]
        S_post_n = np.column_stack([q.score(X_post[neg]) for q in probes])
        S_post_p = np.column_stack([q.score(X_post[pos]) for q in probes])
        F_pre, F_post = ps["S_n"] >= T, S_post_n >= T
        Q_pre, Q_post = ps["S_p"] >= T, S_post_p >= T
        r_rep = P.ratio_from_flags(F_pre, F_post, CNT_N)
        dt_rep = P.delta_tpr_from_flags(Q_pre, Q_post, CNT_P)
        wn = np.linalg.norm(W, axis=0)
        reps: dict[str, np.ndarray] = {"ratio": r_rep, "dtpr": dt_rep}
        geo: dict[str, Any] = {}
        for tag, A_pre, A_post, cnt, n in (("neg", X_ev[neg], X_post[neg], CNT_N, n_neg),
                                           ("all", X_ev, X_post, CNT_A, n_all)):
            D = (A_post.mean(0) - A_pre.mean(0))[:, None] / SD            # (d, S)
            dn, sh = np.linalg.norm(D, axis=0), (W * D).sum(axis=0)
            cs = sh / (wn * dn)
            DXr = cnt @ (A_post - A_pre).astype(np.float32) / n            # (B, d)
            dn_r = np.stack([np.linalg.norm(DXr / SD[:, s], axis=1) for s in range(SD.shape[1])], 1)
            sh_r = np.stack([(DXr / SD[:, s]) @ W[:, s] for s in range(SD.shape[1])], 1)
            cs_r = sh_r / (wn[None, :] * dn_r)
            flip = (np.sign(sh_r) != np.sign(sh)[None, :]).mean(axis=0)
            geo[tag] = {"d_norm": {**mean_rng(dn), **P.ci(dn_r.mean(axis=1))},
                        "cos": {**mean_rng(cs), **P.ci(cs_r.mean(axis=1))},
                        "abs_cos": {**mean_rng(np.abs(cs)), **P.ci(np.abs(cs_r).mean(axis=1))},
                        "score_shift": {**mean_rng(sh), **P.ci(sh_r.mean(axis=1))},
                        "w_norm": mean_rng(wn),
                        "shift_sign_flip_fraction_per_seed": flip.tolist()}
            if tag == "neg":
                reps.update({"abs_shift": np.abs(sh_r), "d_norm": dn_r, "abs_cos": np.abs(cs_r),
                             "flip": flip})
        vr = centred_var(S_post_n) / centred_var(ps["S_n"])
        vr_rep = centred_var(S_post_n, CNT_N) / centred_var(ps["S_n"], CNT_N)     # (B, S)
        reps["vr"] = vr_rep
        point = {"abs_shift": np.abs(np.array(geo["neg"]["score_shift"]["per_seed"])),
                 "d_norm": np.array(geo["neg"]["d_norm"]["per_seed"]),
                 "abs_cos": np.abs(np.array(geo["neg"]["cos"]["per_seed"])), "vr": vr,
                 "ratio": P.ratio_from_flags(F_pre, F_post), "dtpr": P.delta_tpr_from_flags(Q_pre, Q_post)}
        record = {"fpr": P.paired_fpr(ps["S_n"], S_post_n, T), "ratio": P.ci(r_rep),
                  "discordance_neg": P.discordance(ps["S_n"], S_post_n, T),
                  "tpr": P.tpr_pair(ps["S_p"], S_post_p, T), "delta_tpr": P.ci(dt_rep),
                  "geometry": geo,
                  "variance_ratio": {**mean_rng(vr), **P.ci(vr_rep.mean(axis=1))}}
        return record, reps, point

    cells: list[dict[str, Any]] = []
    c1_rows: list[dict[str, Any]] = []
    store: dict[tuple[str, str], tuple[dict, dict]] = {}   # (probe set, stem) -> (reps, point)

    for li in need:
        k = f"layer_{li}"
        X_tr, X_cal, X_ev = pre["probe_train"][k], pre["threshold_calib"][k], pre["eval"][k]
        probes, c1_file, c1_rev = [], [], []
        for seed in args.seeds:
            _, fit = P.d117_split(len(y_tr), seed)
            probes.append(P.fit_probe(X_tr[fit], y_tr[fit], seed, C_sel, "d117_permutation"))
            if li == L_sel:
                fo = np.sort(fit)
                c1_file.append(P.fit_probe(X_tr[fo], y_tr[fo], seed, 1.0, "file_order"))
                c1_rev.append(P.fit_probe(X_tr[fo[::-1]], y_tr[fo[::-1]], seed, 1.0, "reversed_file_order"))
        ps = pre_side(probes, X_cal, X_ev)
        ps1 = pre_side(c1_file, X_cal, X_ev) if li == L_sel else None
        ps1r = pre_side(c1_rev, X_cal, X_ev) if li == L_sel else None
        for arm, stem, dose in runs:
            X_post = post[stem][k]
            record, reps, point = measure(ps, probes, X_ev, X_post)
            cells.append({"arm": arm, "run": stem, "dose": dose, "layer": li,
                          "is_selected_layer": li == L_sel, "probe_C": C_sel,
                          "fit_row_order": "d117_permutation", **record,
                          **({"probe_fit_diagnostics": ps["diag"]} if li == L_sel else {})})
            if li == L_sel:
                store[("selected", stem)] = (reps, point)
                rec1, reps1, point1 = measure(ps1, c1_file, X_ev, X_post)
                recr, _, _ = measure(ps1r, c1_rev, X_ev, X_post)
                store[("C=1", stem)] = (reps1, point1)
                cosw = [float(a.w @ b.w / (np.linalg.norm(a.w) * np.linalg.norm(b.w))) for a, b in zip(c1_file, c1_rev)]
                c1_rows.append({"arm": arm, "run": stem, "dose": dose, "layer": li, "probe_C": 1.0,
                                "fit_row_order": "file_order", **rec1,
                                "probe_fit_diagnostics": ps1["diag"],
                                "reversed_file_order_refit": {
                                    "fpr": recr["fpr"], "tpr": recr["tpr"],
                                    "probe_fit_diagnostics": ps1r["diag"],
                                    "cos_w_file_vs_reversed_per_seed": cosw,
                                    "ratio_file_minus_reversed": rec1["fpr"]["ratio_mean"] - recr["fpr"]["ratio_mean"]}})
        print(f"  layer {li}: done ({len(runs)} runs)", flush=True)

    # --- continuity summary: median over the five continuity layers (Table intent form) --
    continuity = []
    for _, stem, _ in runs:
        cs = [c for c in cells if c["run"] == stem and c["layer"] in CONTINUITY_LAYERS]
        continuity.append({"run": stem, "layers": CONTINUITY_LAYERS, "probe_C": C_sel,
                           "median_d_norm_all_items": float(np.median([c["geometry"]["all"]["d_norm"]["mean"] for c in cs])),
                           "median_abs_cos_all_items": float(np.median([c["geometry"]["all"]["abs_cos"]["mean"] for c in cs])),
                           "median_ratio": float(np.median([c["fpr"]["ratio_mean"] for c in cs]))})

    # --- between arms at matched dose, both probe sets ---------------------------------
    between: list[dict[str, Any]] = []
    for pset in ("selected", "C=1"):
        for arm, stem, dose in runs:
            if arm.startswith("general"):
                continue
            g = next(s for a, s, d in runs if a.startswith("general") and d == dose)
            (rs, ps_), (rg, pg) = store[(pset, stem)], store[(pset, g)]
            row: dict[str, Any] = {"probe_set": pset, "safety_arm": arm, "safety_run": stem,
                                   "general_run": g, "dose": dose, "layer": L_sel}
            for q in ("abs_shift", "d_norm", "abs_cos", "vr"):
                per_seed = np.log(ps_[q]) - np.log(pg[q])
                rep_mlog = (np.log(rs[q]) - np.log(rg[q])).mean(axis=1)
                rep_smean = np.log(rs[q].mean(axis=1)) - np.log(rg[q].mean(axis=1))
                row[f"log_ratio_{q}"] = {
                    "mean_of_per_seed_logs": {"value": float(per_seed.mean()), "per_seed": per_seed.tolist(),
                                              "range": [float(per_seed.min()), float(per_seed.max())], **P.ci(rep_mlog)},
                    "log_of_seed_means": {"value": float(np.log(ps_[q].mean()) - np.log(pg[q].mean())), **P.ci(rep_smean)}}
            m = {q: row[f"log_ratio_{q}"]["mean_of_per_seed_logs"]["value"] for q in ("abs_shift", "d_norm", "abs_cos")}
            row["identity_residual_mean_of_logs"] = float(m["abs_shift"] - m["d_norm"] - m["abs_cos"])  # definitional ~0
            same_sign = np.sign(m["d_norm"]) == np.sign(m["abs_cos"]) and m["d_norm"] != 0
            row["alignment_share"] = (float(m["abs_cos"] / (m["d_norm"] + m["abs_cos"])) if same_sign else None)
            row["alignment_share_note"] = None if same_sign else "not a share: the size and alignment terms have opposite signs"
            fs, fg = float(np.max(rs["flip"])), float(np.max(rg["flip"]))
            row["max_sign_flip_fraction"] = {"safety": fs, "general": fg}
            row["log_shift_interpretable"] = bool(max(fs, fg) <= SIGN_FLIP_LIMIT)
            row["log_ratio_damage"] = {"value": float(np.log(ps_["ratio"]) - np.log(pg["ratio"])),
                                       **P.ci(np.log(rs["ratio"]) - np.log(rg["ratio"]))}
            row["delta_tpr_difference"] = {"value": float(ps_["dtpr"] - pg["dtpr"]), **P.ci(rs["dtpr"] - rg["dtpr"])}
            between.append(row)

    # --- code must be the code that ran; then write, exclusively -----------------------
    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"] or sha256_file(sel_path) != sel_sha:
        raise SystemExit("REFUSING to write: source or selection file changed during the run")
    if sources_now() != reg_sources:
        raise SystemExit("REFUSING to write: the registry's sources changed during the run")
    changed = [r["file"] for r in input_recs.values() if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    if changed:
        raise SystemExit(f"REFUSING to write: input caches changed during the run: {changed}")
    payload = {
        **provenance_block("intent_selected"),
        "preregistration": "D-152 (4a); re-run under D-165 per D-216 (6)",
        "code": code0,
        "input_verification": verification,
        "supersedes": ["results/intent_selected.json"],
        "config": {"selection_file": sel_name, "selection_sha256": sel_sha,
                   "selected_C": C_sel, "selected_layer": L_sel, "layers": need,
                   "seeds": args.seeds, "split": f"D-117 (holdout {P.HOLDOUT})",
                   "selected_fit_row_order": "d117_permutation (the selection's own fits)",
                   "c1_fit_row_order": "file_order, refit reversed_file_order",
                   "target_fpr": args.target_fpr, "n_boot": args.n_boot,
                   "boot_seeds": {"eval_negatives": args.boot_seed, "eval_positives": args.boot_seed + 1,
                                  "all_eval": args.boot_seed + 2},
                   "sign_flip_limit": SIGN_FLIP_LIMIT,
                   "pre_repo": pre_repo, "pre_rev": pre_rev, "axis": axis,
                   **{f"{s}:post_repo": cfgs[s]["post_repo"] for _, s, _ in runs},
                   **{f"{s}:post_rev": cfgs[s]["post_rev"] for _, s, _ in runs},
                   "inputs_from": "src/input_registry.py (D-165)", "input_registry_sources_sha256": reg_sources,
                   "n_eval_negatives": n_neg, "n_eval_positives": n_pos},
        "population": {"declared_in": "src/populations.py", "group": "LORA_ARMS",
                       "runs": [s for _, s, _ in runs]},
        "input_cache_sha256": input_sha,
        "input_cache_source": {k: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"]}
                               for k, r in input_recs.items()},
        "cells": cells, "c1_sensitivity": c1_rows, "continuity_summary": continuity,
        "between_arms": between,
    }
    digest = write_results(out_path, payload, exclusive=True)
    print(f"\nWritten: {out_path}\nsha256:  {digest}")

    # --- printed tables (after the write, so a print error cannot lose the run) --------
    def civ(d: dict, fmt: str = "{:.2f}") -> str:
        c = d.get("ci95")
        return "[" + ", ".join(fmt.format(x) for x in c) + "]" if c else "[undefined]"

    for title, rows in (("SELECTED PROBE", [c for c in cells if c["is_selected_layer"]]), ("C=1 SENSITIVITY", c1_rows)):
        print(f"\n{title} at layer {L_sel}: FPR on {n_neg} eval negatives, TPR on {n_pos} eval positives; "
              f"mean over seeds {{range}} [95% CI]; McNemar = max p over seeds; cens = censored seeds pre/post")
        for c in rows:
            f, t, g = c["fpr"], c["tpr"], c["geometry"]["neg"]
            print(f"  {c['arm']:<22} s{c['dose']:<4} FPR {np.mean(f['n_flagged_pre_per_seed']):.1f}/{n_neg} -> "
                  f"{np.mean(f['n_flagged_post_per_seed']):.1f}/{n_neg}  ratio {f['ratio_mean']:.2f} "
                  f"{{{f['ratio_range'][0]:.2f}-{f['ratio_range'][1]:.2f}}} {civ(c['ratio'])}  "
                  f"p_neg {c['discordance_neg']['mcnemar_exact_p_max']:.1e}  cens {f['n_censored_pre']}/{f['n_censored_post']}")
            print(f"  {'':<22} {'':<5} TPR {np.mean(t['hits_pre_per_seed']):.1f}/{n_pos} -> "
                  f"{np.mean(t['hits_post_per_seed']):.1f}/{n_pos}  dTPR {t['delta_tpr_mean']:+.4f} "
                  f"{civ(c['delta_tpr'], '{:+.4f}')}  p_pos {t['pos_mcnemar_exact_p_max']:.1e}  "
                  f"||d|| {g['d_norm']['mean']:.2f} {civ(g['d_norm'])}  |cos| {g['abs_cos']['mean']:.4f} "
                  f"{civ(g['abs_cos'], '{:.4f}')}  w.d {g['score_shift']['mean']:+.3f}  "
                  f"var ratio {c['variance_ratio']['mean']:.3f} {civ(c['variance_ratio'], '{:.3f}')}")
    print("\nBETWEEN ARMS at matched dose, safety minus general (mean of per-seed logs [95% CI])")
    for b in between:
        s = lambda q: f"{b[q]['mean_of_per_seed_logs']['value']:+.3f} {civ(b[q]['mean_of_per_seed_logs'], '{:+.3f}')}"  # noqa: E731
        print(f"  [{b['probe_set']}] {b['safety_arm']:<22} s{b['dose']:<4} log|w.d| {s('log_ratio_abs_shift')} = "
              f"log||d|| {s('log_ratio_d_norm')} + log|cos| {s('log_ratio_abs_cos')}; log vr {s('log_ratio_vr')}; "
              f"log damage {b['log_ratio_damage']['value']:+.3f} {civ(b['log_ratio_damage'], '{:+.3f}')}"
              + ("" if b["log_shift_interpretable"] else "  [w.d sign unstable: log terms not interpretable]"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
