"""Item 9's frozen split, re-derived for the author (2026-09-27): the frozen thresholds and the XSTest scores
behind one cell of results/overrefusal_d156_<pair>.json, which stores only the per-seed counts.

For the pair and layer given (default: Gemma-2 at its D-117 layer, 18), at the D-117 selected C: the five seed
probes are re-fitted exactly as overrefusal_d156.py fits them (same caches through src/input_registry.py, same
D-117 split, rows and C), each seed's FROZEN threshold is paired.thr_at_fpr on that probe's scores of the
threshold_calib negatives (the frozen split = draw 0 of D-156; target 1% FPR), and the 250 safe and 200
contrast XSTest prompts are scored pre and post. A prompt is flagged when its score (the probe's logit,
standardised features @ w + b) is >= the threshold.
ANCHORS (refuses to write otherwise):
  - every input cache's sha256 is the one the item-9 file read, re-hashed at the end;
  - the probe-fit diagnostics (iterations, gradient norm, weight norm) equal the stored ones exactly;
  - the per-seed counts of safe and contrast prompts flagged pre and post, and their discordant pairs, equal
    the frozen_split counts the item-9 file stores for that cell;
  - at the same thresholds, the Aegis eval negatives flagged pre and the eval positives hit pre equal the
    frozen-split counts of the 4b file item 9 was checked against (twomoment_<pair>.json).
Writes results/overrefusal_frozen_scores_<pair>_L<layer>.npz (scores, thresholds, item ids) and
results/overrefusal_frozen_thresholds_<pair>_L<layer>.json (the threshold record; names the npz and its sha256),
staged and linked together. Never overwritten. results/ only at the 6-thread BLAS pin item 9 ran at.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import input_registry as REG  # noqa: E402
import overrefusal_d156 as OR  # noqa: E402
import paired as P  # noqa: E402
from overrefusal_d156 import PAIRS, sources_now  # noqa: E402
from provenance import (  # noqa: E402
    commit_staged, ensure_utf8_stdout, provenance_block, resolve_cache, sha256_bytes, sha256_file, stage_bytes,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
# everything overrefusal_d156 loads is loaded here too (paired.code_block refuses an unlisted src module)
CODE_FILES = ["src/overrefusal_frozen_rederive.py", *OR.CODE_FILES]
MANIFEST = ROOT / "logs" / "overrefusal_d156_20260926.sha256"
LEVEL = "item_level_D156"


def rel(p: Path) -> str:
    return str(p.resolve().relative_to(ROOT.resolve())).replace("\\", "/") if p.resolve().is_relative_to(ROOT.resolve()) else str(p)


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Re-derive item 9's frozen thresholds and XSTest scores for one cell")
    ap.add_argument("--pair", default="overrefusal_gemma2", choices=list(PAIRS))
    ap.add_argument("--layer", type=int, default=None, help="default: the pair's D-117 selected layer")
    ap.add_argument("--out-dir", default=str(RESULTS))
    args = ap.parse_args()
    stem = args.pair[len("overrefusal_"):]
    threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    out_dir = Path(args.out_dir).resolve()
    final = out_dir == RESULTS.resolve()
    if final and not all(t == "6" for t in threads.values()):
        raise SystemExit(f"REFUSING: results/ only at the 6-thread BLAS pin item 9 ran at: {threads}")
    if not final and out_dir.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test output must not be under results/")
    code0 = P.code_block(CODE_FILES)

    rel9 = f"results/overrefusal_d156_{stem}.json"
    b9 = (ROOT / rel9).read_bytes()
    man = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, f = line.split(maxsplit=1)
            man[f.strip().lstrip("*")] = h
    if man.get(rel9) != sha256_bytes(b9):
        raise SystemExit(f"REFUSING: {rel9} is not the file {MANIFEST.name} names")
    d9 = json.loads(b9.decode("utf-8"))
    if final and code0["libraries"] != d9["code"]["libraries"]:
        raise SystemExit(f"REFUSING: libraries {code0['libraries']} are not item 9's {d9['code']['libraries']}")
    code_differs = sorted(f for f, h in d9["code"]["sha256"].items() if code0["sha256"].get(f) != h)
    cfg9 = d9["config"]
    layer = cfg9["selected_layer"] if args.layer is None else args.layer
    if layer not in cfg9["layers"]:
        raise SystemExit(f"REFUSING: L{layer} is not in {rel9}'s layers {cfg9['layers']}")
    cell9 = next(c for c in d9["cells"] if c["layer"] == layer and c["probe_set"] == "selected")
    fz9 = cell9[LEVEL]["frozen_split"]
    C_sel, seeds, target = float(cfg9["selected_C"]), list(cfg9["seeds"]), float(cfg9["target_fpr"])
    if cell9["probe_C"] != C_sel:
        raise SystemExit("REFUSING: the stored cell's C is not the file's selected C")
    npz_path = out_dir / f"overrefusal_frozen_scores_{stem}_L{layer}.npz"
    js_path = out_dir / f"overrefusal_frozen_thresholds_{stem}_L{layer}.json"
    for p in (npz_path, js_path):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists; never overwritten")
    tm_rel = d9["twomoment_check"]["file"]
    tm_b = (ROOT / tm_rel).read_bytes()
    if sha256_bytes(tm_b) != d9["twomoment_check"]["sha256"]:
        raise SystemExit(f"REFUSING: {tm_rel} is not the 4b file item 9 was checked against")
    tm_cell = next(c for c in json.loads(tm_b.decode("utf-8"))["cells"] if c["layer"] == layer and c["probe_set"] == "selected")
    tm_counts = tm_cell["realised"]["frozen_split_counts"]

    reg_stem = cfg9["registry_pair"]
    cfg = REG.run(reg_stem)
    if (cfg["pre_repo"], cfg["pre_rev"], cfg["post_repo"], cfg["post_rev"]) != \
            (cfg9["pre_repo"], cfg9["pre_rev"], cfg9["post_repo"], cfg9["post_rev"]):
        raise SystemExit(f"REFUSING: the registry's {reg_stem} is not the pair {rel9} ran")
    if d9.get("xstest_cache_tag"):
        raise SystemExit(f"REFUSING: {rel9} read tagged XSTest caches; this script reads the pinned ones")
    axis = cfg["axis"]
    data: dict[str, Any] = {}
    input_sha: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    for side, ax, part in (("pre", axis, "probe_train"), ("pre", axis, "threshold_calib"), ("pre", axis, "eval"),
                           ("pre", "over_refusal", "eval"), ("post", "over_refusal", "eval")):
        z, r = REG.load(cfg[f"{side}_repo"], ax, part)
        key = f"{side}/{ax}/{part}"
        if r["sha256"] != d9["input_cache_sha256"][key]:
            raise SystemExit(f"REFUSING: {key} is not the cache {rel9} read")
        data[key], input_sha[key], input_recs[key] = z, r["sha256"], r
    reg_sources = REG.sources_sha256()
    if sources_now() != reg_sources:
        raise SystemExit("REFUSING: the extraction records or pins on disk are not the ones the registry read")
    verification = REG.verification(input_recs)
    xa, xb = data["pre/over_refusal/eval"], data["post/over_refusal/eval"]
    if not np.array_equal(xa["item_ids"], xb["item_ids"]) or not np.array_equal(xa["labels"], xb["labels"]):
        raise SystemExit("REFUSING: XSTest items or labels differ between pre and post")
    k = f"layer_{layer}"
    tr, cal, ev = data[f"pre/{axis}/probe_train"], data[f"pre/{axis}/threshold_calib"], data[f"pre/{axis}/eval"]
    y_tr = tr["labels"].astype(bool)
    ncal = ~cal["labels"].astype(bool)
    y_ev = ev["labels"].astype(bool)
    x_con = xa["labels"].astype(bool)
    x_safe = ~x_con

    probes = []
    for seed in seeds:
        _, fit = P.d117_split(len(y_tr), seed)
        probes.append(P.fit_probe(tr[k][fit], y_tr[fit], seed, C_sel, "d117_permutation"))
    diag_equal = [q.diag == s for q, s in zip(probes, cell9["probe_fit_diagnostics"])]
    if not all(diag_equal):
        raise SystemExit(f"REFUSING: the re-fitted probes' diagnostics are not item 9's (per seed: {diag_equal})")
    S = lambda x: np.column_stack([q.score(x) for q in probes])  # noqa: E731
    cal_neg = S(cal[k][ncal])
    T = np.array([P.thr_at_fpr(cal_neg[:, s], target) for s in range(len(seeds))])
    sa, sb = S(xa[k]), S(xb[k])
    Sp, Sq, Cp, Cq = sa[x_safe] >= T, sb[x_safe] >= T, sa[x_con] >= T, sb[x_con] >= T
    got = {"safe_flagged_pre": Sp.sum(axis=0).tolist(), "safe_flagged_post": Sq.sum(axis=0).tolist(),
           "contrast_flagged_pre": Cp.sum(axis=0).tolist(), "contrast_flagged_post": Cq.sum(axis=0).tolist(),
           "safe_discordant_b_c": [[int(x), int(y)] for x, y in zip((Sp & ~Sq).sum(axis=0), (~Sp & Sq).sum(axis=0))],
           "contrast_discordant_b_c": [[int(x), int(y)] for x, y in zip((Cp & ~Cq).sum(axis=0), (~Cp & Cq).sum(axis=0))],
           "n_safe": int(x_safe.sum()), "n_contrast": int(x_con.sum())}
    anchor = {q: {"rederived": got[q], "stored": fz9[q], "equal": got[q] == fz9[q]} for q in got}
    # scored as item 9 and 4b scored them: the negative and positive subsets separately (BLAS rows are not bit-identical by position)
    ev_neg, ev_pos = S(ev[k][~y_ev]), S(ev[k][y_ev])
    got4b = {"n_flagged_pre": (ev_neg >= T).sum(axis=0).tolist(), "hits_pre": (ev_pos >= T).sum(axis=0).tolist()}
    anchor4b = {q: {"rederived": got4b[q], "stored": tm_counts[q], "equal": got4b[q] == tm_counts[q]} for q in got4b}
    for s, seed in enumerate(seeds):
        print(f"seed {seed}: frozen threshold {float(T[s])!r}  safe flagged pre {got['safe_flagged_pre'][s]}/{got['n_safe']} "
              f"post {got['safe_flagged_post'][s]}/{got['n_safe']}  (stored {fz9['safe_flagged_pre'][s]} -> "
              f"{fz9['safe_flagged_post'][s]}); Aegis eval negatives flagged pre {got4b['n_flagged_pre'][s]} "
              f"(4b file {tm_counts['n_flagged_pre'][s]})", flush=True)
    if not all(v["equal"] for v in anchor.values()) or not all(v["equal"] for v in anchor4b.values()):
        raise SystemExit(f"REFUSING to write: the re-derived counts are not the stored ones: {anchor} {anchor4b}")

    arrays = {"thresholds": T, "seeds": np.array(seeds), "calibration_negative_scores": cal_neg,
              "calibration_negative_item_ids": np.array([str(i) for i in cal["item_ids"][ncal]]),
              "xstest_item_ids": np.array([str(i) for i in xa["item_ids"]]), "xstest_labels": xa["labels"].astype(np.int8),
              "xstest_scores_pre": sa, "xstest_scores_post": sb}
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    npz_bytes = buf.getvalue()
    code1 = P.code_block(CODE_FILES)
    changed = [n for n, a, b in (("code", code1["sha256"], code0["sha256"]), ("item-9 file", sha256_file(ROOT / rel9), sha256_bytes(b9)),
                                 ("4b file", sha256_file(ROOT / tm_rel), sha256_bytes(tm_b)),
                                 ("registry sources (re-read from disk)", sources_now(), reg_sources)) if a != b]
    changed += [f"cache {r['file']}" for r in input_recs.values() if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    if changed:
        raise SystemExit(f"REFUSING to write: changed during the run: {changed}")
    payload = {**provenance_block("overrefusal_frozen_rederive"), "purpose": "the author's re-derivation of item 9's frozen split (2026-09-27)",
               "code": code0, "blas_threads": threads, "input_verification": verification,
               "item9_file": {"file": rel9, "sha256": sha256_bytes(b9), "manifest": rel(MANIFEST),
                              "input_verification_flag": d9["input_verification"]["flag"], "code": d9["code"],
                              "code_files_differing_from_item9": code_differs},
               "pair": args.pair, "registry_pair": reg_stem, "layer": layer, "probe_C": C_sel, "seeds": seeds,
               "target_fpr": target, "split": "frozen (draw 0 of D-156): thresholds on the threshold_calib negatives",
               "n_calibration_negatives": int(ncal.sum()),
               "flag_rule": "score >= threshold; score = the probe's logit on its fit-set-standardised features",
               "frozen_thresholds_per_seed": T.tolist(),
               "counts": got, "anchor_to_item9": anchor,
               "anchor_to_4b": {"file": tm_rel, "sha256": sha256_bytes(tm_b), "counts": anchor4b},
               "probe_fit_diagnostics_identical_to_item9": diag_equal,
               "probe_fit_diagnostics": [q.diag for q in probes],
               "input_cache_sha256": input_sha,
               "input_cache_source": {k2: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"]}
                                      for k2, r in input_recs.items()},
               "input_registry_sources_sha256": reg_sources,
               "score_file": {"file": rel(npz_path), "sha256": sha256_bytes(npz_bytes), "bytes": len(npz_bytes),
                              "keys": sorted(arrays), "rows": "xstest_*: cache order (labels: 1 contrast, 0 safe); columns: seeds"}}
    json_bytes = json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8")
    json.loads(json_bytes.decode("utf-8"))
    staged: list[Path] = []
    try:
        t_npz, h_npz = stage_bytes(npz_path, npz_bytes)
        staged.append(t_npz)
        t_json, h_json = stage_bytes(js_path, json_bytes)
        staged.append(t_json)
        commit_staged(t_npz, npz_path, h_npz)
        commit_staged(t_json, js_path, h_json)
    finally:
        for t in staged:
            if t.exists():
                t.unlink()
    print(f"Written: {rel(npz_path)}\nsha256:  {h_npz}\nWritten: {rel(js_path)}\nsha256:  {h_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
