"""D-440 (6): D-354's two HEADLINE GROUPED CELLS (Phi-3-mini-128k -> Phi-3.5 C=0.001 L18; OLMo-2 base -> SFT C=0.003
L16), each beside its pair's recorded cell; DESCRIPTIVE (a sensitivity row, D-354 (3)); CPU only, no GPU work.

A NEW MODULE (review of cfc738a): src/p2auroc_d424.py is pinned by records and by other modules' REUSED_SHA256
(src/overrefusal_p2_d440.py, src/recalnoise_d433.py: dc628448...), so it is never edited; this module imports it
UNCHANGED and calls its arithmetic (auroc_block, compute_cell, duan_flags, boot_counts), its loaders (load_scores,
cache_plan, presence) and its writers. Its own code block (CODE_FILES) names this file, and its records name it.
bind() here repeats p2auroc_d424.bind's lines with THIS module's code block (paired.code_block refuses a loaded src
module left out of the block, so p2auroc_d424.bind cannot be called from here).

MODES
  grouped --pair K [--records-only] [--out <test path>] [--n-boot B]
                   results/p2auroc_d424_grouped_<K>.json and _scores.npz (GROUPED_CHOICES (g1)-(g6)).
  queue [--write] [--date D]
                   queues/laneA_<D>_d354_grouped.tsv: the Phi-3 cell (sandbox, records-only preflight, job), then
                   OLMo-2 base L16 (src/subset_p2x.py sandbox and cut, scripts/p2_ingest_record.sh, preflight, job).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import float16_check_d364 as F16  # noqa: E402
import p2auroc_d424 as Q  # noqa: E402  (imported, NOT changed: pinned dc628448 by other modules' REUSED_SHA256)
import paired as P  # noqa: E402
import refsize_d382 as RS  # noqa: E402
import selection_p2 as SP  # noqa: E402
import selection_tpr_v2 as T  # noqa: E402
import twomoment_p2_d361 as TP2  # noqa: E402
from provenance import (  # noqa: E402
    _RESULT_READS, canonical_hash, ensure_utf8_stdout, input_identity_problems, provenance_block, read_result,
    sha256_bytes, sha256_file,
)

ROOT = Path(__file__).resolve().parent.parent
OWN = "src/p2auroc_d424_grouped.py"
CODE_FILES = list(dict.fromkeys([OWN, *Q.CODE_FILES]))
# p2auroc_d424.py at the bytes this module was written and tested against (8c792b3), plus the files it pins itself
REUSED_SHA256 = {
    "src/p2auroc_d424.py": "edc7346487c7d9369526602b0cd54176db04ffb9f319c4276ad4ae4f7d1c106f",
    **Q.REUSED_SHA256,
}


def reused_now() -> dict[str, str | None]:
    return {f: (sha256_file(ROOT / f) if (ROOT / f).is_file() else None) for f in REUSED_SHA256}


def final_problems(registered: bool) -> list[str]:
    out = Q.final_problems(registered)
    now = reused_now()
    f = "src/p2auroc_d424.py"
    if now[f] != REUSED_SHA256[f]:
        out.append(f"{f} is not the reviewed bytes this module imports ({str(now[f])[:12]} vs {REUSED_SHA256[f][:12]})")
    return out


def check_final_gates(final: bool, registered: bool) -> None:
    if final:
        bad = final_problems(registered)
        if bad:
            Q.refuse(f"the final names need: {bad}")


def bind(key: str, n_boot: int, final: bool, records_only: bool) -> dict[str, Any]:
    """p2auroc_d424.bind's lines (user-span mean) under THIS module's code block."""
    if key not in Q.PAIRS:
        Q.refuse(f"{key!r} is not one of the nine D-361 headline pairs: {Q.PAIRS}")
    code0 = P.code_block(CODE_FILES)
    with Q.pooled(Q.POOLING):
        b = RS.p2_bind(key, P.N_DRAWS, False, True, code0)
    if b.get("pooling") != Q.POOLING:
        Q.refuse(f"{key}: the binding is for pooling {b.get('pooling')!r}, not {Q.POOLING!r}")
    registered = n_boot == Q.N_BOOT and list(b["seeds"]) == Q.SEEDS and Q.ANCHOR_DRAWS == P.N_DRAWS
    if b.get("beyond") or len(b["layers"]) != Q.LAYERS_PER_PAIR:
        Q.refuse(f"{key}: the cell set is not D-361's {Q.LAYERS_PER_PAIR} layers (L* plus GRID6): layers "
                 f"{list(b['layers'])}, beyond {list(b.get('beyond') or [])}")
    if final:
        bad = Q.p2_final_problems(b, registered) + final_problems(registered)
        if (b.get("d361") or {}).get("registered_parameters") is not True:
            bad.append(f"{b.get('d361_rel')} was not run with D-361's registered parameters")
        if bad:
            Q.refuse(f"the final names need: {sorted(set(bad))}")
    anchor_rec = None if records_only else TP2.passing_anchor(RS.tp2_code_block(code0), final=final)
    plan = Q.cache_plan(b)
    pres, missing = Q.presence(plan, list(b["layers"]))
    pre_bad = sorted({m["cache_key"] for m in missing if m["side"] == "pre"})
    if pre_bad:
        Q.refuse(f"{key}: PRE-side caches absent or short of a layer: {pre_bad}")
    return {"b": b, "code0": code0, "registered": registered, "anchor_rec": anchor_rec, "plan": plan, "pres": pres,
            "missing": missing}


# ----------------------------------------------------------------------------- D-354's grouped cells (D-440 (6))
# D-424 (7), docs/NEXT.md item 8; the author's item 6 of 2026-10-13: the two headline cells D-354's cluster-grouped
# ho20 would have selected, each beside its pair's recorded selection.
GROUPED_RUN_KIND = "p2auroc_d424_grouped"
HO20_SUMMARY = "results/phase2/ho20cluster_summary.json"
HO20_SUMMARY_SHA256 = "83617cddd7369c798f32b6b293a95dd2d555879f67cc8d3710bf37783a56cfd1"    # D-354 (1)
D424_SUMMARY_FILE = "results/p2auroc_d424_summary.json"
D424_SUMMARY_SHA256 = "3073b6855aca53b9fd3768b6c0f5e90df8a7c5a37964d5bfcd93d1673069d7a1"    # logs/pulls/pull_20261010a
# the two D-354 grouped cells of the headline (D-354 (2): Phi-3-mini-128k usermean 0.003 -> 0.001 at L18; OLMo-2 base
# usermean L14 -> L16); each must also be the ho20cluster summary's grouped cell for the pair's selection record
D354_GROUPED = {"phi3_128k_to_35": {"arm": "C=0.001", "layer": 18},
                "olmo2_base_sft": {"arm": "C=0.003", "layer": 16}}
EXPLICIT_TAG = "p2subx"            # src/subset_p2x.py EXPLICIT_TAG: an explicit-layer subset (D-430 (8), D-440 (6))
GROUPED_PREREG = ("D-440 (6) (the author's item 6 of 2026-10-13: 'report the two D-354 grouped cells (Phi-3, OLMo-2 "
                  "L16)'); D-424 (7) (Phi-3 C=0.001 L18 computable on CPU, anchored on the selection record's grid "
                  "entry; OLMo-2 base L16 needs an ingest first); D-430 (8) (the L16 route); D-354 (3) (a sensitivity "
                  "row: the selections of record stand); D-424's arithmetic, intervals and flags unchanged")
GROUPED_CHOICES = [
    "(g1) CELL: the pair's D-354 grouped cell (D354_GROUPED), which must equal the grouped cell that "
    "results/phase2/ho20cluster_summary.json (sha256 pinned, D-354 (1)) records for the pair's user-span-mean selection "
    "record, with that record's own recorded cell = the binding's selection (C, L*). User-span mean only (the two "
    "headline cells); the eight last-token or tokenizer-pair grouped cells are not this mode's.",
    "(g2) PROBES AND THE GRID ANCHOR: at the grouped layer, the five probes at the grouped C are fitted by "
    "twomoment_p2_d361.fit_probes (as at L*) and must equal selection_tpr_v2.seed_probes at the grouped arm bit for bit; "
    "their fit and ho20 AUROCs, recomputed in selection_p2.sweep's expressions (the seed's permutation, the fitting "
    "part's standardisation, regularisation_sweep.auroc, the mean over seeds), must EQUAL the selection record's "
    "grid[arm][layer] (auroc_fit, auroc_ho20, gap) after the JSON round trip (D-424 (7)). Nothing is written unless "
    "they do.",
    "(g3) THE RECORDED CELL BESIDE, AS AN ANCHOR: the pair's recorded cell (the selection's C at L*) is recomputed "
    "by the pair mode's own lines (load_scores with its L* probe anchor, compute_cell with its D-361 operating-point "
    "anchor) and its AUROC block, Duan flags, beside rows and score hashes must equal the pair's D-424 record's L* cell "
    "(the record bound through the D-424 summary, sha256 pinned). This ties the loading, the v2 gather and the "
    "bootstrap counts of this run to the committed table.",
    "(g4) THE GROUPED CELL: the v2 frozen evaluation split's AUROC block (pre, post, change, the same item-bootstrap "
    "counts as the recorded cell, so the two cells' intervals use the same resamples), Duan's flags, and the realised "
    "D-156 operating point at 1% over D-361's 1000 draws (paired.repartition_estimate, the D-361 report's keys: paired "
    "ratio, ratio to target, TPRs, delta TPR, mean, band and frozen split). No committed record holds this cell's "
    "operating point, so (g2) and (g3) are its anchors; no prediction, no verdict (D-354 (3): reported beside).",
    "(g5) A LAYER THE PRE SIDE'S p2sub CACHES LACK (OLMo-2 base L16, D-424 (7)): the pre side's eval and pool4d "
    "entries are read from the explicit-layer subset record results/phase2/extract__<org>__<model>__p2subx@<rev12>__"
    "rest.json (src/subset_p2x.py; ingested by scripts/p2_ingest_record.sh), which must say run_kind "
    "subset_p2x, tag p2subx, ingest yes, be cut from the same rest extraction record (file and sha256) as the p2sub "
    "record, and hold the layer; each entry checked by selection_tpr_v2.subset_entry against its own ingest index. "
    "Absent or not ingested: the job refuses (never a GPU job, never NVMe scratch). The post side, the calibration and "
    "probe_train caches must hold the layer already (they do for L16: the post's p2sub plan holds SFT -> DPO's L* 16 "
    "and the select sets are full grid); otherwise it refuses.",
    "(g6) OUTPUTS: results/p2auroc_d424_grouped_<pair>.json and _scores.npz (keys 'recorded|L<l>|...' and "
    "'grouped|L<l>|...' in the pair mode's form), written once (staged, then named, never overwritten); final names "
    "need the pair mode's gates. D-424's own records and summaries are not touched or re-read for any value but the "
    "anchor (g3).",
]


def grouped_out_paths(key: str, out: str | Path | None) -> tuple[Path, Path, bool]:
    final_path = (Q.RESULTS / f"p2auroc_d424_grouped_{key}.json").resolve()
    out_path = Path(out).resolve() if out else final_path
    final = out_path == final_path
    if not final and out_path.is_relative_to(Q.RESULTS.resolve()):
        Q.refuse("a test run (--out) must not write under results/; only this job's own final name may")
    npz_path = out_path.with_name(out_path.stem + "_scores.npz")
    if npz_path.exists() and not out_path.exists():
        Q.refuse(f"{npz_path} exists without {out_path.name}: the orphan of a failed write; move it aside (never delete "
               "it) and rerun")
    for p in (out_path, npz_path):
        if p.exists() or p.is_symlink():
            Q.refuse(f"{p} exists; results are never overwritten (D-152 P5)")
    return out_path, npz_path, final


def read_pinned(rel_: str, want: str) -> dict[str, Any]:
    p = ROOT / rel_
    if p.is_symlink() or not p.is_file():
        Q.refuse(f"{rel_} is missing (or not a regular file)")
    doc = read_result(p)
    if _RESULT_READS[Q.rel(p)] != want:
        Q.refuse(f"{rel_} is not the pinned bytes ({_RESULT_READS[Q.rel(p)][:16]} vs {want[:16]})")
    return doc


def d354_row(sel_rel: str) -> dict[str, Any]:
    """The ho20cluster summary's row for one selection record (CHOICES (g1))."""
    ho = read_pinned(HO20_SUMMARY, HO20_SUMMARY_SHA256)
    rows = [r for r in ho.get("records") or [] if r.get("selection_record") == sel_rel]
    if len(rows) != 1:
        Q.refuse(f"{HO20_SUMMARY} holds {len(rows)} rows for {sel_rel}")
    return rows[0]


def d424_cell(key: str, layer: int) -> tuple[dict[str, Any], dict[str, str]]:
    """The pair's D-424 record's cell at `layer`, the record bound through the pinned D-424 summary (CHOICES (g3))."""
    s = read_pinned(D424_SUMMARY_FILE, D424_SUMMARY_SHA256)
    r_ = f"results/p2auroc_d424_{key}.json"
    want = ((s.get("records") or {}).get(r_) or {}).get("sha256")
    if not isinstance(want, str):
        Q.refuse(f"{D424_SUMMARY_FILE} names no record {r_}")
    doc = read_pinned(r_, want)
    cells = [c for c in doc.get("cells") or [] if c.get("layer") == layer]
    if len(cells) != 1 or not cells[0].get("filled"):
        Q.refuse(f"{r_} holds no filled cell at L{layer}")
    return cells[0], {"file": r_, "sha256": want}


def check_grouped_cell(key: str, b: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    want = D354_GROUPED[key]
    if row.get("selected_cell_unchanged") is not False:
        Q.refuse(f"{key}: the ho20cluster summary says the selected cell is unchanged under grouping")
    if Q.canon(row.get("selected_grouped")) != Q.canon(want):
        Q.refuse(f"{key}: the ho20cluster summary's grouped cell {row.get('selected_grouped')} is not D-354's {want}")
    if Q.canon(row.get("selected_recorded")) != Q.canon({"arm": b["arm"], "layer": b["L_sel"]}):
        Q.refuse(f"{key}: the ho20cluster summary's recorded cell {row.get('selected_recorded')} is not the binding's "
               f"selection ({b['arm']}, L{b['L_sel']})")
    if not want["arm"].startswith("C="):
        Q.refuse(f"{key}: the grouped arm {want['arm']} is not a logistic arm")
    grid = (((b["sr"].get("sel") or {}).get("model") or {}).get("grid") or {})
    entry = (grid.get(want["arm"]) or {}).get(str(want["layer"]))
    if not isinstance(entry, dict) or not all(isinstance(entry.get(k), float) for k in ("auroc_fit", "auroc_ho20",
                                                                                           "gap")):
        Q.refuse(f"{b['sr']['sel_rel']} holds no grid entry for {want['arm']} at L{want['layer']}")
    return {"arm": want["arm"], "C": float(want["arm"].split("=")[1]), "layer": int(want["layer"]), "grid_entry": entry}


def grid_anchor(X: Any, y: np.ndarray, probes: list[dict[str, Any]], holdout: float,
                entry: dict[str, Any]) -> dict[str, Any]:
    """CHOICES (g2): the selection's grid values of one (arm, layer) recomputed from the selection's own probes in
    selection_p2.sweep's expressions; refuses unless every value is equal after the JSON round trip."""
    n = len(y)
    fits, hos = [], []
    for p in probes:
        rng = np.random.default_rng(p["seed"])
        perm = rng.permutation(n)
        n_ho = int(holdout * n)
        ho, fit = perm[:n_ho], perm[n_ho:]
        xf_s = (X[fit] - p["mu"]) / p["sd"]
        xh_s = (X[ho] - p["mu"]) / p["sd"]
        fits.append(SP.RS.auroc(xf_s @ p["w"] + p["b"], y[fit]))
        hos.append(SP.RS.auroc(xh_s @ p["w"] + p["b"], y[ho]))
    f, h = float(np.mean(fits)), float(np.mean(hos))
    mine = json.loads(json.dumps({"auroc_fit": f, "auroc_ho20": h, "gap": f - h}))
    diffs = [f"{k}: recomputed {mine[k]!r}, recorded {entry.get(k)!r}" for k in mine if mine[k] != entry.get(k)]
    if diffs:
        raise SystemExit(f"ANCHOR FAILED (the grid entry): {diffs}; nothing written")
    return {"recomputed": mine, "recorded": {k: entry[k] for k in mine}, "values_compared": len(mine),
            "values_differing": 0, "exact_equal": True}


def explicit_subset(b: dict[str, Any], layer: int) -> dict[str, Any] | None:
    """CHOICES (g5): None when the pre side's p2sub eval and pool4d hold the layer; else the explicit-layer subset
    record's binding ({sb, rel, sha256, entries}); refuses when it is absent or not ingested, or when any other cache of
    the cell lacks the layer."""
    pooling = b["pooling"]
    lacks = []
    if layer not in (b["sr"]["rec"].get("layers_resolved") or []):
        lacks.append("the pre side's select record (probe_train, calibration)")
    post_cal = (b["post_sel"]["rec"].get("layers_resolved") if b["post_sel"]["kind"] == "p2tpl"
                else (b["post_cal_entry"] or {}).get("layers_resolved")) or []
    if layer not in post_cal:
        lacks.append("the post side's calibration cache")
    for part in ("eval", "pool4d"):
        if layer not in (b["s_sub"]["post"][part].get("layers_resolved") or []):
            lacks.append(f"the post side's {part} subset")
    if lacks:
        Q.refuse(f"L{layer} is not held by {lacks}; only the pre side's eval and pool4d may come from an explicit-layer "
               "subset (CHOICES (g5))")
    if all(layer in (b["s_sub"]["pre"][part].get("layers_resolved") or []) for part in ("eval", "pool4d")):
        return None
    sb0 = b["sb"]["pre"]
    om = b["pre_repo"].replace("/", "__")
    rev = b["sr"]["rev"]
    p = SP.P2 / f"extract__{om}__{EXPLICIT_TAG}@{rev[:12]}__rest.json"
    if p.is_symlink() or not p.is_file():
        Q.refuse(f"the pre side's p2sub caches lack L{layer} and there is no explicit-layer subset record {Q.rel(p)}: cut it "
               f"with src/subset_p2x.py --explicit-layers {layer} and ingest it with scripts/p2_ingest_record.sh first "
               "(D-430 (8); no GPU work)")
    rec = read_result(p)
    want = {"run_kind": "subset_p2x", "cache_tag": EXPLICIT_TAG, "cache_repo": f"{b['pre_repo']}__{EXPLICIT_TAG}",
            "repo": b["pre_repo"], "revision": rev, "cache_revision": rev, "set_name": "rest", "ingest": "yes"}
    got = {k: rec.get(k) for k in want}
    if got != want:
        Q.refuse(f"{Q.rel(p)}: {[f'{k}={got[k]!r}' for k in want if got[k] != want[k]]}, expected {want}")
    src = rec.get("source_record") or {}
    if (src.get("file"), src.get("sha256")) != (sb0["rest_rel"], _RESULT_READS[sb0["rest_rel"]]):
        Q.refuse(f"{Q.rel(p)} was not cut from {sb0['rest_rel']} as the p2sub record was")
    index_path = SP.INGEST_INDEX / f"{p.stem}.sha256"
    if index_path.is_symlink() or not index_path.is_file():
        Q.refuse(f"{Q.rel(p)} is not ingested (no {Q.rel(index_path)}): run scripts/p2_ingest_record.sh first")
    ip, idx = SP.ingest_index(p)
    sbx = {"path": p, "rel": Q.rel(p), "rec": rec, "rest_path": sb0["rest_path"], "rest_rel": sb0["rest_rel"],
           "rest": sb0["rest"], "index_path": ip, "index": idx}
    entries = {part: T.subset_entry(sbx, part, pooling, layer) for part in ("eval", "pool4d")}
    return {"sb": sbx, "rel": Q.rel(p), "sha256": _RESULT_READS[Q.rel(p)],
            "index": {"file": Q.rel(ip), "sha256": sha256_file(ip)}, "entries": entries}


def grouped_scores(b: dict[str, Any], g: dict[str, Any], xsub: dict[str, Any] | None,
                   log: Callable[[str], None] = print) -> dict[str, Any]:
    """load_scores' section for ONE layer at the grouped C (its loaders called unchanged), with the grid anchor
    (CHOICES (g2)) in place of the L* TPR anchor and the pre side's eval and pool4d from the explicit-layer subset
    when given (CHOICES (g5)). Returns {sc, grid_anchor, caches, gathered, probe_diagnostics}."""
    pooling, sr, seeds, li = b["pooling"], b["sr"], list(b["seeds"]), g["layer"]
    v2, ev2, sb, post_sel = b["v2"], b["ev2"], b["sb"], b["post_sel"]
    z_tr, r_tr, _ = SP.load_ingested(sr["rec"], sr["rec_rel"], b["index_pre"][1], "probe_train", pooling)
    z_cal: dict[str, Any] = {}
    r_cal: dict[str, Any] = {}
    z_cal["pre"], r_cal["pre"], _ = SP.load_ingested(sr["rec"], sr["rec_rel"], b["index_pre"][1], "calibration", pooling)
    recorded = sr["sel"].get("input_cache_sha256") if isinstance(sr["sel"].get("input_cache_sha256"), dict) else {}
    for part, r in (("probe_train", r_tr), ("calibration", r_cal["pre"])):
        k_ = f"pre/{SP.AXIS}/{part}/{pooling}"
        if recorded.get(k_) != r["sha256"]:
            Q.refuse(f"{r['file']} ({r['sha256'][:16]}) is not the cache {sr['sel_rel']} read for {k_}")
    if post_sel["kind"] == "p2tpl":
        z_cal["post"], r_cal["post"], _ = SP.load_ingested(post_sel["rec"], post_sel["rel"], post_sel["index"],
                                                           "calibration", pooling)
    else:
        z_cal["post"], _, _, r_cal["post"] = TP2.open_sub_cache(post_sel, b["post_cal_entry"], "calibration", pooling,
                                                                [li])
    z_sub: dict[str, dict[str, Any]] = {}
    r_sub: dict[str, dict[str, dict[str, Any]]] = {}
    meta_sub: dict[str, dict[str, tuple[list[str], list[int]]]] = {}
    for side in ("pre", "post"):
        z_sub[side], r_sub[side], meta_sub[side] = {}, {}, {}
        for part in ("eval", "pool4d"):
            sbx, ent = ((xsub["sb"], xsub["entries"][part]) if (side == "pre" and xsub) else
                        (sb[side], b["s_sub"][side][part]))
            zz, ids, labs, r = TP2.open_sub_cache(sbx, ent, part, pooling, [li])
            z_sub[side][part], r_sub[side][part], meta_sub[side][part] = zz, r, (ids, labs)
    cal_meta = {side: ([str(x) for x in z_cal[side]["item_ids"]], [int(x) for x in z_cal[side]["labels"]])
                for side in ("pre", "post")}
    if cal_meta["pre"] != cal_meta["post"] or any(meta_sub["pre"][p_] != meta_sub["post"][p_]
                                                  for p_ in ("eval", "pool4d")):
        Q.refuse("v1 calibration, eval or pool4d: the pre and post caches hold other items or labels")
    y_tr = z_tr["labels"].astype(bool)
    if sr["sel"]["model"].get("n_probe_train") != len(y_tr):
        Q.refuse(f"probe_train holds {len(y_tr)} items, {sr['sel_rel']} says {sr['sel']['model'].get('n_probe_train')}")
    k = f"layer_{li}"
    probes = TP2.fit_probes(z_tr[k], y_tr, seeds, g["C"])
    sel_probes = T.seed_probes(z_tr[k], y_tr, g["arm"], seeds, b["holdout"])
    off = [q.diag.get("C") for q, s in zip(probes, sel_probes)
           if not (q.mu.dtype == s["mu"].dtype and q.mu.tobytes() == s["mu"].tobytes()
                   and q.sd.dtype == s["sd"].dtype and q.sd.tobytes() == s["sd"].tobytes()
                   and np.asarray(q.w).dtype == np.asarray(s["w"]).dtype
                   and np.asarray(q.w).tobytes() == np.asarray(s["w"]).tobytes() and q.b == s["b"])]
    if off or len(sel_probes) != len(probes):
        raise SystemExit(f"ANCHOR FAILED: {len(off)} of {len(probes)} probes at {g['arm']} L{li} are not "
                         "selection_tpr_v2.seed_probes' fit bit for bit; nothing written")
    ga = grid_anchor(z_tr[k], y_tr, sel_probes, b["holdout"], g["grid_entry"])
    log(f"  GRID ANCHOR OK at {g['arm']} L{li}: probes bit for bit; auroc_fit, auroc_ho20, gap equal to "
        f"{sr['sel_rel']}")
    mats: dict[str, dict[str, np.ndarray]] = {}
    gathered: dict[str, Any] = {}
    y_cal_v1 = np.asarray(cal_meta["pre"][1]) == 1
    if y_cal_v1.all():
        Q.refuse("v1 calibration holds no negative")
    for side in ("pre", "post"):
        sources = {"calibration": (z_cal[side][k], *cal_meta[side]),
                   "eval": (z_sub[side]["eval"][k], *meta_sub[side]["eval"]),
                   "pool4d": (z_sub[side]["pool4d"][k], *meta_sub[side]["pool4d"])}
        Xc, Xc_pos, cc, _ = F16.gather_split("calibration", v2["ids"], [0] * v2["n"], sources)
        Xn, Xp, cn, cp = F16.gather_split("eval", ev2["ids"], ev2["labels"], sources)
        if Xc_pos.shape[0] or cc != v2["plan_counts"] or cn != ev2["expect_neg"] or cp != ev2["expect_pos"]:
            Q.refuse(f"{side} layer {li}: v2 items per v1 split (calibration {cc}, eval negatives {cn}, eval positives "
                   f"{cp}) are not the v2 record's plan")
        mats[side] = {"cal": Xc, "evn": Xn, "pos": Xp}
        gathered[side] = {"calibration": cc, "eval_negatives": cn, "eval_positives": cp}
        del sources
    sc = TP2.score_matrices(probes, {"cal_pre": mats["pre"]["cal"], "ev_pre": mats["pre"]["evn"],
                                     "cal_post": mats["post"]["cal"], "ev_post": mats["post"]["evn"],
                                     "pos_pre": mats["pre"]["pos"], "pos_post": mats["post"]["pos"]})
    del mats
    for zz in (z_tr, *z_cal.values(), *(z for d in z_sub.values() for z in d.values())):
        zz.close()
    caches = {f"pre/{SP.AXIS}/probe_train/{pooling}": r_tr, f"pre/{SP.AXIS}/calibration/{pooling}": r_cal["pre"],
              b["post_cal_key"]: r_cal["post"]}
    for side in ("pre", "post"):
        for part in ("eval", "pool4d"):
            tag = EXPLICIT_TAG if (side == "pre" and xsub) else T.SUB_TAG
            caches[f"{side}/{tag}/{SP.AXIS}/{part}/{pooling}"] = r_sub[side][part]
    log(f"  grouped cell {g['arm']} L{li} scored")
    return {"sc": sc, "grid_anchor": ga, "caches": caches, "gathered": gathered,
            "probe_diagnostics": [q.diag for q in probes]}


def grouped_cell(key: str, g: dict[str, Any], sc: dict[str, np.ndarray], draws: np.ndarray, CNT_P: np.ndarray,
                 CNT_N: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """CHOICES (g4): the grouped cell's realised operating point (no record to anchor it: reported) and AUROC block."""
    report, _ = P.repartition_estimate(sc["cal_pre"], sc["ev_pre"], sc["cal_post"], sc["ev_post"], sc["pos_pre"],
                                       sc["pos_post"], draws, Q.TARGET)
    au, reps = Q.auroc_block(sc, CNT_P, CNT_N, f"{key} grouped {g['arm']} L{g['layer']}")
    flags = Q.duan_flags(au["pre"]["mean_over_seeds"], au["frozen"]["mean_over_seeds"], au["change"]["mean_over_seeds"],
                       au["change"]["ci95"])
    li = g["layer"]
    cell = {"pair": key, "role": "D-354 grouped cell (cluster-grouped ho20's selection; a sensitivity row, D-354 (3))",
            "arm": g["arm"], "probe_C": g["C"], "layer": li, "filled": True,
            "n_eval_negatives": int(sc["ev_pre"].shape[0]), "n_eval_positives": int(sc["pos_pre"].shape[0]),
            "auroc": au, "duan_flags": flags,
            "realised_d156_at_1pct": {"source": "this job's realised D-156 report at 1% over D-361's 1000 draws (the "
                                                "D-361 estimator); no committed record holds this cell", **report},
            "scores_sha256": {k: sha256_bytes(np.ascontiguousarray(sc[k], dtype="<f8").tobytes()) for k in Q.SCORE_KEYS}}
    arrays = {f"grouped|L{li}|scores|{k}": np.ascontiguousarray(sc[k], dtype="<f8") for k in Q.SCORE_KEYS}
    arrays.update({f"grouped|L{li}|auroc|{name}": v for name, v in reps.items()})
    return cell, arrays


def run_grouped(key: str, n_boot: int = Q.N_BOOT, out: str | Path | None = None, records_only: bool = False,
                log: Callable[[str], None] = print) -> Path:
    """One of D-354's two headline grouped cells beside its pair's recorded cell (GROUPED_CHOICES)."""
    t0 = time.time()
    if key not in D354_GROUPED:
        Q.refuse(f"{key!r} has no D-354 headline grouped cell; the two are {sorted(D354_GROUPED)}")
    out_path, npz_path, final = grouped_out_paths(key, out)
    check_final_gates(final, n_boot == Q.N_BOOT and Q.ANCHOR_DRAWS == P.N_DRAWS)
    ctx = bind(key, n_boot, final, records_only)
    b, code0, registered, missing = ctx["b"], ctx["code0"], ctx["registered"], ctx["missing"]
    if missing:
        Q.refuse(f"{key}: caches absent ({sorted({m['cache_key'] for m in missing})}); the grouped mode computes no "
               "partial cell")
    L_sel = b["L_sel"]
    row = d354_row(b["sr"]["sel_rel"])
    g = check_grouped_cell(key, b, row)
    xsub = explicit_subset(b, g["layer"])
    d424c, d424_ref = d424_cell(key, L_sel)
    if records_only:
        log(f"RECORDS OK: grouped {key} (records only; no cache read, nothing written): recorded {b['arm']} L{L_sel}, "
            f"grouped {g['arm']} L{g['layer']}" + (f"; pre eval/pool4d from {xsub['rel']}" if xsub else ""))
        return out_path
    log(f"D-440 (6) grouped {key}: {b['pre_repo']} -> {b['post_repo']}; recorded {b['arm']} L{L_sel}, grouped "
        f"{g['arm']} L{g['layer']}; B {n_boot}; registered {registered}")
    meta = {"L_sel": L_sel, "g6": list(b["g6"]), "C": b["C_sel"]}
    loaded = Q.load_scores(b, [L_sel], log)
    if sorted(loaded["sc"]) != [L_sel]:
        Q.refuse(f"the loaders scored layers {sorted(loaded['sc'])}, not L* {L_sel}")
    CNT_N, CNT_P = Q.boot_counts(b["n_ev"], b["n_pos"], n_boot)
    count_sha = {"eval_negatives": sha256_bytes(CNT_N.tobytes()), "eval_positives": sha256_bytes(CNT_P.tobytes())}
    draws = P.repartition_draws(b["n_cal"], b["n_ev"], Q.ANCHOR_DRAWS, TP2.P2_DRAW_SEED)
    rec_cell, rec_arr = Q.compute_cell(key, L_sel, loaded["sc"][L_sel], b["d361_cells"].get(L_sel), draws, CNT_P, CNT_N,
                                     meta)
    keys_cmp = ("auroc", "duan_flags", "beside_d156_at_1pct", "scores_sha256", "grid_position", "probe_C")
    off = [k for k in keys_cmp if Q.canon(rec_cell.get(k)) != Q.canon(d424c.get(k))]
    if off:
        Q.refuse(f"the anchor (g3): {key} L{L_sel}: the recomputed recorded cell differs from {d424_ref['file']} in "
               f"{off}; nothing written")
    log(f"  RECORDED CELL OK: {b['arm']} L{L_sel} equals the D-361 record and {d424_ref['file']}")
    gl = grouped_scores(b, g, xsub, log)
    g_cell, g_arr = grouped_cell(key, g, gl["sc"], draws, CNT_P, CNT_N)
    g_cell["grid_anchor"] = gl["grid_anchor"]
    g_cell["probe_fit_diagnostics"] = gl["probe_diagnostics"]
    del draws, CNT_N, CNT_P
    arrays = {f"recorded|{k}": v for k, v in rec_arr.items()}
    arrays.update(g_arr)
    if P.code_block(CODE_FILES)["sha256"] != code0["sha256"]:
        Q.refuse("source changed during the run")
    caches = {**loaded["caches"], **gl["caches"]}
    files_sha = {T.V2_CAL_FILE: sha256_file(SP.SPLIT_ROOT / T.V2_CAL_FILE),
                 F16.V2_EVAL_FILE: sha256_file(SP.SPLIT_ROOT / F16.V2_EVAL_FILE)}
    files_sha.update({k: sha256_file(p) for k, p in b["index_read"].items()})
    moved = [f"{k} (the v2 record names {w[:16]})" for k, w in ((T.V2_CAL_FILE, b["v2"]["sha256"]),
                                                                (F16.V2_EVAL_FILE, b["ev2"]["sha256"]))
             if files_sha[k] != w]
    moved += [k for k, h in b["index_sha0"].items() if files_sha[k] != h]
    if xsub:
        files_sha[xsub["index"]["file"]] = sha256_file(ROOT / xsub["index"]["file"])
        if files_sha[xsub["index"]["file"]] != xsub["index"]["sha256"]:
            moved.append(xsub["index"]["file"])
    moved += TP2.end_hash_problems({f"data/hidden/{r['file']}": (SP.CACHE / r["file"], r["sha256"])
                                    for r in caches.values()})
    moved += [m for m, h in b["manifests"].items() if sha256_file(RS.ROOT / m) != h]
    if final:
        moved += TP2.end_hash_problems({k: (TP2.resolve(k), h) for k, h in _RESULT_READS.items()})
    if moved:
        Q.refuse(f"changed during the run: {moved}")
    files_sha.update({k: h for k, h in _RESULT_READS.items()})
    sr, post_sel = b["sr"], b["post_sel"]
    config = {"item": "D-440 (6): D-354's grouped cell beside the recorded cell, Phase 2 AUROC and operating point; "
                      "descriptive (a sensitivity row, D-354 (3))",
              "population": "phase2_headline", "pair": key, "pair_group": b["group"], "pair_note": b["note"],
              "pre_repo": b["pre_repo"], "pre_rev": sr["rev"], "post_repo": b["post_repo"], "post_rev": post_sel["rev"],
              "pooling": b["pooling"], "axis": SP.AXIS, "split_version": 2,
              "selection_record": {"file": sr["sel_rel"], "sha256": _RESULT_READS[sr["sel_rel"]]},
              "recorded_cell": {"arm": b["arm"], "C": b["C_sel"], "layer": L_sel},
              "grouped_cell": {"arm": g["arm"], "C": g["C"], "layer": g["layer"]},
              "d354_row": {"file": HO20_SUMMARY, "sha256": HO20_SUMMARY_SHA256, "row": row},
              "seeds": list(b["seeds"]), "holdout": b["holdout"],
              "n_eval_negatives": b["n_ev"], "n_eval_positives": b["n_pos"], "n_calibration": b["n_cal"],
              "item_bootstrap": {"n_boot": n_boot, "seeds": Q.ITEM_BOOT_SEED, "counts": "paired.counts_matrix(n, B, seed)",
                                 "counts_sha256": count_sha, "rule": Q.choice("c")},
              "operating_point": {"target_fpr": Q.TARGET, "n_draws": Q.ANCHOR_DRAWS, "draw_seed": TP2.P2_DRAW_SEED},
              "duan_criteria": {"big_drop": Q.BIG_DROP, "operational_failure": Q.OPERATIONAL, "eligible": Q.ELIGIBLE,
                                "note": Q.DUAN_NOTE},
              "explicit_subset": ({"file": xsub["rel"], "sha256": xsub["sha256"], "index": xsub["index"]}
                                  if xsub else None),
              "join_rule": b["v2"]["join_rule"], "post_select_set": post_sel["kind"], "blas_threads": Q.blas_threads()}
    npz_bytes = Q.savez_bytes(arrays)
    payload = {**provenance_block(GROUPED_RUN_KIND), "preregistration": GROUPED_PREREG, "code": code0,
               "reused_files_sha256": reused_now(), "registered_parameters": registered,
               "status": Q.status_block(registered, []), "elapsed_s": round(time.time() - t0, 1),
               "implementation_choices": GROUPED_CHOICES, "pair_mode_choices": Q.CHOICES,
               "config": config, "config_hash": canonical_hash(config),
               "anchor": {"d361_anchor_record": ctx["anchor_rec"],
                          "d361_pair_record": {"file": b["d361_rel"], "sha256": b["d361_sha"]},
                          "probe_anchor_at_L_star": loaded["probe_anchor"],
                          "recorded_cell_equals": {"d361_operating_point": True, "d424_record": d424_ref,
                                                   "keys_compared": list(keys_cmp)},
                          "grid_anchor_at_grouped_cell": gl["grid_anchor"], "manifests": b["manifests"]},
               "input_verification": TP2.D170_P2, "bos_check": b["bos"],
               "v2_gather": {"recorded": loaded["gathered"], "grouped": gl["gathered"]},
               "inputs": dict(_RESULT_READS), "input_files_sha256": files_sha,
               "ingest_indexes_sha256": {"at_read": b["index_sha0"], "unchanged_at_end": True},
               "input_cache_sha256": {k: r["sha256"] for k, r in caches.items()},
               "input_cache_source": {k: f"data/hidden/{r['file']} (expected bytes from {r['expected_from']})"
                                      for k, r in caches.items()},
               "caches_rehashed_at_end": True,
               "scores_npz": {"file": Q.rel(npz_path), "sha256": sha256_bytes(npz_bytes), "size": len(npz_bytes),
                              "keys": "'recorded|' + the pair mode's keys (NPZ_KEYS) of the recorded cell; "
                                      "'grouped|L<layer>|scores|...' and 'grouped|L<layer>|auroc|...' of the grouped "
                                      "cell, in the same form"},
               "cells": {"recorded": rec_cell, "grouped": g_cell}}
    problems = input_identity_problems(payload)
    if problems:
        Q.refuse(f"to write (D-297 (1) results-writer rule): {problems}")
    TP2.write_outputs([(npz_path, npz_bytes), (out_path, Q.json_bytes_of(payload))])
    a, r_ = g_cell["auroc"], g_cell["realised_d156_at_1pct"]
    log(f"  grouped {g['arm']} L{g['layer']}: AUROC pre {a['pre']['mean_over_seeds']:.4f} post "
        f"{a['frozen']['mean_over_seeds']:.4f} change {a['change']['mean_over_seeds']:+.4f} {a['change']['ci95']}; "
        f"paired ratio {r_['paired_ratio']['mean']:.3f}; elapsed {payload['elapsed_s']} s")
    return out_path


GROUPED_LANE = "A"
OLMO_BASE = "allenai/OLMo-2-1124-7B"
OLMO_BASE_REV12 = "7df9a82518af"                           # the base's p2tpl rest record (D-430 (8))
P2CACHE_DIR = f"{Q.SANDBOX_ROOT}/p2cache"                    # the never-ingested full-grid extraction's caches (D-430 (8))
P2SUBX_DIR = f"{Q.SANDBOX_ROOT}/p2subx"                      # the explicit-layer subset's caches before the ingest
EXPLICIT_REC = f"results/phase2/extract__{OLMO_BASE.replace('/', '__')}__{EXPLICIT_TAG}@{OLMO_BASE_REV12}__rest.json"
EXPLICIT_REASON = ("D-354 (2) grouped cell OLMo-2 base -> SFT C=0.003 L16; D-424 (7), D-430 (8), D-440 (6)")


def grouped_outputs(key: str) -> str:
    return f"results/p2auroc_d424_grouped_{key}.json results/p2auroc_d424_grouped_{key}_scores.npz"


def grouped_lane_jobs(date: str) -> list[tuple[str, str, str]]:
    """D-440 (6)'s list: the Phi-3 grouped cell (computable now: a sandbox of the new job kind, a records-only
    preflight, the job), then OLMo-2 base L16 (a sandbox of src/subset_p2x.py, the cut, the ingest, a records-only
    preflight, the job). A failure stops the lane (fail-closed)."""
    if not (date.isascii() and date.isdigit()):
        Q.refuse(f"the sandbox dirs need a plain date, not {date!r}")
    sb_g = f"{Q.SANDBOX_ROOT}/d354_grouped_sandbox_{date}"
    sb_s = f"{Q.SANDBOX_ROOT}/d354_subset_sandbox_{date}"
    phi, olmo = "phi3_128k_to_35", "olmo2_base_sft"
    sub = (f"python3 -u src/subset_p2x.py --repo {OLMO_BASE} --explicit-layers {D354_GROUPED[olmo]['layer']} "
           f"--poolings usermean --explicit-reason \"{EXPLICIT_REASON}\" --src-dir {P2CACHE_DIR}")
    return [
        (f"d354g_sandbox_{phi}", "-",
         f"test ! -e {sb_g} && mkdir -p {sb_g} && python3 -u src/p2auroc_d424_grouped.py grouped --pair {phi} --out "
         f"{sb_g}/p2auroc_d424_grouped_{phi}.json"),
        (f"d354g_preflight_{phi}", "-", f"python3 -u src/p2auroc_d424_grouped.py grouped --pair {phi} --records-only"),
        (f"d354g_{phi}", grouped_outputs(phi), f"python3 -u src/p2auroc_d424_grouped.py grouped --pair {phi}"),
        ("d354g_subset_sandbox_olmo2_base", "-",
         f"test ! -e {sb_s} && mkdir -p {sb_s}/caches && {sub} --out-dir {sb_s}/caches --out {sb_s}/subset_record.json"),
        ("d354g_subset_olmo2_base", EXPLICIT_REC, f"{sub} --out-dir {P2SUBX_DIR}"),
        ("d354g_ingest_olmo2_base", "-", f"bash scripts/p2_ingest_record.sh {EXPLICIT_REC} {P2SUBX_DIR}"),
        (f"d354g_preflight_{olmo}", "-", f"python3 -u src/p2auroc_d424_grouped.py grouped --pair {olmo} --records-only"),
        (f"d354g_{olmo}", grouped_outputs(olmo), f"python3 -u src/p2auroc_d424_grouped.py grouped --pair {olmo}"),
    ]


GROUPED_HEAD = [
    f"# Lane {GROUPED_LANE} (CPU only, no network, no GPU; D-440 (6), the author's item 6 of 2026-10-13: 'report the "
    "two D-354 grouped cells (Phi-3, OLMo-2 L16)'; DESCRIPTIVE, a sensitivity row beside the selections of record, "
    "D-354 (3)): src/p2auroc_d424_grouped.py grouped (GROUPED_CHOICES) for Phi-3-mini-128k -> Phi-3.5 C=0.001 L18 and OLMo-2 "
    "base -> SFT C=0.003 L16, each beside its pair's recorded cell (which must reproduce the D-361 record and the "
    "D-424 record's L* cell), the grouped probes anchored on the selection record's grid entry (D-424 (7)).",
    "# DO NOT LOAD before review: the D-440 (6) entry committed; ONE sonnet review of the grouped mode and of "
    "src/subset_p2x.py (new modules; p2auroc_d424.py and subset_p2.py are pinned and unchanged; docs/NEXT.md items 7-8) and its fix round; the push that carries both, their "
    "tests and this list; the lane's chain rule (D-438 (1): nice 10, at most two CPU lanes at the 6-thread pin). The "
    "sandbox jobs (F-081/F-082 as lane jobs, D-427) write only under ~/.cache/gccf_scratch and are never read; a "
    "same-date relaunch fails at 'test ! -e' (move the old dir aside).",
    "# ORDER: (1) Phi-3: sandbox, records-only preflight, the job (results/p2auroc_d424_grouped_phi3_128k_to_35.json). "
    "(2) OLMo-2 base L16 (D-430 (8)): a sandbox cut by src/subset_p2x.py into scratch, the cut "
    f"({EXPLICIT_REC}, caches to {P2SUBX_DIR}), the ingest (scripts/p2_ingest_record.sh, the NAS-write lock, headroom "
    "checked), a records-only preflight (binds the ingested p2subx record), the job. F-089: ~/.cache was WIPED by the "
    f"node restart of 2026-10-11; if the source caches in {P2CACHE_DIR} are gone, the subset sandbox refuses "
    "('is not a regular file') and the lane stops after the Phi-3 cell: the L16 route then needs a GPU re-extraction "
    "of the base model's rest set (the author's call, D-424 (7)).",
    "# EXPECTED STOP: the OLMo-2 part is expected to REFUSE at d354g_subset_sandbox_olmo2_base, because its source "
    f"caches ({P2CACHE_DIR}, D-430 (8)) were lost in the node restart (F-089); the lane stops there after the Phi-3 "
    "record is written; L16 then needs a GPU re-extraction of the base model's rest set, the author's decision.",
    "# PRICE (estimates, from committed records): D-424's pair jobs on the node took 419.7 s (Phi-3, 7 layers) and "
    "505.8 s (OLMo-2 base -> SFT, 7 layers) (elapsed_s in results/p2auroc_d424_<pair>.json); a grouped job computes 2 "
    "of those 7 layers (the recorded L* and the grouped cell: probes, the 1000-draw report, the B = 5000 bootstrap) "
    "but opens and hashes the cell's caches twice (once per cell), so ESTIMATED at most about the pair job's time "
    "each (under 10 min), the Phi-3 part with its sandbox under 30 min. The cut reads the base's usermean eval, pool4d and over_refusal source "
    "caches (3.39 GB, 0.56 GB, 0.13 GB uncompressed by the rest record) and writes one layer of each: about 0.21 GB "
    "(the D-325 record's 7-layer compressed sizes 1.247 GB, 0.204 GB, 0.043 GB divided by 7); the sandbox cut the "
    "same again on scratch. NAS: the ingest about 0.21 GB; the two grouped records about 2 MB.",
]


def grouped_list_text(date: str) -> str:
    return "\n".join(GROUPED_HEAD + ["\t".join(j) for j in grouped_lane_jobs(date)]) + "\n"


def run_grouped_queue(write: bool = False, date: str | None = None, root: Path | None = None) -> dict[str, str]:
    root = root or ROOT
    date = date or datetime.now(timezone.utc).strftime("%Y%m%d")
    name = f"lane{GROUPED_LANE}_{date}_d354_grouped.tsv"
    text = grouped_list_text(date)
    bad = Q.label_conflicts(name, text, root)
    if bad:
        Q.refuse(f"labels not unique: {bad}")
    if write:
        p = root / "queues" / name
        if p.exists():
            Q.refuse(f"{p} exists")
        p.write_text(text, encoding="utf-8", newline="\n")
        print(f"Written: {p}")
    return {name: text}


# ----------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="D-440 (6): D-354's headline grouped cells (Phase 2 AUROC and operating "
                                             "point beside the recorded cell)")
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("grouped")
    s.add_argument("--pair", required=True, choices=sorted(D354_GROUPED))
    s.add_argument("--n-boot", type=int, default=Q.N_BOOT)
    s.add_argument("--out", default=None)
    s.add_argument("--records-only", action="store_true")
    s = sub.add_parser("queue")
    s.add_argument("--write", action="store_true")
    s.add_argument("--date", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    a = build_parser().parse_args(argv)
    if a.mode == "grouped":
        run_grouped(a.pair, a.n_boot, out=a.out, records_only=a.records_only, log=lambda m: print(m, flush=True))
    elif a.mode == "queue":
        for name, t in run_grouped_queue(a.write, a.date).items():
            if not a.write:
                print(f"== {name}\n{t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
