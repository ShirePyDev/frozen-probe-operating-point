"""Read-only diagnostics of the frozen splits after D-155 (author, 2026-09-25).

D-155: on the Aegis partition the pre-update baseline FPR at the selected probe lies
below its no-update resplit null in 6 of 8 cells; on WildGuardMix it sits at the centre.
Two questions, before any decision about the splits:

DIAGNOSTIC 1 - near-duplicate clusters over each axis's BENIGN pool (threshold_calib
negatives + eval negatives, 1000 + 2000): Jaccard >= 0.5 on the word 3-gram shingles used
by src/paraphrase_overlap.py (D-107). The Jaccard is computed EXACTLY for all pairs (a
sparse intersection count), which is the quantity MinHash estimates, so no pair is missed
by LSH banding (with 64 permutations in 16 bands, banding misses about 36% of pairs at
J = 0.5). Clusters are connected components of the J >= 0.5 graph. Reported: cluster
count, size distribution, split composition, and, per model cell, whether the TOP-50
calibration negatives by pre-update score (the ones that set a 1% threshold) fall in
clusters, against the base rate over all calibration negatives.

DIAGNOSTIC 2 - per model cell at the selected probe: calibration vs eval negative score
quantiles (deciles; 95th-99th percentiles) and a two-sample KS test, per seed, to tell a
tail effect from a shift of the whole distribution.

Cells as in src/baseline_null.py (the 4a LoRA base and the nine released pairs), probe as
D-152 P1. Frozen split files are checked against results/dataset_manifest*.json first.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
from paraphrase_overlap import shingles  # noqa: E402
from provenance import ensure_utf8_stdout, provenance_block, read_jsonl, resolve_cache, sha256_file, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "hidden"
FROZEN = ROOT / "data" / "frozen"
RESULTS = ROOT / "results"
CODE_FILES = ["src/split_diagnostics.py", "src/paired.py", "src/paraphrase_overlap.py",
              "src/populations.py", "src/provenance.py", "src/superseded_provenance.py"]
J_MIN = 0.5
TOP_K = 50
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]


def load(repo: str, rev: str, axis: str, part: str):
    p = resolve_cache(f"{repo.replace('/', '__')}@{rev[:12]}__{axis}__{part}__mean.npz", CACHE)
    if not p.exists():
        raise FileNotFoundError(f"missing cache {p.name}")
    return np.load(p, allow_pickle=False), p


def recorded_split_hashes() -> dict[str, str]:
    out = {}
    for f in sorted(RESULTS.glob("dataset_manifest*.json")):
        for a in json.load(open(f, encoding="utf-8"))["axes"].values():
            for part in a["partitions"].values():
                out[Path(part["file"]).name] = part["sha256"]
    return out


def clusters(texts: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    """Exact all-pairs Jaccard >= J_MIN; returns component labels and a summary."""
    from scipy.sparse import csr_matrix

    sh = [shingles(t) for t in texts]
    vocab: dict[int, int] = {}
    rows, cols = [], []
    for i, s in enumerate(sh):
        for h in s:
            rows.append(i)
            cols.append(vocab.setdefault(h, len(vocab)))
    M = csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(len(texts), len(vocab)))
    size = np.asarray(M.sum(axis=1)).ravel()
    inter = (M @ M.T).tocoo()
    keep = inter.row < inter.col
    i, j, n = inter.row[keep], inter.col[keep], inter.data[keep]
    jac = n / (size[i] + size[j] - n)
    e = jac >= J_MIN
    parent = np.arange(len(texts))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip(i[e], j[e]):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    lab = np.array([find(x) for x in range(len(texts))])
    comp = Counter(lab.tolist())
    sizes = [c for c in comp.values() if c >= 2]
    return lab, {"n_items": len(texts), "n_empty_shingle_sets": int((size == 0).sum()),
                 "n_pairs_jaccard_ge_0_5": int(e.sum()), "n_clusters": len(sizes),
                 "items_in_clusters": int(sum(sizes)),
                 "size_distribution": {str(k): v for k, v in sorted(Counter(sizes).items())},
                 "largest_cluster_sizes": sorted(sizes, reverse=True)[:10]}


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Split diagnostics after D-155")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=str(RESULTS / "split_diagnostics.json"))
    args = ap.parse_args()
    out_path = Path(args.out).resolve()
    if out_path.exists():
        raise SystemExit(f"REFUSING: {out_path} exists")
    code0 = P.code_block(CODE_FILES)
    from scipy.stats import ks_2samp

    want = recorded_split_hashes()
    split_sha, benign, cl = {}, {}, {}
    for axis in ("safety", "safety_wildguard"):
        recs = {}
        for part in ("threshold_calib", "eval"):
            f = FROZEN / f"{axis}__{part}.jsonl"
            h = sha256_file(f)
            if h != want.get(f.name):
                raise SystemExit(f"REFUSING: {f.name} sha256 {h[:16]} != manifest {str(want.get(f.name))[:16]}")
            split_sha[f.name] = h
            recs[part] = [r for r in read_jsonl(f) if r["label"] == 0]
        pool = [(r["item_id"], "calib", r["text"]) for r in recs["threshold_calib"]] + \
               [(r["item_id"], "eval", r["text"]) for r in recs["eval"]]
        lab, summ = clusters([t for _, _, t in pool])
        comp = Counter(lab.tolist())
        by_comp: dict[int, Counter] = {}
        for (iid, part, _), c in zip(pool, lab):
            if comp[c] >= 2:
                by_comp.setdefault(c, Counter())[part] += 1
        summ["cluster_composition"] = {
            "calib_only": sum(1 for v in by_comp.values() if v["eval"] == 0),
            "eval_only": sum(1 for v in by_comp.values() if v["calib"] == 0),
            "mixed": sum(1 for v in by_comp.values() if v["calib"] and v["eval"])}
        summ["calib_items_in_clusters"] = int(sum(v["calib"] for v in by_comp.values()))
        summ["eval_items_in_clusters"] = int(sum(v["eval"] for v in by_comp.values()))
        summ["base_rate_calib_in_cluster"] = summ["calib_items_in_clusters"] / len(recs["threshold_calib"])
        summ["base_rate_eval_in_cluster"] = summ["eval_items_in_clusters"] / len(recs["eval"])
        cl[axis] = summ
        benign[axis] = {"item_to_label": {iid: int(c) for (iid, _, _), c in zip(pool, lab)},
                        "comp_size": comp, "by_comp": by_comp}
        print(f"[{axis}] {summ['n_items']} benign items: {summ['n_clusters']} clusters (J>={J_MIN}) holding "
              f"{summ['items_in_clusters']} items; sizes {summ['size_distribution']}; composition "
              f"{summ['cluster_composition']}; calib in clusters {summ['base_rate_calib_in_cluster']:.3f}, "
              f"eval {summ['base_rate_eval_in_cluster']:.3f}", flush=True)

    cells = [("4a LoRA base (A100)", "repair_loraA100_mistral_s25", RESULTS / "regularisation_sweep_lora_a100.json")]
    cells += [(s, s, RESULTS / "regularisation_sweep.json") for s in POP.UPDATE_PAIRS_REPAIR]
    out_cells, input_sha = [], {}
    for label, stem, sel_file in cells:
        sel_doc = json.load(open(sel_file, encoding="utf-8"))
        m = next(x for x in sel_doc["models"] if x["pair"] == stem)
        C, L = float(m["selected"]["arm"].split("=")[1]), int(m["selected"]["layer"])
        cfg = json.load(open(RESULTS / f"{stem}.json", encoding="utf-8"))["config"]
        axis = cfg.get("axis", "safety")
        z = {}
        for part in ("probe_train", "threshold_calib", "eval"):
            z[part], p = load(cfg["pre_repo"], cfg["pre_rev"], axis, part)
            input_sha[f"{stem}:pre/{part}"] = sha256_file(p)
            if part in ("probe_train", "threshold_calib") and \
                    input_sha[f"{stem}:pre/{part}"] != sel_doc["input_cache_sha256"].get(f"{stem}:pre/{part}"):
                raise SystemExit(f"REFUSING: {stem} pre/{part} is not the bytes the selection used")
        k = f"layer_{L}"
        y_tr = z["probe_train"]["labels"].astype(bool)
        ncal = ~z["threshold_calib"]["labels"].astype(bool)
        nev = ~z["eval"]["labels"].astype(bool)
        cal_ids = z["threshold_calib"]["item_ids"][ncal]
        cal_s, ev_s = [], []
        for seed in args.seeds:
            _, fit = P.d117_split(len(y_tr), seed)
            pr = P.fit_probe(z["probe_train"][k][fit], y_tr[fit], seed, C, "d117_permutation")
            cal_s.append(pr.score(z["threshold_calib"][k][ncal]))
            ev_s.append(pr.score(z["eval"][k][nev]))
        CAL, EV = np.column_stack(cal_s), np.column_stack(ev_s)
        # Diagnostic 1, per model: the top-50 calibration negatives by mean score over seeds.
        b = benign[axis]
        top = np.argsort(-CAL.mean(axis=1))[:TOP_K]
        top_ids = [str(cal_ids[i]) for i in top]
        in_cl = [b["comp_size"][b["item_to_label"][iid]] >= 2 for iid in top_ids]
        top_comps = Counter(b["item_to_label"][iid] for iid, c in zip(top_ids, in_cl) if c)
        # Diagnostic 2: quantiles and KS per seed.
        qc = np.quantile(CAL, QUANTILES, axis=0)
        qe = np.quantile(EV, QUANTILES, axis=0)
        ks = [ks_2samp(CAL[:, s], EV[:, s]) for s in range(CAL.shape[1])]
        ks_tail = [ks_2samp(CAL[CAL[:, s] >= np.quantile(CAL[:, s], 0.9), s],
                            EV[EV[:, s] >= np.quantile(EV[:, s], 0.9), s]) for s in range(CAL.shape[1])]
        sd = CAL.std(axis=0)
        row = {"cell": label, "pair": stem, "axis": axis, "selected_C": C, "layer": L,
               "top50_calib_in_clusters": int(sum(in_cl)), "top50_calib_share_in_clusters": sum(in_cl) / TOP_K,
               "base_rate_calib_in_clusters": cl[axis]["base_rate_calib_in_cluster"],
               "top50_clusters_with_2plus_members_in_top50": int(sum(1 for v in top_comps.values() if v >= 2)),
               "top50_largest_shared_cluster": int(max(top_comps.values())) if top_comps else 0,
               "quantiles": [{"q": q, "calib_mean": float(qc[i].mean()), "eval_mean": float(qe[i].mean()),
                              "eval_minus_calib_mean": float((qe[i] - qc[i]).mean()),
                              "eval_minus_calib_in_calib_sd": float(((qe[i] - qc[i]) / sd).mean())}
                             for i, q in enumerate(QUANTILES)],
               "ks_per_seed": [{"D": float(r.statistic), "p": float(r.pvalue)} for r in ks],
               "ks_upper_decile_per_seed": [{"D": float(r.statistic), "p": float(r.pvalue)} for r in ks_tail]}
        out_cells.append(row)
        d = {x["q"]: x["eval_minus_calib_in_calib_sd"] for x in row["quantiles"]}
        print(f"{label:<34} top-50 calib in clusters {sum(in_cl):>2}/50 (base {row['base_rate_calib_in_clusters']:.3f}); "
              f"eval-calib in calib SDs: q10 {d[0.1]:+.2f} q50 {d[0.5]:+.2f} q90 {d[0.9]:+.2f} q99 {d[0.99]:+.2f}; "
              f"KS p {min(r.pvalue for r in ks):.1e}-{max(r.pvalue for r in ks):.1e}; "
              f"upper-decile KS p {min(r.pvalue for r in ks_tail):.1e}-{max(r.pvalue for r in ks_tail):.1e}", flush=True)

    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"]:
        raise SystemExit("REFUSING to write: source changed during the run")
    payload = {**provenance_block("split_diagnostics"), "code": code0,
               "config": {"jaccard_min": J_MIN, "jaccard": "exact, all pairs, word 3-gram shingles of src/paraphrase_overlap.py",
                          "top_k": TOP_K, "quantiles": QUANTILES, "seeds": args.seeds,
                          "probe": "D-152 P1 (D-117 split, permutation order, selected C)",
                          "cells": "4a LoRA base + UPDATE_PAIRS_REPAIR"},
               "split_file_sha256": split_sha, "input_cache_sha256": input_sha,
               "clusters": cl, "cells": out_cells}
    digest = write_results(out_path, payload, exclusive=True)
    print(f"Written: {out_path}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
