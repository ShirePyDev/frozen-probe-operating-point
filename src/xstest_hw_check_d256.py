"""XSTest hardware check at the full grid (the author's D-256 (4); pre-registered in D-258).

For every model with XSTest caches from both GPUs, the same probes and the same Aegis thresholds are applied to its
V100 cache and to its A100 cache, and the safe-prompt counts are compared at every layer both caches hold:
  UNITS  Mistral-7B-v0.1 (V100 original, A100 __a100; full grid) and LoRA alpaca s25 (V100 original, A100 __a100;
         full grid); Qwen2.5-7B and Mistral-7B-v0.2 (V100 pinned partial grids, A100 __xsfull; their shared layers).
  PROBES the D-117 probes of the pair that reads the cache, fitted exactly as item 9 fits them (seeds 0-4, the D-117
         fit split, selected C and C=1) on that pair's PRE Aegis probe_train; thresholds at 1% on each D-156 draw's
         calibration part of the pooled Aegis negatives (item level, the frozen split as index 0), as in item 9.
  NUMBERS per layer and probe set: the mean over seeds of the safe prompts flagged (of 250) on each cache, and the
         within-draw difference A100 - V100 as a share of the 250, each as D-156 mean and band; the frozen split's
         per-seed counts on both caches and the number of safe prompts whose flag differs between the caches.
  BOUND  judged at selected C: |D-156 mean of the difference share| <= 0.001 at a layer (the D-202/D-204 (iii) XSTest
         bound); C=1 beside, never judged.
  ANCHOR where the V100 cache is the one an item-9 record read (Mistral v0.1 as the pre side of overrefusal_mistral,
         Qwen2.5 as the pre side of overrefusal_qwen, Mistral v0.2 as the post side of overrefusal_mistral_v01_v02),
         the V100 counts must equal that record's safe_flagged_<side>_mean band summary exactly at every shared
         layer and probe set (item level), or nothing is written.
Also written: per over-refusal record, the hardware of its pre and post XSTest caches (the extraction records'
lib_versions.device_name; a pin-only cache is one of D-168's V100 originals).
Writes results/xstest_hw_check_d256.json (never overwritten); a test writes outside results/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_d156 as D  # noqa: E402
import input_registry as REG  # noqa: E402
import paired as P  # noqa: E402
from provenance import ensure_utf8_stdout, provenance_block, resolve_cache, sha256_bytes, sha256_file, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FINAL_OUT = RESULTS / "xstest_hw_check_d256.json"
CODE_FILES = ["src/xstest_hw_check_d256.py", *D.CODE_FILES]
PROBE_SETS = ("selected", "C=1")
SEEDS = [0, 1, 2, 3, 4]
BOUND = 0.001                                           # D-202 (3), D-204 (iii): the XSTest share floor
# unit -> (registry run of the pre Aegis caches, selection file, selection pair, V100 repo, A100 repo, anchor record, side)
UNITS = {
    "Mistral-7B-v0.1": ("repair_mistral_base2instruct", D.SELECTION_OF_RECORD, "repair_mistral_base2instruct",
                        "mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-v0.1__a100",
                        "results/overrefusal_d156_mistral.json", "pre"),
    "lora-alpaca-s25": ("repair_loraA100_mistral_s25", D.SELECTION_OF_RECORD_LORA, "repair_loraA100_mistral_s25",
                        "lora/mistral7b-v01-alpaca-r16-s25", "lora/mistral7b-v01-alpaca-r16-s25__a100", None, None),
    "Qwen2.5-7B": ("repair_qwen_base2instruct", D.SELECTION_OF_RECORD, "repair_qwen_base2instruct",
                   "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B__xsfull", "results/overrefusal_d156_qwen.json", "pre"),
    "Mistral-7B-v0.2": ("depth_mistral_v01_v02_dense", D.SELECTION_OF_RECORD, "repair_mistral_base2instruct",
                        "mistral-community/Mistral-7B-v0.2", "mistral-community/Mistral-7B-v0.2__xsfull",
                        "results/overrefusal_d156_mistral_v01_v02.json", "post"),
}


# D-258 (3): the layers each unit is compared at (both caches' shared grid); a registered run refuses any other set
EXPECTED_LAYERS = {"Mistral-7B-v0.1": list(range(0, 33, 2)), "lora-alpaca-s25": list(range(0, 33, 2)),
                   "Qwen2.5-7B": [14, 28], "Mistral-7B-v0.2": [8, 16, 24, 32]}
QUOTABILITY = ("Mistral-7B-v0.1", (2, 4, 6, 18))       # D-258 (4): the layers where D-253's v0.2 curve rises
ANCHOR_MANIFEST = ROOT / "logs" / "overrefusal_d156_20260926.sha256"


def grid(z: Any) -> list[int]:
    return sorted(int(k.split("_")[1]) for k in z.files if k.startswith("layer_"))


def devices(read: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for f in sorted(RESULTS.glob("extract__*.json")):
        if "__rerun_" in f.name:
            continue
        b = f.read_bytes()
        read[f"results/{f.name}"] = sha256_bytes(b)
        d = json.loads(b.decode("utf-8"))
        dev = (d.get("lib_versions") or {}).get("device_name")
        for s in (d.get("splits") or {}).values():
            out.setdefault(s["cache_file"], dev or "unknown (record has no device_name)")
    return out


def hardware_table(read: dict[str, str]) -> dict[str, Any]:
    dev = devices(read)
    tab = {}
    for f in sorted(RESULTS.glob("overrefusal_d156_*.json")):
        b = f.read_bytes()
        read[f"results/{f.name}"] = sha256_bytes(b)
        d = json.loads(b.decode("utf-8"))
        src = d.get("input_cache_source") or {}
        if "pre/over_refusal/eval" not in src:
            continue
        row = {}
        for side in ("pre", "post"):
            s = src[f"{side}/over_refusal/eval"]
            row[side] = {"cache": s["file"], "hardware": dev.get(s["file"]) or
                         ("V100 (pin-only: D-168's pinned originals)" if s["pin_only"] else "unknown")}
        row["mixed"] = ("A100" in row["pre"]["hardware"]) != ("A100" in row["post"]["hardware"])
        tab[f"results/{f.name}"] = row
    return tab


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="D-256 (4): XSTest V100 vs A100 at every shared layer")
    ap.add_argument("--units", nargs="*", default=list(UNITS), choices=list(UNITS))
    ap.add_argument("--n-draws", type=int, default=P.N_DRAWS)
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="TESTING: a subset of layers")
    ap.add_argument("--out", default=str(FINAL_OUT))
    args = ap.parse_args()
    out = Path(args.out).resolve()
    final = out == FINAL_OUT.resolve()
    registered = args.n_draws == P.N_DRAWS and args.layers is None and list(args.units) == list(UNITS)
    if out.exists():
        raise SystemExit(f"REFUSING: {out} exists; never overwritten")
    if not final and out.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test output must not be under results/")
    threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    if final and (not registered or any(t != "6" for t in threads.values())):
        raise SystemExit(f"REFUSING the final name: registered {registered}, threads {threads}")
    code0 = P.code_block(CODE_FILES)
    inputs: dict[str, str] = {}
    cache_recs: dict[str, dict[str, Any]] = {}
    units: dict[str, Any] = {}
    read: dict[str, str] = {}
    hw_tab = hardware_table(read)
    dev = devices(read)                                  # the same files, already hashed into 'read'
    anchor_man: dict[str, str] = {}
    anchor_man_sha = sha256_file(ANCHOR_MANIFEST)
    for line in ANCHOR_MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, f = line.split(maxsplit=1)
            f = f.strip().lstrip("*")
            if f in anchor_man:
                raise SystemExit(f"REFUSING: {ANCHOR_MANIFEST.name} names {f} twice")
            anchor_man[f] = h
    inputs[str(ANCHOR_MANIFEST.relative_to(ROOT)).replace("\\", "/")] = anchor_man_sha
    reg_sources = REG.sources_sha256()
    if D.sources_now() != reg_sources:
        raise SystemExit("REFUSING: the extraction records or pins on disk are not the ones the registry read (D-189)")
    for name in args.units:
        reg_stem, sel_path, sel_stem, v100, a100, anchor_rel, side = UNITS[name]
        cfg = REG.run(reg_stem)
        axis = cfg["axis"]
        aeg = {}
        for part in ("probe_train", "threshold_calib", "eval"):
            z, r = REG.load(cfg["pre_repo"], axis, part)
            aeg[part], cache_recs[f"{name}/pre/{axis}/{part}"] = z, r
        xv, rv = REG.load(v100, "over_refusal", "eval")
        xa, ra = REG.load(a100, "over_refusal", "eval")
        cache_recs[f"{name}/xstest/V100"], cache_recs[f"{name}/xstest/A100"] = rv, ra
        if not np.array_equal(xv["item_ids"], xa["item_ids"]) or not np.array_equal(xv["labels"], xa["labels"]):
            raise SystemExit(f"REFUSING: {name}: the two XSTest caches hold different items or labels")
        for z in (*aeg.values(), xv, xa):
            if "nonfinite" in z.files and z["nonfinite"].any():
                raise SystemExit(f"REFUSING: {name}: a cache has non-finite items")
        # a hardware-only comparison: the same model and revision, V100 bytes against A100 bytes
        if REG.revision(v100) != REG.revision(a100):
            raise SystemExit(f"REFUSING: {name}: {v100} and {a100} are not the same revision")
        dv, da = (dev.get(rv["file"]) or ("V100 (pin-only)" if rv["pin_only"] else "unknown")), dev.get(ra["file"], "unknown")
        if "A100" not in da or "V100" not in dv:
            raise SystemExit(f"REFUSING: {name}: the caches are not V100 ({dv}) and A100 ({da})")
        sb = sel_path.read_bytes()
        inputs[str(sel_path.relative_to(ROOT)).replace("\\", "/")] = sha256_bytes(sb)
        sel_doc = json.loads(sb.decode("utf-8"))
        if list(sel_doc["config"]["seeds"]) != SEEDS or float(sel_doc["config"]["holdout"]) != P.HOLDOUT:
            raise SystemExit(f"REFUSING: {sel_path.name}: selection seeds/holdout differ from this run's")
        sel = next((m for m in sel_doc["models"] if m["pair"] == sel_stem), None)
        if sel is None or not sel["selected"]["arm"].startswith("C="):
            raise SystemExit(f"REFUSING: no logistic-C selection for {sel_stem}")
        C_sel = float(sel["selected"]["arm"].split("=")[1])
        g_aeg = [grid(aeg[p]) for p in ("probe_train", "threshold_calib", "eval")]
        if any(g != g_aeg[0] for g in g_aeg):
            raise SystemExit(f"REFUSING: {name}: the Aegis caches have different layer grids")
        layers = sorted(set(g_aeg[0]) & set(grid(xv)) & set(grid(xa)))
        if args.layers is None and layers != EXPECTED_LAYERS[name]:
            raise SystemExit(f"REFUSING: {name}: shared layers {layers}, not the registered {EXPECTED_LAYERS[name]} (D-258 (3))")
        if args.layers is not None:
            layers = [li for li in layers if li in args.layers]
        if not layers:
            raise SystemExit(f"REFUSING: {name}: no layer to compare")
        anc = None
        if anchor_rel:
            ab = (ROOT / anchor_rel).read_bytes()
            if anchor_man.get(anchor_rel) != sha256_bytes(ab):
                raise SystemExit(f"REFUSING: {anchor_rel} is not the file {ANCHOR_MANIFEST.name} names")
            inputs[anchor_rel] = sha256_bytes(ab)
            anc = json.loads(ab.decode("utf-8"))
            if anc["input_cache_source"][f"{side}/over_refusal/eval"]["file"] != rv["file"] or \
                    anc["input_cache_sha256"][f"{side}/over_refusal/eval"] != rv["sha256"]:
                raise SystemExit(f"REFUSING: {anchor_rel} did not read {rv['file']} as its {side} XSTest cache")
            if any(anc["input_cache_sha256"].get(f"pre/{axis}/{p}") != cache_recs[f"{name}/pre/{axis}/{p}"]["sha256"]
                   for p in ("probe_train", "threshold_calib", "eval")):
                raise SystemExit(f"REFUSING: {anchor_rel} read other Aegis caches than this unit")
            if not set(layers) <= set(anc["config"]["layers"]):
                raise SystemExit(f"REFUSING: {anchor_rel} holds layers {anc['config']['layers']}, not all of {layers}")
        y_tr = aeg["probe_train"]["labels"].astype(bool)
        ncal = ~aeg["threshold_calib"]["labels"].astype(bool)
        neg = ~aeg["eval"]["labels"].astype(bool)
        safe = ~xv["labels"].astype(bool)
        n_cal = int(ncal.sum())
        n_pool = n_cal + int(neg.sum())
        parts = [(np.arange(n_cal), np.arange(n_cal, n_pool))] + \
                [(d[:n_cal], d[n_cal:]) for d in P.repartition_draws(n_cal, n_pool - n_cal, args.n_draws, P.AXIS_DRAW_SEED[axis])]
        n_safe = int(safe.sum())
        if final and (n_safe, int((~safe).sum())) != (250, 200):
            raise SystemExit(f"REFUSING: {name}: XSTest holds {n_safe} safe and {int((~safe).sum())} contrast prompts, not 250 and 200")
        print(f"{name}: Aegis {cfg['pre_repo']} via {reg_stem}, selection {sel_stem} C={C_sel:g}; XSTest V100 {rv['file']} "
              f"vs A100 {ra['file']}; layers {layers}; {n_safe} safe prompts", flush=True)
        rows: dict[str, Any] = {}
        n_anchor = 0
        for li in layers:
            k = f"layer_{li}"
            for pset in PROBE_SETS:
                probes = []
                for seed in SEEDS:
                    _, fit = P.d117_split(len(y_tr), seed)
                    rws, C, order = (fit, C_sel, "d117_permutation") if pset == "selected" else (np.sort(fit), 1.0, "file_order")
                    probes.append(P.fit_probe(aeg["probe_train"][k][rws], y_tr[rws], seed, C, order))
                S = lambda x: np.column_stack([q.score(x) for q in probes])  # noqa: E731
                pool = np.vstack([S(aeg["threshold_calib"][k][ncal]), S(aeg["eval"][k][neg])])
                sv, sa = S(xv[k])[safe], S(xa[k])[safe]           # all 450 scored, then the safe rows (as item 9)
                cv, ca = [], []
                for cal, _ in parts:
                    T = np.array([P.thr_at_fpr(pool[cal, s], P.TARGET_FPR) for s in range(len(SEEDS))])
                    cv.append((sv >= T).sum(axis=0))
                    ca.append((sa >= T).sum(axis=0))
                cv, ca = np.array(cv), np.array(ca)                      # (1 + n_draws, seeds)
                mv, ma = cv.mean(axis=1), ca.mean(axis=1)
                diff = (ma - mv) / n_safe
                T0 = np.array([P.thr_at_fpr(pool[parts[0][0], s], P.TARGET_FPR) for s in range(len(SEEDS))])
                cell = {"v100_safe_flagged_mean": P.band_summary(mv[1:], mv[0]),
                        "a100_safe_flagged_mean": P.band_summary(ma[1:], ma[0]),
                        "a100_minus_v100_share": P.band_summary(diff[1:], diff[0]),
                        "frozen_split": {"v100_per_seed": cv[0].tolist(), "a100_per_seed": ca[0].tolist(),
                                         "n_safe_prompts_flag_differs_per_seed": ((sv >= T0) != (sa >= T0)).sum(axis=0).tolist()}}
                if pset == "selected":
                    cell["within_bound"] = bool(abs(cell["a100_minus_v100_share"]["mean"]) <= BOUND)
                if anc is not None:
                    want = next((c for c in anc["cells"] if c["layer"] == li and c["probe_set"] == pset), None)
                    if want is None:
                        raise SystemExit(f"REFUSING: {anchor_rel} has no cell at L{li} {pset}")
                    lv = want["item_level_D156"]
                    # the frozen split's per-seed counts always; the 1000-draw band only at the registered draw count
                    if cv[0].tolist() != lv["frozen_split"][f"safe_flagged_{side}"]:
                        raise SystemExit(f"REFUSING: {name} L{li} {pset}: the V100 frozen-split counts are not {anchor_rel}'s")
                    if args.n_draws == P.N_DRAWS and \
                            json.dumps(cell["v100_safe_flagged_mean"], sort_keys=True) != json.dumps(lv[f"safe_flagged_{side}_mean"], sort_keys=True):
                        raise SystemExit(f"REFUSING: {name} L{li} {pset}: the V100 counts are not {anchor_rel}'s "
                                         f"safe_flagged_{side}_mean")
                    n_anchor += 1
                rows[f"L{li}|{pset}"] = cell
            s_ = rows[f"L{li}|selected"]["a100_minus_v100_share"]
            print(f"  L{li:>2} selected: A100 - V100 safe share {s_['mean']:+.5f} [{s_['band95'][0]:+.5f}, {s_['band95'][1]:+.5f}]",
                  flush=True)
        sel_means = [abs(rows[f"L{li}|selected"]["a100_minus_v100_share"]["mean"]) for li in layers]
        tok = ({"n_items_n_tokens_differ": int((xv["n_tokens"] != xa["n_tokens"]).sum()),
                "n_items_truncated_differ": int((xv["truncated"] != xa["truncated"]).sum())}
               if all(k in xv.files and k in xa.files for k in ("n_tokens", "truncated")) else "not in both caches")
        units[name] = {"tokenisation_v100_vs_a100": tok,
                       "aegis_pre_repo": cfg["pre_repo"], "registry_run": reg_stem, "selection_pair": sel_stem,
                       "selected_C": C_sel, "xstest_v100": rv["file"], "xstest_a100": ra["file"], "layers": layers,
                       "anchor": {"record": anchor_rel, "side": side, "n_cells_identical": n_anchor} if anchor_rel else None,
                       "max_abs_mean_share_selected": max(sel_means) if sel_means else None,
                       "all_layers_within_bound_selected": all(rows[f"L{li}|selected"]["within_bound"] for li in layers),
                       "cells": rows}
        if anchor_rel and n_anchor != len(layers) * len(PROBE_SETS):
            raise SystemExit(f"REFUSING to write: {name}: {n_anchor} anchored cells")
    code1 = P.code_block(CODE_FILES)
    changed = [f for f, h in {**inputs, **read}.items() if sha256_file(ROOT / f) != h]
    changed += [r["file"] for r in cache_recs.values() if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    if D.sources_now() != reg_sources:
        changed.append("registry sources (re-read from disk)")
    if code1["sha256"] != code0["sha256"] or changed:
        raise SystemExit(f"REFUSING to write: code or inputs changed during the run: {changed}")
    # D-258 (4): D-253's v0.2 rise becomes quotable when Mistral-7B-v0.1's hardware shift is within the bound at its layers
    qname, qlayers = QUOTABILITY
    quot = None
    if qname in units:
        per_l = {li: units[qname]["cells"].get(f"L{li}|selected", {}).get("within_bound") for li in qlayers}
        quot = {"unit": qname, "layers": list(qlayers), "within_bound_per_layer": per_l,
                "v02_rise_quotable": all(v is True for v in per_l.values()),
                "rule": "D-258 (4): quotable when the hardware shift is within the bound at every layer where v0.2 rises"}
    payload = {**provenance_block("xstest_hw_check_d256"), "code": code0, "registered_parameters": registered,
               "blas_threads": threads, "preregistration": "D-256 (4), D-258",
               "config": {"seeds": SEEDS, "target_fpr": P.TARGET_FPR, "n_draws": args.n_draws, "level": "item_level_D156",
                          "units": list(args.units), "layers_requested": args.layers,
                          "input_registry_sources_sha256": reg_sources},
               "bound": {"share": BOUND, "rule": "D-202 (3), D-204 (iii); judged at selected C on the D-156 mean"},
               "input_files_sha256": inputs, "files_read_for_the_hardware_table_sha256": read,
               "input_caches": {k: {"file": r["file"], "sha256": r["sha256"], "pin_only": r["pin_only"]} for k, r in cache_recs.items()},
               "input_verification": REG.verification(cache_recs),
               "hardware_of_xstest_caches_per_record": hw_tab, "d253_v02_rise_quotability": quot, "units": units}
    digest = write_results(out, payload, exclusive=True)
    print(f"Written: {out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
