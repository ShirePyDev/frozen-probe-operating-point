"""Phase 1-mode equivalence check (D-307 (5)): before any Phase 2 extraction, src/extract_p2.py --phase1-mode must
reproduce an A100 Phase 1 cache bit for bit. It is also the first GPU test of extract_p2.py.

Inputs:
  --p1mode-record   the record extract_p2.py --phase1-mode wrote (outside results/), read through read_result. It must
                    name the weights' snapshot and its files; its cache sits beside it and must have the sha256 it records.
  --phase1-record   the A100 Phase 1 extraction record, results/extract__<repo>__a100@<rev12>__<axes>.json: it must lie in
                    results/ and its cache_repo must be its repo + '__a100'. Read through read_result; its cache
                    (data/hidden, or GUARDRAIL_CACHE_DIRS, via provenance.resolve_cache) must have the cache_sha256 it
                    records.
Refuses (exit 1) unless the Phase 1-mode run used the Phase 1 run's model, revision, dtype, max_length, batch size,
pooling and split (its sha256), a subset of its layers, and an item count that is a multiple of its batch size (bit for
bit needs the same batches: the first N / batch_size batches of the Phase 1 run). A FINAL run (the output under
results/phase2/) refuses unless N is the smallest multiple of the batch size >= 100 and the layers are ALL the Phase 1
record's layers (D-307 (5)).
ENVIRONMENT GATE (exit 2): the Phase 1-mode run's GPU name and torch, transformers and tokenizers versions must equal
the Phase 1 record's wherever the Phase 1 record states them (the ones it does not state are listed), so that a NOT
EQUAL can only mean the code path differs.

Compared, for the first N items in file order: item_ids (by value), labels, nonfinite, truncated and n_tokens (each
differing index listed), and per layer the float32 rows bitwise (a uint32 view, so -0.0 differs from 0.0): every
differing item with its item_id and its number of differing elements, n_equal_bitwise, max_abs_diff over the finite
differences (float64) and the count of non-finite differences. A layer that cannot be compared says why. bit_for_bit
is true only when every one of them is equal. Writes results/phase2/p1mode_equivalence_<slug>.json (never overwritten;
a test --out must lie outside results/) and exits 3 when not bit for bit, 0 when it is.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as P  # noqa: E402
import provenance as PV  # noqa: E402
from provenance import ensure_utf8_stdout, provenance_block, read_result, resolve_cache, sha256_file, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "hidden"
OUT_DIR = RESULTS / "phase2"
CODE_FILES = ["src/p1mode_equivalence.py", "src/provenance.py", "src/paired.py"]
AUX = ("item_ids", "labels", "nonfinite", "truncated", "n_tokens")
GATE_KEYS = ("device_name", "torch", "transformers", "tokenizers", "cuda")      # cuda: the A100 record states it (12.6)
MIN_ITEMS = 100                                                  # D-307 (5): at least 100 items
EXIT_ENV, EXIT_NOT_EQUAL = 2, 3


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def refuse_env(msg: str) -> None:
    print(f"REFUSING (environment gate, D-307 (5)): {msg}", file=sys.stderr, flush=True)
    raise SystemExit(EXIT_ENV)


def code_files() -> list[str]:
    """CODE_FILES plus every src module loaded, so the code block names each imported src module (D-152 P5)."""
    src = Path(__file__).resolve().parent
    loaded = {f"src/{Path(m.__file__).name}" for m in list(sys.modules.values())
              if getattr(m, "__file__", None) and Path(m.__file__).suffix == ".py"
              and Path(m.__file__).resolve().parent == src}
    return sorted(set(CODE_FILES) | loaded)


def code_block(final: bool) -> dict[str, Any]:
    return P.code_block(code_files(), allow_no_commit=not final)


def final_items(batch_size: int) -> int:
    """The smallest multiple of the Phase 1 batch size that is at least MIN_ITEMS."""
    return -(-MIN_ITEMS // batch_size) * batch_size


def where(p: Path) -> dict[str, Any]:
    """A file's name, and its absolute path when it lies outside the repo."""
    p = Path(p).resolve()
    inside = p.is_relative_to(ROOT.resolve())
    return {"file": p.name, "path": str(p.relative_to(ROOT.resolve())).replace("\\", "/") if inside else str(p),
            "in_repo": inside}


def check_params(p1: dict[str, Any], pm: dict[str, Any], p1_path: Path, final: bool) -> dict[str, Any]:
    """The comparison is the intended one only when the Phase 1-mode run reproduced the Phase 1 run's parameters."""
    if pm.get("run_kind") != "extract_p2_phase1_mode" or pm.get("mode") != "phase1":
        refuse("--p1mode-record is not an extract_p2.py --phase1-mode record")
    p1_path = Path(p1_path).resolve()
    if p1_path.parent != RESULTS.resolve() or not p1_path.name.startswith("extract__"):
        refuse(f"--phase1-record must be an extraction record in {RESULTS}, not {p1_path}")
    if p1.get("run_kind") != "extract_hidden":
        refuse("--phase1-record is not an extract_hidden.py record")
    if not p1.get("repo") or p1.get("cache_repo") != f"{p1['repo']}__a100":
        refuse(f"--phase1-record is not an A100 Phase 1 record: cache_repo {p1.get('cache_repo')!r}, not "
               f"{p1.get('repo')!r} + '__a100'")
    if p1.get("adapter") is not None or p1.get("pooling") != "mean" or p1.get("chat_template_applied"):
        refuse("the Phase 1 record is not extract_hidden's mean pooling of a base checkpoint without a chat template")
    diff = {k: (p1.get(k), pm.get(k)) for k in ("repo", "revision", "dtype", "max_length", "batch_size", "pooling")
            if p1.get(k) != pm.get(k)}
    if diff:
        refuse(f"the Phase 1-mode run differs from the Phase 1 run in {diff} (Phase 1, Phase 1 mode)")
    w = pm.get("weights") or {}
    if not w.get("snapshot") or not isinstance(w.get("files"), dict) or not w["files"]:
        refuse("the Phase 1-mode record does not name the weights' snapshot and its files")
    if len(pm.get("splits") or {}) != 1:
        refuse("the Phase 1-mode record names more or fewer than one split")
    split = next(iter(pm["splits"]))
    s1, sm = (p1.get("splits") or {}).get(split), pm["splits"][split]
    if not isinstance(s1, dict):
        refuse(f"the Phase 1 record has no split {split}")
    if s1.get("split_sha256") != sm.get("split_sha256"):
        refuse(f"split {split}: sha256 {str(sm.get('split_sha256'))[:12]} is not the Phase 1 run's "
               f"{str(s1.get('split_sha256'))[:12]}")
    n, b = int(sm["n"]), int(p1["batch_size"])
    if n != pm.get("items") or n < 1 or n % b or n > int(s1["n"]):
        refuse(f"{n} items: bit for bit needs the Phase 1 run's batches, so a multiple of its batch size {b}, at most "
               f"its {s1['n']} items")
    if final and n != final_items(b):
        refuse(f"a final run takes exactly {final_items(b)} items, the smallest multiple of the Phase 1 batch size {b} "
               f"that is at least {MIN_ITEMS} (D-307 (5)); this run has {n}")
    layers_m, layers_1 = sorted(sm["layers_resolved"]), sorted(s1.get("layers_resolved") or [])
    if not layers_m or not set(layers_m) <= set(layers_1):
        refuse(f"layers {layers_m} are not among the Phase 1 cache's {layers_1}")
    if final and layers_m != layers_1:
        refuse(f"a final run compares ALL the Phase 1 record's layers {layers_1}; this run has {layers_m}")
    if final and (Path(str(w["snapshot"])).name != p1.get("revision") or w.get("nvme_checked") is not True):
        refuse("a final run needs the weights read from snapshots/<the Phase 1 revision> on NVMe (nvme_checked)")
    return {"split": split, "n_items": n, "batch_size": b, "n_batches": n // b, "layers": layers_m,
            "phase1_layers": layers_1, "phase1_split_n": int(s1["n"]), "split_sha256": sm["split_sha256"],
            "final_run": final, "final_items_rule": final_items(b),
            "equal_parameters": {k: p1.get(k) for k in ("repo", "revision", "dtype", "max_length", "batch_size",
                                                         "pooling")}}


def env_gate(p1: dict[str, Any], pm: dict[str, Any]) -> dict[str, Any]:
    """GPU name and library versions equal wherever the Phase 1 record states them (exit 2 otherwise)."""
    lv1, lvm = p1.get("lib_versions") or {}, pm.get("lib_versions") or {}
    stated = [k for k in GATE_KEYS if lv1.get(k) is not None]
    diff = {k: {"phase1": lv1[k], "p1mode": lvm.get(k)} for k in stated if lvm.get(k) != lv1[k]}
    if diff:
        refuse_env(f"the Phase 1-mode run's environment differs from the Phase 1 run's: {diff}; a NOT EQUAL could not "
                   "be told apart from an environment difference, so nothing is compared")
    return {"gated": {k: {"phase1": lv1[k], "p1mode": lvm.get(k), "equal": True} for k in stated},
            "not_stated_by_the_phase1_record": {k: {"p1mode": lvm.get(k)} for k in GATE_KEYS if k not in stated},
            "p1mode_bit_environment": pm.get("bit_environment")}


def load_cache(path: Path, want_sha: str, what: str, keys: list[str], n: int | None = None) -> dict[str, np.ndarray]:
    """Only the keys compared, sliced to the first n rows; the file's sha256 must be its record's."""
    if not path.is_file():
        refuse(f"the {what} cache {path.name} is not on disk")
    got = sha256_file(path)
    if got != want_sha:
        refuse(f"the {what} cache {path.name} has sha256 {got[:12]}, not its record's {str(want_sha)[:12]}")
    with np.load(path, allow_pickle=False) as z:
        return {k: (z[k][:n] if n is not None else z[k]) for k in keys if k in z.files}


def compare(a1: dict[str, np.ndarray], am: dict[str, np.ndarray], n: int, layers: list[int]) -> dict[str, Any]:
    """Item by item and layer by layer: the first n Phase 1 rows against the Phase 1-mode rows."""
    ids = a1.get("item_ids")
    aux: dict[str, Any] = {}
    for k in AUX:
        x, y = a1.get(k), am.get(k)
        reason = (f"missing in the {'Phase 1' if x is None else 'Phase 1-mode'} cache" if x is None or y is None
                  else f"shape {x.shape} / {y.shape}" if x.shape != y.shape
                  else f"dtype {x.dtype} / {y.dtype}" if k != "item_ids" and x.dtype != y.dtype else None)
        if reason:
            aux[k] = {"equal": False, "comparable": False, "reason": reason, "differing_indices": None}
            continue
        neq = (x.astype(str) != y.astype(str)) if k == "item_ids" else (x != y)          # ids by value, not dtype
        idx = np.flatnonzero(neq).tolist()
        aux[k] = {"equal": not idx, "comparable": True, "differing_indices": idx,
                  "phase1_dtype": str(x.dtype), "p1mode_dtype": str(y.dtype)}
    per_layer: dict[str, Any] = {}
    for li in layers:
        key = f"layer_{li}"
        x, y = a1.get(key), am.get(key)
        reason = (f"missing key {key} in the {'Phase 1' if x is None else 'Phase 1-mode'} cache" if x is None or y is None
                  else f"dtype {x.dtype} / {y.dtype}, not float32" if x.dtype != np.float32 or y.dtype != np.float32
                  else f"shape {x.shape} / {y.shape}" if x.shape != y.shape else None)
        if reason:
            per_layer[str(li)] = {"n_items": n, "comparable": False, "reason": reason, "n_equal_bitwise": 0,
                                  "max_abs_diff": None, "n_nonfinite_abs_diffs": None, "differing_items": None}
            continue
        x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)                     # one row per item, any layout
        per_item = (x.view(np.uint32) != y.view(np.uint32)).sum(axis=1)                  # -0.0 differs from 0.0
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        fin = np.isfinite(d)
        differing = [{"index": int(i), "item_id": None if ids is None else str(ids[i]),
                      "n_elements_differing": int(per_item[i])} for i in np.flatnonzero(per_item)]
        per_layer[str(li)] = {"n_items": n, "comparable": True, "n_equal_bitwise": int((per_item == 0).sum()),
                              "max_abs_diff": float(d[fin].max()) if fin.any() else None,
                              "n_nonfinite_abs_diffs": int((~fin).sum()), "differing_items": differing}
    bit = all(v["comparable"] and v["equal"] for v in aux.values()) and \
        all(v["comparable"] and v["n_equal_bitwise"] == n for v in per_layer.values())
    return {"aux": aux, "per_layer": per_layer, "bit_for_bit": bit}


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Phase 1-mode equivalence check (D-307 (5))")
    ap.add_argument("--p1mode-record", required=True, help="the record extract_p2.py --phase1-mode wrote")
    ap.add_argument("--phase1-record", required=True, help="results/extract__<repo>__a100@<rev12>__<axes>.json")
    ap.add_argument("--slug", required=True, help="names the output: results/phase2/p1mode_equivalence_<slug>.json")
    ap.add_argument("--phase1-cache-dir", default=None,
                    help="where the Phase 1 cache is; default data/hidden and GUARDRAIL_CACHE_DIRS (resolve_cache)")
    ap.add_argument("--out", default=None, help="a test output outside results/ (default: the final name)")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.slug):
        refuse(f"--slug {args.slug!r} is not letters, digits, '_' and '-'")
    final_out = OUT_DIR / f"p1mode_equivalence_{args.slug}.json"
    out = Path(args.out).resolve() if args.out else final_out.resolve()
    final = out == final_out.resolve()
    if out.exists():
        refuse(f"{out.name} exists; results are never overwritten")
    if not final and out.is_relative_to(RESULTS.resolve()):
        refuse("a test run (--out) must not write under results/")
    code0 = code_block(final)
    p1 = read_result(args.phase1_record)
    pm = read_result(args.p1mode_record)
    prm = check_params(p1, pm, Path(args.phase1_record), final)
    env = env_gate(p1, pm)
    split, n, layers = prm["split"], prm["n_items"], prm["layers"]
    s1, sm = p1["splits"][split], pm["splits"][split]
    p1_path = (Path(args.phase1_cache_dir) / s1["cache_file"] if args.phase1_cache_dir
               else resolve_cache(s1["cache_file"], CACHE))
    pm_path = Path(args.p1mode_record).parent / sm["cache_file"]
    keys = [*AUX, *(f"layer_{li}" for li in layers)]
    a1 = load_cache(p1_path, s1["cache_sha256"], "Phase 1", keys, n)
    am = load_cache(pm_path, sm["cache_sha256"], "Phase 1-mode", keys)
    cmp = compare(a1, am, n, layers)
    differing = {li: [x["index"] for x in v["differing_items"] or []] for li, v in cmp["per_layer"].items()
                 if v["n_equal_bitwise"] != n}
    what = f"{n} items of {split} ({prm['n_batches']} batches of {prm['batch_size']}), layers {layers}"
    payload: dict[str, Any] = {
        **provenance_block("p1mode_equivalence"), "code": code0,
        "preregistration": "D-307 (5): extract_p2.py --phase1-mode must equal the A100 Phase 1 cache bit for bit "
                           "before any Phase 2 extraction; otherwise the run stops and reports where they differ",
        "config": {"repo": pm["repo"], "rev": pm["revision"]},
        "input_cache_sha256": {s1["cache_file"]: s1["cache_sha256"], sm["cache_file"]: sm["cache_sha256"]},
        "phase1_cache": {**where(p1_path), "sha256": s1["cache_sha256"], "verified_on_disk": True},
        "p1mode_cache": {**where(pm_path), "sha256": sm["cache_sha256"], "verified_on_disk": True,
                         "note": "a test cache outside data/hidden with no indexed extraction record (its record is "
                                 "the Phase 1-mode record below, outside results/); the provenance audit shows it as "
                                 "NO_EXTRACTION_RECORD. This result is not primary"},
        "phase1_record": where(Path(args.phase1_record)), "p1mode_record": where(Path(args.p1mode_record)),
        "p1mode_weights": pm.get("weights"),
        "parameters": prm, "environment": env,
        "comparison": "the first n_items rows in file order; item_ids by value; labels, nonfinite, truncated, n_tokens "
                      "exactly; per layer, each float32 element bitwise (uint32 view: -0.0 differs from 0.0), "
                      "max_abs_diff over the finite differences in float64",
        "aux": cmp["aux"], "per_layer": cmp["per_layer"], "bit_for_bit": cmp["bit_for_bit"],
        "layers_with_differences": differing,
        "verdict": (f"BIT_FOR_BIT_EQUAL on {what}: Phase 2 extraction may start (D-307 (5))" if cmp["bit_for_bit"]
                    else f"NOT EQUAL on {what}: Phase 2 extraction does not start (D-307 (5)); every difference is "
                         "listed in aux and per_layer"),
    }
    if code_block(final)["sha256"] != code0["sha256"]:
        refuse("the code changed during the run; nothing is written")
    payload["input_files_sha256"] = dict(PV._RESULT_READS)
    digest = write_results(out, payload, exclusive=True)
    for li, v in cmp["per_layer"].items():
        print(f"  layer {li:>3}: {v['n_equal_bitwise']}/{n} bitwise equal, max |diff| {v['max_abs_diff']}"
              f"{'' if v['comparable'] else ' (not comparable: ' + v['reason'] + ')'}")
    print(f"  aux equal: {all(v['equal'] for v in cmp['aux'].values())}; environment gated on "
          f"{sorted(env['gated'])}, not stated by the Phase 1 record: {sorted(env['not_stated_by_the_phase1_record'])}")
    print(f"{payload['verdict']}\nWritten: {out.name}\nsha256:  {digest}")
    return 0 if cmp["bit_for_bit"] else EXIT_NOT_EQUAL


if __name__ == "__main__":
    raise SystemExit(main())
