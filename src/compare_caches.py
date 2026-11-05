"""Compare two hidden-state caches item by item and layer by layer.

Used as a GATE, not a report. The LoRA step-0 adapter is an exact no-op (LoRA
initialises B = 0), so extracting through the adapter pipeline - causal-LM load,
PEFT attach, merge - must reproduce the base model's cache. If it does not, every
LoRA result would mix the update's effect with a pipeline artefact, so the exit
code is non-zero and downstream runners must stop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as P  # noqa: E402
from provenance import ensure_utf8_stdout, provenance_block, sha256_file, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CODE_FILES = ["src/compare_caches.py", "src/provenance.py", "src/paired.py"]     # the code block (D-238 (3), D-292)
GATE_FAIL = 3            # the record is written and the gate says FAIL; refusals exit 1, so a queue can tell them apart


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Gate: are two caches the same measurement?")
    ap.add_argument("--a", required=True, help="reference cache (.npz)")
    ap.add_argument("--b", required=True, help="candidate cache (.npz)")
    ap.add_argument("--tol", type=float, default=1e-4, help="max |diff| allowed on any layer")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if Path(args.out).exists():
        raise SystemExit(f"REFUSING: {args.out} exists; results are never overwritten")
    code0 = P.code_block(CODE_FILES)

    pa, pb = Path(args.a), Path(args.b)
    pa = pa if pa.is_absolute() else ROOT / pa
    pb = pb if pb.is_absolute() else ROOT / pb
    A, B = np.load(pa), np.load(pb)
    report = {"a": pa.name, "b": pb.name, "a_sha256": sha256_file(pa), "b_sha256": sha256_file(pb),
              "tol": args.tol, "layers": {}}
    ok = True
    if not np.array_equal(A["item_ids"], B["item_ids"]):
        print("FAIL: item_ids differ")
        ok = False
    for flag in ("labels", "truncated", "nonfinite"):
        same = bool(np.array_equal(A[flag], B[flag]))
        report[f"{flag}_equal"] = same
        ok &= same
    common = sorted({k for k in A.files if k.startswith("layer_")} & {k for k in B.files if k.startswith("layer_")},
                    key=lambda s: int(s.split("_")[1]))
    if not common:
        print("FAIL: no common layers")
        ok = False
    worst = 0.0
    for k in common:
        d = float(np.abs(A[k].astype(np.float64) - B[k].astype(np.float64)).max())
        report["layers"][k] = {"max_abs_diff": d, "bit_identical": bool(np.array_equal(A[k], B[k]))}
        worst = max(worst, d)
    report["max_abs_diff_any_layer"] = worst
    report["n_bit_identical_layers"] = sum(v["bit_identical"] for v in report["layers"].values())
    ok &= worst <= args.tol
    report["PASS"] = bool(ok)
    print(f"layers compared {len(common)}  bit-identical {report['n_bit_identical_layers']}  "
          f"max |diff| {worst:.3e}  tol {args.tol:.0e}  flags equal "
          f"{[report[f + '_equal'] for f in ('labels', 'truncated', 'nonfinite')]}")
    print("GATE PASS" if ok else "GATE FAIL")
    if P.code_block(CODE_FILES)["sha256"] != code0["sha256"] or sha256_file(pa) != report["a_sha256"] \
            or sha256_file(pb) != report["b_sha256"]:
        raise SystemExit("REFUSING to write: the code or a cache changed during the run")
    digest = write_results(args.out, {**provenance_block("compare_caches"), "code": code0, **report}, exclusive=True)
    print(f"Written: {args.out}\nsha256:  {digest}")
    return 0 if ok else GATE_FAIL      # a written record whose gate failed, distinct from a refusal (exit 1)


if __name__ == "__main__":
    raise SystemExit(main())
