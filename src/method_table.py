"""The Method table: which analysis ran over which layer grid, pairs, dataset and seeds.

A reviewer reading the results cannot currently tell that the depth curves run a dense
grid while the alignment variants run a sparse one, or that the over-refusal axis uses a
different corpus from the safety axis. Both facts change how a number should be read --
Fig 4's population is a different set of runs from Fig 1's, not a subset of it, and that
was only caught by hand. This builds the table that makes the design legible, by reading
every results file the populations name rather than by restating the design from memory.

Every cell is read back from results/*.json. Nothing here is declared.

  python src/method_table.py            # printed table
  python src/method_table.py --latex    # LaTeX body for the manuscript
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import ensure_utf8_stdout  # noqa: E402
import populations as POP  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# axis -> (corpus, manifest) read from the freeze manifests, not hard-coded here.
def axis_sources() -> dict[str, str]:
    out: dict[str, str] = {}
    for man in ("dataset_manifest.json", "dataset_manifest_wildguard.json"):
        p = RESULTS / man
        if not p.exists():
            continue
        d = json.load(open(p, encoding="utf-8"))
        for axis, a in (d.get("axes") or {}).items():
            src = a.get("source")
            if isinstance(src, str):
                out[axis] = src
    return out


# The analyses, in the order the paper presents them. Each entry names the population
# constant it draws from, so the table cannot drift from what the runs actually used.
# (label, population constant, stem prefix). LORA_ARMS names arms, not result files --
# the same arm appears as repair_, depth_, moment_ and overrefusal_ -- so the dose row
# is pinned to the repair analysis, which is the one the headline quotes.
GROUPS: list[tuple[str, str, str]] = [
    ("Repair (released pairs)", "UPDATE_PAIRS_REPAIR", ""),
    ("Depth curves", "UPDATE_PAIRS_DEPTH", ""),
    ("Moment decomposition", "UPDATE_PAIRS_MOMENT", ""),
    ("Alignment variants", "UPDATE_PAIRS_ALIGNVAR", ""),
    ("LoRA dose series (repair)", "LORA_ARMS", "repair_"),
    ("Over-refusal", "OVER_REFUSAL_PAIRS", ""),
    ("Adversarial stratification", "ADVERSARIAL", ""),
    ("Non-parametric layer", "NON_PARAMETRIC", ""),
    ("Controls (must show no effect)", "CONTROLS", ""),
    ("AUC criterion re-analysis", "AUC_AUDITS", ""),
]


def _stem_shaped(s: str) -> bool:
    # No case test: real stems carry capitals (loraA100, WildGuardMix cells), and an
    # earlier lowercase-only rule silently dropped them, taking the whole wildguardmix
    # corpus out of the table. Labels are excluded by the per-dict rule below instead.
    return isinstance(s, str) and " " not in s and "_" in s and "-" not in s


def candidates(name: str, prefix: str = "") -> list[str]:
    """Every result stem a population names, with the prefix applied.

    The constants are not uniform and the difference matters:
      LORA_ARMS, CONTROLS   {label: [stem, ...]}      -- stems are in the values
      AUC_AUDITS            {label: stem}             -- stem is the value
      OVER_REFUSAL_PAIRS    {stem: (label, ..., n)}   -- stem is the KEY
    Walking keys and values indiscriminately picks up labels such as "gpu_change",
    which is stem-shaped but names no file. So decide per dict: if the value is a
    collection of clean stem-shaped strings, the stems are the values; otherwise the
    key is the stem and the value is metadata.
    """
    out: list[str] = []

    def walk(x) -> None:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
        elif isinstance(x, dict):
            for k, y in x.items():
                if isinstance(y, str) and _stem_shaped(y):
                    out.append(y)
                elif (isinstance(y, (list, tuple)) and y
                        and all(_stem_shaped(e) for e in y)):
                    walk(y)
                else:
                    walk(k)

    walk(getattr(POP, name, None))
    seen, res = set(), []
    for c in out:
        s = f"{prefix}{c}"
        if _stem_shaped(c) and s not in seen:
            seen.add(s)
            res.append(s)
    return res


def stems_of(name: str, prefix: str = "") -> list[str]:
    return [s for s in candidates(name, prefix) if (RESULTS / f"{s}.json").exists()]


def unresolved(name: str, prefix: str = "") -> list[str]:
    return [s for s in candidates(name, prefix) if not (RESULTS / f"{s}.json").exists()]


def describe(stem: str) -> dict | None:
    p = RESULTS / f"{stem}.json"
    if not p.exists():
        return None
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    cfg = d.get("config") or {}
    rows = d.get("rows")
    layers: list[int] = []
    if isinstance(rows, list):
        for r in rows:
            if isinstance(r, dict) and isinstance(r.get("layer"), int):
                layers.append(r["layer"])
    layers = sorted(set(layers))
    seeds = cfg.get("seeds")
    n_seeds = len(seeds) if isinstance(seeds, (list, tuple)) else (1 if seeds is not None else None)
    axis = cfg.get("axis") or cfg.get("train_axis") or "-"
    # An over-refusal run fits on the safety splits and evaluates on over_refusal.
    eval_axis = "over_refusal" if "train_axis" in cfg else axis
    return {"stem": stem, "layers": layers, "n_seeds": n_seeds,
            "axis": axis, "eval_axis": eval_axis, "run_kind": d.get("run_kind")}


def grid_str(layer_sets: list[list[int]]) -> str:
    """Describe the layer grid, and say plainly when it is not the same everywhere."""
    uniq = {tuple(s) for s in layer_sets if s}
    if not uniq:
        return "n/a"
    def ordinal(k: int) -> str:
        return {1: "1st", 2: "2nd", 3: "3rd"}.get(k, f"{k}th")

    def shape(u: tuple[int, ...]) -> str:
        n = len(u)
        if n == 1:
            return f"1 layer ({u[0]})"
        steps = {u[i + 1] - u[i] for i in range(n - 1)}
        if steps == {1}:
            return f"{n} layers, {u[0]}-{u[-1]} (all)"
        if len(steps) == 1:
            return f"{n} layers, {u[0]}-{u[-1]} (every {ordinal(steps.pop())})"
        return f"{n} layers, {u[0]}-{u[-1]} (irregular)"

    if len(uniq) == 1:
        return shape(next(iter(uniq)))

    sizes = sorted({len(u) for u in uniq})
    # Same count, different absolute indices, is the matched-relative-depth design:
    # moment and alignvar evaluate 5 layers at matched fractions of depth, which land
    # on 2,8,16,24,32 in a 32-layer model and 2,10,20,30,42 in Gemma-2's 42. That is
    # the design working, not a grid inconsistency.
    if len(sizes) == 1:
        lo = min(u[0] for u in uniq)
        hi = max(u[-1] for u in uniq)
        return f"{sizes[0]} layers, {lo}-{hi} (matched depth)"
    strides = set()
    for u in uniq:
        strides |= {u[i + 1] - u[i] for i in range(len(u) - 1)}
    lo, hi = min(u[0] for u in uniq), max(u[-1] for u in uniq)
    if strides == {1}:
        return f"{min(sizes)}-{max(sizes)} layers, {lo}-{hi} (all)"
    if len(strides) == 1:
        return f"{min(sizes)}-{max(sizes)} layers, {lo}-{hi} (every {ordinal(strides.pop())})"
    return f"{min(sizes)}-{max(sizes)} layers, {lo}-{hi} (varies)"


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Method table, read from results/")
    ap.add_argument("--latex", action="store_true", help="emit the LaTeX table body")
    args = ap.parse_args()

    src = axis_sources()
    table = []
    for label, popname, prefix in GROUPS:
        stems = stems_of(popname, prefix)
        descs = [x for x in (describe(s) for s in stems) if x]
        missing = len(unresolved(popname, prefix))
        if not descs:
            table.append({"label": label, "n": 0, "missing": missing, "grid": "not run",
                          "axes": "-", "corpora": "-", "seeds": "-"})
            continue
        axes = sorted({d["eval_axis"] for d in descs})
        corpora = sorted({src.get(a, a) for a in axes})
        seeds = sorted({d["n_seeds"] for d in descs if d["n_seeds"]})
        table.append({
            "label": label,
            "n": len(descs),
            "missing": missing,
            "grid": grid_str([d["layers"] for d in descs]),
            "axes": ", ".join(axes),
            "corpora": ", ".join(c.split("/")[-1] for c in corpora),
            "seeds": "/".join(str(s) for s in seeds) if seeds else "-",
        })

    if args.latex:
        print("% generated by src/method_table.py -- do not edit by hand")
        for r in table:
            print(f"{r['label']} & {r['n']} & {r['grid']} & {r['corpora']} & "
                  f"{r['seeds']} \\\\")
        return 0

    w = max(len(r["label"]) for r in table) + 2
    g = max(len(r["grid"]) for r in table) + 2
    c = max(len(r["corpora"]) for r in table) + 2
    rule = "-" * (w + 6 + 2 + g + c + 6)
    print(f"{'analysis':<{w}}{'cells':>6}  {'layer grid':<{g}}{'corpus':<{c}}{'seeds':>6}")
    print(rule)
    for r in table:
        print(f"{r['label']:<{w}}{r['n']:>6}  {r['grid']:<{g}}{r['corpora']:<{c}}"
              f"{r['seeds']:>6}")
        if r["missing"]:
            print(f"{'':<{w}}{'':>6}  MISSING {r['missing']} result file(s) named by the "
                  f"population")
    print(rule)
    print(f"{'TOTAL':<{w}}{sum(r['n'] for r in table):>6}")
    print()
    print("Layer grid: '(all)' evaluates every layer in the span; '(every Nth)' a regular")
    print("subsample; '(matched depth)' the same number of layers at matched fractions of")
    print("model depth, so absolute indices differ between families by design.")
    print("Analyses on different grids are different populations, and a figure drawn from")
    print("one must not be described as a subset of another.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
