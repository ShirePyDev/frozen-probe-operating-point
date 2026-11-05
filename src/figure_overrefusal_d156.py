"""The Fig. 5 replacement (author, 2026-09-27): per-layer over-refusal curves for the seven item-9 pairs under
D-156, from results/overrefusal_d156_<pair>.json only (no new draws), and the final-layer test of the
draft's "reverses at the last layer" claim.

Per pair, at the D-117 selected C, every layer of the pair's item-9 grid (item level; the D-164 cluster level
in the data file): the XSTest refusal rate pre and post (D-156 mean and 95% band) and the paired refusal ratio
(mean and band, log scale, 1 marked; a hollow marker where any draw is floored at 1/250, D-163 (2)). Dashed:
the D-117 layer (only when it is in the grid); dotted: the declared mid layer. The band is over the 1000
re-partitions of the Aegis negatives that set the thresholds; XSTest itself is fixed (D-167), so the band
says nothing about XSTest item sampling: the frozen-split counts and McNemar are reported beside it.
FINAL-LAYER TABLE (the author's test): at each pair's last grid layer (each model's last hidden state), the
paired refusal ratio below 1 with band95 upper < 1, item level at the selected C; the cluster level and C=1
beside, with whether the ratio rose at an earlier layer (band lower > 1). Floored ratios are "at most"
values (D-208). Qwen2.5 and Mistral v0.1 -> v0.2 have partial grids until the D-206 re-run.
Writes paper/figures/fig_overrefusal_d156.{pdf,png} and results/overrefusal_curves_d156.json, staged and
linked together (never overwritten).
--full (D-212): the regeneration once D-206 has filled the two partial grids. Qwen2.5 and Mistral v0.1 -> v0.2
are read from their tagged re-runs, results/overrefusal_d156_<pair>_xsfull.json (tied to
logs/overrefusal_d156_xsfull.sha256); the five others are read unchanged. A NEW record and figure are written,
results/overrefusal_curves_d156_full.json and paper/figures/fig_overrefusal_d156_full.{pdf,png}; the partial-grid
ones are never overwritten. Any pair still on a partial grid is refused. The figure stays a notes figure until
D-168 is final (D-216 (2); final since D-331).
--reissue (D-331, D-318 (7)): once the final D-168 record is pinned (src/input_registry.py), the same variant is
written again under _r2 names (results/overrefusal_curves_d156[_full]_r2.json, paper/figures/
fig_overrefusal_d156[_full]_r2.{pdf,png}) with no D-170 marks; the record names the record it supersedes and its
sha256. The earlier record and figure stay as they are. Refused while D-168 is not final.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import input_registry as REG  # noqa: E402  (D-300 (3): the final D-168 record removes the D-170 marks)
import paired as P  # noqa: E402
from provenance import (  # noqa: E402
    commit_staged, ensure_utf8_stdout, provenance_block, sha256_bytes, sha256_file, stage_bytes,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "paper" / "figures"
FINAL_OUT = RESULTS / "overrefusal_curves_d156.json"
FINAL_FIG = FIGS / "fig_overrefusal_d156"
FULL_OUT = RESULTS / "overrefusal_curves_d156_full.json"
FULL_FIG = FIGS / "fig_overrefusal_d156_full"
CODE_FILES = ["src/figure_overrefusal_d156.py", "src/paired.py", "src/provenance.py", "src/input_registry.py"]
MANIFEST = ROOT / "logs" / "overrefusal_d156_20260926.sha256"
XS_MANIFEST = ROOT / "logs" / "overrefusal_d156_xsfull.sha256"
XSFULL_STEMS = ("qwen", "mistral_v01_v02")
SELECTION = RESULTS / "selection_sweep_d165.json"
STEMS = ("mistral", "gemma2", "llama3", "llama31", "qwen", "mistral_v01_v02", "llama3_to_31")
LABEL = {"mistral": "Mistral v0.1 -> Instruct", "gemma2": "Gemma-2 -> it", "llama3": "Llama-3 -> Instruct",
         "llama31": "Llama-3.1 -> Instruct", "qwen": "Qwen2.5 -> Instruct", "mistral_v01_v02": "Mistral v0.1 -> v0.2",
         "llama3_to_31": "Llama-3 -> 3.1"}
LEVELS = ("item_level_D156", "cluster_level_D164")
BAND_NOTE = ("band95: 2.5-97.5% over the 1000 re-partitions of the Aegis negatives that set the thresholds (D-156; "
             "cluster level D-164); XSTest is fixed (D-167), so the band does not cover XSTest item sampling")
LAST_LAYER_NOTE = ("the last grid layer is each model's last hidden state (extract_hidden stores hidden_states[li]); for "
                   "these models that entry follows the final norm, so it is the only post-norm layer in the grid")


def marked(flag: Any) -> bool:
    """D-216 (2): a panel of a D-170-flagged pair counts as a table; its mark goes once D-168 is final (D-300 (3))."""
    return bool(flag) and REG.d168_final() is None


def band(x: dict[str, Any]) -> dict[str, Any]:
    return {"mean": x["mean"], "band95": x["band95"], "frozen": x["frozen"]}


def rel(p: Path) -> str:
    return str(p.resolve().relative_to(ROOT.resolve())).replace("\\", "/") if p.resolve().is_relative_to(ROOT.resolve()) else str(p)


def below_one(r: dict[str, Any]) -> bool:
    return r["mean"] < 1 and r["band95"][1] < 1


def manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        raise SystemExit(f"REFUSING: no manifest {path.name} (for --full: written after the two D-206 re-runs are pulled)")
    man = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, f = line.split(maxsplit=1)
            man[f.strip().lstrip("*")] = h
    return man


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Fig. 5 replacement: per-layer over-refusal under D-156")
    ap.add_argument("--full", action="store_true",
                    help="D-212: read the two _xsfull re-runs and write the _full record and figure")
    ap.add_argument("--out", default=None, help="default: the variant's final record name")
    ap.add_argument("--fig", default=None, help="path stem; .pdf and .png are added; default: the variant's final name")
    ap.add_argument("--reissue", action="store_true",
                    help="D-331: write the variant again under _r2 names once D-168 is final (no D-170 marks)")
    args = ap.parse_args()
    final_out, final_fig_stem = (FULL_OUT, FULL_FIG) if args.full else (FINAL_OUT, FINAL_FIG)
    supersedes = None
    if args.reissue:
        if REG.d168_final() is None:
            raise SystemExit("REFUSING: --reissue needs the final D-168 record pinned in src/input_registry.py")
        if not final_out.is_file():
            raise SystemExit(f"REFUSING: {rel(final_out)} does not exist, so there is nothing to re-issue")
        supersedes = {"record": rel(final_out), "sha256": sha256_file(final_out)}
        final_out = final_out.with_name(final_out.stem + "_r2.json")
        final_fig_stem = final_fig_stem.with_name(final_fig_stem.name + "_r2")
    out = Path(args.out) if args.out else final_out
    fig_stem = Path(args.fig) if args.fig else final_fig_stem
    final = out.resolve() == final_out.resolve()
    final_fig = fig_stem.resolve() == final_fig_stem.resolve()
    if final != final_fig:
        raise SystemExit("REFUSING: the record and the figure must both take their final names, or neither")
    if not final and any(p.resolve().is_relative_to(d.resolve()) for p in (out, fig_stem) for d in (RESULTS, FIGS)):
        raise SystemExit("REFUSING: a test output must not be under results/ or paper/figures/")
    fig_paths = {s: fig_stem.with_name(fig_stem.name + s) for s in (".pdf", ".png")}
    for p in (out, *fig_paths.values()):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists; never overwritten")
    code0 = P.code_block(CODE_FILES)
    mans = {MANIFEST: manifest(MANIFEST)}
    if args.full:
        mans[XS_MANIFEST] = manifest(XS_MANIFEST)
    sel = {m["pair"]: m["selected"] for m in json.loads(SELECTION.read_text(encoding="utf-8"))["models"]}
    inputs = {rel(SELECTION): sha256_file(SELECTION), **{rel(m): sha256_file(m) for m in mans}}
    curves, final_table = {}, {}
    for stem in STEMS:
        tagged = args.full and stem in XSFULL_STEMS
        rel9 = f"results/overrefusal_d156_{stem}{'_xsfull' if tagged else ''}.json"
        man_path = XS_MANIFEST if tagged else MANIFEST
        b = (ROOT / rel9).read_bytes()
        if mans[man_path].get(rel9) != sha256_bytes(b):
            raise SystemExit(f"REFUSING: {rel9} is not the file {man_path.name} names")
        inputs[rel9] = sha256_bytes(b)
        d = json.loads(b.decode("utf-8"))
        cfg = d["config"]
        if d["config"]["selection_sha256"] != inputs[rel(SELECTION)]:
            raise SystemExit(f"REFUSING: {rel9} was run on another selection")
        if d.get("xstest_cache_tag") != ("xsfull" if tagged else None):
            raise SystemExit(f"REFUSING: {rel9} has xstest_cache_tag {d.get('xstest_cache_tag')!r}")
        if cfg.get("pair") != f"overrefusal_{stem}":
            raise SystemExit(f"REFUSING: {rel9} holds pair {cfg.get('pair')!r}, not overrefusal_{stem}")
        L117 = sel[cfg["selection_of_pair"]]["layer"]
        layers = sorted(cfg["layers"])
        partial = layers != sorted(d["layer_grids"]["aegis"])
        if args.full and partial:
            raise SystemExit(f"REFUSING --full: {rel9} holds a partial grid {layers}, not the Aegis grid")
        by = {(c["layer"], c["probe_set"]): c for c in d["cells"]}
        curves[stem] = {"label": LABEL[stem], "source": rel9, "flag": d["input_verification"]["flag"], "layers": layers,
                        "d117_layer": L117, "d117_layer_in_grid": L117 in layers, "mid_layer": cfg["headline_layer"]["layer"],
                        "partial_grid": partial,
                        "note": ("partial grid (the layers both XSTest caches hold) until the D-206 full-grid re-run" if partial
                                 else f"XSTest cache(s) {d.get('xstest_cache_repos')} from the A100 __xsfull extraction (D-206); "
                                      "the thresholds come from the pinned Aegis caches, so the pair mixes hardware; the re-run "
                                      "check against the partial-grid file at shared layers, under the D-202 bound, is in "
                                      "results/overrefusal_d117_table.json" if tagged else None),
                        "selected": {lvl: {q: [band(by[(li, "selected")][lvl][q]) for li in layers]
                                           for q in ("refusal_pre", "refusal_post", "paired_refusal_ratio")} for lvl in LEVELS},
                        "paired_refusal_ratio_share_of_draws_floored": {
                            lvl: [by[(li, "selected")][lvl]["paired_refusal_ratio_share_of_draws_floored"] for li in layers] for lvl in LEVELS}}
        last = layers[-1]
        row: dict[str, Any] = {}
        for ps in ("selected", "C=1"):
            for lvl in LEVELS:
                blk = by[(last, ps)][lvl]
                r = blk["paired_refusal_ratio"]
                share = blk["paired_refusal_ratio_share_of_draws_floored"]
                earlier = [li for li in layers[:-1] if by[(li, ps)][lvl]["paired_refusal_ratio"]["band95"][0] > 1]
                row[f"{ps}|{lvl}"] = {**band(r), "share_of_draws_floored": share, "floor_bound_at_most": share > 0,
                                      "below_1_band_excludes_1": below_one(r),
                                      "earlier_layers_band_above_1": earlier,
                                      "frozen_safe_flagged_pre_per_seed": blk["frozen_split"]["safe_flagged_pre"],
                                      "frozen_safe_flagged_post_per_seed": blk["frozen_split"]["safe_flagged_post"],
                                      "frozen_safe_discordant_b_c": blk["frozen_split"]["safe_discordant_b_c"],
                                      "mcnemar_safe_p_max": band(blk["mcnemar_safe_p_max"]),
                                      "share_of_draws_mcnemar_safe_p_max_below_0_05": blk["share_of_draws_mcnemar_safe_p_max_below_0_05"]}
        r0 = row[f"selected|{LEVELS[0]}"]
        final_table[stem] = {"final_layer": last, "flag": curves[stem]["flag"], "partial_grid": partial,
                             "below_1_band_excludes_1": r0["below_1_band_excludes_1"],
                             "below_1_band_excludes_1_cluster_level": row[f"selected|{LEVELS[1]}"]["below_1_band_excludes_1"],
                             "below_1_band_excludes_1_C1": row[f"C=1|{LEVELS[0]}"]["below_1_band_excludes_1"],
                             "rose_at_an_earlier_layer": bool(r0["earlier_layers_band_above_1"]),
                             "ratios": row}
        v = final_table[stem]
        print(f"{stem:17s} final L{last}: paired refusal ratio {'at most ' if r0['floor_bound_at_most'] else ''}{r0['mean']:.3f} "
              f"[{r0['band95'][0]:.3f}, {r0['band95'][1]:.3f}] (floored {r0['share_of_draws_floored']:.3f}); "
              f"below 1, band excl. 1: item {v['below_1_band_excludes_1']}, cluster {v['below_1_band_excludes_1_cluster_level']}, "
              f"C=1 {v['below_1_band_excludes_1_C1']}; rose earlier (band > 1): {v['rose_at_an_earlier_layer']}", flush=True)
    below = [s for s, v in final_table.items() if v["below_1_band_excludes_1"]]

    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "figure.dpi": 300, "savefig.dpi": 300})
    fig, axes = plt.subplots(2, len(STEMS), figsize=(13, 4.4))
    for j, stem in enumerate(STEMS):
        cv = curves[stem]
        x = cv["layers"]
        S = cv["selected"][LEVELS[0]]
        ax, ax2 = axes[0, j], axes[1, j]

        def draw(a: Any, series: list[dict[str, Any]], col: str, lab: str | None, hollow: list[bool]) -> None:
            # a partial grid gets error bars, never a band or a line interpolated across the missing layers
            m = [v["mean"] for v in series]
            lo, hi = [v["band95"][0] for v in series], [v["band95"][1] for v in series]
            if cv["partial_grid"]:
                a.errorbar(x, m, yerr=[[max(0.0, mi - l) for mi, l in zip(m, lo)], [max(0.0, h - mi) for mi, h in zip(m, hi)]], fmt="none",
                           ecolor=col, elinewidth=0.8, capsize=2)
            else:
                a.plot(x, m, "-", color=col, lw=1)
                a.fill_between(x, lo, hi, color=col, alpha=0.2, lw=0)
            a.scatter(x, m, s=6, zorder=3, edgecolors=col, linewidths=0.6, label=lab,
                      facecolors=["none" if h else col for h in hollow])

        for q, col, lab in (("refusal_pre", "tab:blue", "pre"), ("refusal_post", "tab:red", "post")):
            draw(ax, S[q], col, lab, [False] * len(x))
        draw(ax2, S["paired_refusal_ratio"], "tab:purple", None,
             [f > 0 for f in cv["paired_refusal_ratio_share_of_draws_floored"][LEVELS[0]]])
        ax2.axhline(1, color="grey", lw=0.6)
        ax2.set_yscale("log")
        ax2.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax2.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        for a in (ax, ax2):
            if cv["d117_layer_in_grid"]:
                a.axvline(cv["d117_layer"], color="k", lw=0.6, ls="--")
            a.axvline(cv["mid_layer"], color="grey", lw=0.6, ls=":")
        title = cv["label"] + (" *" if marked(cv["flag"]) else "")
        if not cv["d117_layer_in_grid"]:
            title += f"\n(D-117 L{cv['d117_layer']} not in grid)"
        ax.set_title(title, fontsize=7)
        ax2.set_xlabel("layer")
        if j == 0:
            ax.set_ylabel("XSTest refusal rate (safe prompts)")
            ax.legend(frameon=False)
            ax2.set_ylabel("paired refusal ratio (post / pre)")
    fig.suptitle("Over-refusal by layer under D-156" + (", full grids (D-206)" if args.full else "") +
                 " (selected C, item level; bands: 95% over 1000 re-partitions of the Aegis "
                 "negatives, XSTest fixed; dashed: D-117 layer; dotted: mid layer; hollow: floored draws"
                 + ("; * D-170 flagged)" if any(marked(cv["flag"]) for cv in curves.values()) else ")"),
                 fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig_bytes = {}
    for s, fmt in ((".pdf", "pdf"), (".png", "png")):
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, metadata={"CreationDate": None} if fmt == "pdf" else None)
        fig_bytes[s] = buf.getvalue()
    plt.close(fig)

    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"] or any(sha256_file(ROOT / f) != h for f, h in inputs.items()):
        raise SystemExit("REFUSING to write: code or an input changed during the run")
    payload = {**provenance_block("figure_overrefusal_d156"), "code": code0, "matplotlib": matplotlib.__version__,
               "purpose": "Fig. 5 replacement and the final-layer test of the draft's reversal claim (author, 2026-09-27)",
               "variant": ("full: Qwen2.5 and Mistral v0.1 -> v0.2 from their _xsfull re-runs (D-206, D-212)" if args.full
                           else "partial: Qwen2.5 and Mistral v0.1 -> v0.2 on the layers both pinned XSTest caches hold"),
               "input_files_sha256": inputs,
               "d168_final": REG.d168_final(),       # D-300 (3): the record that removes the D-170 marks, or None
               "supersedes": supersedes,             # --reissue (D-331): the earlier record, kept unchanged
               "d170_marks_shown": REG.d168_final() is None,     # each 'flag' below is the input record's own text;
                                                                  # its mark is drawn only while D-168 is not final
               "figure_files_sha256": {rel(p): sha256_bytes(fig_bytes[s]) for s, p in fig_paths.items()},
               "rule": "final-layer paired refusal ratio < 1 with band95 upper < 1 (selected C, item level); cluster level "
                       "and C=1 beside; 'rose_at_an_earlier_layer' = some earlier layer with band95 lower > 1",
               "band_note": BAND_NOTE, "last_layer_note": LAST_LAYER_NOTE,
               "floor_note": "a ratio with floored draws is an 'at most' value (D-163 (2), D-208)",
               "final_layer_table": final_table, "pairs_below_1_band_excludes_1": below, "curves": curves}
    json_bytes = json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8")
    json.loads(json_bytes.decode("utf-8"))
    staged: list[tuple[Path, Path, str]] = []
    try:
        for s, p in fig_paths.items():
            p.parent.mkdir(parents=True, exist_ok=True)
            t, h = stage_bytes(p, fig_bytes[s])
            staged.append((t, p, h))
        t, h = stage_bytes(out, json_bytes)
        staged.append((t, out, h))
        for t, p, h in staged:
            commit_staged(t, p, h)
    finally:
        for t, _, _ in staged:
            if t.exists():
                t.unlink()
    print(f"pairs below 1 at the final layer with the band excluding 1 (selected C, item level): {below or 'none'}")
    for t, p, h in staged:
        print(f"Written: {rel(p)}\nsha256:  {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
