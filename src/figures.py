"""Core figures, drawn ONLY from results/synthesis2.json. Computes nothing (CLAUDE.md rule 1/2).

synthesis2.py is the single place where headline numbers are derived from the results
files, with declared populations (F-038). This script reads that one artefact, so a
figure can never disagree with the text: if a number is not in synthesis2.json, it
cannot be plotted here.

Figures are written into paper/figures/ so that the Overleaf project is self-contained:
main.tex is compiled from paper/, and \\includegraphics resolves relative to it.

Each figure is drawn at the exact width it will occupy in an IEEE two-column layout, so
nothing is rescaled on inclusion and in-figure type renders at its true point size. A
figure drawn at 9in and included at \\columnwidth shrinks its 8pt labels to under 3pt,
which is what this file previously did.

  COL_W   one column, included with width=\\columnwidth in a figure environment
  FULL_W  both columns, included with width=\\textwidth in a figure* environment

Figures
  fig1_damage_vs_discrimination  COL   the dissociation: threshold error vs AUROC drop
  fig2_repair_vs_floor           FULL  each repair arm against that pair's no-update floor
  fig3_lora_dose                 FULL  the controlled dose series, damage and alignment
  fig4_mechanism                 COL   alignment variants, alignment-variant population
  fig5_over_refusal_depth        FULL  over-refusal multiplier by relative depth
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import ensure_utf8_stdout, sha256_file  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "paper" / "figures"
SRC = ROOT / "results" / "synthesis2.json"
GREY, BLUE, RED, GREEN = "#666666", "#2b6cb0", "#c53030", "#2f855a"

# IEEEtran conference: column 3.5in, text 7.16in. Leave a hair of margin.
COL_W, FULL_W = 3.42, 7.10

# Short labels for plots, consistent with the NICE names used in src/tables.py.
SHORT = {
    "mistral_base2instruct": "Mistral inst.",
    "gemma2_base2instruct": "Gemma-2 it",
    "llama3_base2instruct": "Llama-3 inst.",
    "llama31_base2instruct": "Llama-3.1 inst.",
    "qwen_base2instruct": "Qwen2.5 inst.",
    "olmo_base2instruct": "OLMo-2 inst.",
    "llama3_to_31": "L3$\\to$L3.1",
    "mistral_v01_v02_dense": "Mistral v0.2",
    "WG_mistral_base2instruct": "Mistral (WG)",
    "WG_llama3_base2instruct": "Llama-3 (WG)",
}


def short(pair: str) -> str:
    return SHORT.get(pair, pair.replace("_", " "))


def clean_label(name: str) -> str:
    """Turn an internal population label into one fit for a legend.

    Strips the decision-log annotation '(2 depths only, D-095)' and the '[VERSION]' tag,
    both of which are internal bookkeeping: the caption and the dashed line style already
    say which pairs are version increments. Also renders the arrow properly."""
    name = re.sub(r"\s*\([^)]*D-\d+[^)]*\)", "", name)
    name = name.replace("[VERSION]", "")
    return re.sub(r"\s*->\s*", r" $\\to$ ", name).strip()


def style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "figure.dpi": 300, "savefig.dpi": 300,
    })


def save(fig, out: Path) -> None:
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig1(S, out):
    d = sorted(S["damage"]["pairs"], key=lambda r: r["max_auroc_drop"])
    xs = [r["max_auroc_drop"] for r in d]
    fig, ax = plt.subplots(figsize=(COL_W, 2.9))
    hi = max(xs)
    # Ten labelled points in a single column, four of them in one cluster. An automatic
    # rule cannot place these without collisions, so the offsets are set per point, the
    # way one would do it by hand: (dx pt, dy pt, horizontal alignment).
    PLACE = {
        "mistral_base2instruct":     (-6, -3, "right"),
        "WG_mistral_base2instruct":  (-6, -3, "right"),
        "llama3_base2instruct":      (0, 7, "center"),
        "llama3_to_31":              (7, 0, "left"),
        "llama31_base2instruct":     (7, -9, "left"),
        "qwen_base2instruct":        (7, 0, "left"),
        "gemma2_base2instruct":      (0, 8, "center"),
        "WG_llama3_base2instruct":   (0, 8, "center"),
        "olmo_base2instruct":        (0, -13, "center"),
        "mistral_v01_v02_dense":     (7, 0, "left"),
    }
    for r in d:
        x, y = r["max_auroc_drop"], r["worst_fpr_ratio"]
        ax.scatter(x, y, s=26, color=BLUE, zorder=3)
        dx, dy, ha = PLACE.get(r["pair"], (6, 3, "left"))
        ax.annotate(short(r["pair"]), (x, y), fontsize=5.8,
                    xytext=(dx, dy), textcoords="offset points",
                    color="#333333", ha=ha, va="center")
    ax.axhline(1.0, color=GREY, lw=0.9, ls="--")
    ax.set_yscale("log")
    # Left margin wide enough for the centred and right-aligned labels above.
    ax.set_xlim(-hi * 0.14, hi * 1.10)
    ax.set_ylim(0.75, max(r["worst_fpr_ratio"] for r in d) * 2.4)
    ax.text(ax.get_xlim()[0] + hi * 0.03, 1.10, "threshold on target",
            fontsize=6, color=GREY)
    ax.set_xlabel("largest frozen-probe AUROC drop")
    ax.set_ylabel("worst realised FPR / target")
    ax.grid(alpha=0.25, zorder=0)
    save(fig, out)


def fig2(S, out):
    rep = S["repair"]["per_pair"]
    names = [short(r["pair"]) for r in rep]
    x = range(len(rep))
    fig, ax = plt.subplots(figsize=(FULL_W, 2.9))
    ax.bar([i - 0.27 for i in x], [r["frozen_all"] for r in rep], 0.26,
           label="frozen (no repair)", color=RED)
    ax.bar(list(x), [r["restandardise"] for r in rep], 0.26,
           label="restandardise (label-free)", color=BLUE)
    ax.bar([i + 0.27 for i in x], [r["refit_probe"] for r in rep], 0.26,
           label="refit with labels", color=GREEN)
    for i, r in enumerate(rep):
        if "floor_q95" in r:
            ax.plot([i - 0.42, i + 0.42], [r["floor_q95"]] * 2, color="black",
                    lw=1.2, zorder=5)
    ax.plot([], [], color="black", lw=1.2, label="no-update floor (95th pct.)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=22, ha="right")
    ax.set_ylabel("mean $|\\log($FPR$/$target$)|$")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="y", alpha=0.25)
    save(fig, out)


def fig3(S, out):
    arms = S["lora_arms"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 2.6))
    fig.subplots_adjust(wspace=0.28)
    colours = {"general (alpaca)": GREY, "safety (hh-harmless)": RED,
               "safety (pku-saferlhf)": "#dd6b20"}
    for arm, rows in arms.items():
        doses = [r["dose"] for r in rows]
        a1.plot(doses, [r["frozen"] for r in rows], "o-", ms=3.5, lw=1.3,
                color=colours[arm], label=arm)
        if all("cos_abs" in r for r in rows):
            a2.plot(doses, [r["cos_abs"] for r in rows], "o-", ms=3.5, lw=1.3,
                    color=colours[arm], label=arm)
    for a in (a1, a2):
        a.set_xscale("log")
        a.set_xticks([25, 100, 400])
        a.set_xticklabels(["25", "100", "400"])
        # A log axis keeps drawing labelled MINOR ticks, which collided with the three
        # dose labels and rendered as "25x10^1 4x10^1 6x10^1". Silence them.
        a.xaxis.set_minor_formatter(NullFormatter())
        a.tick_params(axis="x", which="minor", length=2)
        a.set_xlabel("LoRA steps")
        a.grid(alpha=0.25)
    a1.set_ylabel("mean $|\\log($FPR$/$target$)|$")
    a1.set_title("damage to the frozen threshold")
    a1.legend(frameon=False)
    a2.axhline(0.0156, color=GREY, ls="--", lw=0.9)
    a2.text(26, 0.0159, "unaligned shift (RMS)", fontsize=6, color=GREY)
    a2.set_ylabel("$|\\cos\\theta|$ with the probe direction")
    a2.set_title("how well the shift aims at the probe")
    save(fig, out)


def fig4(S, out):
    m = S["mechanism"]
    order = ["frozen", "paired_align", "coral_unpaired", "procrustes_paired",
             "restandardise", "refit_oracle"]
    label = {"frozen": "no repair", "paired_align": "ridge alignment",
             "coral_unpaired": "CORAL (full cov.)",
             "procrustes_paired": "Procrustes (orth.)",
             "restandardise": "restandardise", "refit_oracle": "refit with labels"}
    vals = [m["alignment_means"][a] for a in order]
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    cols = [RED, "#a0aec0", "#a0aec0", "#a0aec0", BLUE, GREEN]
    ax.barh([label[a] for a in order], vals, color=cols, height=0.68)
    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=6.5)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel("mean $|\\log($FPR$/$target$)|$, 7 pairs, lower is better")
    ax.grid(axis="x", alpha=0.25)
    save(fig, out)


def fig5(S, out):
    fig, ax = plt.subplots(figsize=(FULL_W, 2.8))
    # An explicit restrained palette, so this figure matches the others rather than
    # falling back to the default cycle.
    PALETTE = [BLUE, "#dd6b20", GREEN, RED, "#6b46c1", "#8a6d3b", "#b83280"]
    for i, r in enumerate(S["over_refusal"]):
        layers = sorted(int(k) for k in r["all_layers"])
        ratios = [r["all_layers"][str(k)]["ratio"] for k in layers]
        rel = [k / max(layers) for k in layers]
        st = "--" if r["kind"] == "version" else "-"
        ax.plot(rel, ratios, st, marker="o", ms=3.0, lw=1.2,
                color=PALETTE[i % len(PALETTE)], label=clean_label(r["pair"]))
    ax.axhline(1.0, color=GREY, lw=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("relative depth (layer / final layer)")
    ax.set_ylabel("safe-prompt refusals, post / pre")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.20))
    ax.grid(alpha=0.25)
    save(fig, out)


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Figures from synthesis2.json only")
    ap.add_argument("--src", default=str(SRC))
    args = ap.parse_args()
    src = Path(args.src)
    S = json.load(open(src, encoding="utf-8"))
    FIGS.mkdir(parents=True, exist_ok=True)
    style()
    print(f"source: {src.name}  sha256 {sha256_file(src)[:16]}  "
          f"(run_kind {S.get('run_kind')})")
    for name, fn, width in (("fig1_damage_vs_discrimination", fig1, "column"),
                            ("fig2_repair_vs_floor", fig2, "full"),
                            ("fig3_lora_dose", fig3, "full"),
                            ("fig4_mechanism", fig4, "column"),
                            ("fig5_over_refusal_depth", fig5, "full")):
        out = FIGS / f"{name}.png"
        fn(S, out)
        print(f"  {out.name:<36} {width:<6} {out.stat().st_size / 1024:6.1f} KB  "
              f"sha256 {sha256_file(out)[:12]}")
    print(f"5 figures written to {FIGS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
