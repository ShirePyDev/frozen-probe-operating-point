"""The IV-B pairs table (the author's items 4 and 5 of 2026-10-16f, D-468): one row per cell at L*, one ID per pair.

    python src/pairs_table_d468.py           # writes paper/tables_pairs.tex (the file paper/main.tex \\inputs)
    python src/pairs_table_d468.py --print   # also prints the ID of every row (no file is read but the records)

A new module (F-092): no pinned module is edited. Every entry is COPIED from a committed record; nothing is computed
but the pair IDs and the short forms below. The header names each source record with its sha256, so
src/check_paper.py's 'generated' step (D-438) can rebuild the file and compare it byte for byte.

ROWS. Phase 1: the nine headline cells, the D-117 selected layer of the nine repair pairs (UPDATE_PAIRS_REPAIR, the
config.registry_pair 'repair_*' of results/depth_d156_<pair>.json), then the extra base-model pair (registry_pair
'depth_*', outside the headline). Phase 2: the nine pairs whose config.line is D-361 (2)'s headline, then the
tokenizer-change line, from results/twomoment_p2_d361_<pair>_usermean.json. Per row: the selected cell
(is_selected_layer and probe_set 'selected'): its layer (L*) and probe_C; config.pre_repo/pre_rev and
post_repo/post_rev; the dataset from config.axis. The cell's layer and C must equal config.selected_layer and
selected_C, or the script refuses.

IDS. One ID per pair, a pair being one (pre repo@revision, post repo@revision): '1a', '1b', ... in Phase 1 and '2a',
... in Phase 2, in row order, so a pair run on two datasets (Phase 1's Mistral and Llama-3 base -> instruct, on
Aegis 2.0 and WildGuardMix) keeps one ID, and the two record names 'llama3_to_31' (Phase 1 base -> base; Phase 2
instruct -> instruct) get two. The script refuses if one pair falls in both phases (it would need one ID).

SHORT FORMS. A checkpoint is its repo name without the organisation, and its revision is the first 7 characters of
the pinned hash (the full hashes are Supplement S-1's). C is written m x 10^e.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / os.environ.get("GUARDRAIL_PAPER_DIR", "paper")  # where the .tex files are written (README, config/example.env)
OUT = PAPER_DIR / "tables_pairs.tex"

P1 = ["mistral_base2instruct", "WG_mistral_base2instruct", "gemma2_base2instruct", "llama3_base2instruct",
      "WG_llama3_base2instruct", "llama31_base2instruct", "qwen_base2instruct", "olmo_base2instruct",
      "llama3_to_31", "mistral_v01_v02_dense"]       # a WildGuardMix cell right after its pair's Aegis 2.0 cell
P2 = ["mistral_v01_v02", "llama3_to_31", "qwen2_to_25", "phi3_128k_to_35", "olmo2_base_sft", "olmo2_sft_dpo",
      "olmo2_dpo_instruct", "tulu3_sft_dpo", "tulu3_dpo_final", "mistral_v02_v03"]
DATASET = {"safety": "Aegis 2.0", "safety_wildguard": "WildGuardMix", "p2wg": "WildGuardMix"}
P2_HEADLINE = "headline (D-361 (2))"
REV = 7


def record(rel: str) -> tuple[dict, str]:
    b = (ROOT / rel).read_bytes()
    return json.loads(b.decode("utf-8")), hashlib.sha256(b).hexdigest()


def selected_cell(r: dict, rel: str) -> dict:
    cells = [c for c in r["cells"] if c["is_selected_layer"] and c["probe_set"] == "selected"]
    if len(cells) != 1:
        raise SystemExit(f"{rel}: {len(cells)} selected cells at L*, expected 1")
    c, cfg = cells[0], r["config"]
    if c["layer"] != cfg["selected_layer"] or c["probe_C"] != cfg["selected_C"]:
        raise SystemExit(f"{rel}: the selected cell (L{c['layer']}, C={c['probe_C']}) differs from the config's")
    return c


def rows() -> tuple[list[dict], dict[str, str]]:
    out, shas = [], {}
    for phase, names, path in ((1, P1, "results/depth_d156_{}.json"), (2, P2, "results/twomoment_p2_d361_{}_usermean.json")):
        for p in names:
            rel = path.format(p)
            r, shas[rel] = record(rel)
            c, cfg = selected_cell(r, rel), r["config"]
            if phase == 1:
                headline = cfg["registry_pair"].startswith("repair_")
            else:
                headline = cfg["line"] == P2_HEADLINE
            out.append({"phase": phase, "record": p, "pre": (cfg["pre_repo"], cfg["pre_rev"]),
                        "post": (cfg["post_repo"], cfg["post_rev"]), "dataset": DATASET[cfg["axis"]],
                        "L": c["layer"], "C": c["probe_C"], "headline": headline})
    ids: dict[tuple, str] = {}
    for phase in (1, 2):
        for row in [x for x in out if x["phase"] == phase]:
            key = (row["pre"], row["post"])
            if key not in ids:
                ids[key] = f"{phase}{chr(ord('a') + sum(1 for k in ids.values() if k[0] == str(phase)))}"
            if ids[key][0] != str(phase):
                raise SystemExit(f"{row['record']}: pair {key} falls in both phases")
            row["id"] = ids[key]
    return out, shas


def short(repo_rev: tuple[str, str]) -> str:
    repo, rev = repo_rev
    return f"{repo.split('/', 1)[1]} @\\texttt{{{rev[:REV]}}}"


def sci(x: float) -> str:
    m, e = f"{x:.0e}".split("e")
    e = int(e)
    return f"$10^{{{e}}}$" if m == "1" else f"${m}\\cdot10^{{{e}}}$"


def tex(out: list[dict], shas: dict[str, str]) -> str:
    lines = ["% GENERATED by src/pairs_table_d468.py (D-468): do not edit; rerun the script.",
             "% Sources (sha256):"]
    lines += [f"%   {rel} {sha}" for rel, sha in sorted(shas.items())]
    lines += [r"\begin{table*}[t]",
              r"\caption{Update pairs and their cells at $L^*$. Each pair has one ID; a pair run on two datasets "
              r"has a cell on each. $L^*$ and $C$ are selected on the pre-update model. A revision is the first "
              r"seven characters of the pinned hash (Supplement~\ref{supp:revisions}). $^\dagger$Not a headline "
              r"cell.}",
              r"\label{tab:pairs}",
              r"\centering\footnotesize",
              r"\begin{tabular}{@{}llllrl@{}}",
              r"\toprule",
              r"ID & Pre-update checkpoint & Post-update checkpoint & Dataset & $L^*$ & $C$ \\"]
    for phase in (1, 2):
        lines += [r"\midrule", rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{Phase~{phase}}}}} \\"]
        for x in [r for r in out if r["phase"] == phase]:
            mark = "" if x["headline"] else r"$^\dagger$"
            lines.append(f"{x['id']}{mark} & {short(x['pre'])} & {short(x['post'])} & {x['dataset']} & "
                         f"{x['L']} & {sci(x['C'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    out, shas = rows()
    OUT.write_bytes(tex(out, shas).encode("utf-8"))
    if "--print" in argv:
        for x in out:
            print(x["id"], x["phase"], x["record"], x["dataset"], "headline" if x["headline"] else "outside")
        n = len({x["id"] for x in out})
        print(f"{len(out)} rows, {sum(x['headline'] for x in out)} headline cells, {n} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
