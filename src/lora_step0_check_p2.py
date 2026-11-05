"""The D-343 (2) step-0 check for the Phase 2 LoRA adapters ('step-0 adapters checked against the base as in Phase 1'):
a step-0 adapter (LoRA B = 0) merged into the base must reproduce the base's own caches, so that every later LoRA
difference is the update's and not the extraction path's. It is a GATE: any difference beyond the tolerance writes the
record and exits nonzero, which stops the lane (scripts/queue_runner.sh), so no subset of that seed is cut.

THE RULE IS PHASE 1'S (src/compare_caches.py, the gate behind D-081 and D-087, reused, not copied: its tolerance and its
gate-failure exit code are read from it): item_ids equal; labels, truncated and nonfinite equal; every common layer's
maximum |a - b| in float64 at most compare_caches' --tol default (read from its source by phase1_rule, never typed);
a written record whose gate failed exits compare_caches.GATE_FAIL, a refusal exits 1. Two Phase 2 additions:
  - every layer: both caches must hold the same layers, the full grid of both records, so 'every common layer' is
    every layer extracted;
  - plain-path items: when the two records name different span functions (the base's p2tpl records predate D-324
    (5)(v) and name phase2_tokenizer_check.user_span; an adapter's name user_span_plain), an item either record lists
    on the plain-text path (special_strings_as_text) is tokenized differently BY CONSTRUCTION (the WildGuardMix item
    ef72f52bd8c3ec85, for one) and is left out of the comparison, listed in the record; such items stay out of LoRA
    evaluation (D-324 (5)(iii)). With one span function on both sides nothing is excluded.

INPUTS: --adapter-record, an adapter's step-0 'select' extraction record (results/phase2/extract__lora__<name>-step0__
<tag>@<sha12>__select.json, src/extract_p2.py --lora-record ... --lora-step 0; its caches in --src-dir on NVMe), and the
base's record of the same tag and set, named from the adapter record's model_repo and model_revision: ingest 'yes' and
ingested (manifests/ingest_p2/<stem>.sha256), its caches read from data/hidden at the full grid. Only the select set is
compared: it is the base's set held at the full grid on the NAS (D-318 (7) item 5). Each cache is hashed against its
record before it is read and again after (compare_caches' rule); both records are read through read_result.
OUTPUT: results/phase2/step0check_p2__lora__<name>__<tag>@<sha12>.json, written once (a re-run refuses 'exists').

    python3 -u src/lora_step0_check_p2.py --adapter-record <record> --src-dir <NVMe dir of its caches>
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_caches as CC  # noqa: E402  (Phase 1's gate: its rule, tolerance and exit code)
import paired as P  # noqa: E402
import provenance as PV  # noqa: E402
from provenance import ensure_utf8_stdout, provenance_block, read_result, sha256_file, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CODE_FILES = ["src/lora_step0_check_p2.py", "src/compare_caches.py", "src/provenance.py", "src/paired.py"]
FLAGS = ("labels", "truncated", "nonfinite")                  # compare_caches: the flags that must be equal
ADAPTER_RE = re.compile(r"extract__lora__([A-Za-z0-9.-]+-step0)__(p2tpl|p2sst)@([0-9a-f]{12})__select\.json")


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def phase1_rule(path: Path = ROOT / "src" / "compare_caches.py") -> dict[str, Any]:
    """compare_caches' --tol default, read from its source with the line it is on, and the file's sha256."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    found = [(kw.value.value, n.lineno) for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument" and n.args
             and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--tol"
             for kw in n.keywords if kw.arg == "default" and isinstance(kw.value, ast.Constant)]
    if len(found) != 1 or not isinstance(found[0][0], float):
        refuse(f"{Path(path).name} no longer states one float --tol default (Phase 1's gate, D-081)")
    return {"file": "src/compare_caches.py", "sha256": sha256_file(path), "tol": found[0][0],
            "tol_default_line": found[0][1], "gate_fail_exit": CC.GATE_FAIL}


def plain_items(base: dict[str, Any], cand: dict[str, Any], key: str) -> tuple[set[str], str]:
    """The items left out of one cache's comparison and why (module docstring)."""
    sa, sb = base.get("user_span_source"), cand.get("user_span_source")
    if sa == sb:
        return set(), f"both records name {sa}: no item is excluded"
    ids: set[str] = set()
    for rec in (base, cand):
        e = (rec["splits"][key].get("special_strings_as_text") or {}) if isinstance(rec["splits"][key], dict) else {}
        ids |= {str(x["item_id"]) for x in e.get("items") or [] if isinstance(x, dict) and "item_id" in x}
    return ids, (f"the base's record names {sa} and the adapter's {sb}: the items either lists on the plain-text path "
                 "(D-324 (5)(v)) are tokenized differently by construction and are excluded; they stay out of LoRA "
                 "evaluation (D-324 (5)(iii))")


def compare(base: Path, base_sha: str, cand: Path, cand_sha: str, exclude: set[str], tol: float) -> dict[str, Any]:
    """compare_caches' rule on the rows whose item_id is not excluded, over every layer of both caches."""
    for p, sha in ((base, base_sha), (cand, cand_sha)):
        if p.is_symlink() or not p.is_file():
            refuse(f"{p} is not a regular file")
        if sha256_file(p) != sha:
            refuse(f"{p.name} is not the bytes its record names ({sha[:16]})")
    rep: dict[str, Any] = {"base": base.name, "candidate": cand.name, "base_sha256": base_sha,
                           "candidate_sha256": cand_sha, "layers": {}}
    with np.load(base, allow_pickle=False) as A, np.load(cand, allow_pickle=False) as B:
        ok = rep["item_ids_equal"] = bool(np.array_equal(A["item_ids"], B["item_ids"]))
        if ok:
            ids = A["item_ids"]
            keep = ~np.isin(ids.astype(str), sorted(exclude))
            rep.update({"n_items": int(len(ids)), "n_compared": int(keep.sum()),
                        "excluded_item_ids": sorted(str(x) for x in ids[~keep])})
            for flag in FLAGS:
                same = flag in A.files and flag in B.files and bool(np.array_equal(A[flag][keep], B[flag][keep]))
                rep[f"{flag}_equal"] = same
                ok &= same
            la = {k for k in A.files if k.startswith("layer_")}
            lb = {k for k in B.files if k.startswith("layer_")}
            rep["same_layers"] = la == lb and bool(la)
            ok &= rep["same_layers"]
            worst = 0.0
            for k in sorted(la & lb, key=lambda s: int(s.split("_")[1])):
                a, b = A[k][keep], B[k][keep]
                d = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) if a.size else 0.0
                rep["layers"][k] = {"max_abs_diff": d, "bit_identical": bool(np.array_equal(a, b))}
                worst = max(worst, d)
            rep["max_abs_diff_any_layer"] = worst
            ok &= worst <= tol
    for p, sha in ((base, base_sha), (cand, cand_sha)):
        if sha256_file(p) != sha:
            refuse(f"{p.name} changed during the comparison")
    rep["PASS"] = bool(ok)
    return rep


def run(adapter_record: str, src_dir: str, root: Path = ROOT, out: Path | None = None,
        allow_no_commit: bool = False) -> int:
    root = Path(root)
    p2, hidden, index = root / "results" / "phase2", root / "data" / "hidden", root / "manifests" / "ingest_p2"
    code0 = P.code_block(CODE_FILES, allow_no_commit=allow_no_commit)
    rule = phase1_rule()
    rec_path = Path(adapter_record) if Path(adapter_record).is_absolute() else root / adapter_record
    m = ADAPTER_RE.fullmatch(rec_path.name)
    if not m or rec_path.resolve().parent != p2.resolve():
        refuse(f"{adapter_record} is not results/phase2/extract__lora__<name>-step0__<tag>@<sha12>__select.json")
    if not rec_path.is_file():
        refuse(f"no record {rec_path.name}")
    tag = m.group(2)
    cand = read_result(rec_path)
    src = cand.get("adapter_source") or {}
    if cand.get("run_kind") != "extract_p2" or cand.get("repo") != f"lora/{m.group(1)}" or src.get("step") != 0 or \
            cand.get("set_name") != "select" or cand.get("cache_tag") != tag:
        refuse(f"{rec_path.name} is not a step-0 adapter's select extraction (run_kind, repo, adapter_source.step, "
               "set_name, cache_tag)")
    brepo, brev = cand.get("model_repo"), cand.get("model_revision")
    if not isinstance(brepo, str) or not re.fullmatch(r"[0-9a-f]{40}", str(brev or "")):
        refuse(f"{rec_path.name} names no base model_repo and 40-hex model_revision")
    base_path = p2 / f"extract__{brepo.replace('/', '__')}__{tag}@{brev[:12]}__select.json"
    if not base_path.is_file():
        refuse(f"no base record {base_path.name}")
    base = read_result(base_path)
    if (base.get("repo"), base.get("revision"), base.get("adapter"), base.get("ingest")) != (brepo, brev, None, "yes"):
        refuse(f"{base_path.name} is not the base's own select extraction with ingest 'yes'")
    if not (index / f"{base_path.stem}.sha256").is_file():
        refuse(f"{base_path.stem} is not ingested (no manifests/ingest_p2/{base_path.stem}.sha256): the base's "
               "full-grid select caches are read from data/hidden")
    if not cand.get("layers_resolved") or base.get("layers_resolved") != cand.get("layers_resolved"):
        refuse(f"the two records name different layers: {base.get('layers_resolved')} vs {cand.get('layers_resolved')}")
    keys = sorted(cand.get("splits") or {})
    if not keys or any(k not in (base.get("splits") or {}) for k in keys):
        refuse(f"{rec_path.name} names caches the base's select record does not: "
               f"{[k for k in keys if k not in (base.get('splits') or {})]}")
    out_path = Path(out) if out else p2 / f"step0check_p2__lora__{m.group(1)}__{tag}@{m.group(3)}.json"
    if out_path.exists():
        refuse(f"{out_path.name} exists; results are never overwritten")
    caches: dict[str, Any] = {}
    for key in keys:
        bs, cs = base["splits"][key], cand["splits"][key]
        exclude, why = plain_items(base, cand, key)
        rep = compare(hidden / bs["cache_file"], bs["cache_sha256"], Path(src_dir) / cs["cache_file"],
                      cs["cache_sha256"], exclude, rule["tol"])
        caches[key] = {**rep, "exclusion_rule": why}
        print(f"  {key}: {'PASS' if rep['PASS'] else 'FAIL'}  layers {len(rep['layers'])}  max |diff| "
              f"{rep.get('max_abs_diff_any_layer')}  excluded {rep.get('excluded_item_ids')}", flush=True)
    ok = all(c["PASS"] for c in caches.values())
    if P.code_block(CODE_FILES, allow_no_commit=allow_no_commit)["sha256"] != code0["sha256"]:
        refuse("the code changed during the run; nothing is written")
    digest = write_results(out_path, {
        **provenance_block("lora_step0_check_p2"), "code": code0,
        "decision": "D-343 (2): step-0 adapters checked against the base as in Phase 1 (compare_caches, D-081)",
        "phase1_rule": rule, "adapter_record": {"file": PV._repo_rel(rec_path), "sha256": PV._RESULT_READS[
            PV._repo_rel(rec_path)]}, "base_record": {"file": PV._repo_rel(base_path), "sha256": PV._RESULT_READS[
                PV._repo_rel(base_path)]},
        "identity": cand["repo"], "base": [brepo, brev], "tag": tag, "caches": caches, "PASS": ok,
        "input_files_sha256": dict(PV._RESULT_READS)}, exclusive=True)
    print(f"{'GATE PASS' if ok else 'GATE FAIL'}\nWritten: {PV._repo_rel(out_path)}\nsha256:  {digest}", flush=True)
    return 0 if ok else rule["gate_fail_exit"]


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="D-343 (2): a step-0 adapter's caches against the base's (Phase 1's gate)")
    ap.add_argument("--adapter-record", required=True,
                    help="results/phase2/extract__lora__<name>-step0__<tag>@<sha12>__select.json")
    ap.add_argument("--src-dir", required=True, help="NVMe directory holding the adapter's caches")
    args = ap.parse_args(argv)
    return run(args.adapter_record, args.src_dir)


if __name__ == "__main__":
    raise SystemExit(main())
