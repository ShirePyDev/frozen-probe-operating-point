"""D-440 (6) review fix: the EXPLICIT-LAYER durable subset of a Phase 2 model's rest set (D-430 (8)), a NEW module that
imports src/subset_p2.py UNCHANGED (subset_p2.py is pinned: twomoment_p2_d361.CODE_FILES and A2's code block name its
bytes, so it is never edited). Written for D-354's grouped cell OLMo-2 base -> SFT C=0.003 L16 (D-424 (7)).

  python3 src/subset_p2x.py --repo <repo> --explicit-layers L [L ...] --poolings <...> --explicit-reason "<D-entries>"
                            --src-dir <NVMe p2cache> --out-dir <NVMe dir> [--out <test record outside the repo>]

cuts the named layers ONLY, for the named poolings, from the model's p2tpl REST extraction record (the record may say
ingest 'never': its caches are cut, never ingested whole, as D-430 (8) routes it) into the tag p2subx (EXPLICIT_TAG):
caches <org>__<model>__p2subx@<rev12>__<axis>__<part>__<pooling>.npz and the record
results/phase2/extract__<org>__<model>__p2subx@<rev12>__rest.json (run_kind 'subset_p2x'), so neither collides with the
D-325 p2sub record of the same model (every p2sub reader globs 'p2sub@', which 'p2subx@' does not match). The model's
D-325 p2sub rest record must exist and be cut from the same source record (file and sha256); a layer it already holds
for a named pooling refuses (nothing to add). Every split of a named pooling is kept; the others are listed as not
carried. The cut and its checks are subset_p2.cut (called unchanged); the write is subset_p2.finish's lines under this
module's own code block (paired.code_block refuses a loaded src module left out, so finish itself cannot be called).
Then: bash scripts/p2_ingest_record.sh <the record> <--out-dir>.
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as P  # noqa: E402
import safe_outputs as SO  # noqa: E402
import subset_p2 as SUB  # noqa: E402  (imported, not changed: cut, record_for, grid6, pairs_of, cache_name, paths)
from provenance import (_RESULT_READS, _repo_rel, canonical_hash, ensure_utf8_stdout, provenance_block,  # noqa: E402
                        read_result, sha256_bytes, write_results)

OWN = "src/subset_p2x.py"
CODE_FILES = list(dict.fromkeys([OWN, *SUB.CODE_FILES]))
RUN_KIND = "subset_p2x"
EXPLICIT_TAG = "p2subx"                                        # D-430 (8), D-440 (6)
EXPLICIT_LAYER_RULE = ("--explicit-layers: exactly the layers named on the command line (each an even layer of the "
                       "source's full grid and none already in the model's D-325 p2sub record for that pooling), for "
                       "the poolings named; the reason recorded (--explicit-reason)")
refuse = SUB.refuse


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Phase 2 explicit-layer subset (tag p2subx; D-430 (8), D-440 (6))")
    ap.add_argument("--repo", required=True, help="a Phase 2 model (its p2tpl rest record is cut)")
    ap.add_argument("--explicit-layers", nargs="+", type=int, required=True)
    ap.add_argument("--poolings", nargs="+", required=True, choices=SUB.POOLINGS)
    ap.add_argument("--explicit-reason", required=True, help="the D-entries that call for the layers (recorded)")
    ap.add_argument("--src-dir", required=True, help="NVMe directory holding the source caches")
    ap.add_argument("--out-dir", required=True, help="NVMe directory for the subset caches (staged from there)")
    ap.add_argument("--out", default=None, help="a test: the record's path, outside the repository")
    return ap


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(128 + sig))
    if hasattr(signal, "SIGHUP"):              # absent on Windows (the workstation's tests)
        signal.signal(signal.SIGHUP, lambda sig, frame: sys.exit(128 + sig))
    args = build_parser().parse_args(argv)
    code0 = P.code_block(CODE_FILES)
    if not str(args.explicit_reason).strip():
        refuse("--explicit-reason must name the D-entries that call for the layers")
    if len(set(args.poolings)) != len(args.poolings):
        refuse(f"--poolings {args.poolings} names a pooling twice")
    poolings = [p for p in SUB.POOLINGS if p in args.poolings]
    layers = sorted(set(args.explicit_layers))
    if len(layers) != len(args.explicit_layers):
        refuse(f"--explicit-layers {args.explicit_layers} names a layer twice")
    cc = read_result(SUB.CONFIG_CHECK)
    repo = args.repo
    if repo not in (cc.get("models") or {}):
        refuse(f"the config check names no model {repo}")
    rec_path = SUB.record_for(repo, "rest")
    rec = read_result(rec_path)
    rec_rel = _repo_rel(rec_path)
    if rec.get("repo") != repo or rec.get("set_name") != "rest" or rec.get("run_kind") != "extract_p2":
        refuse(f"{rec_rel} names repo {rec.get('repo')!r}, set {rec.get('set_name')!r}, run_kind {rec.get('run_kind')!r}")
    rev = rec.get("revision")
    if not isinstance(rev, str) or len(rev) != 40 or rec.get("cache_repo") != f"{repo}__{SUB.SRC_TAG}" \
            or (cc["models"][repo] or {}).get("sha") != rev:
        refuse(f"{rec_rel}: revision or cache_repo is not the documented-tag form, or not the config check's commit")
    src_layers = list(rec.get("layers_resolved") or [])
    if not rec.get("splits"):
        refuse("the source record names no cache")
    nl = rec.get("n_model_layers") or next(iter(rec["splits"].values())).get("n_model_layers")
    if not isinstance(nl, int) or src_layers != list(range(0, nl + 1, 2)):
        refuse(f"the source record does not hold the full every-second-layer grid of {nl} layers: {src_layers}")
    off = [x for x in layers if x not in src_layers]
    if off:
        refuse(f"--explicit-layers {off} are not layers of the source record ({src_layers})")
    slug = repo.replace("/", "__")
    comp_path = SUB.P2 / f"extract__{slug}__{SUB.TAG}@{rev[:12]}__rest.json"
    if comp_path.is_symlink() or not comp_path.is_file():
        refuse(f"no D-325 subset record {_repo_rel(comp_path)}: the explicit layers add to it, they do not replace it")
    comp = read_result(comp_path)
    comp_rel = _repo_rel(comp_path)
    if (comp.get("run_kind"), comp.get("cache_tag"), comp.get("repo"), comp.get("revision"), comp.get("set_name")) \
            != ("subset_p2", SUB.TAG, repo, rev, "rest") \
            or (comp.get("source_record") or {}) != {"file": rec_rel, "sha256": _RESULT_READS[rec_rel]}:
        refuse(f"{comp_rel} is not the D-325 subset of {rec_rel} as it is now")
    held = {pl: list((comp.get("layers_resolved_per_pooling") or {}).get(pl) or []) for pl in poolings}
    dup = {pl: [x for x in layers if x in held[pl]] for pl in poolings}
    if any(dup.values()):
        refuse(f"{comp_rel} already holds {dup}: nothing to add")
    plan: dict[str, Any] = {"n_model_layers": nl, "source_layers": src_layers, "grid6": SUB.grid6(nl),
                            "explicit_layers": layers, "pairs": [list(p) for p in SUB.pairs_of(repo, cc)],
                            "per_pooling": {pl: {"layers": layers, "explicit": True,
                                                 "d325_subset_layers": held[pl]} for pl in poolings}}
    out_rec = Path(args.out) if args.out else SUB.P2 / f"extract__{slug}__{EXPLICIT_TAG}@{rev[:12]}__rest.json"
    if args.out and out_rec.resolve().is_relative_to(SUB.ROOT.resolve()):
        refuse("--out (a test) must lie outside the repository")
    if out_rec.exists():
        refuse(f"{out_rec} exists; results are never overwritten (D-152 P5)")
    src_dir, out_dir = Path(args.src_dir), Path(args.out_dir)
    if out_dir.resolve().is_relative_to(SUB.ROOT.resolve()):
        refuse(f"--out-dir {out_dir} lies inside the repository (the NAS); subset caches are staged from NVMe")
    todo, not_kept = [], {}
    for key, sp in sorted((rec.get("splits") or {}).items()):
        parts = key.split("/")
        if len(parts) != 3:
            refuse(f"{rec_rel} split key {key!r} is not '<axis>/<part>/<pooling>'")
        k_axis, part, pooling = parts
        if pooling not in SUB.POOLINGS or sp.get("pooling") != pooling or sp.get("n_nonfinite") != 0 \
                or sp.get("n_truncated") != 0:
            refuse(f"{rec_rel} split {key}: unexpected pooling, or non-finite or truncated items")
        if pooling not in poolings:
            not_kept[key] = "its pooling is not named by --poolings"
            continue
        name = SUB.cache_name(repo, rev, k_axis, part, pooling, EXPLICIT_TAG)
        if (out_dir / name).exists():
            refuse(f"{out_dir / name} exists (never overwritten)")
        todo.append((key, sp, name))
    if not todo:
        refuse(f"{rec_rel}: no cache is kept under these parameters")
    print(f"{repo}@{rev[:12]} [{EXPLICIT_TAG} rest]: {len(todo)} caches; layers {layers}; poolings {poolings}"
          + (f"; not carried {sorted(not_kept)}" if not_kept else ""), flush=True)
    reason = str(args.explicit_reason).strip()
    config = {"repo": repo, "set": "rest", "tag": EXPLICIT_TAG, "source_tag": SUB.SRC_TAG, "poolings": poolings,
              "layer_rule": EXPLICIT_LAYER_RULE, "explicit_layers": layers, "explicit_reason": reason,
              "compression": "np.savez_compressed"}
    tail = {"explicit_layers": layers, "explicit_reason": reason,
            "d325_subset_record": {"file": comp_rel, "sha256": _RESULT_READS[comp_rel],
                                   "layers_resolved_per_pooling": held},
            "splits_not_kept": not_kept, "source_record_ingest": rec.get("ingest"),
            "readers_key_by": "the tag p2subx: no p2sub reader's glob matches it; a reader names this record explicitly "
                              "(src/p2auroc_d424_grouped.py, CHOICES (g5))"}
    written_now: list[tuple[Path, str]] = []
    try:
        return write(repo, rev, rec_rel, plan, todo, src_dir, out_dir, out_rec, code0, config, tail, poolings,
                     reason, written_now)
    except BaseException as e:
        if out_rec.exists():                     # the record names the caches under their final names: keep them
            raise
        left = []
        for path, digest in written_now:
            try:
                aside = SO.aside_name(path)
                SO.noreplace_rename(path, aside)
                left.append(f"{aside} (this run's cache, sha256 {digest[:16]})")
            except OSError as e2:
                left.append(f"{path} (left in place: {type(e2).__name__})")
        if left:
            print("FAILED after writing caches; renamed aside (nothing deleted):\n  " + "\n  ".join(left), file=sys.stderr)
        raise e


def write(repo: str, rev: str, rec_rel: str, plan: dict[str, Any], todo: list[Any], src_dir: Path, out_dir: Path,
          out_rec: Path, code0: dict[str, Any], config: dict[str, Any], tail: dict[str, Any], poolings: list[str],
          reason: str, written_now: list[tuple[Path, str]]) -> int:
    """subset_p2.finish's lines (each cache cut by subset_p2.cut, written by safe_outputs.write_all, then the record),
    under THIS module's code block."""
    splits: dict[str, Any] = {}
    for key, s, name in todo:
        layers = plan["per_pooling"][s["pooling"]]["layers"]
        data, info = SUB.cut(src_dir / s["cache_file"], s["cache_sha256"], layers)
        if info["n_items"] != s.get("n") or info["source_layers"] != plan["source_layers"]:
            refuse(f"{s['cache_file']}: {info['n_items']} items or layers {info['source_layers']} differ from the record")
        written = SO.write_all([(out_dir / name, data)])
        digest = written[out_dir / name]
        written_now.append((out_dir / name, digest))
        if digest != sha256_bytes(data):
            refuse(f"{name}: the written bytes' sha256 is not the serialised bytes'")
        splits[key] = {"split": s["split"], "pooling": s["pooling"], "n": s["n"], "n_items": s["n"],
                       "split_file": s["split_file"], "split_sha256": s["split_sha256"],
                       "cache_file": name, "cache_sha256": digest, "cache_bytes": len(data), "cache_compressed": True,
                       "n_nonfinite": 0, "n_truncated": 0, "layers_resolved": layers,
                       "n_model_layers": plan["n_model_layers"], "hidden_size": s.get("hidden_size"),
                       "kept_keys": info["kept_keys"],
                       "read_back": "safe_outputs.write_all: staged under an exclusive name, fsynced, size and sha256 "
                                    "verified, hard-linked to the final name, read back (D-307 (4))",
                       "source_cache_file": s["cache_file"], "source_cache_sha256": s["cache_sha256"],
                       "source_layers": info["source_layers"],
                       "equal_to_source": "every kept array, re-loaded from the written bytes, equals the source's "
                                          "bit for bit (dtype, shape, bytes)"}
        print(f"  {key}: {name}  {len(data)} bytes  layers {layers}", flush=True)
    if P.code_block(CODE_FILES)["sha256"] != code0["sha256"]:
        refuse("source changed during the run")
    sel_reads = {k: v for k, v in _RESULT_READS.items() if Path(k).name.startswith(("selection_p2__", "selection_p2sst__"))}
    cc_rel = _repo_rel(SUB.CONFIG_CHECK)
    digest = write_results(out_rec, {
        **provenance_block(RUN_KIND),
        "decision": f"--explicit-layers (D-430 (8), D-440 (6)); the job's reason: {reason}",
        "code": code0,
        "repo": repo, "revision": rev, "set_name": "rest", "ingest": "yes",
        "ingest_rule": "D-307 (1): the durable layer subset of NVMe-only caches; ingested and indexed before any "
                       "result reads it",
        "cache_repo": f"{repo}__{EXPLICIT_TAG}", "cache_revision": rev, "cache_tag": EXPLICIT_TAG,
        "cache_tag_documentation": f"D-297 (1) documented tag: cache_repo is repo + '__{EXPLICIT_TAG}' at the same full "
                                   f"revision; a {EXPLICIT_TAG} cache is a layer subset of the {SUB.SRC_TAG} cache it "
                                   "names",
        "config": config, "config_hash": canonical_hash(config),
        "layer_rule": EXPLICIT_LAYER_RULE, "layer_plan": plan,
        "source_record": {"file": rec_rel, "sha256": _RESULT_READS[rec_rel]},
        "inputs": {rec_rel: _RESULT_READS[rec_rel], cc_rel: _RESULT_READS[cc_rel], **sel_reads,
                   tail["d325_subset_record"]["file"]: tail["d325_subset_record"]["sha256"]},
        "n_model_layers": plan["n_model_layers"],
        "layers_resolved_per_pooling": {pl: plan["per_pooling"][pl]["layers"] for pl in poolings},
        "splits": splits,
        **tail,
    }, exclusive=True)
    print(f"Written: {_repo_rel(out_rec)}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
