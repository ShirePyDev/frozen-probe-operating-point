"""Phase 2 durable layer subsets (D-303 (1), D-324 (1), D-325): the selected layer(s) plus the 6-layer grid, cut from a
model's NVMe-only full-grid caches, written compressed for the NAS and recorded so scripts/p2_ingest_record.sh can
ingest and index them. Nothing is computed from the hidden states: every kept array is copied bit for bit.

WHY: every model's 'rest' caches (eval, pool4d, over_refusal/XSTest) and the 'select' caches of models that are never a
pair's pre side exist only on NVMe scratch (ingest 'never', or not ingested), and NVMe did not survive the 2026-09-28
restart (D-264). The pre sides' 'select' caches are already ingested at the full grid and are not duplicated
(D-318 (7) item 5).

LAYERS, per model and pooling (LAYER_RULE, recorded):
  - GRID6(NL): for k = 0..5, the even layer nearest k * NL / 5 (NL = the model's layer count; no tie occurs for
    NL = 28 or 32, and a tie refuses); {0, 6, 12, 20, 26, 32} at 32 layers, {0, 6, 12, 16, 22, 28} at 28 (D-318 (7)
    item 3, the 2026-10-01 proposal);
  - plus L*, the D-117 selected layer of the PRE side of every pair the model belongs to (as pre or as post), read
    from that pre side's selection record for the same pooling (results/phase2/selection_p2__<pre>@<rev12>__<pooling>
    .json, 'model.selected.layer'); a missing selection record refuses;
  - a model in FULL_GRID keeps every layer of its source caches (recorded with the reason).
PAIRS: the config check's comparable pairs (results/phase2_config_check.json 'pairs') minus EXCLUDED_PAIRS.

SETS: --set rest for every model; --set select only for a model that is a post side and no pair's pre side, and not
in FULL_GRID (refused otherwise, and refused if its select record is already ingested): a pre side's select set is
ingested at the full grid by the selection pipeline, and a FULL_GRID or pairless model's full select record (ingest
'yes') is ingested directly with scripts/p2_ingest_record.sh instead of being re-cut (D-325).

INPUTS: the model's extraction record results/phase2/extract__<org>__<model>__p2tpl@<rev12>__<set>.json (read through
provenance.read_result), each source cache in --src-dir checked against the record's sha256 before it is read, the
config check and the selection records (read_result; all listed with their sha256).

OUTPUTS: per source cache, <org>__<model>__p2sub@<rev12>__<axis>__<part>__<pooling>.npz in --out-dir (NVMe; written
through safe_outputs.write_all: exclusive, verified, never overwritten), holding the source's non-layer arrays and its
kept layer arrays, each equal bit for bit to the source (checked on the bytes to be written); and the record
results/phase2/extract__<org>__<model>__p2sub@<rev12>__<set>.json (run_kind 'subset_p2', 'ingest': 'yes', a 'splits'
block in the extraction record's form). It is named like an extraction record ON PURPOSE: it documents the p2sub tag
the way an extraction record documents p2tpl (D-297 (1)), so the provenance audit indexes it and traces every p2sub
cache to it. Every Phase 2 reader globs by tag (p2tpl), so none mistakes it for an extraction. Then:
bash scripts/p2_ingest_record.sh <the record> <--out-dir>. Each source cache is read ONCE: its bytes are hashed and
loaded from the same buffer. On a refusal after a cache was written, the caches this run wrote are renamed aside to
'<name>.failed.<pid>' (never deleted), so a rerun is not blocked by unrecorded caches. A re-cut (a new pair or a
corrected L*) needs a new tag, since names are never reused.

  python3 src/subset_p2.py --repo <repo> --set {rest,select} --src-dir <NVMe p2cache> --out-dir <NVMe p2sub>

LORA ADAPTERS AND SST-2 (D-343, D-338 (4), D-340 (3)): two further sources, each with its layer set as a PARAMETER, so
that one tool serves every storage option D-343 (1) put to the author without choosing one (nothing here has a default
for them; the job list names the option, queues/laneB_*_lora_{wg,sst}_<option>.tsv):
  --adapter-name <name>   a merged adapter's caches (src/extract_p2.py --lora-record: identity lora/<name> at the sha256
                          of its weights, the base in the record's model_repo / model_revision). Its pair is base ->
                          base+adapter, so L* is the BASE's (the pre side's) D-117 selected layer for the pooling, from
                          the base's selection record of the source's axis: selection_p2__<base>@<rev12>__<pooling>.json
                          for p2tpl (WildGuardMix and XSTest), selection_p2sst__<base>@<rev12>__<pooling>.json
                          (src/selection_p2sst.py, run_kind selection_p2sst) for p2sst. Both sets (select, rest) may be
                          cut: an adapter is a post side only.
  --repo <repo> --source-tag p2sst   the base's own SST-2 rest set (its select set, ingested at the full grid because
                          the selection reads it, is not re-cut, D-307 (1), D-318 (7) item 5); L* is its own p2sst
                          selection (it is the pre side of every SST-2 pair, base -> base+adapter).
  Required with either: --layer-set grid6_lstar (GRID6 plus L*, as D-325) or lstar (L* alone); --poolings (one or
  both: a pooling not named is not carried); for an adapter also --storage-option A|B|C|D (D-343 (1)'s option the job
  list implements, recorded and checked against D-343 (1)'s definitions in OPTION_RULES; option D per seed, the seed
  read from the adapter record's adapter_source, OPTION_D_SEVEN_LAYER_SEED), and --post-probe-train
  keep|drop when its source record holds a probe_train split (D-343 (1): 'a post side trains no probe'; option A as
  priced there kept it). The record lists every source split not carried, with the reason.
  Tags: a p2tpl source is cut to p2sub (as D-325); a p2sst source to p2sstsub, so that no p2sst subset can take a p2sub
  record name (the base's p2sub rest record exists) and phase2_bos_check's p2tpl/p2sub scan never meets one. The Phase 2
  models' D-325 path above takes none of these options and is unchanged.
"""

from __future__ import annotations

import argparse
import io
import re
import signal
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as P  # noqa: E402
import safe_outputs as SO  # noqa: E402
from provenance import (_RESULT_READS, _repo_rel, canonical_hash, ensure_utf8_stdout, provenance_block,  # noqa: E402
                        read_result, sha256_bytes, write_results)

ROOT = Path(__file__).resolve().parent.parent
P2 = ROOT / "results" / "phase2"
CONFIG_CHECK = ROOT / "results" / "phase2_config_check.json"
INGEST_INDEX = ROOT / "manifests" / "ingest_p2"
CODE_FILES = ["src/subset_p2.py", "src/paired.py", "src/safe_outputs.py", "src/provenance.py"]
SRC_TAG, TAG = "p2tpl", "p2sub"
POOLINGS = ("usermean", "lastgen")
EXCLUDED_PAIRS = {("microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3.5-mini-instruct"):
                  "D-318 (7) item 7: the Phi pair is Phi-3-mini-128k -> Phi-3.5"}
FULL_GRID = {"allenai/OLMo-2-1124-7B-Instruct":
             "D-325: its pair's pre side (OLMo-2-1124-7B-DPO) is not yet extracted or selected, so its L* is unknown; the "
             "full grid holds whatever L* the selection gives"}
LAYER_RULE = ("per model and pooling: GRID6 (for k = 0..5 the even layer nearest k*NL/5) plus L*, the D-117 selected "
              "layer of the pre side of every pair the model belongs to, from that pre side's selection record for the "
              "same pooling; a FULL_GRID model keeps every source layer")
SUB_TAGS = {"p2tpl": "p2sub", "p2sst": "p2sstsub"}            # source tag -> subset tag
AXIS_OF = {"p2tpl": "p2wg", "p2sst": "p2sst"}                  # the selection axis of a source tag (selection_p2)
LAYER_SETS = {"grid6_lstar": "GRID6 plus L* (D-325's rule)", "lstar": "L* alone (D-343 (1) option C)"}
PAIR_LAYER_RULE = ("per pooling: the --layer-set ('grid6_lstar': GRID6, for k = 0..5 the even layer nearest k*NL/5, plus "
                   "L*; 'lstar': L* alone), where L* is the D-117 selected layer of the PRE side of the pair (an adapter's "
                   "base; the base itself for its own SST-2 rest set), from the pre side's selection record of the "
                   "source's axis for the same pooling")
# D-343 (1)'s options as it defines them: per source tag, the (layer set, poolings, post probe_train) each allows.
# probe_train None: the set holds none. SST-2 is 'extrapolated beside' there: A at GRID6 + L* both poolings without its
# probe_train; B user-span mean at the 7 layers; C 'at L* user-span mean only' (both poolings priced beside); D defines
# no SST-2 storage, so it refuses (a question for the author, not a choice made here).
BOTH = ("usermean", "lastgen")
OPTION_RULES: dict[str, dict[str, list[tuple[str, tuple[str, ...], tuple[Any, ...]]]]] = {
    "p2tpl": {"A": [("grid6_lstar", BOTH, ("keep", "drop", None))],
              "B": [("grid6_lstar", ("usermean",), ("drop", None))],
              "C": [("lstar", BOTH, ("drop", None))],
              "D": [("grid6_lstar", BOTH, ("drop", None)), ("lstar", BOTH, ("drop", None))]},
    "p2sst": {"A": [("grid6_lstar", BOTH, ("drop", None))],
              "B": [("grid6_lstar", ("usermean",), ("drop", None))],
              "C": [("lstar", ("usermean",), ("drop", None)), ("lstar", BOTH, ("drop", None))]},
}
# D-343 (1) option D is per seed: 'seed 0 at the 7 layers ... and seeds 1-2 at L* only'. The seed is read from the
# adapter record (adapter_source.seed, src/lora_adapter_p2.py), never from a job line (code review of fed8bd1, minor).
OPTION_D_SEVEN_LAYER_SEED = 0


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def grid6(n_layers: int) -> list[int]:
    out = []
    for k in range(6):
        x = k * n_layers / 5
        lo = 2 * int(x // 2)
        cands = sorted({lo, lo + 2}, key=lambda e: abs(e - x))
        if len(cands) == 2 and abs(cands[0] - x) == abs(cands[1] - x):
            refuse(f"GRID6 tie at k={k}, NL={n_layers} (x={x})")
        out.append(cands[0])
    return sorted(set(out))


def pairs_of(repo: str, cc: dict[str, Any]) -> list[tuple[str, str]]:
    ps = []
    for p in cc.get("pairs") or []:
        key = (p.get("pre"), p.get("post"))
        if p.get("comparable") is True and key not in EXCLUDED_PAIRS and repo in key:
            ps.append(key)
    return ps


def record_for(repo: str, set_name: str, src_tag: str = SRC_TAG) -> Path:
    found = sorted(P2.glob(f"extract__{repo.replace('/', '__')}__{src_tag}@*__{set_name}.json"))
    if len(found) != 1:
        refuse(f"{len(found)} {set_name} extraction records for {repo} (want exactly one): {[p.name for p in found]}")
    return found[0]


def selected_layer(pre: str, pooling: str, cc: dict[str, Any], axis: str = "p2wg") -> tuple[int, str, str]:
    """L* of the pre side for one pooling: (layer, selection record path, its sha256). axis p2sst reads the pre side's
    SST-2 selection, results/phase2/selection_p2sst__<pre>@<rev12>__<pooling>.json (src/selection_p2sst.py), which must
    say run_kind selection_p2sst and axis p2sst."""
    rev = ((cc.get("models") or {}).get(pre) or {}).get("sha") or ""
    prefix = "selection_p2" if axis == "p2wg" else f"selection_{axis}"
    found = sorted(P2.glob(f"{prefix}__{pre.replace('/', '__')}@{rev[:12]}__{pooling}.json")) if rev else []
    if len(found) != 1:
        refuse(f"no single selection record for the pre side {pre} ({axis}, {pooling}): {[p.name for p in found]}; its "
               "pair's L* is unknown (run the selection first)")
    sel = read_result(found[0])
    m = sel.get("model") or {}
    if m.get("repo") != pre or m.get("pooling") != pooling or not isinstance((m.get("selected") or {}).get("layer"), int):
        refuse(f"{found[0].name} does not name {pre} / {pooling} with an integer selected layer")
    if axis != "p2wg" and (sel.get("run_kind") != f"selection_{axis}" or (sel.get("config") or {}).get("axis") != axis):
        refuse(f"{found[0].name} is not a selection_{axis} record on axis {axis}")
    rel = _repo_rel(found[0])
    return m["selected"]["layer"], rel, _RESULT_READS[rel]


def layer_plan(repo: str, rec: dict[str, Any], cc: dict[str, Any]) -> dict[str, Any]:
    src_layers = list(rec.get("layers_resolved") or [])
    if not rec.get("splits"):
        refuse("the source record names no cache")
    nl = rec.get("n_model_layers") or next(iter(rec["splits"].values())).get("n_model_layers")
    if not isinstance(nl, int) or src_layers != list(range(0, nl + 1, 2)):
        refuse(f"the source record does not hold the full every-second-layer grid of {nl} layers: {src_layers}")
    pairs = pairs_of(repo, cc)
    plan: dict[str, Any] = {"n_model_layers": nl, "source_layers": src_layers, "grid6": grid6(nl),
                            "pairs": [list(p) for p in pairs], "per_pooling": {}}
    for pooling in POOLINGS:
        if repo in FULL_GRID:
            plan["per_pooling"][pooling] = {"layers": src_layers, "full_grid_reason": FULL_GRID[repo], "l_star": []}
            continue
        lstar = []
        for pre, post in pairs:
            layer, sel_file, sel_sha = selected_layer(pre, pooling, cc)
            if layer not in src_layers:
                refuse(f"L* {layer} of {pre} ({pooling}) is not in the source layers {src_layers}")
            lstar.append({"pair": [pre, post], "role": "pre" if repo == pre else "post", "layer": layer,
                          "selection_record": sel_file, "selection_record_sha256": sel_sha})
        layers = sorted(set(plan["grid6"]) | {x["layer"] for x in lstar})
        plan["per_pooling"][pooling] = {"layers": layers, "l_star": lstar}
    return plan


def pair_layer_plan(rec: dict[str, Any], pre: str, ident: str, role: str, axis: str, layer_set: str,
                    poolings: list[str], cc: dict[str, Any]) -> dict[str, Any]:
    """The layer plan of an adapter's sets (pre = its base, role 'post') or of the base's own SST-2 rest set (pre = the
    model, role 'pre'): per named pooling, the --layer-set around the pre side's L* on the source's axis."""
    src_layers = list(rec.get("layers_resolved") or [])
    if not rec.get("splits"):
        refuse("the source record names no cache")
    nl = rec.get("n_model_layers") or next(iter(rec["splits"].values())).get("n_model_layers")
    if not isinstance(nl, int) or src_layers != list(range(0, nl + 1, 2)):
        refuse(f"the source record does not hold the full every-second-layer grid of {nl} layers: {src_layers}")
    plan: dict[str, Any] = {"n_model_layers": nl, "source_layers": src_layers, "grid6": grid6(nl),
                            "pairs": [[pre, ident]], "layer_set": layer_set, "per_pooling": {}}
    for pooling in poolings:
        layer, sel_file, sel_sha = selected_layer(pre, pooling, cc, axis)
        if layer not in src_layers:
            refuse(f"L* {layer} of {pre} ({axis}, {pooling}) is not in the source layers {src_layers}")
        lstar = [{"pair": [pre, ident], "role": role, "layer": layer, "axis": axis, "selection_record": sel_file,
                  "selection_record_sha256": sel_sha}]
        layers = sorted(set(plan["grid6"]) | {layer}) if layer_set == "grid6_lstar" else [layer]
        plan["per_pooling"][pooling] = {"layers": layers, "l_star": lstar}
    return plan


def check_option(src_tag: str, option: str, layer_set: str, poolings: list[str], probe_train: str | None,
                 seed: Any = None) -> dict:
    """The adapter subset's parameters against D-343 (1)'s definition of the option the job list names (OPTION_RULES);
    option D per seed, the seed the adapter record's (OPTION_D_SEVEN_LAYER_SEED)."""
    allowed = (OPTION_RULES.get(src_tag) or {}).get(option)
    if not allowed:
        refuse(f"D-343 (1) defines no storage under option {option} for {src_tag} (SST-2 under D is not defined there: "
               "a question for the author)")
    if option == "D":
        if not isinstance(seed, int) or isinstance(seed, bool):
            refuse(f"option D is defined per seed (D-343 (1)); the adapter record names no seed (adapter_source.seed: "
                   f"{seed!r})")
        allowed = [a for a in allowed if (a[0] == "grid6_lstar") == (seed == OPTION_D_SEVEN_LAYER_SEED)]
    combo = (layer_set, tuple(p for p in POOLINGS if p in poolings), probe_train)
    if not any(combo[0] == ls and combo[1] == pl and combo[2] in pt for ls, pl, pt in allowed):
        refuse(f"--layer-set {layer_set}, --poolings {list(combo[1])}, --post-probe-train {probe_train} are not D-343 "
               f"(1)'s option {option} for {src_tag}{f' at seed {seed}' if option == 'D' else ''} (allowed: {allowed})")
    return {"option": option, "allowed_by_d343": [[ls, list(pl), list(pt)] for ls, pl, pt in allowed],
            **({"seed": seed, "seed_from": "the adapter record's adapter_source.seed"} if option == "D" else {})}


def cache_name(repo: str, rev: str, axis: str, part: str, pooling: str, tag: str = TAG) -> str:
    return f"{repo.replace('/', '__')}__{tag}@{rev[:12]}__{axis}__{part}__{pooling}.npz"


def same_bytes(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def cut(src: Path, want_sha: str, layers: list[int]) -> tuple[bytes, dict[str, Any]]:
    """The subset of one source cache, serialised (np.savez_compressed) in memory and checked: the source's sha256
    first; then every non-layer array and every kept layer array, re-loaded from the bytes to be written, equal to the
    source's bit for bit."""
    if src.is_symlink() or not src.is_file():
        refuse(f"{src} is not a regular file")
    raw = src.read_bytes()                       # read once: the bytes hashed are the bytes cut
    have = sha256_bytes(raw)
    if have != want_sha:
        refuse(f"{src.name} is not the bytes its record names ({have[:16]} vs {want_sha[:16]})")
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        keys = list(z.files)
        src_layers = sorted(int(k.split("_")[1]) for k in keys if k.startswith("layer_"))
        missing = [x for x in layers if x not in src_layers]
        if missing:
            refuse(f"{src.name} lacks layers {missing}")
        if "nonfinite" not in keys or bool(np.asarray(z["nonfinite"]).any()):
            refuse(f"{src.name} has no nonfinite flag, or flags a non-finite item")
        arrays = {k: z[k] for k in keys if not k.startswith("layer_")}
        arrays.update({f"layer_{x}": z[f"layer_{x}"] for x in layers})
    del raw
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    data = buf.getvalue()
    with np.load(io.BytesIO(data), allow_pickle=False) as back:
        if set(back.files) != set(arrays):
            refuse(f"{src.name}: the subset's key set changed when serialised")
        for k, v in arrays.items():
            if not same_bytes(back[k], v):
                refuse(f"{src.name}: {k} differs after serialising")
    return data, {"source_layers": src_layers, "kept_keys": sorted(arrays), "n_items": int(len(arrays["item_ids"]))}


def main() -> int:
    ensure_utf8_stdout()
    # a lane kill (TERM to the job's group) or a hangup unwinds like an exception, so the aside-rename below runs
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(128 + sig))
    if hasattr(signal, "SIGHUP"):              # absent on Windows (the workstation's tests)
        signal.signal(signal.SIGHUP, lambda sig, frame: sys.exit(128 + sig))
    ap = argparse.ArgumentParser(description="Phase 2 durable layer subsets (D-324 (1), D-325; LoRA adapters and SST-2: "
                                             "D-343, D-338 (4))")
    ap.add_argument("--repo", default=None, help="a Phase 2 model (exactly one of --repo and --adapter-name)")
    ap.add_argument("--adapter-name", default=None,
                    help="a merged adapter's identity lora/<name> (src/extract_p2.py --lora-record)")
    ap.add_argument("--source-tag", default=SRC_TAG, choices=sorted(SUB_TAGS),
                    help="the source records' tag: p2tpl (WildGuardMix and XSTest) or p2sst (SST-2, D-340 (3))")
    ap.add_argument("--set", required=True, choices=("rest", "select"))
    ap.add_argument("--layer-set", default=None, choices=sorted(LAYER_SETS),
                    help="adapters and SST-2 only, required there (no default: the job list names the D-343 (1) option)")
    ap.add_argument("--poolings", nargs="+", default=None, choices=POOLINGS,
                    help="adapters and SST-2 only, required there: the poolings carried")
    ap.add_argument("--post-probe-train", default=None, choices=("keep", "drop"),
                    help="an adapter's set holding probe_train only, required there (D-343 (1))")
    ap.add_argument("--storage-option", default=None, choices=("A", "B", "C", "D"),
                    help="adapters only, required: the D-343 (1) option the job list implements (checked, recorded)")
    ap.add_argument("--src-dir", required=True, help="NVMe directory holding the source caches")
    ap.add_argument("--out-dir", required=True, help="NVMe directory for the subset caches (staged from there)")
    ap.add_argument("--out", default=None, help="a test: the record's path, outside the repository")
    args = ap.parse_args()
    if (args.repo is None) == (args.adapter_name is None):
        refuse("give exactly one of --repo and --adapter-name")
    if args.adapter_name is not None or args.source_tag != SRC_TAG:
        return main_pair(args)
    if any(v is not None for v in (args.layer_set, args.poolings, args.post_probe_train, args.storage_option)):
        refuse("--layer-set, --poolings, --post-probe-train and --storage-option are for --adapter-name or "
               "--source-tag p2sst; a Phase 2 model's p2tpl set follows D-325's fixed rule (LAYER_RULE)")
    code0 = P.code_block(CODE_FILES)
    cc = read_result(CONFIG_CHECK)
    repo = args.repo
    if repo not in (cc.get("models") or {}):
        refuse(f"the config check names no model {repo}")
    rec_path = record_for(repo, args.set)
    rec = read_result(rec_path)
    rec_rel = _repo_rel(rec_path)
    if rec.get("repo") != repo or rec.get("set_name") != args.set or rec.get("run_kind") != "extract_p2":
        refuse(f"{rec_rel} names repo {rec.get('repo')!r}, set {rec.get('set_name')!r}, run_kind {rec.get('run_kind')!r}")
    rev = rec.get("revision")
    if not isinstance(rev, str) or len(rev) != 40 or rec.get("cache_repo") != f"{repo}__{SRC_TAG}":
        refuse(f"{rec_rel}: revision or cache_repo is not the documented-tag form")
    my_pairs = pairs_of(repo, cc)
    is_pre = any(pre == repo for pre, _ in my_pairs)
    if args.set == "select":
        if is_pre:
            refuse(f"{repo} is a pair's pre side: its select caches are ingested at the full grid (D-318 (7) item 5)")
        if repo in FULL_GRID or not my_pairs:
            refuse(f"{repo} is FULL_GRID or in no pair: ingest its full select record directly (D-325), do not re-cut it")
        if (INGEST_INDEX / f"{rec_path.stem}.sha256").exists():
            refuse(f"{rec_path.stem} is already ingested")
    plan = layer_plan(repo, rec, cc)
    out_rec = Path(args.out) if args.out else P2 / f"extract__{repo.replace('/', '__')}__{TAG}@{rev[:12]}__{args.set}.json"
    if args.out and out_rec.resolve().is_relative_to(ROOT.resolve()):
        refuse("--out (a test) must lie outside the repository")
    if out_rec.exists():
        refuse(f"{out_rec} exists; results are never overwritten (D-152 P5)")
    src_dir, out_dir = Path(args.src_dir), Path(args.out_dir)
    if out_dir.resolve().is_relative_to(ROOT.resolve()):
        refuse(f"--out-dir {out_dir} lies inside the repository (the NAS); subset caches are staged from NVMe")
    todo = []
    for key, s in sorted((rec.get("splits") or {}).items()):
        axis, part, pooling = key.split("/")
        if pooling not in POOLINGS or s.get("pooling") != pooling or s.get("n_nonfinite") != 0 or s.get("n_truncated") != 0:
            refuse(f"{rec_rel} split {key}: unexpected pooling, or non-finite or truncated items")
        name = cache_name(repo, rev, axis, part, pooling)
        if (out_dir / name).exists():
            refuse(f"{out_dir / name} exists (never overwritten)")
        todo.append((key, s, name))
    if not todo:
        refuse(f"{rec_rel} names no cache")
    print(f"{repo}@{rev[:12]} [{args.set}]: {len(todo)} caches; layers usermean {plan['per_pooling']['usermean']['layers']}, "
          f"lastgen {plan['per_pooling']['lastgen']['layers']}", flush=True)

    splits: dict[str, Any] = {}
    written_now: list[tuple[Path, str]] = []
    try:
        return finish(args, repo, rev, rec_rel, plan, todo, src_dir, out_dir, out_rec, code0, cc, splits, written_now)
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


def main_pair(args: argparse.Namespace) -> int:
    """An adapter's set (--adapter-name) or a Phase 2 model's own SST-2 rest set (--repo --source-tag p2sst): the
    module docstring's LORA ADAPTERS AND SST-2."""
    code0 = P.code_block(CODE_FILES)
    src_tag, sub_tag, axis = args.source_tag, SUB_TAGS[args.source_tag], AXIS_OF[args.source_tag]
    if args.layer_set is None or not args.poolings:
        refuse("--layer-set and --poolings are required for an adapter or an SST-2 set (no default: the job list names "
               "the D-343 (1) option it implements)")
    if len(set(args.poolings)) != len(args.poolings):
        refuse(f"--poolings {args.poolings} names a pooling twice")
    poolings = [p for p in POOLINGS if p in args.poolings]
    cc = read_result(CONFIG_CHECK)
    if args.adapter_name is not None:
        if not re.fullmatch(r"[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+|_[A-Za-z0-9]+)*", args.adapter_name):
            refuse(f"--adapter-name {args.adapter_name!r} is not plain name characters with no '__'")
        if args.storage_option is None:
            refuse("--storage-option is required for an adapter's subset: the D-343 (1) option the job list implements")
        repo, role = f"lora/{args.adapter_name}", "post"
    else:
        repo, role = args.repo, "pre"
        if repo not in (cc.get("models") or {}):
            refuse(f"the config check names no model {repo}")
        if args.set != "rest":
            refuse(f"{repo} is the pre side of its SST-2 pairs: its p2sst select set is ingested at the full grid (the "
                   "selection reads it, D-307 (1)); only its rest set is cut")
        if args.storage_option is not None or args.post_probe_train is not None:
            refuse("--storage-option and --post-probe-train are for an adapter (--adapter-name) only")
    rec_path = record_for(repo, args.set, src_tag)
    rec = read_result(rec_path)
    rec_rel = _repo_rel(rec_path)
    if rec.get("repo") != repo or rec.get("set_name") != args.set or rec.get("run_kind") != "extract_p2":
        refuse(f"{rec_rel} names repo {rec.get('repo')!r}, set {rec.get('set_name')!r}, run_kind {rec.get('run_kind')!r}")
    rev = rec.get("revision")
    if not isinstance(rev, str) or rec.get("cache_repo") != f"{repo}__{src_tag}" or rec.get("cache_revision") != rev:
        refuse(f"{rec_rel}: revision, cache_revision or cache_repo is not the documented-tag form for {src_tag}")
    ad = rec.get("adapter")
    if args.adapter_name is not None:
        pre, pre_rev = rec.get("model_repo"), rec.get("model_revision")
        if not re.fullmatch(r"[0-9a-f]{64}", rev) or not isinstance(ad, dict) or ad.get("sha256") != rev \
                or rec.get("adapter_sha256") != rev:
            refuse(f"{rec_rel}: an adapter's revision is the 64-hex sha256 of its weights, as its adapter block names")
    else:
        pre, pre_rev = repo, rev
        if not re.fullmatch(r"[0-9a-f]{40}", rev) or ad is not None or rec.get("model_repo") != repo:
            refuse(f"{rec_rel}: not the model's own extraction at a 40-hex revision")
    if pre not in (cc.get("models") or {}) or (cc["models"][pre] or {}).get("sha") != pre_rev:
        refuse(f"{rec_rel}: the pre side {pre} @ {str(pre_rev)[:12]} is not the config check's model and commit "
               "(D-303 (8))")
    keys = sorted((rec.get("splits") or {}).items())
    has_pt = any(k.split("/")[1:2] == ["probe_train"] for k, _ in keys)
    meta_option = None
    if args.adapter_name is not None:
        if has_pt and args.post_probe_train is None:
            refuse(f"{rec_rel} holds probe_train: --post-probe-train keep|drop is required (D-343 (1))")
        if not has_pt and args.post_probe_train is not None:
            refuse(f"{rec_rel} holds no probe_train split; --post-probe-train has nothing to keep or drop")
        meta_option = check_option(src_tag, args.storage_option, args.layer_set, poolings, args.post_probe_train,
                                   (rec.get("adapter_source") or {}).get("seed"))
    if args.set == "select" and (INGEST_INDEX / f"{rec_path.stem}.sha256").exists():
        refuse(f"{rec_path.stem} is already ingested")
    plan = pair_layer_plan(rec, pre, repo, role, axis, args.layer_set, poolings, cc)
    slug = repo.replace("/", "__")
    out_rec = Path(args.out) if args.out else P2 / f"extract__{slug}__{sub_tag}@{rev[:12]}__{args.set}.json"
    if args.out and out_rec.resolve().is_relative_to(ROOT.resolve()):
        refuse("--out (a test) must lie outside the repository")
    if out_rec.exists():
        refuse(f"{out_rec} exists; results are never overwritten (D-152 P5)")
    src_dir, out_dir = Path(args.src_dir), Path(args.out_dir)
    if out_dir.resolve().is_relative_to(ROOT.resolve()):
        refuse(f"--out-dir {out_dir} lies inside the repository (the NAS); subset caches are staged from NVMe")
    todo, not_kept = [], {}
    for key, s in keys:
        parts = key.split("/")
        if len(parts) != 3:
            refuse(f"{rec_rel} split key {key!r} is not '<axis>/<part>/<pooling>'")
        k_axis, part, pooling = parts
        if (src_tag == "p2sst") != (k_axis == "p2sst"):
            refuse(f"{rec_rel} split {key}: axis {k_axis} under the tag {src_tag} (D-340 (3): SST-2 has its own tag)")
        if pooling not in POOLINGS or s.get("pooling") != pooling or s.get("n_nonfinite") != 0 or s.get("n_truncated") != 0:
            refuse(f"{rec_rel} split {key}: unexpected pooling, or non-finite or truncated items")
        if pooling not in poolings:
            not_kept[key] = "its pooling is not named by --poolings"
            continue
        if part == "probe_train" and args.post_probe_train == "drop":
            not_kept[key] = "post probe_train dropped (--post-probe-train drop; D-343 (1): a post side trains no probe)"
            continue
        name = cache_name(repo, rev, k_axis, part, pooling, sub_tag)
        if (out_dir / name).exists():
            refuse(f"{out_dir / name} exists (never overwritten)")
        todo.append((key, s, name))
    if not todo:
        refuse(f"{rec_rel}: no cache is kept under these parameters")
    print(f"{repo}@{rev[:12]} [{src_tag} {args.set}]: {len(todo)} caches; "
          + "; ".join(f"{pl} layers {plan['per_pooling'][pl]['layers']}" for pl in poolings)
          + (f"; not carried {sorted(not_kept)}" if not_kept else ""), flush=True)
    params = {"layer_set": args.layer_set, "layer_set_rule": LAYER_SETS[args.layer_set], "poolings": poolings,
              "post_probe_train": args.post_probe_train, "storage_option": args.storage_option}
    config = {"repo": repo, "set": args.set, "tag": sub_tag, "source_tag": src_tag, "poolings": poolings,
              "layer_rule": PAIR_LAYER_RULE, "pair": [pre, repo], "role": role, "axis": axis, **params,
              "compression": "np.savez_compressed"}
    meta = {"tags": (src_tag, sub_tag), "layer_rule": PAIR_LAYER_RULE, "config": config, "poolings": poolings,
            "decision": ("D-343 (1), (2) (the LoRA stage; the storage option the job list names), D-325 (the cut), "
                         "D-303 (1)" if args.adapter_name is not None else
                         "D-338 (4) (SST-2: the 6-layer grid plus L* from a D-117 selection on SST-2's probe_train), "
                         "D-340 (3), D-325 (the cut), D-303 (1)"),
            "tail": {"subset_parameters": params,
                     "storage_option": ({**meta_option, "rule": "D-343 (1): checked against OPTION_RULES"}
                                        if meta_option else None),
                     "pair": {"pre": [pre, pre_rev], "post": [repo, rev], "role_of_this_set": role, "axis": axis},
                     "splits_not_kept": not_kept,
                     "adapter_source": rec.get("adapter_source"),
                     "readers_key_by": "axis and split (the cache name and the split key carry both; D-340 (3)): never "
                                       "a bare item_id across axes"}}
    splits: dict[str, Any] = {}
    written_now: list[tuple[Path, str]] = []
    try:
        return finish(args, repo, rev, rec_rel, plan, todo, src_dir, out_dir, out_rec, code0, cc, splits, written_now,
                      meta)
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


def finish(args: argparse.Namespace, repo: str, rev: str, rec_rel: str, plan: dict[str, Any], todo: list[Any],
           src_dir: Path, out_dir: Path, out_rec: Path, code0: dict[str, Any], cc: dict[str, Any],
           splits: dict[str, Any], written_now: list[tuple[Path, str]], meta: dict[str, Any] | None = None) -> int:
    for key, s, name in todo:
        layers = plan["per_pooling"][s["pooling"]]["layers"]
        data, info = cut(src_dir / s["cache_file"], s["cache_sha256"], layers)
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
    cc_rel = _repo_rel(CONFIG_CHECK)
    if meta is None:                              # D-325's path: exactly the record it has always written
        src_tag, tag, layer_rule, poolings, tail = SRC_TAG, TAG, LAYER_RULE, list(POOLINGS), {}
        decision = "D-303 (1), D-318 (7) item 3 and 5, D-324 (1), D-325"
        config = {"repo": repo, "set": args.set, "tag": TAG, "source_tag": SRC_TAG, "poolings": list(POOLINGS),
                  "layer_rule": LAYER_RULE, "excluded_pairs": {f"{a} -> {b}": w for (a, b), w in EXCLUDED_PAIRS.items()},
                  "full_grid": dict(FULL_GRID), "compression": "np.savez_compressed"}
    else:                                         # an adapter's set or a model's own SST-2 rest set (main_pair)
        (src_tag, tag), layer_rule, poolings, tail = meta["tags"], meta["layer_rule"], meta["poolings"], meta["tail"]
        decision, config = meta["decision"], meta["config"]
    digest = write_results(out_rec, {
        **provenance_block("subset_p2"),
        "decision": decision,
        "code": code0,
        "repo": repo, "revision": rev, "set_name": args.set, "ingest": "yes",
        "ingest_rule": "D-307 (1): the durable layer subset of NVMe-only caches; ingested and indexed before any "
                       "result reads it",
        "cache_repo": f"{repo}__{tag}", "cache_revision": rev, "cache_tag": tag,
        "cache_tag_documentation": f"D-297 (1) documented tag: cache_repo is repo + '__{tag}' at the same full revision; "
                                   f"a {tag} cache is a layer subset of the {src_tag} cache it names",
        "config": config, "config_hash": canonical_hash(config),
        "layer_rule": layer_rule, "layer_plan": plan,
        "excluded_pairs": {f"{a} -> {b}": why for (a, b), why in EXCLUDED_PAIRS.items()},
        "source_record": {"file": rec_rel, "sha256": _RESULT_READS[rec_rel]},
        "inputs": {rec_rel: _RESULT_READS[rec_rel], cc_rel: _RESULT_READS[cc_rel], **sel_reads},
        "n_model_layers": plan["n_model_layers"],
        "layers_resolved_per_pooling": {pl: plan["per_pooling"][pl]["layers"] for pl in poolings},
        "splits": splits,
        **tail,
    }, exclusive=True)
    print(f"Written: {_repo_rel(out_rec)}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
