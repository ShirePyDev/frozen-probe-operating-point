"""Phase 2 hidden-state extractor (D-303; docs/phase2_design.md revision 3: additions c, d, e; details 1, 3, 8, 9, 12).

One forward pass per item, two poolings, two caches per split:
  usermean  the mean over the user message's token span only; template and system tokens are excluded. The span is
            src/phase2_tokenizer_check.py's user_span (one definition for this extractor and the tokenizer check):
            (ids, start, end, n_straddle), whose ids must equal the ids computed here, or the run refuses. Positions
            outside the span are excluded element-wise (torch.where), so a non-finite value there cannot reach the mean.
  lastgen   the final position of the full templated prompt (add_generation_prompt=True), where the model would
            begin its answer.

Inputs (D-303):
  instruct  the model's own chat template, the prompt as the one user turn, add_generation_prompt=True. Token ids
            come from apply_chat_template(tokenize=True), never from re-tokenising a rendered string (extract_hidden's
            chat arm did that, which can add a second BOS). A template error REFUSES; raw text is never substituted.
            Span: user_span(tok, text, **kw), kw = clock_kwargs(template_text(tok), pin date).
  base      the raw prompt through tok(text) with the tokenizer's normal special tokens (the OLMo-2 base). Span:
            user_span(tok, text, raw=True), the prompt's own tokens without the special tokens the tokenizer adds.
            --kind base is accepted exactly for phase2_tokenizer_check.BASE_MODELS.
  special-token strings IN THE PROMPT (D-324 (5)(v), phase2_tokenizer_check.PLAIN_RULE): the span function is
            user_span_plain. An item whose own text the tokenizer would parse into an added token ('<s>', '[/INST]')
            gets that text tokenized as plain text, the template's special tokens unchanged; every other item keeps
            exactly the ids above (checked in _checked_span). Each cache's record entry lists the items that took this
            path (special_strings_as_text) and the record documents the rule and the added-token set
            (special_strings_rule).
Template identity: the loaded chat template's sha256 must equal the one results/phase2/tokenizer_static.json records for
this repo and revision. Clock (D-303 (9)): ONE recorded pin date, tokenizer_static.json's clock.pin_date; a template
that reads the clock (strftime_now) is rendered with it through the tokenizer check's clock_kwargs, and a
--template-date that differs from it refuses.
Revision (D-303 (8)): --revision must be the config check's resolved commit or the input registry's Phase 1 pin.
No truncation (D-303 (3)): an item longer than the model's max context refuses the run before any weight is loaded.
Each item's token count, span and token-id hash (the tokenizer check's ids_sha256, so the two records compare item by
item) are stored in the caches and summarised in the record.

Batching: right padding, length-sorted batches, LONGEST FIRST (an out-of-memory error comes before any cache is
written). A real token never attends to a pad, position ids are counted from the attention mask, and the last-token
index is each row's own last non-pad position; the test checks that left and right padding give the same vectors.
LongRoPE (Phi-3 128k, Phi-3.5): transformers 4.56.1 picks the long or the short factors from max(position_ids) + 1 over
the BATCH, so an item no longer than original_max_position_embeddings batched with a longer one would get other
vectors. Items from the two sides of that boundary are never batched together; dynamic (stateful) RoPE refuses.

Reused from extract_hidden.py: load_model (with the LoRA merge) and resolve_shape (imported), the hidden_states index
convention, the non-finite handling (flag per item, then nan_to_num -> 0.0, values replaced counted per layer) and the
record's fields ('splits', one entry per cache, with cache_file and cache_sha256, as extract_hidden records, so
verify_provenance.extraction_index and scripts/stage_for_ingest.sh read it). The cache is serialised in memory, checked
by loading it back, written to an exclusive temporary name, hard-linked to its final name (never overwriting), and then
READ BACK from disk (D-307 (4)): a fresh open and read of the final name's bytes must give the in-memory bytes' length
and sha256 before the record reports success (exclusive_savez). NOTHING IS DELETED (src/safe_outputs.py): on any
failure a final-named cache holding this run's bytes is renamed aside to '<name>.failed.<pid>' (no-replace), a
temporary of 1 MB or more is left, and the run exits nonzero naming every file left; removal is a human's
scripts/safe_rm.sh call, one file per call.

Storage and ingest (D-303 (1), (12); D-307 (1)): weights are read only from an HF cache snapshot on NVMe
(HF_HUB_OFFLINE=1, hf_env_guard, and the snapshot's own mount checked). A snapshot holding a .bin file refuses (no
pickle is ever loaded); a .bin-only checkpoint in src/bin_weights.py's BIN_ALLOWED is loaded as the safetensors
conversion src/fetch_weights.py made, each converted file hashed against its conversion record, which the extraction
record embeds with its sha256 (D-318 (7) item 8; design section 3). A gated meta-llama/* snapshot is the NAS keep
copy restored by scripts/p2_restore_from_nas.sh: it must hold that script's nas_restore_proof.json, which is checked
against the committed keep manifest, every file rehashed, and embedded with its sha256 (src/nas_restore_proof.py; D-324
(6), N5); without the proof it refuses. Caches are written only to NVMe scratch
(--out-dir, default GUARDRAIL_CACHE_WRITE_DIR), never to data/hidden or data/lora (locked, F-054); ingest is a separate
step (scripts/stage_for_ingest.sh, scripts/ingest_locked.sh). The required --ingest yes|never is recorded as the top-level
'ingest'. THE RULE (D-307 (1)): any cache that a result reads, including the full-grid caches the D-117 layer selection
reads, is ingested to the NAS and indexed ('yes'); only caches no result reads may stay unindexed on NVMe ('never'),
and stage_for_ingest.sh refuses a record whose ingest is 'never'. A 'never' set's caches are written uncompressed
(np.savez), a 'yes' set's compressed (np.savez_compressed); the record's per-cache 'cache_compressed' says which, and a
record without that key (before D-314) was written compressed. Records go to results/phase2/ (exclusive), so no Phase
1 reader's registry snapshot changes. Adapters stay on the NAS (444); --adapter-sha256 and --adapter-config-sha256 give
their recorded sha256, which must equal the files' before use (D-307 (2)).

Cache identity, D-297 (1) documented tags: '<org>__<model>__p2tpl@<rev12>__<axis>__<part>__{usermean,lastgen}.npz';
record 'results/phase2/extract__<org>__<model>__p2tpl@<rev12>__<set>.json' (--set-name, default 'all', so a later split
set of the same model gets its own record). The record's repo/revision name the model and its cache_repo is
'<repo>__p2tpl' at the same full revision, the form verify_provenance.documented_tag accepts. A merged LoRA adapter is
lora/<name> at the sha256 of its weights (extract_hidden), with adapter_config.json's sha256 and the base recorded.

SST-2, THE NON-SAFETY CONTROL (D-338 (4), D-340 (3)): a split of axis p2sst ('p2sst/<part>') is read from
results/phase2/dataset_manifest_p2sst.json (its files under data/frozen_p2/, each row's 'axis' 'p2sst'), and its caches
and record carry their own tag, p2sst, in place of p2tpl ('<org>__<model>__p2sst@<rev12>__p2sst__<part>__<pooling>.npz',
'results/phase2/extract__<org>__<model>__p2sst@<rev12>__<set>.json', cache_repo '<repo>__p2sst'), because
selection_p2, phase2_resplit_v2 and phase2_bos_check glob p2tpl records. One run is one tag: p2sst splits are never
mixed with p2wg splits or XSTest. SST-2 item_ids are a sha256 prefix of the text and a few equal a WildGuardMix item's
(D-340 (3)), so every reader keys a cache by its axis and split (the cache name and the record's split keys carry both),
never by a bare item_id across axes. A p2sst run refuses, before any weight is read, unless both SST-2 scans have passed
for --repo over the manifest in use (sst_scans; D-340 (3): 'before any p2sst extraction'; the record carries them as
'sst_scans'). Nothing written for p2wg or XSTest changes.

ADAPTERS FROM THEIR TRAINING RECORD (D-307 (2), D-343 (2), D-345, D-350): --lora-record results/phase2/
lora_train_p2__<corpus>__s<seed>.json --lora-step <N> takes the adapter directory (data/lora/<run dir>/step_<N>), the
identity and both recorded sha256s from that record and the adapter-ingest index manifests/ingest_lora/<stem>.sha256
(src/lora_adapter_p2.resolve; never typed by hand), and refuses --adapter, --adapter-name, --adapter-sha256 and
--adapter-config-sha256 beside it; --repo and --revision must be the record's base. A seed's step 0 is extracted once,
through the alpaca record, with every sibling record checked (lora_adapter_p2's STEP0_RULE). The record then carries
'adapter_source' (the records and indexes read, with their sha256s, and for step 0 the rule and the siblings); the
training records are also in input_files_sha256 (read_result). An adapter run's record also names the base snapshot's
blob for each weight, config and tokenizer file (model.weight_blobs, read from the snapshot's symlinks; provenance
review of fed8bd1, minor 7); a run without an adapter writes exactly the fields it always wrote.

PHASE 1 MODE (--phase1-mode; D-307 (5), the equivalence check before any Phase 2 extraction): src/extract_hidden.py's
extract() for pooling='mean' with no chat template, mirrored step for step in phase1_extract(): raw text;
tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=--max-length) with the tokenizer's default
padding side and special tokens; batches of --batch-size in FILE ORDER (no length sorting, no token budget);
extract_hidden.load_model itself (the same dtype, attention implementation and device placement); model(**enc,
output_hidden_states=True) under torch.no_grad(); (h.to(float32) * mask).sum(1) / mask.sum(1).clamp(min=1) per layer;
the same non-finite handling. It takes the first --items N items of ONE Phase 1 split (results/dataset_manifest.json)
and writes its cache and record only outside results/ (a test directory on NVMe). src/p1mode_equivalence.py compares
them with the A100 Phase 1 cache. Not mirrored (recorded in PHASE1_NOT_MIRRORED): the item count, the process history,
the cache file's bytes, the software environment.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as P  # noqa: E402
import provenance as PV  # noqa: E402
from extract_hidden import load_model, resolve_shape  # noqa: E402  (D-303: reused, not copied)
from hf_env_guard import NVME_SOURCE_PREFIX, assert_hf_env, mount_of  # noqa: E402
from provenance import (  # noqa: E402
    cache_write_dir,
    ensure_utf8_stdout,
    provenance_block,
    read_result,
    resolve_cache,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_results,
)
import safe_outputs as SO  # noqa: E402  (the no-deletion rule: rename aside, never remove)
import bin_weights as BW  # noqa: E402  (D-318 (7) item 8: a .bin-only checkpoint loads only as its checked conversion)
import nas_restore_proof as RP  # noqa: E402  (D-324 (6), N5: a NAS-restored snapshot carries its proof into the record)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "hidden"
LOCKED_DIRS = (ROOT / "data" / "hidden", ROOT / "data" / "lora")
RESULTS = ROOT / "results"
P2_RESULTS = RESULTS / "phase2"
P2_MANIFEST = P2_RESULTS / "dataset_manifest_p2.json"
P2SST_MANIFEST = P2_RESULTS / "dataset_manifest_p2sst.json"   # D-340 (3): the SST-2 axis, its own manifest and tag
P1_MANIFEST = RESULTS / "dataset_manifest.json"
CONFIG_CHECK = RESULTS / "phase2_config_check.json"
TOKENIZER_STATIC = P2_RESULTS / "tokenizer_static.json"
XSTEST_SPLIT = ("over_refusal", "eval")
XSTEST_FILE = "data/frozen/over_refusal__eval.jsonl"
P2_SPLIT_DIR = "data/frozen_p2/"
TAG = "p2tpl"
SST_AXIS = SST_TAG = "p2sst"                                   # D-340 (3): never p2tpl, which three readers glob
POOLINGS = ("usermean", "lastgen")
CODE_FILES = ["src/extract_p2.py", "src/extract_hidden.py", "src/provenance.py", "src/hf_env_guard.py", "src/paired.py",
              "src/safe_outputs.py"]
INGEST_RULE = ("D-307 (1): any cache that a result reads (including the full-grid caches the D-117 layer selection "
               "reads) is ingested to the NAS and indexed ('yes'); only caches no result reads may stay unindexed on NVMe "
               "('never'); scripts/stage_for_ingest.sh refuses a record whose ingest is 'never'")
PHASE1_NOT_MIRRORED = [
    "the item count: only the first --items items of the split, in file order (the Phase 1 run's first N / batch_size "
    "batches when N is a multiple of its batch size)",
    "the process history: the Phase 1 run may have processed other splits in the same process before this one",
    "the cache file's bytes: written by exclusive_savez (in memory, read back) for N items, not by atomic_savez for the "
    "whole split; the arrays, their names, order and dtypes are extract_hidden's",
    "the software and hardware environment: whatever is installed now; recorded in lib_versions and compared by "
    "src/p1mode_equivalence.py",
]
_SEG = r"[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+|_[A-Za-z0-9]+)*"      # a name segment with no '__', '@' or '/'
SPLIT_RE = re.compile(r"([A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)/([A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)")


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


# --- identity and names -----------------------------------------------------------------------------------------

def identity(args: Any, root: Path = ROOT) -> tuple[str, str, dict[str, Any] | None]:
    """(cache identity repo, its full revision, adapter block or None). A merged adapter is a different model from
    its base, so it never shares the base's cache names: lora/<name> at the sha256 of its weights (extract_hidden).
    adapter_config.json's sha256 is recorded too: its rank and alpha change the merged weights."""
    if not re.fullmatch(r"[0-9a-f]{40}", str(args.revision or "")):
        refuse(f"--revision {args.revision!r} is not a full 40-hex commit sha (never a branch name)")
    if not re.fullmatch(f"{_SEG}/{_SEG}", str(args.repo or "")):
        refuse(f"--repo {args.repo!r} is not '<org>/<model>' with plain name characters and no '__'")
    if not re.fullmatch(r"[A-Za-z0-9]+", str(args.set_name or "")):
        refuse(f"--set-name {args.set_name!r} is not letters and digits")
    expected = {"weights": getattr(args, "adapter_sha256", None), "config": getattr(args, "adapter_config_sha256", None)}
    if args.adapter is None:
        if args.adapter_name or any(expected.values()):
            refuse("--adapter-name, --adapter-sha256 and --adapter-config-sha256 are for --adapter only")
        return args.repo, args.revision, None
    if not args.adapter_name or not re.fullmatch(_SEG, args.adapter_name):
        refuse("--adapter requires --adapter-name, plain name characters with no '__', e.g. mistral7bi-alpaca-r16-s0")
    for k, v in expected.items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(v or "")):
            refuse(f"--adapter requires --adapter-sha256 and --adapter-config-sha256, the recorded sha256 of "
                   f"adapter_model.safetensors and adapter_config.json (D-307 (2)); {k}: {v!r}")
    adir = Path(args.adapter)
    adir = Path(os.path.realpath(adir if adir.is_absolute() else root / adir))
    lora = Path(os.path.realpath(Path(root) / "data" / "lora"))
    if lora not in adir.parents:
        refuse(f"the adapter directory {adir} does not resolve under {lora}; adapters stay on the NAS in data/lora "
               "(444) and are read from there (D-307 (2))")
    afile, cfile = adir / "adapter_model.safetensors", adir / "adapter_config.json"
    for f in (afile, cfile):
        if not f.is_file():
            refuse(f"adapter file not found: {f}")
    observed = {"weights": sha256_file(afile), "config": sha256_file(cfile)}
    bad = [k for k in expected if observed[k] != expected[k]]
    if bad:
        refuse(f"the adapter's {bad} sha256 differ from the recorded values (D-307 (2)): expected {expected}, "
               f"observed {observed}")
    sha = observed["weights"]
    return f"lora/{args.adapter_name}", sha, {"dir": str(adir), "dir_rel": PV._repo_rel(adir),
                                             "weights_file": afile.name, "sha256": sha,
                                             "config_file": cfile.name, "config_sha256": observed["config"],
                                             "expected_sha256": expected["weights"],
                                             "expected_config_sha256": expected["config"],
                                             "verified": "both files hashed before use and equal to the recorded "
                                                         "values (D-307 (2))"}


def recheck_adapter(adapter: dict[str, Any]) -> dict[str, Any]:
    """Hash both adapter files again right after load_model: the merged weights are the verified ones only if the
    files did not change between the check and the load (D-307 (2))."""
    d = Path(adapter["dir"])
    now = {"weights": sha256_file(d / adapter["weights_file"]), "config": sha256_file(d / adapter["config_file"])}
    before = {"weights": adapter["sha256"], "config": adapter["config_sha256"]}
    if now != before:
        refuse(f"the adapter's files changed while it was loaded: before {before}, after {now} (D-307 (2))")
    return {"rehashed_after_load": True, "sha256_after_load": now["weights"], "config_sha256_after_load": now["config"]}


def lora_source(args: Any, root: Path = ROOT) -> dict[str, Any] | None:
    """--lora-record / --lora-step (module docstring, ADAPTERS FROM THEIR TRAINING RECORD): fill args.adapter,
    adapter_name, adapter_sha256 and adapter_config_sha256 from the training record and the adapter-ingest index
    (src/lora_adapter_p2.resolve), and return the record's adapter_source block; None without --lora-record."""
    rec, step = getattr(args, "lora_record", None), getattr(args, "lora_step", None)
    if rec is None:
        if step is not None:
            refuse("--lora-step is for --lora-record only")
        return None
    typed = [f for f, v in (("--adapter", args.adapter), ("--adapter-name", args.adapter_name),
                            ("--adapter-sha256", getattr(args, "adapter_sha256", None)),
                            ("--adapter-config-sha256", getattr(args, "adapter_config_sha256", None))) if v is not None]
    if typed:
        refuse(f"--lora-record reads the adapter, its identity and its recorded sha256s from the training record and "
               f"the adapter-ingest index (never typed by hand, D-307 (2)); {typed} are not given with it")
    if step is None:
        refuse("--lora-record needs --lora-step (the checkpoint: 0, 25, 100 or 400)")
    import lora_adapter_p2 as LA                                 # loaded only for an adapter run (its code block)
    got = LA.resolve(rec, int(step), root)
    if (args.repo, args.revision) != (got["base_repo"], got["base_rev"]):
        refuse(f"--repo/--revision {args.repo} @ {str(args.revision)[:12]} are not the training record's base "
               f"{got['base_repo']} @ {got['base_rev'][:12]}")
    args.adapter, args.adapter_name = got["dir"], got["name"]
    args.adapter_sha256, args.adapter_config_sha256 = got["sha256"], got["config_sha256"]
    return got["source"]


SST_SCANS = ("tokenizer_prompts_p2sst.json", "bos_check_p2sst.json")     # D-340 (3): before any p2sst extraction


def sst_scans(repo: str, record_dir: Path = P2_RESULTS, p2sst_manifest: Path = P2SST_MANIFEST) -> dict[str, Any]:
    """D-340 (3): no p2sst extraction before both SST-2 scans have passed for `repo` over the manifest in use:
    tokenizer_prompts_p2sst.json (src/phase2_tokenizer_check.py prompts --axis p2sst: axis p2sst, the manifest's
    sha256, `repo` with every item's span found: n_failed 0, n_ok = n_items) and bos_check_p2sst.json
    (src/phase2_bos_check.py --axis p2sst: axis p2sst, the manifest's sha256, no tokenizer left unloaded, `repo`
    scanned). Both are read through read_result, so the extraction record lists them."""
    man = Path(p2sst_manifest)
    if not man.is_file():
        refuse(f"{man.name} is not on disk; the SST-2 scans cannot be tied to it")
    man_sha = sha256_file(man)
    out: dict[str, Any] = {"rule": "D-340 (3): both SST-2 scans passed for this model over this manifest before any "
                                   "p2sst extraction"}
    for name in SST_SCANS:
        p = Path(record_dir) / name
        if not p.is_file():
            refuse(f"{name} is not on disk: the SST-2 prompts check and special-token scan run before any p2sst "
                   "extraction (D-340 (3); the scans list src/lora_jobs_p2.py writes)")
        r = read_result(p)
        if not isinstance(r, dict) or r.get("axis") != SST_AXIS or \
                ((r.get("inputs") or {}).get("dataset_manifest_p2sst") or {}).get("sha256") != man_sha:
            refuse(f"{name} is not a scan of axis {SST_AXIS} over {man.name} as it is now (sha256 {man_sha[:16]})")
        if name == SST_SCANS[0]:
            m = (r.get("models") or {}).get(repo) or {}
            ok = m.get("n_failed") == 0 and isinstance(m.get("n_items"), int) and m.get("n_ok") == m["n_items"] > 0
        else:
            ok = r.get("scan_not_loaded") == [] and repo in ((r.get("special_token_scan") or {}).get("models") or {})
        if not ok:
            refuse(f"{name} records no pass for {repo} (prompts: every item's span found; BOS scan: every tokenizer "
                   "loaded and this model scanned); no p2sst extraction (D-340 (3))")
        rel = PV._repo_rel(p)
        out[name] = {"file": rel, "sha256": PV._RESULT_READS[rel], "pass": True}
    out["dataset_manifest_p2sst_sha256"] = man_sha
    return out


def weight_blobs(snap_dir: Path) -> dict[str, Any]:
    """The base snapshot's blob per weight, config and tokenizer file, from its symlinks (an adapter run only): in the
    Hugging Face cache each file of snapshots/<revision>/ is a symlink into blobs/, the blob named by the file's
    sha256 for an LFS file (the weights). Nothing is hashed here; a file that is not a symlink (a local test
    directory) is recorded as None."""
    blobs = {}
    for p in sorted(Path(snap_dir).iterdir()):
        if p.suffix in (".safetensors", ".bin") or p.name.endswith(".index.json") or p.name in AUX_SNAPSHOT_FILES or \
                (p.name.startswith("tokenizer") and p.suffix == ".json"):
            blobs[p.name] = Path(os.readlink(p)).name if p.is_symlink() else None
    return {"rule": "the snapshot's symlinks into the Hugging Face cache's blobs/ (an LFS file's blob is its sha256); "
                    "provenance review of fed8bd1, minor 7", "blobs": blobs}


def split_tag(names: list[str]) -> str:
    """The cache tag of a run: p2sst when every requested split is of axis p2sst, p2tpl when none is (D-340 (3)); a
    run mixing them refuses (one record is one tag)."""
    axes = [str(n).split("/", 1)[0] for n in names]
    n_sst = sum(a == SST_AXIS for a in axes)
    if 0 < n_sst < len(axes):
        refuse(f"--splits {list(names)} mixes axis {SST_AXIS} with other axes; SST-2 has its own tag and record "
               "(D-340 (3)): extract it in a run of its own")
    return SST_TAG if n_sst else TAG


def cache_name(ident_repo: str, ident_rev: str, axis: str, part: str, pooling: str, tag: str = TAG) -> str:
    return f"{ident_repo.replace('/', '__')}__{tag}@{ident_rev[:12]}__{axis}__{part}__{pooling}.npz"


def record_name(ident_repo: str, ident_rev: str, set_name: str, tag: str = TAG) -> str:
    return f"extract__{ident_repo.replace('/', '__')}__{tag}@{ident_rev[:12]}__{set_name}.json"


def _on_nvme(path: Path) -> tuple[bool, str]:
    mnt, src = mount_of(Path(path))
    return src.startswith(NVME_SOURCE_PREFIX), f"{src or 'an unknown device'} (mounted at {mnt or '?'})"


def check_out_dir(out_dir: Path, locked: tuple[Path, ...] = LOCKED_DIRS, require_nvme: bool = True) -> Path:
    """The cache directory: never inside a locked directory (data/hidden, data/lora), and on NVMe scratch."""
    d = Path(os.path.realpath(out_dir))
    for lk in locked:
        lr = Path(os.path.realpath(lk))
        if d == lr or lr in d.parents:
            refuse(f"the cache directory {d} is inside the locked {lk}; the full grid stays on NVMe scratch (D-303 (1))")
    if require_nvme:
        ok, where = _on_nvme(d)
        if not ok:
            refuse(f"the cache directory {d} is on {where}, not NVMe")
    d.mkdir(parents=True, exist_ok=True)
    return d


def snapshot(repo: str, revision: str, load_from: str | None, require_nvme: bool = True,
             root: Path = ROOT) -> tuple[Path, dict[str, Any]]:
    """The local snapshot the weights, config and tokenizer are loaded from, and its weight files (names, sizes).
    From the HF cache it must be snapshots/<revision>, and it and every weight file (a symlink into blobs/) must
    resolve on NVMe. No weight file, or any failure to list them, refuses (design section 3 records the path taken).
    A gated meta-llama/* snapshot (restored from the NAS), or any snapshot holding a restore proof, must pass
    nas_restore_proof.check_proof against the keep manifest under `root` (D-324 (6), N5)."""
    try:
        if load_from:
            d = Path(load_from)
        else:
            from huggingface_hub import snapshot_download
            d = Path(snapshot_download(repo, revision=revision, local_files_only=True))
        files = sorted(p for p in d.iterdir() if p.suffix in (".safetensors", ".bin") or p.name.endswith(".index.json"))
    except Exception as e:  # noqa: BLE001 - any failure to find the local snapshot refuses
        refuse(f"no local snapshot of {repo} @ {revision[:12]} could be listed ({type(e).__name__}: {e})")
    if not load_from and d.name != revision:
        refuse(f"the snapshot directory is {d.name}, not the revision {revision}")
    weights = [p for p in files if p.suffix in (".safetensors", ".bin")]
    if not weights:
        refuse(f"the snapshot of {repo} @ {revision[:12]} holds no weight file")
    if any(p.suffix == ".bin" for p in weights):
        refuse(f"the snapshot of {repo} @ {revision[:12]} holds .bin files "
               f"{[p.name for p in weights if p.suffix == '.bin']}; a pickle is never loaded here (D-318 (7) "
               "item 8): a BIN_ALLOWED checkpoint is converted to safetensors by src/fetch_weights.py "
               "(src/bin_weights.py) and only the checked conversion is loaded")
    allowed_bin = BW.bin_allowed(repo, revision)
    side_files = [d / BW.CONVERSION_RECORD] if allowed_bin and (d / BW.CONVERSION_RECORD).is_file() else []
    from_nas = RP.requires_proof(repo) or os.path.lexists(d / RP.PROOF)
    side_files += [d / RP.PROOF] if from_nas and (d / RP.PROOF).is_file() else []
    aux = sorted(p for p in d.iterdir() if p.is_file() and (p.name in AUX_SNAPSHOT_FILES or (
        p.name.startswith("tokenizer") and p.suffix == ".json")))
    if require_nvme:
        for p in [d, *files, *aux, *side_files]:
            ok, where = _on_nvme(p)
            if not ok:
                refuse(f"{p.name} of the snapshot resolves on {where}, not NVMe; weights are read only from NVMe")
    conv = BW.check_converted(d, repo, revision) if allowed_bin else None
    restore = RP.check_proof(d, repo, revision, root) if from_nas else None
    return d, {"snapshot": d.name if not load_from else "(test: a local directory)",
               "files": {p.name: p.stat().st_size for p in files},
               "config_and_tokenizer_files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                                              for p in aux},
               "has_safetensors": any(p.suffix == ".safetensors" for p in weights),
               "has_bin": any(p.suffix == ".bin" for p in weights), "nvme_checked": require_nvme,
               "load_path": ("safetensors converted on NVMe from the revision's .bin files (bin_conversion; D-318 (7) "
                             "item 8)" if conv else "the NAS keep copy restored to NVMe and verified (nas_restore; "
                             "D-307 (3))" if restore else "safetensors as published at the revision"),
               "bin_conversion": conv, "nas_restore": restore}


# --- splits -----------------------------------------------------------------------------------------------------

def parse_jsonl_bytes(data: bytes) -> list[Any]:
    """provenance.read_jsonl's rule (split on '\\n' only, never splitlines, F-017), applied to the bytes hashed."""
    import json
    return [json.loads(line) for line in data.decode("utf-8").split("\n") if line.strip()]  # not a result (D-297): split rows


def _manifest_entry(man: Any, axis: str, part: str, label: str) -> dict[str, Any]:
    try:
        e = man["axes"][axis]["partitions"][part]
    except (KeyError, TypeError):
        refuse(f"{label} names no split {axis}/{part}")
    if not isinstance(e, dict) or not isinstance(e.get("file"), str) or \
            not re.fullmatch(r"[0-9a-f]{64}", str(e.get("sha256", ""))):
        refuse(f"{label}: split {axis}/{part} has no file and sha256")
    f = e["file"]
    if Path(f).is_absolute() or ".." in Path(f).parts:
        refuse(f"{label}: split {axis}/{part} names a file outside the repo: {f}")
    return e


def check_rows(rows: list[Any], name: str) -> None:
    seen: set[str] = set()
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or not isinstance(r.get("item_id"), str) or not r["item_id"]:
            refuse(f"{name} row {i}: no item_id")
        lab = r.get("label")
        if isinstance(lab, bool) or lab not in (0, 1):
            refuse(f"{name} row {i} ({r['item_id']}): label {lab!r} is not 0 or 1")
        if not isinstance(r.get("text"), str) or not r["text"].strip():
            refuse(f"{name} row {i} ({r['item_id']}): no prompt text")
        if r["item_id"] in seen:
            refuse(f"{name}: item_id {r['item_id']} appears twice")
        seen.add(r["item_id"])


def resolve_splits(names: list[str], *, root: Path = ROOT, p2_manifest: Path = P2_MANIFEST,
                   p1_manifest: Path = P1_MANIFEST, p2sst_manifest: Path = P2SST_MANIFEST) -> list[dict[str, Any]]:
    """Each requested '<axis>/<part>' with its rows, read from bytes whose sha256 equals the manifest's. XSTest
    (over_refusal/eval, data/frozen/over_refusal__eval.jsonl) is verified against results/dataset_manifest.json (Phase
    1's freeze); a p2sst split against results/phase2/dataset_manifest_p2sst.json, each of its rows naming axis p2sst
    (D-340 (3)); every other split against results/phase2/dataset_manifest_p2.json. Every Phase 2 split must lie under
    data/frozen_p2/. The manifests are read through provenance.read_result (D-297 (1))."""
    mans: dict[Path, Any] = {}

    def man(path: Path) -> Any:
        path = Path(path)
        if path not in mans:
            if not path.is_file():
                refuse(f"{path} is not on disk; the splits cannot be verified")
            mans[path] = read_result(path)
        return mans[path]

    out: list[dict[str, Any]] = []
    for name in names:
        m = SPLIT_RE.fullmatch(name)
        if not m:
            refuse(f"split {name!r} is not '<axis>/<part>' (letters, digits, single underscores)")
        if name in [s["name"] for s in out]:
            refuse(f"split {name} requested twice")
        axis, part = m.groups()
        xs = (axis, part) == XSTEST_SPLIT
        mpath = Path(p1_manifest) if xs else Path(p2sst_manifest) if axis == SST_AXIS else Path(p2_manifest)
        e = _manifest_entry(man(mpath), axis, part, mpath.name)
        if xs and e["file"] != XSTEST_FILE:
            refuse(f"{mpath.name} names {e['file']} for XSTest, not {XSTEST_FILE}")
        if not xs and not e["file"].startswith(P2_SPLIT_DIR):
            refuse(f"{mpath.name} names {e['file']} for {name}; every Phase 2 split lies under {P2_SPLIT_DIR}")
        data = (Path(root) / e["file"]).read_bytes()
        got = sha256_bytes(data)
        if got != e["sha256"]:
            refuse(f"{e['file']} has sha256 {got[:12]}..., not {mpath.name}'s {e['sha256'][:12]}...; a split is "
                   "used only as frozen")
        rows = parse_jsonl_bytes(data)
        check_rows(rows, name)
        if axis == SST_AXIS:
            off = [r["item_id"] for r in rows if r.get("axis") != SST_AXIS]
            if off:
                refuse(f"{e['file']}: {len(off)} row(s) do not name axis {SST_AXIS} (e.g. {off[:3]}); an SST-2 item is "
                       "keyed by its axis and split, never by a bare item_id (D-340 (3))")
        for k, want in (("n", len(rows)), ("n_pos", sum(r["label"] == 1 for r in rows)),
                        ("n_neg", sum(r["label"] == 0 for r in rows))):
            if k in e and e[k] != want:
                refuse(f"{e['file']}: {mpath.name} says {k} = {e[k]}, the file holds {want}")
        out.append({"name": name, "axis": axis, "part": part, "file": e["file"], "sha256": got,
                    "manifest": PV._repo_rel(mpath), "rows": rows})
    if XSTEST_SPLIT in [(s["axis"], s["part"]) for s in out] and Path(p2_manifest) in mans:
        try:                                                     # the Phase 2 manifest, if it lists XSTest, agrees
            e2 = mans[Path(p2_manifest)]["axes"]["over_refusal"]["partitions"]["eval"]
        except (KeyError, TypeError):
            e2 = None
        x = next(s for s in out if (s["axis"], s["part"]) == XSTEST_SPLIT)
        if isinstance(e2, dict) and (e2.get("file"), e2.get("sha256")) != (x["file"], x["sha256"]):
            refuse("the Phase 2 manifest's XSTest entry differs from Phase 1's")
    return out


# --- the config check, the revision, the tokenizer check ---------------------------------------------------------

def _ns(d: Any) -> Any:
    return SimpleNamespace(**{k: _ns(v) for k, v in d.items()}) if isinstance(d, dict) else d


def check_revision(cc: Any, api: Any, repo: str, revision: str) -> dict[str, Any]:
    """D-303 (8): --revision is the input registry's Phase 1 pin for a model Phase 1 extracted
    (phase2_tokenizer_check.PHASE1_MODELS) and the commit the config check resolved for every other model: the
    revision phase2_tokenizer_check.registry_revisions uses, so the extractor and the tokenizer check name one
    revision per model (it also refuses when the extraction records name another)."""
    models = (cc.get("models") if isinstance(cc, dict) else None) or {}
    m = models.get(repo)
    if not isinstance(m, dict):
        refuse(f"the config check names no model {repo}")
    used, block = api.registry_revisions([repo], models)
    source = "phase1_pin" if repo in api.PHASE1_MODELS else "config_check"
    if used.get(repo) != revision:
        refuse(f"--revision {revision[:12]} is not {repo}'s revision under D-303 (8): "
               f"{'the input registry Phase 1 pin' if source == 'phase1_pin' else 'the config check commit'} "
               f"{str(used.get(repo))[:12]} (config check {str(m.get('sha'))[:12]})")
    return {"revision_source": [source], "config_check_sha": m.get("sha"),
            "phase1_pin": used[repo] if source == "phase1_pin" else None,
            "equals_config_check_sha": revision == m.get("sha"), "registry": block}


def check_config(cc: Any, model_repo: str, n_layers: int, hidden: int) -> dict[str, Any]:
    """The model's layer count and hidden size equal the config check's, and at least one pair holding the model is
    comparable and not dropped there (D-303 (g): a pair failing on either is dropped)."""
    m = ((cc.get("models") if isinstance(cc, dict) else None) or {}).get(model_repo)
    if not isinstance(m, dict) or not isinstance(m.get("config"), dict):
        refuse(f"the config check has no config.json for {model_repo}")
    try:
        cn, ch = resolve_shape(_ns(m["config"]))
    except (ValueError, AttributeError, TypeError) as e:
        refuse(f"the config check's config for {model_repo} names no shape: {e}")
    if (cn, ch) != (n_layers, hidden):
        refuse(f"{model_repo} loads with {n_layers} layers and hidden size {hidden}; the config check has {cn} and {ch}")
    dropped = set(cc.get("dropped_pairs") or [])

    def kept(p: dict) -> bool:
        return (p.get("comparable") is True and not str(p.get("verdict", "")).startswith("DROPPED")
                and f"{p.get('pre')} -> {p.get('post')}" not in dropped)

    pairs = [p for p in (cc.get("pairs") or []) if isinstance(p, dict) and model_repo in (p.get("pre"), p.get("post"))]
    if not any(kept(p) for p in pairs):
        refuse(f"no config-check pair holding {model_repo} is kept (pairs: "
               f"{[(p.get('pre'), p.get('post'), p.get('verdict') or p.get('why')) for p in pairs]})")
    return {"num_hidden_layers": cn, "hidden_size": ch, "revision_resolved_by_check": m.get("sha"),
            "config_sha256": m.get("config_sha256"),
            "pairs": [{"pair": f"{p.get('pre')} -> {p.get('post')}", "kept": kept(p),
                       "verdict": p.get("verdict") or p.get("why")} for p in pairs]}


def check_tokenizer_static(st: Any, repo: str, revision: str, kind: str, template_sha: str | None) -> dict[str, Any]:
    """The tokenizer check recorded this repo at this revision, as this kind, with this chat template (sha256 of the
    template string apply_chat_template uses, phase2_tokenizer_check.template_text)."""
    m = ((st.get("models") if isinstance(st, dict) else None) or {}).get(repo)
    if not isinstance(m, dict):
        refuse(f"tokenizer_static.json names no model {repo}")
    if m.get("revision") != revision:
        refuse(f"tokenizer_static.json records {repo} at {str(m.get('revision'))[:12]}, not {revision[:12]}")
    if m.get("kind") != kind:
        refuse(f"tokenizer_static.json records {repo} as {m.get('kind')!r}, not {kind!r}")
    ct = (m.get("tokenizer") or {}).get("chat_template")
    if not isinstance(ct, dict) or "sha256" not in ct:
        refuse(f"tokenizer_static.json records no chat template sha256 for {repo}")
    if ct["sha256"] != template_sha:
        refuse(f"the loaded chat template's sha256 {str(template_sha)[:12]} is not the one tokenizer_static.json "
               f"records for {repo} @ {revision[:12]} ({str(ct['sha256'])[:12]})")
    return {"revision": m.get("revision"), "kind": m.get("kind"), "chat_template_sha256": ct["sha256"],
            "template_equal": True}


def max_context(config: Any) -> int:
    v = getattr(config, "max_position_embeddings", None)
    if v is None and getattr(config, "text_config", None) is not None:
        v = getattr(config.text_config, "max_position_embeddings", None)
    if not isinstance(v, int) or v <= 0:
        refuse("the model config names no max_position_embeddings, so the no-truncation rule cannot be applied")
    return v


def rope_boundary(config: Any) -> tuple[int | None, dict[str, Any]]:
    """The token count above which transformers 4.56.1 switches a LongRoPE model to its long factors (its
    longrope_frequency_update: max(position_ids) + 1 > original_max_position_embeddings, else
    max_position_embeddings), or None for a RoPE that does not depend on the batch. Dynamic RoPE caches a growing
    frequency table across calls (order-dependent), so it refuses."""
    rs = getattr(config, "rope_scaling", None)
    rtype = (rs.get("rope_type") or rs.get("type")) if isinstance(rs, dict) else None
    if rtype and "dynamic" in str(rtype):
        refuse(f"rope type {rtype!r} updates its frequencies across calls; vectors would depend on batch and order")
    if rtype not in ("longrope", "su"):
        return None, {"rope_type": rtype, "batch_boundary": None}
    if hasattr(config, "original_max_position_embeddings"):
        b = config.original_max_position_embeddings
    else:
        b = config.max_position_embeddings
    if not isinstance(b, int) or b <= 0:
        refuse(f"LongRoPE with no usable original_max_position_embeddings ({b!r})")
    return b, {"rope_type": rtype, "batch_boundary": b,
               "rule": "items with n_tokens <= the boundary (short factors) and > it (long factors) are never batched "
                       "together"}


def resolve_layers(requested: list[int], n_layers: int) -> list[int]:
    """extract_hidden's convention: hidden_states[i], 0 the embedding output, a negative index from the end
    (-1 -> n_layers)."""
    out = [i if i >= 0 else n_layers + 1 + i for i in requested]
    bad = [i for i in out if not 0 <= i <= n_layers]
    if bad:
        refuse(f"layer index {bad} out of range for a {n_layers}-layer model")
    if len(set(out)) != len(out):
        refuse(f"--layers {requested} names a layer twice")
    return sorted(out)


def sliding_window_info(config: Any, splits: list[dict[str, Any]]) -> dict[str, Any]:
    sw = getattr(config, "sliding_window", None)
    out = {"sliding_window": sw, "use_sliding_window": getattr(config, "use_sliding_window", None),
           "n_items_longer": None, "by_split": None}
    if isinstance(sw, int) and not isinstance(sw, bool) and sw > 0:
        out["by_split"] = {sp["name"]: int((sp["tok"]["n_tokens"] > sw).sum()) for sp in splits}
        out["n_items_longer"] = sum(out["by_split"].values())
    return out


# --- tokenization -------------------------------------------------------------------------------------------------

SPAN_API = ("user_span", "user_span_plain", "clock_kwargs", "template_text", "ids_sha256", "registry_revisions")
SPAN_TUPLES = ("BASE_MODELS", "PHASE1_MODELS")


def span_api() -> Any:
    """src/phase2_tokenizer_check.py, the one definition of the user span, the clock pin, the id hash, the base
    checkpoints and the Phase 1 pins shared with the tokenizer check: user_span(tok, text, raw=False,
    **template_kwargs) -> (ids, start, end, n_straddle), clock_kwargs(template, pin_date), template_text(tok),
    ids_sha256(ids), registry_revisions(repos, models), BASE_MODELS, PHASE1_MODELS."""
    try:
        import phase2_tokenizer_check as TC
    except ImportError as e:
        raise ImportError(
            "src/phase2_tokenizer_check.py (user_span(tok, text) -> (ids, start, end, n_straddle), clock_kwargs, "
            "template_text, ids_sha256, registry_revisions, BASE_MODELS, PHASE1_MODELS) is required: D-303 (d) means "
            "over the user message's token span only, and this extractor never computes the span itself. It could not "
            f"be imported ({type(e).__name__}: {e}).") from e
    missing = [n for n in SPAN_API if not callable(getattr(TC, n, None))]
    missing += [n for n in SPAN_TUPLES if not isinstance(getattr(TC, n, None), tuple)]
    missing += [] if isinstance(getattr(TC, "PLAIN_RULE", None), str) else ["PLAIN_RULE"]
    if missing:
        raise ImportError(f"src/phase2_tokenizer_check.py does not define {missing}, which this extractor uses")
    return TC


def _as_ids(out: Any) -> list[int]:
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    if len(out) and isinstance(out[0], (list, tuple)):
        if len(out) != 1:
            refuse(f"expected one token sequence, got {len(out)}")
        out = out[0]
    return [int(i) for i in out]


CLOCK_PROBE = "CLOCKPROBE7f3a"


def template_setup(tok: Any, kind: str, pin_date: str | None, api: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """(keyword arguments for apply_chat_template and user_span, the record's template block). D-303 (9): a template
    that reads the clock (strftime_now) is rendered with the pin date tokenizer_static.json records (YYYY-MM-DD) through
    the tokenizer check's clock_kwargs, and a probe override of strftime_now must reach the render, or the date is not
    under the pin's control and the run refuses."""
    text = api.template_text(tok)
    block: dict[str, Any] = {"tokenizer_has_chat_template": text is not None,
                             "chat_template_sha256": sha256_bytes(text.encode("utf-8")) if text else None,
                             "recorded_pin_date": pin_date}
    if kind == "base":
        block.update({"template_used": False, "reads_clock": None, "date_pin": None, "template_kwargs": []})
        return {}, block
    if text is None:
        refuse("--kind instruct but the tokenizer defines no chat template (D-303: a template error refuses; raw "
               "text is never substituted)")
    reads_clock = "strftime_now" in text
    if reads_clock and not pin_date:
        refuse("the chat template reads the clock (strftime_now) and tokenizer_static.json records no clock.pin_date "
               "(D-303 (9): one recorded pin date)")
    try:
        kw = api.clock_kwargs(text, pin_date if reads_clock else None)
    except ValueError as e:
        refuse(f"the recorded pin date {pin_date!r} is not a YYYY-MM-DD date ({e})")
    if reads_clock:
        try:
            probe = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                            add_generation_prompt=True, strftime_now=lambda fmt: CLOCK_PROBE)
        except Exception as e:  # noqa: BLE001
            refuse(f"the chat template failed under the clock probe ({type(e).__name__}: {e})")
        if CLOCK_PROBE not in probe:
            refuse("the chat template reads the clock but an override of strftime_now does not reach its render, so "
                   "the date cannot be pinned (D-303 (9))")
    block.update({"template_used": True, "reads_clock": reads_clock, "date_pin": pin_date if reads_clock else None,
                  "template_kwargs": sorted(kw),
                  "pin_mechanism": ("phase2_tokenizer_check.clock_kwargs: strftime_now formats the pinned date at "
                                    "00:00:00; date_string is never passed" if reads_clock else None)})
    return kw, block


def _checked_span(got: Any, ids: list[int], what: str, tok: Any = None,
                  info: dict[str, Any] | None = None) -> tuple[list[int], int, int, int]:
    """The span function's (ids, start, end, n_straddle[, info]) checked against `ids`, today's ids computed here.
    Today's path (no info, or info['plain_text_path'] False): the ids must equal `ids`. The plain-text path (D-324
    (5)(v), phase2_tokenizer_check.PLAIN_RULE): today's ids must be `ids` and hold an added token parsed from the user
    text inside the replaced range; outside that range the ids returned are today's, inside it they hold no added token,
    and the span lies inside it. `info`, when given, receives the path taken and the strings concerned."""
    if not (isinstance(got, (tuple, list)) and len(got) in (4, 5)):
        refuse("user_span did not return (ids, start, end, n_straddle[, info])")
    s_ids, start, end, n_straddle = _as_ids(got[0]), int(got[1]), int(got[2]), int(got[3])
    plain = got[4] if len(got) == 5 else {"plain_text_path": False}
    if not isinstance(plain, dict):
        refuse("the span function's info is not a dict")
    on_plain = plain.get("plain_text_path") is True
    if not on_plain:
        if s_ids != ids:
            refuse(f"user_span's ids differ from {what}'s")
    else:
        added = {int(i) for i in (getattr(tok, "added_tokens_decoder", None) or {})}
        unk = getattr(tok, "unk_token_id", None)                 # the model's own output for text it cannot encode,
        if unk in {getattr(tok, f"{n}_token_id", None) for n in ("bos", "eos", "pad")}:
            unk = None                                           # unless it is also BOS/EOS/pad (OLMo-2: <|endoftext|>)
        try:
            lo, hi = (int(x) for x in plain["replaced"])
            pos = [int(k) for k in plain["today_positions"]]
        except (KeyError, TypeError, ValueError):
            refuse("the plain-text path's info names no replaced range and today's added-token positions")
        m = len(s_ids) - (len(ids) - hi)
        if (plain.get("today_ids") != ids or not pos or not all(lo <= k < hi and ids[k] in added for k in pos)
                or not 0 <= lo <= m <= len(s_ids) or s_ids[:lo] != ids[:lo] or s_ids[m:] != ids[hi:]
                or any(i in added and i != unk for i in s_ids[lo:m]) or not lo <= start < end <= m):
            refuse(f"the plain-text ids (D-324 (5)(v)) are not {what}'s with only the user text re-tokenized and no "
                   "added token left in it")
    if not 0 <= start < end <= len(s_ids) or n_straddle < 0:
        refuse(f"user_span returned span [{start}, {end}) with n_straddle {n_straddle} for {len(s_ids)} tokens")
    if info is not None:
        info.update({"plain_text_path": on_plain, "strings": [str(s) for s in plain.get("strings") or []]})
    return s_ids, start, end, n_straddle


def instruct_item(tok: Any, text: str, kw: dict[str, Any], span_fn: Callable,
                  info: dict[str, Any] | None = None) -> tuple[list[int], int, int, int]:
    """The ids from apply_chat_template(tokenize=True), and the span function's span over the same ids; with
    phase2_tokenizer_check.user_span_plain, special-token strings in the user text are re-tokenized as plain text
    (D-324 (5)(v)) and every other item keeps these ids exactly (_checked_span)."""
    try:
        ids = _as_ids(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                              add_generation_prompt=True, **kw))
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - any template failure refuses
        refuse(f"the chat template failed ({type(e).__name__}: {e}); D-303: a template error refuses, with no "
               "fallback to raw text")
    try:
        got = span_fn(tok, text, **kw)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - SpanError and anything else refuse
        refuse(f"user_span failed ({type(e).__name__}: {e})")
    return _checked_span(got, ids, "apply_chat_template(tokenize=True)", tok, info)


def base_item(tok: Any, text: str, span_fn: Callable,
              info: dict[str, Any] | None = None) -> tuple[list[int], int, int, int]:
    """The raw prompt, tok(text) with the tokenizer's normal special tokens, and user_span(raw=True)'s span (the same
    plain-text rule as instruct_item)."""
    ids = _as_ids(tok(text, add_special_tokens=True)["input_ids"])
    try:
        got = span_fn(tok, text, raw=True)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        refuse(f"user_span(raw=True) failed ({type(e).__name__}: {e})")
    return _checked_span(got, ids, "tok(text)", tok, info)


def tokenize_split(tok: Any, sp: dict[str, Any], kind: str, kw: dict[str, Any], span_fn: Callable,
                   ids_hash: Callable) -> dict:
    ids_l: list[list[int]] = []
    st, en, strad = [], [], []
    plain: list[dict[str, Any]] = []                             # D-324 (5)(v): the items on the plain-text path
    for r in sp["rows"]:
        info: dict[str, Any] = {}
        try:
            ids, s, e, n = (instruct_item(tok, r["text"], kw, span_fn, info) if kind == "instruct"
                            else base_item(tok, r["text"], span_fn, info))
        except SystemExit as ex:
            raise SystemExit(f"{ex} [split {sp['name']}, item {r['item_id']}]") from None
        if info.get("plain_text_path"):
            plain.append({"item_id": r["item_id"], "strings": info["strings"]})
        ids_l.append(ids)
        st.append(s)
        en.append(e)
        strad.append(n)
    return {"ids": ids_l, "start": np.asarray(st, np.int32), "end": np.asarray(en, np.int32),
            "n_straddle": np.asarray(strad, np.int32), "n_tokens": np.asarray([len(i) for i in ids_l], np.int32),
            "ids_sha256": np.asarray([ids_hash(i) for i in ids_l]), "plain": plain}


def special_strings_block(tok: Any, api: Any, span_fn: Callable, splits: list[dict[str, Any]]) -> dict[str, Any]:
    """D-324 (5)(v): the rule, the set of strings it concerns (the tokenizer's added tokens, documented by count, the
    non-special ones by content, and a sha256 of the whole list), and how many items took the plain-text path."""
    import json
    dec = getattr(tok, "added_tokens_decoder", None) or {}
    rows = sorted([int(i), str(t.content), bool(getattr(t, "special", False))] for i, t in dec.items())
    return {"rule": api.PLAIN_RULE if span_fn is getattr(api, "user_span_plain", None) else
            f"not applied: the span function is {getattr(span_fn, '__name__', '?')} (a test)",
            "added_tokens": {"n": len(rows), "n_special": sum(r[2] for r in rows),
                             "not_special": [r[1] for r in rows if not r[2]],
                             "sha256": sha256_bytes(json.dumps(rows).encode("utf-8")),
                             "sha256_of": "json.dumps of the sorted [id, content, special] list"},
            "n_items_plain_text_path": sum(len(sp["tok"]["plain"]) for sp in splits),
            "by_split": {sp["name"]: len(sp["tok"]["plain"]) for sp in splits}}


def check_lengths(splits: list[dict[str, Any]], max_ctx: int) -> None:
    over = [(sp["name"], sp["rows"][i]["item_id"], int(n)) for sp in splits
            for i, n in enumerate(sp["tok"]["n_tokens"]) if n > max_ctx]
    if over:
        refuse(f"{len(over)} item(s) are longer than the model's max context of {max_ctx} tokens (D-303 (3): no "
               f"truncation): {over[:10]}")


def template_tokens(tok: Any, splits: list[dict[str, Any]]) -> dict[str, Any]:
    """D-303 (e): the tokens outside the user span, decoded for the first item (any default system text a template
    injects shows here), and how many distinct such sequences occur."""
    before, after, first = set(), set(), None
    for sp in splits:
        t = sp["tok"]
        for ids, s, e in zip(t["ids"], t["start"], t["end"]):
            b, a = tuple(ids[:s]), tuple(ids[e:])
            before.add(b)
            after.add(a)
            first = first or (b, a)
    return {"first_item_before_user": tok.decode(list(first[0])), "first_item_after_user": tok.decode(list(first[1])),
            "first_item_n_tokens_before_user": len(first[0]), "first_item_n_tokens_after_user": len(first[1]),
            "n_distinct_token_sequences_before_user": len(before), "n_distinct_token_sequences_after_user": len(after)}


# --- forward pass and pooling -----------------------------------------------------------------------------------

def length_batches(lengths: Any, batch_size: int, max_batch_tokens: int,
                   boundary: int | None = None) -> list[list[int]]:
    """Item indices in batches of similar length, LONGEST FIRST, at most batch_size items and max_batch_tokens padded
    tokens each (an item longer than the budget is a batch of its own). With a LongRoPE boundary, items on its two
    sides (n_tokens <= boundary, > boundary) never share a batch."""
    def side(i: int) -> bool:
        return boundary is not None and int(lengths[i]) > boundary

    order = sorted(range(len(lengths)), key=lambda i: (-int(lengths[i]), i))
    batches: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        if cur and (len(cur) >= batch_size or side(i) != side(cur[0])
                    or (len(cur) + 1) * int(lengths[cur[0]]) > max_batch_tokens):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def pool_batch(model: Any, seqs: list[list[int]], spans: list[tuple[int, int]], layers: list[int], device: str,
               pad_id: int, padding_side: str = "right") -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], int]:
    """One forward pass over a padded batch: (user-span means, last-token states, number of hidden_states), float32,
    as they left the model (non-finite values kept, for the caller to flag). Positions outside the span are excluded
    element-wise (torch.where), so a non-finite value there never reaches the mean (inf * 0 would be NaN). The
    extractor always pads right; padding_side='left' exists for the test that the two give the same vectors."""
    import torch

    if padding_side not in ("right", "left"):
        raise ValueError(padding_side)
    b_n, t_n = len(seqs), max(len(s) for s in seqs)
    ids = torch.full((b_n, t_n), int(pad_id), dtype=torch.long)
    mask = torch.zeros((b_n, t_n), dtype=torch.long)
    span = torch.zeros((b_n, t_n), dtype=torch.bool)
    last = torch.zeros(b_n, dtype=torch.long)
    for b, (s, (st, en)) in enumerate(zip(seqs, spans)):
        n = len(s)
        off = 0 if padding_side == "right" else t_n - n
        ids[b, off:off + n] = torch.tensor(s, dtype=torch.long)
        mask[b, off:off + n] = 1
        span[b, off + int(st):off + int(en)] = True
        last[b] = off + n - 1                                    # the last non-pad position: the final templated token
    pos = (mask.cumsum(dim=1) - 1).clamp(min=0)
    ids, mask, pos, span, last = (t.to(device) for t in (ids, mask, pos, span, last))
    with torch.inference_mode():
        res = model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True, use_cache=False)
        hs = res.hidden_states
        denom = span.sum(dim=1, keepdim=True).to(torch.float32)
        sel = span.unsqueeze(-1)
        rows = torch.arange(b_n, device=last.device)
        um, lg = {}, {}
        for li in layers:
            h = hs[li].to(torch.float32)
            um[li] = (torch.where(sel, h, torch.zeros((), dtype=h.dtype, device=h.device)).sum(dim=1) / denom).cpu().numpy()
            lg[li] = h[rows, last].cpu().numpy()
    return um, lg, len(hs)


def extract_split(model: Any, tokd: dict[str, Any], layers: list[int], device: str, pad_id: int, batch_size: int,
                  max_batch_tokens: int, n_layers: int, hidden: int, padding_side: str = "right",
                  sync: Callable | None = None, label: str = "",
                  boundary: int | None = None) -> tuple[dict[str, Any], float, int]:
    """Both poolings for every item of a split, in the split's own item order, with extract_hidden's non-finite
    handling: an item is flagged when any value of any layer is non-finite, then nan_to_num -> 0.0, and the values
    replaced are counted per layer (the stored cache cannot show them afterwards, F-016 addendum)."""
    n = len(tokd["ids"])
    um = {li: np.zeros((n, hidden), np.float32) for li in layers}
    lg = {li: np.zeros((n, hidden), np.float32) for li in layers}
    batches = length_batches(tokd["n_tokens"], batch_size, max_batch_tokens, boundary)
    t0 = time.time()
    for k, b in enumerate(batches):
        u, g, n_hs = pool_batch(model, [tokd["ids"][i] for i in b],
                                [(int(tokd["start"][i]), int(tokd["end"][i])) for i in b],
                                layers, device, pad_id, padding_side)
        if n_hs != n_layers + 1:
            refuse(f"the model returned {n_hs} hidden states, not {n_layers + 1}")
        idx = np.asarray(b)
        for li in layers:
            um[li][idx] = u[li]
            lg[li][idx] = g[li]
        if k % 50 == 0 or k == len(batches) - 1:
            done = sum(len(x) for x in batches[:k + 1])
            print(f"    {label} {done}/{n}  {done / max(time.time() - t0, 1e-9):.1f} items/s", flush=True)
    if sync is not None:
        sync()
    elapsed = time.time() - t0
    out: dict[str, Any] = {}
    for pooling, arrs in (("usermean", um), ("lastgen", lg)):
        bad = np.zeros(n, dtype=bool)
        replaced = {}
        for li in layers:
            fin = np.isfinite(arrs[li])
            bad |= ~fin.all(axis=1)
            replaced[li] = int((~fin).sum())
            if replaced[li]:
                arrs[li] = np.nan_to_num(arrs[li], nan=0.0, posinf=0.0, neginf=0.0)
        out[pooling] = {"hidden": arrs, "nonfinite": bad, "replaced": replaced}
    return out, elapsed, len(batches)


def read_back(path: Path) -> tuple[int, str]:
    """(bytes, sha256) of a file as read now from disk: a fresh open and a full read of its bytes."""
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return n, h.hexdigest()


def exclusive_savez(outp: Path, arrays: dict[str, np.ndarray], compress: bool = True) -> dict[str, Any]:
    """Write a cache that is never overwritten, never deleted, and is verified on disk (D-152 P5; F-027, F-032;
    D-307 (4); the no-deletion rule of src/safe_outputs.py).
      1. serialise in memory (np.savez_compressed, as extract_hidden.atomic_savez; or np.savez when compress is False,
         D-314: a cache of an 'ingest never' set stays on NVMe, where zlib only costs time, 1,705 s of the first
         sitting's 2,124) and load the bytes back: the same keys, shapes, dtypes and values;
      2. an exclusive temporary name ('xb'), fsync, its size and sha256 read back;
      3. a hard link to the final name, which fails if the name exists (os.replace would overwrite); then the
         temporary NAME is unlinked, which deletes no bytes (they stay under the final name);
      4. READ-BACK: a fresh open and read of the final name's bytes must give the in-memory bytes' length and sha256.
    On any failure nothing is deleted. A final name this call linked is this call's file by construction (a link of its
    own exclusive temporary), so it is renamed aside to '<name>.failed.<pid>' (no-replace) with no sha256 check, which a
    read-back mismatch would fail. A temporary is touched only if THIS call's open('xb') created it (never on a name
    collision), and then unlinked only under 1 MB. SystemExit names every file left (removal is a human's
    scripts/safe_rm.sh call). Returns {sha256, bytes, compressed, read_back}."""
    outp = Path(outp)
    buf = io.BytesIO()
    (np.savez_compressed if compress else np.savez)(buf, **arrays)
    data = buf.getvalue()
    with np.load(io.BytesIO(data), allow_pickle=False) as back:
        if set(back.files) != set(arrays):
            raise OSError(f"cache verify: key set differs for {outp.name}")
        for k, v in arrays.items():
            got, v = back[k], np.asarray(v)
            if got.shape != v.shape or got.dtype != v.dtype or not np.array_equal(got, v):
                raise OSError(f"cache verify: {k} differs after serialising {outp.name}")
    want = sha256_bytes(data)
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_name(f"{outp.name}.partial.{os.getpid()}.{os.urandom(4).hex()}")
    created = linked = False
    try:
        with open(tmp, "xb") as fh:
            created = True                                       # this call's file; a collision never gets here
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if tmp.stat().st_size != len(data) or sha256_file(tmp) != want:
            raise OSError(f"stage verify: {tmp.name} does not hold the bytes written")
        os.link(tmp, outp)                                       # FileExistsError if the name exists: never overwrite
        linked = True
        os.unlink(tmp)                                           # the temporary NAME only; the bytes stay under outp
        size, got = read_back(outp)                              # D-307 (4): the bytes on disk, read now
        if size != len(data) or got != want:
            raise OSError(f"cache read-back: {outp.name} on disk is {size} bytes, sha256 {got[:12]}; written "
                          f"{len(data)} bytes, sha256 {want[:12]} (D-307 (4))")
    except BaseException as e:
        left: list[str | None] = []
        if linked:
            aside = SO.aside_name(outp)
            try:
                SO.noreplace_rename(outp, aside)
                left.append(f"{aside} (renamed aside from {outp.name}: this call's file, linked from its own "
                            "exclusive temporary)")
            except OSError as e2:
                left.append(f"{outp} (left in place, this call's file: {type(e2).__name__}: {e2})")
        if created:
            left.append(SO.keep_or_unlink_temporary(tmp))
        raise SO.failure(f"cache {outp.name}: {type(e).__name__}: {e}", left) from None
    return {"sha256": want, "bytes": len(data), "compressed": compress,
            "read_back": {"bytes": size, "sha256": got, "equal_to_written": True,
                          "method": "a fresh open and full read of the final name after the link, compared with the "
                                    "in-memory bytes' length and sha256 (D-307 (4))"}}


# --- record helpers -----------------------------------------------------------------------------------------------

def stats(a: Any) -> dict[str, Any]:
    a = np.asarray(a)
    return {"n": int(a.size), "min": int(a.min()), "median": float(np.median(a)), "mean": float(a.mean()),
            "max": int(a.max()), "sum": int(a.sum())}


def code_files() -> list[str]:
    """CODE_FILES plus every src module loaded (phase2_tokenizer_check.py and whatever it imports), so the code block
    names each script and imported src module (D-152 P5) without guessing them in advance."""
    src = Path(__file__).resolve().parent
    loaded = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f and Path(f).suffix == ".py" and Path(f).resolve().parent == src:
            loaded.add(f"src/{Path(f).name}")
    return sorted(set(CODE_FILES) | loaded)


AUX_SNAPSHOT_FILES = ("config.json", "generation_config.json", "tokenizer.model", "special_tokens_map.json")
BIT_ENV_VARS = ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG",
                "CUBLASLT_WORKSPACE_SIZE", "PYTORCH_TUNABLEOP_ENABLED")


def bit_environment() -> dict[str, Any]:
    """The settings that can change the bits of a GPU forward pass: environment variables (unset -> None) and the
    torch backend flags as they are when the model runs."""
    import torch

    return {"env": {k: os.environ.get(k) for k in BIT_ENV_VARS},
            "torch.backends.cuda.matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "torch.backends.cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "torch.backends.cudnn.benchmark": bool(torch.backends.cudnn.benchmark)}


def lib_versions(device: str) -> dict[str, Any]:
    """Library versions and the name of the device THIS run used (not GPU 0 when the run is on the CPU)."""
    import torch
    from importlib import metadata

    dev = torch.device(device)
    name = (torch.cuda.get_device_name(dev) if dev.type == "cuda" and torch.cuda.is_available()
            else "cpu" if dev.type == "cpu" else None)
    out: dict[str, Any] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                           "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                           "device": str(dev), "device_name": name,
                           "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                           "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)}
    for dist in ("transformers", "tokenizers", "huggingface_hub", "peft", "numpy", "nvidia-cublas-cu12"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


# --- the run ------------------------------------------------------------------------------------------------------

def run(args: Any, *, root: Path = ROOT, record_dir: Path = P2_RESULTS, p2_manifest: Path = P2_MANIFEST,
        p1_manifest: Path = P1_MANIFEST, config_check: Path = CONFIG_CHECK, tokenizer_static: Path = TOKENIZER_STATIC,
        cache_default: Path = CACHE, locked: tuple[Path, ...] = LOCKED_DIRS, user_span_fn: Callable | None = None,
        load_from: str | None = None, require_nvme: bool = True, allow_no_commit: bool = False,
        results_dir: Path = RESULTS, p2sst_manifest: Path = P2SST_MANIFEST) -> tuple[Path, dict[str, Any]]:
    """Every refusal that needs no weights happens before any weight is loaded. The keyword arguments other than
    the defaults exist for the CPU test (temporary files, a tiny local model, the reference user_span)."""
    if args.phase1_mode:
        return run_phase1(args, root=root, p1_manifest=p1_manifest, results_dir=results_dir, locked=locked,
                          load_from=load_from, require_nvme=require_nvme, allow_no_commit=allow_no_commit)
    t_start, started = time.time(), utc_now()
    for flag, v in (("--kind", args.kind), ("--ingest", args.ingest)):
        if v is None:
            refuse(f"{flag} is required{' (yes|never; ' + INGEST_RULE + ')' if flag == '--ingest' else ''}")
    if args.items is not None or args.max_length is not None:
        refuse("--items and --max-length are for --phase1-mode only; a Phase 2 extraction takes whole splits and "
               "never truncates")
    if args.batch_size < 1 or args.max_batch_tokens < 1:
        refuse("--batch-size and --max-batch-tokens must be positive")
    tag = split_tag(args.splits)
    scans = sst_scans(args.repo, record_dir, p2sst_manifest) if tag == SST_TAG else None   # D-340 (3)
    lora_src = lora_source(args, root)                           # None unless --lora-record (fills the adapter args)
    ident_repo, ident_rev, adapter = identity(args, root)
    rec_path = Path(record_dir) / record_name(ident_repo, ident_rev, args.set_name, tag)
    if rec_path.exists():
        refuse(f"{rec_path.name} exists; this extraction is recorded and results are never overwritten")
    out_dir = check_out_dir(Path(args.out_dir) if args.out_dir else cache_write_dir(Path(cache_default)),
                            locked, require_nvme)
    splits = resolve_splits(args.splits, root=root, p2_manifest=p2_manifest, p1_manifest=p1_manifest,
                            p2sst_manifest=p2sst_manifest)
    planned = {(sp["name"], pooling): cache_name(ident_repo, ident_rev, sp["axis"], sp["part"], pooling, tag)
               for sp in splits for pooling in POOLINGS}
    clash = sorted(n for n in planned.values() if (out_dir / n).exists() or resolve_cache(n, Path(cache_default)).exists())
    if clash:
        refuse(f"cache(s) already exist and are never overwritten: {clash[:6]}")
    for f, what in ((config_check, "the model dims and pair verdicts"), (tokenizer_static, "the template and pin")):
        if not Path(f).is_file():
            refuse(f"{Path(f).name} is not on disk; {what} cannot be checked")
    cc = read_result(config_check)
    st = read_result(tokenizer_static)
    api = span_api()
    if (args.repo in api.BASE_MODELS) != (args.kind == "base"):
        refuse(f"--kind {args.kind} for {args.repo}: base checkpoints are exactly phase2_tokenizer_check.BASE_MODELS "
               f"{list(api.BASE_MODELS)}")
    pin = (st.get("clock") or {}).get("pin_date") if isinstance(st, dict) else None
    if args.template_date is not None and args.template_date != pin:
        refuse(f"--template-date {args.template_date} differs from the pin date tokenizer_static.json records "
               f"({pin}); D-303 (9) has one recorded pin")
    rev_block = check_revision(cc, api, args.repo, args.revision)
    snap_dir, weights = snapshot(args.repo, args.revision, load_from, require_nvme, root)
    blobs = weight_blobs(snap_dir) if adapter else None            # the LoRA path only; a base run's record unchanged

    import torch
    from transformers import AutoConfig, AutoTokenizer

    src = str(snap_dir)                                          # config, tokenizer and weights: this one snapshot
    cfg = AutoConfig.from_pretrained(src)
    n_layers, hidden = resolve_shape(cfg)
    cc_block = check_config(cc, args.repo, n_layers, hidden)
    max_ctx = max_context(cfg)
    boundary, rope_block = rope_boundary(cfg)
    layers = resolve_layers(list(args.layers), n_layers)
    tok = AutoTokenizer.from_pretrained(src)
    kw, tmpl = template_setup(tok, args.kind, pin, api)
    st_block = check_tokenizer_static(st, args.repo, args.revision, args.kind, tmpl["chat_template_sha256"])
    span_fn = user_span_fn or api.user_span_plain
    t0 = time.time()
    for sp in splits:
        sp["tok"] = tokenize_split(tok, sp, args.kind, kw, span_fn, api.ids_sha256)
    tokenize_s = time.time() - t0
    check_lengths(splits, max_ctx)
    tmpl.update(template_tokens(tok, splits))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (
        tok.eos_token_id if tok.eos_token_id is not None else 0)
    code0 = P.code_block(code_files(), allow_no_commit=allow_no_commit)

    cuda = args.device.startswith("cuda")
    t0 = time.time()
    print(f"extract_p2: {args.repo} @ {args.revision[:12]} kind={args.kind} dtype={args.dtype} device={args.device}"
          f" set={args.set_name}{' adapter ' + ident_repo if adapter else ''}", flush=True)
    _tok_unused, model = load_model(src, None, args.dtype, args.device, adapter["dir"] if adapter else None)
    if adapter:
        adapter.update(recheck_adapter(adapter))                 # the files merged are the files verified
    if cuda:
        torch.cuda.synchronize()
    model_load_s = time.time() - t0
    if resolve_shape(model.config) != (n_layers, hidden):
        refuse(f"the loaded model's shape {resolve_shape(model.config)} is not its config's {(n_layers, hidden)}")
    fwd = model.base_model                                       # a merged adapter's CausalLM -> its backbone
    sync = torch.cuda.synchronize if cuda else None
    print(f"  model loaded in {model_load_s:.0f}s: {n_layers} layers, hidden size {hidden}, max context {max_ctx}",
          flush=True)

    record: dict[str, Any] = {
        **provenance_block("extract_p2"),
        "code": code0,
        "preregistration": "D-303 (docs/phase2_design.md revision 3): additions (c) XSTest for every model, (d) the "
                           "user-span mean and the last-token state from one forward pass, (e) template text "
                           "recorded; details (1) the full layer grid on NVMe, (3) no truncation, (8) revisions, (9) a "
                           "clock-reading template pinned to the one recorded date, (12) tagged caches, records under "
                           "results/phase2/"
                           + ("; D-338 (4), D-340 (3): the SST-2 non-safety control (axis p2sst, results/phase2/"
                              "dataset_manifest_p2sst.json) under its own tag p2sst, read by axis and split" if tag ==
                              SST_TAG else ""),
        "ingest": args.ingest, "ingest_rule": INGEST_RULE,
        "set_name": args.set_name,
        "lib_versions": lib_versions(args.device), "bit_environment": bit_environment(),
        "repo": ident_repo, "revision": ident_rev,
        "cache_repo": f"{ident_repo}__{tag}", "cache_revision": ident_rev, "cache_tag": tag,
        "cache_tag_documentation": (f"D-297 (1) documented tag: cache_repo is repo + '__{TAG}' at the same full "
                                    "revision (verify_provenance.documented_tag); Phase 2 templated extraction, D-303"
                                    if tag == TAG else
                                    f"D-297 (1) documented tag: cache_repo is repo + '__{tag}' at the same full "
                                    "revision (verify_provenance.documented_tag); the Phase 2 templated extraction of "
                                    "the SST-2 non-safety control (D-338 (4)), its own tag so that no p2tpl reader "
                                    "(selection_p2, phase2_resplit_v2, phase2_bos_check) reads it (D-340 (3))"),
        "config": {"repo": ident_repo, "rev": ident_rev,
                   **({"adapter_base_repo": args.repo, "adapter_base_rev": args.revision} if adapter else {})},
        "model_repo": args.repo, "model_revision": args.revision, "revision_check": rev_block,
        "adapter": {k: v for k, v in adapter.items() if k != "dir"} if adapter else None,   # no host path (D-172)
        **({"adapter_source": lora_src} if lora_src else {}),
        **({"sst_scans": scans} if scans else {}),
        "adapter_name": args.adapter_name, "adapter_sha256": adapter["sha256"] if adapter else None,
        "adapter_config_sha256": adapter["config_sha256"] if adapter else None,
        "adapter_expected_sha256": adapter["expected_sha256"] if adapter else None,
        "adapter_expected_config_sha256": adapter["expected_config_sha256"] if adapter else None,
        "kind": args.kind,
        "input_format": ("the model's own chat template, the prompt as the one user turn, add_generation_prompt=True; "
                         "ids from apply_chat_template(tokenize=True)" if args.kind == "instruct" else
                         "the raw prompt, tok(text) with the tokenizer's normal special tokens")
                        + "; special-token strings in the prompt itself are tokenized as plain text (D-324 (5)(v), "
                          "special_strings_rule; an item without one keeps exactly these ids)",
        "poolings": {"usermean": ("mean over the user message's token span [span_start, span_end) only (template "
                                  "and system tokens excluded, element-wise), from "
                                  f"{getattr(span_fn, '__module__', '?')}.{getattr(span_fn, '__name__', '?')}"
                                  if args.kind == "instruct" else
                                  "mean over the prompt's own tokens (the tokenizer's added special tokens excluded, "
                                  "element-wise), from "
                                  f"{getattr(span_fn, '__module__', '?')}.{getattr(span_fn, '__name__', '?')}"
                                  "(raw=True)")
                                 + " (D-324 (5)(v): special-token strings in the prompt tokenized as plain text)",
                     "lastgen": "the final position of the full templated prompt (index n_tokens - 1), where the "
                                "model would begin its answer"},
        "padding": "right, in length-sorted batches, longest first; position ids counted from the attention mask; the "
                   "last-token index is each row's own last non-pad position",
        "rope": rope_block,
        "template": tmpl,
        "tokenizer_static": {"file": PV._repo_rel(Path(tokenizer_static)),
                             "sha256": PV._RESULT_READS[PV._repo_rel(Path(tokenizer_static))], **st_block},
        "tokenizer": {"class": type(tok).__name__, "bos_token": tok.bos_token, "eos_token": tok.eos_token,
                      "pad_token": tok.pad_token, "pad_id_used": int(pad_id),
                      "model_max_length": int(tok.model_max_length) if tok.model_max_length < 2 ** 62 else None},
        "model": {"n_layers": n_layers, "hidden_size": hidden, "max_position_embeddings": max_ctx,
                  "config_class": type(cfg).__name__, "loaded_class": type(model).__name__,
                  "forward_module": type(fwd).__name__, "weight_files": weights,
                  **({"weight_blobs": blobs} if blobs is not None else {}),
                  "sliding_window": sliding_window_info(cfg, splits)},
        "config_check": {"file": PV._repo_rel(Path(config_check)),
                         "sha256": PV._RESULT_READS[PV._repo_rel(Path(config_check))], **cc_block},
        "layers_requested": list(args.layers), "layers_resolved": layers,
        "layers_every_second_grid": layers == list(range(0, n_layers + 1, 2)),
        "dtype": args.dtype, "device": args.device, "batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "cache_dir_nvme_checked": require_nvme,
        "token_ids_sha256_encoding": "phase2_tokenizer_check.ids_sha256 (the tokenizer check's per-item hash, so the "
                                     "two records compare item by item)",
        "user_span_source": f"{getattr(span_fn, '__module__', '?')}.{getattr(span_fn, '__name__', '?')}",
        "special_strings_rule": special_strings_block(tok, api, span_fn, splits),
        "splits": {}, "split_runs": {},
    }

    forward_s = write_s = 0.0
    written: list[tuple[Path, str]] = []                         # every cache this run wrote, with its sha256
    try:
        for sp in splits:
            t = sp["tok"]
            n = len(sp["rows"])
            print(f"  {sp['name']}: {n} items", flush=True)
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            res, f_s, n_batches = extract_split(fwd, t, layers, args.device, int(pad_id), args.batch_size,
                                                args.max_batch_tokens, n_layers, hidden, "right", sync, sp["name"],
                                                boundary)
            peak = int(torch.cuda.max_memory_allocated()) if cuda else None
            forward_s += f_s
            t0 = time.time()
            common = {"item_ids": np.array([r["item_id"] for r in sp["rows"]]),
                      "labels": np.array([r["label"] for r in sp["rows"]], dtype=np.int8),
                      "n_tokens": t["n_tokens"], "truncated": np.zeros(n, dtype=bool),
                      "token_ids_sha256": t["ids_sha256"]}
            extra = {"usermean": {"span_start": t["start"], "span_end": t["end"], "n_straddle": t["n_straddle"]},
                     "lastgen": {"last_index": t["n_tokens"] - 1}}
            caches = {}
            for pooling in POOLINGS:
                r = res[pooling]
                name = planned[(sp["name"], pooling)]
                arrays = {**common, **extra[pooling], "nonfinite": r["nonfinite"],
                          **{f"layer_{li}": r["hidden"][li] for li in layers}}
                # read back from disk before anything is recorded; zlib only for a set that may go to the NAS (D-314)
                wrote = exclusive_savez(out_dir / name, arrays, compress=args.ingest != "never")
                written.append((out_dir / name, wrote["sha256"]))
                caches[pooling] = name
                bad = np.flatnonzero(r["nonfinite"])
                record["splits"][f"{sp['name']}/{pooling}"] = {
                    "split": sp["name"], "pooling": pooling, "n": n, "n_items": n,
                    "split_file": sp["file"], "split_sha256": sp["sha256"],
                    "cache_file": name, "cache_sha256": wrote["sha256"], "cache_bytes": wrote["bytes"],
                    "cache_compressed": wrote["compressed"], "read_back": wrote["read_back"],
                    "n_nonfinite": int(r["nonfinite"].sum()),
                    "nonfinite_replacements": {
                        "replacement": "nan_to_num -> 0.0, after the flag is set",
                        "item_indices": bad.tolist(), "item_ids": [sp["rows"][i]["item_id"] for i in bad],
                        "values_replaced_per_layer": {str(li): v for li, v in sorted(r["replaced"].items())}},
                    "n_truncated": 0, "layers_resolved": layers, "n_model_layers": n_layers, "hidden_size": hidden,
                    "forward_pass": f"shared with the other pooling; timing in split_runs['{sp['name']}']",
                    "special_strings_as_text": {"n_items": len(t["plain"]), "items": t["plain"],
                                                "rule": "D-324 (5)(v); special_strings_rule"}}
                print(f"    -> {name}  nonfinite={int(r['nonfinite'].sum())}", flush=True)
            w_s = time.time() - t0
            write_s += w_s
            span_len = t["end"] - t["start"]
            record["split_runs"][sp["name"]] = {
                "split_file": sp["file"], "split_sha256": sp["sha256"], "manifest": sp["manifest"], "n_items": n,
                "caches": caches, "tokens": stats(t["n_tokens"]), "span_length": stats(span_len),
                "template_tokens": stats(t["n_tokens"] - span_len),
                "n_straddle": {"total": int(t["n_straddle"].sum()),
                               "items_with_any": int((t["n_straddle"] > 0).sum())},
                "special_strings_as_text": len(t["plain"]),
                "n_batches": n_batches, "forward_s": round(f_s, 3), "items_per_s": round(n / max(f_s, 1e-9), 3),
                "cache_write_s": round(w_s, 3), "cuda_max_memory_allocated_bytes": peak}
            print(f"    {sp['name']}: forward {f_s:.1f}s ({n / max(f_s, 1e-9):.1f} items/s), write {w_s:.1f}s"
                  f"{f', peak {peak / 2 ** 30:.1f} GiB' if peak is not None else ''}", flush=True)

        n_all = sum(len(sp["rows"]) for sp in splits)
        record["timing"] = {
            "started_utc": started, "finished_utc": utc_now(), "wall_s": round(time.time() - t_start, 3),
            "tokenize_s": round(tokenize_s, 3), "model_load_s": round(model_load_s, 3),
            "forward_s": round(forward_s, 3), "cache_write_s": round(write_s, 3), "n_items": n_all,
            "items_per_s_forward": round(n_all / max(forward_s, 1e-9), 3),
            "reading": "wall_s is this model's extraction time from start to the record; forward_s the GPU passes alone"}
        if P.code_block(code_files(), allow_no_commit=allow_no_commit)["sha256"] != code0["sha256"]:
            refuse("the code changed during the run; the record is not written")
        # D-297 (1): every results file this process read through read_result, listed with its sha256
        record["input_files_sha256"] = dict(PV._RESULT_READS)
        digest = write_results(rec_path, record, exclusive=True)
    except BaseException as e:
        if not written:
            raise
        raise failed_keeping(e, written, rec_path) from None
    print(f"\nWritten: {PV._repo_rel(rec_path)}\nsha256:  {digest}\nwall {record['timing']['wall_s']:.0f}s for "
          f"{n_all} items; ingest {args.ingest}", flush=True)
    return rec_path, record


def failed_keeping(e: BaseException, written: list[tuple[Path, str]], rec_path: Path) -> SystemExit:
    """A run that failed after writing caches deletes nothing (src/safe_outputs.py). Unless the record naming them
    exists, each cache this run wrote is renamed aside to '<name>.failed.<pid>' (no-replace, and only if it still holds
    the bytes written); the exit names every file left and says removal is a human's scripts/safe_rm.sh call."""
    if Path(rec_path).exists():
        left = [f"{p} (left in place: the record {Path(rec_path).name} names it)" for p, _ in written]
    else:
        left = [SO.rename_aside(p, sha) for p, sha in written]
    return SO.failure(str(e) if isinstance(e, SystemExit) else f"{type(e).__name__}: {e}", left)


# --- Phase 1 mode (D-307 (5)) -------------------------------------------------------------------------------------

def phase1_extract(tok: Any, model: Any, rows: list[dict[str, Any]], layer_indices: list[int], batch_size: int,
                   max_length: int, device: str) -> dict[str, Any]:
    """src/extract_hidden.py extract() for pooling='mean' and apply_chat_template=False, step for step: the same
    tokenizer call (default padding side and special tokens), batches in file order, the same forward call under
    torch.no_grad(), the same float32 mask arithmetic, the same non-finite handling. Any change to extract_hidden's
    numerics must be made here too; the test compares the two bit for bit."""
    import torch

    n_layers, hidden_size = resolve_shape(model.config)
    resolved = [i if i >= 0 else n_layers + 1 + i for i in layer_indices]
    for i in resolved:
        if not 0 <= i <= n_layers:
            raise ValueError(f"layer index {i} out of range for {n_layers}-layer model")

    out = {li: np.zeros((len(rows), hidden_size), dtype=np.float32) for li in resolved}
    nonfinite = np.zeros(len(rows), dtype=bool)
    replaced = {li: 0 for li in resolved}
    truncated = np.zeros(len(rows), dtype=bool)
    n_tokens = np.zeros(len(rows), dtype=np.int32)

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start: start + batch_size]
            texts = [r["text"] for r in batch]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            lens = enc["attention_mask"].sum(dim=1)
            for k, _row in enumerate(batch):
                n_tokens[start + k] = int(lens[k])
                truncated[start + k] = bool(lens[k] >= max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            res = model(**enc, output_hidden_states=True)
            mask = enc["attention_mask"].unsqueeze(-1).to(torch.float32)
            for li in resolved:
                h = res.hidden_states[li].to(torch.float32)
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                finite = torch.isfinite(pooled)
                bad = ~finite.all(dim=1)
                nonfinite[start: start + len(batch)] |= bad.cpu().numpy()
                replaced[li] += int((~finite).sum().item())
                out[li][start: start + len(batch)] = (
                    torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()
                )
    return {"hidden": out, "nonfinite": nonfinite, "replaced": replaced, "truncated": truncated,
            "n_tokens": n_tokens, "n_model_layers": n_layers, "hidden_size": hidden_size,
            "elapsed_s": time.time() - t0}


def phase1_split(name: str, items: int, root: Path, p1_manifest: Path) -> dict[str, Any]:
    """ONE Phase 1 split, data/frozen/<axis>__<part>.jsonl (the file extract_hidden.load_split reads), its sha256
    verified against results/dataset_manifest.json (read_result), and its first `items` rows in file order."""
    m = SPLIT_RE.fullmatch(name)
    if not m:
        refuse(f"split {name!r} is not '<axis>/<part>'")
    axis, part = m.groups()
    if not Path(p1_manifest).is_file():
        refuse(f"{p1_manifest} is not on disk; the Phase 1 split cannot be verified")
    e = _manifest_entry(read_result(p1_manifest), axis, part, Path(p1_manifest).name)
    want_file = f"data/frozen/{axis}__{part}.jsonl"
    if e["file"] != want_file:
        refuse(f"{Path(p1_manifest).name} names {e['file']} for {name}, not {want_file} (extract_hidden.load_split)")
    data = (Path(root) / e["file"]).read_bytes()
    got = sha256_bytes(data)
    if got != e["sha256"]:
        refuse(f"{e['file']} has sha256 {got[:12]}..., not the manifest's {e['sha256'][:12]}...")
    rows = parse_jsonl_bytes(data)
    check_rows(rows, name)
    if "n" in e and e["n"] != len(rows):
        refuse(f"{e['file']}: the manifest says n = {e['n']}, the file holds {len(rows)}")
    if not 1 <= items <= len(rows):
        refuse(f"--items {items} is not within 1..{len(rows)} for {name}")
    return {"name": name, "axis": axis, "part": part, "file": e["file"], "sha256": got,
            "manifest": PV._repo_rel(Path(p1_manifest)), "n_rows": len(rows), "rows": rows[:items]}


def run_phase1(args: Any, *, root: Path = ROOT, p1_manifest: Path = P1_MANIFEST, results_dir: Path = RESULTS,
               locked: tuple[Path, ...] = LOCKED_DIRS, load_from: str | None = None, require_nvme: bool = True,
               allow_no_commit: bool = False) -> tuple[Path, dict[str, Any]]:
    """--phase1-mode: extract_hidden's mean pooling for the first --items items of one Phase 1 split, with its
    parameters, written only outside results/ (D-307 (5)). The record names every Phase 1 parameter."""
    started = utc_now()
    for flag, v in (("--kind", args.kind), ("--adapter", args.adapter), ("--adapter-name", args.adapter_name),
                    ("--template-date", args.template_date), ("--lora-record", getattr(args, "lora_record", None)),
                    ("--lora-step", getattr(args, "lora_step", None))):
        if v is not None:
            refuse(f"{flag} has no meaning in --phase1-mode (raw text, no adapter, no template)")
    if args.ingest == "yes":
        refuse("--phase1-mode writes a test record outside results/ that is never ingested")
    if args.items is None or args.batch_size is None:
        refuse("--phase1-mode needs --items and --batch-size (the Phase 1 run's)")
    if args.batch_size < 1 or args.max_length < 1:
        refuse("--batch-size and --max-length must be positive")
    if args.items % args.batch_size:
        refuse(f"--items {args.items} is not a multiple of --batch-size {args.batch_size}: bit for bit needs the Phase 1 "
               "run's own batches (D-307 (5))")
    if len(args.splits) != 1:
        refuse("--phase1-mode takes exactly one split")
    ident_repo, ident_rev, _ = identity(args, root)
    out_dir = Path(os.path.realpath(args.out_dir)) if args.out_dir else None
    if out_dir is None:
        refuse("--phase1-mode needs --out-dir (a test directory outside results/)")
    rres = Path(os.path.realpath(results_dir))
    if out_dir == rres or rres in out_dir.parents:
        refuse(f"--phase1-mode writes only outside {results_dir} (D-307 (5)); {out_dir} is inside it")
    out_dir = check_out_dir(out_dir, locked, require_nvme)
    sp = phase1_split(args.splits[0], args.items, root, p1_manifest)
    n = len(sp["rows"])
    safe = ident_repo.replace("/", "__")
    cname = f"{safe}__p1mode@{ident_rev[:12]}__{sp['axis']}__{sp['part']}__mean__n{n}.npz"
    rec_path = out_dir / f"p1mode__{safe}@{ident_rev[:12]}__{sp['axis']}__{sp['part']}__n{n}.json"
    for p in (out_dir / cname, rec_path):
        if p.exists():
            refuse(f"{p.name} exists; nothing is overwritten")
    _snap, weights = snapshot(args.repo, args.revision, load_from, require_nvme, root)
    code0 = P.code_block(code_files(), allow_no_commit=allow_no_commit)

    src, src_rev = (load_from, None) if load_from else (args.repo, args.revision)   # as extract_hidden.main calls it
    print(f"extract_p2 --phase1-mode: {args.repo} @ {args.revision[:12]} {sp['name']} first {n} items, batch "
          f"{args.batch_size}, max_length {args.max_length}, dtype {args.dtype}, device {args.device}", flush=True)
    t0 = time.time()
    tok, model = load_model(src, src_rev, args.dtype, args.device, None)
    model_load_s = time.time() - t0
    res = phase1_extract(tok, model, sp["rows"], list(args.layers), args.batch_size, args.max_length, args.device)
    rows = sp["rows"]
    arrays = {                                                   # extract_hidden.main's arrays, names and order
        "item_ids": np.array([r["item_id"] for r in rows]),
        "labels": np.array([r["label"] for r in rows], dtype=np.int8),
        "nonfinite": res["nonfinite"],
        "truncated": res["truncated"],
        "n_tokens": res["n_tokens"],
        **{f"layer_{li}": res["hidden"][li] for li in res["hidden"]},
    }
    bad = np.flatnonzero(res["nonfinite"])
    record: dict[str, Any] = {
        **provenance_block("extract_p2_phase1_mode"),
        "code": code0,
        "preregistration": "D-307 (5): Phase 1 mode for the bit-for-bit equivalence check before any Phase 2 "
                           "extraction; compared by src/p1mode_equivalence.py",
        "mode": "phase1", "ingest": "never", "ingest_rule": "a test record outside results/; never ingested",
        "lib_versions": lib_versions(args.device), "bit_environment": bit_environment(),
        "repo": args.repo, "revision": args.revision, "config": {"repo": args.repo, "rev": args.revision},
        "adapter": None, "adapter_sha256": None,
        "cache_repo": f"{args.repo}__p1mode", "cache_revision": args.revision,
        "layers_requested": list(args.layers), "pooling": "mean", "chat_template_applied": False,
        "tokenizer_has_chat_template": bool(getattr(tok, "chat_template", None)),
        "dtype": args.dtype, "max_length": args.max_length, "batch_size": args.batch_size,
        "items": n, "split_n_rows": sp["n_rows"], "items_multiple_of_batch_size": n % args.batch_size == 0,
        "phase1_parameters": {
            "tokenization": "tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)",
            "padding_side": getattr(tok, "padding_side", None), "pad_token": tok.pad_token,
            "pad_token_id": tok.pad_token_id, "add_special_tokens": "the tokenizer's default",
            "batching": "file order, batch_size items per batch, no length sorting, no token budget",
            "forward": "model(**enc, output_hidden_states=True) under torch.no_grad(); no position_ids passed",
            "pooling": "(h.to(float32) * mask).sum(1) / mask.sum(1).clamp(min=1), mask = attention_mask in float32",
            "nonfinite": "flag per item, then nan_to_num -> 0.0 (torch), values replaced counted per layer",
            "attn_implementation": getattr(model.config, "_attn_implementation", None),
            "model_class": type(model).__name__, "model_dtype": str(next(model.parameters()).dtype),
            "loader": "extract_hidden.load_model(repo, revision, dtype, device) itself"},
        "not_mirrored": PHASE1_NOT_MIRRORED,
        "weights": weights,
        "splits": {sp["name"]: {
            "n": n, "split_file": sp["file"], "split_sha256": sp["sha256"], "manifest": sp["manifest"],
            "cache_file": cname,
            "n_nonfinite": int(res["nonfinite"].sum()),
            "nonfinite_replacements": {
                "replacement": "nan_to_num -> 0.0, after the flag is set",
                "item_indices": bad.tolist(), "item_ids": [rows[i]["item_id"] for i in bad],
                "values_replaced_per_layer": {str(li): v for li, v in sorted(res["replaced"].items())}},
            "n_truncated": int(res["truncated"].sum()),
            "mean_tokens": float(res["n_tokens"].mean()), "max_tokens": int(res["n_tokens"].max()),
            "elapsed_s": round(res["elapsed_s"], 1), "layers_resolved": sorted(res["hidden"].keys()),
            "n_model_layers": res["n_model_layers"], "hidden_size": res["hidden_size"]}},
        "timing": {"started_utc": started, "finished_utc": utc_now(), "model_load_s": round(model_load_s, 3),
                   "forward_s": round(res["elapsed_s"], 3)},
    }
    wrote = exclusive_savez(out_dir / cname, arrays)            # read back from disk before anything is recorded
    try:
        record["splits"][sp["name"]].update({"cache_sha256": wrote["sha256"], "cache_bytes": wrote["bytes"],
                                             "read_back": wrote["read_back"]})
        if P.code_block(code_files(), allow_no_commit=allow_no_commit)["sha256"] != code0["sha256"]:
            refuse("the code changed during the run; the record is not written")
        record["input_files_sha256"] = dict(PV._RESULT_READS)
        digest = write_results(rec_path, record, exclusive=True)
    except BaseException as e:
        raise failed_keeping(e, [(out_dir / cname, wrote["sha256"])], rec_path) from None
    print(f"\nWritten (test record, outside results/): {rec_path}\nsha256:  {digest}\ncache {cname} "
          f"read back {wrote['read_back']['bytes']} bytes", flush=True)
    return rec_path, record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase 2 hidden-state extraction (D-303): user-span mean and "
                                             "last-token from one templated forward pass")
    ap.add_argument("--repo", required=True, help="'<org>/<model>'; for --adapter, the base it is merged into")
    ap.add_argument("--revision", required=True,
                    help="exact 40-hex commit: the config check's resolved commit or the Phase 1 pin (D-303 (8))")
    ap.add_argument("--kind", default=None, choices=["instruct", "base"],
                    help="required (not in --phase1-mode). instruct: the model's chat template; base: the raw prompt "
                         "(phase2_tokenizer_check.BASE_MODELS)")
    ap.add_argument("--ingest", default=None, choices=["yes", "never"],
                    help="required (not in --phase1-mode), recorded as the top-level 'ingest'. " + INGEST_RULE +
                         "; a 'never' set's caches are written uncompressed (D-314)")
    ap.add_argument("--layers", nargs="+", type=int, required=True,
                    help="hidden_states indices (0 = embeddings; the full grid: every second layer 0..last)")
    ap.add_argument("--splits", nargs="+", required=True,
                    help="'<axis>/<part>' from results/phase2/dataset_manifest_p2.json (files under data/frozen_p2/); "
                         "over_refusal/eval is XSTest, from results/dataset_manifest.json; 'p2sst/<part>' (SST-2) from "
                         "results/phase2/dataset_manifest_p2sst.json, in a run of its own under the tag p2sst")
    ap.add_argument("--set-name", default="all",
                    help="label of this split set in the record name, so a later set of the same model gets its own "
                         "record (letters and digits)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory; merged into --repo before extraction")
    ap.add_argument("--adapter-name", default=None, help="identity of the merged model's caches")
    ap.add_argument("--adapter-sha256", default=None,
                    help="with --adapter, required: the recorded sha256 of adapter_model.safetensors (D-307 (2))")
    ap.add_argument("--adapter-config-sha256", default=None,
                    help="with --adapter, required: the recorded sha256 of adapter_config.json (D-307 (2))")
    ap.add_argument("--lora-record", default=None,
                    help="results/phase2/lora_train_p2__<corpus>__s<seed>.json: the adapter, its identity and its "
                         "recorded sha256s are read from it and from manifests/ingest_lora/ (src/lora_adapter_p2.py); "
                         "not with --adapter, --adapter-name, --adapter-sha256 or --adapter-config-sha256")
    ap.add_argument("--lora-step", type=int, default=None,
                    help="with --lora-record: the checkpoint (0, 25, 100, 400); step 0 through the alpaca record only "
                         "(once per seed)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="default 32 (the token cap protects memory); in --phase1-mode required, the Phase 1 run's")
    ap.add_argument("--max-batch-tokens", type=int, default=16384,
                    help="at most this many padded tokens per batch (no truncation: a longer item runs alone)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"],
                    help="weights; hidden states are upcast to float32 as they leave the model (extract_hidden)")
    ap.add_argument("--out-dir", default=None,
                    help="NVMe scratch for the caches; default GUARDRAIL_CACHE_WRITE_DIR. Never data/hidden or data/lora")
    ap.add_argument("--template-date", default=None,
                    help="optional cross-check only: must equal tokenizer_static.json's clock.pin_date, the one "
                         "recorded pin (D-303 (9)); the pin itself is always read from that record")
    ap.add_argument("--phase1-mode", action="store_true",
                    help="D-307 (5): extract_hidden's mean pooling, no template, for the first --items items of ONE "
                         "Phase 1 split, written only outside results/ (for src/p1mode_equivalence.py)")
    ap.add_argument("--items", type=int, default=None, help="--phase1-mode: the first N items of the split, file order")
    ap.add_argument("--max-length", type=int, default=None,
                    help="--phase1-mode only: extract_hidden's --max-length (tokenizer truncation), default 512 as "
                         "there; a Phase 2 extraction never truncates")
    a = ap.parse_args(argv)
    if a.phase1_mode:
        a.max_length = 512 if a.max_length is None else a.max_length
    elif a.batch_size is None:
        a.batch_size = 32
    return a


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    args = parse_args(argv)
    # Before anything loads: weights only from the local NVMe cache, never the network.
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        refuse("HF_HUB_OFFLINE is not 1; weights are read only from the NVMe Hugging Face cache")
    assert_hf_env("extract_p2.py")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
