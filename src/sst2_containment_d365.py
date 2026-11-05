"""D-365 (4)(vii): SST-2 phrases in the alpaca training text of the Phase 2 LoRA runs, counted as COMPOSITION.

What it answers. Exact item_id matches between an SST-2 split and a training conversation were excluded at training:
every Phase 2 LoRA record's frozen_exclusion lists data/frozen_p2/p2sst__{probe_train,calibration,eval}.jsonl, and the
trainer dropped a conversation whose user turn or Phase 1 prompt hashed to a frozen item_id (D-344 (4)). An SST-2 item
can still occur INSIDE a longer alpaca message (GLUE's SST-2 is phrase-level: sub-phrases and single words). D-365
(4)(vii) counts that substring containment before any SST-2 claim, as composition (D-341's precedent): a count
reported, not a split rule. Nothing is excluded, no split changes, no probe number is computed.

INPUTS (each bound, any difference refuses; nothing is written then):
  the three alpaca training records results/phase2/lora_train_p2__alpaca__s{0,1,2}.json (train_lora_p2.record_name):
      run_kind lora_train_p2, corpus alpaca, seeds 0-2, one dataset and revision, one fetch record, snapshot, layout
      and train file; used_row_indices (the trainer's own list of the rows it consumed, in order) of length
      n_examples_used = max(save_steps) x micro_bs x grad_accum, distinct, an in-order subsequence of
      numpy.random.default_rng(seed).permutation(n_rows) whose first 10 are selection.order_first_10 and whose skipped
      rows inside the prefix are at most the recorded exclusions and drops; contamination 0 for user turns and the
      Phase 1 prompt; the code block's src/train_lora_p2.py and src/train_lora.py equal to the present files (so
      messages_for, item_id and phase1_prompt below are the code that trained).
  the SST-2 LoRA extraction records of each seed, results/phase2/extract__lora__mistral7bi-v02-alpaca-r16-seed<s>-
      step*__p2sst@*__*.json (D-386): at least one per seed, and every one must name that seed's training record, file
      and sha256, as its adapter_source.record (so the text counted is the training of the adapters whose SST-2
      caches exist); each is listed with its sha256.
  the corpus, through the pinned path and sha256 the records name: the fetch record (the records' file and sha256,
      train_lora_p2.check_fetch_record, pinned to the records' Phase 1 record), the snapshot
      datasets--yahma--alpaca-cleaned/snapshots/<revision> under --local-hub with every file's sha256 equal to the
      fetch record's and to the records' snapshot_files_sha256 (train_lora_p2.verify_snapshot), the layout
      (check_layout) and the train file read (read_rows: alpaca_data_cleaned.json, bytes hashed as read) equal to the
      records' train_files_read, and n_rows equal to selection.n_rows.
  the SST-2 splits data/frozen_p2/p2sst__{probe_train,calibration,eval}.jsonl at the sha256, n and label counts of
      results/phase2/dataset_manifest_p2sst.json (D-341) and of every training record's frozen_exclusion entry; every
      row axis p2sst, label 0 or 1, a non-empty stripped text whose sha256 prefix is its item_id (freeze rule), ids
      distinct.
  src/phase2_pool_clusters.py (SubstringIndex, CONTAINMENT_Q: the code of D-338 (3)'s containment) at the sha256 the
      manifest's code block names.

THE COUNT (CHOICES below). Per seed record, the consumed conversations' messages as train_lora_p2.messages_for builds
them (user: the stripped instruction, joined by '\\n\\n### Input:\\n' to the stripped input when there is one;
assistant: the stripped output). An SST-2 item is CONTAINED in a message when its normalised text is a substring of
the message's normalised text: 'substring' (D-338 (3): str.lower, character-level `in`), 'token_aligned' beside
(' ' + text + ' ' on both). For each item, seed, normalisation and role the first consumed position holding it is
found (exactly: phase2_pool_clusters.SubstringIndex, each candidate verified with `in`); the checkpoint of step k has
seen the first micro_bs x grad_accum x k consumed examples (D-344 (3)'s dose rule; 16k in the Phase 2 runs), so the
item counts at step k when that position is below that. Counted per SST-2 split, per label (D-338 (3)'s form: n,
contained, contained_pos, contained_neg), per seed and over the union of the three seeds, at every positive save step
the records give (25, 100 and 400 in the Phase 2 runs), for the roles either, user and assistant (none is named
primary). Beside it, a reading aid that is not a rule (D-344 (3)'s): the last step's counts without the items under 8,
16 and 32 characters. MATCHED: every item contained (substring, any role, any seed, last step) with its split, item_id,
label, length and first consumed position per normalisation, role and seed; no text is written.

BIAS, stated in the record (bias(), built from the records' seq_len). An UPPER bound on what was trained: the messages
are counted untruncated, and training cut each example to its first seq_len tokens (each record's n_truncated); every
consumed example is counted, including those of an optimiser step GradScaler skipped after a gradient overflow (each
skip halves the scale: the halvings in each record's training.loss_curve grad_scale_after, and its non-finite
micro-batch losses, are copied per seed into training_records). Short items inflate it: a single word or short phrase
of SST-2 occurs in ordinary English without any shared source (the reading aid shows its size). A LOWER bound on
shared sources: SST-2's text is tokenised (a space before punctuation and clitics) and only lowercased here, so a
review sentence present in alpaca in ordinary spelling is not found; paraphrases are not measured. token_aligned is a
LOWER bound here and NOT comparable to D-341's token_aligned column: on alpaca's raw text an occurrence next to
punctuation, a quote or a newline has no space on that side and is missed, while D-341 compared SST-2 with SST-2, both
tokenised.

CHOICES BY THE CLOSEST D-ENTRY (D-181), also written into the record (choices(), with the steps the records give):
  (a) the text: each consumed conversation's user and assistant messages (train_lora_p2.messages_for, the trained
      code), not the chat-template render, whose added markers hold no SST-2 text (D-344 (3) counts by turn).
  (b) the examples: each record's used_row_indices, per seed and per dose (the first micro_bs x grad_accum x step
      consumed examples at each positive save step), and over the union of the seeds (D-344 (3): per record and dose,
      the consumed prefix).
  (c) the normalisation: D-338 (3)'s containment rule (phase2_pool_clusters.CONTAINMENT_RULE; D-340 (2)): str.lower
      and character-level `in`, 'token_aligned' beside; exact, through phase2_pool_clusters.SubstringIndex.
  (d) roles: 'either' (user or assistant), D-365 (4)(vii)'s 'training text' (the loss is on the full sequence, the
      records' loss), with user and assistant separately as D-344 (3) reports turns; none is named primary (no D-entry
      names one for this count; which column a text quotes is the author's).
  (e) exact matches: the SST-2 item_ids against every consumed user turn's and Phase 1 prompt's item_id (the trainer's
      own rule, train_lora_p2.user_turn_ids and phase1_prompt) must give 0, as the records state, else the rows read
      are not the ones trained and the run refuses; the records' frozen_hits_by_file for the three SST-2 files (the
      whole corpus) are copied beside.
  (f) short items: the D-344 (3) reading aid (8, 16 and 32 characters), not a rule; nothing is excluded by it.
  (g) alpaca only: D-365 (4)(vii) names alpaca's training text; the HH and PKU runs are not counted here.
  (h) where: results/phase2/sst2_containment_d365.json, beside the SST-2 records it qualifies; written once.
  (i) the hub directory is a command-line argument and is not written into the record (no host path); the snapshot is
      named by its path relative to the hub.

    python3 src/sst2_containment_d365.py --local-hub "$HOME/.cache/gccf_scratch/p2_datasets/hub"
CPU only, no network, no Hugging Face library. It needs the alpaca snapshot that src/fetch_dataset_snapshot.py
--lora-corpus alpaca wrote under the hub (the node's NVMe); without it it refuses ('no local snapshot') and writes
nothing. A test writes elsewhere with --out (never under results/).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as P  # noqa: E402  (code_block, D-161)
import phase2_pool_clusters as PC  # noqa: E402  (SubstringIndex, CONTAINMENT_Q, CONTAINMENT_RULE: D-338 (3)'s rule)
import provenance as PV  # noqa: E402
import train_lora_p2 as T2  # noqa: E402  (the pinned-snapshot readers, fetch-record checks, messages, item ids)
from provenance import (  # noqa: E402
    canonical_hash,
    ensure_utf8_stdout,
    provenance_block,
    read_result,
    sha256_bytes,
    sha256_file,
    write_results,
)

ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = ROOT                        # where the source files are hashed (a test's root is a scratch tree)
RESULTS = ROOT / "results"
OWN = "src/sst2_containment_d365.py"
RUN_KIND = "sst2_containment_d365"
FINAL_NAME = "sst2_containment_d365.json"           # under results/phase2
DECISION = "D-365 (4)(vii)"
CORPUS = "alpaca"
SEEDS = tuple(T2.SEEDS)                              # D-303 (7): 0, 1, 2
AXIS = "p2sst"
SPLITS = ("probe_train", "calibration", "eval")
MANIFEST = "dataset_manifest_p2sst.json"             # under results/phase2 (D-341)
ROLES = ("user", "assistant")
ROLE_COLUMNS = ("either", "user", "assistant")
NORMS = {"substring": lambda t: t.lower(), "token_aligned": lambda t: " " + t.lower() + " "}
SHORT_ITEM_CHARS = (8, 16, 32)                       # D-344 (3)'s reading aid
TRAINING_CODE = ("src/train_lora_p2.py", "src/train_lora.py")   # must be the training records' bytes
CONTAINMENT_CODE = "src/phase2_pool_clusters.py"                 # must be the SST-2 manifest's bytes
NONE = np.iinfo(np.int64).max                         # 'not contained' as a first position

PREREG = ("D-365 (4)(vii): SST-2 phrases in alpaca's training text; exact item_id matches were already excluded at "
          "training (every Phase 2 LoRA record's frozen_exclusion lists data/frozen_p2/p2sst__{probe_train,"
          "calibration,eval}.jsonl); substring containment is counted as composition before any SST-2 claim (D-341's "
          "precedent: composition, not a split rule)")
RULE = ("An SST-2 item is contained in a message when its normalised text is a substring of the message's normalised "
        "text. 'substring': str.lower on both, Python's `in` (character level); 'token_aligned': the same on ' ' + "
        "text + ' ' (whole space-separated tokens) (D-338 (3), phase2_pool_clusters.CONTAINMENT_RULE). Messages: each "
        "consumed alpaca conversation's user and assistant message as train_lora_p2.messages_for builds them; role "
        "'either' is user or assistant. Dose: the checkpoint of step k has seen the first micro_bs x grad_accum x k "
        "consumed examples (used_row_indices order). Exact: candidates from phase2_pool_clusters.SubstringIndex, each "
        "verified with `in`.")
BIAS_TEMPLATE = (
    "UPPER bound on what was trained: messages counted untruncated (training cut each example to its first {seq_len} "
    "tokens: each record's n_truncated), and every consumed example counted, including those of an optimiser step "
    "GradScaler skipped after a gradient overflow (each skip halves the scale: training_records[].grad_scale_halvings, "
    "from each record's training.loss_curve grad_scale_after, with its nonfinite_microbatch_losses beside); short "
    "SST-2 items occur in ordinary English without a shared source (see short_item_reading_aid). LOWER bound on "
    "shared sources: SST-2's tokenised text is only lowercased, so a review sentence in ordinary spelling is not "
    "found; paraphrases are not measured. token_aligned is a LOWER bound here and NOT comparable to D-341's "
    "token_aligned column: on alpaca's raw text an occurrence next to punctuation, a quote or a newline has no space "
    "on that side and is missed, while D-341 compared SST-2 with SST-2, both tokenised. Composition only: nothing is "
    "excluded, no split changes.")
CHOICES_TEMPLATE = [
    "(a) text: each consumed conversation's user and assistant messages (train_lora_p2.messages_for, the trained "
    "code), not the chat-template render (D-344 (3) counts by turn)",
    "(b) examples: each record's used_row_indices, per seed and per dose (the first micro_bs x grad_accum x step "
    "consumed examples, steps {steps}) and over the union of the seeds (D-344 (3))",
    "(c) normalisation: D-338 (3)'s containment rule (str.lower, character-level `in`; token_aligned beside), exact "
    "through phase2_pool_clusters.SubstringIndex (D-340 (2))",
    "(d) roles: 'either' (user or assistant: D-365 (4)(vii)'s 'training text', the loss is on the full sequence), user "
    "and assistant separately as D-344 (3) reports turns; all three reported, none named primary (no D-entry names one "
    "for this count)",
    "(e) exact matches: SST-2 item_ids against every consumed user turn's and Phase 1 prompt's item_id must give 0, as "
    "the records state, else the run refuses; the records' frozen_hits_by_file for the SST-2 files copied beside",
    "(f) short items: D-344 (3)'s reading aid (8, 16, 32 characters), not a rule",
    "(g) alpaca only (D-365 (4)(vii)); HH and PKU are not counted",
    "(h) one record, results/phase2/sst2_containment_d365.json, written once",
    "(i) the hub directory is an argument and is not written (no host path); the snapshot is named relative to it",
]
GRADSCALER_INIT_SCALE = 65536.0     # torch.amp.GradScaler('cuda')'s default init_scale (train_lora_p2.train sets none)


def bias(seq_len: int) -> str:
    """BIAS_TEMPLATE with the sequence length the training records give."""
    return BIAS_TEMPLATE.replace("{seq_len}", str(seq_len))


def choices(steps: list[int]) -> list[str]:
    """CHOICES_TEMPLATE with the save steps the training records give."""
    return [c.replace("{steps}", ", ".join(str(k) for k in steps)) for c in CHOICES_TEMPLATE]


def grad_scale_halvings(curve: list[Any], where: str) -> int:
    """The optimiser steps whose GradScaler scale fell (a skipped step after a gradient overflow), from the loss
    curve's grad_scale_after, starting at GRADSCALER_INIT_SCALE."""
    prev, n = GRADSCALER_INIT_SCALE, 0
    for i, e in enumerate(curve):
        g = e.get("grad_scale_after") if isinstance(e, dict) else None
        if not isinstance(g, (int, float)) or isinstance(g, bool) or not g > 0:
            refuse(f"{where}: training.loss_curve[{i}] has no positive grad_scale_after")
        n += g < prev
        prev = g
    return n


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def code_files() -> list[str]:
    """This file plus every src module loaded (train_lora_p2.code_files' rule, D-161)."""
    src = Path(__file__).resolve().parent
    loaded = {f"src/{Path(f).name}" for f in (getattr(m, "__file__", None) for m in list(sys.modules.values()))
              if f and Path(f).suffix == ".py" and Path(f).resolve().parent == src}
    return sorted({OWN} | loaded)


def need(d: Any, path: str, kinds: tuple[type, ...], where: str) -> Any:
    """d's value at the dotted path, refused in words unless it exists and is one of kinds (a bool only where named)."""
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            refuse(f"{where} has no {path}")
        cur = cur[k]
    if not isinstance(cur, kinds) or (isinstance(cur, bool) and bool not in kinds):
        refuse(f"{where}: {path} is a {type(cur).__name__}, not {'/'.join(x.__name__ for x in kinds)}")
    return cur


def rel_to(p: Path, root: Path) -> str:
    return p.resolve().relative_to(root.resolve()).as_posix()


# --------------------------------------------------------------------------- the training records

def check_order(used: list[int], seed: int, n_rows: int, first10: list[int], max_skipped: int, where: str) -> int:
    """used is an in-order subsequence of default_rng(seed).permutation(n_rows) (train_lora_p2.select_examples'
    order) whose first 10 are first10; returns the rows skipped inside the consumed prefix (at most max_skipped)."""
    perm = np.random.default_rng(seed).permutation(n_rows)
    if [int(x) for x in perm[:10]] != list(first10):
        refuse(f"{where}: selection.order_first_10 is not default_rng({seed}).permutation({n_rows})[:10]")
    pos = np.empty(n_rows, dtype=np.int64)
    pos[perm] = np.arange(n_rows)
    p = pos[np.asarray(used, dtype=np.int64)]
    if len(p) and not bool(np.all(np.diff(p) > 0)):
        refuse(f"{where}: used_row_indices is not in the permutation's order")
    skipped = int(p[-1]) + 1 - len(used) if len(p) else 0
    if skipped > max_skipped:
        refuse(f"{where}: {skipped} rows skipped inside the consumed prefix, more than the {max_skipped} recorded "
               "exclusions and drops")
    return skipped


def extraction_records(results_dir: Path, root: Path, seed: int, rec_rel: str, rec_sha: str,
                       where: str) -> list[dict[str, Any]]:
    """The seed's SST-2 LoRA extraction records (D-386), each naming the training record, file and sha256, as its
    adapter_source.record; at least one."""
    found = sorted((results_dir / "phase2").glob(f"extract__lora__{T2.run_name(CORPUS, seed)}-step*__{AXIS}@*__*.json"))
    if not found:
        refuse(f"no SST-2 LoRA extraction record of {T2.run_name(CORPUS, seed)} under {results_dir / 'phase2'} (D-386)")
    out = []
    for p in found:
        e = read_result(p)
        src = (e.get("adapter_source") or {}) if isinstance(e, dict) else {}
        if src.get("record") != {"file": rec_rel, "sha256": rec_sha} or src.get("corpus") != CORPUS \
                or src.get("seed") != seed:
            refuse(f"{p.name} names adapter_source {json.dumps(src.get('record'))} ({src.get('corpus')!r} seed "
                   f"{src.get('seed')!r}), not {where} at {rec_sha[:12]}")
        out.append({"path": p, "file": PV._repo_rel(p), "sha256": sha256_file(p)})
    return out


def training_records(results_dir: Path, root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The three alpaca records, checked (module docstring, INPUTS); per seed what the count needs, and the shared
    corpus block (equal across the seeds)."""
    recs: list[dict[str, Any]] = []
    shared: dict[str, Any] | None = None
    for seed in SEEDS:
        p = results_dir / "phase2" / T2.record_name(CORPUS, seed)
        if not p.is_file():
            refuse(f"no training record {p.name} under {results_dir / 'phase2'}")
        d = read_result(p)
        w = p.name
        if not isinstance(d, dict) or d.get("run_kind") != "lora_train_p2":
            refuse(f"{w} is not a lora_train_p2 record")
        cfg = need(d, "config", (dict,), w)
        if cfg.get("corpus") != CORPUS or cfg.get("seed") != seed:
            refuse(f"{w} is corpus {cfg.get('corpus')!r} seed {cfg.get('seed')!r}, not {CORPUS} seed {seed}")
        mb = need(d, "config.micro_bs", (int,), w) * need(d, "config.grad_accum", (int,), w)
        save = need(d, "config.save_steps", (list,), w)
        if not save or any(not isinstance(s, int) or isinstance(s, bool) or s < 0 for s in save) or mb < 1 \
                or max(save) < 1:
            refuse(f"{w}: save_steps {save} or micro_bs x grad_accum {mb} is malformed")
        used = need(d, "used_row_indices", (list,), w)
        n_rows = need(d, "selection.n_rows", (int,), w)
        if any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < n_rows for i in used) \
                or len(set(used)) != len(used):
            refuse(f"{w}: used_row_indices holds a non-integer, out-of-range or repeated row")
        n_used = need(d, "n_examples_used", (int,), w)
        if not len(used) == n_used == max(save) * mb == need(d, "training.n_examples_consumed", (int,), w):
            refuse(f"{w}: {len(used)} used rows, n_examples_used {n_used}, {max(save)} x {mb} and the consumed count "
                   "are not all equal")
        cont = need(d, "contamination", (dict,), w)
        if cont.get("used_with_a_frozen_user_turn") != 0 or cont.get("used_with_a_frozen_phase1_prompt") != 0:
            refuse(f"{w}: the record counts used conversations with a frozen user turn or prompt ({cont})")
        skipped = check_order(used, seed, n_rows, need(d, "selection.order_first_10", (list,), w),
                              need(d, "selection.n_excluded", (int,), w) + need(d, "selection.n_dropped", (int,), w), w)
        code = need(d, "code.sha256", (dict,), w)
        frozen = {e.get("file"): e for e in need(d, "frozen_exclusion.files", (list,), w) if isinstance(e, dict)}
        hits = need(d, "selection.frozen_hits_by_file", (dict,), w)
        cs = need(d, "corpus_source", (dict,), w)
        corpus = {"dataset": cfg.get("dataset"), "dataset_revision": cfg.get("dataset_revision"),
                  "repo": cs.get("repo"), "revision": cs.get("revision"), "fetch_record": cs.get("fetch_record"),
                  "snapshot": cs.get("snapshot"), "snapshot_files_sha256": cs.get("snapshot_files_sha256"),
                  "layout": cs.get("layout"), "train_files_read": cs.get("train_files_read"),
                  "reader": cs.get("reader"), "phase1_record": {k: (d.get("phase1_record") or {}).get(k)
                                                                for k in ("file", "sha256")},
                  "n_rows": n_rows, "examples_per_step": mb, "save_steps": save,
                  "seq_len": need(d, "config.seq_len", (int,), w),
                  "training_code_sha256": {f: code.get(f) for f in TRAINING_CODE},
                  "message_rule": d.get("message_rule")}
        if shared is None:
            shared = corpus
        elif corpus != shared:
            diff = sorted(k for k in corpus if corpus[k] != shared[k])
            refuse(f"{w} differs from {T2.record_name(CORPUS, SEEDS[0])} in {diff}")
        curve = need(d, "training.loss_curve", (list,), w)
        if len(curve) != max(save):
            refuse(f"{w}: training.loss_curve holds {len(curve)} steps, not {max(save)}")
        sha = sha256_file(p)
        recs.append({"seed": seed, "path": p, "file": PV._repo_rel(p), "sha256": sha, "used": used,
                     "n_truncated": d.get("n_truncated"), "skipped_inside_prefix": skipped, "frozen": frozen,
                     "frozen_hits_by_file": hits, "grad_scale_halvings": grad_scale_halvings(curve, w),
                     "nonfinite_microbatch_losses": need(d, "training.nonfinite_microbatch_losses", (int,), w),
                     "extraction": extraction_records(results_dir, root, seed, rel_to(p, root), sha, w)})
    assert shared is not None
    rev = shared["revision"]
    if shared["dataset"] != T2.CORPORA[CORPUS]["repo"] or shared["repo"] != shared["dataset"] \
            or not isinstance(rev, str) or not re.fullmatch(r"[0-9a-f]{40}", rev) or shared["dataset_revision"] != rev:
        refuse(f"the records name {shared['dataset']} / {shared['repo']} at {rev!r}, not {T2.CORPORA[CORPUS]['repo']} "
               "at one full revision")
    for f, h in shared["training_code_sha256"].items():
        have = sha256_file(CODE_ROOT / f)
        if h != have:
            refuse(f"{f} is {have[:12]} here, not the {str(h)[:12]} the training records ran: messages_for, item_id "
                   "and phase1_prompt would not be the trained code")
    return recs, shared


# --------------------------------------------------------------------------- the corpus and the SST-2 splits

def read_corpus(shared: dict[str, Any], hub: Path, root: Path, results_dir: Path) -> tuple[list[Any], dict[str, Any]]:
    """The alpaca rows through the pinned fetch record and snapshot the records name (module docstring, INPUTS)."""
    rev = shared["revision"]
    fr = shared["fetch_record"] if isinstance(shared["fetch_record"], dict) else {}
    fpath = T2.fetch_record_path(results_dir, CORPUS, rev)
    if not fpath.is_file() or rel_to(fpath, root) != fr.get("file"):
        refuse(f"the fetch record {fr.get('file')} the training records name is not {rel_to(fpath, root)} on disk")
    frec = read_result(fpath)
    if sha256_file(fpath) != fr.get("sha256"):
        refuse(f"{fpath.name} is not the bytes the training records name ({str(fr.get('sha256'))[:12]})")
    p1 = shared["phase1_record"]
    expected = T2.check_fetch_record(frec, CORPUS, rev, str(p1["file"]), str(p1["sha256"]))
    org, name = T2.CORPORA[CORPUS]["repo"].split("/")
    snap_rel = f"datasets--{org}--{name}/snapshots/{rev}"
    if shared["snapshot"] != snap_rel:
        refuse(f"the training records name the snapshot {shared['snapshot']!r}, not {snap_rel!r}")
    snap = hub / snap_rel
    files = T2.verify_snapshot(snap, expected)
    if files != shared["snapshot_files_sha256"]:
        refuse("the snapshot's files are not the training records' snapshot_files_sha256")
    layout = T2.check_layout(CORPUS, files, (snap / "README.md").read_bytes() if "README.md" in files else None)
    if layout != shared["layout"]:
        refuse(f"the snapshot's layout {layout} is not the training records' {shared['layout']}")
    rows, read = T2.read_rows(CORPUS, snap, expected)
    if read != shared["train_files_read"] or shared["reader"] != T2.CORPORA[CORPUS]["reader"]:
        refuse(f"the train files read {read} are not the training records' {shared['train_files_read']}")
    if len(rows) != shared["n_rows"]:
        refuse(f"{len(rows)} rows read, the training records' selection.n_rows is {shared['n_rows']}")
    block = {"repo": T2.CORPORA[CORPUS]["repo"], "revision": rev,
             "fetch_record": {"file": rel_to(fpath, root), "sha256": fr["sha256"]}, "snapshot": snap_rel,
             "snapshot_files_sha256": files, "layout": layout, "train_files_read": read, "n_rows": len(rows),
             "offline": "read from local files only; no network, no datasets library (train_lora_p2.read_rows)"}
    return rows, block


def read_splits(recs: list[dict[str, Any]], root: Path, results_dir: Path) -> tuple[dict[str, list[dict[str, Any]]],
                                                                                    dict[str, Any]]:
    """The SST-2 split rows at the manifest's and every training record's frozen_exclusion sha256 (INPUTS)."""
    mpath = results_dir / "phase2" / MANIFEST
    if not mpath.is_file():
        refuse(f"no {MANIFEST} under {results_dir / 'phase2'} (D-341's split record)")
    man = read_result(mpath)
    if not isinstance(man, dict) or man.get("run_kind") != "phase2_freeze_splits" or man.get("axis") != AXIS:
        refuse(f"{MANIFEST} is not the p2sst phase2_freeze_splits record")
    have = sha256_file(CODE_ROOT / CONTAINMENT_CODE)
    want = need(man, "code.sha256", (dict,), MANIFEST).get(CONTAINMENT_CODE)
    if want != have:
        refuse(f"{CONTAINMENT_CODE} is {have[:12]} here, not the {str(want)[:12]} that computed D-341's containment")
    parts = need(man, f"axes.{AXIS}.partitions", (dict,), MANIFEST)
    out: dict[str, list[dict[str, Any]]] = {}
    block: dict[str, Any] = {}
    for part in SPLITS:
        e = parts.get(part)
        if not isinstance(e, dict) or not all(isinstance(e.get(k), int) for k in ("n", "n_pos", "n_neg")) \
                or not isinstance(e.get("file"), str) or not isinstance(e.get("sha256"), str):
            refuse(f"{MANIFEST} has no well-formed partition {part}")
        f = e["file"]
        if Path(f).is_absolute() or ".." in Path(f).parts or f != f"data/frozen_p2/{AXIS}__{part}.jsonl":
            refuse(f"{MANIFEST}: partition {part} names {f!r}, not data/frozen_p2/{AXIS}__{part}.jsonl")
        for r in recs:
            x = r["frozen"].get(f)
            if not isinstance(x, dict) or x.get("sha256") != e["sha256"] or x.get("n_items") != e["n"]:
                refuse(f"{r['file']}'s frozen_exclusion entry for {f} is {x}, not the manifest's sha256 and n")
        if not (root / f).is_file():
            refuse(f"{f} is not a file under {root}: the SST-2 split {MANIFEST} names (data/frozen_p2 is not in git; "
                   "the node holds it)")
        b = (root / f).read_bytes()
        if sha256_bytes(b) != e["sha256"]:
            refuse(f"{f} is not the bytes {MANIFEST} names")
        rows = [json.loads(line) for line in b.decode("utf-8").split("\n") if line.strip()]  # not a result (D-297)
        for i, row in enumerate(rows):
            t, iid, lab = row.get("text"), row.get("item_id"), row.get("label")
            if not isinstance(t, str) or not t or t != t.strip() or not isinstance(iid, str) \
                    or T2.item_id(t) != iid or row.get("axis") != AXIS or isinstance(lab, bool) or lab not in (0, 1):
                refuse(f"{f} row {i}: not a p2sst row with a stripped non-empty text, its sha256 item_id and label 0/1")
        n_pos = sum(1 for row in rows if row["label"] == 1)
        if len(rows) != e["n"] or n_pos != e["n_pos"] or len(rows) - n_pos != e["n_neg"]:
            refuse(f"{f}: {len(rows)} rows ({n_pos} positive), the manifest says {e['n']} ({e['n_pos']})")
        if len({row["item_id"] for row in rows}) != len(rows):
            refuse(f"{f}: item_ids repeat")
        out[part] = rows
        block[part] = {"file": f, "sha256": e["sha256"], "n": e["n"], "n_pos": e["n_pos"], "n_neg": e["n_neg"]}
    return out, {"manifest": {"file": PV._repo_rel(mpath), "sha256": sha256_file(mpath)}, "splits": block}


# --------------------------------------------------------------------------- the count

def first_positions(needles: list[str], index: Any, posarr: np.ndarray, norm: Any) -> np.ndarray:
    """(n_needles, n_seeds): per needle and seed, the smallest consumed position among the haystacks containing the
    needle's normalised text (NONE when none does). index is a phase2_pool_clusters.SubstringIndex over the normalised
    messages, the k-th the k-th distinct row's; posarr[s, k] that row's consumed position in seed s (NONE when seed s
    did not consume it)."""
    out = np.full((len(needles), posarr.shape[0]), NONE, dtype=np.int64)
    for j, t in enumerate(needles):
        hits = index.containing(norm(t))
        if hits:
            out[j] = posarr[:, hits].min(axis=1)
    return out


def by_label(mask: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    """D-338 (3)'s form."""
    return {"n": int(len(mask)), "contained": int(mask.sum()), "contained_pos": int((mask & (labels == 1)).sum()),
            "contained_neg": int((mask & (labels == 0)).sum())}


def count(splits: dict[str, list[dict[str, Any]]], messages: dict[int, list[dict[str, str]]],
          recs: list[dict[str, Any]], mb: int, steps: list[int]) -> dict[str, Any]:
    """The counts, the reading aid and the matched list (module docstring, THE COUNT)."""
    rows_u = sorted(messages)
    k_of = {r: k for k, r in enumerate(rows_u)}
    posarr = np.full((len(recs), len(rows_u)), NONE, dtype=np.int64)
    for s, r in enumerate(recs):
        for p, row in enumerate(r["used"]):
            posarr[s, k_of[row]] = p
    first: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for nn, norm in NORMS.items():
        first[nn] = {}
        for ri, role in enumerate(ROLES):
            index = PC.SubstringIndex([norm(messages[r][ri]["content"]) for r in rows_u], PC.CONTAINMENT_Q)
            first[nn][role] = {part: first_positions([x["text"] for x in rows], index, posarr, norm)
                               for part, rows in splits.items()}
            del index                                    # one index in memory at a time
        first[nn]["either"] = {part: np.minimum(first[nn]["user"][part], first[nn]["assistant"][part])
                               for part in splits}
    seeds = [f"s{r['seed']}" for r in recs]
    labels = {part: np.array([x["label"] for x in rows], dtype=np.int64) for part, rows in splits.items()}
    lens = {part: np.array([len(x["text"]) for x in rows], dtype=np.int64) for part, rows in splits.items()}
    counts: dict[str, Any] = {}
    for nn in NORMS:
        counts[nn] = {}
        for role in ROLE_COLUMNS:
            counts[nn][role] = {}
            for part in splits:
                F = first[nn][role][part]
                c = {sd: {f"step_{k}": by_label(F[:, s] < mb * k, labels[part]) for k in steps}
                     for s, sd in enumerate(seeds)}
                c["union_of_seeds"] = {f"step_{k}": by_label((F < mb * k).any(axis=1), labels[part]) for k in steps}
                counts[nn][role][part] = c
    top = max(steps)
    aid: dict[str, Any] = {}
    for nn in NORMS:
        aid[nn] = {}
        for part in splits:
            F = first[nn]["either"][part]
            cols = {**{sd: F[:, s] < mb * top for s, sd in enumerate(seeds)},
                    "union_of_seeds": (F < mb * top).any(axis=1)}
            aid[nn][part] = {}
            for n in SHORT_ITEM_CHARS:
                keep = lens[part] >= n
                aid[nn][part][f"items_of_at_least_{n}_chars"] = {sd: by_label(m[keep], labels[part][keep])
                                                                 for sd, m in cols.items()}
    matched = []
    for part, rows in splits.items():
        hit = (first["substring"]["either"][part] < mb * top).any(axis=1)
        for j in np.flatnonzero(hit):
            matched.append({
                "split": part, "item_id": rows[j]["item_id"], "label": rows[j]["label"],
                "n_chars": len(rows[j]["text"]),
                "first_position": {nn: {role: [None if v == NONE else int(v) for v in first[nn][role][part][j]]
                                        for role in ROLES} for nn in NORMS}})
    return {"counts": counts, "short_item_reading_aid": aid, "matched": matched, "seeds_order": seeds}


def exact_matches(splits: dict[str, list[dict[str, Any]]], messages: dict[int, list[dict[str, str]]],
                  rows: list[Any], recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Choice (e): the trainer's own exact rule on the SST-2 item_ids; refuses unless every count is 0."""
    out: dict[str, Any] = {}
    for r in recs:
        uid = [set(T2.user_turn_ids(messages[i])) for i in r["used"]]
        pid = [T2.item_id(T2.phase1_prompt(CORPUS, rows[i])) for i in r["used"]]
        out[f"s{r['seed']}"] = {}
        for part, srows in splits.items():
            ids = {x["item_id"] for x in srows}
            c = {"used_with_an_sst2_user_turn": sum(1 for u in uid if u & ids),
                 "used_with_an_sst2_phase1_prompt": sum(1 for p in pid if p in ids)}
            if any(c.values()):
                refuse(f"{r['file']}: {c} for {part}, but the training record counts 0 used conversations with a "
                       "frozen user turn or prompt: the rows read are not the ones trained")
            out[f"s{r['seed']}"][part] = c
    return out


# --------------------------------------------------------------------------- the run

def run(local_hub: str | Path, *, root: Path = ROOT, results_dir: Path | None = None,
        out: Path | None = None) -> Path:
    results_dir = Path(results_dir) if results_dir is not None else Path(root) / "results"
    final = RESULTS / "phase2" / FINAL_NAME
    out = Path(out) if out is not None else final
    if out.resolve() == final.resolve():
        if Path(root).resolve() != ROOT.resolve() or results_dir.resolve() != RESULTS.resolve():
            refuse("the final name reads only this repository's own records and splits")
    elif out.resolve().is_relative_to(RESULTS.resolve()):
        refuse("a test output (--out) must not be under results/")
    if out.exists():
        refuse(f"{out} exists; results are never overwritten (D-152 P5)")
    code0 = P.code_block(code_files())
    hub = Path(local_hub)
    recs, shared = training_records(results_dir, Path(root))
    splits, split_block = read_splits(recs, Path(root), results_dir)
    rows, corpus = read_corpus(shared, hub, Path(root), results_dir)
    messages: dict[int, list[dict[str, str]]] = {}
    for i in sorted({i for r in recs for i in r["used"]}):
        m, why = T2.messages_for(CORPUS, rows[i], i)
        if m is None or [x["role"] for x in m] != list(ROLES):
            refuse(f"alpaca row {i}: not one user and one assistant message ({why})")
        messages[i] = m
    exact = exact_matches(splits, messages, rows, recs)
    mb, steps = shared["examples_per_step"], [k for k in shared["save_steps"] if k > 0]
    print(f"{DECISION}: {len(splits)} SST-2 splits ({sum(len(v) for v in splits.values())} items) against "
          f"{len(messages)} distinct consumed alpaca conversations of {len(recs)} seeds; steps {steps} x {mb}",
          flush=True)
    body = count(splits, messages, recs, mb, steps)

    code1 = P.code_block(code_files())
    fpath = T2.fetch_record_path(results_dir, CORPUS, shared["revision"])
    bound = {r["file"]: (r["path"], r["sha256"]) for r in recs}       # every repository input, re-hashed below
    bound.update({x["file"]: (x["path"], x["sha256"]) for r in recs for x in r["extraction"]})
    bound[split_block["manifest"]["file"]] = (results_dir / "phase2" / MANIFEST, split_block["manifest"]["sha256"])
    bound[PV._repo_rel(fpath)] = (fpath, corpus["fetch_record"]["sha256"])
    bound.update({b["file"]: (Path(root) / b["file"], b["sha256"]) for b in split_block["splits"].values()})
    changed = sorted(k for k, (p, h) in bound.items() if not p.is_file() or sha256_file(p) != h)
    if code1["sha256"] != code0["sha256"] or changed:
        refuse(f"the code or an input changed during the run: {changed}")
    inputs = {k: h for k, (_, h) in bound.items()}
    config = {"decision": DECISION, "corpus": CORPUS, "seeds": list(SEEDS), "axis": AXIS, "splits": list(SPLITS),
              "steps": steps, "examples_per_step": mb, "normalisations": list(NORMS), "roles": list(ROLE_COLUMNS),
              "short_item_chars": list(SHORT_ITEM_CHARS), "substring_index_q": PC.CONTAINMENT_Q,
              "training_records": [r["file"] for r in recs], "manifest": split_block["manifest"]["file"]}
    payload = {
        **provenance_block(RUN_KIND), "decision": DECISION, "preregistration": PREREG,
        "rule": RULE, "d338_containment_rule": PC.CONTAINMENT_RULE, "bias": bias(shared["seq_len"]),
        "choices": choices(steps),
        "code": code0,
        "reused_code_sha256": {
            **{f: {"sha256": shared["training_code_sha256"][f], "equal_to": "every alpaca training record's code block"}
               for f in TRAINING_CODE},
            CONTAINMENT_CODE: {"sha256": sha256_file(CODE_ROOT / CONTAINMENT_CODE),
                               "equal_to": f"{MANIFEST}'s code block (D-341's containment)"}},
        "config": config, "config_hash": canonical_hash(config),
        "input_files_sha256": inputs,
        "training_records": [{"seed": r["seed"], "file": r["file"], "sha256": r["sha256"], "n_used": len(r["used"]),
                              "n_truncated": r["n_truncated"], "rows_skipped_inside_prefix": r["skipped_inside_prefix"],
                              "grad_scale_halvings": r["grad_scale_halvings"],
                              "nonfinite_microbatch_losses": r["nonfinite_microbatch_losses"],
                              "frozen_hits_by_file_sst2": {b["file"]: r["frozen_hits_by_file"].get(b["file"])
                                                           for b in split_block["splits"].values()},
                              "sst2_extraction_records": [{"file": x["file"], "sha256": x["sha256"]}
                                                          for x in r["extraction"]]}
                             for r in recs],
        "seq_len": shared["seq_len"],
        "message_rule": shared["message_rule"],
        "corpus": corpus, "sst2": split_block,
        "n_distinct_consumed_conversations": len(messages),
        "exact_item_id_matches": exact,
        **body,
        "n_matched_items": {part: sum(1 for m in body["matched"] if m["split"] == part) for part in SPLITS},
    }
    digest = write_results(out, payload, exclusive=True)
    c = body["counts"]["substring"]["either"]
    for part in SPLITS:                                  # printed only once the record is written
        u = c[part]["union_of_seeds"][f"step_{max(steps)}"]
        print(f"  {part}: {u['contained']} of {u['n']} items contained (substring, either role, union of seeds, "
              f"step {max(steps)}); {u['contained_pos']} positive, {u['contained_neg']} negative", flush=True)
    print(f"Written: {out}\nsha256:  {digest}")
    return out


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="D-365 (4)(vii): SST-2 items contained in the alpaca LoRA runs' training "
                                             "text, counted as composition")
    ap.add_argument("--local-hub", required=True, help="the hub directory src/fetch_dataset_snapshot.py --lora-corpus "
                                                       "alpaca wrote the snapshot under (not written into the record)")
    ap.add_argument("--out", default=None, help="TESTING: a JSON path outside results/")
    args = ap.parse_args()
    run(args.local_hub, out=Path(args.out) if args.out else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
