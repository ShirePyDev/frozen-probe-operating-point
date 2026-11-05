"""D-344 (3): Phase 1's own LoRA training rows, reconstructed, and counted against data/frozen by Human turn and dose.

What it answers. Phase 1's src/train_lora.py excluded a training example when the PROMPT it hashed
(train_lora.text_and_prompt(row, format)[1]: for hh-rlhf train_lora.first_human_turn of the transcript) was a
data/frozen item, but trained on the whole text (D-344 (1)). This read-only count rebuilds the exact examples each
Phase 1 record consumed and counts, per record and per dose (checkpoint), the conversations with a frozen item in a
user turn that Phase 1 trained on and did not check, by frozen split file, with the distinct items, the matches' turn
index and character offsets, and the share of 4a's populations involved. Writes results/lora_contamination_p1.json once
(exclusive create, provenance.write_results); refuses (exit nonzero, nothing written) on any reconstruction mismatch.
It computes no probe number; the author decides 4a's consequence from the record (D-344 (3)).

THE RECONSTRUCTION, step by step, with the train_lora.py lines it reproduces. Its data path is the one the three runs
used: 4a79841 added the hh format and --exclude-frozen (the alpaca run, at 0769d43's version, had neither, and the same
alpaca text, prompt and order), 4c7dbae added the pku format; every later change touches the Hugging Face guard, the
adapter path or the weight loading only (git log -p src/train_lora.py).
  (1) Rows. load_dataset(config.dataset, data_dir=config.data_dir, split='train') (lines 158-159) is replaced by
      train_lora_p2's pinned-snapshot readers (read_rows), at the revision the record holds, from the local snapshot
      whose every file has the sha256 its fetch record lists (results/phase2/fetch_dataset__<org>__<name>@<rev12>.json,
      fetch_dataset_snapshot.py --lora-corpus, pinned to this Phase 1 record at its sha256: train_lora_p2.
      check_fetch_record, verify_snapshot, check_layout).
  (2) Order. numpy.random.default_rng(config.seed).permutation(len(rows)) (lines 145, 161).
  (3) Exclusion, when config.exclude_frozen (lines 163-171): the rows whose text_and_prompt(row, format)[1] hashes
      (sha256, first 16 hex) to an item_id of train_lora.frozen_item_ids() are dropped, keeping the order. The alpaca
      record predates --exclude-frozen and --format: no exclusion, format 'alpaca'.
  (4) The first n_needed = max(save_steps) x micro_bs x grad_accum of the order; the overlap count (lines 172-175).
  (5) Consumption (lines 217-227): grad_accum micro-batches of micro_bs rows per step; the checkpoint of step k is
      saved after the k-th optimiser step (lines 254-256), so it has seen the first k x micro_bs x grad_accum = 16k
      consumed examples (a prefix of the order, no wrap: checked). Step 0 has seen none.
  train_lora.py's importable functions are called, not copied (text_and_prompt, and through it fmt and
  first_human_turn; frozen_item_ids, whose FROZEN must be <root>/data/frozen); the loop arithmetic of (2)-(5) lives
  inside train_lora.main and is reproduced here (phase1_order).

HOW THE RECONSTRUCTION IS CHECKED (any difference refuses):
  against each Phase 1 record: the recipe, dataset, data_dir and format (train_lora_p2.check_phase1_record);
      n_examples_used; contamination_overlap_with_frozen_items; n_excluded_for_frozen_overlap (hh, pku; absent from
      the alpaca record, whose run excluded nothing); data_format where present; the loss curve's steps 1..max; the
      checkpoints' steps. Not reconstructable: trainable_params, nonfinite_microbatch_losses (stated), the loss values,
      the adapter sha256s. The Phase 1 records hold no row count, so the rows read are checked against:
  results/training_data_overlap.json (hh, pku): n_rows, n_unique_prompts, overlap_by_axis, frozen_axis_sizes, with
      training_data_overlap.SOURCES' own prompt functions;
  results/paraphrase_overlap.json (all three): n_train_prompts and, per axis, n_frozen_prompts and
      exact_normalised_matches (paraphrase_overlap.normalise, hash_str);
  THE SELECTED SET (the seed, the permutation, the exclusion and the cut), against D-108's replay of the same
      selection: results/paraphrase_overlap_trained.json, per arm and axis, n_trained_prompts,
      exact_normalised_matches, frac_jaccard_ge_0.5, frac_jaccard_ge_0.8, frac_benign_jaccard_ge_0.5 and
      p95_max_jaccard, and results/paraphrase_per_item_safety.json, every safety item's max Jaccard per arm, replayed
      as paraphrase_overlap_trained.py computed them (MinHashReplay: a and b from default_rng(0), 3-shingle MinHash in
      16 LSH bands, the exact Jaccard over each frozen item's bucket candidates). p95 is compared within 1e-12 relative
      (np.quantile's interpolation arithmetic may differ by an ulp across numpy versions); every other value exactly.
  THE ORDER INSIDE THE PREFIX, which decides what steps 25 and 100 have seen, is checked by no recorded number: every
      recorded quantity above is a function of the set of n_needed rows (the record says so); the order is Phase 1's
      code path (default_rng(seed).permutation, then the cursor), reproduced, not verified.
  the frozen splits: every data/frozen/*.jsonl is a partition of a results/dataset_manifest*.json freeze record with
      that sha256 and item and label counts, and every data/frozen partition there is on disk; each freeze record's
      utc precedes every Phase 1 record's training START, its utc minus its elapsed_s (train_lora.py: utc is written
      after the loop, line 259; elapsed_s = time.time() - t0, line 268, t0 taken at the loop's start, line 219, after
      data/frozen was globbed, line 163, and the model loaded; the glob itself has no recorded time); a Phase 1 record
      without a finite elapsed_s >= 0 refuses; each file's index manifest (manifests/splits) names its sha256, size and
      item ids in order; every row's item_id is sha256(stripped text)[:16]; frozen_item_ids() equals the union read.
  The records compared against are superseded under D-161 (code provenance unverifiable) where src/superseded_
  provenance.py says so; each is used only as the written statement of counts, compared; the record states each one's
  status.

4a, from its own records: results/damage_d156_r2.json (the 4a operating points of record), results/depth_d156_<cell>.json
and results/intent_selected_d165.json, for the nine populations.LORA_ARMS cells. Each cell's post side must be the Phase 1
checkpoint of its dose (post_repo 'lora/<Phase 1 name>-s<step>', post_rev = that record's checkpoint sha256); the parts
<side>/<part> each record reads give data/frozen/<axis>__<part>.jsonl (extract_hidden.load_split). The shares are given
on 4a's populations, each with the sides it is read on: the pooled negatives (threshold_calib's and eval's label-0
items, from which damage_d156 redraws cal_d and E: its config.pooled_negatives, checked), the eval positives (fixed:
its config.positives), and probe_train (the pre side's probe fit; the post side only through depth_d156's post_refit).
Contamination matters on the post side only: the pre side is the base model, never trained.

THE COUNTS, per record and dose (each step > 0 of save_steps), over the examples seen by that checkpoint.
  turns, ONE definition for the three corpora: Phase 1's trained text, train_lora.text_and_prompt(row, format)[0]
      (the tokenizer appended EOS), split at '\\n\\nHuman:' / '\\n\\nAssistant:' (train_lora_p2.HH_MARK after the
      leading whitespace is dropped, as train_lora_p2.hh_turns splits; equal to hh_turns' turns wherever it accepts an
      hh transcript, checked), each turn stripped, with its character offsets in the text; every transcript counts,
      alternating or not, since Phase 1 trained on all. alpaca's text has no Human/Assistant labels, so its user
      message (the instruction joined to the input as train_lora.fmt joins them) and assistant message (the output)
      are turns too, at their offsets, beside any marked turn inside them; pku's text is 'Human: <prompt>\\n\\n
      Assistant: <reply>', so a marker inside its prompt or reply makes more turns.
  item_id: sha256 of the stripped turn text, first 16 hex (freeze_splits.py's rule).
  trained_unchecked_user_turn, THE D-344 COUNT (the headline): conversations with a user turn whose item_id differs
      from the item_id of the prompt Phase 1 checked and is an item of the file; the distinct items. A count of
      consumed examples (the first micro_bs x grad_accum x step = 16 x step examples; BIAS: an upper bound on what was
      trained, the 256-token truncation first).
  beside it: later_user_turn_by_position (user turns after the first by position: differs from the headline where
      first_human_turn and the marker split disagree), any_user_turn (every user turn and the checked prompt),
      assistant_turn (not part of the D-344 rule); and three BOUNDS that are not the D-344 count:
      normalised_user_turn_bound (a user turn whose paraphrase_overlap.normalise text equals a frozen item's, non-empty),
      contained_user_turn_bound (a frozen item's stripped text is a substring of a user turn; very short frozen texts
      inflate it) and their union, normalised_or_contained_user_turn_bound: the first two are incomparable (a normalised
      match need not contain the item's text, a containing turn need not normalise to it), so neither bounds the other
      and their union is the widest user-turn column (an upper bound on the headline and on
      later_user_turn_by_position; not on assistant_turn, which it does not scan, nor on any_user_turn's checked prompt
      where that prompt lies outside every user turn; not on paraphrases). Each per frozen split file (axis and split: safety = Aegis 2.0,
      safety_wildguard = WildGuardMix, over_refusal = XSTest) and over their union. Per-file counts overlap across
      datasets (an item_id can be in an Aegis and a WildGuardMix file): totals are the union, never a sum of files.
  headline_items, beside the headline: every frozen item the headline counts in some arm and dose, with its frozen
      file(s), its text length in characters and the conversations it matches per arm and dose; and, per arm and dose,
      headline_excluding_items_under_n_chars: the headline recomputed without the items under N characters, N in
      SHORT_ITEM_CHARS = (8, 16, 32), A READING AID for the short-item effect (generic very short items exactly match
      ordinary later turns), not a rule: the headline is unchanged.
  n_checked_prompt_not_first_user_turn, hh and pku only: the conversations whose first user turn (marker split) is not
      the prompt Phase 1 checked. alpaca's first user turn is the instruction joined to the input (train_lora.fmt), so
      it differs from the checked instruction whenever the input is non-empty: for alpaca
      n_instruction_is_prefix_of_first_user_turn is counted instead.
  matches: every exact and normalised user-turn match and every exact assistant-turn match in the consumed examples,
      with its position in the consumed order (the dose follows), row, turn index, the turn's [char_start, char_end) in
      the trained text and the text's length, so a tokenized recount needs no re-run; containment matches are counted,
      not listed. No prompt text is written: item ids, counts, offsets and shares only.

BIAS, stated in the record. An UPPER bound on what was trained: the count is of examples consumed, and Phase 1
truncated each text to its first 256 Mistral-7B-v0.1 tokens (lines 228-230), skipped the gradient of a micro-batch with
a non-finite loss (lines 236-238; the records hold the count), and GradScaler skips the optimiser update of a step
whose scaled gradients overflow (scaler.step, line 243; no count recorded). A LOWER bound on paraphrases: exact match.
The normalised and containment columns are incomparable; their union, normalised_or_contained_user_turn_bound, is the
widest user-turn column, an upper bound on the headline and on later_user_turn_by_position (not on assistant_turn,
which it does not scan, nor on any_user_turn's checked prompt where that prompt lies outside every user turn); it is
NOT a bound on paraphrases or near-duplicates, which no column here measures (D-108's MinHash Jaccard ran on the checked prompts, not on later
turns). Only data/frozen is counted, not data/frozen_p2.

  python3 src/lora_contamination_p1.py --local-hub <the NVMe hub fetch_dataset_snapshot.py --lora-corpus wrote>
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as P  # noqa: E402  (code_block, D-161)
import paraphrase_overlap as PO  # noqa: E402  (normalise, hash_str, shingles, signatures: the checks; normalise)
import populations as POP  # noqa: E402  (LORA_ARMS: the nine 4a cells)
import provenance as PV  # noqa: E402
import superseded_provenance as SP  # noqa: E402  (D-161: the status of the records compared against)
import train_lora as T1  # noqa: E402  (Phase 1: text_and_prompt, frozen_item_ids; the file is unchanged)
import train_lora_p2 as P2  # noqa: E402  (the pinned-snapshot readers, the fetch-record checks, messages, item_id)
import training_data_overlap as TDO  # noqa: E402  (SOURCES: its prompt functions, for the row-count check)
from provenance import ensure_utf8_stdout, provenance_block, sha256_bytes, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_NAME = "lora_contamination_p1.json"
DECISION = "D-344 (3)"
CODE_FILES = ["src/lora_contamination_p1.py", "src/train_lora.py", "src/train_lora_p2.py", "src/training_data_overlap.py",
              "src/paraphrase_overlap.py", "src/populations.py", "src/superseded_provenance.py", "src/provenance.py",
              "src/paired.py"]
CORPORA = ("alpaca", "hh", "pku")                            # train_lora_p2.CORPORA keys; each names its Phase 1 record
FROZEN_DIR = "data/frozen"
SPLIT_MANIFESTS = "manifests/splits"
CHECK_RECORDS = {"training_data_overlap": "training_data_overlap.json", "paraphrase_overlap": "paraphrase_overlap.json",
                 "paraphrase_overlap_trained": "paraphrase_overlap_trained.json"}
PER_ITEM = "paraphrase_per_item_safety.json"               # D-108/D-109: each safety item's max Jaccard, per arm
PER_ITEM_AXIS = "safety"                                   # the axis paraphrase_overlap_trained.py wrote it for
TDO_SOURCE = {"hh": "hh-rlhf-harmless", "pku": "pku-saferlhf"}         # training_data_overlap.SOURCES keys
PO_LABEL = {"alpaca": "alpaca (general)", "hh": "hh-rlhf harmless (safety)", "pku": "pku-saferlhf (safety)"}
# paraphrase_overlap.main's corpora (its lines 83-87; defined inside main, so not importable): the prompt per row
PO_PROMPT = {"alpaca": lambda r: r["instruction"], "hh": lambda r: PO.first_human(r["chosen"]),
             "pku": lambda r: r["prompt"]}
DAMAGE_4A = "damage_d156_r2.json"
DEPTH_4A = "depth_d156_{}.json"
INTENT_4A = "intent_selected_d165.json"
CACHE_PART = re.compile(r"(pre|post)/([a-z_]+)")
LORA_POST = re.compile(r"lora/(?P<name>[A-Za-z0-9._-]+?)-s(?P<step>[0-9]+)(?P<tag>__a100)?")
ALPACA_USER_AT = len("### Instruction:\n")                 # train_lora.fmt: the instruction starts here
CONTAIN_K = 8                                              # containment index: frozen texts by their first 8 characters
HEADLINE = "trained_unchecked_user_turn"
EXACT_RULES = ("trained_unchecked_user_turn", "later_user_turn_by_position", "any_user_turn", "assistant_turn")
BOUND_RULES = ("normalised_user_turn_bound", "contained_user_turn_bound", "normalised_or_contained_user_turn_bound")
RULES = EXACT_RULES + BOUND_RULES
USER_RULES = ("trained_unchecked_user_turn", "later_user_turn_by_position", "normalised_user_turn_bound",
              "contained_user_turn_bound", "normalised_or_contained_user_turn_bound")
SHORT_ITEM_CHARS = (8, 16, 32)                             # the reading aid's N: items under N characters left out
SHORT_ITEM_NOTE = ("A READING AID, NOT A RULE: the headline recomputed without the frozen items whose stripped text is "
                   "under N characters (Python str length), for a small fixed set of N; generic very short items exactly "
                   "match ordinary later turns, so this shows how much of the headline they make. The headline "
                   "(D-344's count) is unchanged and is the count of record")
ALPACA_PROMPT_NOTE = ("alpaca's first user turn is the instruction joined to the input (train_lora.fmt; train_lora_p2."
                      "messages_for), so it differs from the prompt Phase 1 checked (the stripped instruction) whenever "
                      "the input is non-empty: n_checked_prompt_not_first_user_turn is not reported for alpaca; "
                      "n_instruction_is_prefix_of_first_user_turn counts the conversations whose first user turn begins "
                      "with the checked instruction; by construction (train_lora_p2.messages_for builds the turn as the "
                      "stripped instruction plus the input, and trained_turns checks the offsets) it equals "
                      "n_conversations_seen, so it is a consistency check, not information")
RULE_TEXT = {
    "trained_unchecked_user_turn": "THE D-344 COUNT (headline): the conversations with a user turn of Phase 1's trained "
                                   "text whose item_id (sha256 of the stripped turn, first 16 hex) differs from the "
                                   "item_id of the prompt Phase 1 checked (train_lora.text_and_prompt(row, format)[1]) and "
                                   "is an item of the file; the distinct items those turns are",
    "later_user_turn_by_position": "the same over the user turns after the first by position (differs from the headline "
                                   "where train_lora.first_human_turn and the marker split disagree)",
    "any_user_turn": "the same over every user turn and the checked prompt itself",
    "assistant_turn": "the same over the assistant turns (reported; not part of the D-344 rule)",
    "normalised_user_turn_bound": "A BOUND, not the D-344 count: a user turn whose paraphrase_overlap.normalise text "
                                  "equals a frozen item's (non-empty; includes the exact matches). Incomparable with "
                                  "contained_user_turn_bound (a normalised match need not contain the item's stripped "
                                  "text): neither includes the other; their union, "
                                  "normalised_or_contained_user_turn_bound, is the widest user-turn column",
    "contained_user_turn_bound": "A BOUND, not the D-344 count: a frozen item's stripped text is a substring of a user "
                                 "turn (includes the exact matches; very short frozen texts inflate it). Incomparable "
                                 "with normalised_user_turn_bound (a containing turn need not normalise to the item): "
                                 "neither includes the other; their union, normalised_or_contained_user_turn_bound, is "
                                 "the widest user-turn column",
    "normalised_or_contained_user_turn_bound": "THE WIDEST USER-TURN COLUMN, not the D-344 count: the union of the two "
                                               "incomparable bounds, a user turn whose normalised text equals a frozen "
                                               "item's OR that contains a frozen item's stripped text (includes the "
                                               "exact matches; very short frozen texts inflate it); an upper bound on "
                                               "the headline and on later_user_turn_by_position; not on assistant_turn "
                                               "(it scans user turns only), nor on any_user_turn's checked prompt where "
                                               "that prompt lies outside every user turn; NOT a bound on paraphrases or "
                                               "near-duplicates, which no column here measures (D-108's MinHash "
                                               "Jaccard ran on the checked prompts, not on later turns)"}
BIAS = {
    "count_of": "examples CONSUMED by the training loop up to each checkpoint (train_lora.py lines 217-227)",
    "upper_bound_on_trained": [
        "Phase 1 truncated each training text (BOS + text + EOS) to its first 256 Mistral-7B-v0.1 tokens (train_lora.py "
        "lines 228-230): a matched turn beyond token 256 was consumed but not trained on; this module does not tokenize "
        "(the v0.1 tokenizer is not read); each match's char_start and char_end in the trained text are listed so a "
        "tokenized recount needs no re-run",
        "a micro-batch with a non-finite loss contributed no gradient (lines 236-238); each record's "
        "nonfinite_microbatch_losses is stated",
        "GradScaler skips the optimiser update of a step whose scaled gradients hold inf or NaN (scaler.step, line 243): "
        "those examples were consumed without an update; Phase 1 recorded no count of such steps"],
    "lower_bound_on_paraphrase": [
        "the headline is an exact match on the stripped turn text: paraphrases and edited copies are not caught (D-080); "
        "normalised_user_turn_bound and contained_user_turn_bound are incomparable (neither includes the other); their "
        "union, normalised_or_contained_user_turn_bound, is the widest user-turn column, an upper bound on the headline "
        "and on later_user_turn_by_position (not on assistant_turn, which it does not scan, nor on any_user_turn's "
        "checked prompt where that prompt lies outside every user turn), NOT a bound on paraphrases or near-duplicates: "
        "no column here measures those "
        "(D-108's MinHash Jaccard ran on the checked prompts, not on later turns); a later turn that is a near-duplicate "
        "of a frozen item without matching it exactly, normalised or as a substring, is in no column"],
    "scope": "only data/frozen (Phase 1's FROZEN, which holds the splits 4a reads); data/frozen_p2 is not counted"}
READING = [
    "per-file counts overlap across datasets: an item_id can be in an Aegis file and a WildGuardMix file; totals are "
    "union_of_frozen_files, never a sum over by_frozen_file",
    "doses are prefixes: what step 25 has seen, step 100 and step 400 have seen too",
    "matches: position is the index in the consumed order (seen by step k when position < k x micro_bs x grad_accum)"]
ORDER_NOTE = ("the order inside the consumed prefix, which decides what steps 25 and 100 have seen, is checked by no "
              "recorded number: every recorded quantity compared is a function of the set of the first n_needed rows; the "
              "order is Phase 1's code path (default_rng(seed).permutation and the cursor), reproduced, not verified")


def refuse(msg: str) -> None:
    P2.refuse(msg)


def item_id(text: str) -> str:
    return P2.item_id(text)


def _utc(s: Any, what: str) -> datetime:
    try:
        return datetime.fromisoformat(str(s))
    except ValueError:
        refuse(f"{what} has no ISO utc ({s!r})")
        raise


START_FIELD = "elapsed_s"
START_RULE = ("a Phase 1 record's training start = its utc minus its elapsed_s (train_lora.py: utc is written after the "
              "loop, line 259; elapsed_s = time.time() - t0, line 268, t0 at the loop's start, line 219, after "
              "data/frozen was globbed, line 163, and the model loaded; the glob itself has no recorded time)")


def phase1_start(rec: dict[str, Any], what: str) -> datetime:
    """START_RULE. A record without a finite number elapsed_s >= 0 refuses (no fall-back to the end time)."""
    e = rec.get(START_FIELD)
    if isinstance(e, bool) or not isinstance(e, (int, float)) or not math.isfinite(e) or e < 0:
        refuse(f"{what} holds no {START_FIELD} (a finite number of seconds >= 0, got {e!r}): its training start "
               f"(utc - {START_FIELD}), which the freeze-time check compares with, is unknown")
    return _utc(rec.get("utc"), what) - timedelta(seconds=float(e))


# --- frozen splits ---------------------------------------------------------------------------------------------------

def read_frozen(root: Path) -> dict[str, Any]:
    """Every data/frozen/*.jsonl, parsed from the bytes hashed: per file its item ids, labels and texts, in sorted file
    order (the order training_data_overlap and paraphrase_overlap grouped them by axis). A row whose item_id is not the
    sha256 of its text (first 16 hex), whose text is not stripped, or whose label is not 0 or 1 refuses."""
    d = Path(root) / FROZEN_DIR
    paths = sorted(d.glob("*.jsonl"))
    if not paths:
        refuse(f"{FROZEN_DIR} holds no *.jsonl file")
    files: dict[str, dict[str, Any]] = {}
    for p in paths:
        data = p.read_bytes()
        rel = f"{FROZEN_DIR}/{p.name}"
        try:
            rows = P2.XP.parse_jsonl_bytes(data)
        except (ValueError, UnicodeDecodeError) as e:
            refuse(f"{rel} does not parse as JSON lines ({type(e).__name__})")
        stem = p.name[:-len(".jsonl")]
        if stem.count("__") != 1:
            refuse(f"{rel} is not named <axis>__<split>.jsonl")
        axis, split = stem.split("__")
        ids, labels, texts = [], {}, []
        for k, r in enumerate(rows):
            if not isinstance(r, dict) or not isinstance(r.get("text"), str) or not isinstance(r.get("item_id"), str):
                refuse(f"{rel} row {k} has no string item_id and text")
            if r["text"] != r["text"].strip() or item_id(r["text"]) != r["item_id"]:
                refuse(f"{rel} row {k}: the item_id is not sha256(stripped text)[:16] (freeze_splits.py's rule)")
            if isinstance(r.get("label"), bool) or r.get("label") not in (0, 1):
                refuse(f"{rel} row {k}: label {r.get('label')!r} is not 0 or 1")
            if r["item_id"] in labels:
                refuse(f"{rel}: item_id {r['item_id']} twice")
            ids.append(r["item_id"])
            labels[r["item_id"]] = int(r["label"])
            texts.append(r["text"])
        files[rel] = {"file": rel, "sha256": sha256_bytes(data), "bytes": len(data), "axis": axis, "split": split,
                      "ids": set(ids), "id_list": ids, "labels": labels, "texts": texts,
                      "n_items": len(ids), "n_pos": sum(labels.values()), "n_neg": len(ids) - sum(labels.values())}
    return files


def check_frozen_manifests(root: Path, results_dir: Path, frozen: dict[str, Any],
                           p1_start: dict[str, datetime]) -> dict[str, Any]:
    """The freeze records (results/dataset_manifest*.json) list exactly the data/frozen files on disk with their
    sha256, item and label counts; each freeze record's utc precedes every Phase 1 run's training start (p1_start:
    START_RULE, utc - elapsed_s); each file's index manifest names its sha256, size and item ids. Returns the record's
    block."""
    parts: dict[str, dict[str, Any]] = {}
    freeze = []
    for mp in sorted(Path(results_dir).glob("dataset_manifest*.json")):
        doc = P2.read_json_result(mp, "the freeze record")
        rel = PV._repo_rel(mp)
        if not isinstance(doc, dict) or not isinstance(doc.get("axes"), dict):
            refuse(f"{mp.name} is not a freeze record (no axes)")
        utc = _utc(doc.get("utc"), mp.name)
        late = sorted(k for k, t in p1_start.items() if not utc < t)
        mine = []
        for axis, ax in doc["axes"].items():
            for split, pv in ((ax or {}).get("partitions") or {}).items():
                f = (pv or {}).get("file")
                if not isinstance(f, str) or not f.startswith(FROZEN_DIR + "/"):
                    continue
                if f in parts:
                    refuse(f"{f} is a partition of two freeze records")
                if late:
                    refuse(f"{mp.name} (utc {doc.get('utc')}) froze {f} at or after the training start (utc - "
                           f"{START_FIELD}) of the Phase 1 run(s) {late}: Phase 1's glob of data/frozen, before that "
                           "start, did not see it")
                parts[f] = {"sha256": pv.get("sha256"), "n": pv.get("n"), "n_pos": pv.get("n_pos"),
                            "n_neg": pv.get("n_neg"), "axis": axis, "split": split, "source": (ax or {}).get("source"),
                            "source_revision": (ax or {}).get("source_revision_sha"), "freeze_record": rel}
                mine.append(f)
        if mine:
            freeze.append({"file": rel, "sha256": PV._RESULT_READS[rel], "utc": doc.get("utc"),
                           "partitions": sorted(mine)})
    if sorted(parts) != sorted(frozen):
        refuse(f"the data/frozen files on disk and the freeze records' partitions differ: "
               f"{sorted(set(parts) ^ set(frozen))}")
    out_files, manifests = [], []
    for f, e in sorted(frozen.items()):
        m = parts[f]
        got = {"sha256": e["sha256"], "n": e["n_items"], "n_pos": e["n_pos"], "n_neg": e["n_neg"],
               "axis": e["axis"], "split": e["split"]}
        want = {k: m[k] for k in got}
        if got != want:
            refuse(f"{f} on disk {got} is not its freeze record's {want}")
        mpath = Path(root) / SPLIT_MANIFESTS / f"{e['axis']}__{e['split']}__manifest.json"
        if not mpath.is_file():
            refuse(f"no index manifest {SPLIT_MANIFESTS}/{mpath.name} for {f}")
        mb = mpath.read_bytes()
        try:
            man = json.loads(mb.decode("utf-8"))  # listed (D-297): its sha256 is in the record's split_manifests
        except (ValueError, UnicodeDecodeError) as exc:
            refuse(f"{mpath.name} is not valid JSON ({type(exc).__name__})")
        items = man.get("items") if isinstance(man, dict) else None
        if not isinstance(items, list) or man.get("sha256_of_original") != e["sha256"] or man.get("n") != e["n_items"] \
                or man.get("split_file") != Path(f).name or \
                [it.get("item_id") if isinstance(it, dict) else None for it in items] != e["id_list"]:
            refuse(f"{SPLIT_MANIFESTS}/{mpath.name} does not describe {f} (sha256, n, split_file, item ids in order)")
        mrel = f"{SPLIT_MANIFESTS}/{mpath.name}"
        manifests.append({"file": mrel, "sha256": sha256_bytes(mb), "split_file": f})
        out_files.append({"file": f, "sha256": e["sha256"], "bytes": e["bytes"], "axis": e["axis"], "split": e["split"],
                          "n_items": e["n_items"], "n_pos": e["n_pos"], "n_neg": e["n_neg"], "source": m["source"],
                          "source_revision": m["source_revision"], "freeze_record": m["freeze_record"],
                          "index_manifest": mrel})
    return {"files": out_files, "freeze_records": freeze, "split_manifests": manifests,
            "checks": ["every data/frozen/*.jsonl is a partition of a results/dataset_manifest*.json with that sha256, "
                       "n, n_pos and n_neg, and every data/frozen partition there is on disk",
                       f"each such freeze record's utc precedes every Phase 1 record's training start, utc - "
                       f"{START_FIELD} ({START_RULE})",
                       "each file's manifests/splits index manifest names its sha256, n, file name and item ids in order",
                       "every row's item_id = sha256(text)[:16] with the text stripped (freeze_splits.py's rule)"]}


# --- Phase 1's order -------------------------------------------------------------------------------------------------

def phase1_order(rows: list[dict[str, Any]], fmt: str, seed: int, exclude_frozen: bool, frozen_ids: set[str],
                 max_step: int, micro_bs: int, grad_accum: int) -> dict[str, Any]:
    """train_lora.main's data path (lines 145, 155-175, 217-227): the permutation, the exclusion, the overlap count
    over the first n_needed, and the rows consumed in order over max_step steps of grad_accum micro-batches of
    micro_bs rows. Refuses where Phase 1 would have aborted (fewer than n_needed rows after the exclusion)."""
    rng = np.random.default_rng(seed)
    order = [int(i) for i in rng.permutation(len(rows))]
    n_needed = max_step * micro_bs * grad_accum
    pids = []
    for i, r in enumerate(rows):
        try:
            pids.append(hashlib.sha256(T1.text_and_prompt(r, fmt)[1].encode("utf-8")).hexdigest()[:16])
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            refuse(f"row {i}: Phase 1's prompt function failed ({type(e).__name__}: {e})")
    first_10 = order[:10]
    n_excluded = 0
    if exclude_frozen:
        kept = [i for i in order if pids[i] not in frozen_ids]
        n_excluded = len(order) - len(kept)
        order = kept
    if len(order) < n_needed:
        refuse(f"need {n_needed} examples, only {len(order)} remain after the exclusion: Phase 1 would have aborted")
    overlap = sum(1 for i in order[:n_needed] if pids[i] in frozen_ids)
    consumed: list[int] = []
    cursor = 0
    for _step in range(1, max_step + 1):
        for _ in range(grad_accum):
            consumed.extend(order[(cursor + j) % len(order)] for j in range(micro_bs))
            cursor += micro_bs
    if consumed != order[:cursor] or len(set(consumed)) != len(consumed):
        refuse("the consumed rows are not a prefix of the order without repetition (a wrap)")
    return {"n_rows": len(rows), "order_first_10": first_10, "n_excluded": n_excluded, "n_eligible": len(order),
            "n_needed": n_needed, "overlap": overlap, "cursor": cursor, "n_examples_used": min(cursor, n_needed),
            "consumed": consumed, "used": order[:n_needed], "pids": pids}


def seen_by_step(consumed: list[int], step: int, micro_bs: int, grad_accum: int) -> list[int]:
    """The examples the checkpoint of `step` has seen: the first step x micro_bs x grad_accum consumed (lines 220-256)."""
    return consumed[:step * micro_bs * grad_accum]


# --- turns and matches -----------------------------------------------------------------------------------------------

def hh_segments(text: str) -> tuple[list[dict[str, Any]], bool]:
    """Every turn of a text under the Human/Assistant marker rule: split at train_lora_p2.HH_MARK after the leading
    whitespace is dropped, exactly as train_lora_p2.hh_turns splits a transcript, each turn stripped, with its [start,
    end) character offsets in `text`. Returns (turns, whether non-space text precedes the first marker)."""
    lead = len(text) - len(text.lstrip())
    s = "\n\n" + text[lead:]
    marks = list(P2.HH_MARK.finditer(s))
    turns = []
    for k, m in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(s)
        raw = s[m.end():end]
        content = raw.strip()
        start = m.end() + (len(raw) - len(raw.lstrip())) - 2 + lead
        turns.append({"role": "user" if m.group(1) == "Human" else "assistant", "content": content,
                      "start": start, "end": start + len(content)})
    return turns, bool(s[:marks[0].start()].strip()) if marks else bool(s.strip())


def trained_turns(corpus: str, row: Any, i: int, fmt: str) -> dict[str, Any]:
    """Phase 1's trained text (train_lora.text_and_prompt(row, format)[0]) and its turns under one definition: the
    marker rule (hh_segments) on that text; for alpaca, whose text has no Human/Assistant labels, its user message
    (train_lora_p2.messages_for: the instruction joined to the input as train_lora.fmt joins them) and assistant message
    (the output) as well, at their offsets. hh's marked turns equal train_lora_p2.hh_turns' wherever it accepts the
    transcript (checked). Every turn's offsets are checked against the text."""
    try:
        text, prompt = T1.text_and_prompt(row, fmt)
    except (KeyError, AttributeError, TypeError, ValueError) as e:
        refuse(f"{corpus} row {i}: Phase 1's text function failed ({type(e).__name__}: {e})")
    turns, prefix = hh_segments(text)
    if corpus == "hh":
        alt, why = P2.hh_turns(P2._str(row, "chosen", corpus, i))
        if alt is not None and [(t["role"], t["content"]) for t in alt] != [(t["role"], t["content"]) for t in turns]:
            refuse(f"hh row {i}: the turns differ from train_lora_p2.hh_turns' for an alternating transcript")
        status = "alternating" if alt is not None else str(why)
    else:
        if corpus == "alpaca":
            msgs, _why = P2.messages_for(corpus, row, i)
            user, asst = msgs[0]["content"], msgs[1]["content"]
            turns = sorted(turns + [{"role": "user", "content": user, "start": ALPACA_USER_AT,
                                     "end": ALPACA_USER_AT + len(user)},
                                    {"role": "assistant", "content": asst, "start": len(text) - len(asst),
                                     "end": len(text)}], key=lambda t: (t["start"], t["role"] != "user"))
            prefix = False                                      # the text before any marker is fmt's own layout
        status = "single_exchange" if [t["role"] for t in turns] == ["user", "assistant"] else "markers_inside"
    for t in turns:
        if text[t["start"]:t["end"]] != t["content"]:
            refuse(f"{corpus} row {i}: a turn's offsets do not give its text")
    first_user = next((t["content"] for t in turns if t["role"] == "user"), None)
    return {"text_chars": len(text), "turns": turns, "pid": item_id(prompt), "status": status,
            "text_before_first_turn": prefix,
            "checked_prompt_prefix_of_first_user_turn": first_user is not None and first_user.startswith(prompt)}


class FrozenIndex:
    """Lookups over every data/frozen item: item_id -> its files; item_id -> its stripped text's length in characters
    (Python str length; no text is kept for the record); normalised text (paraphrase_overlap.normalise, non-empty) ->
    item ids; stripped texts for containment (CONTAIN_K characters or more indexed by their first CONTAIN_K characters,
    shorter ones scanned)."""

    def __init__(self, frozen: dict[str, Any]):
        self.files_of: dict[str, list[str]] = collections.defaultdict(list)
        text_of: dict[str, str] = {}
        for f, e in sorted(frozen.items()):
            for x, t in zip(e["id_list"], e["texts"]):
                self.files_of[x].append(f)
                text_of[x] = t
        self.files_of = dict(self.files_of)
        self.chars_of: dict[str, int] = {x: len(t) for x, t in text_of.items()}
        self.norm: dict[str, set[str]] = collections.defaultdict(set)
        self.by_prefix: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
        self.short: list[tuple[str, str]] = []
        self.n_empty_normalised = 0
        for x, t in sorted(text_of.items()):
            n = PO.normalise(t)
            if n:
                self.norm[n].add(x)
            else:
                self.n_empty_normalised += 1
            if len(t) >= CONTAIN_K:
                self.by_prefix[t[:CONTAIN_K]].append((t, x))
            else:
                self.short.append((t, x))

    def normalised(self, s: str) -> set[str]:
        return set(self.norm.get(PO.normalise(s), ()))

    def contained(self, s: str) -> set[str]:
        out = {x for t, x in self.short if t in s}
        for i in range(len(s) - CONTAIN_K + 1):
            for t, x in self.by_prefix.get(s[i:i + CONTAIN_K], ()):
                if s.startswith(t, i):
                    out.add(x)
        return out


def conversation_matches(conv: dict[str, Any], fi: FrozenIndex, position: int, row: int
                         ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(the matched item ids per rule, with the conversation's turn summary; the listed matches with offsets)."""
    users = [t for t in conv["turns"] if t["role"] == "user"]
    assts = [t for t in conv["turns"] if t["role"] == "assistant"]
    uid = [item_id(t["content"]) for t in users]
    aid = [item_id(t["content"]) for t in assts]
    known, pid = fi.files_of, conv["pid"]
    norm = [fi.normalised(t["content"]) for t in users]
    m = {"trained_unchecked_user_turn": {x for x in uid if x != pid and x in known},
         "later_user_turn_by_position": {x for x in uid[1:] if x in known},
         "any_user_turn": {x for x in [*uid, pid] if x in known},
         "assistant_turn": {x for x in aid if x in known},
         "normalised_user_turn_bound": set().union(*norm, {x for x in uid if x in known}),
         "contained_user_turn_bound": set().union(*[fi.contained(t["content"]) for t in users])}
    m["normalised_or_contained_user_turn_bound"] = m["normalised_user_turn_bound"] | m["contained_user_turn_bound"]
    listed = []
    for k, t in enumerate(users):
        for x in sorted(norm[k] | ({uid[k]} if uid[k] in known else set())):
            listed.append({"position": position, "row": row, "text_chars": conv["text_chars"], "role": "user",
                           "turn_index": k, "char_start": t["start"], "char_end": t["end"], "item_id": x,
                           "exact": x == uid[k], "is_checked_prompt": uid[k] == pid, "files": known[x]})
    for k, t in enumerate(assts):
        if aid[k] in known:
            listed.append({"position": position, "row": row, "text_chars": conv["text_chars"], "role": "assistant",
                           "turn_index": k, "char_start": t["start"], "char_end": t["end"], "item_id": aid[k],
                           "exact": True, "is_checked_prompt": False, "files": known[aid[k]]})
    return {"m": m, "n_user": len(users), "status": conv["status"],
            "text_before_first_turn": conv["text_before_first_turn"],
            "checked_prompt_is_first_user_turn": bool(uid) and uid[0] == pid,
            "checked_prompt_prefix_of_first_user_turn": conv["checked_prompt_prefix_of_first_user_turn"]}, listed


def headline_without_short_items(entries: list[dict[str, Any]], chars_of: dict[str, int]) -> dict[str, Any]:
    """SHORT_ITEM_NOTE: for each N of SHORT_ITEM_CHARS, the headline's conversations and distinct items when the items
    whose stripped text is under N characters are left out, and how many distinct headline items that leaves out."""
    allh = set().union(*[en["m"][HEADLINE] for en in entries])
    by_n = []
    for n in SHORT_ITEM_CHARS:
        items: set[str] = set()
        conv = 0
        for en in entries:
            h = {x for x in en["m"][HEADLINE] if chars_of[x] >= n}
            if h:
                conv += 1
                items |= h
        by_n.append({"n_chars": n, "n_conversations": conv, "n_distinct_items": len(items),
                     "n_distinct_items_left_out": len(allh - items)})
    return {"reading_aid": SHORT_ITEM_NOTE, "rule": f"{HEADLINE} over the items of at least N characters", "by_n": by_n}


def headline_item_conversations(entries: list[dict[str, Any]]) -> collections.Counter:
    """Per frozen item, the conversations among `entries` whose headline matches hold it."""
    c: collections.Counter = collections.Counter()
    for en in entries:
        c.update(en["m"][HEADLINE])
    return c


def headline_items_table(per_dose: dict[str, dict[str, collections.Counter]], fi: FrozenIndex) -> dict[str, Any]:
    """Beside the headline: every frozen item the headline counts in some arm and dose, shortest text first, with its
    frozen file(s), its stripped text's length in characters and the conversations it matches per arm and dose (zero
    where none); the count of those items under each N of SHORT_ITEM_CHARS. Item ids and numbers only, no text."""
    ids = set().union(*[set(c) for doses in per_dose.values() for c in doses.values()])
    items = [{"item_id": x, "files": fi.files_of[x], "text_chars": fi.chars_of[x],
              "n_conversations": {corpus: {step: c.get(x, 0) for step, c in doses.items()}
                                  for corpus, doses in per_dose.items()}}
             for x in sorted(ids, key=lambda x: (fi.chars_of[x], x))]
    return {"rule": HEADLINE,
            "definition": "every frozen item the headline counts in some arm (Phase 1 corpus) and dose, sorted by text "
                          "length then item_id: its data/frozen file(s), its stripped text's length in characters "
                          "(Python str length) and, per arm and dose, the conversations among the consumed examples of "
                          "that dose (the first micro_bs x grad_accum x step) with a trained, unchecked user turn that "
                          "is this item. A conversation can hold two items, so the items' counts do not sum to the "
                          "headline",
            "n_items": len(items),
            "n_items_under_n_chars": [{"n_chars": n, "n_items": sum(1 for it in items if it["text_chars"] < n)}
                                      for n in SHORT_ITEM_CHARS],
            "items": items}


def count_dose(entries: list[dict[str, Any]], frozen: dict[str, Any], pops: dict[str, Any], *, corpus: str,
               chars_of: dict[str, int]) -> dict[str, Any]:
    """Per frozen file and over their union, for each rule: the conversations with a match and the distinct items; the
    shares of 4a's populations (user-turn rules); the headline without the short items (a reading aid); the checked
    prompt against the first user turn (hh, pku), or the instruction as its prefix (alpaca: ALPACA_PROMPT_NOTE)."""
    by_file: dict[str, Any] = {}
    for f, e in sorted(frozen.items()):
        blk = {}
        for rule in RULES:
            items: set[str] = set()
            n = 0
            for en in entries:
                h = en["m"][rule] & e["ids"]
                if h:
                    n += 1
                    items |= h
            blk[rule] = {"n_conversations": n, "n_distinct_items": len(items), "item_ids": sorted(items)}
        by_file[f] = blk
    allm = {rule: set().union(*[en["m"][rule] for en in entries]) for rule in RULES}
    union = {rule: {"n_conversations": sum(1 for en in entries if en["m"][rule]), "n_distinct_items": len(allm[rule])}
             for rule in RULES}
    shares = {}
    for name, p in pops.items():
        blk = {"n_items": len(p["ids"])}
        for rule in USER_RULES:
            k = len(allm[rule] & p["ids"])
            blk[rule] = {"n_distinct_items": k, "share": (k / len(p["ids"])) if p["ids"] else None}
        shares[name] = blk
    status = collections.Counter(en["status"] for en in entries)
    if corpus == "alpaca":
        checked = {"n_instruction_is_prefix_of_first_user_turn":
                   sum(1 for en in entries if en["checked_prompt_prefix_of_first_user_turn"]),
                   "checked_prompt_vs_first_user_turn": ALPACA_PROMPT_NOTE}
    else:
        checked = {"n_checked_prompt_not_first_user_turn": sum(1 for en in entries
                                                               if not en["checked_prompt_is_first_user_turn"])}
    return {"n_conversations_seen": len(entries),
            "n_with_more_than_one_user_turn": sum(1 for en in entries if en["n_user"] > 1),
            "turn_structure": dict(sorted(status.items())),
            "n_text_before_first_turn": sum(1 for en in entries if en["text_before_first_turn"]),
            **checked,
            "headline": {"rule": HEADLINE, **union[HEADLINE]},
            "headline_excluding_items_under_n_chars": headline_without_short_items(entries, chars_of),
            "by_frozen_file": by_file, "union_of_frozen_files": union, "populations_4a": shares}


# --- the splits 4a reads ---------------------------------------------------------------------------------------------

def splits_read_by_4a(results_dir: Path, p1: dict[str, dict[str, Any]], frozen_files: set[str]) -> dict[str, Any]:
    """From 4a's records: the nine LoRA cells of populations.LORA_ARMS in damage_d156_r2.json (scope 4a), their
    depth_d156 records and intent_selected_d165.json; each cell's post side must be the Phase 1 checkpoint of its dose
    (post_repo 'lora/<name>-s<step>', post_rev = that record's checkpoint sha256) in every record; one axis; the parts
    <side>/<part> each record reads, and the files data/frozen/<axis>__<part>.jsonl (extract_hidden.load_split)."""
    dpath = Path(results_dir) / DAMAGE_4A
    dmg = P2.read_json_result(dpath, "the 4a operating-point record")
    if not isinstance(dmg, dict):
        refuse(f"{DAMAGE_4A} is not a JSON object")
    cells = {c.get("cell"): c for c in dmg.get("cells") or [] if isinstance(c, dict) and c.get("scope") == "4a"}
    want = [(arm, stem) for arm, runs in POP.LORA_ARMS.items() for stem in runs]
    if sorted(cells) != sorted(s for _a, s in want):
        refuse(f"{DAMAGE_4A}'s 4a cells {sorted(cells)} are not populations.LORA_ARMS' {sorted(s for _a, s in want)}")
    keys = dmg.get("input_cache_sha256") if isinstance(dmg.get("input_cache_sha256"), dict) else {}
    ipath = Path(results_dir) / INTENT_4A
    intent = P2.read_json_result(ipath, "the 4a intent record")
    icfg = intent.get("config") if isinstance(intent, dict) and isinstance(intent.get("config"), dict) else {}
    ikeys = intent.get("input_cache_sha256") if isinstance(intent, dict) and \
        isinstance(intent.get("input_cache_sha256"), dict) else {}
    by_name = {rec["config"]["name"]: corpus for corpus, rec in p1.items()}
    records = [{"file": PV._repo_rel(p), "sha256": PV._RESULT_READS[PV._repo_rel(p)]} for p in (dpath, ipath)]
    reads: dict[str, dict[str, set[str]]] = {k: collections.defaultdict(set)
                                             for k in ("damage_d156_r2", "depth_d156", "intent_selected_d165")}
    out_cells, axes, arm_corpus = [], set(), {}

    def parts_of(ks: list[str], what: str) -> list[str]:
        if not ks or not all(isinstance(k, str) and CACHE_PART.fullmatch(k) for k in ks):
            refuse(f"{what} reads no '<side>/<part>' cache parts ({ks[:6]})")
        return sorted(ks)

    def add(kind: str, parts: list[str]) -> None:
        for p in parts:
            side, part = p.split("/")
            reads[kind][part].add(side)

    shared_intent = [k for k in ikeys if isinstance(k, str) and ":" not in k]
    for arm, stem in want:
        c = cells[stem]
        dparts = parts_of([k.split(":", 1)[1] for k in keys if k.startswith(stem + ":")], f"{DAMAGE_4A} cell {stem}")
        ppath = Path(results_dir) / DEPTH_4A.format(stem)
        dep = P2.read_json_result(ppath, "a 4a depth record")
        cfg = dep.get("config") if isinstance(dep, dict) and isinstance(dep.get("config"), dict) else {}
        ics = dep.get("input_cache_sha256") if isinstance(dep, dict) else None
        eparts = parts_of(sorted(ics) if isinstance(ics, dict) else [], ppath.name)
        iparts = parts_of(sorted(shared_intent + [k.split(":", 1)[1] for k in ikeys
                                                  if isinstance(k, str) and k.startswith(stem + ":")]),
                          f"{INTENT_4A} cell {stem}")
        m = LORA_POST.fullmatch(str(cfg.get("post_repo")))
        if not m or m["name"] not in by_name:
            refuse(f"{ppath.name}: post_repo {cfg.get('post_repo')!r} is not 'lora/<a Phase 1 record name>-s<step>'")
        corpus, step = by_name[m["name"]], int(m["step"])
        ck = {x.get("step"): x.get("sha256") for x in p1[corpus].get("checkpoints") or [] if isinstance(x, dict)}
        if c.get("dose") != step or not ck.get(step) or ck.get(step) != cfg.get("post_rev"):
            refuse(f"{stem}: dose {c.get('dose')!r}, post_repo step {step}, post_rev {str(cfg.get('post_rev'))[:12]}: not "
                   f"the {corpus} Phase 1 checkpoint of that step ({str(ck.get(step))[:12]})")
        mi = LORA_POST.fullmatch(str(icfg.get(f"{stem}:post_repo")))
        if not mi or mi["name"] != m["name"] or int(mi["step"]) != step or icfg.get(f"{stem}:post_rev") != ck[step]:
            refuse(f"{INTENT_4A}: {stem}'s post_repo/post_rev are not the {corpus} Phase 1 checkpoint of step {step}")
        if arm_corpus.setdefault(arm, corpus) != corpus:
            refuse(f"the 4a arm {arm!r} names two Phase 1 records")
        cell_axes = {c.get("axis"), cfg.get("axis"), icfg.get("axis")}
        if len(cell_axes) != 1 or not isinstance(c.get("axis"), str):
            refuse(f"{stem}: the axis differs between {DAMAGE_4A}, {ppath.name} and {INTENT_4A} "
                   f"({sorted(map(str, cell_axes))})")
        axes.add(c["axis"])
        add("damage_d156_r2", dparts)
        add("depth_d156", eparts)
        add("intent_selected_d165", iparts)
        records.append({"file": PV._repo_rel(ppath), "sha256": PV._RESULT_READS[PV._repo_rel(ppath)]})
        out_cells.append({"cell": stem, "arm": arm, "dose": step, "phase1_corpus": corpus, "axis": c["axis"],
                          "post_repo": cfg.get("post_repo"), "post_rev_is_phase1_checkpoint": True,
                          "damage_parts": dparts, "depth_parts": eparts, "intent_parts": iparts})
    if len(set(arm_corpus.values())) != len(arm_corpus):
        refuse(f"two 4a arms name one Phase 1 record: {arm_corpus}")
    if len(axes) != 1:
        refuse(f"the 4a cells read more than one axis: {sorted(axes)}")
    axis = axes.pop()
    rd = {k: {part: sorted(sides) for part, sides in sorted(v.items())} for k, v in reads.items()}
    files = sorted({f"{FROZEN_DIR}/{axis}__{part}.jsonl" for v in rd.values() for part in v})
    missing = sorted(set(files) - set(frozen_files))
    if missing:
        refuse(f"4a reads {missing}, which is not in data/frozen")
    dcfg = dmg.get("config") if isinstance(dmg.get("config"), dict) else {}
    return {"axis": axis, "files": files, "reads": rd, "arm_to_phase1_corpus": arm_corpus, "cells": out_cells,
            "records": records,
            "damage_population_config": {"pooled_negatives": (dcfg.get("pooled_negatives") or {}).get(axis),
                                         "positives": dcfg.get("positives")},
            "rule": "the nine populations.LORA_ARMS cells of damage_d156_r2.json (scope 4a), their depth_d156 records and "
                    "intent_selected_d165.json; each cell's post_repo/post_rev is its Phase 1 checkpoint at its dose in "
                    "every record; every cache part <side>/<part> they read is data/frozen/<axis>__<part>.jsonl "
                    "(extract_hidden.load_split)"}


def populations_4a(frozen: dict[str, Any], s4a: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """4a's populations on its axis, each with the sides each 4a record reads it on: the pooled negatives
    (threshold_calib's and eval's label-0 items: damage_d156's config.pooled_negatives, checked against the files), the
    eval positives (its config.positives), probe_train."""
    axis = s4a["axis"]

    def f(part: str) -> dict[str, Any]:
        e = frozen.get(f"{FROZEN_DIR}/{axis}__{part}.jsonl")
        if e is None:
            refuse(f"no {FROZEN_DIR}/{axis}__{part}.jsonl for 4a's populations")
        return e

    cal, ev, pt = f("threshold_calib"), f("eval"), f("probe_train")
    pc = s4a["damage_population_config"]
    if pc["pooled_negatives"] != {"n_calibration": cal["n_neg"], "n_evaluation": ev["n_neg"]} or \
            "eval positives" not in str(pc["positives"]):
        refuse(f"{DAMAGE_4A}'s config.pooled_negatives {pc['pooled_negatives']} / positives {pc['positives']!r} are not "
               f"threshold_calib's and eval's negatives ({cal['n_neg']}, {ev['n_neg']}) and the eval positives")

    def lab(e: dict[str, Any], y: int) -> set[str]:
        return {x for x in e["id_list"] if e["labels"][x] == y}

    def read_on(*parts: str) -> dict[str, dict[str, list[str]]]:
        return {kind: {p: r.get(p, []) for p in parts} for kind, r in s4a["reads"].items()}

    return {
        "pooled_negatives": {"ids": lab(cal, 0) | lab(ev, 0), "read_on": read_on("threshold_calib", "eval"),
                             "definition": "threshold_calib's and eval's label-0 items: the pool damage_d156 4a redraws "
                                           "cal_d and E from (its config.pooled_negatives)"},
        "eval_positives": {"ids": lab(ev, 1), "read_on": read_on("eval"),
                           "definition": "eval's label-1 items, fixed (damage_d156's config.positives)"},
        "probe_train": {"ids": set(pt["id_list"]), "read_on": read_on("probe_train"),
                        "definition": "the probe's training split: the pre side's probe fit (the base model) in every 4a "
                                      "record; the post side only through depth_d156's post_refit"}}


# --- the checks against the records that hold counts -----------------------------------------------------------------

def _chk(check: str, source: str, recorded: Any, reconstructed: Any) -> dict[str, Any]:
    return {"check": check, "source": source, "recorded": recorded, "reconstructed": reconstructed,
            "equal": recorded == reconstructed}


def check_phase1(rec: dict[str, Any], corpus: str, rel: str, recon: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = rec["config"]
    max_step = max(cfg["save_steps"])
    out = [_chk("n_examples_used", rel, rec.get("n_examples_used"), recon["n_examples_used"]),
           _chk("contamination_overlap_with_frozen_items", rel, rec.get("contamination_overlap_with_frozen_items"),
                recon["overlap"])]
    if "n_excluded_for_frozen_overlap" in rec:
        out.append(_chk("n_excluded_for_frozen_overlap", rel, rec["n_excluded_for_frozen_overlap"], recon["n_excluded"]))
    else:
        out.append(_chk("no exclusion (the record predates --exclude-frozen: no exclude_frozen, no "
                        "n_excluded_for_frozen_overlap)", rel, {"exclude_frozen": cfg.get("exclude_frozen", False),
                                                                "n_excluded": 0},
                        {"exclude_frozen": False, "n_excluded": recon["n_excluded"]}))
    if "data_format" in rec:
        out.append(_chk("data_format", rel, rec["data_format"], P2.CORPORA[corpus]["phase1_format"]))
    steps = [x.get("step") if isinstance(x, dict) else None for x in rec.get("loss_curve") or []]
    out.append(_chk("loss_curve steps (every step ran, so the cursor reached n_needed)", rel, steps,
                    list(range(1, max_step + 1))))
    out.append(_chk("checkpoint steps", rel, [x.get("step") if isinstance(x, dict) else None
                                              for x in rec.get("checkpoints") or []], list(cfg["save_steps"])))
    out.append(_chk("consumed examples (cursor)", "train_lora.py lines 217-227, 263", rec.get("n_examples_used"),
                    recon["cursor"]))
    return out


def axis_groups(frozen: dict[str, Any]) -> tuple[dict[str, set[str]], dict[str, list[str]], dict[str, list[tuple]]]:
    """By axis (the file name before '__'), in sorted file order: the item ids, the texts, the (item_id, text, label)
    items (the order and grouping training_data_overlap.py and paraphrase_overlap*.py used)."""
    items: dict[str, list[tuple]] = {}
    for f, e in sorted(frozen.items()):
        items.setdefault(e["axis"], []).extend((x, t, e["labels"][x]) for x, t in zip(e["id_list"], e["texts"]))
    return ({ax: {x for x, _t, _y in v} for ax, v in items.items()},
            {ax: [t for _x, t, _y in v] for ax, v in items.items()}, items)


def _identity(e: dict[str, Any], corpus: str, rev: str, src: str) -> None:
    spec = P2.CORPORA[corpus]
    got = (e.get("repo"), e.get("data_dir"), e.get("revision"))
    if got != (spec["repo"], spec["data_dir"], rev):
        refuse(f"{src} counts {got}, not {(spec['repo'], spec['data_dir'], rev)}: it checks nothing here")


def check_training_data_overlap(doc: Any, corpus: str, rows: list[dict[str, Any]], rev: str, rel: str,
                                axis_ids: dict[str, set[str]]) -> list[dict[str, Any]]:
    """training_data_overlap.py's counts (its SOURCES' prompt functions) over the rows read."""
    name = TDO_SOURCE[corpus]
    e = ((doc.get("sources") if isinstance(doc, dict) else None) or {}).get(name)
    if not isinstance(e, dict) or e.get("accessible") is not True:
        refuse(f"{rel} holds no accessible source {name!r}")
    _identity(e, corpus, rev, f"{rel} source {name}")
    prompts = [TDO.SOURCES[name]["extract"](r) for r in rows]
    ids = {hashlib.sha256(t.strip().encode("utf-8")).hexdigest()[:16] for t in prompts if t.strip()}
    return [_chk(f"{name}.n_rows (rows read)", rel, e.get("n_rows"), len(rows)),
            _chk(f"{name}.n_unique_prompts", rel, e.get("n_unique_prompts"), len(ids)),
            _chk(f"{name}.overlap_by_axis", rel, e.get("overlap_by_axis"),
                 {ax: len(ids & s) for ax, s in axis_ids.items()})]


def _po_axes(e: dict[str, Any], axis_texts: dict[str, list[str]], src: str) -> list[str]:
    axes = sorted(k for k, v in e.items() if isinstance(v, dict) and "exact_normalised_matches" in v)
    if axes != sorted(axis_texts):
        refuse(f"{src} counts the axes {axes}, not data/frozen's {sorted(axis_texts)}")
    return axes


def _exact_counts(prompts: list[str], e: dict[str, Any], axis_texts: dict[str, list[str]], src: str, rel: str
                  ) -> list[dict[str, Any]]:
    exact = {PO.hash_str(PO.normalise(t)) for t in prompts}
    out = []
    for ax in _po_axes(e, axis_texts, src):
        out.append(_chk(f"{src}.{ax}.n_frozen_prompts", rel, e[ax].get("n_frozen_prompts"), len(axis_texts[ax])))
        out.append(_chk(f"{src}.{ax}.exact_normalised_matches", rel, e[ax].get("exact_normalised_matches"),
                        sum(1 for t in axis_texts[ax] if PO.hash_str(PO.normalise(t)) in exact)))
    return out


def check_paraphrase_overlap(doc: Any, corpus: str, rows: list[dict[str, Any]], rev: str, rel: str,
                             axis_texts: dict[str, list[str]]) -> list[dict[str, Any]]:
    """paraphrase_overlap.py's corpus counts over the rows read: the rows with a non-empty prompt and the exact
    normalised matches per frozen axis."""
    label = PO_LABEL[corpus]
    e = ((doc.get("corpora") if isinstance(doc, dict) else None) or {}).get(label)
    if not isinstance(e, dict):
        refuse(f"{rel} holds no corpus {label!r}")
    _identity(e, corpus, rev, f"{rel} corpus {label}")
    texts = [t for t in (PO_PROMPT[corpus](r) for r in rows) if t and t.strip()]
    return [_chk(f"{label}.n_train_prompts (rows with a non-empty prompt)", rel, e.get("n_train_prompts"), len(texts)),
            *_exact_counts(texts, e, axis_texts, label, rel)]


class MinHashReplay:
    """paraphrase_overlap_trained.py's near-duplicate pass, replayed exactly (its lines 76-78 and 96-141): a and b from
    numpy.random.default_rng(0) (its --seed default), MinHash signatures of word 3-shingles (paraphrase_overlap.shingles,
    signatures: 64 permutations, 16 bands), LSH buckets of the consumed prompts, and for each frozen item of an axis the
    largest exact Jaccard over its bucket candidates. The frozen side is computed once."""

    def __init__(self, axis_items: dict[str, list[tuple]]):
        rng = np.random.default_rng(0)
        self.a = rng.integers(1, 1 << 31, size=PO.N_PERM, dtype=np.uint64)
        self.b = rng.integers(0, 1 << 31, size=PO.N_PERM, dtype=np.uint64)
        self.rpb = PO.N_PERM // PO.N_BANDS
        self.frozen = {}
        for ax, items in axis_items.items():
            f_sh = [PO.shingles(t) for _x, t, _y in items]
            self.frozen[ax] = (f_sh, PO.signatures(f_sh, self.a, self.b))

    def best(self, prompts: list[str]) -> dict[str, np.ndarray]:
        rpb = self.rpb
        c_sh = [PO.shingles(t) for t in prompts]
        c_sig = PO.signatures(c_sh, self.a, self.b)
        buckets: list[dict[bytes, list[int]]] = [collections.defaultdict(list) for _ in range(PO.N_BANDS)]
        for i in range(len(c_sh)):
            for bi in range(PO.N_BANDS):
                buckets[bi][c_sig[i, bi * rpb:(bi + 1) * rpb].tobytes()].append(i)
        out = {}
        for ax, (f_sh, f_sig) in self.frozen.items():
            best = np.zeros(len(f_sh))
            for j, sh in enumerate(f_sh):
                cand: set[int] = set()
                for bi in range(PO.N_BANDS):
                    cand.update(buckets[bi].get(f_sig[j, bi * rpb:(bi + 1) * rpb].tobytes(), ()))
                for i in cand:
                    inter = len(sh & c_sh[i])
                    if inter:
                        sim = inter / len(sh | c_sh[i])
                        if sim > best[j]:
                            best[j] = sim
            out[ax] = best
        return out


def jaccard_fields(best: np.ndarray, items: list[tuple]) -> dict[str, Any]:
    """paraphrase_overlap_trained.py lines 127-134."""
    neg = np.array([not y for _x, _t, y in items])
    return {"frac_jaccard_ge_0.5": float((best >= 0.5).mean()), "frac_jaccard_ge_0.8": float((best >= 0.8).mean()),
            "frac_benign_jaccard_ge_0.5": float((best[neg] >= 0.5).mean()) if neg.any() else None,
            "p95_max_jaccard": float(np.quantile(best, 0.95))}


def check_paraphrase_overlap_trained(doc: Any, per_item_doc: Any, rec: dict[str, Any], rows: list[dict[str, Any]],
                                     fmt: str, used: list[int], rel: str, per_item_rel: str,
                                     axis_texts: dict[str, list[str]], axis_items: dict[str, list[tuple]],
                                     mh: MinHashReplay) -> list[dict[str, Any]]:
    """D-108's replay of the same selection, the check of WHICH rows were selected (seed, order, exclusion, cut): the
    n_needed prompts consumed; per axis their exact normalised matches and the MinHash fields; every safety item's max
    Jaccard (paraphrase_per_item_safety.json)."""
    name = rec["config"]["name"]
    e = ((doc.get("arms") if isinstance(doc, dict) else None) or {}).get(name)
    if not isinstance(e, dict):
        refuse(f"{rel} holds no arm {name!r}")
    if e.get("dataset") != rec["config"]["dataset"]:
        refuse(f"{rel} arm {name} names dataset {e.get('dataset')!r}, not {rec['config']['dataset']!r}")
    prompts = [T1.text_and_prompt(rows[i], fmt)[1] for i in used]
    out = [_chk(f"{name}.n_trained_prompts", rel, e.get("n_trained_prompts"), len(prompts)),
           _chk(f"{name}.n_examples_used_recorded", rel, e.get("n_examples_used_recorded"), rec.get("n_examples_used")),
           *_exact_counts(prompts, e, axis_texts, name, rel)]
    best = mh.best(prompts)
    for ax in _po_axes(e, axis_texts, name):
        for k, v in jaccard_fields(best[ax], axis_items[ax]).items():
            r = e[ax].get(k)
            if k == "p95_max_jaccard":
                ok = isinstance(r, (int, float)) and math.isclose(r, v, rel_tol=1e-12, abs_tol=0.0)
                out.append({**_chk(f"{name}.{ax}.{k} (within 1e-12 relative)", rel, r, v), "equal": ok})
            else:
                out.append(_chk(f"{name}.{ax}.{k}", rel, r, v))
    if PER_ITEM_AXIS not in axis_items:
        refuse(f"data/frozen has no {PER_ITEM_AXIS!r} axis, the axis {PER_ITEM} holds")
    mine = {x: float(best[PER_ITEM_AXIS][j]) for j, (x, _t, _y) in enumerate(axis_items[PER_ITEM_AXIS])}
    theirs = (per_item_doc.get(name) if isinstance(per_item_doc, dict) else None)
    if not isinstance(theirs, dict):
        refuse(f"{per_item_rel} holds no arm {name!r}")
    n_diff = sum(1 for x in set(mine) | set(theirs) if mine.get(x) != theirs.get(x))
    out.append(_chk(f"{name}: every {PER_ITEM_AXIS} item's max Jaccard", per_item_rel,
                    {"n_items": len(theirs), "n_differing": 0}, {"n_items": len(mine), "n_differing": n_diff}))
    return out


def _refuse_mismatches(checks: list[dict[str, Any]], what: str) -> None:
    bad = [c for c in checks if not c["equal"]]
    if bad:
        refuse(f"the reconstruction of {what} does not reproduce: " + "; ".join(
            f"{c['check']} ({c['source']}): recorded {json.dumps(c['recorded'])[:80]}, reconstructed "
            f"{json.dumps(c['reconstructed'])[:80]}" for c in bad[:8]))


# --- the run ---------------------------------------------------------------------------------------------------------

def code_files() -> list[str]:
    """CODE_FILES plus every src module loaded (D-161; train_lora_p2.code_files' rule)."""
    src = Path(__file__).resolve().parent
    loaded = {f"src/{Path(f).name}" for f in (getattr(m, "__file__", None) for m in list(sys.modules.values()))
              if f and Path(f).suffix == ".py" and Path(f).resolve().parent == src}
    return sorted(set(CODE_FILES) | loaded)


def _superseded(rel: str) -> dict[str, Any]:
    stem = Path(rel).stem
    return {"superseded": stem in SP.CODE_PROVENANCE_UNVERIFIABLE, "reason": SP.CODE_PROVENANCE_UNVERIFIABLE.get(stem),
            "use_here": "the written statement of counts, each compared with the reconstruction; no result of it is used"}


def run(local_hub: str | Path, *, root: Path = ROOT, results_dir: Path = RESULTS,
        allow_no_commit: bool = False) -> tuple[Path, dict[str, Any]]:
    code0 = P.code_block(code_files(), allow_no_commit=allow_no_commit)
    root, results_dir = Path(root), Path(results_dir)
    out = results_dir / OUT_NAME
    if os.path.lexists(out):
        refuse(f"{OUT_NAME} exists; results are never overwritten")
    if Path(T1.FROZEN).resolve() != (root / FROZEN_DIR).resolve():
        refuse(f"train_lora.FROZEN ({T1.FROZEN}) is not {root / FROZEN_DIR}: Phase 1's own frozen_item_ids would read "
               "another directory than the one hashed here")
    hub = Path(local_hub)

    # Phase 1 records
    p1: dict[str, dict[str, Any]] = {}
    p1_block: dict[str, dict[str, Any]] = {}
    p1_start: dict[str, datetime] = {}                       # START_RULE: utc - elapsed_s
    for corpus in CORPORA:
        path = results_dir / P2.CORPORA[corpus]["phase1_record"]
        rec = P2.read_json_result(path, "the Phase 1 record")
        rel = PV._repo_rel(path)
        chk = P2.check_phase1_record(rec, corpus)
        cfg = rec["config"]
        if isinstance(cfg.get("seed"), bool) or not isinstance(cfg.get("seed"), int):
            refuse(f"{rel} holds no integer config.seed")
        if not isinstance(chk["exclude_frozen"], bool) or not isinstance(cfg.get("name"), str):
            refuse(f"{rel}: config.exclude_frozen {chk['exclude_frozen']!r}, config.name {cfg.get('name')!r}")
        p1[corpus] = rec
        start = p1_start[corpus] = phase1_start(rec, rel)
        p1_block[corpus] = {"file": rel, "sha256": PV._RESULT_READS[rel], "name": cfg["name"], "seed": cfg["seed"],
                            "dataset": cfg["dataset"], "data_dir": cfg.get("data_dir"),
                            "format": P2.CORPORA[corpus]["phase1_format"], "exclude_frozen": chk["exclude_frozen"],
                            "dataset_revision": chk["dataset_revision"], "utc": rec.get("utc"),
                            START_FIELD: rec.get(START_FIELD), "training_start_utc": start.isoformat(),
                            "training_start_rule": START_RULE,
                            "nonfinite_microbatch_losses": rec.get("nonfinite_microbatch_losses"),
                            "superseded_under_d161": _superseded(rel)}
    per_step = sorted({p1[c]["config"]["micro_bs"] * p1[c]["config"]["grad_accum"] for c in CORPORA})
    first_n = (f"the first {per_step[0]} x step examples" if len(per_step) == 1
               else "the first micro_bs x grad_accum x step examples")

    # frozen splits: bytes, freeze records, index manifests, Phase 1's own reader
    frozen = read_frozen(root)
    frozen_block = check_frozen_manifests(root, results_dir, frozen, p1_start)
    union = set().union(*[e["ids"] for e in frozen.values()])
    if T1.frozen_item_ids() != union:
        refuse("train_lora.frozen_item_ids() differs from the union of the data/frozen files read here")
    axis_ids, axis_texts, axis_items = axis_groups(frozen)
    s4a = splits_read_by_4a(results_dir, p1, set(frozen))
    pops = populations_4a(frozen, s4a)
    fi = FrozenIndex(frozen)

    # the records holding counts
    checks_docs = {}
    for key, name in {**CHECK_RECORDS, "per_item": PER_ITEM}.items():
        cp = results_dir / name
        checks_docs[key] = (P2.read_json_result(cp, "a check record"), PV._repo_rel(cp))
    tdo, tdo_rel = checks_docs["training_data_overlap"]
    po, po_rel = checks_docs["paraphrase_overlap"]
    pot, pot_rel = checks_docs["paraphrase_overlap_trained"]
    pid_doc, pid_rel = checks_docs["per_item"]
    sizes_chk = [_chk("frozen_axis_sizes", tdo_rel, tdo.get("frozen_axis_sizes") if isinstance(tdo, dict) else None,
                      {ax: len(s) for ax, s in axis_ids.items()})]
    _refuse_mismatches(sizes_chk, "the frozen axes")
    mh = MinHashReplay(axis_items)

    recon_block: dict[str, Any] = {}
    counts: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    item_conv: dict[str, dict[str, collections.Counter]] = {}
    for corpus in CORPORA:
        rec, spec = p1[corpus], P2.CORPORA[corpus]
        cfg, b = rec["config"], p1_block[corpus]
        rev = b["dataset_revision"]
        fpath = P2.fetch_record_path(results_dir, corpus, rev)
        if not fpath.is_file():
            refuse(f"no fetch record {fpath.name}; queues/laneC_20261003c_lora_corpora.tsv writes it")
        frec = P2.read_json_result(fpath, "the fetch record")
        expected = P2.check_fetch_record(frec, corpus, rev, b["file"], b["sha256"])
        org, rname = spec["repo"].split("/")
        snap_rel = f"datasets--{org}--{rname}/snapshots/{rev}"
        snap = hub / snap_rel
        files = P2.verify_snapshot(snap, expected)
        layout = P2.check_layout(corpus, files, (snap / "README.md").read_bytes() if "README.md" in files else None)
        rows, read = P2.read_rows(corpus, snap, expected)
        fmt = spec["phase1_format"]
        recon = phase1_order(rows, fmt, cfg["seed"], b["exclude_frozen"], union, max(cfg["save_steps"]),
                             cfg["micro_bs"], cfg["grad_accum"])
        checks = check_phase1(rec, corpus, b["file"], recon)
        if corpus in TDO_SOURCE:
            checks += check_training_data_overlap(tdo, corpus, rows, rev, tdo_rel, axis_ids)
        checks += check_paraphrase_overlap(po, corpus, rows, rev, po_rel, axis_texts)
        checks += check_paraphrase_overlap_trained(pot, pid_doc, rec, rows, fmt, recon["used"], pot_rel, pid_rel,
                                                   axis_texts, axis_items, mh)
        _refuse_mismatches(checks, f"{b['file']} ({corpus})")

        entries, listed = [], []
        for pos, i in enumerate(recon["consumed"]):
            conv = trained_turns(corpus, rows[i], i, fmt)
            if conv["pid"] != recon["pids"][i]:
                refuse(f"{corpus} row {i}: the checked prompt's item_id differs between the two Phase 1 code paths")
            en, ls = conversation_matches(conv, fi, pos, i)
            entries.append(en)
            listed.extend(ls)
        doses = {}
        item_conv[corpus] = {}
        for step in cfg["save_steps"]:
            if step <= 0:
                continue
            seen = seen_by_step(recon["consumed"], step, cfg["micro_bs"], cfg["grad_accum"])
            doses[str(step)] = {"step": step, "n_examples_seen": len(seen),
                                "rule": f"the first {step} x {cfg['micro_bs']} x {cfg['grad_accum']} examples consumed",
                                **count_dose(entries[:len(seen)], frozen, pops, corpus=corpus, chars_of=fi.chars_of)}
            item_conv[corpus][str(step)] = headline_item_conversations(entries[:len(seen)])
        counts[corpus] = {"phase1_record": b["file"], "name": b["name"], "doses": doses,
                          "matches": listed, "n_matches_listed": len(listed)}
        recon_block[corpus] = {"phase1_record": b["file"], "seed": cfg["seed"], "format": fmt,
                               "exclude_frozen": b["exclude_frozen"], "n_rows": recon["n_rows"],
                               "order_first_10": recon["order_first_10"], "n_excluded": recon["n_excluded"],
                               "n_eligible": recon["n_eligible"], "n_needed": recon["n_needed"],
                               "overlap": recon["overlap"], "n_examples_consumed": recon["cursor"],
                               "numpy": np.__version__, "checks": checks, "all_equal": True,
                               "order_inside_prefix": ORDER_NOTE,
                               "not_reconstructable": ["trainable_params", "nonfinite_microbatch_losses", "loss values",
                                                       "elapsed_s", "checkpoint sha256s"]}
        sources[corpus] = {"repo": spec["repo"], "revision": rev, "data_dir": spec["data_dir"],
                           "fetch_record": {"file": PV._repo_rel(fpath), "sha256": PV._RESULT_READS[PV._repo_rel(fpath)]},
                           "snapshot": snap_rel, "snapshot_files_sha256": files, "layout": layout,
                           "train_files_read": read, "reader": spec["reader"],
                           "offline": "read from local files only; no network, no datasets library"}
        del rows, entries

    record: dict[str, Any] = {
        **provenance_block("lora_contamination_p1"),
        "code": code0,
        "decision": DECISION,
        "preregistration": "D-344 (3): a reviewed count on Phase 1's own training rows (reconstructed with train_lora.py's "
                           "functions, each record's seed and corpus revision): per arm and dose, the conversations with "
                           "any later Human turn whose item_id is a data/frozen item, by split, and the distinct frozen "
                           "items involved; the author decides 4a's consequence from it",
        "config": {"records": [p1_block[c]["file"] for c in CORPORA], "frozen_dir": FROZEN_DIR,
                   "doses": "every step > 0 of each record's save_steps", "item_id": "sha256 of the UTF-8 of the stripped "
                   "text, first 16 hex (freeze_splits.py)", "local_hub": "not recorded (a host path); each snapshot is "
                   "named relative to it", "containment_index_prefix_chars": CONTAIN_K},
        "headline": {"rule": HEADLINE, "definition": RULE_TEXT[HEADLINE],
                     "count_of": f"count of consumed examples ({first_n}); see bias.upper_bound_on_trained",
                     "where": "counts.<corpus>.doses.<step>.headline (union over data/frozen); by_frozen_file per split",
                     "per_item": "headline_items: each item the headline counts, its file(s), its text length in "
                                 "characters and its conversations per arm and dose",
                     "short_items": "counts.<corpus>.doses.<step>.headline_excluding_items_under_n_chars, N in "
                                    f"{list(SHORT_ITEM_CHARS)}: a reading aid, not a rule; the headline is unchanged"},
        "headline_items": headline_items_table(item_conv, fi),
        "rules": RULE_TEXT,
        "bias": BIAS,
        "reading": READING,
        "phase1_records": p1_block,
        "corpus_source": sources,
        "frozen_splits": {**frozen_block, "n_items_with_empty_normalised_text": fi.n_empty_normalised,
                          "n_distinct_items_under_n_chars": [
                              {"n_chars": n, "n_items": sum(1 for c in fi.chars_of.values() if c < n)}
                              for n in SHORT_ITEM_CHARS]},
        "splits_4a": s4a,
        "populations_4a": {k: {kk: vv for kk, vv in v.items() if kk != "ids"} | {"n_items": len(v["ids"])}
                           for k, v in pops.items()},
        "check_records": [{"file": rel, "sha256": PV._RESULT_READS[rel], "superseded_under_d161": _superseded(rel)}
                          for _doc, rel in checks_docs.values()],
        "frozen_axis_sizes_check": sizes_chk,
        "reconstruction": recon_block,
        "dose_rule": "train_lora.py lines 217-227 and 254-256: grad_accum micro-batches of micro_bs consumed rows per "
                     "step, the checkpoint saved after the step's update, so step k has seen the first k x micro_bs x "
                     "grad_accum (16k) consumed examples, a prefix of the order (no wrap: checked)",
        "order_inside_prefix": ORDER_NOTE,
        "counts": counts,
    }
    code1 = P.code_block(code_files(), allow_no_commit=allow_no_commit)
    if code1["sha256"] != code0["sha256"]:
        refuse("the code changed during the run; nothing is recorded")
    record["input_files_sha256"] = dict(PV._RESULT_READS)      # D-297 (1): every results file read, with its sha256
    problems = PV.input_identity_problems(record)
    if problems:
        refuse(f"D-297 (1): {problems}")
    digest = write_results(out, record, exclusive=True)
    print(f"headline ({HEADLINE}): count of consumed examples ({first_n}); see bias.upper_bound_on_trained")
    for corpus in CORPORA:
        d = counts[corpus]["doses"]
        print(f"{corpus}: " + "; ".join(f"step {s}: {v['headline']['n_conversations']} of {v['n_examples_seen']} "
                                        f"consumed examples with a trained, unchecked frozen user turn"
                                        for s, v in d.items()))
    print(f"\nWritten: {PV._repo_rel(out)}\nsha256:  {digest}", flush=True)
    return out, record


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="D-344 (3): Phase 1's LoRA training rows against data/frozen, by later "
                                             "Human turn and dose")
    ap.add_argument("--local-hub", required=True, help="the NVMe hub directory src/fetch_dataset_snapshot.py "
                                                       "--lora-corpus fetched the three corpora into")
    args = ap.parse_args(argv)
    run(args.local_hub)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
