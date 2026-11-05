"""Phase 2 LoRA trainer (D-343 (2)): one (corpus, seed) dose series on Mistral-7B-Instruct-v0.2 with Phase 1's recipe.

What it is. Phase 1's src/train_lora.py (unchanged, and imported here only for its prompt function) trained three dose
series on Mistral-7B-v0.1. D-303 (f) asks for the same steps, doses and seeds on Mistral-7B-Instruct-v0.2, formatted with
that model's chat template; D-343 (2) fixes the rest. One call trains one corpus at one seed and writes the checkpoints
at steps 0, 25, 100 and 400 to an NVMe directory given on the command line, never under the repository. A separate tool
ingests them into data/lora before any extraction (D-307 (2); extract_p2 refuses an adapter outside data/lora).

THE RECIPE, value by value, with its source. Every value is a module constant, not an option. At run time each one is
compared with the corpus's Phase 1 record (results/lora_train__mistral7b-v01-<alpaca|hhharmless|pkusafe>-r16.json,
read through provenance.read_result) and the run refuses on any difference (check_phase1_record).
  base           mistralai/Mistral-7B-Instruct-v0.2 @ 63a8b081895390a26e140280378bc85ec8bce07a: D-343 (1)-(2); the
                 commit results/phase2_config_check.json resolved (models[repo].sha) and results/phase2/
                 tokenizer_static.json records (D-303 (8)); both are checked.
  rank 16, alpha 32, dropout 0.05
                 train_lora.py lines 122-124 (argparse defaults); records' config.rank/.alpha/.dropout; D-106.
  targets        q_proj k_proj v_proj o_proj gate_proj up_proj down_proj: train_lora.py lines 125-126; config.targets.
  lr 2e-4, warmup 10
                 train_lora.py lines 127-128; the schedule lr * min(1, step / max(warmup, 1)), constant after warmup:
                 line 221; config.lr/.warmup.
  sequence length 256
                 train_lora.py line 129 (tokenizer truncation, line 229-230; right side: the v0.2 tokenizer's
                 truncation_side, recorded 'right' in tokenizer_static.json, is checked); config.seq_len.
  micro-batch 4 x accumulation 4
                 train_lora.py lines 130-131 and 225-227; config.micro_bs/.grad_accum. Loss per micro-batch / 4,
                 lines 234-235.
  steps          checkpoints at 0, 25, 100, 400: train_lora.py line 121; config.save_steps. Step 0 is saved before any
                 optimiser step (lines 211-213); n_needed = 400 x 4 x 4 = 6400 examples (lines 155-156; records'
                 n_examples_used).
  precision      base fp16 (from_pretrained dtype float16, low_cpu_mem_usage, use_safetensors), adapter fp32 (every
                 trainable tensor .float()), fp16 autocast with torch.amp.GradScaler('cuda'): train_lora.py lines
                 181-183, 190-192, 198, 234; gradient checkpointing, enable_input_require_grads, use_cache False: lines
                 184-186; LoraConfig(task_type CAUSAL_LM) + get_peft_model: lines 187-189.
  optimiser      AdamW(lr, weight_decay 0.0) on the trainable tensors; unscale, clip_grad_norm 1.0, step, update,
                 zero_grad(set_to_none=True): train_lora.py lines 196-197, 241-245.
  non-finite     a micro-batch whose loss is not finite is skipped (lines 236-238); more than 10 x grad_accum of them
                 aborts (lines 247-249). Phase 1 then wrote its record; here the abort REFUSES (nothing under the final
                 names), as every other failed check does.
  loss           the causal LM loss on the full sequence: labels = input_ids with padded positions (attention_mask 0)
                 set to -100 (lines 231-232); the model's own shift; mean over the micro-batch's label tokens.
                 D-343 (2) rejected assistant-only loss.
  padding        pad token = EOS when the tokenizer has none (lines 179-180); padded to the longest row of the
                 micro-batch on the tokenizer's padding_side (v0.2: 'left', as tokenizer_static.json records), with an
                 attention mask and no position ids passed, exactly what tok(texts, padding=True) did (pad_batch).
  seeds          0, 1, 2 (D-303 (7)). Seed handling as Phase 1: torch.manual_seed(seed) before the model is loaded
                 (line 144) and the data order from numpy.random.default_rng(seed).permutation(len(rows)) over the
                 corpus's train rows in file order (lines 145, 161); rows whose prompt is in a frozen split are dropped
                 keeping that order (lines 163-171); the first 6400 of what remains are consumed in order, micro-batch k
                 holding positions 4k..4k+3 (line 226's cursor arithmetic; no wrap, since at least 6400 remain).
  trainable      41,943,040 parameters, the Phase 1 records' trainable_params (same architecture): checked after
                 get_peft_model.
  corpora        the revisions the Phase 1 records hold (dataset_revision; config.dataset, config.data_dir):
                 yahma/alpaca-cleaned @ 12567cabf869d7c92e573c7c783905fc160e9639, Anthropic/hh-rlhf (harmless-base) @
                 09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa, PKU-Alignment/PKU-SafeRLHF @
                 9421ffafec3fa40a1f1a7d567b4d525079477ecb.

WHAT D-343 (2) CHANGES, and how.
  (a) Each example is token ids from the base's chat template: tok.apply_chat_template(messages, tokenize=True,
      add_generation_prompt=False). The v0.2 template writes BOS once and EOS after each assistant message, so no
      double BOS or EOS; every example is checked for exactly one BOS (at position 0) and one EOS per assistant message,
      the last token EOS (before truncation). The loaded template's sha256 must equal tokenizer_static.json's for this
      repo and revision, and so must the vocabulary hash and the added tokens; the snapshot's tokenizer and config
      files must have the sha256s that record lists.
  (b) Special-token strings in user OR assistant content are tokenized as plain text (D-324 (5)(v)), by
      phase2_tokenizer_check's machinery (PLAIN_RULE: user_added_positions, added_ids, encode_stretch with its
      ANCHOR, the faithfulness check), which handles one user message with a generation prompt (user_span_plain).
      EXTENDED HERE, minimally, to several content spans (conversation_ids): each message's character range is found
      by rendering sentinel contents and rebuilding the real render from the template's pieces (it must be equal); an
      added token parsed from non-whitespace characters of any content range sends its stretch (between the
      template's nearest added tokens) through encode_stretch exactly as user_span_plain does; a conversation with no
      such string keeps apply_chat_template's ids exactly. phase2_tokenizer_check.py itself is not changed (extract_p2
      records bind its sha256).
  (c) Messages. alpaca: the user message is the instruction, joined to the input (if any) as Phase 1 joined them
      ('\\n\\n### Input:\\n', train_lora.fmt); the role headers '### Instruction:' and '### Response:' are what the
      template's [INST] ... [/INST] replace, as HH's 'Human:'/'Assistant:' labels are; the output is the assistant
      message. Strings stripped as Phase 1 stripped them. hh: the 'chosen' transcript split at '\\n\\nHuman:' and
      '\\n\\nAssistant:' into turns (each stripped); it must start with Human, alternate, and end with Assistant, else it
      is dropped and counted by reason. pku: one exchange, the prompt and the response the dataset labels safer
      (safer_response_id), as Phase 1 chose it (train_lora.text_and_prompt).
  (d) Loss on the full sequence, as Phase 1 (above).
  (e) Truncation to the first 256 tokens, as Phase 1; the number of examples truncated is recorded.
  (f) Frozen-split exclusion (D-344 (4)): a conversation is excluded when ANY user turn's item_id (sha256 of the
      stripped turn text, first 16 hex, as freeze_splits.py hashes) or the item_id of Phase 1's own prompt
      (train_lora.text_and_prompt: alpaca the stripped instruction, hh the stripped first Human turn, pku the stripped
      prompt) is the item_id of a row of data/frozen/*.jsonl, data/frozen_p2/*.jsonl or data/frozen_p2_v2/*.jsonl (the
      last if present). Phase 1 checked only its prompt, so an HH transcript's later Human turns went unchecked
      (D-344 (1)); alpaca and PKU have one user turn. Recorded: each frozen file's sha256 and item count; per frozen
      file (its directory and split) and per directory, over the whole corpus, the rows whose Phase 1 prompt is one of
      its items, the conversations with any user turn (and with a later user turn) that is, and the distinct items
      those turns are; the run's exclusions by rule; the used conversations re-checked (the record's 'contamination'
      names exactly what was checked). Phase 1's own rule (its prompt, data/frozen alone) is replayed beside its
      record's count, and for hh and pku, whose records hold one, a different count REFUSES: it is the real-data check
      that the readers give Phase 1's rows.
  (g) The corpus is read only from the pinned local snapshot (--local-hub), offline: every file of the snapshot must
      have the sha256 its fetch record (results/phase2/fetch_dataset__<org>__<name>@<rev12>.json, written by
      src/fetch_dataset_snapshot.py --lora-corpus) lists, and that record must pin this corpus to the Phase 1 record
      read here, at its sha256. The train rows are parsed from the bytes hashed, by plain readers that reproduce the
      rows load_dataset gave Phase 1 in its order (alpaca: the JSON array; hh: harmless-base/train.jsonl.gz, the
      split data_dir selects; pku: the README's default-config train files in their listed order); the README's
      configs and the data files in scope are checked against those assumptions. No datasets library, no Arrow cache.
  (h) Adapters go to <--out-dir>/<run name>/step_<N>/ on NVMe, outside the repository. They are written into a
      staging directory, hashed, renamed into place without replacing anything, and hashed again; each checkpoint's
      files are recorded with their sha256 (adapter_model.safetensors and adapter_config.json named apart, the two
      values extract_p2 requires). The model's config._name_or_path and name_or_path are set to the base repo id before
      the adapter is attached, so peft's README.md card (base_model) and adapter_config.json name it, not the local
      snapshot path; before anything is named, no file of any checkpoint may hold the local load path, the home
      directory or the account name (D-172). Step 0 (saved before any optimiser step) must be the same bytes for every
      corpus of a seed: checked against every sibling record of the seed on disk as soon as step 0 is saved, otherwise
      recorded for the ingest tool; whether seed 0's step 0 equals Phase 1's is recorded, not checked.
  (i) The record results/phase2/lora_train_p2__<corpus>__s<seed>.json, exclusive create, through
      provenance.write_results (D-297 (1): every results file read is listed with its sha256): provenance block; code
      block (D-161: every src file loaded, hashed at start and compared at the end); inputs with sha256 (corpus files,
      frozen files, base weight files); every recipe value; the template sha256; counts of rows, examples used,
      dropped, truncated, excluded; per-checkpoint file hashes; the loss trace; torch/transformers/peft versions, GPU
      name, BLAS threads (provenance env), the bit-environment and determinism flags as they were (Phase 1 set none;
      a GPU training run is not in general bit-reproducible, so the record names the flags rather than claims it);
      the ingest manifest (each checkpoint directory's files, sizes, sha256); and that the Phase 1 records pinned to are
      superseded under D-161 (code provenance unverifiable), used only as the written statement of the recipe and the
      corpus revision, every value compared.

REFUSALS (exit nonzero; nothing under the record's name, nothing under the adapters' final name): an existing record or
run directory; an output directory inside the repository or not on NVMe; a seed outside 0-2; any recipe value, corpus,
revision or trainable-parameter count that differs from the Phase 1 record; a config check or tokenizer record naming
another revision; a fetch record that does not pin this corpus to that Phase 1 record; any snapshot file whose sha256 is
not the fetch record's; a README or file layout the readers do not assume; a malformed row; fewer than 6400 eligible
examples; a template, vocabulary, added-token or tokenizer-file mismatch; a weight file that is not the blob its name
promises; a Phase 1 exclusion count that does not replay (hh, pku); any example that fails the plain-text, BOS or
EOS checks (one such example refuses the whole run: accepted after review, so no example is silently dropped or
altered); persistent non-finite losses; a step-0 adapter other than a sibling record's of the same seed; a checkpoint
file holding a host string; the code changing during the run; SIGTERM (a handler raises SystemExit, so a lane's TERM
takes the failure path); malformed JSON, JSON-lines or YAML inputs and missing records (a message, not a traceback).
NOTHING IS DELETED (src/safe_outputs.py): on a failure the staging directory, or the run directory renamed aside to
'<name>.failed.<pid>', is left and named; removal is a human's scripts/safe_rm.sh call.

  HF_HUB_OFFLINE=1 python3 src/train_lora_p2.py --corpus hh --seed 0 --out-dir <NVMe dir> --local-hub <NVMe hub dir>
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_p2 as XP  # noqa: E402  (snapshot, lib_versions, bit_environment, parse_jsonl_bytes: reused, not copied)
import fetch_weights as FW  # noqa: E402  (git_blob_sha1 for a weight snapshot's small files)
import paired as P  # noqa: E402
import phase2_tokenizer_check as TC  # noqa: E402  (the plain-text machinery, D-324 (5)(v))
import provenance as PV  # noqa: E402
import safe_outputs as SO  # noqa: E402
import superseded_provenance as SP  # noqa: E402  (D-161: the Phase 1 records' status, stated in the record)
import train_lora as T1  # noqa: E402  (Phase 1: its prompt function, for the exclusion hash; the file is unchanged)
from hf_env_guard import NVME_SOURCE_PREFIX, assert_hf_env, mount_of  # noqa: E402
from provenance import (  # noqa: E402
    canonical_hash,
    ensure_utf8_stdout,
    provenance_block,
    read_result,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_results,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/train_lora_p2.py", "src/train_lora.py", "src/extract_p2.py", "src/phase2_tokenizer_check.py",
              "src/provenance.py", "src/paired.py", "src/safe_outputs.py", "src/hf_env_guard.py", "src/fetch_weights.py"]
DECISION = "D-343 (2)"

BASE_REPO = "mistralai/Mistral-7B-Instruct-v0.2"
BASE_REV = "63a8b081895390a26e140280378bc85ec8bce07a"
SEEDS = (0, 1, 2)                                            # D-303 (7)
RANK, ALPHA, DROPOUT = 16, 32, 0.05
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LR, WARMUP = 2e-4, 10
SEQ_LEN = 256
MICRO_BS, GRAD_ACCUM = 4, 4
SAVE_STEPS = (0, 25, 100, 400)
WEIGHT_DECAY, CLIP_NORM = 0.0, 1.0
NONFINITE_ABORT = 10                                         # x GRAD_ACCUM micro-batches (train_lora.py line 247)
N_NEEDED = max(SAVE_STEPS) * MICRO_BS * GRAD_ACCUM           # 6400
PHASE1_CONFIG = {"rank": RANK, "alpha": ALPHA, "dropout": DROPOUT, "targets": list(TARGETS), "lr": LR,
                 "warmup": WARMUP, "seq_len": SEQ_LEN, "micro_bs": MICRO_BS, "grad_accum": GRAD_ACCUM,
                 "save_steps": list(SAVE_STEPS)}

CORPORA: dict[str, dict[str, Any]] = {
    "alpaca": {"phase1_record": "lora_train__mistral7b-v01-alpaca-r16.json", "repo": "yahma/alpaca-cleaned",
               "data_dir": None, "phase1_format": "alpaca", "tag": "alpaca",
               "train_files": ("alpaca_data_cleaned.json",), "reader": "json_array",
               "scope": "", "scope_files": ("alpaca_data_cleaned.json",), "readme_configs": False},
    "hh": {"phase1_record": "lora_train__mistral7b-v01-hhharmless-r16.json", "repo": "Anthropic/hh-rlhf",
           "data_dir": "harmless-base", "phase1_format": "hh", "tag": "hhharmless",
           "train_files": ("harmless-base/train.jsonl.gz",), "reader": "jsonl_gz",
           "scope": "harmless-base/", "scope_files": ("harmless-base/test.jsonl.gz", "harmless-base/train.jsonl.gz"),
           "readme_configs": False},
    "pku": {"phase1_record": "lora_train__mistral7b-v01-pkusafe-r16.json", "repo": "PKU-Alignment/PKU-SafeRLHF",
            "data_dir": None, "phase1_format": "pku", "tag": "pkusafe",
            "train_files": ("data/Alpaca-7B/train.jsonl", "data/Alpaca2-7B/train.jsonl", "data/Alpaca3-8B/train.jsonl"),
            "reader": "jsonl", "scope": None, "scope_files": None, "readme_configs": True},
}
METADATA_FILES = (".gitattributes", "README.md")
FROZEN_DIRS = ("data/frozen", "data/frozen_p2", "data/frozen_p2_v2")
FROZEN_REQUIRED = ("data/frozen", "data/frozen_p2")
ALPACA_INPUT_JOIN = "\n\n### Input:\n"                        # train_lora.fmt joins the input to the instruction so
HH_MARK = re.compile(r"\n\n(Human|Assistant):")
MSG_SENTINEL = "SENTINELMSG{}_4c1e"
EXCLUSION_RULE = ("D-344 (4), D-343 (2), D-080: a conversation is excluded when the item_id (sha256 of the UTF-8 of the "
                  "stripped text, first 16 hex, as freeze_splits.py hashes) of ANY of its user turns, or of Phase 1's "
                  "own prompt (train_lora.text_and_prompt(row, format)[1]: alpaca the stripped instruction, hh the "
                  "stripped first Human turn, pku the stripped prompt), is the item_id of a row of data/frozen/*.jsonl, "
                  "data/frozen_p2/*.jsonl or data/frozen_p2_v2/*.jsonl (the last if present); excluded rows are dropped "
                  "keeping the permuted order")
REPLAY_REQUIRED = ("hh", "pku")                              # their Phase 1 records hold the exclusion count
PLAIN_EXTENSION = ("phase2_tokenizer_check.PLAIN_RULE applied to every message's content (user and assistant), not "
                   "only one user message: conversation_ids finds each content's character range in the render, sends "
                   "each stretch between the template's added tokens that holds an added token parsed from non-whitespace "
                   "content characters through encode_stretch (the anchor and the faithfulness check unchanged), and keeps "
                   "apply_chat_template's ids exactly for a conversation with no such string")


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def run_name(corpus: str, seed: int) -> str:
    """The adapters' directory name; 'seed', not 's', since Phase 1's cache names use '-s<step>'."""
    return f"mistral7bi-v02-{CORPORA[corpus]['tag']}-r16-seed{seed}"


def record_name(corpus: str, seed: int) -> str:
    return f"lora_train_p2__{corpus}__s{seed}.json"


def item_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# --- Phase 1 record, config check, tokenizer record ----------------------------------------------------------------

def check_phase1_record(rec: Any, corpus: str) -> dict[str, Any]:
    """Every recipe constant equals the corpus's Phase 1 record (config.*), and so do the dataset, data_dir and format
    (the alpaca record predates --format and --data-dir: their defaults, 'alpaca' and None). Seeds are not compared:
    D-303 (7) gives 0, 1, 2."""
    spec = CORPORA[corpus]
    if not isinstance(rec, dict) or rec.get("run_kind") != "lora_train" or not isinstance(rec.get("config"), dict):
        refuse(f"{spec['phase1_record']} is not a Phase 1 LoRA training record")
    cfg = rec["config"]
    bad = {k: (cfg.get(k), v) for k, v in PHASE1_CONFIG.items() if cfg.get(k) != v}
    if bad:
        refuse(f"the recipe differs from {spec['phase1_record']} (record value, this module's): {bad}")
    got = {"dataset": cfg.get("dataset"), "data_dir": cfg.get("data_dir"), "format": cfg.get("format", "alpaca")}
    want = {"dataset": spec["repo"], "data_dir": spec["data_dir"], "format": spec["phase1_format"]}
    if got != want:
        refuse(f"{spec['phase1_record']} names {got}, not {want}")
    rev = rec.get("dataset_revision")
    if not isinstance(rev, str) or not re.fullmatch(r"[0-9a-f]{40}", rev):
        refuse(f"{spec['phase1_record']} holds no 40-hex dataset_revision ({rev!r})")
    n_tp = rec.get("trainable_params")
    if not isinstance(n_tp, int) or rec.get("n_examples_used") != N_NEEDED:
        refuse(f"{spec['phase1_record']}: trainable_params {n_tp!r}, n_examples_used {rec.get('n_examples_used')!r}")
    return {"compared": sorted(PHASE1_CONFIG) + ["dataset", "data_dir", "format", "n_examples_used"],
            "values_equal": True, "dataset_revision": rev, "trainable_params": n_tp,
            "exclude_frozen": cfg.get("exclude_frozen", False),
            "n_excluded_for_frozen_overlap": rec.get("n_excluded_for_frozen_overlap"),
            "step0_adapter_sha256": next((c.get("sha256") for c in rec.get("checkpoints") or []
                                          if isinstance(c, dict) and c.get("step") == 0), None),
            "base_repo": cfg.get("base_repo"), "base_rev": cfg.get("base_rev")}


def check_base_revision(cc: Any, static: Any) -> dict[str, Any]:
    """D-303 (8): BASE_REV is the commit the config check resolved for BASE_REPO, and the tokenizer record holds that
    repo at that revision as an instruct model. Returns the tokenizer record's entry for it."""
    m = ((cc.get("models") if isinstance(cc, dict) else None) or {}).get(BASE_REPO)
    if not isinstance(m, dict) or m.get("sha") != BASE_REV:
        refuse(f"the config check resolves {BASE_REPO} at {str((m or {}).get('sha'))[:12]}, not {BASE_REV[:12]}")
    st = ((static.get("models") if isinstance(static, dict) else None) or {}).get(BASE_REPO)
    if not isinstance(st, dict) or st.get("revision") != BASE_REV or st.get("kind") != "instruct":
        refuse(f"tokenizer_static.json does not record {BASE_REPO} @ {BASE_REV[:12]} as an instruct model")
    ct = (st.get("tokenizer") or {}).get("chat_template") or {}
    if not re.fullmatch(r"[0-9a-f]{64}", str(ct.get("sha256", ""))):
        refuse("tokenizer_static.json records no chat template sha256 for the base")
    if ct.get("reads_clock"):
        refuse("the base's template reads the clock; this trainer passes no template kwargs")
    return st


def check_tokenizer(tok: Any, st: dict[str, Any]) -> dict[str, Any]:
    """The loaded tokenizer is the one tokenizer_static.json records: the chat template's sha256 (the string
    apply_chat_template uses, phase2_tokenizer_check.template_text), the vocabulary hash, the added tokens, BOS/EOS,
    padding and truncation sides; a fast tokenizer (character offsets)."""
    rec = st.get("tokenizer") or {}
    tmpl = TC.template_text(tok)
    sha = sha256_bytes(tmpl.encode("utf-8")) if tmpl else None
    want = (rec.get("chat_template") or {}).get("sha256")
    if sha != want:
        refuse(f"the loaded chat template's sha256 {str(sha)[:12]} is not tokenizer_static.json's {str(want)[:12]} for "
               f"{BASE_REPO} @ {BASE_REV[:12]}")
    if "strftime_now" in (tmpl or ""):
        refuse("the chat template reads the clock; this trainer passes no template kwargs")
    if not getattr(tok, "is_fast", False):
        refuse("the tokenizer is not a fast tokenizer (character offsets are needed for the plain-text rule)")
    checks = {"vocab_sha256": (TC._vocab_sha(tok), rec.get("vocab_sha256")),
              "added_tokens": (TC._added(tok), rec.get("added_tokens")),
              "bos": ({"token": str(tok.bos_token), "id": int(tok.bos_token_id)}, rec.get("bos")),
              "eos": ({"token": str(tok.eos_token), "id": int(tok.eos_token_id)}, rec.get("eos")),
              "padding_side": (tok.padding_side, rec.get("padding_side")),
              "truncation_side": (tok.truncation_side, rec.get("truncation_side"))}
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        refuse(f"the loaded tokenizer differs from tokenizer_static.json's record (loaded, recorded): {bad}")
    if tok.truncation_side != "right":
        refuse(f"truncation_side {tok.truncation_side!r}: Phase 1 kept the first {SEQ_LEN} tokens (right truncation)")
    return {"chat_template_sha256": sha, "vocab_sha256": checks["vocab_sha256"][0], "added_tokens_equal": True,
            "bos_id": int(tok.bos_token_id), "eos_id": int(tok.eos_token_id), "padding_side": tok.padding_side,
            "truncation_side": tok.truncation_side, "pad_token_before": None if tok.pad_token is None else str(tok.pad_token),
            "equal_to_tokenizer_static": sorted(["chat_template_sha256", *checks])}


def check_snapshot_files(snap: Path, st: dict[str, Any], weight_files: list[str], blobs: bool) -> dict[str, Any]:
    """The base snapshot: each tokenizer/config file tokenizer_static.json lists (fetched and hash-checked by the
    tokenizer check) has that sha256 here; with blobs (an HF cache snapshot), every file is the blob its name promises
    (a 64-hex blob: sha256 = name; a 40-hex blob: git blob sha1 = name; verify_hf_cache.py's rule), so the weights are
    the bytes the Hub published at the revision."""
    listed = {f["name"]: f["sha256"] for f in ((st.get("fetch") or {}).get("files") or [])}
    tok_files = {}
    for name, want in sorted(listed.items()):
        p = snap / name
        if not p.is_file():
            refuse(f"the base snapshot holds no {name}, which tokenizer_static.json lists")
        got = sha256_file(p)
        if got != want:
            refuse(f"the base snapshot's {name} has sha256 {got[:12]}, not tokenizer_static.json's {want[:12]}")
        tok_files[name] = got
    weights: dict[str, Any] = {}
    if blobs:
        for p in sorted(x for x in snap.iterdir() if x.is_file()):
            blob = Path(os.path.realpath(p)).name
            if re.fullmatch(r"[0-9a-f]{64}", blob):
                got, how = sha256_file(p), "sha256"
            elif re.fullmatch(r"[0-9a-f]{40}", blob):
                got, how = FW.git_blob_sha1(p), "git blob sha1"
            else:
                refuse(f"the base snapshot's {p.name} is not a hub-cache blob link (blob {blob!r})")
            if got != blob:
                refuse(f"the base snapshot's {p.name}: {how} {got[:12]} is not its blob name {blob[:12]}")
            weights[p.name] = {"blob": blob, "check": how, "bytes": p.stat().st_size,
                               **({"sha256": got} if how == "sha256" else {"sha256": sha256_file(p)})}
        missing = [w for w in weight_files if w not in weights]
        if missing:
            refuse(f"weight files not checked: {missing}")
    return {"tokenizer_files_sha256_equal_to_tokenizer_static": tok_files, "files": weights,
            "blob_check": "every file of the snapshot equals the blob its link names (64-hex: sha256; 40-hex: git blob "
                          "sha1)" if blobs else "not applied (a test directory, not a hub-cache snapshot)"}


# --- the corpus: fetch record, snapshot, rows ----------------------------------------------------------------------

def fetch_record_path(results_dir: Path, corpus: str, rev: str) -> Path:
    org, name = CORPORA[corpus]["repo"].split("/")
    return Path(results_dir) / "phase2" / f"fetch_dataset__{org}__{name}@{rev[:12]}.json"


def check_fetch_record(frec: Any, corpus: str, rev: str, p1_rel: str, p1_sha: str) -> dict[str, str]:
    """The fetch record pins this corpus (fetch_dataset_snapshot --lora-corpus) to the Phase 1 record read here, at
    that record's sha256, and names the repo and revision; returns its {file: sha256}."""
    spec = CORPORA[corpus]
    if not isinstance(frec, dict):
        refuse(f"the fetch record is a {type(frec).__name__}, not a JSON object")
    pin = frec.get("pin")
    if not isinstance(pin, dict):
        refuse("the fetch record holds no pin object")
    if frec.get("repo") != spec["repo"] or frec.get("revision") != rev:
        refuse(f"the fetch record is for {frec.get('repo')}@{str(frec.get('revision'))[:12]}, not "
               f"{spec['repo']}@{rev[:12]}")
    if pin.get("lora_corpus") != corpus or pin.get("file") != p1_rel or pin.get("sha256") != p1_sha:
        refuse(f"the fetch record's pin ({json.dumps(pin, sort_keys=True)[:240]}) is not corpus {corpus} pinned to "
               f"{p1_rel} at {p1_sha[:12]}")
    files = frec.get("files")
    if not isinstance(files, list) or not files:
        refuse("the fetch record lists no files")
    out = {}
    for f in files:
        if not isinstance(f, dict) or not isinstance(f.get("name"), str) or \
                not re.fullmatch(r"[0-9a-f]{64}", str(f.get("sha256", ""))) or f["name"] in out:
            refuse(f"the fetch record lists a file without a unique name and sha256: {f!r}"[:300])
        out[f["name"]] = f["sha256"]
    return out


def verify_snapshot(snap: Path, expected: dict[str, str]) -> dict[str, str]:
    """Every file under the snapshot, with its sha256, must be exactly the fetch record's list."""
    if not snap.is_dir():
        refuse(f"no local snapshot at {snap}")
    got = {p.relative_to(snap).as_posix(): sha256_file(p) for p in sorted(snap.rglob("*")) if p.is_file()}
    if sorted(got) != sorted(expected):
        refuse(f"the snapshot {snap} holds {sorted(set(got) ^ set(expected))[:10]} beyond or short of the fetch record")
    bad = sorted(n for n in got if got[n] != expected[n])
    if bad:
        refuse(f"snapshot file(s) whose sha256 is not the fetch record's: {bad[:10]}")
    return got


def readme_configs(data: bytes) -> Any:
    """The 'configs' key of a dataset card's YAML front matter (None when there is none)."""
    import yaml
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        refuse("the README is not UTF-8")
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end < 0:
        refuse("the README's front matter is not closed")
    try:
        meta = yaml.safe_load(text[3:end]) or {}
    except yaml.YAMLError as e:
        refuse(f"the README's front matter is not valid YAML ({type(e).__name__})")
    return meta.get("configs") if isinstance(meta, dict) else None


def check_layout(corpus: str, files: dict[str, str], readme: bytes | None) -> dict[str, Any]:
    """What load_dataset(repo, data_dir=..., split='train') read for Phase 1, from this file list and README:
    pku: the README's default config lists the train files (in that order); alpaca and hh: the README has no configs and
    the data files in scope (the repo, or data_dir) are exactly the expected ones (hh: train and test by file name)."""
    spec = CORPORA[corpus]
    cfgs = readme_configs(readme) if readme is not None else None
    if spec["readme_configs"]:
        default = [c for c in (cfgs or []) if isinstance(c, dict) and c.get("config_name") == "default"]
        if len(default) != 1:
            refuse(f"{spec['repo']}'s README names no single default config")
        train = [d for d in default[0].get("data_files") or [] if isinstance(d, dict) and d.get("split") == "train"]
        paths = train[0].get("path") if len(train) == 1 else None
        paths = [paths] if isinstance(paths, str) else paths
        if list(paths or []) != list(spec["train_files"]):
            refuse(f"{spec['repo']}'s default config lists train files {paths}, not {list(spec['train_files'])}")
        rule = "the README's default config: split train = these files, in this order"
    else:
        if cfgs:
            refuse(f"{spec['repo']}'s README declares configs; the split is not inferred from file names as assumed")
        scope = sorted(n for n in files if n.startswith(spec["scope"]) and n not in METADATA_FILES)
        if scope != sorted(spec["scope_files"]):
            refuse(f"{spec['repo']}: the data files in scope {spec['scope'] or '(repo root)'} are {scope}, not "
                   f"{sorted(spec['scope_files'])}")
        rule = (f"no README configs; the data files under '{spec['scope'] or '(root)'}' are {sorted(spec['scope_files'])}"
                "; split train = the file(s) named train" + ("" if spec["scope"] else " (the only data file)"))
    missing = [n for n in spec["train_files"] if n not in files]
    if missing:
        refuse(f"the snapshot holds no {missing}")
    return {"rule": rule, "train_files": list(spec["train_files"])}


def parse_jsonl(data: bytes, name: str) -> list[dict[str, Any]]:
    rows = XP.parse_jsonl_bytes(data)                         # split on '\n' only (F-017); blank lines skipped
    if not all(isinstance(r, dict) for r in rows):
        refuse(f"{name}: a line is not a JSON object")
    return rows


def read_rows(corpus: str, snap: Path, expected: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The train rows in load_dataset's order, parsed from the bytes whose sha256 is the fetch record's."""
    spec = CORPORA[corpus]
    rows: list[dict[str, Any]] = []
    read = []
    for name in spec["train_files"]:
        data = (snap / name).read_bytes()
        h = sha256_bytes(data)
        if h != expected.get(name):
            refuse(f"{name} has sha256 {h[:12]} as read, not the fetch record's")
        try:
            if spec["reader"] == "json_array":
                got = json.loads(data.decode("utf-8"))        # not a result (D-297): corpus rows
                if not isinstance(got, list) or not all(isinstance(r, dict) for r in got):
                    refuse(f"{name} is not a JSON array of objects")
            elif spec["reader"] == "jsonl_gz":
                got = parse_jsonl(gzip.decompress(data), name)
            else:
                got = parse_jsonl(data, name)
        except (ValueError, UnicodeDecodeError, OSError, EOFError) as e:
            refuse(f"{name} does not parse ({type(e).__name__})")
        read.append({"file": name, "sha256": h, "bytes": len(data), "rows": len(got)})
        rows.extend(got)
    return rows, read


# --- messages ------------------------------------------------------------------------------------------------------

def _str(row: dict[str, Any], key: str, corpus: str, i: int, optional: bool = False) -> str:
    v = row.get(key)
    if v is None and optional:
        return ""
    if not isinstance(v, str):
        refuse(f"{corpus} row {i}: {key} is {type(v).__name__}, not a string")
    return v


def hh_turns(chosen: str) -> tuple[list[dict[str, str]] | None, str | None]:
    """The 'chosen' transcript as alternating turns, or (None, reason). Turn markers are '\\n\\nHuman:' and
    '\\n\\nAssistant:' (a transcript's leading whitespace is dropped and a marker at its start counts); each turn's
    text is stripped. It must start with Human, alternate, and end with Assistant."""
    s = "\n\n" + chosen.lstrip()
    marks = list(HH_MARK.finditer(s))
    if not marks or marks[0].start() != 0:
        return None, "no_turn_marker" if not marks else "text_before_first_turn"
    turns = []
    for k, m in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(s)
        turns.append({"role": "user" if m.group(1) == "Human" else "assistant", "content": s[m.end():end].strip()})
    if turns[0]["role"] != "user":
        return None, "first_turn_not_user"
    if any(turns[k]["role"] == turns[k - 1]["role"] for k in range(1, len(turns))):
        return None, "consecutive_same_role"
    if turns[-1]["role"] != "assistant":
        return None, "last_turn_not_assistant"
    return turns, None


def messages_for(corpus: str, row: Any, i: int) -> tuple[list[dict[str, str]] | None, str | None]:
    """The conversation for one row, or (None, drop reason). A malformed row refuses."""
    if not isinstance(row, dict):
        refuse(f"{corpus} row {i} is not an object")
    if corpus == "alpaca":
        ins, out = _str(row, "instruction", corpus, i).strip(), _str(row, "output", corpus, i).strip()
        inp = _str(row, "input", corpus, i, optional=True).strip()
        user = ins + (ALPACA_INPUT_JOIN + inp if inp else "")
        return [{"role": "user", "content": user}, {"role": "assistant", "content": out}], None
    if corpus == "hh":
        return hh_turns(_str(row, "chosen", corpus, i))
    if corpus == "pku":
        prompt = _str(row, "prompt", corpus, i).strip()
        safer = row.get("safer_response_id")
        if isinstance(safer, bool) or safer not in (0, 1):
            refuse(f"pku row {i}: safer_response_id {safer!r} is not 0 or 1")
        reply = _str(row, f"response_{int(safer)}", corpus, i).strip()
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}], None
    refuse(f"unknown corpus {corpus!r}")
    return None, None


def phase1_prompt(corpus: str, row: dict[str, Any]) -> str:
    """The prompt Phase 1 hashed for its exclusion (train_lora.text_and_prompt(row, format)[1])."""
    return T1.text_and_prompt(row, CORPORA[corpus]["phase1_format"])[1]


# --- frozen splits and the selection ------------------------------------------------------------------------------

def frozen_ids(root: Path) -> tuple[dict[str, set[str] | None], dict[str, set[str]], dict[str, Any]]:
    """({directory: the item_ids of its *.jsonl files, None when an optional directory is absent}, {file: its item_ids}
    (a file is one directory and split, e.g. 'data/frozen/safety__eval.jsonl'), the record's block: every file with its
    sha256 and item count). data/frozen and data/frozen_p2 must exist and hold files; a file that does not parse, or a
    row without a 16-hex item_id, refuses."""
    by_dir: dict[str, set[str] | None] = {}
    by_file: dict[str, set[str]] = {}
    files = []
    for d in FROZEN_DIRS:
        p = Path(root) / d
        if not p.is_dir():
            if d in FROZEN_REQUIRED:
                refuse(f"{d} is not a directory; the frozen-split exclusion needs it")
            by_dir[d] = None
            continue
        paths = sorted(p.glob("*.jsonl"))
        if not paths and d in FROZEN_REQUIRED:
            refuse(f"{d} holds no *.jsonl file")
        ids: set[str] = set()
        for f in paths:
            data = f.read_bytes()
            try:
                rows = XP.parse_jsonl_bytes(data)
            except (ValueError, UnicodeDecodeError) as e:
                refuse(f"{d}/{f.name} does not parse as JSON lines ({type(e).__name__})")
            got = [r.get("item_id") if isinstance(r, dict) else None for r in rows]
            if not all(isinstance(x, str) and re.fullmatch(r"[0-9a-f]{16}", x) for x in got):
                refuse(f"{d}/{f.name}: a row without a 16-hex item_id")
            by_file[f"{d}/{f.name}"] = set(got)
            ids.update(got)
            files.append({"file": f"{d}/{f.name}", "sha256": sha256_bytes(data), "n_items": len(got)})
        by_dir[d] = ids
    union = set().union(*[v for v in by_dir.values() if v is not None])
    return by_dir, by_file, {"rule": EXCLUSION_RULE, "files": files,
                             "directories": {d: ({"present": False} if v is None else
                                                 {"present": True, "n_item_ids": len(v)}) for d, v in by_dir.items()},
                             "n_item_ids_union": len(union)}


def user_turn_ids(messages: list[dict[str, str]]) -> list[str]:
    """The item_id of each user turn's stripped text, in turn order (D-344 (4))."""
    return [item_id(m["content"].strip()) for m in messages if m["role"] == "user"]


def select_examples(rows: list[dict[str, Any]], corpus: str, seed: int, by_dir: dict[str, set[str] | None],
                    n_needed: int = N_NEEDED, by_file: dict[str, set[str]] | None = None) -> dict[str, Any]:
    """Phase 1's data order (train_lora.py lines 145, 161-175, 226) with D-344 (4)'s exclusion: the permutation of every
    train row by numpy.random.default_rng(seed); dropped from it, keeping its order, (1) the rows whose Phase 1 prompt is
    a frozen item (Phase 1's rule), (2) the HH transcripts that do not alternate (D-343 (2)), (3) the conversations with
    any user turn that is a frozen item (D-344 (4)); the first n_needed of what remains are used, in order. The counts
    per frozen file (directory and split) and per directory are over the whole corpus."""
    rng = np.random.default_rng(seed)
    order = [int(i) for i in rng.permutation(len(rows))]
    union = set().union(*[v for v in by_dir.values() if v is not None])
    pids: list[str] = []
    msgs_all: dict[int, list[dict[str, str]]] = {}
    why_all: dict[int, str] = {}
    for i, r in enumerate(rows):
        m, why = messages_for(corpus, r, i)                    # a malformed row refuses here, before Phase 1's function
        try:
            pids.append(item_id(phase1_prompt(corpus, r)))
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            refuse(f"{corpus} row {i}: Phase 1's prompt function failed ({type(e).__name__}: {e})")
        if m is None:
            why_all[i] = str(why)
        else:
            msgs_all[i] = m
    uids = {i: user_turn_ids(m) for i, m in msgs_all.items()}
    kept = [i for i in order if pids[i] not in union]                            # (1)
    reasons = collections.Counter(why_all[i] for i in kept if i in why_all)       # (2)
    valid = [i for i in kept if i not in why_all]
    eligible = [i for i in valid if not any(u in union for u in uids[i])]        # (3)
    if len(eligible) < n_needed:
        refuse(f"need {n_needed} examples, only {len(eligible)} remain after the exclusion and the drops")
    used = [eligible[c % len(eligible)] for c in range(n_needed)]                # train_lora.py line 226 (no wrap here)

    def hits(ids: set[str]) -> dict[str, int]:
        return {"rows_phase1_prompt": sum(1 for p in pids if p in ids),
                "conversations_any_user_turn": sum(1 for u in uids.values() if any(x in ids for x in u)),
                "conversations_later_user_turn": sum(1 for u in uids.values() if any(x in ids for x in u[1:])),
                "distinct_items_any_user_turn": len({x for u in uids.values() for x in u if x in ids})}

    p1_dir = by_dir.get("data/frozen")
    return {"corpus": corpus, "n_rows": len(rows), "order_first_10": order[:10],
            "n_excluded_phase1_prompt": len(order) - len(kept),
            "n_dropped": sum(reasons.values()), "drop_reasons": dict(sorted(reasons.items())),
            "n_excluded_any_user_turn": len(valid) - len(eligible),
            "n_excluded": (len(order) - len(kept)) + (len(valid) - len(eligible)),
            "exclusion_order": "(1) Phase 1's prompt over every row; (2) the drops among the rows (1) kept; (3) any user "
                               "turn among the conversations (2) kept",
            "frozen_hits_by_directory": {d: (None if v is None else hits(v)) for d, v in by_dir.items()},
            "frozen_hits_by_file": {f: hits(v) for f, v in sorted((by_file or {}).items())},
            "frozen_hits_reading": "over the whole corpus: rows_phase1_prompt, the rows whose Phase 1 prompt is an item "
                                   "of that file or directory; conversations_any_user_turn and _later_user_turn, the rows "
                                   "forming a conversation with any user turn (a user turn after the first) that is one; "
                                   "distinct_items_any_user_turn, the items those turns are",
            "n_excluded_data_frozen_only": None if p1_dir is None else sum(1 for p in pids if p in p1_dir),
            "n_eligible": len(eligible), "n_needed": n_needed, "used": used,
            "messages": [msgs_all[i] for i in used],
            "contamination": {
                "checked": "every user turn's item_id and Phase 1's prompt item_id of every used conversation, against "
                           "the union of the item_ids of every frozen file read (D-344 (4)); assistant turns are not "
                           "checked",
                "used_with_a_frozen_user_turn": sum(1 for i in used if any(u in union for u in uids[i])),
                "used_with_a_frozen_phase1_prompt": sum(1 for i in used if pids[i] in union)},
            "n_used_with_an_empty_message": sum(1 for i in used if any(not m["content"] for m in msgs_all[i])),
            "n_turns": {"min": min(len(msgs_all[i]) for i in used), "max": max(len(msgs_all[i]) for i in used),
                        "n_used_multi_turn": sum(1 for i in used if len(msgs_all[i]) > 2)}}


# --- token ids: the template and the plain-text rule ---------------------------------------------------------------

def render_parts(tok: Any, messages: list[dict[str, str]], add_generation_prompt: bool = False
                 ) -> tuple[str, list[tuple[int, int]]]:
    """(the render, each message content's [start, end) in it). The template is rendered with sentinel contents, cut
    at the sentinels into its own pieces, and the real render must equal those pieces with each content inserted
    verbatim (else SpanError)."""
    sent = [MSG_SENTINEL.format(k) for k in range(len(messages))]
    srender = tok.apply_chat_template([{"role": m["role"], "content": s} for m, s in zip(messages, sent)],
                                      tokenize=False, add_generation_prompt=add_generation_prompt)
    pieces, pos = [], 0
    for s in sent:
        if srender.count(s) != 1 or srender.index(s) < pos:
            raise TC.SpanError("a message sentinel is not found exactly once, in order, in the template's render")
        i = srender.index(s)
        pieces.append(srender[pos:i])
        pos = i + len(s)
    pieces.append(srender[pos:])
    full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    rebuilt, ranges = "", []
    for k, m in enumerate(messages):
        rebuilt += pieces[k]
        ranges.append((len(rebuilt), len(rebuilt) + len(m["content"])))
        rebuilt += m["content"]
    rebuilt += pieces[-1]
    if rebuilt != full:
        raise TC.SpanError("the template does not insert each message's content verbatim between its own pieces")
    return full, ranges


def conversation_ids(tok: Any, messages: list[dict[str, str]], add_generation_prompt: bool = False
                     ) -> tuple[list[int], dict[str, Any]]:
    """(ids, info): apply_chat_template(messages, tokenize=True)'s ids, with every special-token string in any
    message's content tokenized as plain text (PLAIN_EXTENSION; phase2_tokenizer_check.user_span_plain's rule per
    stretch). A conversation without such a string gets exactly apply_chat_template's ids (info plain_text_path
    False). Raises SpanError when a check fails."""
    full, ranges = render_parts(tok, messages, add_generation_prompt)
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(i) for i in enc["input_ids"]]
    offs = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    check = [int(i) for i in XP._as_ids(tok.apply_chat_template(messages, tokenize=True,
                                                                 add_generation_prompt=add_generation_prompt))]
    if check != ids:
        raise TC.SpanError("apply_chat_template(tokenize=True) gives other ids than the render tokenized without added "
                           "special tokens")
    found = sorted({k for cs, ce in ranges for k in TC.user_added_positions(tok, full, cs, ce, ids, offs)})
    if not found:
        return ids, {"plain_text_path": False}
    dec = getattr(tok, "added_tokens_decoder", None) or {}
    strings = sorted({str(getattr(dec.get(ids[k]), "content", ids[k])) for k in found})
    added = TC.added_ids(tok)
    fset = set(found)
    tmpl = [k for k in range(len(ids)) if ids[k] in added and k not in fset]
    stretches = sorted({(max([t + 1 for t in tmpl if t < k], default=0), min([t for t in tmpl if t > k],
                                                                                 default=len(ids))) for k in found})
    if TC.ANCHOR in full:
        raise TC.SpanError("a message holds the stretch anchor string (PLAIN_RULE)")
    unk = TC.unk_exempt(tok)
    new_ids, new_offs = list(ids), list(offs)
    for lo, hi in sorted(stretches, reverse=True):               # the last first: earlier positions stay valid
        c0 = offs[lo - 1][1] if lo > 0 else 0
        c1 = offs[hi][0] if hi < len(ids) else len(full)
        again, _o = TC.encode_stretch(tok, False, full, c0, c1, False)
        if again != ids[lo:hi]:
            raise TC.SpanError(f"the stretch [{c0}, {c1}) encoded on its own does not give today's tokens of it, so its "
                               "plain-text tokens would not be the tokenizer's (PLAIN_RULE)")
        mid_ids, mid_offs = TC.encode_stretch(tok, True, full, c0, c1, False)
        if any(i in added and i != unk for i in mid_ids):
            raise TC.SpanError("the plain-text stretch still holds an added token (PLAIN_RULE)")
        new_ids[lo:hi] = mid_ids
        new_offs[lo:hi] = mid_offs
    left = sorted({k for cs, ce in ranges for k in TC.user_added_positions(tok, full, cs, ce, new_ids, new_offs)})
    if left:
        raise TC.SpanError("added token(s) are still parsed from message content after the plain tokenization")
    if [new_ids[k] for k in range(len(new_ids)) if new_ids[k] in added and new_ids[k] != unk] != \
            [ids[k] for k in tmpl if ids[k] != unk]:
        raise TC.SpanError("the template's added tokens changed under the plain tokenization")
    return new_ids, {"plain_text_path": True, "strings": strings, "stretches": [list(x) for x in stretches],
                     "today_ids_sha256": TC.ids_sha256(ids)}


def truncate(ids_list: list[list[int]], seq_len: int = SEQ_LEN) -> tuple[list[list[int]], int]:
    """Phase 1's truncation (tok(..., truncation=True, max_length=256), right side): the first seq_len tokens."""
    return [ids[:seq_len] for ids in ids_list], sum(1 for ids in ids_list if len(ids) > seq_len)


def build_examples(tok: Any, messages: list[list[dict[str, str]]], used: list[int], seq_len: int = SEQ_LEN
                   ) -> dict[str, Any]:
    """Token ids for each used conversation (conversation_ids), each checked for exactly one BOS (first), one EOS per
    assistant message and EOS last, then truncated as Phase 1."""
    bos, eos = int(tok.bos_token_id), int(tok.eos_token_id)
    ids_l, plain = [], []
    for row, msgs in zip(used, messages):
        try:
            ids, info = conversation_ids(tok, msgs)
        except TC.SpanError as e:
            refuse(f"row {row}: {e}")
        n_asst = sum(1 for m in msgs if m["role"] == "assistant")
        if not ids or ids[0] != bos or ids.count(bos) != 1 or ids.count(eos) != n_asst or ids[-1] != eos:
            refuse(f"row {row}: {ids.count(bos)} BOS (first {ids[:1]}) and {ids.count(eos)} EOS (last {ids[-1:]}) for "
                   f"{n_asst} assistant message(s); want one BOS first and one EOS per assistant message, EOS last")
        if info["plain_text_path"]:
            plain.append({"row": row, "strings": info["strings"]})
        ids_l.append(ids)
    lengths = np.asarray([len(x) for x in ids_l])
    out, n_trunc = truncate(ids_l, seq_len)
    return {"ids": out, "n_truncated": n_trunc,
            "tokens_before_truncation": XP.stats(lengths),
            "tokens_after_truncation": XP.stats(np.asarray([len(x) for x in out])),
            "plain_text_path": {"n_examples": len(plain), "examples": plain},
            "ids_sha256_before_truncation": sha256_bytes(",".join(TC.ids_sha256(x) for x in ids_l).encode("ascii")),
            "ids_sha256_trained": sha256_bytes(",".join(TC.ids_sha256(x) for x in out).encode("ascii")),
            "ids_sha256_method": "sha256 of the per-example phase2_tokenizer_check.ids_sha256 values joined by ',' "
                                 "in training order"}


def pad_batch(batch: list[list[int]], pad_id: int, side: str) -> tuple[list[list[int]], list[list[int]]]:
    """(input_ids, attention_mask) padded to the longest row on `side`: what tok(texts, padding=True) did in Phase 1
    (the tokenizer's padding_side, pad_token_id, no pad_to_multiple_of)."""
    if side not in ("left", "right"):
        raise ValueError(side)
    n = max(len(x) for x in batch)
    ids, mask = [], []
    for x in batch:
        pad = n - len(x)
        ids.append([pad_id] * pad + list(x) if side == "left" else list(x) + [pad_id] * pad)
        mask.append([0] * pad + [1] * len(x) if side == "left" else [1] * len(x) + [0] * pad)
    return ids, mask


# --- adapters on NVMe -----------------------------------------------------------------------------------------------

def check_out_dir(out_dir: Path, root: Path, require_nvme: bool) -> tuple[Path, dict[str, Any]]:
    d = Path(os.path.realpath(out_dir))
    r = Path(os.path.realpath(root))
    if d == r or r in d.parents:
        refuse(f"the adapter directory {d} is inside the repository {r}; adapters go to NVMe outside it (D-343 (2))")
    info: dict[str, Any] = {"out_dir": str(d), "nvme_checked": require_nvme}
    if require_nvme:
        mnt, src = mount_of(d)
        if not src.startswith(NVME_SOURCE_PREFIX):
            refuse(f"the adapter directory {d} is on {src or 'an unknown device'} (mounted at {mnt or '?'}), not NVMe")
        info.update(mount_point=mnt, mount_source=src)
    return d, info


def dir_files(d: Path) -> dict[str, dict[str, Any]]:
    return {p.relative_to(d).as_posix(): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in sorted(Path(d).rglob("*")) if p.is_file()}


def checkpoint_entry(stage: Path, step: int) -> dict[str, Any]:
    files = dir_files(stage / f"step_{step}")
    for need in ("adapter_model.safetensors", "adapter_config.json"):
        if need not in files:
            refuse(f"step_{step} holds no {need}")
    return {"step": step, "dir": f"step_{step}", "files": files,
            "adapter_sha256": files["adapter_model.safetensors"]["sha256"],
            "adapter_config_sha256": files["adapter_config.json"]["sha256"]}


def finalize(stage: Path, final: Path, checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    """The staging directory renamed to its final name without replacing anything (safe_outputs.noreplace_rename), and
    every checkpoint file hashed again there: exactly the files recorded, with the same sha256."""
    want = {f"{c['dir']}/{n}": v for c in checkpoints for n, v in c["files"].items()}
    if dir_files(stage) != want:
        refuse(f"the staging directory {stage} does not hold exactly the checkpoint files recorded")
    if os.path.lexists(final):
        refuse(f"{final} exists; adapters are never overwritten")
    try:
        SO.noreplace_rename(stage, final)
    except SO.MovedUnverified:
        pass                                                   # moved: verified under the final name just below
    if dir_files(final) != want:
        refuse(f"{final} does not hold exactly the checkpoint files recorded after the rename")
    return {"files_rehashed_under_final_name": len(want), "equal": True}


# --- training -------------------------------------------------------------------------------------------------------

def prepare_model(src: str, torch: Any) -> Any:
    """train_lora.py lines 181-192: the fp16 base, gradient checkpointing, the LoRA adapter, fp32 trainable tensors.
    The model's config._name_or_path and name_or_path are set to the base repo id before the adapter is attached, as
    Phase 1's (loaded by repo id) were: peft writes them into README.md (card base_model) and adapter_config.json, which
    would otherwise name the local snapshot path (D-172)."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float16, low_cpu_mem_usage=True,
                                                 use_safetensors=True).cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.config._name_or_path = BASE_REPO
    model.name_or_path = BASE_REPO
    lcfg = LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=DROPOUT, target_modules=list(TARGETS),
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.peft_config["default"].base_model_name_or_path = BASE_REPO
    for _n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()                            # adapter weights in fp32
    return model


def train(model: Any, examples: list[list[int]], pad_id: int, padding_side: str, stage: Path, torch: Any, *,
          save_steps: tuple[int, ...] = SAVE_STEPS, lr: float = LR, warmup: int = WARMUP, micro_bs: int = MICRO_BS,
          grad_accum: int = GRAD_ACCUM, log: Callable[[str], None] = print,
          on_save: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """train_lora.py lines 196-256, on token ids: AdamW (weight decay 0), GradScaler, linear warmup then constant,
    grad_accum micro-batches of micro_bs examples in order, full-sequence loss with padding masked, clip 1.0, the
    non-finite rule; the adapter saved at each save step (step 0 before any optimiser step) into `stage`; on_save
    (run(): the step-0 identity check) is called with each checkpoint's entry right after it is saved."""
    max_step = max(save_steps)
    if len(examples) < max_step * micro_bs * grad_accum:
        refuse(f"{len(examples)} examples for {max_step} steps of {micro_bs} x {grad_accum}")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")
    checkpoints: list[dict[str, Any]] = []

    def save(step: int) -> None:
        model.save_pretrained(stage / f"step_{step}")
        checkpoints.append(checkpoint_entry(stage, step))
        log(f"  saved step {step} sha {checkpoints[-1]['adapter_sha256'][:12]}")
        if on_save is not None:
            on_save(checkpoints[-1])

    if 0 in save_steps:
        save(0)
    losses: list[dict[str, Any]] = []
    nonfinite = 0
    cursor = 0
    model.train()
    t0 = time.time()
    for step in range(1, max_step + 1):
        cur_lr = lr * min(1.0, step / max(warmup, 1))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        step_loss, n_label = 0.0, 0
        for _ in range(grad_accum):
            batch = [examples[cursor + j] for j in range(micro_bs)]
            cursor += micro_bs
            ids, mask = pad_batch(batch, pad_id, padding_side)
            input_ids = torch.tensor(ids, dtype=torch.long)
            attention_mask = torch.tensor(mask, dtype=torch.long)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            n_label += int((attention_mask[:, 1:] == 1).sum())
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(),
                             labels=labels.cuda()).loss / grad_accum
            if not torch.isfinite(loss):
                nonfinite += 1
                continue
            scaler.scale(loss).backward()
            step_loss += float(loss)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, CLIP_NORM)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        losses.append({"step": step, "loss": step_loss, "lr": cur_lr, "grad_scale_after": float(scaler.get_scale()),
                       "n_label_tokens": n_label})
        if nonfinite > NONFINITE_ABORT * grad_accum:
            refuse(f"{nonfinite} non-finite micro-batch losses by step {step} (train_lora.py line 247's limit)")
        if step % 5 == 0 or step == 1:
            log(f"  step {step}/{max_step}  loss {step_loss:.4f}  {(time.time() - t0) / step:.1f}s/step  "
                f"nonfinite {nonfinite}")
        if step in save_steps:
            save(step)
    return {"checkpoints": checkpoints, "loss_curve": losses, "nonfinite_microbatch_losses": nonfinite,
            "n_examples_consumed": cursor, "elapsed_s": round(time.time() - t0, 1)}


def determinism_flags(torch: Any) -> dict[str, Any]:
    return {"torch.are_deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
            "torch.backends.cudnn.deterministic": bool(torch.backends.cudnn.deterministic),
            "torch.backends.cudnn.benchmark": bool(torch.backends.cudnn.benchmark),
            "torch.get_num_threads": int(torch.get_num_threads()),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "note": "as Phase 1: none of these is set by this trainer; a GPU training run is not in general bit-"
                    "reproducible (non-deterministic backward kernels), so the flags are recorded, not a claim"}


def code_files() -> list[str]:
    """CODE_FILES plus every src module loaded (D-161; extract_p2.code_files' rule)."""
    src = Path(__file__).resolve().parent
    loaded = {f"src/{Path(f).name}" for f in (getattr(m, "__file__", None) for m in list(sys.modules.values()))
              if f and Path(f).suffix == ".py" and Path(f).resolve().parent == src}
    return sorted(set(CODE_FILES) | loaded)


# --- the run --------------------------------------------------------------------------------------------------------

def read_json_result(path: Path, what: str) -> Any:
    """A results file read as an input (provenance.read_result, so it is listed with its sha256); a missing file or
    one that is not JSON refuses with a message."""
    path = Path(path)
    if not path.is_file():
        refuse(f"{what} {path.name} is not on disk")
    try:
        return read_result(path)
    except (ValueError, UnicodeDecodeError) as e:
        refuse(f"{what} {path.name} is not valid JSON ({type(e).__name__})")


def host_strings(load_path: str) -> dict[str, str]:
    """The strings no checkpoint file may hold (D-172): the base's local load path as given and resolved, the home
    directory as given and resolved, and its last component (the account name, when 5 characters or more)."""
    home = str(Path.home())
    out = {"the base's local load path": load_path, "the base's local load path, resolved": os.path.realpath(load_path),
           "the home directory": home, "the home directory, resolved": os.path.realpath(home)}
    out = {k: v for k, v in out.items() if v and len(v) >= 4}
    if len(Path(home).name) >= 5:
        out["the account name"] = Path(home).name
    return out


def check_no_host_strings(stage: Path, needles: dict[str, str]) -> dict[str, Any]:
    """Every file under `stage` read whole; none may hold any of `needles` (UTF-8 bytes). The refusal names the file
    and which kind of string, never the string."""
    hits, n = [], 0
    for p in sorted(Path(stage).rglob("*")):
        if p.is_file():
            n += 1
            data = p.read_bytes()
            hits += [(p.relative_to(stage).as_posix(), what) for what, v in needles.items() if v.encode("utf-8") in data]
    if hits:
        refuse(f"checkpoint files hold host strings (D-172): {hits[:10]}")
    return {"files_checked": n, "strings_checked": sorted(needles), "found": 0}


def step0_identity(results_dir: Path, corpus: str, seed: int, entry: dict[str, Any],
                   phase1_step0: str | None) -> dict[str, Any]:
    """Step 0 is saved before any optimiser step (B = 0, A from the seeded initialisation), so for one seed it is the
    same bytes for every corpus: every sibling record of this seed on disk (results/phase2/lora_train_p2__<other>__s
    <seed>.json) must name the same step-0 adapter_model.safetensors sha256, else this refuses. Whether seed 0's step 0
    equals Phase 1's is recorded, not checked."""
    sib = []
    for other in sorted(CORPORA):
        p = Path(results_dir) / "phase2" / record_name(other, seed)
        if other == corpus or not p.is_file():
            continue
        rec = read_json_result(p, "the sibling record")
        c0 = [c for c in ((rec.get("checkpoints") if isinstance(rec, dict) else None) or [])
              if isinstance(c, dict) and c.get("step") == 0]
        if len(c0) != 1 or not isinstance(c0[0].get("adapter_sha256"), str):
            refuse(f"{p.name} names no single step-0 checkpoint")
        rel = PV._repo_rel(p)
        same = c0[0]["adapter_sha256"] == entry["adapter_sha256"]
        libs = rec.get("library_versions_short")
        sib.append({"file": rel, "sha256": PV._RESULT_READS[rel], "corpus": other,
                    "step0_adapter_sha256": c0[0]["adapter_sha256"], "equal": same, "libraries": libs})
        if not same:
            refuse(f"the step-0 adapter {entry['adapter_sha256'][:12]} differs from {p.name}'s "
                   f"{c0[0]['adapter_sha256'][:12]} for seed {seed}: step 0 must be the same bytes for every corpus of a "
                   f"seed (that record's libraries: {libs})")
    return {"adapter_sha256": entry["adapter_sha256"], "adapter_config_sha256": entry["adapter_config_sha256"],
            "files": entry["files"], "siblings_checked": sib,
            "rule": "for one seed, step 0 is the same bytes for every corpus; checked against every sibling record of "
                    "the seed on disk when step 0 is saved",
            "for_the_ingest_tool": "with fewer than three records of this seed on disk, the ingest tool checks step 0 "
                                   "across the three",
            "phase1_step0_adapter_sha256": phase1_step0,
            "equals_phase1_step0": None if seed != 0 or phase1_step0 is None else entry["adapter_sha256"] == phase1_step0,
            "phase1_note": "informational, not a refusal: Phase 1's step 0 (Mistral-7B-v0.1 at seed 0; the same module "
                           "shapes) is this sha256 in its record; the bytes can differ with other torch or peft versions "
                           "(the seeded initialisation of A, the safetensors serialisation) or another use of the torch "
                           "RNG before the initialisation; compared for seed 0 only"}


def _on_term(signum: int, frame: Any) -> None:
    raise SystemExit(f"REFUSING: signal {signum} (SIGTERM) received; the run stops through its failure path")


def leftovers(stage: Path, final: Path, checkpoints: list[dict[str, Any]] | None, rec_path: Path) -> list[str | None]:
    """What a failed run leaves, renamed aside where it is this run's and unrecorded (src/safe_outputs.py)."""
    out: list[str | None] = []
    if os.path.lexists(stage):
        out.append(f"{stage} (the staging directory, never given the adapters' name; left in place)")
    elif checkpoints is not None and os.path.lexists(final):
        if os.path.lexists(rec_path):
            out.append(f"{final} (left in place: the record {rec_path.name} exists)")
        else:
            want = {f"{c['dir']}/{n}": v for c in checkpoints for n, v in c["files"].items()}
            out.append(SO.rename_aside(final, None, matches=lambda d: dir_files(d) == want))
    if os.path.lexists(rec_path):
        out.append(f"{rec_path} (the record; it exists: written by this call, or by another)")
    return out


def preflight(args: Any, *, root: Path = ROOT, results_dir: Path = RESULTS, require_nvme: bool = True
              ) -> dict[str, Any]:
    """Every check that needs neither the tokenizer nor torch, in this order: the arguments; the record and the
    adapters' final directory do not exist; the output directory; the Phase 1 record; the config check and the
    tokenizer record; the fetch record; the corpus snapshot and its rows; the frozen splits; the selection and the
    Phase 1 exclusion replay."""
    if args.corpus not in CORPORA:
        refuse(f"--corpus {args.corpus!r} is not one of {sorted(CORPORA)}")
    if isinstance(args.seed, bool) or args.seed not in SEEDS:
        refuse(f"--seed {args.seed!r}: the LoRA seeds are {list(SEEDS)} (D-303 (7))")
    spec = CORPORA[args.corpus]
    results_dir = Path(results_dir)
    rec_path = results_dir / "phase2" / record_name(args.corpus, args.seed)
    if os.path.lexists(rec_path):
        refuse(f"{rec_path.name} exists; this run is recorded and results are never overwritten")
    out_dir, out_info = check_out_dir(Path(args.out_dir), root, require_nvme)
    name = run_name(args.corpus, args.seed)
    if os.path.lexists(out_dir / name):
        refuse(f"{out_dir / name} exists; adapters are never overwritten")
    p1_path = results_dir / spec["phase1_record"]
    p1 = read_json_result(p1_path, "the Phase 1 record")
    p1_rel = PV._repo_rel(p1_path)
    stem = Path(spec["phase1_record"]).stem
    p1_block = {"file": p1_rel, "sha256": PV._RESULT_READS[p1_rel], **check_phase1_record(p1, args.corpus),
                "superseded_under_d161": {
                    "superseded": stem in SP.CODE_PROVENANCE_UNVERIFIABLE,
                    "reason": SP.CODE_PROVENANCE_UNVERIFIABLE.get(stem), "source": "src/superseded_provenance.py",
                    "use_here": "only the written statement of Phase 1's recipe and corpus revision: every value is "
                                "compared (values_equal), the corpus is fetched again with every file hashed, and no "
                                "Phase 1 result is used"}}
    rev = p1_block["dataset_revision"]
    cc_path, st_path = results_dir / "phase2_config_check.json", results_dir / "phase2" / "tokenizer_static.json"
    cc = read_json_result(cc_path, "the config check record")
    static = read_json_result(st_path, "the tokenizer record")
    st = check_base_revision(cc, static)
    f_path = fetch_record_path(results_dir, args.corpus, rev)
    if not f_path.is_file():
        refuse(f"no fetch record {f_path.name}; run src/fetch_dataset_snapshot.py --lora-corpus {args.corpus} first")
    frec = read_json_result(f_path, "the fetch record")
    expected = check_fetch_record(frec, args.corpus, rev, p1_rel, p1_block["sha256"])
    org, rname = spec["repo"].split("/")
    hub = Path(os.path.realpath(args.local_hub))
    snap = hub / f"datasets--{org}--{rname}" / "snapshots" / rev
    if require_nvme:
        mnt, src = mount_of(snap)
        if not src.startswith(NVME_SOURCE_PREFIX):
            refuse(f"the corpus snapshot {snap} is on {src or 'an unknown device'}, not NVMe")
    files = verify_snapshot(snap, expected)
    layout = check_layout(args.corpus, files, (snap / "README.md").read_bytes() if "README.md" in files else None)
    rows, read = read_rows(args.corpus, snap, expected)
    by_dir, by_file, frozen_block = frozen_ids(root)
    sel = select_examples(rows, args.corpus, args.seed, by_dir, by_file=by_file)
    p1n, replay = p1_block["n_excluded_for_frozen_overlap"], sel["n_excluded_data_frozen_only"]
    if args.corpus in REPLAY_REQUIRED and p1n != replay:
        refuse(f"Phase 1's exclusion rule (its prompt against data/frozen alone) excludes {replay} rows here, and "
               f"{spec['phase1_record']} holds {p1n}: the rows read are not Phase 1's, or data/frozen has changed")
    sel["phase1_exclusion_replay"] = {
        "n_excluded_data_frozen_only": replay, "phase1_record_n_excluded": p1n,
        "equal": None if p1n is None else p1n == replay,
        "role": "Phase 1's rule over the same rows, beside its record's count: for hh and pku a different count "
                "refuses (the real-data check that the readers give Phase 1's rows); alpaca's record, which predates "
                "--exclude-frozen, holds none"}
    return {"spec": spec, "corpus": args.corpus, "seed": args.seed, "name": name, "rec_path": rec_path,
            "out_dir": out_dir, "out_info": out_info, "phase1": p1_block, "static_entry": st,
            "config_check": {"file": PV._repo_rel(cc_path), "sha256": PV._RESULT_READS[PV._repo_rel(cc_path)],
                             "base_sha": BASE_REV},
            "tokenizer_static": {"file": PV._repo_rel(st_path), "sha256": PV._RESULT_READS[PV._repo_rel(st_path)]},
            "corpus_source": {"repo": spec["repo"], "revision": rev, "data_dir": spec["data_dir"],
                              "fetch_record": {"file": PV._repo_rel(f_path),
                                               "sha256": PV._RESULT_READS[PV._repo_rel(f_path)]},
                              "snapshot": f"datasets--{org}--{rname}/snapshots/{rev}", "local_hub": str(hub),
                              "snapshot_files_sha256": files, "layout": layout, "train_files_read": read,
                              "reader": spec["reader"], "offline": "read from local files only; no network, no "
                                                                   "datasets library"},
            "frozen": frozen_block, "selection": sel}


def run(args: Any, **kw: Any) -> tuple[Path, dict[str, Any]]:
    """_run under a SIGTERM handler that raises SystemExit, so a lane's TERM (scripts/queue_runner.sh) takes the
    failure path (the staging directory left and named, an unrecorded final directory renamed aside) instead of ending
    the process with nothing named; the previous handler is restored afterwards. The handler does nothing else, so the
    runner's TERM-to-KILL window is used as before."""
    main = threading.current_thread() is threading.main_thread()
    prev = signal.signal(signal.SIGTERM, _on_term) if main else None
    try:
        return _run(args, **kw)
    finally:
        if main:
            signal.signal(signal.SIGTERM, prev)


def _run(args: Any, *, root: Path = ROOT, results_dir: Path = RESULTS, require_nvme: bool = True,
         load_from: str | None = None, allow_no_commit: bool = False) -> tuple[Path, dict[str, Any]]:
    t_start, started = time.time(), utc_now()
    code0 = P.code_block(code_files(), allow_no_commit=allow_no_commit)
    ctx = preflight(args, root=root, results_dir=results_dir, require_nvme=require_nvme)
    sel = ctx["selection"]
    snap_dir, weights = XP.snapshot(BASE_REPO, BASE_REV, load_from, require_nvme, root)
    weight_names = [n for n in weights["files"] if n.endswith(".safetensors")]
    snap_check = check_snapshot_files(snap_dir, ctx["static_entry"], weight_names, blobs=load_from is None)
    tok = TC.load_tokenizer(snap_dir)
    tok_block = check_tokenizer(tok, ctx["static_entry"])
    t0 = time.time()
    ex = build_examples(tok, sel["messages"], sel["used"])
    tokenize_s = time.time() - t0
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token                          # train_lora.py lines 179-180
    pad_id = int(tok.pad_token_id)

    import torch
    import peft as _peft
    import transformers as _tf

    if not torch.cuda.is_available():
        refuse("no CUDA device; Phase 1's recipe is fp16 autocast on the GPU")
    torch.manual_seed(ctx["seed"])                             # train_lora.py line 144, before the model is loaded
    print(f"train_lora_p2: {ctx['corpus']} seed {ctx['seed']} -> {ctx['out_dir'] / ctx['name']}; {len(ex['ids'])} "
          f"examples, {ex['n_truncated']} truncated, {ex['plain_text_path']['n_examples']} on the plain-text path",
          flush=True)
    t0 = time.time()
    model = prepare_model(str(snap_dir), torch)
    model_load_s = time.time() - t0
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable != ctx["phase1"]["trainable_params"]:
        refuse(f"{n_trainable} trainable parameters, not the Phase 1 record's {ctx['phase1']['trainable_params']}")
    stage = ctx["out_dir"] / f"{ctx['name']}.partial.{os.getpid()}.{os.urandom(4).hex()}"
    final = ctx["out_dir"] / ctx["name"]
    step0: dict[str, Any] = {}

    def on_save(entry: dict[str, Any]) -> None:
        if entry["step"] == 0:                                 # seconds after the start, not after the training
            step0.update(step0_identity(results_dir, ctx["corpus"], ctx["seed"], entry,
                                        ctx["phase1"]["step0_adapter_sha256"]))

    tr: dict[str, Any] | None = None
    stage.mkdir(parents=True)
    try:
        if require_nvme:
            mnt, src = mount_of(stage)
            if not src.startswith(NVME_SOURCE_PREFIX):
                refuse(f"the staging directory {stage} is on {src or 'an unknown device'}, not NVMe")
        torch.cuda.reset_peak_memory_stats()
        tr = train(model, ex["ids"], pad_id, tok.padding_side, stage, torch, on_save=on_save)
        steps = [c["step"] for c in tr["checkpoints"]]
        if steps != list(SAVE_STEPS) or tr["n_examples_consumed"] != N_NEEDED or not step0:
            refuse(f"checkpoints at {steps} after {tr['n_examples_consumed']} examples, not {list(SAVE_STEPS)} after "
                   f"{N_NEEDED}")
        peak = int(torch.cuda.max_memory_allocated())
        host_check = check_no_host_strings(stage, host_strings(str(snap_dir)))
        if P.code_block(code_files(), allow_no_commit=allow_no_commit)["sha256"] != code0["sha256"]:
            refuse("the code changed during the run; nothing is named or recorded")
        shas = [c["adapter_sha256"] for c in tr["checkpoints"]]
        config = {"repo": BASE_REPO, "rev": BASE_REV, "corpus": ctx["corpus"], "seed": ctx["seed"],
                  "name": ctx["name"], **PHASE1_CONFIG, "weight_decay": WEIGHT_DECAY, "clip_grad_norm": CLIP_NORM,
                  "nonfinite_abort_after": NONFINITE_ABORT * GRAD_ACCUM, "n_needed": N_NEEDED,
                  "dataset": ctx["spec"]["repo"], "dataset_revision": ctx["corpus_source"]["revision"],
                  "data_dir": ctx["spec"]["data_dir"]}
        sel_rec = {k: v for k, v in sel.items() if k not in ("messages", "used", "contamination")}
        manifest = [{"path": f"{c['dir']}/{n}", "bytes": v["bytes"], "sha256": v["sha256"]}
                    for c in tr["checkpoints"] for n, v in sorted(c["files"].items())]
        record: dict[str, Any] = {
            **provenance_block("lora_train_p2"),
            "code": code0,
            "decision": DECISION,
            "preregistration": "D-343 (2): Phase 1's recipe (D-106) on Mistral-7B-Instruct-v0.2 @ 63a8b081, seeds 0-2 "
                               "(D-303 (7)), examples from the base's chat template (D-303 (f)), plain-text special "
                               "strings (D-324 (5)(v)), HH as alternating turns, pinned corpora with every file hashed, "
                               "adapters on NVMe (D-307 (2)); D-344 (4): a conversation with any frozen user turn is "
                               "excluded",
            "config": config, "config_hash": canonical_hash(config),
            "recipe_sources": {"rank/alpha/dropout/targets/lr/warmup/seq_len/micro_bs/grad_accum/save_steps":
                               "train_lora.py argparse defaults (lines 121-131) = the Phase 1 record's config (checked)",
                               "weight_decay 0.0, clip 1.0, GradScaler, fp16 autocast, fp32 adapter, gradient "
                               "checkpointing": "train_lora.py lines 181-198, 234-245", "seeds": "D-303 (7)",
                               "base revision": "results/phase2_config_check.json (D-303 (8))"},
            "phase1_record": ctx["phase1"],
            "base": {"repo": BASE_REPO, "revision": BASE_REV, "config_check": ctx["config_check"],
                     "snapshot": weights, "snapshot_check": snap_check},
            "template": {"tokenizer_static": {**ctx["tokenizer_static"], "revision": BASE_REV}, **tok_block,
                         "input_format": "tok.apply_chat_template(messages, tokenize=True, add_generation_prompt="
                                         "False): the template writes BOS once and EOS after each assistant message",
                         "special_strings_rule": {"rule": TC.PLAIN_RULE, "extension": PLAIN_EXTENSION,
                                                  "added_tokens": TC._added(tok)},
                         "pad_token_used": str(tok.pad_token), "pad_id_used": pad_id},
            "corpus_source": ctx["corpus_source"],
            "message_rule": {"alpaca": "user = instruction.strip() + ('\\n\\n### Input:\\n' + input.strip() if input); "
                                       "assistant = output.strip()",
                             "hh": "the 'chosen' transcript split at '\\n\\nHuman:' / '\\n\\nAssistant:' into stripped "
                                   "turns; dropped unless it starts with Human, alternates and ends with Assistant",
                             "pku": "user = prompt.strip(); assistant = response_<safer_response_id>.strip()"}[
                                 ctx["corpus"]],
            "frozen_exclusion": ctx["frozen"],
            "selection": sel_rec,
            "n_examples_used": len(sel["used"]), "n_excluded_for_frozen_overlap": sel["n_excluded"],
            "n_dropped_not_alternating": sel["n_dropped"],
            "contamination": sel["contamination"],
            "used_row_indices": sel["used"],
            "examples": {k: v for k, v in ex.items() if k != "ids"},
            "n_truncated": ex["n_truncated"],
            "loss": "causal LM loss on the full sequence; labels = input_ids with padded positions set to -100 "
                    "(train_lora.py lines 231-232)",
            "padding": f"{tok.padding_side}, to the longest row of each micro-batch, pad id {pad_id} (EOS), attention "
                       "mask, no position ids (as tok(texts, padding=True) in Phase 1)",
            "trainable_params": n_trainable,
            "training": {"nonfinite_microbatch_losses": tr["nonfinite_microbatch_losses"],
                         "n_examples_consumed": tr["n_examples_consumed"], "elapsed_s": tr["elapsed_s"],
                         "cuda_max_memory_allocated_bytes": peak, "loss_curve": tr["loss_curve"]},
            "checkpoints": tr["checkpoints"],
            "checkpoint_adapter_sha256_distinct": len(set(shas)) == len(shas),
            "step0_identity": step0,
            "host_strings_check": host_check,
            "adapters": {**ctx["out_info"], "run_dir": ctx["name"],
                         "next": "a separate tool ingests them into data/lora (444) before any extraction (D-307 (2))"},
            "ingest_manifest": {"out_dir": str(ctx["out_dir"]), "run_dir": ctx["name"],
                                "checkpoint_dirs": [c["dir"] for c in tr["checkpoints"]], "files": manifest,
                                "n_files": len(manifest), "total_bytes": sum(x["bytes"] for x in manifest),
                                "rule": "what the ingest tool needs (D-307 (2)): <out_dir>/<run_dir>/ copied to "
                                        "data/lora/<run_dir>/ (444), every file listed here checked for its size and "
                                        "sha256 before any extraction reads it; extract_p2 takes each step's "
                                        "adapter_sha256 and adapter_config_sha256 from 'checkpoints'"},
            "lib_versions": XP.lib_versions("cuda"), "bit_environment": XP.bit_environment(),
            "determinism": determinism_flags(torch),
            "attn_implementation": getattr(model.config, "_attn_implementation", None),
            "library_versions_short": {"torch": torch.__version__, "transformers": _tf.__version__,
                                       "peft": _peft.__version__},
            "timing": {"started_utc": started, "finished_utc": utc_now(), "wall_s": round(time.time() - t_start, 3),
                       "tokenize_s": round(tokenize_s, 3), "model_load_s": round(model_load_s, 3)},
        }
        record["input_files_sha256"] = dict(PV._RESULT_READS)  # D-297 (1): every results file read, with its sha256
        problems = PV.input_identity_problems(record)
        if problems:
            refuse(f"D-297 (1): {problems}")
        record["adapters"]["final_check"] = finalize(stage, final, tr["checkpoints"])
        digest = write_results(ctx["rec_path"], record, exclusive=True)
    except BaseException as e:
        msg = e.code if isinstance(e, SystemExit) else f"{type(e).__name__}: {e}"
        raise SO.failure(str(msg), leftovers(stage, final, None if tr is None else tr["checkpoints"],
                                             ctx["rec_path"])) from None
    print(f"\nAdapters: {final}\nWritten: {PV._repo_rel(ctx['rec_path'])}\nsha256:  {digest}", flush=True)
    return ctx["rec_path"], record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase 2 LoRA dose series (D-343 (2)): one corpus, one seed, Phase 1's "
                                             "recipe on Mistral-7B-Instruct-v0.2")
    ap.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    ap.add_argument("--seed", required=True, type=int, help="0, 1 or 2 (D-303 (7))")
    ap.add_argument("--out-dir", required=True, help="NVMe directory outside the repository; the adapters go to "
                                                     "<out-dir>/<run name>/step_<N>/")
    ap.add_argument("--local-hub", required=True, help="the NVMe hub directory src/fetch_dataset_snapshot.py "
                                                       "--lora-corpus fetched the corpus into")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    args = parse_args(argv)
    if os.environ.get("HF_HUB_OFFLINE") != "1":                # before anything loads: local files only
        refuse("HF_HUB_OFFLINE is not 1; the base is read only from the NVMe Hugging Face cache")
    assert_hf_env("train_lora_p2.py")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
