"""Phase 2 tokenizer check (D-303 (a), (e); docs/phase2_design.md revision 3, sections 2 and 5).

Two modes. Each writes one record under results/phase2/ and never overwrites one.

static   runs after src/phase2_config_check.py and reads results/phase2_config_check.json through
         provenance.read_result.
    Which models: every model of every pair the config check kept (hidden size and layer count equal). The commit is
    the Phase 1 pinned revision for the models Phase 1 extracted (PHASE1_MODELS; D-303 (8), src/input_registry.py)
    and the config check's resolved commit for every other model. Both are recorded. Whenever the extraction records
    name a revision for a model, it must be the one used, or the run refuses.
    What is fetched: the commit's file list (https://huggingface.co/api/models/<repo>/revision/<commit>?blobs=true)
    names the files, and ONLY these are downloaded, by commit, into a scratch directory (see DELETION below):
    - tokenizer.json, tokenizer_config.json, special_tokens_map.json, tokenizer.model;
    - chat_template.jinja, chat_template.json, additional_chat_templates/*;
    - added_tokens.json, vocab.json, merges.txt;
    - config.json, which AutoTokenizer may need to resolve the tokenizer class.
    Each file is checked against the hash the Hub publishes (its git blob sha1, or its LFS object's sha256). The record
    keeps each file's sha256 and size.
    Loading: transformers.AutoTokenizer.from_pretrained(<that directory>, local_files_only=True,
    trust_remote_code=False), refused unless huggingface_hub is offline.
    Per model, the record holds:
    - the vocabulary: len(tok), and the sha256 of its token -> id map;
    - the added tokens (content, id, special), the bos/eos/pad/unk tokens, padding_side and is_fast;
    - the chat template, its sha256, and whether it reads a date or the clock (date_string, strftime_now);
    - the render of the sentinel user message SENTINEL with add_generation_prompt=True: the text and token ids before
      and after the sentinel;
    - the default system text the template injects (injected_system_text below);
    - fed_ids_start_with_bos: whether the ids fed to the model begin with the BOS token (some templates write none);
    - the DOUBLE-BOS check (double_bos below);
    - whether the tokenizer loaded from only the files fetch_weights.ALLOW downloads (what an extractor reading the
      weights' download would load) has the same template, vocabulary, added tokens and sentinel ids.
    Per pair: whether the vocabulary, the added tokens, the special tokens, the template, the injected system text and
    the sentinel render are equal.
    Clock (D-303 (9)), from ONE recorded source:
    - a template that calls strftime_now is rendered at --pin-date;
    - the record's clock.pin_date is that date, or null when no template reads the clock;
    - a clock-reading template with no --pin-date refuses;
    - prompts mode and src/extract_p2.py take the date from clock.pin_date.
    Output: results/phase2/tokenizer_static.json.
    Refusals (nothing is written):
    - the config check record, or any of its pairs or commits, is missing;
    - a registry revision differs from the one used;
    - an instruct model has no chat template;
    - the sentinel cannot be located in a render;
    - a clock-reading template has no pinned date.

prompts  runs after the Phase 2 splits exist. It reads tokenizer_static.json, results/phase2/dataset_manifest_p2.json
         and results/dataset_manifest.json through read_result.
    Inputs checked against their sha256 (CLAUDE.md rule 7). Each file's bytes are read once, hashed, and parsed from
    those same bytes:
    - every split file the Phase 2 manifest lists (axes -> <axis> -> partitions -> <split> -> {file, sha256}). Each
      must lie under data/frozen_p2/;
    - the XSTest file (data/frozen/over_refusal__eval.jsonl, 450 prompts), with its sha256 from
      results/dataset_manifest.json.
    Each model's tokenizer files are fetched again by commit and must match the static record's sha256s. The
    transformers and tokenizers versions must equal the static record's.
    For every item and model: the full templated ids and the user-span ids (user_span below).
    Per pair: the count and the item ids where the full sequence differs, and where the user span differs.
    Per model: the token-length distribution (min, median, max), for the full sequence and the user span.
    The per-item sha256 of both id sequences, the token lengths and the span bounds go to
    results/phase2/tokenizer_prompts_ids.npz. The JSON lists that file with its sha256.
    Output: results/phase2/tokenizer_prompts.json. Both files are staged and verified, then named together
    (commit_outputs). If the JSON cannot be named after the npz was, the npz is renamed aside to
    '<name>.failed.<pid>', never deleted, and the run exits nonzero.
    --axis p2sst (D-338 (4), D-340 (3): the SST-2 non-safety control, before any p2sst extraction): the same check
    over the split files of results/phase2/dataset_manifest_p2sst.json only (its one axis, p2sst; files under
    data/frozen_p2/), every model and pair of tokenizer_static.json as above; XSTest is not re-read (it is in
    tokenizer_prompts.json) and no Phase 1 manifest is read. Output results/phase2/tokenizer_prompts_p2sst.json and
    tokenizer_prompts_p2sst_ids.npz, the record naming its axis. Items are keyed by (file, item_id), so an SST-2 item
    whose item_id equals a WildGuardMix item's (D-340 (3)) is never confused with it. --axis p2wg (the default) is
    unchanged in every input, check and output.

DELETION (the author's standing rules; this script runs as a background queue job): nothing that was given a final
name, and no file over 1 MB, is ever deleted here. Removing those is a human's scripts/safe_rm.sh call, one file per
call. The scratch directory of fetched tokenizer files is cleaned of files of at most 1 MB; larger ones are left and
printed with their safe_rm.sh call (clean_scratch). The static record goes through provenance.write_results when it is
at most 1 MB, and through commit_outputs otherwise.

THE USER SPAN. It is importable, and src/extract_p2.py uses it:

    from phase2_tokenizer_check import SpanError, clock_kwargs, template_text, user_span
    kw = clock_kwargs(template_text(tok), pin_date)                      # {} unless the template reads the clock
    ids, start, end, n_straddle = user_span(tok, text, **kw)            # an instruct model (chat template)
    ids, start, end, n_straddle = user_span(tok, text, raw=True)        # a base checkpoint (raw prompt, section 5)

  ids         The FULL sequence the model is fed. For an instruct model, the chat template renders
              [{"role": "user", "content": text}] with add_generation_prompt=True, and the render is tokenized WITHOUT
              added special tokens. That equals tok.apply_chat_template(..., tokenize=True), which every call checks.
              Feed exactly these ids. Never re-tokenize the rendered text with add_special_tokens=True: that puts a
              second BOS in front of a template that writes bos_token itself (the DOUBLE-BOS hazard of Phase 1's chat
              arm, src/extract_hidden.py). For raw=True, the ids are tok(text) with add_special_tokens=True, which is
              how Phase 1 fed its models.
  start, end  ids[start:end] is the user message: the tokens whose characters overlap the user text. Mean pooling
              (D-303 (d)) averages positions start .. end-1. Last-token pooling reads position len(ids) - 1, where the
              model begins its answer.
  n_straddle  How many of those tokens also cover template characters. Such a token merges a template space,
              newline or punctuation mark with the first or last user character. These tokens are inside the span,
              and this counts them. One exception: an ADDED token (a template control token, such as Phi-3's
              '<|user|>', which strips the whitespace after it) that reaches into the user text has absorbed only
              whitespace there. It belongs to the template and is left out of the span, and it is not counted.
  Method:
  - The sentinel user message is rendered to find the template's prefix and suffix.
  - The text's own render must start with that prefix and end with that suffix.
  - What lies between them must equal the text up to whitespace, since a template may trim the text.
  - The render is tokenized once, with character offsets. This needs a fast tokenizer.
  - Decoding ids[start:end] must give the user text up to whitespace and the tokenizer's own Unicode normal form.
    All whitespace is removed on both sides, and both sides are put in the normal form(s) the tokenizer's backend
    normalizer applies (Qwen2 and Qwen2.5: NFC), so a non-NFC prompt is not a failure. With straddling tokens, the
    user text must be contained in the decode.
  Any failure raises SpanError.
  PLAIN TEXT (D-324 (5)(v), PLAIN_RULE): user_span_plain(tok, text, raw=False, **kw) -> (ids, start, end, n_straddle,
  info) tokenizes special-token strings in the user text as ordinary text; an item without one gets user_span's result
  exactly (info['plain_text_path'] False). src/extract_p2.py uses it; the prompts mode keeps user_span.
  Template kwargs are passed to apply_chat_template. clock_kwargs() gives the ones that pin a clock-reading template's
  date (D-303 (9)). date_string is never passed, so a template's own default date (Llama-3.1: '26 Jul 2024') is what
  renders.

injected_system_text: the first-turn prefix, minus a leading bos_token string, minus its longest common TOKEN suffix
with the opener of a later user turn.
- The later opener is the gap between the assistant message and the last user message in a [user, assistant, user]
  render, less the assistant turn's closer.
- The closer is the text after the assistant message in a [user, assistant] render with add_generation_prompt=False.
- Without that subtraction, a system turn ending in the same closer (ChatML's '<|im_end|>') would lose its closer to
  the opener.
What remains is what the template writes before the first user turn that is not a user-turn opener: a default system
message, or a fixed header like Llama-3.1's 'Cutting Knowledge Date'.

double_bos: the sentinel conversation as apply_chat_template(tokenize=True) gives it, against the render
(tokenize=False) tokenized with add_special_tokens=True. The second is what src/extract_hidden.py's chat arm (lines
~201-216) fed the model in Phase 1. This check records the difference and does not change that file.

TOKEN: as in src/phase2_config_check.py, whose get() makes every request to the Hub here.
- The URL must be on https://huggingface.co, and redirects are refused.
- The token (fetch_weights.load_token, attached by fetch_weights.auth_headers, which attaches it to
  https://huggingface.co only) is requested only for a repo the config check recorded as gated.
- An ungated repo that refuses a file is refused here; it is not retried with the token.
- An LFS object is served by the Hub as a redirect to its storage. That one hop is made explicitly:
  - as a separate request, on a fresh session (no cookie, trust_env=False), with no token at all;
  - to a URL with no backslash, whitespace, control character or user info, https, port 443 at most;
  - to a host under .hf.co or .huggingface.co, which urllib.parse and urllib3 (on requests' prepared URL) must read
    identically;
  - with redirects refused (storage_host, _storage_get).
- token_sent is recorded per request family. The token itself is never printed, logged or written.
- huggingface_hub's endpoint resolution and environment are never used. The node's login shell sets HF_ENDPOINT to an
  HTTP mirror, and nothing here reads it.
- transformers runs with HF_HUB_OFFLINE=1 (checked in huggingface_hub.constants) on local files only, and with
  trust_remote_code=False.
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
import unicodedata
import urllib.parse
import weakref
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import requests
import urllib3

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_weights as FW  # noqa: E402
import paired as P  # noqa: E402
import phase2_config_check as CC  # noqa: E402
from provenance import (_RESULT_READS, _repo_rel, ensure_utf8_stdout, input_identity_problems,  # noqa: E402
                        provenance_block, read_result, sha256_bytes, sha256_file, write_results)

CODE_FILES = ["src/phase2_tokenizer_check.py", "src/phase2_config_check.py", "src/fetch_weights.py",
              "src/hf_env_guard.py", "src/provenance.py", "src/paired.py", "src/input_registry.py"]

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CONFIG_CHECK = RESULTS / "phase2_config_check.json"
FINAL_STATIC = RESULTS / "phase2" / "tokenizer_static.json"
FINAL_PROMPTS = RESULTS / "phase2" / "tokenizer_prompts.json"
FINAL_NPZ = RESULTS / "phase2" / "tokenizer_prompts_ids.npz"
P2_MANIFEST = RESULTS / "phase2" / "dataset_manifest_p2.json"
P1_MANIFEST = RESULTS / "dataset_manifest.json"
SST_AXIS = "p2sst"                                      # D-340 (3): SST-2's prompts are checked in a record of their own
P2SST_MANIFEST = RESULTS / "phase2" / "dataset_manifest_p2sst.json"
FINAL_PROMPTS_SST = RESULTS / "phase2" / "tokenizer_prompts_p2sst.json"
FINAL_NPZ_SST = RESULTS / "phase2" / "tokenizer_prompts_p2sst_ids.npz"
XSTEST = "data/frozen/over_refusal__eval.jsonl"
P2_SPLIT_DIR = "data/frozen_p2/"                        # src/phase2_freeze_splits.py's split directory
HUB = CC.HUB

SENTINEL = "SENTINEL_9f3c"
TURN_A, TURN_B = "SENTINELA_7d21", "SENTINELB_7d21"      # the two-turn render of injected_system_text
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
                   "chat_template.jinja", "chat_template.json", "added_tokens.json", "vocab.json", "merges.txt")
TEMPLATE_DIR = "additional_chat_templates/"
CLASS_FILES = ("config.json",)
STORAGE_HOST_SUFFIXES = (".hf.co", ".huggingface.co")
BAD_URL_CHARS = re.compile(r"[\\\s\x00-\x1f\x7f]")
REDIRECTS = (301, 302, 303, 307, 308)
# The author's standing rule: a file over 1 MB is deleted only by scripts/safe_rm.sh (one per call, by a human), and no
# destructive command runs inside a background task, which this script is (a queue job). Nothing over RM_LIMIT bytes
# is ever deleted here, and nothing that was given a final name.
RM_LIMIT = 1_000_000
UNICODE_FORMS = ("NFC", "NFD", "NFKC", "NFKD")
KEPT = "kept for the next step"                         # phase2_config_check.compare()'s verdict for a kept pair
# D-303 (8): the models Phase 1 extracted keep their Phase 1 pinned revisions (docs/phase2_design.md section 1; the
# OLMo-2 base is the pre side of Phase 1's repair_olmo_base2instruct)
PHASE1_MODELS = ("mistralai/Mistral-7B-Instruct-v0.1", "meta-llama/Meta-Llama-3-8B-Instruct",
                 "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "allenai/OLMo-2-1124-7B-Instruct",
                 "allenai/OLMo-2-1124-7B")
# design section 5: base checkpoints get the raw prompt
BASE_MODELS = ("allenai/OLMo-2-1124-7B",)
PLAIN_RULE = ("D-324 (5)(v): special-token strings in the user content are tokenized as plain text (user_span_plain). "
              "The strings concerned are those of the tokenizer's added tokens (tok.added_tokens_decoder, special or "
              "not), which a fast tokenizer matches atomically anywhere in its input. An item takes the plain-text "
              "path only when today's tokenization parses an added token from non-whitespace characters of the user "
              "text; every other item keeps today's ids exactly. On that path the template's added tokens stay as "
              "they are, and the STRETCH holding the user's special-token strings, from the template's nearest added "
              "token before them to its nearest after them (or the start or end of the render), is tokenized again "
              "as one piece, add_special_tokens=False, by a copy of the tokenizer's backend in which every added "
              "token is special and special tokens are encoded as text (plain_backend: split_special_tokens for "
              "special AND non-special added tokens), so no added token is matched in it. The tokenizer encodes each "
              "stretch between added tokens on its own; a stretch that does not start the text is encoded right after "
              "an anchor added token (ANCHOR, dropped again), so it sits at a non-zero offset after an added token as "
              "in the render (a Metaspace prepend_scheme 'first' prepends only at offset 0), and the stretch encoded "
              "that way with added tokens matched must give today's tokens of it exactly, else the item refuses. A "
              "base checkpoint: the whole raw text, with the tokenizer's own BOS still added by its post-processor. "
              "The span is recomputed by the same rule.")


# --------------------------------------------------------------------------- templates and the user span

class SpanError(ValueError):
    """The user message cannot be located in the templated sequence (raised by frame() and user_span())."""


def _squash(s: str) -> str:
    """The text with every whitespace character removed: the whitespace normalisation of the span checks."""
    return "".join(s.split())


def unicode_forms(tok: Any) -> list[str]:
    """The Unicode normal forms (NFC, NFD, NFKC, NFKD) the tokenizer's backend normalizer applies, in order, read
    from the normalizer's own serialised state (Qwen2 and Qwen2.5: NFC). Empty when there is none."""
    norm = getattr(getattr(tok, "backend_tokenizer", None), "normalizer", None)
    if norm is None:
        return []
    out: list[str] = []

    def walk(n: Any) -> None:
        if isinstance(n, dict):
            if n.get("type") in UNICODE_FORMS:
                out.append(n["type"])
            for m in n.get("normalizers") or []:
                walk(m)

    walk(json.loads(norm.__getstate__()))  # not a result (D-297): the tokenizer's normalizer state
    return out


def _comparable(s: str, forms: list[str]) -> str:
    for f in forms:
        s = unicodedata.normalize(f, s)
    return _squash(s)


def ids_sha256(ids: Any) -> str:
    """sha256 of an id sequence: of the ASCII text of its decimal ids joined by ',' (no dtype or byte order)."""
    return sha256_bytes(",".join(str(int(i)) for i in ids).encode("ascii"))


def template_text(tok: Any) -> str | None:
    """The chat template apply_chat_template uses with no tools and no template name: the string itself, or a template
    dict's 'default' entry (as transformers picks it); None when there is none."""
    t = getattr(tok, "chat_template", None)
    if isinstance(t, dict):
        t = t.get("default")
    return t or None


def clock_refs(template: str | None) -> dict[str, bool]:
    t = template or ""
    return {"references_strftime_now": "strftime_now" in t, "references_date_string": "date_string" in t,
            "reads_clock": "strftime_now" in t}


def clock_kwargs(template: str | None, pin_date: str | None) -> dict[str, Any]:
    """Template kwargs pinning a clock-reading template's date (D-303 (9)). When the template calls strftime_now, the
    call is overridden by one that formats `pin_date` (YYYY-MM-DD, at 00:00:00). A template that does not read the
    clock gets {}. date_string is never passed: a template's own default date is what an extractor passing nothing
    renders."""
    if not clock_refs(template)["reads_clock"]:
        return {}
    if not pin_date:
        raise SystemExit("REFUSING: the chat template reads the clock (strftime_now) and no date is pinned")
    when = datetime.strptime(pin_date, "%Y-%m-%d")
    return {"strftime_now": lambda fmt: when.strftime(fmt)}


def render(tok: Any, messages: list[dict[str, str]], **template_kwargs: Any) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)


def frame(tok: Any, **template_kwargs: Any) -> tuple[str, str]:
    """(prefix, suffix): the template's render of the sentinel user message, split around the sentinel."""
    s = render(tok, [{"role": "user", "content": SENTINEL}], **template_kwargs)
    if s.count(SENTINEL) != 1:
        raise SpanError(f"the sentinel appears {s.count(SENTINEL)} times in the template's render (want exactly 1)")
    i = s.index(SENTINEL)
    return s[:i], s[i + len(SENTINEL):]


def _encode(tok: Any, text: str, raw: bool, template_kwargs: dict[str, Any]) -> tuple[str, int, int, list, list]:
    """(full, cs, ce, ids, offsets): the text as it is fed (the render, or the raw text), the user text's character
    range full[cs:ce] in it, and its ids and character offsets as tokenized TODAY (special-token strings parsed)."""
    if not getattr(tok, "is_fast", False):
        raise SpanError("user_span needs a fast tokenizer (character offsets)")
    if raw:
        full, cs, ce = text, 0, len(text)
        enc = tok(full, add_special_tokens=True, return_offsets_mapping=True)
    else:
        msgs = [{"role": "user", "content": text}]
        full = render(tok, msgs, **template_kwargs)
        prefix, suffix = frame(tok, **template_kwargs)
        if not (full.startswith(prefix) and full.endswith(suffix) and len(full) >= len(prefix) + len(suffix)):
            raise SpanError("the render of the text does not share the sentinel render's prefix and suffix")
        cs, ce = len(prefix), len(full) - len(suffix)
        if _squash(full[cs:ce]) != _squash(text):
            raise SpanError("the rendered user region differs from the text beyond whitespace")
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        check = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **template_kwargs)
        if [int(i) for i in check] != [int(i) for i in enc["input_ids"]]:
            raise SpanError("apply_chat_template(tokenize=True) gives other ids than the render tokenized without "
                            "added special tokens")
    return (full, cs, ce, [int(i) for i in enc["input_ids"]],
            [(int(a), int(b)) for a, b in enc["offset_mapping"]])


def added_ids(tok: Any) -> set[int]:
    """The ids of the tokenizer's added tokens (tok.added_tokens_decoder: special and not): the strings a fast tokenizer
    matches atomically wherever they occur in the text it encodes."""
    return {int(i) for i in (getattr(tok, "added_tokens_decoder", None) or {})}


def _hit(ids: list, offs: list, added: set[int], cs: int, ce: int) -> list[int]:
    # An added token (a template control token such as Phi-3's '<|user|>', which strips the whitespace after it) that
    # reaches into the user text has absorbed only whitespace there: it is the template's, so it is left out.
    return [k for k, (a, b) in enumerate(offs)
            if b > a and a < ce and b > cs and not (ids[k] in added and (a < cs or b > ce))]


def _span(tok: Any, full: str, cs: int, ce: int, ids: list, offs: list) -> tuple[int, int, int]:
    """(start, end, n_straddle) of the user text in these ids (module docstring, THE USER SPAN), decode-checked."""
    hit = _hit(ids, offs, added_ids(tok), cs, ce)
    if not hit:
        raise SpanError("no token overlaps the user text")
    start, end = hit[0], hit[-1] + 1
    n_straddle = sum(1 for a, b in offs[start:end] if b > a and (a < cs or b > ce))
    # both sides in the tokenizer's own Unicode normal form: a normalizing tokenizer (NFC) decodes a non-NFC prompt to
    # its NFC form, which is the same text
    forms = unicode_forms(tok)
    dec = _comparable(tok.decode(ids[start:end], skip_special_tokens=False, clean_up_tokenization_spaces=False), forms)
    want = _comparable(full[cs:ce], forms)
    if (dec != want) if n_straddle == 0 else (want not in dec):
        raise SpanError("decoding the span does not give the user text (whitespace and the tokenizer's Unicode "
                        "normal form aside)")
    return start, end, n_straddle


def user_span(tok: Any, text: str, raw: bool = False, **template_kwargs: Any) -> tuple[list[int], int, int, int]:
    """(ids, start, end, n_straddle): the full sequence fed to the model, and the user message's tokens ids[start:end].
    See the module docstring (THE USER SPAN) for the definition. Raises SpanError when the span cannot be located.
    Special-token strings in the text are parsed as the tokenizer parses them (the rule before D-324 (5)(v); the
    prompts mode keeps it). src/extract_p2.py uses user_span_plain."""
    full, cs, ce, ids, offs = _encode(tok, text, raw, template_kwargs)
    start, end, n_straddle = _span(tok, full, cs, ce, ids, offs)
    return ids, start, end, n_straddle


def user_added_positions(tok: Any, full: str, cs: int, ce: int, ids: list, offs: list) -> list[int]:
    """Positions of added tokens parsed from NON-WHITESPACE characters of the user text full[cs:ce] (a template control
    token that absorbed only whitespace there is the template's, as in the span rule). The unknown token counts only
    where the text is its own string ('<unk>'): elsewhere it is the model's output for text it cannot encode; an unknown
    token that is also the BOS, EOS or pad token (OLMo-2's '<|endoftext|>') always counts (unk_exempt)."""
    dec = {int(i): t for i, t in (getattr(tok, "added_tokens_decoder", None) or {}).items()}
    unk = unk_exempt(tok)
    out = []
    for k, (a, b) in enumerate(offs):
        i = ids[k]
        if i not in dec or b <= a or not full[max(a, cs):min(b, ce)].strip():
            continue
        if i == unk and full[a:b].strip() != str(getattr(dec[i], "content", "")).strip():
            continue
        out.append(k)
    return out


def unk_exempt(tok: Any) -> int | None:
    """The unknown token's id when it is only the unknown token, else None: an id that is also the BOS, EOS or pad
    token (OLMo-2's unk is '<|endoftext|>') is a special id wherever it appears."""
    unk = getattr(tok, "unk_token_id", None)
    others = {getattr(tok, f"{n}_token_id", None) for n in ("bos", "eos", "pad")}
    return None if unk is None or unk in others else unk


_BACKENDS: "weakref.WeakKeyDictionary[Any, dict]" = weakref.WeakKeyDictionary()


# a string no prompt holds (checked): added to each backend copy as a NON-special added token, so it is matched even
# where special tokens are encoded as text, and a stretch encoded after it sits at a non-zero offset after an added
# token, exactly as it does in the render (a Metaspace prepend_scheme 'first' prepends only at offset 0 in tokenizers
# 0.22)
ANCHOR = "\u2063gccf\u2063anchor\u2063"


def backend_variant(tok: Any, plain: bool) -> Any:
    """A copy of the tokenizer's backend (tokenizers.Tokenizer), padding and truncation off, with ANCHOR added; one per
    tokenizer object and kind. plain: every added token marked special and special tokens encoded as text, so it matches
    NO added token of the tokenizer and every added token's string is tokenized as ordinary text by the same
    normalizer, pre-tokenizer, model and post-processor (split_special_tokens alone would still match the non-special
    ones: Phi-3's '</s>', OLMo-2's '<|extra_id_0|>')."""
    cache = _BACKENDS.setdefault(tok, {})
    if plain not in cache:
        import tokenizers
        spec = json.loads(tok.backend_tokenizer.to_str())  # not a result (D-297): the tokenizer's own serialisation
        if plain:
            for t in spec.get("added_tokens") or []:
                t["special"] = True
        b = tokenizers.Tokenizer.from_str(json.dumps(spec))
        b.encode_special_tokens = plain
        b.no_padding()
        b.no_truncation()
        b.add_tokens([tokenizers.AddedToken(ANCHOR, special=False, normalized=False, single_word=False, lstrip=False,
                                            rstrip=False)])
        cache[plain] = b
    return cache[plain]


def plain_backend(tok: Any) -> Any:
    """backend_variant(tok, plain=True): no added token of the tokenizer matched (PLAIN_RULE)."""
    return backend_variant(tok, True)


def encode_stretch(tok: Any, plain: bool, full: str, c0: int, c1: int, special: bool) -> tuple[list[int], list]:
    """(ids, offsets in `full`) of full[c0:c1] encoded as the tokenizer encodes that stretch inside `full`: at offset 0
    on its own, elsewhere right after an added token (ANCHOR, dropped again)."""
    b = backend_variant(tok, plain)
    if c0 == 0:
        enc = b.encode(full[c0:c1], add_special_tokens=special)
        return [int(i) for i in enc.ids], [(int(a), int(e)) for a, e in enc.offsets]
    enc = b.encode(ANCHOR + full[c0:c1], add_special_tokens=special)
    k = len(ANCHOR)
    if not enc.ids or enc.ids[0] != b.token_to_id(ANCHOR) or enc.offsets[0] != (0, k):
        raise SpanError("the anchor is not the first token of the stretch's encoding (PLAIN_RULE)")
    return [int(i) for i in enc.ids[1:]], [(int(a) - k + c0, int(e) - k + c0) for a, e in enc.offsets[1:]]


def user_span_plain(tok: Any, text: str, raw: bool = False,
                    **template_kwargs: Any) -> tuple[list[int], int, int, int, dict[str, Any]]:
    """(ids, start, end, n_straddle, info) under PLAIN_RULE (D-324 (5)(v)): special-token strings in the user text are
    tokenized as ordinary text, the template's own special tokens stay special, and an item whose text holds no such
    string gets exactly user_span's ids and span (info['plain_text_path'] False). Raises SpanError as user_span does,
    and (a safety check that should never fire) when an added token is still parsed from the user text afterwards."""
    full, cs, ce, ids, offs = _encode(tok, text, raw, template_kwargs)
    found = user_added_positions(tok, full, cs, ce, ids, offs)
    if not found:
        start, end, n_straddle = _span(tok, full, cs, ce, ids, offs)
        return ids, start, end, n_straddle, {"plain_text_path": False}
    dec = getattr(tok, "added_tokens_decoder", None) or {}
    strings = sorted({str(getattr(dec.get(ids[k]), "content", ids[k])) for k in found})
    if raw:
        # the whole text is the user's: tokenized again with no added token matched, the tokenizer's own added
        # special tokens (BOS) still added by its post-processor
        lo, hi, c0, c1, special = 0, len(ids), 0, len(full), True
    else:
        # the STRETCH holding the user-parsed added tokens: from the template's nearest added token before them to its
        # nearest added token after them (or the start or end of the render). The tokenizer encodes each stretch
        # between added tokens on its own, so tokenizing this stretch again as one piece, with no added token matched,
        # is what the tokenizer gives the text as plain text; the template's added tokens around it stay as they are
        added = added_ids(tok)
        tmpl = [k for k in range(len(ids)) if ids[k] in added and k not in set(found)]
        first, last = min(found), max(found)
        lo = max([k + 1 for k in tmpl if k < first], default=0)
        hi = min([k for k in tmpl if k > last], default=len(ids))
        if any(lo <= k < hi for k in tmpl):
            raise SpanError("a template added token lies between the user text's special-token strings (PLAIN_RULE)")
        c0 = offs[lo - 1][1] if lo > 0 else 0
        c1 = offs[hi][0] if hi < len(ids) else len(full)
        special = False
    if ANCHOR in full:
        raise SpanError("the text holds the stretch anchor string (PLAIN_RULE)")
    # what makes the stretch's re-encoding faithful is the anchor (encode_stretch: a non-zero offset right after an
    # added token, as in the render). This check catches SOME failures of it, not all: encoded the same way with added
    # tokens matched (as today), the stretch must give exactly today's tokens of it. It cannot see a broken anchor when
    # the stretch starts with the user's special-token string (matched mode then never meets the offset-0 prepend);
    # the review of 58ee75e checked every Phase 2 prompt on all eight tokenizers against a renamed-token oracle
    # (0 disagreements; recorded in D-335)
    again, _o = encode_stretch(tok, False, full, c0, c1, special)
    if again != ids[lo:hi]:
        raise SpanError(f"the stretch [{c0}, {c1}) encoded on its own does not give today's tokens of it, so its "
                        "plain-text tokens would not be the tokenizer's (PLAIN_RULE)")
    mid_ids, mid_offs = encode_stretch(tok, True, full, c0, c1, special)
    new_ids = ids[:lo] + mid_ids + ids[hi:]
    new_offs = offs[:lo] + mid_offs + offs[hi:]
    if raw:            # the replaced range reported is the text's tokens: the post-processor's (BOS) stay as today
        lo = next((k for k, (a, b) in enumerate(offs) if b > a), len(ids))
        hi = len(ids) - next((k for k, (a, b) in enumerate(reversed(offs)) if b > a), len(ids))
    left = user_added_positions(tok, full, cs, ce, new_ids, new_offs)
    if left:
        names = sorted({str(getattr(dec.get(new_ids[k]), "content", new_ids[k])) for k in left})
        raise SpanError(f"added token(s) {names} are still parsed from the user text after the plain tokenization "
                        "(PLAIN_RULE)")
    start, end, n_straddle = _span(tok, full, cs, ce, new_ids, new_offs)
    return new_ids, start, end, n_straddle, {
        "plain_text_path": True, "strings": strings, "today_positions": found, "replaced": [lo, hi],
        "n_today": len(ids), "today_ids_sha256": ids_sha256(ids), "today_ids": ids}


def injected_system_text(tok: Any, **template_kwargs: Any) -> dict[str, Any]:
    """The text the template writes before the first user turn that is not a user-turn opener (module docstring)."""
    out: dict[str, Any] = {"text": None, "present": None, "user_turn_opener": None, "assistant_turn_closer": None,
                           "opener_tokens_matched": None, "error": None,
                           "method": "the first-turn prefix, minus a leading bos_token string, minus its longest "
                                     "common TOKEN suffix with a later user turn's opener. The opener is the gap "
                                     "between the assistant message and the last user message of a [user, assistant, "
                                     "user] render, less the assistant turn's closer (the text after the assistant "
                                     "message of a [user, assistant] render, add_generation_prompt=False)"}
    try:
        prefix, _ = frame(tok, **template_kwargs)
        turns = [{"role": "user", "content": TURN_A}, {"role": "assistant", "content": TURN_B}]
        multi = render(tok, turns + [{"role": "user", "content": SENTINEL}], **template_kwargs)
        two = tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=False, **template_kwargs)
    except Exception as e:  # noqa: BLE001 - a template refusing the two-turn render is recorded, not fatal
        out["error"] = f"{type(e).__name__}: {e}"[:300]
        return out
    if multi.count(TURN_B) != 1 or multi.count(SENTINEL) != 1 or multi.index(TURN_B) > multi.index(SENTINEL):
        out["error"] = "the two-turn render does not hold each marker once, in order"
        return out
    gap = multi[multi.index(TURN_B) + len(TURN_B):multi.index(SENTINEL)]
    closer = two[two.index(TURN_B) + len(TURN_B):] if two.count(TURN_B) == 1 else ""
    # the assistant turn's closer (Qwen: '<|im_end|>\n') is not the user opener: without this a system turn ending in
    # the same closer would lose it to the opener
    opener = gap[len(closer):] if closer and gap.startswith(closer) else gap
    bos = str(getattr(tok, "bos_token", None) or "")
    # BOS first, so the match never runs into it (Mistral: '</s> [INST] ' against '<s> [INST] ')
    rest = prefix[len(bos):] if bos and prefix.startswith(bos) else prefix
    enc = tok(rest, add_special_tokens=False, return_offsets_mapping=True)
    r_ids, o_ids = [int(i) for i in enc["input_ids"]], [int(i) for i in tok(opener, add_special_tokens=False)["input_ids"]]
    k = 0
    while k < min(len(r_ids), len(o_ids)) and r_ids[-1 - k] == o_ids[-1 - k]:
        k += 1
    cut = len(rest) if k == 0 else int(enc["offset_mapping"][len(r_ids) - k][0])
    out.update(text=rest[:cut], present=bool(rest[:cut].strip()), user_turn_opener=rest[cut:],
               assistant_turn_closer=closer if closer and gap.startswith(closer) else None, opener_tokens_matched=k)
    return out


def _seq_diff(a: list[int], b: list[int], tok: Any) -> dict[str, Any]:
    p = 0
    while p < min(len(a), len(b)) and a[p] == b[p]:
        p += 1
    s = 0
    while s < min(len(a), len(b)) - p and a[-1 - s] == b[-1 - s]:
        s += 1
    ea, eb = a[p:len(a) - s], b[p:len(b) - s]
    return {"first_difference_at": None if a == b else p,
            "only_in_chat_template": {"ids": ea, "tokens": tok.convert_ids_to_tokens(ea)},
            "only_in_render_then_tokenize": {"ids": eb, "tokens": tok.convert_ids_to_tokens(eb)}}


def double_bos(tok: Any, **template_kwargs: Any) -> dict[str, Any]:
    """The sentinel conversation: apply_chat_template(tokenize=True) against render-then-tokenize with
    add_special_tokens=True (Phase 1's chat arm)."""
    msgs = [{"role": "user", "content": SENTINEL}]
    a = [int(i) for i in tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **template_kwargs)]
    text = render(tok, msgs, **template_kwargs)
    b = [int(i) for i in tok(text, add_special_tokens=True)["input_ids"]]
    c = [int(i) for i in tok(text, add_special_tokens=False)["input_ids"]]
    bos, eos = tok.bos_token_id, tok.eos_token_id
    return {"ids_apply_chat_template": a, "ids_render_then_tokenize_add_special_tokens": b,
            "differ": a != b, "len": {"apply_chat_template": len(a), "render_then_tokenize": len(b)},
            "double_bos": bos is not None and b[:2] == [bos, bos] and a[:2] != [bos, bos],
            "eos_appended": eos is not None and b[-1:] == [eos] and a[-1:] != [eos],
            "render_tokenized_without_special_tokens_equals_chat_template": c == a,
            **_seq_diff(a, b, tok),
            "phase1_chat_arm": "src/extract_hidden.py (chat arm, lines ~201-216) renders with tokenize=False and then "
                               "calls tok(texts) with add_special_tokens left at its default (True): the "
                               "render_then_tokenize sequence here. Recorded, not fixed."}


def _special(tok: Any, name: str) -> dict[str, Any]:
    t, i = getattr(tok, name, None), getattr(tok, f"{name}_id", None)
    return {"token": None if t is None else str(t), "id": None if i is None else int(i)}


def _vocab_sha(tok: Any) -> str:
    text = json.dumps(sorted(([t, int(i)] for t, i in tok.get_vocab().items()), key=lambda x: (x[1], x[0])),
                      ensure_ascii=False)
    return sha256_bytes(text.encode("utf-8"))


def _added(tok: Any) -> list[dict[str, Any]]:
    return [{"id": int(i), "content": str(t.content), "special": bool(t.special)}
            for i, t in sorted(tok.added_tokens_decoder.items())]


def _template_sha(tmpl: str | None) -> str | None:
    return sha256_bytes(tmpl.encode("utf-8")) if tmpl else None


def analyse(tok: Any, repo: str, kind: str, pin_date: str | None) -> dict[str, Any]:
    """The per-model tokenizer record. Refuses (SystemExit) when an instruct model has no chat template, the sentinel
    cannot be located, or the template reads the clock and no date is pinned."""
    raw_t = getattr(tok, "chat_template", None)
    tmpl = template_text(tok)
    kw = clock_kwargs(tmpl, pin_date)
    added = _added(tok)
    out: dict[str, Any] = {
        "class": type(tok).__name__, "is_fast": bool(getattr(tok, "is_fast", False)),
        "vocab_size_len": len(tok), "vocab_size_attr": getattr(tok, "vocab_size", None),
        "vocab_sha256": _vocab_sha(tok),
        "vocab_sha256_method": "sha256 of the UTF-8 JSON list of [token, id] pairs sorted by id, then token",
        "added_tokens": added, "n_added_tokens": len(added),
        "bos": _special(tok, "bos_token"), "eos": _special(tok, "eos_token"),
        "pad": _special(tok, "pad_token"), "unk": _special(tok, "unk_token"),
        "padding_side": getattr(tok, "padding_side", None), "truncation_side": getattr(tok, "truncation_side", None),
        "add_bos_token": getattr(tok, "add_bos_token", None), "add_eos_token": getattr(tok, "add_eos_token", None),
        "model_max_length": getattr(tok, "model_max_length", None), "unicode_normal_forms": unicode_forms(tok),
        "chat_template": {"present": tmpl is not None, "is_dict": isinstance(raw_t, dict),
                          "names": sorted(raw_t) if isinstance(raw_t, dict) else None,
                          "sha256": _template_sha(tmpl),
                          "sha256_method": "sha256 of the UTF-8 template string apply_chat_template uses",
                          **clock_refs(tmpl), "clock_kwargs_passed": sorted(kw),
                          "pinned_date": pin_date if kw else None, "text": tmpl}}
    if kind == "instruct":
        if tmpl is None:
            raise SystemExit(f"REFUSING: {repo} is an instruct model with no chat template; nothing is written")
        try:
            prefix, suffix = frame(tok, **kw)
            ids, s, e, n = user_span(tok, SENTINEL, **kw)
        except SpanError as err:
            raise SystemExit(f"REFUSING: {repo}: the sentinel cannot be located in the render ({err}); "
                             "nothing is written") from None
        mode = "chat template, add_generation_prompt=True, tokenized without added special tokens"
        out["injected_system_text"] = injected_system_text(tok, **kw)
        out["double_bos"] = double_bos(tok, **kw)
    else:
        prefix, suffix = "", ""
        try:
            ids, s, e, n = user_span(tok, SENTINEL, raw=True)
        except SpanError as err:
            raise SystemExit(f"REFUSING: {repo}: the sentinel cannot be located ({err}); nothing is written") from None
        mode = "raw prompt (design section 5: a base checkpoint gets the raw prompt), add_special_tokens=True"
        out["injected_system_text"] = {"text": None, "present": False, "method": "not applicable: raw prompt"}
        out["double_bos"] = ({**double_bos(tok, **kw), "note": "this checkpoint's template is not used in Phase 2"}
                             if tmpl else {"applicable": False, "note": "no chat template; raw prompt"})
    toks = tok.convert_ids_to_tokens(ids)
    bos_id = out["bos"]["id"]
    out["fed_ids_start_with_bos"] = bos_id is not None and ids[:1] == [bos_id]
    out["sentinel"] = {"mode": mode, "message": SENTINEL, "render": prefix + SENTINEL + suffix,
                       "text_before": prefix, "text_after": suffix, "ids": ids, "tokens": toks, "n_tokens": len(ids),
                       "user_span": {"start": s, "end": e, "n_straddle": n},
                       "ids_before": ids[:s], "ids_after": ids[e:], "tokens_before": toks[:s], "tokens_after": toks[e:]}
    return out


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Per pair: what differs between the two tokenizer records."""
    ta, tb = a["tokenizer"], b["tokenizer"]
    aa = {(t["id"], t["content"], t["special"]) for t in ta["added_tokens"]}
    bb = {(t["id"], t["content"], t["special"]) for t in tb["added_tokens"]}

    def listed(s: set) -> list[dict[str, Any]]:
        return [{"id": i, "content": c, "special": sp} for i, c, sp in sorted(s)]

    sa, sb = ta["sentinel"], tb["sentinel"]
    ia, ib = ta["injected_system_text"], tb["injected_system_text"]
    da, db = ta.get("double_bos") or {}, tb.get("double_bos") or {}
    return {"pre": a["repo"], "post": b["repo"], "kinds": {"pre": a["kind"], "post": b["kind"]},
            "vocab_size": {"pre": ta["vocab_size_len"], "post": tb["vocab_size_len"]},
            "vocab_size_equal": ta["vocab_size_len"] == tb["vocab_size_len"],
            "vocab_equal": ta["vocab_sha256"] == tb["vocab_sha256"],
            "added_tokens_equal": aa == bb,
            "added_tokens_only_in_pre": listed(aa - bb), "added_tokens_only_in_post": listed(bb - aa),
            "special_tokens_equal": all(ta[k] == tb[k] for k in ("bos", "eos", "pad", "unk")),
            "padding_side_equal": ta["padding_side"] == tb["padding_side"],
            "template_equal": ta["chat_template"]["sha256"] == tb["chat_template"]["sha256"],
            "injected_system_text_equal": ia.get("text") == ib.get("text"),
            "injected_system_text": {"pre": ia.get("text"), "post": ib.get("text")},
            "sentinel_render_equal": sa["render"] == sb["render"],
            "sentinel_ids_equal": sa["ids"] == sb["ids"],
            "render_then_tokenize_differs": {"pre": da.get("differ"), "post": db.get("differ")},
            "double_bos": {"pre": da.get("double_bos"), "post": db.get("double_bos")}}


# --------------------------------------------------------------------------- fetching, by commit, from the Hub

def git_blob_sha1(data: bytes) -> str:
    """Git's object id: sha1 over 'blob <size>\\0' + content."""
    h = hashlib.sha1()
    h.update(f"blob {len(data)}\0".encode())
    h.update(data)
    return h.hexdigest()


def storage_host(url: Any) -> str:
    """The host of an LFS redirect, or SystemExit. The URL must be printable ASCII with no backslash, whitespace or
    control character, https with no user info and no port but 443, and its host must be the same whether parsed by
    urllib.parse or by urllib3 from requests' prepared URL (what the request would actually reach). The host must lie
    under .hf.co or .huggingface.co. Only the host is ever printed, never the presigned URL."""
    if not isinstance(url, str) or not url or not url.isascii() or BAD_URL_CHARS.search(url):
        raise SystemExit("REFUSING: an LFS redirect whose URL is empty, not ASCII, or holds a backslash, whitespace or "
                         "a control character; nothing is fetched from it")
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    try:
        prepared = urllib3.util.parse_url(requests.Request("GET", url).prepare().url)
        port = parts.port
    except Exception as e:  # noqa: BLE001 - an unparsable redirect is refused
        raise SystemExit(f"REFUSING: an LFS redirect that cannot be parsed ({type(e).__name__})") from None
    if parts.scheme != "https" or prepared.scheme != "https" or "@" in parts.netloc or port not in (None, 443) \
            or (prepared.host or "").lower() != host or not host.endswith(STORAGE_HOST_SUFFIXES):
        raise SystemExit(f"REFUSING: an LFS redirect to {parts.scheme or '?'}://{host or '?'} (urllib3 reads host "
                         f"{(prepared.host or '?').lower()}), which is not the Hub's https storage; nothing is fetched "
                         "from it")
    return host


def _storage_session() -> requests.Session:
    """A FRESH session for the storage hop: no cookie from huggingface.co, no credential from the environment."""
    s = requests.Session()
    s.trust_env = False
    return s


def _storage_get(url: str) -> bytes:
    """The one redirect hop of an LFS object: a SEPARATE request, on a fresh session, with no token, https, to the
    Hub's storage host, redirects refused."""
    host = storage_host(url)
    last = None
    for attempt in range(5):
        s = _storage_session()
        try:
            req = s.prepare_request(requests.Request("GET", url, headers=dict(CC.UA)))
            if any(k.lower() in ("authorization", "cookie") for k in req.headers) or s.auth is not None:
                raise SystemExit("REFUSING: the storage request would carry a credential or a cookie")
            if (urllib3.util.parse_url(req.url).host or "").lower() != host:
                raise SystemExit("REFUSING: the prepared storage request names another host")
            r = s.send(req, timeout=120, allow_redirects=False)
        except requests.RequestException as e:
            last = type(e).__name__
            time.sleep(5 * (attempt + 1))
            continue
        finally:
            s.close()
        if r.status_code == 429 or r.status_code >= 500:
            last = f"HTTP {r.status_code}"
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code != 200:
            raise SystemExit(f"REFUSING: the storage host {host} answered HTTP {r.status_code}")
        return r.content
    raise SystemExit(f"REFUSING: the storage host {host} kept failing ({last})")


def _object(repo: str, rev: str, name: str, gated: bool) -> tuple[bytes, dict[str, Any]]:
    """An LFS object: /resolve on huggingface.co by commit (the token only for a gated repo, redirects refused); a
    redirect to the Hub's storage is followed by _storage_get, with no token."""
    r = CC.get(f"{HUB}/{repo}/resolve/{rev}/{name}", use_token=gated)
    info: dict[str, Any] = {"token_sent": bool(getattr(r, "token_sent", False)), "object_host": "huggingface.co",
                            "storage_request_token_sent": None}
    if r.status_code == 200:
        return r.content, info
    if r.status_code not in REDIRECTS:
        raise SystemExit(f"REFUSING: {repo}@{rev[:12]} {name}: /resolve answered HTTP {r.status_code}")
    loc = r.headers.get("Location") or ""
    data = _storage_get(loc)
    info.update(object_host=(urllib.parse.urlsplit(loc).hostname or "").lower(), storage_request_token_sent=False)
    return data, info


def fetch_file(repo: str, rev: str, name: str, sib: dict[str, Any], gated: bool, dest: Path) -> dict[str, Any]:
    """Fetch one file by commit, check it against the hash the Hub publishes, write it under dest, record it."""
    lfs = sib.get("lfs") or {}
    rec: dict[str, Any] = {"name": name, "lfs": bool(lfs)}
    if lfs:
        want, size = lfs.get("sha256") or lfs.get("oid"), lfs.get("size", sib.get("size"))
        if not want:
            raise SystemExit(f"REFUSING: {repo} {name} is an LFS file with no published sha256")
        data, info = _object(repo, rev, name, gated)
        rec.update(info, hub_check="sha256 the Hub publishes for the LFS object")
    else:
        blob, size = sib.get("blobId") or sib.get("oid"), sib.get("size")
        if not blob:
            raise SystemExit(f"REFUSING: {repo} {name} has no published git blob id")
        r = CC.get(f"{HUB}/{repo}/raw/{rev}/{name}", use_token=gated)
        if r.status_code != 200:
            raise SystemExit(f"REFUSING: {repo}@{rev[:12]} {name}: /raw answered HTTP {r.status_code}"
                             + ("" if gated else " (the repo is not gated, so no token is tried)"))
        data = r.content
        if git_blob_sha1(data) != blob or (size is not None and len(data) != int(size)):
            raise SystemExit(f"HASH MISMATCH: {repo}@{rev[:12]} {name}: {len(data)} bytes, blob sha1 "
                             f"{git_blob_sha1(data)[:16]} != {str(blob)[:16]}")
        rec.update(token_sent=bool(getattr(r, "token_sent", False)), object_host="huggingface.co",
                   storage_request_token_sent=None, hub_check="git blob sha1 the Hub publishes for the file")
        want = None
        pointer = FW.lfs_pointer(data) if len(data) <= FW.POINTER_MAX else None
        if pointer:                       # F-068: an LFS pointer committed as a plain blob; fetch its object
            want, size = pointer
            data, info = _object(repo, rev, name, gated)
            rec.update(object_host=info["object_host"], storage_request_token_sent=info["storage_request_token_sent"],
                       token_sent=rec["token_sent"] or info["token_sent"], f068_pointer=True,
                       hub_check="sha256 named by the committed LFS pointer (F-068)")
    got = sha256_bytes(data)
    if (want and got != want) or (size is not None and len(data) != int(size)):
        raise SystemExit(f"HASH MISMATCH: {repo}@{rev[:12]} {name}: {len(data)} bytes, sha256 {got[:16]} "
                         f"!= {str(want)[:16]}")
    path = dest / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    rec.update(sha256=got, size=len(data))
    return rec


def hub_listing(repo: str, rev: str, gated: bool) -> tuple[dict[str, dict[str, Any]], bool]:
    """The commit's files ({name: sibling}), from https://huggingface.co/api/models/<repo>/revision/<rev>?blobs=true;
    refuses unless the Hub resolves exactly that commit."""
    r = CC.get(f"{HUB}/api/models/{repo}/revision/{rev}?blobs=true", use_token=gated)
    if r.status_code != 200:
        raise SystemExit(f"REFUSING: the file list of {repo}@{rev[:12]} answered HTTP {r.status_code}")
    body = r.json()
    if body.get("sha") != rev:
        raise SystemExit(f"REFUSING: {repo}: asked for commit {rev[:12]}, the Hub listed {str(body.get('sha'))[:12]}")
    return {s["rfilename"]: s for s in body.get("siblings") or []}, bool(getattr(r, "token_sent", False))


def wanted_files(names: Any) -> list[str]:
    return sorted(n for n in names if n in TOKENIZER_FILES or n in CLASS_FILES
                  or (n.startswith(TEMPLATE_DIR) and n.count("/") == 1))


def fetch_tokenizer(repo: str, rev: str, gated: bool, dest: Path) -> dict[str, Any]:
    """Every tokenizer file of the commit (and config.json), fetched and checked; the record of what was fetched."""
    listing, listing_token = hub_listing(repo, rev, gated)
    names = wanted_files(listing)
    if not any(n in names for n in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        raise SystemExit(f"REFUSING: {repo}@{rev[:12]} has no tokenizer.json, tokenizer.model or vocab.json")
    files = [fetch_file(repo, rev, n, listing[n], gated, dest) for n in names]
    return {"listing_token_sent": listing_token, "n_files_in_commit": len(listing), "files": files,
            "token_sent_any": listing_token or any(f["token_sent"] for f in files),
            "not_fetched_by_fetch_weights_allow": [n for n in names
                                                   if not any(fnmatch.fnmatchcase(n, p) for p in FW.ALLOW)]}


def load_tokenizer(local_dir: Path) -> Any:
    """AutoTokenizer from local files only, with the Hub offline (no network inside transformers) and no remote code.
    Refuses when huggingface_hub is not offline: it reads HF_HUB_OFFLINE once, at its first import, so a process that
    imported it earlier without the variable would be online."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    import huggingface_hub.constants as HC
    if not HC.HF_HUB_OFFLINE:
        raise SystemExit("REFUSING: huggingface_hub is not offline (it was imported before HF_HUB_OFFLINE=1 was set); "
                         "set HF_HUB_OFFLINE=1 before starting the process")
    return AutoTokenizer.from_pretrained(str(local_dir), local_files_only=True, use_fast=True, trust_remote_code=False)


def _sentinel_ids(tok: Any, kind: str, pin_date: str | None) -> list[int] | None:
    if kind == "base":
        return [int(i) for i in tok(SENTINEL, add_special_tokens=True)["input_ids"]]
    t = template_text(tok)
    if t is None:
        return None
    return [int(i) for i in tok.apply_chat_template([{"role": "user", "content": SENTINEL}], tokenize=True,
                                                    add_generation_prompt=True, **clock_kwargs(t, pin_date))]


def fetch_weights_equivalence(d: Path, fetched: dict[str, Any], rec: dict[str, Any], kind: str,
                              pin_date: str | None) -> dict[str, Any]:
    """Whether the tokenizer loaded from only the files fetch_weights.ALLOW downloads (the snapshot an extractor
    reading the weights' download would load) has the same template, vocabulary, added tokens and sentinel ids."""
    lacking = fetched["not_fetched_by_fetch_weights_allow"]
    if not lacking:
        return {"lacking": [], "equivalent": True, "note": "every fetched file is one fetch_weights downloads"}
    sub = d.parent / (d.name + "__fetch_weights_allow_only")
    for f in fetched["files"]:
        if f["name"] not in lacking:
            (sub / f["name"]).parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(d / f["name"], sub / f["name"])     # a second name, not a second copy of the bytes
            except OSError:
                shutil.copyfile(d / f["name"], sub / f["name"])
    out: dict[str, Any] = {"lacking": lacking}
    try:
        tok2 = load_tokenizer(sub)
        same = {"template_equal": _template_sha(template_text(tok2)) == rec["chat_template"]["sha256"],
                "vocab_equal": _vocab_sha(tok2) == rec["vocab_sha256"],
                "added_tokens_equal": _added(tok2) == rec["added_tokens"],
                "sentinel_ids_equal": _sentinel_ids(tok2, kind, pin_date) == rec["sentinel"]["ids"]}
    except Exception as e:  # noqa: BLE001 - a snapshot that cannot load or render is the finding, recorded
        return {**out, "loads": False, "equivalent": False, "error": f"{type(e).__name__}: {e}"[:300]}
    return {**out, "loads": True, **same, "equivalent": all(same.values())}


def _libraries() -> dict[str, str | None]:
    from importlib import metadata
    out: dict[str, str | None] = {}
    for dist in ("transformers", "tokenizers", "jinja2", "requests", "numpy"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def _left_note(p: Path) -> str:
    """How a human removes `p`: scripts/safe_rm.sh <path> <bytes> <sha256>, one file per call."""
    try:
        return f"scripts/safe_rm.sh {p} {p.stat().st_size} {sha256_file(p)}"
    except OSError as e:
        return f"{p} (cannot be read: {type(e).__name__})"


def _remove_small_temp(p: Path) -> bool:
    """Unlink `p`, a temporary that never had a final name, only if it is at most RM_LIMIT bytes. True when it is
    gone; a larger one is LEFT (the author's standing rule: a file over 1 MB is deleted only by scripts/safe_rm.sh)."""
    try:
        if not p.exists():
            return True
        if p.is_symlink() or not p.is_file() or p.stat().st_size > RM_LIMIT:
            return False
        p.unlink()
        return True
    except OSError:
        return False


def clean_scratch(root: Path) -> list[str]:
    """After a run, the scratch directory of fetched tokenizer files (never-named temporaries):
    - each file of at most RM_LIMIT bytes is unlinked;
    - directories left empty are removed;
    - every larger file is LEFT and printed with the scripts/safe_rm.sh call that removes it.
    No file over 1 MB is ever deleted here, since this script runs as a background queue job (the author's standing
    rule). Returns the safe_rm.sh lines of what was left."""
    left: list[str] = []
    for p in sorted(Path(root).rglob("*"), key=lambda q: len(q.parts), reverse=True):
        if p.is_dir() and not p.is_symlink():
            with contextlib.suppress(OSError):
                p.rmdir()                                   # only succeeds when empty
        elif not _remove_small_temp(p):
            left.append(_left_note(p))
    with contextlib.suppress(OSError):
        Path(root).rmdir()
    if left:
        print(f"LEFT in the scratch directory {root}: {len(left)} file(s) over 1 MB, never deleted by this job. "
              "Removing each is a human's call, one file per call:\n  " + "\n  ".join(left), file=sys.stderr,
              flush=True)
    return left


def _link(src: Path, dst: Path) -> None:
    os.link(src, dst)                                       # FileExistsError if dst exists: never overwrite


def _set_aside(p: Path) -> Path | None:
    """Rename a file that was already given its final name to '<name>.failed.<pid>', without overwriting: refused when
    that name exists. Returns the new path, or None when it could not be done (the file stays under its name). The
    bytes are never deleted."""
    aside = p.with_name(f"{p.name}.failed.{os.getpid()}")
    if aside.exists() or not p.exists():
        return None
    try:
        os.rename(p, aside)
    except OSError:
        return None
    return aside


def commit_outputs(outputs: list[tuple[Path, bytes]], final: bool) -> list[str]:
    """Write several outputs as one unit, deleting nothing that matters. Returns each output's sha256.
    1. Each output is written to its own temporary (exclusive create, fsync) and checked by size and sha256.
    2. No final name may exist.
    3. Each output is named in order, by a hard link (never an overwrite), and its final name is checked by sha256.
    If naming one fails after earlier outputs were named, those are NOT deleted. Each is renamed aside, without
    overwriting, to '<name>.failed.<pid>' (or left under its name if that cannot be done). The run then exits nonzero,
    naming them and the scripts/safe_rm.sh call that removes each, which is a human's call.
    Afterwards, a temporary name whose bytes were linked to a final name is unlinked (the bytes stay under that name),
    as is a never-named temporary of at most 1 MB. A larger never-named temporary is left and printed."""
    staged: list[tuple[Path, Path, str]] = []               # (temporary, final name, sha256)
    linked: set[Path] = set()
    try:
        for path, data in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.partial.{os.getpid()}.{os.urandom(4).hex()}")
            want = sha256_bytes(data)
            with open(tmp, "xb") as fh:
                staged.append((tmp, path, want))
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            if tmp.stat().st_size != len(data) or sha256_file(tmp) != want:
                raise OSError(f"stage verify: {tmp.name} is not what was written")
        _refuse_outputs([p for p, _ in outputs], final)
        named: list[Path] = []
        for tmp, path, want in staged:
            try:
                _link(tmp, path)
                linked.add(tmp)
                named.append(path)
                if sha256_file(path) != want:
                    raise OSError(f"{path.name} differs from what was staged")
            except BaseException as e:
                kept = []
                for p in named:
                    aside = _set_aside(p)
                    kept.append(f"{p.name} -> {aside.name}" if aside else f"{p.name} (left under its name)")
                    kept[-1] += f"; remove with: {_left_note(aside or p)}"
                raise SystemExit(f"REFUSING: naming {path.name} failed ({type(e).__name__}: {e}). Already named and "
                                 f"NOT deleted: {kept or 'nothing'}. Removing them is a human's scripts/safe_rm.sh "
                                 "call, one file per call") from None
        return [w for _, _, w in staged]
    finally:
        left = []
        for tmp, _, _ in staged:
            if tmp in linked:
                with contextlib.suppress(OSError):
                    tmp.unlink()                    # a temporary NAME: its bytes stay under the final name
            elif not _remove_small_temp(tmp):
                left.append(_left_note(tmp))
        if left:
            print("LEFT never-named temporaries over 1 MB, not deleted by this job; removing each is a human's "
                  "call, one file per call:\n  " + "\n  ".join(left), file=sys.stderr, flush=True)


def _refuse_outputs(paths: list[Path], final: bool) -> None:
    for p in paths:
        if p.exists():
            raise SystemExit(f"REFUSING: {p.name} exists; results are never overwritten")
        if not final and p.resolve().is_relative_to(RESULTS.resolve()):
            raise SystemExit("REFUSING: a test run (--out) must not write under results/")


def _listed(path: Path) -> dict[str, str]:
    """The {file, sha256} block of a results file read through read_result (D-297 (1))."""
    rel = _repo_rel(path)
    return {"file": rel, "sha256": _RESULT_READS[rel]}


def registry_revisions(repos: list[str], models: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    """D-303 (8): ({repo: the revision used}, the record's registry block).
    - A Phase 1 model (PHASE1_MODELS) uses its Phase 1 pinned revision, input_registry.revision(), which refuses
      unless the extraction records name exactly one.
    - Every other model uses the commit the config check resolved.
    - Whenever the extraction records name a revision for a model, it must be the one used, or this refuses.
    The block lists every source the registry read, with its sha256 (input_registry.sources_sha256())."""
    import input_registry as REG
    recorded: dict[str, set[str]] = {}
    for x in REG._load_records():
        recorded.setdefault(x["repo"], set()).add(x["revision"])
    used: dict[str, str] = {}
    for r in repos:
        used[r] = REG.revision(r) if r in PHASE1_MODELS else models[r]["sha"]
        revs = sorted(recorded.get(r, set()))
        if revs and revs != [used[r]]:
            raise SystemExit(f"REFUSING: the extraction records name {r} at {[v[:12] for v in revs]}, not at "
                             f"{used[r][:12]}, the revision this check would use (D-303 (8))")
    return used, {"registry_revisions": {r: sorted(recorded.get(r, set())) for r in repos},
                  "phase1_models": [r for r in repos if r in PHASE1_MODELS],
                  "sources_sha256": {"src/input_registry.py": sha256_file(Path(REG.__file__)),
                                     **REG.sources_sha256()}}


# --------------------------------------------------------------------------- static mode

def run_static(cc_path: Path, out: Path, expected_pairs: list[tuple[str, str]], pin_date: str | None) -> str:
    """pin_date: --pin-date (YYYY-MM-DD) or None. D-303 (9): ONE recorded source of the date. The record's
    clock.pin_date is this date when some template reads the clock, else null; a clock-reading template with no
    pin_date refuses. Prompts mode and src/extract_p2.py take clock.pin_date from this record."""
    out, cc_path = Path(out), Path(cc_path)
    final = out.resolve() == FINAL_STATIC.resolve()
    _refuse_outputs([out], final)
    if final and cc_path.resolve() != CONFIG_CHECK.resolve():
        raise SystemExit("REFUSING: the final record reads results/phase2_config_check.json only")
    if pin_date is not None:
        try:
            datetime.strptime(pin_date, "%Y-%m-%d")
        except ValueError:
            raise SystemExit(f"REFUSING: --pin-date {pin_date!r} is not a YYYY-MM-DD date") from None
    code0 = P.code_block(CODE_FILES)
    if not cc_path.is_file():
        raise SystemExit(f"REFUSING: {cc_path.name} does not exist; run src/phase2_config_check.py first")
    cc = read_result(cc_path)
    if cc.get("run_kind") != "phase2_config_check":
        raise SystemExit(f"REFUSING: {cc_path.name} is not a config check record (run_kind {cc.get('run_kind')!r})")
    models, pairs = cc.get("models"), cc.get("pairs")
    if not isinstance(models, dict) or not isinstance(pairs, list):
        raise SystemExit(f"REFUSING: {cc_path.name} has no models or pairs")
    by = {(p.get("pre"), p.get("post")): p for p in pairs if isinstance(p, dict)}
    missing = [f"{a} -> {b}" for a, b in expected_pairs if (a, b) not in by]
    if missing:
        raise SystemExit(f"REFUSING: pairs missing from the config check record: {missing}")
    kept, skipped = [], []
    for a, b in expected_pairs:
        p = by[(a, b)]
        if p.get("comparable") is True and p.get("verdict") == KEPT:
            kept.append((a, b))
        else:
            skipped.append({"pre": a, "post": b, "comparable": p.get("comparable"),
                            "verdict": p.get("verdict"), "why": p.get("why")})
    repos = list(dict.fromkeys(r for pair in kept for r in pair))
    bad = [r for r in repos if not (isinstance((models.get(r) or {}).get("sha"), str)
                                    and len(models[r]["sha"]) == 40 and "gated" in models[r])]
    if bad:
        raise SystemExit(f"REFUSING: no resolved commit (or no gating state) in the config check record for {bad}")
    used, registry = registry_revisions(repos, models)
    print(f"tokenizer check (static): {len(kept)} kept pairs, {len(skipped)} not kept, {len(repos)} models; "
          f"sentinel {SENTINEL}; --pin-date {pin_date}", flush=True)
    recs: dict[str, dict[str, Any]] = {}
    tmp = tempfile.mkdtemp(prefix="gccf_tok_")       # no auto-cleanup: clean_scratch never deletes a file over 1 MB
    try:
        for repo in repos:
            m = models[repo]
            rev = used[repo]
            gated = bool(m.get("gated"))
            kind = "base" if repo in BASE_MODELS else "instruct"
            d = Path(tmp) / repo.replace("/", "__")
            fetched = fetch_tokenizer(repo, rev, gated, d)
            cfg = next((f for f in fetched["files"] if f["name"] == "config.json"), None)
            tok = load_tokenizer(d)
            rec = {"repo": repo, "revision": rev,
                   "revision_source": ("D-303 (8): the Phase 1 pinned revision (src/input_registry.py)"
                                       if repo in PHASE1_MODELS else "D-303 (8): the commit the config check resolved"),
                   "config_check_sha": m["sha"], "revision_differs_from_config_check": rev != m["sha"],
                   "gated": m.get("gated"), "kind": kind, "fetch": fetched,
                   "config_json_matches_config_check": (None if cfg is None or rev != m["sha"]
                                                        else cfg["sha256"] == m.get("config_sha256")),
                   "tokenizer": analyse(tok, repo, kind, pin_date)}
            rec["fetch_weights_snapshot"] = fetch_weights_equivalence(d, fetched, rec["tokenizer"], kind, pin_date)
            recs[repo] = rec
            t = rec["tokenizer"]
            print(f"  {repo:<42} rev {rev[:12]} files {len(fetched['files'])} token sent {fetched['token_sent_any']} "
                  f"vocab {t['vocab_size_len']} template {str(t['chat_template']['sha256'])[:12]} "
                  f"injected {t['injected_system_text'].get('present')} "
                  f"double-BOS {(t.get('double_bos') or {}).get('double_bos')}", flush=True)
    finally:
        clean_scratch(Path(tmp))
    pair_recs = [compare(recs[a], recs[b]) for a, b in kept]
    for p in pair_recs:
        print(f"  {p['pre']} -> {p['post']}: vocab equal {p['vocab_equal']}, added tokens equal "
              f"{p['added_tokens_equal']}, template equal {p['template_equal']}, injected text equal "
              f"{p['injected_system_text_equal']}, sentinel ids equal {p['sentinel_ids_equal']}", flush=True)
    if P.code_block(CODE_FILES)["sha256"] != code0["sha256"]:
        raise SystemExit("REFUSING to write: the code changed during the run")
    clock_models = [r for r, v in recs.items() if v["tokenizer"]["chat_template"]["reads_clock"]]
    matching = (cc.get("phi_pick") or {}).get("matching")
    payload = {**provenance_block("phase2_tokenizer_static"), "code": code0, "libraries": _libraries(),
               "preregistration": "D-303 (a), (e) (docs/phase2_design.md revision 3; section 2's tokenizer comparison; "
                                  "section 5's templates); revisions by D-303 (8); a clock-reading template's date by "
                                  "D-303 (9)",
               "scope": "tokenizer files (and config.json, for the tokenizer class) of every model of every pair the "
                        "config check kept, by commit; no weights; no prompts (the per-prompt check is the 'prompts' "
                        "mode)",
               "token_use": "requested only for a repo the config check recorded as gated; sent only to "
                            "https://huggingface.co (fetch_weights.auth_headers); redirects refused; an LFS object's "
                            "storage hop is a separate request with no token; never printed or written",
               "inputs": {"config_check": _listed(cc_path), "input_registry": registry},
               "sentinel": SENTINEL,
               "clock": {"pin_date": pin_date if clock_models else None,
                         "models_whose_template_reads_clock": clock_models,
                         "pin_date_given_but_unused": pin_date if (pin_date and not clock_models) else None,
                         "rule": "D-303 (9): ONE recorded source of the date. A template calling strftime_now is "
                                 "rendered with strftime_now formatting pin_date (00:00:00); pin_date is null when no "
                                 "template reads the clock; a clock-reading template with no --pin-date refuses. "
                                 "date_string is never passed, so a template's default date is what renders. Prompts "
                                 "mode and src/extract_p2.py take the date from this key"},
               "user_span_definition": "phase2_tokenizer_check.user_span (module docstring, THE USER SPAN)",
               "pairs_kept": [{"pre": a, "post": b} for a, b in kept], "pairs_not_kept": skipped,
               "phi_pick": cc.get("phi_pick"),
               "models": recs, "pairs": pair_recs,
               "summary": {
                   "pairs_with_template_differences": [f"{p['pre']} -> {p['post']}" for p in pair_recs
                                                       if not p["template_equal"]],
                   "pairs_with_vocab_differences": [f"{p['pre']} -> {p['post']}" for p in pair_recs
                                                    if not p["vocab_equal"]],
                   "models_with_injected_system_text": [r for r, v in recs.items()
                                                        if v["tokenizer"]["injected_system_text"].get("present")],
                   "models_where_render_then_tokenize_differs": [
                       r for r, v in recs.items() if (v["tokenizer"].get("double_bos") or {}).get("differ")],
                   "models_with_double_bos": [r for r, v in recs.items()
                                              if (v["tokenizer"].get("double_bos") or {}).get("double_bos")],
                   "models_whose_template_reads_clock": clock_models,
                   "models_whose_fed_ids_start_with_bos": [r for r, v in recs.items()
                                                           if v["tokenizer"]["fed_ids_start_with_bos"]],
                   "models_whose_fed_ids_do_not_start_with_bos": [r for r, v in recs.items()
                                                                  if not v["tokenizer"]["fed_ids_start_with_bos"]],
                   "phi_pick_matching": matching,
                   "phi_pick_unresolved": not (isinstance(matching, list) and len(matching) == 1),
                   "files_a_fetch_weights_snapshot_would_lack": {
                       r: v["fetch"]["not_fetched_by_fetch_weights_allow"] for r, v in recs.items()
                       if v["fetch"]["not_fetched_by_fetch_weights_allow"]},
                   "models_not_equivalent_from_a_fetch_weights_snapshot": [
                       r for r, v in recs.items() if not v["fetch_weights_snapshot"]["equivalent"]]}}
    data = json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8")   # write_results' bytes
    if len(data) <= RM_LIMIT:
        digest = write_results(out, payload, exclusive=True)
    else:
        # write_results removes its never-named temporary on a failed write, whatever its size; a record over 1 MB
        # goes through commit_outputs instead, after the same D-297 (1) check write_results makes
        problems = input_identity_problems(payload)
        if problems:
            raise SystemExit(f"REFUSING to write {out.name} (D-297 (1) results-writer rule): {problems}")
        digest = commit_outputs([(out, data)], final)[0]
    print(f"Written: {out.name}\nsha256:  {digest}")
    return digest


# --------------------------------------------------------------------------- prompts mode

def _under(f: str, prefix: str) -> bool:
    """f is a clean relative POSIX path under `prefix` (no backslash, no '..', no '.', no doubled '/')."""
    p = PurePosixPath(f)
    return ("\\" not in f and not p.is_absolute() and ".." not in p.parts and str(p) == f
            and f.startswith(prefix) and len(f) > len(prefix))


def split_files(p2: dict[str, Any], p1: dict[str, Any] | None, axis: str = "p2wg") -> list[dict[str, str]]:
    """The Phase 2 split files (axes -> partitions -> {file, sha256}) and XSTest, each with its recorded sha256. Every
    Phase 2 split must lie under data/frozen_p2/; XSTest is data/frozen/over_refusal__eval.jsonl, with the sha256
    results/dataset_manifest.json records (and the same one if the Phase 2 manifest lists it too). axis p2sst: the
    SST-2 manifest, which must hold that one axis; its files only, no XSTest (p1 is not read)."""
    axes = p2.get("axes")
    if not isinstance(axes, dict) or not axes:
        raise SystemExit("REFUSING: the Phase 2 manifest has no 'axes' (the freeze_splits schema: axes -> <axis> -> "
                         "partitions -> <split> -> {file, sha256}); which files are the Phase 2 splits is not guessed")
    if axis == SST_AXIS:
        if set(axes) != {SST_AXIS}:
            raise SystemExit(f"REFUSING: the SST-2 manifest holds the axes {sorted(axes)}, not exactly {SST_AXIS}")
        sst: dict[str, str] = {}
        for split, part in ((axes[SST_AXIS] or {}).get("partitions") or {}).items():
            f, h = (part or {}).get("file"), (part or {}).get("sha256")
            if not (isinstance(f, str) and isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h)):
                raise SystemExit(f"REFUSING: the SST-2 manifest's {SST_AXIS}/{split} has no file and sha256")
            if not _under(f, P2_SPLIT_DIR):
                raise SystemExit(f"REFUSING: the SST-2 manifest's {SST_AXIS}/{split} is {f!r}, not under {P2_SPLIT_DIR}")
            if sst.setdefault(f, h) != h:
                raise SystemExit(f"REFUSING: {f} is listed with two sha256s")
        if not sst:
            raise SystemExit("REFUSING: the SST-2 manifest lists no split file")
        return [{"file": f, "sha256": h} for f, h in sst.items()]
    out: dict[str, str] = {}
    for axis, a in axes.items():
        for split, part in ((a or {}).get("partitions") or {}).items():
            f, h = (part or {}).get("file"), (part or {}).get("sha256")
            if not (isinstance(f, str) and isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h)):
                raise SystemExit(f"REFUSING: the Phase 2 manifest's {axis}/{split} has no file and sha256")
            if f != XSTEST and not _under(f, P2_SPLIT_DIR):
                raise SystemExit(f"REFUSING: the Phase 2 manifest's {axis}/{split} is {f!r}, not under {P2_SPLIT_DIR}")
            if out.setdefault(f, h) != h:
                raise SystemExit(f"REFUSING: {f} is listed with two sha256s")
    if not any(f != XSTEST for f in out):
        raise SystemExit("REFUSING: the Phase 2 manifest lists no split file")
    xs = ((((p1.get("axes") or {}).get("over_refusal") or {}).get("partitions") or {}).get("eval") or {})
    if xs.get("file") != XSTEST or not isinstance(xs.get("sha256"), str):
        raise SystemExit(f"REFUSING: the dataset manifest does not list {XSTEST} with a sha256")
    if out.setdefault(XSTEST, xs["sha256"]) != xs["sha256"]:
        raise SystemExit(f"REFUSING: {XSTEST} has another sha256 in the Phase 2 manifest")
    return [{"file": f, "sha256": h} for f, h in out.items()]


def read_split(root: Path, fr: dict[str, str]) -> list[dict[str, Any]]:
    """The rows of one split file: its bytes read ONCE, hashed against the manifest's sha256, and parsed from those
    same bytes (split on '\\n' only, as provenance.read_jsonl does, F-017)."""
    path = Path(root) / fr["file"]
    if not path.is_file():
        raise SystemExit(f"REFUSING: {fr['file']} does not exist")
    data = path.read_bytes()
    got = sha256_bytes(data)
    if got != fr["sha256"]:
        raise SystemExit(f"REFUSING: {fr['file']} has sha256 {got[:12]}, the manifest says {fr['sha256'][:12]}")
    rows = [json.loads(line) for line in data.decode("utf-8").split("\n") if line.strip()]  # not a result (D-297)
    ids = [r.get("item_id") if isinstance(r, dict) else None for r in rows]
    if not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids) \
            or not all(isinstance(r.get("text"), str) for r in rows):
        raise SystemExit(f"REFUSING: {fr['file']}: every row needs a unique string item_id and a string text")
    return rows


def _dist(x: np.ndarray) -> dict[str, Any]:
    if x.size == 0:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {"n": int(x.size), "min": int(x.min()), "median": float(np.median(x)), "max": int(x.max())}


def _model_items(tok: Any, kind: str, kw: dict[str, Any], items: list[tuple[str, str, str]]) -> tuple[dict, list]:
    n = len(items)
    a = {"full_sha256": np.zeros(n, dtype="S64"), "span_sha256": np.zeros(n, dtype="S64"),
         "n_tokens": np.full(n, -1, np.int32), "span_start": np.full(n, -1, np.int32),
         "span_end": np.full(n, -1, np.int32), "n_straddle": np.full(n, -1, np.int32), "ok": np.zeros(n, bool)}
    fails = []
    for k, (f, iid, text) in enumerate(items):
        try:
            ids, s, e, m = user_span(tok, text, raw=(kind == "base"), **kw)
        except SpanError as err:
            fails.append({"file": f, "item_id": iid, "why": str(err)})
            continue
        a["full_sha256"][k] = ids_sha256(ids).encode("ascii")
        a["span_sha256"][k] = ids_sha256(ids[s:e]).encode("ascii")
        a["n_tokens"][k], a["span_start"][k], a["span_end"][k], a["n_straddle"][k] = len(ids), s, e, m
        a["ok"][k] = True
    return a, fails


def run_prompts(static_path: Path, manifest_path: Path, p1_manifest_path: Path | None, out: Path,
                limit: int | None = None, data_root: Path = ROOT, axis: str = "p2wg") -> str:
    """data_root: where the manifests' relative split paths resolve (the repo root; a test passes a temporary one).
    axis p2sst: the SST-2 manifest's files only, no Phase 1 manifest (p1_manifest_path None), its own final names."""
    if axis not in ("p2wg", SST_AXIS):
        raise SystemExit(f"REFUSING: --axis {axis!r} is neither p2wg nor {SST_AXIS}")
    sst = axis == SST_AXIS
    if sst and p1_manifest_path is not None:
        raise SystemExit("REFUSING: the SST-2 prompts check reads no Phase 1 manifest (XSTest is in "
                         "tokenizer_prompts.json); give no --phase1-manifest")
    if not sst and p1_manifest_path is None:
        raise SystemExit("REFUSING: the p2wg prompts check needs the Phase 1 manifest (XSTest)")
    static_path, manifest_path, out = map(Path, (static_path, manifest_path, out))
    p1_manifest_path = None if sst else Path(p1_manifest_path)
    final_out, final_npz, default_manifest = ((FINAL_PROMPTS_SST, FINAL_NPZ_SST, P2SST_MANIFEST) if sst else
                                              (FINAL_PROMPTS, FINAL_NPZ, P2_MANIFEST))
    final = out.resolve() == final_out.resolve()
    npz = final_npz if final else out.with_name(out.stem + "_ids.npz")
    _refuse_outputs([out, npz], final)
    if final and (limit is not None or Path(data_root).resolve() != ROOT.resolve()
                  or static_path.resolve() != FINAL_STATIC.resolve()
                  or manifest_path.resolve() != default_manifest.resolve()
                  or (not sst and p1_manifest_path.resolve() != P1_MANIFEST.resolve())):
        raise SystemExit("REFUSING: the final record reads the default inputs, every item (no --limit)")
    code0 = P.code_block(CODE_FILES)
    for p in (static_path, manifest_path, *([] if sst else [p1_manifest_path])):
        if not p.is_file():
            raise SystemExit(f"REFUSING: {p.name} does not exist")
    st = read_result(static_path)
    if st.get("run_kind") != "phase2_tokenizer_static":
        raise SystemExit(f"REFUSING: {static_path.name} is not a static tokenizer record")
    libs, st_libs = _libraries(), st.get("libraries") or {}
    off = {k: {"static_record": st_libs.get(k), "now": libs.get(k)} for k in ("transformers", "tokenizers")
           if libs.get(k) is None or libs.get(k) != st_libs.get(k)}
    if off:
        raise SystemExit(f"REFUSING: the tokenization libraries differ from the static record's: {off}")
    p2, p1 = read_result(manifest_path), (None if sst else read_result(p1_manifest_path))
    files = split_files(p2, p1, axis)
    items: list[tuple[str, str, str]] = []
    for fr in files:
        rows = read_split(Path(data_root), fr)
        fr["n_items_in_file"] = len(rows)
        rows = rows[:limit] if limit is not None else rows
        fr["n_items_checked"] = len(rows)
        items += [(fr["file"], r["item_id"], r["text"]) for r in rows]
    pin_date = (st.get("clock") or {}).get("pin_date")
    print(f"tokenizer check (prompts): {len(items)} items from {len(files)} files, {len(st['models'])} models",
          flush=True)
    arrays: dict[str, np.ndarray] = {"files": np.array([f["file"] for f in files]),
                                     "file_index": np.array([[f["file"] for f in files].index(i[0]) for i in items],
                                                            np.int32),
                                     "item_id": np.array([i[1] for i in items])}
    per_model: dict[str, dict[str, Any]] = {}
    model_arrays: dict[str, dict[str, np.ndarray]] = {}
    file_of = np.array([i[0] for i in items])
    tmp = tempfile.mkdtemp(prefix="gccf_tokp_")      # no auto-cleanup: clean_scratch never deletes a file over 1 MB
    try:
        for repo, rec in st["models"].items():
            t0 = time.time()
            d = Path(tmp) / repo.replace("/", "__")
            fetched = fetch_tokenizer(repo, rec["revision"], bool(rec.get("gated")), d)
            want = {f["name"]: f["sha256"] for f in rec["fetch"]["files"]}
            have = {f["name"]: f["sha256"] for f in fetched["files"]}
            if want != have:
                raise SystemExit(f"REFUSING: {repo}@{rec['revision'][:12]}: the tokenizer files differ from the static "
                                 f"record's: {sorted(set(want.items()) ^ set(have.items()))}")
            tok = load_tokenizer(d)
            tmpl = template_text(tok)
            if _template_sha(tmpl) != rec["tokenizer"]["chat_template"]["sha256"]:
                raise SystemExit(f"REFUSING: {repo}: the loaded template differs from the static record's")
            kw = clock_kwargs(tmpl, pin_date)
            a, fails = _model_items(tok, rec["kind"], kw, items)
            model_arrays[repo] = a
            key = repo.replace("/", "__")
            for k, v in a.items():
                arrays[f"{key}__{k}"] = v
            ok = a["ok"]
            span_len = a["span_end"] - a["span_start"]
            per_model[repo] = {
                "revision": rec["revision"], "kind": rec["kind"], "n_items": len(items), "n_ok": int(ok.sum()),
                "n_failed": len(fails), "failures": fails, "token_sent_any": fetched["token_sent_any"],
                "clock_kwargs_passed": sorted(kw),
                "token_length": {"full": _dist(a["n_tokens"][ok]), "user_span": _dist(span_len[ok]),
                                 "by_file": {f["file"]: {"full": _dist(a["n_tokens"][ok & (file_of == f["file"])]),
                                                         "user_span": _dist(span_len[ok & (file_of == f["file"])])}
                                             for f in files}},
                "n_items_with_straddle": int((a["n_straddle"][ok] > 0).sum()),
                "max_straddle": int(a["n_straddle"][ok].max()) if ok.any() else None,
                "seconds": round(time.time() - t0, 1)}
            print(f"  {repo:<42} ok {int(ok.sum())}/{len(items)} failed {len(fails)} full tokens "
                  f"{per_model[repo]['token_length']['full']} ({per_model[repo]['seconds']} s)", flush=True)
    finally:
        clean_scratch(Path(tmp))

    def by_file(mask: np.ndarray) -> dict[str, list[str]]:
        return {f["file"]: [items[k][1] for k in np.flatnonzero(mask & (file_of == f["file"]))] for f in files}

    pair_recs = []
    for p in st["pairs"]:
        a, b = model_arrays[p["pre"]], model_arrays[p["post"]]
        both = a["ok"] & b["ok"]
        full_d = both & (a["full_sha256"] != b["full_sha256"])
        span_d = both & (a["span_sha256"] != b["span_sha256"])
        pair_recs.append({"pre": p["pre"], "post": p["post"], "n_items": len(items), "n_compared": int(both.sum()),
                          "n_not_compared": int((~both).sum()),
                          "full_sequence_differs": {"n": int(full_d.sum()), "item_ids": by_file(full_d)},
                          "user_span_differs": {"n": int(span_d.sum()), "item_ids": by_file(span_d)},
                          "token_length_differs": {"n": int((both & (a["n_tokens"] != b["n_tokens"])).sum())}})
        print(f"  {p['pre']} -> {p['post']}: full differs {int(full_d.sum())}, user span differs "
              f"{int(span_d.sum())} of {int(both.sum())} compared", flush=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    npz_bytes = buf.getvalue()
    npz_sha = sha256_bytes(npz_bytes)
    if P.code_block(CODE_FILES)["sha256"] != code0["sha256"]:
        raise SystemExit("REFUSING to write: the code changed during the run")
    payload = {**provenance_block("phase2_tokenizer_prompts"), "code": code0, "libraries": libs,
               "preregistration": "D-303 (c), (e) (docs/phase2_design.md revision 3; section 2's added-tokens check; "
                                  "section 5: per prompt and in total, whether the pre and post token sequences "
                                  "differ); no truncation, token lengths recorded (D-303 (3))"
                                  + ("; D-338 (4), D-340 (3): the SST-2 non-safety control's prompts, before any p2sst "
                                     "extraction, keyed by (file, item_id)" if sst else ""),
               **({"axis": SST_AXIS} if sst else {}),
               "token_use": st.get("token_use"),
               "inputs": ({"static_record": _listed(static_path), "dataset_manifest_p2sst": _listed(manifest_path),
                           "split_files": files} if sst else
                          {"static_record": _listed(static_path), "dataset_manifest_p2": _listed(manifest_path),
                           "dataset_manifest": _listed(p1_manifest_path), "split_files": files}),
               "limit_per_file": limit, "clock": st.get("clock"),
               "user_span_definition": "phase2_tokenizer_check.user_span (module docstring, THE USER SPAN)",
               "id_sha256_method": "sha256 of the ASCII decimal ids joined by ','",
               "ids_npz": {"file": _repo_rel(npz), "sha256": npz_sha,
                           "arrays": "files, file_index, item_id; per model <org>__<model>__{full_sha256, "
                                     "span_sha256, n_tokens, span_start, span_end, n_straddle, ok} (-1 and empty "
                                     "where the span failed)"},
               "models": per_model, "pairs": pair_recs}
    problems = input_identity_problems(payload)
    if problems:
        raise SystemExit(f"REFUSING to write (D-297 (1) results-writer rule): {problems}")
    digest = commit_pair(out, json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8"),
                         npz, npz_bytes, final)
    print(f"Written: {out.name} and {npz.name}\nsha256:  {digest} (json), {npz_sha} (npz)")
    return digest


def commit_pair(out: Path, json_bytes: bytes, npz: Path, npz_bytes: bytes, final: bool) -> str:
    """The npz, then the JSON, through commit_outputs. If naming the JSON fails, the npz already named is renamed
    aside to '<name>.failed.<pid>' (never deleted) and the run exits nonzero. Returns the JSON's sha256."""
    return commit_outputs([(npz, npz_bytes), (out, json_bytes)], final)[1]


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Phase 2 tokenizer check (D-303 (a), (e))")
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("static", help="tokenizer files and templates of every kept pair's models")
    s.add_argument("--config-check", default=str(CONFIG_CHECK))
    s.add_argument("--out", default=str(FINAL_STATIC))
    s.add_argument("--pin-date", default=None,
                   help="YYYY-MM-DD for a template that reads the clock (D-303 (9)); required only then, and recorded "
                        "as clock.pin_date (null when no template reads the clock)")
    q = sub.add_parser("prompts", help="every Phase 2 split item and XSTest, per model and pair")
    q.add_argument("--axis", default="p2wg", choices=("p2wg", SST_AXIS),
                   help=f"p2wg (the default: dataset_manifest_p2.json and XSTest) or {SST_AXIS} (SST-2: "
                        "dataset_manifest_p2sst.json only, its own record; D-340 (3))")
    q.add_argument("--static", default=str(FINAL_STATIC))
    q.add_argument("--manifest", default=None, help=f"default {P2_MANIFEST.name}, or {P2SST_MANIFEST.name} for p2sst")
    q.add_argument("--phase1-manifest", default=None, help=f"p2wg only: default {P1_MANIFEST.name}")
    q.add_argument("--out", default=None, help=f"default {FINAL_PROMPTS.name}, or {FINAL_PROMPTS_SST.name} for p2sst")
    q.add_argument("--limit", type=int, default=None, help="first N items per file (test runs only)")
    args = ap.parse_args(argv)
    from hf_env_guard import assert_hf_env
    assert_hf_env("phase2_tokenizer_check.py")
    if args.mode == "static":
        run_static(Path(args.config_check), Path(args.out), list(CC.PAIRS), args.pin_date)
    else:
        sst = args.axis == SST_AXIS
        manifest = args.manifest or str(P2SST_MANIFEST if sst else P2_MANIFEST)
        p1 = args.phase1_manifest if sst else (args.phase1_manifest or str(P1_MANIFEST))
        out = args.out or str(FINAL_PROMPTS_SST if sst else FINAL_PROMPTS)
        run_prompts(Path(args.static), Path(manifest), None if p1 is None else Path(p1), Path(out), args.limit,
                    axis=args.axis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
