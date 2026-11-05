"""Phase 2 BOS check and special-token scan (D-318 (6), D-324 (5)): what the tokenizers make of every prompt.

PART A, BOS (D-318 (6)). Phase 1's chat arm rendered the chat template as text and then tokenized it with special
tokens, which gave a double BOS (D-305, D-311). src/extract_p2.py takes its ids from apply_chat_template(tokenize=True).
But no record holds per-item ids or BOS counts: the caches hold only token_ids_sha256 (phase2_tokenizer_check.
ids_sha256). Part A rebuilds every item's ids through extract_p2's own code path, with the span function the model's
extraction records name (D-335 (5)), counts the BOS ids, and ties each item to the caches by that hash.

For each Phase 2 model (the 17 models of results/phase2/tokenizer_static.json, which are those of the config check's
kept pairs):
  tokenizer  See TOKENIZER FILES below. Every loaded file must also have the sha256 that tokenizer_static.json records.
             For an extracted model, the files must be exactly the tokenizer files the extraction record lists
             (model.weight_files.config_and_tokenizer_files), with the same sha256.
  ids        extract_p2.template_setup (the template rule; the clock pin comes from tokenizer_static.json), then
             extract_p2.instruct_item or base_item with the span function the model's extraction records name: the ids
             extract_p2 fed the model. Every extraction record of the model must name the same template sha256,
             template kwargs, date pin and tokenizer_static.json, and as user_span_source either SPAN_TODAY,
             'phase2_tokenizer_check.user_span' (the rule before the olmo-bin merge d74b22a: the records of every cache
             extracted before it), or SPAN_PLAIN, 'phase2_tokenizer_check.user_span_plain' (D-324 (5)(v): extract_p2's
             span function from that merge; the first are OLMo-2 SFT and DPO). Any other name refuses, and so do two
             span functions among one model's records. A model with no extraction record is rebuilt with user_span, the
             rule tokenizer_prompts_ids.npz was built with. The model's block names the span function used
             (span_function; also span_function_by_model at the top of the record).
  plain path With user_span_plain, an item whose user text today's tokenization parses into an added token from
             non-whitespace characters takes the plain-text path (extract_p2's info: plain_text_path, strings); every
             other item keeps today's ids exactly (extract_p2._checked_span). Each plain-path item is listed apart
             (plain_path.items: file, item_id, strings, today's ids_sha256 through the same path with user_span, or
             null with the refusal, and the plain ids_sha256). For each cache (split x pooling) of each extraction
             record, the items it lists in special_strings_as_text, with their strings, must be exactly the plain-path
             items of its split file (plain_path.record_listing), or the model fails. A record naming user_span may lack
             special_strings_as_text (it was written before the rule) and then lists none.
  counted    The tokenizer's bos_token_id. A tokenizer with no BOS token (Qwen) is counted on config.json's bos_token_id
             instead, and the record says so; with neither, nothing is counted.
  policy     The count of that id the template writes at position 0 (0 or 1), read from the sentinel message through
             the same path. The sentinel's ids must equal tokenizer_static.json's.
  per item   On the ids fed (for a plain-path item, its plain ids): whether position 0 holds the id, the positions of
             every later occurrence, and ids_sha256(ids). For OLMo-2, id 100257 is both BOS and EOS, so position 0 and
             the later occurrences are reported separately. Items are
             listed only where they differ from the policy; every item not listed has exactly the policy pattern.
             items_with_more_than_one_bos names, per model, every item holding the id more than once.
  ties       - results/phase2/tokenizer_prompts_ids.npz, built with today's path: full_sha256 for the same (file,
               item_id), over the same items, for every item whose fed ids are today's. A plain-path item is set apart
               (plain_path_items_set_apart): its today's ids_sha256 is compared with the npz instead (null when today's
               path refuses it, which must meet an item the npz marks not ok); its plain ids never are.
             - The caches, through the DURABLE copies only (D-307 (1): a cache a result reads is ingested and indexed).
               For each extraction record (set select or rest) the caches read are those of the extraction record itself
               when it is ingested (an index manifests/ingest_p2/<record stem>.sha256 exists), else those of its p2sub
               layer-subset record (src/subset_p2.py, results/phase2/extract__<org>__<model>__p2sub@<rev12>__<set>.json)
               when that is ingested. A p2sub cache copies item_ids and token_ids_sha256 bit for bit, and its record
               must name the extraction record's cache file and sha256 as its source for every split.
               The record read must say ingest 'yes', and neither its index nor a cache may be a symlink.
               Each cache read is in data/hidden, mode 444, of the record's size; its sha256 must equal the record's and
               the index's. With --hash-caches (required for the record) its bytes are read ONCE, hashed and loaded
               from the same buffer, and the observed sha256 goes into the row and into input_cache_sha256 (named with
               config['<cache file>_repo'] and config['<cache file>_rev']). Only item_ids and token_ids_sha256 are read.
               Per cache: token_ids_sha256 equals ids_sha256 of the ids fed (a plain-path item: its plain ids) for the
               same item_id, the item set equals the split file's, and the record's split_sha256 is part A's. The two
               poolings of a split must agree.
               A read error is a failed row, never a crash.
             - Coverage: every Phase 2 split file x {usermean, lastgen} must be tied, or the model's tie is
               'incomplete' (a failure). A cache named {repo}__p2tpl@*__*.npz or {repo}__p2sub@*__*.npz in data/hidden
               or any --cache-dir (default ~/.cache/gccf_scratch/p2cache and ~/.cache/gccf_scratch/p2sub) that no
               record of the model names is a failure; so is any cache of another revision than the static record's
               (listed apart as other_revision_caches). A model with no extraction record and no such cache is
               'skipped'.

PART B, SPECIAL-TOKEN SCAN (D-324 (5) (iv)). Every Phase 1 and Phase 2 prompt is searched for the literal strings each
tokenizer matches atomically.
  prompts    The Phase 2 splits (dataset_manifest_p2.json) and XSTest. Every Phase 1 split file a dataset manifest
             results/dataset_manifest*.json lists, each checked against its index manifest
             manifests/splits/<split>__manifest.json (split_file, sha256_of_original, and n against the rows read); no
             index manifest may be unlisted. Every file is sha256-checked when read. (LoRA training corpora are
             training text, not prompts, and are not scanned.)
  models     The 17 Phase 2 models, plus every (repo, revision) an original Phase 1 extraction record names (record
             'repo' and 'revision': a LoRA arm or a tagged re-extraction names its base model). A Phase 1 model already
             in the Phase 2 list at the same revision is scanned once; the others are keyed '<repo>@<rev12>'.
  strings    Every added token's content, special or not: a fast tokenizer splits its whole added vocabulary out of
             raw text before the model sees it. split_special_tokens=True would turn off only the special ones, so the
             two kinds are counted apart. Also every all_special_tokens string that is not an added token (flagged:
             these may not be matched atomically).
  parsing    For each prompt holding such a string, the added-token ids it really tokenizes to, special and non-special
             apart:
             - raw: tok(text, add_special_tokens=False), the user content as Phase 1's extract_hidden fed it;
             - template: the user span of extract_p2.instruct_item with phase2_tokenizer_check.user_span (Phase 2
               instruct models; today's rule whatever span function part A used: the scan asks what the tokenizer
               makes of the strings). For the OLMo-2 base, extract_p2's path is tok(text), so raw is its path.
             A template path that refuses is counted apart. A prompt where neither path gives an added id is a
             substring match only. Prompts with no literal string are not tokenized here.
  headline   Counts over every prompt, and over the files the model's own extraction records name.

TOKENIZER FILES. Only the tokenizer files are used, never weights.
  - A model with a NAS keep copy, data/weights_keep/<slug>/ (the gated Llama pair): copied from it. Each file is
    checked against logs/weights_keep_<slug>.sha256.
  - Any other model: fetched by commit from https://huggingface.co (phase2_tokenizer_check.fetch_tokenizer, which checks
    each file against the hash the Hub publishes). The Hub token is used ONLY for a gated repo: gated as
    tokenizer_static.json (Phase 2) or results/access_check.json (Phase 1) records it; for a repo neither lists, as the
    Hub's model API reports it without a token (https://huggingface.co/api/models/<repo>), and failing that, as the
    Hub's HTTP 401/403 to a request without the token. fetch_weights.load_token reads it from the .env at the code's
    repo root, and fetch_weights.auth_headers sends it to https://huggingface.co only; gated, its source and
    token_sent are recorded per model, and a token sent for a repo not known to be gated refuses. A repo the Hub still
    refuses is reported as not loaded, listed in scan_not_loaded, and makes the run exit 3.
  - Only the files fetch_weights.ALLOW downloads are loaded: the files an extraction snapshot held.
  Loading is done by phase2_tokenizer_check.load_tokenizer: local files only, offline, no remote code.

Keys added for the span functions (D-335 (5)): per model span_function, span_function_from (where it comes from),
plain_path {n_items, items (file, item_id, strings, today's and plain ids_sha256), ok (the record's
special_strings_as_text lists exactly these items with these strings)}; prompts_tie.plain_path_items_set_apart and
prompts_tie.plain_path_today_ids_not_npz (n_compared and n_equal count only the items not set apart); a row's
today_refused when today's path refuses a plain item; at the top span_rule, span_function_by_model and
plain_path_items. Every model's span function is checked before any tokenizer is fetched or cache read.

Output: results/phase2/bos_check_p2.json, written once (refused if it exists) through provenance.write_results; the
record requires --hash-caches, every model and the default directories. A test run (--out elsewhere) must not write
under results/. The code block names every src module the process loaded (derived, not listed by hand).

--axis p2sst (D-338 (4), D-340 (3): the SST-2 non-safety control's prompts, before any p2sst extraction): PART B ONLY,
over the split files of results/phase2/dataset_manifest_p2sst.json (each sha256-checked when read, and the same files
results/phase2/tokenizer_prompts_p2sst.json read, so the prompts check runs first), against the same tokenizers as the
p2wg record (the 17 Phase 2 models, each loaded and checked as part A loads it: tokenizer_static.json's files and the
model's p2tpl extraction records' files, its template rule, today's user_span on the template path; and every Phase
1-only (repo, revision)), by the same scan_model. Own-files headline: the p2sst files the model's own p2sst extraction
records name (none before its extraction). Part A is not run: no p2sst cache is tied here (the p2sst extraction records
carry each cache's token_ids_sha256 and plain-path items). Prompts are keyed by (file, item_id), never by a bare item_id
(an SST-2 item_id can equal a WildGuardMix one). Output results/phase2/bos_check_p2sst.json, written once; a test run
writes elsewhere, never under results/. Exit 0, or 3 when a tokenizer was not loaded. With --axis p2wg (the default)
nothing changes.
Exit: 0 when every part A tie holds (the npz, the caches and the plain-path listing), none is incomplete and every
tokenizer was loaded; 3 otherwise (the record is still written). The last line prints the failed, skipped and
incomplete models and the tokenizers not loaded.
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_p2 as EP  # noqa: E402
import fetch_weights as FW  # noqa: E402
import paired as P  # noqa: E402
import phase2_config_check as CC  # noqa: E402
import phase2_tokenizer_check as TC  # noqa: E402
import subset_p2 as SUB  # noqa: E402
from provenance import (_repo_rel, ensure_utf8_stdout, provenance_block, read_result, sha256_bytes,  # noqa: E402
                        sha256_file, write_results)

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent
FINAL = ROOT / "results" / "phase2" / "bos_check_p2.json"
FINAL_SST = ROOT / "results" / "phase2" / "bos_check_p2sst.json"   # --axis p2sst: part B over SST-2's prompts
SCRATCH_PARENT = Path.home() / ".cache" / "gccf_scratch"
P2CACHE = SCRATCH_PARENT / "p2cache"                    # extract_p2's full-grid caches (scripts/p2_sitting.sh)
P2SUB = SCRATCH_PARENT / "p2sub"                        # src/subset_p2.py's layer subsets before ingest (lane A)
POOLINGS = ("usermean", "lastgen")
SPAN_TODAY = "phase2_tokenizer_check.user_span"          # today's rule: records before d74b22a, the npz, part B
SPAN_PLAIN = "phase2_tokenizer_check.user_span_plain"    # D-324 (5)(v): extract_p2's span function from d74b22a
SPAN_FUNCTIONS = {SPAN_TODAY: "user_span", SPAN_PLAIN: "user_span_plain"}   # user_span_source -> the span api's name
LIST_MAX = 50                                           # part B lists a model's matched prompts up to this many


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


class NotLoaded(Exception):
    """The Hub refused a tokenizer's files, with the token where the repo is gated, and there is no NAS keep copy."""


def code_files() -> list[str]:
    """Every src module this process has loaded (paired.code_block refuses a loaded module it is not given)."""
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and Path(f).suffix == ".py" and Path(f).resolve().parent == SRC:
            out.add(f"src/{Path(f).name}")
    return sorted(out)


def _str(a: np.ndarray) -> np.ndarray:
    return np.char.decode(a, "ascii") if a.dtype.kind == "S" else a.astype(str)


def _rel(p: Path, root: Path) -> str:
    """A path relative to the repo when it lies under it, else as given."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(Path(root).resolve())).replace("\\", "/")
    except ValueError:
        return str(p)


# --- part A: per item ----------------------------------------------------------------------------------------------

def bos_positions(ids: list[int], bos_id: int | None) -> list[int] | None:
    """Every position of the counted id in ids; None when nothing is counted."""
    return None if bos_id is None else [k for k, i in enumerate(ids) if i == bos_id]


def item_builder(tok: Any, kind: str, kw: dict[str, Any], api: Any,
                 span_fn: Callable | None = None) -> Callable[..., list[int]]:
    """extract_p2's own path from a text to the ids it fed: instruct_item or base_item with the span function (default
    api.user_span, today's rule), with nothing added. build(text, info): extract_p2 records in info the path taken
    (plain_text_path) and the strings concerned."""
    fn = api.user_span if span_fn is None else span_fn
    if kind == "instruct":
        return lambda text, info=None: EP.instruct_item(tok, text, kw, fn, info)[0]
    return lambda text, info=None: EP.base_item(tok, text, fn, info)[0]


def policy(build: Callable[[str], list[int]], bos_id: int | None, sm: dict[str, Any]) -> dict[str, Any]:
    """How many of the counted id the template writes at position 0, from the sentinel through the same path."""
    ids = build(TC.SENTINEL)
    if ids != sm["tokenizer"]["sentinel"]["ids"]:
        refuse(f"the sentinel's ids {ids} are not tokenizer_static.json's {sm['tokenizer']['sentinel']['ids']}")
    pos = bos_positions(ids, bos_id)
    return {"bos_at_position_0": None if pos is None else int(0 in pos),
            "sentinel_later_bos_positions": None if pos is None else [p for p in pos if p > 0],
            "sentinel_ids_equal_tokenizer_static": True,
            "tokenizer_static_fed_ids_start_with_bos": sm["tokenizer"].get("fed_ids_start_with_bos")}


def scan_items(build: Callable[..., list[int]], items: list[tuple[str, str, str]], bos_id: int | None,
               policy_at0: int | None, today: Callable[..., list[int]] | None = None
               ) -> tuple[dict[tuple[str, str], str], dict[str, Any]]:
    """(ids_sha256 of the ids fed, by (file, item_id); the BOS summary, counted on the ids fed). An item is listed when
    its pattern differs from the policy: position 0 not as the template writes it, or any later occurrence. An item on
    the plain-text path (extract_p2's info) goes into the summary's plain_path_items with its strings, today's
    ids_sha256 through `today` (extract_p2's path with user_span; null, with the refusal, when that path refuses) and
    its plain ids_sha256. With no `today` the span function is today's, which never takes that path: such an item
    refuses."""
    hashes: dict[tuple[str, str], str] = {}
    plain: list[dict[str, Any]] = []
    counts: Counter = Counter()
    n_at0, differing = 0, []
    for f, iid, text in items:
        info: dict[str, Any] = {}
        try:
            ids = build(text, info)
        except SystemExit as e:
            raise SystemExit(f"{e} [{f} {iid}]") from None
        hashes[(f, iid)] = TC.ids_sha256(ids)
        if info.get("plain_text_path"):
            if today is None:
                refuse(f"{f} {iid} took the plain-text path under today's span function")
            row: dict[str, Any] = {"file": f, "item_id": iid, "strings": info.get("strings"),
                                   "today_ids_sha256": None, "plain_ids_sha256": hashes[(f, iid)]}
            try:
                row["today_ids_sha256"] = TC.ids_sha256(today(text))
            except SystemExit as e:
                row["today_refused"] = str(e)[:300]
            plain.append(row)
        pos = bos_positions(ids, bos_id)
        if pos is None:
            continue
        counts[len(pos)] += 1
        at0 = int(0 in pos)
        n_at0 += at0
        later = [p for p in pos if p > 0]
        if at0 != policy_at0 or later:
            differing.append({"file": f, "item_id": iid, "n_tokens": len(ids), "bos_at_position_0": at0,
                              "later_bos_positions": later})
    out: dict[str, Any] = {"n_items": len(items)}
    if bos_id is None:
        out.update(bos_count_distribution=None, n_bos_at_position_0=None,
                   note="neither the tokenizer nor config.json names a BOS id, so nothing is counted")
    else:
        out.update(bos_count_distribution={str(k): counts[k] for k in sorted(counts)}, n_bos_at_position_0=n_at0)
    out.update(n_items_differing_from_policy=len(differing), items_differing_from_policy=differing,
               plain_path_items=plain)
    return hashes, out


# --- tokenizer files -----------------------------------------------------------------------------------------------

def _allowed(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in FW.ALLOW)


def _place(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)                               # a second name for the same bytes
    except OSError:
        shutil.copyfile(src, dst)


def hub_gated(repo: str) -> Any:
    """The repo's 'gated' as the Hub's model API reports it to a request without a token (False, 'auto' or 'manual'),
    or None when the API does not answer 200."""
    r = CC.get(f"{CC.HUB}/api/models/{repo}", use_token=False)
    return r.json().get("gated") if r.status_code == 200 else None


def tokenizer_dir(repo: str, rev: str, root: Path, scratch: Path, gated: Any) -> tuple[Path, dict[str, Any]]:
    """The directory the tokenizer is loaded from: only tokenizer files fetch_weights would have downloaded. From the
    NAS keep copy when there is one, else by commit from the Hub, with the token only for a gated repo. gated: the
    recorded flag, or None when no record lists the repo; then the Hub's model API without a token, and failing that
    a 401/403 to a request without the token, decides. NotLoaded when the Hub still refuses."""
    slug = f"{repo.replace('/', '__')}@{rev[:12]}"
    load = scratch / slug / "load"
    load.mkdir(parents=True)                            # a fresh directory (run() refuses a reused --scratch)
    keep, man = root / "data" / "weights_keep" / slug, root / "logs" / f"weights_keep_{slug}.sha256"
    if keep.is_dir():
        if not man.is_file():
            refuse(f"{repo}: the NAS keep copy {keep} has no manifest {man}")
        listed = {}
        for line in man.read_text(encoding="utf-8").splitlines():
            h, _, p = line.partition("  ")
            listed[p.removeprefix(f"data/weights_keep/{slug}/")] = h
        for name in TC.wanted_files(p.name for p in keep.iterdir() if p.is_file()):
            if _allowed(name):
                shutil.copyfile(keep / name, load / name)
                if sha256_file(load / name) != listed.get(name):
                    refuse(f"{repo}: {name} copied from the keep copy is not the manifest's sha256")
        source = {"from": "the NAS keep copy", "dir": f"data/weights_keep/{slug}", "gated": gated, "token_sent": False,
                  "manifest": {"file": f"logs/weights_keep_{slug}.sha256", "sha256": sha256_file(man)}}
    else:
        def fetch(with_token: bool, sub: str) -> tuple[Any, Path]:
            dest = scratch / slug / sub
            try:
                return TC.fetch_tokenizer(repo, rev, with_token, dest), dest
            except SystemExit as e:
                if re.search(r"HTTP 40[13]\b", str(e)):
                    return e, dest
                raise

        gated_from = "a record (tokenizer_static.json or access_check.json)"
        if gated is None:
            gated, gated_from = hub_gated(repo), "the Hub's model API, asked without a token"
        got, hub = fetch(bool(gated), "hub")
        if isinstance(got, SystemExit) and gated is None:      # still unknown: the Hub's refusal says gated
            gated, gated_from = True, "the Hub's HTTP 401/403 to a request without the token"
            got, hub = fetch(True, "hub_token")
        if isinstance(got, SystemExit):
            raise NotLoaded(f"the Hub refused the tokenizer files (gated: {gated!r}, from {gated_from}; the token is "
                            f"sent only for a gated repo): {got}")
        if got["token_sent_any"] and not gated:
            refuse(f"{repo}: a token was sent for a repo not known to be gated")
        for f in got["files"]:
            if _allowed(f["name"]):
                _place(hub / f["name"], load / f["name"])
        source = {"from": "https://huggingface.co by commit (phase2_tokenizer_check.fetch_tokenizer)", "gated": gated,
                  "gated_from": gated_from, "token_sent": got["token_sent_any"],
                  "fetched_not_loaded": sorted(f["name"] for f in got["files"] if not _allowed(f["name"]))}
    files = {p.name: sha256_file(p) for p in sorted(load.iterdir())}
    return load, {**source, "loaded_files_sha256": files}


def check_files(repo: str, files: dict[str, str], sm: dict[str, Any], records: list[dict[str, Any]]) -> None:
    """The loaded files have tokenizer_static.json's sha256s and are exactly the tokenizer files every extraction
    record lists (what extract_p2 loaded)."""
    static = {f["name"]: f["sha256"] for f in sm["fetch"]["files"]}
    bad = sorted(n for n, h in files.items() if static.get(n) != h)
    if bad:
        refuse(f"{repo}: {bad} differ from tokenizer_static.json's sha256")
    for r in records:
        ref = r["model"]["weight_files"]["config_and_tokenizer_files"]
        ref = {n: v["sha256"] for n, v in ref.items() if n in TC.TOKENIZER_FILES or n in TC.CLASS_FILES}
        if ref != files:
            refuse(f"{repo}: the files loaded here {files} are not the ones its extraction record lists {ref}")


def check_record(r: dict[str, Any], repo: str, sm: dict[str, Any], tmpl: dict[str, Any], static_sha: str) -> None:
    """The extraction record names this model, revision and kind, no adapter, this template rule and static record, and
    a span function part A rebuilds with (SPAN_FUNCTIONS)."""
    t = r.get("template") or {}
    want = {"model_repo": repo, "model_revision": sm["revision"], "kind": sm["kind"], "adapter": None,
            "template sha256": tmpl["chat_template_sha256"], "template kwargs": tmpl["template_kwargs"],
            "date pin": tmpl["date_pin"], "tokenizer_static sha256": static_sha}
    got = {"model_repo": r.get("model_repo"), "model_revision": r.get("model_revision"), "kind": r.get("kind"),
           "adapter": r.get("adapter"), "template sha256": t.get("chat_template_sha256"),
           "template kwargs": t.get("template_kwargs"), "date pin": t.get("date_pin"),
           "tokenizer_static sha256": (r.get("tokenizer_static") or {}).get("sha256")}
    if got != want:
        refuse(f"{repo}: the extraction record differs: {sorted(k for k in want if got[k] != want[k])}")
    src = r.get("user_span_source")
    if not isinstance(src, str) or src not in SPAN_FUNCTIONS:
        refuse(f"{repo}: the extraction record differs: ['user_span_source'] ({src!r} is neither {SPAN_TODAY} nor "
               f"{SPAN_PLAIN})")


def record_span(repo: str, records: list[dict[str, Any]]) -> tuple[str, str]:
    """(the span function part A rebuilds the model's items with, where that comes from): the one every extraction
    record of the model names; with no record, today's user_span, the rule tokenizer_prompts_ids.npz was built with.
    A name part A does not know, or two span functions among one model's records, refuse."""
    names = sorted({str(r.get("user_span_source")) for r in records})
    unknown = [n for n in names if n not in SPAN_FUNCTIONS]
    if unknown:
        refuse(f"{repo}: an extraction record names the span function {unknown} (user_span_source), neither "
               f"{SPAN_TODAY} nor {SPAN_PLAIN}")
    if len(names) > 1:
        refuse(f"{repo}: its extraction records name different span functions {names} (user_span_source); all of one "
               "model's records must name the same one")
    if not names:
        return SPAN_TODAY, "no extraction record: today's rule, the one tokenizer_prompts_ids.npz was built with"
    return names[0], "every extraction record of the model names it (user_span_source)"


# --- part A: ties --------------------------------------------------------------------------------------------------

def tie_prompts(z: Any, repo: str, hashes: dict[tuple[str, str], str],
                apart: dict[tuple[str, str], str | None] | None = None) -> dict[str, Any]:
    """Each item's ids_sha256 (of the ids fed) against tokenizer_prompts_ids.npz's full_sha256 (built with today's
    path), over the same (file, item_id) set. apart: the plain-path items, {(file, item_id): today's ids_sha256, or None
    when today's path refused it}. Their fed ids are not today's and are never compared; their today's ids_sha256 is,
    and None must meet an item the npz marks not ok."""
    key = repo.replace("/", "__")
    apart = apart or {}
    files, iid = _str(z["files"]), _str(z["item_id"])
    keys = [(str(files[i]), str(d)) for i, d in zip(z["file_index"], iid)]
    full, ok = _str(z[f"{key}__full_sha256"]), z[f"{key}__ok"]
    mism = [list(k) for k, h, o in zip(keys, full, ok) if o and k in hashes and k not in apart and hashes[k] != h]
    not_ok = [list(k) for k, o in zip(keys, ok) if not o and k not in apart]
    differ = sorted(set(keys) ^ set(hashes))
    npz = {k: (str(h) if o else None) for k, h, o in zip(keys, full, ok)}
    apart_mism = sorted(list(k) for k in apart if npz.get(k, "not in the npz") != apart[k])
    n = sum(1 for k, o in zip(keys, ok) if o and k in hashes and k not in apart)
    return {"n_compared": n, "n_equal": n - len(mism), "mismatched": mism, "not_ok_in_npz": not_ok,
            "items_not_in_both": [list(k) for k in differ],
            "plain_path_items_set_apart": sorted(list(k) for k in apart),
            "plain_path_today_ids_not_npz": apart_mism,
            "ok": not mism and not not_ok and not differ and not apart_mism}


def plain_listing_tie(paths: list[Path], records: list[dict[str, Any]],
                      plain: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """D-324 (5)(v): for each cache (split x pooling) of each extraction record, the items it lists in
    special_strings_as_text (extract_p2's list of the items of its split on the plain-text path, with their strings)
    against the plain-path items rebuilt here from its split file. A record naming user_span may lack the list (it was
    written before the rule): it then lists none."""
    rebuilt: dict[str, dict[str, Any]] = {}
    for p in plain:
        rebuilt.setdefault(p["file"], {})[str(p["item_id"])] = p["strings"]
    rows = []
    for path, rec in zip(paths, records):
        for key, s in sorted((rec.get("splits") or {}).items()):
            want = rebuilt.get(s.get("split_file"), {})
            row: dict[str, Any] = {"record": path.name, "key": key, "split_file": s.get("split_file"),
                                   "n_rebuilt": len(want), "problems": []}
            lst = s.get("special_strings_as_text")
            if lst is None and rec.get("user_span_source") == SPAN_TODAY:
                lst = {"n_items": 0, "items": []}
            items = lst.get("items") if isinstance(lst, dict) else None
            if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
                row["problems"].append("it lists no special_strings_as_text items")
                rows.append({**row, "ok": False})
                continue
            got = {str(i.get("item_id")): i.get("strings") for i in items}
            row["n_listed"] = len(items)
            if lst.get("n_items") != len(items) or len(got) != len(items):
                row["problems"].append("its n_items or the uniqueness of its items is not its list's")
            row.update(listed_not_rebuilt=sorted(set(got) - set(want)), rebuilt_not_listed=sorted(set(want) - set(got)),
                       strings_differ=sorted(i for i in set(got) & set(want) if got[i] != want[i]))
            if row["listed_not_rebuilt"] or row["rebuilt_not_listed"] or row["strings_differ"]:
                row["problems"].append("its plain-path items are not the ones rebuilt here")
            row["ok"] = not row["problems"]
            rows.append(row)
    return rows


def read_index(path: Path) -> dict[str, str]:
    """{cache file name: sha256} of an ingest index manifest ('<sha256>  data/hidden/<cache file>' lines)."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, _, f = line.partition("  ")
            out[Path(f.strip()).name] = h.strip()
    return out


def read_cache(p: Path, want_bytes: Any, hash_it: bool) -> tuple[dict[str, Any], tuple | None]:
    """One cache: (row facts, (item_ids, token_ids_sha256) or None). The size first (a cache of another size is not
    loaded); with hash_it the bytes are read once, hashed and loaded from the same buffer. A read error is a row
    fact, never a crash."""
    row: dict[str, Any] = {}
    try:
        st = p.stat()
        row.update(mode=oct(stat.S_IMODE(st.st_mode)), bytes_equal_record=st.st_size == want_bytes)
        if not row["bytes_equal_record"]:
            return row, None
        src: Any = p
        if hash_it:
            raw = p.read_bytes()
            row["sha256"] = sha256_bytes(raw)
            src = io.BytesIO(raw)
        with np.load(src, allow_pickle=False) as z:
            return row, (_str(z["item_ids"]), _str(z["token_ids_sha256"]))
    except Exception as e:  # noqa: BLE001 - any failure to read a cache is a failed row
        row["error"] = f"{type(e).__name__}: {e}"[:300]
        return row, None


def tie_rows(rec: dict[str, Any], rec_rel: str, idx: dict[str, str], hashes: dict[tuple[str, str], str], root: Path,
             file_sha: dict[str, str], hash_caches: bool) -> list[dict[str, Any]]:
    """Every cache of one ingested record (an extraction record or its p2sub subset) against the ids_sha256 of the same
    items. A row carries its arrays under '_arrays' for the pooling check; run() drops them."""
    rows = []
    for key, s in sorted((rec.get("splits") or {}).items()):
        p = root / "data" / "hidden" / s["cache_file"]
        row: dict[str, Any] = {"key": key, "record": rec_rel, "cache_file": s["cache_file"], "split": s.get("split"),
                               "pooling": s.get("pooling"), "split_file": s.get("split_file"),
                               "found": p.is_file(), "problems": []}
        if file_sha.get(s.get("split_file")) != s.get("split_sha256"):
            row["problems"].append("the record's split_sha256 is not part A's")
        if idx.get(s["cache_file"]) != s.get("cache_sha256"):
            row["problems"].append("the ingest index names other bytes than the record")
        if p.is_symlink() or not row["found"]:
            row["problems"].append("a symlink, not a regular file" if p.is_symlink() else
                                   "not in data/hidden although its record is indexed")
            rows.append({**row, "ok": False})
            continue
        facts, arrays = read_cache(p, s.get("cache_bytes"), hash_caches)
        row.update(facts)
        if facts.get("mode") != "0o444":
            row["problems"].append(f"mode {facts.get('mode')}, not 0o444")
        if not facts.get("bytes_equal_record"):
            row["problems"].append("its size is not the record's cache_bytes")
        if "sha256" in facts and facts["sha256"] != s.get("cache_sha256"):
            row["problems"].append("its sha256 is not the record's")
        if "error" in facts:
            row["problems"].append("it could not be read")
        if arrays is not None:
            iid, h = arrays
            want = {str(i): hashes.get((s.get("split_file"), str(i))) for i in iid}
            ours = {i for f, i in hashes if f == s.get("split_file")}
            row.update(n_items=len(iid),
                       mismatched_item_ids=[str(i) for i, x in zip(iid, h) if want[str(i)] not in (None, x)],
                       cache_items_not_in_split=sorted(i for i, w in want.items() if w is None),
                       split_items_not_in_cache=sorted(ours - set(want)))
            row["n_equal"] = len(iid) - len(row["mismatched_item_ids"]) - len(row["cache_items_not_in_split"])
            if len(want) != len(iid) or len(iid) != s.get("n"):
                row["problems"].append("its item count or uniqueness is not the record's")
            if row["n_equal"] != len(iid) or row["split_items_not_in_cache"]:
                row["problems"].append("its token_ids_sha256 or item set differs from the items rebuilt here")
            row["_arrays"] = arrays
        row["ok"] = not row["problems"] and arrays is not None
        rows.append(row)
    return rows


def check_subset(sub: dict[str, Any], rec: dict[str, Any], repo: str, rev: str, set_name: str) -> list[str]:
    """What keeps a p2sub record from standing for an extraction record's caches (empty when nothing)."""
    out = []
    if (sub.get("run_kind"), sub.get("repo"), sub.get("revision"), sub.get("set_name")) != ("subset_p2", repo, rev,
                                                                                           set_name):
        out.append("its run_kind, repo, revision or set is not the extraction record's")
    a, b = sub.get("splits") or {}, rec.get("splits") or {}
    if set(a) != set(b):
        out.append(f"its splits {sorted(a)} are not the extraction record's {sorted(b)}")
    for k in set(a) & set(b):
        if (a[k].get("source_cache_file"), a[k].get("source_cache_sha256"), a[k].get("split_file")) != \
                (b[k].get("cache_file"), b[k].get("cache_sha256"), b[k].get("split_file")):
            out.append(f"{k}: its source cache or split file is not the extraction record's")
    return out


def cache_tie(repo: str, rev: str, paths: list[Path], records: list[dict[str, Any]], hashes: dict[tuple[str, str], str],
              root: Path, dirs: list[Path], file_sha: dict[str, str], hash_caches: bool) -> dict[str, Any]:
    """The model's caches, through their durable ingested copies, with coverage and orphan checks. status: ok, failed,
    incomplete (some split file x pooling not tied) or skipped (no extraction record and no cache)."""
    slug, p2 = repo.replace("/", "__"), root / "results" / "phase2"
    named, sources, rows = set(), [], []
    for path, rec in zip(paths, records):
        named |= {s["cache_file"] for s in rec["splits"].values()}
        set_name = rec.get("set_name")
        src: dict[str, Any] = {"set": set_name, "extraction_record": TC._listed(path), "durable": None}
        sub_path = p2 / f"extract__{slug}__{SUB.TAG}@{rev[:12]}__{set_name}.json"
        sub = read_result(sub_path) if sub_path.is_file() else None
        if sub is not None:
            named |= {s["cache_file"] for s in (sub.get("splits") or {}).values()}
            src["subset_record"] = TC._listed(sub_path)
        for cand_path, cand, what in ((path, rec, "the extraction record (ingested)"),
                                      (sub_path, sub, "its p2sub subset record (ingested)")):
            ip = root / "manifests" / "ingest_p2" / f"{cand_path.stem}.sha256"
            if cand is None or not ip.is_file():
                continue
            problems = check_subset(cand, rec, repo, rev, set_name) if cand is sub else []
            if cand.get("ingest") != "yes":
                problems.append(f"its ingest is {cand.get('ingest')!r}, not 'yes'")
            if ip.is_symlink():
                problems.append("its ingest index is a symlink")
            src.update(durable=what, index={"file": _rel(ip, root), "sha256": sha256_file(ip)}, problems=problems)
            rows += tie_rows(cand, _rel(cand_path, root), read_index(ip), hashes, root, file_sha, hash_caches)
            break
        sources.append(src)
    found = sorted({p.name for d in dirs for tag in (EP.TAG, SUB.TAG) for p in Path(d).glob(f"{slug}__{tag}@*__*.npz")})
    orphans = [n for n in found if n not in named]                    # includes every cache of another revision
    other_rev = [n for n in found if not any(n.startswith(f"{slug}__{tag}@{rev[:12]}__") for tag in (EP.TAG, SUB.TAG))]
    by_split: dict[str, list[tuple]] = {}
    for r in rows:
        if "_arrays" in r:
            by_split.setdefault(r["split"], []).append(r.pop("_arrays"))
    agree = {sp: all(np.array_equal(v[0][0], a) and np.array_equal(v[0][1], b) for a, b in v[1:])
             for sp, v in by_split.items()}
    tied = {(r["split_file"], r["pooling"]) for r in rows if r["ok"]}
    want = {(f, pl) for f in file_sha for pl in POOLINGS}
    failed = orphans or not all(r["ok"] for r in rows) or not all(agree.values()) \
        or any(s.get("problems") for s in sources)
    status = ("skipped" if not records and not orphans else "failed" if failed
              else "incomplete" if want - tied else "ok")
    return {"status": status, "ok": status == "ok", "sources": sources, "caches": rows, "poolings_agree": agree,
            "orphan_caches": orphans, "other_revision_caches": other_rev,
            "not_tied": sorted([list(x) for x in want - tied]),
            "records_extract_p2_sha256": sorted({r["code"]["sha256"]["src/extract_p2.py"] for r in records})}


# --- part B: prompts, models, scan ---------------------------------------------------------------------------------

def phase1_files(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """(every Phase 1 split file the dataset manifests list, with its sha256 and its index manifest's n; the manifests
    read). Each file must match its index manifest manifests/splits/<split>__manifest.json, and no index manifest may
    be left unlisted. run() checks n against the rows read."""
    files: dict[str, str] = {}
    read: list[Path] = []
    for mp in sorted((root / "results").glob("dataset_manifest*.json")):
        read.append(mp)
        for axis, a in (read_result(mp).get("axes") or {}).items():
            for part, q in ((a or {}).get("partitions") or {}).items():
                f, h = q.get("file"), q.get("sha256")
                if not (isinstance(f, str) and f.startswith("data/frozen/") and isinstance(h, str)):
                    refuse(f"{mp.name}: {axis}/{part} names no file under data/frozen/ with a sha256")
                if files.setdefault(f, h) != h:
                    refuse(f"{f} is listed with two sha256s")
    idx_dir, out = root / "manifests" / "splits", []
    for f, h in files.items():
        ip = idx_dir / f"{Path(f).stem}__manifest.json"
        if not ip.is_file():
            refuse(f"{f} has no index manifest {ip.name}")
        read.append(ip)
        idx = read_result(ip)
        if idx.get("split_file") != Path(f).name or idx.get("sha256_of_original") != h:
            refuse(f"{ip.name} names {idx.get('split_file')} at {str(idx.get('sha256_of_original'))[:12]}, not {f}")
        out.append({"file": f, "sha256": h, "n_index_manifest": idx.get("n")})
    unlisted = sorted(p.name for p in idx_dir.glob("*__manifest.json")
                      if f"data/frozen/{p.name.removesuffix('__manifest.json')}.jsonl" not in files)
    if unlisted:
        refuse(f"index manifests with no dataset manifest entry: {unlisted}")
    return out, [TC._listed(p) for p in read]


def phase1_models(root: Path) -> tuple[dict[tuple[str, str], list[str]], list[dict[str, str]]]:
    """({(repo, revision): the split files its records name} for every model an original Phase 1 extraction record
    names; the records read). A LoRA arm or a tagged re-extraction names its base model as 'repo'."""
    out: dict[tuple[str, str], set] = {}
    paths = [p for p in sorted((root / "results").glob("extract__*.json")) if "__rerun_" not in p.name]
    for p in paths:
        e = read_result(p)
        if not (isinstance(e.get("repo"), str) and isinstance(e.get("revision"), str)):
            refuse(f"{p.name} names no repo and revision")
        files = out.setdefault((e["repo"], e["revision"]), set())
        files |= {f"data/frozen/{k.split('/')[0]}__{k.split('/')[1]}.jsonl" for k in (e.get("splits") or {})}
    return {k: sorted(v) for k, v in out.items()}, [TC._listed(p) for p in paths]


def atomic_strings(tok: Any) -> dict[str, dict[str, Any]]:
    """Every string the tokenizer may match atomically in raw text: each added token (special or not), and each
    all_special_tokens string that is not an added token (flagged)."""
    out = {t.content: {"special": bool(t.special), "added": True} for t in tok.added_tokens_decoder.values()}
    for s in tok.all_special_tokens:
        out.setdefault(s, {"special": True, "added": False})
    return out


def hit_counts(rows: list[dict[str, Any]], with_template: bool) -> dict[str, Any]:
    """The headline counts over some matched prompts. A prompt is a substring match only when neither path gives an
    added id and its template path (if any) did not refuse; a refusal is counted apart."""
    refused = [isinstance(r["template"], dict) and "refused" in r["template"] for r in rows]
    tpl = [None if x or r["template"] is None else r["template"] for r, x in zip(rows, refused)]
    raw_hit = [bool(r["raw"]["special"] or r["raw"]["nonspecial"]) for r in rows]
    tpl_hit = [bool(t and (t["special"] or t["nonspecial"])) for t in tpl]
    return {"n_prompts_with_any": len(rows), "n_distinct_items_with_any": len({r["item_id"] for r in rows}),
            "n_tokenized_special_raw": sum(bool(r["raw"]["special"]) for r in rows),
            "n_tokenized_nonspecial_raw": sum(bool(r["raw"]["nonspecial"]) for r in rows),
            "n_tokenized_special_template": sum(bool(t and t["special"]) for t in tpl) if with_template else None,
            "n_tokenized_nonspecial_template": sum(bool(t and t["nonspecial"]) for t in tpl) if with_template else None,
            "n_template_refused": sum(refused),
            "n_substring_only": sum(not a and not b and not x for a, b, x in zip(raw_hit, tpl_hit, refused))}


def scan_model(tok: Any, template_span: Callable | None, prompts: list[tuple[str, str, str]],
               occ: dict[str, list[int]], own_files: list[str] | None) -> dict[str, Any]:
    """The prompts holding any atomic string of this tokenizer, and the added-token ids each really tokenizes to.
    occ caches, per string, the indices of the prompts holding it (shared across models)."""
    strings = atomic_strings(tok)
    added = {int(i): bool(t.special) for i, t in tok.added_tokens_decoder.items()}
    hit: dict[int, list[str]] = {}
    for s in strings:
        if s not in occ:
            occ[s] = [k for k, (_, _, text) in enumerate(prompts) if s in text]
        for k in occ[s]:
            hit.setdefault(k, []).append(s)

    def parsed(ids: list[int]) -> dict[str, list[str]]:
        found = [int(i) for i in ids if int(i) in added]
        return {"special": [str(tok.convert_ids_to_tokens(i)) for i in found if added[i]],
                "nonspecial": [str(tok.convert_ids_to_tokens(i)) for i in found if not added[i]]}

    rows = []
    for k in sorted(hit):
        f, iid, text = prompts[k]
        row: dict[str, Any] = {"file": f, "item_id": iid, "strings": sorted(hit[k]),
                               "raw": parsed(tok(text, add_special_tokens=False)["input_ids"]), "template": None}
        if template_span is not None:
            try:
                ids, s, e, _ = template_span(text)
                row["template"] = parsed(ids[s:e])
            except SystemExit as ex:
                row["template"] = {"refused": str(ex)[:300]}
        rows.append(row)
    by_file: dict[str, dict[str, int]] = {}
    for f, _, _ in prompts:
        by_file.setdefault(f, {"n_prompts": 0, "n_with_any": 0})["n_prompts"] += 1
    for r in rows:
        by_file[r["file"]]["n_with_any"] += 1
    own = None
    if own_files is not None:
        own = {"files": own_files, "n_prompts": sum(by_file.get(f, {}).get("n_prompts", 0) for f in own_files),
               **hit_counts([r for r in rows if r["file"] in own_files], template_span is not None)}
    return {
        "n_atomic_strings": {"special": sum(v["special"] and v["added"] for v in strings.values()),
                             "nonspecial": sum(not v["special"] for v in strings.values()),
                             "special_not_added": sorted(s for s, v in strings.items() if not v["added"])},
        "n_prompts": len(prompts), "all_prompts": hit_counts(rows, template_span is not None),
        "own_files": own, "by_file": by_file,
        "strings_found": {s: {"special": strings[s]["special"], "n_prompts": len(occ[s])}
                          for s in sorted(strings) if occ[s]},
        "prompts_listed": f"all {len(rows)}" if len(rows) <= LIST_MAX else f"the first {LIST_MAX} of {len(rows)}",
        "prompts": rows[:LIST_MAX]}


# --- inputs, one Phase 2 model, the run ----------------------------------------------------------------------------

def load_inputs(root: Path) -> dict[str, Any]:
    """Everything read before any tokenizer: the records, the npz, the split files and the Phase 1 models."""
    rp = root / "results"
    st_path, pr_path = rp / "phase2" / "tokenizer_static.json", rp / "phase2" / "tokenizer_prompts.json"
    m2_path, m1_path, ac_path = rp / "phase2" / "dataset_manifest_p2.json", rp / "dataset_manifest.json", \
        rp / "access_check.json"
    st, pr, m2, m1, ac = (read_result(p) for p in (st_path, pr_path, m2_path, m1_path, ac_path))
    libs = TC._libraries()
    off = {k: (st["libraries"].get(k), libs.get(k)) for k in ("transformers", "tokenizers")
           if libs.get(k) is None or libs.get(k) != st["libraries"].get(k)}
    if off:
        refuse(f"the tokenization libraries differ from tokenizer_static.json's (recorded, now): {off}")
    npz_path = root / pr["ids_npz"]["file"]
    npz_bytes = npz_path.read_bytes()
    if sha256_bytes(npz_bytes) != pr["ids_npz"]["sha256"]:
        refuse(f"{npz_path.name} is not the file tokenizer_prompts.json names")
    files = TC.split_files(m2, m1)
    if {(f["file"], f["sha256"]) for f in files} != {(f["file"], f["sha256"]) for f in pr["inputs"]["split_files"]}:
        refuse("the split files are not the ones tokenizer_prompts.json read")
    items = [(fr["file"], r["item_id"], r["text"]) for fr in files for r in TC.read_split(root, fr)]
    p1_files, p1_manifests = phase1_files(root)
    file_sha = {f["file"]: f["sha256"] for f in files}
    all_files = dict(file_sha)
    prompts = list(items)
    for f in p1_files:
        if all_files.setdefault(f["file"], f["sha256"]) != f["sha256"]:
            refuse(f"{f['file']} has two sha256s across the Phase 1 and Phase 2 manifests")
        rows = TC.read_split(root, f)
        if len(rows) != f["n_index_manifest"]:
            refuse(f"{f['file']} has {len(rows)} rows, its index manifest says {f['n_index_manifest']}")
        if f["file"] not in file_sha:                                 # XSTest is in both lists: read once
            prompts += [(f["file"], r["item_id"], r["text"]) for r in rows]
    p1_models, p1_records = phase1_models(root)
    gated = {t.get("repo_id"): t.get("gated") for t in ac.get("targets") or []}
    return {"st": st, "static_sha": TC._listed(st_path)["sha256"], "libs": libs,
            "z": np.load(io.BytesIO(npz_bytes), allow_pickle=False), "files": files, "file_sha": file_sha,
            "items": items, "prompts": prompts, "all_files": all_files, "p1_models": p1_models, "gated": gated,
            "inputs": {"tokenizer_static": TC._listed(st_path), "tokenizer_prompts": TC._listed(pr_path),
                       "tokenizer_prompts_ids": {"file": _rel(npz_path, root), "sha256": pr["ids_npz"]["sha256"]},
                       "dataset_manifest_p2": TC._listed(m2_path), "dataset_manifest": TC._listed(m1_path),
                       "access_check": TC._listed(ac_path), "split_files": files, "phase1_manifests": p1_manifests,
                       "phase1_split_files": p1_files, "phase1_extraction_records": p1_records}}


def check_model(repo: str, inp: dict[str, Any], root: Path, scratch: Path, api: Any, dirs: list[Path],
                hash_caches: bool) -> tuple[dict[str, Any], Any, Callable | None]:
    """Part A for one Phase 2 model: (its block, the loaded tokenizer, part B's template path: extract_p2.instruct_item
    with today's user_span, or None for base)."""
    t0 = time.time()
    st = inp["st"]
    sm = st["models"][repo]
    rev, kind = sm["revision"], sm["kind"]
    stem = f"extract__{repo.replace('/', '__')}__{EP.TAG}@{rev[:12]}__"
    paths = sorted((root / "results" / "phase2").glob(f"{stem}*.json"))
    records = [read_result(p) for p in paths]
    span_name, span_from = record_span(repo, records)          # before any fetch: an unknown or mixed name refuses
    try:
        d, source = tokenizer_dir(repo, rev, root, scratch, sm.get("gated"))
    except NotLoaded as e:
        refuse(f"{repo}, a Phase 2 model: {e}")
    check_files(repo, source["loaded_files_sha256"], sm, records)
    tok = TC.load_tokenizer(d)
    kw, tmpl = EP.template_setup(tok, kind, (st.get("clock") or {}).get("pin_date"), api)
    EP.check_tokenizer_static(st, repo, rev, kind, tmpl["chat_template_sha256"])
    for r in records:
        check_record(r, repo, sm, tmpl, inp["static_sha"])
    counted, counted_from = tok.bos_token_id, "the tokenizer's bos_token_id"
    if counted is None:
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8")) if (d / "config.json").is_file() else {}  # not a result (D-297)
        counted, counted_from = ((cfg.get("bos_token_id"), "config.json's bos_token_id (the tokenizer has no BOS "
                                  "token)") if isinstance(cfg.get("bos_token_id"), int) else (None, "nothing"))
    build = item_builder(tok, kind, kw, api, getattr(api, SPAN_FUNCTIONS[span_name]))    # the ids extract_p2 fed
    today = None if span_name == SPAN_TODAY else item_builder(tok, kind, kw, api)     # today's ids, for the npz
    pol = policy(build, counted, sm)
    hashes, summary = scan_items(build, inp["items"], counted, pol["bos_at_position_0"], today)
    plain = summary.pop("plain_path_items")
    listing = plain_listing_tie(paths, records, plain)
    out: dict[str, Any] = {
        "revision": rev, "kind": kind, "tokenizer_source": source, "span_function": span_name,
        "span_function_from": span_from,
        "added_tokens_equal_tokenizer_static": TC._added(tok) == sm["tokenizer"]["added_tokens"],
        "bos_token": tok.bos_token, "bos_token_id": tok.bos_token_id, "eos_token_id": tok.eos_token_id,
        "counted_id": counted, "counted_id_from": counted_from,
        "counted_id_equals_eos_id": counted is not None and counted == tok.eos_token_id,
        "template": {k: tmpl[k] for k in ("chat_template_sha256", "template_used", "template_kwargs", "date_pin")},
        "policy": pol, **summary,
        "plain_path": {"rule": "D-324 (5)(v), phase2_tokenizer_check.PLAIN_RULE: the items the span function put on "
                               "the plain-text path; every other item's ids are today's",
                       "n_items": len(plain), "items": plain, "record_listing": listing,
                       "ok": all(r["ok"] for r in listing)},
        "prompts_tie": tie_prompts(inp["z"], repo, hashes,
                                   {(p["file"], p["item_id"]): p["today_ids_sha256"] for p in plain}),
        "record_split_files": sorted({s["split_file"] for r in records for s in r["splits"].values()}),
        "cache_tie": cache_tie(repo, rev, paths, records, hashes, root, dirs, inp["file_sha"], hash_caches),
        "seconds": round(time.time() - t0, 1)}
    span = (lambda text: EP.instruct_item(tok, text, kw, api.user_span)) if kind == "instruct" else None
    return out, tok, span


def load_phase1(repo: str, rev: str, root: Path, scratch: Path, gated: Any) -> tuple[Any, dict[str, Any]]:
    """A Phase 1-only model's tokenizer (or None) and the head of its scan block."""
    head = {"repo": repo, "revision": rev, "role": "Phase 1 only"}
    try:
        d, source = tokenizer_dir(repo, rev, root, scratch, gated)
    except NotLoaded as e:
        return None, {**head, "loaded": False, "why": str(e)[:600]}
    return TC.load_tokenizer(d), {**head, "loaded": True, "tokenizer_source": source}


def _scan_line(name: str, s: dict[str, Any]) -> str:
    if not s.get("loaded", True):
        return f"  scan {name:<44} NOT LOADED: {s['why'][:160]}"
    a, found = s["all_prompts"], [(k, v["n_prompts"]) for k, v in s["strings_found"].items()]
    return (f"  scan {name:<44} prompts with any {a['n_prompts_with_any']} (items {a['n_distinct_items_with_any']}) "
            f"special raw/template {a['n_tokenized_special_raw']}/{a['n_tokenized_special_template']} non-special "
            f"{a['n_tokenized_nonspecial_raw']}/{a['n_tokenized_nonspecial_template']} "
            f"refused {a['n_template_refused']} "
            f"substring only {a['n_substring_only']} strings {found}")


def run(root: Path, out: Path, models: list[str] | None, cache_dirs: list[str] | None, scratch: str | None,
        hash_caches: bool) -> int:
    root, out = Path(root), Path(out)
    final = out.resolve() == FINAL.resolve()
    if out.exists():
        refuse(f"{out.name} exists; results are never overwritten")
    if final and (root.resolve() != ROOT.resolve() or models is not None or cache_dirs is not None
                  or not hash_caches):
        refuse("the record reads the repo's own inputs, every model and the default cache directories, with "
               "--hash-caches (D-307 (1): the bytes of every cache read are named)")
    if not final and any(out.resolve().is_relative_to((r / "results").resolve()) for r in (root, ROOT)):
        refuse("a test run (--out) must not write under results/")
    if scratch and Path(scratch).exists() and any(Path(scratch).iterdir()):
        refuse(f"--scratch {scratch} is not empty; give a new directory (tokenizer files are never mixed between runs)")
    code0 = P.code_block(code_files())
    inp = load_inputs(root)
    st = inp["st"]
    p1_only = {f"{r}@{v[:12]}": (r, v) for r, v in inp["p1_models"]
               if (st["models"].get(r) or {}).get("revision") != v}
    names = [*st["models"], *p1_only]
    chosen = names if models is None else models
    unknown = [m for m in chosen if m not in names]
    if unknown:
        refuse(f"neither a Phase 2 model nor a Phase 1-only '<repo>@<rev12>': {unknown}")
    dirs = [root / "data" / "hidden", *[Path(c) for c in (cache_dirs or [str(P2CACHE), str(P2SUB)])]]
    for name in chosen:        # every model's span function first: an unknown or mixed name refuses before any fetch
        rev = (st["models"].get(name) or {}).get("revision")     # check_model itself requires it
        if rev:
            stem = f"extract__{name.replace('/', '__')}__{EP.TAG}@{rev[:12]}__"
            record_span(name, [read_result(p) for p in sorted((root / "results" / "phase2").glob(f"{stem}*.json"))])
    api = EP.span_api()
    if not scratch:
        SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)
    work = Path(scratch) if scratch else Path(tempfile.mkdtemp(prefix="gccf_bos_", dir=SCRATCH_PARENT))
    print(f"phase2_bos_check: part A {len(inp['items'])} items; part B {len(inp['prompts'])} prompts from "
          f"{len(inp['all_files'])} files; {len(chosen)} models", flush=True)
    per_model: dict[str, Any] = {}
    scan: dict[str, Any] = {}
    occ: dict[str, list[int]] = {}
    try:
        for name in chosen:
            if name in st["models"]:
                m, tok, span = check_model(name, inp, root, work, api, dirs, hash_caches)
                per_model[name] = m
                diff = [(d["item_id"], d["later_bos_positions"]) for d in m["items_differing_from_policy"]]
                print(f"  {name:<40} {m['span_function'].rsplit('.', 1)[-1]} (plain-path items "
                      f"{m['plain_path']['n_items']}, listing ok {m['plain_path']['ok']}) "
                      f"policy {m['policy']['bos_at_position_0']} counts {m['bos_count_distribution']} "
                      f"differing {diff} prompts tie {m['prompts_tie']['n_equal']}/{m['prompts_tie']['n_compared']} "
                      f"cache tie {m['cache_tie']['status']} ({m['seconds']} s)", flush=True)
                p1_files = inp["p1_models"].get((name, m["revision"]))
                own = sorted(set(m["record_split_files"]) | set(p1_files or [])) or None
                head = {"repo": name, "revision": m["revision"], "loaded": True,
                        "role": "Phase 2 and Phase 1" if p1_files is not None else "Phase 2",
                        "tokenizer_source": "as in part A (models.<repo>.tokenizer_source)",
                        "span_function": SPAN_TODAY if span is not None else None}
            else:
                repo, rev = p1_only[name]
                tok, head = load_phase1(repo, rev, root, work, inp["gated"].get(repo))
                span, own = None, inp["p1_models"][(repo, rev)]
                if tok is None:
                    scan[name] = head
                    print(_scan_line(name, head), flush=True)
                    continue
            scan[name] = {**head, **scan_model(tok, span, inp["prompts"], occ, own)}
            print(_scan_line(name, scan[name]), flush=True)
    finally:
        if not scratch:
            TC.clean_scratch(work)                      # never deletes a file over 1 MB (the author's rule)
    if P.code_block(code_files())["sha256"] != code0["sha256"]:
        refuse("the code changed during the run, or a module was loaded mid-run; nothing is written")
    return finish(out, root, inp, code0, models, dirs, hash_caches, per_model, scan)


def finish(out: Path, root: Path, inp: dict[str, Any], code0: dict[str, Any], models: list[str] | None,
           dirs: list[Path], hash_caches: bool, per_model: dict[str, Any], scan: dict[str, Any]) -> int:
    """The verdict, the record, the last line and the exit code (3 when any part A tie (the npz, the caches, the
    plain-path listing) failed or is incomplete, or a tokenizer could not be loaded)."""
    status = {r: m["cache_tie"]["status"] for r, m in per_model.items()}
    failed = sorted(r for r, m in per_model.items()
                    if not m["prompts_tie"]["ok"] or status[r] == "failed" or not m["plain_path"]["ok"])
    skipped = sorted(r for r, s in status.items() if s == "skipped")
    incomplete = sorted(r for r, s in status.items() if s == "incomplete")
    not_loaded = sorted(n for n, s in scan.items() if not s.get("loaded", True))
    observed = {c["cache_file"]: (c["sha256"], r) for r, m in per_model.items()
                for c in m["cache_tie"]["caches"] if "sha256" in c}
    config = {k: v for f, (_, r) in sorted(observed.items())
              for k, v in ((f"{f}_repo", r), (f"{f}_rev", per_model[r]["revision"]))}
    payload = {
        **provenance_block("phase2_bos_check"), "code": code0, "libraries": inp["libs"],
        "preregistration": "D-318 (6): confirm from token ids the BOS count of every Phase 2 cache (part A); D-324 (5) "
                           "(iv): scan every Phase 1 and Phase 2 prompt for each tokenizer's literal special-token "
                           "strings (part B)",
        "inputs": inp["inputs"], "config": config,
        "options": {"models": models or "all", "cache_dirs": [_rel(d, root) for d in dirs], "hash_caches": hash_caches},
        "id_sha256_method": "phase2_tokenizer_check.ids_sha256: sha256 of the ASCII decimal ids joined by ','",
        "listing_rule": "items are listed only where their pattern differs from the policy; every other item has "
                        "exactly the policy pattern (the counted id at position 0 if and only if the policy is 1, no "
                        "later occurrence)",
        "span_rule": f"part A rebuilds each model's items with the span function its extraction records name "
                     f"(user_span_source: {SPAN_TODAY} or {SPAN_PLAIN}, one per model; {SPAN_TODAY} for a model with "
                     "no record, the rule tokenizer_prompts_ids.npz was built with); the BOS counts and the cache tie "
                     "use the ids fed, the npz tie today's ids (a plain-path item's today's ids, never its plain "
                     "ones); D-335 (5)",
        "span_function_by_model": {r: m["span_function"] for r, m in per_model.items()},
        "plain_path_items": {r: [[p["file"], p["item_id"]] for p in m["plain_path"]["items"]]
                             for r, m in per_model.items()},
        "models": per_model, "ties_failed": failed, "cache_tie_skipped": skipped, "cache_tie_incomplete": incomplete,
        "scan_not_loaded": not_loaded,
        "items_with_more_than_one_bos": {r: [d["item_id"] for d in m["items_differing_from_policy"]
                                             if d["bos_at_position_0"] + len(d["later_bos_positions"]) > 1]
                                         for r, m in per_model.items()},
        "special_token_scan": {
            "strings_rule": "every added token's content (special or not: a fast tokenizer splits its whole added "
                            "vocabulary out of raw text; split_special_tokens=True would turn off only the special "
                            "ones), plus every all_special_tokens string that is not an added token",
            "parsing_rule": f"raw: tok(text, add_special_tokens=False), the user content as Phase 1 fed it; template: "
                            f"the user span of extract_p2.instruct_item with {SPAN_TODAY} (Phase 2 instruct models); "
                            "special and non-special added ids apart; a refused template path counted apart; prompts "
                            "with no literal string are not tokenized",
            "n_prompts": len(inp["prompts"]),
            "prompt_files": [{"file": f, "sha256": h} for f, h in inp["all_files"].items()], "models": scan}}
    if observed:
        payload["input_cache_sha256"] = {f: h for f, (h, _) in sorted(observed.items())}
    digest = write_results(out, payload, exclusive=True)
    print(f"Written: {_rel(out, root)}\nsha256:  {digest}")
    print(f"ties failed: {failed or 'none'}; cache tie skipped: {skipped or 'none'}; "
          f"cache tie incomplete: {incomplete or 'none'}; scan not loaded: {not_loaded or 'none'}", flush=True)
    return 3 if failed or incomplete or not_loaded else 0


# --- --axis p2sst: part B over the SST-2 prompts (D-338 (4), D-340 (3)) ---------------------------------------------

def load_inputs_sst(root: Path) -> dict[str, Any]:
    """The SST-2 prompts (the files of dataset_manifest_p2sst.json, each sha256-checked when read, and exactly the files
    tokenizer_prompts_p2sst.json read), tokenizer_static.json, access_check.json and the Phase 1 models."""
    rp = root / "results"
    st_path, pr_path = rp / "phase2" / "tokenizer_static.json", rp / "phase2" / "tokenizer_prompts_p2sst.json"
    m_path, ac_path = rp / "phase2" / "dataset_manifest_p2sst.json", rp / "access_check.json"
    for p in (st_path, pr_path, m_path, ac_path):
        if not p.is_file():
            refuse(f"{_rel(p, root)} is not on disk (the SST-2 scan runs after the p2sst prompts check)")
    st, pr, m, ac = (read_result(p) for p in (st_path, pr_path, m_path, ac_path))
    libs = TC._libraries()
    off = {k: (st["libraries"].get(k), libs.get(k)) for k in ("transformers", "tokenizers")
           if libs.get(k) is None or libs.get(k) != st["libraries"].get(k)}
    if off:
        refuse(f"the tokenization libraries differ from tokenizer_static.json's (recorded, now): {off}")
    if pr.get("axis") != TC.SST_AXIS or pr.get("run_kind") != "phase2_tokenizer_prompts":
        refuse(f"{pr_path.name} is not the p2sst prompts check")
    files = TC.split_files(m, None, TC.SST_AXIS)
    if {(f["file"], f["sha256"]) for f in files} != {(f["file"], f["sha256"]) for f in pr["inputs"]["split_files"]}:
        refuse(f"the SST-2 split files are not the ones {pr_path.name} read")
    prompts = []
    for fr in files:
        rows = TC.read_split(root, fr)
        bad = [r["item_id"] for r in rows if r.get("axis") != TC.SST_AXIS]
        if bad:
            refuse(f"{fr['file']}: rows not naming axis {TC.SST_AXIS} (e.g. {bad[:3]}); prompts are keyed by axis")
        prompts += [(fr["file"], r["item_id"], r["text"]) for r in rows]
    p1_models, p1_records = phase1_models(root)
    gated = {t.get("repo_id"): t.get("gated") for t in ac.get("targets") or []}
    return {"st": st, "libs": libs, "files": files, "prompts": prompts,
            "all_files": {f["file"]: f["sha256"] for f in files}, "p1_models": p1_models, "gated": gated,
            "inputs": {"tokenizer_static": TC._listed(st_path), "tokenizer_prompts_p2sst": TC._listed(pr_path),
                       "dataset_manifest_p2sst": TC._listed(m_path), "access_check": TC._listed(ac_path),
                       "split_files": files, "phase1_extraction_records": p1_records}}


def scan_head_sst(repo: str, inp: dict[str, Any], root: Path, scratch: Path,
                  api: Any) -> tuple[dict[str, Any], Any, Callable | None, list[str] | None]:
    """One Phase 2 model's tokenizer for the SST-2 scan, loaded and checked as part A loads it (tokenizer_static.json's
    files and those its p2tpl extraction records list, the template rule): (its head, the tokenizer, the template path
    with today's user_span or None for base, the p2sst files its own p2sst extraction records name or None)."""
    st = inp["st"]
    sm = st["models"][repo]
    rev, kind = sm["revision"], sm["kind"]
    p2 = root / "results" / "phase2"
    slug = repo.replace("/", "__")
    paths = sorted(p2.glob(f"extract__{slug}__{EP.TAG}@{rev[:12]}__*.json"))
    records = [read_result(p) for p in paths]
    sst_paths = sorted(p2.glob(f"extract__{slug}__{EP.SST_TAG}@{rev[:12]}__*.json"))
    own = sorted({s["split_file"] for p in sst_paths for s in (read_result(p).get("splits") or {}).values()}) or None
    try:
        d, source = tokenizer_dir(repo, rev, root, scratch, sm.get("gated"))
    except NotLoaded as e:
        refuse(f"{repo}, a Phase 2 model: {e}")
    check_files(repo, source["loaded_files_sha256"], sm, records)
    tok = TC.load_tokenizer(d)
    kw, tmpl = EP.template_setup(tok, kind, (st.get("clock") or {}).get("pin_date"), api)
    EP.check_tokenizer_static(st, repo, rev, kind, tmpl["chat_template_sha256"])
    span = (lambda text: EP.instruct_item(tok, text, kw, api.user_span)) if kind == "instruct" else None
    head = {"repo": repo, "revision": rev, "loaded": True, "role": "Phase 2", "tokenizer_source": source,
            "span_function": SPAN_TODAY if span is not None else None,
            "tokenizer_files_also_checked_against": [TC._listed(p) for p in paths],
            "own_p2sst_records": [TC._listed(p) for p in sst_paths]}
    return head, tok, span, own


def run_sst(root: Path, out: Path, models: list[str] | None, scratch: str | None) -> int:
    """--axis p2sst: part B over the SST-2 prompts (module docstring). Exit 0, or 3 when a tokenizer was not loaded."""
    root, out = Path(root), Path(out)
    final = out.resolve() == FINAL_SST.resolve()
    if out.exists():
        refuse(f"{out.name} exists; results are never overwritten")
    if final and (root.resolve() != ROOT.resolve() or models is not None):
        refuse("the SST-2 record reads the repo's own inputs and every model")
    if not final and any(out.resolve().is_relative_to((r / "results").resolve()) for r in (root, ROOT)):
        refuse("a test run (--out) must not write under results/")
    if scratch and Path(scratch).exists() and any(Path(scratch).iterdir()):
        refuse(f"--scratch {scratch} is not empty; give a new directory (tokenizer files are never mixed between runs)")
    code0 = P.code_block(code_files())
    inp = load_inputs_sst(root)
    st = inp["st"]
    p1_only = {f"{r}@{v[:12]}": (r, v) for r, v in inp["p1_models"]
               if (st["models"].get(r) or {}).get("revision") != v}
    names = [*st["models"], *p1_only]
    chosen = names if models is None else models
    unknown = [m for m in chosen if m not in names]
    if unknown:
        refuse(f"neither a Phase 2 model nor a Phase 1-only '<repo>@<rev12>': {unknown}")
    api = EP.span_api()
    if not scratch:
        SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)
    work = Path(scratch) if scratch else Path(tempfile.mkdtemp(prefix="gccf_bos_", dir=SCRATCH_PARENT))
    print(f"phase2_bos_check --axis p2sst: part B {len(inp['prompts'])} SST-2 prompts from {len(inp['all_files'])} "
          f"files; {len(chosen)} models", flush=True)
    scan: dict[str, Any] = {}
    occ: dict[str, list[int]] = {}
    try:
        for name in chosen:
            if name in st["models"]:
                head, tok, span, own = scan_head_sst(name, inp, root, work, api)
            else:
                repo, rev = p1_only[name]
                tok, head = load_phase1(repo, rev, root, work, inp["gated"].get(repo))
                span, own = None, None                 # a Phase 1-only model is never fed an SST-2 prompt
                if tok is None:
                    scan[name] = head
                    print(_scan_line(name, head), flush=True)
                    continue
            scan[name] = {**head, **scan_model(tok, span, inp["prompts"], occ, own)}
            print(_scan_line(name, scan[name]), flush=True)
    finally:
        if not scratch:
            TC.clean_scratch(work)                      # never deletes a file over 1 MB (the author's rule)
    if P.code_block(code_files())["sha256"] != code0["sha256"]:
        refuse("the code changed during the run, or a module was loaded mid-run; nothing is written")
    not_loaded = sorted(n for n, s in scan.items() if not s.get("loaded", True))
    payload = {
        **provenance_block("phase2_bos_check"), "code": code0, "libraries": inp["libs"], "axis": TC.SST_AXIS,
        "preregistration": "D-324 (5) (iv): scan every prompt for each tokenizer's literal special-token strings (part "
                           "B), here the SST-2 non-safety control's prompts before any p2sst extraction (D-338 (4), "
                           "D-340 (3)); part A is not run for p2sst",
        "inputs": inp["inputs"], "options": {"models": models or "all"},
        "part_a": "not run for p2sst: no p2sst cache is tied here; the p2sst extraction records carry each cache's "
                  "token_ids_sha256 and the items on the plain-text path",
        "keyed_by": "(file, item_id): an SST-2 item_id can equal a WildGuardMix one (D-340 (3))",
        "scan_not_loaded": not_loaded,
        "special_token_scan": {
            "strings_rule": "every added token's content (special or not: a fast tokenizer splits its whole added "
                            "vocabulary out of raw text; split_special_tokens=True would turn off only the special "
                            "ones), plus every all_special_tokens string that is not an added token",
            "parsing_rule": f"raw: tok(text, add_special_tokens=False); template: the user span of "
                            f"extract_p2.instruct_item with {SPAN_TODAY} (Phase 2 instruct models); special and "
                            "non-special added ids apart; a refused template path counted apart; prompts with no "
                            "literal string are not tokenized",
            "own_files_rule": "the p2sst files the model's own p2sst extraction records name (null before any)",
            "n_prompts": len(inp["prompts"]),
            "prompt_files": [{"file": f, "sha256": h} for f, h in inp["all_files"].items()], "models": scan}}
    digest = write_results(out, payload, exclusive=True)
    print(f"Written: {_rel(out, root)}\nsha256:  {digest}")
    print(f"scan not loaded: {not_loaded or 'none'}", flush=True)
    return 3 if not_loaded else 0


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Phase 2 BOS check and special-token scan (D-318 (6), D-324 (5))")
    ap.add_argument("--root", default=str(ROOT), help="where the inputs resolve (the repo; a sandbox run names it)")
    ap.add_argument("--axis", default="p2wg", choices=("p2wg", TC.SST_AXIS),
                    help=f"p2wg (the default: parts A and B) or {TC.SST_AXIS} (part B over the SST-2 prompts only; "
                         f"output {FINAL_SST.name})")
    ap.add_argument("--out", default=None, help=f"default {FINAL.name}, or {FINAL_SST.name} with --axis p2sst")
    ap.add_argument("--models", nargs="+", default=None,
                    help="a subset: Phase 2 repos and Phase 1-only '<repo>@<rev12>' (test or sandbox runs only)")
    ap.add_argument("--cache-dir", action="append", default=None,
                    help=f"searched for unrecorded caches after <root>/data/hidden; repeatable (default {P2CACHE} and "
                         f"{P2SUB})")
    ap.add_argument("--scratch", default=None, help="a new directory for the tokenizer files, left after the run "
                                                    f"(default: a temporary one under {SCRATCH_PARENT}, cleaned of "
                                                    "files of at most 1 MB)")
    ap.add_argument("--hash-caches", action="store_true",
                    help="read each cache once, hash it and load from the same bytes (required for the record)")
    args = ap.parse_args(argv)
    if args.axis == TC.SST_AXIS and (args.cache_dir is not None or args.hash_caches):
        refuse("--cache-dir and --hash-caches are part A's; --axis p2sst runs part B only")
    from hf_env_guard import assert_hf_env
    assert_hf_env("phase2_bos_check.py")
    if args.axis == TC.SST_AXIS:
        return run_sst(Path(args.root), Path(args.out or FINAL_SST), args.models, args.scratch)
    return run(Path(args.root), Path(args.out or FINAL), args.models, args.cache_dir, args.scratch, args.hash_caches)


if __name__ == "__main__":
    raise SystemExit(main())
