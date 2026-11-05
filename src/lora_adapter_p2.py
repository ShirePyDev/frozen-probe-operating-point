"""Phase 2 LoRA adapters for extraction (D-307 (2), D-343 (2), D-345, D-350): the adapter directory, the cache identity
and the recorded sha256s that one extraction of one checkpoint uses, READ from the training record and the adapter-ingest
index, never typed by hand. src/extract_p2.py --lora-record <record> --lora-step <N> calls resolve(); src/lora_jobs_p2.py
uses checkpoint() and identity() (no file on the NAS needed) to write the job lists.

INPUTS for one checkpoint (resolve):
  record   results/phase2/lora_train_p2__<corpus>__s<seed>.json (src/train_lora_p2.py), read through
           provenance.read_result, so the extraction record lists it with its sha256 (D-297 (1)). It must be a
           'lora_train_p2' record of that corpus and seed whose config.name, adapters.run_dir and ingest_manifest.run_dir
           agree; the step must be one of its config.save_steps and of STEPS; its 'checkpoints' entry for the step
           (dir step_<N>) names adapter_sha256 and adapter_config_sha256, which must equal that entry's files' sha256s
           and the 'ingest_manifest' block's (the files p2_ingest_lora.sh copied).
  index    manifests/ingest_lora/<record stem>.sha256 (scripts/p2_ingest_lora.sh: '<sha256>  data/lora/<run dir>/
           step_<N>/<file>', one line per file): a regular file listing exactly the ingest_manifest block's files with
           the same sha256s. Its sha256 is recorded. (The index is written before the copy, so its existence alone does
           not prove the ingest; the files below and extract_p2's own hashing do.)
  files    data/lora/<run dir>/step_<N>/adapter_model.safetensors and adapter_config.json: regular files (no symlink),
           mode 444 (the ingested copy, D-307 (2)). extract_p2.identity hashes both against the recorded values before
           use and again after the load (recheck_adapter).
  base     the record's config.repo and config.rev, which extract_p2 requires to equal --repo and --revision.

THE CACHE IDENTITY (extract_p2: lora/<name> at the sha256 of adapter_model.safetensors): name = '<run dir>-step<N>' for
N > 0, e.g. 'mistral7bi-v02-alpaca-r16-seed0-step25' (run dirs end in '-seed<N>', so the step reads as '-step<N>'; Phase
1's '-s<step>' form is not reused, as train_lora_p2.run_name keeps 'seed' apart from Phase 1's 's').

STEP 0 (choice by the closest D-entry, D-345 (1) and D-350: the step-0 adapter of a seed is the same bytes for every
corpus, checked by src/train_lora_p2.py when it was saved (step0_identity) and by scripts/p2_ingest_lora.sh across the
three corpora of the seed): it is extracted ONCE PER SEED, as 'lora/<head>-<tail>-step0' with the corpus tag removed from
the run dir (e.g. 'mistral7bi-v02-r16-seed0-step0'), from the record of the FIRST corpus of CORPUS_ORDER (alpaca); a
step-0 request through another corpus's record refuses, so the one-per-seed rule is enforced here, not only in the job
list. Every sibling record of the seed (hh, pku) must be on disk, read through read_result (so the extraction record lists
them), pass the same checks for step 0 with its own index, name the same step-0 adapter_sha256 (the weights, compared by
sha256 as D-345 (1) and scripts/p2_ingest_lora.sh (D-350) compare them) and the same run-dir head, tail and base; and its
ingested adapter_config.json (data/lora, 444, the bytes its record's adapter_config_sha256 names) must equal the
extracted one's AS PARSED JSON, with target_modules compared as a set: peft writes target_modules from a set, so its
order differs between the three corpora's step-0 saves while the bytes of the weights do not (fidelity review of
fed8bd1). The source block records the rule and every sibling checked, with each sibling's own config sha256.

CORPUS_ORDER is src/train_lora_p2.py's CORPORA keys and RUN_TAGS their 'tag' (tests/test_extract_p2.py,
test_corpus_order_and_run_tags_are_the_trainers, reads both from that file's source, without importing it: it loads
torch and the corpus readers). No weight is read here.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance as PV  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CORPUS_ORDER = ("alpaca", "hh", "pku")                       # src/train_lora_p2.py CORPORA, in its order
RUN_TAGS = {"alpaca": "alpaca", "hh": "hhharmless", "pku": "pkusafe"}   # CORPORA[c]['tag'] (run_name)
STEPS = (0, 25, 100, 400)                                    # D-343 (2): checkpoints at 0, 25, 100, 400 steps
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")
RECORD_RE = re.compile(r"lora_train_p2__([a-z0-9]+)__s([0-9]+)\.json")
RUN_DIR_RE = re.compile(r"(?P<head>[a-z0-9][a-z0-9.]*(?:-[a-z0-9.]+)*)-(?P<tag>[a-z0-9]+)-(?P<tail>r[0-9]+-seed[0-9]+)")
HEX64 = re.compile(r"[0-9a-f]{64}")
STEP0_RULE = ("D-345 (1), D-350: a seed's step-0 adapter weights are the same bytes for every corpus (train_lora_p2 "
              "step0_identity; scripts/p2_ingest_lora.sh compared their sha256 across the seed's three corpora), so it is "
              "extracted once per seed, from the record of the first corpus of CORPUS_ORDER, as lora/<run dir without its "
              "corpus tag>-step0; every sibling record of the seed is read and must name the same step-0 adapter_sha256, "
              "and its ingested adapter_config.json (the bytes its record names) must equal this one's as parsed JSON with "
              "target_modules compared as a set (peft writes it from a set: its order varies, the weights do not)")


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def record_path(corpus: str, seed: int, root: Path = ROOT) -> Path:
    return Path(root) / "results" / "phase2" / f"lora_train_p2__{corpus}__s{seed}.json"


def parse_name(path: Path) -> tuple[str, int]:
    m = RECORD_RE.fullmatch(Path(path).name)
    if not m or m.group(1) not in CORPUS_ORDER:
        refuse(f"{Path(path).name} is not lora_train_p2__<corpus>__s<seed>.json with a corpus of {CORPUS_ORDER}")
    return m.group(1), int(m.group(2))


def split_run_dir(run_dir: str, corpus: str) -> tuple[str, str]:
    """(head, tail) of '<head>-<corpus tag>-<tail>' (train_lora_p2.run_name)."""
    m = RUN_DIR_RE.fullmatch(str(run_dir))
    if not m or m.group("tag") != RUN_TAGS[corpus]:
        refuse(f"the run dir {run_dir!r} is not '<head>-{RUN_TAGS[corpus]}-r<rank>-seed<N>' (train_lora_p2.run_name)")
    return m.group("head"), m.group("tail")


def identity(run_dir: str, corpus: str, step: int) -> str:
    """The cache identity's name (lora/<name>): '<run dir>-step<N>'; step 0 without the corpus tag (once per seed)."""
    if step == 0:
        head, tail = split_run_dir(run_dir, corpus)
        return f"{head}-{tail}-step0"
    split_run_dir(run_dir, corpus)
    return f"{run_dir}-step{step}"


def checkpoint(rec: Any, step: int, corpus: str, seed: int, label: str) -> dict[str, Any]:
    """The checkpoint of one training record for one step, every recorded copy of its sha256s agreeing (no file is
    read): {run_dir, dir, sha256, config_sha256, base_repo, base_rev, files (the ingest_manifest block's)}."""
    if not isinstance(rec, dict) or rec.get("run_kind") != "lora_train_p2":
        refuse(f"{label} is not a lora_train_p2 record")
    cfg = rec.get("config") if isinstance(rec.get("config"), dict) else {}
    if (cfg.get("corpus"), cfg.get("seed")) != (corpus, seed):
        refuse(f"{label} names corpus {cfg.get('corpus')!r} and seed {cfg.get('seed')!r}, not its file name's")
    run_dir = cfg.get("name")
    ad = rec.get("adapters") if isinstance(rec.get("adapters"), dict) else {}
    im = rec.get("ingest_manifest") if isinstance(rec.get("ingest_manifest"), dict) else {}
    if not isinstance(run_dir, str) or ad.get("run_dir") != run_dir or im.get("run_dir") != run_dir:
        refuse(f"{label}: config.name, adapters.run_dir and ingest_manifest.run_dir disagree ({run_dir!r}, "
               f"{ad.get('run_dir')!r}, {im.get('run_dir')!r})")
    split_run_dir(run_dir, corpus)
    if step not in STEPS or step not in (cfg.get("save_steps") or []):
        refuse(f"{label}: step {step} is not one of the saved checkpoints {cfg.get('save_steps')} (D-343 (2): {STEPS})")
    base_repo, base_rev = cfg.get("repo"), cfg.get("rev")
    if not isinstance(base_repo, str) or not re.fullmatch(r"[0-9a-f]{40}", str(base_rev or "")):
        refuse(f"{label}: the record names no base repo and 40-hex revision")
    ents = [c for c in rec.get("checkpoints") or [] if isinstance(c, dict) and c.get("step") == step]
    if len(ents) != 1:
        refuse(f"{label}: {len(ents)} 'checkpoints' entries for step {step} (want exactly one)")
    c = ents[0]
    d = f"step_{step}"
    files = c.get("files") if isinstance(c.get("files"), dict) else {}
    sha, csha = c.get("adapter_sha256"), c.get("adapter_config_sha256")
    if c.get("dir") != d or not HEX64.fullmatch(str(sha or "")) or not HEX64.fullmatch(str(csha or "")):
        refuse(f"{label}: the step-{step} checkpoint names no dir {d} and 64-hex adapter sha256s")
    if (files.get(ADAPTER_FILES[0]) or {}).get("sha256") != sha or (files.get(ADAPTER_FILES[1]) or {}).get("sha256") != csha:
        refuse(f"{label}: the step-{step} checkpoint's adapter_sha256 / adapter_config_sha256 are not its files' sha256s")
    listed = {}
    for f in im.get("files") or []:
        if isinstance(f, dict) and isinstance(f.get("path"), str):
            if f["path"] in listed:
                refuse(f"{label}: ingest_manifest lists {f['path']} twice")
            listed[f["path"]] = f.get("sha256")
    for name, want in zip(ADAPTER_FILES, (sha, csha)):
        if listed.get(f"{d}/{name}") != want:
            refuse(f"{label}: ingest_manifest lists {d}/{name} with sha256 {str(listed.get(f'{d}/{name}'))[:16]}, not the "
                   f"checkpoint's {want[:16]}")
    return {"run_dir": run_dir, "dir": d, "sha256": sha, "config_sha256": csha, "base_repo": base_repo,
            "base_rev": base_rev, "ingest_files": listed}


def read_index(path: Path, run_dir: str, want: dict[str, Any], label: str) -> dict[str, str]:
    """The adapter-ingest index: exactly the ingest_manifest block's files, '<sha256>  data/lora/<run dir>/<path>'."""
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        refuse(f"no adapter-ingest index {p.name} (or not a regular file): scripts/p2_ingest_lora.sh writes it when "
               f"it ingests {label} (D-307 (2), D-350)")
    got: dict[str, str] = {}
    prefix = f"data/lora/{run_dir}/"
    for line in p.read_text(encoding="utf-8").split("\n"):
        if not line:
            continue
        h, sep, f = line.partition("  ")
        if not sep or not HEX64.fullmatch(h) or not f.startswith(prefix) or f[len(prefix):] in got:
            refuse(f"{p.name}: malformed or repeated line {line!r}")
        got[f[len(prefix):]] = h
    if got != want:
        refuse(f"{p.name} does not list exactly the record's ingest_manifest files with their sha256s "
               f"(differ: {sorted(set(got.items()) ^ set(want.items()))[:4]})")
    return got


def check_files(adir: Path, label: str) -> None:
    for name in ADAPTER_FILES:
        f = adir / name
        if f.is_symlink() or not f.is_file():
            refuse(f"{label}: {f} is not a regular file in data/lora (the ingested copy, D-307 (2))")
        mode = stat.S_IMODE(os.stat(f).st_mode)
        if mode != 0o444:
            refuse(f"{label}: {f} is mode {oct(mode)}, not 0o444 (an ingested adapter is read-only, D-307 (2))")


def read_config(adir: Path, want_sha: str, label: str) -> dict[str, Any]:
    """An ingested adapter_config.json, its bytes' sha256 the recorded one, parsed; target_modules (a list in peft's
    file) as a set, so two saves of one adapter compare equal whatever order peft wrote the set in."""
    f = Path(adir) / ADAPTER_FILES[1]
    if f.is_symlink() or not f.is_file():
        refuse(f"{label}: {f} is not a regular file in data/lora (the ingested copy, D-307 (2))")
    data = f.read_bytes()
    if PV.sha256_bytes(data) != want_sha:
        refuse(f"{label}: {f} is not the bytes its record names (adapter_config_sha256 {want_sha[:16]})")
    try:
        cfg = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        refuse(f"{label}: {f} is not JSON ({type(e).__name__})")
    if not isinstance(cfg, dict):
        refuse(f"{label}: {f} is not a JSON object")
    if isinstance(cfg.get("target_modules"), list):
        cfg = {**cfg, "target_modules": set(cfg["target_modules"])}
    return cfg


def resolve(record: str | Path, step: int, root: Path = ROOT) -> dict[str, Any]:
    """What extract_p2 needs for one checkpoint (module docstring): {dir (repo-relative), name, sha256, config_sha256,
    base_repo, base_rev, source (the record's adapter_source block)}."""
    root = Path(root)
    path = Path(record) if Path(record).is_absolute() else root / record
    p2 = (root / "results" / "phase2").resolve()
    if path.resolve().parent != p2:
        refuse(f"{record} is not a record in results/phase2/")
    corpus, seed = parse_name(path)
    if not path.is_file() or path.is_symlink():
        refuse(f"no training record {record} (or it is a symlink)")
    if step == 0 and corpus != CORPUS_ORDER[0]:
        refuse(f"step 0 is extracted once per seed, through the {CORPUS_ORDER[0]} record ({STEP0_RULE}); "
               f"{path.name} is not that record")

    def one(p: Path, c: str) -> tuple[dict[str, Any], dict[str, Any]]:
        rel = PV._repo_rel(p)
        rec = PV.read_result(p)
        sha_rec = PV._RESULT_READS[rel]
        ck = checkpoint(rec, step, c, seed, rel)
        idx = root / "manifests" / "ingest_lora" / f"{p.stem}.sha256"
        read_index(idx, ck["run_dir"], ck["ingest_files"], rel)
        return ck, {"record": {"file": rel, "sha256": sha_rec}, "corpus": c, "seed": seed,
                    "run_dir": ck["run_dir"],
                    "index": {"file": f"manifests/ingest_lora/{idx.name}", "sha256": PV.sha256_file(idx)}}

    ck, src = one(path, corpus)
    adir = root / "data" / "lora" / ck["run_dir"] / ck["dir"]
    check_files(adir, PV._repo_rel(path))
    name = identity(ck["run_dir"], corpus, step)
    block: dict[str, Any] = {
        "rule": ("D-307 (2), D-343 (2), D-350: the adapter's directory, identity and recorded sha256s are read from the "
                 "training record's 'checkpoints' and 'ingest_manifest' blocks and the adapter-ingest index, which agree; "
                 "never typed by hand (src/lora_adapter_p2.py)"),
        **src, "step": step, "adapter_dir": f"data/lora/{ck['run_dir']}/{ck['dir']}", "identity": f"lora/{name}",
        "adapter_sha256": ck["sha256"], "adapter_config_sha256": ck["config_sha256"],
        "base": {"repo": ck["base_repo"], "rev": ck["base_rev"]}}
    if step == 0:
        head_tail = split_run_dir(ck["run_dir"], corpus)
        cfg0 = read_config(adir, ck["config_sha256"], PV._repo_rel(path))
        sibs = []
        for c in CORPUS_ORDER[1:]:
            sp = record_path(c, seed, root)
            if not sp.is_file() or sp.is_symlink():
                refuse(f"step 0 of seed {seed}: the sibling record {sp.name} is not on disk ({STEP0_RULE})")
            sck, ssrc = one(sp, c)
            if sck["sha256"] != ck["sha256"]:
                refuse(f"step 0 of seed {seed}: {sp.name} names step-0 adapter weights {sck['sha256'][:16]}, not "
                       f"{ck['sha256'][:16]} ({STEP0_RULE})")
            if split_run_dir(sck["run_dir"], c) != head_tail or (sck["base_repo"], sck["base_rev"]) != \
                    (ck["base_repo"], ck["base_rev"]):
                refuse(f"step 0 of seed {seed}: {sp.name}'s run dir or base differs from {path.name}'s")
            sdir = root / "data" / "lora" / sck["run_dir"] / sck["dir"]
            check_files(sdir, ssrc["record"]["file"])
            if read_config(sdir, sck["config_sha256"], ssrc["record"]["file"]) != cfg0:
                refuse(f"step 0 of seed {seed}: {sp.name}'s step-0 adapter_config.json differs from {path.name}'s as "
                       f"parsed JSON, beyond the order of target_modules ({STEP0_RULE})")
            sibs.append({**ssrc, "step0_adapter_sha256": sck["sha256"], "step0_adapter_config_sha256":
                         sck["config_sha256"], "weights_equal": True,
                         "config_equal_as_parsed_json_target_modules_as_set": True,
                         "config_bytes_equal": sck["config_sha256"] == ck["config_sha256"]})
        block["step0"] = {"rule": STEP0_RULE, "extracted_through": path.name, "siblings": sibs}
    return {"dir": block["adapter_dir"], "name": name, "sha256": ck["sha256"], "config_sha256": ck["config_sha256"],
            "base_repo": ck["base_repo"], "base_rev": ck["base_rev"], "source": block}
