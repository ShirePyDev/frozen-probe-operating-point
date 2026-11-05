"""Job lists for the Phase 2 LoRA extraction stage and the SST-2 control (D-343, D-338 (4), D-340 (3)), written from the
committed records so that no adapter name, sha256 or record name is typed by hand. NOTHING HERE RUNS ANYTHING: it writes
scripts/queue_runner.sh lists (label TAB outputs TAB command), each headed by what it is for and when it may run.

    python src/lora_jobs_p2.py --write     (writes the lists below; an existing list must be byte-identical)
    python src/lora_jobs_p2.py --check     (exit 1 unless every committed list equals what the records give)
    python src/lora_jobs_p2.py --price --headroom-bytes <N>   (READ-ONLY, writes nothing: for each storage option, the
                                           NAS bytes it would ingest, fit or not against N and the margin left; N is the
                                           headroom above the 20 GB floor the user reads from scripts/survey_export_root.sh,
                                           never measured here)

PRICING (for the author's D-343 (1) choice; D-355 (2), D-357 (B)(5)): what each option list ingests into data/hidden, per
checkpoint (the nine training records: 27 at steps 25-400 and 3 step-0, one per seed, so 30 cut) and per cache, every
size read from records on disk (Sizes; rule 2: no size is typed), each figure marked with its basis (BASIS): M, measured
(Mistral-7B-Instruct-v0.3's ingested p2sub cache of the same split, pooling and layer indices: the per-adapter template
D-343 (1) priced from, the base's L* being v0.3's); L, extrapolated by layer count (L* alone: that cache's bytes / its 7
layers; no one-layer cache exists); S, extrapolated by item count (SST-2 is not yet extracted: measured bytes per item
and layer times dataset_manifest_p2sst.json's items and the layers; adapters at v0.3's p2sub rate, the base's SST-2
select set at the full grid at its own ingested p2tpl select rate, its rest subset at its own p2sub rate; an SST-2 L* not
yet selected counted outside GRID6, the upper bound); F, extrapolated float16 user-span mean for the adapters' subsets
(D-357 (B)(5), after the one-cell check; no float16 cache exists: the raw size, items x layers x hidden x 2; last-token
states are float16 values already and stay as measured; the base's caches are unchanged). Each cache is rounded up to the
NAS allocation unit as scripts/p2_ingest_record.sh rounds it (its minimum unit, or --alloc-unit). The adapters' weights
are already in data/lora (manifests/ingest_lora/), inside the allocated bytes the headroom is read from: not added. The
table: each list's parts (steps 25-400, step 0, the base's SST-2 select and rest ingests; float32 and float16), then
each option alone and with its SST-2 list(s) (PRICE_ROWS), with and without the step-0 subsets, float32 and float16:
the NAS bytes, the margin (headroom - bytes) and fits / DOES NOT FIT.

    python3 -u src/lora_jobs_p2.py --preflight <list name>   (on the node, the FIRST JOB of every option list: below)

D-343 (1) IS OPEN: no LoRA extraction starts until the author chooses the storage option. Every option list is a whole
lane B sitting; the author launches the one WildGuardMix list and (or) the one SST-2 list of the option chosen.

THE LISTS (dates and letters follow the lists already in queues/; fed8bd1's lists are withdrawn, never run):
  laneC_20261003d_p2sst_scans.tsv   CPU, before any p2sst extraction (D-340 (3)): the tokenizer prompts check and the
                                    BOS check's part B over the SST-2 prompts.
  laneB_20261004a_lora_wg_opt<X>.tsv   the WildGuardMix + XSTest sitting under one option (GPU, lane B): the preflight;
                                    the base's weights fetched once from the campus mirror, every file sha256-checked
                                    against huggingface.co (src/fetch_weights.py --mirror, the rule of
                                    scripts/p2_sitting.sh); then SEED BY SEED (provenance review of fed8bd1, SF3): the
                                    seed's step-0 adapter (once per seed, lora_adapter_p2's STEP0_RULE) extracted, the
                                    D-343 (2) step-0 check against the base (src/lora_step0_check_p2.py; a failure stops
                                    the lane before any subset of the seed), the seed's other nine checkpoints extracted,
                                    then every checkpoint of the seed cut to the option's subset and ingested; the purge.
                                    Each extraction is one record (select: p2wg/probe_train + calibration; rest: eval,
                                    pool4d, XSTest) at the full grid, ingest 'never', the adapter, identity and sha256s
                                    from the training record (--lora-record). X: optA_keeptrain (GRID6 + L*, both
                                    poolings, post probe_train kept: option A as D-343 (1) priced it), optA_droptrain (the
                                    same without post probe_train; not defined in D-343 (1), offered for the author), optB
                                    (GRID6 + L*, user-span mean, no post probe_train), optC (L* alone, both poolings, no
                                    post probe_train), optD (seed 0 as optA_droptrain, seeds 1 and 2 as optC).
  laneB_20261004b_lora_sst_opt<Y>.tsv  the SST-2 sitting under one option (GPU, lane B): the preflight (with the SST-2
                                    scans' pass, D-340 (3)); the weights; the base's SST-2 select (probe_train,
                                    calibration; ingest yes) and rest (eval; ingest never) sets; the select set ingested
                                    and the D-117 selection on SST-2's probe_train for the base, both poolings
                                    (src/selection_p2sst.py, selection_p2's unchanged, anchored sweep: the existing anchor
                                    serves); the base's rest set cut (GRID6 + its SST-2 L*, both poolings) and ingested;
                                    then seed by seed as above, each checkpoint's SST-2 calibration (select) and eval
                                    (rest), the step-0 check on calibration against the base's ingested select caches;
                                    the purge. Y: optA (GRID6 + L*, both), optB (GRID6 + L*, user-span mean),
                                    optC_usermean (L* alone, user-span mean: D-343 (1)'s C), optC_both (L* alone, both;
                                    not defined in D-343 (1), offered for the author). D-343 (1) defines no SST-2 storage
                                    under option D: no list (a question for the author).

THE PREFLIGHT (--preflight, the first job of every option list; provenance review of fed8bd1, blocker 2): refuses before
anything is fetched or cut unless (1) queues/<list> is what the records give (so the price is this list's) and its labels
pass check_labels; (2) no other job of the list has a .done or .exit in logs/queue/<lane>/ (a label of this list already
run there is a collision or a re-run whose preflight has no .done; the preflight's own .exit, a failed earlier preflight,
is allowed); (3) for an SST-2 list, extract_p2.sst_scans passes for the base (D-340 (3)); (4) the
account's allocated bytes (scripts/survey_export_root.sh's ACCOUNT SUM, under logs/nas_write.lock, as
scripts/p2_ingest_record.sh reads them) + the whole list's expected NAS bytes (walk(), as --price computes them, each
cache rounded up to the NAS allocation unit as that script rounds it) + FLOOR do not exceed QUOTA_BOUND (both read from
scripts/p2_ingest_record.sh). Every ingest still checks its own record's headroom. Once it has passed, a relaunch of the
list skips it (.done), so the bytes already ingested are never counted twice.

LABELS: every label of every list is unique against every other list of its lane in queues/ (check_labels refuses
otherwise; --check --done-labels <file> also against a listing of the node's logs/queue/<lane>/*.done): a repeated label
lets scripts/queue_runner.sh take another list's .done for a job (provenance review of fed8bd1, blocker 1).

CHOICES RECORDED HERE (the closest D-entry; D-181): the post probe_train of WildGuardMix is extracted for every adapter
although only optA_keeptrain keeps it (NVMe only); the SST-2 probe_train of an adapter is not extracted (every D-343 (1)
option prices SST-2 without it; a post side trains no probe). Every list keeps the step-0 subsets as fed8bd1 did; whether
they are kept durably is the author's choice (the price shows each total with and without them). The step-0 check runs
for every seed (D-343 (2): 'step-0 adapters checked'), on the select set (the base's set held at the full grid). Adapter
caches go to $HOME/.cache/gccf_scratch/p2lora, the base's SST-2 caches to .../p2cache, subsets to .../p2sub.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lora_adapter_p2 as LA  # noqa: E402
import provenance as PV  # noqa: E402
import subset_p2 as S  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QUEUES = ROOT / "queues"
BASE = "mistralai/Mistral-7B-Instruct-v0.2"
SEEDS = (0, 1, 2)                                            # D-303 (7)
P2CACHE, P2LORA, P2SUB = ("$HOME/.cache/gccf_scratch/p2cache", "$HOME/.cache/gccf_scratch/p2lora",
                          "$HOME/.cache/gccf_scratch/p2sub")
SETS = {  # (tag, set name) -> splits; D-325's select/rest split, SST-2 without an adapter's probe_train
    ("p2tpl", "select"): ("p2wg/probe_train", "p2wg/calibration"),
    ("p2tpl", "rest"): ("p2wg/eval", "p2wg/pool4d", "over_refusal/eval"),
    ("p2sst", "select"): ("p2sst/calibration",),
    ("p2sst", "rest"): ("p2sst/eval",),
}
BASE_SST_SETS = {"select": (("p2sst/probe_train", "p2sst/calibration"), "yes"), "rest": (("p2sst/eval",), "never")}
SUB_TAG = {"p2tpl": "p2sub", "p2sst": "p2sstsub"}            # subset_p2.SUB_TAGS
BOTH = ("usermean", "lastgen")
# option -> per seed (layer set, poolings, post probe_train for a set holding it)
WG_OPTIONS = {
    "optA_keeptrain": ("A", lambda s: ("grid6_lstar", BOTH, "keep")),
    "optA_droptrain": ("A", lambda s: ("grid6_lstar", BOTH, "drop")),
    "optB": ("B", lambda s: ("grid6_lstar", ("usermean",), "drop")),
    "optC": ("C", lambda s: ("lstar", BOTH, "drop")),
    "optD": ("D", lambda s: ("grid6_lstar", BOTH, "drop") if s == S.OPTION_D_SEVEN_LAYER_SEED else
             ("lstar", BOTH, "drop")),
}
SST_OPTIONS = {
    "optA": ("A", ("grid6_lstar", BOTH)),
    "optB": ("B", ("grid6_lstar", ("usermean",))),
    "optC_usermean": ("C", ("lstar", ("usermean",))),
    "optC_both": ("C", ("lstar", BOTH)),
}
NOT_IN_D343 = {"wg_optA_droptrain", "sst_optC_both"}         # fidelity review of fed8bd1: the author's to define
NAMES = {"scans": "laneC_20261003d_p2sst_scans.tsv",
         **{f"wg_{k}": f"laneB_20261004a_lora_wg_{k}.tsv" for k in WG_OPTIONS},
         **{f"sst_{k}": f"laneB_20261004b_lora_sst_{k}.tsv" for k in SST_OPTIONS}}
LABEL_PREFIX = {"wg_optA_keeptrain": "wgAk", "wg_optA_droptrain": "wgAd", "wg_optB": "wgB", "wg_optC": "wgC",
                "wg_optD": "wgD", "sst_optA": "ssA", "sst_optB": "ssB", "sst_optC_usermean": "ssCu",
                "sst_optC_both": "ssCb"}


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def slug(repo: str) -> str:
    return repo.replace("/", "__")


def adapters(root: Path = ROOT) -> tuple[list[dict[str, Any]], str, int]:
    """Every checkpoint to extract, in the sitting's order (seed; step 0 once, through the first corpus; then steps 25,
    100, 400, each corpus), from the nine training records; the base revision (every record's, and the config check's)
    and its layer count (the config check's)."""
    cc = json.loads((root / "results" / "phase2_config_check.json").read_text(encoding="utf-8"))
    m = (cc.get("models") or {}).get(BASE) or {}
    rev, nl = m.get("sha"), ((m.get("config") or {}).get("num_hidden_layers"))
    if not isinstance(rev, str) or not isinstance(nl, int):
        refuse(f"the config check names no revision and layer count for {BASE}")
    out = []
    for seed in SEEDS:
        recs = {}
        for c in LA.CORPUS_ORDER:
            p = LA.record_path(c, seed, root)
            if not p.is_file():
                refuse(f"no training record {p.name}")
            recs[c] = json.loads(p.read_text(encoding="utf-8"))
        for step in LA.STEPS:
            for c in (LA.CORPUS_ORDER[:1] if step == 0 else LA.CORPUS_ORDER):
                ck = LA.checkpoint(recs[c], step, c, seed, f"lora_train_p2__{c}__s{seed}.json")
                if (ck["base_repo"], ck["base_rev"]) != (BASE, rev):
                    refuse(f"lora_train_p2__{c}__s{seed}.json trained {ck['base_repo']} @ {ck['base_rev'][:12]}, not "
                           f"{BASE} @ {rev[:12]}")
                if step == 0:
                    for o in LA.CORPUS_ORDER[1:]:
                        if LA.checkpoint(recs[o], 0, o, seed, o)["sha256"] != ck["sha256"]:
                            refuse(f"seed {seed}: step 0 of {o} is not step 0 of {c} ({LA.STEP0_RULE})")
                out.append({"seed": seed, "corpus": c, "step": step, "sha256": ck["sha256"],
                            "record": f"results/phase2/lora_train_p2__{c}__s{seed}.json",
                            "name": LA.identity(ck["run_dir"], c, step)})
    return out, rev, nl


RECOVERY = ("RECOVERY (D-355 (3); F-071: NVMe does not survive a restart): a seed whose every ingest has a .done is "
            "complete and stays as it is. For the seed in progress, each record of this list that names a cache no "
            "longer on NVMe and not ingested is renamed <name>.lost.<UTC>, and so are the .done and .exit of the job "
            "that wrote it, the seed's step-0 check record with its job's .done and .exit, and the weights fetch's "
            ".done and .exit; nothing is deleted. The list is then relaunched, and those jobs run again from scratch; "
            "the preflight's .done is kept, so the bytes already ingested are not counted twice.")


def lane_of(name: str) -> str:
    m = re.fullmatch(r"lane([A-Za-z0-9]+)_[A-Za-z0-9_.-]+\.tsv", name)
    if not m:
        refuse(f"{name} is not a queues/lane<L>_<...>.tsv list name")
    return m.group(1)


def jobs(text: str) -> list[tuple[str, str, str]]:
    """A list's job lines as (label, outputs, command); comment and blank lines skipped."""
    out = []
    for ln in text.split("\n"):
        if not ln.strip() or ln.startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) != 3:
            refuse(f"a job line without three TAB fields: {ln[:80]!r}")
        out.append((parts[0], parts[1], parts[2]))
    return out


def read_done_labels(path: Path) -> dict[str, set[str]]:
    """A listing of the node's .done files (`ls logs/queue/*/*.done`, one path per line) as {lane: labels}; a line of
    any other form refuses, so a wrong listing is never read as an empty one."""
    out: dict[str, set[str]] = {}
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        m = re.fullmatch(r"(?:.*/)?queue/([A-Za-z0-9_-]+)/([A-Za-z0-9_.-]+)\.done", ln.strip())
        if not m:
            refuse(f"{path}: {ln.strip()[:80]!r} is not a logs/queue/<lane>/<label>.done path")
        out.setdefault(m.group(1), set()).add(m.group(2))
    if not out:
        refuse(f"{path} lists no .done file")
    return out


def check_labels(texts: dict[str, str], root: Path = ROOT, done: dict[str, set[str]] | None = None) -> None:
    """LABELS (module docstring; provenance review of fed8bd1, blocker 1): every label of every list here is unique in
    its list, against every other list of its lane here and in queues/ (a committed list of the same name stands for
    itself: --check holds it equal), and, given `done`, against the node's .done labels of its lane."""
    seen: dict[tuple[str, str], str] = {}
    for p in sorted((Path(root) / "queues").glob("lane*_*.tsv")):
        if p.name in texts:
            continue
        for ln in p.read_text(encoding="utf-8").split("\n"):
            if ln.strip() and not ln.startswith("#"):
                seen.setdefault((lane_of(p.name), ln.split("\t")[0]), p.name)
    for name, text in texts.items():
        lane = lane_of(name)
        for label, _, _ in jobs(text):
            if (lane, label) in seen:
                refuse(f"label {label} of {name} is also a label of {seen[(lane, label)]} in lane {lane}")
            if done is not None and label in done.get(lane, set()):
                refuse(f"label {label} of {name} has a .done in the node's logs/queue/{lane}/ (another list's job)")
            seen[(lane, label)] = name


def extract_record(ident: str, rev: str, tag: str, set_name: str) -> str:
    return f"results/phase2/extract__{slug(ident)}__{tag}@{rev[:12]}__{set_name}.json"


def line(label: str, outputs: str, cmd: str) -> str:
    return f"{label}\t{outputs}\t{cmd}"


def header(text: str) -> str:
    return "# " + " ".join(text.split())


def lists(root: Path = ROOT) -> dict[str, str]:
    """{list file name: its text}."""
    ads, rev, nl = adapters(root)
    layers = " ".join(str(i) for i in range(0, nl + 1, 2))
    base_x = (f"HF_HUB_OFFLINE=1 python3 -u src/extract_p2.py --repo {BASE} --revision {rev} --kind instruct "
              f"--layers {layers}")
    out: dict[str, str] = {}

    # 1. the SST-2 scans, before any p2sst extraction
    scans = [header(
        "Lane C (CPU; D-338 (4), D-340 (3)): before any p2sst extraction, the SST-2 non-safety control's prompts through "
        "every Phase 2 tokenizer (src/phase2_tokenizer_check.py prompts --axis p2sst: dataset_manifest_p2sst.json's "
        "files only, no XSTest, keyed by (file, item_id)) and the special-token scan over them (src/phase2_bos_check.py "
        "--axis p2sst: part B only, every Phase 2 and Phase 1-only tokenizer). Tokenizer files are fetched by commit "
        "from huggingface.co (the token only for a gated repo). Each record is written once; the BOS scan exits 3 if a "
        "tokenizer is not loaded (the record is still written, for a human). Generated by src/lora_jobs_p2.py. NOT RUN "
        "until reviewed."),
        line("p2sst_tokprompts", "results/phase2/tokenizer_prompts_p2sst.json "
             "results/phase2/tokenizer_prompts_p2sst_ids.npz",
             "python3 -u src/phase2_tokenizer_check.py prompts --axis p2sst"),
        line("p2sst_bos", "results/phase2/bos_check_p2sst.json", "python3 -u src/phase2_bos_check.py --axis p2sst")]
    out[NAMES["scans"]] = "\n".join(scans) + "\n"

    # 2. one lane B sitting per option list: preflight, weights, (SST-2: the base's sets and selection), seed by seed
    fetch = (f". scripts/site_env.sh && site_require GCCF_HF_MIRROR && python3 -u src/fetch_weights.py --repo {BASE} "
             f"--revision {rev} --mirror \"$GCCF_HF_MIRROR\"")
    purge = f"python3 -u src/fetch_weights.py --repo {BASE} --revision {rev} --purge"
    for key in [f"wg_{k}" for k in WG_OPTIONS] + [f"sst_{k}" for k in SST_OPTIONS]:
        name, pre, wg = NAMES[key], LABEL_PREFIX[key], key.startswith("wg_")
        tag = "p2tpl" if wg else "p2sst"
        letter = (WG_OPTIONS[key[3:]] if wg else SST_OPTIONS[key[4:]])[0]
        after_scans = "" if wg else f", after queues/{NAMES['scans']} is done (D-340 (3))"
        what = (f"WildGuardMix + XSTest (select: {' + '.join(SETS[(tag, 'select')])}; rest: "
                f"{', '.join(SETS[(tag, 'rest')])})" if wg else
                f"SST-2 (select: {' + '.join(SETS[(tag, 'select')])}; rest: {', '.join(SETS[(tag, 'rest')])}; an adapter "
                f"trains no SST-2 probe)")
        rows = [header(
            f"Lane B (GPU; D-343 (1) OPTION {letter} as this file's name says: {key}"
            f"{' (not defined in D-343 (1): offered for the author, D-355 (4))' if key in NOT_IN_D343 else ''}; D-343 "
            f"(2), D-325, D-307, D-345, D-350, D-355 (3){'; D-338 (4), D-340 (3)' if not wg else ''}). ONE LORA SITTING, "
            f"{what}. NOT TO BE LAUNCHED BEFORE THE AUTHOR CHOOSES D-343 (1)'s STORAGE OPTION, and then only this "
            f"option's list(s){after_scans}. "
            f"The preflight first (src/lora_jobs_p2.py --preflight: this list is what the records give, its labels are "
            f"unique in lane B and none has run, {'the SST-2 scans pass for the base, ' if not wg else ''}and the "
            f"account's allocated bytes + this whole list's NAS bytes + the 20 GB floor fit under QUOTA_BOUND); the base "
            f"{BASE} @ {rev[:12]} fetched from the campus mirror, every file sha256-checked against huggingface.co "
            f"(src/fetch_weights.py, the rule of scripts/p2_sitting.sh)"
            + ("" if wg else
               f"; the base's SST-2 select set (probe_train + calibration, ingest yes) and rest set (eval, ingest never) "
               f"at the full grid to {P2CACHE}, the select set ingested, the D-117 selection on SST-2's probe_train for "
               f"both poolings (src/selection_p2sst.py: selection_p2's anchored sweep unchanged, the existing anchor "
               f"serves), the base's rest set cut to GRID6 + its SST-2 L*, both poolings, and ingested")
            + f"; then SEED BY SEED (D-355 (3)): the seed's step-0 checkpoint (once per seed: {LA.STEP0_RULE}) "
            f"extracted, the D-343 (2) step-0 check against the base's ingested select caches (src/lora_step0_check_p2.py"
            f", Phase 1's gate within compare_caches' 1e-4 on every layer, the plain-path items left out and listed; a "
            f"failure stops the lane before any subset of the seed), the seed's nine other checkpoints extracted, then "
            f"each of the seed's ten checkpoints cut to the option's layers around the BASE's D-117 L* "
            f"(src/subset_p2.py, the pair base -> base+adapter) and ingested into data/hidden at 444 "
            f"(scripts/p2_ingest_record.sh: the NAS-write lock and its own headroom check); finally the purge. Every "
            f"extraction is one record at the full {nl}-layer grid, ingest never, to {P2LORA}; the adapter, its "
            f"identity lora/<name> and sha256s read by extract_p2 --lora-record from the training record and "
            f"manifests/ingest_lora/, never typed. {RECOVERY} Generated by src/lora_jobs_p2.py."),
            line(f"{pre}_preflight", "-", f"python3 -u src/lora_jobs_p2.py --preflight {name}"),
            line(f"{pre}_weights", "-", fetch)]
        if not wg:
            for set_name, (splits, ingest) in BASE_SST_SETS.items():
                rows.append(line(f"{pre}_base_{set_name}", extract_record(BASE, rev, "p2sst", set_name),
                                 f"{base_x} --splits {' '.join(splits)} --set-name {set_name} --ingest {ingest} "
                                 f"--device cuda --out-dir \"{P2CACHE}\""))
            rows.append(line(f"{pre}_ing_base_select", "-", f"bash scripts/p2_ingest_record.sh "
                                                             f"{extract_record(BASE, rev, 'p2sst', 'select')} \"{P2CACHE}\""))
            for pooling in BOTH:
                rows.append(line(f"{pre}_sel_{pooling}",
                                 f"results/phase2/selection_p2sst__{slug(BASE)}@{rev[:12]}__{pooling}.json",
                                 f"python3 -u src/selection_p2sst.py --repo {BASE} --pooling {pooling}"))
            base_sub = extract_record(BASE, rev, "p2sstsub", "rest")
            rows.append(line(f"{pre}_sub_base_rest", base_sub,
                             f"python3 -u src/subset_p2.py --repo {BASE} --source-tag p2sst --set rest --layer-set "
                             f"grid6_lstar --poolings {' '.join(BOTH)} --src-dir \"{P2CACHE}\" --out-dir \"{P2SUB}\""))
            rows.append(line(f"{pre}_ing_base_rest", "-", f"bash scripts/p2_ingest_record.sh {base_sub} \"{P2SUB}\""))
        for seed in SEEDS:
            sa = [a for a in ads if a["seed"] == seed]
            if not sa or sa[0]["step"] != 0 or any(a["step"] == 0 for a in sa[1:]):
                refuse(f"seed {seed}: the checkpoints do not begin with one step-0 checkpoint")

            def lab(a: dict[str, Any], kind: str, set_name: str) -> str:
                who = "all" if a["step"] == 0 else a["corpus"]
                return f"{pre}_{kind}_s{a['seed']}_{who}_{a['step']}_{set_name[:3]}"

            def extract(a: dict[str, Any]) -> list[str]:
                return [line(lab(a, "x", set_name), extract_record(f"lora/{a['name']}", a["sha256"], tag, set_name),
                             f"{base_x} --lora-record {a['record']} --lora-step {a['step']} --splits "
                             f"{' '.join(SETS[(tag, set_name)])} --set-name {set_name} --ingest never --device cuda "
                             f"--out-dir \"{P2LORA}\"") for set_name in ("select", "rest")]
            rows += extract(sa[0])
            s0rec = extract_record(f"lora/{sa[0]['name']}", sa[0]["sha256"], tag, "select")
            rows.append(line(f"{pre}_step0chk_s{seed}",
                             f"results/phase2/step0check_p2__lora__{sa[0]['name']}__{tag}@{sa[0]['sha256'][:12]}.json",
                             f"python3 -u src/lora_step0_check_p2.py --adapter-record {s0rec} --src-dir \"{P2LORA}\""))
            for a in sa[1:]:
                rows += extract(a)
            for a in sa:
                if wg:
                    layer_set, pools, ptrain = WG_OPTIONS[key[3:]][1](a["seed"])
                else:
                    (layer_set, pools), ptrain = SST_OPTIONS[key[4:]][1], None
                for set_name in ("select", "rest"):
                    rec = extract_record(f"lora/{a['name']}", a["sha256"], S.SUB_TAGS[tag], set_name)
                    flags = ("" if wg else "--source-tag p2sst ") + f"--set {set_name} --layer-set {layer_set} " \
                        f"--poolings {' '.join(pools)}" + (f" --post-probe-train {ptrain}"
                                                           if wg and set_name == "select" else "") \
                        + f" --storage-option {letter}"
                    rows.append(line(lab(a, "sub", set_name), rec,
                                     f"python3 -u src/subset_p2.py --adapter-name {a['name']} {flags} --src-dir "
                                     f"\"{P2LORA}\" --out-dir \"{P2SUB}\""))
                    rows.append(line(lab(a, "ing", set_name), "-",
                                     f"bash scripts/p2_ingest_record.sh {rec} \"{P2SUB}\""))
        rows.append(line(f"{pre}_purge", "-", purge))
        out[name] = "\n".join(rows) + "\n"
    check_labels(out, root)
    return out


# --- pricing D-343 (1)'s options (module docstring, PRICING; read-only) ---------------------------------------------

MEASURED = "mistralai/Mistral-7B-Instruct-v0.3"   # D-343 (1): the per-adapter subsets were MEASURED from its caches
WG_SPLITS = SETS[("p2tpl", "select")] + SETS[("p2tpl", "rest")]
SST_SPLITS = SETS[("p2sst", "select")] + SETS[("p2sst", "rest")]
DTYPES = ("f32", "f16")
PARTS = (("s25_400", "steps 25-400"), ("step0", "step 0"), ("base_sel", "base SST-2 select"),
         ("base_rest", "base SST-2 rest"))
# the fit table's rows: each WildGuardMix + XSTest option list alone, then with the SST-2 list(s) of its option
PRICE_ROWS = (("wg_optA_keeptrain", None), ("wg_optA_keeptrain", "sst_optA"), ("wg_optA_droptrain", None),
              ("wg_optA_droptrain", "sst_optA"), ("wg_optB", None), ("wg_optB", "sst_optB"), ("wg_optC", None),
              ("wg_optC", "sst_optC_usermean"), ("wg_optC", "sst_optC_both"), ("wg_optD", None))
BASIS = {"M": "measured: Mistral-7B-Instruct-v0.3's ingested p2sub cache of the same split, pooling and layer indices "
              "(the record's cache_bytes, equal to the file on the NAS)",
         "L": "EXTRAPOLATED by layer count: L* alone = that 7-layer cache's bytes / 7 (no one-layer cache exists; "
              "layers compress differently)",
         "S": "EXTRAPOLATED by item count (SST-2 not yet extracted): measured bytes per item and layer x SST-2's items "
              "x layers (adapters: v0.3's p2sub rate; the base: its own ingested p2tpl select rate at the full grid, "
              "its own p2sub rest rate for its subset)",
         "F": "EXTRAPOLATED float16 user-span mean (adapters' subsets only; no float16 cache exists): items x layers x "
              "hidden x 2, the raw size, compression not counted"}


class Sizes:
    """Cache sizes for the price, every one from records on disk (module docstring, PRICING); `sources` names each
    file read with its sha256 and what it gives."""

    def __init__(self, root: Path = ROOT, alloc_unit: int | None = None):
        self.root = Path(root)
        self.sources: list[tuple[str, str, str]] = []
        cc = self._json("results/phase2_config_check.json", "revisions, layer count, hidden size")
        models = cc.get("models") or {}
        cfg = (models.get(BASE) or {}).get("config") or {}
        self.rev, self.measured_rev = (models.get(BASE) or {}).get("sha"), (models.get(MEASURED) or {}).get("sha")
        self.nl, self.hidden = cfg.get("num_hidden_layers"), cfg.get("hidden_size")
        if not (isinstance(self.rev, str) and isinstance(self.measured_rev, str) and isinstance(self.nl, int)
                and isinstance(self.hidden, int)):
            refuse(f"the config check names no revision, layer count and hidden size for {BASE} and {MEASURED}")
        self.grid, self.grid6 = list(range(0, self.nl + 1, 2)), S.grid6(self.nl)
        p2 = "results/phase2"
        # the base's own records: items per split (both sets); bytes per item and layer of its ingested (compressed)
        # full-grid select caches and of its ingested rest subset (SST-2's base lines)
        self.items: dict[str, int] = {}
        self.full_rate = self._rate(f"{p2}/extract__{slug(BASE)}__p2tpl@{self.rev[:12]}__select.json",
                                    "items; bytes per item and layer at the full grid (the base's SST-2 select)",
                                    self.grid)
        self._rate(f"{p2}/extract__{slug(BASE)}__p2tpl@{self.rev[:12]}__rest.json", "items per split", None,
                   ingested=False)
        self.base_sub_rate = self._rate(f"{p2}/extract__{slug(BASE)}__p2sub@{self.rev[:12]}__rest.json",
                                        "bytes per item and layer of a subset (the base's SST-2 rest subset)", None)
        # the per-adapter template (D-343 (1)): v0.3's ingested 7-layer p2sub caches, per split and pooling
        self.tpl: dict[tuple[str, str], dict[str, Any]] = {}
        acc: dict[str, list[int]] = {p: [0, 0] for p in BOTH}
        for set_name in ("select", "rest"):
            rel = f"{p2}/extract__{slug(MEASURED)}__p2sub@{self.measured_rev[:12]}__{set_name}.json"
            for key, s in self._splits(rel, f"per-adapter template ({set_name} set)").items():
                split, pooling = key.rsplit("/", 1)
                if s["n_items"] != self.items.get(split):
                    refuse(f"{rel} {key}: {s['n_items']} items, the base's records {self.items.get(split)}")
                self.tpl[(split, pooling)] = s
                acc[pooling][0] += s["cache_bytes"]
                acc[pooling][1] += s["n_items"] * len(s["layers_resolved"])
        if set(self.tpl) != {(sp, p) for sp in WG_SPLITS for p in BOTH}:
            refuse(f"v0.3's p2sub records do not hold every WildGuardMix/XSTest split x pooling: {sorted(self.tpl)}")
        self.tpl_rate = {p: Fraction(b, n) for p, (b, n) in acc.items()}
        man = self._json(f"{p2}/dataset_manifest_p2sst.json", "SST-2 items per split")
        for part, e in man["axes"]["p2sst"]["partitions"].items():
            self.items[f"p2sst/{part}"] = e["n"]
        # L* per (axis, pooling): the base's D-117 selections; an SST-2 selection not yet run is None (upper bound)
        self.lstar: dict[tuple[str, str], int | None] = {}
        for tag, prefix in (("p2tpl", "selection_p2"), ("p2sst", "selection_p2sst")):
            for pooling in BOTH:
                rel = f"{p2}/{prefix}__{slug(BASE)}@{self.rev[:12]}__{pooling}.json"
                if tag == "p2sst" and not (self.root / rel).is_file():
                    self.lstar[(tag, pooling)] = None
                    continue
                self.lstar[(tag, pooling)] = self._json(rel, f"L* ({tag}, {pooling})")["model"]["selected"]["layer"]
        # the adapters' weights: already ingested into data/lora (manifests/ingest_lora/), not priced as an ingest
        self.weights = {"dirs": 0, "bytes": 0, "files": 0, "per_weights": set()}
        for seed in SEEDS:
            for c in LA.CORPUS_ORDER:
                p = LA.record_path(c, seed, self.root)
                rel = p.relative_to(self.root).as_posix()
                rec = self._json(rel, "adapter checkpoint files and bytes")
                man_rel = f"manifests/ingest_lora/{p.stem}.sha256"
                if not (self.root / man_rel).is_file():
                    refuse(f"no {man_rel}: {p.name}'s adapters are not recorded as ingested")
                self.sources.append((man_rel, PV.sha256_file(self.root / man_rel), "the adapters' ingest into data/lora"))
                n_listed = sum(1 for x in (self.root / man_rel).read_text(encoding="utf-8").splitlines() if x.strip())
                n_files = 0
                for ck in rec["checkpoints"]:
                    self.weights["dirs"] += 1
                    for fname, f in ck["files"].items():
                        n_files += 1
                        self.weights["bytes"] += f["bytes"]
                        if fname == "adapter_model.safetensors":
                            self.weights["per_weights"].add(f["bytes"])
                if n_files != n_listed:
                    refuse(f"{man_rel} lists {n_listed} files, {p.name}'s checkpoints {n_files}")
                self.weights["files"] += n_files
        # the ingest's bound, floor and minimum allocation unit (scripts/p2_ingest_record.sh, read, never typed)
        sh_rel = "scripts/p2_ingest_record.sh"
        sh = (self.root / sh_rel).read_text(encoding="utf-8")
        self.sources.append((sh_rel, PV.sha256_file(self.root / sh_rel), "QUOTA_BOUND, FLOOR, minimum allocation unit"))
        q, f = re.search(r"^QUOTA_BOUND=([0-9]+)$", sh, re.M), re.search(r"^FLOOR=([0-9]+)$", sh, re.M)
        u = re.search(r'\[ "\$unit" -ge ([0-9]+) \] \|\| unit=([0-9]+)', sh)
        w = re.search(r"^LOCK_WAIT=([0-9]+)\b", sh, re.M)
        if not (q and f and u and w) or u.group(1) != u.group(2):
            refuse(f"{sh_rel} no longer states QUOTA_BOUND, FLOOR, LOCK_WAIT and its minimum allocation unit as read "
                   "here")
        self.quota_bound, self.floor, self.min_unit = int(q.group(1)), int(f.group(1)), int(u.group(1))
        self.lock_wait = int(w.group(1))
        self.unit = max(self.min_unit, alloc_unit or 0)

    def _json(self, rel: str, what: str) -> Any:
        p = self.root / rel
        if not p.is_file():
            refuse(f"no {rel} to price from ({what})")
        self.sources.append((rel, PV.sha256_file(p), what))
        return json.loads(p.read_text(encoding="utf-8"))

    def _splits(self, rel: str, what: str, ingested: bool = True) -> dict[str, Any]:
        """An extraction record's splits, each a cache of the base's shape; compressed (ingested) where it is read for
        its bytes."""
        d = self._json(rel, what)
        if ingested and d.get("ingest") != "yes":
            refuse(f"{rel} is not an ingested (compressed) record")
        sp = d.get("splits") or {}
        for key, s in sp.items():
            if (s.get("hidden_size") != self.hidden or s.get("n_model_layers", self.nl) != self.nl
                    or (ingested and s.get("cache_compressed") is False) or not isinstance(s.get("cache_bytes"), int)):
                refuse(f"{rel} {key}: not a compressed cache of {BASE}'s shape with its bytes")
        if not sp:
            refuse(f"{rel} holds no split")
        return sp

    def _rate(self, rel: str, what: str, layers: list[int] | None, ingested: bool = True) -> dict[str, Fraction]:
        """Bytes per item and layer, per pooling, over every cache of one record (its items per split recorded)."""
        acc: dict[str, list[int]] = {p: [0, 0] for p in BOTH}
        for key, s in self._splits(rel, what, ingested).items():
            split, pooling = key.rsplit("/", 1)
            if self.items.setdefault(split, s["n_items"]) != s["n_items"]:
                refuse(f"{rel} {key}: {s['n_items']} items, another record {self.items[split]}")
            if layers is not None and s["layers_resolved"] != layers:
                refuse(f"{rel} {key}: layers {s['layers_resolved']}, not {layers}")
            acc[pooling][0] += s["cache_bytes"]
            acc[pooling][1] += s["n_items"] * len(s["layers_resolved"])
        return {p: Fraction(b, n) for p, (b, n) in acc.items() if n}

    def layers(self, tag: str, layer_set: str, pooling: str) -> tuple[list[int] | None, int]:
        """The layers a subset holds (None while the axis's L* is not selected) and their count (then the upper bound,
        L* outside GRID6)."""
        lstar = self.lstar[(tag, pooling)]
        if lstar is None:
            return None, 1 if layer_set == "lstar" else len(self.grid6) + 1
        got = [lstar] if layer_set == "lstar" else sorted(set(self.grid6) | {lstar})
        return got, len(got)

    def adapter_cache(self, tag: str, split: str, pooling: str, layer_set: str, dtype: str) -> tuple[int, str]:
        """One adapter subset cache: (bytes, its BASIS code)."""
        layers, n = self.layers(tag, layer_set, pooling)
        if dtype == "f16" and pooling == "usermean":
            return self.items[split] * n * self.hidden * 2, "F"
        if tag == "p2sst":
            return math.ceil(self.tpl_rate[pooling] * self.items[split] * n), "S"
        s = self.tpl[(split, pooling)]
        if layers == s["layers_resolved"]:
            return s["cache_bytes"], "M"
        return math.ceil(Fraction(s["cache_bytes"] * n, len(s["layers_resolved"]))), "L"

    def base_sst(self, set_name: str) -> list[int]:
        """The base's own SST-2 ingests (basis S): its select set at the full grid (ingest yes), both poolings; its rest
        subset at GRID6 + its SST-2 L*, both poolings."""
        if set_name == "select":
            return [math.ceil(self.full_rate[p] * self.items[sp] * len(self.grid))
                    for sp in BASE_SST_SETS["select"][0] for p in BOTH]
        return [math.ceil(self.base_sub_rate[p] * self.items[sp] * self.layers("p2sst", "grid6_lstar", p)[1])
                for sp in BASE_SST_SETS["rest"][0] for p in BOTH]

    def on_nas(self, n: int) -> int:
        return -(-n // self.unit) * self.unit                          # p2_ingest_record.sh: rounded up per cache


def option_parts(sz: Sizes, ads: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, tuple[int, str]]]]:
    """{list key: {part: {dtype: (NAS bytes, BASIS codes)}}}: what each option list ingests (module docstring,
    PRICING), every cache rounded up to the allocation unit."""
    out = {}
    for key in [f"wg_{k}" for k in WG_OPTIONS] + [f"sst_{k}" for k in SST_OPTIONS]:
        acc: dict[str, dict[str, list[Any]]] = {pt: {dt: [0, set()] for dt in DTYPES} for pt, _ in PARTS}
        wg = key.startswith("wg_")
        for a in ads:
            if wg:
                layer_set, pools, ptrain = WG_OPTIONS[key[3:]][1](a["seed"])
            else:
                (layer_set, pools), ptrain = SST_OPTIONS[key[4:]][1], None
            part = "step0" if a["step"] == 0 else "s25_400"
            for split in (WG_SPLITS if wg else SST_SPLITS):
                if split.endswith("/probe_train") and ptrain != "keep":
                    continue
                for p in pools:
                    for dt in DTYPES:
                        b, how = sz.adapter_cache("p2tpl" if wg else "p2sst", split, p, layer_set, dt)
                        acc[part][dt][0] += sz.on_nas(b)
                        acc[part][dt][1].add(how)
        if not wg:
            for part, set_name in (("base_sel", "select"), ("base_rest", "rest")):
                for dt in DTYPES:
                    acc[part][dt][0] = sum(sz.on_nas(b) for b in sz.base_sst(set_name))
                    acc[part][dt][1].add("S")
        out[key] = {pt: {dt: (v[0], "".join(sorted(v[1]))) for dt, v in d.items()} for pt, d in acc.items()}
    return out


def row_bytes(parts: dict[str, Any], wg: str, sst: str | None, dtype: str, step0: bool) -> int:
    return sum(parts[k][pt][dtype][0] for k in (wg, sst) if k for pt, _ in PARTS if step0 or pt != "step0")


def price(headroom: int, root: Path = ROOT, alloc_unit: int | None = None) -> tuple[list[str], dict[str, Any]]:
    """The price table (module docstring, PRICING): lines to print and the figures per option list and row."""
    sz = Sizes(root, alloc_unit)
    if not 0 <= headroom <= sz.quota_bound - sz.floor:
        refuse(f"--headroom-bytes {headroom:,} is not between 0 and QUOTA_BOUND - FLOOR {sz.quota_bound - sz.floor:,}")
    ads, _, _ = adapters(root)
    n0, n = sum(a["step"] == 0 for a in ads), sum(a["step"] != 0 for a in ads)
    parts = option_parts(sz, ads)
    w = sz.weights
    out = [f"D-343 (1) PRICE (read-only; nothing written). NAS bytes each option ingests into data/hidden, every cache "
           f"rounded up to {sz.unit:,} (scripts/p2_ingest_record.sh{'' if alloc_unit else ', its minimum unit'}). "
           f"Headroom GIVEN {headroom:,} = QUOTA_BOUND {sz.quota_bound:,} - FLOOR {sz.floor:,} - allocated "
           f"(scripts/survey_export_root.sh; not measured here). Margin = headroom - NAS bytes.",
           f"Checkpoints cut per list: {n} at steps 25-400 + {n0} step-0 (one per seed) = {n + n0}. Adapter weights "
           f"({w['dirs']} checkpoint dirs, {w['files']} files, {w['bytes']:,} bytes, adapter_model.safetensors "
           f"{'/'.join(f'{x:,}' for x in sorted(w['per_weights']))} each) are already in data/lora "
           f"(manifests/ingest_lora/): inside the allocated bytes, not added.",
           "Basis: " + " ".join(f"{k} = {v}." for k, v in BASIS.items())]
    unsel = [p for p in BOTH if sz.lstar[("p2sst", p)] is None]
    if unsel:
        out.append(f"SST-2 L* not yet selected ({', '.join(unsel)}): GRID6 + L* counted as {len(sz.grid6) + 1} layers "
                   "(the upper bound).")
    out.append(f"{'PER LIST':<22}| " + " | ".join(f"{t:>21}" for _, t in PARTS) + " | "
               + " | ".join(f"{'f16 ' + t:>21}" for pt, t in PARTS[:2]))
    for key in parts:
        g = parts[key]
        cells = [f"{g[pt]['f32'][0]:,} {g[pt]['f32'][1] or '-':<2}" for pt, _ in PARTS]
        cells += [f"{g[pt]['f16'][0]:,} {g[pt]['f16'][1]:<2}" for pt, _ in PARTS[:2]]
        out.append(f"{key + ('*' if key in NOT_IN_D343 else ''):<22}| " + " | ".join(f"{c:>21}" for c in cells))
    out.append(f"{'OPTION (+ SST-2 list)':<36}| dt  | {'NAS with step 0':>17} | {'margin':>17} | {'fit':<12} | "
               f"{'NAS without step 0':>18} | {'margin':>17} | fit")
    rows = []
    for wg, sst in PRICE_ROWS:
        label = wg + ("*" if wg in NOT_IN_D343 else "") + (f" + {sst}" + ("*" if sst in NOT_IN_D343 else "")
                                                           if sst else "")
        for dt in DTYPES:
            r = {"wg": wg, "sst": sst, "dtype": dt}
            cells = []
            for step0 in (True, False):
                b = row_bytes(parts, wg, sst, dt, step0)
                r[step0] = (b, headroom - b)
                cells.append(f"{b:>{17 if step0 else 18},} | {headroom - b:>17,} | "
                             f"{'fits' if b <= headroom else 'DOES NOT FIT':<12}")
            rows.append(r)
            out.append(f"{label:<36}| {dt} | " + " | ".join(cells).rstrip())
    out.append("* not defined in D-343 (1) (D-355 (4), the author's). No SST-2 list under option D (D-343 (1) defines "
               "none). f16: the adapters' user-span mean subsets in float16 (D-357 (B)(5): only after the one-cell "
               "check).")
    out.append("sources (sha256):")
    seen = set()
    for rel, sha, what in sz.sources:
        if rel not in seen:
            seen.add(rel)
            out.append(f"  {rel} {sha} ({what})")
    return out, {"sizes": sz, "parts": parts, "rows": rows, "n_checkpoints": (n, n0)}


# --- the preflight (module docstring, THE PREFLIGHT; on the node, the first job of every option list) -----------------

def survey_allocated(root: Path, lock_wait: int) -> int:
    """The account's allocated bytes: scripts/survey_export_root.sh's ACCOUNT SUM, read under logs/nas_write.lock as
    scripts/p2_ingest_record.sh reads it (the same shell lock, the same wait)."""
    r = subprocess.run(["bash", "-c", 'exec 9>logs/nas_write.lock && flock -w "$1" 9 || exit 75; '
                                      'bash scripts/survey_export_root.sh', "preflight", str(lock_wait)],
                       cwd=root, capture_output=True, text=True)
    sums = re.findall(r"^\s*([0-9]+)\s+ACCOUNT SUM\b", r.stdout, re.M)
    if r.returncode != 0 or len(sums) != 1:
        refuse(f"the headroom survey under logs/nas_write.lock exited {r.returncode} with {len(sums)} account sums")
    return int(sums[0])


def preflight(name: str, root: Path = ROOT, survey: Any = survey_allocated, nas_unit: int | None = None,
              scans: Any = None) -> list[str]:
    """THE PREFLIGHT of one option list (module docstring): refuses unless (1) queues/<name> is what the records give
    (and its labels pass check_labels), (2) no other job of it has a .done or .exit in logs/queue/<lane>/, (3) for an
    SST-2 list the SST-2 scans pass for the base (extract_p2.sst_scans, D-340 (3)), (4) allocated + the whole list's
    NAS bytes (option_parts, float32, the step-0 subsets included, every cache rounded up to the NAS unit) + FLOOR fit
    under QUOTA_BOUND. Writes nothing (the lock file aside, as every NAS writer opens it)."""
    root = Path(root)
    key = next((k for k, v in NAMES.items() if v == name and k != "scans"), None)
    if key is None:
        refuse(f"{name} is not an option list of src/lora_jobs_p2.py ({sorted(v for k, v in NAMES.items() if k != 'scans')})")
    texts = lists(root)
    p = root / "queues" / name
    if not p.is_file() or p.read_bytes() != texts[name].encode("utf-8"):
        refuse(f"(1) queues/{name} is missing or not what the records give (src/lora_jobs_p2.py --check)")
    labels = [j[0] for j in jobs(texts[name])]
    qd = root / "logs" / "queue" / lane_of(name)
    ran = sorted(f"{lb}{ext}" for lb in labels[1:] for ext in (".done", ".exit") if (qd / f"{lb}{ext}").exists())
    if ran:
        refuse(f"(2) {len(ran)} job(s) of {name} already ran in {qd.relative_to(root).as_posix()}/ (first: {ran[0]}): "
               "a label collision, or a relaunch whose preflight has no .done; inspect by hand")
    out = [f"preflight {name}: (1) the list is what the records give, its labels unique in lane {lane_of(name)}; "
           f"(2) none of its {len(labels) - 1} other jobs has run"]
    if key.startswith("sst_"):
        if scans is None:
            import extract_p2 as X
            scans = X.sst_scans
        scans(BASE)
        out.append(f"  (3) the SST-2 scans pass for {BASE} (extract_p2.sst_scans, D-340 (3))")
    unit = nas_unit if nas_unit is not None else os.statvfs(root / "data").f_frsize
    sz = Sizes(root, unit)
    ads, _, _ = adapters(root)
    need = sum(option_parts(sz, ads)[key][pt]["f32"][0] for pt, _ in PARTS)
    alloc = survey(root, sz.lock_wait)
    total = alloc + need + sz.floor
    msg = (f"allocated {alloc:,} + this list's NAS bytes {need:,} (each cache rounded up to {sz.unit:,}) + FLOOR "
           f"{sz.floor:,} = {total:,} against QUOTA_BOUND {sz.quota_bound:,}")
    if total > sz.quota_bound:
        refuse(f"(4) {msg}: the list does not fit; nothing fetched or cut")
    out.append(f"  (4) {msg}: fits, margin {sz.quota_bound - total:,}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="write or check the LoRA-stage and SST-2 job lists (nothing is run), "
                                             "price each storage option (read-only), or run one option list's "
                                             "preflight (on the node)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--price", action="store_true", help="read-only: the NAS bytes of every storage option")
    g.add_argument("--preflight", metavar="LIST", help="the first job of every option list (module docstring)")
    ap.add_argument("--done-labels", type=Path, default=None,
                    help="with --write or --check: a listing of the node's logs/queue/*/*.done; every label is also "
                         "checked against it")
    ap.add_argument("--headroom-bytes", type=int, default=None,
                    help="with --price, required: QUOTA_BOUND - FLOOR - the account's allocated bytes, as the user "
                         "reads them from scripts/survey_export_root.sh (never measured here)")
    ap.add_argument("--alloc-unit", type=int, default=None,
                    help="with --price: the NAS allocation unit if larger than p2_ingest_record.sh's minimum")
    args = ap.parse_args(argv)
    if args.price:
        if args.headroom_bytes is None:
            refuse("--price needs --headroom-bytes (the user's figure from scripts/survey_export_root.sh)")
        lines, _ = price(args.headroom_bytes, ROOT, args.alloc_unit)
        print("\n".join(lines))
        return 0
    if args.headroom_bytes is not None or args.alloc_unit is not None:
        refuse("--headroom-bytes and --alloc-unit are for --price only")
    if args.preflight:
        if args.done_labels is not None:
            refuse("--done-labels is for --write and --check (the preflight reads the node's logs/queue/ itself)")
        print("\n".join(preflight(args.preflight)))
        return 0
    want = lists()
    if args.done_labels is not None:
        check_labels(want, ROOT, read_done_labels(args.done_labels))
    bad = []
    for name, text in want.items():
        p = QUEUES / name
        data = text.encode("utf-8")
        if p.exists():
            if p.read_bytes() != data:
                bad.append(name)
            continue
        if args.write:
            with open(p, "xb") as fh:                          # never overwritten
                fh.write(data)
            print(f"written queues/{name} ({len(jobs(text))} jobs)")
        else:
            bad.append(name)
    if bad:
        print(f"{'REFUSING: these lists exist and differ' if args.write else 'DIFFER or missing'}: {bad}")
        return 1
    print(f"{len(want)} lists {'in place' if args.write else 'equal to what the records give'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
