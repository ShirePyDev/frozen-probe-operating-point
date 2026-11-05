# frozen-probe-operating-point

Code and records for the paper *Operating-Point Failure of Frozen Guardrail Probes Under Backbone Updates*, by Yusuf
Khalid Shire and Sang-Chul Kim.

The guardrail we study is a linear probe on a language model's hidden states that flags harmful prompts, with its
threshold set once for a target false-positive rate. We keep the probe and its threshold frozen while the model
underneath is replaced by a released update or a controlled fine-tune, and measure where the operating point lands:
the false-positive and detection rates the old threshold now gives, whether ranking (AUROC) shows the change, whether
the checks a deployer would run notice it, which repairs restore the rate without new labels, and what an attacker who
controls the update can do.

The paper's numbers can be recomputed from the records here. Each result record names the commit and SHA-256 of the
code that wrote it, its parameters, and the SHA-256 of every file it read. Each table and figure is written by a
script from those records, and each file the scripts write begins with the SHA-256 of every record it read.

## What is here

- `src/`: all of the Python. Splitting the datasets, extracting hidden states, fitting probes, the analyses, Phase 3's
  attackers and defences, and the five scripts that write the tables and figures. It is one flat directory because the
  modules import one another by name and the records name them by these paths.
- `results/`: the per-item result records that the table and figure writers read, and the records of the datasets
  and splits.
- `manifests/splits/`, `manifests/splits_p2/`, `manifests/splits_p2_v2/`: one manifest per frozen split, with each
  item's identifier, label and source row, and the SHA-256 of the split file. No prompt text. The Phase 3 split is
  recorded in `results/phase3_capcost_wgtest_split.json`.
- `docs/registered_predictions.md`: the registry of our predictions. For each one it gives the wording, the decision-log
  entries that registered and scored it, the result files with their hashes, and the verdict.
- `docs/paper/`: two records that two of the writers read (`config_deltas_d471.json`, `review_20261016b_calc.json`) and
  the three scripts that produced them. `scripts/tally_ledger_verdicts_d449.py` is imported by one of those scripts,
  and `logs/table3_tpr_d282_20261003.sha256` is a hash list that one writer checks before it trusts a record.
- `requirements-tables.txt` and `requirements.txt`: pinned environments for the writers and for the experiments.
- `config/example.env`: every setting the code reads from the environment, with example values.
- `PROVENANCE.md`: the files whose bytes differ from the ones that produced the results, with their diffs.

Every file sits at the path it had in our working repository. The code finds its inputs by those paths and the
records name them, so moving a file would mean editing code and breaking the hashes. That is also why two records sit
under `docs/paper/` and a hash list under `logs/`.

## Reproducing the tables and figures

The writers need Python 3.12 and numpy, nothing else. From the repository root:

```sh
python3.12 -m venv .venv
. .venv/bin/activate                        # Windows: .venv\Scripts\activate
pip install -r requirements-tables.txt
mkdir -p build/tables
export GUARDRAIL_PAPER_DIR=build/tables     # Windows PowerShell: $env:GUARDRAIL_PAPER_DIR = "build/tables"
python src/config_table_d472.py
python src/pairs_table_d468.py
python src/results_tables_d504.py
python src/results_figures_d570.py
python src/supp_tables_d585.py
```

Keep this order: `supp_tables_d585.py` reads back the `tables_results.tex` that `results_tables_d504.py` writes. The
scripts read only files in this repository and write only into `GUARDRAIL_PAPER_DIR`, which is taken relative to the
root and must lie inside the repository. Left unset, it is `paper/`.

| File written | Where it appears in the paper |
|---|---|
| `tables_config.tex` | the table of configuration changes of the update pairs (`tab:config`) |
| `tables_results.tex` | the three Results tables: the headline cells (`tab:headline`), the matched-dose LoRA arms (`tab:lora`) and the estimated-rate check per Phase 1 cell (`tab:fourc`) |
| `figures_results.tex` | the figure of realised paired ratio against the estimated rate ratio (`fig:fourc-scatter`) |
| `tables_pairs.tex` | Supplement: the update pairs and their cells (`tab:pairs`) |
| `tables_supplement.tex` | Supplement: `tab:x5`, `tab:geomean`, `tab:contrast` and `tab:lastgen` |
| `figures_supplement.tex` | Supplement: paired ratio against layer (`fig:layer-sweep`) |

The figures are pgfplots code, not images, so their numbers can be read in the `.tex` files. We regenerated all six
files in a fresh environment, from this repository's files alone. Each was byte-identical to the file the manuscript
is built from.

Two tables cannot be regenerated here: the registry table in the Supplement (`tab:registry`) and the claims table in
the Introduction (`tab:claims`). Their writers read the manuscript's LaTeX source to find the section where each claim
is stated, and the manuscript is not in this repository. The registry that the first of them typesets is
`docs/registered_predictions.md`.

## Re-running the experiments

This is a far larger job than regenerating the tables. It needs the following.

- Linux and NVIDIA GPUs. The records name two GPUs: a Tesla V100-DGXS-32GB, used for Phase 1, and an NVIDIA
  A100-SXM4-40GB, used for Phase 2 and for some Phase 1 work. The paper's Method section says which cells ran where.
  The GPU stages are extraction (one forward pass per prompt, no generation), LoRA training, and Phase 3's attack
  training and capability-cost scoring. The probe fits and the analyses run on CPU.
- NVMe storage. Every script that fetches or loads weights calls `src/hf_env_guard.py`, which refuses to run unless
  `HF_HOME` lies on an NVMe device as `/proc/self/mounts` reports it. The Phase 2 extraction, the Phase 2 and
  Phase 3 trainers and the capability-cost harness make the same check of their own directories.
- Python 3.10 with the versions in `requirements.txt`, which are the ones the compute node's records name. torch is the
  CUDA 12.6 build. Two imported packages, urllib3 and PyYAML, have no recorded version and are not pinned.
- The settings in `config/example.env`. A Hugging Face token for gated checkpoints and datasets goes in a file named
  `.env` at the repository root, where `src/fetch_weights.py` reads it. The guard refuses to run while a token is set
  in the environment.
- The datasets. Each comes from its owner under its owner's licence, and we redistribute none of their text.

| Dataset | Used for |
|---|---|
| `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` | the Phase 1 safety splits |
| `allenai/wildguardmix` | the Phase 1 and Phase 2 safety splits; its test set gives the Phase 3 split |
| `natolambert/xstest-v2-copy` | the over-refusal split (XSTest) |
| `nyu-mll/glue` (SST-2) | the non-safety control of Phase 2 |
| `yahma/alpaca-cleaned`, `Anthropic/hh-rlhf`, `PKU-Alignment/PKU-SafeRLHF` | training the controlled LoRA updates |
| `allenai/ai2_arc` (ARC-Easy) | the capability-cost measurements of Phase 3 |

The revision each split was built from is recorded: Phase 1's in `results/dataset_manifest.json` and
`results/dataset_manifest_wildguard.json`, Phase 2's in `results/phase2/dataset_manifest_p2*.json`, and Phase 3's in
its split record. To rebuild a split, run the script that its manifest's note names against that revision, then
compare the SHA-256 of the file it writes with `sha256_of_original` in the manifest. The revisions of the LoRA corpora
and of ARC are in the training and capability-cost records, and the checkpoints of each update pair, with their
revisions, are named in the configurations of the records. Some of those records are among the ones not in this
repository (see below).

We kept no total of GPU or CPU time and do not estimate one here. Each record carries the UTC time at which it was
written.

## What is not here, and why

- The canary prompts. One of the defences in the paper checks detection on a set of prompts that only the deployer
  holds, and it works only while an attacker cannot know which prompts those are. We publish neither the list nor the
  arrays of that defence arm that mark them.
- Dataset prompt text. The licences belong to the datasets' owners. The split manifests carry identifiers and hashes,
  so a rebuilt split can be checked against ours.
- Model artefacts: the LoRA adapters of the controlled updates, the attacked models of Section V-H, and any
  checkpoint. They are too large for git, and whether to release a model update that evades the monitor is a decision
  we have not yet made. The code that trains them is in `src/`.
- Caches: the hidden-state activations and the raw probe scores. They are large, and the extraction code regenerates
  them.
- Most of the per-item records. This repository carries the records that the table and figure writers read. The
  others, behind the rest of the paper's numbers, are too large for a git repository, and where they will be deposited
  is not yet decided: [ARCHIVE LOCATION: placeholder, to be filled in].
- The decision log. Each registry row cites, by D-number, the entries of our decision log that registered and scored
  the prediction. Those entries and their dates are not in this repository yet, so the order of registration and
  scoring cannot be checked from here alone.
- Our test suite, and the scripts that ran jobs on our compute node.

## Files changed for publication

Twelve files differ from the code and records that produced the results. In eight modules an output directory or the
compute node's home path became a setting. Three modules check the SHA-256 of `src/p2auroc_d424.py`, one of the eight,
before they run, and their pins now hold its new value; two of the three are among the eight, and the third changed
in that line only. In three records, five absolute paths each became `<HOME>`. `PROVENANCE.md` gives both SHA-256 values and the full
diff of each file. Every other file copied from our working repository is byte-identical to its original; the
files written for this release (this README, `LICENSE`, `CITATION.cff`, `PROVENANCE.md`, the two requirements files,
`.gitignore`, `.gitattributes` and `config/example.env`) are listed in `PROVENANCE.md`.

## Citing

Until the paper appears, please cite it as:

```bibtex
@unpublished{shire2026operating,
  author = {Shire, Yusuf Khalid and Kim, Sang-Chul},
  title  = {Operating-Point Failure of Frozen Guardrail Probes Under Backbone Updates},
  year   = {2026},
  note   = {Manuscript}
}
```

`CITATION.cff` holds the same entry. Repository: [REPOSITORY URL: placeholder, to be added when the repository is
published].

## Licence

No licence has been chosen yet; `LICENSE` is a placeholder.
