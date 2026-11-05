"""Train LoRA adapters on a base checkpoint, saving a DOSE series of checkpoints.

Purpose
-------
arXiv:2606.15980 found LoRA-style updates stale frozen probes and described the
mechanism as the readout DIRECTION falling out of alignment. On instruction tuning
and version increments we found a moment shift (D-077) and refuted the rotation
account (D-045). Their update type is LoRA, so the only way to settle whether the
mechanisms differ by update type is to measure LoRA updates under OUR protocol.

Training our own adapters, rather than downloading community ones, is deliberate:
it lets the update DOSE be controlled (checkpoints at several step counts), which
turns "does LoRA break the probe" into "how does the mechanism change as the
update grows" - a question that could reconcile the two accounts rather than
merely pick one.

Design choices (recorded, all overridable)
------------------------------------------
  targets    q,k,v,o,gate,up,down  - every linear layer, so the update can reach
             the representation the probe reads, not just attention
  rank 16, alpha 32, dropout 0.05, lr 2e-4 constant after 10 warmup steps
  precision  base weights fp16 (V100 has no hardware bf16, F-010); adapter
             weights fp32; fp16 autocast with a GradScaler; non-finite losses are
             counted and training aborts if they persist
  data       yahma/alpaca-cleaned, general instruction following - the prior
             work's "general instruction-following" LoRA condition
  loss       causal LM loss on the full formatted sequence

Controls built in
-----------------
  step 0 is saved BEFORE any optimiser step. LoRA initialises B = 0, so the merged
  weights equal the base exactly, and hidden states extracted through that adapter
  must reproduce the base cache. That validates the adapter pipeline before any
  result depends on it.

  contamination: every training instruction is hashed exactly as freeze_splits.py
  hashes prompts, and compared with every frozen item_id on every axis. The count
  is recorded; training on evaluation prompts would invalidate the experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_env_guard import assert_hf_env  # noqa: E402
from provenance import (  # noqa: E402
    canonical_hash,
    ensure_utf8_stdout,
    provenance_block,
    read_jsonl,
    sha256_file,
    write_results,
)

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "data" / "frozen"


def fmt(row: dict[str, Any]) -> str:
    ins, inp, out = row["instruction"].strip(), (row.get("input") or "").strip(), row["output"].strip()
    if inp:
        return f"### Instruction:\n{ins}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    return f"### Instruction:\n{ins}\n\n### Response:\n{out}"


def first_human_turn(dialog: str) -> str:
    i = dialog.find("Human:")
    if i < 0:
        return dialog.strip()
    rest = dialog[i + len("Human:"):]
    j = rest.find("Assistant:")
    return (rest[:j] if j >= 0 else rest).strip()


def text_and_prompt(row: dict[str, Any], data_format: str) -> tuple[str, str]:
    """Training text and the PROMPT used for the contamination hash."""
    if data_format == "alpaca":
        return fmt(row), row["instruction"].strip()
    if data_format == "hh":
        # HH-RLHF harmless "chosen" transcript: the more harmless of two replies.
        return row["chosen"].strip(), first_human_turn(row["chosen"])
    if data_format == "pku":
        # PKU-SafeRLHF: the prompt with the response the dataset itself labels SAFER,
        # rendered in the same Human/Assistant transcript form the hh arm uses, so the
        # text FORMAT is held constant and only the corpus content varies (D-106).
        prompt = row["prompt"].strip()
        safer = int(row["safer_response_id"])
        reply = row[f"response_{safer}"].strip()
        return "Human: " + prompt + "\n\nAssistant: " + reply, prompt
    raise ValueError(f"unknown format {data_format}")


def frozen_item_ids() -> set[str]:
    ids: set[str] = set()
    for p in FROZEN.glob("*.jsonl"):
        ids.update(r["item_id"] for r in read_jsonl(p))
    return ids


def main() -> int:
    ensure_utf8_stdout()
    # Before anything loads: HTTP endpoint, NAS HF_HOME or an exported token -> exit 78.
    assert_hf_env("train_lora.py")
    ap = argparse.ArgumentParser(description="Train a LoRA dose series")
    ap.add_argument("--base-repo", required=True)
    ap.add_argument("--base-rev", required=True)
    ap.add_argument("--name", required=True, help="short id, used in output paths")
    ap.add_argument("--dataset", default="yahma/alpaca-cleaned")
    ap.add_argument("--data-dir", default=None, help="e.g. harmless-base for Anthropic/hh-rlhf")
    ap.add_argument("--format", default="alpaca", choices=["alpaca", "hh", "pku"])
    ap.add_argument("--exclude-frozen", action="store_true",
                    help="drop every example whose prompt is verbatim in a frozen split (D-080)")
    ap.add_argument("--save-steps", nargs="+", type=int, default=[0, 25, 100, 400])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--targets", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import peft as _peft
    import transformers as _tf

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    # F-032: adapters may have to be written to scratch while the NAS quota is full.
    # Unset, this is exactly the previous path.
    out_dir = Path(os.environ.get("GUARDRAIL_LORA_DIR", ROOT / "data" / "lora")) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = ROOT / "results" / f"lora_train__{args.name}.json"
    if record_path.exists():
        print(f"REFUSING: {record_path.name} exists; this run is already recorded")
        return 2

    max_step = max(args.save_steps)
    n_needed = max_step * args.micro_bs * args.grad_accum

    ds = (load_dataset(args.dataset, data_dir=args.data_dir, split="train") if args.data_dir
          else load_dataset(args.dataset, split="train"))
    ds_rev = HfApi().dataset_info(args.dataset).sha
    order = [int(i) for i in rng.permutation(len(ds))]

    frozen_ids = frozen_item_ids()
    phash = lambda i: hashlib.sha256(  # noqa: E731
        text_and_prompt(ds[i], args.format)[1].encode("utf-8")).hexdigest()[:16]
    n_excluded = 0
    if args.exclude_frozen:
        kept = [i for i in order if phash(i) not in frozen_ids]
        n_excluded = len(order) - len(kept)
        order = kept
        print(f"excluded {n_excluded} examples whose prompt is verbatim in a frozen split")
    if len(order) < n_needed:
        print(f"ABORT: need {n_needed} examples, only {len(order)} available")
        return 2
    overlap = sum(1 for i in order[:n_needed] if phash(i) in frozen_ids)
    print(f"training examples needed {n_needed}; overlap with frozen eval item_ids: {overlap}")

    tok = AutoTokenizer.from_pretrained(args.base_repo, revision=args.base_rev)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(        # use_safetensors: never a .bin pickle (D-318 (7) item 8)
        args.base_repo, revision=args.base_rev, dtype=torch.float16, low_cpu_mem_usage=True,
        use_safetensors=True).cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    lcfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                      target_modules=args.targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()  # adapter weights in fp32
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_trainable}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                            weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")

    def save(step: int) -> dict[str, Any]:
        d = out_dir / f"step_{step}"
        model.save_pretrained(d)
        f = d / "adapter_model.safetensors"
        try:
            where = str(d.relative_to(ROOT))
        except ValueError:  # adapters on scratch storage are outside the repo (F-032)
            where = str(d)
        return {"step": step, "path": where, "sha256": sha256_file(f)}

    checkpoints: list[dict[str, Any]] = []
    if 0 in args.save_steps:
        checkpoints.append(save(0))
        print(f"  saved step 0 (no-op control) sha {checkpoints[-1]['sha256'][:12]}")

    losses: list[dict[str, float]] = []
    nonfinite_losses = 0
    cursor = 0
    model.train()
    t0 = time.time()
    for step in range(1, max_step + 1):
        lr = args.lr * min(1.0, step / max(args.warmup, 1))
        for g in opt.param_groups:
            g["lr"] = lr
        step_loss = 0.0
        for _ in range(args.grad_accum):
            rows = [ds[order[(cursor + j) % len(order)]] for j in range(args.micro_bs)]
            cursor += args.micro_bs
            texts = [text_and_prompt(r, args.format)[0] + tok.eos_token for r in rows]
            enc = tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=args.seq_len)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            enc = {k: v.cuda() for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**enc, labels=labels.cuda()).loss / args.grad_accum
            if not torch.isfinite(loss):
                nonfinite_losses += 1
                continue
            scaler.scale(loss).backward()
            step_loss += float(loss)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        losses.append({"step": step, "loss": step_loss, "lr": lr})
        if nonfinite_losses > 10 * args.grad_accum:
            print(f"ABORT: {nonfinite_losses} non-finite micro-batch losses")
            break
        if step % 5 == 0 or step == 1:
            rate = (time.time() - t0) / step
            print(f"  step {step}/{max_step}  loss {step_loss:.4f}  {rate:.1f}s/step  "
                  f"nonfinite {nonfinite_losses}", flush=True)
        if step in args.save_steps:
            checkpoints.append(save(step))
            print(f"  saved step {step} sha {checkpoints[-1]['sha256'][:12]}", flush=True)

    payload = {
        **provenance_block("lora_train"),
        "lib_versions": {"torch": torch.__version__, "transformers": _tf.__version__,
                         "peft": _peft.__version__},
        "config": vars(args), "config_hash": canonical_hash(vars(args)),
        "dataset_revision": ds_rev, "n_examples_used": min(cursor, n_needed),
        "contamination_overlap_with_frozen_items": overlap,
        "n_excluded_for_frozen_overlap": n_excluded,
        "data_format": args.format,
        "trainable_params": n_trainable, "nonfinite_microbatch_losses": nonfinite_losses,
        "elapsed_s": round(time.time() - t0, 1),
        "checkpoints": checkpoints, "loss_curve": losses,
    }
    digest = write_results(record_path, payload)
    print(f"\nWritten: {record_path}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
