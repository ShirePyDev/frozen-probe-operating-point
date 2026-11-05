"""Extract frozen hidden states for the probe and similarity layers.

This is the only GPU stage in the generation-free pilot (E6). It runs one forward
pass per item per checkpoint and caches the pooled hidden states to disk, keyed by
(repo, revision, split, layer index, pooling). Everything downstream - probe
fitting, thresholding, stack composition, the homogeneous/heterogeneous contrast -
is an offline join over this cache (D-009).

Precision (D-027, adapted - see FAILURES.md F-016)
--------------------------------------------------
The V100 has no hardware bf16 (F-010: emulated, 8.3x slower than fp16), and the
agreed policy was fp32 for hidden-state extraction. That does not fit: Mistral-7B
in fp32 is ~29 GB of weights against 31.7 GB of VRAM, leaving no room for
activations. So:

  - main run: fp16 weights on GPU, hidden states upcast to fp32 the moment they
    leave the model, with a per-item non-finite flag recorded
  - validation: --fp32-check runs a subsample on CPU in fp32 and reports the
    max absolute deviation and rank correlation against the fp16 run

The non-finite flag matters because all candidate backbones were trained in bf16,
whose dynamic range exceeds fp16's 65504. If overflow struck the two checkpoints
of an update pair at different rates it would look exactly like the effect we are
trying to measure (RESEARCH_LEDGER R2). A flag that is always zero is evidence
the confound did not fire; a flag that is not is a finding in itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_env_guard import assert_hf_env  # noqa: E402
from provenance import (  # noqa: E402
    cache_write_dir,
    ensure_utf8_stdout,
    provenance_block,
    read_jsonl,
    sha256_file,
    write_results,
)

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "data" / "frozen"
CACHE = ROOT / "data" / "hidden"


def load_split(axis: str, part: str) -> list[dict[str, Any]]:
    path = FROZEN / f"{axis}__{part}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"frozen split missing: {path}. Run freeze_splits.py first.")
    return read_jsonl(path)


def atomic_savez(outp: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a cache so that a failed write can never leave a file at `outp`.

    F-025/F-027: a quota-exhausted NAS truncated a cache mid-write and left a
    corrupt .npz under the real name. Now: write to a temporary name, fsync, re-open
    and read every array back (the zip CRC is checked on read) and compare shapes,
    and only then rename into place. os.replace is atomic on the same filesystem.
    """
    # F-036: the temp name carries the PID, so two processes extracting the same cache
    # cannot write the same temporary file and rename it out from under each other.
    tmp = outp.with_name(f"{outp.name}.partial.{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        with np.load(tmp) as back:
            assert set(back.files) == set(arrays), "cache verify: key set differs"
            for k, v in arrays.items():
                assert back[k].shape == np.asarray(v).shape, f"cache verify: shape differs for {k}"
        os.replace(tmp, outp)
    finally:
        if tmp.exists():
            tmp.unlink()


def cache_path(repo: str, revision: str, axis: str, part: str, pooling: str,
               chat_template: bool = False) -> Path:
    safe_repo = repo.replace("/", "__")
    # The templated arm MUST NOT overwrite the raw cache - they are different
    # measurements of the same checkpoint.
    suffix = f"{pooling}__chat" if chat_template else pooling
    return cache_write_dir(CACHE) / f"{safe_repo}@{revision[:12]}__{axis}__{part}__{suffix}.npz"


def resolve_shape(config) -> tuple[int, int]:
    """Return (num_hidden_layers, hidden_size), descending into text_config.

    Multimodal releases nest the language-model shape. Verified on
    google/gemma-3-12b-pt: its config has NO top-level num_hidden_layers, only
    config.text_config.num_hidden_layers = 48 and hidden_size = 3840. Reading the
    top level would raise AttributeError, and guessing a default would silently
    extract the wrong layers.
    """
    n = getattr(config, "num_hidden_layers", None)
    h = getattr(config, "hidden_size", None)
    if n is None or h is None:
        inner = getattr(config, "text_config", None)
        if inner is None:
            raise ValueError(
                f"cannot resolve layer count / hidden size from {type(config).__name__}; "
                "no top-level fields and no text_config")
        n = n if n is not None else inner.num_hidden_layers
        h = h if h is not None else inner.hidden_size
    return int(n), int(h)


def load_model(repo: str, revision: str, dtype_name: str, device: str,
               adapter: str | None = None):
    """Load tokenizer and model ONCE per process.

    Loading inside the per-split loop was the original design and it OOM'd on the
    third split (FAILURES.md F-018): a 7B fp16 model is ~15 GB, `del` does not
    synchronously free CUDA memory, and two resident copies exceed 31.7 GiB.
    Loading once is both correct and 4x faster.

    With `adapter`, the base is loaded as a causal LM, the LoRA adapter applied and
    MERGED into the weights, as a merged adapter is served in deployment. The causal
    LM path is required because adapters trained by train_lora.py address modules
    under the causal-LM prefix; attaching one to the bare backbone would silently
    fail to match any weights. Its hidden_states are the backbone's, so caches stay
    comparable - and that is checked, not assumed, by extracting the step-0 no-op
    adapter and comparing against the base cache.
    """
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    dtype = {"float16": torch.float16, "float32": torch.float32}[dtype_name]
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # use_safetensors=True: transformers never falls back to a .bin pickle here (D-318 (7) item 8; review of olmo-bin)
    if adapter is None:
        model = AutoModel.from_pretrained(
            repo, revision=revision, dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
    else:
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            repo, revision=revision, dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    model.eval().to(device)
    return tok, model


def extract(
    tok,
    model,
    rows: list[dict[str, Any]],
    layer_indices: list[int],
    batch_size: int,
    max_length: int,
    device: str,
    pooling: str,
    apply_chat_template: bool = False,
) -> dict[str, Any]:
    """Run forward passes and return pooled hidden states plus health flags."""
    import torch

    n_layers, hidden_size = resolve_shape(model.config)
    resolved = [i if i >= 0 else n_layers + 1 + i for i in layer_indices]
    for i in resolved:
        if not 0 <= i <= n_layers:
            raise ValueError(f"layer index {i} out of range for {n_layers}-layer model")

    out = {li: np.zeros((len(rows), hidden_size), dtype=np.float32) for li in resolved}
    nonfinite = np.zeros(len(rows), dtype=bool)
    # Values that nan_to_num replaces with 0.0, per layer: the stored cache cannot show
    # them afterwards (F-016 addendum: a fully non-finite row is stored as all zeros).
    replaced = {li: 0 for li in resolved}
    truncated = np.zeros(len(rows), dtype=bool)
    n_tokens = np.zeros(len(rows), dtype=np.int32)

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            texts = [r["text"] for r in batch]
            if apply_chat_template:
                # CONFOUND CONTROL. Everywhere else in this project the SAME raw
                # prompt is fed to both sides of an update, which isolates the
                # weight change. But an instruct checkpoint has only ever seen
                # its own chat format, so un-templated input may be
                # out-of-distribution for it - which would be an alternative
                # explanation for the representation-mean shift measured in
                # D-046. This arm renders each prompt through the checkpoint's
                # own chat template so the two explanations can be separated.
                rendered = []
                for t in texts:
                    try:
                        rendered.append(tok.apply_chat_template(
                            [{"role": "user", "content": t}],
                            tokenize=False, add_generation_prompt=True))
                    except Exception:
                        rendered.append(t)  # no template defined; recorded below
                texts = rendered
            enc = tok(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            lens = enc["attention_mask"].sum(dim=1)
            for k, row in enumerate(batch):
                n_tokens[start + k] = int(lens[k])
                # A sequence hitting max_length exactly was probably cut short.
                truncated[start + k] = bool(lens[k] >= max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            res = model(**enc, output_hidden_states=True)
            mask = enc["attention_mask"].unsqueeze(-1).to(torch.float32)
            for li in resolved:
                h = res.hidden_states[li].to(torch.float32)
                if pooling == "mean":
                    pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                elif pooling == "last":
                    idx = (enc["attention_mask"].sum(dim=1) - 1).clamp(min=0)
                    pooled = h[torch.arange(h.size(0), device=h.device), idx]
                else:
                    raise ValueError(f"unknown pooling {pooling}")
                finite = torch.isfinite(pooled)
                bad = ~finite.all(dim=1)
                nonfinite[start : start + len(batch)] |= bad.cpu().numpy()
                replaced[li] += int((~finite).sum().item())
                out[li][start : start + len(batch)] = (
                    torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()
                )
            if start % (batch_size * 20) == 0:
                done = start + len(batch)
                rate = done / max(time.time() - t0, 1e-9)
                print(f"    {done}/{len(rows)}  {rate:.1f} items/s", flush=True)

    return {
        "hidden": out,
        "nonfinite": nonfinite,
        "replaced": replaced,
        "truncated": truncated,
        "n_tokens": n_tokens,
        "n_model_layers": n_layers,
        "hidden_size": hidden_size,
        "elapsed_s": time.time() - t0,
    }


def summary_path(args) -> Path:
    """Extraction-record filename, unique per (model, revision, axes, template arm).

    F-024: this was once keyed by model alone, so re-extracting a model on another
    axis or through the chat-template arm overwrote the earlier record.
    """
    tag = args.cache_repo.replace("/", "__") + "@" + args.cache_rev[:12]
    axes_tag = "+".join(sorted(args.axes))
    arm = "__chat" if args.chat_template else ""
    # A re-extraction that rebuilds a lost cache must NOT overwrite the original
    # record: that record holds the cache_sha256 the rebuild is checked against, so
    # overwriting it would destroy the very evidence the check needs.
    rerun = f"__rerun_{args.rerun_tag}" if getattr(args, "rerun_tag", None) else ""
    return ROOT / "results" / f"extract__{tag}__{axes_tag}{arm}{rerun}.json"


def main() -> int:
    ensure_utf8_stdout()
    # Before anything loads: HTTP endpoint, NAS HF_HOME or an exported token -> exit 78.
    assert_hf_env("extract_hidden.py")
    ap = argparse.ArgumentParser(description="Extract pooled hidden states")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True, help="exact commit SHA; never a branch name")
    ap.add_argument("--axes", nargs="*", default=["safety", "over_refusal"])
    ap.add_argument("--parts", nargs="*", default=["probe_train", "threshold_calib", "eval"])
    ap.add_argument("--layers", nargs="*", type=int, default=[8, 16, 24, -1])
    ap.add_argument("--pooling", default="mean", choices=["mean", "last"])
    ap.add_argument("--chat-template", action="store_true",
                    help="render each prompt through the checkpoint's own chat template "
                         "(confound control for instruct models; see D-057)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter directory; merged into the base before extraction")
    ap.add_argument("--adapter-name", default=None,
                    help="identity for the adapted model's caches, e.g. mistral7b-alpaca-r16-s100")
    ap.add_argument("--rerun-tag", default=None,
                    help="suffix the extraction record, for rebuilding a lost cache "
                         "without overwriting the original record it is checked against")
    ap.add_argument("--cache-tag", default=None,
                    help="suffix for the cache identity, e.g. a100. Use when re-extracting the "
                         "SAME weights under a different execution environment (GPU architecture, "
                         "library version) so the new caches cannot overwrite the old ones and the "
                         "two can be compared as a pair (D-087)")
    args = ap.parse_args()

    # Cache identity. A merged adapter is a DIFFERENT model from its base, so it
    # must never write to the base's cache path. Its revision is the SHA-256 of
    # the adapter weights, so two adapters can never collide.
    if args.adapter:
        if not args.adapter_name:
            print("--adapter requires --adapter-name")
            return 2
        adir = Path(args.adapter)
        adir = adir if adir.is_absolute() else ROOT / adir
        afile = adir / "adapter_model.safetensors"
        if not afile.exists():
            print(f"adapter weights not found: {afile}")
            return 2
        args.adapter = str(adir)
        args.adapter_sha256 = sha256_file(afile)
        args.cache_repo = f"lora/{args.adapter_name}"
        args.cache_rev = args.adapter_sha256
    else:
        args.adapter_sha256 = None
        args.cache_repo, args.cache_rev = args.repo, args.revision
    if args.cache_tag:
        assert "@" not in args.cache_tag and "/" not in args.cache_tag, "cache tag must be a plain token"
        args.cache_repo = f"{args.cache_repo}__{args.cache_tag}"

    CACHE.mkdir(parents=True, exist_ok=True)
    # Refuse BEFORE any GPU work, not after: failing at the end would discard a
    # completed extraction's record, which is the very loss F-024 is about.
    if summary_path(args).exists():
        print(f"REFUSING: {summary_path(args).name} already exists. This exact extraction "
              "(model, revision, axes, arm) has been recorded before.")
        return 2
    # Library versions are load-bearing provenance: a transformers upgrade can
    # change numerics, and caches produced under different versions are not
    # comparable. Recording this is what makes a later upgrade auditable
    # (RESEARCH_LEDGER R6).
    import torch as _torch
    import transformers as _tf

    summary: dict[str, Any] = {
        **provenance_block("extract_hidden"),
        "lib_versions": {
            "torch": _torch.__version__,
            "transformers": _tf.__version__,
            "cuda": _torch.version.cuda,
            "device_name": (_torch.cuda.get_device_name(0)
                            if _torch.cuda.is_available() else None),
        },
        "repo": args.repo,
        "revision": args.revision,
        "adapter": args.adapter,
        "adapter_sha256": args.adapter_sha256,
        "cache_repo": args.cache_repo,
        "cache_revision": args.cache_rev,
        "layers_requested": args.layers,
        "pooling": args.pooling,
        "chat_template_applied": args.chat_template,
        "tokenizer_has_chat_template": None,
        "dtype": args.dtype,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "splits": {},
    }

    print(f"extracting: {args.repo} @ {args.revision[:12]}  dtype={args.dtype} device={args.device}")
    tok, model = load_model(args.repo, args.revision, args.dtype, args.device, args.adapter)
    _nl, _hs = resolve_shape(model.config)
    summary["tokenizer_has_chat_template"] = bool(getattr(tok, "chat_template", None))
    if args.chat_template and not summary["tokenizer_has_chat_template"]:
        print("  WARNING: --chat-template requested but this tokenizer defines none; "
              "prompts are passed through raw and this is recorded in the results file")
    print(f"  model loaded: {_nl} layers, hidden_size {_hs}")
    for axis in args.axes:
        for part in args.parts:
            path = FROZEN / f"{axis}__{part}.jsonl"
            if not path.exists():
                print(f"  skip {axis}/{part} (no frozen split)")
                continue
            rows = load_split(axis, part)
            print(f"  {axis}/{part}: {len(rows)} items")
            res = extract(
                tok, model, rows, args.layers, args.batch_size,
                args.max_length, args.device, args.pooling, args.chat_template,
            )
            outp = cache_path(args.cache_repo, args.cache_rev, axis, part, args.pooling,
                              args.chat_template)
            arrays = {
                "item_ids": np.array([r["item_id"] for r in rows]),
                "labels": np.array([r["label"] for r in rows], dtype=np.int8),
                "nonfinite": res["nonfinite"],
                "truncated": res["truncated"],
                "n_tokens": res["n_tokens"],
                **{f"layer_{li}": res["hidden"][li] for li in res["hidden"]},
            }
            atomic_savez(outp, arrays)
            summary["splits"][f"{axis}/{part}"] = {
                "n": len(rows),
                "split_sha256": sha256_file(path),
                "cache_file": outp.name,
                "cache_sha256": sha256_file(outp),
                "n_nonfinite": int(res["nonfinite"].sum()),
                "nonfinite_replacements": {
                    "replacement": "nan_to_num -> 0.0, after the flag is set",
                    "item_indices": np.flatnonzero(res["nonfinite"]).tolist(),
                    "item_ids": [rows[i]["item_id"] for i in np.flatnonzero(res["nonfinite"])],
                    "values_replaced_per_layer": {str(li): n for li, n in sorted(res["replaced"].items())},
                },
                "n_truncated": int(res["truncated"].sum()),
                "mean_tokens": float(res["n_tokens"].mean()),
                "max_tokens": int(res["n_tokens"].max()),
                "elapsed_s": round(res["elapsed_s"], 1),
                "layers_resolved": sorted(res["hidden"].keys()),
                "n_model_layers": res["n_model_layers"],
                "hidden_size": res["hidden_size"],
            }
            print(f"    -> {outp.name}  nonfinite={int(res['nonfinite'].sum())} "
                  f"truncated={int(res['truncated'].sum())}  {res['elapsed_s']:.0f}s")

    # F-024: the summary used to be named by model alone, so a second extraction of
    # the same model on another axis, or through the chat-template arm, silently
    # overwrote the first run's record. The name now carries everything that makes
    # two runs different measurements.
    outjson = summary_path(args)
    digest = write_results(outjson, summary)
    print(f"\nWritten: {outjson.name}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
