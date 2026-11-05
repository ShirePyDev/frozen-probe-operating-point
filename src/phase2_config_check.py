"""Phase 2 config check (docs/phase2_design.md section 2; the author's D-281 (2) and D-297 (3)): public metadata and
config.json ONLY, nothing else downloaded. Gated repos are read through https://huggingface.co with the token.

For every candidate model, including Phi-3-mini-128k and the three Tulu-3 checkpoints:
  - the public Hub metadata (https://huggingface.co/api/models/<repo>): the commit sha, the gating state and
    whether the repo exists;
  - config.json at that commit (https://huggingface.co/<repo>/raw/<sha>/config.json), with its sha256 and content.
For every pair (pre -> post): every config key that differs, with the six the design names flagged (hidden_size,
num_hidden_layers, vocab_size, rope_theta, rope_scaling, sliding_window). A pair whose hidden size or layer count
differs is DROPPED (a frozen probe cannot read the post model). For Phi, both Phi-3-mini variants are compared with
Phi-3.5-mini, and the record says which one's positional settings match (rope_theta, rope_scaling,
max_position_embeddings, sliding_window, original_max_position_embeddings).

TOKEN: read from .env by fetch_weights.load_token(), only when a repo's metadata says it is gated or its config is
refused without one. It is attached by fetch_weights.auth_headers(), which sends it to https://huggingface.co only.
Redirects are refused, so the token can never follow one. It is never printed, logged or written. The tokenizer
comparison of the design (the v0.2 -> v0.3 check) needs tokenizer files and the prompts, and is not part of this
check. Writes results/phase2_config_check.json (never overwritten) only when every config was read and every pair is
comparable; otherwise the final name refuses and nothing is written. A test writes outside results/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as P  # noqa: E402
from provenance import ensure_utf8_stdout, provenance_block, write_results  # noqa: E402

CODE_FILES = ["src/phase2_config_check.py", "src/fetch_weights.py", "src/hf_env_guard.py", "src/provenance.py",
              "src/paired.py"]

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FINAL_OUT = RESULTS / "phase2_config_check.json"
HUB = "https://huggingface.co"
MODELS = ["mistralai/Mistral-7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.3",
          "meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
          "Qwen/Qwen2-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
          "microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3-mini-128k-instruct", "microsoft/Phi-3.5-mini-instruct",
          "allenai/OLMo-2-1124-7B", "allenai/OLMo-2-1124-7B-SFT", "allenai/OLMo-2-1124-7B-DPO",
          "allenai/OLMo-2-1124-7B-Instruct",
          "allenai/Llama-3.1-Tulu-3-8B-SFT", "allenai/Llama-3.1-Tulu-3-8B-DPO", "allenai/Llama-3.1-Tulu-3-8B"]
PAIRS = [("mistralai/Mistral-7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.2"),
         ("mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.3"),
         ("meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
         ("Qwen/Qwen2-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
         ("microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3.5-mini-instruct"),
         ("microsoft/Phi-3-mini-128k-instruct", "microsoft/Phi-3.5-mini-instruct"),
         ("allenai/OLMo-2-1124-7B", "allenai/OLMo-2-1124-7B-SFT"),
         ("allenai/OLMo-2-1124-7B-SFT", "allenai/OLMo-2-1124-7B-DPO"),
         ("allenai/OLMo-2-1124-7B-DPO", "allenai/OLMo-2-1124-7B-Instruct"),
         ("allenai/Llama-3.1-Tulu-3-8B-SFT", "allenai/Llama-3.1-Tulu-3-8B-DPO"),
         ("allenai/Llama-3.1-Tulu-3-8B-DPO", "allenai/Llama-3.1-Tulu-3-8B")]
DESIGN_FIELDS = ("hidden_size", "num_hidden_layers", "vocab_size", "rope_theta", "rope_scaling", "sliding_window")
DROP_FIELDS = ("hidden_size", "num_hidden_layers")
POSITIONAL = ("rope_theta", "rope_scaling", "max_position_embeddings", "sliding_window", "original_max_position_embeddings")
UA = {"User-Agent": "guardrail-ccf-phase2-config-check/1.0"}

SESSION = requests.Session()
SESSION.trust_env = False                      # no credential from the environment or ~/.netrc is ever sent
_TOKEN: dict[str, str | None] = {}


def token() -> str | None:
    if "t" not in _TOKEN:
        import fetch_weights as FW             # imported only when a gated repo needs the token
        _TOKEN["t"] = FW.load_token()
    return _TOKEN["t"]


def headers(url: str, use_token: bool) -> dict[str, str]:
    if not use_token:
        return dict(UA)
    import fetch_weights as FW
    return {**UA, **FW.auth_headers(url, token())}  # the token goes to https://huggingface.co only


def get(url: str, use_token: bool) -> requests.Response:
    """GET from the Hub; the response carries .token_sent (whether an Authorization header went out)."""
    if not url.startswith(HUB + "/"):
        raise SystemExit(f"REFUSING: {url} is not on {HUB}")
    last = None
    for attempt in range(5):
        try:
            h = headers(url, use_token)
            r = SESSION.get(url, headers=h, timeout=60, allow_redirects=False)
            r.token_sent = "Authorization" in h            # type: ignore[attr-defined]
        except requests.RequestException as e:
            last = f"{type(e).__name__}"
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            last = f"HTTP {r.status_code}"
            time.sleep(10 * (attempt + 1))
            continue
        return r
    raise SystemExit(f"REFUSING: {url.split('?')[0]} kept failing ({last}); nothing is written")


def model(repo: str) -> dict[str, Any]:
    out: dict[str, Any] = {"repo": repo}
    r = get(f"{HUB}/api/models/{repo}", use_token=False)
    if r.status_code in (401, 403):
        r = get(f"{HUB}/api/models/{repo}", use_token=True)
    out["metadata_http_status"] = r.status_code
    if r.status_code != 200:
        out["exists"] = r.status_code != 404 and None
        return out
    meta = r.json()
    out.update({"exists": True, "sha": meta.get("sha"), "gated": meta.get("gated"), "private": meta.get("private"),
                "last_modified": meta.get("lastModified")})
    if not out["sha"]:
        out["config_error"] = "no commit sha in the metadata"
        return out
    url = f"{HUB}/{repo}/raw/{out['sha']}/config.json"
    gated = bool(out["gated"])
    c = get(url, use_token=gated)
    if c.status_code in (401, 403) and not gated:
        c = get(url, use_token=True)
    out["config_http_status"], out["config_token_sent"] = c.status_code, bool(getattr(c, "token_sent", False))
    if c.status_code in (401, 403) and not out["config_token_sent"]:
        out["config_error"] = f"HTTP {c.status_code}: the repo needs a token and none was available"
        return out
    if c.status_code != 200:
        out["config_error"] = f"HTTP {c.status_code}"
        return out
    out["config_sha256"] = hashlib.sha256(c.content).hexdigest()
    out["config"] = json.loads(c.content.decode("utf-8"))
    return out


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    ca, cb = a.get("config"), b.get("config")
    if ca is None or cb is None:
        return {"pre": a["repo"], "post": b["repo"], "comparable": False,
                "why": "a config is missing: " + "; ".join(f"{m['repo']}: {m.get('config_error') or m.get('metadata_http_status')}"
                                                           for m in (a, b) if m.get("config") is None)}
    missing = [k for k in DROP_FIELDS if k not in ca or k not in cb]
    if missing:
        return {"pre": a["repo"], "post": b["repo"], "comparable": False,
                "why": f"{missing} missing from a config, so the drop rule cannot be applied"}
    keys = sorted(set(ca) | set(cb))
    diffs = {k: {"pre": ca.get(k), "post": cb.get(k)} for k in keys
             if json.dumps(ca.get(k), sort_keys=True) != json.dumps(cb.get(k), sort_keys=True)}
    drop = [k for k in DROP_FIELDS if k in diffs]
    return {"pre": a["repo"], "post": b["repo"], "comparable": True, "n_keys_differing": len(diffs),
            "differences": diffs, "design_fields": {k: {"pre": ca.get(k), "post": cb.get(k), "differs": k in diffs}
                                                     for k in DESIGN_FIELDS},
            "positional_fields_differing": [k for k in POSITIONAL if k in diffs],
            "verdict": f"DROPPED: {drop} differ (a frozen probe cannot read the post model)" if drop else "kept for the next step"}


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Phase 2 config check (metadata and config.json only)")
    ap.add_argument("--out", default=str(FINAL_OUT))
    args = ap.parse_args()
    out = Path(args.out).resolve()
    final = out == FINAL_OUT.resolve()
    if out.exists():
        raise SystemExit(f"REFUSING: {out.name} exists; results are never overwritten")
    if not final and out.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test run (--out) must not write under results/")
    code0 = P.code_block(CODE_FILES)
    models = {}
    for repo in MODELS:
        models[repo] = model(repo)
        m = models[repo]
        print(f"  {repo:<42} exists {m.get('exists')} gated {m.get('gated')} sha {str(m.get('sha'))[:12]} "
              f"config {m.get('config_http_status')} {(m.get('config_sha256') or m.get('config_error') or '')[:12]}", flush=True)
    pairs = [compare(models[a], models[b]) for a, b in PAIRS]
    for p in pairs:
        print(f"  {p['pre']} -> {p['post']}: {p.get('verdict') or p.get('why')}; "
              f"{p.get('n_keys_differing')} keys differ; positional {p.get('positional_fields_differing')}", flush=True)
    phi = {p["pre"]: p.get("positional_fields_differing") for p in pairs if p["post"] == "microsoft/Phi-3.5-mini-instruct"}
    matching = [k for k, v in phi.items() if v == []]
    incomplete = [m for m, v in models.items() if v.get("config") is None] + \
                 [f"{p['pre']} -> {p['post']}" for p in pairs if not p["comparable"]]
    if final and incomplete:
        raise SystemExit(f"REFUSING the final name: not every config was read or comparable: {incomplete}; nothing written")
    if P.code_block(CODE_FILES)["sha256"] != code0["sha256"]:
        raise SystemExit("REFUSING to write: the code changed during the run")
    payload = {**provenance_block("phase2_config_check"), "code": code0,
               "preregistration": "D-281 (2), D-297 (3) (docs/phase2_design.md section 2); Phase 2 not yet approved",
               "scope": "public metadata and config.json only; no weights, no tokenizer files. Still owed by section 2: "
                        "the added-tokens tokenization check (the v0.2 -> v0.3 check), which needs tokenizer files and "
                        "the prompts",
               "token_use": "read from .env only when a repo is gated or refuses its config without one; sent only to "
                            "https://huggingface.co; redirects refused; never written",
               "models": models, "pairs": pairs,
               "phi_pick": {"compared_with": "microsoft/Phi-3.5-mini-instruct", "positional_fields_differing": phi,
                            "matching": matching,
                            "reading": (f"kept by the design's rule: {matching}" if len(matching) == 1 else
                                        "no exact match (or more than one): the pick returns to the author; the "
                                        "differences, rope_scaling included, are in this record's pairs")},
               "dropped_pairs": [f"{p['pre']} -> {p['post']}" for p in pairs if str(p.get("verdict", "")).startswith("DROPPED")],
               "not_comparable": [f"{p['pre']} -> {p['post']}" for p in pairs if not p["comparable"]]}
    digest = write_results(out, payload, exclusive=True)
    print(f"phi matching positional settings: {matching}\nWritten: {out.name}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
