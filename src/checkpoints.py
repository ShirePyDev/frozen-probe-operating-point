"""Phase 0 - checkpoint feasibility audit.

For every candidate family in CLAUDE.md, verify against the live HuggingFace API
which checkpoints actually exist at <=13B, their exact parameter counts, their
architectural shape, and their repo creation dates. Then classify each consecutive
version pair as CLEAN / CAVEATED / UNUSABLE.

Deliverable: results/phase0_checkpoint_audit.json plus a printed summary ending in
the CLEAN pair count. This script STOPS at the count. It does not proceed.

Rule 2 discipline: every field is either read from the API or set to None with an
explicit error string. No parameter count is estimated from config arithmetic, no
release date is inferred, no existence is assumed.

Rule 4 discipline: the classification thresholds are encoded as named constants at
the top of this file so a human can audit exactly what "same parameter count" means.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import (  # noqa: E402
    canonical_hash,
    ensure_utf8_stdout,
    provenance_block,
    write_results,
)

HF_API = "https://huggingface.co/api/models"
HF_RESOLVE = "https://huggingface.co"
USER_AGENT = "guardrail-ccf-phase0/1.0 (checkpoint feasibility audit)"
TIMEOUT = 30
RETRIES = 3
RETRY_BACKOFF_S = 2.0

# --- classification thresholds (D-010) -------------------------------------
# "Same parameter count" is not bit-identical: a vocabulary change alters the
# embedding and output matrices by a fraction of a percent without changing model
# capacity in any way that matters for a capacity confound. This is the tolerance.
PARAM_REL_TOL = 0.01  # 1% relative difference in total parameters

# Size ceiling from CLAUDE.md Phase 0: "checkpoints ... at <=13B".
MAX_PARAMS = 13_000_000_000

# Config fields that are compared across a pair. Differences in the first group
# are disqualifying for CLEAN (capacity / depth confounds). Differences in the
# second group are recorded as notes but do not block CLEAN.
SHAPE_FIELDS = ("hidden_size", "num_hidden_layers")
NOTE_FIELDS = (
    "vocab_size",
    "max_position_embeddings",
    "rope_theta",
    "sliding_window",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "torch_dtype",
    "model_type",
)


# --- candidate table --------------------------------------------------------
# Each entry lists repo-id HYPOTHESES in priority order. The auditor tries each in
# turn and records every attempt, including the 404s. A 404 is evidence, not an
# error: CLAUDE.md says do not assume any of these exist.
#
# Both variants are audited per D-011: L1 (probe on frozen hidden states) needs the
# base checkpoint, L2 (zero-shot judge) needs the instruct checkpoint. A pair is
# only fully usable if all four repos resolve.

CANDIDATES: list[dict[str, Any]] = [
    {
        "family": "mistral-7b",
        "version": "v0.1",
        "order": 1,
        "base": ["mistralai/Mistral-7B-v0.1"],
        "instruct": ["mistralai/Mistral-7B-Instruct-v0.1"],
    },
    {
        "family": "mistral-7b",
        "version": "v0.2",
        "order": 2,
        "base": [
            "mistralai/Mistral-7B-v0.2",
            "mistral-community/Mistral-7B-v0.2",
            "alpindale/Mistral-7B-v0.2-hf",
        ],
        "instruct": ["mistralai/Mistral-7B-Instruct-v0.2"],
    },
    {
        "family": "mistral-7b",
        "version": "v0.3",
        "order": 3,
        "base": ["mistralai/Mistral-7B-v0.3"],
        "instruct": ["mistralai/Mistral-7B-Instruct-v0.3"],
    },
    {
        "family": "llama-3-8b",
        "version": "3.0",
        "order": 1,
        "base": ["meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-3-8B"],
        "instruct": [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Llama-3-8B-Instruct",
        ],
    },
    {
        "family": "llama-3-8b",
        "version": "3.1",
        "order": 2,
        "base": ["meta-llama/Llama-3.1-8B", "meta-llama/Meta-Llama-3.1-8B"],
        "instruct": [
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
        ],
    },
    {
        "family": "phi-3-mini",
        "version": "3.0",
        "order": 1,
        # Microsoft did not obviously ship a separate base checkpoint for Phi-3-mini.
        # These hypotheses are expected to 404; the 404 is the recorded finding.
        "base": ["microsoft/Phi-3-mini-4k-base", "microsoft/Phi-3-mini-4k"],
        "instruct": [
            "microsoft/Phi-3-mini-4k-instruct",
            "microsoft/Phi-3-mini-128k-instruct",
        ],
    },
    {
        "family": "phi-3-mini",
        "version": "3.5",
        "order": 2,
        "base": ["microsoft/Phi-3.5-mini-base", "microsoft/Phi-3.5-mini"],
        "instruct": ["microsoft/Phi-3.5-mini-instruct"],
    },
    {
        "family": "olmo-7b",
        "version": "2",
        "order": 1,
        "base": ["allenai/OLMo-2-1124-7B", "allenai/Olmo-2-1124-7B"],
        "instruct": ["allenai/OLMo-2-1124-7B-Instruct", "allenai/Olmo-2-1124-7B-Instruct"],
    },
    {
        "family": "olmo-7b",
        "version": "3",
        "order": 2,
        "base": ["allenai/Olmo-3-7B", "allenai/OLMo-3-7B", "allenai/Olmo-3-1025-7B"],
        "instruct": [
            "allenai/Olmo-3-7B-Instruct",
            "allenai/OLMo-3-7B-Instruct",
            "allenai/Olmo-3-1025-7B-Instruct",
        ],
    },
    {
        "family": "qwen-7b8b",
        "version": "2.5",
        "order": 1,
        "base": ["Qwen/Qwen2.5-7B"],
        "instruct": ["Qwen/Qwen2.5-7B-Instruct"],
    },
    {
        "family": "qwen-7b8b",
        "version": "3",
        "order": 2,
        "base": ["Qwen/Qwen3-8B-Base"],
        "instruct": ["Qwen/Qwen3-8B"],
    },
    {
        "family": "gemma",
        "version": "2-9b",
        "order": 1,
        "base": ["google/gemma-2-9b"],
        "instruct": ["google/gemma-2-9b-it"],
    },
    {
        "family": "gemma",
        "version": "3-12b",
        "order": 2,
        "base": ["google/gemma-3-12b-pt"],
        "instruct": ["google/gemma-3-12b-it"],
    },
]


def make_session(token: str | None) -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    if token:
        sess.headers.update({"Authorization": f"Bearer {token}"})
    return sess


def _get(sess: requests.Session, url: str) -> tuple[int | None, Any, str | None]:
    """GET with retries. Returns (status_code, parsed_or_text, error_string)."""
    last_err = None
    for attempt in range(RETRIES):
        try:
            resp = sess.get(url, timeout=TIMEOUT)
        except requests.RequestException as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(RETRY_BACKOFF_S * (attempt + 1))
            continue
        if resp.status_code >= 500 and attempt < RETRIES - 1:
            last_err = f"HTTP {resp.status_code}"
            time.sleep(RETRY_BACKOFF_S * (attempt + 1))
            continue
        try:
            return resp.status_code, resp.json(), None
        except ValueError:
            return resp.status_code, resp.text, None
    return None, None, last_err


def probe_repo(sess: requests.Session, repo_id: str) -> dict[str, Any]:
    """Query one repo. Records existence, revision SHA, gating, params, config."""
    rec: dict[str, Any] = {
        "repo_id": repo_id,
        "exists": None,
        "existence_ambiguous": False,
        "http_status": None,
        "revision_sha": None,
        "gated": None,
        "access_verified": False,
        "hf_created_at": None,
        "hf_last_modified": None,
        "params_total": None,
        "params_by_dtype": None,
        "params_source": None,
        "config": None,
        "config_source": None,
        "errors": [],
    }

    status, body, err = _get(sess, f"{HF_API}/{repo_id}")
    rec["http_status"] = status
    if err is not None:
        rec["errors"].append(f"api: {err}")
        return rec
    if status == 404:
        rec["exists"] = False
        return rec
    if status in (401, 403):
        # HuggingFace returns 401 for a repo that is absent OR private, so as not
        # to leak the existence of private repos. Unauthenticated, these two cases
        # are INDISTINGUISHABLE. Recording this as "does not exist" would be an
        # invented fact (rule 2), so it stays explicitly ambiguous.
        rec["exists"] = None
        rec["existence_ambiguous"] = True
        rec["errors"].append(
            f"api: HTTP {status} - absent or private, indistinguishable without HF_TOKEN"
        )
        return rec
    if status != 200 or not isinstance(body, dict):
        rec["exists"] = None
        rec["errors"].append(f"api: unexpected status {status}")
        return rec

    rec["exists"] = True
    rec["revision_sha"] = body.get("sha")
    rec["gated"] = body.get("gated", False)
    rec["hf_created_at"] = body.get("createdAt")
    rec["hf_last_modified"] = body.get("lastModified")
    rec["pipeline_tag"] = body.get("pipeline_tag")

    safet = body.get("safetensors")
    if isinstance(safet, dict) and safet.get("total"):
        rec["params_total"] = safet["total"]
        rec["params_by_dtype"] = safet.get("parameters")
        rec["params_source"] = "hf_api.safetensors.total"
    else:
        rec["errors"].append("params: no safetensors metadata on API response")

    # config.json via the resolve endpoint, pinned to the revision SHA we just read.
    rev = rec["revision_sha"] or "main"
    cfg_status, cfg_body, cfg_err = _get(
        sess, f"{HF_RESOLVE}/{repo_id}/resolve/{rev}/config.json"
    )
    if cfg_err is not None:
        rec["errors"].append(f"config: {cfg_err}")
    elif cfg_status == 200 and isinstance(cfg_body, dict):
        rec["config"] = cfg_body
        rec["config_source"] = f"resolve@{rev[:12] if rec['revision_sha'] else 'main'}"
        rec["access_verified"] = True
    elif cfg_status in (401, 403):
        rec["errors"].append(
            f"config: HTTP {cfg_status} - gated, needs accepted licence + HF_TOKEN"
        )
    else:
        rec["errors"].append(f"config: HTTP {cfg_status}")

    return rec


def flatten_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Pull shape fields out of a config, descending into text_config if present.

    Multimodal releases (e.g. Gemma 3) nest the language-model shape under
    `text_config`. Reading only the top level would silently compare a vision
    wrapper against a plain decoder.
    """
    if not isinstance(cfg, dict):
        return {}
    inner = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    out: dict[str, Any] = {}
    for field in SHAPE_FIELDS + NOTE_FIELDS:
        val = cfg.get(field)
        if val is None and inner:
            val = inner.get(field)
        out[field] = val
    out["architectures"] = cfg.get("architectures")
    out["nested_text_config"] = bool(inner)
    return out


def classify_pair(
    older: dict[str, Any], newer: dict[str, Any], family: str
) -> dict[str, Any]:
    """Classify one consecutive version pair into the three CLAUDE.md buckets.

    Ordering of checks matters and is deliberate:
      1. missing repos          -> UNUSABLE (nothing to compare)
      2. unverifiable config    -> UNUSABLE, blocked_on=hf_token (we refuse to
                                   call a pair CLEAN on evidence we do not have)
      3. wholesale redesign     -> UNUSABLE
      4. capacity/depth change  -> CAVEATED
      5. otherwise              -> CLEAN, with non-disqualifying diffs as notes
    """
    res: dict[str, Any] = {
        "family": family,
        "from_version": older["version"],
        "to_version": newer["version"],
        "classification": None,
        "reason": None,
        "blocked_on": None,
        "notes": [],
        "resolved": {},
        "shape_diff": {},
    }

    missing: list[str] = []
    for side, entry in (("from", older), ("to", newer)):
        for variant in ("base", "instruct"):
            resolved = entry["resolved"].get(variant)
            res["resolved"][f"{side}_{variant}"] = (
                resolved["repo_id"] if resolved else None
            )
            if resolved is None:
                missing.append(f"{side}.{variant}")

    if missing:
        res["classification"] = "UNUSABLE"
        res["reason"] = f"missing checkpoint(s): {', '.join(missing)}"
        return res

    # The probe (L1) runs on the base checkpoints, so the base pair defines the
    # architectural comparison. The instruct pair is checked for gating only.
    a = older["resolved"]["base"]
    b = newer["resolved"]["base"]

    # Provenance warnings propagate into the pair, because a pair whose two sides
    # come from different orgs is not a clean version increment regardless of how
    # well the parameter counts match.
    for side, entry in (("from", older), ("to", newer)):
        for variant in ("base", "instruct"):
            r = entry["resolved"].get(variant)
            if r and r.get("provenance_class") == "third_party":
                res["notes"].append(
                    f"PROVENANCE: {side}.{variant} is a third-party re-upload "
                    f"({r['repo_id']})"
                )
            for w in entry.get("warnings", {}).get(variant, []):
                if w.startswith("higher-priority"):
                    res["notes"].append(f"{side}.{variant}: {w}")

    ca, cb = flatten_config(a["config"]), flatten_config(b["config"])

    if not ca or not cb:
        blocked = [x["repo_id"] for x in (a, b) if not flatten_config(x["config"])]
        res["classification"] = "UNUSABLE"
        res["reason"] = f"config unreadable for {', '.join(blocked)}"
        res["blocked_on"] = "hf_token"
        return res

    for field in SHAPE_FIELDS + NOTE_FIELDS:
        if ca.get(field) != cb.get(field):
            res["shape_diff"][field] = {"from": ca.get(field), "to": cb.get(field)}

    pa, pb = a["params_total"], b["params_total"]
    if pa is None or pb is None:
        res["classification"] = "UNUSABLE"
        res["reason"] = "parameter count unavailable on at least one side"
        res["blocked_on"] = "hf_token" if (a["gated"] or b["gated"]) else "hf_api"
        return res

    res["params"] = {"from": pa, "to": pb}
    res["params_rel_delta"] = abs(pa - pb) / max(pa, pb)

    oversize = [r["repo_id"] for r, p in ((a, pa), (b, pb)) if p > MAX_PARAMS]
    if oversize:
        res["notes"].append(f"exceeds 13B ceiling: {', '.join(oversize)}")

    shape_changed = [f for f in SHAPE_FIELDS if ca.get(f) != cb.get(f)]
    size_changed = res["params_rel_delta"] > PARAM_REL_TOL

    # Wholesale redesign: depth AND width AND declared model type all moved.
    if shape_changed == list(SHAPE_FIELDS) and ca.get("model_type") != cb.get("model_type"):
        res["classification"] = "UNUSABLE"
        res["reason"] = (
            "architecture change too large: hidden_size, num_hidden_layers and "
            "model_type all differ"
        )
        return res

    if size_changed or shape_changed:
        res["classification"] = "CAVEATED"
        bits = []
        if size_changed:
            bits.append(f"param delta {res['params_rel_delta']:.2%} > {PARAM_REL_TOL:.0%}")
        if shape_changed:
            bits.append("shape change: " + ", ".join(shape_changed))
        res["reason"] = "same family, different size - " + "; ".join(bits)
        return res

    res["classification"] = "CLEAN"
    res["reason"] = (
        f"same family, param delta {res['params_rel_delta']:.3%} within "
        f"{PARAM_REL_TOL:.0%}, identical hidden_size and num_hidden_layers"
    )
    for field in NOTE_FIELDS:
        if field in res["shape_diff"]:
            d = res["shape_diff"][field]
            res["notes"].append(f"{field}: {d['from']} -> {d['to']}")
    if a["gated"] or b["gated"]:
        res["notes"].append("gated repo(s) - access required before Phase 1")
    return res


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Phase 0 checkpoint feasibility audit")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "results" / "phase0_checkpoint_audit.json"),
    )
    args = ap.parse_args()

    try:
        from huggingface_hub import get_token

        token = get_token()
    except Exception:
        token = None

    sess = make_session(token)
    print(f"Phase 0 checkpoint audit - HF token present: {bool(token)}")
    print(f"Candidates: {len(CANDIDATES)} checkpoint versions across "
          f"{len(set(c['family'] for c in CANDIDATES))} families\n")

    audited: list[dict[str, Any]] = []
    for cand in CANDIDATES:
        entry = {
            "family": cand["family"],
            "version": cand["version"],
            "order": cand["order"],
            "attempts": {},
            "resolved": {},
        }
        for variant in ("base", "instruct"):
            # Probe EVERY hypothesis rather than stopping at the first hit. Two
            # benefits: we learn whether a fallback exists at all, and when the
            # fallbacks are independent third-party conversions of the same
            # weights we can cross-check them against each other, which is the
            # only corroboration available when the official repo is unreachable.
            probed = [probe_repo(sess, repo_id) for repo_id in cand[variant]]
            attempts = [
                {
                    "repo_id": r["repo_id"],
                    "exists": r["exists"],
                    "existence_ambiguous": r["existence_ambiguous"],
                    "http_status": r["http_status"],
                    "params_total": r["params_total"],
                    "access_verified": r["access_verified"],
                }
                for r in probed
            ]
            skipped_ambiguous = [r["repo_id"] for r in probed if r["existence_ambiguous"]]
            existing = [r for r in probed if r["exists"]]

            # Selection priority. Gating is resolvable with a token; third-party
            # provenance is NOT resolvable, so an official-but-gated repo always
            # outranks a readable third-party one.
            preferred = cand[variant][0] if cand[variant] else None
            pref_org = preferred.split("/")[0] if preferred else None
            resolved = None
            for pick in (
                lambda r: r["repo_id"] == preferred,
                lambda r: r["repo_id"].split("/")[0] == pref_org,
                lambda r: r["access_verified"],
                lambda r: True,
            ):
                match = [r for r in existing if pick(r)]
                if match:
                    resolved = match[0]
                    break

            # Cross-check independent conversions: if more than one candidate
            # resolved, do they agree on parameter count and core shape?
            # Dedupe by revision SHA first: HuggingFace resolves repo names
            # case-insensitively, so "allenai/OLMo-2-1124-7B" and
            # "allenai/Olmo-2-1124-7B" are ONE repo. Counting them as two
            # independent sources would manufacture corroboration that does not
            # exist.
            seen_sha: set[str] = set()
            independent = []
            for r in existing:
                sha = r["revision_sha"]
                if sha and sha in seen_sha:
                    continue
                if sha:
                    seen_sha.add(sha)
                independent.append(r)
            corroboration = None
            if len(independent) > 1:
                existing_for_check = independent
                shapes = {
                    r["repo_id"]: {
                        "params_total": r["params_total"],
                        **{f: flatten_config(r["config"]).get(f) for f in SHAPE_FIELDS},
                    }
                    for r in existing_for_check
                }
                distinct = {canonical_hash(v) for v in shapes.values()}
                corroboration = {
                    "n_existing": len(existing),
                    "n_independent_by_sha": len(independent),
                    "agree": len(distinct) == 1,
                    "shapes": shapes,
                }

            # Provenance: the FIRST listed hypothesis is the preferred/official repo.
            # Falling through to a later one is not neutral - a different org means a
            # third-party re-upload whose weights are not verified against the
            # original release. That is a confound in the update pair, not a detail.
            warnings: list[str] = []
            preferred = cand[variant][0] if cand[variant] else None
            prov_class = None
            if resolved and preferred:
                pref_org = preferred.split("/")[0]
                res_org = resolved["repo_id"].split("/")[0]
                if resolved["repo_id"] == preferred:
                    prov_class = "official_preferred"
                elif res_org == pref_org:
                    prov_class = "same_org_alias"
                else:
                    prov_class = "third_party"
                    warnings.append(
                        f"resolved to third-party org '{res_org}' (preferred was "
                        f"'{preferred}'); weights not verified against original release"
                    )
            if skipped_ambiguous:
                warnings.append(
                    "higher-priority repo(s) returned 401 (absent or private, "
                    "undetermined without HF_TOKEN): " + ", ".join(skipped_ambiguous)
                )
            if resolved:
                resolved["provenance_class"] = prov_class
            if corroboration is not None:
                warnings.append(
                    f"{corroboration['n_independent_by_sha']} independent repos "
                    "(distinct revision SHAs) cross-checked; shapes "
                    + ("AGREE" if corroboration["agree"] else "DISAGREE")
                )
            entry["attempts"][variant] = attempts
            entry["resolved"][variant] = resolved
            entry.setdefault("corroboration", {})[variant] = corroboration
            entry.setdefault("warnings", {})[variant] = warnings

            mark = "OK " if resolved else "MISS"
            rid = resolved["repo_id"] if resolved else " / ".join(cand[variant]) or "(none listed)"
            extra = ""
            if resolved:
                g = resolved["gated"]
                extra = f"  params={resolved['params_total']}  gated={g}"
                if not resolved["access_verified"]:
                    extra += "  [config unreadable]"
                if prov_class == "third_party":
                    extra += "  [THIRD-PARTY]"
            print(f"  [{mark}] {cand['family']:<12} {cand['version']:<6} {variant:<8} {rid}{extra}")
            for w in warnings:
                print(f"         !! {w}")
        audited.append(entry)

    pairs: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = {}
    for entry in audited:
        by_family.setdefault(entry["family"], []).append(entry)
    for family, entries in by_family.items():
        entries.sort(key=lambda e: e["order"])
        for older, newer in zip(entries, entries[1:]):
            pairs.append(classify_pair(older, newer, family))

    clean = [p for p in pairs if p["classification"] == "CLEAN"]
    caveated = [p for p in pairs if p["classification"] == "CAVEATED"]
    unusable = [p for p in pairs if p["classification"] == "UNUSABLE"]
    blocked = [p for p in unusable if p["blocked_on"] == "hf_token"]
    clean_no_notes = [p for p in clean if not p["notes"]]

    payload = {
        **provenance_block("phase0_checkpoint_audit"),
        "hf_token_present": bool(token),
        "thresholds": {
            "param_rel_tol": PARAM_REL_TOL,
            "max_params": MAX_PARAMS,
            "shape_fields": list(SHAPE_FIELDS),
            "note_fields": list(NOTE_FIELDS),
        },
        "candidate_table_hash": canonical_hash(CANDIDATES),
        "summary": {
            "clean_pair_count": len(clean),
            "clean_pairs_without_notes": len(clean_no_notes),
            "caveated_pair_count": len(caveated),
            "unusable_pair_count": len(unusable),
            "unusable_blocked_on_token": len(blocked),
        },
        "pairs": pairs,
        "checkpoints": audited,
    }
    digest = write_results(args.out, payload)

    print("\n" + "=" * 72)
    print("PAIR CLASSIFICATION")
    print("=" * 72)
    for p in pairs:
        tag = p["classification"]
        blk = f"  (blocked_on={p['blocked_on']})" if p["blocked_on"] else ""
        print(f"  {tag:<9} {p['family']:<12} {p['from_version']} -> {p['to_version']}{blk}")
        print(f"            {p['reason']}")
        for note in p["notes"]:
            print(f"            note: {note}")
    print("=" * 72)
    print(f"CLEAN pair count:            {len(clean)}")
    print(f"  of which carry no notes:   {len(clean_no_notes)}")
    print(f"CAVEATED pair count:         {len(caveated)}")
    print(f"UNUSABLE pair count:         {len(unusable)}")
    print(f"  blocked on HF token only:  {len(blocked)}")
    print("=" * 72)
    print(f"\nWritten: {args.out}\nsha256:  {digest}")
    print("\nGO/NO-GO GATE: stopping here. A human must confirm the CLEAN pair count")
    print("is sufficient before Phase 1 begins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
