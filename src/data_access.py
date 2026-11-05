"""Phase 0 - dataset availability and gating audit (D-012).

CLAUDE.md names five datasets. Two of them are gated on HuggingFace. Discovering a
licence wall at the start of Phase 1 costs a cycle, so availability is a feasibility
fact and is audited here, alongside the checkpoints.

For each dataset this records: existence, revision SHA, gating status, licence,
available configs and splits, and per-split row counts - the last of these feeds
directly into the power analysis (D-008), which cannot size the eval partition
without knowing how many items exist to draw from.

Deliverable: results/phase0_dataset_audit.json

Rule 2 discipline: row counts come from the HuggingFace datasets-server, not from
recollection. When the server cannot answer, the field is None with an error string.
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

HF_DATASET_API = "https://huggingface.co/api/datasets"
DATASETS_SERVER = "https://datasets-server.huggingface.co"
USER_AGENT = "guardrail-ccf-phase0/1.0 (dataset feasibility audit)"
TIMEOUT = 45
RETRIES = 3
RETRY_BACKOFF_S = 2.0

# Role maps each dataset onto the axis it serves in the design. The over-refusal
# axis is not optional (CLAUDE.md), so it is labelled explicitly here.
CANDIDATE_DATASETS: list[dict[str, Any]] = [
    {
        "key": "wildguardmix",
        "role": "primary safety axis",
        "phase": 1,
        "candidates": ["allenai/wildguardmix"],
    },
    {
        "key": "xstest",
        "role": "over-refusal / false positive axis",
        "phase": 1,
        "candidates": [
            "walledai/XSTest",
            "natolambert/xstest-v2-copy",
            "Paul/XSTest",
        ],
    },
    {
        "key": "aegis2",
        "role": "second safety axis",
        "phase": 2,
        "candidates": [
            "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
            "nvidia/Aegis-AI-Content-Safety-2.0",
        ],
    },
    {
        "key": "jbb_behaviors",
        "role": "adversarial axis",
        "phase": 2,
        "candidates": ["JailbreakBench/JBB-Behaviors"],
    },
    {
        "key": "orbench",
        "role": "over-refusal / false positive axis",
        "phase": 2,
        "candidates": ["bench-llm/or-bench"],
    },
]


def make_session(token: str | None) -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    if token:
        sess.headers.update({"Authorization": f"Bearer {token}"})
    return sess


def _get(sess: requests.Session, url: str) -> tuple[int | None, Any, str | None]:
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


def probe_dataset(sess: requests.Session, ds_id: str) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "dataset_id": ds_id,
        "exists": None,
        "existence_ambiguous": False,
        "http_status": None,
        "revision_sha": None,
        "gated": None,
        "access_verified": False,
        "license": None,
        "hf_created_at": None,
        "hf_last_modified": None,
        "downloads_30d": None,
        "configs": None,
        "splits": None,
        "row_counts": None,
        "total_rows": None,
        "errors": [],
    }

    status, body, err = _get(sess, f"{HF_DATASET_API}/{ds_id}")
    rec["http_status"] = status
    if err is not None:
        rec["errors"].append(f"api: {err}")
        return rec
    if status == 404:
        rec["exists"] = False
        return rec
    if status in (401, 403):
        rec["exists"] = None
        rec["existence_ambiguous"] = True
        rec["errors"].append(
            f"api: HTTP {status} - absent or private, indistinguishable without HF_TOKEN"
        )
        return rec
    if status != 200 or not isinstance(body, dict):
        rec["errors"].append(f"api: unexpected status {status}")
        return rec

    rec["exists"] = True
    rec["revision_sha"] = body.get("sha")
    rec["gated"] = body.get("gated", False)
    rec["hf_created_at"] = body.get("createdAt")
    rec["hf_last_modified"] = body.get("lastModified")
    rec["downloads_30d"] = body.get("downloads")
    card = body.get("cardData") or {}
    if isinstance(card, dict):
        rec["license"] = card.get("license")

    # Split structure and row counts via the datasets-server. This is the check
    # that actually proves readability: a gated dataset we cannot access returns
    # an error here even though the metadata call above succeeded.
    s_status, s_body, s_err = _get(sess, f"{DATASETS_SERVER}/splits?dataset={ds_id}")
    if s_err is not None:
        rec["errors"].append(f"splits: {s_err}")
    elif s_status == 200 and isinstance(s_body, dict) and "splits" in s_body:
        splits = s_body["splits"]
        rec["configs"] = sorted({s["config"] for s in splits})
        rec["splits"] = [f"{s['config']}/{s['split']}" for s in splits]
        rec["access_verified"] = True
    else:
        detail = ""
        if isinstance(s_body, dict):
            detail = f" - {s_body.get('error') or s_body.get('detail') or ''}".rstrip()
        rec["errors"].append(f"splits: HTTP {s_status}{detail}")

    z_status, z_body, z_err = _get(sess, f"{DATASETS_SERVER}/size?dataset={ds_id}")
    if z_err is not None:
        rec["errors"].append(f"size: {z_err}")
    elif z_status == 200 and isinstance(z_body, dict):
        size = z_body.get("size", {})
        counts = {}
        for entry in size.get("splits", []) or []:
            counts[f"{entry.get('config')}/{entry.get('split')}"] = entry.get("num_rows")
        if counts:
            rec["row_counts"] = counts
            known = [v for v in counts.values() if isinstance(v, int)]
            rec["total_rows"] = sum(known) if known else None
    else:
        rec["errors"].append(f"size: HTTP {z_status}")

    return rec


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Phase 0 dataset feasibility audit")
    ap.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parent.parent / "results" / "phase0_dataset_audit.json"
        ),
    )
    args = ap.parse_args()

    try:
        from huggingface_hub import get_token

        token = get_token()
    except Exception:
        token = None

    sess = make_session(token)
    print(f"Phase 0 dataset audit - HF token present: {bool(token)}")
    print(f"Candidate datasets: {len(CANDIDATE_DATASETS)}\n")

    audited: list[dict[str, Any]] = []
    for cand in CANDIDATE_DATASETS:
        # Probe EVERY candidate, not just up to the first hit. Stopping early hides
        # the fact that a later mirror may be ungated and therefore usable now,
        # while the earlier one needs a licence click. The extra HTTP calls are
        # cheap; an unnecessarily blocked Phase 1 is not.
        probed = [probe_dataset(sess, ds_id) for ds_id in cand["candidates"]]
        attempts = [
            {
                "dataset_id": r["dataset_id"],
                "exists": r["exists"],
                "existence_ambiguous": r["existence_ambiguous"],
                "http_status": r["http_status"],
                "access_verified": r["access_verified"],
                "gated": r["gated"],
            }
            for r in probed
        ]
        readable = [r for r in probed if r["exists"] and r["access_verified"]]
        existing = [r for r in probed if r["exists"]]
        resolved = readable[0] if readable else (existing[0] if existing else None)
        selection_rule = (
            "first_readable" if readable else ("first_existing" if existing else "none")
        )
        alternatives = [
            r["dataset_id"] for r in existing if resolved and r is not resolved
        ]
        entry_extra = {
            "selection_rule": selection_rule,
            "readable_alternatives": [r["dataset_id"] for r in readable],
            "other_existing": alternatives,
        }
        entry = {
            "key": cand["key"],
            "role": cand["role"],
            "needed_by_phase": cand["phase"],
            "attempts": attempts,
            "resolved": resolved,
            **entry_extra,
        }
        audited.append(entry)

        if resolved is None:
            mark = "MISS"
        elif resolved["access_verified"]:
            mark = "OK "
        else:
            mark = "GATE"
        rid = resolved["dataset_id"] if resolved else " / ".join(cand["candidates"])
        print(f"  [{mark}] {cand['key']:<14} phase{cand['phase']}  {rid}")
        if entry_extra["other_existing"]:
            print(f"         alternatives that also exist: {', '.join(entry_extra['other_existing'])}")
        if resolved:
            print(
                f"         gated={resolved['gated']}  licence={resolved['license']}  "
                f"readable={resolved['access_verified']}  rows={resolved['total_rows']}"
            )
            if resolved["configs"]:
                print(f"         configs: {', '.join(resolved['configs'])}")
            if resolved["row_counts"]:
                for k, v in sorted(resolved["row_counts"].items()):
                    print(f"           {k}: {v}")
            for e in resolved["errors"]:
                print(f"         !! {e}")

    readable = [e for e in audited if e["resolved"] and e["resolved"]["access_verified"]]
    blocked = [
        e
        for e in audited
        if e["resolved"] and not e["resolved"]["access_verified"]
    ]
    missing = [e for e in audited if not e["resolved"]]
    p1 = [e for e in audited if e["needed_by_phase"] == 1]
    p1_ready = [e for e in p1 if e["resolved"] and e["resolved"]["access_verified"]]

    payload = {
        **provenance_block("phase0_dataset_audit"),
        "hf_token_present": bool(token),
        "candidate_table_hash": canonical_hash(CANDIDATE_DATASETS),
        "summary": {
            "readable_count": len(readable),
            "blocked_count": len(blocked),
            "missing_count": len(missing),
            "phase1_datasets_total": len(p1),
            "phase1_datasets_readable": len(p1_ready),
        },
        "datasets": audited,
    }
    digest = write_results(args.out, payload)

    print("\n" + "=" * 72)
    print(f"readable (splits enumerated): {len(readable)} / {len(audited)}")
    print(f"blocked (exists, unreadable): {len(blocked)}")
    print(f"unresolved:                   {len(missing)}")
    print(f"Phase 1 datasets readable:    {len(p1_ready)} / {len(p1)}")
    print("=" * 72)
    print(f"\nWritten: {args.out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
