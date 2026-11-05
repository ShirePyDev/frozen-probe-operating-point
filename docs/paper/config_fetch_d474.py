"""Complete docs/paper/config_deltas_d471.json: the gated checkpoints D-471 left NOT FOUND (the author's item 3 of
2026-10-16h; the procedure pre-registered in D-474 (f42d892); D-470's fields, sources and delta rule unchanged).

    python docs/paper/config_fetch_d474.py            # fetch the NOT FOUND checkpoints with HF_TOKEN; complete the record
    python docs/paper/config_fetch_d474.py --dry-run  # list what would be fetched; no request, no write

A new script (F-092): docs/paper/config_fetch_d471.py, whose sha256 the record pins, is imported unchanged.

TOKEN. HF_TOKEN is read from the process environment only (no .env file, no other source). It is never printed and
never written; it is sent in an Authorization header to the host huggingface.co only. A redirect is followed only to an
https huggingface.co host, and the header is dropped for any host but huggingface.co itself. Without HF_TOKEN no
request is made: every checkpoint stays NOT FOUND with that reason, and nothing is written.

COMPLETION. For each checkpoint the record lists as NOT FOUND (with a full revision): S1 at its recorded revision. On
HTTP 200 the bytes go to docs/paper/configs_d471/ (D-471's naming), and the entry gets source 'S1', its file, sha256,
bytes, status, fields and token_sent True. Otherwise it stays NOT FOUND, with the new status and reason beside D-471's.
Each pair's deltas are recomputed by deltas(), D-471's rule (config_fetch_d471.main inlines it, so it is copied here;
tests/test_config_fetch_d474.py checks that it reproduces the 14 pairs D-471 found). The record is rewritten in place
(the author's 'complete') only if a checkpoint was completed: every D-471 key kept, plus a 'completion_d474' block
(this script's and config_fetch_d471.py's sha256, the prior record's sha256, the checkpoints tried and completed, the
host the token went to). LF line ends.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs" / "paper"))
import config_fetch_d471 as D471  # noqa: E402

REC = D471.OUT
RAW = D471.RAW
HOST = D471.HOST
TOKEN_HOST = "huggingface.co"
NO_TOKEN = "HF_TOKEN is not in the environment; no request made"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def get(url: str, token: str | None, hops: int = 5) -> tuple[int, bytes, str]:
    """GET; the token goes in an Authorization header to the host huggingface.co only; redirects followed only to
    https huggingface.co hosts. Returns (status, body, final URL or a note); no header or token is ever returned."""
    for _ in range(hops):
        u = urlparse(url)
        host = u.hostname or ""
        if u.scheme != "https" or not (host == HOST or host.endswith("." + HOST)):
            return -1, b"", f"refused redirect to {host or '?'}"
        headers = {"User-Agent": "gccf-d474"}
        if token and host == TOKEN_HOST:
            headers["Authorization"] = "Bearer " + token
        conn = http.client.HTTPSConnection(host, timeout=30)
        try:
            conn.request("GET", u.path + (f"?{u.query}" if u.query else ""), headers=headers)
            r = conn.getresponse()
            body = r.read()
            if r.status in (301, 302, 303, 307, 308):
                url = urljoin(url, r.getheader("Location") or "")
                continue
            return r.status, body, url
        except (OSError, ValueError, http.client.HTTPException) as e:   # never str(e): it can hold a header value
            return -2, b"", type(e).__name__
        finally:
            conn.close()
    return -3, b"", "too many redirects"


def bare(url: str) -> str:
    """scheme://host/path only: a signed redirect URL's query string never reaches the record or the screen."""
    u = urlparse(url)
    return f"{u.scheme}://{u.hostname}{u.path}" if u.scheme and u.hostname else url


def deltas(a: dict, b: dict) -> dict | str:
    """D-471's delta rule, copied from config_fetch_d471.main: a and b are checkpoint entries."""
    if "fields" not in a or "fields" not in b:
        return "NOT FOUND"
    keys = D471.FIELDS + D471.BESIDE
    fa = {k: D471.none(a["fields"][k]["value"]) for k in keys}
    fb = {k: D471.none(b["fields"][k]["value"]) for k in keys}
    d: dict = {}
    for k in D471.FIELDS:
        if k == "sliding_window":
            if fa[k] != fb[k] or fa["use_sliding_window"] != fb["use_sliding_window"]:
                d[k] = {"pre": fa[k], "post": fb[k], "use_sliding_window_pre": fa["use_sliding_window"],
                        "use_sliding_window_post": fb["use_sliding_window"]}
        elif fa[k] != fb[k]:
            d[k] = {"pre": fa[k], "post": fb[k]}
            if k == "rope_scaling":
                d[k].update(type_pre=D471.rope_type(fa[k]), type_post=D471.rope_type(fb[k]))
    return d


def todo(rec: dict) -> list[str]:
    return [k for k, c in rec["checkpoints"].items()
            if c.get("source") == "NOT FOUND" and len(c.get("revision") or "") == 40]


def main(argv: list[str]) -> int:
    prior = REC.read_bytes()
    rec = json.loads(prior.decode("utf-8"))
    keys = todo(rec)
    print(f"{len(keys)} checkpoint(s) NOT FOUND in {REC.relative_to(ROOT).as_posix()}: "
          + ", ".join(f"{rec['checkpoints'][k]['repo']}@{rec['checkpoints'][k]['revision'][:7]}" for k in keys))
    if "--dry-run" in argv:
        return 0
    token = (os.environ.get("HF_TOKEN") or "").strip() or None      # a stray newline would make an invalid header
    if token is None:
        for k in keys:
            print(f"  {k.split('@')[0]}: NOT FOUND ({NO_TOKEN})")
        print("nothing written")
        return 2
    done, tried = [], {}
    for k in keys:
        c = rec["checkpoints"][k]
        url = f"https://{HOST}/{c['repo']}/resolve/{c['revision']}/config.json"
        status, body, final = get(url, token)
        final = bare(final)
        tried[k] = {"http_status": status, "note": None if status == 200 else final,
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if status == 200:
            cfg = json.loads(body.decode("utf-8"))
            path = D471.raw_path(c["repo"], c["revision"])
            RAW.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            c.update(source="S1", file=path.relative_to(ROOT).as_posix(), sha256=sha(body), bytes=len(body),
                     d474_http_status=status, d474_fetched_utc=tried[k]["utc"], token_sent=True,
                     fields={f: D471.field(cfg, f) for f in D471.FIELDS + D471.BESIDE})
            done.append(k)
        else:
            c.update(d474_http_status=status, d474_note=final)
        print(f"  {c['repo']}@{c['revision'][:7]}: HTTP {status}" + (f" sha256 {c['sha256'][:16]}" if status == 200
                                                                          else f" ({final}); stays NOT FOUND"))
        time.sleep(0.2)
    if not done:
        print("no checkpoint completed; nothing written")
        return 1
    for p in rec["pairs"]:
        a = rec["checkpoints"][f"{p['pre'][0]}@{p['pre'][1]}"]
        b = rec["checkpoints"][f"{p['post'][0]}@{p['post'][1]}"]
        p["pre_source"], p["post_source"], p["deltas"] = a["source"], b["source"], deltas(a, b)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout.strip()
    rec["completion_d474"] = {
        "record": "D-475 (3)", "preregistration": "D-474 (f42d892)", "git_head": head,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_sha256": {"docs/paper/config_fetch_d474.py": sha(Path(__file__).read_bytes()),
                        "docs/paper/config_fetch_d471.py": sha(Path(D471.__file__).read_bytes())},
        "prior_record_sha256": sha(prior), "tried": tried, "completed": done,
        "token_sent_to": TOKEN_HOST, "token_source": "HF_TOKEN, process environment"}
    text = json.dumps(rec, indent=1, ensure_ascii=True) + "\n"
    assert token not in text
    REC.write_bytes(text.encode("utf-8"))
    n = sum(1 for p in rec["pairs"] if p["deltas"] == "NOT FOUND")
    print(f"completed {len(done)} of {len(keys)}; {n} pair(s) still NOT FOUND; wrote {REC.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
