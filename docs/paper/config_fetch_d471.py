"""Config deltas of the 18 update pairs (the author's item 4 of 2026-10-16g; registered in D-470 (9dc5f00) before any
value was read; the record is D-471). DESCRIPTIVE: no verdict.

    python docs/paper/config_fetch_d471.py           # fetch, then write docs/paper/config_deltas_d471.json
    python docs/paper/config_fetch_d471.py --offline # rebuild the record from the saved files (no network)

PAIRS. The 18 pairs 1a-1h and 2a-2j of src/pairs_table_d468.py (imported, not edited); a checkpoint is
(config.pre_repo, config.pre_rev) or (config.post_repo, config.post_rev) of the first record of each pair ID.

FIELDS (D-470; no other field is reported): rope_theta, rope_scaling, sliding_window (use_sliding_window beside, where
the file has it), max_position_embeddings, hidden_size, num_hidden_layers, vocab_size. A key is read at the top level;
if absent there, from text_config; rope_theta also from rope_parameters.rope_theta; else ABSENT.

SOURCES, in order. (S1) https://huggingface.co/<repo>/resolve/<full revision>/config.json, anonymous (no token is
sent: none is on the workstation, and this script never reads one); a redirect is followed only to a huggingface.co
host; the bytes go to docs/paper/configs_d471/ with their sha256. (S2) if S1 fails: results/phase2_config_check.json
`models`, where its `sha` equals the pinned revision and it holds the config and its sha256 (results/
phase0_checkpoint_audit.json holds configs but no config sha256, so it is not a source). (S3) otherwise NOT FOUND,
with the HTTP status. Where S1 and S2 both exist, the two sha256 are compared.

DELTA RULE. A field changes if its pre and post values differ as JSON values, ABSENT and null both read as 'none';
rope_scaling compares the whole object (its type is rope_type, else type); the sliding window changes if
sliding_window or use_sliding_window differs. A pair with a NOT FOUND side has NOT FOUND deltas.

This is the one network use of the WRITE session (the author's item 4): config.json at the pinned revisions only.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import pairs_table_d468 as PT  # noqa: E402

RAW = ROOT / "docs" / "paper" / "configs_d471"
OUT = ROOT / "docs" / "paper" / "config_deltas_d471.json"
P2CHECK = ROOT / "results" / "phase2_config_check.json"
FIELDS = ["rope_theta", "rope_scaling", "sliding_window", "max_position_embeddings", "hidden_size",
          "num_hidden_layers", "vocab_size"]
BESIDE = ["use_sliding_window"]
ABSENT = "ABSENT"
HOST = "huggingface.co"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def raw_path(repo: str, rev: str) -> Path:
    return RAW / f"{repo.replace('/', '__')}@{rev}.json"


def get(url: str, hops: int = 5) -> tuple[int, bytes, str]:
    """GET without a token; redirects followed only to huggingface.co hosts."""
    for _ in range(hops):
        u = urlparse(url)
        if u.scheme != "https" or not (u.hostname == HOST or (u.hostname or "").endswith("." + HOST)):
            return -1, b"", f"refused redirect to {u.hostname}"
        conn = http.client.HTTPSConnection(u.hostname, timeout=30)
        try:
            conn.request("GET", u.path + (f"?{u.query}" if u.query else ""), headers={"User-Agent": "gccf-d471"})
            r = conn.getresponse()
            body = r.read()
            if r.status in (301, 302, 303, 307, 308):
                url = urljoin(url, r.getheader("Location") or "")
                continue
            return r.status, body, url
        except OSError as e:
            return -2, b"", f"{type(e).__name__}: {e}"
        finally:
            conn.close()
    return -3, b"", "too many redirects"


def field(cfg: dict, key: str):
    if key in cfg:
        return {"value": cfg[key], "at": "top"}
    tc = cfg.get("text_config")
    if isinstance(tc, dict) and key in tc:
        return {"value": tc[key], "at": "text_config"}
    rp = cfg.get("rope_parameters")
    if key == "rope_theta" and isinstance(rp, dict) and "rope_theta" in rp:
        return {"value": rp["rope_theta"], "at": "rope_parameters"}
    return {"value": ABSENT, "at": None}


def none(v):
    return None if v == ABSENT else v


def rope_type(v):
    v = none(v)
    if v is None:
        return "none"
    if isinstance(v, dict):
        return v.get("rope_type", v.get("type", "?"))
    return "?"


def checkpoints() -> tuple[list[dict], dict[str, str]]:
    rows, shas = PT.rows()
    pairs, seen = [], set()
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        pairs.append({"id": r["id"], "record": r["record"], "phase": r["phase"], "pre": list(r["pre"]),
                      "post": list(r["post"]), "headline": r["headline"]})
    return pairs, shas


def s2(p2: dict, repo: str, rev: str):
    m = (p2.get("models") or {}).get(repo)
    if not m or m.get("sha") != rev or not isinstance(m.get("config"), dict) or not m.get("config_sha256"):
        return None
    return {"config": m["config"], "sha256": m["config_sha256"], "http_status": m.get("config_http_status")}


def main(argv: list[str]) -> int:
    offline = "--offline" in argv
    pairs, src_shas = checkpoints()
    assert len(pairs) == 18, len(pairs)
    p2b = P2CHECK.read_bytes()
    p2 = json.loads(p2b.decode("utf-8"))
    RAW.mkdir(parents=True, exist_ok=True)
    cks: dict[str, dict] = {}
    for p in pairs:
        for side in ("pre", "post"):
            repo, rev = p[side]
            key = f"{repo}@{rev}"
            if key in cks:
                continue
            url = f"https://{HOST}/{repo}/resolve/{rev}/config.json"
            ent = {"repo": repo, "revision": rev, "url": url}
            if not rev or len(rev) != 40:
                ent.update(source="NOT FOUND", reason="no full revision in the record")
                cks[key] = ent
                continue
            path = raw_path(repo, rev)
            if offline:
                status, body, final = (200, path.read_bytes(), "saved file") if path.is_file() else (None, b"", "")
                ent["s1_http_status"] = "offline: " + ("saved file" if path.is_file() else "no saved file")
            else:
                status, body, final = get(url)
                ent["s1_http_status"] = status
                ent["s1_final_url"] = final if status == 200 else None
                ent["s1_note"] = None if status == 200 else final
                ent["fetched_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                if status == 200:
                    path.write_bytes(body)
            alt = s2(p2, repo, rev)
            cfg = None
            if status == 200:
                cfg = json.loads(body.decode("utf-8"))
                ent.update(source="S1", file=path.relative_to(ROOT).as_posix(), sha256=sha(body), bytes=len(body))
                if alt:
                    ent["s2_sha256"] = alt["sha256"]
                    ent["s1_s2_sha256_equal"] = alt["sha256"] == ent["sha256"]
            elif alt:
                cfg = alt["config"]
                ent.update(source="S2", file="results/phase2_config_check.json", sha256=alt["sha256"],
                           s2_http_status=alt["http_status"])
            else:
                ent.update(source="NOT FOUND")
            if cfg is not None:
                ent["fields"] = {k: field(cfg, k) for k in FIELDS + BESIDE}
            cks[key] = ent
            time.sleep(0.2)
    out_pairs = []
    for p in pairs:
        a, b = cks[f"{p['pre'][0]}@{p['pre'][1]}"], cks[f"{p['post'][0]}@{p['post'][1]}"]
        ent = {**p, "pre_source": a["source"], "post_source": b["source"]}
        if "fields" not in a or "fields" not in b:
            ent["deltas"] = "NOT FOUND"
        else:
            fa = {k: none(a["fields"][k]["value"]) for k in FIELDS + BESIDE}
            fb = {k: none(b["fields"][k]["value"]) for k in FIELDS + BESIDE}
            d = {}
            for k in FIELDS:
                if k == "sliding_window":
                    if fa[k] != fb[k] or fa["use_sliding_window"] != fb["use_sliding_window"]:
                        d[k] = {"pre": fa[k], "post": fb[k], "use_sliding_window_pre": fa["use_sliding_window"],
                                "use_sliding_window_post": fb["use_sliding_window"]}
                elif fa[k] != fb[k]:
                    d[k] = {"pre": fa[k], "post": fb[k]}
                    if k == "rope_scaling":
                        d[k].update(type_pre=rope_type(fa[k]), type_post=rope_type(fb[k]))
            ent["deltas"] = d
        out_pairs.append(ent)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    rec = {"record": "D-471", "preregistration": "D-470 (9dc5f00)", "status": "DESCRIPTIVE",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_head": head, "offline": offline,
           "code_sha256": {"docs/paper/config_fetch_d471.py": sha(Path(__file__).read_bytes()),
                           "src/pairs_table_d468.py": sha((ROOT / "src" / "pairs_table_d468.py").read_bytes())},
           "pair_source_records_sha256": src_shas, "s2_record_sha256": {"results/phase2_config_check.json": sha(p2b)},
           "fields": FIELDS, "beside": BESIDE, "token_sent": False,
           "checkpoints": cks, "pairs": out_pairs}
    OUT.write_text(json.dumps(rec, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    n = {s: sum(1 for c in cks.values() if c["source"] == s) for s in ("S1", "S2", "NOT FOUND")}
    print(f"{len(cks)} checkpoints: {n}; {sum(1 for p in out_pairs if p['deltas'] == 'NOT FOUND')} pairs NOT FOUND;"
          f" wrote {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
