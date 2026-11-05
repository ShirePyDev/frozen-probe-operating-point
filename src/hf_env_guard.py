"""Refuse to run under an unsafe Hugging Face environment.

Why this exists (2026-09-24)
----------------------------
The node's LOGIN shell (zsh) exports HF_ENDPOINT set to a plain-HTTP campus mirror that is
not ours (GCCF_HF_MIRROR in the private site config, D-172), and HF_HOME set to a shared
cache on the NAS (GCCF_SHARED_HF_HOME). A non-login ssh shell exports neither. So a job behaved differently depending on how
it was launched, and one launched from a login shell would (a) route huggingface_hub
traffic - and any implicit credential - to a third-party host in clear text, and (b) put
multi-GB weights on the quota-bound NAS.

scripts/project_env.sh sets the environment explicitly. This module CHECKS it at the
start of every script that fetches or loads weights, so a runner that forgot to source
the env file stops before it touches anything:

  * HF_ENDPOINT, if set, must be an https:// URL;
  * HF_HOME must be set and must resolve to a filesystem whose source is an NVMe device;
  * no Hub token may be present in the environment. fetch_weights.py reads it from .env
    itself, for gated repos only, and sends it only to https://huggingface.co.

Usage:  from hf_env_guard import assert_hf_env; assert_hf_env("extract_hidden.py")
        python3 src/hf_env_guard.py <caller>      # exit 0 if safe, 78 if not
"""

from __future__ import annotations

import os
import sys
import urllib.parse
from pathlib import Path

NVME_SOURCE_PREFIX = "/dev/nvme"
TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")
EXIT_UNSAFE = 78  # EX_CONFIG


def mount_of(path: Path) -> tuple[str, str]:
    """(mount point, source) of the filesystem holding `path`, or its nearest existing parent.

    Read from /proc/self/mounts; the longest matching mount point wins, and among equal
    mount points the later line wins, because a later mount shadows an earlier one.
    """
    p = Path(os.path.realpath(path))
    while not p.exists() and p != p.parent:
        p = p.parent
    target = str(p)
    best_mnt, best_src = "", ""
    with open("/proc/self/mounts", encoding="utf-8") as fh:
        for line in fh:
            fields = line.split()
            if len(fields) < 2:
                continue
            src, mnt = fields[0], fields[1].replace("\\040", " ")
            inside = target == mnt or target.startswith(mnt.rstrip("/") + "/")
            if inside and len(mnt) >= len(best_mnt):
                best_mnt, best_src = mnt, src
    return best_mnt, best_src


def problems() -> list[str]:
    out: list[str] = []
    ep = os.environ.get("HF_ENDPOINT")
    if ep and urllib.parse.urlsplit(ep).scheme != "https":
        out.append(f"HF_ENDPOINT={ep} is not an https URL")
    home = os.environ.get("HF_HOME")
    if not home:
        out.append("HF_HOME is not set")
    else:
        try:
            mnt, src = mount_of(Path(home))
        except OSError as exc:
            mnt, src = "?", f"<cannot read mounts: {type(exc).__name__}>"
        if not src.startswith(NVME_SOURCE_PREFIX):
            out.append(f"HF_HOME={home} is on {src or 'an unknown device'} "
                       f"(mounted at {mnt or '?'}), not on local NVMe")
    for k in TOKEN_VARS:
        if os.environ.get(k):
            out.append(f"{k} is set in the environment; the token is never exported")
    return out


def assert_hf_env(caller: str) -> None:
    bad = problems()
    if not bad:
        return
    print(f"REFUSING to run {caller}: unsafe Hugging Face environment", file=sys.stderr)
    for b in bad:
        print(f"  - {b}", file=sys.stderr)
    print("  fix: source scripts/project_env.sh before launching", file=sys.stderr)
    raise SystemExit(EXIT_UNSAFE)


if __name__ == "__main__":
    assert_hf_env(sys.argv[1] if len(sys.argv) > 1 else "this job")
    home = os.environ["HF_HOME"]
    print(f"hf_env_guard OK: HF_HOME={home} on {mount_of(Path(home))[1]}, "
          f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT') or 'unset'}, no token in env")
