"""Tally the verdict words of docs/registered_predictions.md (D-449 (1)).

Read only. A row's verdict is the first bold word in its verdict column (the column after the result-file column);
where the bold text lists alternatives ("NOT SUPPORTED / REFUTED / INCONCLUSIVE"), the first of them named in the
text after it is the entry's own. An unbolded "OUTCOME ONLY" cell counts as OUTCOME ONLY. Prints the count over all rows and over the Phase 3 rows of the author's exit rule part (1)
(56-76, 90-100, 105-137; docs/phase3_exit.md section 4), and exits 1 if any Phase 3 row is NOT YET RUN or has no word.

    python -I scripts/tally_ledger_verdicts_d449.py
"""
import collections
import re
import sys
from pathlib import Path

LEDGER = Path(__file__).resolve().parent.parent / "docs" / "registered_predictions.md"
PHASE3 = list(range(56, 77)) + list(range(90, 101)) + list(range(105, 138))
WORDS = ("WITHDRAWN BEFORE RUNNING", "NOT SUPPORTED", "NOT YET RUN", "OUTCOME ONLY", "DESCRIPTIVE", "INCONCLUSIVE",
         "UNDEFINED", "REFUTED", "HELD", "FAILED")


def verdict(cells):
    for i, c in enumerate(cells):
        if "`results/" in c or c.strip() in ("none", "same") or c.strip().startswith("same "):
            continue
        if c.strip().startswith("OUTCOME ONLY"):
            return "OUTCOME ONLY"
        m = re.match(r"\s*\*\*([^*]+)\*\*(.*)", c)
        if not m:
            continue
        bold, rest = m.group(1).strip(), m.group(2)
        if "/" in bold:
            hits = [(rest.find(w), w) for w in WORDS if w in bold and rest.find(w) >= 0]
            bold = min(hits)[1] if hits else bold
        for w in WORDS:
            if bold.startswith(w):
                return w
        return "OTHER:" + bold
    return None


def main():
    rows = {}
    for line in LEDGER.read_bytes().decode("utf-8").split("\n"):
        m = re.match(r"^\|\s*(\d+)\s*\|", line)
        if m:
            n = int(m.group(1))
            if n in rows:
                sys.exit(f"duplicate row {n}")
            rows[n] = line.rstrip("\r").split("|")[3:]
    words = {n: verdict(c) for n, c in rows.items()}
    print(f"rows {len(rows)} (1..{max(rows)}; missing {sorted(set(range(1, max(rows) + 1)) - set(rows))})")
    print("all rows:", dict(collections.Counter(words.values()).most_common()))
    missing = [n for n in PHASE3 if n not in rows]
    p3 = collections.Counter(words.get(n) for n in PHASE3)
    print(f"Phase 3 rows {len(PHASE3)} (missing {missing}):", dict(p3.most_common()))
    open_rows = [n for n in PHASE3 if words.get(n) in (None, "NOT YET RUN") or str(words.get(n)).startswith("OTHER")]
    print("Phase 3 rows without a verdict or withdrawal:", open_rows or "none")
    print(f"part (1): {len(PHASE3) - len(open_rows) - len(missing)} of {len(PHASE3)}")
    return 1 if open_rows or missing else 0


if __name__ == "__main__":
    sys.exit(main())
