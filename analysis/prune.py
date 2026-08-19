"""
Remove corpus records whose stored prompt no longer matches what the
generator would send today.

The point of config.py is that the three corpora are comparable. That claim
is only worth anything if the records were actually generated from the
current wording, and the resume logic in the generators cannot check it:
submit() diffs the grid against the ids already in the corpus file, so a
record whose prompt has since changed is silently kept forever.

So this script re-renders prompt_for() for every stored record and compares
it, byte for byte, against the `prompt` field the record carries. Anything
that differs is stale and is removed, which lets --submit refill the cell.

Deliberately NOT driven by a hand-maintained list of what changed. The
comparison is the whole test: it catches wordings I forgot I edited, and it
reports nothing when nothing changed.

    uv run prune.py              # report only, touches nothing
    uv run prune.py --apply      # back up, then rewrite without the stale

Backups go to <corpus>.stale-<timestamp>.jsonl, so a wrong call here is
recoverable without an API spend.
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import config

GENERATORS = {
    "infinitude_of_primes": "generate_primes",
    "irrationality_of_sqrt2": "generate_sqrt2",
    "pythagorean_theorem": "generate_pythagoras",
}


def classify(path: Path, mod):
    """Split a corpus into (fresh, stale) records by re-rendering the prompt."""
    fresh, stale = [], []
    for line in path.open():
        if not line.strip():
            continue
        rec = json.loads(line)
        try:
            want = mod.prompt_for(rec)
        except KeyError:
            # direction dropped from config entirely: nothing can refill it
            stale.append(rec)
            continue
        (fresh if want == rec["prompt"] else stale).append(rec)
    return fresh, stale


def label(rec):
    return f"{rec['arm']}/{rec['direction'] or rec['technique'] or rec['theorem']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the corpora; without it, report only")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    total_stale = 0

    for theorem, path in config.CORPORA.items():
        if not path.exists():
            print(f"{path}: missing, skipped")
            continue
        mod = __import__(GENERATORS[theorem])
        fresh, stale = classify(path, mod)
        total_stale += len(stale)

        print(f"\n{path}  {len(fresh) + len(stale)} records")
        if not stale:
            print("  all prompts current")
            continue
        for k, n in sorted(Counter(label(r) for r in stale).items()):
            print(f"  STALE  {k:34} {n:4}")

        if args.apply:
            backup = path.with_suffix(f".stale-{stamp}.jsonl")
            shutil.copy(path, backup)
            with path.open("w") as f:
                for rec in fresh:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  backed up to {backup}")
            print(f"  rewrote {path} with {len(fresh)} records")

    if not total_stale:
        print("\nNothing stale. The corpora match config.py.")
        return
    if args.apply:
        print(f"\nRemoved {total_stale} records.")
        print("Cached embeddings are row-aligned with the corpus files and are "
              "now invalid. Delete embeddings/*.npy and re-embed after the "
              "regenerated records land.")
    else:
        print(f"\n{total_stale} records are stale. Re-run with --apply to "
              f"remove them (a backup is written first).")


if __name__ == "__main__":
    main()
