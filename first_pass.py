"""
Stage 2: first-pass analysis of the proof corpus.

Reads proofs.jsonl and reports, in order:

  0. integrity      record count, uniqueness, truncation, prompt homogeneity
  1. raw lengths    median characters per direction, pooled over languages
  2. cells          median and CV per (direction, language)
  3. normalised     within-language z-score of log length
  4. summary        direction means, language spread, between:within ratio
  5. interactions   per-direction CJK-vs-Latin gap on the normalised scale

Character count is not comparable across scripts: a Chinese proof of the same
content is roughly 0.57x the length of the English one -- the CJK-to-Latin
median ratio measured on this corpus, stable across all five directions, which
is what identifies it as a property of the script rather than of the proof.
The effect is multiplicative, so section 3 works in log space, where a constant ratio
becomes a constant offset and subtracting a per-language mean removes it.

Baselines use all records in a language, both arms (n = 75 per language at
--samples 5), not just the extreme arm. The grid is balanced identically
across languages, so no language gets an unfair baseline.

Usage:
    python first_pass.py
    python first_pass.py --corpus proofs.jsonl
"""

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

# non-latin character languages
CJK = {"ja", "zh"}


def load(path: Path):
    if not path.exists():
        raise SystemExit(f"{path} not found.")
    with path.open() as f:
        recs = [json.loads(l) for l in f if l.strip()]
    if not recs:
        raise SystemExit(f"{path} is empty.")
    for r in recs:
        r["chars"] = len(r["proof"])
        r["log_chars"] = math.log(r["chars"]) if r["chars"] else float("nan")
    return recs


def rule(title):
    print(f"\n{title}\n" + "-" * len(title))


def integrity(recs):
    # Here we are just doing housekeeping
    rule("0. Integrity")
    ids = [r["id"] for r in recs]
    print(f"records            {len(recs)}")
    print(f"unique ids         {len(set(ids))}")
    print(f"truncated          {sum(r['stop_reason'] == 'max_tokens' for r in recs)}")
    print(f"empty proofs       {sum(r['chars'] == 0 for r in recs)}")

    for field in ("model", "effort"):
        vals = {r.get(field) for r in recs}
        flag = "" if len(vals) == 1 else "  <- mixed, model differences and effort confounding all the factors"
        print(f"{field:18} {sorted(map(str, vals))}{flag}")

    # One prompt per (direction, language) is what is expected by design: the template
    # move between the language names so a direction has 6 distinct prompts.
    per_cell = defaultdict(set)
    for r in recs:
        if r["arm"] == "extreme":
            per_cell[(r["direction"], r["language"])].add(r["prompt"])
    bad = {k: len(v) for k, v in per_cell.items() if len(v) != 1}
    print(f"cells w/ mixed prompt  {len(bad)}" + (f"  {bad}" if bad else ""))


def raw_lengths(recs):
    # Quick sanity check that we don't have something oddly long or oddly short, language effects kept
    rule("1. Raw length by direction (characters, pooled over languages)")
    d = defaultdict(list)
    for r in recs:
        if r["arm"] == "extreme":
            d[r["direction"]].append(r["chars"])
    for k, v in sorted(d.items(), key=lambda kv: st.median(kv[1])):
        print(f"  {k:14} median {st.median(v):7.0f}   n={len(v)}")
    print("  Pooled over scripts, so this axis still carries the language effect.")


def cells(recs):
    # within (direction, language) how much variation in length do we see in the samples (e.g. n=5)
    rule("2. Cell medians and within-cell variation")
    d = defaultdict(list)
    for r in recs:
        if r["arm"] == "extreme":
            d[(r["direction"], r["language"])].append(r["chars"])
    print(f"  {'direction':14} {'lang':4} {'median':>7} {'cv':>6}")
    for k, v in sorted(d.items()):
        cv = st.stdev(v) / st.mean(v) if len(v) > 1 else float("nan")
        print(f"  {k[0]:14} {k[1]:4} {st.median(v):7.0f} {cv:6.2f}")
    print("  CV ~0.05 means the model returns one proof five times over")
    print("  higher means real sampling variation")
    print("  Note that 5 samples is a poor estimate of the true variance, this is not a statistical result we are just doing a first pass")




def normalise(recs):
    # Since script density factor can affect sequence lengths which we do not want to confound the underlying proof structure, we 
    # get the log-length (since the factor is a mult) and standardise so that languages are apples to apples
    """Within-language z-score of log length. Mutates recs, returns cell means."""
    by_lang = defaultdict(list)
    for r in recs:
        by_lang[r["language"]].append(r["log_chars"])
    mu = {k: st.mean(v) for k, v in by_lang.items()}
    sd = {k: st.stdev(v) for k, v in by_lang.items()}
    for r in recs:
        r["z"] = (r["log_chars"] - mu[r["language"]]) / sd[r["language"]]

    rule("3. Normalised length: z-score of log chars, within language")
    d = defaultdict(list)
    for r in recs:
        if r["arm"] == "extreme":
            d[(r["direction"], r["language"])].append(r["z"])
    cell = {k: st.mean(v) for k, v in d.items()}
    for k in sorted(cell):
        print(f"  {k[0]:14} {k[1]:4} z {cell[k]:+.2f}")
    return cell
    # It is unsurprising that brevity has a negative z-score and machinery has a positive for e.g. that is a hopeful sign that we
    # can extract the underlying structure but no signal yet


def summary(cell):
    # Now with the normalised cell lengths, we want to see if we can see larger differences in either direction or language, we want direction to be much larger
    rule("4. Direction separation vs language spread")
    dirs = sorted({k[0] for k in cell})
    means, spreads = {}, {}
    print(f"  {'direction':14} {'mean z':>8} {'lang spread':>12}")
    for dd in dirs:
        vals = [v for k, v in cell.items() if k[0] == dd]
        means[dd] = st.mean(vals)
        spreads[dd] = max(vals) - min(vals)
    for dd in sorted(dirs, key=lambda x: means[x]):
        print(f"  {dd:14} {means[dd]:+8.2f} {spreads[dd]:12.2f}")

    between = max(means.values()) - min(means.values())
    within = st.mean(list(spreads.values()))
    # So now with the normalised log-lengths, let's see the diferences
    print(f"\n  between-direction range   {between:.2f}")
    print(f"  mean within-dir spread    {within:.2f}")
    print(f"  ratio                     {between / within:.1f} : 1")
    print("  A high ratio is evidence that language behaves as a nuisance")
    print("  factor on this axis, this is the assumption the 6-language design rests on!")

    print("\n  Directions within 0.15 z of each other (unseparated by length):")
    ordered = sorted(dirs, key=lambda x: means[x])
    close = [(a, b) for a, b in zip(ordered, ordered[1:])
             if abs(means[a] - means[b]) < 0.15]
    for a, b in close:
        print(f"    {a} ~ {b}")
    if not close:
        print("    none")
    
    # I see ratio > 8 which is good!


def interactions(cell):
    # Now let's hold direction steady and see how much change happens across latin and cjk scripts (normalised)
    rule("5. CJK vs Latin, per direction (normalised scale)")
    print("  A level difference between scripts is already removed by section 3.")
    print("  Anything left here is a language x direction interaction.")
    print(f"\n  {'direction':14} {'latin':>7} {'cjk':>7} {'gap':>7}")
    for dd in sorted({k[0] for k in cell}):
        latin = [v for k, v in cell.items()
                 if k[0] == dd and k[1] not in CJK]
        cjk = [v for k, v in cell.items() if k[0] == dd and k[1] in CJK]
        if not latin or not cjk:
            continue
        gap = st.mean(cjk) - st.mean(latin)
        flag = "  <--" if abs(gap) > 0.25 else ""
        print(f"  {dd:14} {st.mean(latin):+7.2f} {st.mean(cjk):+7.2f} "
              f"{gap:+7.2f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("proofs.jsonl"))
    args = ap.parse_args()

    recs = load(args.corpus)
    integrity(recs)
    raw_lengths(recs)
    cells(recs)
    cell = normalise(recs)
    summary(cell)
    interactions(cell)

    print("\nCaveat: character count is a proxy for proof length, but not a measure")


if __name__ == "__main__":
    main()