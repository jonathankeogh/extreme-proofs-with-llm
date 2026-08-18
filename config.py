from pathlib import Path

MODEL = "claude-opus-5"
EFFORT = "medium"
MAX_TOKENS = 64000

LANGUAGES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "zh": "Chinese",
    "ja": "Japanese",
}

# Chinese and Japanese are deliberate. A European-only grid lets shared
# Latin roots become an implicit factor

STYLES = {
    "terse": "Write in a terse, austere style: minimal prose, heavy use of "
             "notation, no motivation or commentary, like a research "
             "monograph.",
    "verbose": "Write in an expansive, pedagogical style: motivate each step, "
               "explain the idea behind the argument in words, as if for an "
               "undergraduate seeing it for the first time.",
}

SAMPLES = 5          # per cell, technique and extreme arms
SCOPE_SAMPLES = 3    # per cell, scope arm, when --scope-arm is passed

DIRECTIONS = {
    "brevity": "Give the shortest proof you can. Minimise total length. "
               "Do not sacrifice correctness or completeness for length, but "
               "subject to that, be as short as possible.",

    "generality": "Give the proof that generalises furthest. Choose an "
                  "argument whose method extends to the widest class of "
                  "other results.",

    "visuality": "Give the most visual or geometric proof you can. The "
                 "argument should be describable as a picture or a "
                 "construction rather than a symbolic manipulation.",

    # NEW, never generated for any theorem
    "nonvisuality": "Give the least visual proof you can. The argument "
                    "should proceed by symbolic or formal manipulation, "
                    "with no step that depends on a picture, a diagram or a "
                    "spatial construction.",

    "machinery": "Give the proof that quotes the heaviest machinery. Use the "
                 "most powerful named theorems available, even where lighter "
                 "tools would suffice.",

    # unchanged wording, but absent from primes, which never had this
    # direction at all. Identical in sqrt2 and pythagoras.
    "surprise": "Give the most surprising proof you can. Prefer an argument "
                "whose central idea comes from as far outside the statement "
                "as possible.",

    # CHANGED. Was "beyond the definition of divisibility and basic
    # arithmetic" for primes and sqrt2, and "beyond lengths, areas,
    # congruence and basic arithmetic" for pythagoras
    "elementarity": "Give the most elementary proof you can. Assume nothing "
                    "beyond the basic definitions and elementary arithmetic. "
                    "Do not quote any named theorem.",

    # CHANGED. Was "from any supposed rational representation ... produce a
    # witness" for sqrt2, which presupposes a proof by contradiction about
    # rationals, and "pieces that could actually be cut out and reassembled"
    # for pythagoras, which presupposes a dissection
    "constructiveness": "Give the most constructive proof you can. The "
                        "argument should explicitly exhibit or construct the "
                        "object it asserts -- a witness, a bound, or a "
                        "procedure carried out step by step -- rather than "
                        "only deriving a contradiction or verifying an "
                        "identity.",
}

# Where each wording came from, so the cost of this file is legible.
PROVENANCE = {
    "brevity": "unchanged",
    "generality": "unchanged",
    "visuality": "unchanged",
    "nonvisuality": "NEW: never generated for any theorem",
    "machinery": "unchanged",
    "surprise": "unchanged, but never generated for primes",
    "elementarity": "NEW: replaces two theorem-specific wordings",
    "constructiveness": "NEW: replaces two theorem-specific wordings; "
                        "never generated for primes",
}

REGENERATION_REQUIRED = {
    "elementarity": ["infinitude_of_primes", "irrationality_of_sqrt2",
                     "pythagorean_theorem"],
    "constructiveness": ["infinitude_of_primes", "irrationality_of_sqrt2",
                         "pythagorean_theorem"],
    "nonvisuality": ["infinitude_of_primes", "irrationality_of_sqrt2",
                     "pythagorean_theorem"],
    "surprise": ["infinitude_of_primes"],
}

CORPORA = {
    "infinitude_of_primes": Path("proofs_primes.jsonl"),
    "irrationality_of_sqrt2": Path("proofs_sqrt2.jsonl"),
    "pythagorean_theorem": Path("proofs_pythagoras.jsonl"),
}


def corpus_state():
    """
    Re-derive comparability from the corpora themselves.

    Deliberately reads the `prompt` field of the records rather than the
    constants above: the question is what was sent, not what the source
    says it sends. Those came apart once already, which is why this file
    exists.
    """
    import hashlib
    import json
    import re
    from collections import defaultdict

    seen = defaultdict(dict)
    for theorem, path in CORPORA.items():
        if not path.exists():
            print(f"  {path} missing; skipped")
            continue
        for line in path.open():
            r = json.loads(line)
            if r.get("arm") != "extreme":
                continue
            m = re.search(r"Selection criterion: (.*?)\n\nLanguage:",
                          r["prompt"], re.S)
            if m:
                seen[r["direction"]].setdefault(theorem, set()).add(
                    hashlib.md5(m.group(1).encode()).hexdigest()[:8])
    return seen


def _report():
    seen = corpus_state()
    want = {k: __import__("hashlib").md5(v.encode()).hexdigest()[:8]
            for k, v in DIRECTIONS.items()}

    print(f"\n  {'direction':18}{'primes':>10}{'sqrt2':>10}{'pyth':>10}  "
          f"{'vs config':<12} article")
    for d in sorted(set(seen) | set(DIRECTIONS) | set(ARTICLE_VALUES)):
        cells, hs = [], []
        for t in CORPORA:
            got = seen.get(d, {}).get(t)
            if not got:
                cells.append("-")
            elif len(got) > 1:
                cells.append("MIXED")
                hs.append("mixed")
            else:
                cells.append(next(iter(got)))
                hs.append(next(iter(got)))
        if d not in DIRECTIONS:
            verdict = "not in config"
        elif not hs:
            verdict = "generate all"
        elif set(hs) == {want[d]} and len(hs) == len(CORPORA):
            verdict = "MATCHES"
        else:
            verdict = "REGENERATE"
        print(f"  {d:18}" + "".join(f"{c:>10}" for c in cells)
              + f"  {verdict:<12} {'yes' if d in ARTICLE_VALUES else ''}")

    n = sum(len(v) for v in REGENERATION_REQUIRED.values()) * \
        len(LANGUAGES) * SAMPLES
    print(f"\n  records to generate before the corpora match this config: {n}")
    missing = [v for v in ARTICLE_VALUES if v not in DIRECTIONS]
    print(f"  article values absent from DIRECTIONS: "
          f"{', '.join(missing) if missing else 'none'}")
    print(f"  opposed pairs: "
          + ", ".join(f"{a}/{b}" for a, b in OPPOSED_PAIRS))


if __name__ == "__main__":
    print(f"MODEL       {MODEL}")
    print(f"EFFORT      {EFFORT}")
    print(f"MAX_TOKENS  {MAX_TOKENS}   (unsettled -- see the note in this file)")
    print(f"LANGUAGES   {', '.join(LANGUAGES)}")
    print(f"STYLES      {', '.join(STYLES)}")
    print(f"SAMPLES     {SAMPLES} technique/extreme, {SCOPE_SAMPLES} scope")
    print(f"DIRECTIONS  {len(DIRECTIONS)}: {', '.join(DIRECTIONS)}")
    _report()
