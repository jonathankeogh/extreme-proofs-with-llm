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

    "nonvisuality": "Give the least visual proof you can. The argument "
                    "should proceed by symbolic or formal manipulation, "
                    "with no step that depends on a picture, a diagram or a "
                    "spatial construction.",

    "machinery": "Give the proof that quotes the heaviest machinery. Use the "
                 "most powerful named theorems available, even where lighter "
                 "tools would suffice.",

    "surprise": "Give the most surprising proof you can. Prefer an argument "
                "whose central idea comes from as far outside the statement "
                "as possible.",

    "elementarity": "Give the most elementary proof you can. Assume nothing "
                    "beyond the basic definitions and elementary arithmetic. "
                    "Do not quote any named theorem.",

    "constructiveness": "Give the most constructive proof you can. The "
                        "argument should explicitly exhibit or construct the "
                        "object it asserts -- a witness, a bound, or a "
                        "procedure carried out step by step -- rather than "
                        "only deriving a contradiction or verifying an "
                        "identity.",
}

# Both ends of an axis. If a pair does not separate in the embedding, the
# prompt is not moving the model along that axis at all - which is a
# result about the instrument, and worth being able to state
OPPOSED_PAIRS = [("elementarity", "machinery"), ("visuality", "nonvisuality")]

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


if __name__ == "__main__":
    print(f"MODEL       {MODEL}")
    print(f"EFFORT      {EFFORT}")
    print(f"MAX_TOKENS  {MAX_TOKENS}   (never binding -- see the note in this file)")
    print(f"LANGUAGES   {', '.join(LANGUAGES)}")
    print(f"STYLES      {', '.join(STYLES)}")
    print(f"SAMPLES     {SAMPLES} technique/extreme, {SCOPE_SAMPLES} scope")
    print(f"DIRECTIONS  {len(DIRECTIONS)}: {', '.join(DIRECTIONS)}")
    _report()
