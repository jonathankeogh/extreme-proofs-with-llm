"""
Stage 1, second theorem: generate the sqrt-2 proof corpus.

Same logical design as generate_primes.py -- prescribed known proofs as
interior reference points, extremal prompts as candidate vertices, language
as a crossed factor -- applied to the theorem Conway and Shipman actually
wrote about: there is no rational number whose square is 2.

    technique arm  6 techniques x 6 languages x 2 styles x N samples
                   -> 360 at N=5. Known proofs, prescribed. The instrument.

    extreme arm    7 directions x 6 languages x N samples
                   -> 210 at N=5. Technique not prescribed; each direction
                   asks for a vertex.

    scope arm      7 statements x 6 languages x M samples  (--scope-arm)
                   -> 126 at M=3. No criterion at all. Off by default.

Why this theorem is worth the second run, given the primes corpus already
exists (see WRITEUP.md for what the primes run left open):

  1. The proofs are documented with their SCOPE. Conway and Shipman's own
     test for whether two proofs are really different is whether they settle
     different sets of numbers: covering does sqrt2 and sqrt3, folding does
     sqrt(n^2 +- 1), traditional does roots of primes, reciprocation does
     square roots of every nonsquare, unique factorisation does all roots of
     all integers, analytic does all algebraic integers. So "generality" has
     a ground-truth ordering here. On the primes theorem it had none, which
     is why that axis stayed unfalsifiable.

  2. Visuality has more than one target. The primes corpus could not tell
     its three visual proofs apart, and that axis died. Here the article
     gives two geometric proofs it explicitly separates -- covering is
     "visually obvious", folding is "purely geometrical" -- so asking which
     one the visuality prompt selects is a question with an answer.

  3. Two of the six prescribed proofs are near-duplicates by construction.
     Conway and Shipman say covering and folding are "mechanically the
     same", both carrying out the first step of the division algorithm.
     Expect the technique probe to sit below the 1.000 the primes arm hit.
     That is the design working, not failing: five obviously different
     arguments were an easy instrument, and this one is not.

Design decisions, so they are auditable rather than inferred:

  * Bashmakova's mod-8 proof is the seventh in the article and is NOT here.
    The article states that for N = 2 it "is really the same as the
    Traditional proof". Prescribing it would put two labels on one argument
    and measure the labelling, not the model.

  * The first five directions are byte-identical to the ones in
    generate_primes.py. They never name the theorem, so they did not need
    adapting, and keeping them identical means any difference between the
    two corpora cannot be a difference in prompt wording. `surprise` and
    `constructiveness` are new, and are two more of the values Conway and
    Shipman list in their opening paragraph.

  * No direction prompt asks the model to state its proof's scope, however
    tempting that is here. The primes run truncated 28 records because the
    generality prompt asked for a deliverable beyond the proof; see the
    corpus note in README.md. The scope arm measures scope by behaviour
    instead -- give the model a harder number and see which method it
    reaches for -- which needs nothing added to the prompt.

  * The scope arm's prompt carries no selection criterion, so it is also
    the missing baseline: what the model writes unprompted. Its first
    target is sqrt2 itself, which gives the interior point that every
    extremal direction should be measured against. generate_primes.py now
    carries the same arm, added afterwards, so both theorems are analysed
    on the same footing.

Output: proofs_sqrt2.jsonl, one JSON object per proof, same schema as
proofs_primes.jsonl. Scope-arm records carry arm="scope" and a varying `theorem`;
every downstream script filters on `arm`, so they are inert unless asked
for.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...   # or put it in secrets.env
    uv sync
    python generate_sqrt2.py --dry-run    # print the grid, no API calls
    python generate_sqrt2.py --submit     # submit the batch, print batch id
    python generate_sqrt2.py --collect    # poll and write proofs_sqrt2.jsonl

    python first_pass.py --corpus proofs_sqrt2.jsonl
"""

import argparse
import itertools
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import anthropic


MODEL = "claude-opus-5"
EFFORT = "medium"
# 32000 throughout, matching the primes corpus after its cap was raised.
# Starting here avoids repeating that run's truncations.
MAX_TOKENS = 64000

THEOREM = "irrationality_of_sqrt2"
STATEMENT = "there is no rational number whose square is 2"

OUT = Path("proofs_sqrt2.jsonl")
# Distinct from the primes generator's, so the two can be in flight at once.
BATCH_ID_FILE = Path(".batch_id_sqrt2")
META_FILE = Path(".batch_meta_sqrt2.json")

# The six proofs of Conway and Shipman (Math. Intelligencer 35(3), 2013),
# minus Bashmakova's; see the header. Ordered as in the article, which is
# increasing scope: covering settles the fewest numbers, analytic the most.
TECHNIQUES = {
    "covering": "Tennenbaum's covering argument: p^2 = 2q^2 says a p x p "
                "square has the area of two q x q squares. Take the smallest "
                "such p, lay the two smaller squares into opposite corners of "
                "the larger one, and observe that the doubly covered square "
                "in the middle has the same area as the two uncovered corner "
                "squares -- a strictly smaller instance, contradicting "
                "minimality.",
    "folding": "The folding argument: suppose the side and diagonal of a "
               "square are both integer multiples q and p of a common unit, "
               "with p and q least. Fold the half-square triangle so that the "
               "hypotenuse lands along a leg; the triangle that remains is a "
               "smaller half-square triangle with integer sides, "
               "contradicting minimality.",
    "traditional": "The traditional even/odd argument: write the supposed "
                   "root as p/q in lowest terms, so p^2 = 2q^2; conclude that "
                   "p is even, hence that q is even, contradicting lowest "
                   "terms. Use only the parity of products -- a product is "
                   "odd exactly when both factors are odd.",
    "reciprocation": "The reciprocation argument of Conway and Guy: if the "
                     "root is P/Q then it also equals 2/(the root) = 2Q/P, so "
                     "P/Q and 2Q/P have the same fractional part; writing "
                     "those parts as q/Q and p/P gives p/q equal to P/Q with "
                     "p < P and q < Q, a strictly simpler representation. Use "
                     "one step of division with remainder and nothing about "
                     "factorisation.",
    "unique_factorization": "The unique factorisation argument: in a^2 the "
                            "exponent of 2 is even, while in 2b^2 it is odd, "
                            "so by the Fundamental Theorem of Arithmetic a^2 "
                            "cannot equal 2b^2. Quote that theorem; do not "
                            "prove it.",
    "analytic": "Laczkovich's analytic argument: for every positive integer n "
                "the number (sqrt2 - 1)^n has the form a*sqrt2 + b with a and "
                "b integers, and since 0 < sqrt2 - 1 < 1 these numbers tend "
                "to 0; but if sqrt2 were rational with denominator D then "
                "every a*sqrt2 + b would be a rational with denominator "
                "dividing D, and a nonzero such rational cannot be "
                "arbitrarily close to 0.",
}

# First five byte-identical to generate_primes.py -- see the header. The last
# two are further values from Conway and Shipman's opening list ("brevity,
# generality, constructiveness, visuality, nonvisuality, surprise,
# elementarity"); the article tags the analytic proof "surprising", and
# descent is the constructive pole against unique factorisation.
DIRECTIONS = {
    "brevity": "Give the shortest proof you can. Minimise total length. "
               "Do not sacrifice correctness or completeness for length, but "
               "subject to that, be as short as possible.",
    "elementarity": "Give the most elementary proof you can. Assume nothing "
                    "beyond the basic definitions and elementary "
                    "arithmetic. Do not quote any named theorem.",
    "generality": "Give the proof that generalises furthest. Choose an "
                  "argument whose method extends to the widest class of "
                  "other results.",
    "visuality": "Give the most visual or geometric proof you can. The "
                 "argument should be describable as a picture or a "
                 "construction rather than a symbolic manipulation.",
    "machinery": "Give the proof that quotes the heaviest machinery. Use the "
                 "most powerful named theorems available, even where lighter "
                 "tools would suffice.",
    "surprise": "Give the most surprising proof you can. Prefer an argument "
                "whose central idea comes from as far outside the statement "
                "as possible.",
    "constructiveness": "Give the most constructive proof you can. The "
                        "argument should explicitly exhibit or construct "
                        "the object it asserts -- a witness, a bound, or a "
                        "procedure carried out step by step -- rather than "
                        "only deriving a contradiction or verifying an "
                        "identity.",
}

# Scope arm. Each statement sits on a boundary that the article documents,
# so which method the model produces is checkable against a known answer.
# The comment on each line is what survives at that target.
SCOPE_TARGETS = {
    # baseline: the theorem of the other two arms, asked with no criterion
    "sqrt2": "there is no rational number whose square is 2",
    # all six still work; covering is Figure 2c
    "sqrt3": "there is no rational number whose square is 3",
    # 6 is not prime, so traditional dies; covering dies; folding only via
    # sqrt6 = sqrt24 / 2
    "sqrt6": "there is no rational number whose square is 6",
    # x^2 + 1 = 7y^2 is unsolvable, so folding needs the sqrt63 / 3 route
    "sqrt7": "there is no rational number whose square is 7",
    # Theodorus stopped here; 17 = 4^2 + 1 so folding is easy, and 17 is
    # prime so parity works -- the historical control
    "sqrt17": "there is no rational number whose square is 17",
    # every geometric proof dies; parity survives, since a power of an odd
    # number is odd
    "cbrt2": "there is no rational number whose cube is 2",
    # outside every scope but the analytic one
    "cubic_root": "the real number x satisfying x^3 = x + 1 is irrational",
}

# The centre cell proves the corpus's own theorem, so it carries the corpus's
# own theorem name; the rest name themselves.
SCOPE_THEOREM = {t: (THEOREM if t == "sqrt2" else f"irrationality_of_{t}")
                 for t in SCOPE_TARGETS}
SCOPE_BY_THEOREM = {v: k for k, v in SCOPE_THEOREM.items()}

LANGUAGES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "zh": "Chinese",
    "ja": "Japanese",
}

STYLES = {
    "terse": "Write in a terse, austere style: minimal prose, heavy use of "
             "notation, no motivation or commentary, like a research "
             "monograph.",
    "verbose": "Write in an expansive, pedagogical style: motivate each step, "
               "explain the idea behind the argument in words, as if for an "
               "undergraduate seeing it for the first time.",
}

TECHNIQUE_TEMPLATE = """Write a complete, correct, self-contained proof that {statement}.

Proof technique (follow this approach and no other): {technique}

Language: write the entire proof in {language}.

Style: {style}

Output only the proof itself. No title, no preamble, no closing remarks."""

EXTREME_TEMPLATE = """Write a complete, correct, self-contained proof that {statement}.

Choose the proof yourself. You are not restricted to any particular argument.

Selection criterion: {direction}

Language: write the entire proof in {language}.

Output only the proof itself. No title, no preamble, no closing remarks."""

# No selection criterion, deliberately: this arm asks what the model does
# when nothing is being maximised.
SCOPE_TEMPLATE = """Write a complete, correct, self-contained proof that {statement}.

Choose the proof yourself. You are not restricted to any particular argument.

Language: write the entire proof in {language}.

Output only the proof itself. No title, no preamble, no closing remarks."""


def load_env():
    """Read secrets.env if present."""
    p = Path("secrets.env")
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def base(**kw):
    meta = {
        "arm": None, "theorem": THEOREM, "technique": None, "direction": None,
        "language": None, "style": None, "sample_index": None,
        "model": MODEL, "effort": EFFORT, "max_tokens": MAX_TOKENS,
    }
    meta.update(kw)
    return meta


def build_grid(n_samples: int, scope_samples: int = 0):
    """Yield metadata dicts for every cell. scope_samples=0 omits that arm."""
    for tech, lang, style, k in itertools.product(
        TECHNIQUES, LANGUAGES, STYLES, range(n_samples)
    ):
        yield base(id=f"t__{tech}__{lang}__{style}__{k}", arm="technique",
                   technique=tech, language=lang, style=style, sample_index=k)

    for direction, lang, k in itertools.product(
        DIRECTIONS, LANGUAGES, range(n_samples)
    ):
        yield base(id=f"x__{direction}__{lang}__{k}", arm="extreme",
                   direction=direction, language=lang, sample_index=k)

    for target, lang, k in itertools.product(
        SCOPE_TARGETS, LANGUAGES, range(scope_samples)
    ):
        yield base(id=f"s__{target}__{lang}__{k}", arm="scope",
                   theorem=SCOPE_THEOREM[target], language=lang,
                   sample_index=k)


def prompt_for(meta: dict) -> str:
    if meta["arm"] == "technique":
        return TECHNIQUE_TEMPLATE.format(
            statement=STATEMENT,
            technique=TECHNIQUES[meta["technique"]],
            language=LANGUAGES[meta["language"]],
            style=STYLES[meta["style"]],
        )
    if meta["arm"] == "extreme":
        return EXTREME_TEMPLATE.format(
            statement=STATEMENT,
            direction=DIRECTIONS[meta["direction"]],
            language=LANGUAGES[meta["language"]],
        )
    return SCOPE_TEMPLATE.format(
        statement=SCOPE_TARGETS[SCOPE_BY_THEOREM[meta["theorem"]]],
        language=LANGUAGES[meta["language"]],
    )


def to_request(meta: dict) -> dict:
    # custom_id must match ^[a-zA-Z0-9_-]{1,64}$
    assert len(meta["id"]) <= 64, meta["id"]
    return {
        "custom_id": meta["id"],
        "params": {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": EFFORT},
            "messages": [{"role": "user", "content": prompt_for(meta)}],
        },
    }


def guard_output():
    """Refuse to append to a corpus generated by a different model/effort."""
    if not OUT.exists():
        return
    with OUT.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("model") != MODEL or rec.get("effort") != EFFORT:
                sys.exit(
                    f"{OUT} contains records from model={rec.get('model')} "
                    f"effort={rec.get('effort')}, but this run is "
                    f"model={MODEL} effort={EFFORT}. Mixing them confounds "
                    f"model with every factor in the grid. Delete {OUT} and "
                    f"regenerate the whole corpus."
                )
            break


def describe_pending():
    """Print what the pending batch actually contains, from its meta file."""
    if not META_FILE.exists():
        print(f"No {META_FILE}; nothing to collect.")
        return
    metas = json.loads(META_FILE.read_text())
    arms = Counter(m["arm"] for m in metas.values())
    print(f"Collecting a batch of {len(metas)} submitted requests: "
          + ", ".join(f"{n} {a}" for a, n in sorted(arms.items())))
    print("(--scope-arm has no effect here; batch contents were fixed at "
          "submit time)")


def load_done() -> set[str]:
    """ids already written to the corpus."""
    if not OUT.exists():
        return set()
    with OUT.open() as f:
        return {json.loads(l)["id"] for l in f if l.strip()}


def submit(client, grid):
    if BATCH_ID_FILE.exists():
        sys.exit(f"Batch {BATCH_ID_FILE.read_text().strip()} not collected "
                 f"yet. Run --collect first.")

    todo = [m for m in grid if m["id"] not in load_done()]
    if not todo:
        print("Nothing missing.")
        return

    META_FILE.write_text(json.dumps({m["id"]: m for m in todo}))
    batch = client.messages.batches.create(
        requests=[to_request(m) for m in todo])
    BATCH_ID_FILE.write_text(batch.id)
    print(f"Submitted {len(todo)} of {len(grid)}. Batch id: {batch.id}")


def collect(client, batch_id: str, poll_seconds: int = 60):
    metas = json.loads(META_FILE.read_text())
    already = load_done()

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        c = batch.request_counts
        print(f"status={batch.processing_status} succeeded={c.succeeded} "
              f"errored={c.errored} processing={c.processing}")
        time.sleep(poll_seconds)

    n_ok = n_err = n_trunc = 0
    in_tok = out_tok = 0
    with OUT.open("a") as f:
        for result in client.messages.batches.results(batch_id):
            if result.custom_id in already:
                continue
            meta = metas[result.custom_id]
            if result.result.type != "succeeded":
                n_err += 1
                print(f"FAILED {result.custom_id}: {result.result.type} "
                      f"{getattr(result.result, 'error', '')}")
                continue
            msg = result.result.message
            text = "".join(b.text for b in msg.content if b.type == "text")
            if msg.stop_reason == "max_tokens":
                n_trunc += 1
                print(f"TRUNCATED {result.custom_id} -- raise MAX_TOKENS")
            in_tok += msg.usage.input_tokens
            out_tok += msg.usage.output_tokens
            f.write(json.dumps({
                **meta,
                "prompt": prompt_for(meta),
                "proof": text.strip(),
                "stop_reason": msg.stop_reason,
                "usage": {
                    "input_tokens": msg.usage.input_tokens,
                    "output_tokens": msg.usage.output_tokens,
                },
            }, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"\n{n_ok} succeeded, {n_err} failed, {n_trunc} truncated. "
          f"Corpus at {OUT.resolve()}")
    print(f"Tokens: {in_tok} in, {out_tok} out")
    if n_trunc:
        print("Truncated records are incomplete proofs. Do not analyse them "
              "as if they were short ones -- raise MAX_TOKENS, delete those "
              "lines, and rerun --submit to refill the cells.")
    BATCH_ID_FILE.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5,
                    help="samples per cell in the technique and extreme arms "
                         "(default 5 -> 570 proofs)")
    ap.add_argument("--scope-arm", action="store_true",
                    help="also generate the scope arm (off by default)")
    ap.add_argument("--scope-samples", type=int, default=3,
                    help="samples per cell in the scope arm (default 3)")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_env()
    n_scope = args.scope_samples if args.scope_arm else 0
    grid = list(build_grid(args.samples, n_scope))

    if args.collect:
        # The grid describes THIS invocation's flags, and they do not apply
        # when collecting: the batch contents were fixed at submit time, so
        # printing a grid here would claim the scope arm is absent from a
        # batch that contains it. Report what is actually pending instead.
        describe_pending()
    else:
        n_tech = len(TECHNIQUES) * len(LANGUAGES) * len(STYLES) * args.samples
        n_ext = len(DIRECTIONS) * len(LANGUAGES) * args.samples
        n_scp = len(SCOPE_TARGETS) * len(LANGUAGES) * n_scope
        print(f"Grid: {len(grid)} proofs = {n_tech} technique arm "
              f"+ {n_ext} extreme arm + {n_scp} scope arm "
              f"(model={MODEL}, effort={EFFORT}, max_tokens={MAX_TOKENS})")
        if not args.scope_arm:
            print("Scope arm omitted; pass --scope-arm to include it.")

    if args.dry_run:
        for g in grid:
            print(" ", g["id"])
        return

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    if args.collect:
        guard_output()
        if not BATCH_ID_FILE.exists():
            sys.exit(f"No {BATCH_ID_FILE}; nothing to collect.")
        collect(client, BATCH_ID_FILE.read_text().strip())
    elif args.submit:
        guard_output()
        submit(client, grid)
    else:
        ap.error("pass --submit, --collect, or --dry-run")


if __name__ == "__main__":
    main()
