"""
Stage 1, third theorem: generate the Pythagorean-theorem proof corpus.

Same design as generate_primes.py and generate_sqrt2.py -- prescribed known
proofs as interior reference points, extremal prompts as candidate vertices,
language crossed throughout, plus the centre cell and the scope ladder.

    technique arm  6 techniques x 6 languages x 2 styles x N samples
                   -> 360 at N=5.

    extreme arm    7 directions x 6 languages x N samples
                   -> 210 at N=5. Same seven as generate_sqrt2.py.

    scope arm      6 statements x 6 languages x M samples  (--scope-arm)
                   -> 108 at M=3. Off by default.

Why this theorem, as the third:

  1. It is the visuality theorem. On primes the visuality direction died --
     three visual proofs the embedding could not tell apart, and no second
     coordinate reached them. Here visual proof is the DEFAULT rather than
     an exotic corner: the rearrangement dissection and Euclid I.47 are
     both pictures, and they are pictures of different kinds. If the
     presentation axis is measurable anywhere, it is here.

  2. The scope ladder is a genuine generalisation sequence rather than a
     list of harder cases: law of cosines (drop the right angle), the box
     diagonal (add a dimension), de Gua (areas instead of lengths), and the
     inner-product statement (drop geometry altogether). Each rung kills a
     different family of proofs.

  3. It has an inner-product proof, so the machinery pole is a genuine
     abstraction rather than a heavier computation.

Two warnings specific to this theorem, both real:

  * CIRCULARITY. The famous wrong proof derives Pythagoras from
     sin^2 + cos^2 = 1, which is Pythagoras. Nothing in the pipeline
     checks correctness, and the surprise and machinery directions are
     the likely places for a circular argument to appear. Proof
     correctness here needs more hand-reading than the other two corpora,
     not less.

  * REFERENCE-SET COVERAGE. Loomis catalogued 371 proofs of this theorem.
     Six prescribed techniques cover a far smaller fraction of the known
     space than five did for primes, so extreme.py's out-of-set threshold
     is weaker evidence here: a record far from all six centroids may be
     a seventh known proof rather than a departure from the literature.

Design decisions, matching the other two generators:

  * The first five directions are byte-identical to generate_primes.py and
    generate_sqrt2.py, so a difference across the three corpora cannot be a
    difference in prompt wording. surprise and constructiveness are the two
    added in generate_sqrt2.py.

  * No prompt asks the model to describe its own proof's scope; the scope
    arm measures that by behaviour. See generate_sqrt2.py for why.

Output: proofs_pythagoras.jsonl, same schema as the other corpora.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...   # or put it in secrets.env
    uv sync
    python generate_pythagoras.py --dry-run
    python generate_pythagoras.py --submit
    python generate_pythagoras.py --collect

    python first_pass.py --corpus proofs_pythagoras.jsonl
"""

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import anthropic


MODEL = "claude-opus-5"
EFFORT = "medium"
MAX_TOKENS = 32000

THEOREM = "pythagorean_theorem"
STATEMENT = ("in a right triangle the square on the hypotenuse is equal to "
             "the sum of the squares on the other two sides")

OUT = Path("proofs_pythagoras.jsonl")
BATCH_ID_FILE = Path(".batch_id_pythagoras")
META_FILE = Path(".batch_meta_pythagoras.json")

# Six proofs chosen to span the axes, not to sample the 371 evenly: two
# pictures of different kinds (dissection vs Euclid's construction), two
# computations (similar triangles, Garfield), one that quotes a named
# theorem (Ptolemy), one that abandons the triangle entirely (inner
# product).
TECHNIQUES = {
    "euclid_i47": "Euclid's proof, Elements I.47: drop the perpendicular "
                  "from the right angle to the hypotenuse, splitting the "
                  "square on the hypotenuse into two rectangles, and show by "
                  "congruent triangles that each rectangle has the same area "
                  "as the square on the corresponding leg.",
    "similar_triangles": "The similar-triangles proof: the altitude from the "
                         "right angle divides the triangle into two triangles "
                         "each similar to the whole, so each leg is the mean "
                         "proportional between the hypotenuse and its "
                         "adjacent segment; adding the two resulting "
                         "relations gives the theorem.",
    "rearrangement": "The dissection proof: arrange four copies of the right "
                     "triangle inside a square of side a + b in two different "
                     "ways, and compare the uncovered area, which is c^2 in "
                     "one arrangement and a^2 + b^2 in the other. Argue by "
                     "areas of congruent pieces, not by algebra on "
                     "coordinates.",
    "garfield": "Garfield's trapezoid proof: place two copies of the right "
                "triangle and the triangle on the hypotenuse to form a "
                "trapezoid of parallel sides a and b and height a + b, then "
                "compute its area in two ways.",
    "ptolemy": "The proof via Ptolemy's theorem: inscribe the right triangle "
               "in a circle, complete it to a rectangle, and apply Ptolemy's "
               "relation for a cyclic quadrilateral to obtain the theorem. "
               "Quote Ptolemy's theorem; do not prove it.",
    "inner_product": "The inner-product proof: in a real inner product space "
                     "expand ||x + y||^2 = <x + y, x + y> by bilinearity, and "
                     "observe that orthogonality makes the cross terms "
                     "vanish, leaving ||x + y||^2 = ||x||^2 + ||y||^2.",
}

# Five of these are byte-identical to generate_primes.py and
# generate_sqrt2.py: brevity, generality, visuality, machinery, surprise.
# They never name a theorem, so they carry over untouched and a difference
# between the three corpora on those five cannot be a wording difference.
#
# TWO HAD TO CHANGE, and pretending otherwise would have quietly broken the
# arm:
#
#   elementarity      the shared text reads "assume nothing beyond the
#                     definition of divisibility and basic arithmetic".
#                     Divisibility is the right floor for primes and for
#                     sqrt2, and it is irrelevant to a statement about
#                     areas. Replaced with the geometric floor.
#
#   constructiveness  the shared text asks the proof to produce "a strictly
#                     simpler representation" from "any supposed rational
#                     representation" -- that is the sqrt2 descent, and it
#                     is meaningless here. Replaced with the constructive
#                     pole that this theorem actually has: an explicit
#                     dissection.
#
# So brevity/generality/visuality/machinery/surprise are comparable across
# all three theorems; elementarity is comparable between primes and sqrt2
# only; constructiveness is not comparable across theorems at all.
DIRECTIONS = {
    "brevity": "Give the shortest proof you can. Minimise total length. "
               "Do not sacrifice correctness or completeness for length, but "
               "subject to that, be as short as possible.",
    "elementarity": "Give the most elementary proof you can. Assume nothing "
                    "beyond lengths, areas, congruence and basic arithmetic. "
                    "Do not quote any named theorem.",
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
                        "argument should exhibit an explicit construction -- "
                        "pieces that could actually be cut out and "
                        "reassembled, or a procedure carried out step by "
                        "step -- rather than only verifying an identity.",
}

# Scope ladder. Each rung removes a hypothesis the lighter proofs rely on.
SCOPE_TARGETS = {
    # centre cell: the theorem itself, no selection criterion
    "pythagoras": ("in a right triangle the square on the hypotenuse is "
                   "equal to the sum of the squares on the other two sides"),
    # drop the right angle: every dissection proof dies, the
    # similar-triangle and algebraic ones survive
    "law_of_cosines": ("in any triangle with sides a, b and c, where C is "
                       "the angle opposite c, c^2 = a^2 + b^2 - 2ab cos C"),
    # add a dimension: plane dissections die
    "box_diagonal": ("the square of the length of the space diagonal of a "
                     "rectangular box equals the sum of the squares of its "
                     "three edge lengths"),
    # areas rather than lengths: only the algebraic and vector methods reach
    "de_gua": ("for a tetrahedron with three mutually perpendicular faces "
               "meeting at one vertex, the square of the area of the fourth "
               "face equals the sum of the squares of the areas of the "
               "other three"),
    # geometry dropped entirely: the inner-product proof only
    "parallelogram_law": ("in any real inner product space, "
                          "||x + y||^2 + ||x - y||^2 = 2||x||^2 + 2||y||^2"),
    "inner_product_space": ("in any real inner product space, if x and y are "
                           "orthogonal then ||x + y||^2 = ||x||^2 + ||y||^2"),
}

# The centre cell proves the corpus's own theorem, so it carries the corpus's
# own theorem name; the rest name themselves.
SCOPE_THEOREM = {t: (THEOREM if t == "pythagoras" else t)
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
        print("Truncated records are incomplete proofs, and they are now in "
              "the corpus, so a re-run will NOT replace them: the resume "
              "logic keys on id. Raise MAX_TOKENS, delete those lines, then "
              "--submit again to refill the cells.")
    BATCH_ID_FILE.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5,
                    help="samples per cell in the technique and extreme arms "
                         "(default 5 -> 570 proofs)")
    ap.add_argument("--scope-arm", action="store_true",
                    help="also generate the scope arm and the centre cell")
    ap.add_argument("--scope-samples", type=int, default=3,
                    help="samples per cell in the scope arm (default 3)")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_env()
    n_scope = args.scope_samples if args.scope_arm else 0
    grid = list(build_grid(args.samples, n_scope))
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
