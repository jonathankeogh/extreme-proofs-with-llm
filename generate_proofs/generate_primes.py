"""
Stage 1: generate the proof corpus for the extreme-proofs analysis

Two types that are both over the same 6 languages, which keeps the spirit of Conway and Shipman about machinery and direction:

  technique arm  5 techniques x 6 languages x 2 styles x N samples
                 -> this gives 300 at N=5.  Interior reference points: known proofs,
                 prescribed. Used to measure within-technique variation and
                 to test whether language behaves as a factor.

  extreme arm    8 directions x 6 languages x N samples
                 -> 240 at N=5.  Technique here is not prescribed. Each
                 direction asks the model to maximise one attribute. The idea
                 here is these may be the vertices of the hull.

  scope arm      6 statements x 6 languages x M samples  (--scope-arm)
                 -> 108 at M=3. No selection criterion at all.

Total at N=5, M=3: 648 proofs

Every invariant constant -- model, effort, token cap, languages, styles,
sample counts, and the extremal DIRECTIONS -- lives in config.py and is
imported, not restated. That is the only way the three corpora stay
comparable: a direction whose wording drifts between generators measures
the wording, not the direction.

Note we are sticking with one model, Opus 5 Medium. This is because we don't want model to be a factor, and Opus 5 is better at instruction following
for things like "the most elementary proof"

Output: proofs_primes.jsonl (ie one JSON object per proof, with metadata and usage).

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...   # or put it in secrets.env
    uv sync
    python generate_primes.py --dry-run   # print the grid that is going to be called, but does no API calls
    python generate_primes.py --submit    # submit the batch, print batch id
    python generate_primes.py --collect   # poll and write proofs_primes.jsonl
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

from config import (MODEL, EFFORT, MAX_TOKENS, LANGUAGES, STYLES, DIRECTIONS,
                    SAMPLES, SCOPE_SAMPLES)


THEOREM = "infinitude_of_primes"
STATEMENT = "there are infinitely many prime numbers"

OUT = Path("proofs_primes.jsonl")
BATCH_ID_FILE = Path(".batch_id_primes")
META_FILE = Path(".batch_meta_primes.json")

TECHNIQUES = {
    "euclid": "Euclid's classic argument: assume finitely many primes, "
              "consider the product of all of them plus one, and derive a "
              "contradiction from its prime factorisation.",
    "euler_product": "Euler's analytic argument: use the divergence of the "
                     "harmonic series and the Euler product formula "
                     "sum 1/n = prod (1 - 1/p)^-1 to show there must be "
                     "infinitely many primes.",
    "furstenberg": "Furstenberg's topological argument: define a topology on "
                   "the integers using arithmetic progressions as a basis, and "
                   "derive the infinitude of primes from the fact that a "
                   "finite union of closed sets is closed.",
    "erdos_counting": "Erdos's counting argument: write every integer as a "
                      "square times a squarefree part, and show that finitely "
                      "many primes would give too few representable integers "
                      "up to N.",
    "fermat_numbers": "The Fermat-numbers argument (attributed to Goldbach): "
                      "show that the Fermat numbers 2^(2^n)+1 are pairwise "
                      "coprime, so each contributes at least one new prime.",
}


# Scope arm, added after the sqrt2 design showed what it buys. Conway and
# Shipman's test for whether two proofs are really different is whether they
# settle different sets of statements, and the primes theorem has such a
# ladder too -- it is just not indexed by a number, so it is easier to miss.
# Each target below sits on a boundary where some of the five techniques
# stop working, so which argument the model produces is checkable against a
# known answer:
#
#   infinitude        the base theorem, asked with NO selection criterion.
#                     This is the centre cell: the unprompted interior point
#                     that every extremal direction should be read against.
#                     The corpus had no such reference until now.
#   primes_3_mod_4    Euclid's argument adapts directly (take 4*P - 1).
#   primes_1_mod_4    still Euclidean, but needs the extra input that any
#                     prime dividing x^2 + 1 is 1 mod 4.
#   primes_2_mod_5    no Euclidean proof exists. Murty (1988): a Euclidean
#                     proof for a mod q exists iff a^2 = 1 mod q, and
#                     2^2 = 4 is not 1 mod 5. Only the analytic method
#                     reaches this one, so it is the sharpest boundary here.
#   sum_reciprocals   Euler and Erdos give it; Euclid and Fermat do not.
#                     Separates the counting arguments from the
#                     constructive ones.
#   effective_bound   Euclid and Fermat give an explicit bound; Furstenberg
#                     gives none. The constructive pole.
SCOPE_TARGETS = {
    "infinitude": "there are infinitely many prime numbers",
    "primes_3_mod_4": "there are infinitely many primes congruent to 3 "
                      "modulo 4",
    "primes_1_mod_4": "there are infinitely many primes congruent to 1 "
                      "modulo 4",
    "primes_2_mod_5": "there are infinitely many primes congruent to 2 "
                      "modulo 5",
    "sum_reciprocals": "the sum of the reciprocals of the primes diverges",
    "effective_bound": "the n-th prime satisfies p_n < 2^(2^n) for every n",
}

# The centre cell proves the corpus's own theorem, so it carries the corpus's
# own theorem name; the rest name themselves.
SCOPE_THEOREM = {t: (THEOREM if t == "infinitude" else t)
                 for t in SCOPE_TARGETS}
SCOPE_BY_THEOREM = {v: k for k, v in SCOPE_THEOREM.items()}



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
    """Read secrets.env if present"""
    p = Path("secrets.env")
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def build_grid(n_samples: int, scope_samples: int = 0):
    """
    Yields metadata dicts for every cell.

    scope_samples=0 omits the scope arm, which is what reproduces the
    original 450-record corpus exactly. The technique and extreme cells are
    emitted first and with unchanged ids, so --submit against an existing
    proofs_primes.jsonl asks only for the new cells.
    """
    for tech, lang, style, k in itertools.product(
        TECHNIQUES, LANGUAGES, STYLES, range(n_samples)
    ):
        yield {
            "id": f"t__{tech}__{lang}__{style}__{k}",
            "arm": "technique",
            "theorem": THEOREM,
            "technique": tech,
            "direction": None,
            "language": lang,
            "style": style,
            "sample_index": k,
            "model": MODEL,
            "effort": EFFORT,
            "max_tokens": MAX_TOKENS,
        }
    for direction, lang, k in itertools.product(
        DIRECTIONS, LANGUAGES, range(n_samples)
    ):
        yield {
            "id": f"x__{direction}__{lang}__{k}",
            "arm": "extreme",
            "theorem": THEOREM,
            "technique": None,
            "direction": direction,
            "language": lang,
            "style": None,
            "sample_index": k,
            "model": MODEL,
            "effort": EFFORT,
            "max_tokens": MAX_TOKENS,
        }
    for target, lang, k in itertools.product(
        SCOPE_TARGETS, LANGUAGES, range(scope_samples)
    ):
        yield {
            "id": f"s__{target}__{lang}__{k}",
            "arm": "scope",
            "theorem": SCOPE_THEOREM[target],
            "technique": None,
            "direction": None,
            "language": lang,
            "style": None,
            "sample_index": k,
            "model": MODEL,
            "effort": EFFORT,
            "max_tokens": MAX_TOKENS,
        }


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
    """ids already written to proofs_primes.jsonl."""
    if not OUT.exists():
        return set()
    with OUT.open() as f:
        return {json.loads(l)["id"] for l in f if l.strip()}


def submit(client, grid):
    if BATCH_ID_FILE.exists():
        sys.exit(f"Batch {BATCH_ID_FILE.read_text().strip()} not collected yet. "
                 f"Run --collect first.")

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
    already = load_done()          # <-- new

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
            meta = metas[result.custom_id]
            if result.custom_id in already:
                continue
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
    BATCH_ID_FILE.unlink(missing_ok=True)      # <-- new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=SAMPLES,
                    help="samples per cell in the technique and extreme arms "
                         "(default 5 -> the 450-proof corpus)")
    ap.add_argument("--scope-arm", action="store_true",
                    help="also generate the scope arm and the centre cell "
                         "(off by default; the published corpus is the 450)")
    ap.add_argument("--scope-samples", type=int, default=SCOPE_SAMPLES,
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
        print(f"Grid: {len(grid)} proofs "
              f"= {n_tech} technique arm + {n_ext} extreme arm "
              f"+ {n_scp} scope arm "
              f"(model={MODEL}, effort={EFFORT})")
        if not args.scope_arm:
            print("Scope arm omitted; pass --scope-arm to include it.")

    if args.dry_run:
        for g in grid:
            print(" ", g["id"])
        return

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    if args.collect:
        guard_output()
        collect(client, BATCH_ID_FILE.read_text().strip())
    elif args.submit:
        guard_output()
        submit(client, grid)
    else:
        ap.error("pass --submit, --collect, or --dry-run")


if __name__ == "__main__":
    main()