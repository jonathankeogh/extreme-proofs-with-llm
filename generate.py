"""
Stage 1: generate the proof corpus for the extreme-proofs analysis

Two types that are both over the same 6 languages, which keeps the spirit of Conway and Shipman about machinery and direction:

  technique arm  5 techniques x 6 languages x 2 styles x N samples
                 -> this gives 300 at N=5.  Interior reference points: known proofs,
                 prescribed. Used to measure within-technique variation and
                 to test whether language behaves as a factor.

  extreme arm    5 directions x 6 languages x N samples
                 -> 150 at N=5.  Technique here is not prescribed. Each direction
                 asks the model to maximise one attribute (brevity,
                 elementarity, generality, visuality, machinery). The idea here is these may be the vertices of the hull

Total at N=5: 450 proofs

Note we are sticking with one model, Opus 5 Medium. This is because we don't want model to be a factor, and Opus 5 is better at instruction following
for things like "the most elementary proof"

Output: proofs.jsonl (ie one JSON object per proof, with metadata and usage).

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...   # or put it in secrets.env
    uv sync
    python generate.py --dry-run          # print the grid that is going to be called, but does no API calls
    python generate.py --submit           # submit the batch, print batch id
    python generate.py --collect          # poll and write proofs.jsonl
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
# Raised from 10000 after the first run truncated 28 records, 83% of them
# in the generality direction. See the corpus note in README.md.
MAX_TOKENS = 32000

OUT = Path("proofs.jsonl")
BATCH_ID_FILE = Path(".batch_id")
META_FILE = Path(".batch_meta.json")

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

# Extremal directions. The idea is that for the opposed pair elementarity vs
# machinery the hull will have some width along at least one axis
DIRECTIONS = {
    "brevity": "Give the shortest proof you can. Minimise total length. "
               "Do not sacrifice correctness or completeness for length, but "
               "subject to that, be as short as possible.",
    "elementarity": "Give the most elementary proof you can. Assume nothing "
                    "beyond the definition of divisibility and basic "
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
}

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

TECHNIQUE_TEMPLATE = """Write a complete, correct, self-contained proof that there are infinitely many prime numbers.

Proof technique (follow this approach and no other): {technique}

Language: write the entire proof in {language}.

Style: {style}

Output only the proof itself. No title, no preamble, no closing remarks."""

EXTREME_TEMPLATE = """Write a complete, correct, self-contained proof that there are infinitely many prime numbers.

Choose the proof yourself. You are not restricted to any particular argument.

Selection criterion: {direction}

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


def build_grid(n_samples: int):
    """This is a genrator that yields metadata dicts for both arms"""
    for tech, lang, style, k in itertools.product(
        TECHNIQUES, LANGUAGES, STYLES, range(n_samples)
    ):
        yield {
            "id": f"t__{tech}__{lang}__{style}__{k}",
            "arm": "technique",
            "theorem": "infinitude_of_primes",
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
            "theorem": "infinitude_of_primes",
            "technique": None,
            "direction": direction,
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
            technique=TECHNIQUES[meta["technique"]],
            language=LANGUAGES[meta["language"]],
            style=STYLES[meta["style"]],
        )
    return EXTREME_TEMPLATE.format(
        direction=DIRECTIONS[meta["direction"]],
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
    """ids already written to proofs.jsonl."""
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

    n_ok = n_err = 0
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

    print(f"\n{n_ok} succeeded, {n_err} failed. Corpus at {OUT.resolve()}")
    print(f"Tokens: {in_tok} in, {out_tok} out")
    BATCH_ID_FILE.unlink(missing_ok=True)      # <-- new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5,
                    help="samples per cell (default 5 -> 450 proofs total)")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_env()
    grid = list(build_grid(args.samples))
    n_tech = len(TECHNIQUES) * len(LANGUAGES) * len(STYLES) * args.samples
    n_ext = len(DIRECTIONS) * len(LANGUAGES) * args.samples
    print(f"Grid: {len(grid)} proofs "
          f"= {n_tech} technique arm + {n_ext} extreme arm "
          f"(model={MODEL}, effort={EFFORT})")

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