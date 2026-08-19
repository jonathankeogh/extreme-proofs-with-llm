"""
Where the embedding caches live, and how their names are derived.

Kept in its own module rather than in analyse.py because probes.py,
extreme.py and mask.py all need it and none of them should pay for
analyse.py's imports: that file pulls in sentence_transformers and
matplotlib at module level, and the whole point of probes.py is that it
re-embeds nothing.

The cache path is derived from the corpus path rather than defaulted
separately. Passing --corpus proofs_sqrt2.jsonl while leaving --cache at a
primes default is a mistake this codebase has already made, and the row-count
guard in each loader catches it only when the two corpora happen to differ in
length. Deriving the name removes the possibility instead of detecting it.
"""

from pathlib import Path

EMB_DIR = Path("embeddings")


def cache_for(corpus: Path, suffix: str = "") -> Path:
    """
    proofs_primes.jsonl                     -> embeddings/primes.npy
    proofs_primes.masked-discriminative.jsonl
                                            -> embeddings/primes_masked-discriminative.npy

    `suffix` appends a tag before the extension, for callers that build a
    variant name themselves.
    """
    stem = Path(corpus).name
    for ext in (".jsonl", ".json"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    if stem.startswith("proofs_"):
        stem = stem[len("proofs_"):]
    # proofs_primes.masked-both -> primes_masked-both
    stem = stem.replace(".", "_")
    return EMB_DIR / f"{stem}{suffix}.npy"


def ensure_dir(path: Path) -> Path:
    """Make the parent directory, so a first run does not fail on save."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path
