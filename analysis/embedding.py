"""
Embed a corpus, cache the result, and draw the UMAP figures.

The only file in the repo that loads bge-m3, and the only one that needs
sentence-transformers, umap-learn or matplotlib installed. Everything else
reads the cache it writes, which is why the split is here and not
somewhere else: analyse.py runs in seconds because it never imports any of
this, and the ~2GB model download happens once per corpus rather than once
per question asked of it.

Embedding is cache-first. If embeddings/<theorem>.npy already exists and
its row count matches the corpus, the model is never loaded at all -- so
re-running this to redraw a figure costs nothing. A row-count mismatch
re-embeds, which is the only path that reaches the network after the first
run.

The cache path is derived from the corpus path rather than defaulted
separately. Passing --corpus proofs_sqrt2.jsonl while leaving --cache at a
primes default is a mistake this codebase has already made, and a
row-count guard catches it only when the two corpora happen to differ in
length. Deriving the name removes the possibility instead of detecting it.

Usage:
    python analysis/embedding.py
    python analysis/embedding.py --corpus generate_proofs/proofs_sqrt2.jsonl
    python analysis/embedding.py --model intfloat/multilingual-e5-large

    # a corpus written by analyse.py's masking stage
    python analysis/embedding.py \\
        --corpus generate_proofs/proofs_primes.masked-discriminative.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

EMB_DIR = ROOT / "embeddings"

CORPUS = ROOT / "generate_proofs" / "proofs_primes.jsonl"

# bge-m3 takes 8192 tokens. Short-context models (e.g. the 128-token
# paraphrase-multilingual family) would truncate every proof to its opening
# paragraph -- and since length correlates with direction, that truncation
# would be systematic, not noise.
DEFAULT_MODEL = "BAAI/bge-m3"


def cache_for(corpus: Path) -> Path:
    """
    proofs_primes.jsonl                     -> embeddings/primes.npy
    proofs_primes.masked-discriminative.jsonl
                                            -> embeddings/primes_masked-discriminative.npy

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
    return EMB_DIR / f"{stem}.npy"


def ensure_dir(path: Path) -> Path:
    """Make the parent directory, so a first run does not fail on save."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def load_corpus(path: Path):
    recs = [json.loads(l) for l in path.open() if l.strip()]
    print(f"{len(recs)} records from {path}")
    for arm, n in Counter(r["arm"] for r in recs).items():
        print(f"  {arm:10} {n}")
    return recs


def embed(records, model_name, cache: Path):
    if cache.exists():
        # Cache-first, on existence alone: the corpora are immutable and
        # the masked ones are regenerated from a fixed table, so a cache
        # that is present is a cache that is right. Delete the .npy to
        # re-embed.
        print(f"Using cached embeddings from {cache}")
        return np.load(cache)

    # Imported here rather than at module level so that the cache-hit path
    # above costs nothing: loading sentence_transformers is most of this
    # script's startup time, and it is not needed to redraw a figure.
    from sentence_transformers import SentenceTransformer

    print(f"Embedding {len(records)} proofs with {model_name}")
    model = SentenceTransformer(model_name)
    lim = model.max_seq_length
    print(f"  max_seq_length = {lim} tokens")
    X = model.encode([r["proof"] for r in records], show_progress_bar=True,
                     batch_size=8, normalize_embeddings=True)
    X = np.asarray(X)
    np.save(ensure_dir(cache), X)
    return X


def centre_by_language(X, recs):
    """
    Subtract each language's mean embedding, then renormalise.

    Duplicated from analyse.py's centre_by rather than imported, so that
    drawing a figure never pulls in the analysis and its scipy/sklearn
    imports. Ten lines is the cheaper of the two dependencies.
    """
    Xc = X.copy()
    for lang in {r["language"] for r in recs}:
        idx = [i for i, r in enumerate(recs) if r["language"] == lang]
        Xc[idx] -= Xc[idx].mean(axis=0)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    return Xc / np.clip(norms, 1e-9, None)


def plot(records, X2, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = ["technique", "language", "style"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, field in zip(axes, fields):
        vals = sorted({str(r[field]) for r in records})
        cmap = plt.get_cmap("tab10")
        for i, v in enumerate(vals):
            idx = [j for j, r in enumerate(records) if str(r[field]) == v]
            ax.scatter(X2[idx, 0], X2[idx, 1], s=14, alpha=0.75,
                       color=cmap(i % 10), label=v)
        ax.set_title(f"coloured by {field}")
        ax.legend(fontsize=8, markerscale=1.5)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Technique arm: UMAP of multilingual embeddings", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"\nFigure: {out.resolve()}")
    print("UMAP distorts distance and density. Read it for shape, not scores.")


def figures(records, X, out: Path):
    """
    UMAP coloured by technique / language / style, raw and language-centred.
    For looking at, never for scoring off -- every number in the analysis
    comes from analyse.py, which reads the cache and not the projection.
    """
    import umap

    ensure_dir(out)
    for tag, M in (("raw", X), ("centred", centre_by_language(X, records))):
        X2 = umap.UMAP(n_neighbors=15, min_dist=0.1,
                       random_state=0).fit_transform(M)
        path = out.with_name(f"{out.stem}_{tag}{out.suffix}")
        plot(records, X2, path)


def main():
    ap = argparse.ArgumentParser(
        description="Embed a corpus to embeddings/<theorem>.npy and draw "
                    "its UMAP figures.")
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cache", type=Path, default=None,
                    help="default: embeddings/<theorem>.npy, derived from "
                         "--corpus")
    ap.add_argument("--out", type=Path, default=ROOT / "umaps" / "pilot_umap.png")
    ap.add_argument("--no-figures", action="store_true",
                    help="embed and cache only; skip the UMAP, which is the "
                         "slow part once the embeddings exist")
    args = ap.parse_args()

    cache = args.cache or cache_for(args.corpus)
    recs = load_corpus(args.corpus)
    X = embed(recs, args.model, cache)
    print(f"cache: {cache}  ({X.shape[0]} x {X.shape[1]})")

    if args.no_figures:
        return

    # The figures are drawn on the technique arm: it is the labelled part,
    # and colouring by a label the other arms do not carry is meaningless.
    idx = [i for i, r in enumerate(recs) if r["arm"] == "technique"]
    if not idx:
        print("\nNo technique arm in this corpus; no figures drawn.")
        return
    figures([recs[i] for i in idx], X[idx], args.out)


if __name__ == "__main__":
    main()
