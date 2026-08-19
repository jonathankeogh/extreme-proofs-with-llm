"""
Stage 2: embed the corpus and test whether the embedding encodes argument
structure or just language.

The technique arm (300 records) carries ground-truth labels: you know which
of five proofs each record is, in each of six languages. That makes it a
labelled test set for the embedding itself. The extreme arm (150 records)
has no technique label -- it is what you want to classify later, once the
instrument is validated.

Reported, in order:

  1. pooled probe      logistic regression, 5-fold CV, on technique / language
                       / style. Accuracy against chance, not NMI: NMI is not
                       comparable across labels with different class counts.

  2. LOLO probe        train on five languages, test on the sixth. This is
                       the load-bearing test. Pooled CV cannot distinguish
                       "technique separates within every language" from
                       "technique separates in English only" -- both give
                       high pooled accuracy. Leave-one-language-out can.

  3. neighbourhood     for each record, the fraction of its k nearest
                       neighbours sharing its technique vs its language.
                       Says which factor dominates the local geometry.

  4. centred repeat    the same scores after subtracting each language's
                       mean embedding. Same logic as the log-normalisation
                       on the length axis, now vectorial. The balanced grid
                       is what makes those means unbiased.

  5. figure            UMAP coloured by technique / language / style.
                       For looking at, never for scoring off.

Usage:
    pip install sentence-transformers umap-learn scikit-learn matplotlib
    python analyse.py
    python analyse.py --model intfloat/multilingual-e5-large
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from paths import cache_for, ensure_dir

# bge-m3 takes 8192 tokens. Short-context models (e.g. the 128-token
# paraphrase-multilingual family) would truncate every proof to its opening
# paragraph -- and since length correlates with direction, that truncation
# would be systematic, not noise.
DEFAULT_MODEL = "BAAI/bge-m3"


def load_corpus(path: Path):
    if not path.exists():
        raise SystemExit(f"{path} not found.")
    recs = [json.loads(l) for l in path.open() if l.strip()]
    print(f"{len(recs)} records from {path}")
    for arm, n in Counter(r["arm"] for r in recs).items():
        print(f"  {arm:10} {n}")
    return recs


def embed(records, model_name, cache: Path):
    if cache.exists():
        X = np.load(cache)
        if len(X) == len(records):
            print(f"Using cached embeddings from {cache}")
            return X
        print(f"{cache} has {len(X)} rows, corpus has {len(records)}. "
              f"Re-embedding.")

    print(f"Embedding {len(records)} proofs with {model_name}")
    model = SentenceTransformer(model_name)
    lim = model.max_seq_length
    print(f"  max_seq_length = {lim} tokens")
    X = model.encode([r["proof"] for r in records], show_progress_bar=True,
                     batch_size=8, normalize_embeddings=True)
    X = np.asarray(X)
    np.save(ensure_dir(cache), X)
    return X


def probe(X, y, name):
    """5-fold CV accuracy of a linear probe, against chance."""
    yi = LabelEncoder().fit_transform(y)
    k = len(set(yi))
    acc = cross_val_score(LogisticRegression(max_iter=3000), X, yi, cv=5).mean()
    sil = silhouette_score(X, yi)
    print(f"  {name:<10} k={k}  acc={acc:.3f}  chance={1/k:.3f}  "
          f"lift={acc - 1/k:+.3f}  silhouette={sil:+.3f}")
    return acc - 1 / k


def lolo(X, recs, target="technique"):
    """Train on five languages, test on the held-out sixth."""
    langs = sorted({r["language"] for r in recs})
    y = LabelEncoder().fit_transform([r[target] for r in recs])
    chance = 1 / len(set(y))
    accs = []
    for held in langs:
        tr = [i for i, r in enumerate(recs) if r["language"] != held]
        te = [i for i, r in enumerate(recs) if r["language"] == held]
        clf = LogisticRegression(max_iter=3000).fit(X[tr], y[tr])
        a = clf.score(X[te], y[te])
        accs.append(a)
        print(f"  hold out {held}   acc={a:.3f}")
    print(f"  mean {np.mean(accs):.3f}  min {min(accs):.3f}  "
          f"chance {chance:.3f}")
    return np.mean(accs)


def neighbourhood(X, recs, k=10):
    """Fraction of k nearest neighbours sharing technique vs language."""
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, axis=1)[:, :k]
    share = {}
    for field in ("technique", "language", "style"):
        vals = [r[field] for r in recs]
        share[field] = np.mean([
            np.mean([vals[j] == vals[i] for j in nn[i]])
            for i in range(len(recs))
        ])
        base = sum(c * (c - 1) for c in Counter(vals).values()) / (
            len(recs) * (len(recs) - 1))
        print(f"  {field:<10} same-label among {k}-NN: {share[field]:.3f}  "
              f"(baseline {base:.3f})")
    return share


def centre_by_language(X, recs):
    """Subtract each language's mean embedding, then renormalise."""
    Xc = X.copy()
    for lang in {r["language"] for r in recs}:
        idx = [i for i, r in enumerate(recs) if r["language"] == lang]
        Xc[idx] -= Xc[idx].mean(axis=0)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    return Xc / np.clip(norms, 1e-9, None)


def plot(records, X2, out: Path):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("proofs_primes.jsonl"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cache", type=Path, default=None,
                    help="default: embeddings/<theorem>.npy, derived from "
                         "--corpus")
    ap.add_argument("--out", type=Path, default=Path("pilot_umap.png"))
    ap.add_argument("--knn", type=int, default=10)
    args = ap.parse_args()

    cache = args.cache or cache_for(args.corpus)
    recs = load_corpus(args.corpus)
    X = embed(recs, args.model, cache)

    # Scoring uses the technique arm only: it is the labelled part.
    idx = [i for i, r in enumerate(recs) if r["arm"] == "technique"]
    T = [recs[i] for i in idx]
    XT = X[idx]
    print(f"\nScoring on the technique arm: {len(T)} labelled records")

    print("\n1. Pooled linear probe (5-fold CV)")
    lifts = {f: probe(XT, [r[f] for r in T], f)
             for f in ("technique", "language", "style")}

    print("\n2. Leave-one-language-out probe, target = technique")
    lolo_acc = lolo(XT, T)

    print(f"\n3. Neighbourhood composition (k={args.knn})")
    neighbourhood(XT, T, args.knn)

    print("\n4. After subtracting each language's mean embedding")
    XC = centre_by_language(XT, T)
    for f in ("technique", "language", "style"):
        probe(XC, [r[f] for r in T], f)
    print("  LOLO on centred embeddings:")
    lolo(XC, T)

    print("\nRead:")
    print(f"  technique lift {lifts['technique']:+.3f} vs language lift "
          f"{lifts['language']:+.3f}")
    print(f"  LOLO technique accuracy {lolo_acc:.3f} (chance 0.200)")
    print("  LOLO near pooled accuracy -> argument structure is")
    print("  language-invariant in this space, and the extreme arm can be")
    print("  classified against these five centroids.")
    print("  LOLO collapsing toward chance -> the probe learned")
    print("  language-specific surface features. Use the centred embeddings,")
    print("  and if that does not recover it, the embedding is not a usable")
    print("  instrument for this question -- a fact about the model, not")
    print("  about proofs.")

    import umap
    for tag, M in (("raw", XT), ("centred", XC)):
        X2 = umap.UMAP(n_neighbors=15, min_dist=0.1,
                       random_state=0).fit_transform(M)
        out = args.out.with_name(f"{args.out.stem}_{tag}{args.out.suffix}")
        plot(T, X2, out)


if __name__ == "__main__":
    main()