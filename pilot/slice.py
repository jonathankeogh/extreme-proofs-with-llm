"""
Proof-space pilot, stage 3: slice the space.

Uses the CACHED embeddings.npy from analyse.py -- no API calls, no
re-embedding. Runs in seconds.

Answers four questions the headline numbers can't:

  1. Does technique structure depend on style?
     (Prediction: technique is much stronger in TERSE proofs, because
     notation is language-invariant and technique-specific, while verbose
     prose is dominated by natural language.)

  2. Does technique structure survive holding language fixed?
     (Restrict to English only; if technique still clusters, the structure
     is not a language artifact.)

  3. Does technique transfer ACROSS languages?
     (Train the probe on English proofs, test on the other five. This is
     the strong test: transfer means the space encodes the mathematics,
     not the words.)

  4. Which techniques are geometrically extreme?
     (Conway's extreme points: mean distance from every other technique's
     centroid, plus which proofs the technique probe gets wrong -- likely
     corpus-quality failures where the generator ignored the instruction.)

Usage:
    python slices.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score,
                             normalized_mutual_info_score,
                             silhouette_score)
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import LabelEncoder

CORPUS = Path("proofs.jsonl")
EMB = Path("embeddings.npy")


def load():
    records = [json.loads(l) for l in CORPUS.open() if l.strip()]
    X = np.load(EMB)
    assert len(records) == len(X), (
        f"{len(records)} proofs but {len(X)} embeddings -- rerun analyze.py")
    return records, X


def scores(X, y, name, indent="  "):
    """Return (and print) clustering + probe scores for one labeling."""
    le = LabelEncoder()
    yi = le.fit_transform(y)
    k = len(le.classes_)
    if k < 2 or len(yi) < 10:
        print(f"{indent}{name:<24} (skipped: {k} classes, {len(yi)} points)")
        return None
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    out = {
        "n": len(yi),
        "k": k,
        "ari": adjusted_rand_score(yi, km.labels_),
        "nmi": normalized_mutual_info_score(yi, km.labels_),
        "sil": silhouette_score(X, yi),
        "probe": cross_val_score(
            LogisticRegression(max_iter=2000), X, yi, cv=5).mean(),
        "chance": 1 / k,
    }
    out["lift"] = out["probe"] / out["chance"]
    print(f"{indent}{name:<24} n={out['n']:<4} k={k}  "
          f"ARI={out['ari']:.3f}  NMI={out['nmi']:.3f}  "
          f"sil={out['sil']:+.3f}  probe={out['probe']:.3f} "
          f"(chance {out['chance']:.3f}, lift {out['lift']:.1f}x)")
    return out


def q1_style_slices(records, X):
    print("\n=== Q1. Technique structure within each style ===")
    print("Prediction: technique is stronger in terse (notation-dense) "
          "than verbose (prose-dense).\n")
    results = {}
    for style in sorted({r["style"] for r in records}):
        idx = [i for i, r in enumerate(records) if r["style"] == style]
        print(f"  [{style}]")
        for label in ("technique", "language"):
            y = [records[i][label] for i in idx]
            results[(style, label)] = scores(X[idx], y, label, indent="    ")
        print()

    t_terse = results.get(("terse", "technique"))
    t_verb = results.get(("verbose", "technique"))
    if t_terse and t_verb:
        d = t_terse["nmi"] - t_verb["nmi"]
        print(f"  Technique NMI: terse {t_terse['nmi']:.3f} vs verbose "
              f"{t_verb['nmi']:.3f}  (difference {d:+.3f})")
        print("  -> Prediction CONFIRMED: notation carries the mathematics."
              if d > 0.05 else
              "  -> Prediction NOT confirmed; style does not modulate "
              "technique structure much.")


def q2_within_language(records, X):
    print("\n=== Q2. Technique structure with language held fixed ===")
    print("If technique clusters within a single language, the structure "
          "is not a language artifact.\n")
    for lang in sorted({r["language"] for r in records}):
        idx = [i for i, r in enumerate(records) if r["language"] == lang]
        y = [records[i]["technique"] for i in idx]
        scores(X[idx], y, f"technique | lang={lang}", indent="  ")


def q3_cross_lingual_transfer(records, X):
    print("\n=== Q3. Cross-lingual technique transfer (the strong test) ===")
    print("Train the technique probe on ENGLISH only, test on each other "
          "language.\n")
    le = LabelEncoder().fit([r["technique"] for r in records])
    tr = [i for i, r in enumerate(records) if r["language"] == "en"]
    if len(tr) < 20:
        print("  (not enough English proofs)")
        return
    clf = LogisticRegression(max_iter=2000).fit(
        X[tr], le.transform([records[i]["technique"] for i in tr]))
    chance = 1 / len(le.classes_)
    accs = []
    for lang in sorted({r["language"] for r in records} - {"en"}):
        idx = [i for i, r in enumerate(records) if r["language"] == lang]
        yt = le.transform([records[i]["technique"] for i in idx])
        acc = clf.score(X[idx], yt)
        accs.append(acc)
        print(f"  en -> {lang:<4} n={len(idx):<4} acc={acc:.3f} "
              f"(chance {chance:.3f}, lift {acc/chance:.1f}x)")
    mean = float(np.mean(accs))
    print(f"\n  Mean transfer accuracy: {mean:.3f}")
    if mean > 0.6:
        print("  -> Technique survives the language barrier. The space "
              "encodes mathematics, not vocabulary.")
    elif mean > 2 * chance:
        print("  -> Partial transfer: some technique signal is "
              "language-independent, but much is not.")
    else:
        print("  -> Little transfer: technique structure is largely "
              "language-bound.")


def q4_extreme_points(records, X):
    print("\n=== Q4. Which techniques are geometrically extreme? ===")
    print("Conway's extreme points: how far is each technique's centroid "
          "from the others?\n")
    techs = sorted({r["technique"] for r in records})
    cents = {}
    for t in techs:
        idx = [i for i, r in enumerate(records) if r["technique"] == t]
        cents[t] = X[idx].mean(axis=0)

    rows = []
    for t in techs:
        others = [cents[o] for o in techs if o != t]
        d = float(np.mean([np.linalg.norm(cents[t] - o) for o in others]))
        idx = [i for i, r in enumerate(records) if r["technique"] == t]
        spread = float(np.mean(np.linalg.norm(X[idx] - cents[t], axis=1)))
        rows.append((d, spread, t))
    rows.sort(reverse=True)

    print(f"  {'technique':<18} {'dist to others':>14} {'own spread':>12}")
    for d, spread, t in rows:
        print(f"  {t:<18} {d:>14.3f} {spread:>12.3f}")
    print(f"\n  Most extreme: {rows[0][2]}  (candidate 'extreme point' in "
          f"Conway's sense)")
    print(f"  Most central: {rows[-1][2]}")

    # Where does the technique probe fail? Those are corpus-quality suspects.
    print("\n  --- Technique probe errors (possible instruction-following "
          "failures) ---")
    le = LabelEncoder()
    y = le.fit_transform([r["technique"] for r in records])
    pred = cross_val_predict(
        LogisticRegression(max_iter=2000), X, y, cv=5)
    wrong = [(records[i]["id"], le.classes_[y[i]], le.classes_[pred[i]])
             for i in range(len(y)) if pred[i] != y[i]]
    print(f"  {len(wrong)}/{len(y)} misclassified "
          f"({100*len(wrong)/len(y):.1f}%)")
    conf = Counter((t, p) for _, t, p in wrong)
    for (t, p), c in conf.most_common(8):
        print(f"    {t} -> predicted {p}  ({c}x)")
    print("\n  Read a few of these proofs by hand. If the generator "
          "actually followed a different technique than requested, the "
          "LABEL is wrong, not the embedding:")
    for pid, t, p in wrong[:8]:
        print(f"    {pid}   (labelled {t}, looks like {p})")


def main():
    records, X = load()
    print(f"Loaded {len(records)} proofs, embeddings {X.shape}")
    q1_style_slices(records, X)
    q2_within_language(records, X)
    q3_cross_lingual_transfer(records, X)
    q4_extreme_points(records, X)
    print("\nDone. No API calls, no re-embedding -- rerun freely.")


if __name__ == "__main__":
    main()