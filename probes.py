"""
Stage 2b: structural probes on the cached embeddings.

Runs on embeddings.npy + proofs.jsonl. No re-embedding, no API calls, no
model download -- seconds, not minutes.

  1. language_spectrum   How many dimensions does language actually occupy?
                         SVD of the six language means. "Additive" (which
                         the centring result established) is weaker than
                         "one-dimensional", and the UMAP cannot tell them
                         apart: 2D projection shows the largest component,
                         so a dominant script direction looks like the only
                         direction whether or not it is.

  2. style_spectrum      Same question for style. The centred UMAP shows
                         terse and verbose separating as cleanly as
                         techniques do, so style is not a small effect.

  3. subspace_angles     Are the language and style subspaces orthogonal to
                         the technique subspace, or merely separable?
                         Principal angles answer this directly.

  4. variance_budget     Of total embedding variance, how much is explained
                         by each factor? Puts the three on one scale.

  5. centroid_agreement  THE ONE THAT MATTERS FOR STAGE 3. Technique
                         centroids built from terse records only vs verbose
                         records only. If they disagree about held-out
                         records, pooled centroids are unsafe for
                         classifying the extreme arm, which carries no style
                         label at all.

  6. hard_probe          Probe accuracy after PCA to few dimensions and
                         after subsampling to one record per cell. The
                         1.000s in the previous stage are uninformative: 300
                         points with five near-duplicates per cell are
                         trivially separable in 1024 dimensions. This makes
                         the task hard enough for the number to mean
                         something.

  7. within_vs_between   Cosine similarity within a technique (across
                         languages) vs between techniques. Scale-free, does
                         not saturate, and directly comparable across the
                         raw and centred spaces.

  8. angle_null          Nulls for item 3. Permuting the technique labels
                         within each (language, style) cell barely moves the
                         angle, and a random subspace of the same rank sits
                         higher. Item 3 measures the dimension, not the
                         design.

  9. lexical_baseline    Can TF-IDF alone recover the technique within one
                         language? Trains on terse, tests on verbose, so no
                         cell appears on both sides. Bounds how much of the
                         cross-language result is terminology.

 10. style_slices        Does technique structure depend on style? Notation
                         is shared across languages and is technique-
                         specific; prose is neither.

 11. cross_lingual       Technique probe trained on one language, applied
                         frozen to the rest, on RAW embeddings. Centring
                         would make this trivial and would not say whether
                         the encoder aligns languages or the centring does.

 12. within_language     Technique clustering with language held fixed.
                         Unsupervised scores only -- the probe saturates.

 13. extremal_ranking    Techniques ranked by centroid distance from the
                         others, with own-spread alongside as the caveat.
                         Explains what the machinery arm attaches to in
                         stage 3.

Usage:
    python probes.py
    python probes.py --arm technique --knn 10
"""

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import LabelEncoder

FACTORS = ("technique", "language", "style")


def load(corpus: Path, cache: Path, arm: str):
    recs = [json.loads(l) for l in corpus.open() if l.strip()]
    X = np.load(cache)
    if len(X) != len(recs):
        raise SystemExit(
            f"{cache} has {len(X)} rows, {corpus} has {len(recs)}. "
            f"Re-run the embedding step."
        )
    idx = [i for i, r in enumerate(recs) if r["arm"] == arm]
    return [recs[i] for i in idx], X[idx]


def centre_by_loo(X, recs, field):
    """
    Leave-one-out centring: each record is centred by the mean of its own
    level EXCLUDING itself.

    Plain centre_by forces each level mean to exactly zero, so a language
    probe on the result is guaranteed to collapse -- by construction, not by
    discovery. LOO removes that guarantee: the offset applied to a record is
    estimated from the other 49 records in its language, so a probe that
    still fails is failing on held-out information.
    """
    Xc = X.copy()
    for lvl in {r[field] for r in recs}:
        i = np.array([j for j, r in enumerate(recs) if r[field] == lvl])
        s, n = X[i].sum(0), len(i)
        Xc[i] = X[i] - (s - X[i]) / (n - 1)
    return Xc / np.clip(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-9, None)


def centre_by(X, recs, field):
    """Subtract the mean embedding of each level of `field`, renormalise."""
    Xc = X.copy()
    for lvl in {r[field] for r in recs}:
        i = [j for j, r in enumerate(recs) if r[field] == lvl]
        Xc[i] -= Xc[i].mean(axis=0)
    n = np.linalg.norm(Xc, axis=1, keepdims=True)
    return Xc / np.clip(n, 1e-9, None)


def means_matrix(X, recs, field):
    """One row per level, centred on the grand mean of those rows."""
    levels = sorted({r[field] for r in recs})
    M = np.stack([X[[i for i, r in enumerate(recs) if r[field] == lvl]].mean(0)
                  for lvl in levels])
    return levels, M - M.mean(0)


def basis(X, recs, field, keep=None):
    """Orthonormal basis for the span of the level means."""
    _, M = means_matrix(X, recs, field)
    U, s, _ = np.linalg.svd(M.T, full_matrices=False)
    k = keep if keep is not None else int((s > 1e-8).sum())
    return U[:, :k]


def rule(t):
    print(f"\n{t}\n" + "-" * len(t))


# ---------------------------------------------------------------- 1, 2

def spectrum(X, recs, field):
    levels, M = means_matrix(X, recs, field)
    s = np.linalg.svd(M, compute_uv=False)
    var = s ** 2 / (s ** 2).sum()
    print(f"  {field}: {len(levels)} levels, rank <= {len(levels) - 1}")
    print("  variance fraction per component:")
    print("   ", "  ".join(f"{v:.3f}" for v in var[:len(levels) - 1]))
    print(f"  cumulative: "
          f"{'  '.join(f'{c:.3f}' for c in np.cumsum(var[:len(levels)-1]))}")
    if var[0] > 0.85:
        print(f"  -> effectively one-dimensional ({var[0]:.1%} in one "
              f"direction).")
    elif var[0] + var[1] > 0.85:
        print(f"  -> two dominant directions "
              f"({var[0] + var[1]:.1%} in the first two).")
    else:
        print("  -> genuinely multidimensional; no single direction "
              "dominates.")

    # Which levels sit at the ends of the leading direction?
    U, _, _ = np.linalg.svd(M.T, full_matrices=False)
    proj = M @ U[:, 0]
    order = np.argsort(proj)
    print("  leading direction, levels ordered:")
    print("   ", "  ".join(f"{levels[i]}({proj[i]:+.2f})" for i in order))
    return var


# ---------------------------------------------------------------- 3

def subspace_angles(X, recs):
    print("  Principal angles between factor subspaces (degrees).")
    print("  90 = orthogonal (factors independent in this space);")
    print("  small = the two factors share directions.")
    B = {f: basis(X, recs, f) for f in FACTORS}
    for a, b in combinations(FACTORS, 2):
        s = np.linalg.svd(B[a].T @ B[b], compute_uv=False)
        ang = np.degrees(np.arccos(np.clip(s, -1, 1)))
        print(f"  {a:<10} vs {b:<10} "
              f"min={ang.min():5.1f}  mean={ang.mean():5.1f}  "
              f"max={ang.max():5.1f}")


# ---------------------------------------------------------------- 4

def variance_budget(X, recs):
    print("  Fraction of total variance explained by each factor")
    print("  (between-level variance / total variance).")
    Xc = X - X.mean(0)
    total = (Xc ** 2).sum()
    for f in FACTORS:
        levels = sorted({r[f] for r in recs})
        ss = 0.0
        for lvl in levels:
            i = [j for j, r in enumerate(recs) if r[f] == lvl]
            ss += len(i) * ((X[i].mean(0) - X.mean(0)) ** 2).sum()
        print(f"  {f:<10} {ss / total:.3f}")
    # Joint technique x style, to see whether they add or interact
    ss = 0.0
    cells = defaultdict(list)
    for j, r in enumerate(recs):
        cells[(r["technique"], r["style"])].append(j)
    for i in cells.values():
        ss += len(i) * ((X[i].mean(0) - X.mean(0)) ** 2).sum()
    print(f"  {'tech x style':<10} {ss / total:.3f}   "
          f"(compare with the sum of the two above)")


# ---------------------------------------------------------------- 5

def centroid_agreement(X, recs):
    print("  Technique centroids built from terse records only, and from")
    print("  verbose records only. Each record is then assigned to its")
    print("  nearest centroid under each set. Disagreement means pooled")
    print("  centroids are unsafe for the extreme arm, which has no style.")
    techs = sorted({r["technique"] for r in recs})

    def centroids(style):
        C = []
        for t in techs:
            i = [j for j, r in enumerate(recs)
                 if r["technique"] == t and r["style"] == style]
            C.append(X[i].mean(0))
        C = np.stack(C)
        return C / np.clip(np.linalg.norm(C, axis=1, keepdims=True), 1e-9, None)

    Ct, Cv = centroids("terse"), centroids("verbose")
    at = np.argmax(X @ Ct.T, axis=1)
    av = np.argmax(X @ Cv.T, axis=1)
    truth = np.array([techs.index(r["technique"]) for r in recs])

    print(f"  terse-built centroids   accuracy {np.mean(at == truth):.3f}")
    print(f"  verbose-built centroids accuracy {np.mean(av == truth):.3f}")
    print(f"  the two agree on        {np.mean(at == av):.3f} of records")

    # Cross-style: does a terse-built centroid classify verbose records?
    for style, C, name in (("verbose", Ct, "terse-built -> verbose records"),
                           ("terse", Cv, "verbose-built -> terse records")):
        i = [j for j, r in enumerate(recs) if r["style"] == style]
        a = np.argmax(X[i] @ C.T, axis=1)
        print(f"  {name:<32} {np.mean(a == truth[i]):.3f}")

    # How far apart are the two style sub-clusters of one technique,
    # relative to the distance between techniques?
    within = [float(Ct[i] @ Cv[i]) for i in range(len(techs))]
    between = [float(Ct[i] @ Ct[j])
               for i in range(len(techs)) for j in range(len(techs)) if i != j]
    print(f"\n  cos(terse_t, verbose_t)  same technique, mean "
          f"{np.mean(within):.3f}")
    print(f"  cos(terse_t, terse_u)    diff technique, mean "
          f"{np.mean(between):.3f}")
    if np.mean(within) < np.mean(between):
        print("  -> style splits a technique further than technique splits "
              "the space. Pool with care.")


# ---------------------------------------------------------------- 6

def hard_probe(X, recs, seed=0):
    print("  Probe accuracy is uninformative at full dimension: 300 points")
    print("  with five near-duplicates per cell are trivially separable in")
    print("  1024 dimensions. Two ways to make the task honest.")
    rng = np.random.default_rng(seed)

    print("\n  (a) after PCA, all 300 records")
    print(f"  {'dims':>5}  " + "  ".join(f"{f:>10}" for f in FACTORS))
    for d in (2, 5, 10, 20, 50):
        Xd = PCA(n_components=d, random_state=seed).fit_transform(X)
        row = []
        for f in FACTORS:
            y = LabelEncoder().fit_transform([r[f] for r in recs])
            row.append(cross_val_score(
                LogisticRegression(max_iter=3000), Xd, y, cv=5).mean())
        print(f"  {d:5d}  " + "  ".join(f"{a:10.3f}" for a in row))
    print("  chance:  " + "  ".join(
        f"{1/len({r[f] for r in recs}):10.3f}" for f in FACTORS))

    print("\n  (b) one record per cell (removes near-duplicate inflation)")
    cells = defaultdict(list)
    for j, r in enumerate(recs):
        cells[(r["technique"], r["language"], r["style"])].append(j)
    keep = [rng.choice(v) for v in cells.values()]
    Xs, Rs = X[keep], [recs[j] for j in keep]
    print(f"  n = {len(keep)}")
    for d in (5, 10, 20):
        Xd = PCA(n_components=min(d, len(keep) - 1),
                 random_state=seed).fit_transform(Xs)
        row = []
        for f in FACTORS:
            y = LabelEncoder().fit_transform([r[f] for r in Rs])
            row.append(cross_val_score(
                LogisticRegression(max_iter=3000), Xd, y, cv=5).mean())
        print(f"  {d:5d}  " + "  ".join(f"{a:10.3f}" for a in row))


# ---------------------------------------------------------------- 7

def within_vs_between(X, recs):
    print("  Mean cosine similarity between pairs of records, split by")
    print("  whether they share a factor level. Does not saturate.")
    S = X @ X.T
    n = len(recs)
    iu = np.triu_indices(n, k=1)
    sims = S[iu]
    for f in FACTORS:
        v = np.array([r[f] for r in recs])
        same = (v[iu[0]] == v[iu[1]])
        print(f"  {f:<10} same {sims[same].mean():+.3f}   "
              f"diff {sims[~same].mean():+.3f}   "
              f"gap {sims[same].mean() - sims[~same].mean():+.3f}")

    # Same technique but different language: the quantity the whole
    # six-language design rests on.
    tech = np.array([r["technique"] for r in recs])
    lang = np.array([r["language"] for r in recs])
    m = (tech[iu[0]] == tech[iu[1]]) & (lang[iu[0]] != lang[iu[1]])
    m2 = (tech[iu[0]] != tech[iu[1]]) & (lang[iu[0]] == lang[iu[1]])
    print(f"\n  same technique, different language  {sims[m].mean():+.3f}")
    print(f"  different technique, same language  {sims[m2].mean():+.3f}")
    if sims[m].mean() > sims[m2].mean():
        print("  -> a proof is closer to the same argument in another")
        print("     language than to a different argument in its own.")


# ---------------------------------------------------------------- 8

def angle_null(X, recs, a="technique", b="language", n_perm=200, seed=0):
    """
    Is 83 degrees a finding, or is it what any two subspaces give in 1024
    dimensions? Two nulls:

      permuted  -- shuffle the `a` labels across records, keeping the design
                   intact, and recompute. Tests whether the observed angles
                   depend on the labels at all.
      random    -- a random subspace of the same rank. Tests whether the
                   observed angles are just high-dimensional geometry.

    Report the observed MIN angle against the null distribution. The mean is
    the wrong statistic: shared structure shows up in the smallest angle.
    """
    rng = np.random.default_rng(seed)
    Ba, Bb = basis(X, recs, a), basis(X, recs, b)
    ka = Ba.shape[1]

    def min_angle(U, V):
        s = np.linalg.svd(U.T @ V, compute_uv=False)
        return float(np.degrees(np.arccos(np.clip(s, -1, 1))).min())

    obs = min_angle(Ba, Bb)

    cells = defaultdict(list)
    for j, r in enumerate(recs):
        cells[(r["language"], r["style"])].append(j)

    perm = []
    shuffled = [dict(r) for r in recs]
    for _ in range(n_perm):
        for idxs in cells.values():
            labs_c = [recs[j][a] for j in idxs]
            for j, lab in zip(idxs, rng.permutation(labs_c)):
                shuffled[j][a] = lab
        perm.append(min_angle(basis(X, shuffled, a), Bb))

    rand = []
    for _ in range(n_perm):
        U, _ = np.linalg.qr(rng.standard_normal((X.shape[1], ka)))
        rand.append(min_angle(U, Bb))

    perm, rand = np.array(perm), np.array(rand)
    print(f"  observed min angle, {a} vs {b}:  {obs:.1f} deg")
    print(f"  permuted-label null:  mean {perm.mean():.1f}  "
          f"2.5th pct {np.percentile(perm, 2.5):.1f}")
    print(f"  random-subspace null: mean {rand.mean():.1f}  "
          f"2.5th pct {np.percentile(rand, 2.5):.1f}")
    print(f"  p(null <= observed), permuted {np.mean(perm <= obs):.3f}  "
          f"random {np.mean(rand <= obs):.3f}")
    print("  -> if the nulls sit at the same angle, orthogonality is a")
    print("     property of the dimension, not of the design.")


# ---------------------------------------------------------------- 9

def lexical_baseline(X, recs, lang="en", seed=0):
    """
    Bounds the vocabulary-vs-structure confound. Within one language, can
    character n-grams alone recover the technique? If yes, technique
    identity is carried by terminology, and the embedding's cross-language
    matching is plausibly multilingual term alignment rather than argument
    structure.

    Folds hold out whole prompt cells (technique x style). Plain 5-fold CV
    puts four near-duplicate samples of a cell in train and the fifth in
    test, which any representation solves trivially.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline

    i = [j for j, r in enumerate(recs) if r["language"] == lang]
    txt = [recs[j]["proof"] for j in i]
    y = LabelEncoder().fit_transform([recs[j]["technique"] for j in i])
    styles = [recs[j]["style"] for j in i]
    print(f"  {len(i)} records in '{lang}', chance {1/len(set(y)):.3f}")

    def split_score(fit_texts, fit_y, test_texts, test_y, pipe):
        pipe.fit(fit_texts, fit_y)
        return pipe.score(test_texts, test_y)

    def by_style(make_pipe, vecs=None):
        accs = []
        for tr_style, te_style in (("terse", "verbose"), ("verbose", "terse")):
            tr = [k for k, s in enumerate(styles) if s == tr_style]
            te = [k for k, s in enumerate(styles) if s == te_style]
            if vecs is None:
                accs.append(split_score([txt[k] for k in tr], y[tr],
                                        [txt[k] for k in te], y[te],
                                        make_pipe()))
            else:
                clf = LogisticRegression(max_iter=3000).fit(vecs[tr], y[tr])
                accs.append(clf.score(vecs[te], y[te]))
        return float(np.mean(accs))

    for kind, kw in (("word 1-2gram", dict(analyzer="word", ngram_range=(1, 2))),
                     ("char 3-5gram", dict(analyzer="char_wb",
                                           ngram_range=(3, 5)))):
        acc = by_style(lambda kw=kw: make_pipeline(
            TfidfVectorizer(min_df=2, **kw),
            LogisticRegression(max_iter=3000)))
        print(f"  tfidf {kind:<14} {acc:.3f}")

    print(f"  bge-m3 embedding      {by_style(None, vecs=X[i]):.3f}")
    print("  -> tfidf near the embedding: technique is lexical here.")


# ---------------------------------------------------------------- 10

def style_slices(X, recs):
    """
    Does technique structure depend on style? Notation is shared across all
    six languages and is technique-specific; prose is neither. So terse
    proofs should carry more technique signal and less language signal than
    verbose ones. Predicted before the corpus was generated.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score

    print(f"  {'slice':<10}{'technique NMI':>15}{'technique probe':>17}"
          f"{'language probe':>16}")
    out = {}
    for style in sorted({r["style"] for r in recs}):
        i = [j for j, r in enumerate(recs) if r["style"] == style]
        row = []
        for f in ("technique", "language"):
            y = LabelEncoder().fit_transform([recs[j][f] for j in i])
            k = len(set(y))
            acc = cross_val_score(LogisticRegression(max_iter=3000),
                                  X[i], y, cv=5).mean()
            if f == "technique":
                km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X[i])
                nmi = normalized_mutual_info_score(y, km.labels_)
                out[style] = nmi
                row += [nmi, acc]
            else:
                row.append(acc)
        print(f"  {style:<10}{row[0]:>15.3f}{row[1]:>17.3f}{row[2]:>16.3f}")
    if len(out) == 2:
        t, v = out.get("terse"), out.get("verbose")
        print(f"\n  technique NMI difference (terse - verbose): {t - v:+.3f}")
        print("  -> positive: notation carries the mathematics, prose the "
              "language.")


# ---------------------------------------------------------------- 11

def cross_lingual_transfer(X, recs, train_lang="en"):
    """
    Train the technique probe on one language, apply it frozen to the rest.
    The strong test: English and Japanese share no script and no cognate
    vocabulary, so a direction separating Euclid from Furstenberg in both
    is not riding on English terms.

    Run on RAW embeddings. Language centring would make this trivial and
    would not tell you whether the encoder aligns languages or whether the
    centring does.
    """
    le = LabelEncoder().fit([r["technique"] for r in recs])
    tr = [j for j, r in enumerate(recs) if r["language"] == train_lang]
    clf = LogisticRegression(max_iter=3000).fit(
        X[tr], le.transform([recs[j]["technique"] for j in tr]))
    chance = 1 / len(le.classes_)
    accs = []
    for lang in sorted({r["language"] for r in recs} - {train_lang}):
        i = [j for j, r in enumerate(recs) if r["language"] == lang]
        acc = clf.score(X[i], le.transform([recs[j]["technique"] for j in i]))
        accs.append(acc)
        print(f"  {train_lang} -> {lang:<4} n={len(i):<4} acc={acc:.3f}")
    print(f"\n  mean {np.mean(accs):.3f}   chance {chance:.3f}")
    print("  Degradation on zh/ja quantifies how much of the technique")
    print("  direction rides on shared Indo-European vocabulary.")


# ---------------------------------------------------------------- 12

def within_language(X, recs):
    """
    Technique structure with language held fixed. If technique clusters
    inside a single language, the structure is not a language artifact.

    Probe accuracy is NOT reported here. Fifty records in 1024 dimensions
    are linearly separable almost regardless of the labels -- the probe
    reads 1.000 in every language and carries no information (item 6 makes
    the same point at length). The unsupervised scores do carry information,
    and they vary a lot: technique clusters far more cleanly in ja than in
    zh, which is not what a purely lexical account predicts.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import (normalized_mutual_info_score,
                                 adjusted_rand_score, silhouette_score)

    print(f"  {'language':<10}{'n':>5}{'NMI':>9}{'ARI':>9}{'silhouette':>13}")
    for lang in sorted({r["language"] for r in recs}):
        i = [j for j, r in enumerate(recs) if r["language"] == lang]
        y = LabelEncoder().fit_transform([recs[j]["technique"] for j in i])
        km = KMeans(n_clusters=len(set(y)), n_init=10,
                    random_state=0).fit(X[i])
        print(f"  {lang:<10}{len(i):>5}"
              f"{normalized_mutual_info_score(y, km.labels_):>9.3f}"
              f"{adjusted_rand_score(y, km.labels_):>9.3f}"
              f"{silhouette_score(X[i], y):>13.3f}")


# ---------------------------------------------------------------- 13

def extremal_ranking(X, recs):
    """
    Rank techniques by mean distance from every other technique's centroid.

    This is a ranking of centroid distances, not a vertex of a hull -- the
    own-spread column is the caveat, and it is the same size as the
    distances, so the clouds interpenetrate heavily. Reported because
    Furstenberg's extremality is what explains the machinery arm attaching
    to it in stage 3: unusual vocabulary, not unusual mathematics.

    The second half asks where the technique probe fails. Those would be
    corpus-quality suspects (the generator ignoring the instruction).
    """
    techs = sorted({r["technique"] for r in recs})
    idxs = {t: [i for i, r in enumerate(recs) if r["technique"] == t]
            for t in techs}
    cents = {t: X[i].mean(axis=0) for t, i in idxs.items()}

    rows = []
    for t in techs:
        d = float(np.mean([np.linalg.norm(cents[t] - cents[o])
                           for o in techs if o != t]))
        spread = float(np.mean(np.linalg.norm(X[idxs[t]] - cents[t], axis=1)))
        rows.append((d, spread, t))
    rows.sort(reverse=True)

    print(f"  {'technique':<18}{'dist to others':>16}{'own spread':>13}")
    for d, spread, t in rows:
        print(f"  {t:<18}{d:>16.3f}{spread:>13.3f}")
    print(f"\n  most extreme {rows[0][2]}   most central {rows[-1][2]}")
    print("  Own spread is comparable to the distance to other centroids")
    print("  (and larger, for all but the top technique): the groups")
    print("  interpenetrate. This ranks centroids; it fits no hull.")

    y = LabelEncoder().fit_transform([r["technique"] for r in recs])
    pred = cross_val_predict(LogisticRegression(max_iter=3000), X, y, cv=5)
    n_wrong = int((pred != y).sum())
    print(f"\n  technique probe errors: {n_wrong}/{len(y)}")
    if n_wrong == 0:
        print("  None. The euler_product/euclid confusion seen on the")
        print("  earlier 768-dim encoder does not reproduce here, which")
        print("  points at that encoder rather than at corpus quality.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("proofs.jsonl"))
    ap.add_argument("--cache", type=Path, default=Path("embeddings.npy"))
    ap.add_argument("--arm", default="technique")
    args = ap.parse_args()

    recs, X = load(args.corpus, args.cache, args.arm)
    print(f"{len(recs)} records in the {args.arm} arm, "
          f"{X.shape[1]} dimensions")
    for f in FACTORS:
        print(f"  {f:<10} {dict(Counter(r[f] for r in recs))}")

    XL = centre_by(X, recs, "language")
    XLoo = centre_by_loo(X, recs, "language")

    rule("1. Language subspace: how many dimensions?")
    spectrum(X, recs, "language")

    rule("2. Style subspace")
    spectrum(X, recs, "style")
    print("\n  (technique, for comparison)")
    spectrum(X, recs, "technique")

    rule("3. Are the subspaces orthogonal?")
    subspace_angles(X, recs)

    rule("4. Variance budget")
    variance_budget(X, recs)

    rule("5. Do terse and verbose give the same technique centroids?")
    centroid_agreement(X, recs)
    print("\n  same, on language-centred embeddings:")
    centroid_agreement(XL, recs)

    rule("6. Probes with the task made hard")
    hard_probe(X, recs)

    rule("7. Pairwise similarity, raw space")
    within_vs_between(X, recs)
    rule("   Pairwise similarity, language-centred (in-sample means)")
    within_vs_between(XL, recs)
    rule("   Pairwise similarity, language-centred (leave-one-out means)")
    print("  The honest version: each record's offset is estimated without")
    print("  it. If the gap survives here, the centring result is real.")
    within_vs_between(XLoo, recs)
    print("\n  Probe accuracy under each centring:")
    for name, M in (("raw", X), ("in-sample", XL), ("leave-one-out", XLoo)):
        row = []
        for f in FACTORS:
            y = LabelEncoder().fit_transform([r[f] for r in recs])
            row.append(cross_val_score(
                LogisticRegression(max_iter=3000), M, y, cv=5).mean())
        print(f"  {name:<14} " + "  ".join(
            f"{f}={a:.3f}" for f, a in zip(FACTORS, row)))

    rule("8. Null distribution for the subspace angles")
    angle_null(X, recs, "technique", "language")
    angle_null(X, recs, "technique", "style")

    rule("9. Lexical baseline within one language")
    lexical_baseline(X, recs, "en")

    rule("10. Technique structure within each style")
    style_slices(X, recs)

    rule("11. Cross-lingual transfer, raw embeddings")
    cross_lingual_transfer(X, recs)

    rule("12. Technique structure with language held fixed")
    within_language(X, recs)

    rule("13. Extremal ranking of the techniques")
    extremal_ranking(X, recs)


if __name__ == "__main__":
    main()
