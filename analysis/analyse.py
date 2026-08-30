"""

Reads all three corpora and their cached embeddings and prints every analysis stage

  Stage 1  lengths        integrity, raw and normalised lengths, direction
                          vs language separation, CJK x direction
                          interactions, the centre cell.   (no embeddings)
  Stage 2  probes         pooled and leave-one-language-out probes,
                          neighbourhood composition, the centred repeat.
  Stage 3  structure      language/style/technique spectra, subspace angles
                          and their nulls, centroid agreement, hard probes,
                          the lexical baseline, style slices, cross-lingual
                          transfer, within-language scores, extremal
                          ranking.
  Stage 4  extreme        the extreme arm classified against the technique
                          centroids: confidence, out-of-set threshold,
                          direction x technique with a permutation null,
                          and the scope test.
"""

import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "config"))

import config
from embedding import cache_for

# The labelled arm the instrument can be validated on.
ARM = "technique"

# Language used in masking
LANG = 'en'

# non-latin character languages
CJK = {"ja", "zh"}

FACTORS = ("technique", "language", "style")


def rule(t):
    print(f"\n{t}\n" + "-" * len(t))


def banner(t):
    print(f"\n\n{'=' * 72}\n{t}\n{'=' * 72}")


def load(corpus: Path):
    with corpus.open() as f:
        recs = [json.loads(l) for l in f if l.strip()]
    fields = set().union(*(r.keys() for r in recs))
    for r in recs:
        r["chars"] = len(r["proof"])
        r["log_chars"] = math.log(r["chars"])
    print(f"{len(recs)} records from {corpus}")
    for arm, n in Counter(r["arm"] for r in recs).items():
        print(f"  {arm:10} {n}")
    return recs, fields


def load_embeddings(cache: Path):
    X = np.load(cache)
    print(f"embeddings from {cache}: {X.shape[0]} x {X.shape[1]}")
    return X


def stage_lengths(recs):
    banner("STAGE 1: lengths")

    # ---------------------------------------------------------------- 0
    rule("0. Integrity check")
    ids = [r["id"] for r in recs]
    print(f"records            {len(recs)}")
    print(f"unique ids         {len(set(ids))}")
    print(f"truncated          {sum(r['stop_reason'] == 'max_tokens' for r in recs)}")
    print(f"empty proofs       {sum(r['chars'] == 0 for r in recs)}")

    # Check we didn't mix up models and/or reasonings
    for field in ("model", "effort"):
        vals = {r.get(field) for r in recs}
        flag = "" if len(vals) == 1 else " WARNING: mix of model and reasoning effort, confounding"
        print(f"{field:18} {sorted(map(str, vals))}{flag}")

    # In the extreme arm, one prompt per (direction, language) is what is expected by design
    # Check we didn't mix up prompts per single cell
    per_cell = defaultdict(set)
    for r in recs:
        if r["arm"] == "extreme":
            per_cell[(r["direction"], r["language"])].add(r["prompt"])
    bad = {k: len(v) for k, v in per_cell.items() if len(v) != 1}
    print(f"cells w/ mixed prompt  {len(bad)}" + (f"  {bad}" if bad else ""))


    rule("1. Raw length by extremal direction (characters, pooled over languages)")
    d = defaultdict(list)
    for r in recs:
        if r["arm"] == "extreme":
            d[r["direction"]].append(r["chars"])
    for k, v in sorted(d.items(), key=lambda kv: st.median(kv[1])):
        print(f"  {k:14} median {st.median(v):7.0f}   n={len(v)}")
    print("  Pooled over scripts, so this axis still carries the language effect.")

    # within (direction, language) how much variation in length do we see in the samples (e.g. n=5)
    rule("2. Cell medians and within-cell variation")
    d = defaultdict(list)
    for r in recs:
        if r["arm"] == "extreme":
            d[(r["direction"], r["language"])].append(r["chars"])
    print(f"  {'direction':14} {'lang':4} {'median':>7} {'cv':>6}")
    for k, v in sorted(d.items()):
        cv = st.stdev(v) / st.mean(v)
        print(f"  {k[0]:14} {k[1]:4} {st.median(v):7.0f} {cv:6.2f}")
    print("  CV ~0.05 means the model returns one proof five times over")
    print("  higher means real sampling variation")
    print("  Note that 5 samples is a poor estimate of the true variance, this is not a statistical result we are just doing a first pass")

    # Script density affects sequence length, and we do not want that
    # confounded with the underlying proof structure. The effect is
    # multiplicative, so take the log and standardise within language,
    # which puts every language on the same scale.
    #
    # Baseline over the technique and extreme arms only. Scope-arm records
    # prove different theorems at systematically different lengths, so
    # including them would move the origin that the extreme arm is then
    # measured against.
    by_lang = defaultdict(list)
    for r in recs:
        if r["arm"] != "scope":
            by_lang[r["language"]].append(r["log_chars"])
    mu = {k: st.mean(v) for k, v in by_lang.items()}
    sd = {k: st.stdev(v) for k, v in by_lang.items()}
    for r in recs:
        r["z"] = (r["log_chars"] - mu[r["language"]]) / sd[r["language"]]

    rule("3. Normalised length: z-score of log chars, within language")
    d = defaultdict(list)
    for r in recs:
        if r["arm"] == "extreme":
            d[(r["direction"], r["language"])].append(r["z"])
    direction_language_cell = {k: st.mean(v) for k, v in d.items()}
    for k in sorted(direction_language_cell):
        print(f"  {k[0]:14} {k[1]:4} z {direction_language_cell[k]:+.2f}")
    # It is unsurprising that brevity is negative and machinery positive.
    # That is a hopeful sign the structure is there, not a result yet.

    directions = sorted({k[0] for k in direction_language_cell})
    means = {}
    for direction in directions:
        vals = [v for k, v in direction_language_cell.items() if k[0] == direction]
        means[direction] = st.mean(vals)

    # Hold direction steady and see how much moves between Latin and CJK.
    rule("5. CJK vs Latin, per direction (normalised scale)")
    print("  A level difference between scripts is already removed by section 3.")
    print("  Anything left here is a language x direction interaction.")
    print(f"\n  {'direction':14} {'latin':>7} {'cjk':>7} {'gap':>7}")
    for direction in directions:
        latin = [v for k, v in direction_language_cell.items()
                 if k[0] == direction and k[1] not in CJK]
        cjk = [v for k, v in direction_language_cell.items()
               if k[0] == direction and k[1] in CJK]
        gap = st.mean(cjk) - st.mean(latin)
        flag = "  <--" if abs(gap) > 0.25 else ""
        print(f"  {direction:14} {st.mean(latin):+7.2f} {st.mean(cjk):+7.2f} "
              f"{gap:+7.2f}{flag}")

    # ---------------------------------------------------------------- 6
    # Now do calcs relative to the unprompted centre
    theorem = next(r["theorem"] for r in recs if r["arm"] != "scope")
    centre_proofs = [r for r in recs
            if r["arm"] == "scope" and r["theorem"] == theorem]

    rule("6. The centre cell: same theorem, no selection criterion")
    z0 = st.mean([r["z"] for r in centre_proofs])
    print(f"  unprompted     n={len(centre_proofs)}   mean z {z0:+.2f}")
    print(f"  {'direction':14} {'mean z':>8} {'vs centre':>11}")
    for d in sorted(means, key=lambda x: means[x]):
        print(f"  {d:14} {means[d]:+8.2f} {means[d] - z0:+11.2f}")

    print("\nCaveat: character count is a proxy for proof length, but not a measure")

# ======================================================================
# Stage 2: probes on the embedding - the purpose it figure out if technique is being captured first
#
# We should do a quick linear probe and think about the results
# This turned out to be dumb, because the learning is over-specified, the matrix
# is (num languages) x (embedding dim) = 5 x 1024 = 5125 for the languages. For 300 samples that's meaningless
# because 5125 / 300 = 17 whcih means we can twiddle each 17 knob to fit our label for each sample

# theorem: n points in general position in 𝑅^𝑑 can be split by a hyperplane for any binary labelling as long as 
# 𝑛 ≤ 𝑑 + 1
# 
# ======================================================================


def stage_probes(recs, X):
    # recs is the corpus, before embedding
    # X is the embeddings matrix of the corpus
    banner("STAGE 2: linear probes on the embedding")


    # A linear probe here is just saying, can we linearly seperate the factors
    def probe(M, y, name):
        """5-fold CV accuracy of a linear probe against chance"""
        # encode our labels y
        yi = LabelEncoder().fit_transform(y)
        k = len(set(yi))
        # fit a linear probe f(x) = Ax + b with 5 folds (5 chunks of train and tests) composed with softmax
        # "logistic" because it models log-odds as linear: log(p/(1-p)) = Ax + b
        # then use log-loss to train, capped at 3000 iters.
        # then average the accuracy
        acc = cross_val_score(LogisticRegression(max_iter=3000), M, yi,
                              cv=5).mean()
        # silhouette measures clustering by pairwise distances
        # for each point i:
        #   a(i) = mean distance to the other points in its OWN group
        #   b(i) = mean distance to the nearest OTHER group
        #   s(i) = (b - a) / max(a, b),  averaged over all points.
        # near 1  means own group far closer than any other
        #  0  means it's in no man's land
        # near -1 means it's closer to another group than to its own
        # Note this is a different question from the probe's, that is,  classes can be
        # perfectly separable by a hyperplane and still score near 0 here
        # (think about long parallel stripes, not round blobs)
        sil = silhouette_score(M, yi)
        print(f"  {name:<10} k={k}  acc={acc:.3f}  chance={1/k:.3f}  "
              f"lift={acc - 1/k:+.3f}  silhouette={sil:+.3f}")

    # leave-one-langauge-out, linear probe
    def lolo(M, y, groups):
        """Train on every group but one and test on the held-out group"""
        yi = LabelEncoder().fit_transform(y)
        accs = []
        # hold out a language and train
        # the purpose of this is to see, when the probe is trained on 5/6 languages to capture technique, does that
        # generalise to the held out language?
        # pooled 5-fold above cannot ask this because ervery random fold contains all six
        # languages, so the model always has english in train and in test sets
        #
        # this is a generalisation test not a separability one, it says the
        # technique direction transfers across languages, not that language
        # and technique are independent! we test that lateer. they can still heavily depend. And it does not rule out
        # the transfer riding on untranslated proper nouns, which we also test later
        for held in sorted(set(groups)):
            tr = [i for i, g in enumerate(groups) if g != held]
            te = [i for i, g in enumerate(groups) if g == held]
            clf = LogisticRegression(max_iter=3000).fit(M[tr], yi[tr])
            a = clf.score(M[te], yi[te])
            accs.append(a)
            print(f"  hold out {held}   acc={a:.3f}")
        print(f"  mean {np.mean(accs):.3f}  min {min(accs):.3f}  "
              f"chance {1 / len(set(yi)):.3f}")

    # Scoring uses the technique arm only because it is the labelled part
    # idx is the indices for the technique corpus
    idx = [i for i, r in enumerate(recs) if r["arm"] == "technique"]
    # T is the technique corpus
    T = [recs[i] for i in idx]
    # XT  is the technique embeddings
    XT = X[idx]
    print(f"\nScoring on the technique arm: {len(T)} labelled records")

    # ---------------------------------------------------------------- 1
    print("\n1. Pooled linear probe (5-fold cross validation)")
    for f in FACTORS:
        probe(XT, [r[f] for r in T], f)

    # ---------------------------------------------------------------- 2
    # Pooled CV cannot separate "technique separates within every language"
    # from "technique separates in English only". This can.
    print("\n2. Leave-one-language-out probe, target = technique")
    lolo(XT, [r["technique"] for r in T], [r["language"] for r in T])

    # ---------------------------------------------------------------- 3
    # Which factor dominates the local geometry - for each record, the
    # fraction of its k nearest neighbours sharing each label (but take only one sample from each cell)
    # use 10-nearest neigbhours
    KNN = 10
    print(f"\n3. Neighbourhood composition (k={KNN}, same-cell excluded)")
    # get the dot-product of every embedding in the techniques, so we can read off cosine similarity
    S = XT @ XT.T
    # create a boolean nxn of cells we want to mask
    cell = [(r["technique"], r["language"], r["style"]) for r in T]
    same_cell = np.array([[a == b for b in cell] for a in cell])
    S[same_cell] = -np.inf          # mask set similarity for same cell as negative infinity, including itself ie diagonals
    # sort by values but return indices, -S makes it descending, axis=1 sorts within row, then keep the first KNN columns
    nn = np.argsort(-S, axis=1)[:, :KNN]
    for field in FACTORS:
        # the vakues of this field, e.g. euclid as your technique in the infintude of the primes
        vals = [r[field] for r in T]
        # of the 10 nearest neigbhours, get the percent (mean of boolean values) that are the same technqieue for e.g.,
        # and do this across the values in your factor (.e.g across technique factor, what is the average percent
        # agreement with 10 nearest neighbours)
        share = np.mean([np.mean([vals[j] == vals[i] for j in nn[i]])
                         for i in range(len(T))])
        # Chance that two records drawn at random share a label:
        # sum_c n_c(n_c - 1) / n(n - 1). this handles class imbalance unlike
        # the 1/k used in probe()
        base = sum(c * (c - 1) for c in Counter(vals).values()) / (
            len(T) * (len(T) - 1))
        print(f"  {field:<10} same-label among {KNN}-NN: {share:.3f}  "
              f"(baseline {base:.3f}  lift {share - base:+.3f}  "
              f"ratio {share / base:.2f}x)")


# ======================================================================
# Stage 3: structural probes
#
#
#   1. language_spectrum   How many dimensions does language actually
#                          occupy? SVD of the six language means.
#                          "Additive" is weaker than "one-dimensional", and
#                          the UMAP can;t tell them apart
#   2. style_spectrum      Same question for style.
#   3. subspace_angles     Are the language and style subspaces orthogonal
#                          to the technique subspace, or merely separable?
#   4. variance_budget     Fraction of total variance per factor.
#   5. centroid_agreement  THE ONE THAT MATTERS FOR STAGE 4. Technique
#                          centroids from terse records only vs verbose
#                          only. Disagreement means pooled centroids are
#                          unsafe for the extreme arm, which has no style.
#   6. hard_probe          Accuracy after PCA and after subsampling to one
#                          record per cell. The 1.000s above are
#                          uninformative on their own.
#   7. within_vs_between   Cosine within a technique across languages vs
#                          between techniques. Does not saturate.
#   8. angle_null          Nulls for section 3. It measures the dimension,
#                          not the design.
#   9. lexical_baseline    Can TF-IDF alone recover the technique within
#                          one language? Bounds how much is terminology.
#  10. style_slices        Does technique structure depend on style?
#  11. cross_lingual       Technique probe trained on one language, applied
#                          frozen to the rest, on RAW embeddings.
#  12. within_language     Technique clustering with language held fixed.
#  13. extremal_ranking    Techniques ranked by centroid distance.
# ======================================================================


def centre_by(X, recs, field):
    """Subtract the mean embedding of each level of `field`, renormalise."""
    Xc = X.copy()
    for lvl in {r[field] for r in recs}:
        i = [j for j, r in enumerate(recs) if r[field] == lvl]
        Xc[i] -= Xc[i].mean(axis=0)
    n = np.linalg.norm(Xc, axis=1, keepdims=True)
    return Xc / np.clip(n, 1e-9, None)


# ---------------------------------------------------------------- 6


def hard_probe(X, recs, seed=0):
    print(f"  Probe accuracy is uninformative at full dimension: "
          f"{len(recs)} points")
    print("  with several near-duplicates per cell are trivially separable")
    print(f"  in {X.shape[1]} dimensions. There are two ways to improve this")
    rng = np.random.default_rng(seed)

    print(f"\n  (a) after PCA, all {len(recs)} records")
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


def stage_structure(recs, X):
    """
    Everything this stage needs is defined inside it and the helpers
    below are used by no other stage. The three that stage 7 also
    calls -- centre_by, hard_probe, cross_lingual_transfer -- are the
    exceptions, and live with the shared helpers at the top.
    """

    def centre_by_loo(X, recs, field):
        """
        Leave-one-out centring: each record is centred by the mean of its own
        level EXCLUDING itself.

        Plain centre_by forces each level mean to exactly zero, so a language
        probe on the result is guaranteed to collapse -- by construction, not by
        discovery. LOO removes that guarantee: the offset applied to a record is
        estimated from the other records at its level and never from itself, so
        a probe that still fails is failing on held-out information.
        """
        Xc = X.copy()
        for lvl in {r[field] for r in recs}:
            i = np.array([j for j, r in enumerate(recs) if r[field] == lvl])
            s, n = X[i].sum(0), len(i)
            Xc[i] = X[i] - (s - X[i]) / (n - 1)
        return Xc / np.clip(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-9, None)

    def means_matrix(X, recs, field):
        """One row per level, centred on the grand mean of those rows"""
        # Get the sorted values of the called field in the recs, e.g. for technique it might be euclid and frustenberg etc
        levels = sorted({r[field] for r in recs})
        # for each value in the field, select them from the embedding matrix X, then get the average vector, stack them up
        M = np.stack([X[[i for i, r in enumerate(recs) if r[field] == lvl]].mean(0)
                      for lvl in levels])
        # so then, M is the average vector per value in a field, then we remove the average across M, so we
        # we get a matrix where each row is a centered average vector of the value of the given field
        # Each row now represents how that level's average embedding
        # differs from the average embedding across all levels
        return levels, M - M.mean(0)

    def basis(X, recs, field):
        """Orthonormal basis for the span of the level means."""
        _, M = means_matrix(X, recs, field)
        U, s, _ = np.linalg.svd(M.T, full_matrices=False)
        return U[:, :int((s > 1e-8).sum())]

    # ---------------------------------------------------------------- 1, 2

    # this entire block is basically doing PCA-style analysis to say, if we get an average centered vector
    # per value in our field, which directions explain the variance
    def spectrum(X, recs, field):
        levels, M = means_matrix(X, recs, field)
        # get the singulat values, along each orthogonal direction
        s = np.linalg.svd(M, compute_uv=False)
        # normalise so it's percentages
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

        # Where do the group-mean vectors for each field value lie when
        # projected onto the leading direction, i.e. PC1 above?
        U, _, _ = np.linalg.svd(M.T, full_matrices=False)
        proj = M @ U[:, 0]
        order = np.argsort(proj)
        print("  leading direction, levels ordered:")
        print("   ", "  ".join(f"{levels[i]}({proj[i]:+.2f})" for i in order))

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
        print(f"\n  technique NMI difference (terse - verbose): "
              f"{out['terse'] - out['verbose']:+.3f}")
        print("  -> positive: notation carries the mathematics, prose the "
              "language.")

    # ---------------------------------------------------------------- 12

    def within_language(X, recs):
        """
        Technique structure with language held fixed. If technique clusters
        inside a single language, the structure is not a language artifact.

        Probe accuracy is NOT reported here. Fifty records in 1024 dimensions
        are linearly separable almost regardless of the labels -- the probe
        reads 1.000 in every language and carries no information (section 6
        makes the same point at length). The unsupervised scores do carry
        information, and they vary a lot: technique clusters far more cleanly
        in ja than in zh, which is not what a purely lexical account predicts.
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
            print("  None. On the primes corpus this is the result that clears")
            print("  the earlier 768-dim encoder's euler_product/euclid")
            print("  confusion: it points at that encoder, not at the corpus.")
        else:
            pairs = Counter((sorted({r["technique"] for r in recs})[t],
                             sorted({r["technique"] for r in recs})[p])
                            for t, p in zip(y, pred) if t != p)
            print("  Confusions (true -> predicted): " + ", ".join(
                f"{a}->{b} {n}" for (a, b), n in pairs.most_common(5)))

    banner("STAGE 3: structural probes")

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
    print("\n  same, on language-centred embeddings (what stage 4 uses):")
    centroid_agreement(XLoo, recs)

    rule("6. Probes with the task made hard")
    hard_probe(X, recs)

    rule("7. Pairwise similarity, raw space")
    within_vs_between(X, recs)
    rule("   Pairwise similarity, language-centred (leave-one-out means)")
    print("  The honest version: each record's offset is estimated without")
    print("  it. If the gap survives here, the centring result is real.")
    within_vs_between(XLoo, recs)

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


# ======================================================================
# Stage 3b: is the centre of the grid the centre cell?
#
# Numbered 3b rather than 4 on purpose: the README indexes stages 1-7 by
# number, and this is a question that only became askable once stage 3
# showed language is a removable additive offset. It runs on the same two
# objects stage 3 does and adds no estimate anything downstream uses.
#
# Stage 1 section 6 found the centre cell on the length axis -- the scope
# arm's own-theorem target, asked with no selection criterion. That is a
# cell the corpus contains. The centroid of the whole grid is a different
# object: the mean of every technique and extreme record, a point no
# prompt targeted. If the extremal directions really do surround a
# default, the two should land in the same place.
#
# Raw cosines are reported but should not be read: bge-m3 is anisotropic
# enough that every cell pair sits above 0.75, so only the ranking carries
# information. The language-centred column is the one that means anything.
# ======================================================================


def stage_centre(recs, X):
    banner("STAGE 3b: the centre of the grid")

    def cells_of(recs):
        """
        Name every cell the corpus contains, by the factor that defines
        it: direction for the extreme arm, technique for the labelled
        arm, target statement for the scope arm.
        """
        out = defaultdict(list)
        for i, r in enumerate(recs):
            if r["arm"] == "scope":
                out[("scope", r["theorem"])].append(i)
            elif r["arm"] == "extreme":
                out[("extreme", r["direction"])].append(i)
            else:
                out[("technique", r["technique"])].append(i)
        return out

    def unit(v):
        return v / max(float(np.linalg.norm(v)), 1e-9)

    # The centre cell, identified exactly as stage 1 section 6 does: the
    # scope records that prove the grid's own theorem.
    theorem = Counter(r["theorem"] for r in recs
                      if r["arm"] != "scope").most_common(1)[0][0]
    centre_key = next((("scope", r["theorem"]) for r in recs
                       if r["arm"] == "scope" and r["theorem"] == theorem),
                      None)
    if centre_key is None:
        print("  No own-theorem scope cell in this corpus; nothing to compare.")
        return

    # The grid is the technique and extreme arms. Scope records prove
    # OTHER statements, so they are not part of the thing whose centre is
    # being located -- they are only scored against it.
    grid = [i for i, r in enumerate(recs) if r["arm"] != "scope"]
    cells = cells_of(recs)

    Xr = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)

    # Not centre_by: that would estimate each language mean from every
    # record, scope arm included. The means have to come from the balanced
    # grid, and the scope records then get scored against them.
    mu = {lang: X[[i for i in grid if recs[i]["language"] == lang]].mean(0)
          for lang in {r["language"] for r in recs}}
    Xc = np.stack([X[i] - mu[recs[i]["language"]] for i in range(len(recs))])
    Xc /= np.clip(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-9, None)

    for name, M in (("raw", Xr), ("language-centred", Xc)):
        rule(f"{'1' if name == 'raw' else '2'}. Cell centroids vs the grid "
             f"centroid ({name})")
        grand = unit(M[grid].mean(0))
        sims = sorted(((float(grand @ unit(M[i].mean(0))), k)
                       for k, i in cells.items()), reverse=True)
        rank = [k for _, k in sims].index(centre_key) + 1

        print(f"\n  centre cell rank {rank} of {len(sims)}")
        if name == "raw":
            print("  Raw cosines span a narrow band -- bge-m3's anisotropy, not")
            print("  a finding. Only the rank is worth reading in this section.")

    print("\n  If the centre cell ranks at or near the top of section 2, the")
    print("  grid's centre of mass is a real place: the extremal directions")
    print("  cancel, and what they cancel to is what the model writes when")
    print("  nothing is asked of it. Where it does NOT rank first, the cells")
    print("  above it name the imbalance -- the techniques the grid")
    print("  over-samples relative to the model's default.")


# ======================================================================
# Stage 4: what argument does the model reach for when asked for a vertex?
#
# Stage 3 established that the instrument works: language is additive and
# removable, and after centring, technique centroids built from one style
# classify the other at 1.000. That licenses exactly one thing --
# classifying the extreme arm, which carries no technique label.
#
# Three guards, because a nearest-centroid assignment will always return
# something: assignment confidence, a permutation null against the
# marginal, and a length-collinearity check.
#
# Section 8 runs the same classification on the scope arm, where present.
# Those records are held out of every estimate here and only classified
# against the centroids -- Conway and Shipman's scope test, run on the
# model instead of on the literature.
#
# Nothing here names a technique, a direction or a theorem: the labels come
# from the corpus, so it runs on any corpus generate_*.py emits.
# ======================================================================


def stage_extreme(all_recs, X_all):
    """
    Helpers used by nothing else, kept inside the stage.
    """

    def language_means(recs, X, arm=None):
        mu = {}
        for lang in sorted({r["language"] for r in recs}):
            i = [j for j, r in enumerate(recs)
                 if r["language"] == lang and (arm is None or r["arm"] == arm)]
            mu[lang] = X[i].mean(0)
        return mu

    def centre(recs, X, mu, idx):
        Xc = np.stack([X[j] - mu[recs[j]["language"]] for j in idx])
        n = np.linalg.norm(Xc, axis=1, keepdims=True)
        return Xc / np.clip(n, 1e-9, None)

    def perm_test(tab, n_perm=20000, seed=0):
        """
        Permutation null for the direction x technique table.

        chi2_contingency's p-value assumes expected cell counts are not tiny.
        Here they are: the table has one cell per (direction, technique) pair,
        and some techniques are picked once or never across the whole extreme
        arm, so the asymptotic p is not trustworthy even though the effect is
        obvious. Shuffling the direction labels against
        the assigned techniques costs a second and needs no such assumption.

        Reported as chi2 recomputed on each shuffle; p is the fraction of
        shuffles reaching the observed statistic.
        """
        obs = chi2_contingency(tab)[0]
        rows = np.repeat(np.arange(tab.shape[0]), tab.sum(1))
        cols = np.repeat(np.arange(tab.shape[1]), tab.sum(0))
        rng = np.random.default_rng(seed)
        null = np.empty(n_perm)
        for b in range(n_perm):
            shuf = rng.permutation(cols)
            t = np.zeros_like(tab)
            np.add.at(t, (rows, shuf), 1)
            # chi2 by hand: the shuffled table can have empty rows/columns.
            exp = np.outer(t.sum(1), t.sum(0)) / t.sum()
            null[b] = np.where(exp > 0, (t - exp) ** 2 / np.maximum(exp, 1e-12),
                               0.0).sum()
        hits = int((null >= obs).sum())
        p = (hits + 1) / (n_perm + 1)
        print(f"\n  permutation null ({n_perm} shuffles of the direction "
              f"labels):")
        print(f"    observed chi2 {obs:.1f}   null mean {null.mean():.1f}   "
              f"null max {null.max():.1f}")
        print(f"    p = {p:.5f}  ({hits} of {n_perm} shuffles >= observed)")
        print("    Distribution-free, so it does not lean on expected counts")
        print("    that the sparse cells here would not support.")

    banner("STAGE 4: the extreme arm against the technique centroids")
    # The scope arm is held back rather than dropped. It must not reach the
    # language means or the centroids -- those have to be estimated on
    # proofs of ONE theorem, and a different statement would move them for
    # reasons that have nothing to do with language. But being unfit to
    # estimate from is not the same as being unfit to classify, and
    # classifying it is the whole point of the arm: section 8 asks which
    # known argument the model reaches for as the statement moves out of
    # each proof's documented scope.
    keep = [i for i, r in enumerate(all_recs) if r["arm"] != "scope"]
    drop = [i for i, r in enumerate(all_recs) if r["arm"] == "scope"]
    recs, X = [all_recs[i] for i in keep], X_all[keep]
    scope_recs, scope_X = [all_recs[i] for i in drop], X_all[drop]

    tech_i = [j for j, r in enumerate(recs) if r["arm"] == "technique"]
    extr_i = [j for j, r in enumerate(recs) if r["arm"] == "extreme"]
    print(f"{len(tech_i)} technique records, {len(extr_i)} extreme records, "
          f"{len(scope_recs)} scope records")

    mu = language_means(recs, X)
    T = centre(recs, X, mu, tech_i)
    E = centre(recs, X, mu, extr_i)

    techs = sorted({recs[j]["technique"] for j in tech_i})
    dirs = sorted({recs[j]["direction"] for j in extr_i})

    # ---------------------------------------------------------------- 0
    rule("0. Does the out-of-sample offset actually remove language?")
    print("  If the technique-arm language means are the language component,")
    print("  a probe on the centred extreme arm should be near chance.")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    yl = LabelEncoder().fit_transform([recs[j]["language"] for j in extr_i])
    raw = np.stack([X[j] for j in extr_i])
    for name, M in (("raw", raw), ("centred", E)):
        a = cross_val_score(LogisticRegression(max_iter=3000), M, yl,
                            cv=5).mean()
        print(f"  language probe, {name:<8} {a:.3f}   (chance "
              f"{1/len(set(yl)):.3f})")
    if a > 2 / len(set(yl)):
        print("  -> centred still high: the offset did not transfer. Stop,")
        print("     and re-estimate the means on the pooled corpus instead.")
    else:
        print("  -> offset transferred; language is removed out of sample.")

    # ---------------------------------------------------------------- 1
    rule("1. Technique centroids, built on the technique arm")
    C = np.stack([T[[k for k, j in enumerate(tech_i)
                     if recs[j]["technique"] == t]].mean(0) for t in techs])
    C /= np.clip(np.linalg.norm(C, axis=1, keepdims=True), 1e-9, None)

    # ---------------------------------------------------------------- 2
    rule("2. Assignment confidence on the extreme arm")
    S = E @ C.T
    order = np.argsort(-S, axis=1)
    best = order[:, 0]
    margin = S[np.arange(len(S)), order[:, 0]] - S[np.arange(len(S)),
                                                   order[:, 1]]
    top = S[np.arange(len(S)), best]
    print(f"  cos to nearest centroid   mean {top.mean():+.3f}  "
          f"min {top.min():+.3f}")
    print(f"  margin over second-best   mean {margin.mean():+.3f}  "
          f"median {np.median(margin):+.3f}")
    print(f"  records with margin < 0.02: {(margin < 0.02).sum()} of "
          f"{len(margin)}")
    print("  A small margin means the record sits between arguments -- the")
    print("  assignment is a coin flip and should not be counted. Table")
    print("  below is reported twice: all records, and margin >= 0.02 only.")

    # ---------------------------------------------------------------- 3
    for label, keep in (("all records", np.ones(len(E), bool)),
                        ("margin >= 0.02", margin >= 0.02)):
        rule(f"3. Direction x technique  ({label}, n={keep.sum()})")
        tab = np.zeros((len(dirs), len(techs)), int)
        for k, j in enumerate(extr_i):
            if keep[k]:
                tab[dirs.index(recs[j]["direction"]), best[k]] += 1
        w = max(len(t) for t in techs) + 2
        print("  " + " " * 14 + "".join(f"{t:>{w}}" for t in techs))
        for a, d in enumerate(dirs):
            print(f"  {d:<14}" + "".join(f"{v:>{w}d}" for v in tab[a]))
        print("  " + " " * 14 + "".join(f"{v:>{w}d}" for v in tab.sum(0))
              + "   <- marginal")
        if (tab.sum(1) > 0).all() and (tab.sum(0) > 0).all():
            chi2, p, dof, _ = chi2_contingency(tab)
            v = np.sqrt(chi2 / (tab.sum() * (min(tab.shape) - 1)))
            print(f"\n  chi2={chi2:.1f}  dof={dof}  p={p:.4f}  "
                  f"Cramer's V={v:.3f}")
            print("  Null: direction and technique choice are independent,")
            print("  i.e. every direction draws from the same marginal.")
            print("  p small -> the direction prompt changes which argument")
            print("  the model reaches for. That is the second coordinate.")
            perm_test(tab)
        else:
            print("\n  Some row or column is empty; chi2 not computed.")

    # ---------------------------------------------------------------- 4
    rule("4. Is technique choice just length in disguise?")
    print("  If the two coordinates are collinear, you have one axis.")
    L = np.array([len(recs[j]["proof"]) for j in extr_i], float)
    lang = np.array([recs[j]["language"] for j in extr_i])
    z = np.zeros(len(L))
    for lg in set(lang):
        m = lang == lg
        z[m] = (np.log(L[m]) - np.log(L[m]).mean()) / np.log(L[m]).std()
    print(f"  {'technique':<18}{'n':>4}{'mean z-length':>16}")
    for a, t in enumerate(techs):
        m = best == a
        if m.sum():
            print(f"  {t:<18}{m.sum():>4}{z[m].mean():>16.2f}")

    # ---------------------------------------------------------------- 5
    rule("5. Per-direction detail, for eyeballing")
    for a, d in enumerate(dirs):
        ks = [k for k, j in enumerate(extr_i) if recs[j]["direction"] == d]
        c = Counter(techs[best[k]] for k in ks)
        share = ", ".join(f"{t} {n}/{len(ks)}" for t, n in c.most_common())
        print(f"  {d:<14} {share}")
    print("\n  Read at least three raw proofs from the two most lopsided")
    print("  cells before believing any of this.")


    rule("6. Does the mapping hold within every language?")
    langs = sorted({recs[j]["language"] for j in extr_i})
    for d in dirs:
        row = []
        for lg in langs:
            ks = [k for k, j in enumerate(extr_i)
                  if recs[j]["direction"] == d and recs[j]["language"] == lg]
            c = Counter(techs[best[k]] for k in ks)
            top_tech, n = c.most_common(1)[0]
            row.append(f"{lg}:{top_tech[:5]}{n}")
        print(f"  {d:<14} " + "  ".join(row))
    print("  Same technique across all six languages -> the mapping is a")
    print("  property of the direction, not of residual language leakage.")


    rule("7. Out-of-set detection")
    print("  Nearest-centroid must choose one of five. A record whose proof")
    print("  is none of the five still gets assigned. Calibrate: how close")
    print("  is a technique-arm record to its OWN centroid, leave-one-out?")

    ref = []
    for k, j in enumerate(tech_i):
        t = techs.index(recs[j]["technique"])
        peers = [m for m, jj in enumerate(tech_i)
                 if recs[jj]["technique"] == recs[j]["technique"] and m != k]
        c = T[peers].mean(0)
        ref.append(float(T[k] @ c / np.linalg.norm(c)))
    ref = np.array(ref)
    thr = np.percentile(ref, 5)
    print(f"  technique arm, cos to own centroid (LOO): "
          f"mean {ref.mean():+.3f}  5th pct {thr:+.3f}  min {ref.min():+.3f}")
    print(f"  extreme arm,   cos to nearest centroid:   "
          f"mean {top.mean():+.3f}  median {np.median(top):+.3f}")

    for a, d in enumerate(dirs):
        ks = [k for k, j in enumerate(extr_i) if recs[j]["direction"] == d]
        out = sum(top[k] < thr for k in ks)
        print(f"  {d:<14} {out:>2}/{len(ks)} below the 5th-percentile "
              f"threshold  (mean cos {np.mean([top[k] for k in ks]):+.3f})")
    print("  A direction with most records below threshold is not selecting")
    print("  a known technique -- it is leaving the reference set.")

    if not scope_recs:
        return

    rule("8. The scope test")
    print("  Conway and Shipman's criterion for two proofs being really")
    print("  different is that they settle different sets of statements.")
    print("  Each target below sits on a boundary where some of the known")
    print("  arguments stop working. The centroids are estimated on this")
    print("  theorem's technique arm and the language means on its two main")
    print("  arms; the scope records are only classified against them,")
    print("  never used to build them.")
    print()
    print("  Read it as a prediction test: as the statement moves out of a")
    print("  proof's documented scope, that proof should stop being the one")
    print("  the model produces, and the records should drift away from")
    print("  every centroid.")

    Sc = centre(scope_recs, scope_X, mu, range(len(scope_recs))) @ C.T
    best_s = np.argmax(Sc, 1)
    top_s = Sc[np.arange(len(Sc)), best_s]
    order_s = np.argsort(-Sc, axis=1)
    margin_s = (Sc[np.arange(len(Sc)), order_s[:, 0]]
                - Sc[np.arange(len(Sc)), order_s[:, 1]])

    print(f"\n  {'target':<24}{'n':>3}{'cos':>8}{'out':>7}   "
          f"technique chosen")
    for t in sorted({r["theorem"] for r in scope_recs}):
        ks = [k for k, r in enumerate(scope_recs) if r["theorem"] == t]
        c = Counter(techs[best_s[k]] for k in ks)
        out = sum(top_s[k] < thr for k in ks)
        share = ", ".join(f"{n}/{len(ks)} {a}" for a, n in c.most_common(3))
        print(f"  {t:<24}{len(ks):>3}{np.mean([top_s[k] for k in ks]):>8.3f}"
              f"{out:>4}/{len(ks)}   {share}")
    print(f"\n  cos      mean cosine to the nearest technique centroid")
    print(f"  out      records below the {thr:+.3f} out-of-set threshold "
          f"from section 7")
    print(f"  margin over second-best, pooled: mean {margin_s.mean():+.3f}")
    print("\n  Two failure modes to check before reading anything into it.")
    print("  A target where everything lands on one centroid with a tiny")
    print("  margin is nearest-centroid having to choose, not the model")
    print("  agreeing. And a target far outside every scope should show a")
    print("  LOW cosine: if it does not, the centroids are measuring")
    print("  subject matter rather than argument.")



def main():
    """
    Every stage, on every corpus, with the settings the write-up reports.

    There are no options. The whole run is about ninety seconds because
    nothing here embeds anything -- so there is no configuration worth
    the reader having to know about, and no way to produce a number by
    passing a flag that is not in this file.
    """
    for theorem, corpus in config.CORPORA.items():
        banner(f"CORPUS: {corpus.name}")

        recs, fields = load(corpus)
        X = load_embeddings(cache_for(corpus))


        # The technique arm is the labelled part -- the only arm that says
        # which known proof each record is -- so it is the slice the
        # instrument is validated on, in stages 3 and 7.
        arm = [i for i, r in enumerate(recs) if r["arm"] == ARM]
        arm_recs, arm_X = [recs[i] for i in arm], X[arm]

        # 1. Char length before any embedding is involved. Does some basic
        #    counting analysis and finds the
        #    centre cell every extremal direction is measured against.
        # Character count is not comparable across scripts: a Chinese proof of the
        # same content is roughly 0.57x the length of the English one, so section 3
        # works in log space, where a constant ratio becomes a constant offset and
        # subtracting a per-language mean removes it.
        #
        # Scope-arm records, are excluded from the baseline because they prove a different
        # theorem, so their length is not a fact about language.
        stage_lengths(recs)

        # 2. Can we remove all language data with a simple removal of the mean directoin?
        stage_probes(recs, X)

        # 3. What is actually in the space: how many dimensions language
        #    occupies, whether the factors are separable or merely
        #    orthogonal, how much of the technique signal is vocabulary.
        stage_structure(arm_recs, arm_X)

        # 3b. Stage 1 found the centre cell on the length axis. Now that
        #     language is known to be removable, ask the same question in
        #     the embedding: is the centroid of the whole grid -- a point
        #     no prompt targeted -- the same place as the cell where
        #     nothing was asked?
        stage_centre(recs, X)

        # 4. The question the corpus was built for: which known argument
        #    does the model reach for when asked for a vertex?
        stage_extreme(recs, X)


if __name__ == "__main__":
    main()
