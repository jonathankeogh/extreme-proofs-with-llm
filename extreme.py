"""
Stage 3: what argument does the model reach for when asked for a vertex?

The technique arm was the instrument. Stage 2 established that it works:
language is additive and removable, and after centring, technique centroids
built from one style classify the other at 1.000. That licenses exactly one
thing -- classifying the extreme arm, which carries no technique label.

The question: when asked for the SHORTEST proof, which of the known
arguments does the model produce? When asked for the HEAVIEST MACHINERY?

If the answer differs by direction, technique choice is a second coordinate.
If it is also not a monotone function of length, it is an INDEPENDENT second
coordinate -- which is the thing missing before "hull" means anything.

Three guards, because a nearest-centroid assignment will always return
something:

  1. Confidence. If an extreme record is far from every centroid, the
     assignment is noise. Report the margin between best and second-best.
  2. Null. Compare the observed direction x technique table against the
     marginal, not against uniform.
  3. Length collinearity. If technique choice tracks length, it is the
     coordinate you already have wearing a hat.

Nothing here names a technique, a direction or a theorem: the labels come
from the corpus, so the same script runs on any corpus generate_*.py emits.

Usage:
    python extreme.py
    python extreme.py --corpus proofs_sqrt2.jsonl --cache embeddings_sqrt2.npy
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency, spearmanr

CORPUS = Path("proofs_primes.jsonl")
CACHE = Path("embeddings_primes.npy")


def load(corpus: Path, cache: Path):
    recs = [json.loads(l) for l in corpus.open() if l.strip()]
    X = np.load(cache)
    if len(X) != len(recs):
        raise SystemExit(f"{cache}: {len(X)} rows, {corpus}: {len(recs)}.")
    # The scope arm proves different theorems. It must not reach the language
    # means: those are the offset subtracted from every record here, and a
    # different theorem shifts them for reasons that have nothing to do with
    # language.
    keep = [i for i, r in enumerate(recs) if r.get("arm") != "scope"]
    if len(keep) != len(recs):
        print(f"dropped {len(recs) - len(keep)} scope-arm records "
              f"(different theorems; not comparable to these centroids)")
        recs, X = [recs[i] for i in keep], X[keep]
    return recs, X


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


def rule(t):
    print(f"\n{t}\n" + "-" * len(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--cache", type=Path, default=CACHE)
    args = ap.parse_args()

    recs, X = load(args.corpus, args.cache)
    tech_i = [j for j, r in enumerate(recs) if r["arm"] == "technique"]
    extr_i = [j for j, r in enumerate(recs) if r["arm"] == "extreme"]
    print(f"{len(tech_i)} technique records, {len(extr_i)} extreme records")

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
    self_acc = np.mean(
        np.argmax(T @ C.T, 1)
        == np.array([techs.index(recs[j]["technique"]) for j in tech_i]))
    print(f"  in-sample accuracy on the technique arm: {self_acc:.3f}")
    print("  (sanity check only -- these are the records the centroids came")
    print("   from. Stage 2's cross-style transfer is the honest number.)")

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
    rho, p = spearmanr(best, z)
    print(f"\n  Spearman(technique index, z-length) rho={rho:+.3f} p={p:.3f}")
    print("  Technique index is nominal, so read this only as a crude")
    print("  collinearity flag. The real check is the table above: if two")
    print("  techniques sit at the same z-length but different directions")
    print("  select them, the coordinates are independent.")

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


if __name__ == "__main__":
    main()