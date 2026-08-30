"""
Is the technique signal lexical?

Mask the most technique-diagnostic word types per language, re-embed, and
ask whether a probe trained on one language still recovers technique in
another. Translation removes shared script and shared vocabulary, so a
transfer that survives masking is not riding on tokens that cross
languages unchanged.

What this rules out is the lexical account. What it does NOT establish is
that the encoder reads argument structure: proof length, equation count
and paragraph shape also survive translation. Read it as "not the words",
not as "the semantics".

Two further caveats, both structural:
  - terms are selected on terse records and the whole corpus is masked, so
    the terse half is masked in-sample. The LANGUAGES are still held out,
    which is what the transfer test is about, but the number is not clean.
  - masking may destroy the argument rather than just its vocabulary. Read
    a masked proof before believing a collapse means "the encoder was
    lexical".

    python analysis/ablation.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "config"))

import config
from embedding import cache_for

PLACEHOLDER = "⟨m⟩"
CJK = {"ja", "zh"}
LATIN = {"en", "de", "es", "fr"}
TOPK = 400          # word types masked per language


def analyser_for(lang):
    """ja/zh are not space-delimited, so score character n-grams there."""
    if lang in LATIN:
        return dict(analyzer="word", ngram_range=(1, 1))
    return dict(analyzer="char_wb", ngram_range=(2, 4))


def discriminative_terms(recs, lang, topk):
    """
    The terms a linear classifier leans on, found from the data rather than
    from a hand-written list.

    Selection sees TERSE records only; the transfer test below is run on
    all of them. Choosing the terms by looking at the records you then
    score would guarantee a collapse and measure the selection, not the
    representation.

    Ties break on the term itself, not just length: `sorted` is stable, so
    key=len alone would order equal-length terms by set iteration order,
    which Python randomises per process. The masked corpus would then not
    reproduce.
    """
    i = [j for j, r in enumerate(recs)
         if r["arm"] == "technique" and r["language"] == lang
         and r["style"] == "terse"]
    vec = TfidfVectorizer(min_df=2, **analyser_for(lang))
    M = vec.fit_transform([recs[j]["proof"] for j in i])
    y = LabelEncoder().fit_transform([recs[j]["technique"] for j in i])
    W = np.atleast_2d(LogisticRegression(max_iter=3000).fit(M, y).coef_)
    names = np.array(vec.get_feature_names_out())
    per = max(1, topk // len(W))
    picked = [t for row in W for t in names[np.argsort(row)[::-1][:per]]]
    return sorted(set(picked), key=lambda t: (-len(t), t))[:topk]


def mask_text(text, terms, lang):
    """
    Replace every occurrence with one placeholder. Longest term first, so
    "open cover" is consumed before "open". Latin terms get word
    boundaries; CJK n-grams are matched as bare substrings.
    """
    for t in terms:
        if lang in LATIN:
            text = re.sub(rf"\b{re.escape(t)}\b", PLACEHOLDER, text,
                          flags=re.IGNORECASE)
        else:
            text = text.replace(t, PLACEHOLDER)
    return text


def write_masked(recs, fields, corpus):
    """Mask every record, write in input order so rows stay aligned."""
    out = corpus.with_name(f"{corpus.stem}.masked.jsonl")
    if out.exists():
        print(f"  reusing {out.name}")
        return out

    terms = {}
    for lg in sorted({r["language"] for r in recs}):
        terms[lg] = discriminative_terms(recs, lg, TOPK)
        print(f"  {lg}: {len(terms[lg])} terms, e.g. "
              f"{', '.join(terms[lg][:5])}")

    with out.open("w") as f:
        for r in recs:
            m = mask_text(r["proof"], terms[r["language"]], r["language"])
            rec = {k: v for k, v in r.items() if k in fields}
            f.write(json.dumps({**rec, "proof": m}, ensure_ascii=False) + "\n")
    print(f"  wrote {out.name} ({len(recs)} records, input order)")
    return out


def embed(masked):
    """embedding.py is the only file that loads the model."""
    if cache_for(masked).exists():
        print(f"  embeddings already cached")
        return
    print(f"  embedding {masked.name} ...")
    subprocess.run([sys.executable, str(ROOT / "analysis" / "embedding.py"),
                    "--corpus", str(masked), "--no-figures"], check=True)


def transfer_matrix(X, recs, label="technique"):
    """
    Train a probe on each language, apply it frozen to every other. All
    ordered pairs, so no pair is chosen by hand.

    Read the Latin<->CJK block, not the overall mean: those are the pairs
    with no shared script and no cognates. The worst pair is the honesty
    check -- a high mean with one pair at chance is a different result from
    a uniform one.
    """
    langs = sorted({r["language"] for r in recs})
    le = LabelEncoder().fit([r[label] for r in recs])
    idx = {l: [j for j, r in enumerate(recs) if r["language"] == l]
           for l in langs}
    clf = {l: LogisticRegression(max_iter=3000).fit(
               X[idx[l]], le.transform([recs[j][label] for j in idx[l]]))
           for l in langs}

    n = len(langs)
    A = np.full((n, n), np.nan)
    for a, src in enumerate(langs):
        for b, tgt in enumerate(langs):
            if a != b:
                A[a, b] = clf[src].score(
                    X[idx[tgt]], le.transform([recs[j][label] for j in idx[tgt]]))

    print("        " + "".join(f"{l:>7}" for l in langs) + "   (rows: trained on)")
    for a, src in enumerate(langs):
        print(f"  {src:<6}" + "".join(
            "      -" if a == b else f"{A[a, b]:7.3f}" for b in range(n)))

    off = ~np.eye(n, dtype=bool)
    cross = np.array([[(langs[a] in CJK) != (langs[b] in CJK)
                       for b in range(n)] for a in range(n)])
    return dict(mean=np.nanmean(A[off]),
                cjk=np.nanmean(A[off & cross]),
                worst=np.nanmin(A[off]),
                chance=1 / len(le.classes_))


def infinitude_of_primes_ablation():
    corpus = config.CORPORA["infinitude_of_primes"]

    recs = [json.loads(l) for l in corpus.open() if l.strip()]
    fields = set().union(*(r.keys() for r in recs))
    T = [r for r in recs if r["arm"] == "technique"]
    keep = [i for i, r in enumerate(recs) if r["arm"] == "technique"]

    print("\nmasking")
    masked_corpus = write_masked(recs, fields, corpus)
    embed(masked_corpus)

    X = np.load(cache_for(corpus))[keep]
    Xm = np.load(cache_for(masked_corpus))[keep]

    print("\nunmasked")
    u = transfer_matrix(X, T)
    print("\nmasked")
    m = transfer_matrix(Xm, T)

    print(f"\n  {'':<16}{'unmasked':>10}{'masked':>10}{'drop':>8}")
    for k in ("mean", "cjk", "worst"):
        print(f"  {k:<16}{u[k]:10.3f}{m[k]:10.3f}{u[k] - m[k]:+8.3f}")
    print(f"  {'chance':<16}{u['chance']:10.3f}")


if __name__ == "__main__":
    infinitude_of_primes_ablation()