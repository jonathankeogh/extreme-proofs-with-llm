"""
Masking ablation: is technique identity carried by terminology?

Section 9 of probes.py bounds the vocabulary confound from below: on the
primes corpus, TF-IDF word 1-2grams recover the technique at 0.940 against
a chance of 0.200, and bge-m3 manages 0.980. A bag of words comes within
four points of the encoder. That is consistent with two very different
stories, and section 9 cannot separate them:

  (a) the encoder reads the argument, and vocabulary happens to correlate
      with the argument well enough that counting words nearly matches it;

  (b) the encoder reads the vocabulary, and there is no argument structure
      in the representation at all.

The cross-language results do not settle it either, because the tokens that
give a technique away are the ones that are never translated. "Furstenberg",
"Fermat", the Euler product sign -- these appear identically in the German
and Chinese proofs. What looks like the encoder aligning arguments across
languages may be nothing more than it aligning proper nouns.

This script removes those tokens and lets the two stories come apart. It
rewrites the corpus with the give-away terms replaced by a single neutral
placeholder, so that:

  * the masked corpus embeds through analyse.py unchanged -- same records,
    same order, same row count, so probes.py can compare the two spaces
    row by row;

  * the reference machinery is taken from the coordinates.py registry,
    which already enumerates the surface forms in all six languages, rather
    than from a fresh word list invented here.

Masking tiers, weakest to strongest:

  names       Mathematicians' names only. The narrowest intervention: it
              leaves every mathematical term in place and removes only the
              labels. If technique survives this, the representation is at
              least not keyed on proper nouns.
  machinery   The coordinates.py registry: named theorems, lemmas and
              constructions. Leaves prose and notation.
  both        names + machinery. The headline number.
  notation    Technique-diagnostic symbols and formulae. The most
              aggressive and the least defensible tier -- see NOTATION.
  all         Everything above.
  discriminative
              `all`, plus the terms a classifier actually leans on, found
              from the data. The registry tiers turn out to remove nothing:
              masking every named theorem, every mathematician and every
              diagnostic symbol leaves the lexical baseline exactly where it
              was, at 0.940. The technique label is not in the citations, it
              is in the ordinary descriptive vocabulary, and only a
              data-driven list reaches it. This is the tier to use.

What this script cannot do, and reports instead of hiding:

  1. Masking leaves a hole of a known size. Even with the words gone, the
     text still says how many were removed and where. Section 3 measures
     that leak from the features a reader of the masked text can actually
     see -- placeholder count, length, density -- and not from which term
     each placeholder replaced, since every placeholder is the same string
     and that information is absent from the corpus that gets embedded.

  2. Paraphrase survives masking. "The product of the zeta function over
     primes" states the Euler product without naming it. This ablation puts
     an upper bound on the lexical contribution, not an exact figure.

  3. At the point where the lexical baseline dies, roughly a tenth of the
     text is gone. Section 2 checks that the loss is even across techniques,
     because brevity is one of the value functions under study and a
     technique-dependent shortening would replace one confound with another.

Usage:
    python mask.py --corpus proofs_primes.jsonl --tier discriminative --curve
    python mask.py --corpus proofs_primes.jsonl --tier discriminative --topk 400
    python mask.py --corpus proofs_primes.jsonl --tier both --sample 1

Writes proofs_primes.masked-<tier>.jsonl, then embed that as usual:
    python analyse.py --corpus proofs_primes.masked-discriminative.jsonl
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

from coordinates import compile_forms, select_registry
from paths import cache_for

PLACEHOLDER = "⟨m⟩"

# Mathematicians' names. The registries in coordinates.py carry these only
# inside longer phrases ("Euclid's lemma", "Fermat numbers"), because their
# job is to count invoked machinery. A bare surname is not machinery, but it
# is exactly the kind of token that gives a technique away, so it needs its
# own list. Latin-script forms cover en/de/fr/es together, including the
# diacritic variants; the CJK transliterations are separate strings.
NAMES = [
    ("euclid", [r"re:Euclid\w*", r"re:Euklid\w*", r"re:Eucli?de\w*",
                "欧几里得", "エウクレイデス", "ユークリッド"]),
    ("euler", [r"re:Euler\w*", r"re:Eulersch\w*", "欧拉", "オイラー"]),
    ("fermat", [r"re:Fermat\w*", "费马", "フェルマー"]),
    ("erdos", [r"re:Erd[őoö]s", "埃尔德什", "厄尔多斯", "エルデシュ",
               "エルデーシュ"]),
    ("furstenberg", [r"re:F[üu]rstenberg", r"re:Furstenberg",
                     "菲尔斯滕贝格", "フルステンベルグ"]),
    ("mersenne", [r"re:Mersenne\w*", "梅森", "メルセンヌ"]),
    ("riemann", [r"re:Riemann\w*", "黎曼", "リーマン"]),
    ("dirichlet", [r"re:Dirichlet\w*", "狄利克雷", "ディリクレ"]),
    ("chebyshev", [r"re:Chebyshev", r"re:Tschebysch\w*", r"re:Tchebych\w*",
                   "切比雪夫", "チェビシェフ"]),
    ("bertrand", [r"re:Bertrand", "伯特兰", "ベルトラン"]),
    ("legendre", [r"re:Legendre", "勒让德", "ルジャンドル"]),
    ("gauss", [r"re:Gauss\w*", r"re:Gau[ßs]\w*", "高斯", "ガウス"]),
    ("bezout", [r"re:B[ée]zout", "裴蜀", "ベズー"]),
    ("zsigmondy", [r"re:Zsigmondy"]),
    ("sylvester", [r"re:Sylvester"]),
    ("kummer", [r"re:Kummer"]),
    ("stieltjes", [r"re:Stieltjes"]),
    ("perott", [r"re:Perott"]),
    ("auric", [r"re:Auric"]),
    ("metrod", [r"re:M[ée]trod"]),
    ("thue", [r"re:Thue"]),
    # sqrt2
    ("tennenbaum", [r"re:Tennenbaum", "テネンバウム"]),
    ("bashmakova", [r"re:Bashmakova", r"re:Ba[sš]makova"]),
    ("conway", [r"re:Conway", "康威", "コンウェイ"]),
    ("guy", [r"re:Richard Guy", r"re:Conway-?Guy"]),
    ("laczkovich", [r"re:Laczkovich"]),
    ("dedekind", [r"re:Dedekind", "戴德金", "デデキント"]),
    ("eisenstein", [r"re:Eisenstein"]),
    ("gelfond", [r"re:Gelfond", r"re:Gel'?fond"]),
    ("liouville", [r"re:Liouville", "刘维尔", "リウヴィル"]),
    # pythagoras
    ("pythagoras", [r"re:Pythagor\w*", r"re:pitag[óo]ric\w*",
                    "毕达哥拉斯", "ピタゴラス", "勾股"]),
    ("garfield", [r"re:Garfield", "加菲尔德", "ガーフィールド"]),
    ("ptolemy", [r"re:Ptolem\w*", r"re:Ptolem[äa]\w*", r"re:Ptol[ée]m\w*",
                 "托勒密", "プトレマイオス"]),
    ("de_gua", [r"re:de Gua", r"re:De Gua"]),
    ("cauchy", [r"re:Cauchy", "柯西", "コーシー"]),
    ("schwarz", [r"re:Schwarz\w*", "施瓦茨", "シュワルツ"]),
    ("heron", [r"re:Heron\w*", r"re:H[ée]ron\w*", "海伦", "ヘロン"]),
    # "Tales" is the Spanish spelling, and also the ordinary Spanish word
    # for "such". compile_forms matches case-insensitively, so the bare form
    # took 24 hits in the Spanish primes proofs -- "enteros m tales que
    # m^2 | n" -- and not one of them was Thales. --audit caught it. The
    # full name is required instead; the other spellings are unambiguous.
    ("thales", [r"re:Thales", r"re:Thal[èe]s", r"re:Tales de Mileto",
                "泰勒斯", "タレス"]),
    ("loomis", [r"re:Loomis"]),
    ("bhaskara", [r"re:Bh[āa]skara", "婆什迦罗", "バースカラ"]),
    ("liu_hui", [r"re:Liu Hui", "刘徽"]),
    ("zhoubi", [r"re:Zhoubi", r"re:Zhou Bi", "周髀"]),
]

# Technique-diagnostic notation. This tier is the least defensible of the
# five and is kept separate for that reason: some of these strings are
# ordinary mathematics that any proof might use, so masking them damages
# proofs that were not relying on them, and a drop in accuracy here is
# partly a drop in readability. Reported, but never as the headline.
NOTATION = [
    ("euler_product_notation", [r"re:\\zeta", "ζ", "∏", r"re:\\prod"]),
    ("fermat_tower", [r"re:2\^\{?2\^", r"re:2\^\(2\^", r"re:F_?\{?n\}?\s*=",
                      "2^(2^n)", "2^{2^n}"]),
    ("factorial_plus_one", [r"re:n!\s*\+\s*1", r"re:N!\s*\+\s*1"]),
    ("euclid_product_plus_one", [
        r"re:p_?1\s*p_?2\s*\\?cdots?", r"re:\\prod_?\{?i",
        r"re:p_?1\s*\\cdot"]),
    ("topology_notation", [r"re:N_?\{?a,\s*b\}?", r"re:\\mathcal\{[OT]\}"]),
    ("congruence", [r"re:\\pmod", r"re:\\bmod", r"re:\bmod\b", "≡"]),
    ("radical", [r"re:\\sqrt", "√"]),
]

TIERS = {
    "names": ["names"],
    "machinery": ["machinery"],
    "both": ["names", "machinery"],
    "notation": ["notation"],
    "all": ["names", "machinery", "notation"],
    "discriminative": ["names", "machinery", "notation", "discriminative"],
}

# Languages whose tokens the word analyser can find. Japanese and Chinese
# are not space-delimited, so the discriminative tier scores character
# n-grams for them instead of words -- see discriminative_terms.
LATIN = {"en", "de", "es", "fr"}


def analyser_for(lang):
    if lang in LATIN:
        return dict(analyzer="word", ngram_range=(1, 1))
    return dict(analyzer="char_wb", ngram_range=(2, 4))


def discriminative_terms(recs, lang, topk, fit_style="terse"):
    """
    The terms a classifier actually leans on, found from the data rather
    than from a list of names.

    The registry tiers above turn out to remove nothing: masking every
    named theorem, every mathematician and every diagnostic symbol leaves
    the TF-IDF baseline exactly where it was. So the technique label is not
    carried by citations. It is carried by the ordinary descriptive
    vocabulary that goes with an argument -- a topological proof says "open",
    "cover", "clopen"; a counting proof says "at most", "square-free",
    "bound" -- and no hand-written list will catch that, because the list
    would have to enumerate the whole technical lexicon of each method.

    This finds those terms directly: fit a linear model on TF-IDF, then take
    the tokens with the largest weight for each technique.

    The selection sees only `fit_style`, and the caller evaluates on the
    other style. That matters. Choosing the terms to mask by looking at the
    labels of the very records you then test on would guarantee a collapse
    and prove nothing -- the accuracy would be measuring the selection, not
    the representation. Because the mask must be applied to the whole corpus
    before embedding, this asymmetry cannot be made two-sided, so the
    discriminative tier reports one direction only and says so.
    """
    i = [j for j, r in enumerate(recs)
         if r["arm"] == "technique" and r["language"] == lang
         and r["style"] == fit_style]
    if len(i) < 5:
        return []
    vec = TfidfVectorizer(min_df=2, **analyser_for(lang))
    M = vec.fit_transform([recs[j]["proof"] for j in i])
    y = LabelEncoder().fit_transform([recs[j]["technique"] for j in i])
    clf = LogisticRegression(max_iter=3000).fit(M, y)
    names = np.array(vec.get_feature_names_out())
    W = np.atleast_2d(clf.coef_)
    per = max(1, topk // max(len(W), 1))
    picked = []
    for row in W:
        picked += list(names[np.argsort(row)[::-1][:per]])
    # Longest first, so "open cover" is consumed before "open".
    return sorted(set(picked), key=len, reverse=True)[:topk]


def build_groups(registry, which):
    """
    Compile the surface forms for the requested groups into one list of
    (name, regexes, substrings), the same shape compile_registry produces.
    """
    out = []
    if "names" in which:
        out += [(f"name:{n}", *compile_forms(f)) for n, f in NAMES]
    if "machinery" in which:
        out += [(f"mach:{n}", *compile_forms(f)) for n, f in registry]
    if "notation" in which:
        out += [(f"notn:{n}", *compile_forms(f)) for n, f in NOTATION]
    return out


def term_groups(terms, lang):
    """
    Compile data-driven terms into the same group shape. Latin-script terms
    get word boundaries so that masking "open" does not also gut "opening";
    CJK n-grams are matched as bare substrings, which is what they are.
    """
    out = []
    for t in terms:
        if lang in LATIN:
            out.append((f"disc:{t}",
                        [re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)],
                        []))
        else:
            out.append((f"disc:{t}", [], [t]))
    return out


def spans(text, groups):
    """
    Every (start, end, group_name) match in the text, from all groups.

    Overlaps are resolved longest-first so that "Euclid's lemma" is replaced
    as one unit rather than leaving a fragment behind when the bare name
    "Euclid" also matches inside it.
    """
    found = []
    for name, rx, subs in groups:
        for r in rx:
            found += [(m.start(), m.end(), name) for m in r.finditer(text)]
        for s in subs:
            i = text.find(s)
            while i != -1:
                found.append((i, i + len(s), name))
                i = text.find(s, i + 1)
    found.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    kept, last = [], -1
    for start, end, name in found:
        if start >= last:
            kept.append((start, end, name))
            last = end
    return kept


def mask_text(text, groups):
    """Replace every matched span with PLACEHOLDER. Returns (text, counts)."""
    kept = spans(text, groups)
    counts = Counter(name for _, _, name in kept)
    for start, end, _ in reversed(kept):
        text = text[:start] + PLACEHOLDER + text[end:]
    return text, counts


# ---------------------------------------------------------------- 1

def report_coverage(recs, per_rec_counts):
    """How much was masked, and how unevenly across techniques."""
    tech = defaultdict(list)
    for r, c in zip(recs, per_rec_counts):
        if r["arm"] == "technique":
            tech[r["technique"]].append(sum(c.values()))
    print(f"  {'technique':<20} {'masks/proof':>12} {'min':>5} {'max':>5}")
    for t in sorted(tech):
        v = tech[t]
        print(f"  {t:<20} {np.mean(v):12.1f} {min(v):5} {max(v):5}")
    allc = [sum(c.values()) for c in per_rec_counts]
    print(f"  {'ALL RECORDS':<20} {np.mean(allc):12.1f} {min(allc):5} "
          f"{max(allc):5}")
    zero = sum(1 for n in allc if n == 0)
    print(f"  records with nothing masked: {zero}/{len(allc)}")


# ---------------------------------------------------------------- 2

def report_shrinkage(recs, masked):
    """
    Masking must not shorten the proofs differently by technique, or the
    ablation trades a vocabulary confound for a length confound. Brevity is
    one of the value functions under study, so length is not a nuisance
    parameter here -- it is a coordinate.
    """
    by = defaultdict(list)
    for r, m in zip(recs, masked):
        if r["arm"] == "technique":
            by[r["technique"]].append(len(m) / max(len(r["proof"]), 1))
    print(f"  {'technique':<20} {'length kept':>12}")
    for t in sorted(by):
        print(f"  {t:<20} {np.mean(by[t]):12.3f}")


# ---------------------------------------------------------------- 3

def leak_check(recs, per_rec_counts, masked, lang="en", seed=0):
    """
    The control that decides whether the ablation means anything.

    Masking leaves a hole of a known size. Even with the words gone, the
    text still says how many terms were removed and where they sat, and a
    proof carrying six hundred placeholders is not the same object as one
    carrying eighty. If technique can be read off that alone, the masked
    corpus still carries its label and a high masked accuracy would say
    nothing about argument structure.

    The features here are restricted to what a reader of the masked text can
    actually see: how many placeholders, how long the text is, how dense the
    placeholders are. Crucially NOT which term each placeholder replaced --
    every one of them is the same string, so that information is not in the
    corpus that gets embedded. An earlier version of this check used the
    per-term counts and reported a leak of 0.90 on the discriminative tier,
    which was an artefact of the check rather than a property of the data:
    it was classifying from the mask log, not from the masked proofs.

    Trains on terse and tests on verbose, and the reverse, matching
    probes.py section 9 so the numbers sit on the same scale.
    """
    i = [j for j, r in enumerate(recs)
         if r["arm"] == "technique" and r["language"] == lang]
    if not i:
        print(f"  no technique-arm records in '{lang}'; skipped.")
        return
    y = LabelEncoder().fit_transform([recs[j]["technique"] for j in i])
    styles = [recs[j]["style"] for j in i]
    chance = 1 / len(set(y))
    print(f"  {len(i)} records in '{lang}', chance {chance:.3f}")

    def observable(j):
        n = sum(per_rec_counts[j].values())
        L = max(len(masked[j]), 1)
        return [n, L, n / L, len(recs[j]["proof"]) / L]

    V = np.array([observable(j) for j in i], dtype=float)

    def score(F):
        accs = []
        for tr_s, te_s in (("terse", "verbose"), ("verbose", "terse")):
            tr = [k for k, s in enumerate(styles) if s == tr_s]
            te = [k for k, s in enumerate(styles) if s == te_s]
            if not tr or not te:
                continue
            clf = LogisticRegression(max_iter=3000).fit(F[tr], y[tr])
            accs.append(clf.score(F[te], y[te]))
        return float(np.mean(accs)) if accs else float("nan")

    acc = score(V)
    print(f"  technique from placeholder COUNT + length {acc:.3f}")
    print(f"  technique from COUNT alone                "
          f"{score(V[:, :1]):.3f}")
    if acc > chance * 2:
        print("  -> the hole leaks. Masked-text accuracy is an upper bound")
        print("     on structure and cannot be read as evidence for it.")
    else:
        print("  -> the hole carries little; the masked text is a fair test.")
    print("  (for reference, not a leak the encoder can exploit: every")
    print("   placeholder is the same string, so which term was removed is")
    print("   absent from the corpus that gets embedded.)")


# ---------------------------------------------------------------- 4

def tfidf_on_masked(recs, masked, lang="en"):
    """
    The result available without re-embedding anything.

    Section 9 asked whether TF-IDF recovers technique from the original
    text; it does, at 0.940. Asking the same question of the masked text
    says how much surface signal the masking actually removed. If TF-IDF
    stays high, the masking failed and there is no point embedding the
    result. If it falls to chance, the masked corpus is a real test, and
    whatever the encoder scores on it is about something other than words.
    """
    i = [j for j, r in enumerate(recs)
         if r["arm"] == "technique" and r["language"] == lang]
    if not i:
        print(f"  no technique-arm records in '{lang}'; skipped.")
        return
    y = LabelEncoder().fit_transform([recs[j]["technique"] for j in i])
    styles = [recs[j]["style"] for j in i]
    print(f"  {len(i)} records in '{lang}', chance {1/len(set(y)):.3f}")

    for label, texts in (("original", [recs[j]["proof"] for j in i]),
                         ("masked", [masked[j] for j in i])):
        for kind, kw in (("word 1-2gram",
                          dict(analyzer="word", ngram_range=(1, 2))),
                         ("char 3-5gram",
                          dict(analyzer="char_wb", ngram_range=(3, 5)))):
            accs = []
            for tr_s, te_s in (("terse", "verbose"), ("verbose", "terse")):
                tr = [k for k, s in enumerate(styles) if s == tr_s]
                te = [k for k, s in enumerate(styles) if s == te_s]
                if not tr or not te:
                    continue
                pipe = make_pipeline(
                    TfidfVectorizer(min_df=2, **kw),
                    LogisticRegression(max_iter=3000))
                pipe.fit([texts[k] for k in tr], y[tr])
                accs.append(pipe.score([texts[k] for k in te], y[te]))
            print(f"  {label:<9} tfidf {kind:<14} {np.mean(accs):.3f}")


# ---------------------------------------------------------------- 5

def ablation_curve(recs, base_groups, lang, ks):
    """
    Accuracy against how much vocabulary has been taken away.

    A single masked number is hard to read: if accuracy stays high, was the
    mask too small, or is there real structure? The curve settles it by
    shape rather than by level. Two extremes bracket the answer:

      a cliff        Technique rests on a few keywords. Masking the top
                     handful drops accuracy to chance, and the "structure"
                     in the representation was a lookup table.
      a slope        Technique is spread across the whole technical lexicon.
                     Every token removed costs a little and none is
                     decisive. This is the more interesting outcome, and it
                     is genuinely ambiguous: an argument and the words used
                     to state it are not separable in prose, so a gentle
                     slope is as consistent with the encoder tracking the
                     mathematics as with it tracking diffuse wording.

    Trains on terse, tests on verbose, one direction only, because the terms
    were selected on terse -- see discriminative_terms.
    """
    i = [j for j, r in enumerate(recs)
         if r["arm"] == "technique" and r["language"] == lang]
    if not i:
        print(f"  no technique-arm records in '{lang}'; skipped.")
        return
    y_all = LabelEncoder().fit_transform([recs[j]["technique"] for j in i])
    styles = [recs[j]["style"] for j in i]
    tr = [k for k, s in enumerate(styles) if s == "terse"]
    te = [k for k, s in enumerate(styles) if s == "verbose"]
    if not tr or not te:
        print("  needs both styles; skipped.")
        return
    print(f"  {len(i)} records in '{lang}', chance "
          f"{1/len(set(y_all)):.3f}, train terse -> test verbose")
    print(f"  {'top-k masked':>14} {'tfidf':>8} {'kept':>8} "
          f"{'example terms'}")
    for k in ks:
        terms = discriminative_terms(recs, lang, k) if k else []
        groups = base_groups + term_groups(terms, lang)
        texts, kept = [], []
        for j in i:
            m, _ = mask_text(recs[j]["proof"], groups)
            texts.append(m)
            kept.append(len(m) / max(len(recs[j]["proof"]), 1))
        pipe = make_pipeline(TfidfVectorizer(min_df=2, **analyser_for(lang)),
                            LogisticRegression(max_iter=3000))
        pipe.fit([texts[k2] for k2 in tr], y_all[tr])
        acc = pipe.score([texts[k2] for k2 in te], y_all[te])
        ex = ", ".join(terms[:3]) if terms else "(registry tiers only)"
        print(f"  {k:>14} {acc:8.3f} {np.mean(kept):8.3f}  {ex}")


# ---------------------------------------------------------------- 6

def excerpts(name, lang, groups, texts, limit=2, pad=45):
    """A little context around the first few matches, for eyeballing."""
    entry = next((g for g in groups if g[0] == name), None)
    if entry is None:
        return []
    _, rx, subs = entry
    out = []
    for t in texts[lang]:
        spots = [m.start() for r in rx for m in [r.search(t)] if m]
        spots += [t.find(s) for s in subs if s in t]
        if spots:
            i = min(spots)
            out.append(t[max(0, i - pad):i + pad].replace("\n", " "))
        if len(out) >= limit:
            break
    return out


def audit(recs, groups, min_en=5):
    """
    Whether the hand-written lists are any good, measured against the corpus
    rather than asserted.

    NAMES and NOTATION were written from memory, and the sqrt2 and Pythagoras
    registries in coordinates.py are marked PROVISIONAL by that file's own
    VALIDATED set. A list nobody has checked fails in two directions, and the
    two look identical in the headline number:

      false negatives  A term the proofs use and the list misses. Masking
                       then removes less than it claims, and a null result
                       ("masking changed nothing") is indistinguishable from
                       an incomplete list. This is the failure that would
                       make the registry-tier null meaningless.
      false positives  A pattern that matches ordinary text. This corrupts
                       the masked corpus by deleting words the proof needed,
                       and it does so silently.

    The per-language counts catch both, because the same mathematics is being
    written six times. An entry that fires in every language at a similar
    rate is doing its job.

    An entry that fires in exactly one language is AMBIGUOUS, and this was
    got wrong once already. Both of these look identical in the counts:

      a collision      "Tales" is Thales in Spanish and also the ordinary
                       Spanish word for "such". It took 24 hits in the
                       Spanish primes proofs and none of them was Thales.
                       The mask was deleting a function word.
      a coverage gap   The Chinese proofs say 剪拼 for scissors-congruence
                       and the English ones say "dissection", which the
                       registry does not list. The entry is correct; the
                       other five languages' forms are missing. Fixing this
                       means adding forms, not removing one.

    Nothing in the counts separates those, so this prints an excerpt from
    each flagged entry rather than reporting a verdict it cannot support.
    Read the excerpt: if the match is ordinary prose, it is a collision; if
    it is the mathematics, the entry is right and its siblings are absent.
    """
    langs = sorted({r["language"] for r in recs})
    texts = defaultdict(list)
    for r in recs:
        texts[r["language"]].append(r["proof"])

    rows, dead, suspect, gaps = [], [], [], []
    for name, rx, subs in groups:
        row = {}
        for l in langs:
            row[l] = sum(
                1 for t in texts[l]
                if any(r.search(t) for r in rx) or any(s in t for s in subs))
        total = sum(row.values())
        if total == 0:
            dead.append(name)
            continue
        rows.append((name, row, total))
        firing = [l for l in langs if row[l]]
        if len(firing) == 1 and row[firing[0]] >= 3:
            suspect.append((name, firing[0], row[firing[0]]))
        elif row.get("en", 0) >= min_en and not (row.get("ja", 0)
                                                 or row.get("zh", 0)):
            gaps.append((name, row.get("en", 0)))

    print(f"  {'entry':<24} " + " ".join(f"{l:>5}" for l in langs))
    for name, row, _ in sorted(rows, key=lambda t: -t[2]):
        print(f"  {name:<24} " + " ".join(f"{row[l]:5}" for l in langs))

    print(f"\n  never matched ({len(dead)}): harmless here, but unvalidated. "
          f"Entries for\n  another theorem's proofs are expected to sit at "
          f"zero on this corpus.")
    if dead:
        print("    " + ", ".join(sorted(dead)))

    print(f"\n  FIRES IN ONE LANGUAGE ONLY ({len(suspect)}): needs an eye, "
          f"not a verdict.")
    print("  Either a collision with an ordinary word in that language, or a")
    print("  correct entry whose other five forms are missing. The excerpt")
    print("  below decides which; the counts cannot.")
    if suspect:
        for name, l, n in suspect:
            print(f"\n    {name}: {n} hits, all in '{l}'")
            for ex in excerpts(name, l, groups, texts, limit=2):
                print(f"      ...{ex}...")
    else:
        print("    none.")

    print(f"\n  LIKELY MISSING TRANSLITERATIONS ({len(gaps)}): in English, "
          f"absent in ja and zh.")
    if gaps:
        for name, n in gaps:
            print(f"    {name} -- {n} English records, nothing in CJK")
    else:
        print("    none.")


def rule(t):
    print(f"\n{t}\n{'-' * len(t)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path,
                    default=Path("proofs_primes.jsonl"))
    ap.add_argument("--tier", default="both", choices=sorted(TIERS))
    ap.add_argument("--theorem", default=None,
                    help="override the registry choice; normally taken "
                         "from the corpus's own theorem field")
    ap.add_argument("--lang", default="en",
                    help="language for the within-language diagnostics")
    ap.add_argument("--sample", type=int, default=0,
                    help="print this many masked excerpts for eyeballing")
    ap.add_argument("--topk", type=int, default=200,
                    help="terms per language for the discriminative tier")
    ap.add_argument("--audit", action="store_true",
                    help="check the hand-written lists against the corpus "
                         "for false positives and missing forms; writes "
                         "nothing")
    ap.add_argument("--curve", action="store_true",
                    help="sweep top-k and print accuracy against it; "
                         "writes nothing")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.corpus.exists():
        raise SystemExit(f"{args.corpus} not found.")
    recs = [json.loads(l) for l in args.corpus.open() if l.strip()]
    theorem, registry = select_registry(recs, args.theorem)
    groups = build_groups(registry, TIERS[args.tier])

    print(f"{len(recs)} records from {args.corpus}")
    print(f"theorem  {theorem}")
    print(f"tier     {args.tier}  ({', '.join(TIERS[args.tier])})")
    print(f"patterns {len(groups)} groups")

    if args.audit:
        rule("6. Audit: do the hand-written lists match the corpus?")
        audit(recs, groups)
        return

    if args.curve:
        rule("5. Ablation curve: accuracy against vocabulary removed")
        ablation_curve(recs, groups, args.lang,
                       [0, 5, 10, 25, 50, 100, 200, 400])
        return

    # The discriminative terms are per language: the tokens that give a
    # topological proof away in German are not the German spellings of the
    # English ones, and for ja/zh they are character n-grams rather than
    # words at all. So each record is masked with its own language's terms
    # on top of the shared registry groups.
    per_lang = {}
    if "discriminative" in TIERS[args.tier]:
        for lang in sorted({r["language"] for r in recs}):
            terms = discriminative_terms(recs, lang, args.topk)
            per_lang[lang] = term_groups(terms, lang)
            print(f"  {lang}: {len(terms)} terms, e.g. "
                  f"{', '.join(terms[:5])}")

    masked, counts = [], []
    for r in recs:
        g = groups + per_lang.get(r["language"], [])
        m, c = mask_text(r["proof"], g)
        masked.append(m)
        counts.append(c)

    rule("1. Masking coverage")
    report_coverage(recs, counts)

    rule("2. Length kept after masking")
    report_shrinkage(recs, masked)

    rule("3. Does the hole leak the label?")
    leak_check(recs, counts, masked, args.lang)

    rule("4. Lexical baseline, original vs masked")
    tfidf_on_masked(recs, masked, args.lang)

    if args.sample:
        rule("5. Masked excerpts")
        for r, m in list(zip(recs, masked))[:args.sample]:
            print(f"\n  [{r['id']}]")
            print("  " + m[:600].replace("\n", "\n  "))

    stem = args.corpus.stem
    out = args.out or args.corpus.with_name(f"{stem}.masked-{args.tier}.jsonl")
    with out.open("w") as f:
        for r, m in zip(recs, masked):
            f.write(json.dumps({**r, "proof": m, "proof_original": r["proof"],
                                "mask_tier": args.tier},
                               ensure_ascii=False) + "\n")
    print(f"\nWrote {out} ({len(recs)} records, same order as the input).")
    print(f"Its embedding cache will be {cache_for(out)}.")
    print("Embed it, then compare the two spaces:")
    print(f"  python analyse.py --corpus {out}")
    print(f"  python probes.py --corpus {args.corpus} --masked-corpus {out}")


if __name__ == "__main__":
    main()
