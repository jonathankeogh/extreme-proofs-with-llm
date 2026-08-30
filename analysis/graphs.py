"""
Publication figures for the write-up, drawn from the same data analyse.py
reads: the three corpora and their cached bge-m3 embeddings. Nothing here
embeds anything, and nothing here invents a number -- every figure is
recomputed with the same method as the analyse.py stage it illustrates,
except the masking sweep, whose values are transcribed from stage 5's
printed table (recomputing it would mean duplicating the entire masking
pipeline; the numbers are the ones the write-up quotes).

    fig1_lengths.png        stage 1 s3–s5: normalised length by direction, with
                            the unprompted centre cell as the origin
    fig2_direction_technique.png
                            stage 4 s4: which known argument each extremal
                            direction selects, per theorem
    fig3_transfer.png       stage 3 s11: cross-lingual technique transfer
                            on raw embeddings, en -> the other five
    fig4_lexical.png        stage 3 s9: a bag of words against the encoder,
                            within English, style held out
    fig5_masking.png        stage 5: the lexical baseline as vocabulary is
                            removed (values from the stage 5 run)
    fig6_coordinates.png    stage 1 length x registry lookup: the two
                            coordinates that survive -- normalised length
                            and named results invoked
    fig7_sphere.png         the anisotropy picture: the whole corpus is a
                            cap on the unit sphere, and what centring does
                            to it

Usage:
    uv run analysis/graphs.py           # writes figures/*.png
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

from embedding import ROOT, cache_for
from lookup_tables import REGISTRIES, compile_registry

OUT = ROOT / "figures"

THEOREMS = ("primes", "sqrt2", "pythagoras")
TITLES = {"primes": "infinitude of primes",
          "sqrt2": "irrationality of √2",
          "pythagoras": "Pythagorean theorem"}
CENTRE = {"primes": "infinitude", "sqrt2": "sqrt2",
          "pythagoras": "pythagoras"}
REGISTRY_KEY = {"primes": "infinitude_of_primes"}

# A readable order for the eight directions: shortest argument first,
# heaviest machinery last -- the order the write-up discusses them in.
DIRECTIONS = ("brevity", "elementarity", "nonvisuality", "constructiveness",
              "surprise", "visuality", "generality", "machinery")

# ----------------------------------------------------------------- style
# Colours follow the reference data-viz palette, light mode. Categorical
# hues are assigned in fixed slot order and never cycled; sequential is
# one blue, light -> dark.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3
SERIES = {"primes": S1, "sqrt2": S2, "pythagoras": S3}

BLUES = LinearSegmentedColormap.from_list(
    "seq-blue", [SURFACE, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                 "#256abf", "#184f95", "#0d366b"])

mpl.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 200,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
})


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def load(theorem):
    corpus = ROOT / "generate_proofs" / f"proofs_{theorem}.jsonl"
    with corpus.open() as f:
        recs = [json.loads(l) for l in f if l.strip()]
    X = np.load(cache_for(corpus))
    assert len(recs) == X.shape[0], f"{corpus}: corpus/embedding mismatch"
    return recs, X


def zlength(recs):
    """
    Within-language z of log character length, estimated on the technique
    and extreme arms only (stage 1's baseline: scope records prove other
    theorems, so their length is not a fact about language) -- but scored
    for every record, so the centre cell can be placed on the same scale.
    """
    by_lang = defaultdict(list)
    for r in recs:
        if r["arm"] != "scope":
            by_lang[r["language"]].append(math.log(len(r["proof"])))
    stats = {k: (np.mean(v), np.std(v) or 1.0) for k, v in by_lang.items()}
    return [(math.log(len(r["proof"])) - stats[r["language"]][0])
            / stats[r["language"]][1] for r in recs]


def classify_extreme(recs, X):
    """
    Stage 4's nearest-centroid assignment, verbatim in method: language
    means over the non-scope arms, centre and renormalise, centroids from
    the technique arm, cosine assignment of the extreme arm.
    """
    keep = [i for i, r in enumerate(recs) if r["arm"] != "scope"]
    recs = [recs[i] for i in keep]
    X = X[keep]
    mu = {lang: X[[j for j, r in enumerate(recs)
                   if r["language"] == lang]].mean(0)
          for lang in {r["language"] for r in recs}}
    Xc = np.stack([X[j] - mu[r["language"]] for j, r in enumerate(recs)])
    Xc /= np.clip(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-9, None)

    tech_i = [j for j, r in enumerate(recs) if r["arm"] == "technique"]
    extr_i = [j for j, r in enumerate(recs) if r["arm"] == "extreme"]
    techs = sorted({recs[j]["technique"] for j in tech_i})
    C = np.stack([Xc[[j for j in tech_i if recs[j]["technique"] == t]].mean(0)
                  for t in techs])
    C /= np.clip(np.linalg.norm(C, axis=1, keepdims=True), 1e-9, None)
    best = np.argmax(Xc[extr_i] @ C.T, axis=1)

    tab = np.zeros((len(DIRECTIONS), len(techs)), int)
    for k, j in enumerate(extr_i):
        tab[DIRECTIONS.index(recs[j]["direction"]), best[k]] += 1
    chi2 = chi2_contingency(tab)[0]
    v = math.sqrt(chi2 / (tab.sum() * (min(tab.shape) - 1)))
    return tab, techs, chi2, v


# ================================================================= fig 1
def fig_lengths(data):
    """
    One panel per theorem: mean normalised length per direction (dot),
    per-language means (small ticks -- the spread language contributes),
    and the unprompted centre cell as a dashed origin line.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), sharex=True,
                             sharey=True)
    order = DIRECTIONS[::-1]                    # brevity at the bottom
    ys = np.arange(len(order))
    for ax, t in zip(axes, THEOREMS):
        recs, _ = data[t]
        z = zlength(recs)
        ax.axvline(0, color=GRID, lw=0.8, zorder=0)
        for y, d in zip(ys, order):
            per_lang = [np.mean([z[j] for j, r in enumerate(recs)
                                 if r["arm"] == "extreme"
                                 and r["direction"] == d
                                 and r["language"] == lg])
                        for lg in sorted({r["language"] for r in recs})]
            ax.plot(per_lang, [y] * len(per_lang), "|", color=SERIES[t],
                    alpha=0.35, ms=7, mew=1.2, zorder=2)
            ax.plot(np.mean(per_lang), y, "o", color=SERIES[t], ms=6,
                    zorder=3)
        centre = np.mean([z[j] for j, r in enumerate(recs)
                          if r["arm"] == "scope"
                          and r["id"].split("__")[1] == CENTRE[t]])
        ax.axvline(centre, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(centre, -0.62, "unprompted", color=INK2, fontsize=8,
                ha="center", va="top")
        ax.set_ylim(-1.1, len(order) - 0.5)
        ax.set_title(TITLES[t], fontsize=10)
        ax.set_yticks(ys, order)
        ax.tick_params(axis="y", length=0)
        despine(ax, keep=("bottom",))
        ax.set_xlabel("length (z of log chars, within language)")
    fig.text(0.005, 1.09, "Direction prompts move length far more than "
             "language does", fontsize=12, fontweight="bold", va="bottom")
    fig.text(0.005, 1.03, "dot = direction mean · ticks = the six language "
             "means · dashed = the model unprompted (the scope arm's "
             "centre cell)", fontsize=9, color=INK2, va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_lengths.png", bbox_inches="tight")
    plt.close(fig)


# ================================================================= fig 2
def fig_direction_technique(data):
    """
    The replicated finding: direction x assigned technique, one heatmap
    per theorem. Counts out of 30 (6 languages x 5 samples).
    """
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
    for ax, t in zip(axes, THEOREMS):
        recs, X = data[t]
        tab, techs, chi2, v = classify_extreme(recs, X)
        ax.imshow(tab, cmap=BLUES, vmin=0, vmax=30, aspect="auto")
        for a in range(tab.shape[0]):
            for b in range(tab.shape[1]):
                if tab[a, b]:
                    ax.text(b, a, tab[a, b], ha="center", va="center",
                            fontsize=8,
                            color="#ffffff" if tab[a, b] > 16 else INK)
        ax.set_xticks(range(len(techs)),
                      [s.replace("_", " ") for s in techs],
                      rotation=35, ha="right")
        ax.set_yticks(range(len(DIRECTIONS)),
                      DIRECTIONS if ax is axes[0] else [])
        ax.tick_params(length=0)
        despine(ax, keep=())
        ax.set_title(f"{TITLES[t]}\n"
                     f"Cramér's V = {v:.2f}, permutation p < 1/20000",
                     fontsize=9.5)
    axes[0].text(0, 1.22, "Each extremal direction selects a known argument",
                 transform=axes[0].transAxes, fontsize=12, fontweight="bold",
                 va="bottom")
    axes[0].text(0, 1.15, "extreme-arm records assigned to the nearest "
                 "technique centroid, counts of 30 per direction",
                 transform=axes[0].transAxes, fontsize=9, color=INK2,
                 va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_direction_technique.png", bbox_inches="tight")
    plt.close(fig)


# ================================================================= fig 3
def fig_transfer(data):
    """
    Stage 3 s11: technique probe trained on English, applied frozen to the
    other five languages, on raw (uncentred) embeddings.
    """
    langs = ("de", "es", "fr", "ja", "zh")
    M = np.zeros((len(THEOREMS), len(langs)))
    for a, t in enumerate(THEOREMS):
        recs, X = data[t]
        i = [j for j, r in enumerate(recs) if r["arm"] == "technique"]
        le = LabelEncoder().fit([recs[j]["technique"] for j in i])
        tr = [j for j in i if recs[j]["language"] == "en"]
        clf = LogisticRegression(max_iter=3000).fit(
            X[tr], le.transform([recs[j]["technique"] for j in tr]))
        for b, lg in enumerate(langs):
            te = [j for j in i if recs[j]["language"] == lg]
            M[a, b] = clf.score(
                X[te], le.transform([recs[j]["technique"] for j in te]))

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.imshow(M, cmap=BLUES, vmin=0.5, vmax=1.0, aspect="auto")
    for a in range(M.shape[0]):
        for b in range(M.shape[1]):
            ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center",
                    fontsize=9, color="#ffffff" if M[a, b] > 0.85 else INK)
    ax.set_xticks(range(len(langs)), langs)
    ax.set_yticks(range(len(THEOREMS)), [TITLES[t] for t in THEOREMS])
    ax.tick_params(length=0)
    despine(ax, keep=())
    ax.set_xlabel("technique probe trained on English, applied frozen to…")
    ax.set_title("Cross-lingual technique transfer on raw embeddings\n"
                 "chance 0.17–0.20 · an earlier 768-d encoder managed "
                 "0.62 (zh), 0.50 (ja)",
                 fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_transfer.png", bbox_inches="tight")
    plt.close(fig)


# ================================================================= fig 4
def fig_lexical(data):
    """
    Stage 3 s9: within English, train on one style and test on the other
    (both directions, averaged). A bag of words against the encoder.
    """
    kinds = (("TF-IDF word 1–2gram", S1,
              dict(analyzer="word", ngram_range=(1, 2))),
             ("TF-IDF char 3–5gram", S2,
              dict(analyzer="char_wb", ngram_range=(3, 5))),
             ("bge-m3 embedding", S3, None))
    acc = np.zeros((len(THEOREMS), len(kinds)))
    chance = []
    for a, t in enumerate(THEOREMS):
        recs, X = data[t]
        i = [j for j, r in enumerate(recs)
             if r["arm"] == "technique" and r["language"] == "en"]
        y = LabelEncoder().fit_transform([recs[j]["technique"] for j in i])
        styles = [recs[j]["style"] for j in i]
        txt = [recs[j]["proof"] for j in i]
        chance.append(1 / len(set(y)))
        for b, (_, _, kw) in enumerate(kinds):
            scores = []
            for s_tr, s_te in (("terse", "verbose"), ("verbose", "terse")):
                tr = [k for k, s in enumerate(styles) if s == s_tr]
                te = [k for k, s in enumerate(styles) if s == s_te]
                if kw is None:
                    E = X[i]
                    clf = LogisticRegression(max_iter=3000).fit(E[tr], y[tr])
                    scores.append(clf.score(E[te], y[te]))
                else:
                    pipe = make_pipeline(
                        TfidfVectorizer(min_df=2, **kw),
                        LogisticRegression(max_iter=3000))
                    pipe.fit([txt[k] for k in tr], y[tr])
                    scores.append(pipe.score([txt[k] for k in te], y[te]))
            acc[a, b] = np.mean(scores)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    xs = np.arange(len(THEOREMS))
    w = 0.24
    for b, (label, colour, _) in enumerate(kinds):
        bars = ax.bar(xs + (b - 1) * (w + 0.02), acc[:, b], width=w,
                      color=colour, label=label, zorder=3)
        for r in bars:
            r.set_path_effects([])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, color=INK2, padding=2)
    for a, c in enumerate(chance):
        ax.plot([a - 0.42, a + 0.42], [c, c], color=INK2, lw=1,
                ls=(0, (3, 2)), zorder=4)
    ax.text(chance.index(min(chance)) - 0.46, min(chance) - 0.01, "chance",
            fontsize=8, color=INK2, ha="right", va="center")
    ax.set_xticks(xs, [TITLES[t] for t in THEOREMS])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.12)
    ax.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
    ax.grid(axis="y", zorder=0)
    despine(ax, keep=("bottom",))
    ax.set_ylabel("technique accuracy, English only\n"
                  "(train on one style, test on the other)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=3,
              columnspacing=1.4, handlelength=1.2)
    ax.set_title("A bag of words nearly matches the encoder — and on "
                 "√2 beats it", loc="left", fontsize=11,
                 fontweight="bold", pad=32)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_lexical.png", bbox_inches="tight")
    plt.close(fig)


# ================================================================= fig 5
def fig_masking(data):
    """
    The discriminative masking sweep, recomputed from the corpora with
    stage 5's own selection and masking (ablation.py): terms chosen by
    classifier weight on terse English records, the corpus masked, and
    TF-IDF scored train-terse / test-verbose within English.
    """
    from ablation import PLACEHOLDER, discriminative_terms, mask_text

    ks = (0, 10, 50, 100, 400)
    sweep, kept = {}, {}
    for t in THEOREMS:
        recs, _ = data[t]
        en = [r for r in recs
              if r["arm"] == "technique" and r["language"] == "en"]
        y = LabelEncoder().fit_transform([r["technique"] for r in en])
        tr = [i for i, r in enumerate(en) if r["style"] == "terse"]
        te = [i for i, r in enumerate(en) if r["style"] == "verbose"]
        row = []
        for k in ks:
            terms = discriminative_terms(recs, "en", k) if k else []
            txt = [mask_text(r["proof"], terms, "en") for r in en]
            pipe = make_pipeline(
                TfidfVectorizer(min_df=2, analyzer="word",
                                ngram_range=(1, 2)),
                LogisticRegression(max_iter=3000))
            pipe.fit([txt[i] for i in tr], y[tr])
            row.append(pipe.score([txt[i] for i in te], y[te]))
        sweep[t] = tuple(row)
        kept[t] = (sum(len(s.replace(PLACEHOLDER, "")) for s in txt)
                   / sum(len(r["proof"]) for r in en))
        print(f"  masking sweep {t}: "
              + "  ".join(f"{a:.3f}" for a in sweep[t])
              + f"   text kept at {ks[-1]}: {kept[t]:.2f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    xs = np.arange(len(ks))
    nudge = {"primes": 6, "sqrt2": -8, "pythagoras": 0}
    for t in THEOREMS:
        ax.plot(xs, sweep[t], "-o", color=SERIES[t], lw=2, ms=5, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(TITLES[t], (xs[-1], sweep[t][-1]),
                    xytext=(8, nudge[t]), textcoords="offset points",
                    fontsize=8.5, color=SERIES[t], va="center")
    ax.axhline(0.200, color=INK2, lw=1, ls=(0, (3, 2)), zorder=1)
    ax.axhline(0.167, color=INK2, lw=1, ls=(0, (3, 2)), zorder=1)
    ax.text(-0.05, 0.207, "chance (5 techniques)", fontsize=7.5, color=INK2)
    ax.text(-0.05, 0.128, "chance (6 techniques)", fontsize=7.5, color=INK2)
    ax.set_xticks(xs, ks)
    ax.set_xlim(-0.2, 4.9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", zorder=0)
    despine(ax)
    ax.set_xlabel("word types masked per language (by classifier weight)")
    ax.set_ylabel("TF-IDF technique accuracy")
    ax.set_title("No small set of keywords carries the technique label",
                 loc="left", fontsize=11, fontweight="bold", pad=26)
    lo, hi = min(kept.values()), max(kept.values())
    ax.text(0, 1.03, "a slope, not a cliff: ~400 word types before a bag "
            "of words is at or near chance, with "
            f"{lo:.0%}–{hi:.0%} of the English text still on the page",
            transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_masking.png", bbox_inches="tight")
    plt.close(fig)


# ================================================================= fig 8
def fig_masked_transfer():
    """
    Stage 5's answer to the open question: cross-lingual technique
    transfer on the primes corpus, before and after masking the 400
    highest-weight word types per language -- the corpus on which the
    TF-IDF baseline is at chance. Values transcribed from the stage 5
    run, like fig5 (recomputing them would duplicate the pipeline).
    """
    langs = ("de", "en", "es", "fr", "ja", "zh")
    unmasked = np.array([
        [np.nan, 1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, np.nan, 1.000, 0.980, 0.920, 0.900],
        [1.000, 1.000, np.nan, 0.980, 0.960, 0.980],
        [1.000, 1.000, 1.000, np.nan, 0.920, 0.900],
        [1.000, 0.960, 1.000, 0.880, np.nan, 1.000],
        [0.940, 0.900, 0.960, 0.840, 1.000, np.nan]])
    masked = np.array([
        [np.nan, 0.860, 0.820, 0.720, 0.500, 0.540],
        [0.880, np.nan, 0.920, 0.900, 0.680, 0.720],
        [0.960, 0.960, np.nan, 0.940, 0.700, 0.820],
        [0.960, 0.900, 0.960, np.nan, 0.400, 0.380],
        [0.640, 0.560, 0.580, 0.580, np.nan, 0.960],
        [0.400, 0.500, 0.460, 0.480, 0.980, np.nan]])

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.1),
                             gridspec_kw={"wspace": 0.08})
    for ax, M, head, mu in ((axes[0], unmasked, "unmasked", 0.967),
                            (axes[1], masked, "masked (400 word types "
                             "per language)", 0.722)):
        ax.imshow(np.ma.masked_invalid(M), cmap=BLUES, vmin=0.2, vmax=1.0)
        for a in range(6):
            for b in range(6):
                if a == b:
                    continue
                ax.text(b, a, f"{M[a, b]:.2f}".lstrip("0"), ha="center",
                        va="center", fontsize=8,
                        color="#ffffff" if M[a, b] > 0.75 else INK)
        # the CJK block: the corner masking hits hardest
        for x0, y0 in ((3.5, -0.5), (-0.5, 3.5)):
            ax.add_patch(mpl.patches.Rectangle(
                (x0, y0), 2 if x0 > 0 else 4, 2 if y0 > 0 else 4,
                fill=False, edgecolor=S2, lw=1.4,
                linestyle=(0, (3, 2)), clip_on=False))
        ax.set_xticks(range(6), langs)
        ax.set_yticks(range(6), langs if ax is axes[0] else [])
        ax.tick_params(length=0)
        despine(ax, keep=())
        ax.set_title(f"{head}\nmean {mu:.3f}", fontsize=9.5)
        ax.set_xlabel("technique probe applied to…")
    axes[0].set_ylabel("trained on")
    axes[0].text(0, 1.30, "With the give-away vocabulary gone, transfer "
                 "drops but does not die", transform=axes[0].transAxes,
                 fontsize=12, fontweight="bold", va="bottom")
    axes[0].text(0, 1.19, "infinitude of primes, bge-m3 · dashed: the "
                 "Latin/CJK blocks, 0.94 to 0.56 · chance 0.20 · the "
                 "same masked corpus puts TF-IDF at chance",
                 transform=axes[0].transAxes, fontsize=9, color=INK2,
                 va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / "fig8_masked_transfer.png", bbox_inches="tight")
    plt.close(fig)


# ================================================================= fig 6
def fig_coordinates(data):
    """
    Stage 1 length x registry lookup on primes: the two coordinates that
    mean something. x = mean
    normalised length, y = mean distinct named results invoked, one point
    per direction; Furstenberg's technique-arm records as the reference
    the embedding confused machinery with.
    """
    recs, _ = data["primes"]
    compiled = compile_registry(REGISTRIES[REGISTRY_KEY["primes"]])

    def n_invoked(text):
        n = 0
        for _, rxs, subs in compiled:
            if (any(r.search(text) for r in rxs)
                    or any(s in text for s in subs)):
                n += 1
        return n

    z = zlength(recs)
    pts = {}
    for d in DIRECTIONS:
        i = [j for j, r in enumerate(recs)
             if r["arm"] == "extreme" and r["direction"] == d]
        pts[d] = (np.mean([z[j] for j in i]),
                  np.mean([n_invoked(recs[j]["proof"]) for j in i]))
    fu = [j for j, r in enumerate(recs)
          if r["arm"] == "technique" and r["technique"] == "furstenberg"]
    fu_pt = (np.mean([z[j] for j in fu]),
             np.mean([n_invoked(recs[j]["proof"]) for j in fu]))

    # hand-placed label offsets (points); the low-count directions crowd
    # the bottom-right corner
    off = {"brevity": (10, 0, "left"), "elementarity": (0, 10, "center"),
           "nonvisuality": (-9, -2, "right"), "surprise": (0, 12, "center"),
           "visuality": (8, -10, "left"), "constructiveness": (10, 2, "left"),
           "generality": (10, 0, "left"), "machinery": (-12, 0, "right")}

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.set_yscale("symlog", linthresh=2)
    ax.set_yticks((0, 1, 2, 5, 10, 20), ("0", "1", "2", "5", "10", "20"))
    ax.grid(axis="both", zorder=0)
    for d, (x, y) in pts.items():
        heavy = d == "machinery"
        ax.plot(x, y, "o", ms=9 if heavy else 7,
                color=S2 if heavy else S1, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        dx, dy, ha = off[d]
        ax.annotate(d, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, color=INK, va="center", ha=ha)
    ax.plot(*fu_pt, "o", ms=8, markerfacecolor="none", markeredgecolor=S2,
            markeredgewidth=1.6, zorder=3)
    ax.annotate("Fürstenberg (technique arm) —\nwhere the embedding put\n"
                "the machinery records",
                fu_pt, xytext=(-14, 26), textcoords="offset points",
                fontsize=8, color=INK2, ha="right")
    despine(ax)
    ax.set_xlim(-2.35, 1.35)
    ax.set_xlabel("coordinate 1: length (z of log chars, within language)")
    ax.set_ylabel("coordinate 2: distinct named results invoked (log scale)")
    ax.set_title("Two coordinates that mean something — and only two",
                 loc="left", fontsize=11, fontweight="bold", pad=26)
    ax.text(0, 1.03, "infinitude of primes, extreme arm · machinery "
            "invokes 16.6 named results where Fürstenberg invokes 0.3",
            transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_coordinates.png", bbox_inches="tight")
    plt.close(fig)


# ================================================================= fig 7
# Two panels asking two questions of the same axis, so they are drawn in
# one frame and the second is what the first becomes.
#
#   1. Where do the raw embeddings sit? In a cap: every record is 31-49 deg
#      from the corpus mean direction, because they all share a topic.
#   2. What does removing the per-language mean do to that? It moves the
#      cloud off the pole and onto the great circle perpendicular to it:
#      81-99 deg from the same axis. The common direction is gone.
#
# The frame is estimated once, on the raw cloud, and the centred vectors
# are scored against it. Re-deriving a frame per panel would have been the
# mistake: a panel drawn on its own axes cannot show what happened to the
# other one, and any axis fitted to the centred cloud puts its own records
# near 90 deg from itself, which says nothing.
#
# The polar angle is exact in both panels -- the arccos of a 1024-dimensional
# inner product, not of a projected one -- so the cap and the belt are the
# real cap and the real belt. Only the azimuth is a projection, and it uses
# the same two directions in both panels, so a record keeps its longitude
# from one to the other.

# Staging, shared by both panels so the pair is a before/after and not two
# different pictures. The axis leans well out of the page (LEAN) so the cap
# is seen face-on rather than edge-on -- but not so far that the belt in the
# second panel collapses onto the silhouette, which is what a fully face-on
# axis would do to a cloud sitting at 90 degrees from it.
LEAN, TIP = 1.15, 0.42


def polar_frame(X, frame=None):
    """
    Unit rows -> unit rows in R^3, exact in polar angle, approximate in
    azimuth. Returns the points, their cosine to the pole, and the frame
    used -- pass that frame back in to place another cloud on the same
    picture.
    """
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
    if frame is None:
        pole = X.mean(0)
        pole /= np.linalg.norm(pole)
        R = X - np.outer(X @ pole, pole)         # residual, orthogonal to pole
        _, _, Vt = np.linalg.svd(R - R.mean(0), full_matrices=False)
        frame = (pole, Vt[0], Vt[1])
    pole, v1, v2 = frame
    cos = np.clip(X @ pole, -1.0, 1.0)
    R = X - np.outer(cos, pole)
    phi = np.arctan2(R @ v2, R @ v1)
    sin = np.sqrt(np.clip(1 - cos ** 2, 0, None))
    return (np.stack([sin * np.cos(phi), sin * np.sin(phi), cos], 1),
            cos, frame)


def camera(elev, azim):
    """Unit vector from the origin toward the viewer."""
    e, a = math.radians(elev), math.radians(azim)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a),
                     math.sin(e)])


def tilt_to(target):
    """
    Rotation taking the pole (0, 0, 1) to `target`. Used to lean the mean
    direction out of the page toward the viewer, so the cap is seen face-on
    as a disc rather than edge-on as a sliver at the top of the globe --
    an orientation choice only: every angle in the picture is preserved.
    """
    d = target / np.linalg.norm(target)
    k = np.cross((0, 0, 1), d)
    s = np.linalg.norm(k)
    if s < 1e-9:
        return np.eye(3)
    k /= s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - d[2]) * (K @ K)      # Rodrigues


def front_only(xyz, cam, margin=-0.02):
    """A curve with its far-side vertices lifted out, so it breaks at the limb."""
    out = xyz.copy()
    out[xyz @ cam < margin] = np.nan
    return out


def draw_sphere(ax, P, colours, band, elev=14, azim=-64, lean=0.72,
                tip=0.0, axis_line=True):
    """
    A shaded unit sphere with the corpus on it and, shaded in, the zone of
    polar angle `band` (in radians) the records occupy, the whole thing
    rotated so that zone faces the viewer.

    `lean` pulls the axis out of the page toward the camera, `tip` leans it
    across the page to the right. Both are staging: the rotation is rigid,
    so every angle drawn is the angle measured.

    Two rendering details matter. The surface is lit by hand (a Lambert
    term against a light over the viewer's shoulder) because matplotlib's
    own shading keys off the facet normals of a coarse mesh and bands
    badly on a sphere. And the scatter is split by the camera direction
    and drawn either side of the surface, because a 3-d scatter is not
    occluded by a surface otherwise -- the far side of the cloud would
    read as the near side.
    """
    cam = camera(elev, azim)
    right = np.cross((0, 0, 1.0), cam)          # screen right, in world axes
    right /= np.linalg.norm(right)
    R = tilt_to(cam * lean + np.array([0.0, 0.0, 1.0]) + right * tip)
    P = P @ R.T

    u = np.linspace(0, 2 * np.pi, 120)
    v = np.linspace(0, np.pi, 60)
    S = np.stack([np.outer(np.cos(u), np.sin(v)),
                  np.outer(np.sin(u), np.sin(v)),
                  np.outer(np.ones_like(u), np.cos(v))], -1)
    light = camera(elev + 34, azim - 26)
    lam = np.clip(S @ light, 0, 1) ** 0.7
    face = np.empty(lam.shape + (4,))
    for c in range(3):
        lo = int(GRID[1 + 2 * c:3 + 2 * c], 16) / 255 * 0.93
        hi = int(SURFACE[1 + 2 * c:3 + 2 * c], 16) / 255
        face[..., c] = lo + (hi - lo) * lam
    face[..., 3] = 0.62
    ax.plot_surface(*S.transpose(2, 0, 1), facecolors=face, linewidth=0,
                    shade=False, antialiased=True, rstride=1, cstride=1,
                    zorder=2)

    # graticule on the tilted frame: faint all the way round, and drawn
    # again over the surface on the near side so the globe has a grain
    for k in range(0, 120, 10):                 # meridians
        t = np.linspace(0, np.pi, 90)
        m = np.stack([np.sin(t) * np.cos(u[k]), np.sin(t) * np.sin(u[k]),
                      np.cos(t)], 1) @ R.T
        ax.plot(*m.T, color=GRID, lw=0.4, alpha=0.5, zorder=1)
        ax.plot(*front_only(m * 1.002, cam).T, color=GRID, lw=0.5,
                alpha=0.9, zorder=4)
    for c in np.linspace(-0.75, 0.75, 5):       # parallels
        r = math.sqrt(1 - c ** 2)
        p = np.stack([r * np.cos(u), r * np.sin(u), np.full_like(u, c)], 1) @ R.T
        ax.plot(*p.T, color=GRID, lw=0.4, alpha=0.5, zorder=1)
        ax.plot(*front_only(p * 1.002, cam).T, color=GRID, lw=0.5,
                alpha=0.9, zorder=4)

    # the limb: the silhouette circle, which is what gives the shaded
    # surface a hard edge instead of a soft one against the page
    e1 = np.cross(cam, (0, 0, 1.0))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(cam, e1)
    ax.plot(*(np.outer(np.cos(u), e1) + np.outer(np.sin(u), e2)).T,
            color=BASELINE, lw=1.0, alpha=0.9, zorder=9)

    # The zone the cloud occupies, between two angles from the axis: a cap
    # (0 deg to something) before centring, a belt straddling 90 deg after.
    # Same construction, same colour, so the move from one to the other is
    # the thing the pair of panels shows.
    if band is not None:
        vb = np.linspace(*band, 40)
        B = np.stack([np.outer(np.cos(u), np.sin(vb)),
                      np.outer(np.sin(u), np.sin(vb)),
                      np.outer(np.ones_like(u), np.cos(vb))], -1) @ R.T
        ax.plot_surface(*(B * 1.004).transpose(2, 0, 1), color=S1,
                        alpha=0.14, linewidth=0, shade=False, zorder=3)
        for t in band:
            if t <= 1e-6:
                continue
            rim = np.stack([math.sin(t) * np.cos(u), math.sin(t) * np.sin(u),
                            np.full_like(u, math.cos(t))], 1) @ R.T * 1.006
            ax.plot(*rim.T, color=S1, lw=1.0, alpha=0.3, zorder=1)
            ax.plot(*front_only(rim, cam).T, color=S1, lw=1.6, alpha=0.9,
                    zorder=6)
    if axis_line:
        ax.plot(*np.stack([(0, 0, 0), R[:, 2] * 1.14], 1), color=INK2,
                lw=0.9, ls=(0, (4, 3)), zorder=10)

    front = P @ cam > 0
    for mask, alpha, z in ((~front, 0.14, 0), (front, 0.9, 8)):
        ax.scatter(*(P[mask] * 1.012).T, s=5.5, c=colours[mask],
                   linewidths=0, alpha=alpha, depthshade=False, zorder=z)

    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-0.78, 0.78)
    ax.set_ylim(-0.78, 0.78)
    ax.set_zlim(-0.78, 0.78)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    return R


def fig_sphere(data):
    """
    Two questions of one axis. Left: where the raw embeddings sit, which
    is a cap around the corpus mean direction -- they share a topic, so
    they share a direction. Right: what stage 3's centring does to that,
    which is to move the whole cloud off the pole and onto the great
    circle perpendicular to it.

    Both panels are drawn in the frame estimated from the raw cloud, so
    the second is literally what the first becomes.
    """
    raw, cent, colour = [], [], []
    for t in THEOREMS:
        recs, X = data[t]
        X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
        raw.append(X)
        mu = {lg: X[[j for j, r in enumerate(recs)
                     if r["language"] == lg]].mean(0)
              for lg in {r["language"] for r in recs}}
        cent.append(np.stack([X[j] - mu[r["language"]]
                              for j, r in enumerate(recs)]))
        colour += [SERIES[t]] * len(recs)
    colour = np.array(colour)

    specs = (
        # The raw zone is shaded from the pole out, not from its inner
        # edge: the claim is that the cloud lives inside a cap, and a
        # donut would draw the 31 deg inner edge as though it bounded
        # something. The belt has no such reading -- both its edges are
        # real -- so it is drawn as it is measured.
        dict(A=np.vstack(raw), from_pole=True,
             head="raw embeddings",
             sub="every record within {far_ax:.0f}° of the axis "
                 "because they all share a topic\n"
                 "furthest pair anywhere in the corpus: {far:.0f}° "
                 "(cos {lo:.2f}) · mean pair cos {mu:.2f}"),
        dict(A=np.vstack(cent), from_pole=False,
             head="per-language mean removed, within theorem",
             sub="a belt at {near:.0f}–{far_ax:.0f}° from the same axis\n"
                 "furthest pair: {far:.0f}° (cos {lo:.2f}) · "
                 "mean pair cos {mu:.2f}"),
    )

    # One frame, estimated on the raw cloud, used for both panels: the
    # right-hand picture is what the left-hand one becomes.
    frame = None
    fig = plt.figure(figsize=(10.6, 5.6))
    for k, spec in enumerate(specs):
        A = spec["A"]
        A = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-9, None)
        S = A @ A.T
        pair = S[np.triu_indices_from(S, 1)]
        P, cos, frame = polar_frame(A, frame)
        band = (0.0 if spec["from_pole"] else math.acos(float(cos.max())),
                math.acos(float(cos.min())))

        # computed_zorder=False: Axes3D otherwise sorts whole collections
        # by their mean depth and paints the sphere over the near-side
        # scatter, which is the difference between a cloud on the surface
        # and a cloud seen through frosted glass.
        ax = fig.add_subplot(1, 2, k + 1, projection="3d",
                             computed_zorder=False)
        R = draw_sphere(ax, P, colour, band, lean=LEAN, tip=TIP)
        ax.text(*(R[:, 2] * 1.18), "mean direction", fontsize=8,
                ha="center", va="bottom", color=INK2, zorder=11)
        ax.text2D(0.5, 0.995, spec["head"], transform=ax.transAxes,
                  fontsize=10, ha="center", va="top", color=INK)
        ax.text2D(0.5, 0.115,
                  spec["sub"].format(
                      near=math.degrees(band[0]),
                      far_ax=math.degrees(band[1]),
                      far=math.degrees(math.acos(float(pair.min()))),
                      lo=float(pair.min()), mu=float(pair.mean())),
                  transform=ax.transAxes, fontsize=8.5, ha="center",
                  va="top", color=INK2, linespacing=1.6)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=5,
                          markerfacecolor=SERIES[t], markeredgecolor="none",
                          label=TITLES[t]) for t in THEOREMS]
    fig.legend(handles=handles, loc="lower center", ncols=3, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3,
               columnspacing=1.8)
    fig.text(0.008, 1.115, "Anisotropy of the corpus and removal of the language directions",
             fontsize=12, fontweight="bold", va="bottom")
    fig.text(0.008, 1.050, "each point is one proof, projected down to the three dimensional subspace spanned by the mean direction of the corpus and pc1 & pc2, then normalised to unit",
             fontsize=9, color=INK2, va="bottom", linespacing=1.6)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0.05, wspace=0.02)
    fig.savefig(OUT / "fig7_sphere.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    data = {t: load(t) for t in THEOREMS}
    fig_lengths(data)
    fig_direction_technique(data)
    fig_transfer(data)
    fig_lexical(data)
    fig_masking(data)
    fig_masked_transfer()
    fig_coordinates(data)
    fig_sphere(data)
    for p in sorted(OUT.glob("fig*.png")):
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
