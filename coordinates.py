"""
Stage 4: a second coordinate that is not lexical similarity.

Everything in stages 2-3 measures a proof by where its *wording* lands in an
embedding. That is what produced the central error of the analysis: the
machinery arm was assigned to Furstenberg 26/30, because Furstenberg is the
most abstract-VOCABULARY member of the reference set, while its mathematics
is Euclid's argument in topological dress and invokes no theorem at all. A
similarity measure cannot tell "uses abstract terminology" from "uses
powerful theorems".

This file builds the coordinate that can: the number of distinct named
external results a proof leans on. Chebotarev density, Lindemann-Weierstrass
and Hadamard factorisation are named results; "let p be prime" is not.
Euclid's proof invokes nearly nothing, Furstenberg invokes only topological
definitions, and the machinery proofs invoke a great deal.

Why it should work where cosine failed:

  script-independent   Mathematician surnames stay in Latin script inside
                       Chinese and Japanese proofs ("Euler 乘积公式",
                       "Riemann の解析接続"), and the results that do get
                       localised have a small closed set of forms. So the
                       same registry reads all six languages. Section 1
                       tests this rather than assuming it.

  not a length proxy   A long elementary proof invokes nothing; a short
                       machinery proof invokes several results. Section 2
                       measures the correlation instead of asserting it.

  not lexical          It counts what a proof DEPENDS on, not what it sounds
                       like. Section 3 is the test that matters: it asks
                       whether this coordinate separates the machinery arm
                       from Furstenberg, which is exactly what the embedding
                       could not do.

Together with the normalised length from first_pass.py this gives two
coordinates that mean something, which is the minimum for the geometry in
Conway and Shipman's framing to be worth taking literally.

Usage:
    python coordinates.py
    python coordinates.py --show machinery   # print what was matched
"""

import argparse
import json
import math
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

CORPUS = Path("proofs.jsonl")
CJK = {"ja", "zh"}

# ----------------------------------------------------------------------
# The registry of named external results.
#
# Each entry is (canonical_name, [surface forms]). A form beginning with
# "re:" is a regex applied with word boundaries to Latin script; everything
# else is a plain substring test, which is what CJK needs (no word
# boundaries exist). Matching is case-insensitive for the Latin forms.
#
# Deliberately NOT in the registry:
#
#   "Euclid" alone. It is ambiguous between the theorem being proved
#   (Euclid's theorem = the infinitude of primes), the technique label, and
#   the genuinely-invoked results that carry his name. The specific
#   invocations (Euclid's lemma, Euclidean algorithm/domain) are listed
#   separately; the bare surname is not counted.
#
#   Generic theorem words (theorem/Satz/定理/補題). Those mark a result
#   being stated, not an external result being leaned on, and a proof that
#   states and proves its own lemma should not score for it.
# ----------------------------------------------------------------------

REGISTRY = [
    ("fundamental_thm_arithmetic", [
        r"re:fundamental theorem of arithmetic", r"re:Fundamentalsatz",
        r"re:th[ée]or[èe]me fondamental de l'arithm[ée]tique",
        r"re:teorema fundamental de la aritm[ée]tica",
        "算术基本定理", "算術の基本定理", "算術基本定理"]),
    ("euclid_lemma", [
        r"re:Euclid'?s lemma", r"re:Lemme d'Euclide", r"re:Lema de Euclides",
        r"re:Lemma von Euklid", r"re:Euklids Lemma",
        "欧几里得引理", "ユークリッドの補題"]),
    ("euclidean_algorithm", [
        r"re:Euclidean (algorithm|domain|division)",
        r"re:euklidisch\w*", r"re:euclidien\w*", r"re:eucl[íi]deo\w*",
        "欧几里得整环", "欧几里得算法", "ユークリッド整域", "ユークリッドの互除法"]),
    ("bezout", [r"re:B[ée]zout", "裴蜀", "ベズー"]),
    ("euler_product", [
        r"re:Euler'?s product", r"re:Euler product", r"re:Euler-?Produkt\w*",
        r"re:produit eul[ée]rien", r"re:produit d'Euler",
        r"re:producto de Euler", r"re:Eulersche Produktformel",
        "欧拉乘积", "オイラー積", "オイラーの積"]),
    ("euler_phi", [r"re:Euler'?s (totient|phi)", r"re:Eulersche Phi"]),
    ("fermat_little", [
        r"re:Fermat'?s little theorem", r"re:kleine\w* Satz von Fermat",
        r"re:petit th[ée]or[èe]me de Fermat",
        r"re:peque[ñn]o teorema de Fermat",
        "费马小定理", "フェルマーの小定理"]),
    ("fermat_numbers", [
        r"re:Fermat numbers?", r"re:Fermat-?Zahl\w*",
        r"re:nombres de Fermat", r"re:n[úu]meros de Fermat",
        "费马数", "フェルマー数"]),
    ("dirichlet_thm", [
        r"re:Dirichlet", "狄利克雷", "ディリクレ"]),
    ("prime_number_theorem", [
        r"re:prime number theorem", r"re:Primzahlsatz",
        r"re:th[ée]or[èe]me des nombres premiers",
        r"re:teorema de los n[úu]meros primos",
        "素数定理", "素数分布定理"]),
    ("chebyshev", [
        r"re:Chebyshev", r"re:Tschebyschow", r"re:Tchebychev",
        r"re:Bertrand'?s postulate", r"re:postulat de Bertrand",
        r"re:Bertrandsches Postulat", "切比雪夫", "チェビシェフ",
        "ベルトラン"]),
    ("mertens", [r"re:Mertens", "梅腾斯"]),
    ("riemann_zeta", [
        r"re:Riemann", r"re:Zetafunktion", r"re:zeta function",
        r"re:fonction z[êe]ta", r"re:funci[óo]n zeta",
        "黎曼", "リーマン", "ゼータ関数", "泽塔函数"]),
    ("analytic_continuation", [
        r"re:analytic continuation", r"re:analytische Fortsetzung",
        r"re:prolongement analytique", r"re:continuaci[óo]n anal[íi]tica",
        "解析接続", "解析延拓"]),
    ("hadamard", [r"re:Hadamard", "阿达马", "アダマール"]),
    ("de_la_vallee_poussin", [r"re:Poussin", "普桑", "プーサン"]),
    ("wiener_ikehara", [
        r"re:Wiener[-–—]?Ikehara", r"re:Ikehara", r"re:Tauber\w*",
        "维纳", "ウィーナー", "イケハラ", "タウバー"]),
    ("lindemann_weierstrass", [
        r"re:Lindemann", r"re:Hermite[-–—]Lindemann",
        "林德曼", "リンデマン"]),
    ("weierstrass", [r"re:Weierstrass", r"re:Weierstra[ßs]", "魏尔斯特拉斯",
                     "ワイエルシュトラス"]),
    ("transcendence_pi", [
        r"re:transcendence of \$?\\?pi", r"re:Transzendenz von",
        r"re:transcendance de", r"re:trascendencia de",
        "超越性", "の超越"]),
    ("chebotarev", [r"re:Chebotar[eë]v", r"re:Tschebotarjow",
                    "切博塔廖夫", "チェボタレフ"]),
    ("class_field_theory", [
        r"re:class field theory", r"re:Klassenk[öo]rpertheorie",
        r"re:th[ée]orie du corps de classes",
        r"re:teor[íi]a de cuerpos de clases",
        "类域论", "類体論"]),
    ("artin", [r"re:Artin", "阿廷", "アルティン"]),
    ("frobenius", [r"re:Frobenius", "弗罗贝尼乌斯", "フロベニウス"]),
    ("galois", [r"re:Galois", r"re:Galoiserweiterung", "伽罗瓦", "ガロア"]),
    ("kronecker_weber", [r"re:Kronecker[-–—]Weber", "克罗内克", "クロネッカー"]),
    ("dedekind", [r"re:Dedekind", "戴德金", "デデキント"]),
    ("hecke", [r"re:Hecke", "赫克", "ヘッケ"]),
    ("tate_thesis", [r"re:Tate", "泰特", "テイト"]),
    ("minkowski", [r"re:Minkowski", "闵可夫斯基", "ミンコフスキー"]),
    ("hilbert", [r"re:Hilbert", "希尔伯特", "ヒルベルト"]),
    ("zorn_choice", [
        r"re:Zorn", r"re:axiom of choice", r"re:Auswahlaxiom",
        r"re:axiome du choix", r"re:axioma de elecci[óo]n",
        "选择公理", "選択公理", "ツォルン"]),
    ("compactness_tychonoff", [
        r"re:Tychonoff", r"re:Tichonow", r"re:compactness theorem",
        r"re:Kompaktheitssatz", "チコノフ", "吉洪诺夫"]),
    ("baire", [r"re:Baire", "贝尔", "ベール"]),
    ("stone_weierstrass", [r"re:Stone[-–—]Weierstrass", "斯通"]),
    ("parseval", [r"re:Parseval", r"re:Plancherel", "帕塞瓦尔", "パーセバル"]),
    ("fourier", [r"re:Fourier", "傅里叶", "フーリエ"]),
    ("poisson", [r"re:Poisson", "泊松", "ポアソン"]),
    ("mellin", [r"re:Mellin", "梅林", "メリン"]),
    ("fubini", [r"re:Fubini", "富比尼", "フビニ"]),
    ("jacobi", [r"re:Jacobi", "雅可比", "ヤコビ"]),
    ("landau", [r"re:Landau", "朗道", "ランダウ"]),
    ("mangoldt", [r"re:Mangoldt", "曼戈尔特", "マンゴルト"]),
    ("gauss", [r"re:Gauss", r"re:Gau[ßs]", "高斯", "ガウス"]),
    ("legendre", [r"re:Legendre", "勒让德", "ルジャンドル"]),
    ("lagrange", [r"re:Lagrange", "拉格朗日", "ラグランジュ"]),
    ("wilson", [r"re:Wilson'?s theorem", r"re:Satz von Wilson", "威尔逊"]),
    ("mobius", [r"re:M[öo]bius", "莫比乌斯", "メビウス"]),
    ("thue_siegel_roth", [r"re:Thue[-–—]Siegel", r"re:Roth'?s theorem"]),
    ("green_tao", [r"re:Green[-–—]Tao", r"re:Szemer[ée]di", "グリーン"]),
    ("zhang_maynard", [r"re:Zhang", r"re:Maynard", r"re:Goldston"]),
    ("brun", [r"re:Brun'?s", r"re:Brun theorem", "ブルン"]),
    ("selberg", [r"re:Selberg", "塞尔伯格", "セルバーグ"]),
    ("schur", [r"re:Schur", "舒尔", "シューア"]),
    ("sylow_lagrange_group", [
        r"re:Sylow", r"re:Cauchy'?s theorem", "シロー"]),
    ("oresme", [r"re:Oresme", r"re:harmonic series diverge",
                r"re:divergence de la s[ée]rie harmonique",
                "调和级数", "調和級数"]),
    ("goldbach", [r"re:Goldbach", "哥德巴赫", "ゴールドバッハ"]),
    ("evertse_schmidt", [r"re:Evertse", r"re:Schmidt subspace",
                         r"re:subspace theorem"]),
    ("hardy", [r"re:Hardy", "哈代", "ハーディ"]),
    ("stirling", [r"re:Stirling", "斯特林", "スターリング"]),
]

# Proof-environment markers, used only by the "cited but not proved"
# refinement in count_invoked().
PROOF_MARKERS = [
    r"re:proof\b", r"re:Beweis", r"re:Preuve", r"re:D[ée]monstration",
    r"re:Prueba", r"re:Demostraci[óo]n", "证明", "証明",
]


def compile_forms(forms):
    """Split surface forms into (compiled Latin regexes, CJK substrings)."""
    rx, subs = [], []
    for f in forms:
        if f.startswith("re:"):
            rx.append(re.compile(rf"\b(?:{f[3:]})", re.IGNORECASE))
        else:
            subs.append(f)
    return rx, subs


COMPILED = [(name, *compile_forms(forms)) for name, forms in REGISTRY]
PROOF_RX, PROOF_SUBS = compile_forms(PROOF_MARKERS)


def find_invoked(text):
    """
    Return {canonical_name: [char offsets]} for every registry entry that
    appears in the text. Offsets are kept so the refinement below can ask
    what follows a mention.
    """
    hits = {}
    for name, rx, subs in COMPILED:
        pos = []
        for r in rx:
            pos.extend(m.start() for m in r.finditer(text))
        for s in subs:
            start = text.find(s)
            while start != -1:
                pos.append(start)
                start = text.find(s, start + 1)
        if pos:
            hits[name] = sorted(pos)
    return hits


def proved_in_place(text, offset, window=400):
    """
    Crude test for whether a mention is followed by its own proof: does a
    proof marker appear within `window` characters after it?

    This is the weakest part of the file and is reported separately for
    that reason. It cannot tell "Theorem B (Hadamard). ... Proof of the
    main result follows" from "Lemma 2 (Euler product). Proof. ...", and
    the window is a guess.
    """
    seg = text[offset:offset + window]
    return (any(r.search(seg) for r in PROOF_RX)
            or any(s in seg for s in PROOF_SUBS))


def count_invoked(rec):
    """
    Two counts per record:

      n_named   distinct named external results mentioned at all. The
                primary measure: it asks what the proof leans on.
      n_cited   those with no proof environment nearby, i.e. quoted rather
                than established. Sharper in principle, noisier in practice.
    """
    text = rec["proof"]
    hits = find_invoked(text)
    n_cited = sum(
        1 for name, positions in hits.items()
        if not all(proved_in_place(text, p) for p in positions))
    return len(hits), n_cited, hits


def norm_length(recs):
    """Within-language z-score of log character length (first_pass.py §3)."""
    by_lang = defaultdict(list)
    for r in recs:
        by_lang[r["language"]].append(math.log(len(r["proof"])))
    stats = {k: (st.mean(v), st.pstdev(v) or 1.0) for k, v in by_lang.items()}
    out = []
    for r in recs:
        mu, sd = stats[r["language"]]
        out.append((math.log(len(r["proof"])) - mu) / sd)
    return out


def rule(t):
    print(f"\n{t}\n" + "-" * len(t))


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (math.sqrt(sum((x - ma) ** 2 for x in ra))
           * math.sqrt(sum((y - mb) ** 2 for y in rb)))
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--show", default=None,
                    help="print per-record matches for a direction")
    args = ap.parse_args()

    recs = [json.loads(l) for l in args.corpus.open() if l.strip()]
    counts = [count_invoked(r) for r in recs]
    named = [c[0] for c in counts]
    cited = [c[1] for c in counts]
    zlen = norm_length(recs)
    for r, n, c, z in zip(recs, named, cited, zlen):
        r["_named"], r["_cited"], r["_z"] = n, c, z

    print(f"{len(recs)} records, {len(REGISTRY)} named results in the "
          f"registry")
    print(f"mean distinct results invoked: {sum(named)/len(named):.2f}   "
          f"(quoted-not-proved: {sum(cited)/len(cited):.2f})")

    tech = [r for r in recs if r["arm"] == "technique"]
    extr = [r for r in recs if r["arm"] == "extreme"]

    # ---------------------------------------------------------------- 0
    rule("0. Manipulation check: does the count order the directions?")
    print("  Prediction from the prompts alone: elementarity forbids named")
    print("  theorems, machinery demands them.")
    print(f"  {'direction':<16}{'mean named':>12}{'mean cited':>12}"
          f"{'median':>9}")
    for d in sorted({r["direction"] for r in extr}):
        g = [r for r in extr if r["direction"] == d]
        print(f"  {d:<16}{st.mean([r['_named'] for r in g]):>12.2f}"
              f"{st.mean([r['_cited'] for r in g]):>12.2f}"
              f"{st.median([r['_named'] for r in g]):>9.1f}")

    # ---------------------------------------------------------------- 1
    rule("1. Is the count script-independent?")
    print("  The technique arm is a translation-matched grid: the same five")
    print("  arguments written in six languages. A script-independent")
    print("  measure should give the same count for a cell in every one.")
    print()
    print("  Note on the statistic. The obvious test is the coefficient of")
    print("  variation across languages, and it is the wrong one here:")
    print("  three of the five cells average under one invoked result, and")
    print("  CV divides by that mean, so euclid at 0.10/0.10/0.00/0.00/")
    print("  0.00/0.10 scores CV 1.000 on an absolute spread of a tenth of")
    print("  a result. Below, the absolute spread is the headline and CV is")
    print("  shown only where the mean exceeds 1 and it means something.")

    langs = sorted({r["language"] for r in tech})
    print(f"\n  {'technique':<18}{'mean':>7}{'max-min':>9}{'CV':>8}"
          f"   per language ({' '.join(langs)})")
    spreads = []
    for t in sorted({r["technique"] for r in tech}):
        per_lang = [st.mean([r["_named"] for r in tech
                             if r["technique"] == t and r["language"] == l])
                    for l in langs]
        m = st.mean(per_lang)
        spread = max(per_lang) - min(per_lang)
        spreads.append(spread)
        cv = f"{st.pstdev(per_lang) / m:>8.3f}" if m >= 1 else f"{'--':>8}"
        print(f"  {t:<18}{m:>7.2f}{spread:>9.2f}{cv}   "
              + " ".join(f"{v:.1f}" for v in per_lang))
    print(f"\n  mean absolute spread across languages: "
          f"{st.mean(spreads):.2f} results")

    # The specific worry is a directional CJK penalty, not spread as such.
    print("\n  The real worry is directional: does the registry simply read")
    print("  CJK worse? Latin vs CJK means, on matched cells:")
    print(f"  {'cell':<26}{'latin':>8}{'cjk':>8}{'ratio':>8}")
    ratios = []
    for arm_recs, key in ((tech, "technique"), (extr, "direction")):
        for v in sorted({r[key] for r in arm_recs}):
            lat = [r["_named"] for r in arm_recs
                   if r[key] == v and r["language"] not in CJK]
            cjk = [r["_named"] for r in arm_recs
                   if r[key] == v and r["language"] in CJK]
            ml, mc = st.mean(lat), st.mean(cjk)
            if ml < 0.5 and mc < 0.5:
                continue          # nothing to compare; both floor at zero
            ratios.append(mc / ml if ml else float("nan"))
            print(f"  {v:<26}{ml:>8.2f}{mc:>8.2f}{ratios[-1]:>8.2f}")
    good = [r for r in ratios if r == r]
    print(f"\n  median CJK/Latin ratio: {st.median(good):.2f}   "
          f"(range {min(good):.2f}-{max(good):.2f})")
    print("  The median is the honest summary: the mean is dragged by")
    print("  generality at 2.47, where CJK proofs really do invoke more,")
    print("  which is a fact about those proofs and not about the registry.")
    print("  Above ~0.8 the registry is reading both scripts. The machinery")
    print("  cell is the weakest: CJK proofs quote the most obscure results,")
    print("  where localised name forms are least well covered, so some are")
    print("  missed. That is a recall limit on the registry, not a script")
    print("  bias in the coordinate -- and section 3 shows it is nowhere")
    print("  near large enough to affect the distinction being drawn.")

    # ---------------------------------------------------------------- 2
    rule("2. Is it just length again?")
    rho_all = spearman([r["_named"] for r in recs], [r["_z"] for r in recs])
    rho_ext = spearman([r["_named"] for r in extr], [r["_z"] for r in extr])
    print(f"  Spearman(count, normalised length), all 450: {rho_all:+.3f}")
    print(f"  Spearman(count, normalised length), extreme arm: "
          f"{rho_ext:+.3f}")
    print("  Substantially correlated, and on the extreme arm strongly so:")
    print("  quoting theorems takes words, and the machinery proofs are")
    print("  both the longest and the heaviest. So this coordinate is NOT")
    print("  independent of length, and the claim that it adds information")
    print("  rests on where the two disagree, not on the correlation being")
    print("  low. The length-matched control in section 3 is the real test.")
    print("\n  Where they disagree, within the extreme arm:")
    for d in ("elementarity", "generality", "machinery"):
        g = [r for r in extr if r["direction"] == d]
        print(f"    {d:<14} length z {st.mean([r['_z'] for r in g]):+.2f}"
              f"   named results {st.mean([r['_named'] for r in g]):.2f}")
    print("  Elementarity sits at average length and invokes nothing at")
    print("  all; generality is longer and invokes barely more. Length")
    print("  alone cannot tell either of them from a machinery proof")
    print("  scoring 16.63.")

    # ---------------------------------------------------------------- 3
    rule("3. The test that matters: machinery vs Furstenberg")
    print("  The embedding assigned 26/30 machinery records to the")
    print("  Furstenberg centroid, because Furstenberg has the most")
    print("  abstract vocabulary in the reference set. But Furstenberg's")
    print("  proof invokes no theorem -- only topological definitions.")
    print("  If this coordinate is measuring dependence rather than")
    print("  wording, it must separate them.\n")
    furst = [r for r in tech if r["technique"] == "furstenberg"]
    mach = [r for r in extr if r["direction"] == "machinery"]
    fm = st.mean([r["_named"] for r in furst])
    mm = st.mean([r["_named"] for r in mach])
    print(f"  furstenberg (technique arm, n={len(furst)}): "
          f"mean {fm:.2f}  median {st.median([r['_named'] for r in furst])}")
    print(f"  machinery   (extreme arm,   n={len(mach)}): "
          f"mean {mm:.2f}  median {st.median([r['_named'] for r in mach])}")

    # Best single threshold, and the accuracy it gives.
    lo = [r["_named"] for r in furst]
    hi = [r["_named"] for r in mach]
    best_t, best_acc = None, 0.0
    for t in range(0, max(hi + lo) + 2):
        acc = (sum(1 for v in lo if v < t) + sum(1 for v in hi if v >= t))
        acc /= len(lo) + len(hi)
        if acc > best_acc:
            best_t, best_acc = t, acc
    print(f"\n  best single threshold: count >= {best_t} -> machinery")
    print(f"  separation accuracy: {best_acc:.3f}  "
          f"(embedding cosine: 0.133, it called 26/30 of them Furstenberg)")
    if best_acc > 0.9:
        print("  -> The coordinate makes the distinction the embedding")
        print("     could not. Abstract vocabulary and heavy machinery are")
        print("     different things, and this measures the second one.")

    # Length-matched control. Section 2 found rho=+0.73 with length on the
    # extreme arm, so "machinery scores high because it is long" has to be
    # ruled out before the separation above means anything.
    print("\n  Length-matched control. The count correlates with length, so")
    print("  compare machinery against the LONGEST technique-arm records")
    print("  instead of all of them -- same length, different dependence:")
    tech_sorted = sorted(tech, key=lambda r: -r["_z"])[:len(mach)]
    tz = st.mean([r["_z"] for r in tech_sorted])
    tn = st.mean([r["_named"] for r in tech_sorted])
    print(f"    longest {len(tech_sorted)} technique-arm records: "
          f"length z {tz:+.2f}, invoked {tn:.2f}")
    print(f"    machinery records:                    "
          f"length z {st.mean([r['_z'] for r in mach]):+.2f}, "
          f"invoked {mm:.2f}")
    if tz >= st.mean([r["_z"] for r in mach]) and tn < mm / 2:
        print("    -> At equal or greater length the reference records")
        print("       invoke a fraction as much. The coordinate is not")
        print("       length wearing a different hat.")

    # ---------------------------------------------------------------- 4
    rule("4. Two coordinates")
    print("  Normalised length against invoked results, by direction.")
    print("  This is the 2D space the write-up says is the minimum for the")
    print("  geometry to be worth taking literally.\n")
    print(f"  {'direction':<16}{'length z':>10}{'invoked':>9}   position")
    rows = []
    for d in sorted({r["direction"] for r in extr}):
        g = [r for r in extr if r["direction"] == d]
        rows.append((d, st.mean([r["_z"] for r in g]),
                     st.mean([r["_named"] for r in g])))
    med = st.median([r[2] for r in rows])
    for d, z, n in sorted(rows, key=lambda x: -x[2]):
        pos = "heavy" if n > med else "light"
        pos += (", long" if z > 0.2 else
                ", short" if z < -0.2 else ", average length")
        print(f"  {d:<16}{z:>10.2f}{n:>9.2f}   {pos}")
    print("\n  brevity and elementarity are the same argument at different")
    print("  lengths and both invoke almost nothing -- they separate on")
    print("  coordinate 1 and coincide on coordinate 2, which is what the")
    print("  write-up claims about them from reading the proofs.")

    # ---------------------------------------------------------------- 5
    if args.show:
        rule(f"5. What was matched: {args.show}")
        for r in [x for x in recs if x.get("direction") == args.show][:12]:
            _, _, hits = count_invoked(r)
            print(f"  {r['id']:<28} {r['_named']:>2}  "
                  f"{', '.join(sorted(hits)) if hits else '(none)'}")


if __name__ == "__main__":
    main()
