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

One registry per theorem, chosen from the corpus's own `theorem` field. The
coordinate transfers between theorems; the registry cannot, because what
counts as an invoked result depends on what the proofs cite. Running the
wrong one is the failure mode this guards: the sqrt2 registry against the
primes corpus reports 0.85 invoked results where the right one reports
16.63, which reads exactly like a null result and is not one.

Usage:
    python coordinates.py
    python coordinates.py --show machinery   # print what was matched
    python coordinates.py --corpus proofs_sqrt2.jsonl
"""

import argparse
import json
import math
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

CORPUS = Path("proofs_primes.jsonl")
CJK = {"ja", "zh"}

# ----------------------------------------------------------------------
# The registries of named external results, one per theorem.
#
# The coordinate is theorem-invariant; the registry is not, and cannot be.
# What counts as an invoked external result depends on what the proofs
# actually cite, and a primes registry run against sqrt-2 proofs would
# report near-zero dependence everywhere -- a recall failure that reads
# exactly like a null result. Each theorem therefore gets its own list,
# selected from the corpus's `theorem` field, and a registry that has not
# been checked against a real corpus says so at the top of the output.
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

PRIMES_REGISTRY = [
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

# ----------------------------------------------------------------------
# Irrationality of sqrt2. PROVISIONAL: built from the proofs Conway and
# Shipman describe and their standard prerequisites, not yet from a corpus,
# because the corpus does not exist yet. Re-derive it from token
# frequencies the way the primes registry was derived before trusting a
# number that comes out of it.
#
# Excluded on the same principle as bare "Euclid" above: Tennenbaum,
# Theodorus, Theaetetus, Pythagoras-as-a-person, Conway, Guy, Laczkovich,
# Apostol. Those name the proof being given, not a result it leans on. The
# Pythagorean THEOREM is a genuine dependency of the folding proof and is
# listed; the school and the man are not.
# ----------------------------------------------------------------------

SQRT2_REGISTRY = [
    ("fundamental_thm_arithmetic", [
        r"re:fundamental theorem of arithmetic", r"re:Fundamentalsatz",
        r"re:th[ée]or[èe]me fondamental de l'arithm[ée]tique",
        r"re:teorema fundamental de la aritm[ée]tica",
        r"re:unique factou?ri[sz]ation",
        "算术基本定理", "算術の基本定理", "算術基本定理", "素因数分解の一意性"]),
    ("euclid_lemma", [
        r"re:Euclid'?s lemma", r"re:Lemme d'Euclide", r"re:Lema de Euclides",
        r"re:Lemma von Euklid", r"re:Euklids Lemma",
        "欧几里得引理", "ユークリッドの補題"]),
    ("euclidean_algorithm", [
        r"re:Euclidean (algorithm|domain|division)",
        r"re:euklidisch\w*", r"re:euclidien\w*", r"re:eucl[íi]deo\w*",
        "欧几里得整环", "欧几里得算法", "ユークリッド整域", "ユークリッドの互除法"]),
    ("bezout", [r"re:B[ée]zout", "裴蜀", "ベズー"]),
    ("pythagorean_theorem", [
        r"re:Pythagorean theorem", r"re:theorem of Pythagoras",
        r"re:Satz des Pythagoras", r"re:th[ée]or[èe]me de Pythagore",
        r"re:teorema de Pit[áa]goras",
        "勾股定理", "毕达哥拉斯定理", "ピタゴラスの定理", "三平方の定理"]),
    ("well_ordering", [
        r"re:well[- ]orderi?ng", r"re:least element", r"re:Wohlordnung\w*",
        r"re:bon ordre", r"re:buen orden", r"re:principio del buen orden",
        "良序", "最小数原理", "整列"]),
    ("division_algorithm", [
        r"re:division algorithm", r"re:division with remainder",
        r"re:Division mit Rest", r"re:division euclidienne",
        r"re:divisi[óo]n con resto", "带余除法", "除法定理", "除法の原理"]),
    ("infinite_descent", [
        r"re:infinite descent", r"re:unendliche\w* Abstieg\w*",
        r"re:descente infinie", r"re:descenso infinito",
        "无穷递降", "無限降下"]),
    ("archimedean_axiom", [
        r"re:Archimedean", r"re:archimedisch\w*", r"re:archim[ée]dien\w*",
        r"re:arquimediano", "阿基米德", "アルキメデス"]),
    ("rational_root_theorem", [
        r"re:rational root (theorem|test)",
        r"re:Satz [üu]ber rationale Nullstellen",
        r"re:th[ée]or[èe]me des racines rationnelles",
        r"re:teorema de la ra[íi]z racional",
        "有理根定理", "有理根定理"]),
    ("gauss_lemma", [
        r"re:Gauss'?s? lemma", r"re:Gau[ßs]'?sches Lemma",
        r"re:lemme de Gauss", "高斯引理", "ガウスの補題"]),
    ("eisenstein", [r"re:Eisenstein", "艾森斯坦", "アイゼンシュタイン"]),
    ("minimal_polynomial", [
        r"re:minimal polynomial", r"re:Minimalpolynom",
        r"re:polyn[ôo]me minimal", r"re:polinomio m[íi]nimo",
        "极小多项式", "最小多項式"]),
    ("field_extension", [
        r"re:field extension", r"re:degree of the extension",
        r"re:K[öo]rpererweiterung", r"re:extension de corps",
        r"re:extensi[óo]n de cuerpos", "域扩张", "体の拡大", "拡大次数"]),
    ("galois", [r"re:Galois", r"re:Galoiserweiterung", "伽罗瓦", "ガロア"]),
    ("integrally_closed", [
        r"re:integrally closed", r"re:ganz[- ]?abgeschlossen",
        r"re:int[ée]gralement clos", r"re:algebraic integer",
        r"re:ganze algebraische Zahl", r"re:entier alg[ée]brique",
        "整闭", "整閉", "代数的整数", "代数整数"]),
    ("dedekind", [r"re:Dedekind", "戴德金", "デデキント"]),
    ("noetherian_ufd", [
        r"re:Noetherian", r"re:noethersch\w*",
        r"re:unique factou?ri[sz]ation domain", r"re:\bUFD\b", r"re:\bPID\b",
        r"re:principal ideal domain", r"re:Hauptidealring",
        "主理想整环", "単項イデアル整域"]),
    ("p_adic", [
        r"re:p-?adic", r"re:p-?adisch\w*", r"re:p-?adique",
        r"re:p-?[áa]dico", r"re:valuation", r"re:Bewertung",
        "p进", "p進", "赋值", "付値"]),
    ("hensel", [r"re:Hensel", "亨泽尔", "ヘンゼル"]),
    ("ostrowski", [r"re:Ostrowski", "オストロフスキー"]),
    ("pell_equation", [
        r"re:Pell'?s equation", r"re:Pellsche Gleichung",
        r"re:[ée]quation de Pell", r"re:ecuaci[óo]n de Pell",
        "佩尔方程", "ペル方程式"]),
    ("continued_fraction", [
        r"re:continued fraction", r"re:Kettenbruch\w*",
        r"re:fraction continue", r"re:fracci[óo]n continua",
        "连分数", "連分数"]),
    ("pigeonhole", [
        r"re:pigeonhole", r"re:Schubfachprinzip", r"re:principe des tiroirs",
        r"re:principio del palomar", "鸽巢原理", "抽屉原理", "鳩の巣原理"]),
    ("dirichlet_approximation", [
        r"re:Dirichlet'?s approximation", r"re:Dirichletscher",
        r"re:approximation de Dirichlet", "狄利克雷", "ディリクレ"]),
    ("liouville", [
        r"re:Liouville", "刘维尔", "リウヴィル"]),
    ("thue_siegel_roth", [
        r"re:Thue[-–—]Siegel", r"re:Roth'?s theorem", r"re:Satz von Roth"]),
    ("intermediate_value", [
        r"re:intermediate value theorem", r"re:Zwischenwertsatz",
        r"re:th[ée]or[èe]me des valeurs interm[ée]diaires",
        r"re:teorema del valor intermedio", "介值定理", "中間値の定理"]),
    ("binomial_theorem", [
        r"re:binomial theorem", r"re:binomische\w* (Lehrsatz|Formel)",
        r"re:formule du bin[ôo]me", r"re:teorema del binomio",
        "二项式定理", "二項定理"]),
    ("completeness_reals", [
        r"re:completeness of (the )?(the reals|R)", r"re:Vollst[äa]ndigkeit",
        r"re:compl[ée]tude", r"re:Cauchy sequence", r"re:Cauchyfolge",
        r"re:suite de Cauchy", "完备性", "完備性", "コーシー列"]),
    ("fermat_little", [
        r"re:Fermat'?s little theorem", r"re:kleine\w* Satz von Fermat",
        r"re:petit th[ée]or[èe]me de Fermat",
        r"re:peque[ñn]o teorema de Fermat",
        "费马小定理", "フェルマーの小定理"]),
    ("quadratic_reciprocity", [
        r"re:quadratic (reciprocity|residue)",
        r"re:quadratische\w* (Reziprozit[äa]t|Rest)",
        r"re:r[ée]ciprocit[ée] quadratique",
        r"re:reciprocidad cuadr[áa]tica",
        "二次互反律", "二次剩余", "平方剰余"]),
    ("legendre", [r"re:Legendre", "勒让德", "ルジャンドル"]),
    ("lagrange", [r"re:Lagrange", "拉格朗日", "ラグランジュ"]),
    ("zorn_choice", [
        r"re:Zorn", r"re:axiom of choice", r"re:Auswahlaxiom",
        r"re:axiome du choix", r"re:axioma de elecci[óo]n",
        "选择公理", "選択公理", "ツォルン"]),
    ("transcendence", [
        r"re:Lindemann", r"re:transcendence of", r"re:Transzendenz von",
        r"re:transcendance de", "超越性", "リンデマン"]),
]

# ----------------------------------------------------------------------
# Pythagorean theorem. PROVISIONAL, same caveat as the sqrt2 list.
#
# Excluded on the bare-Euclid principle: Pythagoras and Bhaskara as people,
# Garfield, Loomis, and "Euclid" alone -- they name the proof being given.
# "Euclid I.47" and the congruence criteria it leans on are listed, since
# those are invoked results. Also excluded: the word "triangle".
# ----------------------------------------------------------------------

PYTHAGORAS_REGISTRY = [
    ("euclid_elements_prop", [
        r"re:Elements,? (Book )?I", r"re:I\.4[7-8]\b", r"re:Euclid I\.",
        r"re:Elemente", r"re:[ÉE]l[ée]ments", "原论", "原本"]),
    ("congruence_criteria", [
        r"re:\bSAS\b", r"re:\bSSS\b", r"re:\bASA\b",
        r"re:side[- ]angle[- ]side", r"re:Kongruenzsatz",
        r"re:cas d'[ée]galit[ée]", r"re:criterio de congruencia",
        "全等", "合同条件"]),
    ("similar_triangles", [
        r"re:similar triangles", r"re:\bAA\b similarity",
        r"re:[ÄA]hnlichkeitssatz", r"re:triangles semblables",
        r"re:tri[áa]ngulos semejantes", "相似三角形", "相似形"]),
    ("thales_intercept", [
        r"re:Thales", r"re:intercept theorem", r"re:Strahlensatz",
        r"re:th[ée]or[èe]me de Thal[èe]s", "泰勒斯", "タレス"]),
    ("inscribed_angle", [
        r"re:inscribed angle", r"re:Thales'? circle",
        r"re:Umfangswinkelsatz", r"re:angle inscrit",
        r"re:[áa]ngulo inscrito", "圆周角", "円周角"]),
    ("ptolemy", [
        r"re:Ptolemy", r"re:Ptolem[äa]us", r"re:Ptol[ée]m[ée]e",
        r"re:Ptolomeo", "托勒密", "トレミー", "プトレマイオス"]),
    ("law_of_cosines", [
        r"re:law of cosines", r"re:cosine rule", r"re:Kosinussatz",
        r"re:loi des cosinus", r"re:teorema del coseno",
        "余弦定理", "餘弦定理"]),
    ("law_of_sines", [
        r"re:law of sines", r"re:Sinussatz", r"re:loi des sinus",
        r"re:teorema del seno", "正弦定理"]),
    ("heron", [r"re:Heron'?s formula", r"re:Satz des Heron",
               r"re:formule de H[ée]ron", "海伦公式", "ヘロンの公式"]),
    ("stewart_apollonius", [
        r"re:Stewart'?s theorem", r"re:Apollonius", "アポロニウス"]),
    ("ceva_menelaus", [r"re:Ceva", r"re:Menelaus", "梅涅劳斯", "チェバ"]),
    ("british_flag", [r"re:British flag", "英国国旗定理"]),
    ("parallel_postulate", [
        r"re:parallel postulate", r"re:Parallelenaxiom",
        r"re:postulat des parall[èe]les", r"re:postulado de las paralelas",
        "平行公理", "平行線公準"]),
    ("hilbert_axioms", [
        r"re:Hilbert'?s axioms", r"re:Grundlagen der Geometrie",
        r"re:axiomes de Hilbert", "希尔伯特公理", "ヒルベルトの公理"]),
    ("area_axioms", [
        r"re:equidecomposab\w*", r"re:scissors congruen\w*",
        r"re:Zerlegungsgleichheit", r"re:[ée]quicompl[ée]mentaire",
        r"re:Wallace[-–—]Bolyai", r"re:Bolyai[-–—]Gerwien",
        "剪拼", "分割合同"]),
    ("cavalieri", [r"re:Cavalieri", "卡瓦列里", "カヴァリエリ"]),
    ("inner_product", [
        r"re:inner product", r"re:Skalarprodukt", r"re:produit scalaire",
        r"re:producto interno", r"re:dot product",
        "内积", "内積", "スカラー積"]),
    ("cauchy_schwarz", [
        r"re:Cauchy[-–—]Schwarz", r"re:Cauchy[-–—]Bunyakovsky",
        "柯西-施瓦茨", "コーシー・シュワルツ"]),
    ("parallelogram_law", [
        r"re:parallelogram law", r"re:Parallelogrammgleichung",
        r"re:r[èe]gle du parall[ée]logramme",
        r"re:ley del paralelogramo", "平行四边形法则", "中線定理"]),
    ("triangle_inequality", [
        r"re:triangle inequality", r"re:Dreiecksungleichung",
        r"re:in[ée]galit[ée] triangulaire",
        r"re:desigualdad triangular", "三角不等式"]),
    ("hilbert_space", [
        r"re:Hilbert space", r"re:Hilbertraum", r"re:espace de Hilbert",
        r"re:espacio de Hilbert", "希尔伯特空间", "ヒルベルト空間"]),
    ("gram_determinant", [
        r"re:Gram (matrix|determinant)", r"re:Gramsche",
        r"re:matrice de Gram", "格拉姆", "グラム行列"]),
    ("linear_algebra", [
        r"re:orthogonal (projection|complement)",
        r"re:Orthogonalprojektion", r"re:projection orthogonale",
        r"re:proyecci[óo]n ortogonal", "正交投影", "直交射影"]),
    ("trig_identity", [
        r"re:Pythagorean identity", r"re:trigonometrische\w* Identit[äa]t",
        r"re:identit[ée] trigonom[ée]trique",
        r"re:identidad trigonom[ée]trica",
        r"re:addition formula", r"re:Additionstheorem",
        "三角恒等式", "加法定理"]),
    ("taylor_series", [
        r"re:Taylor (series|expansion)", r"re:Taylorreihe",
        r"re:s[ée]rie de Taylor", r"re:serie de Taylor",
        "泰勒级数", "テイラー展開"]),
    ("greens_theorem", [
        r"re:Green'?s theorem", r"re:Satz von Green", r"re:Stokes",
        r"re:th[ée]or[èe]me de Green", "格林公式", "グリーンの定理"]),
    ("differential_equation", [
        r"re:differential equation", r"re:Differentialgleichung",
        r"re:[ée]quation diff[ée]rentielle",
        r"re:ecuaci[óo]n diferencial", "微分方程", "微分方程式"]),
    ("de_gua", [r"re:de Gua", "德古阿"]),
    ("minkowski", [r"re:Minkowski", "闵可夫斯基", "ミンコフスキー"]),
    ("euler_formula_complex", [
        r"re:Euler'?s formula", r"re:Eulersche Formel",
        r"re:formule d'Euler", "欧拉公式", "オイラーの公式"]),
    ("dimensional_analysis", [
        r"re:dimensional analysis", r"re:Dimensionsanalyse",
        r"re:analyse dimensionnelle", "量纲分析", "次元解析"]),
]

REGISTRIES = {
    "infinitude_of_primes": PRIMES_REGISTRY,
    "irrationality_of_sqrt2": SQRT2_REGISTRY,
    "pythagorean_theorem": PYTHAGORAS_REGISTRY,
}

# Registries checked against a real corpus. Anything else is a draft, and
# the output says so.
VALIDATED = {"infinitude_of_primes"}

# The section-3 contrast, per theorem: the technique whose proof invokes
# nothing but which the EMBEDDING confused with the machinery direction.
# It cannot be picked as the minimum-dependence technique automatically --
# on the primes corpus that would select euclid (0.05), and euclid is not
# the record the embedding got wrong. It is corpus knowledge, so it is
# stated, with the note that goes with it.
CONTRAST = {
    "infinitude_of_primes": (
        "furstenberg",
        ["The embedding assigned 26/30 machinery records to the",
         "Furstenberg centroid, because Furstenberg has the most",
         "abstract vocabulary in the reference set. But Furstenberg's",
         "proof invokes no theorem -- only topological definitions.",
         "If this coordinate is measuring dependence rather than",
         "wording, it must separate them."],
        0.133),
}

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


def compile_registry(registry):
    return [(name, *compile_forms(forms)) for name, forms in registry]


PROOF_RX, PROOF_SUBS = compile_forms(PROOF_MARKERS)


def select_registry(recs, override=None):
    """
    Pick the registry from the corpus's own `theorem` field.

    The corpus states which theorem it proves, so the caller does not have
    to remember. An unknown theorem is a hard stop rather than a fallback:
    silently matching a primes registry against other proofs is precisely
    the failure this selection exists to prevent.
    """
    name = override or Counter(
        r.get("theorem") for r in recs).most_common(1)[0][0]
    if name not in REGISTRIES:
        raise SystemExit(
            f"No registry for theorem {name!r}. Known: "
            f"{', '.join(sorted(REGISTRIES))}. Add one to REGISTRIES; do "
            f"not reuse another theorem's, which would report a recall "
            f"failure as a null result.")
    return name, REGISTRIES[name]


def find_invoked(text, compiled):
    """
    Return {canonical_name: [char offsets]} for every registry entry that
    appears in the text. Offsets are kept so the refinement below can ask
    what follows a mention.
    """
    hits = {}
    for name, rx, subs in compiled:
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


def count_invoked(rec, compiled):
    """
    Two counts per record:

      n_named   distinct named external results mentioned at all. The
                primary measure: it asks what the proof leans on.
      n_cited   those with no proof environment nearby, i.e. quoted rather
                than established. Sharper in principle, noisier in practice.
    """
    text = rec["proof"]
    hits = find_invoked(text, compiled)
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
    ap.add_argument("--theorem", default=None,
                    help="override the registry choice (default: read the "
                         "corpus's own theorem field)")
    ap.add_argument("--contrast", default=None,
                    help="technique to test against the machinery direction "
                         "in section 3")
    args = ap.parse_args()

    all_recs = [json.loads(l) for l in args.corpus.open() if l.strip()]
    theorem, registry = select_registry(all_recs, args.theorem)
    compiled = compile_registry(registry)

    # The scope arm proves other theorems, so it is not comparable on either
    # coordinate: a different statement changes both the length baseline and
    # what there is to invoke. Reported on its own in section 6.
    recs = [r for r in all_recs if r.get("arm") != "scope"]
    scope = [r for r in all_recs if r.get("arm") == "scope"]

    for group in (recs, scope):
        if not group:
            continue
        counts = [count_invoked(r, compiled) for r in group]
        for r, (n, c, _), z in zip(group, counts, norm_length(group)):
            r["_named"], r["_cited"], r["_z"] = n, c, z

    named = [r["_named"] for r in recs]
    cited = [r["_cited"] for r in recs]

    print(f"{len(recs)} records, theorem {theorem}, {len(registry)} named "
          f"results in the registry")
    if theorem not in VALIDATED:
        print("  !! This registry is PROVISIONAL: it has not been checked")
        print("     against a corpus. A low count here may be the registry")
        print("     failing to recognise a name, not a proof invoking")
        print("     nothing. Re-derive it from corpus token frequencies")
        print("     before reporting any number from this run.")
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
    n_tech = len({r["technique"] for r in tech})
    langs = sorted({r["language"] for r in tech})
    print(f"  The technique arm is a translation-matched grid: the same "
          f"{n_tech}")
    print(f"  arguments written in {len(langs)} languages. A "
          f"script-independent")
    print("  measure should give the same count for a cell in every one.")
    print()
    print("  Note on the statistic. The obvious test is the coefficient of")
    print("  variation across languages, and it is the wrong one here:")
    print("  most cells average under one invoked result, and CV divides by")
    print("  that mean -- on the primes corpus euclid runs 0.10/0.10/0.00/")
    print("  0.00/0.00/0.10 and scores CV 1.000 on an absolute spread of a")
    print("  tenth of a result. Below, the absolute spread is the headline")
    print("  and CV is shown only where the mean exceeds 1 and means")
    print("  something.")
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
    print(f"  Spearman(count, normalised length), all {len(recs)}: "
          f"{rho_all:+.3f}")
    print(f"  Spearman(count, normalised length), extreme arm: "
          f"{rho_ext:+.3f}")
    print("  Substantially correlated, and on the extreme arm strongly so:")
    print("  quoting theorems takes words, and the machinery proofs are")
    print("  both the longest and the heaviest. So this coordinate is NOT")
    print("  independent of length, and the claim that it adds information")
    print("  rests on where the two disagree, not on the correlation being")
    print("  low. The length-matched control in section 3 is the real test.")
    print("\n  Where they disagree, within the extreme arm:")
    present = {r["direction"] for r in extr}
    for d in [x for x in ("elementarity", "generality", "machinery")
              if x in present]:
        g = [r for r in extr if r["direction"] == d]
        print(f"    {d:<14} length z {st.mean([r['_z'] for r in g]):+.2f}"
              f"   named results {st.mean([r['_named'] for r in g]):.2f}")
    print("  Read the two columns against each other: where directions sit")
    print("  at similar length but far apart in invoked results, length")
    print("  alone cannot tell them apart and this coordinate can.")

    # ---------------------------------------------------------------- 3
    known, note, prior = CONTRAST.get(theorem, (None, None, None))
    name = args.contrast or known
    if name is None:
        # No stated contrast for this theorem yet. Fall back to the
        # lowest-dependence technique, and say that it is a stand-in: the
        # real contrast is whichever technique the EMBEDDING confuses with
        # machinery, and that is not knowable from this file.
        name = min({r["technique"] for r in tech},
                   key=lambda t: st.mean([r["_named"] for r in tech
                                          if r["technique"] == t]))
        note = [f"No contrast recorded for {theorem}, so this uses the",
                f"lowest-dependence technique ({name}) as a stand-in. The",
                "contrast that matters is whichever technique the embedding",
                "confuses with machinery; run extreme.py first, then add it",
                "to CONTRAST."]
    rule(f"3. The test that matters: machinery vs {name}")
    for line in note:
        print(f"  {line}")
    print()
    furst = [r for r in tech if r["technique"] == name]
    mach = [r for r in extr if r["direction"] == "machinery"]
    if not furst or not mach:
        raise SystemExit(f"section 3 needs technique {name!r} and direction "
                         f"'machinery' in the corpus.")
    fm = st.mean([r["_named"] for r in furst])
    mm = st.mean([r["_named"] for r in mach])
    print(f"  {name:<11} (technique arm, n={len(furst)}): "
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
    if prior is not None:
        print(f"  separation accuracy: {best_acc:.3f}  "
              f"(embedding cosine: {prior:.3f}, it called 26/30 of them "
              f"Furstenberg)")
    else:
        print(f"  separation accuracy: {best_acc:.3f}  "
              f"(no embedding figure recorded for this theorem)")
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
    if {"brevity", "elementarity"} <= {r[0] for r in rows}:
        print("\n  brevity and elementarity are the same argument at "
              "different")
        print("  lengths and both invoke almost nothing -- they separate on")
        print("  coordinate 1 and coincide on coordinate 2, which is what")
        print("  the write-up claims about them from reading the proofs.")

    # ---------------------------------------------------------------- 5
    if args.show:
        rule(f"5. What was matched: {args.show}")
        for r in [x for x in recs if x.get("direction") == args.show][:12]:
            _, _, hits = count_invoked(r, compiled)
            print(f"  {r['id']:<28} {r['_named']:>2}  "
                  f"{', '.join(sorted(hits)) if hits else '(none)'}")

    # ---------------------------------------------------------------- 6
    if scope:
        rule("6. Scope arm: does dependence track scope?")
        print("  Each target is a statement the reference proofs do or do")
        print("  not reach. Conway and Shipman's scope test says proofs")
        print("  with different scopes are different proofs; if dependence")
        print("  is a real coordinate, it should climb as the statement")
        print("  moves out of reach of the light arguments.\n")
        print(f"  {'target':<28}{'invoked':>9}{'length z':>10}")
        for t in sorted({r["theorem"] for r in scope}):
            g = [r for r in scope if r["theorem"] == t]
            print(f"  {t:<28}{st.mean([r['_named'] for r in g]):>9.2f}"
                  f"{st.mean([r['_z'] for r in g]):>10.2f}")
        print("\n  The first target is the theorem of the other two arms,")
        print("  asked with no selection criterion: the unprompted baseline")
        print("  every extremal direction should be read against.")


if __name__ == "__main__":
    main()
