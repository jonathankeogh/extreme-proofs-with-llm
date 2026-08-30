"""
The lookup tables: data the analysis reads, and nothing else.

Five hundred lines of surface forms with no logic in them, kept out of
analyse.py so that file reads as five stages rather than as five stages
wrapped around a wall of regexes.

  PROOF_MARKERS            "Proof."/"Beweis"/"証明" in six languages. Used
                           by graphs.py for the cited-but-not-proved
                           refinement.
  *_REGISTRY, REGISTRIES   Named external results, one list per theorem,
                           keyed by the corpus's own `theorem` field. Used
                           by graphs.py fig6; the coordinate transfers between
                           theorems; the registry cannot, because what
                           counts as an invoked result depends on what the
                           proofs cite.
  NAMES, NOTATION          Stage 5's masking tables: mathematicians'
                           surnames, and technique-diagnostic symbols.

Each entry is (canonical_name, [surface forms]). A form beginning with
"re:" is a regex applied with word boundaries to Latin script; everything
else is a plain substring test, which is what CJK needs, no word
boundaries existing there. Latin matching is case-insensitive.

`compile_forms` and `compile_registry` live here rather than in analyse.py
because they are what makes a table usable, both stages call them, and
PROOF_RX below is built at import from PROOF_MARKERS -- so leaving them in
the analysis would mean this module importing it back.
"""

import re


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


# Proof-environment markers, used only by the "cited but not proved"
# refinement in count_invoked().
PROOF_MARKERS = [
    r"re:proof\b", r"re:Beweis", r"re:Preuve", r"re:D[ée]monstration",
    r"re:Prueba", r"re:Demostraci[óo]n", "证明", "証明",
]

PROOF_RX, PROOF_SUBS = compile_forms(PROOF_MARKERS)


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
# Mathematicians' names. The registries carry these only
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
