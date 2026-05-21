from sage.all import *
import os
import time
import pickle
import random
import re

DEFAULT_TRIALS = 10000

CODE_VERSION = "v21_spec_dmix_default_no_raw_spec_ridn"
DEFAULT_CHECKPOINT_DIR = "sqisign_ideal_checkpoints_reject64p2_pi4_v21_spec_dmix_default_no_raw_spec_ridn"
DEFAULT_PLOT_DIR = "sqisign_ratio_plots_reject64p2_pi4_v21_spec_dmix_default_no_raw_spec_ridn"

# ============================================================
# SQIsign ideal-basis experiment
#
# Reject rule tested:
#   reject iff max_j nrd(b_j) > nrd(I) * (64 p^2 / pi^4)
# where {b_j} are the columns of a chosen basis matrix of I.
#
# Main plotted quantity:
#   max_j nrd(b_j) / nrd(I)
#
# Basis modes:
#   - natural : a fast ZZ-module basis extracted from the generating set
#   - hnf     : canonical column-HNF basis (much slower)
#   - l2      : an L2/LLL-like reduced basis of the same lattice
#
# IMPORTANT SCOPE NOTE
# --------------------
# Self-contained exact implementations are provided for:
#   1) RandomIdealGivenNorm(N, prime=True)
#   2) RandomIdealGivenNorm(N, prime=False)
#   3) RandomEquivalentPrimeIdeal(I)
#
# KernelDecomposedToIdeal requires the precomputed torsion-action matrices
# M_i and M_theta on the chosen basis of E0[2^f], exactly as in the official
# implementation / precomputation.  A trace-driven adapter is provided below.
# ============================================================

PARAMS = {
    "NIST-I": {
        "lambda": Integer(128),
        "p": Integer(5) * (Integer(2) ** Integer(248)) - Integer(1),
        "f": Integer(248),
        "ersp": Integer(126),
        "echl": Integer(122),
        "quat_repres_bound_input": Integer(20),
        "quat_equiv_bound_coeff": Integer(64),
        "finduv_box_size": Integer(2),
        "norders": Integer(6),
        "qts": [Integer(5), Integer(17), Integer(37), Integer(41), Integer(53), Integer(97)],
        "quat_prime_cofactor": (Integer(2) ** Integer(252)) + Integer(65),
    },
    "NIST-III": {
        "lambda": Integer(192),
        "p": Integer(65) * (Integer(2) ** Integer(376)) - Integer(1),
        "f": Integer(376),
        "ersp": Integer(192),
        "echl": Integer(184),
        "quat_repres_bound_input": Integer(21),
        "quat_equiv_bound_coeff": Integer(64),
        "finduv_box_size": Integer(3),
        "norders": Integer(7),
        "qts": [Integer(5), Integer(13), Integer(17), Integer(41), Integer(73), Integer(89), Integer(97)],
        "quat_prime_cofactor": (Integer(2) ** Integer(384)) + Integer(369),
    },
    "NIST-V": {
        "lambda": Integer(256),
        "p": Integer(27) * (Integer(2) ** Integer(500)) - Integer(1),
        "f": Integer(500),
        "ersp": Integer(253),
        "echl": Integer(247),
        "quat_repres_bound_input": Integer(21),
        "quat_equiv_bound_coeff": Integer(64),
        "finduv_box_size": Integer(3),
        "norders": Integer(6),
        "qts": [Integer(5), Integer(37), Integer(61), Integer(97), Integer(113), Integer(149)],
        "quat_prime_cofactor": (Integer(2) ** Integer(506)) + Integer(51),
    },
}



OFFICIAL_REF_KERNEL_PRECOMP = {
    "NIST-I": {
        "M_i": [
            [358148485862516530295777212166085464807446066917001814010108736316374275803, 154232568911797650634232736925059697497509532296810210911386094633617161594],
            [37475475934861355003334106447043810791830859938399588542807915684018275407, 94164362720749858077546948024101675244389810683156639269022451214536386853],
        ],
        "M_theta": [
            [282349176237718206067359945038802588997504137205055463117532166877505723469, 365622797362587724498832576197043126766234736028243398748773626097238170941],
            [421730964382299709976244257876910782264477400883404706303090456049754234377, 169963672345548182305964215151384551054331740395102990161599020653404939188],
        ],
    },
    "NIST-III": {
        "M_i": [
            [93055293095500727746142441799865949768100937028869825019591032989399096866624879789495160940534906981182376548305, 61449142554984666589740398827086540092669723721283085816960137470569744740356230215491032540836372515323971508930],
            [151881986583586218268548843038305035626245309798447235888121028499827245097407954862158024084914088624290247539471, 60858793609165206676822558591320041657991794496385826027081988120935753803286099161341816617609294740718514038831],
        ],
        "M_theta": [
            [75679083973221001963827754980773367908580964086493950835920783056974901392509838194007505276176207296626804267983, 96149917319427434744810902009111599063281594815403301597480097858260731396696635096358786037069911036350240462591],
            [27711570859537858511513101629726033696302048773964261959614120156102138551901239211304886582166875394901005909137, 78235002731444932459137245410412623517511767438761700210752238053359949277401140756829472281967994425274086319154],
        ],
    },
    "NIST-V": {
        "M_i": [
            [330813439832858287942944108737783197698989506581897084021279156986435747669018854075658847200154773456214714023837883176155405441171441151144971809591, 1761855382089977027431128958383445434898469559142042483870875991719632760077865192104666958006812616713353220186378018356990266136254743912921012437646],
            [1577391983280026427800963684432069401469550703597248546720509391264950193188865838483631482448334487448714035938987024530907615513097468486358828729777, 2942577168063283582070245588089815954517652539461167705462012211109698048735655700807611245125749383694671970103722187833061851104713951902183555779785],
        ],
        "M_theta": [
            [295954231282169347155342958338837802990680799209836456831947354486697131087641499499676060189500735365821655302706146187886846486062711737661388626637, 1923023077270741107507624161551818232824716102621026603786834227269664991836688703515915010320951653190264962746137588617074591700637347267139865308551],
            [2070549466882705306840633431729736331246331467943043190516352692504177822910806822186059805092879679491381331971194818352869125397602407051762322516339, 2977436376613972522857846738488761349225961246833228332651344013609436665317033055383594032136403421785065028824853924821330410059822681315667138962740],
        ],
    },
}


def load_official_ref_kernel_precomp():
    '''
    Official reference precomputation for KernelDecomposedToIdeal, extracted from
    the first precomputed tuple CURVES_WITH_ENDOMORPHISMS[0] in the uploaded
    reference code (ref/lvl{1,3,5}/endomorphism_action.c).

    The uploaded precomp satisfies:
      - action_gen2 = action_i
      - action_gen3 = action_( (i+j)/2 )
      - action_gen4 = action_( (1+k)/2 )

    and we verified on all three parameter sets that:
      - 2*action_gen3 - action_i = action_j      mod 2^f
      - 2*action_gen4 - Id        = action_k      mod 2^f

    Therefore the exact matrix for
      theta = j + (1+k)/2
    is the embedded matrix M_theta below.
    '''
    M_i_by_level = {level: OFFICIAL_REF_KERNEL_PRECOMP[level]["M_i"] for level in OFFICIAL_REF_KERNEL_PRECOMP}
    M_theta_by_level = {level: OFFICIAL_REF_KERNEL_PRECOMP[level]["M_theta"] for level in OFFICIAL_REF_KERNEL_PRECOMP}
    return M_i_by_level, M_theta_by_level


def make_default_kernel_decomposed_c_samplers(M_theta_by_level, max_tries=4096):
    '''
    Default random coefficient samplers for KernelDecomposedToIdeal.

    For each level, sample (c1,c2) uniformly modulo 2^f until:
      (1) [c1]P0 + [c2]Q0 has exact order 2^f, i.e. not both coefficients are even;
      (2) the matrix [[c1,d1],[c2,d2]] is invertible modulo 2^f, where
          [d1,d2]^T = M_theta [c1,c2]^T.
    '''
    samplers = {}
    for level in ["NIST-I", "NIST-III", "NIST-V"]:
        f = Integer(PARAMS[level]["f"])
        mod = Integer(2) ** f
        Mtheta = matrix(Integers(mod), M_theta_by_level[level])

        def _sampler(ctx, rng, mod=mod, Mtheta=Mtheta, max_tries=max_tries):
            for _ in range(int(max_tries)):
                c1 = Integer(rng.randrange(int(mod)))
                c2 = Integer(rng.randrange(int(mod)))
                if (c1 % 2 == 0) and (c2 % 2 == 0):
                    continue
                d = Mtheta * vector(Integers(mod), [c1, c2])
                M = matrix(Integers(mod), [[c1, d[0]], [c2, d[1]]])
                if gcd(Integer(M.det().lift()), mod) == 1:
                    return (c1, c2)
            raise RuntimeError("KernelDecomposedToIdeal coefficient sampler exceeded max_tries")

        samplers[level] = _sampler
    return samplers


def _gcd_list(vals):
    g = Integer(0)
    for v in vals:
        g = gcd(g, abs(Integer(v)))
    return g


def _prime_above(n):
    return Integer(Integer(n).next_prime(proof=False))


def _prime_1mod4_above(n):
    p = _prime_above(n)
    while p % 4 != 1:
        p = Integer(p.next_prime(proof=False))
    return p


def _coerce_py_seed(seed):
    if seed is None:
        return None
    if isinstance(seed, (int, float, str, bytes, bytearray)):
        return seed
    try:
        return int(seed)
    except Exception:
        pass
    try:
        return float(seed)
    except Exception:
        pass
    try:
        return bytes(seed)
    except Exception:
        return str(seed)


def _seed_plus(seed, offset):
    if seed is None:
        return int(offset)
    try:
        return int(seed) + int(offset)
    except Exception:
        return "{}+{}".format(seed, offset)


def _coerce_py_int(n, name):
    try:
        return int(n)
    except Exception:
        raise TypeError("{} must be int-like".format(name))


def _safe_qq_ratio(num, den):
    den = Integer(den)
    if den == 0:
        return None
    return QQ(Integer(num)) / QQ(den)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        try:
            return float(N(x, 53))
        except Exception:
            return None


def _probably_prime(n):
    return bool(Integer(n).is_prime(proof=False))


def _legendre_symbol_prime(a, p):
    a = Integer(a) % Integer(p)
    p = Integer(p)
    if a == 0:
        return Integer(0)
    ls = power_mod(a, (p - 1) // 2, p)
    if ls == p - 1:
        return Integer(-1)
    return Integer(ls)


def _modular_sqrt_prime_one(a, p):
    """
    Return one square root r of a modulo the odd prime p, or raise ValueError
    if no root exists. This avoids Sage-version-specific sqrt_mod behaviour.
    """
    a = Integer(a) % Integer(p)
    p = Integer(p)
    if p == 2:
        return a
    if a == 0:
        return Integer(0)
    if _legendre_symbol_prime(a, p) != 1:
        raise ValueError("not a square modulo p")
    if p % 4 == 3:
        return Integer(power_mod(a, (p + 1) // 4, p))

    # Tonelli-Shanks
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1

    z = Integer(2)
    while _legendre_symbol_prime(z, p) != -1:
        z += 1

    m = s
    c = Integer(power_mod(z, q, p))
    t = Integer(power_mod(a, q, p))
    r = Integer(power_mod(a, (q + 1) // 2, p))

    while t != 1:
        i = 1
        t2i = Integer(power_mod(t, 2, p))
        while i < m and t2i != 1:
            t2i = Integer(power_mod(t2i, 2, p))
            i += 1
        if i == m:
            raise ValueError("Tonelli-Shanks failed to converge")
        b = Integer(power_mod(c, Integer(1) << (m - i - 1), p))
        r = Integer((r * b) % p)
        t = Integer((t * b * b) % p)
        c = Integer((b * b) % p)
        m = i
    return r


def _modular_sqrt_prime_all(a, p):
    a = Integer(a) % Integer(p)
    p = Integer(p)
    if p == 2:
        return [a]
    if a == 0:
        return [Integer(0)]
    r = _modular_sqrt_prime_one(a, p)
    s = Integer((-r) % p)
    if r == s:
        return [r]
    return [r, s]


def _record_failure_example(state, exc):
    rep = repr(exc)
    if state["first_exception"] is None:
        state["first_exception"] = rep
    state["last_exception"] = rep
    hist = state["exception_hist"]
    hist[rep] = hist.get(rep, 0) + 1


def _sorted_quantiles(vals, qs=(0.5, 0.9, 0.95, 0.99)):
    if not vals:
        return {}
    ys = sorted(float(v) for v in vals)
    n = len(ys)
    out = {}
    for q in qs:
        if n == 1:
            out[str(q)] = ys[0]
            continue
        t = (n - 1) * float(q)
        i = int(t)
        j = min(i + 1, n - 1)
        a = t - i
        out[str(q)] = (1.0 - a) * ys[i] + a * ys[j]
    return out


def _looks_like_result_dict(obj):
    return isinstance(obj, dict) and ("level" in obj) and ("name" in obj)


def _flatten_result_dicts(obj):
    out = []
    if _looks_like_result_dict(obj):
        out.append(obj)
        return out
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_flatten_result_dicts(v))
    return out


def _merge_result_maps(dst, src):
    for res in _flatten_result_dicts(src):
        level = res["level"]
        if level not in dst:
            dst[level] = {}
        dst[level][res["name"]] = res
    return dst


def _slugify(text):
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "plot"


class SQISignIdealLab:
    def __init__(self, level):
        if level not in PARAMS:
            raise ValueError("unknown level")
        P = PARAMS[level]
        self.level = level
        self.lam = P["lambda"]
        self.p = P["p"]
        self.f = P["f"]
        self.ersp = P["ersp"]
        self.echl = P["echl"]
        self.quat_repres_bound_input = P["quat_repres_bound_input"]
        self.quat_equiv_bound_coeff = P["quat_equiv_bound_coeff"]
        self.finduv_box_size = P["finduv_box_size"]
        self.norders = P["norders"]
        self.qts = P["qts"]
        self.quat_prime_cofactor = P["quat_prime_cofactor"]
        if level == "NIST-I":
            self.Dmix_spec = (Integer(2) ** (Integer(4) * self.lam)) + Integer(75)
            self.Dmix_4k1 = (Integer(2) ** (Integer(4) * self.lam)) + Integer(145)
        elif level == "NIST-III":
            self.Dmix_spec = (Integer(2) ** (Integer(4) * self.lam)) + Integer(183)
            self.Dmix_4k1 = (Integer(2) ** (Integer(4) * self.lam)) + Integer(241)
        else:
            self.Dmix_spec = (Integer(2) ** (Integer(4) * self.lam)) + Integer(643)
            self.Dmix_4k1 = (Integer(2) ** (Integer(4) * self.lam)) + Integer(1081)
        self.Dmix = self.Dmix_spec

        self.R = RealField(256)
        self.pi_R = self.R.pi()
        self.reject_ratio_bound = self.R(64) * (self.R(self.p) ** 2) / (self.pi_R ** 4)

        # Ambient basis = (1, i, j, k)
        # O0 basis = (1, i, (i+j)/2, (1+k)/2)
        self.O0 = matrix(QQ, [
            [1, 0,   0,   QQ(1)/2],
            [0, 1, QQ(1)/2, 0],
            [0, 0, QQ(1)/2, 0],
            [0, 0,   0,   QQ(1)/2],
        ])

        # Bilinear form <alpha,beta> = tr(alpha * conjugate(beta))
        self.G0 = diagonal_matrix(QQ, [2, 2, 2*self.p, 2*self.p])

    # --------------------------------------------------------
    # Quaternion arithmetic in ambient basis (1, i, j, k)
    # --------------------------------------------------------
    def quat_conj(self, v):
        v = vector(QQ, v)
        return vector(QQ, [v[0], -v[1], -v[2], -v[3]])

    def quat_nrd(self, v):
        v = vector(QQ, v)
        return v[0]**2 + v[1]**2 + self.p*(v[2]**2 + v[3]**2)

    def right_mult_matrix(self, alpha):
        a, b, c, d = map(QQ, list(alpha))
        p = self.p
        # Column j = (basis_j) * alpha in basis (1,i,j,k)
        return matrix(QQ, [
            [a,  -b, -p*c, -p*d],
            [b,   a,  p*d, -p*c],
            [c,  -d,    a,    b],
            [d,   c,   -b,    a],
        ])

    def quat_mul(self, x, y):
        return self.right_mult_matrix(y) * vector(QQ, x)

    def reduce_quat_mod_N(self, alpha, N):
        N = Integer(N)
        v = vector(QQ, alpha)
        out = []
        for j in range(4):
            q = QQ(v[j])
            if q.denominator() != 1:
                raise RuntimeError("reduce_quat_mod_N expects integral ambient coefficients")
            out.append(Integer(q.numerator()) % N)
        return vector(QQ, out)

    # --------------------------------------------------------
    # Lattice / ideal helpers
    # --------------------------------------------------------
    def common_denominator(self, M):
        dens = [QQ(x).denominator() for x in M.list()]
        return lcm(dens) if dens else Integer(1)

    def col_hnf_int(self, M):
        # Sage hermite_form is row-HNF, so transpose twice.
        H = M.transpose().hermite_form().transpose()
        nz = [j for j in range(H.ncols()) if not H.column(j).is_zero()]
        if len(nz) == 0:
            return matrix(ZZ, H.nrows(), 0, [])
        return H.matrix_from_columns(nz)

    def col_hnf_rational(self, M):
        D = self.common_denominator(M)
        H = self.col_hnf_int((D*M).change_ring(ZZ))
        return (H / D).change_ring(QQ)

    def module_basis_rational(self, M):
        D = self.common_denominator(M)
        A = (D*M).change_ring(ZZ)
        V = ZZ^4
        L = V.span([vector(ZZ, A.column(j)) for j in range(A.ncols())])
        B = matrix(ZZ, [v.list() for v in L.basis()]).transpose()
        return (B / D).change_ring(QQ)

    def normalize_lattice(self, M, method="module"):
        if method == "module":
            return self.module_basis_rational(M)
        elif method == "hnf":
            return self.col_hnf_rational(M)
        else:
            raise ValueError("unknown normalization method")

    def lattice_sum(self, L1, L2, method="module"):
        M = block_matrix(QQ, 1, 2, [L1, L2])
        return self.normalize_lattice(M, method=method)

    def right_multiply_lattice(self, L, alpha, method="module"):
        M = self.right_mult_matrix(alpha) * L
        return self.normalize_lattice(M, method=method)

    def lattice_product(self, L1, L2, method="module"):
        blocks = []
        for j in range(L2.ncols()):
            alpha_j = L2.column(j)
            blocks.append(self.right_mult_matrix(alpha_j) * L1)
        M = block_matrix(QQ, 1, len(blocks), blocks)
        return self.normalize_lattice(M, method=method)

    def lattice_intersection(self, L1, L2, method="module"):
        D1 = self.common_denominator(L1)
        D2 = self.common_denominator(L2)
        A1 = (D1*L1).change_ring(ZZ)
        A2 = (D2*L2).change_ring(ZZ)
        D = lcm(D1, D2)
        B1 = ((D // D1) * A1).change_ring(ZZ)
        B2 = ((D // D2) * A2).change_ring(ZZ)
        V = ZZ^4
        M1 = V.span([vector(ZZ, B1.column(j)) for j in range(B1.ncols())])
        M2 = V.span([vector(ZZ, B2.column(j)) for j in range(B2.ncols())])
        I = M1.intersection(M2)
        B = matrix(ZZ, [v.list() for v in I.basis()]).transpose()
        return self.normalize_lattice(B / D, method=method)

    def ideal_O0_alpha_N(self, alpha, N, method="module"):
        Oalpha = self.right_mult_matrix(alpha) * self.O0
        ON = QQ(N) * self.O0
        return self.lattice_sum(Oalpha, ON, method=method)

    def lattice_conjugate(self, L):
        cols = [self.quat_conj(L.column(j)).list() for j in range(L.ncols())]
        return matrix(QQ, cols).transpose()

    def ideal_inverse_connecting_maximal(self, I, norm_I, method="module"):
        # For integral ideals connecting maximal orders:
        #   I^{-1} = conjugate(I) / nrd(I)
        Ibar = self.lattice_conjugate(I)
        return self.normalize_lattice(Ibar / QQ(norm_I), method=method)

    def left_order_of_connecting_maximal_ideal(self, I, norm_I, method="module"):
        Iinv = self.ideal_inverse_connecting_maximal(I, norm_I, method=method)
        return self.lattice_product(I, Iinv, method=method)

    def compute_connecting_ideal(self, OL, OR, method="module"):
        # A connecting ideal can be obtained as N * OL * OR, where N is any
        # common denominator of the lattice product OL * OR.
        P = self.lattice_product(OL, OR, method=method)
        D = self.common_denominator(P)
        return self.normalize_lattice(QQ(D) * P, method=method)

    def compute_connecting_ideal_with_norm(self, OL, OR, method="module"):
        # Spec 3.1.6.1 allows taking any common denominator N of OL*OR and
        # setting I = N*OL*OR. In this representation, the ideal norm is N.
        P = self.lattice_product(OL, OR, method=method)
        D = Integer(self.common_denominator(P))
        I = self.normalize_lattice(QQ(D) * P, method=method)
        return I, D

    def right_order_of_connecting_maximal_ideal(self, I, norm_I, method="module"):
        Iinv = self.ideal_inverse_connecting_maximal(I, norm_I, method=method)
        return self.lattice_product(Iinv, I, method=method)

    def ideal_pullback(self, J, norm_J, I, norm_I, method="module"):
        # [J]^* I = JI + nrd(I) OL(J)
        OLJ = self.left_order_of_connecting_maximal_ideal(J, norm_J, method=method)
        return self.lattice_sum(
            self.lattice_product(J, I, method=method),
            QQ(norm_I) * OLJ,
            method=method,
        )

    def ideal_pushforward(self, J, norm_J, I, method="module"):
        # [J]_* I = J^{-1}(J ∩ I)
        Jinv = self.ideal_inverse_connecting_maximal(J, norm_J, method=method)
        return self.lattice_product(
            Jinv,
            self.lattice_intersection(J, I, method=method),
            method=method,
        )

    def response_ideal_from_alpha(self, alpha_rsp, q_rsp, Dmix=None, method="module"):
        if Dmix is None:
            Dmix = self.Dmix_spec
        return self.ideal_O0_alpha_N(alpha_rsp, Integer(q_rsp) * Integer(Dmix), method=method)

    def ideal_norm_from_left_order(self, I, OL=None):
        '''
        Recover nrd(I) from a full-rank integral left ideal I of the maximal order OL
        via the identity [OL : I] = nrd(I)^2.
        '''
        if OL is None:
            OL = self.O0
        if I.nrows() != 4 or I.ncols() != 4:
            raise ValueError("ideal lattice must be rank-4")
        idx = QQ(abs(I.det() / OL.det()))
        if idx.denominator() != 1:
            raise RuntimeError("ideal index is not integral")
        idx = Integer(idx)
        if idx < 0 or (not idx.is_square()):
            raise RuntimeError("ideal index is not a perfect square")
        return Integer(idx.sqrt())

    def sample_sign_like_composite_norm(self, rng=None, e_min=2, e_max=None, require_composite=True, max_tries=4096):
        # Default synthetic sampler for the signing-side composite shape
        #   N = 2^e - q_rsp,
        # where q_rsp is odd and 0 < q_rsp < 2^e.
        if rng is None:
            rng = random.Random()
        if e_max is None:
            e_max = Integer(self.ersp)
        e_min = Integer(max(2, int(e_min)))
        e_max = Integer(max(int(e_min), int(e_max)))

        for _ in range(int(max_tries)):
            e = Integer(rng.randint(int(e_min), int(e_max)))
            q = Integer(rng.randrange(1, int(Integer(2) ** e), 2))
            N = Integer(2) ** e - q
            if N <= 1:
                continue
            if N % 2 == 0:
                continue
            if N % self.p == 0:
                continue
            if require_composite and _probably_prime(N):
                continue
            return Integer(N)
        raise RuntimeError("sample_sign_like_composite_norm exceeded max_tries")

    def sample_random_ideal_given_norm(self, N, prime, rng=None, lattice_norm_method="module"):
        if prime:
            return self.sample_random_ideal_given_norm_prime(N, rng=rng, lattice_norm_method=lattice_norm_method)
        return self.sample_random_ideal_given_norm_composite(N, rng=rng, lattice_norm_method=lattice_norm_method)

    # --------------------------------------------------------
    # Basis-extraction modes for the event test
    # --------------------------------------------------------
    def l2_reduced_basis(self, L):
        D = self.common_denominator(L)
        Bint = (D*L).change_ring(ZZ)
        G = (Bint.transpose() * self.G0 * Bint).change_ring(ZZ)
        try:
            U = matrix(ZZ, pari(G).qflllgram())
        except Exception:
            # Fallback: return the input basis if qflllgram is unavailable.
            return L
        return (Bint * U / D).change_ring(QQ)

    def basis_for_test(self, L, basis_mode="natural"):
        if basis_mode == "natural":
            return L
        elif basis_mode == "hnf":
            return self.col_hnf_rational(L)
        elif basis_mode == "l2":
            return self.l2_reduced_basis(L)
        else:
            raise ValueError("basis_mode must be one of: natural, hnf, l2")

    def basis_nrd_list(self, B):
        return [self.quat_nrd(B.column(j)) for j in range(B.ncols())]

    def reject_ratio_threshold(self):
        return self.reject_ratio_bound

    def event_data(self, L, ideal_norm, basis_mode="natural"):
        B = self.basis_for_test(L, basis_mode=basis_mode)
        nrds = self.basis_nrd_list(B)
        mx = max(nrds) if nrds else QQ(0)
        if Integer(ideal_norm) <= 0:
            raise ValueError("ideal_norm must be positive")
        ratio_over_norm = QQ(mx) / QQ(ideal_norm)
        legacy_ratio_over_norm_square = QQ(mx) / (QQ(ideal_norm) ** 2)
        threshold = self.reject_ratio_threshold()
        ratio_R = self.R(ratio_over_norm)
        log10_ratio = float(ratio_R.log10()) if ratio_R > 0 else float("-inf")
        return {
            "basis": B,
            "basis_nrds": nrds,
            "max_basis_nrd": mx,
            "ratio_over_norm": ratio_over_norm,
            "legacy_ratio_over_norm_square": legacy_ratio_over_norm_square,
            "reject_threshold_ratio": threshold,
            "reject_threshold_log10": float(threshold.log10()),
            "ratio_over_norm_log10": log10_ratio,
            "reject": bool(ratio_R > threshold),
        }

    # --------------------------------------------------------
    # Cornacchia / norm-representation helpers
    # --------------------------------------------------------
    def _cornacchia_from_root(self, q, m, r0):
        """
        Solve x^2 + q y^2 = m from a chosen square root r0^2 ≡ -q (mod m)
        using the standard Cornacchia Euclidean descent.
        """
        q = Integer(q)
        m = Integer(m)
        r0 = Integer(r0) % m
        if q <= 0 or m <= 0:
            return None
        if r0 == 0:
            return None

        a = Integer(m)
        b = Integer(min(r0, m - r0))
        if b == 0:
            return None

        while b*b > m:
            a, b = b, a % b
            if b == 0:
                return None

        x = Integer(abs(b))
        rhs = Integer(m - x*x)
        if rhs < 0 or rhs % q != 0:
            return None
        y2 = Integer(rhs // q)
        if y2 < 0 or (not y2.is_square()):
            return None
        y = Integer(y2.sqrt())
        if x*x + q*y*y != m:
            return None
        return (x, y)

    def cornacchia(self, q, m):
        q = Integer(q)
        m = Integer(m)
        if kronecker(-q, m) != 1:
            return None
        if m == 2:
            if q == 1:
                return (Integer(1), Integer(1))
            return None

        roots = _modular_sqrt_prime_all((-q) % m, m)
        for r0 in roots:
            sol = self._cornacchia_from_root(q, m, r0)
            if sol is not None:
                return sol
        return None

    def cornacchia_random(self, q, m, rng=None):
        """
        A lightly randomized wrapper around Cornacchia(q, m).

        For prime m this first randomizes the modular-square-root branch and then
        randomizes signs on the returned solution. A coordinate swap is applied
        only in the symmetric case q = 1.
        """
        q = Integer(q)
        m = Integer(m)
        if rng is None:
            rng = random.Random()
        if kronecker(-q, m) != 1:
            return None
        if m == 2:
            if q == 1:
                return (Integer(1), Integer(1))
            return None

        roots = list(_modular_sqrt_prime_all((-q) % m, m))
        rng.shuffle(roots)
        for r0 in roots:
            sol = self._cornacchia_from_root(q, m, r0)
            if sol is None:
                continue
            x, y = sol
            if rng.getrandbits(1):
                x = -x
            if rng.getrandbits(1):
                y = -y
            if q == 1 and rng.getrandbits(1):
                x, y = y, x
            return (Integer(x), Integer(y))
        return None

    def dmax_dividing_integral_gamma_in_O0(self, a, b, c, d):
        # If gamma = a + b i + c j + d k (ambient integral coefficients),
        # then in the O0-basis we have coefficients:
        #   (a-d, b-c, 2c, 2d).
        return _gcd_list([a-d, b-c, 2*c, 2*d])

    def generalized_represent_integer_O0(self, M, isogeny_cond=False, rng=None, max_tries=None):
        # This is the q=1, O=O0 specialization of GeneralizedRepresentInteger.
        M = Integer(M)
        if M % 2 == 0:
            raise ValueError("GeneralizedRepresentInteger only supports odd M")
        if M <= self.p:
            raise ValueError("need M > p")
        if rng is None:
            rng = random.Random()

        bound = Integer(ceil(QQ(4*M) / self.p))
        if max_tries is not None:
            bound = min(bound, Integer(max_tries))

        mz_sq = Integer((4*M - self.p) // self.p)
        if mz_sq < 1:
            raise RuntimeError("no valid z-range")
        mz = Integer(mz_sq).isqrt()

        for _ in range(bound):
            z = Integer(rng.randint(1, int(mz)))
            mt_sq = Integer((4*M - self.p * z*z) // self.p)
            if mt_sq < 0:
                continue
            mt = Integer(mt_sq).isqrt()
            t = Integer(rng.randint(-int(mt), int(mt)))
            Mprime = Integer(4*M - self.p*(z*z + t*t))
            if Mprime <= 0:
                continue
            if not _probably_prime(Mprime):
                continue

            res = self.cornacchia(1, Mprime)
            if res is None:
                continue
            x, y = res

            if isogeny_cond:
                if (x - t) % 2 != 0:
                    x, y = y, x
                if ((x - t) % 4 != 2) or ((y - z) % 4 != 2):
                    continue

            dmax = self.dmax_dividing_integral_gamma_in_O0(x, y, z, t)
            if dmax != 2:
                continue

            return vector(QQ, [QQ(x)/2, QQ(y)/2, QQ(z)/2, QQ(t)/2])

        raise RuntimeError("GeneralizedRepresentInteger(O0 specialization) failed")

    # --------------------------------------------------------
    # Exact ideal-producing samplers
    # --------------------------------------------------------
    def sample_random_ideal_given_prime_norm_modsqrt_beta_modN(self, N, rng=None, lattice_norm_method="module", max_tries=4096, max_beta_tries=4096):
        """
        RandomIdealGivenPrimeNorm(N), using the user-requested modular-square-root variant.

        Input: a prime N which is the norm of some left O0 ideal.
        Output: a random left O0-ideal J' of norm N, or raise an exception.

        Algorithm implemented:
          1. Sample g1,g2,g3 uniformly in [0,N-1].
          2. gamma0 = g1*i + g2*j + g3*ij.
          3. Repeat until Legendre(-nrd(gamma0),N)=1.
          4. gamma = sqrt(-nrd(gamma0)) + gamma0.
          5. Sample beta = a + b*i + c*j + d*k uniformly from [0,N)^4,
             conditioned on nrd(beta) != 0 mod N.
          6. g = gamma*beta mod N.
          7. Return O0<g,N>.
        """
        N = Integer(N)
        if rng is None:
            rng = random.Random()
        # N is required by the algorithm to be prime.  We do not repeat an
        # expensive primality test inside every trial; callers pass the fixed
        # SQIsign prime-norm inputs.
        if N <= 1:
            raise ValueError("RandomIdealGivenPrimeNorm expects N > 1")
        if N == self.p:
            raise ValueError("RandomIdealGivenPrimeNorm expects N distinct from p")

        gamma = None
        for _ in range(int(max_tries)):
            g1 = Integer(rng.randrange(int(N)))
            g2 = Integer(rng.randrange(int(N)))
            g3 = Integer(rng.randrange(int(N)))
            gamma0 = vector(QQ, [0, g1, g2, g3])
            target = Integer((-self.quat_nrd(gamma0)) % N)
            if _legendre_symbol_prime(target, N) != 1:
                continue
            root = Integer(_modular_sqrt_prime_one(target, N))
            gamma = vector(QQ, [root, g1, g2, g3])
            break
        if gamma is None:
            raise RuntimeError("RandomIdealGivenPrimeNorm exceeded max_tries in Legendre/modsqrt loop")

        for _ in range(int(max_beta_tries)):
            a = Integer(rng.randrange(int(N)))
            b = Integer(rng.randrange(int(N)))
            c = Integer(rng.randrange(int(N)))
            d = Integer(rng.randrange(int(N)))
            beta = vector(QQ, [a, b, c, d])
            # nrd(beta) = a^2 + b^2 + p(c^2+d^2).  Since N is prime and N != p,
            # this is the requested nonzero-norm condition modulo N.
            if Integer(self.quat_nrd(beta)) % N == 0:
                continue
            g = self.reduce_quat_mod_N(self.quat_mul(gamma, beta), N)
            J = self.ideal_O0_alpha_N(g, N, method=lattice_norm_method)
            return {
                "name": "RandomIdealGivenPrimeNorm",
                "ideal": J,
                "norm": N,
                "alpha": g,
                "gamma": gamma,
                "beta": beta,
                "prime_sampler_variant": "g123_modsqrt_beta0N_modN",
            }
        raise RuntimeError("RandomIdealGivenPrimeNorm beta sampler exceeded max_beta_tries")

    def sample_random_ideal_given_norm_prime_spec2025(self, N, rng=None, lattice_norm_method="module", max_tries=4096, max_beta_tries=4096):
        # Backward-compatible name: this now points to the user-requested
        # RandomIdealGivenPrimeNorm implementation above.  The rest of the
        # experiment code can keep calling the old method name unchanged.
        return self.sample_random_ideal_given_prime_norm_modsqrt_beta_modN(
            N, rng=rng, lattice_norm_method=lattice_norm_method,
            max_tries=max_tries, max_beta_tries=max_beta_tries,
        )

    def sample_random_ideal_given_prime_norm_cornacchia_variant(self, N, rng=None, lattice_norm_method="module", max_beta_tries=4096):
        # Deprecated compatibility alias.  Earlier versions used a Cornacchia×2
        # variant for this name, but the current experiment intentionally uses
        # the g1,g2,g3 + ModularSQRT algorithm requested by the user.
        return self.sample_random_ideal_given_prime_norm_modsqrt_beta_modN(
            N, rng=rng, lattice_norm_method=lattice_norm_method,
            max_beta_tries=max_beta_tries,
        )

    def sample_random_ideal_given_norm_prime(self, N, rng=None, lattice_norm_method="module", max_tries=4096, max_beta_tries=4096):
        # Prime-norm branch used by every spec-Dmix upstream path and by any
        # explicit prime-norm experiment.  This is the user-requested
        # RandomIdealGivenPrimeNorm algorithm with Legendre/ModularSQRT and
        # beta sampled from [0,N)^4 conditioned on nonzero reduced norm mod N.
        return self.sample_random_ideal_given_prime_norm_modsqrt_beta_modN(
            N, rng=rng, lattice_norm_method=lattice_norm_method,
            max_tries=max_tries, max_beta_tries=max_beta_tries,
        )

    def sample_random_ideal_given_norm_composite(self, N, rng=None, lattice_norm_method="module", max_beta_tries=4096):
        # RandomIdealGivenNorm(N, prime=False)
        N = Integer(N)
        if rng is None:
            rng = random.Random()

        gamma = self.generalized_represent_integer_O0(self.quat_prime_cofactor * N, False, rng=rng)

        for _ in range(int(max_beta_tries)):
            x = Integer(rng.randint(1, int(N)))
            y = Integer(rng.randint(1, int(N)))
            z = Integer(rng.randint(1, int(N)))
            w = Integer(rng.randint(1, int(N)))
            beta = vector(QQ, [x, y, z, w])
            if gcd(Integer(self.quat_nrd(beta)), N) != 1:
                continue
            alpha = self.quat_mul(gamma, beta)
            I = self.ideal_O0_alpha_N(alpha, N, method=lattice_norm_method)
            return {
                "name": "RandomIdealGivenNorm(prime=False)",
                "ideal": I,
                "norm": N,
                "alpha": alpha,
                "gamma": gamma,
                "beta": beta,
            }
        raise RuntimeError("RandomIdealGivenNorm(prime=False) beta sampler exceeded max_beta_tries")

    def sample_random_equivalent_prime_ideal(self, I, norm_I, rng=None, lattice_norm_method="module"):
        # RandomEquivalentPrimeIdeal(I)
        if rng is None:
            rng = random.Random()
        norm_I = Integer(norm_I)
        Bred = self.l2_reduced_basis(I)
        bnd = Integer(self.quat_equiv_bound_coeff)
        max_tries = Integer((2*bnd + 1)**4)

        for _ in range(max_tries):
            coeffs = [Integer(rng.randint(-int(bnd), int(bnd))) for _ in range(Bred.ncols())]
            beta = sum((coeffs[j] * vector(QQ, Bred.column(j)) for j in range(Bred.ncols())), vector(QQ, [0,0,0,0]))
            nJ = QQ(self.quat_nrd(beta)) / QQ(norm_I)
            if nJ.denominator() != 1:
                continue
            nJ = Integer(nJ)
            if nJ <= 1 or (not _probably_prime(nJ)):
                continue
            J = (self.right_mult_matrix(self.quat_conj(beta)) * I) / norm_I
            J = self.normalize_lattice(J, method=lattice_norm_method)
            return {
                "name": "RandomEquivalentPrimeIdeal",
                "ideal": J,
                "norm": nJ,
                "beta": beta,
                "input_norm": norm_I,
            }

        raise RuntimeError("RandomEquivalentPrimeIdeal failed")

    def sample_random_equiv_prime_after_random_given_norm(self, N, rng=None, lattice_norm_method="module"):
        base = self.sample_random_ideal_given_norm_prime(N, rng=rng, lattice_norm_method=lattice_norm_method)
        return self.sample_random_equivalent_prime_ideal(
            base["ideal"], base["norm"], rng=rng, lattice_norm_method=lattice_norm_method
        )

    def sample_random_equiv_prime_after_random_given_prime_norm_cornacchia_variant(self, N, rng=None, lattice_norm_method="module"):
        # Deprecated compatibility alias.  It now uses the current
        # RandomIdealGivenPrimeNorm sampler, not the older Cornacchia×2 variant.
        base = self.sample_random_ideal_given_prime_norm_modsqrt_beta_modN(
            N, rng=rng, lattice_norm_method=lattice_norm_method
        )
        return self.sample_random_equivalent_prime_ideal(
            base["ideal"], base["norm"], rng=rng, lattice_norm_method=lattice_norm_method
        )

    # --------------------------------------------------------
    # KernelDecomposedToIdeal (trace-driven adapter)
    # --------------------------------------------------------
    def kernel_decomposed_to_ideal_from_precomp(self, c1, c2, M_i, M_theta, lattice_norm_method="module"):
        # This implements Algorithm 3.17 once the two torsion-action matrices
        # are supplied externally.
        mod = Integer(2) ** self.f
        c = vector(Integers(mod), [Integer(c1), Integer(c2)])
        M_i = matrix(Integers(mod), M_i)
        M_theta = matrix(Integers(mod), M_theta)

        dvec = M_theta * c
        M = matrix(Integers(mod), [[c[0], dvec[0]], [c[1], dvec[1]]])
        rhs = M_i * c
        if gcd(Integer(M.det().lift()), mod) != 1:
            raise RuntimeError("Matrix not invertible mod 2^f in KernelDecomposedToIdeal")
        ab = M.inverse() * rhs
        a = Integer(ab[0].lift())
        b = Integer(ab[1].lift())

        # alpha = a + b*(j + (1+k)/2) - i
        alpha = vector(QQ, [a + QQ(b)/2, -1, b, QQ(b)/2])
        I = self.ideal_O0_alpha_N(alpha, Integer(2) ** self.f, method=lattice_norm_method)
        return {
            "name": "KernelDecomposedToIdeal",
            "ideal": I,
            "norm": Integer(2) ** self.f,
            "alpha": alpha,
            "a": a,
            "b": b,
        }

    # --------------------------------------------------------
    # Trial runners / checkpointing
    # --------------------------------------------------------
    def _init_state(self, seed):
        py_seed = _coerce_py_seed(seed)
        return {
            "checkpoint_version": CODE_VERSION,
            "successes": 0,
            "attempts_total": 0,
            "rejects": 0,
            "exceptions": 0,
            "sum_ratio_over_norm": QQ(0),
            "max_ratio_over_norm": QQ(0),
            "plot_log10_ratios": [],
            "exception_hist": {},
            "first_exception": None,
            "last_exception": None,
            "worst": None,
            "printed_first_exception": False,
            "rng_state": random.Random(py_seed).getstate(),
            "seed": py_seed,
        }

    def _save_state(self, path, state):
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_state(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def run_trials(self, name, sampler, trials=DEFAULT_TRIALS, basis_mode="natural",
                   seed=0, checkpoint_path=None, save_every=1000, progress_every=None,
                   verbose=True, resume=True):
        trials = _coerce_py_int(trials, "trials")
        save_every = max(1, _coerce_py_int(save_every, "save_every"))
        if progress_every is None:
            progress_every = save_every
        progress_every = max(1, _coerce_py_int(progress_every, "progress_every"))

        if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
            state = self._load_state(checkpoint_path)
            if state.get("checkpoint_version") != CODE_VERSION:
                state = self._init_state(seed)
        else:
            state = self._init_state(seed)

        rng = random.Random()
        rng.setstate(state["rng_state"])

        t0 = time.time()
        while state["attempts_total"] < trials:
            state["attempts_total"] += 1
            try:
                out = sampler(rng)
                ev = self.event_data(out["ideal"], out["norm"], basis_mode=basis_mode)
                ratio = ev["ratio_over_norm"]
                log10_ratio = float(ev["ratio_over_norm_log10"])
                state["successes"] += 1
                if ev["reject"]:
                    state["rejects"] += 1
                state["sum_ratio_over_norm"] += ratio
                state["plot_log10_ratios"].append(log10_ratio)
                if ratio > state["max_ratio_over_norm"]:
                    state["max_ratio_over_norm"] = ratio
                    state["worst"] = {
                        "trial_success_index": state["successes"],
                        "attempt_total": state["attempts_total"],
                        "norm": str(out["norm"]),
                        "max_basis_nrd": str(ev["max_basis_nrd"]),
                        "ratio_over_norm": str(ratio),
                        "legacy_ratio_over_norm_square": str(ev["legacy_ratio_over_norm_square"]),
                        "log10_ratio_over_norm": log10_ratio,
                        "reject_threshold_log10": ev["reject_threshold_log10"],
                        "reject": bool(ev["reject"]),
                        "basis": str(ev["basis"]),
                        "basis_nrds": [str(x) for x in ev["basis_nrds"]],
                    }
            except Exception as e:
                state["exceptions"] += 1
                _record_failure_example(state, e)
                if state["worst"] is None:
                    state["worst"] = {"first_exception": repr(e)}

            if checkpoint_path is not None and (state["attempts_total"] % save_every == 0) and state["attempts_total"] > 0:
                state["rng_state"] = rng.getstate()
                self._save_state(checkpoint_path, state)

            if verbose and (state["attempts_total"] % progress_every == 0) and state["attempts_total"] > 0:
                print("[{} | {} | {}] constructed={} accepted={} rejected={} exceptions={} attempts={} elapsed={:.1f}s".format(
                    self.level, name, basis_mode,
                    state["successes"], state["successes"] - state["rejects"], state["rejects"], state["exceptions"], state["attempts_total"], time.time() - t0
                ), flush=True)
                if (state["first_exception"] is not None) and (not state.get("printed_first_exception", False)):
                    print("    first_exception = {}".format(state["first_exception"]), flush=True)
                    state["printed_first_exception"] = True

        state["rng_state"] = rng.getstate()
        if checkpoint_path is not None:
            self._save_state(checkpoint_path, state)

        successes = Integer(state["successes"])
        attempts_total = Integer(state["attempts_total"])
        exceptions = Integer(state["exceptions"])
        rejects = Integer(state["rejects"])
        accepted = successes - rejects
        threshold = self.reject_ratio_threshold()
        qtls = _sorted_quantiles(state["plot_log10_ratios"])
        avg_ratio = (state["sum_ratio_over_norm"] / QQ(successes)) if successes > 0 else None
        return {
            "level": self.level,
            "name": name,
            "basis_mode": basis_mode,
            "trials_requested": Integer(trials),
            "trials_done": attempts_total,
            "successful_samples": successes,
            "constructed_samples": successes,
            "accepted_samples": accepted,
            "attempts_total": attempts_total,
            "exceptions": exceptions,
            "failures": exceptions,
            "rejects": rejects,
            "success_rate": _safe_qq_ratio(successes, attempts_total),
            "constructed_rate": _safe_qq_ratio(successes, attempts_total),
            "accepted_rate": _safe_qq_ratio(accepted, attempts_total),
            "accept_probability_success_only": _safe_qq_ratio(accepted, successes),
            "failure_rate": _safe_qq_ratio(exceptions, attempts_total),
            "exception_rate": _safe_qq_ratio(exceptions, attempts_total),
            "reject_probability_success_only": _safe_qq_ratio(rejects, successes),
            "reject_probability_over_attempts": _safe_qq_ratio(rejects, attempts_total),
            "average_ratio_over_norm": avg_ratio,
            "average_ratio": avg_ratio,
            "max_ratio_over_norm": state["max_ratio_over_norm"],
            "max_ratio": state["max_ratio_over_norm"],
            "threshold_ratio_bound": threshold,
            "threshold_log10_bound": float(threshold.log10()),
            "plot_log10_ratios": list(state["plot_log10_ratios"]),
            "log10_ratio_quantiles": qtls,
            "first_exception": state["first_exception"],
            "last_exception": state["last_exception"],
            "exception_hist": dict(state["exception_hist"]),
            "worst": state["worst"],
        }


# ============================================================
# Ready-to-run suites
# ============================================================

def run_actual_fixed_input_suite(trials=DEFAULT_TRIALS, basis_mode="natural", seed=0,
                                 checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                                 include_raw_ridn=False,
                                 include_reqp_after_ridn=True,
                                 save_every=1000, progress_every=None, verbose=True, resume=True):
    """
    Main fixed-input suite for the original SQIsign-spec "Dmix" experiments.

    This uses the exact SQIsign specification value
        Dmix = smallest prime larger than 2^(4 lambda).

    For each level, it can run:
      - RandomIdealGivenNorm(Dmix, true)  [implemented via the user-requested
        Legendre/ModularSQRT + beta in [0,N)^4 prime-norm variant]
      - RandomEquivalentPrimeIdeal after RandomIdealGivenNorm(Dmix, true)

    Default behavior:
      - include_raw_ridn=False
      - include_reqp_after_ridn=True

    This matches the requested experiment scope: keep the REQP output
    produced *after* RandomIdealGivenNorm(Dmix,true), but do not benchmark the
    standalone raw Dmix RIDN row in the default fixed-input suite.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    results = {}
    for level in ["NIST-I", "NIST-III", "NIST-V"]:
        ctx = SQISignIdealLab(level)
        results[level] = {}

        if include_raw_ridn:
            name1 = "RandomIdealGivenNorm(Dmix,true)"
            cp1 = os.path.join(checkpoint_dir, "{}_{}_{}.pkl".format(level, "RIDNprimeDmix", basis_mode))
            results[level][name1] = ctx.run_trials(
                name=name1,
                sampler=lambda rng, ctx=ctx: ctx.sample_random_ideal_given_norm_prime(ctx.Dmix_spec, rng=rng),
                trials=trials,
                basis_mode=basis_mode,
                seed=seed,
                checkpoint_path=cp1,
                save_every=save_every, progress_every=progress_every, verbose=verbose, resume=resume,
            )

        if include_reqp_after_ridn:
            name2 = "RandomEquivalentPrimeIdeal(after RandomIdealGivenNorm(Dmix,true))"
            cp2 = os.path.join(checkpoint_dir, "{}_{}_{}.pkl".format(level, "REQPafterDmix", basis_mode))
            results[level][name2] = ctx.run_trials(
                name=name2,
                sampler=lambda rng, ctx=ctx: ctx.sample_random_equiv_prime_after_random_given_norm(ctx.Dmix_spec, rng=rng),
                trials=trials,
                basis_mode=basis_mode,
                seed=_seed_plus(seed, 1),
                checkpoint_path=cp2,
                save_every=save_every, progress_every=progress_every, verbose=verbose, resume=resume,
            )
    return results


def run_spec_fixed_input_suite(trials=DEFAULT_TRIALS, basis_mode="natural", seed=0,
                               checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                               include_raw_ridn=False,
                               include_reqp_after_ridn=True,
                               save_every=1000, progress_every=None, verbose=True, resume=True):
    """
    Backward-compatible alias for the exact-spec fixed-input suite.

    Since the default fixed-input benchmark now already uses the original
    SQIsign specification value Dmix, this helper simply forwards to
    run_actual_fixed_input_suite().
    """
    return run_actual_fixed_input_suite(
        trials=trials, basis_mode=basis_mode, seed=seed, checkpoint_dir=checkpoint_dir,
        include_raw_ridn=include_raw_ridn, include_reqp_after_ridn=include_reqp_after_ridn,
        save_every=save_every, progress_every=progress_every, verbose=verbose, resume=resume,
    )



def run_random_ideal_given_norm_composite_suite(composite_norm_samplers,
                                                trials=DEFAULT_TRIALS,
                                                basis_mode="natural",
                                                seed=0,
                                                checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                                                save_every=1000, progress_every=None, verbose=True, resume=True):
    """
    composite_norm_samplers: dict mapping
       level -> function(ctx, rng) returning a positive odd composite N not divisible by p.

    This lets you test the prime=False branch of RandomIdealGivenNorm under an
    input distribution of your choice, including sign-like distributions of the
    form N = 2^e - q_rsp.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    results = {}
    for level in ["NIST-I", "NIST-III", "NIST-V"]:
        ctx = SQISignIdealLab(level)
        if level not in composite_norm_samplers:
            raise ValueError("missing composite norm sampler for {}".format(level))
        N_sampler = composite_norm_samplers[level]
        name = "RandomIdealGivenNorm(N,false)"
        cp = os.path.join(checkpoint_dir, "{}_{}_{}.pkl".format(level, "RIDNcomposite", basis_mode))
        results[level] = ctx.run_trials(
            name=name,
            sampler=lambda rng, ctx=ctx, N_sampler=N_sampler: ctx.sample_random_ideal_given_norm_composite(
                Integer(N_sampler(ctx, rng)), rng=rng
            ),
            trials=trials,
            basis_mode=basis_mode,
            seed=seed,
            checkpoint_path=cp,
            save_every=save_every, progress_every=progress_every, verbose=verbose, resume=resume,
        )
    return results


def run_default_sign_like_composite_suite(trials=DEFAULT_TRIALS,
                                          basis_mode="natural",
                                          seed=0,
                                          checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                                          e_min=2,
                                          e_max_by_level=None,
                                          save_every=1000, progress_every=None, verbose=True, resume=True):
    """
    Turn-key composite experiment for the signing-side shape
        N = 2^e - q_rsp,
    with q_rsp odd, 0 < q_rsp < 2^e, and N forced to be composite.

    By default, e is sampled uniformly from [e_min, ersp].
    """
    if e_max_by_level is None:
        e_max_by_level = {}

    composite_samplers = {}
    for level in ["NIST-I", "NIST-III", "NIST-V"]:
        def _sampler(ctx, rng, level=level):
            e_max = e_max_by_level.get(level, None)
            return ctx.sample_sign_like_composite_norm(rng=rng, e_min=e_min, e_max=e_max, require_composite=True)
        composite_samplers[level] = _sampler

    return run_random_ideal_given_norm_composite_suite(
        composite_samplers,
        trials=trials,
        basis_mode=basis_mode,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        save_every=save_every, progress_every=progress_every, verbose=verbose, resume=resume,
    )


def run_kernel_decomposed_suite(M_i_by_level, M_theta_by_level, c_samplers,
                                trials=DEFAULT_TRIALS,
                                basis_mode="natural",
                                seed=0,
                                checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                                save_every=1000, progress_every=None, verbose=True, resume=True):
    """
    Trace-driven suite for KernelDecomposedToIdeal.

    Inputs:
      - M_i_by_level[level]      : 2x2 matrix mod 2^f for action of i on E0[2^f]
      - M_theta_by_level[level]  : 2x2 matrix mod 2^f for action of theta=j+(1+k)/2 on E0[2^f]
      - c_samplers[level](ctx,rng): returns (c1,c2) modulo 2^f

    This is the clean way to stay faithful to the official precomputed torsion data.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    results = {}
    for level in ["NIST-I", "NIST-III", "NIST-V"]:
        ctx = SQISignIdealLab(level)
        Mi = M_i_by_level[level]
        Mtheta = M_theta_by_level[level]
        csamp = c_samplers[level]
        name = "KernelDecomposedToIdeal"
        cp = os.path.join(checkpoint_dir, "{}_{}_{}.pkl".format(level, "KernelDecomp", basis_mode))
        results[level] = ctx.run_trials(
            name=name,
            sampler=lambda rng, ctx=ctx, Mi=Mi, Mtheta=Mtheta, csamp=csamp: ctx.kernel_decomposed_to_ideal_from_precomp(
                *csamp(ctx, rng), M_i=Mi, M_theta=Mtheta
            ),
            trials=trials,
            basis_mode=basis_mode,
            seed=seed,
            checkpoint_path=cp,
            save_every=save_every, progress_every=progress_every, verbose=verbose, resume=resume,
        )
    return results


def run_kernel_decomposed_suite_official_ref(trials=DEFAULT_TRIALS,
                                             basis_mode="natural",
                                             seed=0,
                                             checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                                             c_samplers=None,
                                             save_every=1000, progress_every=None, verbose=True, resume=True):
    '''
    Turn-key KernelDecomposedToIdeal experiment using the official reference
    precomputation extracted from the uploaded ref.zip.

    If c_samplers is omitted, a default distribution is used that samples
    random exact-order points [c1]P0 + [c2]Q0 on E0[2^f] for which the
    algorithmic matrix is invertible modulo 2^f.
    '''
    M_i_by_level, M_theta_by_level = load_official_ref_kernel_precomp()
    if c_samplers is None:
        c_samplers = make_default_kernel_decomposed_c_samplers(M_theta_by_level)
    return run_kernel_decomposed_suite(
        M_i_by_level,
        M_theta_by_level,
        c_samplers,
        trials=trials,
        basis_mode=basis_mode,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        save_every=save_every, progress_every=progress_every, verbose=verbose, resume=resume,
    )



def _resolve_default_prime_input(ctx, prime_input):
    '''
    Resolve the base prime-norm input used to build the default deterministic
    ideal-operation experiments.

    Accepted values:
      - "Dmix_4k1" : smallest prime 1 mod 4 above 2^(4 lambda)
      - "Dmix_spec": exact SQIsign spec Dmix
      - any positive integer prime
    '''
    if prime_input in [None, "Dmix", "Dmix_spec"]:
        return Integer(ctx.Dmix_spec), "Dmix"
    if prime_input == "Dmix_4k1":
        return Integer(ctx.Dmix_4k1), "Dmix4k1"
    N = Integer(prime_input)
    if N <= 1 or (not N.is_prime()):
        raise ValueError("prime_input must be 'Dmix', 'Dmix_spec', 'Dmix_4k1', or a positive prime integer")
    return N, "customPrime"


def _default_connecting_sampler_from_random_equiv_prime(ctx, source_prime_norm, rng):
    Jd = ctx.sample_random_equiv_prime_after_random_given_norm(source_prime_norm, rng=rng)
    OR = ctx.right_order_of_connecting_maximal_ideal(Jd["ideal"], Jd["norm"])
    K, nK = ctx.compute_connecting_ideal_with_norm(ctx.O0, OR)
    return {
        "ideal": K,
        "norm": nK,
        "source": Jd,
        "right_order": OR,
    }


def _default_pullback_sampler_from_random_equiv_prime(ctx, source_prime_norm, rng, max_resamples=1000):
    raise RuntimeError("pullback default sampler has been retired from the bundled experiments")


def _make_protocol_pushforward_fixed_secret(ctx, source_prime_norm, seed, max_secret_resamples=256):
    '''
    Emulate the SQIsign signing path by fixing a secret ideal
        Isk = RandomEquivalentPrimeIdeal(RandomIdealGivenNorm(Dmix, true))
    once per level, then reusing it across all trials of the pushforward suite.

    This matches Algorithm 4.2 more closely than resampling a new secret key ideal
    for every pushforward sample.
    '''
    secret_seed = _coerce_py_seed(seed)
    secret_rng = random.Random(secret_seed)
    first_exc = None
    for _ in range(int(max_secret_resamples)):
        try:
            Jsk = ctx.sample_random_equiv_prime_after_random_given_norm(source_prime_norm, rng=secret_rng)
        except Exception as e:
            if first_exc is None:
                first_exc = repr(e)
            continue
        if gcd(Integer(Jsk["norm"]), Integer(2) ** ctx.f) != 1:
            continue
        OR = ctx.right_order_of_connecting_maximal_ideal(Jsk["ideal"], Jsk["norm"])
        return {
            "secret": Jsk,
            "right_order": OR,
        }
    if first_exc is None:
        raise RuntimeError("protocol pushforward fixed-secret generation failed")
    raise RuntimeError("protocol pushforward fixed-secret generation failed; first exception = {}".format(first_exc))


def _default_pushforward_sampler_from_reqp_and_kernel(ctx, fixed_secret, M_i, M_theta, c_sampler, rng):
    '''
    Protocol-faithful pushforward experiment for the SQIsign signing path.

    In Algorithm 4.2, the signer computes
        I'_chl <- KernelDecomposedToIdeal(c1,c2)
        Ichl   <- [Isk]_* I'_chl,
    where Isk is the secret key ideal output by RandomEquivalentPrimeIdeal after
    RandomIdealGivenNorm(Dmix, true).

    Here we therefore test pushforward with
      - J = Isk  (fixed per level for the whole suite run), and
      - I = I'_chl from KernelDecomposedToIdeal.
    '''
    Jsk = fixed_secret["secret"]
    OR = fixed_secret["right_order"]
    Ik = ctx.kernel_decomposed_to_ideal_from_precomp(*c_sampler(ctx, rng), M_i=M_i, M_theta=M_theta)
    if gcd(Integer(Jsk["norm"]), Integer(Ik["norm"])) != 1:
        raise RuntimeError("protocol pushforward sampler got non-coprime norms")
    Ipf = ctx.ideal_pushforward(Jsk["ideal"], Jsk["norm"], Ik["ideal"])
    return {
        "ideal": Ipf,
        # In the signing protocol, Ichl corresponds to a challenge isogeny of degree 2^f,
        # so the tested pushforward ideal has norm 2^f as well.
        "norm": Integer(Ik["norm"]),
        "secret_source": Jsk,
        "challenge_source": Ik,
        "right_order": OR,
    }


def run_default_connecting_ideal_suite(trials=DEFAULT_TRIALS,
                                       basis_mode="natural",
                                       seed=0,
                                       checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                                       prime_input="Dmix_spec",
                                       save_every=1000, progress_every=None, verbose=True, resume=True):
    '''
    Turn-key experiment for ComputeConnectingIdeal.

    Default input path per trial:
      1) J <- RandomIdealGivenNorm(prime_input, true)
      2) J <- RandomEquivalentPrimeIdeal(J)
      3) OR <- right_order(J)
      4) K <- ComputeConnectingIdeal(O0, OR)
    and the tested ideal is K.
    '''
    os.makedirs(checkpoint_dir, exist_ok=True)
    results = {}
    for level in ["NIST-I", "NIST-III", "NIST-V"]:
        ctx = SQISignIdealLab(level)
        source_prime_norm, prime_tag = _resolve_default_prime_input(ctx, prime_input)
        name = "ComputeConnectingIdeal(default path, {})".format(prime_tag)
        cp = os.path.join(checkpoint_dir, "{}_{}_{}.pkl".format(level, "ComputeConnectingIdeal_" + prime_tag, basis_mode))
        results[level] = ctx.run_trials(
            name=name,
            sampler=lambda rng, ctx=ctx, source_prime_norm=source_prime_norm: _default_connecting_sampler_from_random_equiv_prime(
                ctx, source_prime_norm, rng
            ),
            trials=trials,
            basis_mode=basis_mode,
            seed=seed,
            checkpoint_path=cp,
            save_every=save_every, progress_every=progress_every, verbose=verbose, resume=resume,
        )
    return results


def run_default_pullback_suite(trials=DEFAULT_TRIALS,
                               basis_mode="natural",
                               seed=0,
                               checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                               prime_input="Dmix_spec",
                               max_resamples=1000,
                               save_every=1000, progress_every=None, verbose=True, resume=True):
    '''
    Pullback is intentionally not part of the default bundled experiments anymore.

    The SQIsign signing algorithm uses pushforward in the response path
        Ichl <- [Isk]_* I'_chl,
    and does not use pullback as a pre-multiplication ideal experiment. The old
    default pullback sampler was also structurally ill-posed for this benchmark:
    it attempted to pair a prime-norm connecting ideal J with another connecting
    ideal K between the opposite orders, but in practice these norms share the
    same prime factor, so the required coprimality condition almost never holds.
    '''
    raise RuntimeError(
        "run_default_pullback_suite has been removed from the default experiments; "
        "use pushforward with RandomEquivalentPrimeIdeal and KernelDecomposedToIdeal instead"
    )


def run_default_pushforward_suite(trials=DEFAULT_TRIALS,
                                  basis_mode="natural",
                                  seed=0,
                                  checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                                  prime_input="Dmix_spec",
                                  c_samplers=None,
                                  max_secret_resamples=256,
                                  save_every=1000, progress_every=None, verbose=True, resume=True):
    '''
    Turn-key pushforward experiment following the actual SQIsign signing path.

    Per level, we build:
      - a secret-key ideal Isk from RandomEquivalentPrimeIdeal(RandomIdealGivenNorm(prime_input, true)), and
      - a challenge-side ideal I'_chl from KernelDecomposedToIdeal.

    The tested ideal is then
        Ichl = [Isk]_* I'_chl,
    exactly as in Algorithm 4.2.

    By default, Isk is sampled once per level (deterministically from the provided
    seed) and reused across all trials, which is closer to how a real secret key
    is used across many signatures.
    '''
    os.makedirs(checkpoint_dir, exist_ok=True)
    M_i_by_level, M_theta_by_level = load_official_ref_kernel_precomp()
    if c_samplers is None:
        c_samplers = make_default_kernel_decomposed_c_samplers(M_theta_by_level)
    results = {}
    level_offsets = {"NIST-I": 401, "NIST-III": 403, "NIST-V": 405}
    for level in ["NIST-I", "NIST-III", "NIST-V"]:
        ctx = SQISignIdealLab(level)
        source_prime_norm, prime_tag = _resolve_default_prime_input(ctx, prime_input)
        Mi = M_i_by_level[level]
        Mtheta = M_theta_by_level[level]
        csamp = c_samplers[level]
        secret_bundle = _make_protocol_pushforward_fixed_secret(
            ctx,
            source_prime_norm,
            _seed_plus(seed, level_offsets[level]),
            max_secret_resamples=max_secret_resamples,
        )
        name = "ideal_pushforward(protocol path: REQP secret + KernelDecomposedToIdeal challenge, upstream={})".format(prime_tag)
        cp = os.path.join(checkpoint_dir, "{}_{}_{}.pkl".format(level, "IdealPushforwardProtocol_" + prime_tag, basis_mode))
        results[level] = ctx.run_trials(
            name=name,
            sampler=lambda rng, ctx=ctx, secret_bundle=secret_bundle, Mi=Mi, Mtheta=Mtheta, csamp=csamp: _default_pushforward_sampler_from_reqp_and_kernel(
                ctx, secret_bundle, Mi, Mtheta, csamp, rng
            ),
            trials=trials,
            basis_mode=basis_mode,
            seed=seed,
            checkpoint_path=cp,
            save_every=save_every, progress_every=progress_every, verbose=verbose, resume=resume,
        )
    return results


def run_default_deterministic_ideal_ops_suite(trials=DEFAULT_TRIALS,
                                              basis_mode="natural",
                                              seed=0,
                                              checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                                              prime_input="Dmix_spec",
                                              c_samplers=None,
                                              max_secret_resamples=256,
                                              save_every=1000, progress_every=None, verbose=True, resume=True):
    '''
    Combined turn-key suite for the deterministic ideal operations that are still
    relevant to the current protocol-faithful experiments:
      - ComputeConnectingIdeal
      - ideal_pushforward following the signing path

    Pullback is intentionally omitted from the bundled run.
    '''
    res_connect = run_default_connecting_ideal_suite(
        trials=trials,
        basis_mode=basis_mode,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        prime_input=prime_input,
        save_every=save_every, progress_every=progress_every, verbose=verbose, resume=resume,
    )
    res_push = run_default_pushforward_suite(
        trials=trials,
        basis_mode=basis_mode,
        seed=_seed_plus(seed, 2),
        checkpoint_dir=checkpoint_dir,
        prime_input=prime_input,
        c_samplers=c_samplers,
        max_secret_resamples=max_secret_resamples,
        save_every=save_every, progress_every=progress_every, verbose=verbose, resume=resume,
    )

    results = {}
    for level in ["NIST-I", "NIST-III", "NIST-V"]:
        results[level] = {
            res_connect[level]["name"]: res_connect[level],
            res_push[level]["name"]: res_push[level],
        }
    return results

# ============================================================
# Minimal result printer / plotting helpers
# ============================================================

def print_result_block(res):
    print("="*72)
    print("level                 :", res["level"])
    print("algorithm             :", res["name"])
    print("basis_mode            :", res["basis_mode"])
    print("trials_done           :", res["trials_done"])
    print("constructed_samples   :", res.get("constructed_samples", res["successful_samples"]))
    print("accepted_samples      :", res.get("accepted_samples", res["successful_samples"] - res["rejects"]))
    print("rejected_samples      :", res["rejects"])
    print("exceptions            :", res["exceptions"])
    print("attempts_total        :", res["attempts_total"])
    print("constructed_rate      :", N(res.get("constructed_rate", res["success_rate"]), 20) if res.get("constructed_rate", res["success_rate"]) is not None else None)
    print("accepted_rate         :", N(res.get("accepted_rate"), 20) if res.get("accepted_rate") is not None else None)
    print("exception_rate        :", N(res["exception_rate"], 20) if res["exception_rate"] is not None else None)
    print("Pr[accept | constructed]:", N(res.get("accept_probability_success_only"), 20) if res.get("accept_probability_success_only") is not None else None)
    print("Pr[reject | constructed]:", N(res["reject_probability_success_only"], 20) if res["reject_probability_success_only"] is not None else None)
    print("Pr[reject | attempt]  :", N(res["reject_probability_over_attempts"], 20) if res["reject_probability_over_attempts"] is not None else None)
    print("avg ratio             :", N(res["average_ratio_over_norm"], 20) if res["average_ratio_over_norm"] is not None else None,
          " (= avg max_i nrd(alpha_i) / nrd(I) over constructed samples )")
    print("max ratio             :", N(res["max_ratio_over_norm"], 20) if res["max_ratio_over_norm"] is not None else None)
    print("threshold             :", N(res["threshold_ratio_bound"], 20), " (= 64*p^2/pi^4 )")
    if res.get("first_exception") is not None:
        print("first_exception       :", res["first_exception"])
    if res.get("last_exception") is not None and res.get("last_exception") != res.get("first_exception"):
        print("last_exception        :", res["last_exception"])
    if res.get("exception_hist"):
        top = sorted(res["exception_hist"].items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        print("top_exceptions        :", top)
    print("worst                 :", res["worst"])


def plot_ratio_histogram(res, bins=100, show=True, savepath=None, title=None, allow_empty=True):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting") from e

    xs = list(res.get("plot_log10_ratios", []))
    if len(xs) == 0:
        if allow_empty:
            return None
        raise RuntimeError("no recorded ratio samples available for plotting")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(xs, bins=bins)
    ax.axvline(float(res["threshold_log10_bound"]), linestyle="--")
    ax.set_xlabel("log10(max_i nrd(alpha_i) / nrd(I))")
    ax.set_ylabel("count")
    if title is None:
        title = "{} | {} | {}".format(res.get("level", "?"), res.get("name", "?"), res.get("basis_mode", "?"))
    ax.set_title(title)
    ax.text(0.02, 0.98,
            "constructed = {}\naccepted = {}\nrejected = {}\nthreshold log10 = {:.6f}".format(
                int(res.get("constructed_samples", res["successful_samples"])),
                int(res.get("accepted_samples", res["successful_samples"] - res["rejects"])),
                int(res["rejects"]),
                float(res["threshold_log10_bound"])
            ),
            transform=ax.transAxes, va="top", ha="left")
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=180)
    if show:
        plt.show()
    return fig, ax


def plot_suite_histograms(results, bins=100, save_dir=DEFAULT_PLOT_DIR, show=True):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    figs = {}
    for res in _flatten_result_dicts(results):
        savepath = None
        if save_dir is not None:
            fname = "{}_{}_{}.png".format(_slugify(res["level"]), _slugify(res["name"]), _slugify(res["basis_mode"]))
            savepath = os.path.join(save_dir, fname)
        fig = plot_ratio_histogram(res, bins=bins, show=show, savepath=savepath, allow_empty=True)
        if fig is not None:
            figs[(res["level"], res["name"])] = fig
    return figs


def plot_all_ratio_histograms(results, output_dir=DEFAULT_PLOT_DIR, bins=100, show=True):
    return plot_suite_histograms(results, bins=bins, save_dir=output_dir, show=show)


def _suite_banner(msg):
    print("\n" + "="*18 + " " + str(msg) + " " + "="*18, flush=True)


def _suite_done(msg):
    print("[done] {}".format(msg), flush=True)


def run_all_requested_reject_experiments(trials=DEFAULT_TRIALS,
                                         basis_mode="natural",
                                         seed=0,
                                         checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                                         plot_dir=None,
                                         show_plots=False,
                                         include_spec=False,
                                         include_composite=True,
                                         include_kernel=True,
                                         include_deterministic=False,
                                         composite_samplers=None,
                                         e_min=2,
                                         e_max_by_level=None,
                                         prime_input="Dmix_spec",
                                         max_resamples=1000,
                                         bins=100,
                                         include_actual_raw_ridn=False,
                                         include_actual_reqp_after_ridn=True,
                                         include_spec_raw_ridn=False,
                                         include_spec_reqp_after_ridn=True,
                                         save_every=1000,
                                         progress_every=1000,
                                         verbose=True,
                                         resume=True):
    """
    Bundled runner for the *pre-multiplication ideal-generation* algorithms only.

    By default this now runs exactly the ideal samplers that appear before ideal
    multiplication in the SQIsign protocol paths studied here:
      - RandomEquivalentPrimeIdeal(after RandomIdealGivenNorm(Dmix, true))
      - RandomIdealGivenNorm(N, false) on the sign-like composite shape N = 2^e - q_rsp
      - KernelDecomposedToIdeal

    Deliberately *not* included by default anymore:
      - standalone RandomIdealGivenNorm(Dmix, true)
      - an additional duplicate exact-spec suite (no longer needed)
      - ComputeConnectingIdeal
      - ideal_pullback
      - ideal_pushforward

    Those deterministic ideal operations may still exist as helpers below, but
    they are no longer part of the default benchmark because they are not the
    "ideal is sampled before multiplication" algorithms the user asked to test.
    """
    results = {}

    _suite_banner("pre-multiplication fixed-input SQIsign Dmix suite")
    _merge_result_maps(results, run_actual_fixed_input_suite(
        trials=trials,
        basis_mode=basis_mode,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        include_raw_ridn=include_actual_raw_ridn,
        include_reqp_after_ridn=include_actual_reqp_after_ridn,
        save_every=save_every,
        progress_every=progress_every,
        verbose=verbose,
        resume=resume,
    ))
    _suite_done("pre-multiplication fixed-input SQIsign Dmix suite")

    if include_spec and verbose:
        print("[note] include_spec=True is redundant: the default fixed-input suite already uses the original SQIsign Dmix.", flush=True)

    if include_composite:
        if composite_samplers is None:
            _suite_banner("pre-multiplication sign-like composite RIDN(N,false) suite")
            comp = run_default_sign_like_composite_suite(
                trials=trials, basis_mode=basis_mode, seed=_seed_plus(seed, 21),
                checkpoint_dir=checkpoint_dir, e_min=e_min, e_max_by_level=e_max_by_level,
                save_every=save_every, progress_every=progress_every, verbose=verbose, resume=resume
            )
            _suite_done("pre-multiplication sign-like composite RIDN(N,false) suite")
        else:
            _suite_banner("pre-multiplication custom composite RIDN(N,false) suite")
            comp = run_random_ideal_given_norm_composite_suite(
                composite_samplers, trials=trials, basis_mode=basis_mode,
                seed=_seed_plus(seed, 21), checkpoint_dir=checkpoint_dir,
                save_every=save_every, progress_every=progress_every, verbose=verbose, resume=resume
            )
            _suite_done("pre-multiplication custom composite RIDN(N,false) suite")
        _merge_result_maps(results, comp)

    if include_kernel:
        _suite_banner("pre-multiplication KernelDecomposedToIdeal suite")
        _merge_result_maps(results, run_kernel_decomposed_suite_official_ref(
            trials=trials, basis_mode=basis_mode, seed=_seed_plus(seed, 31), checkpoint_dir=checkpoint_dir,
            save_every=save_every, progress_every=progress_every, verbose=verbose, resume=resume
        ))
        _suite_done("pre-multiplication KernelDecomposedToIdeal suite")

    if include_deterministic:
        _suite_banner("extra deterministic ideal-op suite")
        _merge_result_maps(results, run_default_deterministic_ideal_ops_suite(
            trials=trials, basis_mode=basis_mode, seed=_seed_plus(seed, 41),
            checkpoint_dir=checkpoint_dir, prime_input=prime_input, max_secret_resamples=max_resamples,
            save_every=save_every, progress_every=progress_every, verbose=verbose, resume=resume
        ))
        _suite_done("extra deterministic ideal-op suite")

    if plot_dir is not None or show_plots:
        _suite_banner("plotting histograms")
        plot_suite_histograms(results, bins=bins, save_dir=plot_dir, show=show_plots)
        _suite_done("plotting histograms")

    return results


def run_all_without_raw_randomidealgivennorm(trials=DEFAULT_TRIALS,
                                           basis_mode="natural",
                                           seed=0,
                                           checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                                           plot_dir=None,
                                           show_plots=False,
                                           include_spec=False,
                                           include_composite=True,
                                           include_kernel=True,
                                           include_deterministic=False,
                                           composite_samplers=None,
                                           e_min=2,
                                           e_max_by_level=None,
                                           prime_input="Dmix_spec",
                                           max_resamples=1000,
                                           bins=100,
                                           save_every=1000,
                                           progress_every=1000,
                                           verbose=True,
                                           resume=True):
    """
    Convenience wrapper for the same pre-multiplication benchmark, but with the
    standalone RandomIdealGivenNorm rows removed.

    It keeps by default:
      - RandomEquivalentPrimeIdeal(after RandomIdealGivenNorm(...))
      - RandomIdealGivenNorm(N,false)
      - KernelDecomposedToIdeal

    Deterministic ideal operations remain opt-in only.
    """
    return run_all_requested_reject_experiments(
        trials=trials,
        basis_mode=basis_mode,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        plot_dir=plot_dir,
        show_plots=show_plots,
        include_spec=include_spec,
        include_composite=include_composite,
        include_kernel=include_kernel,
        include_deterministic=include_deterministic,
        composite_samplers=composite_samplers,
        e_min=e_min,
        e_max_by_level=e_max_by_level,
        prime_input=prime_input,
        max_resamples=max_resamples,
        bins=bins,
        include_actual_raw_ridn=False,
        include_actual_reqp_after_ridn=True,
        include_spec_raw_ridn=False,
        include_spec_reqp_after_ridn=True,
        save_every=save_every,
        progress_every=progress_every,
        verbose=verbose,
        resume=resume,
    )


def run_all_pre_multiplication_ideal_generation_experiments(trials=DEFAULT_TRIALS,
                                                           basis_mode="natural",
                                                           seed=0,
                                                           checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                                                           plot_dir=None,
                                                           show_plots=False,
                                                           include_spec=False,
                                                           include_composite=True,
                                                           include_kernel=True,
                                                           include_raw_ridn=False,
                                                           include_reqp_after_ridn=True,
                                                           include_spec_raw_ridn=False,
                                                           include_spec_reqp_after_ridn=True,
                                                           save_every=1000,
                                                           progress_every=1000,
                                                           verbose=True,
                                                           resume=True,
                                                           bins=100,
                                                           composite_samplers=None,
                                                           e_min=2,
                                                           e_max_by_level=None):
    """
    Friendly alias for the default bundled benchmark over the pre-multiplication ideal-generation algorithms.
    """
    return run_all_requested_reject_experiments(
        trials=trials,
        basis_mode=basis_mode,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        plot_dir=plot_dir,
        show_plots=show_plots,
        include_spec=include_spec,
        include_composite=include_composite,
        include_kernel=include_kernel,
        include_deterministic=False,
        composite_samplers=composite_samplers,
        e_min=e_min,
        e_max_by_level=e_max_by_level,
        bins=bins,
        include_actual_raw_ridn=include_raw_ridn,
        include_actual_reqp_after_ridn=include_reqp_after_ridn,
        include_spec_raw_ridn=include_spec_raw_ridn,
        include_spec_reqp_after_ridn=include_spec_reqp_after_ridn,
        save_every=save_every,
        progress_every=progress_every,
        verbose=verbose,
        resume=resume,
    )


def probe_first_exceptions(basis_mode="natural", seed=0, prime_input="Dmix_spec"):
    """
    Run one attempt for the main algorithms and print the first captured exception.
    Useful when a Sage installation behaves differently from the one expected by
    this script.
    """
    suite = run_all_requested_reject_experiments(
        trials=1, basis_mode=basis_mode, seed=seed, prime_input=prime_input,
        show_plots=False, plot_dir=None
    )
    for res in _flatten_result_dicts(suite):
        print("[{} | {}] successes={} exceptions={} first_exception={}".format(
            res["level"], res["name"], int(res["successful_samples"]), int(res["exceptions"]), res.get("first_exception")
        ))
    return suite


# ============================================================
# Example usage
# ============================================================


def run_randomidealgivennorm_prime_only_all_bases(trials=DEFAULT_TRIALS, seed=0, checkpoint_dir=DEFAULT_CHECKPOINT_DIR):
    out = {}
    modes = ["natural", "hnf", "l2"]
    for basis_mode in modes:
        sub = {}
        for level in ["NIST-I", "NIST-III", "NIST-V"]:
            ctx = SQISignIdealLab(level)
            name = "RandomIdealGivenNorm(Dmix,true)"
            cp = os.path.join(checkpoint_dir, "basis_only_{}_{}_{}.pkl".format(level, name.replace("(", "_").replace(")", "_").replace(",", "_"), basis_mode))
            sub[level] = ctx.run_trials(
                name=name,
                sampler=lambda rng, ctx=ctx: ctx.sample_random_ideal_given_norm_prime(ctx.Dmix_spec, rng=rng),
                trials=trials,
                basis_mode=basis_mode,
                seed=_seed_plus(seed, [0, 101, 202][modes.index(basis_mode)]),
                checkpoint_path=cp,
            )
        out[basis_mode] = sub
    return out


def explain_counts_formula(res):
    constructed = Integer(res.get("constructed_samples", res.get("successful_samples", 0)))
    rejected = Integer(res.get("rejects", 0))
    accepted = Integer(res.get("accepted_samples", constructed - rejected))
    exceptions = Integer(res.get("exceptions", res.get("failures", 0)))
    attempts = Integer(res.get("attempts_total", res.get("trials_done", 0)))
    print("constructed =", constructed)
    print("accepted    =", accepted)
    print("rejected    =", rejected)
    print("exceptions  =", exceptions)
    print("attempts    =", attempts)
    print("check: accepted + rejected + exceptions =", accepted + rejected + exceptions)


def run_randomidealgivennorm_prime_basis_comparison(trials=DEFAULT_TRIALS, seed=0, checkpoint_dir=DEFAULT_CHECKPOINT_DIR):
    out = {}
    for basis_mode in ["natural", "hnf", "l2"]:
        out[basis_mode] = run_actual_fixed_input_suite(
            trials=trials, basis_mode=basis_mode, seed=_seed_plus(seed, [0, 101, 202][["natural", "hnf", "l2"].index(basis_mode)]),
            checkpoint_dir=os.path.join(checkpoint_dir, "basis_compare_" + basis_mode)
        )
    return out



def print_compact_suite_summary(results):
    for res in _flatten_result_dicts(results):
        print("[{} | {}] constructed={} accepted={} rejected={} exceptions={}".format(
            res["level"], res["name"],
            int(res.get("constructed_samples", res["successful_samples"])),
            int(res.get("accepted_samples", res["successful_samples"] - res["rejects"])),
            int(res["rejects"]),
            int(res["exceptions"]),
        ), flush=True)


def probe_prime_norm_sampler(level="NIST-I", trials=10, seed=0):
    """
    Quick smoke test for the current RandomIdealGivenPrimeNorm branch.
    Returns a short list of successful samples and their ambient generators.
    """
    ctx = SQISignIdealLab(level)
    rng = random.Random(_coerce_py_seed(seed))
    out = []
    for _ in range(int(trials)):
        s = ctx.sample_random_ideal_given_norm_prime(ctx.Dmix_spec, rng=rng)
        out.append({
            "norm": Integer(s["norm"]),
            "alpha": tuple(Integer(QQ(v)) for v in s["alpha"]),
            "variant": s.get("prime_sampler_variant", None),
        })
    return out


def probe_cornacchia_divzero_fix(level="NIST-I", trials=10, seed=0):
    # Backward-compatible alias for old notebooks.  The current sampler is no
    # longer Cornacchia-based.
    return probe_prime_norm_sampler(level=level, trials=trials, seed=seed)


_EXAMPLE_USAGE = r"""
1) Main fixed-input reject experiment with the new bound
       max_i nrd(alpha_i) > (64 p^2 / pi^4) * nrd(I)
   and histogram output for log10(max_i nrd(alpha_i) / nrd(I)):

   results = run_actual_fixed_input_suite(trials=10000, basis_mode="natural", seed=0)
   for level in results:
       for name in results[level]:
           print_result_block(results[level][name])
   plot_all_ratio_histograms(results)

2) Sign-like composite experiment for RandomIdealGivenNorm(N,false):

   results = run_default_sign_like_composite_suite(trials=10000, basis_mode="natural", seed=0)
   plot_all_ratio_histograms(results)

3) KernelDecomposedToIdeal using the official ref.zip precomp:

   results = run_kernel_decomposed_suite_official_ref(trials=10000, basis_mode="natural", seed=0)
   plot_all_ratio_histograms(results)

4) Run the whole *pre-multiplication ideal-generation* bundle (fixed 10,000 attempts per experiment by default).
   By default this EXCLUDES standalone RIDN(Dmix,true), but KEEPS
   REQP-after-RIDN(Dmix,true), RIDN(N,false), and KernelDecomposedToIdeal:

   results = run_all_pre_multiplication_ideal_generation_experiments(
       trials=10000,
       basis_mode="natural",
       seed=0,
       show_plots=False,
   )

5) The default fixed-input suite already uses the original SQIsign Dmix.

   results = run_all_requested_reject_experiments(
       trials=10000,
       basis_mode="natural",
       seed=0,
       include_spec=True,
   )

6) If you explicitly want the raw standalone Dmix RIDN row back in, opt in:

   results = run_actual_fixed_input_suite(
       trials=10000,
       basis_mode="natural",
       seed=0,
       include_raw_ridn=True,
       include_reqp_after_ridn=True,
   )

Notes:
- The returned ratio is max_i nrd(alpha_i) / nrd(I), not max_i nrd(alpha_i) / nrd(I)^2.
- The histogram uses log10 of that ratio by default.
- Checkpoints now use a new version tag and a new default directory, so stale
  checkpoints from the older hit-test code are ignored.
- Every experiment now runs a fixed number of attempts (default 10,000), not
  "until 10,000 successes".
- Inner random samplers use finite caps so that one bad attempt cannot loop forever.
- The default bundled runner is back to the pre-multiplication ideal-generation
  algorithms only, but with standalone RIDN(Dmix,true) excluded by default:
  REQP-after-RIDN(Dmix,true), RIDN(composite), and KernelDecomposedToIdeal.
  Deterministic ideal operations are no longer part of the default all-in-one benchmark.
- For every prime-norm path, sample_random_ideal_given_norm_prime uses the current
  RandomIdealGivenPrimeNorm sampler: g1,g2,g3 + Legendre/ModularSQRT, then beta
  sampled from [0,N)^4 with nonzero reduced norm modulo N.
"""

def show_example_usage():
    print(_EXAMPLE_USAGE)

