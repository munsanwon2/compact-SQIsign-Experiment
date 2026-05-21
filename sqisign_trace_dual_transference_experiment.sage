# -*- coding: utf-8 -*-
"""
SQIsign RandomIdealGivenPrimeNorm experiment using the TRANSFERENCE-THEOREM dual.
Version v19: trace-dual experiment plotting per-level sample minima and a fixed-slope
lower-envelope line.  No least-squares regression is used.

This file intentionally uses the trace-pairing lattice dual

    I^# = { y in B_{p,inf} : trd(x * conjugate(y)) in Z for every x in I }

and NOT the quaternion ideal inverse

    I^{-1} = conjugate(I) / nrd(I).

For an input ideal I of norm N, let alpha_1^# be a nonzero shortest vector of
I^# measured by reduced norm.  For each trial we compute

    y(I) = log2( N * nrd(alpha_1^#) ).

For each security level, the plotted point is min_I y(I) over all trials,
not the average.  This is the conservative statistic for deriving a sampled
upper bound on nrd(alpha_4)/nrd(I) via transference, since the bound is
proportional to 1/(N*nrd(alpha_1^#)).  The plot also draws the fixed-slope
line

    y = -1/2 * log2(p) + b,

where b is chosen as a lower envelope, so every plotted minimum point lies on
or above the line.  Optionally a small positive margin in bits is subtracted
from b, so the line is strictly below every point in the rendered plot.

The factor 1/N is the same normalization scale as before, since
log2(N*nrd(alpha_1^#)) = log2(nrd(alpha_1^#)/(1/N)).  I^# is a lattice, not an
ideal, so "nrd(I^#)" is not an ideal norm in the quaternion-ideal sense.

The bottom main(...) call is the only place where ordinary experiment settings
need to be changed.
"""

from sage.all import *

import builtins as _py_builtins
import csv
import os
import warnings
import gc
from io import BytesIO

# Matplotlib is used only through an Agg canvas.  Display in notebooks is done
# by rendering the figure to PNG bytes and passing those bytes to IPython.display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None

PY_FLOAT = _py_builtins.float
PY_INT = _py_builtins.int
PY_STR = _py_builtins.str

EXPERIMENT_VERSION = "v21-trace-dual-transference-min-lower-envelope-log2p-xaxis"


def configure_runtime_warnings(suppress_cypari_stack_warnings=True):
    """Optionally hide the noisy cypari2 PARI-stack RuntimeWarning.

    This only filters the warning message; it does not hide real Python
    exceptions such as PARI stack overflow errors.  For long notebook runs,
    restarting the kernel before execution is still recommended.
    """
    if suppress_cypari_stack_warnings:
        warnings.filterwarnings(
            "ignore",
            message=r".*cypari2 leaked .* bytes on the PARI stack.*",
            category=RuntimeWarning,
        )



# -----------------------------------------------------------------------------
# SQIsign parameters
# -----------------------------------------------------------------------------

def sqisign_parameter_sets():
    """Return the SQIsign NIST-I/III/V p values and security parameters."""
    return [
        {
            "level": "NIST-I",
            "lam": ZZ(128),
            "p": ZZ(5) * ZZ(2)**ZZ(248) - ZZ(1),
        },
        {
            "level": "NIST-III",
            "lam": ZZ(192),
            "p": ZZ(65) * ZZ(2)**ZZ(376) - ZZ(1),
        },
        {
            "level": "NIST-V",
            "lam": ZZ(256),
            "p": ZZ(27) * ZZ(2)**ZZ(500) - ZZ(1),
        },
    ]


# -----------------------------------------------------------------------------
# Quaternion arithmetic in B_{p,infty} = (-1, -p) over Q
# Coordinates are always in the basis (1, i, j, k=ij).
# -----------------------------------------------------------------------------

def qvec(a, b, c, d):
    return vector(QQ, [a, b, c, d])


def qconj(x):
    return vector(QQ, [x[0], -x[1], -x[2], -x[3]])


def qnrd(x, p):
    x = vector(QQ, x)
    return x[0]**2 + x[1]**2 + QQ(p) * (x[2]**2 + x[3]**2)


def qmul(x, y, p):
    """Quaternion product in B_{p,infty}, where i^2=-1, j^2=-p, k=ij."""
    x = vector(QQ, x)
    y = vector(QQ, y)
    a, b, c, d = x
    e, f, g, h = y
    p = QQ(p)
    return vector(QQ, [
        a*e - b*f - p*c*g - p*d*h,
        a*f + b*e + p*(c*h - d*g),
        a*g + c*e - b*h + d*f,
        a*h + d*e + b*g - c*f,
    ])


def qmod_N_integer_coords(x, N):
    """Reduce an integral-coordinate quaternion modulo N coefficientwise."""
    return vector(QQ, [ZZ(v) % N for v in x])


# -----------------------------------------------------------------------------
# Lattices and the SQIsign order O0
# -----------------------------------------------------------------------------

def O0_basis():
    """Rows are the SQIsign O0 basis: 1, i, (i+j)/2, (1+k)/2."""
    return matrix(QQ, [
        [1,   0,   0,   0],
        [0,   1,   0,   0],
        [0, QQ(1)/2, QQ(1)/2, 0],
        [QQ(1)/2, 0, 0, QQ(1)/2],
    ])


def nrd_diagonal_matrix(p):
    """Matrix D such that nrd(x) = x * D * x^T for row coordinates x."""
    return diagonal_matrix(QQ, [1, 1, QQ(p), QQ(p)])


def trace_pairing_matrix(p):
    """Matrix H such that <x,y> = trd(x*conj(y)) = x * H * y^T."""
    return diagonal_matrix(QQ, [2, 2, 2*QQ(p), 2*QQ(p)])


def common_denominator_matrix(M):
    d = ZZ(1)
    for x in M.list():
        d = lcm(d, QQ(x).denominator())
    return d


def zrow_span_basis_QQ(generators):
    """
    Return a QQ row basis for the Z-span of the given rational row generators.
    """
    G = matrix(QQ, generators)
    den = common_denominator_matrix(G)
    A = matrix(ZZ, [[ZZ(den * G[i, j]) for j in range(G.ncols())]
                    for i in range(G.nrows())])
    rowmod = A.row_module(ZZ)
    B_int = rowmod.basis_matrix()
    if B_int.nrows() != G.ncols():
        raise ValueError("The generated lattice is not full rank.")
    return matrix(QQ, B_int) / QQ(den)


def left_ideal_basis_O0_generated_by_g_and_N(g, N, p):
    """Return a Z-basis for O0*g + O0*N as rows in the (1,i,j,k) basis."""
    O = O0_basis()
    gens = []
    for r in O.rows():
        gens.append(qmul(r, g, p))
    for r in O.rows():
        gens.append(QQ(N) * r)
    return zrow_span_basis_QQ(gens)


def trace_lattice_dual_basis(B, p):
    """
    Embedded trace-pairing lattice dual of the row-lattice generated by B.

    If I has row basis B and H is the trace pairing matrix, the returned rows B#
    satisfy

        B * H * (B#)^T = Id.

    Therefore B# generates {y : <x,y> in Z for all x in I}.
    """
    B = matrix(QQ, B)
    H = trace_pairing_matrix(p)
    Bdual = (B * H).inverse().transpose()
    # Exact sanity check: the returned basis is dual to the input basis.
    if B * H * Bdual.transpose() != identity_matrix(QQ, B.nrows()):
        raise ArithmeticError("trace dual basis check failed")
    return Bdual


def inverse_ideal_lattice_basis(B, N):
    """
    Quaternion ideal inverse lattice conjugate(I)/N.
    This is NOT the transference dual.  It is kept only for optional diagnostics.
    """
    return matrix(QQ, [qconj(B.row(i)) / QQ(N) for i in range(B.nrows())])


# -----------------------------------------------------------------------------
# RandomIdealGivenPrimeNorm(N), specialized to the pseudocode in the prompt
# -----------------------------------------------------------------------------

def legendre_is_one(a, N):
    a = ZZ(a) % ZZ(N)
    if a == 0:
        return False
    return ZZ(Mod(a, N)**((N - 1)//2)) == 1


def sqrt_mod_prime_residue(a, N):
    roots = Mod(ZZ(a) % ZZ(N), ZZ(N)).sqrt(all=True)
    if len(roots) == 0:
        raise ValueError("input is not a quadratic residue modulo N")
    return ZZ(roots[0])


def random_ideal_given_prime_norm_trace_experiment(p, N):
    """
    Implements the prompt's RandomIdealGivenPrimeNorm(N) in B_{p,infty}.

    Output:
        I_basis: row basis of I = O0<g,N>
        g:       quaternion generator used modulo N
    """
    p = ZZ(p)
    N = ZZ(N)

    while True:
        g1 = ZZ.random_element(N)
        g2 = ZZ.random_element(N)
        g3 = ZZ.random_element(N)
        gamma0 = qvec(0, g1, g2, g3)
        target = ZZ(-qnrd(gamma0, p)) % N
        if legendre_is_one(target, N):
            r = sqrt_mod_prime_residue(target, N)
            gamma = qvec(r, g1, g2, g3)
            break

    while True:
        a = ZZ.random_element(N)
        b = ZZ.random_element(N)
        c = ZZ.random_element(N)
        d = ZZ.random_element(N)
        beta = qvec(a, b, c, d)
        if ZZ(qnrd(beta, p)) % N != 0:
            break

    g = qmod_N_integer_coords(qmul(gamma, beta, p), N)
    I_basis = left_ideal_basis_O0_generated_by_g_and_N(g, N, p)
    return I_basis, g


# -----------------------------------------------------------------------------
# Exact 4-dimensional shortest-vector search for the reduced norm quadratic form
# -----------------------------------------------------------------------------

def gram_nrd(B, p):
    B = matrix(QQ, B)
    D = nrd_diagonal_matrix(p)
    Q = B * D * B.transpose()
    # Symmetrize defensively.
    return (Q + Q.transpose()) / QQ(2)


def ldl_decomposition_QQ(Q):
    """Exact LDL^T decomposition Q = L * diag(D) * L^T, L lower unit."""
    Q = matrix(QQ, Q)
    n = Q.nrows()
    L = identity_matrix(QQ, n)
    D = [QQ(0)] * n
    for i in range(n):
        s = QQ(0)
        for k in range(i):
            s += L[i, k]**2 * D[k]
        D[i] = Q[i, i] - s
        if D[i] <= 0:
            raise ArithmeticError("Gram matrix is not positive definite")
        for j in range(i + 1, n):
            s = QQ(0)
            for k in range(i):
                s += L[j, k] * L[i, k] * D[k]
            L[j, i] = (Q[j, i] - s) / D[i]
    return L, D


def _floor_div_int(a, b):
    a = ZZ(a)
    b = ZZ(b)
    if b <= 0:
        raise ValueError("b must be positive")
    return a // b


def _ceil_div_int(a, b):
    a = ZZ(a)
    b = ZZ(b)
    if b <= 0:
        raise ValueError("b must be positive")
    return -((-a) // b)


def integer_interval_for_square_bound(mu, R):
    """
    Exact interval for integers x satisfying (x + mu)^2 <= R,
    where mu and R are rational and R >= 0.
    """
    mu = QQ(mu)
    R = QQ(R)
    if R < 0:
        return (ZZ(1), ZZ(0))  # empty interval

    a = ZZ(mu.numerator())
    b = ZZ(mu.denominator())  # positive in Sage's normalized QQ
    num = ZZ(R.numerator())
    den = ZZ(R.denominator())

    # Need (b*x + a)^2 <= b^2 * num / den.
    # Since b*x+a is integral, this is equivalent to
    # (b*x+a)^2 <= floor(b^2*num/den).
    M2 = (b*b*num) // den
    M = ZZ(M2).sqrtrem()[0]

    lo = _ceil_div_int(-M - a, b)
    hi = _floor_div_int(M - a, b)
    return lo, hi


def lll_precondition_basis_for_gram(B, Q):
    """
    Apply Sage's LLL_gram to a denominator-cleared Gram matrix.
    Returns a unimodularly equivalent row basis and its exact Gram matrix.
    """
    B = matrix(QQ, B)
    Q = matrix(QQ, Q)

    den = common_denominator_matrix(Q)
    Qint = matrix(ZZ, [[ZZ(den * Q[i, j]) for j in range(Q.ncols())]
                       for i in range(Q.nrows())])

    try:
        U = Qint.LLL_gram()
        # Some Sage versions/options may return a tuple; the transformation
        # matrix is the first component in that case.
        if isinstance(U, tuple):
            U = U[0]
    except Exception:
        U = identity_matrix(ZZ, Q.nrows())

    candidates = []
    for T in [matrix(QQ, U.transpose()), matrix(QQ, U), identity_matrix(QQ, Q.nrows())]:
        Bc = T * B
        Qc = T * Q * T.transpose()
        # Score by maximum diagonal first, then sum of diagonals.
        maxdiag = max([Qc[i, i] for i in range(Qc.nrows())])
        sumdiag = sum([Qc[i, i] for i in range(Qc.nrows())], QQ(0))
        candidates.append((maxdiag, sumdiag, Bc, Qc))

    candidates.sort(key=lambda z: (z[0], z[1]))
    return candidates[0][2], candidates[0][3]


def shortest_nrd_in_lattice(B, p):
    """
    Exact shortest nonzero vector for the reduced norm on the lattice with row basis B.

    Returns:
        best_norm: exact QQ reduced norm
        best_vec:  exact QQ coordinates in the (1,i,j,k) basis
        best_coeffs_reduced_basis: coefficient vector relative to the internally reduced basis
    """
    B = matrix(QQ, B)
    Q = gram_nrd(B, p)
    Bred, Qred = lll_precondition_basis_for_gram(B, Q)

    n = Qred.nrows()
    # Initial bound from shortest basis vector in the reduced/preconditioned basis.
    diag = [Qred[i, i] for i in range(n)]
    best_norm = min(diag)
    best_index = diag.index(best_norm)
    best_coeffs = [ZZ(0)] * n
    best_coeffs[best_index] = ZZ(1)

    L, D = ldl_decomposition_QQ(Qred)
    x = [ZZ(0)] * n

    def recurse(i, partial):
        nonlocal best_norm, best_coeffs
        if i < 0:
            if all(xi == 0 for xi in x):
                return
            if partial < best_norm:
                best_norm = partial
                best_coeffs = [ZZ(xi) for xi in x]
            return

        rem = best_norm - partial
        if rem <= 0:
            return

        mu = QQ(0)
        for j in range(i + 1, n):
            mu += L[j, i] * x[j]

        R = rem / D[i]
        lo, hi = integer_interval_for_square_bound(mu, R)
        if lo > hi:
            return

        center = -mu
        # Enumerate values nearest to the center first to improve pruning quickly.
        vals = list(range(PY_INT(lo), PY_INT(hi) + 1))
        vals.sort(key=lambda z: abs(QQ(z) - center))

        for xi in vals:
            xi = ZZ(xi)
            term_arg = QQ(xi) + mu
            new_partial = partial + D[i] * term_arg**2
            if new_partial < best_norm:
                x[i] = xi
                recurse(i - 1, new_partial)
        x[i] = ZZ(0)

    recurse(n - 1, QQ(0))

    coeff_row = matrix(QQ, [best_coeffs])
    best_vec = (coeff_row * Bred).row(0)
    check_norm = qnrd(best_vec, p)
    if check_norm != best_norm:
        raise ArithmeticError("shortest-vector norm check failed")
    return best_norm, best_vec, vector(ZZ, best_coeffs)


def successive_minima_4_nrd(B, p):
    """
    Optional diagnostic: exact-ish enumeration of the first time the span reaches
    rank 1,2,3,4 under increasing norm.  This is not used for the plotted value.

    Dimension is 4, but this can be expensive for many trials.  Keep disabled
    unless you specifically want to inspect lambda_4(I).
    """
    B = matrix(QQ, B)
    Q = gram_nrd(B, p)
    Bred, Qred = lll_precondition_basis_for_gram(B, Q)
    n = 4

    # Bound by a small multiple of the largest reduced-basis norm.
    # This is enough in usual SQIsign-sized experiments after LLL; if it fails,
    # the function will report failure rather than affect the main experiment.
    bound = max([Qred[i, i] for i in range(n)])
    L, D = ldl_decomposition_QQ(Qred)
    x = [ZZ(0)] * n
    found = []

    def add_vector(coeffs, norm):
        nonlocal found
        if all(c == 0 for c in coeffs):
            return
        found.append((QQ(norm), vector(ZZ, coeffs)))

    def recurse(i, partial):
        if i < 0:
            add_vector([ZZ(xi) for xi in x], partial)
            return
        rem = bound - partial
        if rem < 0:
            return
        mu = QQ(0)
        for j in range(i + 1, n):
            mu += L[j, i] * x[j]
        R = rem / D[i]
        lo, hi = integer_interval_for_square_bound(mu, R)
        for xi in range(PY_INT(lo), PY_INT(hi) + 1):
            xi = ZZ(xi)
            term_arg = QQ(xi) + mu
            new_partial = partial + D[i] * term_arg**2
            if new_partial <= bound:
                x[i] = xi
                recurse(i - 1, new_partial)
        x[i] = ZZ(0)

    recurse(n - 1, QQ(0))
    found.sort(key=lambda z: z[0])

    span_rows = []
    mins = []
    for norm, coeffs in found:
        span_rows.append(list(coeffs))
        r = matrix(ZZ, span_rows).rank()
        if r > len(mins):
            mins.append(norm)
            if r == 4:
                return mins

    raise RuntimeError("lambda_4 diagnostic did not find rank 4 within the bound")


# -----------------------------------------------------------------------------
# Numeric formatting and plotting
# -----------------------------------------------------------------------------

def log2_QQ(q, prec=200):
    q = QQ(q)
    if q <= 0:
        raise ValueError("log2 input must be positive")
    R = RealField(prec)
    return R(q).log() / R(2).log()


def qq_to_csv_string(q):
    q = QQ(q)
    if q.denominator() == 1:
        return PY_STR(q.numerator())
    return PY_STR(q.numerator()) + "/" + PY_STR(q.denominator())


def vector_to_csv_string(v):
    return "[" + ", ".join(qq_to_csv_string(QQ(x)) for x in v) + "]"


def ensure_pdf_name(path):
    if not path.lower().endswith(".pdf"):
        path += ".pdf"
    return path


def render_figure_to_png_bytes(fig, dpi=220):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=PY_INT(dpi), bbox_inches="tight")
    return buf.getvalue()


def save_png_bytes_as_pdf(png_bytes, out_pdf, dpi=220):
    if PILImage is None:
        raise RuntimeError("Pillow/PIL is required for PDF output in this Sage-safe rasterized path.")
    out_pdf = ensure_pdf_name(out_pdf)
    img = PILImage.open(BytesIO(png_bytes)).convert("RGB")
    img.save(out_pdf, "PDF", resolution=PY_FLOAT(dpi))


def display_png_bytes(png_bytes):
    """Display in Jupyter/Sage notebooks; fall back to PIL's viewer when possible."""
    try:
        from IPython.display import Image, display
        display(Image(data=png_bytes))
        return True
    except Exception:
        pass
    try:
        if PILImage is not None:
            img = PILImage.open(BytesIO(png_bytes))
            img.show()
            return True
    except Exception:
        pass
    return False


def real_from_csv_string(s, prec=200):
    """
    Convert a CSV numeric string to a Sage RealField element.

    This intentionally accepts decimal strings such as
    "250.321928094...".  Do NOT send those through QQ: Sage's QQ
    parser expects exact rational syntax, not decimal real syntax.
    """
    R = RealField(prec)
    return R(PY_STR(s))


def lower_envelope_line(grouped, slope=QQ(-1)/QQ(2), margin_bits=0, prec=200):
    """
    Return the fixed-slope lower-envelope line for the plotted minimum points.

    The plotted points are

        (x_l, y_l) = (log2(p_l), min_t log2(N*nrd(alpha_{1,t}^#))).

    For a prescribed slope, here slope = -1/2, the largest intercept b such
    that y_l >= slope*x_l + b for every level l is

        b_tight = min_l (y_l - slope*x_l).

    We use b = b_tight - margin_bits.  With margin_bits=0 the line touches at
    least one point; with margin_bits>0 it is strictly below every plotted
    point by at least margin_bits bits.

    This is not a regression and does not average the three minimum points.
    """
    if len(grouped) == 0:
        return None
    R = RealField(prec)
    slope_R = R(slope)
    margin_R = R(PY_STR(margin_bits))
    xs = [real_from_csv_string(r["log2p_real_string"], prec) for r in grouped]
    ys = [real_from_csv_string(r["plot_y_real_string"], prec) for r in grouped]
    intercept_candidates = [y - slope_R*x for x, y in zip(xs, ys)]
    tight_intercept = min(intercept_candidates)
    intercept = tight_intercept - margin_R
    gaps = [y - (slope_R*x + intercept) for x, y in zip(xs, ys)]
    return {
        "slope": slope_R,
        "tight_intercept": tight_intercept,
        "intercept": intercept,
        "margin_bits": margin_R,
        "min_gap_bits": min(gaps),
        "max_gap_bits": max(gaps),
    }


def make_plot(
    grouped,
    out_pdf,
    log_x_axis=True,
    save_pdf=True,
    show_plot=True,
    lower_envelope_margin_bits=0.02,
):
    out_pdf = ensure_pdf_name(out_pdf)
    grouped = sorted(grouped, key=lambda r: ZZ(r["p"]))

    # Plot on the x-axis using log_2(p) directly, not p itself.
    x_plot = [PY_FLOAT(real_from_csv_string(r["log2p_real_string"], 200)) for r in grouped]
    y_plot = [PY_FLOAT(RealField(100)(r["plot_y_real_string"])) for r in grouped]
    labels = [r["level"] for r in grouped]

    lower = lower_envelope_line(
        grouped,
        slope=QQ(-1) / QQ(2),
        margin_bits=lower_envelope_margin_bits,
        prec=200,
    )

    fig, ax = plt.subplots(figsize=(PY_FLOAT(8.0), PY_FLOAT(5.0)))

    # Plot only the per-level sample minima as points.  Do not connect them by
    # line segments, since those segments have no lower-bound meaning.
    ax.plot(
        x_plot,
        y_plot,
        marker="o",
        linestyle="None",
        markersize=PY_FLOAT(6.0),
        label="sample minimum per level",
    )
    for x, y, lab in zip(x_plot, y_plot, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5))

    if lower is not None:
        R = RealField(200)
        log2ps = [real_from_csv_string(r["log2p_real_string"], 200) for r in grouped]
        lx_min = min(log2ps)
        lx_max = max(log2ps)
        if lx_min == lx_max:
            lx_min = lx_min - R(1)
            lx_max = lx_max + R(1)
        slope = lower["slope"]
        intercept = lower["intercept"]

        x_line = []
        y_line = []
        steps = 120
        for i in range(steps + 1):
            t = R(i) / R(steps)
            lx = lx_min + t * (lx_max - lx_min)
            x_line.append(PY_FLOAT(lx))
            y_line.append(PY_FLOAT(slope * lx + intercept))

        label = (
            "lower envelope: y = -0.5 log2(p) %+0.3f"
            % PY_FLOAT(intercept)
        )
        ax.plot(
            x_line,
            y_line,
            linewidth=PY_FLOAT(1.5),
            label=label,
        )

    # The x-axis is always shown as log_2(p) directly.  The legacy
    # log_x_axis flag is kept only for backward compatibility.
    ax.set_xlabel(r"$\log_2 p$")

    ax.set_ylabel(r"$\log_2(N\cdot\mathrm{nrd}(\alpha^{\sharp}_1))$")
    ax.set_title("SQIsign trace-dual shortest reduced norm: sample minima and lower envelope")
    ax.grid(True)
    ax.legend(frameon=False)

    try:
        fig.tight_layout()
    except Exception as e:
        # Layout failures are plotting-only issues.  Do not discard the
        # completed experiment; use a fixed layout instead.
        print("[plot warning] tight_layout failed; using fixed subplot margins: %s" % e)
        fig.subplots_adjust(
            left=PY_FLOAT(0.13),
            right=PY_FLOAT(0.98),
            bottom=PY_FLOAT(0.14),
            top=PY_FLOAT(0.90),
        )

    png_bytes = render_figure_to_png_bytes(fig, dpi=220)
    if save_pdf:
        save_png_bytes_as_pdf(png_bytes, out_pdf, dpi=220)
    if show_plot:
        ok = display_png_bytes(png_bytes)
        if not ok:
            print("[plot] Could not open an interactive display; PDF was saved instead.")
    plt.close(fig)
    return lower



def print_lower_envelope_summary(lower):
    if lower is None:
        return
    slope = lower["slope"]
    b = lower["intercept"]
    b_tight = lower["tight_intercept"]
    margin = lower["margin_bits"]
    min_gap = lower["min_gap_bits"]

    # The plotted lower-envelope line is
    #     y = -1/2 log2(p) + b,
    # i.e.
    #     N*nrd(alpha_1^#) >= 2^b / sqrt(p)
    # for every plotted sample minimum, hence for every trial in the CSV.
    # Transference gives
    #     nrd(alpha_4)/N <= 4 * 2^{-b} * sqrt(p) = 2^(2-b) * sqrt(p).
    log2_upper_constant = RealField(200)(2) - b

    print("\nFixed-slope lower envelope, not a regression:")
    print("  y = %.10f * log2(p) %+ .10f" % (PY_FLOAT(slope), PY_FLOAT(b)))
    print("  tight intercept before visual margin: %.10f" % PY_FLOAT(b_tight))
    print("  margin subtracted: %.10f bits" % PY_FLOAT(margin))
    print("  minimum point-to-line gap: %.10f bits" % PY_FLOAT(min_gap))
    print("  sampled lower bound: N*nrd(alpha_1^#) >= 2^b / sqrt(p)")
    print("  sampled transference upper bound: nrd(alpha_4)/nrd(I) <= 2^(2-b)*sqrt(p)")
    print("  log2 upper-bound constant 2-b = %.10f" % PY_FLOAT(log2_upper_constant))


# -----------------------------------------------------------------------------
# Experiment driver
# -----------------------------------------------------------------------------

def run_self_tests():
    """Small exact checks that the trace-dual convention is the one intended."""
    p = ZZ(5) * ZZ(2)**ZZ(248) - ZZ(1)
    O = O0_basis()
    Odual = trace_lattice_dual_basis(O, p)
    H = trace_pairing_matrix(p)
    assert O * H * Odual.transpose() == identity_matrix(QQ, 4)

    # O0^# contains j/p and k/p, so min nrd(alpha^#) should be 1/p.
    lam1, vec, _ = shortest_nrd_in_lattice(Odual, p)
    if lam1 != QQ(1) / QQ(p):
        raise ArithmeticError("O0 trace-dual self-test failed: expected lambda_1 = 1/p")


def run_one(level_data, trial_index, compute_inverse_diagnostic=False, compute_lambda4_diagnostic=False):
    level = level_data["level"]
    lam = ZZ(level_data["lam"])
    p = ZZ(level_data["p"])
    N = next_prime(ZZ(2)**(ZZ(4)*lam))

    I_basis, g = random_ideal_given_prime_norm_trace_experiment(p, N)
    I_trace_dual_basis = trace_lattice_dual_basis(I_basis, p)

    lambda1_trace_dual_nrd, shortest_trace_dual_vec, coeffs = shortest_nrd_in_lattice(I_trace_dual_basis, p)

    normalizer = QQ(1) / QQ(N)
    ratio = lambda1_trace_dual_nrd / normalizer  # = N * nrd(alpha_1^#)
    y = log2_QQ(ratio)

    row = {
        "level": level,
        "lambda": PY_STR(lam),
        "trial": PY_STR(trial_index),
        "p": PY_STR(p),
        "log2p": PY_STR(log2_QQ(QQ(p))),
        "N": PY_STR(N),
        "log2N": PY_STR(log2_QQ(QQ(N))),
        "normalizer_1_over_N": qq_to_csv_string(normalizer),
        "lambda1_trace_dual_nrd": qq_to_csv_string(lambda1_trace_dual_nrd),
        "ratio_lambda1_trace_dual_over_1_over_N": qq_to_csv_string(ratio),
        "y_log2_ratio": PY_STR(y),
        "shortest_trace_dual_vec_1_i_j_k": vector_to_csv_string(shortest_trace_dual_vec),
        "shortest_trace_dual_coeffs_internal_reduced_basis": vector_to_csv_string(coeffs),
        "generator_g_mod_N_1_i_j_k": vector_to_csv_string(g),
    }

    if compute_inverse_diagnostic:
        I_inv_basis = inverse_ideal_lattice_basis(I_basis, N)
        lambda1_inv_nrd, shortest_inv_vec, _ = shortest_nrd_in_lattice(I_inv_basis, p)
        ratio_inv = lambda1_inv_nrd / normalizer
        row["lambda1_inverse_lattice_nrd_DIAGNOSTIC"] = qq_to_csv_string(lambda1_inv_nrd)
        row["y_inverse_lattice_DIAGNOSTIC"] = PY_STR(log2_QQ(ratio_inv))
        row["shortest_inverse_vec_DIAGNOSTIC"] = vector_to_csv_string(shortest_inv_vec)

    if compute_lambda4_diagnostic:
        try:
            mins = successive_minima_4_nrd(I_basis, p)
            row["lambda1_I_nrd_DIAGNOSTIC"] = qq_to_csv_string(mins[0])
            row["lambda2_I_nrd_DIAGNOSTIC"] = qq_to_csv_string(mins[1])
            row["lambda3_I_nrd_DIAGNOSTIC"] = qq_to_csv_string(mins[2])
            row["lambda4_I_nrd_DIAGNOSTIC"] = qq_to_csv_string(mins[3])
            row["lambda4_I_over_N_DIAGNOSTIC"] = qq_to_csv_string(mins[3] / QQ(N))
            row["transference_product_lambda1_trace_dual_lambda4_I_DIAGNOSTIC"] = qq_to_csv_string(lambda1_trace_dual_nrd * mins[3])
        except Exception as e:
            row["lambda4_diagnostic_error"] = PY_STR(e)

    return row


def write_csv(rows, out_csv):
    if len(rows) == 0:
        raise ValueError("No rows to write")
    # Preserve insertion order from the first row, then append any extra diagnostic keys.
    fields = list(rows[0].keys())
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def read_csv_rows(out_csv):
    with open(out_csv, "r", newline="") as f:
        return list(csv.DictReader(f))


def group_rows_by_level(rows):
    """
    Group raw trial rows by security level.

    The plotted statistic is the minimum y over the trials at that level:

        min_y = min_I log2(N*nrd(alpha_1^#)).

    This is the conservative statistic for a sampled transference upper bound,
    because

        nrd(alpha_4)/N <= 4/(N*nrd(alpha_1^#)).

    Therefore the worst sampled upper bound is obtained from the smallest
    denominator N*nrd(alpha_1^#), equivalently from min_y.
    """
    groups = {}
    for r in rows:
        level = r["level"]
        groups.setdefault(level, []).append(r)

    grouped = []
    for level, rs in groups.items():
        p = ZZ(rs[0]["p"])
        log2p = log2_QQ(QQ(p))

        ys = [RealField(200)(r["y_log2_ratio"]) for r in rs]
        ratios = [QQ(r["ratio_lambda1_trace_dual_over_1_over_N"]) for r in rs]

        mean_y = sum(ys) / RealField(200)(len(ys))
        min_y = min(ys)
        max_y = max(ys)
        min_ratio = min(ratios)

        # Consistency: since log2 is monotone, min_y equals log2(min_ratio).
        # Use the exact min_ratio to define the plotted y value.
        min_y_from_exact_ratio = log2_QQ(min_ratio)

        if len(ys) >= 2:
            mean = mean_y
            var = sum([(yy - mean)**2 for yy in ys]) / RealField(200)(len(ys) - 1)
            sd = var.sqrt()
        else:
            sd = RealField(200)(0)

        transference_upper_bound = QQ(4) / min_ratio
        log2_transference_upper_bound = log2_QQ(transference_upper_bound)

        grouped.append({
            "level": level,
            "p": PY_STR(p),
            "log2p_real_string": PY_STR(log2p),
            "trials": PY_STR(len(rs)),

            # This is what is plotted and used to define the lower envelope.
            "plot_statistic": "min_y",
            "plot_y_real_string": PY_STR(min_y_from_exact_ratio),
            "min_y_real_string": PY_STR(min_y_from_exact_ratio),
            "min_ratio_exact_string": qq_to_csv_string(min_ratio),

            # Diagnostics only; these are not plotted.
            "mean_y_real_string": PY_STR(mean_y),
            "sd_y_real_string": PY_STR(sd),
            "max_y_real_string": PY_STR(max_y),

            # Sampled worst-case upper bound from transference:
            # nrd(alpha_4)/N <= 4/(N*nrd(alpha_1^#)).
            "sampled_transference_upper_bound_lambda4_over_N_exact": qq_to_csv_string(transference_upper_bound),
            "sampled_transference_upper_bound_lambda4_over_N_log2": PY_STR(log2_transference_upper_bound),
        })

    level_order = {"NIST-I": 0, "NIST-III": 1, "NIST-V": 2}
    grouped.sort(key=lambda r: level_order.get(r["level"], 99))
    return grouped

def print_progress(level, done, total, rows_this_level):
    ys = [RealField(100)(r["y_log2_ratio"]) for r in rows_this_level]
    mean_y = sum(ys) / RealField(100)(len(ys))
    min_y = min(ys)
    if len(ys) >= 2:
        var = sum([(yy - mean_y)**2 for yy in ys]) / RealField(100)(len(ys) - 1)
        sd_y = var.sqrt()
    else:
        sd_y = RealField(100)(0)
    last_y = ys[-1]
    print("[progress] %s: %s/%s trials | last y = %.8f | min y = %.8f | mean y = %.8f | sd y = %.8f" % (
        level, done, total, PY_FLOAT(last_y), PY_FLOAT(min_y), PY_FLOAT(mean_y), PY_FLOAT(sd_y)
    ))

def main(
    trials_I=1,
    trials_III=1,
    trials_V=1,
    progress_every=1000,
    print_progress_at_level_end=False,
    print_each_trial=False,
    random_seed=20260424,
    out_csv="sqisign_trace_dual_transference_results.csv",
    out_pdf="sqisign_trace_dual_transference_plot.pdf",
    log_x_axis=True,
    save_pdf=True,
    show_plot=True,
    plot_only_from_csv=False,
    run_tests=True,
    compute_inverse_diagnostic=False,
    compute_lambda4_diagnostic=False,
    suppress_cypari_stack_warnings=True,
    gc_every=0,
    lower_envelope_margin_bits=0.02,
):
    """
    Main experiment settings.

    Change trials_I, trials_III, trials_V here or at the bottom main(...) call.
    progress_every=K prints compact running results after every K completed trials per level.
    progress_every=0 disables periodic progress summaries.
    print_progress_at_level_end=True also prints one final summary at the end of each level.
    print_each_trial=True prints every individual trial start; keep it False for large runs.
    suppress_cypari_stack_warnings=True hides Sage/cypari2 PARI-stack RuntimeWarnings.
    gc_every=K runs Python garbage collection every K completed trials per level; 0 disables it.
    lower_envelope_margin_bits=epsilon subtracts epsilon bits from the tight
    lower-envelope intercept, making every plotted point strictly above the line.
    """
    configure_runtime_warnings(suppress_cypari_stack_warnings)
    print("Running %s" % EXPERIMENT_VERSION)
    out_pdf = ensure_pdf_name(out_pdf)

    if plot_only_from_csv:
        rows = read_csv_rows(out_csv)
        grouped = group_rows_by_level(rows)
        lower = make_plot(
            grouped,
            out_pdf,
            log_x_axis=log_x_axis,
            save_pdf=save_pdf,
            show_plot=show_plot,
            lower_envelope_margin_bits=lower_envelope_margin_bits,
        )
        print("Loaded CSV and regenerated plot.")
        print_lower_envelope_summary(lower)
        return

    if random_seed is not None:
        set_random_seed(ZZ(random_seed))

    if run_tests:
        print("Running exact trace-dual self-tests...")
        run_self_tests()
        print("  self-tests passed")

    trials_by_level = {
        "NIST-I": PY_INT(trials_I),
        "NIST-III": PY_INT(trials_III),
        "NIST-V": PY_INT(trials_V),
    }

    all_rows = []
    for level_data in sqisign_parameter_sets():
        level = level_data["level"]
        total = trials_by_level[level]
        rows_this_level = []
        if total <= 0:
            print("Skipping %s because trials=0" % level)
            continue
        for t in range(total):
            if print_each_trial:
                print("Running %s, lambda=%s, trial=%s" % (level, level_data["lam"], t))
            row = run_one(
                level_data,
                t,
                compute_inverse_diagnostic=compute_inverse_diagnostic,
                compute_lambda4_diagnostic=compute_lambda4_diagnostic,
            )
            all_rows.append(row)
            rows_this_level.append(row)

            if gc_every and gc_every > 0 and ((t + 1) % gc_every == 0):
                gc.collect()

            if progress_every and progress_every > 0:
                should_print_periodic = ((t + 1) % progress_every == 0)
                should_print_final = (print_progress_at_level_end and (t + 1 == total))
                if should_print_periodic or should_print_final:
                    print_progress(level, t + 1, total, rows_this_level)

    if len(all_rows) == 0:
        raise ValueError("No trials were run.  Set at least one of trials_I/trials_III/trials_V positive.")

    write_csv(all_rows, out_csv)
    grouped = group_rows_by_level(all_rows)

    print("\nGrouped minima for plotted trace-dual quantity:")
    for r in grouped:
        print("  %s: trials=%s, log2(p)=%.8f, min y=%.8f, mean y=%.8f, sd y=%.8f, log2 upper bound=%.8f" % (
            r["level"],
            r["trials"],
            PY_FLOAT(RealField(100)(r["log2p_real_string"])),
            PY_FLOAT(RealField(100)(r["min_y_real_string"])),
            PY_FLOAT(RealField(100)(r["mean_y_real_string"])),
            PY_FLOAT(RealField(100)(r["sd_y_real_string"])),
            PY_FLOAT(RealField(100)(r["sampled_transference_upper_bound_lambda4_over_N_log2"])),
        ))

    lower = make_plot(
        grouped,
        out_pdf,
        log_x_axis=log_x_axis,
        save_pdf=save_pdf,
        show_plot=show_plot,
        lower_envelope_margin_bits=lower_envelope_margin_bits,
    )
    print_lower_envelope_summary(lower)

    print("\nWrote:")
    print("  %s" % out_csv)
    if save_pdf:
        print("  %s" % out_pdf)


# -----------------------------------------------------------------------------
# Edit experiment settings here.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main(
        # Number of RandomIdealGivenPrimeNorm samples per SQIsign security level.
        trials_I=10000,
        trials_III=10000,
        trials_V=10000,

        # Print compact running results every K completed trials per level.
        # Set to 0 to disable periodic summaries.
        progress_every=1000,

        # If True, also print one summary at the end of each security level.
        # Keep False if progress_every=1000 should print only at exact multiples of 1000.
        print_progress_at_level_end=False,

        # If True, print every trial start.  Keep False for large runs.
        print_each_trial=False,

        # Set to None for non-deterministic Sage randomness.
        random_seed=20260424,

        out_csv="sqisign_trace_dual_transference_results.csv",
        out_pdf="sqisign_trace_dual_transference_plot.pdf",

        # x-data are actual p values.  With log_x_axis=True they are displayed
        # on a base-2 logarithmic x-axis, which is the right visual scale for
        # checking slope near -1/2.
        log_x_axis=True,

        # Save a PDF and also display the plot in a notebook when possible.
        save_pdf=True,
        show_plot=True,

        # If True, do not run trials; just load out_csv and regenerate the plot.
        plot_only_from_csv=False,

        # Exact sanity check that O0^# has shortest reduced norm 1/p.
        run_tests=True,

        # Optional diagnostics.  They are not used for the plotted value.
        compute_inverse_diagnostic=False,
        compute_lambda4_diagnostic=False,

        # Hide noisy notebook warnings of the form
        # "RuntimeWarning: cypari2 leaked ... bytes on the PARI stack".
        suppress_cypari_stack_warnings=True,

        # Optional Python garbage collection cadence.  This does not replace a
        # kernel restart for very long notebook runs, but can keep Python-side
        # objects tidy.  Set e.g. 1000 for long experiments.
        gc_every=0,

        # Visual/conservative margin in bits for the fixed-slope lower-envelope line.
        # With 0 the line touches one plotted point; with 0.02 every point is
        # strictly above the line by at least 0.02 bits.
        lower_envelope_margin_bits=0.02,
    )
