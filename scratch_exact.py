import numpy as np
from fractions import Fraction
from numpy.polynomial import chebyshev as ncheb
import numpy.polynomial.polynomial as Pmod

import pennylane as qp
from pennylane.templates.subroutines.qsvt import _complementary_poly, _compute_qsp_angle


def build_poly_exact(a_frac, n):
    """Exact rational construction of V(y) = 1 - T_n(phi(y))/T_n(phi(0)), returns double coeffs."""
    y1 = a_frac * a_frac
    al = Fraction(2, 1) / (1 - y1)
    be = -(1 + y1) / (1 - y1)
    # T_n(z) monomial coeffs (integers) from numpy then to Fraction
    tn = ncheb.cheb2poly([0] * n + [1])
    tnf = [Fraction(int(round(t))) if abs(t - round(t)) < 1e-9 else Fraction(t).limit_denominator(10**12) for t in tn]
    # exact composition p(al*y + be)
    c = [Fraction(0)] * (n + 1)
    powk = [Fraction(1)]
    for k in range(n + 1):
        if k > 0:
            # multiply poly powk by (be + al y)
            new = [Fraction(0)] * (len(powk) + 1)
            for i, coef in enumerate(powk):
                new[i] += coef * be
                new[i + 1] += coef * al
            powk = new
        for i, coef in enumerate(powk):
            c[i] += tnf[k] * coef
    # T_n(phi(0)) = T_n(-zeta): evaluate composition at y=0 == c[0]
    Tphi0 = c[0]
    v = [1 - ci / Tphi0 if i == 0 else -ci / Tphi0 for i, ci in enumerate(c)]
    v[0] = Fraction(0)
    q = [Fraction(0)] * (2 * n)
    for j in range(n):
        q[2 * j + 1] = v[j + 1]
    qd = np.array([float(x) for x in q])
    sumabs = sum(abs(x) for x in v)
    return qd, float(sumabs), float(max(abs(x) for x in c))


def probe(a_frac, n_list):
    a = float(a_frac)
    y1 = float(a_frac) ** 2
    zeta = (1 + y1) / (1 - y1)
    for n in n_list:
        q, sumabs, maxc = build_poly_exact(a_frac, n)
        xt = np.linspace(a, 1.0, 400001)
        Qt = Pmod.polyval(xt, q)
        err = np.max(np.abs(1 - xt * Qt))
        theo = 1.0 / np.cosh(n * np.arccosh(zeta))
        # scaling and poly_to_angles attempt
        xs = np.linspace(-1, 1, 400001)
        mx = np.max(np.abs(Pmod.polyval(xs, q)))
        P = (0.99 / mx) * q
        try:
            ang = qp.poly_to_angles(P, "QSVT")
            msg = f"angles OK ({len(ang)})"
        except Exception as e:
            msg = f"angles FAIL: {type(e).__name__}: {e}"
        # complementary poly diagnostic
        try:
            pa = np.asarray(P)
            parity = (len(pa) - 1) % 2
            Pch = np.concatenate([np.zeros(len(pa) // 2), ncheb.poly2cheb(pa)[parity::2]]) * (1 - 1e-12)
            comp = _complementary_poly(Pch)
            lens = f"lenP={len(Pch)} lenQ={len(comp)}"
        except Exception as e:
            lens = f"comp err {e}"
        print(
            f"a={a} n={n:3d} deg={2*n-1:3d} sum|v|={sumabs:.3e} max|c|={maxc:.3e} "
            f"err={err:.3e} theo={theo:.3e} {msg} {lens}"
        )


print("--- a = 0.097 (worst case, all nu) ---")
probe(Fraction(97, 1000), [8, 10, 12, 14, 16, 18, 20, 22, 24])
print("--- a = 0.15 (nu up to ~18) ---")
probe(Fraction(15, 100), [8, 10, 12, 14, 16, 18, 20])
print("--- a = 0.3 (nu up to ~3) ---")
probe(Fraction(3, 10), [4, 6, 8, 10, 12])
