import numpy as np
from numpy.polynomial import chebyshev as ncheb
import numpy.polynomial.polynomial as Pmod

import pennylane as qp

h, dt = 0.2, 0.01
D2 = np.array([[-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -2]], float) / h**2
phi = np.array([0.03, 0.04, 0.07, 0.09])
src = np.array([1.0, 2.0, 1.0, 0.5])
bvec = phi + dt * src
bnorm = np.linalg.norm(bvec)


def Amat(nu, k):
    return (1 + k * dt) * np.eye(4) - nu * dt * D2


def build_poly(a, n):
    """Optimal odd polynomial Q (deg 2n-1) with V(y)=xQ(x), y=x^2, V(0)=0 minimizing
    max |1 - V(y)| on [a^2, 1]:  V(y) = 1 - T_n(phi(y))/T_n(phi(0))."""
    y1 = a * a
    al = 2.0 / (1.0 - y1)
    be = -(1.0 + y1) / (1.0 - y1)
    zeta = (1.0 + y1) / (1.0 - y1)
    gamma = np.arccosh(zeta)
    Tz = np.cosh(n * gamma) * ((-1.0) ** n)  # = T_n(phi(0)) = T_n(-zeta)

    tn = ncheb.cheb2poly([0.0] * n + [1.0])  # monomial coeffs of T_n(z)
    c = np.zeros(n + 1)
    powk = np.array([1.0])
    for k in range(n + 1):
        if k > 0:
            powk = Pmod.polymul(powk, [be, al])
        c[: len(powk)] += tn[k] * powk
    v = -c / Tz
    v[0] += 1.0
    v[0] = 0.0  # exact by construction
    # V(y) = sum_j v_{j+1} y^{j+1};  Q(x) = V(x^2)/x = sum_j v_{j+1} x^{2j+1}
    q = np.zeros(2 * n)
    q[1::2] = v[1:]
    return q


def report(n, a=0.097):
    q = build_poly(a, n)
    xs = np.linspace(-1.0, 1.0, 2000001)
    Qv = np.polynomial.polynomial.polyval(xs, q)
    xmax = np.max(np.abs(Qv))
    s = 0.99 / xmax
    P = s * q
    # relative error on [a, 1]
    xt = np.linspace(a, 1.0, 500001)
    Qt = np.polynomial.polynomial.polyval(xt, q)
    err = np.max(np.abs(1 - xt * Qt))
    theo = 1.0 / np.cosh(n * np.arccosh((1 + a * a) / (1 - a * a)))
    # parity / structure checks
    even_zero = np.max(np.abs(P[::2]))
    absmax = np.max(np.abs(np.polynomial.polynomial.polyval(xs, P)))
    print(
        f"n={n:3d} deg={2*n-1:3d} err={err:.3e} theory={theo:.3e} "
        f"max|Q|={xmax:.4g} s(C)={s:.4g} max|P|[-1,1]={absmax:.6f} even={even_zero:.1e}"
    )
    return P, s


Ps = {}
for n in [10, 15, 20, 25, 30, 40, 50]:
    Ps[n] = report(n)

# ---- angles ----
for n, (P, s) in Ps.items():
    try:
        ang = qp.poly_to_angles(P, "QSVT")
        print("n", n, "n_angles", len(ang), "ok, first/last", np.round(ang[:3], 4), np.round(ang[-2:], 4))
    except Exception as e:
        print("n", n, "FAILED", e)
