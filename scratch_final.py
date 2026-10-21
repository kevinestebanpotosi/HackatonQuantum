import numpy as np
from fractions import Fraction
from numpy.polynomial import chebyshev as ncheb
import numpy.polynomial.polynomial as Pmod

import pennylane as qp

h, dt = 0.2, 0.01
D2 = np.array([[-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -2]], float) / h**2
phi_old = np.array([0.03, 0.04, 0.07, 0.09])
src = np.array([1.0, 2.0, 1.0, 0.5])
bvec = phi_old + dt * src
bnorm = np.linalg.norm(bvec)


def Amat(nu, k):
    return (1 + k * dt) * np.eye(4) - nu * dt * D2


def build_poly(a_frac, n):
    y1 = a_frac * a_frac
    al = Fraction(2, 1) / (1 - y1)
    be = -(1 + y1) / (1 - y1)
    tn = ncheb.cheb2poly([0] * n + [1])
    tnf = [Fraction(int(round(t))) for t in tn]
    c = [Fraction(0)] * (n + 1)
    powk = [Fraction(1)]
    for k in range(n + 1):
        if k > 0:
            new = [Fraction(0)] * (len(powk) + 1)
            for i, coef in enumerate(powk):
                new[i] += coef * be
                new[i + 1] += coef * al
            powk = new
        for i, coef in enumerate(powk):
            c[i] += tnf[k] * coef
    v = [-ci / c[0] for ci in c]
    v[0] = Fraction(0)
    q = [Fraction(0)] * (2 * n)
    for j in range(n):
        q[2 * j + 1] = v[j + 1]
    return np.array([float(x) for x in q])


def qsp_norm(poly):
    """max_theta |sum_j cheb[parity+2j] e^{i j theta}|  (must be <= 1 for QSP feasibility)."""
    cheb = ncheb.poly2cheb(poly)
    parity = (len(poly) - 1) % 2
    cj = cheb[parity::2]
    big = np.concatenate([cj, np.zeros(4096 - len(cj))])
    return np.max(np.abs(np.fft.fft(big)))


def make_poly(a_frac, n):
    q = build_poly(a_frac, n)
    err = np.max(np.abs(1 - np.linspace(float(a_frac), 1, 400001) *
                        Pmod.polyval(np.linspace(float(a_frac), 1, 400001), q)))
    qn = qsp_norm(q)
    s = 0.99 / qn
    return s * q, s, err


def run_case(a_frac, n, solver="root-finding"):
    P, s, err = make_poly(a_frac, n)
    theo = 1 / np.cosh(n * np.arccosh((1 + float(a_frac) ** 2) / (1 - float(a_frac) ** 2)))
    try:
        angles = qp.poly_to_angles(P, "QSVT", angle_solver=solver)
    except Exception as e:
        print(f"a={float(a_frac)} n={n} {solver}: ANGLES FAIL {type(e).__name__}: {e}")
        return None
    print(f"a={float(a_frac)} n={n} {solver}: deg={len(P)-1} err={err:.3e} theo={theo:.3e} "
          f"C={s:.4g} n_angles={len(angles)}")

    dev = qp.device("default.qubit", wires=3)

    @qp.qnode(dev)
    def circ(nu, k):
        A = (1 + k * dt) * np.eye(4) - nu * dt * D2
        alpha = np.sqrt(np.linalg.norm(A @ A.T, ord=np.inf))
        alpha = float(max(alpha, 1.0))
        M = A / alpha
        qp.StatePrep(bvec / bnorm, wires=[1, 2])
        qp.QSVT(qp.BlockEncode(M, wires=[0, 1, 2]),
                [qp.PCPhase(a_, dim=4, wires=[0, 1, 2]) for a_ in angles])
        return qp.state()

    # convention check vs P(M) psi  on one case
    nu0, k0 = 0.2, 0.5
    A0 = Amat(nu0, k0)
    al0 = float(max(np.sqrt(np.linalg.norm(A0 @ A0.T, ord=np.inf)), 1.0))
    M0 = A0 / al0
    w0, V0 = np.linalg.eigh(M0)
    P0 = (V0 * Pmod.polyval(w0, P)) @ V0.T
    st0 = circ(nu0, k0)
    pred = s * (Pmod.polyval(w0, P))  # placeholder
    psi_b = bvec / bnorm
    pred_vec = s * ((V0 * Pmod.polyval(w0, P)) @ (V0.T @ psi_b))
    got = st0[:4]
    print("   conv check: max|out - P(M)psi| =", np.max(np.abs(got - pred_vec)),
          " max|Im| =", np.max(np.abs(np.imag(got))))

    cases = [(0.2, 0.5), (1.0, 1.2), (0.01, 0.0), (0.5, 2.0), (2.0, 0.1), (5.0, 3.0),
             (10.0, 0.0), (17.0, 0.0), (0.1, 10.0), (3.0, 1.2), (0.0, 1.0)]
    rng = np.random.default_rng(0)
    cases += [(float(u), float(v)) for u, v in rng.uniform(0.01, 8, size=(30, 2))]
    worst = (2.0, None)
    for nu, k in cases:
        out = circ(nu, k)[:4]
        rescaled = out * bnorm / s
        v = np.real(rescaled / np.linalg.norm(rescaled))
        sol = np.linalg.solve(Amat(nu, k), bvec)
        sol = sol / np.linalg.norm(sol)
        fid = float(np.dot(v, sol))
        if fid < worst[0]:
            worst = (fid, (nu, k))
    print(f"   min fidelity over {len(cases)} cases: {worst[0]:.6f} at {worst[1]}")
    return worst[0]


for a_frac, n in [(Fraction(97, 1000), 20), (Fraction(9, 100), 20), (Fraction(95, 1000), 20)]:
    run_case(a_frac, n)
