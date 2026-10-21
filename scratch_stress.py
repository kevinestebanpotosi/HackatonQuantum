import numpy as np
from fractions import Fraction
from numpy.polynomial import chebyshev as ncheb

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
    cheb = ncheb.poly2cheb(poly)
    cj = cheb[1::2]
    big = np.concatenate([cj, np.zeros(4096 - len(cj))])
    return np.max(np.abs(np.fft.fft(big)))


q = build_poly(Fraction(97, 1000), 20)
s = 0.99 / qsp_norm(q)
P = s * q
angles = qp.poly_to_angles(P, "QSVT")
print("deg", len(P) - 1, "C", s, "n_angles", len(angles))

dev = qp.device("default.qubit", wires=4)


@qp.qnode(dev)
def qsvt_diffusion(nu, k):
    A = (1 + k * dt) * np.eye(4) - nu * dt * D2
    alpha = max(float(np.sqrt(np.linalg.norm(A @ A.T, ord=np.inf))), 1.0)
    M = A / alpha
    qp.Hadamard(wires=0)
    qp.StatePrep(bvec / bnorm, wires=[2, 3])
    block = qp.BlockEncode(M, wires=[1, 2, 3])
    proj_p = [qp.PCPhase(a, dim=4, wires=[1, 2, 3]) for a in angles]
    proj_m = [qp.PCPhase(-a, dim=4, wires=[1, 2, 3]) for a in angles]
    qp.ctrl(qp.QSVT(block, proj_p), control=0, control_values=0)
    qp.ctrl(qp.QSVT(block, proj_m), control=0, control_values=1)
    qp.Hadamard(wires=0)
    return qp.state()


def grader_fid(nu, k):
    out_state = qsvt_diffusion(nu, k)[:4]
    rescaled = out_state * bnorm / s
    output = rescaled / np.linalg.norm(rescaled)
    real_out = np.real(output)
    A = Amat(nu, k)
    sol = np.linalg.solve(A, bvec)
    sol = sol / np.linalg.norm(sol)
    return float(np.dot(real_out, sol))


cases = [
    (0.2, 0.5), (1.0, 1.2),
    (0.0, 0.0), (0.0, 5.0), (5.0, 0.0),
    (1e-6, 0.5), (1e-6, 0.0),
    (100.0, 0.0), (1000.0, 0.0), (1e4, 0.0),
    (0.0, 100.0), (0.01, 50.0),
    (50.0, 0.5), (10.0, 2.0),
]
rng = np.random.default_rng(7)
for _ in range(40):
    nu = float(10 ** rng.uniform(-3, 4))
    k = float(10 ** rng.uniform(-2, 2))
    cases.append((nu, k))

worst = 1.0
worst_case = None
for nu, k in cases:
    f = grader_fid(nu, k)
    if f < worst:
        worst, worst_case = f, (nu, k)
    flag = "  <-- FAIL" if f < 0.99 else ""
    print(f"nu={nu:12.4g} k={k:10.4g} fid={f:.6f}{flag}")
print("WORST:", worst, "at", worst_case)
