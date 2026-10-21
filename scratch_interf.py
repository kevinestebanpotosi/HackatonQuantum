import json
import numpy as np
from fractions import Fraction
from numpy.polynomial import chebyshev as ncheb
import numpy.polynomial.polynomial as Pmod

import pennylane as qp

np.set_printoptions(precision=6, suppress=True, linewidth=160)

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
    parity = (len(poly) - 1) % 2
    cj = cheb[parity::2]
    big = np.concatenate([cj, np.zeros(4096 - len(cj))])
    return np.max(np.abs(np.fft.fft(big)))


def make_poly(a_frac, n):
    q = build_poly(a_frac, n)
    s = 0.99 / qsp_norm(q)
    return s * q, s


P, s = make_poly(Fraction(97, 1000), 20)
angles = qp.poly_to_angles(P, "QSVT")
print("deg", len(P) - 1, "C", s, "n_angles", len(angles))


def true_qsvt_unitary(M, angles):
    block = qp.BlockEncode(M, wires=[0, 1, 2])
    proj = [qp.PCPhase(a, dim=4, wires=[0, 1, 2]) for a in angles]
    return qp.matrix(qp.QSVT(block, proj))


def rel_fidelity(out4, nu, k):
    A = Amat(nu, k)
    sol = np.linalg.solve(A, bvec)
    sol = sol / np.linalg.norm(sol)
    o = out4 / np.linalg.norm(out4)
    return float(np.dot(np.real(o), sol))


def eval_case(nu, k, Uq):
    A = Amat(nu, k)
    alpha = float(max(np.sqrt(np.linalg.norm(A @ A.T, ord=np.inf)), 1.0))
    M = A / alpha
    psi = bvec / bnorm

    Uq_local = true_qsvt_unitary(M, angles)

    dev = qp.device("default.qubit", wires=4)

    @qp.qnode(dev)
    def circ_openctrl():
        qp.Hadamard(wires=0)
        qp.StatePrep(psi, wires=[2, 3])
        block = qp.BlockEncode(M, wires=[1, 2, 3])
        proj_p = [qp.PCPhase(a, dim=4, wires=[1, 2, 3]) for a in angles]
        proj_m = [qp.PCPhase(-a, dim=4, wires=[1, 2, 3]) for a in angles]
        qp.ctrl(qp.QSVT(block, proj_p), control=0, control_values=0)
        qp.ctrl(qp.QSVT(block, proj_m), control=0, control_values=1)
        qp.Hadamard(wires=0)
        return qp.state()

    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    Ufull = np.kron(H, np.eye(8)) @ np.block([[Uq_local, np.zeros((8, 8))],
                                              [np.zeros((8, 8)), Uq_local.conj()]])

    @qp.qnode(dev)
    def circ_unitary():
        qp.Hadamard(wires=0)
        qp.StatePrep(psi, wires=[2, 3])
        qp.QubitUnitary(Ufull, wires=[0, 1, 2, 3])
        return qp.state()

    st_oc = circ_openctrl()
    st_uq = circ_unitary()
    fid_oc = rel_fidelity(st_oc[:4], nu, k)
    fid_uq = rel_fidelity(st_uq[:4], nu, k)
    im_oc = float(np.max(np.abs(st_oc[:4].imag)))
    im_uq = float(np.max(np.abs(st_uq[:4].imag)))
    return fid_oc, im_oc, fid_uq, im_uq, st_oc[:4]


cases = [(0.2, 0.5), (1.0, 1.2), (5.0, 0.5), (10.0, 2.0), (50.0, 0.5), (0.01, 5.0)]
for nu, k in cases:
    fid_oc, im_oc, fid_uq, im_uq, out = eval_case(nu, k, None)
    print(f"nu={nu:6.2f} k={k:4.2f} | openctrl fid={fid_oc:.6f} im={im_oc:.2e} | "
          f"unitary fid={fid_uq:.6f} im={im_uq:.2e}")
    print("   out[:4]:", out)
