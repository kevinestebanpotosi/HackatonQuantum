import numpy as np
import pennylane as qp
from scipy.linalg import sqrtm

np.set_printoptions(precision=4, suppress=True, linewidth=160)

# --- Test A: PennyLane docstring-style, P(x) = 0.5 x on a 2x2 block encoding ---
M = np.array([[0.2, 0.1], [0.1, -0.1]])
alpha = max(1.0, np.sqrt(np.linalg.norm(M @ M.T, ord=np.inf)))
M = M / alpha
print("alpha A:", alpha)

poly = np.array([0.0, 0.5])
angles = qp.poly_to_angles(poly, "QSVT")
print("angles A:", angles)

dev = qp.device("default.qubit", wires=2)


@qp.qnode(dev)
def circA():
    qp.QSVT(qp.BlockEncode(M, wires=[0, 1]), [qp.PCPhase(a_, dim=2, wires=[0, 1]) for a_ in angles])
    return qp.state()


U = qp.matrix(circA)()
print("U:\n", U)
print("top-left block:\n", U[:2, :2])
print("expected 0.5*M:\n", 0.5 * M)

# --- Test B: docstring example verbatim ---
poly = np.array([0, -1, 0, 0.5, 0, 0.5])
angles = qp.poly_to_angles(poly, "QSVT")
input_matrix = np.array([[0.2, 0.1], [0.1, -0.1]])
dev = qp.device("default.qubit")


@qp.qnode(dev)
def circuit():
    qp.QSVT(qp.BlockEncode(input_matrix, wires=[0, 1]),
            [qp.PCPhase(angles[i], dim=len(input_matrix), wires=[0, 1]) for i in range(len(angles))])
    return qp.state()


U = qp.matrix(circuit)()
print("B top-left:\n", U[:2, :2])

# --- Test C: our inverse polynomial on a 4x4 (small n=8 for speed) ---
from fractions import Fraction
from numpy.polynomial import chebyshev as ncheb
import numpy.polynomial.polynomial as Pmod

h, dt = 0.2, 0.01
D2 = np.array([[-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -2]], float) / h**2


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


q = build_poly(Fraction(97, 1000), 8)
s = 0.99 / qsp_norm(q)
Pc = s * q
angles = qp.poly_to_angles(Pc, "QSVT")
print("n=8 angles:", np.round(angles, 4))

nu, k = 0.2, 0.5
A = Amat(nu, k)
alpha = np.sqrt(np.linalg.norm(A @ A.T, ord=np.inf))
M4 = A / alpha
print("alpha C:", alpha, "||M||2:", np.linalg.norm(M4, 2))

dev = qp.device("default.qubit", wires=3)


@qp.qnode(dev)
def circC():
    qp.QSVT(qp.BlockEncode(M4, wires=[0, 1, 2]),
            [qp.PCPhase(a_, dim=4, wires=[0, 1, 2]) for a_ in angles])
    return qp.state()


U = qp.matrix(circC)()
print("C top-left block:\n", U[:4, :4].real)
w, V = np.linalg.eigh(M4)
target = (V * s * Pmod.polyval(w, Pc)) @ V.T
print("target s*P(M):\n", target)
print("max|block - target|:", np.max(np.abs(U[:4, :4] - target)))
print("max|block - target.conj|:", np.max(np.abs(U[:4, :4] - target.conj())))
print("max|Im(block)|:", np.max(np.abs(U[:4, :4].imag)))
