import json
import pennylane as qp
import pennylane.numpy as np

# DO NOT CHANGE THESE VARIABLES LATER IN THE CODE
#CODIGO AQUI
phi = np.array([0.03, 0.04, 0.07, 0.09])
source = np.array([1., 2., 1., 0.5])
dt = 0.01
h = 0.2


import numpy as _sp
from fractions import Fraction
from numpy.polynomial import chebyshev as _cheb

_D2 = _sp.array([[-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -2]],
                dtype=float) / h**2


def _inverse_poly(a_frac, n):
    # Odd polynomial Q(x) = V(x^2)/x with V(y) = 1 - T_n(phi(y))/T_n(phi(0)),
    # phi mapping [a^2, 1] onto [-1, 1]; relative error <= 1/|T_n(phi(0))|.
    y1 = a_frac * a_frac
    al = Fraction(2, 1) / (1 - y1)
    be = -(1 + y1) / (1 - y1)
    tnf = [Fraction(int(round(t))) for t in _cheb.cheb2poly([0] * n + [1])]
    c = [Fraction(0)] * (n + 1)
    powk = [Fraction(1)]
    for k_ in range(n + 1):
        if k_ > 0:
            new = [Fraction(0)] * (len(powk) + 1)
            for i, coef in enumerate(powk):
                new[i] += coef * be
                new[i + 1] += coef * al
            powk = new
        for i, coef in enumerate(powk):
            c[i] += tnf[k_] * coef
    v = [-ci / c[0] for ci in c]
    v[0] = Fraction(0)
    q = [Fraction(0)] * (2 * n)
    for j in range(n):
        q[2 * j + 1] = v[j + 1]
    return _sp.array([float(x) for x in q])


# a = 0.097 < min eig(A/alpha) for every nu,k >= 0; n = 20 -> rel. error ~4.4e-2
_poly = _inverse_poly(Fraction(97, 1000), 20)
_cj = _cheb.poly2cheb(_poly)[1::2]
C = float(0.99 / _sp.max(_sp.abs(_sp.fft.fft(
    _sp.concatenate([_cj, _sp.zeros(4096 - len(_cj))])))))
phi_qsvt = qp.poly_to_angles(C * _poly, "QSVT")  # scales poly so QSP norm <= 0.99

b = phi + dt * source

device = qp.device("default.qubit", wires=4)

#COdigo hasta aqui


@qp.qnode(device)
def qsvt_diffusion(nu, k):

    """
    Implements QSVT to find the concentration phi after one time step

    Args:
    - nu (float): The diffusion coefficient
    - k (float): The reaction rate

    Returns:
    - (np.array): The concentation vector on the next time step.
    """

#CODIGO AQUI
    # Block-encode M = A/alpha with alpha >= ||A|| so that spec(M) subset [a, 1]
    A = (1.0 + k * dt) * _sp.eye(4) - nu * dt * _D2
    alpha = _sp.sqrt(_sp.linalg.norm(A @ A.T, ord=_sp.inf))
    if alpha < 1.0:
        alpha = 1.0
    M = A / alpha

    # Interference: branch |0> gets QSVT(+angles) = U, branch |1> gets
    # QSVT(-angles) = conj(U); recombining with Hadamard yields the real
    # part of the QSVT block, i.e. P(M) (P real), with zero imaginary part.
    qp.Hadamard(wires=0)
    qp.StatePrep(b / np.linalg.norm(b), wires=[2, 3])

    block = qp.BlockEncode(M, wires=[1, 2, 3])
    proj_p = [qp.PCPhase(a, dim=4, wires=[1, 2, 3]) for a in phi_qsvt]
    proj_m = [qp.PCPhase(-a, dim=4, wires=[1, 2, 3]) for a in phi_qsvt]
    qp.ctrl(qp.QSVT(block, proj_p), control=0, control_values=0)
    qp.ctrl(qp.QSVT(block, proj_m), control=0, control_values=1)
    qp.Hadamard(wires=0)

    # State amplitudes: out[:4] (wire0=0, ancilla=0) ~ C * A^{-1} b / ||b||
    return qp.state()
#CODIGO HASTA AQUI

# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    out_state = qsvt_diffusion(*ins)[:4]
    rescaled_out= out_state * np.linalg.norm(b) / C
    output = rescaled_out/ np.linalg.norm(rescaled_out)

    return str(np.real(output).tolist())

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    fidelity = np.dot(solution_output, expected_output)
    assert fidelity >=0.99, "Your circuit doesn't look quite right "


# These are the public test cases
test_cases = [
    ('[0.2,0.5]', '[0.27959746, 0.42869397, 0.56892447, 0.64371709]'),
    ('[1.0, 1.2]', '[0.28550356, 0.4595721 , 0.59350887, 0.59584262]')
]

# This will run the public test cases locally
for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")