"""Scratch: optimo real de QAOA (simulacion clasica exacta) vs valores esperados.

H_cost = (3/4) * sum_(i,j) (Zi Zj + Zi + Zj) - sum_i Zi   (diagonal)
Mixer   = sum_i X_i
Convencion: wire 0 = bit mas significativo (PennyLane).
"""

import numpy as np
from scipy.optimize import minimize

GRAPHS = {
    "G1 (triangulo+colgante)": [[0, 1], [1, 2], [0, 2], [2, 3]],
    "G2 (C4)": [[0, 1], [1, 2], [2, 3], [3, 0]],
    "G3 (5 nodos)": [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 4]],
}
EXPECTED = {"G1 (triangulo+colgante)": 0.58160924, "G2 (C4)": 0.94746734}
LOWER = {"G1 (triangulo+colgante)": 0.55, "G2 (C4)": 0.92, "G3 (5 nodos)": 0.55}


def build(edges):
    n = max(max(e) for e in edges) + 1
    Hd = np.zeros(2**n)
    for b in range(2**n):
        z = np.array([1 - 2 * ((b >> (n - 1 - i)) & 1) for i in range(n)])
        s = sum(0.75 * (z[i] * z[j] + z[i] + z[j]) for i, j in edges)
        s -= z.sum()
        Hd[b] = s
    return n, Hd


def apply_rx_all(state, beta, n):
    c, s = np.cos(beta), -1j * np.sin(beta)
    U = np.array([[c, s], [s, c]])
    st = state.reshape([2] * n)
    for w in range(n):
        st = np.moveaxis(st, w, 0)
        st = np.tensordot(U, st, axes=([1], [0]))
        st = np.moveaxis(st, 0, w)
    return st.reshape(-1)


def expval(params, p, n, Hd):
    amps = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
    for k in range(p):
        amps = amps * np.exp(-1j * params[k] * Hd)
        amps = apply_rx_all(amps, params[p + k], n)
    return float(np.sum(np.abs(amps) ** 2 * Hd))


def optimize(edges, p, restarts=30, seed=0):
    n, Hd = build(edges)
    rng = np.random.default_rng(seed)
    best = np.inf
    for _ in range(restarts):
        x0 = rng.uniform(0, 2 * np.pi, 2 * p)
        res = minimize(expval, x0, args=(p, n, Hd), method="BFGS",
                       options={"maxiter": 800, "gtol": 1e-10})
        best = min(best, res.fun)
    return best, float(Hd.min()), best / Hd.min()


if __name__ == "__main__":
    for name, edges in GRAPHS.items():
        n, Hd = build(edges)
        exp = EXPECTED.get(name)
        banda = (exp * 0.95, exp * 1.05) if exp else None
        print(f"\n=== {name}  n={n}  lam_min={Hd.min()}  lower_bound={LOWER[name]}  banda_lab={banda}")
        for p in (1, 2, 3, 4):
            best, lm, ratio = optimize(edges, p)
            flag = ""
            if exp is not None:
                flag = " <== EN BANDA" if abs(ratio - exp) <= 0.05 * exp else " (fuera de banda)"
            print(f"  p={p}: <H>min={best:.8f}  ratio={ratio:.8f}{flag}")
