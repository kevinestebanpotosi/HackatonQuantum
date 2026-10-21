import pennylane as qml
from pennylane import numpy as pnp
import numpy as onp
from scipy.optimize import minimize

P = 4


def build(edges):
    n = max(max(e) for e in edges) + 1
    coeffs = []
    ops = []
    for (i, j) in edges:
        coeffs += [0.75, 0.75, 0.75]
        ops += [qml.Z(i) @ qml.Z(j), qml.Z(i), qml.Z(j)]
    for i in range(n):
        coeffs.append(-1.0)
        ops.append(qml.Z(i))
    Hc = qml.Hamiltonian(coeffs, ops)
    Hm = qml.Hamiltonian([1.0] * n, [qml.X(i) for i in range(n)])
    return n, Hc, Hm


def exact_min(edges, n):
    diag = []
    for k in range(2 ** n):
        z = [1 if (k >> (n - 1 - i)) & 1 == 0 else -1 for i in range(n)]
        # z=+1 for bit 0
        e = 0.0
        for (i, j) in edges:
            e += 0.75 * (z[i] * z[j] + z[i] + z[j])
        e -= sum(z)
        diag.append(e)
    return min(diag)


def run(edges, seed=0, init_scale=0.1, maxiter=1000):
    n, Hc, Hm = build(edges)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circ(params):
        gammas = params[:P]
        betas = params[P:]
        for w in range(n):
            qml.Hadamard(wires=w)
        for l in range(P):
            g = gammas[l]
            for (i, j) in edges:
                qml.IsingZZ(2 * 0.75 * g, wires=[i, j])
            zc = {}
            for (i, j) in edges:
                zc[i] = zc.get(i, 0.0) + 0.75 * g
                zc[j] = zc.get(j, 0.0) + 0.75 * g
            for i in range(n):
                qml.RZ(2 * (zc.get(i, 0.0) - g), wires=i)
            for i in range(n):
                qml.RX(2 * betas[l], wires=i)
        return qml.expval(Hc)

    rng = onp.random.default_rng(seed)
    x0 = rng.uniform(-init_scale, init_scale, size=2 * P)
    x0 = pnp.array(x0, requires_grad=True)
    grad = qml.grad(circ)

    def fun(x):
        x = pnp.array(x, requires_grad=True)
        return float(circ(x))

    def jac(x):
        x = pnp.array(x, requires_grad=True)
        return onp.array(grad(x))

    res = minimize(fun, onp.array(x0), jac=jac, method="L-BFGS-B",
                   options={"maxiter": maxiter})
    e = exact_min(edges, n)
    return res.fun, e, res.fun / e, res.nit, res.success


if __name__ == "__main__":
    cases = {
        "tri_tail": [[0, 1], [1, 2], [0, 2], [2, 3]],
        "square": [[0, 1], [1, 2], [2, 3], [3, 0]],
    }
    for name, edges in cases.items():
        print("=" * 60)
        print(name, "exact", exact_min(edges, max(max(e) for e in edges) + 1))
        for seed in range(10):
            fun, e, ratio, nit, ok = run(edges, seed=seed)
            print(f"seed={seed} expval={fun:.8f} ratio={ratio:.8f} nit={nit} ok={ok}")
