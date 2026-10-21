import numpy as np
from scipy.optimize import linprog

h, dt = 0.2, 0.01
D2 = np.array([[-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -2]], float) / h**2
phi = np.array([0.03, 0.04, 0.07, 0.09])
src = np.array([1.0, 2.0, 1.0, 0.5])
bvec = phi + dt * src


def Amat(nu, k):
    return (1 + k * dt) * np.eye(4) - nu * dt * D2


# --- 1. check expected outputs equal normalized classical solution ---
for nu, k, exp in [
    (0.2, 0.5, [0.27959746, 0.42869397, 0.56892447, 0.64371709]),
    (1.0, 1.2, [0.28550356, 0.4595721, 0.59350887, 0.59584262]),
]:
    sol = np.linalg.solve(Amat(nu, k), bvec)
    sol = sol / np.linalg.norm(sol)
    print("expected check", nu, k, sol, np.dot(sol, np.array(exp)))

# --- 2. worst-case lower end of spectrum after normalization ---
worst = (9e9, None)
best_b = 0
for nu in np.concatenate([[0.0], np.logspace(-4, 4, 81)]):
    for k in [0.0, 0.1, 0.5, 1.0, 1.2, 5.0, 10.0, 100.0, 1000.0]:
        A = Amat(nu, k)
        alpha = np.sqrt(np.linalg.norm(A @ A.T, ord=np.inf))
        ev = np.linalg.eigvalsh(A)
        a = ev[0] / alpha
        btop = ev[-1] / alpha
        best_b = max(best_b, btop)
        if a < worst[0]:
            worst = (a, (nu, k, btop))
print("worst a =", worst, " max top =", best_b)

# --- 3. LP fit: odd polynomial Q (deg 2m+1), V(y)=sum q_j y^{j+1} ~ 1 on [a^2,1] ---
def fit(a, m, npts=3000):
    y1 = a * a
    th = np.linspace(0, np.pi, npts)
    # cluster near y1 and 1
    y = (y1 + 1) / 2 - (1 - y1) / 2 * np.cos(th)
    V = np.vander(y, m + 1, increasing=True)[:, 1:]  # y^1..y^m
    # |1 - V q| <= eps  <=>  -V q - eps <= -1  and  V q - eps <= 1
    A_ub = np.zeros((2 * npts, m + 1))
    A_ub[:npts, :m] = -V
    A_ub[npts:, :m] = V
    A_ub[:npts, m] = -1
    A_ub[npts:, m] = -1
    b_ub = np.concatenate([-np.ones(npts), np.ones(npts)])
    c = np.zeros(m + 1)
    c[m] = 1
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * m + [(0, None)], method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    q = res.x[:m]
    eps = res.x[m]
    # dense true error
    yd = np.linspace(y1, 1, 200001)
    Vd = np.vander(yd, m + 1, increasing=True)[:, 1:]
    err = np.max(np.abs(1 - Vd @ q))
    return q, eps, err


for a in [0.5, 0.3, 0.2, 0.15, 0.11, 0.097]:
    row = []
    for m in [5, 10, 15, 20, 30, 40, 60, 80, 120]:
        try:
            q, eps, err = fit(a, m)
            row.append((m, round(float(err), 6)))
        except Exception as e:
            row.append((m, f"fail: {e}"))
    print("a =", a, row)
