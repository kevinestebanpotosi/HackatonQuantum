import importlib.util
import json
import numpy as np

spec = importlib.util.spec_from_file_location(
    "challenge", r"Diffuse and Conquer.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

cases = [
    ("[0.2, 0.5]", None), ("[1.0, 1.2]", None),
    ("[5.0, 0.5]", None), ("[10.0, 2.0]", None),
    ("[50.0, 0.5]", None), ("[100.0, 0.0]", None),
    ("[1000.0, 0.0]", None), ("[0.0, 5.0]", None),
    ("[0.01, 50.0]", None), ("[3.7, 12.3]", None),
    ("[250.0, 0.7]", None), ("[0.001, 0.5]", None),
]

rng = np.random.default_rng(99)
for _ in range(12):
    nu = float(10 ** rng.uniform(-3, 4))
    k = float(10 ** rng.uniform(-2, 2))
    cases.append((json.dumps([nu, k]), None))

fails = 0
for inp, _ in cases:
    out = mod.run(inp)
    ins = json.loads(inp)
    A = (1 + ins[1] * mod.dt) * np.eye(4) - ins[0] * mod.dt * mod._D2
    sol = np.linalg.solve(A, np.asarray(mod.b, dtype=float))
    sol = (sol / np.linalg.norm(sol)).tolist()
    sol_out = json.loads(out)
    fid = float(np.dot(sol_out, sol))
    ok = fid >= 0.99
    fails += (not ok)
    print(f"nu={ins[0]:12.4g} k={ins[1]:10.4g} fid={fid:.6f} {'OK' if ok else 'FAIL'}")

print("TOTAL FAILS:", fails)
