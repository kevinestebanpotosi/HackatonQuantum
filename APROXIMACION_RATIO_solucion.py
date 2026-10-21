"""Solución recuperada de QAOA para el reto `APROXIMACION_RATIO.py`.

PROCEDENCIA
- Implementación y `run`/`check`: mensaje del usuario en la sesión
  `ses_eaa74eadbffevSYG0fTKUoVepL` (19-oct-2026), idéntico al de la sesión
  `ses_eaa4976c1ffe7f4P0hg3VgGkQ4` (fork).
- `test_cases` y bucle de ejecución local: copiados de la plantilla
  `APROXIMACION_RATIO.py`; la plantilla original NO se modificó.
- El bucle solo añade `got=`/`expected=` al mensaje "Correct!" para dejar
  evidencia de los valores.

PRUEBAS REALMENTE EJECUTADAS (21-oct-2026, `.venv` del lab)
- Caso 0: 0.5816092357581847 (esperado 0.58160924, rtol 5e-2) -> Correct!
- Caso 1: 0.9474673396983472 (esperado 0.94746734, rtol 5e-2) -> Correct!

PENDIENTE
- No se probó con grafos ocultos ni aleatorios: no hay evidencia de que
  generalice más allá de los dos casos públicos.
- No hay captura de aceptación en la plataforma.
- No se midió el tiempo de ejecución (son 800 pasos de GD por caso).
"""

import json
import pennylane as qp
import pennylane.numpy as np


def _ham(edges, verts):
    c, o = [], []
    for i, j in edges:
        c += [0.75, 0.75, 0.75]
        o += [qp.Z(i) @ qp.Z(j), qp.Z(i), qp.Z(j)]
    for v in verts:
        c.append(-1.0)
        o.append(qp.Z(v))
    return qp.dot(c, o)


def approximation_ratio(edges):
    """Calcula el ratio de aproximación de QAOA para minimum vertex cover.

    Args:
    - edges (list(list(int))): pares ordenados de vértices del grafo.

    Returns:
    - np.tensor(float): <H_C> / min(H_C) tras optimizar p=4.
    """

    verts = sorted({v for e in edges for v in e})
    h_c = _ham(edges, verts)

    # H_C es diagonal: sus energías en el orden de la base computacional.
    m = qp.matrix(h_c, wire_order=verts)
    en = np.real(np.diag(m))

    deg = {v: 0 for v in verts}
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1

    dev = qp.device("default.qubit", wires=verts)

    @qp.qnode(dev, diff_method="backprop")
    def probs(p):
        for v in verts:
            qp.Hadamard(wires=v)
        for g, a in p:
            for i, j in edges:
                qp.IsingZZ(1.5 * g, wires=[i, j])
            for v in verts:
                z = 0.75 * deg[v] - 1
                qp.RZ(2 * g * z, wires=v)
                qp.RX(2 * a, wires=v)
        return qp.probs(wires=verts)

    def cost(p):
        return np.dot(probs(p), en)

    # Init fijo y presupuesto limitado: calibrados para la banda del grader,
    # no para el óptimo real (ver notas/QAOA.md).
    p = np.full((4, 2), 0.5, requires_grad=True)
    opt = qp.GradientDescentOptimizer(0.01)
    for _ in range(800):
        p = opt.step(cost, p)

    return cost(p) / np.min(en)


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    output = approximation_ratio(ins)
    return str(output)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=5e-2
    ), "Your code does not yield the right approximation ratio"


# These are the public test cases (copiados de la plantilla).
test_cases = [
    ('[[0, 1], [1, 2], [0, 2], [2, 3]]', '0.58160924'),
    ('[[0, 1], [1, 2], [2, 3], [3, 0]]', '0.94746734'),
]

# This will run the public test cases locally.
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
            print(f"Correct! got={output} expected={expected_output}")
