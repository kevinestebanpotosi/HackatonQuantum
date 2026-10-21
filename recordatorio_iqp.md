# Recordatorio — Reto IQP (sucesionFases.py)

**Estado: resuelto ✔** (tests públicos: expval intermedio exacto y mínimo `C(θ) = -1` en ambos casos)

## Qué pedía el reto

Construir y optimizar un circuito IQP de 4 cúbits:

- `U(θ) = ∏_j exp(-i θ_j X_{S_j} / 2)`, donde cada `pattern[i]` es una lista de grupos de
  wires; todos los generadores son Pauli-X puro → conmutan → el orden no importa.
- Medir una suma de observables Pauli-Z descritos por un vector binario `observable`
  (bit `a_k = 1` ⇒ factor `Z_k`).
- Restricción: **no usar** la plantilla `qml.IQP` (el check la detecta vía `specs`).

## Cómo funcionó la solución

### `iqp_circuit` (QNode) — SOLO construye el circuito

```python
for i, groups in enumerate(pattern):
    for group in groups:
        qp.PauliRot(weights[i], "X" * len(group), wires=group)  # exp(-i θ_i X_S / 2)

obs = None
for i, val in enumerate(observable):
    if val == 1:
        obs = qp.PauliZ(i) if obs is None else obs @ qp.PauliZ(i)

return qp.expval(obs)
```

- `PauliRot(θ, "X…X", wires=S)` implementa exactamente `exp(-iθ X_S/2)` (convención
  de PennyLane con factor 1/2).
- El observable se arma con producto tensorial `@` de `PauliZ` donde el bit es 1.

### `optimize_iqp` — SOLO optimiza, usando el QNode como coste

```python
weights = np.ones(len(pattern), requires_grad=True)
opt = qp.GradientDescentOptimizer(stepsize=0.4)
for _ in range(200):
    weights = opt.step(lambda w: iqp_circuit(w, pattern, observable), weights)
return weights
```

- Init en **unos** + GD (stepsize 0.4, 200 pasos) converge al mínimo `-1` (~1e-14)
  en los dos tests públicos; determinista (sin semilla aleatoria).

## Trampas que causaron errores en el camino

1. **`maximum recursion depth exceeded`**: meter el optimizador (o un `def cost(w):
   return iqp_circuit(...)` dentro del QNode. El QNode se llamaba a sí mismo →
   recursión infinita. Regla: QNode = solo puertas + `expval`; optimización = solo
   en `optimize_iqp` (con lambda, sin helper `cost` local al QNode).
2. **No editar "de más"**: la plataforma solo tiene casillas delimitadas; el docstring,
   `n_qubits`, `dev` y el harness (`run`/`check`) son inmutables.
3. **`qml.IQP` prohibido**: usar solo `PauliRot` + `PauliZ`.
4. **`specs` local**: el PennyLane editable de `../pennylane` (versión dev) ya no
   acepta `qml.specs` sobre un QNode desnudo; el harness local falla por eso, pero en
   la plataforma (PennyLane estable) funciona. No es error de la solución.

## Verificación local (con `.venv\Scripts\python.exe`)

| Test | expval con weights `[0.3,-0.7,0.2,1.1]` | mínimo C(θ) |
|---|---|---|
| `[[[0],[1]],[[1,2]],[[0,2],[1,3]],[[0,1,2,3]]]`, obs `[1,0,1,0]` | `0.7306816499355087` ✓ | `-1.0` ✓ |
| `[[[0],[2]],[[1,2,3]],[[0,1],[2,3]],[[0,1,2]]]`, obs `[1,1,0,1]` | `0.93629336358419` ✓ | `-1.0` ✓ |
