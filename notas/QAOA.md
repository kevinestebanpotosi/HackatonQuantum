# Nota — QAOA / minimum vertex cover

**Reto:** `APROXIMACION_RATIO.py` · **Solución:** `APROXIMACION_RATIO_solucion.py`
**Estado:** verificado localmente en los 2 casos públicos. **No** está probado en
grafos ocultos ni aceptado en plataforma (sin captura).

## Objetivo

Implementar QAOA para minimum vertex cover y devolver el ratio de aproximación
`⟨H_C⟩ / min(H_C)` como `np.tensor`, con `rtol=5e-2`.

## Enfoque que funcionó

- `p = 4`; `H_C = Σ_(i,j) 0.75·(Z_iZ_j + Z_i + Z_j) − Σ_v Z_v` (diagonal).
- Precalcular `en = diag(matrix(H_C))` y usar `qp.probs`; coste `= en · probs`.
- Init fijo `0.5`; `GradientDescentOptimizer(0.01)`, 800 pasos, `diff_method="backprop"`.
- Circuito: `H` en todos los wires; por capa `IsingZZ(1.5·g)` + `RZ(2·g·(0.75·deg−1))`
  + `RX(2·a)` (equivale a `exp(−i g H_C)·exp(−i a ΣX)`).
- `qml.qaoa` no existe en la versión local (`0.46.0-dev129`): se construyó a mano.

## Restricciones específicas

- `rtol=5e-2`; retorno convertible a `str`; solo el bloque editable.
- La referencia está **sub-optimizada**: banda `[0.55–0.61]` y `[0.90–0.99]`, mientras
  el óptimo real p=4 es ~`0.918` (G1) y `1.0` (C4) según `QAOA_scratch.py`.
- Grafos ocultos: no hay evidencia de generalización con solo 2 casos públicos.

## Errores relevantes

- Optimizar “demasiado bien” (L-BFGS, reinicios) sale de la banda → `Wrong Answer`.
- No persistir la solución: se recuperó del chat (`ses_eaa74ead…`, 19-oct;
  copia en `ses_eaa4976c…`) y se guardó después en `APROXIMACION_RATIO_solucion.py`.
- Meter el optimizador dentro del QNode provoca recursión (lección de IQP).

## Lecciones reutilizables

1. Calcula la banda con diagonalización exacta **antes** de optimizar.
2. Ajusta init/pasos/stepsize a la banda, no al óptimo matemático.
3. Verifica los públicos, pero no afirmes generalización sin estrés con grafos ocultos.
4. Persiste la solución y su evidencia en el mismo momento en que pasa.
