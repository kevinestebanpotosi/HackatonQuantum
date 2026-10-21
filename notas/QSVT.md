# Nota — QSVT / difusión-reacción

**Reto:** `Diffuse and Conquer.py`
**Estado:** verificado localmente; 2 públicos `Correct!` y 54 casos de estrés con
`TOTAL FAILS: 0` (peor fidelidad `0.99975`).

## Objetivo

Resolver un paso implícito de difusión-reacción,
`A·φ_new = b` con `A = (1 + k·dt)·I − ν·dt·D2` y `b = φ + dt·source`, usando QSVT
para aproximar `A⁻¹`. El harness compara el vector normalizado con fidelidad `≥0.99`.

## Enfoque que funcionó

- Polinomio impar `Q(x) = V(x²)/x` con `V(y) = 1 − T_n(φ(y))/T_n(φ(0))`, donde `φ`
  mapea `[a², 1]` a `[-1, 1]`; `a = 0.097`, `n = 20`.
- Coeficientes con `Fraction` + Chebyshev (`_inverse_poly`) para evitar errores de
  redondeo; `C` escala el polinomio para que la norma QSP sea `≤0.99`.
- `phi_qsvt = qp.poly_to_angles(C·_poly, "QSVT")`.
- `BlockEncode(M)` con `M = A/alpha`, `alpha = max(1, ‖A‖∞)`; `PCPhase` de dimensión 4.
- Dos `QSVT` controlados con ángulos `+φ` y `−φ` y un `Hadamard` final: la
  interferencia deja la parte real (la parte imaginaria se cancela).
- `StatePrep(b/‖b‖)` en wires `[2,3]`; se devuelve `qp.state()` y el harness toma `[:4]`.

## Restricciones específicas

- Fidelidad `≥0.99`; `C>0`; `phi`, `source`, `dt`, `h` fijos e inmutables.
- `a` debe superar el máximo eigenvalue de `A/alpha` para todo `ν,k ≥ 0`; `0.097`
  deja margen.
- Solo el bloque editable; `default.qubit` de 4 wires.

## Errores relevantes

- `poly_to_angles` exige norma QSP `≤1`: sin el escalado `C` falla o distorsiona.
- Sin la interferencia, la parte imaginaria del bloque contamina la salida.
- Un `a` demasiado grande pierde aproximación; uno demasiado pequeño rompe la
  hipótesis del mapeo. Verificar con `eigvalsh(A/alpha)` en los extremos.
- La normalización final del harness oculta la escala, pero no la dirección: no
  confundir “pasa” con “resuelve exactamente”.

## Lecciones reutilizables

1. Aritmética exacta (`Fraction`) para coeficientes de polinomios; convierte a float
   solo al final.
2. Chebyshev es la vía natural para aproximar `1/x` en un intervalo espectral.
3. Prueba los extremos de los parámetros (`ν`, `k`) antes de enviar; usa los scratch
   `scratch_stress.py` y `scratch_final_check.py`.
4. La fidelidad es más indulgente que el error absoluto, pero exige controlar la
   fase global y la parte real.
