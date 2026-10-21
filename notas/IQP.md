# Nota — IQP / sucesión de fases

**Reto:** `sucesionFases.py` · **Detalle ampliado:** `recordatorio_iqp.md`
**Estado:** verificado localmente (públicos); aceptación en plataforma declarada por
el usuario, sin salida capturada.

## Objetivo

Circuito IQP de 4 cúbits, `U(θ) = ∏_j exp(−i θ_j X_{S_j}/2)`, midiendo el producto
de `Z` indicado por un vector binario. Devolver `[expval(weights fijos), min C(θ)]`.

## Enfoque que funcionó

- `qp.PauliRot(θ_i, "X"*len(group), wires=group)` por cada grupo del patrón.
- Observable: producto con `@` de `qp.PauliZ(k)` donde `observable[k] == 1`.
- Optimización: `weights = ones`, `GradientDescentOptimizer(0.4)`, 200 pasos.
- **QNode solo construye circuito y mide**; la optimización va en `optimize_iqp`
  con una `lambda` que llama al QNode.

## Restricciones específicas

- Prohibido `qml.IQP`; el check lo detecta con `qp.specs(...)['resources']`.
- `atol=1e-3` para el expval intermedio y `atol=2e-2` para el mínimo.
- Solo celdas editables; `n_qubits`, `dev` y harness fijos.

## Errores relevantes

- `maximum recursion depth exceeded`: un `cost`/optimizador dentro del QNode hacía
  que el QNode se llamara a sí mismo.
- El harness local falla en `qp.specs` por la versión dev; en la plataforma estable
  funciona. No es un fallo de la solución.
- Restos de código viejo dentro del QNode: limpiar antes de enviar.

## Lecciones reutilizables

1. Los generadores `X` puro conmutan: el orden de los grupos no altera el estado.
2. Optimización determinista y barata (init en unos) suele bastar; evita aleatoriedad.
3. Si el check usa `specs`, la versión local puede mentir: contrasta con el código
   del upstream y no “arregles” lo que no está roto.
