# Retos PennyLane — inventario y estado

Actualizado: 2026-10-21. Fuente: archivos del laboratorio y sesiones de OpenCode
(`~/.local/share/opencode/opencode.db`, consultado en solo lectura).
Leyenda: **Verificado** = harness ejecutado con salida real; **Parcial** = pasa lo
público pero falta evidencia; **Desconocido** = sin contenido o sin evaluar.

## Retos

| Reto | Archivo | Familia | Estado | Evidencia | Restricciones clave del evaluador |
|---|---|---|---|---|---|
| GHZ + paridad | `reto_ghz_pennylane.py` | Certificación / estabilizadores | Verificado local (réplica del grader) | `Correct! 0.9999999999999997` para n=3 y n=5 | Solo `expval`; audita observables y cintas; `atol=1e-5`; `default.qubit` fijo; solo bloque editable |
| VQE fermiónico | `FermionicVQE.py` | Química cuántica / VQE | Verificado | `-2.0` y `-6.0`; extras `[1,2,3]→0`, `[-1,-2,-3]→-6`, `[0,0,0]→0` | `atol=0.1`; coeficientes reales (Jordan-Wigner); semilla fija `40`; no tocar `dev` ni funciones dadas |
| IQP / sucesión de fases | `sucesionFases.py` | Optimización variacional / IQP | Verificado local; plataforma declarada por el usuario | `0.7306816499355087` y `0.93629336358419`; mínimo `-1.0` (~1e-14) | `atol=1e-3` (expval) y `2e-2` (mínimo); prohíbe `qml.IQP` (lo detecta `specs`); solo celdas editables |
| Pauli desde `probs` | `keepyourall.py` | Medición restringida | Verificado | `0.931986472711103` y `0.9673999542626387`; `III→1.0` | Solo `qp.probs`; audita cintas y `shots is None`; mediciones en base Z; `atol=1e-5` |
| Tomografía 1 cúbit | `informe_exploracion.py` | Medición restringida / tomografía | Verificado | Ambos públicos `Correct!`; Bell → `[[0,0,0], 0.5]` | Solo `expval` sobre 3 observables; `atol=1e-5`; `dev` y pureza dados |
| QAOA vertex cover | `APROXIMACION_RATIO.py` (plantilla) / `APROXIMACION_RATIO_solucion.py` | Optimización variacional / QAOA | Verificado local (2 públicos); ocultos sin probar; no persistido en la plantilla | `0.5816092357581847` y `0.9474673396983472` (21-oct-2026) | `rtol=5e-2`; retorno `np.tensor`; solo bloque editable; referencia sub-optimizada (banda, no óptimo) |
| QMC canasta de opciones | `ABasket_OF_options.py` | Finanzas cuánticas / estimación | Verificado | `10.556134590137555` vs `10.556134590137553`; `6.487093067342844`; extras ~1e-15 | `atol=1e-3`; orden de entrada `[rho, w, T, K]` (distinto al enunciado); 64 escenarios; plantilla con variables fijas |
| QSVT difusión-reacción | `Diffuse and Conquer.py` | Álgebra lineal / QSVT | Verificado | Públicos `Correct!`; 54 casos de estrés, peor fidelidad `0.99975`, `TOTAL FAILS: 0` | Fidelidad `≥0.99`; `C>0`; `phi/source/dt/h` fijos; no editar el bloque dado; 4 wires |
| Goal or No Goal | `Grupo1/Goal or No Goal.ipynb` | Desconocida | Desconocido | Archivo de 0 bytes (21-oct 17:25), sin celdas | n/a |

## Artefactos de apoyo

| Archivo | Para qué sirve |
|---|---|
| `APROXIMACION_RATIO_solucion.py` | Solución QAOA recuperada del chat + procedencia y pruebas (plantilla intacta) |
| `QAOA_scratch.py` | Diagonalización exacta de QAOA p=1..4: demuestra que el óptimo cae fuera de la banda |
| `scratch_conv.py`, `scratch_exact.py`, `scratch_final.py`, `scratch_interf.py`, `scratch_pipeline.py`, `scratch_qsvt_explore.py`, `scratch_stress.py` | Exploración QSVT: coeficientes exactos, norma QSP, interferencia y estrés |
| `scratch_qaoa.py` | Intento QAOA con L-BFGS (descartado: sobre-optimiza) |
| `scratch_final_check.py` | Verificación QSVT contra solución clásica (24 casos, 0 fallos) |
| `recordatorio_iqp.md` | Detalle del reto IQP y su verificación |
| `notas/QAOA.md`, `notas/IQP.md`, `notas/QSVT.md` | Notas breves reutilizables (objetivo, enfoque, trampas, lecciones) |
| `__pycache__/` | Solo evidencia de importación/ejecución; no editar |

## Patrones transversales

1. **Leer el harness primero**: cada reto trae tolerancias, semillas y auditorías distintas.
2. **`default.qubit` analítico, sin shots**, salvo indicación; varias soluciones auditan las cintas.
3. **Solo se editan bloques delimitados**: docstrings, `dev`, variables dadas y harness son inmutables.
4. **QNode = circuito + medición**; el optimizador va fuera (meterlo dentro causó recursión en IQP).
5. **Pre-simular en numpy** para obtener el valor de referencia y la banda antes de optimizar.
6. **La referencia puede estar sub-optimizada**: en QAOA el óptimo real queda fuera de la banda; calibrar el presupuesto, no maximizar.
7. **Versión local dev (0.46.0-dev129)** frente a la estable de la plataforma: `qml.qaoa` no existe localmente y `qp.specs` sobre QNode desnudo falla; no son fallos de la solución.
8. **Convenciones**: wire 0 = bit más significativo; el orden de entradas puede diferir del enunciado (QMC).

## Sesiones de referencia

| Fecha | Sesión | Contenido |
|---|---|---|
| 18-oct 16:47 | `ses_eaf04017…` (+4 `@explore`) | Mapeo del repo PennyLane |
| 18-oct 17:40 | `ses_eaed3ee1…` | Creación de `AGENTS.md` |
| 18-oct 18:37 | `ses_eae9faa6…` | Skill y `qml.qnode` |
| 18-oct 20:03 | `ses_eae50f91…` | GHZ y FermionicVQE |
| 19-oct 13:36 | `ses_eaa8c2e0…` | IQP |
| 19-oct 13:52 | `ses_eaa7dacb…` | keepyourall |
| 19-oct 13:57 | `ses_eaa793b3…` | Tomografía |
| 19-oct 14:02 | `ses_eaa74ead…` | QAOA (solución aportada por el usuario) |
| 19-oct 14:05 | `ses_eaa720bc…` | QMC |
| 19-oct 14:10 | `ses_eaa6dc78…` | QSVT |
| 19-oct 14:49 | `ses_eaa4976c…` | Fork de QAOA |

## Uso rápido

1. Localiza el reto en la tabla y abre su archivo.
2. Lee `run`/`check`/`test_cases` antes de tocar código.
3. Consulta `notas/<RETO>.md` para la estrategia y las trampas.
4. Reutiliza los scratch solo como referencia; no los importes desde una solución.
