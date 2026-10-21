# quantumHackatonLab — instrucciones de agente

Laboratorio propio (hackatón cuántica). cwd = raíz del lab (`.../HackatonCuantica/quantumHackatonLab`).
No es repo git; solo contiene `.venv`. Reglas globales de OpenCode (concisión, sin commits/push/PR sin petición
explícita, sin secretos en archivos, sin inventar URLs, preferencia de herramientas) siguen vigentes: aquí solo
se documenta lo específico del lab.

## Alcance de archivos

- Editar/crear únicamente dentro de `quantumHackatonLab/`.
- `../pennylane/` es upstream de **solo lectura**: no modificar código, docs, configs, tests ni su `AGENTS.md`.
  Está instalado editable en el `.venv`, así que cualquier edición upstream altera el entorno del lab.

## Ejecución desde el laboratorio

- Shell PowerShell; rutas con espacios (perfil OneDrive) siempre entre comillas.
- Ejecutar siempre desde la raíz del lab; usa el parámetro `workdir`, no cambies de directorio dentro del comando.
- Intérprete del lab: `.venv\Scripts\python.exe` (Python 3.12.10). Ejecutar con `uv run ...` o ese intérprete
  directo (`-m`). Nunca el Python del sistema.
- Imports: resolver siempre contra el `.venv` del lab. Verifica el origen antes de ejecutar
  (`.venv\Scripts\python.exe -c "import pennylane; print(pennylane.__file__)"` debe apuntar al `pennylane` del
  venv, no a un cwd accidental ni a `../pennylane` importado por ruta). No agregues `../pennylane` a
  `sys.path` ni a variables `PYTHONPATH`: el upstream no debe interferir con los imports del lab.
- Verifica con el shell antes de afirmar que algo no existe: `Test-Path .venv`. No uses glob sobre `.venv`.

## uv

- uv es el único gestor de entorno/dependencias: `uv run ...`, `uv pip install --python .venv\Scripts\python.exe <pkg>`,
  `uv pip list --python .venv\Scripts\python.exe`.
- No existe `pyproject.toml` ni `uv.lock`: no crearlos ni correr `uv init / sync / lock / add` sin aprobación
  explícita (generan cambios estructurales y fijan versiones).
- Instalar o actualizar dependencias requiere aprobación previa en esta fase, incluido `pytest`.

## Validación de APIs y consultas web

- **Prioridad de fuente para APIs de PennyLane:** primero el código local (`../pennylane/pennylane/`, `.venv`) y
  sus tests (`../pennylane/tests/`); la documentación web oficial es complemento, nunca sustituto.
- Distingue siempre la **versión local** (`pennylane.__version__` en el `.venv`, instalación editable) de la
  **versión documentada** en la web; anota ambas cuando las APIs no coincidan y ante la duda da crédito al código local.
- Al proponger código: verifica que las APIs usadas existan (lectura local o ejecución de un snippet) y aporta
  rutas concretas del repositorio (`../pennylane/pennylane/<modulo>.py`, tests incluidos) cuando sea posible.
- **Consultas web informativas** (leer documentación, guías, APIs públicas sin costo): permitidas sin aprobación.
- **Acciones que requieren aprobación previa:** llamadas que consuman créditos/cuota, usen credenciales, envíen
  datos a servicios externos o creen trabajos/jobs externos (backends en la nube, proxies LLM, CI, uploads).
- Credenciales: nunca en código, notebooks, logs ni este archivo; leer de variables de entorno o archivos locales
  ignorados. Si falta, pedirla; no adivinar ni reutilizar claves de otros proyectos.
- URLs solo si las proporciona el usuario, un archivo del repo o una búsqueda verificada.

## Pruebas

- `pytest` no está instalado. Si una herramienta necesaria no está, informa del bloqueo y solicita autorización
  antes de instalarla; nunca la instales por tu cuenta ni la eludas.
- Nunca afirmes que una prueba pasó sin haberla ejecutado y mostrado su output real.
- Correr solo las pruebas del cambio, con `.venv\Scripts\python.exe -m pytest <ruta> [-k <expr>]`.
- Toda función nueva o fix necesita prueba; localiza antes las pruebas existentes del mismo módulo.

## Retos hackatón (archivos con `run`/`check`/`test_cases`)

Patrón de todo el lab (adaptaciones de `pennylane.ai/challenges`, p. ej.
`APROXIMACION_RATIO.py`, `FermionicVQE.py`, `sucesionFases.py`, `keepyourall.py`).
Flujo obligatorio y en este orden:

1. **Leer PRIMERO el harness completo** (`run`, `check`, `test_cases`), no solo el enunciado: tolerancia exacta
   (`rtol`/`atol`), semillas fijas (p. ej. `np.random.seed(40)` en `FermionicVQE.py:78`) y chequeos estructurales
   (solo `qp.probs`, sin template IQP, etc.). El harness impone más restricciones que el enunciado.
2. **Identificar el reto original** con una búsqueda web (el enunciado suele estar copiado literal de
   `pennylane.ai/challenges`) y extraer su plantilla/test cases del payload de la página; el reto original revela
   qué compara realmente el grader (a veces lower bounds, no valores exactos).
3. **Calcular la banda de aceptación** (esperado ± tolerancia) y contrastarla de inmediato con el óptimo real
   (diagonalización/exhaustivo/optimización convergida; simular primero en numpy puro, es mucho más rápido que
   PennyLane). **Si el óptimo cae FUERA de la banda, la referencia sub-optimizó**: no optimices bien; calibra un
   presupuesto limitado (init fijo p. ej. `0.5`, `GradientDescentOptimizer(0.01)`, pocos pasos) hasta caer dentro
   de la banda en **todos** los casos públicos. Ejemplo real: en `APROXIMACION_RATIO.py` el óptimo p=4 da
   0.918/1.0 pero la banda era [0.55–0.61]/[0.90–0.99].
4. **Timebox**: si en 2–3 experimentos no se identifica la referencia exacta, cambiar a estrategia de banda; no
   perseguir la reproducción bit a bit del grader.

## Autonomía controlada

Sin pedir permiso: leer/explorar (lab y `../pennylane`), ejecutar código local y pruebas ya instaladas,
comandos uv de solo lectura (`--dry-run`, `list`, `run`), y crear/editar archivos dentro de `quantumHackatonLab/`.

Pedir permiso (una pregunta corta con la acción exacta): instalar/actualizar dependencias o tocar `.venv`;
crear `pyproject.toml`/`uv.lock`/`.git`; cualquier red que consuma cuota, use credenciales, envíe datos o cree
trabajos externos; modificar archivos fuera del lab; borrar o reescribir en bloque; iniciar commits.

## Fuera de alcance por ahora

No crear skills, subagentes, MCP, bases vectoriales ni pipelines RAG. Si se solicita, preguntar primero:
queda para una fase posterior.
