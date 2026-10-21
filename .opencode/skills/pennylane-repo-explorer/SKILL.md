---
name: pennylane-repo-explorer
description: Explora y verifica la instalación local de PennyLane (repo ../pennylane, instalación editable en .venv del lab) para resolver problemas del hackatón. Localiza la implementación, pruebas y documentación de una API (device, template, transform, operator, measurement, qnode, arg), distingue símbolos internos de APIs exportadas, confirma comportamiento en ejecución y cita rutas concretas del repo. Usar cuando una pregunta sobre PennyLane requiera evidencia del código local en lugar de documentación web, o cuando haya que rastrear una API desde su implementación hasta sus tests y docs.
---

# pennylane-repo-explorer

Complementa `AGENTS.md` del lab (entorno, uv, aprobaciones, secretos, pruebas). Esta skill solo aporta el
método de exploración del repo upstream. Reglas globales de OpenCode siguen vigentes.

## Solo lectura del upstream (operativo)

- `../pennylane/` jamás se modifica: no usar tools de escritura (edit/write) sobre rutas de ese repo;
  en bash, solo comandos de lectura (`Get-Content`, `Select-String`, `git -C ../pennylane status|log|diff`).
- Verificación de integridad tras exploraciones largas: `git -C ../pennylane status --porcelain` debe
  quedar vacío.
- Si una petición exige modificar upstream, detenerse y remitir a `AGENTS.md`.

## Mapa del repo (referencia inicial, verificado en 0.46.0.dev129)

El siguiente mapa es un punto de partida: verificar con listado/grep cuando una ruta concreta importe
o cuando el repo haya cambiado.

- Código: `../pennylane/pennylane/` (es el paquete instalado editable). Submódulos útiles: `ops/`,
  `templates/`, `transforms/`, `devices/`, `measurements.py`, `math/`, `numpy/`, `tape/`, `workflow/`,
  `gradients/`, `qchem/`, `qnn/`, `noise/`, `io/`, `pauli/`, `operation.py`, `queuing.py`, `wires.py`.
- Pruebas: `../pennylane/tests/` (espejo por área: `ops/`, `templates/`, `transforms/`, `devices/`,
  `core/`, `tape/`, `math/`...). No todos los submódulos del paquete tienen espejo exacto.
- API docs (Sphinx): `../pennylane/doc/code/qp*.rst`. Guías temáticas: `../pennylane/doc/introduction/*.rst`.
- Cambios y deprecaciones: `../pennylane/doc/releases/changelog-dev.md` (+ changelogs por versión).
- Límites de imports entre capas: `../pennylane/tach.toml` (`pennylane.labs` y `pennylane.ftqc`
  restringidos: evitar usarlos en verificaciones runtime).
- Ejemplos canónicos: docstrings del propio código (testeados con Sybil vía `../pennylane/conftest.py`)
  y usos dentro de `tests/`. Este repo NO contiene demos/notebooks (las demos viven en un repo
  separado de PennyLane → solo web, con la cautela de versión de abajo).

## Símbolos internos, APIs exportadas y ejecución

Al reportar sobre un nombre, distingue tres niveles:

- **Símbolo interno**: definido en el código (`def`/`class` en un módulo) pero no re-exportado como
  API pública. No asumir que su uso está soportado.
- **API exportada públicamente**: alcanzable vía el namespace público (`qml.*` u otro import público
  estable). Confirmar el re-export leyendo `../pennylane/pennylane/__init__.py` o el `__init__.py`
  del submódulo correspondiente.
- **Funcionalidad comprobada en ejecución**: además de lo anterior, verificada importando desde el
  `.venv` del lab (y, cuando conviene, con un ejemplo mínimo ejecutable). Solo a este nivel afirmar
  "existe y funciona en la versión local instalada".

## Localizar una API: implementación → pruebas → docs

1. Implementación: grep de definiciones en el paquete — patrones `def <nombre>(` y `class <nombre>`
   dentro de `../pennylane/pennylane/`. Si no se conoce el área, grep del símbolo en
   `../pennylane/pennylane/__init__.py` (u otros `__init__.py`) para ver el re-export y el módulo
   de origen.
2. Pruebas: grep del símbolo en `../pennylane/tests/`; el test que lo usa es el comportamiento
   ejecutable de referencia (argumentos, casos límite, interfaces). Inspeccionar pruebas está
   permitido; NO ejecutar la suite completa ni instalar dependencias para ello sin autorización
   previa (ver `AGENTS.md`).
3. Docs locales: grep en `../pennylane/doc/code/*.rst` (ficha API) y `doc/introduction/*.rst` (guía);
   luego `changelog-dev.md` para saber desde qué versión existe o si está en deprecación.

## Verificar existencia y compatibilidad en la versión local

Orden de prioridad. Ejecutar siempre con el intérprete del lab. Patrones verificados en 0.46.0.dev129:

1. **Inspección de código y firma**: leer la definición en `../pennylane/pennylane/` y su
   `__init__.py`; `inspect.signature` y `inspect.getsource` (ventana corta) contra el módulo de origen.
2. **Importación desde el `.venv`**: `.venv\Scripts\python.exe -c "import pennylane as qml; print(qml.__version__, qml.__file__)"`;
   después `hasattr(qml, 'nombre')` / `getattr(qml, 'nombre', None)`, igual para submódulos
   (`qml.devices.default_qubit`, `qml.testing.assert_valid` existen hoy). El import debe resolver
   contra el venv del lab (regla de `AGENTS.md`), nunca vía `sys.path`/`PYTHONPATH` manual.
3. **Ejemplo mínimo ejecutable** cuando sea pertinente (snippet pequeño con `default.qubit`,
   local, sin red, sin installs) para confirmar el comportamiento real (p. ej., argumentos aceptados).

Nota: `qml.plugin_devices` existe pero puede ser legacy: contrastar con el código fuente antes de
recomendarlo.

## Discrepancias entre fuentes

Si la documentación local (`doc/`), el código, las pruebas y el comportamiento en ejecución no
coinciden, describir la discrepancia explícitamente: qué dice cada fuente (con ruta/cita), en qué
versión local se observó y cuál se recomienda seguir (prevalece el código local instalado; registrar
también la versión documentada cuando provenga de la web, según `AGENTS.md`).

## Ejemplos compatibles con la versión local

Prioridad: (1) docstring de la función en el repo local; (2) patrón copiado de `tests/` local;
(3) `doc/introduction/*.rst` local; (4) docs/demos web solo como complemento, señalando que pueden
corresponder a otra versión. Si web y local difieren, manda el local y anota ambas versiones.

## Evidencia, citas y límites

- Toda conclusión lleva cita: `../pennylane/<ruta>:<línea>` o `"<símbolo>" en <ruta>`, obtenida con
  grep, read o check runtime. Etiquetar: VERIFICADO LOCAL (leído o ejecutado aquí), INFERIDO
  (deducido de nombres/patrones), NO CONFIRMADO.
- Sin evidencia suficiente: declararlo, listar qué se buscó y proponer el siguiente paso (grep
  adicional, check runtime, preguntar al usuario). Nunca inventar APIs, firmas ni rutas.

## Economía de contexto

No volcar el repo al contexto: grep con `-n` y `read` con offset/limit sobre ventanas concretas;
listados de directorio acotados; no pegar archivos enteros grandes; varias búsquedas dirigidas
en vez de leer módulos completos.
