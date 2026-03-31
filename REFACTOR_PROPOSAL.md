# Propuesta de Refactorización — MAFPIN

> Documento de análisis y propuesta. Ningún archivo ha sido modificado aún.

---

## 1. Diagnóstico del estado actual

### 1.1 Estructura de archivos actual

```text
mafpin/
├── networks/
│   ├── generate_cascades.py          ← lógica + CLI integradas
│   ├── calculate_delta.py            ← math + I/O + visualización mezclados
│   ├── infer_networks.py             ← lógica de inferencia
│   ├── run_inference.py              ← CLI wrapper de infer_networks.py
│   ├── centrality_metrics.py         ← funciones SNAP individuales
│   ├── calculate_centrality_metrics.py ← orquestación + I/O
│   ├── run_centrality_metrics.py     ← CLI wrapper de calculate_centrality_metrics.py
│   ├── calculate_communities.py      ← detección de comunidades + LPH
│   ├── run_communities.py            ← CLI wrapper de calculate_communities.py
│   └── plot_communities.py           ← visualizaciones de comunidades
├── matrix_factorization/
│   ├── utils.py                      ← carga de datos + splitting
│   ├── cmf.py                        ← CMF baseline + búsqueda de hiperparámetros
│   ├── cmf_centrality.py             ← CMF con atributos de red
│   └── model_plots.py                ← visualizaciones del modelo
└── data/ plots/ requirements.txt ...
```

### 1.2 Problemas identificados

#### P1 — Patrón "lógica + runner" duplicado (impacto: ALTO)

Cada módulo con lógica tiene un archivo `run_*.py` que solo agrega `argparse`. Esto doubles el número de archivos sin añadir valor real:

| Lógica | Runner | Valor diferencial |
|---|---|---|
| `infer_networks.py` | `run_inference.py` | Solo argparse |
| `calculate_centrality_metrics.py` | `run_centrality_metrics.py` | Solo argparse |
| `calculate_communities.py` | `run_communities.py` | Solo argparse |

La solución es fusionar lógica + CLI en el mismo archivo y proteger la ejecución directa con `if __name__ == '__main__'`.

#### P2 — Rutas relativas frágiles (impacto: ALTO)

Todos los scripts usan `os.path.join('..', 'data', ...)`. Esto impone una restricción invisible: **los scripts DEBEN ejecutarse desde su directorio padre** (`networks/` o `matrix_factorization/`). Si se ejecutan desde la raíz del proyecto, todo falla sin ningún mensaje de error claro.

```python
# Ejemplo actual — frágil
cascades_path = os.path.join('..', 'data', cascade_file)
```

#### P3 — Sin paquete Python (impacto: ALTO)

No hay `__init__.py` en ningún directorio. Los módulos no son importables entre sí de forma estándar. Las importaciones funcionan solo gracias al `sys.path` implícito del directorio de trabajo actual.

#### P4 — Cargador de red duplicado (impacto: MEDIO)

La lógica para parsear los archivos `.txt` de redes inferidas (detectar nodos como self-loops `i,i` y edges como `i,j`) está duplicada en dos funciones diferentes que usan librerías distintas:

- `load_inferred_network_snap()` en `calculate_centrality_metrics.py` → usa SNAP
- `load_inferred_network_nx()` en `calculate_communities.py` → usa NetworkX

El parsing del archivo es idéntico; solo cambia la construcción del grafo. Esto viola DRY y cualquier cambio en el formato del archivo requiere modificar dos lugares.

#### P5 — Mezcla de responsabilidades en `calculate_delta.py` (impacto: MEDIO)

Este archivo contiene tres responsabilidades completamente distintas:
1. **Visualización**: `plot_cascades_timestamps()` — genera plots
2. **I/O + cómputo**: `compute_median_delta()` — lee archivo y calcula estadísticas
3. **Math puro**: `alpha_centers_from_delta()` y `log_alpha_grid()` — funciones sin efectos secundarios

Las funciones matemáticas son reutilizables y testeables de forma aislada; mezclarlas con I/O las hace más difíciles de probar.

#### P6 — Sin pipeline unificado (impacto: ALTO)

Para ejecutar el flujo completo hay que abrir 4 terminales, cambiar de directorio y recordar el orden correcto:

```bash
cd networks && python generate_cascades.py ratings_small
cd networks && python run_inference.py --all-models --N 100
cd networks && python run_centrality_metrics.py --all-models
cd networks && python run_communities.py --all-models
cd matrix_factorization && python cmf_centrality.py
```

No existe ningún punto de entrada único que ejecute toda la pipeline o pasos seleccionados.

#### P7 — API verbosa en métricas de centralidad (impacto: BAJO)

Todas las funciones de `centrality_metrics.py` tienen la misma firma de 5 argumentos de los cuales 4 son siempre opcionales para plots:

```python
def calculate_degree_snap(G, plot_enabled=False, model_name="", network_id="", output_dir=""):
def calculate_betweenness_snap(G, plot_enabled=False, model_name="", network_id="", output_dir=""):
# ... lo mismo para las 5 funciones restantes
```

Un objeto de contexto o parámetros `**plot_kwargs` simplificaría la firma.

#### P8 — Módulos de visualización dispersos (impacto: BAJO)

Las funciones de visualización están distribuidas en tres lugares sin una estructura coherente:
- `networks/centrality_metrics.py` → plots de distribuciones de centralidad
- `networks/plot_communities.py` → plots de comunidades
- `matrix_factorization/model_plots.py` → plots del modelo CMF

---

## 2. Estructura propuesta

```text
mafpin/
│
├── config.py                      ← NUEVO: configuración centralizada de rutas y defaults
├── pipeline.py                    ← NUEVO: entrada única para toda la pipeline
│
├── networks/
│   ├── __init__.py                ← NUEVO
│   ├── cascades.py                ← RENOMBRADO de generate_cascades.py
│   ├── delta.py                   ← RENOMBRADO de calculate_delta.py (solo math + I/O)
│   ├── network_io.py              ← NUEVO: parser unificado de archivos .txt de redes
│   ├── inference.py               ← FUSIÓN de infer_networks.py + run_inference.py
│   ├── centrality.py              ← FUSIÓN de centrality_metrics.py + calculate_centrality_metrics.py
│   └── communities.py             ← FUSIÓN de calculate_communities.py + run_communities.py
│
├── recommender/
│   ├── __init__.py                ← NUEVO
│   ├── data.py                    ← RENOMBRADO de utils.py
│   ├── baseline.py                ← RENOMBRADO de cmf.py
│   └── enhanced.py                ← RENOMBRADO de cmf_centrality.py
│
├── visualization/
│   ├── __init__.py                ← NUEVO
│   ├── network_plots.py           ← MOVIDO: distribuciones de centralidad
│   ├── community_plots.py         ← MOVIDO de networks/plot_communities.py
│   └── model_plots.py             ← MOVIDO de matrix_factorization/model_plots.py
│
└── data/ plots/ requirements.txt ...
```

**Resumen de cambios:**
- 12 archivos actuales → 11 archivos propuestos (estructura más plana y coherente)
- 6 archivos `run_*.py` + lógica → 3 archivos con CLI integrada
- 0 `__init__.py` → 4 `__init__.py` (paquete importable)
- 0 config centralizada → 1 `config.py`
- 0 entry point unificado → 1 `pipeline.py`

---

## 3. Cambios detallados por archivo

### 3.1 `config.py` (nuevo)

**Propósito:** Eliminar las rutas hardcodeadas con `../` dispersas por todo el código. Un único lugar para cambiar rutas, nombres de modelos y defaults de parámetros.

```python
# config.py (propuesto)
from pathlib import Path

ROOT = Path(__file__).parent  # siempre apunta a la raíz del proyecto

class Paths:
    DATA        = ROOT / "data"
    PLOTS       = ROOT / "plots"
    CASCADES    = DATA / "cascades.txt"
    NETWORKS    = DATA / "inferred_networks"
    CENTRALITY  = DATA / "centrality_metrics"
    COMMUNITIES = DATA / "communities"

class Models:
    ALL    = ["exponential", "powerlaw", "rayleigh"]
    SHORT  = {"exponential": "expo", "powerlaw": "power", "rayleigh": "ray"}

class Defaults:
    N_ALPHAS   = 100
    RANGE_R    = 100.0
    MAX_ITER   = 2000
    EPSILON    = 0.25
    MIN_COM    = 3
    K          = 20
    LAMBDA_REG = 1.0
```

**Impacto:** Todos los `os.path.join('..', 'data', ...)` serían reemplazados por `Paths.DATA / "..."`, y los scripts se podrían ejecutar desde cualquier directorio.

---

### 3.2 `networks/network_io.py` (nuevo)

**Propósito:** Fuente única de verdad para leer el formato de archivo `.txt` de redes inferidas. Actualmente hay dos parsers duplicados.

```python
# networks/network_io.py (propuesto)
def parse_network_file(network_file: Path) -> tuple[list[int], list[tuple[int, int]]]:
    """
    Parsea un archivo de red inferida por NetInf.
    Nodos codificados como self-loops (i,i); edges como (i,j) con i != j.
    
    Retorna: (lista de node_ids originales, lista de edges)
    """
    nodes, edges = [], []
    with open(network_file, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            i, j = map(int, line.split(","))
            if i == j:
                nodes.append(i)
            else:
                edges.append((i, j))
    return nodes, edges


def load_as_snap(network_file: Path):
    """Carga la red como grafo SNAP (para métricas de centralidad)."""
    nodes, edges = parse_network_file(network_file)
    mapper = {old: new for new, old in enumerate(sorted(set(nodes)), start=1)}
    G = snap.TUNGraph.New()
    for u in mapper.values():
        G.AddNode(u)
    for i, j in edges:
        if i in mapper and j in mapper:
            G.AddEdge(mapper[i], mapper[j])
    return G, list(mapper.values())


def load_as_networkx(network_file: Path):
    """Carga la red como grafo NetworkX (para detección de comunidades)."""
    nodes, edges = parse_network_file(network_file)
    mapper = {old: new for new, old in enumerate(sorted(set(nodes)), start=1)}
    G = nx.Graph()
    G.add_nodes_from(mapper.values())
    for i, j in edges:
        if i in mapper and j in mapper:
            G.add_edge(mapper[i], mapper[j])
    return G, list(mapper.values())
```

**Impacto:** Si el formato del archivo cambia, solo se toca un lugar. Se elimina ~40 líneas de código duplicado.

---

### 3.3 Fusión de `calculate_centrality_metrics.py` + `run_centrality_metrics.py` → `networks/centrality.py`

**Propósito:** El archivo `run_centrality_metrics.py` solo tiene `argparse` e invoca funciones de `calculate_centrality_metrics.py`. No justifica existir por separado.

**Antes:**
```
calculate_centrality_metrics.py  (lógica, 250 líneas)
run_centrality_metrics.py        (CLI, 150 líneas de argparse)
```

**Después:**
```python
# networks/centrality.py
# ... toda la lógica de calculate_centrality_metrics.py ...
# ... funciones individuales de centralidad de centrality_metrics.py ...

def main():
    parser = argparse.ArgumentParser(...)
    # ... argparse de run_centrality_metrics.py ...

if __name__ == "__main__":
    main()
```

**Lo mismo aplica para:**
- `infer_networks.py` + `run_inference.py` → `networks/inference.py`  
- `calculate_communities.py` + `run_communities.py` → `networks/communities.py`

---

### 3.4 Separación de responsabilidades en `calculate_delta.py` → `networks/delta.py`

**Propósito:** Las funciones matemáticas (`alpha_centers_from_delta`, `log_alpha_grid`) son puras y testeables; mezclarlas con I/O y visualización las hace opaques.

**Reorganización:**

| Función actual | Destino propuesto |
|---|---|
| `plot_cascades_timestamps()` | `visualization/network_plots.py` |
| `compute_median_delta()` | `networks/delta.py` (I/O + cómputo) |
| `alpha_centers_from_delta()` | `networks/delta.py` (math puro) |
| `log_alpha_grid()` | `networks/delta.py` (math puro) |

---

### 3.5 Módulo `recommender/` (reorganización de `matrix_factorization/`)

**Renombramiento con intención más clara:**

| Archivo actual | Propuesto | Razón |
|---|---|---|
| `utils.py` | `recommender/data.py` | "utils" no dice qué hace; es carga y preprocesado de datos |
| `cmf.py` | `recommender/baseline.py` | Explicita que es el modelo base sin atributos de red |
| `cmf_centrality.py` | `recommender/enhanced.py` | Explicita que es el modelo mejorado |
| `model_plots.py` | `visualization/model_plots.py` | Centralizar visualizaciones |

El código interno de estos archivos no necesita cambiar, solo reorganizarse.

---

### 3.6 `pipeline.py` (nuevo, alta prioridad)

**Propósito:** Punto de entrada único para ejecutar la pipeline completa o pasos seleccionados. Actualmente hay que ejecutar 5 comandos en 3 directorios distintos.

```python
# pipeline.py (propuesto)
"""
Entrada única para la pipeline MAFPIN completa.

Ejemplos:
    python pipeline.py --all                           # Pipeline completa
    python pipeline.py --steps cascades inference      # Solo pasos 1 y 2
    python pipeline.py --dataset ratings_small --model exponential
    python pipeline.py --steps evaluate --model exponential
"""
import argparse
from config import Paths, Defaults

STEPS = ["cascades", "inference", "centrality", "communities", "evaluate"]

def run_cascades(dataset: str): ...
def run_inference(model: str, n: int, r: float): ...
def run_centrality(model: str): ...
def run_communities(model: str): ...
def run_evaluate(model: str): ...

def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--steps", nargs="+", choices=STEPS)
    parser.add_argument("--dataset", default="ratings_small")
    parser.add_argument("--model", choices=["exponential", "powerlaw", "rayleigh", "all"])
    # ...
```

---

### 3.7 `visualization/` (reorganización)

Centralizar todos los plots en un solo módulo:

```
visualization/
├── __init__.py
├── network_plots.py    ← distribuciones de centralidad + plot_cascades_timestamps
├── community_plots.py  ← todo lo de plot_communities.py
└── model_plots.py      ← todo lo de matrix_factorization/model_plots.py
```

---

## 4. Plan de implementación por prioridad

### Prioridad 1 — Correctivo (no cambia funcionalidad, elimina fragilidad)

Estos cambios reducen riesgo de errores y mejoran la mantenibilidad sin tocar lógica:

1. **Crear `config.py`** con `pathlib.Path` — elimina las rutas relativas frágiles
2. **Añadir `__init__.py`** a `networks/` y `matrix_factorization/` (o `recommender/`)
3. **Crear `networks/network_io.py`** — elimina el parser duplicado

Esfuerzo estimado: bajo. Impacto: alto.

### Prioridad 2 — Estructural (reorganización de archivos)

4. **Fusionar runners y lógica**:
   - `calculate_centrality_metrics.py` + `run_centrality_metrics.py` → `networks/centrality.py`
   - `infer_networks.py` + `run_inference.py` → `networks/inference.py`
   - `calculate_communities.py` + `run_communities.py` → `networks/communities.py`
5. **Mover `centrality_metrics.py`** — sus funciones de métricas individuales pasan a ser parte de `networks/centrality.py`
6. **Crear `visualization/`** y mover/consolidar los módulos de plots

Esfuerzo: medio. Impacto: medio-alto.

### Prioridad 3 — Pipeline y experiencia de uso

7. **`pipeline.py`** — entry point unificado con `argparse`
8. **Renombrar `matrix_factorization/` a `recommender/`** y sus archivos internos

Esfuerzo: medio. Impacto: alto (mejora la experiencia de uso significativamente).

### Prioridad 4 — Mejoras opcionales

9. **Separar math puro de I/O en `calculate_delta.py`** (el `compute_median_delta` podría recibir datos ya parseados en lugar de un path)
10. **Simplificar API de funciones de centralidad** — agrupar los 4 argumentos de plot en un `dataclass PlotContext` o patrón similar

---

## 5. Tabla resumen de cambios

| Archivo actual | Acción | Archivo propuesto |
|---|---|---|
| `networks/generate_cascades.py` | Renombrar | `networks/cascades.py` |
| `networks/calculate_delta.py` | Renombrar + separar viz | `networks/delta.py` |
| `networks/infer_networks.py` | **Fusionar** con runner | `networks/inference.py` |
| `networks/run_inference.py` | **Eliminar** (fusionado) | — |
| `networks/centrality_metrics.py` | **Fusionar** con orquestador | `networks/centrality.py` |
| `networks/calculate_centrality_metrics.py` | **Fusionar** con runner | `networks/centrality.py` |
| `networks/run_centrality_metrics.py` | **Eliminar** (fusionado) | — |
| `networks/calculate_communities.py` | **Fusionar** con runner | `networks/communities.py` |
| `networks/run_communities.py` | **Eliminar** (fusionado) | — |
| `networks/plot_communities.py` | **Mover** | `visualization/community_plots.py` |
| `matrix_factorization/utils.py` | Renombrar | `recommender/data.py` |
| `matrix_factorization/cmf.py` | Renombrar | `recommender/baseline.py` |
| `matrix_factorization/cmf_centrality.py` | Renombrar | `recommender/enhanced.py` |
| `matrix_factorization/model_plots.py` | **Mover** | `visualization/model_plots.py` |
| *(no existe)* | **Crear** | `config.py` |
| *(no existe)* | **Crear** | `networks/network_io.py` |
| *(no existe)* | **Crear** | `pipeline.py` |
| *(no existe)* | **Crear** | `visualization/network_plots.py` |

**Resultado:** 14 archivos existentes → 11 archivos + estructura de paquete correcta.

---

## 6. Qué NO cambiar

Algunos elementos están bien y no deben tocarse:

- **La lógica interna de `cmf.py` y `cmf_centrality.py`**: las funciones de entrenamiento, CV y evaluación están bien estructuradas.
- **La integración de comunidades en `load_centrality_metrics`**: el parámetro `include_communities=True` es una solución pragmática, limpia para el caso de uso.
- **Los nombres de columnas en los CSV** del directorio `data/`: cambiarlos rompería compatibilidad con resultados ya generados.
- **El binario `netinf`**: es una dependencia externa, no Python.
- **`requirements.txt`**: está bien tal cual.

---

## 7. Árbol final de archivos propuesto

```
mafpin/
│
├── config.py                       ← NUEVO
├── pipeline.py                     ← NUEVO
│
├── networks/
│   ├── __init__.py                 ← NUEVO
│   ├── cascades.py                 ← (era generate_cascades.py)
│   ├── delta.py                    ← (era calculate_delta.py)
│   ├── network_io.py               ← NUEVO
│   ├── inference.py                ← FUSIÓN infer_networks + run_inference
│   ├── centrality.py               ← FUSIÓN centrality_metrics + calculate_centrality_metrics + run_centrality_metrics
│   └── communities.py              ← FUSIÓN calculate_communities + run_communities
│
├── recommender/
│   ├── __init__.py                 ← NUEVO
│   ├── data.py                     ← (era matrix_factorization/utils.py)
│   ├── baseline.py                 ← (era matrix_factorization/cmf.py)
│   └── enhanced.py                 ← (era matrix_factorization/cmf_centrality.py)
│
├── visualization/
│   ├── __init__.py                 ← NUEVO
│   ├── network_plots.py            ← distribuciones de centralidad + cascades timeline
│   ├── community_plots.py          ← (era networks/plot_communities.py)
│   └── model_plots.py              ← (era matrix_factorization/model_plots.py)
│
├── data/
├── plots/
├── requirements.txt
└── README.md
```
