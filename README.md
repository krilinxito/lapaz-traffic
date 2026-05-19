# Tráfico LP — Análisis no supervisado de congestión vial

Dashboard interactivo para el análisis de tráfico en **La Paz y El Alto, Bolivia**, usando técnicas de aprendizaje automático no supervisado: PCA, UMAP, K-Means, DBSCAN, NMF y PCHIP.

## Estructura del proyecto

```
lapaz_traffic/
├── dashboard.py              # Servidor Flask + generación de mapas Folium
├── templates/
│   ├── base.html             # Layout base (diseño Mini Metro)
│   └── index.html            # Dashboard principal
├── data/
│   ├── raw/                  # Datos crudos de la Distance Matrix API
│   │   ├── traffic_lunes_HH.csv  # Mediciones reales (lunes, miércoles, viernes, sábado)
│   │   ├── lapaz_edges.csv       # Red vial La Paz (geometrías OSM)
│   │   ├── el_alto_edges.csv     # Red vial El Alto (geometrías OSM)
│   │   └── sample_segments.csv   # Segmentos monitoreados
│   └── processed/            # Salidas del pipeline ML
│       ├── feature_matrix.csv        # Matriz jam × hora (La Paz)
│       ├── feature_matrix_el_alto.csv
│       ├── segments_clustered.csv    # Segmentos con clusters K-Means + DBSCAN
│       └── ...
├── notebooks/                # Análisis exploratorio paso a paso
│   ├── 01_red_vial.ipynb
│   ├── 02_recoleccion.ipynb
│   ├── 03_preprocesamiento.ipynb
│   ├── 04_reduccion_dimensionalidad.ipynb
│   ├── 05_clustering.ipynb
│   └── 06_visualizacion.ipynb
├── scripts/                  # Pipeline reproducible
│   ├── 01_red_vial_el_alto.py
│   ├── 02_colectar_extension.py
│   ├── 03_rebuild_features.py
│   ├── 04_rebuild_clustering.py
│   └── 05_predict_missing_days.py
└── informe/                  # Informe académico en LaTeX
    ├── informe.tex
    ├── informe.pdf
    └── img/
```

## Instalación

```bash
pip install -r requirements.txt
```

## Configurar API key

Crear un archivo `.env` en la raíz del proyecto:

```
GOOGLE_MAPS_API_KEY=tu_api_key_aqui
```

Solo es necesaria para volver a recolectar datos (script `02_colectar_extension.py`). El dashboard funciona sin ella usando los datos procesados incluidos.

## Ejecutar el dashboard

```bash
python dashboard.py
```

Abrir `http://localhost:5000` en el navegador.

## Pipeline ML

El pipeline completo se puede reproducir ejecutando los scripts en orden:

```bash
python scripts/01_red_vial_el_alto.py    # Descarga red vial OSM de El Alto
python scripts/02_colectar_extension.py  # Recolecta datos (requiere API key)
python scripts/03_rebuild_features.py    # Construye matrices de features
python scripts/04_rebuild_clustering.py  # K-Means temporal + DBSCAN espacial
python scripts/05_predict_missing_days.py # NMF + PCHIP para días sintéticos
```

## Algoritmos aplicados

| Técnica | Uso |
|---------|-----|
| **PCA** | Reducción a 13 componentes (86–96% varianza) |
| **UMAP** | Proyección 2D para visualización de clusters |
| **K-Means** (k=2) | Clustering temporal: patrón alto/bajo |
| **DBSCAN** (ε=500 m) | Clustering espacial geográfico |
| **NMF** (k=6) | Síntesis de días faltantes (martes, jueves, domingo) |
| **PCHIP** | Interpolación suave en el eje día-de-semana |

## Datos

Los datos fueron recolectados con la **Google Maps Distance Matrix API** durante 4 días reales (lunes, miércoles, viernes, sábado), cubriendo 500 segmentos viales (250 La Paz + 250 El Alto) × 24 horas.
