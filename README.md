# Tráfico LP — Análisis no supervisado de congestión vial

Análisis de la congestión vehicular en **La Paz y El Alto** a partir de un dataset propio: mediciones
horarias de tiempos de viaje recolectadas con la Google Distance Matrix API sobre la red vial de
OpenStreetMap. No hay datos etiquetados ni encuestas — todo el conocimiento sale de técnicas no
supervisadas.

**96 mediciones horarias × 250 segmentos viales**, en lunes, miércoles, viernes y sábado.

## Qué se encontró

- **La congestión severa de La Paz es estructural, no dispersa.** El **10,8 %** de las arterias
  monitoreadas concentra los patrones de alta congestión, sobre todo en el centro histórico y los
  accesos a la autopista. Intervenir ahí rinde más que repartir esfuerzo por toda la ciudad.
- **96 dimensiones se comprimen a 13** conservando entre el 86 % y el 96 % de la información, lo que
  confirma que los perfiles horarios son mucho menos variados de lo que parecen.
- Cada técnica aporta algo distinto: PCA reduce ruido, UMAP hace visible la separación en 2D, DBSCAN
  encuentra vecindades geográficas sin fijar `k` de antemano, K-Means segmenta perfiles temporales
  interpretables, y NMF + PCHIP completan los días no medidos respetando que el `jam_factor` no puede
  ser negativo.

El dashboard etiqueta explícitamente qué datos son medidos y cuáles reconstruidos, para no presentar
una estimación como si fuera una observación.

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
