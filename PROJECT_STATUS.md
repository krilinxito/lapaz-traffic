# PROJECT STATUS — Análisis de congestión vehicular en La Paz

## ✅ FASE 1 — Verificación del entorno (COMPLETADA)

### Lo que se hizo
- Verificado Python 3.14.4 instalado
- Creado entorno virtual `venv_traffic/`
- Instaladas todas las dependencias correctamente

### Dependencias instaladas
| Librería | Versión |
|---|---|
| requests | 2.34.2 |
| pandas | 3.0.3 |
| geopandas | 1.1.3 |
| osmnx | 2.1.0 |
| scikit-learn | 1.8.0 |
| umap-learn | 0.5.12 |
| matplotlib | 3.10.9 |
| folium | 0.20.0 |
| streamlit | 1.57.0 |
| python-dotenv | OK |

### Archivos creados
```
lapaz_traffic/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_red_vial.ipynb
│   ├── 02_recoleccion.ipynb
│   ├── 03_preprocesamiento.ipynb
│   ├── 04_reduccion_dimensionalidad.ipynb
│   ├── 05_clustering.ipynb
│   ├── 06_visualizacion.ipynb
│   └── 07_dashboard.ipynb
└── PROJECT_STATUS.md
```

---

## ✅ FASE 2 — Red vial de La Paz (OSMnx) (COMPLETADA)

### Lo que se hizo
- Descargada la red vial vehicular de La Paz desde OpenStreetMap con `osmnx 2.1.0`
- Ejecutado el notebook `notebooks/01_red_vial.ipynb` sin errores

### Estadísticas de la red
| Métrica | Valor |
|---|---|
| Nodos (intersecciones) | 14,650 |
| Aristas (segmentos viales) | 20,438 |
| Longitud total de la red | 2,167.8 km |
| Tipos de vía únicos | 21 |

### Archivos generados
- `data/raw/lapaz_network.gpkg` — 7.8 MB — grafo completo en GeoPackage
- `data/raw/lapaz_edges.csv` — 8.0 MB — aristas con columnas: osmid, name, highway, length, geometry, u, v
- `data/raw/lapaz_network_preview.png` — 618 KB — mapa estático de verificación

---

## ✅ FASE 3 — Configuración API de tráfico (COMPLETADA)

### Lo que se hizo
- Descartado HERE Traffic API (requiere tarjeta + bug de cookies SameSite)
- Descartado TomTom Traffic API (sin cobertura para Bolivia)
- Configurada **Google Maps Distance Matrix API** (Free Trial $300 créditos, proyecto `La-Paz-Traffic`)
- Verificada llamada de prueba exitosa: La Paz → status OK, `duration_in_traffic` presente
- `GMAPS_API_KEY` guardada en `lapaz_traffic/.env` (nunca en el código)
- Instalado `googlemaps==4.10.0` en el entorno virtual

### Fórmula de jam_factor
```python
jam_factor = max(0, min(10, (duration_in_traffic / duration - 1) * 10))
```

---

## ✅ FASE 4 — Recolección de datos (COMPLETADA)

### Lo que se hizo
- Construido `notebooks/02_recoleccion.ipynb` con funciones `get_traffic_flow` y `collect_sample`
- Seleccionados **250 segmentos** estratificados por tipo de vía de la red OSMnx
- Ejecutada recolección de prueba con 3 intervalos

### Estadísticas de la recolección de prueba
| Métrica | Valor |
|---|---|
| Segmentos por intervalo | 250 |
| Intervalos ejecutados | 3 (+ 3 de pruebas anteriores = 6 total) |
| jam_factor promedio | ~0.35 |
| Velocidad promedio | ~18.3 km/h |
| jam_factor en rango [0,10] | ✅ 100% |

### Archivos generados
- `data/raw/sample_segments.csv` — 250 segmentos con coordenadas de nodos u/v
- `data/raw/traffic_2026-05-18_*.csv` — 6 archivos CSV (250 segmentos cada uno)
- `data/raw/traffic_sample_preview.png` — distribución de jam_factor y boxplot por tipo de vía

### Cuota API consumida
- ~1,500 elementos (6 intervalos × 250 segmentos)
- Saldo restante: $300 free trial disponible

### Próximo paso
Para el clustering necesitás **2-3 semanas de datos** cubriendo distintos horarios. Podés continuar recolectando con `collect_sample(client, segments_df, n_intervals=N)`.

---

## ✅ FASE 5 — Preprocesamiento (COMPLETADA)

### Lo que se hizo
- Cargados 72 archivos históricos (lunes/viernes/sábado × 24h × 250 segmentos)
- Construida matriz de features: pivot por `(segment_id, tipo_dia, hora)` → jam_factor
- Normalización: SimpleImputer (media) + StandardScaler
- Validados patrones esperados: rush 7-9h > madrugada, lunes > sábado

### Estadísticas
| Métrica | Valor |
|---|---|
| Shape feature_matrix | (250, 72) |
| NaN en matriz | 0 |
| Media post-scaling | 0.000000 |
| Std post-scaling | 1.000000 |
| Lunes 8h jam promedio | 0.201 |
| Sábado 8h jam promedio | 0.049 |

### Archivos generados
- `data/processed/feature_matrix.csv` — (250, 72) sin normalizar
- `data/processed/feature_matrix_scaled.csv` — (250, 72) normalizada
- `data/processed/scaler.pkl` — StandardScaler fitted
- `data/processed/preprocessing_report.png` — distribuciones y perfiles temporales

---

## ✅ FASE 6 — Reducción de dimensionalidad (COMPLETADA)

### Lo que se hizo
- PCA completo sobre matriz (250×72): 13 componentes capturan 80% de varianza
- Interpretación de componentes por correlación con patrones horarios
- UMAP 2D con `n_neighbors=15, min_dist=0.1` — dispersión visible (rango x=5.13, y=8.48)

### Archivos generados
- `data/processed/pca_variance.png` — scree plot con umbrales 80/90/95%
- `data/processed/pca_embedding.csv` — (250, 13) proyección PCA
- `data/processed/umap_embedding.csv` — (250, 3) coordenadas 2D
- `data/processed/umap_preview.png` — scatter coloreado por jam_factor y PC1

---

## ✅ FASE 7 — Clustering (COMPLETADA)

### Resultados DBSCAN espacial
| Cluster | Segmentos | Descripción |
|---|---|---|
| 0 | 242 | Zona principal de La Paz |
| 1 | 3 | Subzona periférica |
| -1 (outliers) | 5 | 2.0% del total |

### Resultados K-Means temporal (k=2)
| Cluster | Segmentos | Perfil |
|---|---|---|
| 0 | 28 | Alta congestión (jam_factor elevado) |
| 1 | 222 | Baja/moderada congestión |

### Archivos generados
- `data/processed/clusters_spatial.csv` — asignaciones DBSCAN
- `data/processed/clusters_temporal.csv` — asignaciones K-Means
- `data/processed/kmeans_elbow.png` — elbow + silhouette plot
- `data/processed/cluster_profiles.png` — perfiles temporales por cluster
- `data/processed/segments_clustered.csv` — DataFrame maestro (250×10)

---

## ✅ FASE 8 — Visualización estática (COMPLETADA)

### Archivos generados
- `data/processed/map_spatial_clusters.html` — 214 KB — mapa Folium coloreado por cluster DBSCAN
- `data/processed/map_temporal_clusters.html` — 214 KB — mapa Folium coloreado por cluster K-Means
- `data/processed/temporal_profiles.png` — 125 KB — perfiles horarios por cluster (lunes/viernes/sábado)

---

## ✅ FASE 9 — Dashboard Flask (COMPLETADA)

### Stack
- Flask 3.1 + Bootstrap 5 (dark theme) + Plotly.js + Folium (iframe)

### Archivos creados
- `lapaz_traffic/dashboard.py` — app Flask con 5 rutas API
- `lapaz_traffic/templates/base.html` — layout Bootstrap 5 dark
- `lapaz_traffic/templates/index.html` — UI con sidebar, mapa, gráficos interactivos, tabla

### Funcionalidades
- Slider de hora (0-23) + selector de día + selector de cluster
- Mapa Folium coloreado por jam_factor en la franja seleccionada (gradiente verde→rojo)
- Perfil temporal Plotly interactivo (líneas por tipo de día, zoom, hover)
- Histograma Plotly de distribución de jam_factor
- 3 métricas en cards: jam_promedio, n_segmentos, hora_pico
- Tabla Top 10 segmentos más congestionados con badges por cluster

### Cómo ejecutar
```bash
source venv_traffic/bin/activate
cd lapaz_traffic
python dashboard.py
# → http://localhost:5000
```

---

## ⏳ FASE 10 — Cierre y documentación — PENDIENTE
