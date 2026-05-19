"""Descarga la red vial de El Alto (OSMnx) y muestrea 250 segmentos estratificados.

Salida:
  data/raw/el_alto_edges.csv              — toda la red (geometría + atributos)
  data/raw/sample_segments_el_alto.csv   — 250 segmentos seleccionados (IDs 1000–1249)

Ejecutar desde la raíz del proyecto:
  python lapaz_traffic/scripts/01_red_vial_el_alto.py
"""
import random
from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd
from shapely import wkt

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

TIPOS_VALIDOS = ["primary", "secondary", "tertiary", "residential", "trunk", "unclassified"]
N_SEGS       = 250
ID_OFFSET    = 1000   # IDs 1000–1249 para no colisionar con La Paz (0–249)
RANDOM_SEED  = 42

# ── 1. Descarga ───────────────────────────────────────────────────────────────
print("Descargando red vial de El Alto, Bolivia …")
try:
    G = ox.graph_from_place("El Alto, Bolivia", network_type="drive")
except Exception:
    print("  Fallback: usando bounding box manual de El Alto")
    G = ox.graph_from_bbox(
        north=-16.460, south=-16.545,
        east=-68.140, west=-68.240,
        network_type="drive",
    )

edges = ox.graph_to_gdfs(G, nodes=False).reset_index()
cols = [c for c in ["u", "v", "name", "highway", "length", "geometry"] if c in edges.columns]
edges = edges[cols].copy()
edges["highway"] = edges["highway"].apply(lambda x: x[0] if isinstance(x, list) else x)
edges["name"]    = edges["name"].apply(lambda x: x[0] if isinstance(x, list) else x) if "name" in edges.columns else ""
edges = edges[edges["geometry"].notna()].drop_duplicates(["u", "v"]).reset_index(drop=True)

edges.to_csv(DATA_RAW / "el_alto_edges.csv", index=False)
print(f"  Guardado: el_alto_edges.csv  ({len(edges):,} aristas)")

# ── 2. Muestreo estratificado ─────────────────────────────────────────────────
edges_v = edges[edges["highway"].isin(TIPOS_VALIDOS)].copy()
counts  = edges_v["highway"].value_counts()
fracs   = (counts / counts.sum() * N_SEGS).round().astype(int)
diff    = N_SEGS - fracs.sum()
fracs.iloc[0] += diff   # ajuste para llegar exactamente a N_SEGS

print(f"\nDistribución muestreo El Alto:")
samples = []
random.seed(RANDOM_SEED)
for hw, n in fracs.items():
    pool = edges_v[edges_v["highway"] == hw]
    n    = min(n, len(pool))
    if n <= 0:
        continue
    samples.append(pool.sample(n, random_state=RANDOM_SEED))
    print(f"  {hw:<20} {n:>4} segs  (pool: {len(pool)})")

segs = pd.concat(samples).reset_index(drop=True)
segs["segment_id"] = range(ID_OFFSET, ID_OFFSET + len(segs))

# ── 3. Extraer coordenadas ────────────────────────────────────────────────────
def get_coords(geom_str):
    coords = list(wkt.loads(str(geom_str)).coords)
    mid    = coords[len(coords) // 2]
    return {
        "lat":     coords[0][1],      # lat nodo u (inicio)
        "lon":     coords[0][0],
        "lat_u":   coords[0][1],
        "lon_u":   coords[0][0],
        "lat_v":   coords[-1][1],     # lat nodo v (fin)
        "lon_v":   coords[-1][0],
        "lat_mid": mid[1],
        "lon_mid": mid[0],
    }

geo = segs["geometry"].map(get_coords).apply(pd.Series)
segs["lat"]   = geo["lat_mid"]
segs["lon"]   = geo["lon_mid"]
segs["lat_u"] = geo["lat_u"]
segs["lon_u"] = geo["lon_u"]
segs["lat_v"] = geo["lat_v"]
segs["lon_v"] = geo["lon_v"]

out_cols = ["segment_id", "u", "v", "lat_u", "lon_u", "lat_v", "lon_v", "highway", "length", "lat", "lon"]
segs[out_cols].to_csv(DATA_RAW / "sample_segments_el_alto.csv", index=False)
print(f"\nGuardado: sample_segments_el_alto.csv  ({len(segs)} segmentos, IDs {ID_OFFSET}–{ID_OFFSET+len(segs)-1})")
