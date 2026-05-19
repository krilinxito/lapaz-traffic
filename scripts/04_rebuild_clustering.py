"""PCA + UMAP + DBSCAN + K-Means para La Paz y El Alto.

Salida:
  data/processed/segments_clustered.csv   — 500 filas con columna 'zona'

Ejecutar desde la raíz del proyecto:
  python lapaz_traffic/scripts/04_rebuild_clustering.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_RAW  = Path(__file__).parent.parent / "data" / "raw"
DATA_PROC = Path(__file__).parent.parent / "data" / "processed"

N_PCA_COMPONENTS = 13
EARTH_RADIUS_M   = 6_371_000
DBSCAN_EPS_M     = 500
K_MEANS_K        = 2


def run_pipeline(segs_csv: Path, feat_csv: Path, zona: str) -> pd.DataFrame:
    print(f"\n── {zona.upper()} ──────────────────────────────────")
    segs = pd.read_csv(segs_csv)
    feat = pd.read_csv(feat_csv, index_col="segment_id")

    # Alinear índices
    common = segs["segment_id"].isin(feat.index)
    segs   = segs[common].reset_index(drop=True)
    feat   = feat.loc[segs["segment_id"]]

    X = StandardScaler().fit_transform(feat.values)
    print(f"  Feature matrix: {feat.shape}")

    # PCA
    n_comp = min(N_PCA_COMPONENTS, X.shape[1], X.shape[0] - 1)
    pca    = PCA(n_components=n_comp, random_state=42)
    X_pca  = pca.fit_transform(X)
    var_ac = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"  PCA {n_comp} componentes → {var_ac:.1%} varianza")

    # UMAP
    try:
        import umap as umap_lib
        reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        X_umap  = reducer.fit_transform(X_pca)
        print("  UMAP 2D OK")
    except ImportError:
        print("  UMAP no disponible — usando PCA 2D como fallback")
        X_umap = PCA(n_components=2, random_state=42).fit_transform(X_pca)

    # DBSCAN espacial (haversine)
    coords_rad = np.radians(segs[["lat", "lon"]].values)
    db         = DBSCAN(eps=DBSCAN_EPS_M / EARTH_RADIUS_M, min_samples=5, metric="haversine")
    cl_esp     = db.fit_predict(coords_rad)
    n_clusters_esp = len(set(cl_esp)) - (1 if -1 in cl_esp else 0)
    n_outliers     = (cl_esp == -1).sum()
    print(f"  DBSCAN: {n_clusters_esp} clusters, {n_outliers} outliers")

    # K-Means temporal
    km     = KMeans(n_clusters=K_MEANS_K, random_state=42, n_init=20)
    cl_tmp = km.fit_predict(X)
    print(f"  K-Means k={K_MEANS_K}: {np.bincount(cl_tmp + (cl_tmp.min() < 0))}")

    result = segs.copy()
    result["umap_x"]           = X_umap[:, 0]
    result["umap_y"]           = X_umap[:, 1]
    result["cluster_espacial"] = cl_esp
    result["cluster_temporal"] = cl_tmp
    result["jam_mean"]         = feat.mean(axis=1).values
    result["zona"]             = zona
    return result


# ── Ejecutar pipeline por zona ────────────────────────────────────────────────
lp = run_pipeline(
    DATA_RAW  / "sample_segments.csv",
    DATA_PROC / "feature_matrix.csv",
    "lapaz",
)

ea = run_pipeline(
    DATA_RAW  / "sample_segments_el_alto.csv",
    DATA_PROC / "feature_matrix_el_alto.csv",
    "el_alto",
)

# ── Combinar y guardar ────────────────────────────────────────────────────────
combined = pd.concat([lp, ea], ignore_index=True)
combined.to_csv(DATA_PROC / "segments_clustered.csv", index=False)

print(f"\n✓ segments_clustered.csv: {combined.shape}")
print(combined.groupby("zona")[["jam_mean", "cluster_temporal"]].agg(
    {"jam_mean": "mean", "cluster_temporal": "count"}
).rename(columns={"cluster_temporal": "n_segs"}))
