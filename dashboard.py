"""Dashboard de congestión vehicular — La Paz y El Alto, Bolivia"""
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from shapely import wkt
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ── Rutas de datos ──────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
DATA_RAW       = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
TIPOS_DIA       = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
DIAS_SINTETICOS = {"martes", "jueves", "domingo"}

CENTERS = {
    "lapaz":   [-16.505, -68.130],
    "el_alto": [-16.502, -68.187],
    "ambas":   [-16.504, -68.155],
}

# ── Carga de datos ───────────────────────────────────────────────────────────
clustered  = pd.read_csv(DATA_PROCESSED / "segments_clustered.csv")
if "zona" not in clustered.columns:
    clustered["zona"] = "lapaz"   # retrocompatibilidad antes del pipeline de extensión
feat_lapaz = pd.read_csv(DATA_PROCESSED / "feature_matrix.csv",         index_col="segment_id")

# El Alto se carga si el archivo existe (puede no existir antes de correr el pipeline)
_ea_path = DATA_PROCESSED / "feature_matrix_el_alto.csv"
feat_el_alto = pd.read_csv(_ea_path, index_col="segment_id") if _ea_path.exists() else pd.DataFrame()

segs_info  = pd.read_csv(DATA_RAW / "sample_segments.csv")
edges_lapaz = pd.read_csv(DATA_RAW / "lapaz_edges.csv").drop_duplicates(["u", "v"])

_ea_edges = DATA_RAW / "el_alto_edges.csv"
edges_el_alto = pd.read_csv(_ea_edges).drop_duplicates(["u", "v"]) if _ea_edges.exists() else pd.DataFrame()

# Segmentos de El Alto
_ea_segs = DATA_RAW / "sample_segments_el_alto.csv"
segs_el_alto = pd.read_csv(_ea_segs) if _ea_segs.exists() else pd.DataFrame()

# Unir todos los segmentos de referencia
segs_all = pd.concat([segs_info, segs_el_alto], ignore_index=True) if not segs_el_alto.empty else segs_info

# Metadata de días sintéticos (NMF+PCHIP)
_syn_meta_path = DATA_PROCESSED / "synthetic_days_metadata.csv"
synthetic_meta = pd.read_csv(_syn_meta_path) if _syn_meta_path.exists() else pd.DataFrame()

# ── Funciones de cómputo de modelos (ejecutadas una vez en startup) ──────────
_REAL_DAYS = ["lunes", "miercoles", "viernes", "sabado"]


def _compute_nmf_artifacts(feat: pd.DataFrame) -> dict:
    cols_real = [c for c in feat.columns
                 if any(c.startswith(f"jam_{d}_") for d in _REAL_DAYS)]
    if not cols_real or feat.empty:
        return {"W": None, "H": None, "mean_profiles": None, "dominant": None,
                "segment_ids": [], "recon_err": None}
    X = np.clip(feat[cols_real].fillna(0).values, 0, None)
    nmf = NMF(n_components=6, init="nndsvda", random_state=42, max_iter=600)
    W = nmf.fit_transform(X)
    H = nmf.components_
    n_days = len(_REAL_DAYS)
    H_reshaped = H.reshape(6, n_days, 24)
    mean_profiles = H_reshaped.mean(axis=1)
    return {
        "W": W,
        "H": H,
        "mean_profiles": mean_profiles,
        "recon_err": float(nmf.reconstruction_err_),
        "dominant": W.argmax(axis=1),
        "segment_ids": feat.index.tolist(),
    }


def _compute_model_metrics(feat: pd.DataFrame, cl: pd.DataFrame, zona: str) -> dict:
    out: dict = {}
    cl_zona = cl[cl["zona"] == zona].copy() if "zona" in cl.columns else cl.copy()

    # K-Means silhouette
    merged = feat.merge(cl_zona[["segment_id", "cluster_temporal"]],
                        left_index=True, right_on="segment_id", how="inner")
    if not merged.empty and merged["cluster_temporal"].nunique() > 1:
        X = StandardScaler().fit_transform(
            merged.drop(columns=["segment_id", "cluster_temporal"], errors="ignore")
            .select_dtypes(include=[np.number])
        )
        out["kmeans_silhouette"] = round(float(silhouette_score(X, merged["cluster_temporal"].values)), 4)
    else:
        out["kmeans_silhouette"] = None

    # PCA varianza acumulada
    if not feat.empty:
        X_s = StandardScaler().fit_transform(feat.fillna(0).values)
        n = min(13, X_s.shape[1], X_s.shape[0] - 1)
        pca = PCA(n_components=n)
        pca.fit(X_s)
        out["pca_variance_pct"] = round(float(np.cumsum(pca.explained_variance_ratio_)[-1]) * 100, 1)
        out["pca_n_components"] = n
    else:
        out["pca_variance_pct"] = None
        out["pca_n_components"] = 0

    # DBSCAN silhouette (solo no-ruido)
    non_noise = cl_zona[cl_zona["cluster_espacial"] != -1]
    if len(non_noise) > 1 and non_noise["cluster_espacial"].nunique() > 1:
        coords = non_noise[["lat", "lon"]].values
        out["dbscan_silhouette"] = round(float(silhouette_score(coords, non_noise["cluster_espacial"].values)), 4)
        out["dbscan_n_clusters"] = int(non_noise["cluster_espacial"].nunique())
    else:
        out["dbscan_silhouette"] = None
        out["dbscan_n_clusters"] = 0
    out["dbscan_noise_pct"] = round(float((cl_zona["cluster_espacial"] == -1).sum() / max(len(cl_zona), 1)) * 100, 1)

    # NMF RMSE-CV desde metadata
    if not synthetic_meta.empty:
        sm = synthetic_meta[synthetic_meta["zona"] == zona] if "zona" in synthetic_meta.columns else synthetic_meta
        interp = sm[sm["tipo"] == "interpolado"]["rmse_cv"].mean() if "tipo" in sm.columns else float("nan")
        extra  = sm[sm["tipo"] == "extrapolado"]["rmse_cv"].mean() if "tipo" in sm.columns else float("nan")
        out["nmf_rmse_interpolado"] = round(float(interp), 4) if not np.isnan(interp) else None
        out["nmf_rmse_extrapolado"] = round(float(extra), 4) if not np.isnan(extra) else None
    else:
        out["nmf_rmse_interpolado"] = None
        out["nmf_rmse_extrapolado"] = None

    return out


NMF_LAPAZ           = _compute_nmf_artifacts(feat_lapaz)
MODEL_METRICS_LAPAZ = _compute_model_metrics(feat_lapaz, clustered, "lapaz")

# Geometría combinada
def _build_df_geo(segs, edges):
    if segs.empty or edges.empty:
        return pd.DataFrame()
    df = (
        segs
        .merge(edges[["u", "v", "geometry", "name"]], on=["u", "v"], how="left")
    )
    return df[df["geometry"].notna()].reset_index(drop=True)

df_geo_lapaz   = _build_df_geo(
    segs_info.merge(clustered[clustered["zona"] == "lapaz"][["segment_id","cluster_temporal","cluster_espacial","jam_mean"]],
                    on="segment_id", how="left"),
    edges_lapaz,
)
df_geo_el_alto = _build_df_geo(
    segs_el_alto.merge(clustered[clustered["zona"] == "el_alto"][["segment_id","cluster_temporal","cluster_espacial","jam_mean"]],
                       on="segment_id", how="left"),
    edges_el_alto,
) if not segs_el_alto.empty else pd.DataFrame()

df_geo_all = pd.concat([df_geo_lapaz, df_geo_el_alto], ignore_index=True) if not df_geo_el_alto.empty else df_geo_lapaz


# ── Helpers de zona ──────────────────────────────────────────────────────────
def get_clustered(zona: str) -> pd.DataFrame:
    if zona == "el_alto":
        return clustered[clustered["zona"] == "el_alto"]
    if zona == "ambas":
        return clustered
    return clustered[clustered["zona"] == "lapaz"]


def get_feat(zona: str) -> pd.DataFrame:
    if zona == "el_alto":
        return feat_el_alto
    if zona == "ambas":
        # Solo columnas comunes
        common = feat_lapaz.columns.intersection(feat_el_alto.columns)
        return pd.concat([feat_lapaz[common], feat_el_alto[common]])
    return feat_lapaz


def get_df_geo(zona: str) -> pd.DataFrame:
    if zona == "el_alto":
        return df_geo_el_alto
    if zona == "ambas":
        return df_geo_all
    return df_geo_lapaz


def get_edges(zona: str) -> pd.DataFrame:
    if zona == "el_alto":
        return edges_el_alto
    if zona == "ambas":
        return pd.concat([edges_lapaz, edges_el_alto], ignore_index=True)
    return edges_lapaz


def cluster_alto_for(zona: str) -> int:
    cl = get_clustered(zona)
    feat = get_feat(zona)
    if feat.empty or cl.empty:
        return 0
    ids = cl["segment_id"].tolist()
    sub = feat.loc[feat.index.isin(ids)]
    merged = sub.merge(cl[["segment_id", "cluster_temporal"]], left_index=True, right_on="segment_id", how="inner")
    mean_by_cl = merged.groupby("cluster_temporal").mean(numeric_only=True).mean(axis=1)
    return int(mean_by_cl.idxmax()) if not mean_by_cl.empty else 0


# CLUSTER_ALTO global para La Paz (compatibilidad template)
CLUSTER_ALTO = cluster_alto_for("lapaz")


# ── Colores ──────────────────────────────────────────────────────────────────
def jam_color(value: float) -> str:
    r = min(255, max(0, int(value / 10 * 510)))
    g = min(255, max(0, int((1 - value / 10) * 510)))
    return f"#{r:02x}{g:02x}00"


# ── build_map ────────────────────────────────────────────────────────────────
def build_map(hora: int, dia: str, cluster: int, zona: str = "lapaz") -> str:
    col = f"jam_{dia}_{hora:02d}"
    feat = get_feat(zona)
    df_g = get_df_geo(zona)
    cl   = get_clustered(zona)
    ca   = cluster_alto_for(zona)

    if col not in feat.columns:
        col = feat.columns[0] if not feat.empty else None

    df = df_g.copy()
    if col and not feat.empty:
        jam_hora = feat[col].rename("jam_hora")
        df = df.merge(jam_hora, left_on="segment_id", right_index=True, how="left")
    else:
        df["jam_hora"] = 0.0
    df["jam_hora"] = df["jam_hora"].fillna(0)

    if cluster != -1:
        df = df[df["cluster_temporal"] == cluster]

    center = CENTERS.get(zona, CENTERS["lapaz"])
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")

    # ── Fondo: extender calles por nombre (una sola capa GeoJSON) ────────────
    name_jam: dict[str, float] = {}
    for _, row in df.iterrows():
        name = row.get("name")
        if pd.notna(name) and str(name).strip():
            key = str(name)
            name_jam[key] = max(name_jam.get(key, 0.0), float(row["jam_hora"]))

    if name_jam:
        all_edges = get_edges(zona)
        if not all_edges.empty and "name" in all_edges.columns:
            bg = all_edges[all_edges["name"].isin(name_jam)]
            features = []
            for _, erow in bg.iterrows():
                ename = str(erow.get("name", ""))
                jam_bg = name_jam.get(ename, 0.0)
                try:
                    geom = wkt.loads(str(erow["geometry"]))
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": list(geom.coords),
                        },
                        "properties": {
                            "color": jam_color(jam_bg * 10),
                            "tip": f"{ename or 'Sin nombre'}  ·  jam {jam_bg:.3f}",
                        },
                    })
                except Exception:
                    pass
            if features:
                folium.GeoJson(
                    {"type": "FeatureCollection", "features": features},
                    style_function=lambda f: {
                        "color":   f["properties"]["color"],
                        "weight":  3,
                        "opacity": 0.40,
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=["tip"], aliases=[""], labels=False, sticky=True,
                        style=_TIP_STYLE,
                    ),
                ).add_to(m)

    for _, row in df.iterrows():
        jam_val = float(row["jam_hora"])
        color   = jam_color(jam_val * 10)
        nombre  = row["name"] if pd.notna(row.get("name")) else "Sin nombre"
        es_alto = int(row.get("cluster_temporal", 0)) == ca
        jam_pct = min(100, jam_val * 250)
        seg_id  = int(row["segment_id"])
        hw      = str(row.get("highway", ""))
        jam_mean_seg = float(row.get("jam_mean", 0))

        badge_bg    = "#fef2f2" if es_alto else "#f0fdf4"
        badge_color = "#dc2626" if es_alto else "#16a34a"
        badge_bdr   = "#fecaca" if es_alto else "#bbf7d0"
        badge_label = "patrón ALTO" if es_alto else "patrón BAJO"
        cl_esp      = int(row.get("cluster_espacial", -1))
        dbscan_badge = (
            f'<span style="background:#fff7ed;color:#c2410c;border:1px solid #fed7aa;'
            f'padding:1px 7px;border-radius:3px;font-size:10px;font-weight:700">'
            f'DBSCAN zona {cl_esp}</span>'
            if cl_esp != -1 else
            '<span style="font-size:9px;color:#94a3b8">sin zona DBSCAN</span>'
        )
        onclick_js  = (
            "window.parent.postMessage("
            "{type:'segment_click',segment_id:" + str(seg_id) + "},'*')"
        )

        popup_html = (
            '<div style="font-family:Inter,sans-serif;font-size:13px;min-width:200px;line-height:1.6">'
            f'<b style="font-size:14px">{nombre}</b>'
            '<div style="margin:7px 0 2px">'
            f'<div style="background:#e2e8f0;border-radius:3px;height:5px">'
            f'<div style="background:{color};width:{jam_pct:.0f}%;height:5px;border-radius:3px"></div></div>'
            f'<span style="font-size:11px;color:#0f172a;font-weight:600">jam ahora: {jam_val:.3f}</span>'
            f'<span style="font-size:11px;color:#94a3b8;margin-left:8px">histórico: {jam_mean_seg:.3f}</span>'
            '</div>'
            f'<div style="font-size:11px;color:#64748b;margin-bottom:4px">{hora:02d}:00 · {dia} · {hw}</div>'
            f'<span style="background:{badge_bg};color:{badge_color};border:1px solid {badge_bdr};'
            f'padding:1px 7px;border-radius:3px;font-size:10px;font-weight:700">{badge_label}</span>'
            '<span style="font-size:9px;color:#94a3b8;margin-left:5px">K-Means</span>'
            f'<div style="margin-top:4px">{dbscan_badge}'
            '<span style="font-size:9px;color:#94a3b8;margin-left:5px">DBSCAN</span></div>'
            f'<button onclick="{onclick_js}" '
            'style="display:block;margin-top:8px;background:#0369a1;color:#fff;border:none;padding:5px 12px;'
            'border-radius:3px;cursor:pointer;font-size:11px;font-weight:600;width:100%">'
            'Ver perfil de 24 h</button></div>'
        )

        tooltip_html = (
            f'<div style="font-family:Inter,sans-serif;min-width:185px;padding:2px 0">'
            f'<div style="font-size:13px;font-weight:700;color:#0f172a;margin-bottom:4px">{nombre}</div>'
            f'<div style="background:#e2e8f0;border-radius:2px;height:3px;margin-bottom:5px">'
            f'<div style="background:{color};width:{jam_pct:.0f}%;height:3px;border-radius:2px"></div></div>'
            f'<div style="font-size:12px;line-height:1.7">'
            f'jam {hora:02d}:00&nbsp;&nbsp;<b style="color:{color}">{jam_val:.3f}</b>'
            f'&ensp;<span style="color:#94a3b8;font-size:10px">hist: {jam_mean_seg:.3f}</span></div>'
            f'<div style="font-size:10px;color:#64748b;margin-top:2px">{hw} &middot; {dia}</div>'
            f'<div style="margin-top:6px;display:flex;align-items:center;gap:5px;flex-wrap:wrap">'
            f'<span style="background:{badge_bg};color:{badge_color};border:1px solid {badge_bdr};'
            f'padding:1px 6px;border-radius:3px;font-size:10px;font-weight:700">{badge_label}</span>'
            f'{dbscan_badge}</div>'
            f'<div style="font-size:10px;color:#94a3b8;margin-top:5px">&#8592; clic &middot; ver perfil 24 h</div>'
            f'</div>'
        )
        coords = [[lat, lon] for lon, lat in wkt.loads(row["geometry"]).coords]
        folium.PolyLine(coords, color=color, weight=5, opacity=0.9).add_to(m)
        folium.PolyLine(
            coords, color=color, weight=16, opacity=0.001,
            tooltip=folium.Tooltip(tooltip_html, sticky=True, style=_TIP_STYLE),
            popup=folium.Popup(popup_html, max_width=230),
        ).add_to(m)

    # Leyenda
    legend_html = (
        '<div style="position:fixed;bottom:24px;right:24px;z-index:1000;'
        'background:white;padding:10px 14px;border-radius:4px;'
        'border:1px solid #e2e8f0;font-family:Inter,sans-serif;font-size:11px;'
        'box-shadow:0 2px 8px rgba(0,0,0,.08)">'
        '<div style="font-weight:700;margin-bottom:6px;color:#0f172a">jam_factor</div>'
        '<div style="display:flex;align-items:center;gap:7px;margin-bottom:4px">'
        '<div style="width:22px;height:4px;background:#00ff00;border-radius:2px"></div>Bajo</div>'
        '<div style="display:flex;align-items:center;gap:7px;margin-bottom:4px">'
        '<div style="width:22px;height:4px;background:#ffaa00;border-radius:2px"></div>Medio</div>'
        '<div style="display:flex;align-items:center;gap:7px">'
        '<div style="width:22px;height:4px;background:#ff0000;border-radius:2px"></div>Alto</div></div>'
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    # Receptor fly_to
    fly_script = (
        '<script>'
        'window.addEventListener("message",function(e){'
        'if(!e.data||e.data.type!=="fly_to")return;'
        'var map=Object.values(window).find(function(v){'
        'return v&&typeof v.flyTo==="function"&&typeof v.setView==="function";});'
        'if(map)map.flyTo([e.data.lat,e.data.lon],17,{duration:1.0});'
        '});'
        '</script>'
    )
    m.get_root().html.add_child(folium.Element(fly_script))

    return m.get_root().render()


# ── build_map_promedio ───────────────────────────────────────────────────────
def build_map_promedio(zona: str = "lapaz") -> str:
    """Mapa coloreado por jam_mean histórico — sin filtro de hora/día."""
    df_g = get_df_geo(zona).copy()
    df_g["jam_hora"] = df_g["jam_mean"].fillna(0.0)

    center = CENTERS.get(zona, CENTERS["lapaz"])
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")

    # Capa de fondo: extender calles por nombre
    name_jam: dict[str, float] = {}
    for _, row in df_g.iterrows():
        name = row.get("name")
        if pd.notna(name) and str(name).strip():
            key = str(name)
            name_jam[key] = max(name_jam.get(key, 0.0), float(row["jam_hora"]))

    if name_jam:
        all_edges = get_edges(zona)
        if not all_edges.empty and "name" in all_edges.columns:
            bg = all_edges[all_edges["name"].isin(name_jam)]
            features = []
            for _, erow in bg.iterrows():
                ename = str(erow.get("name", ""))
                jam_bg = name_jam.get(ename, 0.0)
                try:
                    geom = wkt.loads(str(erow["geometry"]))
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": list(geom.coords)},
                        "properties": {
                            "color": jam_color(jam_bg * 10),
                            "tip": f"{ename or 'Sin nombre'}  ·  jam {jam_bg:.3f}",
                        },
                    })
                except Exception:
                    pass
            if features:
                folium.GeoJson(
                    {"type": "FeatureCollection", "features": features},
                    style_function=lambda f: {
                        "color": f["properties"]["color"], "weight": 3, "opacity": 0.40,
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=["tip"], aliases=[""], labels=False, sticky=True,
                        style=_TIP_STYLE,
                    ),
                ).add_to(m)

    ca = cluster_alto_for(zona)
    for _, row in df_g.iterrows():
        jam_val = float(row["jam_hora"])
        color   = jam_color(jam_val * 10)
        nombre  = row["name"] if pd.notna(row.get("name")) else "Sin nombre"
        seg_id  = int(row["segment_id"])
        es_alto = int(row.get("cluster_temporal", 0)) == ca
        hw      = str(row.get("highway", ""))
        jam_pct = min(100, jam_val * 250)

        badge_bg    = "#fef2f2" if es_alto else "#f0fdf4"
        badge_color = "#dc2626" if es_alto else "#16a34a"
        badge_bdr   = "#fecaca" if es_alto else "#bbf7d0"
        badge_label = "Alta congestión" if es_alto else "Baja congestión"
        onclick_js  = (
            "window.parent.postMessage("
            "{type:'segment_click',segment_id:" + str(seg_id) + "},'*')"
        )

        popup_html = (
            '<div style="font-family:Inter,sans-serif;font-size:13px;min-width:200px;line-height:1.6">'
            f'<b style="font-size:14px">{nombre}</b>'
            '<div style="margin:7px 0 2px">'
            f'<div style="background:#e2e8f0;border-radius:3px;height:5px">'
            f'<div style="background:{color};width:{jam_pct:.0f}%;height:5px;border-radius:3px"></div></div>'
            f'<span style="font-size:11px;color:#0f172a;font-weight:600">jam promedio: {jam_val:.3f}</span>'
            '</div>'
            f'<div style="font-size:11px;color:#64748b;margin-bottom:4px">{hw}</div>'
            f'<span style="background:{badge_bg};color:{badge_color};border:1px solid {badge_bdr};'
            f'padding:1px 7px;border-radius:3px;font-size:10px;font-weight:700">{badge_label}</span>'
            f'<button onclick="{onclick_js}" '
            'style="display:block;margin-top:8px;background:#0369a1;color:#fff;border:none;padding:5px 12px;'
            'border-radius:3px;cursor:pointer;font-size:11px;font-weight:600;width:100%">'
            'Ver perfil de 24 h</button></div>'
        )

        tooltip_html_prom = (
            f'<div style="font-family:Inter,sans-serif;min-width:185px;padding:2px 0">'
            f'<div style="font-size:13px;font-weight:700;color:#0f172a;margin-bottom:4px">{nombre}</div>'
            f'<div style="background:#e2e8f0;border-radius:2px;height:3px;margin-bottom:5px">'
            f'<div style="background:{color};width:{jam_pct:.0f}%;height:3px;border-radius:2px"></div></div>'
            f'<div style="font-size:12px;line-height:1.7">'
            f'jam hist&nbsp;&nbsp;<b style="color:{color}">{jam_val:.3f}</b></div>'
            f'<div style="font-size:10px;color:#64748b;margin-top:2px">{hw} &middot; promedio semanal</div>'
            f'<div style="margin-top:6px">'
            f'<span style="background:{badge_bg};color:{badge_color};border:1px solid {badge_bdr};'
            f'padding:1px 6px;border-radius:3px;font-size:10px;font-weight:700">{badge_label}</span></div>'
            f'<div style="font-size:10px;color:#94a3b8;margin-top:5px">&#8592; clic &middot; ver perfil 24 h</div>'
            f'</div>'
        )
        coords = [[lat, lon] for lon, lat in wkt.loads(row["geometry"]).coords]
        folium.PolyLine(coords, color=color, weight=5, opacity=0.9).add_to(m)
        folium.PolyLine(
            coords, color=color, weight=16, opacity=0.001,
            tooltip=folium.Tooltip(tooltip_html_prom, sticky=True, style=_TIP_STYLE),
            popup=folium.Popup(popup_html, max_width=230),
        ).add_to(m)

    legend_html = (
        '<div style="position:fixed;bottom:24px;right:24px;z-index:1000;'
        'background:white;padding:10px 14px;border-radius:4px;'
        'border:1px solid #e2e8f0;font-family:Inter,sans-serif;font-size:11px;">'
        '<div style="font-weight:700;margin-bottom:6px;color:#0f172a">Congestión promedio</div>'
        '<div style="display:flex;align-items:center;gap:7px;margin-bottom:4px">'
        '<div style="width:22px;height:4px;background:#00ff00;border-radius:2px"></div>Baja</div>'
        '<div style="display:flex;align-items:center;gap:7px;margin-bottom:4px">'
        '<div style="width:22px;height:4px;background:#ffaa00;border-radius:2px"></div>Media</div>'
        '<div style="display:flex;align-items:center;gap:7px">'
        '<div style="width:22px;height:4px;background:#ff0000;border-radius:2px"></div>Alta</div></div>'
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    fly_script = (
        '<script>window.addEventListener("message",function(e){'
        'if(!e.data||e.data.type!=="fly_to")return;'
        'var map=Object.values(window).find(function(v){'
        'return v&&typeof v.flyTo==="function"&&typeof v.setView==="function";});'
        'if(map)map.flyTo([e.data.lat,e.data.lon],17,{duration:1.0});'
        '});</script>'
    )
    m.get_root().html.add_child(folium.Element(fly_script))
    return m.get_root().render()


# ── App Flask ────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    clusters = sorted(clustered["cluster_temporal"].unique().tolist())

    cluster_info = {}
    for c in clusters:
        n     = int((clustered["cluster_temporal"] == c).sum())
        label = f"Alta congestión ({n} calles)" if c == CLUSTER_ALTO else f"Baja congestión ({n} calles)"
        cluster_info[c] = label

    dia_labels = {
        "lunes":     "Lunes",
        "martes":    "Martes (est.)",
        "miercoles": "Miércoles",
        "jueves":    "Jueves (est.)",
        "viernes":   "Viernes",
        "sabado":    "Sábado",
        "domingo":   "Domingo (est.)",
    }

    return render_template("index.html",
                           tipos_dia=TIPOS_DIA,
                           clusters=clusters,
                           cluster_alto=CLUSTER_ALTO,
                           cluster_info=cluster_info,
                           dia_labels=dia_labels,
                           dias_sinteticos=sorted(DIAS_SINTETICOS))


@app.route("/map-promedio")
def map_promedio_route():
    zona = request.args.get("zona", "lapaz")
    return build_map_promedio(zona)


@app.route("/map")
def mapa():
    hora    = int(request.args.get("hora", 8))
    dia     = request.args.get("dia", "lunes")
    cluster = int(request.args.get("cluster", -1))
    zona    = request.args.get("zona", "lapaz")
    return build_map(hora, dia, cluster, zona)


@app.route("/segmento/<int:sid>")
def segmento(sid):
    rows = clustered[clustered["segment_id"] == sid]
    if rows.empty:
        return jsonify({"error": "not found"}), 404
    row  = rows.iloc[0]
    zona = str(row.get("zona", "lapaz"))
    feat = get_feat(zona)

    geo_rows = get_df_geo(zona)
    geo_rows = geo_rows[geo_rows["segment_id"] == sid]
    nombre   = geo_rows["name"].iloc[0] if not geo_rows.empty else None
    nombre   = nombre if pd.notna(nombre) else "Sin nombre"

    if sid not in feat.index:
        return jsonify({"error": "no feature data"}), 404

    perfil = {}
    for d in TIPOS_DIA:
        cols = [f"jam_{d}_{h:02d}" for h in range(24)]
        available = [c for c in cols if c in feat.columns]
        vals = feat.loc[sid, available].tolist() if available else [0.0] * 24
        perfil[d] = [round(v, 4) for v in vals]

    return jsonify({
        "nombre":           nombre,
        "jam_mean":         round(float(row["jam_mean"]), 4),
        "length_m":         int(row.get("length", 0)),
        "highway":          str(row.get("highway", "")),
        "cluster_temporal": int(row["cluster_temporal"]),
        "cluster_espacial": int(row.get("cluster_espacial", -1)),
        "lat":              float(row["lat"]),
        "lon":              float(row["lon"]),
        "zona":             zona,
        "perfil":           perfil,
    })


@app.route("/clusters-data")
def clusters_data():
    zona = request.args.get("zona", "lapaz")
    cl   = get_clustered(zona)
    feat = get_feat(zona)

    points = cl[[
        "segment_id", "lat", "lon", "highway", "jam_mean",
        "cluster_temporal", "cluster_espacial", "umap_x", "umap_y",
    ]].round(5).to_dict(orient="records")

    ca     = cluster_alto_for(zona)
    resumen = []
    for c in sorted(cl["cluster_temporal"].unique()):
        sub  = cl[cl["cluster_temporal"] == c]
        ids  = sub["segment_id"].tolist()
        sfeat = feat.loc[feat.index.isin(ids)]
        jam_mean_cl = float(sfeat.mean().mean()) if not sfeat.empty else 0.0
        cols_d = [f"jam_lunes_{h:02d}" for h in range(24) if f"jam_lunes_{h:02d}" in feat.columns]
        hora_pico = int(np.argmax(sfeat[cols_d].mean().values)) if cols_d and not sfeat.empty else 0
        hw_dom    = str(sub["highway"].value_counts().index[0]) if not sub.empty else ""
        resumen.append({
            "cluster_temporal":  int(c),
            "n_segmentos":       len(sub),
            "jam_promedio":      round(jam_mean_cl, 4),
            "hora_pico":         hora_pico,
            "highway_dominante": hw_dom,
            "es_alto":           int(c) == ca,
        })

    return jsonify({"points": points, "resumen": resumen, "cluster_alto": ca})


@app.route("/pca-data")
def pca_data():
    zona = request.args.get("zona", "lapaz")
    feat = get_feat(zona)
    if feat.empty:
        return jsonify({"variance_ratio": [], "cumulative": []})
    X = StandardScaler().fit_transform(feat.values)
    n = min(13, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n)
    pca.fit(X)
    ratios     = [round(v, 5) for v in pca.explained_variance_ratio_.tolist()]
    cumulative = [round(v, 5) for v in np.cumsum(pca.explained_variance_ratio_).tolist()]
    return jsonify({"variance_ratio": ratios, "cumulative": cumulative})


@app.route("/perfil")
def perfil():
    cluster = int(request.args.get("cluster", -1))
    zona    = request.args.get("zona", "lapaz")
    cl      = get_clustered(zona)
    feat    = get_feat(zona)

    if cluster == -1:
        mask = pd.Series([True] * len(feat), index=feat.index)
    else:
        seg_ids = cl[cl["cluster_temporal"] == cluster]["segment_id"]
        mask    = feat.index.isin(seg_ids)

    series = {}
    for d in TIPOS_DIA:
        cols = [f"jam_{d}_{h:02d}" for h in range(24) if f"jam_{d}_{h:02d}" in feat.columns]
        if cols:
            series[d] = [round(v, 4) for v in feat.loc[mask, cols].mean().tolist()]
        else:
            series[d] = [0.0] * 24

    return jsonify({"horas": list(range(24)), "series": series})


@app.route("/metricas")
def metricas():
    hora    = int(request.args.get("hora", 8))
    dia     = request.args.get("dia", "lunes")
    cluster = int(request.args.get("cluster", -1))
    zona    = request.args.get("zona", "lapaz")
    col     = f"jam_{dia}_{hora:02d}"

    cl   = get_clustered(zona)
    feat = get_feat(zona)

    df = cl.copy()
    if cluster != -1:
        df = df[df["cluster_temporal"] == cluster]

    seg_ids  = df["segment_id"].tolist()
    jam_vals = feat.loc[feat.index.isin(seg_ids), col] if col in feat.columns else pd.Series(dtype=float)
    jam_prom = float(jam_vals.mean()) if not jam_vals.empty else 0.0

    mask   = feat.index.isin(seg_ids)
    cols_d = [f"jam_{dia}_{h:02d}" for h in range(24) if f"jam_{dia}_{h:02d}" in feat.columns]
    hora_pico = int(np.argmax(feat.loc[mask, cols_d].mean().values)) if cols_d else 0

    return jsonify({
        "jam_promedio": round(jam_prom, 3),
        "n_segmentos":  len(seg_ids),
        "hora_pico":    hora_pico,
    })


@app.route("/top10")
def top10():
    hora    = int(request.args.get("hora", 8))
    dia     = request.args.get("dia", "lunes")
    cluster = int(request.args.get("cluster", -1))
    zona    = request.args.get("zona", "lapaz")
    col     = f"jam_{dia}_{hora:02d}"

    cl   = get_clustered(zona)
    feat = get_feat(zona)

    df = cl.copy()
    if cluster != -1:
        df = df[df["cluster_temporal"] == cluster]

    if col in feat.columns:
        df = df.merge(feat[[col]].rename(columns={col: "jam_hora"}),
                      left_on="segment_id", right_index=True, how="left")
    else:
        df["jam_hora"] = df["jam_mean"]

    # Nombre del segmento
    df_g = get_df_geo(zona)[["segment_id", "name"]].drop_duplicates("segment_id")
    df   = df.merge(df_g, on="segment_id", how="left")
    df["nombre"] = df["name"].fillna("Sin nombre")

    top = df.nlargest(10, "jam_hora")[
        ["segment_id", "nombre", "highway", "jam_hora", "cluster_temporal", "cluster_espacial"]
    ]
    return jsonify(top.rename(columns={"jam_hora": "jam_factor"}).to_dict(orient="records"))


@app.route("/model-metrics")
def model_metrics():
    zona = request.args.get("zona", "lapaz")
    if zona == "lapaz":
        return jsonify(MODEL_METRICS_LAPAZ)
    return jsonify(_compute_model_metrics(get_feat(zona), get_clustered(zona), zona))


@app.route("/nmf-archetypes")
def nmf_archetypes():
    nmf = NMF_LAPAZ
    if nmf["W"] is None:
        return jsonify({"horas": list(range(24)), "arquetipos": [], "distribucion": {}, "puntos": []})

    arquetipos = []
    for k in range(6):
        profile = nmf["mean_profiles"][k].tolist()
        peak_hour = int(np.argmax(profile))
        arquetipos.append({
            "id": k,
            "label": f"Arquetipo {k}",
            "peak_hour": peak_hour,
            "peak_hour_label": f"{peak_hour:02d}:00",
            "max_activation": round(float(max(profile)), 4),
            "profile": [round(float(v), 4) for v in profile],
        })

    dominant = nmf["dominant"].tolist()
    seg_ids  = nmf["segment_ids"]
    dist     = {str(k): int((np.array(dominant) == k).sum()) for k in range(6)}

    cl = get_clustered("lapaz")
    umap_lookup = cl.set_index("segment_id")[["umap_x", "umap_y"]].to_dict("index")
    puntos = []
    for i, sid in enumerate(seg_ids):
        info = umap_lookup.get(sid, {"umap_x": 0.0, "umap_y": 0.0})
        puntos.append({
            "segment_id": int(sid),
            "dominant_archetype": int(dominant[i]),
            "umap_x": round(float(info["umap_x"]), 5),
            "umap_y": round(float(info["umap_y"]), 5),
        })

    return jsonify({"horas": list(range(24)), "arquetipos": arquetipos,
                    "distribucion": dist, "puntos": puntos})


@app.route("/dbscan-data")
def dbscan_data():
    zona = request.args.get("zona", "lapaz")
    cl   = get_clustered(zona)
    df_g = get_df_geo(zona)

    name_lookup = (df_g[["segment_id", "name"]].drop_duplicates("segment_id")
                   .set_index("segment_id")["name"].to_dict()
                   if not df_g.empty else {})

    clusters = []
    for cid in sorted(cl["cluster_espacial"].unique()):
        sub = cl[cl["cluster_espacial"] == cid]
        is_noise = int(cid) == -1
        segs = []
        for _, row in sub.iterrows():
            segs.append({
                "segment_id": int(row["segment_id"]),
                "lat": round(float(row["lat"]), 5),
                "lon": round(float(row["lon"]), 5),
                "jam_mean": round(float(row["jam_mean"]), 4),
                "highway": str(row.get("highway", "")),
            })
        hw_counts = sub["highway"].value_counts()
        clusters.append({
            "cluster_id": int(cid),
            "es_ruido": is_noise,
            "n_segmentos": len(sub),
            "lat_centro": round(float(sub["lat"].mean()), 5),
            "lon_centro": round(float(sub["lon"].mean()), 5),
            "jam_medio": round(float(sub["jam_mean"].mean()), 4),
            "highway_dominante": str(hw_counts.index[0]) if not hw_counts.empty else "",
            "segmentos": segs,
        })

    total       = len(cl)
    noise_count = int((cl["cluster_espacial"] == -1).sum())
    n_real      = int(cl[cl["cluster_espacial"] != -1]["cluster_espacial"].nunique())
    return jsonify({
        "clusters": clusters,
        "total_segmentos": total,
        "n_clusters_reales": n_real,
        "noise_count": noise_count,
        "noise_pct": round(noise_count / max(total, 1) * 100, 1),
        "zona": zona,
    })


@app.route("/synthetic-days")
def synthetic_days():
    zona = request.args.get("zona", "lapaz")
    feat = get_feat(zona)

    if synthetic_meta.empty:
        return jsonify({"dias": [], "horas": list(range(24))})

    sm = synthetic_meta[synthetic_meta["zona"] == zona].copy() if "zona" in synthetic_meta.columns else synthetic_meta.copy()
    dias_info = []
    for _, row in sm.iterrows():
        dia  = row["dia"]
        cols = [f"jam_{dia}_{h:02d}" for h in range(24) if f"jam_{dia}_{h:02d}" in feat.columns]
        profile = feat[cols].mean().tolist() if cols else [0.0] * 24
        dias_info.append({
            "dia": dia,
            "tipo": str(row.get("tipo", "real")),
            "rmse_cv": float(row.get("rmse_cv", 0.0)),
            "confianza": str(row.get("confianza", "alta")),
            "profile": [round(float(v), 4) for v in profile],
        })

    return jsonify({"dias": dias_info, "horas": list(range(24))})


_TIP_STYLE = (
    "background:white;border:1px solid #e2e8f0;border-radius:4px;"
    "padding:8px 10px;box-shadow:0 2px 8px rgba(0,0,0,.12);"
)

_DBSCAN_COLORS = [
    "#1565c0", "#1a9e5c", "#e02020", "#e87722", "#f4b942",
    "#9b59b6", "#00bcd4", "#ff5722", "#795548", "#607d8b", "#8bc34a", "#3f51b5",
]


@app.route("/map-dbscan")
def map_dbscan():
    zona   = request.args.get("zona", "lapaz")
    cl     = get_clustered(zona)
    center = CENTERS.get(zona, CENTERS["lapaz"])
    m      = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")

    for _, row in cl.iterrows():
        c_id    = int(row["cluster_espacial"])
        is_noise = c_id == -1
        color   = "#aaaaaa" if is_noise else _DBSCAN_COLORS[c_id % len(_DBSCAN_COLORS)]
        radius  = 3 if is_noise else 7
        opacity = 0.20 if is_noise else 0.85
        jam_seg   = float(row.get("jam_mean", 0))
        hw_seg    = str(row.get("highway", ""))
        zona_lbl  = f"Zona {c_id}" if not is_noise else "Sin zona (ruido)"
        cl_lbl    = "Alta congestión" if int(row.get("cluster_temporal", 0)) == cluster_alto_for(zona) else "Baja congestión"
        cl_bg     = "#fef2f2" if cl_lbl == "Alta congestión" else "#f0fdf4"
        cl_clr    = "#dc2626" if cl_lbl == "Alta congestión" else "#16a34a"
        cl_bdr    = "#fecaca" if cl_lbl == "Alta congestión" else "#bbf7d0"
        jam_bar_w = min(100, jam_seg * 250)
        jam_clr   = jam_color(jam_seg * 10)
        tip_html  = (
            f'<div style="font-family:Inter,sans-serif;min-width:170px;padding:2px 0">'
            f'<div style="font-size:12px;font-weight:700;color:{color};margin-bottom:3px">{zona_lbl}</div>'
            f'<div style="background:#e2e8f0;border-radius:2px;height:3px;margin-bottom:5px">'
            f'<div style="background:{jam_clr};width:{jam_bar_w:.0f}%;height:3px;border-radius:2px"></div></div>'
            f'<div style="font-size:11px;line-height:1.7">'
            f'jam hist&nbsp;&nbsp;<b style="color:{jam_clr}">{jam_seg:.3f}</b></div>'
            f'<div style="font-size:10px;color:#64748b;margin-top:2px">{hw_seg}</div>'
            f'<div style="margin-top:5px;display:flex;align-items:center;gap:5px">'
            f'<span style="background:{cl_bg};color:{cl_clr};border:1px solid {cl_bdr};'
            f'padding:1px 6px;border-radius:3px;font-size:10px;font-weight:700">{cl_lbl}</span></div>'
            f'<div style="font-size:10px;color:#94a3b8;margin-top:4px">Seg {int(row["segment_id"])}</div>'
            f'</div>'
        )
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=radius, color=color, fill=True, fill_color=color,
            fill_opacity=opacity, opacity=opacity,
            tooltip=folium.Tooltip(tip_html, sticky=True,
                                   style="background:white;border:1px solid #e2e8f0;"
                                         "border-radius:4px;padding:8px 10px;"
                                         "box-shadow:0 2px 8px rgba(0,0,0,.12);"),
            popup=folium.Popup(tip_html, max_width=200),
        ).add_to(m)

    unique_clusters = sorted(cl[cl["cluster_espacial"] != -1]["cluster_espacial"].unique())
    legend_items = "".join(
        f'<div style="display:flex;align-items:center;gap:7px;margin-bottom:4px">'
        f'<div style="width:10px;height:10px;border-radius:50%;'
        f'background:{_DBSCAN_COLORS[int(c) % len(_DBSCAN_COLORS)]}"></div>'
        f'Zona {int(c)}</div>'
        for c in unique_clusters
    )
    legend_html = (
        '<div style="position:fixed;bottom:24px;right:24px;z-index:1000;background:white;'
        'padding:10px 14px;border-radius:4px;border:1px solid #e2e8f0;'
        'font-family:Inter,sans-serif;font-size:11px;box-shadow:0 2px 8px rgba(0,0,0,.08)">'
        '<div style="font-weight:700;margin-bottom:6px;color:#0f172a">DBSCAN espacial</div>'
        + legend_items +
        '<div style="display:flex;align-items:center;gap:7px;margin-top:4px">'
        '<div style="width:10px;height:10px;border-radius:50%;background:#aaa;opacity:.4"></div>'
        'Ruido (sin zona)</div></div>'
    )
    m.get_root().html.add_child(folium.Element(legend_html))
    return m.get_root().render()


@app.route("/segments-3d")
def segments_3d():
    zona = request.args.get("zona", "lapaz")
    dia  = request.args.get("dia", "lunes")
    hora = int(request.args.get("hora", 8))
    col  = f"jam_{dia}_{hora:02d}"
    parts = []
    for z in (["lapaz", "el_alto"] if zona == "ambas" else [zona]):
        df = clustered[clustered["zona"] == z].copy()
        feat = feat_lapaz if z == "lapaz" else feat_el_alto
        if not feat.empty and col in feat.columns:
            df = df.set_index("segment_id").join(
                feat[col].rename("jam_actual"), how="left"
            ).reset_index()
            df["jam_actual"] = df["jam_actual"].fillna(0)
        else:
            df["jam_actual"] = df["jam_mean"]
        parts.append(df)
    df_all = pd.concat(parts, ignore_index=True)

    # Agregar nombre de calle desde df_geo (contiene el merge con edges)
    geo_pieces = []
    for z in (["lapaz", "el_alto"] if zona == "ambas" else [zona]):
        dg = get_df_geo(z)
        if not dg.empty and "name" in dg.columns:
            geo_pieces.append(dg.drop_duplicates("segment_id")[["segment_id", "name"]])
    if geo_pieces:
        name_series = pd.concat(geo_pieces).set_index("segment_id")["name"]
        df_all["name"] = df_all["segment_id"].map(name_series).fillna("")
    else:
        df_all["name"] = ""

    return jsonify([{
        "segment_id": int(r["segment_id"]),
        "lat": float(r["lat"]),
        "lon": float(r["lon"]),
        "jam": float(r["jam_actual"]),
        "jam_mean": float(r["jam_mean"]),
        "cluster_temporal": int(r["cluster_temporal"]),
        "highway": str(r.get("highway", "")),
        "name": str(r.get("name", "") or ""),
    } for _, r in df_all.iterrows()])


if __name__ == "__main__":
    app.run(debug=True, port=5000)
