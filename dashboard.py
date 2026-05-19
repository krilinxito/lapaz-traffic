"""Dashboard de congestión vehicular — La Paz y El Alto, Bolivia"""
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from shapely import wkt
from sklearn.decomposition import PCA
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
                        "properties": {"color": jam_color(jam_bg * 10)},
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
            '<span style="font-size:9px;color:#94a3b8;margin-left:5px">cluster temporal</span>'
            f'<button onclick="{onclick_js}" '
            'style="display:block;margin-top:8px;background:#0369a1;color:#fff;border:none;padding:5px 12px;'
            'border-radius:3px;cursor:pointer;font-size:11px;font-weight:600;width:100%">'
            'Ver perfil de 24 h</button></div>'
        )

        coords = [[lat, lon] for lon, lat in wkt.loads(row["geometry"]).coords]
        folium.PolyLine(
            coords, color=color, weight=5, opacity=0.9,
            tooltip=folium.Tooltip(f"<b>{nombre}</b> · jam {jam_val:.3f}"),
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
                        "properties": {"color": jam_color(jam_bg * 10)},
                    })
                except Exception:
                    pass
            if features:
                folium.GeoJson(
                    {"type": "FeatureCollection", "features": features},
                    style_function=lambda f: {
                        "color": f["properties"]["color"], "weight": 3, "opacity": 0.40,
                    },
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

        coords = [[lat, lon] for lon, lat in wkt.loads(row["geometry"]).coords]
        folium.PolyLine(
            coords, color=color, weight=5, opacity=0.9,
            tooltip=folium.Tooltip(f"<b>{nombre}</b> · jam prom. {jam_val:.3f}"),
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
