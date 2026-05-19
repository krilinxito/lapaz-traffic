"""Genera img/mapa_clusters.png con scatter lat/lon de ambos clusterings."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE = Path(__file__).parent.parent
df = pd.read_csv(BASE / "data" / "processed" / "segments_clustered.csv")

CLUSTER_ALTO = int(df.groupby("cluster_temporal")["jam_mean"].mean().idxmax())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#f8f9fa")

# ── Mapa K-Means temporal ────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor("#eef2f7")

colors_tmp = {CLUSTER_ALTO: "#c0392b", 1 - CLUSTER_ALTO: "#27ae60"}
labels_tmp = {CLUSTER_ALTO: "Alta congestión", 1 - CLUSTER_ALTO: "Baja congestión"}
zonas = {"lapaz": ("La Paz", "o", 40), "el_alto": ("El Alto", "s", 35)}

for zona, (znombre, marker, size) in zonas.items():
    sub = df[df["zona"] == zona]
    for cl in sorted(df["cluster_temporal"].unique()):
        pts = sub[sub["cluster_temporal"] == cl]
        ax.scatter(pts["lon"], pts["lat"], c=colors_tmp[cl],
                   s=size, alpha=0.75, marker=marker, linewidths=0,
                   label=f"{labels_tmp[cl]} ({znombre})" if zona == "lapaz" else None)

ax.set_title("Clustering temporal (K-Means, $k=2$)", fontsize=12, fontweight="bold", pad=10)
ax.set_xlabel("Longitud", fontsize=9)
ax.set_ylabel("Latitud", fontsize=9)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.3, linewidth=0.5)

patches = [mpatches.Patch(color="#c0392b", label="Alta congestión"),
           mpatches.Patch(color="#27ae60", label="Baja congestión"),
           mpatches.Patch(facecolor="none", edgecolor="gray", label="○ La Paz  □ El Alto")]
ax.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.85)

# ── Mapa DBSCAN espacial ──────────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor("#eef2f7")

unique_cl = sorted(df["cluster_espacial"].unique())
palette = plt.cm.tab10(np.linspace(0, 0.6, max(len(unique_cl) - 1, 1)))
cl_colors = {}
pal_idx = 0
for cl in unique_cl:
    if cl == -1:
        cl_colors[cl] = ("#aaaaaa", "Ruido (outlier)")
    else:
        cl_colors[cl] = (palette[pal_idx], f"Zona {cl}")
        pal_idx += 1

for zona, (znombre, marker, size) in zonas.items():
    sub = df[df["zona"] == zona]
    for cl in unique_cl:
        pts = sub[sub["cluster_espacial"] == cl]
        if pts.empty:
            continue
        col, lbl = cl_colors[cl]
        ax2.scatter(pts["lon"], pts["lat"], c=[col] * len(pts),
                    s=size, alpha=0.75, marker=marker, linewidths=0)

ax2.set_title("Clustering espacial (DBSCAN, $\\varepsilon=500$ m)", fontsize=12,
              fontweight="bold", pad=10)
ax2.set_xlabel("Longitud", fontsize=9)
ax2.set_ylabel("Latitud", fontsize=9)
ax2.tick_params(labelsize=8)
ax2.grid(True, alpha=0.3, linewidth=0.5)

patches2 = [mpatches.Patch(color=cl_colors[cl][0], label=cl_colors[cl][1])
            for cl in unique_cl]
patches2.append(mpatches.Patch(facecolor="none", edgecolor="gray",
                                label="○ La Paz  □ El Alto"))
ax2.legend(handles=patches2, loc="lower right", fontsize=8, framealpha=0.85)

plt.suptitle("Segmentos viales clasificados — La Paz y El Alto", fontsize=13,
             fontweight="bold", y=1.01)
plt.tight_layout()

out = Path(__file__).parent / "img" / "mapa_clusters.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Guardado: {out}")
