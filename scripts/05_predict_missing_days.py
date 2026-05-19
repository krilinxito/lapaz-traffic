"""Síntesis de días faltantes (martes, jueves, domingo) con NMF + PCHIP.

Algoritmo:
  1. NMF descompone X (segmentos × 96) en W (pesos) × H (arquetipos temporales)
  2. H se reshapea a (k, 4 días, 24 h)
  3. PCHIP interpola cada arquetipo en el eje día-de-semana hacia posiciones 1,3,6
  4. Se reconstruyen los perfiles sintéticos: W × H_expandido
  5. Las feature matrices se amplían de 96 → 168 columnas

Validación interna: se oculta miércoles (posición 2), se predice con el resto,
y se reporta el RMSE como indicador de calidad antes de sintetizar.

⚠  Los días generados (martes, jueves, domingo) son ESTIMACIONES estadísticas,
   no datos recolectados. Se etiquetan como "estimado" en el dashboard.

Ejecutar desde la raíz del proyecto:
  python lapaz_traffic/scripts/05_predict_missing_days.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error

DATA_PROC = Path(__file__).parent.parent / "data" / "processed"

DIAS_REALES = ["lunes", "miercoles", "viernes", "sabado"]
DIAS_SINT   = ["martes", "jueves", "domingo"]
DIAS_TODOS  = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
DIA_POS     = {"lunes": 0, "martes": 1, "miercoles": 2, "jueves": 3,
               "viernes": 4, "sabado": 5, "domingo": 6}
NMF_K       = 6   # número de arquetipos


# ── Pipeline principal ────────────────────────────────────────────────────────

def predict_missing_days(feat_csv: Path, zona_label: str) -> pd.DataFrame:
    print(f"\n── {zona_label.upper()} ─────────────────────────────────────────")

    feat = pd.read_csv(feat_csv, index_col="segment_id")
    n_segs = len(feat)

    # Verificar que las columnas de los 4 días reales existen
    expected = [f"jam_{d}_{h:02d}" for d in DIAS_REALES for h in range(24)]
    missing_cols = [c for c in expected if c not in feat.columns]
    if missing_cols:
        print(f"  ADVERTENCIA: {len(missing_cols)} columnas faltantes — completando con 0")
        for c in missing_cols:
            feat[c] = 0.0

    # Ordenar columnas según DIAS_REALES para garantizar el reshape
    cols_reales = [f"jam_{d}_{h:02d}" for d in DIAS_REALES for h in range(24)]
    X = feat[cols_reales].values.astype(float)           # (n_segs, 96)
    X3d = X.reshape(n_segs, len(DIAS_REALES), 24)        # (n_segs, 4, 24)
    print(f"  Feature matrix: {X.shape}  — NMF k={NMF_K}")

    # ── Validación cruzada: ocultar miércoles ─────────────────────────────────
    idx_mie     = DIAS_REALES.index("miercoles")          # 1
    idx_sin_mie = [i for i in range(len(DIAS_REALES)) if i != idx_mie]

    X_sin_mie   = X3d[:, idx_sin_mie, :].reshape(n_segs, -1)   # (n_segs, 72)
    pos_sin_mie = [DIA_POS[DIAS_REALES[i]] for i in idx_sin_mie]  # [0, 4, 5]

    nmf_cv = NMF(n_components=NMF_K, init="nndsvd", max_iter=600, random_state=42)
    W_cv   = nmf_cv.fit_transform(X_sin_mie)             # (n_segs, k)
    H_cv   = nmf_cv.components_.reshape(NMF_K, 3, 24)   # (k, 3 días, 24h)

    mie_pred_H = np.zeros((NMF_K, 24))
    for ki in range(NMF_K):
        for h in range(24):
            sp = PchipInterpolator(pos_sin_mie, H_cv[ki, :, h])
            mie_pred_H[ki, h] = float(sp(DIA_POS["miercoles"]))  # posición 2

    X_mie_pred = np.clip(W_cv @ mie_pred_H, 0, None)    # (n_segs, 24)
    X_mie_real = X3d[:, idx_mie, :]
    rmse_cv = float(np.sqrt(mean_squared_error(X_mie_real.ravel(), X_mie_pred.ravel())))
    print(f"  CV RMSE miércoles reconstruido: {rmse_cv:.5f}  "
          f"(referencia jam_mean={X_mie_real.mean():.4f})")

    # ── NMF sobre los 4 días completos ────────────────────────────────────────
    nmf   = NMF(n_components=NMF_K, init="nndsvd", max_iter=600, random_state=42)
    W     = nmf.fit_transform(X)                          # (n_segs, k)
    H     = nmf.components_.reshape(NMF_K, len(DIAS_REALES), 24)  # (k, 4, 24)

    reconstruido = np.clip(W @ nmf.components_, 0, None)
    rmse_rec = float(np.sqrt(mean_squared_error(X.ravel(), reconstruido.ravel())))
    print(f"  RMSE reconstrucción completa:   {rmse_rec:.5f}")

    # ── Interpolación PCHIP por arquetipo y hora ─────────────────────────────
    pos_reales = [DIA_POS[d] for d in DIAS_REALES]       # [0, 2, 4, 5]
    pos_sint   = [DIA_POS[d] for d in DIAS_SINT]         # [1, 3, 6]

    H_sint = np.zeros((NMF_K, len(DIAS_SINT), 24))
    for ki in range(NMF_K):
        for h in range(24):
            sp = PchipInterpolator(pos_reales, H[ki, :, h])
            H_sint[ki, :, h] = sp(pos_sint)
    H_sint = np.clip(H_sint, 0, None)

    # ── Reconstruir perfiles sintéticos para los segmentos ───────────────────
    X_sint: dict[str, np.ndarray] = {}
    for i, dia in enumerate(DIAS_SINT):
        X_sint[dia] = np.clip(W @ H_sint[:, i, :], 0, None)  # (n_segs, 24)
        tipo = "extrapolado" if dia == "domingo" else "interpolado"
        print(f"  {dia:10s} ({tipo}) — jam_mean={X_sint[dia].mean():.4f}")

    # ── Construir feature matrix expandida (n_segs × 168) ───────────────────
    mat = feat[cols_reales].copy()                         # empieza con los 4 reales
    for dia in DIAS_SINT:
        for h in range(24):
            mat[f"jam_{dia}_{h:02d}"] = X_sint[dia][:, h]

    cols_ordered = [f"jam_{d}_{h:02d}" for d in DIAS_TODOS for h in range(24)]
    mat = mat[cols_ordered]
    mat.to_csv(feat_csv)
    print(f"  Guardado: {feat_csv.name}  shape={mat.shape}")

    # ── Metadata de confianza ─────────────────────────────────────────────────
    rows = []
    for dia in DIAS_TODOS:
        es_sint   = dia in DIAS_SINT
        es_extrap = dia == "domingo"
        rows.append({
            "dia":       dia,
            "zona":      zona_label,
            "tipo":      "extrapolado" if es_extrap else ("interpolado" if es_sint else "real"),
            "rmse_cv":   round(rmse_cv, 5) if es_sint else 0.0,
            "confianza": "baja"  if es_extrap else ("media" if es_sint else "alta"),
        })
    return pd.DataFrame(rows)


# ── Ejecutar para ambas zonas ─────────────────────────────────────────────────

meta_lp = predict_missing_days(
    DATA_PROC / "feature_matrix.csv",
    "lapaz",
)

meta_ea = predict_missing_days(
    DATA_PROC / "feature_matrix_el_alto.csv",
    "el_alto",
)

meta_all = pd.concat([meta_lp, meta_ea], ignore_index=True)
meta_all.to_csv(DATA_PROC / "synthetic_days_metadata.csv", index=False)

print("\n✓ Resumen de confianza:")
print(meta_all[meta_all["zona"] == "lapaz"][["dia", "tipo", "rmse_cv", "confianza"]].to_string(index=False))
print("\nPróximo paso:")
print("  python lapaz_traffic/scripts/04_rebuild_clustering.py")
