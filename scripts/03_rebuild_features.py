"""Reconstruye las feature matrices para La Paz (4 días) y El Alto.

Salida:
  data/processed/feature_matrix.csv           — La Paz 250 × 96 (4 días × 24 h)
  data/processed/feature_matrix_el_alto.csv   — El Alto 250 × 96

Ejecutar desde la raíz del proyecto:
  python lapaz_traffic/scripts/03_rebuild_features.py
"""
from pathlib import Path
import pandas as pd

DATA_RAW  = Path(__file__).parent.parent / "data" / "raw"
DATA_PROC = Path(__file__).parent.parent / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)

DIAS = ["lunes", "miercoles", "viernes", "sabado"]


def build_feature_matrix(segs_csv: Path, traffic_dir: Path, output_csv: Path) -> pd.DataFrame:
    segs = pd.read_csv(segs_csv)[["segment_id"]]
    mat  = segs.set_index("segment_id").copy()

    missing = []
    for dia in DIAS:
        for h in range(24):
            f = traffic_dir / f"traffic_{dia}_{h:02d}.csv"
            col = f"jam_{dia}_{h:02d}"
            if not f.exists():
                missing.append(f.name)
                mat[col] = 0.0
                continue
            df = pd.read_csv(f)[["segment_id", "jam_factor"]]
            mat[col] = df.set_index("segment_id")["jam_factor"]

    mat = mat.fillna(0.0)
    mat.to_csv(output_csv)

    if missing:
        print(f"  ADVERTENCIA — {len(missing)} archivos faltantes (completados con 0):")
        for m in missing[:10]:
            print(f"    {m}")
        if len(missing) > 10:
            print(f"    … y {len(missing)-10} más")

    print(f"  Guardado: {output_csv.name}  shape={mat.shape}")
    return mat


# ── La Paz ────────────────────────────────────────────────────────────────────
print("=== Feature matrix La Paz (4 días × 24 h = 96 columnas) ===")
mat_lp = build_feature_matrix(
    DATA_RAW / "sample_segments.csv",
    DATA_RAW,
    DATA_PROC / "feature_matrix.csv",
)
print(f"  jam_mean global La Paz: {mat_lp.mean().mean():.4f}")

# ── El Alto ───────────────────────────────────────────────────────────────────
print("\n=== Feature matrix El Alto (4 días × 24 h = 96 columnas) ===")
mat_ea = build_feature_matrix(
    DATA_RAW / "sample_segments_el_alto.csv",
    DATA_RAW / "el_alto",
    DATA_PROC / "feature_matrix_el_alto.csv",
)
print(f"  jam_mean global El Alto: {mat_ea.mean().mean():.4f}")

print("\nListo.")
