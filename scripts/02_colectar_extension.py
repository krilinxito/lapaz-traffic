"""Recolecta datos de tráfico para la extensión del proyecto.

Pasos:
  1. Miércoles para La Paz (250 segs × 24 h = 6 000 elementos)
  2. 4 días para El Alto (250 segs × 4 días × 24 h = 24 000 elementos)

Costo estimado: ~$300 con Distance Matrix Advanced ($0.01/elemento).

Ejecutar desde la raíz del proyecto:
  python lapaz_traffic/scripts/02_colectar_extension.py

Los archivos existentes se saltan automáticamente (idempotente).
"""
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import googlemaps
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

API_KEY = os.getenv("GMAPS_API_KEY")
if not API_KEY:
    raise ValueError("GMAPS_API_KEY no encontrada en .env")

client  = googlemaps.Client(key=API_KEY)
DATA_RAW = Path(__file__).parent.parent / "data" / "raw"

# ── Helpers ───────────────────────────────────────────────────────────────────
def next_weekday(base, target_wd: int):
    """Próxima fecha con el weekday indicado (0=lunes … 6=domingo)."""
    days = (target_wd - base.weekday()) % 7
    return base + timedelta(days=days if days else 7)


def get_traffic_flow(client, segments_df: pd.DataFrame, departure_time: datetime) -> pd.DataFrame:
    """Consulta Distance Matrix para cada segmento. Retorna DataFrame con jam_factor."""
    records = []
    for _, row in segments_df.iterrows():
        try:
            resp = client.distance_matrix(
                origins=[(row["lat_u"], row["lon_u"])],
                destinations=[(row["lat_v"], row["lon_v"])],
                mode="driving",
                departure_time=departure_time,
                traffic_model="best_guess",
            )
            elem = resp["rows"][0]["elements"][0]
            if elem["status"] != "OK":
                dur_libre = dur_traf = dist_m = 0.0
            else:
                dur_libre = elem["duration"]["value"]
                dur_traf  = elem.get("duration_in_traffic", elem["duration"])["value"]
                dist_m    = elem["distance"]["value"]
            speed_free    = dist_m / dur_libre * 3.6  if dur_libre > 0 else 0.0
            speed_current = dist_m / dur_traf  * 3.6  if dur_traf  > 0 else 0.0
            jam           = max(0.0, min(10.0, (dur_traf / dur_libre - 1) * 10)) if dur_libre > 0 else 0.0
        except Exception as e:
            print(f"    ERROR seg {row['segment_id']}: {e}")
            speed_free = speed_current = jam = 0.0

        records.append({
            "segment_id":    row["segment_id"],
            "street_name":   row.get("name", ""),
            "speed_current": round(speed_current, 2),
            "speed_free":    round(speed_free, 2),
            "jam_factor":    round(jam, 4),
            "timestamp":     departure_time.isoformat(),
            "lat":           row["lat"],
            "lon":           row["lon"],
            "highway":       row["highway"],
        })
        time.sleep(0.05)   # ~20 req/s, bien por debajo del límite

    return pd.DataFrame(records)


def collect_historical(
    client,
    segments_df: pd.DataFrame,
    output_dir: Path,
    day_bases: dict,
):
    """Recolecta datos históricos para los day_bases dados. Salta archivos existentes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(day_bases) * 24 * len(segments_df)
    print(f"  Total elementos: {len(day_bases)} días × 24 h × {len(segments_df)} segs = {total:,}")

    done = 0
    for dia_nombre, dia_base in day_bases.items():
        for hora in range(24):
            out_file = output_dir / f"traffic_{dia_nombre}_{hora:02d}.csv"
            if out_file.exists():
                done += len(segments_df)
                print(f"  [skip] {out_file.name}")
                continue

            # departure_time en UTC
            local_offset = timedelta(hours=-4)   # Bolivia UTC-4
            dt_local = datetime.combine(dia_base, datetime.min.time()).replace(
                hour=hora, tzinfo=timezone(local_offset)
            )
            dt_utc = dt_local.astimezone(timezone.utc)

            print(f"  Consultando {dia_nombre} {hora:02d}:00 … ", end="", flush=True)
            df = get_traffic_flow(client, segments_df, dt_utc)
            df.to_csv(out_file, index=False)
            done += len(segments_df)
            pct = done / total * 100
            print(f"jam_medio={df['jam_factor'].mean():.3f}  ({pct:.1f}%)")


# ── Configuración temporal ─────────────────────────────────────────────────────
today = datetime.now().date()

DAY_BASES_MIERCOLES = {
    "miercoles": next_weekday(today, 2),
}

DAY_BASES_FULL = {
    "lunes":     next_weekday(today, 0),
    "miercoles": next_weekday(today, 2),
    "viernes":   next_weekday(today, 4),
    "sabado":    next_weekday(today, 5),
}

# ── 1. Miércoles La Paz ────────────────────────────────────────────────────────
print("\n=== La Paz — Miércoles ===")
segs_lp = pd.read_csv(DATA_RAW / "sample_segments.csv")
collect_historical(client, segs_lp, output_dir=DATA_RAW, day_bases=DAY_BASES_MIERCOLES)

# ── 2. El Alto — 4 días ────────────────────────────────────────────────────────
print("\n=== El Alto — Lunes / Miércoles / Viernes / Sábado ===")
segs_ea = pd.read_csv(DATA_RAW / "sample_segments_el_alto.csv")
collect_historical(client, segs_ea, output_dir=DATA_RAW / "el_alto", day_bases=DAY_BASES_FULL)

print("\nRecolección finalizada.")
