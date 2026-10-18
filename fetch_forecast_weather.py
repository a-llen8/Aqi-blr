import sys
import time
import numpy as np
import pandas as pd
import requests

SOURCE = "previous"
for a in sys.argv[1:]:
    if a.startswith("--source="):
        SOURCE = a.split("=", 1)[1]

if SOURCE == "previous":
    ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
    SUFFIX = "_previous_day1"
else:
    ENDPOINT = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    SUFFIX = ""

MODEL = "ecmwf_ifs025"
START = pd.Timestamp("2023-07-01")
END = pd.Timestamp("2026-06-30")
CHUNK_DAYS = 180
OUT = "data/processed/weather_forecast_daily.csv"

STATIONS = {
    "BTM Layout": (12.9135218, 77.5950804),
    "Hebbal": (13.029152, 77.585901),
    "Jayanagar 5th Block": (12.920984, 77.584908),
    "Kasturi Nagar": (13.004533, 77.664561),
    "Peenya": (13.0270199, 77.494094),
    "Silk Board": (12.917348, 77.622813),
}
BASE = ["temperature_2m", "dew_point_2m", "surface_pressure",
        "wind_speed_10m", "wind_direction_10m", "precipitation"]


def get_chunk(lat, lon, a, b):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": a.strftime("%Y-%m-%d"),
        "end_date": b.strftime("%Y-%m-%d"),
        "hourly": ",".join(v + SUFFIX for v in BASE),
        "models": MODEL,
        "timezone": "Asia/Kolkata",
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }
    for attempt in range(1, 5):
        try:
            r = requests.get(ENDPOINT, params=params, timeout=120)
            if r.status_code == 400:
                raise SystemExit("API refused the request: " + r.text[:300])
            r.raise_for_status()
            return r.json()["hourly"]
        except requests.RequestException as exc:
            print("  attempt", attempt, "failed:", exc)
            time.sleep(2 ** attempt)
    raise SystemExit("all retries failed")


def to_frame(hourly):
    h = pd.DataFrame(hourly)
    h["time"] = pd.to_datetime(h["time"])
    ren = {}
    for c in h.columns:
        if c == "time":
            continue
        name = c
        if MODEL and name.endswith("_" + MODEL):
            name = name[: -len(MODEL) - 1]
        if SUFFIX and name.endswith(SUFFIX):
            name = name[: -len(SUFFIX)]
        ren[c] = name
    h = h.rename(columns=ren)
    missing = [v for v in BASE if v not in h.columns]
    if missing:
        raise SystemExit("variables not returned: " + ", ".join(missing))
    return h


def aggregate(h, station):
    theta = np.deg2rad(h["wind_direction_10m"])
    h["wind_u"] = -h["wind_speed_10m"] * np.sin(theta)
    h["wind_v"] = -h["wind_speed_10m"] * np.cos(theta)
    h["date_instant"] = h["time"].dt.normalize()
    h["date_accum"] = (h["time"] - pd.Timedelta(hours=1)).dt.normalize()

    inst = h.groupby("date_instant").agg(
        temp_mean=("temperature_2m", "mean"),
        temp_min=("temperature_2m", "min"),
        temp_max=("temperature_2m", "max"),
        dewpoint_mean=("dew_point_2m", "mean"),
        pressure_mean=("surface_pressure", "mean"),
        wind_speed_mean=("wind_speed_10m", "mean"),
        wind_u_mean=("wind_u", "mean"),
        wind_v_mean=("wind_v", "mean"),
        n_hours_instant=("temperature_2m", "count"),
    )
    acc = h.groupby("date_accum").agg(
        rain_sum=("precipitation", lambda s: s.sum(min_count=1)),
        n_hours_accum=("precipitation", "count"),
    )
    d = inst.join(acc, how="left")
    d["temp_range"] = d["temp_max"] - d["temp_min"]
    d["wind_dir_resultant"] = np.degrees(
        np.arctan2(-d["wind_u_mean"], -d["wind_v_mean"])) % 360
    d.index.name = "Date"
    d = d.reset_index()
    d.insert(0, "Station", station)
    return d[(d["Date"] >= START) & (d["Date"] <= END)]


def main():
    print("source:", SOURCE, "| model:", MODEL, "| endpoint:", ENDPOINT)
    parts = []
    for name, (lat, lon) in STATIONS.items():
        hourly_parts = []
        a = START - pd.Timedelta(days=1)
        while a <= END:
            b = min(a + pd.Timedelta(days=CHUNK_DAYS - 1), END)
            hourly_parts.append(to_frame(get_chunk(lat, lon, a, b)))
            a = b + pd.Timedelta(days=1)
            time.sleep(1)
        h = pd.concat(hourly_parts, ignore_index=True)
        h = h.drop_duplicates("time").sort_values("time")
        parts.append(aggregate(h, name))
        print("done", name, flush=True)

    out = pd.concat(parts, ignore_index=True)
    cols = ["Station", "Date", "temp_mean", "temp_range", "dewpoint_mean",
            "rain_sum", "wind_speed_mean", "pressure_mean",
            "wind_dir_resultant", "n_hours_instant", "n_hours_accum"]
    out[cols].to_csv(OUT, index=False)

    print()
    print("rows:", len(out), "(expected 6576)")
    g = out.groupby("Station")
    print()
    print("coverage per station")
    print(pd.DataFrame({
        "first_day_with_data": g.apply(
            lambda x: x.loc[x["temp_mean"].notna(), "Date"].min()),
        "days_with_data": g["temp_mean"].count(),
        "share_missing_pct": (g["temp_mean"].apply(
            lambda s: s.isna().mean()) * 100).round(1),
    }).to_string())
    print()
    print("if first_day_with_data is later than 2023-07-01, stop and tell me")


if __name__ == "__main__":
    main()
