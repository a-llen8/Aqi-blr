import sys
import numpy as np
import pandas as pd
import xgboost as xgb

FEATURES = "data/processed/Bengaluru_features.csv"
FEAT_DICT = "data/processed/feature_dictionary.csv"
AQI_DAILY = "data/processed/Bengaluru_AQI_daily.csv"
FORECAST_WX = "data/processed/weather_forecast_daily.csv"
OUT_TABLE = "eda_outputs/experiment1_results.csv"

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]
EXCLUDE_PREFIXES = ("CO_", "NO2_")
SEEDS = [42, 43, 44]
FOLDS = [
    ("2024-07-01", "2024-12-31"),
    ("2025-01-01", "2025-06-30"),
    ("2025-07-01", "2025-12-31"),
    ("2026-01-01", "2026-06-30"),
]
XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=2.0,
    objective="reg:quantileerror",
    quantile_alpha=0.5,
    tree_method="hist",
)
DAILY_WX = ["temp_mean", "temp_range", "dewpoint_mean", "rain_sum",
            "wind_speed_mean", "pressure_mean"]
BOOT_BLOCK_DAYS = 14
BOOT_REPS = 2000


def clean_flags():
    a = pd.read_csv(AQI_DAILY, parse_dates=["Date"])
    clean = pd.Series(False, index=a.index)
    for p in POLLUTANTS:
        imp = a[p + "_is_imputed"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        clean |= (a["aqi_prominent_pollutant"] == p) & ~imp
    return pd.DataFrame({"Station": a["Station"], "Date": a["Date"],
                         "clean": clean.to_numpy()})


def add_forecast_columns(df):
    fc = pd.read_csv(FORECAST_WX, parse_dates=["Date"])
    keep = ["Station", "Date"] + DAILY_WX + ["wind_dir_resultant"]
    fc = fc[keep].rename(columns={c: "fc_" + c for c in keep[2:]})
    df = df.merge(fc, on=["Station", "Date"], how="left", validate="one_to_one")
    for w in DAILY_WX:
        df["fcast_wxfc_" + w] = df["fc_" + w]
    rad = np.deg2rad(df["fc_wind_dir_resultant"])
    df["fcast_wxfc_wind_dir_sin"] = np.sin(rad)
    df["fcast_wxfc_wind_dir_cos"] = np.cos(rad)
    for p in ["PM2.5", "PM10"]:
        for w in ["wind_speed_mean", "temp_range", "rain_sum"]:
            df["fcast_ix_%s_x_%s" % (p, w)] = df[p + "_lag1"] * df["fc_" + w]
    return df


def feature_groups(df, fd):
    key = "feature" if "feature" in fd.columns else fd.columns[0]
    allf = [c for c in fd[key] if c in df.columns
            and not c.startswith(EXCLUDE_PREFIXES)]
    grp = dict(zip(fd[key], fd["group"]))
    obs = [f for f in allf if grp[f] == "weather-observed"]
    same = [f for f in allf if grp[f] in ("weather-same-day", "interaction")]
    hist = [f for f in allf if f not in obs and f not in same]
    return hist, obs, same


def fit(X, y, seed):
    m = xgb.XGBRegressor(**dict(XGB_PARAMS, random_state=seed))
    m.fit(X, y, verbose=False)
    return m


def avg_predict(models, X):
    return np.mean([m.predict(X) for m in models], axis=0)


def score(frames):
    d = pd.concat(frames).sort_values(["Date", "Station"]).reset_index(drop=True)
    e = d["pred"] - d["actual"]
    hi = d["actual"] > 100
    row = dict(
        n=len(d),
        MAE=round(float(e.abs().mean()), 2),
        RMSE=round(float(np.sqrt((e ** 2).mean())), 2),
        R2=round(float(1 - (e ** 2).sum() /
                       ((d["actual"] - d["actual"].mean()) ** 2).sum()), 3),
        MAE_AQI_gt100=round(float(e[hi].abs().mean()), 2),
        n_gt100=int(hi.sum()),
    )
    return row, d


def block_bootstrap_diff(d_ref, d_new, seed=0):
    rng = np.random.default_rng(seed)
    diff = ((d_new["pred"] - d_new["actual"]).abs().values
            - (d_ref["pred"] - d_ref["actual"]).abs().values)
    days = d_ref["Date"].dt.normalize().values
    uniq = np.unique(days)
    idx_by_day = {d: np.where(days == d)[0] for d in uniq}
    nb = int(np.ceil(len(uniq) / BOOT_BLOCK_DAYS))
    stats = []
    for _ in range(BOOT_REPS):
        starts = rng.integers(0, len(uniq) - BOOT_BLOCK_DAYS + 1, nb)
        sel = np.concatenate([idx_by_day[uniq[j]]
                              for s in starts
                              for j in range(s, s + BOOT_BLOCK_DAYS)])
        stats.append(diff[sel].mean())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def forecast_quality(df):
    t = df[(df["Date"] >= FOLDS[0][0]) & (df["Date"] <= FOLDS[-1][1])]
    print()
    print("How wrong is the forecast weather vs the reanalysis? (test window)")
    rows = []
    for w in DAILY_WX:
        a = t["wxfc_" + w]
        b = t["fcast_wxfc_" + w]
        ok = a.notna() & b.notna()
        rows.append(dict(variable=w, n=int(ok.sum()),
                         corr=round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 3),
                         MAE=round(float((a[ok] - b[ok]).abs().mean()), 3),
                         bias=round(float((b[ok] - a[ok]).mean()), 3)))
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    use_forecast = "--forecast" in sys.argv
    df = pd.read_csv(FEATURES, parse_dates=["Date"])
    fd = pd.read_csv(FEAT_DICT)
    df = df.merge(clean_flags(), on=["Station", "Date"], how="left")
    df["clean"] = df["clean"].fillna(False)
    df = df[df["aqi"].notna()].copy()
    if use_forecast:
        df = add_forecast_columns(df)
        forecast_quality(df)
    hist, obs, same = feature_groups(df, fd)
    codes = {s: i for i, s in enumerate(sorted(df["Station"].unique()))}
    df["station_id"] = df["Station"].map(codes)
    swapped = ["fcast_" + f for f in same]

    sets = {
        "history_only": hist,
        "+obs_weather_t-1": hist + obs,
    }
    frames = {k: [] for k in sets}
    frames["+same_day_reanalysis"] = []
    if use_forecast:
        frames["+same_day_FORECAST (trained on reanalysis)"] = []
        frames["+same_day_FORECAST (trained on forecast)"] = []
    persist = []

    for start, end in FOLDS:
        tr = df[df["Date"] < start]
        te = df[(df["Date"] >= start) & (df["Date"] <= end)]
        te = te[te["clean"] & te["aqi_lag1"].notna()]

        def pack(pred):
            return pd.DataFrame({"Station": te["Station"].values,
                                 "Date": te["Date"].values,
                                 "actual": te["aqi"].values, "pred": pred})

        persist.append(pack(te["aqi_lag1"].values))
        for name, cols in sets.items():
            c = cols + ["station_id"]
            ms = [fit(tr[c], tr["aqi"], s) for s in SEEDS]
            frames[name].append(pack(avg_predict(ms, te[c])))

        c_full = hist + obs + same + ["station_id"]
        ms = [fit(tr[c_full], tr["aqi"], s) for s in SEEDS]
        frames["+same_day_reanalysis"].append(pack(avg_predict(ms, te[c_full])))

        if use_forecast:
            te_fc = te[c_full].copy()
            te_fc[same] = te[swapped].values
            frames["+same_day_FORECAST (trained on reanalysis)"].append(
                pack(avg_predict(ms, te_fc)))

            c_fc = hist + obs + swapped + ["station_id"]
            ms2 = [fit(tr[c_fc], tr["aqi"], s) for s in SEEDS]
            frames["+same_day_FORECAST (trained on forecast)"].append(
                pack(avg_predict(ms2, te[c_fc])))
        print("fold", start, "to", end, "done", flush=True)

    table = {"persistence": score(persist)[0]}
    scored = {}
    for name, fr in frames.items():
        table[name], scored[name] = score(fr)
    out = pd.DataFrame(table).T
    print()
    print(out.to_string())
    out.to_csv(OUT_TABLE)

    print()
    print("Paired block bootstrap, MAE difference vs history_only "
          "(negative = better), 95% CI")
    ref = scored["history_only"]
    for name, d in scored.items():
        if name == "history_only":
            continue
        m, lo, hi = block_bootstrap_diff(ref, d)
        print("%-46s %+.3f  [%+.3f, %+.3f]" % (name, m, lo, hi))

    if use_forecast:
        print()
        print("Same bootstrap, forecast vs reanalysis (negative = forecast better)")
        a = scored["+same_day_reanalysis"]
        for name in ["+same_day_FORECAST (trained on reanalysis)",
                     "+same_day_FORECAST (trained on forecast)"]:
            m, lo, hi = block_bootstrap_diff(a, scored[name])
            print("%-46s %+.3f  [%+.3f, %+.3f]" % (name, m, lo, hi))


if __name__ == "__main__":
    main()