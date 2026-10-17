# AirSense-IQ Bengaluru — Status & Next Steps

Adapting Sana Jacquilin D's **AirSense-IQ** (Urban Air Quality Prediction & Hospital
Capacity Alert System, Agra) — [report](.) + [github.com/sanajacq/airsense-iq](https://github.com/sanajacq/airsense-iq)
— to Bengaluru, for the CHRIST Seed Money Scheme project *"An Improved Model to
Predict the Air Quality Index in Bangalore City through the application of Machine
Learning."*

---

## 1. Done so far

### Data collection & preprocessing (mirrors Agra Phase 1 / Module 1)
- Raw CPCB station exports (mixed `.csv` / `.xlsx` formats) collected for **6 Bengaluru
  stations**: BTM Layout, Hebbal, Jayanagar 5th Block, Kasturi Nagar, Peenya, Silk Board
  — chosen to match the "6-station" structure of the Agra system.
- Built `bengaluru_preprocessing.ipynb`, an auto-discovery preprocessing notebook
  (drop raw files in a folder, re-run top to bottom, no per-file code edits):
  1. **Discover** raw files, auto-detect format (by extension) and station (by
     filename/metadata keyword match, with a manual-override safety valve).
  2. **Load** via one loader per raw format, standardised to a common column set.
  3. **Clean per station**: drop duplicate dates, reindex to a full daily calendar so
     gaps are explicit, interpolate (time-based, then edge-fill), trim to the study
     window.
  4. **Combine** all stations into one long-format table.
  5. **Validate**: expected row count, duplicate (Station, Date) pairs, remaining
     nulls, per-station date coverage.
  6. **Save** combined CSV + single-sheet Excel.
- Study window: **2023-07-01 → 2026-06-30** (1,096 days).
- Target pollutants: PM2.5, PM10, NO2, SO2, CO, Ozone.

### Output produced
`Bengaluru_AirQuality_combined.csv` — **6,576 rows** (6 stations × 1,096 days),
**zero missing values** after interpolation, with imputation transparently flagged
per pollutant via `<pollutant>_is_imputed` and `<pollutant>_impute_run_len` columns
(not silently filled). Share of values that were interpolated: NO2 ~9.3%, CO ~8.4%,
Ozone ~7.4%, PM10 ~5.8%, PM2.5/SO2 ~5.7%.

Weather variables (RH, WS, WD, SR, BP, AT, RF) were **intentionally deferred** —
AT, RF, and TOT-RF had 0% coverage in the raw BTM Layout files, so those need an
actual external data source rather than interpolation.

### How missing pollutant values were filled

Each station's raw files are first concatenated, de-duplicated on Date, and
reindexed to a **full daily calendar** (`asfreq("D")`) so every missing day becomes
an explicit `NaN` row rather than a silent gap. Two fill steps are then applied, in
order, to each of the 6 pollutant columns independently:

1. **Time-based interpolation** (`interpolate(method="time", limit_direction="both")`)
   — estimates a missing value from the two nearest real readings either side of it,
   weighted by elapsed time rather than row count, so it works even across
   unevenly-spaced gaps.
2. **Forward/backward fill** (`ffill().bfill()`) — catches anything interpolation
   couldn't reach, i.e. gaps that run to the very start or end of a station's data
   (where there's no "far side" reading to interpolate from). This is what covers
   long stretches of missing raw data, not just short ones.

Both steps are applied before the study window is trimmed to
2023-07-01 → 2026-06-30, so an edge gap at the very start/end of the *usable* window
can still be legitimately filled from data just outside it.

**Every filled value is flagged, not hidden.** For each pollutant, two extra columns
travel with it in the combined CSV:
- `<pollutant>_is_imputed` — `True` if that value is estimated rather than measured.
- `<pollutant>_impute_run_len` — length in days of the consecutive gap that value came
  from (0 for real measurements). This is what distinguishes a 2-day interpolated
  blip from a long ffill/bfill stretch — both show up as `is_imputed = True`, but only
  the run length tells them apart.

**How much was actually filled**, over the 6,576-row combined dataset:

| Pollutant | Imputed rows | % of rows | Longest single gap filled | Rows from a >7‑day gap |
|---|---|---|---|---|
| PM2.5 | 373 | 5.7% | 57 days | 217 |
| PM10  | 379 | 5.8% | 57 days | 224 |
| NO2   | 609 | 9.3% | 37 days | 367 |
| SO2   | 373 | 5.7% | 26 days | 198 |
| CO    | 555 | 8.4% | 78 days | 382 |
| Ozone | 488 | 7.4% | 50 days | 295 |

So the majority of imputed rows per pollutant come from gaps longer than a week —
i.e. genuine reporting outages in the raw CPCB exports, filled mostly by
forward/backward-fill rather than short-gap interpolation. That's worth flagging
in the dissertation/report methodology section rather than presenting "5.7–9.3%
imputed" as if it were all short, low-risk interpolation.

PM2.5 imputation is also uneven across stations — **Peenya (149 rows)** and
**BTM Layout (82)** carry noticeably more filled values than **Silk Board (11)** or
**Jayanagar 5th Block (18)**, meaning raw data completeness isn't uniform across the
6 stations.

### Known data quality issues (not caught by the imputation flags)

Structural checks pass cleanly: no duplicate (Station, Date) rows, no calendar
gaps (all 6 stations have exactly 1,096/1,096 expected days), no negative values,
and `_is_imputed`/`_impute_run_len` are internally consistent throughout.

However, four issues exist **in the raw values themselves** — they arrived from the
source files as literal readings, so `is_imputed = False` for all of them and they
were invisible to the interpolation step (which only ever touches true `NaN`s):

1. **PM10 < PM2.5 (physically impossible — PM10 includes PM2.5 by definition)** —
   25 rows, concentrated in Jayanagar 5th Block (13) and Silk Board (6).
2. **One-day extreme outlier** — Silk Board, 2023-09-05: PM2.5 = **794 µg/m³**
   against ~30–70 µg/m³ on the days immediately before/after, with no matching
   rise in that day's PM10 (113). Reads as a raw sensor glitch.
3. **CO reported as exact 0.00** — 147 rows, clustered rather than scattered (e.g.
   13 consecutive days at BTM Layout, 2026-01-01 → 2026-01-13). Ambient CO is
   essentially never exactly zero; this looks like a "missing reading" sentinel
   value baked into the source file rather than a real measurement.
4. **Flatlined / stuck-sensor runs** — an identical value repeated for many
   consecutive days:
   - SO2 @ BTM Layout: round-number runs (11.0, 15.0, 18.0, 13.0, 16.0, 17.0, 12.0)
     covering nearly all of Jan–May 2026.
   - CO @ BTM Layout: 0.0 for 13 straight days (2026-01-01 → 01-13), then 1.0 for
     10 straight days (2026-01-15 → 01-24).
   - **Ozone @ Kasturi Nagar: 39.47 µg/m³ repeated for 45 consecutive days**
     (2024-06-09 → 2024-07-23).
   - Smaller stuck runs also present in CO @ Kasturi Nagar, CO @ Jayanagar 5th
     Block, CO @ Peenya, and PM10 @ Peenya/Kasturi Nagar.

   Item 4 is the highest-risk one: it isn't a handful of bad rows, it's a large
   share of some station-pollutant series' recent history, and it will quietly
   bias any lag/rolling/AQI feature built on top of it. Filtering on `is_imputed`
   will **not** screen these out.

**Suggested fix before feature engineering**: run a physical-consistency pass
(re-null PM10<PM2.5 rows, exact-zero CO, and runs of ≥5 identical consecutive
values that aren't already flagged imputed) and re-apply the same
interpolate→ffill/bfill logic to those re-nulled cells, so they're handled the
same transparent way as the original gaps.

---

## 2. Not started yet — remaining pipeline (mirrors Agra Modules 2–6)

| Stage | Agra reference (module) | Bengaluru status |
|---|---|---|
| Weather data | raw CPCB weather columns | **Not started** — source RH/WS/WD/SR/BP; find an alternate source for AT & RF (e.g. IMD or OpenMeteo, per the Agra report's own "Future Scope") |
| Feature engineering | `train_models.py`, Module 2 — 115 features/station/day: lag-1/2/3/7/14, rolling 3/7/14/30-day mean+std, day/week change-rate, sine/cosine seasonal encodings, weather-interaction terms, cross-station city averages | **Not started** |
| AQI computation | CPCB breakpoint formula, prominent pollutant per station | **Not started** |
| 24h forecast model | XGBoost quantile regression, 18 models (6 pollutants × 10th/50th/90th), target MAPE ≤15% (achieved 8–14% in Agra) | **Not started** |
| 7-day forecast model | PyTorch stacked LSTM, 30-day lookback | **Not started** |
| Alert engine | ERF-based patient surge (β = 0.22%/μg/m³ PM2.5 over WHO 15 μg/m³ threshold), WHO/AIIMS consumables list, MOHFW IPHS 2022 staffing ratios, hospital-to-nearest-station mapping (27 facilities in Agra) | **Not started** — needs a Bengaluru hospital list mapped to the 6 stations; ERF coefficient can likely be reused as-is (WHO-derived, not Agra-specific) but baseline admission rates should be sourced locally if available |
| Backend API | Flask REST endpoints (`/api/dashboard`, `/api/historical_aqi`, `/api/date_detail`, `/api/accuracy`, `/api/chat`, `/api/admin/add_reading`) | **Not started** |
| Frontend dashboard | Flask-served HTML/CSS/JS + Chart.js + Leaflet, recentred on the city | **Not started** — recentre map on Bengaluru station coordinates |
| Testing | Unit tests per module (forecast, alert engine, admin entry, past-data lookup) + chronological 80:20 evaluation | **Not started** |

## 3. Immediate next step: weather data

Yes — **weather is next**, and it comes before feature engineering, because several
of the 115 Agra features are weather-interaction terms (PM2.5 × wind speed, PM2.5 ×
temperature, PM2.5 × relative humidity, plus their squared terms) that can't be built
without it. Two things need sourcing:

1. **RH, WS, WD, SR, BP** — should be obtainable from the same CPCB station
   exports if the portal provides them, the way pollutants were.
2. **AT (air temperature) and RF (rainfall)** — had 0% coverage in the raw files
   seen so far, so these need an external source rather than interpolation, e.g.
   IMD or OpenMeteo (the Agra report's own "Future Scope" section names OpenMeteo's
   free tier for exactly this).

Once weather is merged in, the next step after that is feature engineering — lags,
rolling stats, cyclical encodings, weather-interaction terms, and cross-station city
averages — the Bengaluru equivalent of `train_models.py`'s feature block, feeding
into AQI computation and the XGBoost/LSTM forecast models.
