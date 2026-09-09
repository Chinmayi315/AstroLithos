# AstroLithos — SIH26009 Prototype

**PS:** Using AI/ML and Space Technology to Identify Manganese Reserves and Overcome
Production Shortfalls · Ministry of Steel (MOIL Ltd.) · Software · Space Technology

## Run it
```bash
pip install streamlit pandas numpy scikit-learn plotly
streamlit run app.py
```
Opens at `http://localhost:8501`. Three tabs, matching the PS's three asks exactly:
1. **Reserve Mapping** — flags probable manganese zones from a spectral band-ratio index
2. **Shortfall Prediction** — forecasts next 14 days of production, flags at-risk days
3. **Recommendations** — maps each at-risk day to a corrective action

## What's real vs. simulated (read this before your first demo)
| Module | Technique | Data right now | Upgrade before submission |
|---|---|---|---|
| Reserve mapping | Ferrous/iron-oxide band-ratio index — a real mineral-exploration remote-sensing technique | Simulated band values | Swap `make_reserve_grid()` in `core.py` for a `rasterio` read of a real Sentinel-2/ISRO Bhoonidhi scene over MOIL's mining belt (Nagpur/Bhandara, Maharashtra or Balaghat, MP) |
| Shortfall prediction | RandomForest regression on ops features | Simulated downtime/rainfall/blasting data | Swap `make_production_data()` for real logs if MOIL/mentor can share any, or at minimum cite the (Ministry of Steel / IBM yearbook) production figures to calibrate the baseline realistically |

Being upfront about this split in your pitch is a strength, not a weakness — judges know
students can't get proprietary MOIL ops data or paid satellite tiles in a few weeks. What
they're scoring is whether the *pipeline and technique* are sound end to end.

## Files
- `core.py` — all the data simulation + modeling logic (swap data sources here)
- `app.py` — the Streamlit dashboard (UI only, imports from `core.py`)

## Next concrete steps for your team
1. Get a free Sentinel-2 scene over a known manganese belt (Copernicus Browser or
   ISRO Bhoonidhi) and swap it into `make_reserve_grid()` using `rasterio`.
2. Look up actual published ferrous/iron-oxide index formulas for your specific
   satellite's band numbers (Sentinel-2 bands differ from Landsat) — right now the
   ratio logic is illustrative, not tuned to a specific sensor.
3. If anyone on the team can get even indicative MOIL/IBM (Indian Bureau of Mines)
   production statistics, recalibrate `BASELINE_TONS_PER_DAY` and the feature
   coefficients in `make_production_data()` to match reality more closely.
