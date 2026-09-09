"""
Core logic for the SIH26009 prototype — AstroLithos.
"Using AI/ML and Space Technology to Identify Manganese Reserves and
Overcome Production Shortfalls" (Ministry of Steel / MOIL Ltd.)

IMPORTANT — what's real vs simulated in this prototype:
- Module 1 (reserve mapping) uses SIMULATED multi-band reflectance data
  standing in for real Sentinel-2 / ASTER / ISRO Bhoonidhi imagery. The
  band-ratio technique (ferrous/iron-oxide indices) is the real technique
  used in mineral-exploration remote sensing — only the input pixels are
  synthetic here. Swap `make_reserve_grid()` for a real GeoTIFF reader
  (rasterio) before the actual submission.
- Module 2 (production/shortfall) uses SIMULATED daily mine-ops data
  (MOIL's real downtime/weather/blasting logs aren't public). The
  modeling approach (regression on operational features) is real and
  will work the same way once real data is substituted.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

BASELINE_TONS_PER_DAY = 500  # illustrative capacity, not an official MOIL figure


def make_reserve_grid(size=80, n_hotspots=3, seed=7):
    """Simulate multi-band reflectance over a mine region and derive a
    mineral-probability index from band ratios (the real technique used
    for iron/manganese oxide alteration mapping in remote sensing)."""
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    true_signal = np.zeros((size, size))
    for _ in range(n_hotspots):
        cx, cy = rng.uniform(0.2, 0.8, 2)
        spread = rng.uniform(0.03, 0.07)
        true_signal += np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * spread ** 2)))
    true_signal = np.clip(true_signal, 0, 1)

    def noise(scale):
        return rng.normal(0, scale, (size, size))

    swir = 0.30 + 0.35 * true_signal + noise(0.03)
    nir = 0.40 - 0.20 * true_signal + noise(0.03)
    red = 0.25 + 0.25 * true_signal + noise(0.03)
    blue = 0.20 + noise(0.02)

    ferrous_index = swir / np.clip(nir, 1e-3, None)
    iron_oxide_index = red / np.clip(blue, 1e-3, None)
    combined = (ferrous_index / ferrous_index.max()) * 0.5 + (iron_oxide_index / iron_oxide_index.max()) * 0.5
    return true_signal, combined


def flag_reserve_zones(combined_index, percentile=85):
    threshold = np.percentile(combined_index, percentile)
    return combined_index >= threshold, threshold


def make_production_data(n_days=200, seed=7):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-01-01", periods=n_days, freq="D")
    downtime = np.clip(rng.gamma(1.5, 1.2, n_days), 0, None)
    doy = days.dayofyear.values
    rainfall = np.clip(40 * np.sin((doy - 150) / 365 * 2 * np.pi) + rng.normal(0, 8, n_days) + 20, 0, None)
    blast_delay = rng.choice([0, 0, 0, 0, 1, 2], size=n_days)
    production = (BASELINE_TONS_PER_DAY - 18 * downtime - 0.9 * rainfall - 25 * blast_delay
                  + rng.normal(0, 12, n_days))
    production = np.clip(production, 0, None)
    return pd.DataFrame({"date": days, "downtime_hrs": downtime, "rainfall_mm": rainfall,
                          "blast_delay_days": blast_delay, "tons_produced": production})


def train_shortfall_model(df):
    X, y = df[["downtime_hrs", "rainfall_mm", "blast_delay_days"]], df["tons_produced"]
    model = RandomForestRegressor(n_estimators=200, random_state=7).fit(X, y)
    return model, model.score(X, y)


def forecast_future(model, n_days=14, seed=11):
    rng = np.random.default_rng(seed)
    future = pd.DataFrame({
        "downtime_hrs": np.clip(rng.gamma(1.5, 1.2, n_days), 0, None),
        "rainfall_mm": np.clip(rng.normal(35, 15, n_days), 0, None),
        "blast_delay_days": rng.choice([0, 0, 0, 1, 2], size=n_days),
    })
    future["predicted_tons"] = model.predict(future[["downtime_hrs", "rainfall_mm", "blast_delay_days"]])
    future["shortfall_risk"] = future["predicted_tons"] < 0.8 * BASELINE_TONS_PER_DAY
    future.insert(0, "day", [f"Day {i+1}" for i in range(n_days)])
    return future


RECOMMENDATIONS = {
    "downtime_hrs": "Re-deploy backup equipment / prioritize maintenance window",
    "rainfall_mm": "Shift schedule to avoid peak rainfall hours; check drainage",
    "blast_delay_days": "Pre-clear blasting permits earlier; optimize blast timing",
}


def recommend(row):
    causes = {"downtime_hrs": row["downtime_hrs"] / 6, "rainfall_mm": row["rainfall_mm"] / 60,
              "blast_delay_days": row["blast_delay_days"] / 2}
    return RECOMMENDATIONS[max(causes, key=causes.get)]
