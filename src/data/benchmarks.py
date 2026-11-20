"""Generates high-fidelity benchmark datasets for classic exoplanet discoveries."""
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd


BENCHMARK_TARGETS: Dict[str, Dict[str, Any]] = {
    "kepler-10b": {
        "name": "Kepler-10b",
        "description": "First confirmed rocky super-Earth discovered by NASA Kepler.",
        "period": 0.837495,         # days
        "t0": 0.15,                 # transit epoch (days)
        "duration": 0.075,          # days (~1.8 hours)
        "depth": 0.000165,          # 165 ppm dip
        "stellar_radius": 1.065,    # R_sun
        "stellar_mass": 0.910,      # M_sun
        "stellar_teff": 5708.0,     # Kelvin
        "true_radius_earth": 1.47,  # R_earth
        "is_planet": True,
        "is_binary": False,
        "cadence_mins": 29.4,       # Kepler long cadence
        "days_observed": 30.0,
        "noise_std": 0.000045,      # photometric noise
    },
    "trappist-1e": {
        "name": "TRAPPIST-1e",
        "description": "Earth-sized planet in the habitable zone of an ultra-cool red dwarf.",
        "period": 6.099615,         # days
        "t0": 0.82,                 # days
        "duration": 0.038,          # days (~55 minutes)
        "depth": 0.0051,            # 0.51% dip (small star, deep transit)
        "stellar_radius": 0.1192,   # R_sun
        "stellar_mass": 0.0898,     # M_sun
        "stellar_teff": 2566.0,     # Kelvin
        "true_radius_earth": 0.92,  # R_earth
        "is_planet": True,
        "is_binary": False,
        "cadence_mins": 2.0,        # TESS / high-cadence
        "days_observed": 45.0,
        "noise_std": 0.0009,
    },
    "binary-false-positive": {
        "name": "Eclipsing-Binary-FP",
        "description": "Grazing eclipsing binary with alternating primary/secondary eclipse depths.",
        "period": 2.4501,           # days
        "t0": 0.40,
        "duration": 0.12,
        "depth": 0.025,             # Primary eclipse depth (2.5%)
        "secondary_depth": 0.012,   # Secondary eclipse at phase 0.5 (1.2%)
        "stellar_radius": 1.2,
        "stellar_mass": 1.1,
        "stellar_teff": 6200.0,
        "true_radius_earth": 18.5,  # Stellar companion, not a planet
        "is_planet": False,
        "is_binary": True,
        "cadence_mins": 10.0,
        "days_observed": 25.0,
        "noise_std": 0.0015,
    },
}


def _simulate_transit_lightcurve(params: Dict[str, Any], seed: int = 42) -> pd.DataFrame:
    """Simulate a realistic light curve with limb-darkened transit dips, stellar variability, and noise."""
    rng = np.random.default_rng(seed)

    cadence_days = params["cadence_mins"] / (24.0 * 60.0)
    time = np.arange(0.0, params["days_observed"], cadence_days)
    n_points = len(time)

    # 1. Baseline Normalized Stellar Flux
    flux = np.ones(n_points, dtype=float)

    # 2. Add Low-Frequency Stellar Activity (e.g. Starspot rotation)
    spot_period = 8.5  # days
    flux += 0.0003 * np.sin(2 * np.pi * time / spot_period)

    # 3. Inject Primary Transits
    period = params["period"]
    t0 = params["t0"]
    duration = params["duration"]
    depth = params["depth"]

    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    in_transit = np.abs(phase) < (0.5 * duration)

    # Simplified limb-darkened transit shape: U-shaped dip
    transit_shape = np.clip(1.0 - (2.0 * np.abs(phase[in_transit]) / duration) ** 4, 0.0, 1.0)
    flux[in_transit] -= depth * transit_shape

    # 4. Inject Secondary Eclipse if it's an eclipsing binary
    if params.get("is_binary", False):
        sec_depth = params.get("secondary_depth", depth * 0.5)
        sec_phase = ((time - (t0 + 0.5 * period) + 0.5 * period) % period) - 0.5 * period
        in_sec = np.abs(sec_phase) < (0.5 * duration)
        sec_shape = np.clip(1.0 - (2.0 * np.abs(sec_phase[in_sec]) / duration) ** 4, 0.0, 1.0)
        flux[in_sec] -= sec_depth * sec_shape

    # 5. Add Gaussian Instrumental & Photon Noise
    noise = rng.normal(0.0, params["noise_std"], size=n_points)
    flux += noise

    flux_err = np.full_like(flux, params["noise_std"])

    return pd.DataFrame({
        "time": np.round(time, 6),
        "flux": np.round(flux, 7),
        "flux_err": np.round(flux_err, 7),
    })


def generate_all_benchmarks(output_dir: Path) -> Dict[str, Path]:
    """Generate and save all benchmark CSV datasets to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = {}

    for key, params in BENCHMARK_TARGETS.items():
        df = _simulate_transit_lightcurve(params)
        filepath = output_dir / f"{key}.csv"
        df.to_csv(filepath, index=False)
        generated_files[key] = filepath

    return generated_files


def get_benchmark_metadata(target_key: str) -> Dict[str, Any]:
    """Retrieve metadata for a benchmark target."""
    clean_key = target_key.lower().replace("_", "-")
    if clean_key not in BENCHMARK_TARGETS:
        raise KeyError(f"Unknown benchmark target '{target_key}'. Available: {list(BENCHMARK_TARGETS.keys())}")
    return BENCHMARK_TARGETS[clean_key]
