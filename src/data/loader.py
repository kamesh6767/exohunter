"""Light curve loader and container."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd


@dataclass
class LightCurveData:
    """Standardized photometric light curve container."""
    time: np.ndarray          # Time array (e.g. days in BJD / BTJD)
    flux: np.ndarray          # Relative / normalized flux
    flux_err: np.ndarray      # Flux uncertainties
    target_name: str          # Host star or target identifier
    stellar_radius: float     # Host star radius in Solar Radii (R_sun)
    stellar_mass: float       # Host star mass in Solar Masses (M_sun)
    stellar_teff: float       # Host star effective temperature (K)

    def __len__(self) -> int:
        return len(self.time)


def load_lightcurve(
    filepath: Union[str, Path],
    target_name: Optional[str] = None,
    stellar_radius: float = 1.0,
    stellar_mass: float = 1.0,
    stellar_teff: float = 5778.0,
) -> LightCurveData:
    """Load a light curve from a CSV file.

    Expected CSV columns:
    - time (or 't', 'bjd', 'btjd')
    - flux (or 'f', 'sap_flux', 'pdcsap_flux')
    - [optional] flux_err (or 'flux_error', 'err')
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Light curve file not found: {path}")

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    # Resolve time column
    time_col = None
    for cand in ["time", "t", "bjd", "btjd", "time_days"]:
        if cand in cols:
            time_col = cols[cand]
            break
    if time_col is None:
        time_col = df.columns[0]

    # Resolve flux column
    flux_col = None
    for cand in ["flux", "f", "pdcsap_flux", "sap_flux", "normalized_flux"]:
        if cand in cols:
            flux_col = cols[cand]
            break
    if flux_col is None:
        flux_col = df.columns[1]

    # Resolve error column
    err_col = None
    for cand in ["flux_err", "flux_error", "err", "pdcsap_flux_err"]:
        if cand in cols:
            err_col = cols[cand]
            break

    time = df[time_col].to_numpy(dtype=float)
    flux = df[flux_col].to_numpy(dtype=float)

    if err_col:
        flux_err = df[err_col].to_numpy(dtype=float)
    else:
        # Estimate noise via median absolute deviation if no errors are given
        mad = np.median(np.abs(flux - np.median(flux)))
        flux_err = np.full_like(flux, 1.4826 * mad)

    # Clean NaNs or infinities
    mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    time = time[mask]
    flux = flux[mask]
    flux_err = flux_err[mask]

    name = target_name or path.stem

    return LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        target_name=name,
        stellar_radius=stellar_radius,
        stellar_mass=stellar_mass,
        stellar_teff=stellar_teff,
    )
