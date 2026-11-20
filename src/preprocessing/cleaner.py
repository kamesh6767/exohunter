"""Photometric light curve preprocessing: outlier removal and baseline detrending."""
import warnings
from typing import Tuple
import numpy as np

# Suppress minor wheel compatibility warning between newer numpy and scipy
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

from scipy.signal import savgol_filter
from ..data.loader import LightCurveData


class LightCurveCleaner:
    """Preprocesses raw light curve flux to isolate exoplanetary transit signatures."""

    def __init__(self, sigma_threshold: float = 4.0, savgol_window_hours: float = 12.0):
        self.sigma_threshold = sigma_threshold
        self.savgol_window_hours = savgol_window_hours

    def remove_outliers(
        self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove cosmic ray spikes and instrumental outliers via robust sigma clipping."""
        median = np.median(flux)
        mad = np.median(np.abs(flux - median))
        robust_std = 1.4826 * mad if mad > 0 else np.std(flux)

        mask = np.abs(flux - median) < (self.sigma_threshold * robust_std)
        return time[mask], flux[mask], flux_err[mask]

    def detrend(self, time: np.ndarray, flux: np.ndarray) -> np.ndarray:
        """Remove low-frequency stellar activity and instrumental drift using Savitzky-Golay filtering."""
        if len(time) < 10:
            return flux / np.median(flux)

        dt_median = np.median(np.diff(time))
        if dt_median <= 0:
            dt_median = (time[-1] - time[0]) / max(len(time) - 1, 1)

        window_days = self.savgol_window_hours / 24.0
        window_points = int(round(window_days / dt_median))

        if window_points % 2 == 0:
            window_points += 1
        window_points = max(7, min(window_points, len(flux) - 1 if len(flux) % 2 != 0 else len(flux) - 2))

        try:
            trend = savgol_filter(flux, window_length=window_points, polyorder=2)
            trend = np.where(trend <= 0, 1.0, trend)
            detrended_flux = flux / trend
        except Exception:
            detrended_flux = flux / np.median(flux)

        return detrended_flux


def preprocess_lightcurve(
    data: LightCurveData,
    sigma_threshold: float = 4.0,
    savgol_window_hours: float = 12.0,
) -> LightCurveData:
    """Convenience function to clean outliers and detrend a LightCurveData instance."""
    cleaner = LightCurveCleaner(
        sigma_threshold=sigma_threshold,
        savgol_window_hours=savgol_window_hours
    )
    t_clean, f_clean, e_clean = cleaner.remove_outliers(data.time, data.flux, data.flux_err)
    f_detrended = cleaner.detrend(t_clean, f_clean)

    return LightCurveData(
        time=t_clean,
        flux=f_detrended,
        flux_err=e_clean,
        target_name=data.target_name,
        stellar_radius=data.stellar_radius,
        stellar_mass=data.stellar_mass,
        stellar_teff=data.stellar_teff,
    )
