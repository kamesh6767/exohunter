"""High-performance Box Least Squares (BLS) detection engine.

Implements the Kovács, Zucker & Mazeh (2002) transit search algorithm in pure vectorized NumPy,
providing robust, zero-dependency exoplanet periodic signal extraction.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from ..data.loader import LightCurveData


@dataclass
class TransitCandidate:
    """Container for detected exoplanet transit signal parameters."""
    target_name: str
    period: float             # Orbital period in days
    t0: float                 # Transit epoch (midpoint time in days)
    duration: float           # Transit duration in days
    depth: float              # Fractional transit depth (delta F / F)
    snr: float                # Signal-to-Noise Ratio of transit
    power_peak: float         # Peak power in the BLS periodogram
    periods_tested: np.ndarray
    powers_tested: np.ndarray
    folded_phase: np.ndarray
    folded_flux: np.ndarray
    binned_phase: np.ndarray
    binned_flux: np.ndarray
    odd_depth: float          # Depth of odd-numbered transits
    even_depth: float         # Depth of even-numbered transits


class BLSDetectionEngine:
    """Fast, vectorized Box Least Squares (BLS) periodogram search."""

    def __init__(
        self,
        min_period: float = 0.4,
        max_period: float = 20.0,
        n_periods: int = 4000,
        durations: Optional[Tuple[float, ...]] = None,
        n_phase_bins: int = 250,
    ):
        self.min_period = min_period
        self.max_period = max_period
        self.n_periods = n_periods
        self.durations = durations or (0.015, 0.03, 0.05, 0.08, 0.12, 0.16)
        self.n_phase_bins = n_phase_bins

    def search(self, data: LightCurveData) -> TransitCandidate:
        """Execute BLS period search on normalized light curve data."""
        time = data.time
        flux = data.flux

        # Normalize relative to median baseline
        t_zeroed = time - time[0]
        baseline_flux = np.median(flux)
        y = flux - baseline_flux
        total_pts = len(y)

        # Observation baseline check
        baseline_duration = time[-1] - time[0]
        max_search_p = min(self.max_period, baseline_duration / 2.0)
        min_search_p = max(self.min_period, 2.0 * np.median(np.diff(time)))

        trial_periods = np.linspace(min_search_p, max_search_p, self.n_periods)
        powers = np.zeros(self.n_periods, dtype=float)
        best_durations = np.zeros(self.n_periods, dtype=float)
        best_depths = np.zeros(self.n_periods, dtype=float)
        best_t0_phases = np.zeros(self.n_periods, dtype=float)

        n_bins = self.n_phase_bins
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

        for i, p in enumerate(trial_periods):
            phase = (t_zeroed % p) / p
            bin_idx = np.clip(np.digitize(phase, bin_edges) - 1, 0, n_bins - 1)

            counts = np.bincount(bin_idx, minlength=n_bins)
            sums = np.bincount(bin_idx, weights=y, minlength=n_bins)

            counts_ext = np.concatenate([counts, counts])
            sums_ext = np.concatenate([sums, sums])
            cs_counts = np.cumsum(counts_ext)
            cs_sums = np.cumsum(sums_ext)

            max_p_power = -1.0
            p_best_depth = 0.0
            p_best_dur = 0.0
            p_best_phase = 0.0

            for dur_days in self.durations:
                q = dur_days / p
                if q >= 0.20 or q <= 0.001:
                    continue

                w_bins = max(1, int(round(q * n_bins)))
                w_c = cs_counts[w_bins : w_bins + n_bins] - cs_counts[0:n_bins]
                w_s = cs_sums[w_bins : w_bins + n_bins] - cs_sums[0:n_bins]

                valid = (w_c > 1) & (w_c < total_pts - 1) & (w_s < 0)
                if not np.any(valid):
                    continue

                # Kovacs et al. statistic
                stat = (w_s[valid] ** 2) / (w_c[valid] * (total_pts - w_c[valid]))
                idx_local_max = np.argmax(stat)
                val_max = stat[idx_local_max]

                if val_max > max_p_power:
                    max_p_power = val_max
                    bin_pos = np.where(valid)[0][idx_local_max]
                    p_best_phase = (bin_pos / n_bins) + (0.5 * q)
                    p_best_depth = -w_s[valid][idx_local_max] / w_c[valid][idx_local_max]
                    p_best_dur = dur_days

            if max_p_power > 0:
                powers[i] = max_p_power
                best_depths[i] = p_best_depth
                best_durations[i] = p_best_dur
                best_t0_phases[i] = p_best_phase

        best_idx = np.argmax(powers)
        best_period = float(trial_periods[best_idx])
        best_duration = float(best_durations[best_idx])
        best_depth = float(best_depths[best_idx])
        peak_power = float(powers[best_idx])

        # Transit midpoint t0
        phase_frac = best_t0_phases[best_idx] % 1.0
        best_t0 = float(time[0] + phase_frac * best_period)

        # Phase fold observations
        phase = ((time - best_t0 + 0.5 * best_period) % best_period) - 0.5 * best_period
        sort_idx = np.argsort(phase)
        folded_phase = phase[sort_idx]
        folded_flux = flux[sort_idx]

        # Binned curve for plotting
        binned_phase, binned_flux = self._bin_folded_curve(folded_phase, folded_flux, n_bins=80)

        # Calculate transit Signal-to-Noise Ratio (SNR)
        in_transit = np.abs(folded_phase) < (0.5 * best_duration)
        n_in_transit = np.sum(in_transit)
        out_transit = ~in_transit
        out_noise = np.std(folded_flux[out_transit]) if np.any(out_transit) else 0.001
        snr = float((best_depth / out_noise) * np.sqrt(max(n_in_transit, 1)))

        # Compute Odd vs Even transit depths (vetting eclipsing binaries)
        odd_depth, even_depth = self._compute_odd_even_depths(
            time, flux, best_period, best_t0, best_duration
        )

        return TransitCandidate(
            target_name=data.target_name,
            period=best_period,
            t0=best_t0,
            duration=best_duration,
            depth=max(best_depth, 1e-7),
            snr=snr,
            power_peak=peak_power,
            periods_tested=trial_periods,
            powers_tested=powers,
            folded_phase=folded_phase,
            folded_flux=folded_flux,
            binned_phase=binned_phase,
            binned_flux=binned_flux,
            odd_depth=odd_depth,
            even_depth=even_depth,
        )

    def _bin_folded_curve(
        self, phase: np.ndarray, flux: np.ndarray, n_bins: int = 80
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bin folded phase data into discrete chunks."""
        bins = np.linspace(phase[0], phase[-1], n_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_flux = np.ones(n_bins, dtype=float)

        digitized = np.clip(np.digitize(phase, bins) - 1, 0, n_bins - 1)
        for i in range(n_bins):
            pts = flux[digitized == i]
            if len(pts) > 0:
                bin_flux[i] = np.median(pts)
            else:
                bin_flux[i] = 1.0

        return bin_centers, bin_flux

    def _compute_odd_even_depths(
        self, time: np.ndarray, flux: np.ndarray, period: float, t0: float, duration: float
    ) -> Tuple[float, float]:
        """Compute transit depths separately for odd and even events."""
        transit_number = np.round((time - t0) / period).astype(int)
        phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
        in_transit = np.abs(phase) < (0.5 * duration)

        is_odd = (transit_number % 2 != 0) & in_transit
        is_even = (transit_number % 2 == 0) & in_transit

        out_transit = np.abs(phase) > (duration)
        baseline = np.median(flux[out_transit]) if np.any(out_transit) else 1.0

        n_odd = np.sum(is_odd)
        n_even = np.sum(is_even)

        if n_odd < 3 or n_even < 3:
            return 0.001, 0.001

        odd_depth = float(baseline - np.median(flux[is_odd]))
        even_depth = float(baseline - np.median(flux[is_even]))

        return max(odd_depth, 1e-7), max(even_depth, 1e-7)
