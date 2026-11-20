"""Generates 4-panel astrophysical diagnostic reports."""
from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np

from ..data.loader import LightCurveData
from ..detection.bls import TransitCandidate
from ..modeling.physics import PhysicalProperties
from ..vetting.classifier import VettingResult


def generate_diagnostic_plot(
    raw_data: LightCurveData,
    detrended_data: LightCurveData,
    candidate: TransitCandidate,
    physics: PhysicalProperties,
    vetting: VettingResult,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Generate a multi-panel publication-grade discovery sheet."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [1, 1, 1, 1.3]})
    plt.subplots_adjust(hspace=0.38)

    # Styling colors
    c_raw = "#4A5568"
    c_detrend = "#2B6CB0"
    c_bls = "#D69E2E"
    c_transit = "#E53E3E"
    c_bin = "#2D3748"

    # --- Panel 1: Raw Flux ---
    ax1 = axes[0]
    ax1.scatter(raw_data.time, raw_data.flux, s=1.5, color=c_raw, alpha=0.45, label="Raw Stellar Photometry")
    ax1.set_title(f"Target: {raw_data.target_name} | Raw Normalized Flux", fontsize=12, fontweight="bold", loc="left")
    ax1.set_xlabel("Time (Barycentric Days)", fontsize=10)
    ax1.set_ylabel("Relative Flux", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", framealpha=0.9)

    # --- Panel 2: Detrended Flux ---
    ax2 = axes[1]
    ax2.scatter(detrended_data.time, detrended_data.flux, s=1.5, color=c_detrend, alpha=0.5, label="Savitzky-Golay Detrended")
    # Mark expected transits
    transits = np.arange(candidate.t0, detrended_data.time[-1], candidate.period)
    for t in transits:
        if detrended_data.time[0] <= t <= detrended_data.time[-1]:
            ax2.axvline(t, color=c_transit, linestyle=":", alpha=0.4, linewidth=1.2)
    ax2.set_title("Stellar Activity Removed & Isolated Transits", fontsize=12, fontweight="bold", loc="left")
    ax2.set_xlabel("Time (Barycentric Days)", fontsize=10)
    ax2.set_ylabel("Detrended Flux", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", framealpha=0.9)

    # --- Panel 3: BLS Periodogram ---
    ax3 = axes[2]
    ax3.plot(candidate.periods_tested, candidate.powers_tested, color=c_bls, linewidth=1.2, label="BLS Power Spectrum")
    ax3.axvline(candidate.period, color=c_transit, linestyle="--", linewidth=1.5, label=f"Peak: P = {candidate.period:.4f} days")
    ax3.set_title(f"Box Least Squares Period Search | Max SNR = {candidate.snr:.1f}σ", fontsize=12, fontweight="bold", loc="left")
    ax3.set_xlabel("Trial Period (Days)", fontsize=10)
    ax3.set_ylabel("Detection Power", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper right", framealpha=0.9)

    # --- Panel 4: Phase-Folded Transit Profile ---
    ax4 = axes[3]
    ax4.scatter(candidate.folded_phase, candidate.folded_flux, s=2.5, color="#A0AEC0", alpha=0.35, label="Folded Observations")
    ax4.plot(candidate.binned_phase, candidate.binned_flux, "o-", color=c_bin, markersize=4, linewidth=1.5, label="Binned Median (80 bins)")

    # Analytic transit model box
    duration = candidate.duration
    depth = candidate.depth
    model_phase = np.linspace(-duration * 1.5, duration * 1.5, 300)
    in_box = np.abs(model_phase) < (0.5 * duration)
    model_flux = np.ones_like(model_phase)
    # U-shaped transit profile with limb-darkened rounding
    model_flux[in_box] = 1.0 - depth * (1.0 - (2.0 * np.abs(model_phase[in_box]) / duration) ** 4)
    ax4.plot(model_phase, model_flux, color=c_transit, linewidth=2.5, label="Analytic Transit Fit")

    ax4.set_xlim(-duration * 2.2, duration * 2.2)
    ax4.set_title(
        f"Folded Transit Profile | Radius = {physics.radius_earth:.2f} R⊕ ({physics.planet_type})",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    ax4.set_xlabel("Phase (Days from Mid-Transit)", fontsize=10)
    ax4.set_ylabel("Normalized Flux", fontsize=10)
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend(loc="lower right", framealpha=0.9)

    # Super Title with ML Vetting Verdict
    color_verdict = "#22543D" if "CONFIRMED" in vetting.verdict else "#742A2A"
    fig.suptitle(
        f"ExoHunter Detection Report: {raw_data.target_name}\nVerdict: {vetting.verdict} ({vetting.confidence:.1f}% confidence)",
        fontsize=15,
        fontweight="bold",
        y=0.995,
        color=color_verdict,
    )

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")

    return fig
