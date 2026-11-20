"""Comprehensive test suite for ExoHunter pipeline."""
from pathlib import Path
import numpy as np
import pytest

from src.data.loader import LightCurveData, load_lightcurve
from src.data.benchmarks import generate_all_benchmarks, get_benchmark_metadata
from src.preprocessing.cleaner import LightCurveCleaner, preprocess_lightcurve
from src.detection.bls import BLSDetectionEngine, TransitCandidate
from src.modeling.physics import ExoplanetPhysicsModel
from src.vetting.classifier import ExoplanetVettingClassifier


@pytest.fixture(scope="session")
def benchmark_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("benchmarks")
    generate_all_benchmarks(d)
    return d


def test_data_loader(benchmark_dir):
    data = load_lightcurve(benchmark_dir / "kepler-10b.csv", target_name="Kepler-10b")
    assert len(data) > 500
    assert data.target_name == "Kepler-10b"
    assert np.all(np.isfinite(data.flux))
    assert np.all(data.flux_err > 0)


def test_preprocessing(benchmark_dir):
    data = load_lightcurve(benchmark_dir / "kepler-10b.csv")
    cleaned = preprocess_lightcurve(data, sigma_threshold=3.5)
    assert len(cleaned) <= len(data)
    # Median normalized flux should be approximately 1.0
    assert np.isclose(np.median(cleaned.flux), 1.0, atol=0.005)


def test_bls_detection_kepler10(benchmark_dir):
    data = load_lightcurve(
        benchmark_dir / "kepler-10b.csv",
        target_name="Kepler-10b",
        stellar_radius=1.065,
        stellar_mass=0.910,
        stellar_teff=5708.0,
    )
    detrended = preprocess_lightcurve(data)
    engine = BLSDetectionEngine(min_period=0.5, max_period=5.0, n_periods=2000)
    candidate = engine.search(detrended)

    # Injected period is 0.8375 days; should be detected within 1%
    assert np.isclose(candidate.period, 0.8375, rtol=0.02)
    assert candidate.snr > 10.0
    assert candidate.depth > 0.00005


def test_physics_model():
    candidate = TransitCandidate(
        target_name="Test-Planet",
        period=10.0,
        t0=0.5,
        duration=0.1,
        depth=0.0001,  # 100 ppm
        snr=25.0,
        power_peak=100.0,
        periods_tested=np.array([10.0]),
        powers_tested=np.array([100.0]),
        folded_phase=np.array([0.0]),
        folded_flux=np.array([0.9999]),
        binned_phase=np.array([0.0]),
        binned_flux=np.array([0.9999]),
        odd_depth=0.0001,
        even_depth=0.0001,
    )
    stellar_data = LightCurveData(
        time=np.array([0.0, 1.0]),
        flux=np.array([1.0, 1.0]),
        flux_err=np.array([0.001, 0.001]),
        target_name="Test-Star",
        stellar_radius=1.0,  # 1 R_sun
        stellar_mass=1.0,    # 1 M_sun
        stellar_teff=5778.0,
    )
    physics = ExoplanetPhysicsModel().compute(candidate, stellar_data)

    # Depth = 0.0001 -> sqrt = 0.01 -> R_p = 0.01 * 109.2 = ~1.09 R_earth
    assert np.isclose(physics.radius_earth, 1.092, atol=0.05)
    assert physics.planet_type == "Earth-sized / Terrestrial"
    assert physics.semi_major_axis_au > 0.05


def test_ml_vetting_classifier():
    classifier = ExoplanetVettingClassifier()

    # Create a true planet candidate
    planet_cand = TransitCandidate(
        target_name="Real-Planet",
        period=3.5,
        t0=0.2,
        duration=0.08,
        depth=0.0002,
        snr=35.0,
        power_peak=150.0,
        periods_tested=np.array([3.5]),
        powers_tested=np.array([150.0]),
        folded_phase=np.array([0.0]),
        folded_flux=np.array([0.9998]),
        binned_phase=np.array([0.0]),
        binned_flux=np.array([0.9998]),
        odd_depth=0.00020,
        even_depth=0.00019,  # ~5% mismatch
    )
    res_planet = classifier.vet(planet_cand)
    assert "CONFIRMED" in res_planet.verdict
    assert res_planet.confidence >= 65.0

    # Create an eclipsing binary candidate
    binary_cand = TransitCandidate(
        target_name="Eclipsing-Binary",
        period=2.0,
        t0=0.1,
        duration=0.15,
        depth=0.02,
        snr=20.0,
        power_peak=120.0,
        periods_tested=np.array([2.0]),
        powers_tested=np.array([120.0]),
        folded_phase=np.array([0.0]),
        folded_flux=np.array([0.98]),
        binned_phase=np.array([0.0]),
        binned_flux=np.array([0.98]),
        odd_depth=0.020,
        even_depth=0.008,  # >50% mismatch
    )
    res_binary = classifier.vet(binary_cand)
    assert "BINARY" in res_binary.verdict
