"""Command-line interface for the ExoHunter detection engine."""
import argparse
from pathlib import Path
import sys

from src.data.loader import load_lightcurve
from src.data.benchmarks import generate_all_benchmarks, get_benchmark_metadata
from src.preprocessing.cleaner import preprocess_lightcurve
from src.detection.bls import BLSDetectionEngine
from src.modeling.physics import ExoplanetPhysicsModel
from src.vetting.classifier import ExoplanetVettingClassifier
from src.visualization.plots import generate_diagnostic_plot


def run_pipeline(
    filepath: Path,
    target_name: str,
    r_star: float,
    m_star: float,
    teff: float,
    save_plot: bool = True,
    output_dir: Path = Path("assets"),
):
    print("=" * 65)
    print(f"  EXOHUNTER: Exoplanet Detection & Vetting Pipeline")
    print("=" * 65)
    print(f"[*] Target: {target_name}")
    print(f"[*] Stellar Parameters: Radius={r_star:.3f} R_sun, Mass={m_star:.3f} M_sun, Teff={teff:.0f} K")
    print(f"[*] Loading data from: {filepath}")

    # 1. Ingest Data
    raw_data = load_lightcurve(
        filepath=filepath,
        target_name=target_name,
        stellar_radius=r_star,
        stellar_mass=m_star,
        stellar_teff=teff,
    )
    print(f"    - Loaded {len(raw_data)} photometric observations over {raw_data.time[-1] - raw_data.time[0]:.1f} days.")

    # 2. Preprocessing & Detrending
    print("[*] Running sigma-clipping & Savitzky-Golay baseline detrending...")
    detrended = preprocess_lightcurve(raw_data)

    # 3. BLS Periodogram Search
    print("[*] Computing Box Least Squares (BLS) period spectrum...")
    engine = BLSDetectionEngine(min_period=0.4, max_period=20.0, n_periods=6000)
    candidate = engine.search(detrended)
    print(f"    - Dominant Signal Detected at Period: {candidate.period:.5f} days")
    print(f"    - Transit Depth: {candidate.depth * 1e6:.1f} ppm ({candidate.depth * 100:.3f}%)")
    print(f"    - Transit Duration: {candidate.duration * 24.0:.2f} hours")
    print(f"    - Detection Significance (SNR): {candidate.snr:.1f} sigma")

    # 4. Astrophysical Modeling
    print("[*] Deriving physical planet dimensions & habitability...")
    physics = ExoplanetPhysicsModel().compute(candidate, detrended)
    print(f"    - Planet Radius: {physics.radius_earth:.2f} R_earth ({physics.radius_jupiter:.3f} R_jup)")
    print(f"    - Classification: {physics.planet_type}")
    print(f"    - Semi-Major Axis: {physics.semi_major_axis_au:.4f} AU")
    print(f"    - Equilibrium Temperature: {physics.equilibrium_temp_k:.0f} K")
    print(f"    - Habitable Zone Status: {physics.habitable_zone_status}")

    # 5. Machine Learning Vetting
    print("[*] Running Machine Learning False-Positive Vetting...")
    vetting = ExoplanetVettingClassifier().vet(candidate)
    print("-" * 65)
    print(f"  >>> ML VERDICT: {vetting.verdict}")
    print(f"  >>> CONFIDENCE: {vetting.confidence:.1f}%")
    print(f"  >>> Odd/Even Transit Depth Mismatch: {vetting.odd_even_diff_pct:.1f}%")
    for r in vetting.reasons:
        print(f"      * {r}")
    print("-" * 65)

    # 6. Diagnostic Plot
    if save_plot:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / f"{target_name.lower().replace(' ', '_')}_diagnostic.png"
        print(f"[*] Generating 4-panel diagnostic discovery figure -> {plot_path}")
        generate_diagnostic_plot(raw_data, detrended, candidate, physics, vetting, save_path=plot_path)
        print("[+] Discovery report generated successfully!")

    print("=" * 65)
    return candidate, physics, vetting


def main():
    parser = argparse.ArgumentParser(description="ExoHunter: Exoplanet Detection & Vetting CLI")
    parser.add_argument("--benchmark", type=str, choices=["kepler-10b", "trappist-1e", "binary-false-positive"],
                        default="kepler-10b", help="Run on built-in benchmark dataset")
    parser.add_argument("--file", type=str, default=None, help="Path to custom light curve CSV")
    parser.add_argument("--target", type=str, default=None, help="Target star name")
    parser.add_argument("--radius", type=float, default=1.0, help="Host star radius (Solar Radii)")
    parser.add_argument("--mass", type=float, default=1.0, help="Host star mass (Solar Masses)")
    parser.add_argument("--teff", type=float, default=5778.0, help="Host star effective temperature (K)")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving diagnostic plot")
    args = parser.parse_args()

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    if args.file:
        filepath = Path(args.file)
        name = args.target or filepath.stem
        r_star = args.radius
        m_star = args.mass
        teff = args.teff
    else:
        # Benchmark mode
        bench_key = args.benchmark
        meta = get_benchmark_metadata(bench_key)
        filepath = data_dir / f"{bench_key}.csv"
        if not filepath.exists():
            print(f"[*] Generating benchmark datasets in {data_dir}...")
            generate_all_benchmarks(data_dir)
        name = meta["name"]
        r_star = meta["stellar_radius"]
        m_star = meta["stellar_mass"]
        teff = meta["stellar_teff"]

    run_pipeline(
        filepath=filepath,
        target_name=name,
        r_star=r_star,
        m_star=m_star,
        teff=teff,
        save_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
