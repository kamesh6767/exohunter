"""Data ingestion and benchmark data generation module."""
from .loader import load_lightcurve, LightCurveData
from .benchmarks import generate_all_benchmarks, get_benchmark_metadata

__all__ = ["load_lightcurve", "LightCurveData", "generate_all_benchmarks", "get_benchmark_metadata"]
