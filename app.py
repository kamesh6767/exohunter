"""ExoHunter: Interactive Streamlit Web Dashboard for Exoplanet Detection & Vetting."""
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.data.loader import load_lightcurve
from src.data.benchmarks import generate_all_benchmarks, BENCHMARK_TARGETS
from src.preprocessing.cleaner import preprocess_lightcurve
from src.detection.bls import BLSDetectionEngine
from src.modeling.physics import ExoplanetPhysicsModel
from src.vetting.classifier import ExoplanetVettingClassifier
from src.visualization.plots import generate_diagnostic_plot

st.set_page_config(
    page_title="ExoHunter | AI Exoplanet Detection Engine",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1A202C;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #3182CE;
        margin-bottom: 10px;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def ensure_benchmarks():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    generate_all_benchmarks(data_dir)
    return data_dir


data_dir = ensure_benchmarks()

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("🪐 ExoHunter Engine")
    st.caption("AI & Astrophysical Signal Processing for Exoplanet Discovery")
    st.divider()

    st.subheader("1. Select Observation Data")
    source_choice = st.radio(
        "Data Source",
        ["Benchmark Preset Targets", "Upload Custom CSV"],
        index=0,
    )

    if source_choice == "Benchmark Preset Targets":
        preset_options = {
            "Kepler-10b (Rocky Super-Earth)": "kepler-10b",
            "TRAPPIST-1e (Habitable Zone Planet)": "trappist-1e",
            "Eclipsing Binary (Astrophysical False Positive)": "binary-false-positive",
        }
        selected_label = st.selectbox("Preset Target", list(preset_options.keys()))
        target_key = preset_options[selected_label]
        target_meta = BENCHMARK_TARGETS[target_key]

        target_name = target_meta["name"]
        default_r = float(target_meta["stellar_radius"])
        default_m = float(target_meta["stellar_mass"])
        default_teff = float(target_meta["stellar_teff"])
        filepath = data_dir / f"{target_key}.csv"
        st.info(target_meta["description"])
    else:
        uploaded_file = st.file_uploader("Upload Light Curve (CSV)", type=["csv", "txt"])
        target_name = st.text_input("Target Name", value="My-Target-Star")
        default_r = 1.0
        default_m = 1.0
        default_teff = 5778.0
        filepath = None

    st.divider()
    st.subheader("2. Host Star Properties")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        stellar_radius = st.number_input("Radius (R☉)", min_value=0.05, max_value=25.0, value=default_r, step=0.05)
        stellar_mass = st.number_input("Mass (M☉)", min_value=0.05, max_value=20.0, value=default_m, step=0.05)
    with col_s2:
        stellar_teff = st.number_input("Teff (K)", min_value=1500.0, max_value=40000.0, value=default_teff, step=100.0)

    st.divider()
    st.subheader("3. Pipeline Sensitivity")
    min_period = st.slider("Min Period (Days)", 0.2, 5.0, 0.4, 0.1)
    max_period = st.slider("Max Period (Days)", 5.0, 40.0, 20.0, 1.0)
    sigma_clip = st.slider("Outlier Sigma Cut", 2.5, 6.0, 4.0, 0.5)

    st.divider()
    run_btn = st.button("🚀 Run ExoHunter Detection", type="primary", use_container_width=True)

# --- Main App Header ---
st.title("🔭 ExoHunter: Automated Exoplanet Detection & Vetting")
st.markdown("""
*Analyze real and benchmark photometric time-series from NASA's Kepler & TESS missions to detect transiting exoplanets, 
derive physical dimensions, and vet against astrophysical false positives using Machine Learning.*
""")

if run_btn:
    if source_choice == "Upload Custom CSV" and uploaded_file is None:
        st.error("Please upload a CSV file before running the detection engine.")
        st.stop()

    with st.spinner("Processing light curve: Ingesting data & removing stellar variability..."):
        if source_choice == "Upload Custom CSV":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.getvalue())
                target_path = Path(tmp.name)
        else:
            target_path = filepath

        raw_data = load_lightcurve(
            filepath=target_path,
            target_name=target_name,
            stellar_radius=stellar_radius,
            stellar_mass=stellar_mass,
            stellar_teff=stellar_teff,
        )

        detrended_data = preprocess_lightcurve(raw_data, sigma_threshold=sigma_clip)

    with st.spinner("Scanning for periodic transit signatures with Box Least Squares (BLS)..."):
        engine = BLSDetectionEngine(min_period=min_period, max_period=max_period, n_periods=6000)
        candidate = engine.search(detrended_data)

    with st.spinner("Modeling physical planet dimensions & running ML vetting..."):
        physics = ExoplanetPhysicsModel().compute(candidate, detrended_data)
        vetting = ExoplanetVettingClassifier().vet(candidate)

    # --- Top Metrics Row ---
    st.divider()
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Orbital Period", f"{candidate.period:.4f} days")
    with m2:
        st.metric("Planet Radius", f"{physics.radius_earth:.2f} R⊕", f"{physics.radius_jupiter:.3f} R_Jup")
    with m3:
        st.metric("Semi-Major Axis", f"{physics.semi_major_axis_au:.4f} AU")
    with m4:
        st.metric("Eq. Temperature", f"{physics.equilibrium_temp_k:.0f} K")
    with m5:
        st.metric("Detection SNR", f"{candidate.snr:.1f} σ")

    # --- Machine Learning Vetting Verdict Card ---
    st.divider()
    if "CONFIRMED" in vetting.verdict:
        st.success(f"### ✅ Machine Learning Verdict: {vetting.verdict}")
    elif "BINARY" in vetting.verdict:
        st.error(f"### ⚠️ Machine Learning Verdict: {vetting.verdict}")
    else:
        st.warning(f"### ❓ Machine Learning Verdict: {vetting.verdict}")

    col_v1, col_v2 = st.columns([1, 2])
    with col_v1:
        st.write(f"**ML Model Confidence:** `{vetting.confidence:.1f}%`")
        st.write(f"**Odd/Even Depth Mismatch:** `{vetting.odd_even_diff_pct:.1f}%`")
        st.write(f"**Planet Type:** `{physics.planet_type}`")
        st.write(f"**Habitable Zone:** `{physics.habitable_zone_status}`")
    with col_v2:
        st.write("**Vetting Analysis Notes:**")
        for reason in vetting.reasons:
            st.markdown(f"- {reason}")

    # --- Diagnostic Visualizations ---
    st.divider()
    st.subheader("📊 4-Panel Astrophysical Diagnostic Sheet")
    fig = generate_diagnostic_plot(raw_data, detrended_data, candidate, physics, vetting)
    st.pyplot(fig, use_container_width=True)

    # Export Report Button
    out_img_path = Path("assets") / f"{target_name.lower().replace(' ', '_')}_report.png"
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_img_path, dpi=200, bbox_inches="tight")
    with open(out_img_path, "rb") as file_bytes:
        st.download_button(
            label="📥 Download High-Res Discovery Report (PNG)",
            data=file_bytes,
            file_name=f"{target_name}_discovery_report.png",
            mime="image/png",
            use_container_width=True,
        )

else:
    # Landing page welcoming user
    st.info("👈 Select a target in the sidebar and click **'Run ExoHunter Detection'** to start discovering planets!")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        #### 1. Real Physics & Signal Processing
        - Outlier rejection via sigma-clipping
        - Baseline stellar detrending (Savitzky-Golay)
        - Fast, vectorized **Box Least Squares (BLS)** period search
        """)
    with c2:
        st.markdown("""
        #### 2. Planetary Dimensions
        - Kepler's Third Law semi-major axis calculation
        - Radius ratio: $\\delta = (R_p / R_*)^2$
        - Equilibrium temperature & Habitable Zone evaluation
        """)
    with c3:
        st.markdown("""
        #### 3. AI & False-Positive Vetting
        - Odd-even transit depth discrepancy test
        - V-shape vs U-shape geometry analysis
        - Pre-trained Random Forest classifier based on NASA Kepler KOI catalog
        """)
