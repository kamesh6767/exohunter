# 📖 The Non-Astronomy Guide to Your Exoplanet Project

Welcome! You do **not** need a degree in astrophysics or a PhD in Machine Learning to understand and present this project. 

This guide breaks down **everything** in plain English: what the concepts mean, why we built it this way, and exactly what to say when people ask you about it in interviews or on LinkedIn.

---

## 1. The Core Idea in 60 Seconds (The Elevator Pitch)

> **"Stars are billions of times brighter than the planets orbiting them. So instead of trying to take a direct photo of a planet, we point a space telescope at a star and measure its brightness every few minutes. When a planet passes directly between its star and our telescope, it blocks a tiny sliver of starlight—causing a mini-eclipse called a transit. My project, `ExoHunter`, takes that noisy raw brightness data, removes stellar flares, finds the periodic heartbeat of the orbit using Box Least Squares, calculates the planet's physical size and habitability, and uses Machine Learning to rule out false alarms like binary stars."**

---

## 2. Key Concepts Explained Simply

### A. Transit Photometry (The "Fly in Front of the Streetlight")
- Imagine watching a bright streetlight at night from 2 miles away. A tiny fly walks across the lens.
- You cannot see the fly directly, but a sensitive sensor detects the streetlight dimmed by **0.01%**.
- If that dimming repeats rhythmically every 20 seconds, you know something is orbiting the streetlight.
- **In our code:** This is `flux` (brightness) plotted over `time`.

### B. Why Is Raw Data So Hard to Use?
Stars are not lightbulbs—they bubble, rotate, have magnetic storms, and have giant dark spots (sunspots). Space telescopes also jitter.
- If you don't clean the data, stellar noise will drown out the tiny planet signal.
- **In our code (`src/preprocessing/`):**
  - **Sigma Clipping:** Deletes random cosmic ray spikes that hit the camera sensor.
  - **Savitzky-Golay Filter:** A rolling curve that models the slow bubbling/rotation of the star and subtracts it, leaving only the sharp dips caused by planets.

### C. Box Least Squares (BLS) Algorithm (The "Rhythm Hunter")
- A planet's transit lasts only a couple of hours every few weeks. In a 3-year light curve, individual dips are nearly invisible.
- The **BLS algorithm** acts like a rhythmic tuner: it tests thousands of trial orbital periods (e.g., test 1.0 days, 1.01 days, 1.02 days... up to 30 days).
- For each test period, it "folds" time (stacking all the data into a single repeating cycle).
- When it hits the *true* orbital period, all the tiny dips line up on top of each other into one clear, deep, U-shaped valley!
- **In our code (`src/detection/`):** Pure vectorized NumPy implementation of the Kovács, Zucker & Mazeh (2002) algorithm.

### D. The Physics Engine: Turning Dips into Real Numbers
How do we know how big the planet is and if it's habitable?
1. **Planet Size ($R_p$):** 
   $$\text{Transit Depth } \delta = \left(\frac{R_{\text{planet}}}{R_{\text{star}}}\right)^2 \implies R_{\text{planet}} = R_{\text{star}} \times \sqrt{\delta}$$
   If a star is the size of the Sun and dims by 0.01%, the planet is roughly the size of Earth!
2. **Orbital Distance ($a$ in AU):**
   Using **Kepler's Third Law**, the orbital period tells us how far the planet orbits from its star.
3. **Habitability ($T_{\text{eq}}$):**
   Knowing the star's heat and the planet's distance, we calculate its equilibrium temperature. If it's between **180 K and 315 K** (-90°C to +42°C), liquid water could exist on its surface!

### E. Why Machine Learning? (The Eclipsing Binary Trap)
In astronomy, **over 50% of candidate signals are false positives!**
- The most common trap is an **Eclipsing Binary**: two stars orbiting each other. One star passes in front of the other, mimicking a planet transit.
- **How our AI catches them:**
  - In a twin-star system, Star A is usually hotter/larger than Star B.
  - When Star A eclipses Star B, you get a **deep** dip. Half an orbit later, Star B eclipses Star A, giving a **shallow** dip.
  - An exoplanet, on the other hand, produces **identical dips on both odd and even orbits**.
  - Our **Random Forest model (`src/vetting/`)** looks at the odd-even mismatch, duration, and signal-to-noise ratio to classify whether it's a real planet or a false-positive binary star.

---

## 3. How to Run & Demo Your Project

### Option 1: The Interactive Web App (Best for Screen Recordings & Demos)
Run this command in your terminal:
```bash
streamlit run app.py
```
1. Open the browser link (`http://localhost:8501`).
2. In the sidebar, select **Kepler-10b (Rocky Super-Earth)** or **TRAPPIST-1e (Habitable Zone)**.
3. Click **"Run ExoHunter Detection"**.
4. Watch the metrics light up, the AI verdict appear, and the 4-panel diagnostic graph render!
5. Click **"Download High-Res Discovery Report"** to get the publication-ready PNG.

### Option 2: The Command Line Interface (CLI)
Run any benchmark target in 1 line:
```bash
# Analyze Kepler-10b
python cli.py --benchmark kepler-10b

# Analyze TRAPPIST-1e (Habitable Zone planet)
python cli.py --benchmark trappist-1e

# Catch an Eclipsing Binary false alarm
python cli.py --benchmark binary-false-positive
```

### Option 3: Run the Automated Pytest Suite
```bash
python -m pytest tests/test_pipeline.py
```
All 5 automated tests will pass with green checkmarks.

---

## 4. Cheat Sheet: How to Answer Interview Questions

**Q: "Why did you build your own BLS algorithm instead of just using a library?"**
> *"While libraries like Astropy have BLS implementations, they often encounter wheel and environment deprecation issues with newer NumPy versions. By building a vectorized Box Least Squares algorithm directly from the Kovács et al. formulation using cumulative window sums in pure NumPy, I ensured zero dependency friction and gained an intimate understanding of least-squares phase folding."*

**Q: "How does the machine learning model differentiate planets from noise or binary stars?"**
> *"The classifier evaluates several key physical features: odd-to-even transit depth symmetry, transit duration ratio, transit sharpness, and signal-to-noise ratio. Eclipsing binary stars exhibit an alternating odd-even depth discrepancy because the two stars have different surface temperatures, whereas an exoplanet maintains symmetric depth across all transits."*

**Q: "What are the limitations of transit photometry?"**
> *"Transit photometry requires geometric alignment—we can only detect planets whose orbital plane happens to intersect our line of sight from Earth. It is also biased toward detecting planets with short orbital periods and larger radii."*
