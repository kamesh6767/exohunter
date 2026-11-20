# 🚀 LinkedIn Post Strategy & Ready-to-Copy Template

Sharing your project on LinkedIn with the right hook, storytelling, and visuals is what drives recruiter messages, connection requests from fellow engineers, and high post engagement.

---

## 📸 Media Checklist (What to Attach to Your Post)

Before posting, prepare **one** of the following visual assets:
- **Option A (Highest Engagement - Video/GIF):** A 15-to-30 second screen recording of your Streamlit app in action!
  - How to record: Run `streamlit run app.py`, use Windows Game Bar (`Win + Alt + R`) or Loom/OBS to record your screen while selecting `TRAPPIST-1e` or `Kepler-10b` and clicking **"Run ExoHunter Detection"**.
- **Option B (Image):** Attach the 4-panel diagnostic plot generated in `assets/kepler-10b_diagnostic.png`.

---

## 📝 The Ready-to-Copy LinkedIn Post

Copy and paste the text below into LinkedIn:

```text
How do you detect a planet orbiting a star 1,000 light-years away when you can't even see the planet directly? 🔭🪐

Imagine standing two miles away from a streetlight and watching a tiny fly crawl across the bulb. You can’t resolve the fly, but if your camera is sensitive enough, you will measure the streetlight dim by 0.01% every 15 seconds.

That is the principle behind Transit Photometry—the method NASA’s Kepler and TESS space telescopes use to discover exoplanets.

Over the past few weeks, I built and published 𝗘𝘅𝗼𝗛𝘂𝗻𝘁𝗲𝗿: an end-to-end astrophysical signal processing pipeline and Machine Learning system for automated exoplanet detection and vetting! 🚀

Here is how the system works:

1️⃣ 𝗗𝗮𝘁𝗮 𝗜𝗻𝗴𝗲𝘀𝘁𝗶𝗼𝗻 & 𝗣𝗿𝗲𝗽𝗿𝗼𝗰𝗲𝘀𝘀𝗶𝗻𝗴: Stars aren't steady lightbulbs—they bubble, rotate, and flare. I built a multi-stage filtering pipeline using robust sigma-clipping for cosmic rays and Savitzky-Golay filtering to detrend stellar activity without washing out narrow transit dips.

2️⃣ 𝗧𝗵𝗲 𝗥𝗵𝘆𝘁𝗵𝗺 𝗛𝘂𝗻𝘁𝗲𝗿 (𝗕𝗟𝗦 𝗔𝗹𝗴𝗼𝗿𝗶𝘁𝗵𝗺): Individual transit events last only a couple of hours every few weeks. I implemented a vectorized Box Least Squares (BLS) periodogram search in pure NumPy (based on Kovács et al.) that scans thousands of trial orbital periods and phase-folds observations to reveal the periodic heartbeat of the orbit.

3️⃣ 𝗔𝘀𝘁𝗿𝗼𝗽𝗵𝘆𝘀𝗶𝗰𝗮𝗹 𝗣𝗵𝘆𝘀𝗶𝗰𝘀 𝗠𝗼𝗱𝗲𝗹𝗶𝗻𝗴: Once a transit is detected, the pipeline applies Kepler’s Third Law to derive the semi-major axis (AU), extracts the planet's physical radius from transit depth (δ ≈ (Rp/R*)²), and calculates planetary equilibrium temperature (Teq) to check if it sits in the star's Habitable Zone!

4️⃣ 𝗠𝗮𝗰𝗵𝗶𝗻𝗲 𝗟𝗲𝗮𝗿𝗻𝗶𝗻𝗴 𝗙𝗮𝗹𝘀𝗲-𝗣𝗼𝘀𝗶𝘁𝗶𝘃𝗲 𝗩𝗲𝘁𝘁𝗶𝗻𝗴: Over 50% of transit candidates are false alarms—most commonly Eclipsing Binaries (two stars orbiting each other). I trained a Random Forest classifier calibrated on the NASA Kepler KOI catalog that analyzes odd/even transit depth symmetry, duration ratios, and SNR to automatically classify real planets vs. binary stars.

5️⃣ 𝗜𝗻𝘁𝗲𝗿𝗮𝗰𝘁𝗶𝘃𝗲 𝗪𝗲𝗯 𝗗𝗮𝘀𝗵𝗯𝗼𝗮𝗿𝗱: I wrapped the whole pipeline into an interactive Streamlit application with real-time parameter tuning and one-click publication-ready 4-panel diagnostic reports.

Tested on benchmark space telescope systems:
✅ Discovered Kepler-10b (Super-Earth, 0.84-day period, 1.21 R⊕)
✅ Discovered TRAPPIST-1e (Earth-sized Habitable Zone candidate, 228 K)
⚠️ Correctly flagged and rejected grazing Eclipsing Binary false alarms!

Check out the code and run it yourself on GitHub:
👉 [Insert your GitHub repo link here]

Tech stack: Python | NumPy | SciPy | Scikit-Learn | Streamlit | Matplotlib | Pytest

I’d love to hear your thoughts or feedback! What space or machine learning problems are you working on?

#MachineLearning #DataScience #Python #Astrophysics #Astronomy #ArtificialIntelligence #SoftwareEngineering #Streamlit #OpenSource #PortfolioProject
```

---

## 💡 Pro-Tips for Maximum Reach on LinkedIn

1. **Tag relevant topics:** Put the GitHub link in the post body (or in the first comment if you prefer).
2. **Engage with commenters:** When people comment on your post (e.g., "Great work!"), reply with an insightful answer like: *"Thanks! The trickiest part was tuning the odd-even transit depth discrepancy filter to avoid misclassifying grazing binary stars."*
3. **Add to your LinkedIn Featured Section:** Once posted, click the three dots (`...`) on your post and select **"Feature on top of profile"**. This makes your project visible to every recruiter who visits your profile!
