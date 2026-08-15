## Question 2 - Biosignal/Embedded: low-compute artefact rejection for battery-limited wearables

**The real problem:** Signal-quality assessment is usually built on models heavy enough to noticeably shorten battery life on a wearable, yet acquiring an artefact-corrupted signal is worse than acquiring none, since it silently corrupts every biomarker computed downstream. Reducing that compute cost is one of the more direct paths to a clinical-grade device people will actually wear all day.

**Your task:** Describe - and implement a concept in Python - an unsupervised, feature-based method for rejecting artefact-corrupted segments of a PPG or ECG signal, cheap enough to run continuously on a low-power microcontroller without a trained ML classifier in the loop. State which signal properties your method exploits and why, and estimate its compute cost (operations, memory) against a typical lightweight ML alternative.

**We are specifically looking for:** a method you can justify from first principles about the signal rather than a black box; realistic reasoning about what "cheap enough for a battery powered wearable" means in practice; and honesty about the artefacts your approach would miss.

------------
### ANALYSIS ROADMAP
1. Dataset Selection
2. Method-1: Statistical Approach
3. Algorithm Insights for Method-1
4. Method-2: Adaptive Filtering Based Approach Using Motion Data
5. Algorithm Insights for Method-2
6. Summary

-------------
### 1. Dataset Selection

The implementation uses the **Pulse Transit Time (PTT) PPG Dataset** from PhysioNet. The dataset comprises 66 recordings collected from 22 participants under 3 physical activity conditions:

1. **Sitting (`sit`):** Serves as the clean baseline with stable periodic waveforms and minimal baseline drift.
2. **Walking (`walk`):** Exhibits moderate motion artefacts and slight baseline variation.
3. **Running (`run`):** Contains severe motion artefacts, frequency band corruption, and pronounced baseline wander.

![Raw PPG signals across all conditions](r"Figures/raw_signal.png")

**Dataset Specifications & Key Features**

| Feature Category | Technical Specification |
| :--- | :--- |
| **Optical Sites** | 2 spatially separated optical sites on the finger |
| **Wavelengths** | Red, Infrared, and Green wavelengths |
| **Acquisition Frequency** | $500\text{ Hz}$ sampling clock |
| **Recorded Conditions** | Recorded during Sitting, walking and Running (corrupted) |

> **Reference:**
> Mehrgardt, P., et al. "Pulse transit time PPG dataset." *PhysioNet* (2022): e215-e220. DOI: [[1] https://doi.org/10.13026/jpan-6n92](https://doi.org/10.13026/jpan-6n92).

-----
### 2. Method-1: Statistical Approach

> [!NOTE]
> **This approach is only for getting the Ground Truth Baseline and understanding the signal properties.**
> While Higher-Order Statistical (HOS) metrics (e.g., Skewness, Kurtosis) are computationally expensive for continuous execution on low-power MCUs, this approach was implemented to establish an analytical ground truth baseline and systematically evaluate signal distribution properties during motion.

**ALGORITHM:**

#### 1. Preprocessing & Bandpass Filtering
* **Channel Selection & Acquisition:** The raw PPG signal is extracted from the `pleth_1` channel (MAX30101 red wavelength sensor positioned at the distal phalanx of the left index finger, sampled at $500\text{ Hz}$).
* **Polarity Inversion:** The raw signal is inverted ($\times -1$) to convert it into a Blood-Volume-Pulse-oriented waveform (systolic peaks pointing up) before filtering.
* **Filter Configuration:** A 2nd-order Butterworth Bandpass Filter (BPF) with a passband of **$0.5\text{ Hz}$ to $8.0\text{ Hz}$** is applied using zero-phase filtering (`filtfilt`) to remove low-frequency baseline drift (DC offset) and high-frequency high-order noise while keeping the signal shape intact.

##### Physiological & Spectral Justification

| Feature | Passband Range | Physiological Rationale |
| :--- | :--- | :--- |
| **Fundamental Pulse Rate** | $0.67\text{ Hz} - 3.67\text{ Hz}$ | Encompasses normal physiological resting and active heart rates ($40 - 220\text{ BPM}$). |
| **Waveform Morphology** | $3.0\text{ Hz} - 8.0\text{ Hz}$ | Preserves critical high-frequency pulse features, including the systolic peak, dicrotic notch, and diastolic peak. |

> **Note on Bandwidth Selection:** While the core fundamental heart rate lies within $0.5 - 6.0\text{ Hz}$, extending the upper cutoff to $8.0\text{ Hz}$ prevents morphological distortion of the dicrotic wave structure essential for higher-order statistical feature extraction.

![Preprocessing]("Figures/preprocessing.png")

#### 2. Signal Segmentation & Heart Period Estimation

The continuous PPG signal is segmented into single cardiac cycles using dynamic autocorrelation to enable beat-by-beat quality evaluation.

* **Why Dynamic Autocorrelation over Fixed Windows:** Fixed $2\text{s}$ windows (initially planned) are low-cost for MCUs but group multiple cycles together during elevated heart rates. Autocorrelation adapts the period estimate to the recording's actual heart rate rather than assuming a fixed cycle length.

* **Implementation Steps:**
  1. **Standardization:** Normalize amplitude using $Z$-score ($Z = \frac{x - \mu}{\sigma}$).
  2. **Autocorrelation:** Compute self-similarity restricted to physiological lag bounds ($0.285 - 2.0\text{ s}$, or $30 - 210\text{ BPM}$), and select the **strongest** (dominant) peak within that window as the period estimate — not merely the first peak encountered, since a smaller ripple (e.g. a harmonic) can appear earlier in the lag window.
  3. **Beat Slicing:** Use the dominant peak to derive pulse period $T$ and cut the signal into single-cycle segments of length $N_{\text{period}} = \lfloor T \times f_s \rfloor$.

> **Known simplification:** $T$ is estimated once per analysis window and used to slice the whole window into equal-length segments, rather than re-estimated per beat. HR drift within the window will progressively misalign beats inside their fixed-length slices — a limitation carried into the statistical evaluation below, not a per-beat dynamic segmentation.

![segmenting the signals]("Figures/segmentation.png")

#### 3. Statistical Quality Evaluation & Adaptive Thresholding

* **Reference Initialization:** Baseline statistical parameters ($\mu_{\text{std}}$, $\mu_{\text{skew}}$, $\mu_{\text{kurt}}$) are initialized from the first beat segment.
* **Feature Extraction:** For each beat segment $i$, standard deviation ($\text{std}_i$), skewness ($\text{skew}_i$), and Pearson kurtosis ($\text{kurt}_i$) are computed.
* **Tolerance Check:** Segment deviations are evaluated against static thresholds:

  $$\Delta_{\text{std}} = |\text{std}_i - \mu_{\text{std}}| \le \delta_{\text{std}}$$
  $$\Delta_{\text{skew}} = |\text{skew}_i - \mu_{\text{skew}}| \le \delta_{\text{skew}}$$
  $$\Delta_{\text{kurt}} = |\text{kurt}_i - \mu_{\text{kurt}}| \le \delta_{\text{kurt}}$$

* **Classification & Baseline Update:**
  * **Clean Beat:** If all three bounds are satisfied, segment $i$ is flagged as **CLEAN**. Baselines update using an Exponential Moving Average (EMA) with $\alpha = 0.8$:
    $$\mu_{\text{ref}} \leftarrow \alpha \cdot \mu_{\text{ref}} + (1 - \alpha) \cdot \text{metric}_i$$
  * **Motion Artifact:** If any condition fails, segment $i$ is flagged as a **MOTION ARTIFACT**, and existing baselines are retained.

![statistical evaluation the signals]("Figures/fig3_skew_kurt_final_detection.png")

### 3. Algorithm Insights - Method-1

#### First-Principles Signal Exploitation
* **Periodicity:** Autocorrelation isolates true cardiac cycles within physiological bounds ($30 - 210\text{ BPM}$), preventing arbitrary windowing errors.
* **Distributional Geometry:** Clean PPG signals exhibit distinct asymmetry (systolic rise vs. diastolic fall) and peak sharpness. Higher-Order Statistics ($\text{std}$, $\text{skew}$, $\text{kurt}$) track deviations in these geometric traits caused by movement.

#### Microcontroller (MCU) Efficiency Analysis
* **Computational Footprint:** ~10–20 FLOPs per beat segment for simple 1D statistical moments, vs. ~1,000+ FLOPs for lightweight ML classifiers (e.g. Random Forests, TinyML nets).
* **Memory Constraints:** <1 KB SRAM to retain reference scalar baselines — no feature matrices or model weights to buffer.

#### Known Failure Modes & Limitations
* **Corrupted Initial Beat:** If the first segment used for baseline initialization is corrupted, the adaptive reference thresholds will distort until a reset.
* **In-Band Motion:** Low-amplitude motion at cardiac frequencies ($0.5-3.0\text{ Hz}$) can alter pulse shape without exceeding statistical bounds, letting subtle artefacts pass through.
* **Fixed-window segmentation:** as noted above, period drift within a window can itself masquerade as a quality drop, independent of any true artefact.

-------
### 4. Method-2: Adaptive Filtering Based Approach Using Motion Data

**ALGORITHM (short form):**

1. **Load** synchronized PPG (`pleth_1`) + 3-axis accelerometer + 3-axis gyroscope.
2. **Calibrate motion threshold once**, on a separate calm recording (`s1_sit`): threshold = mean + 4·std of smoothed IMU energy.
3. **Detect motion** on the target recording by comparing its smoothed IMU energy to that fixed threshold.
4. **Pick the best IMU reference axis** — whichever of the 6 axes correlates most strongly with the PPG during motion-flagged samples.
5. **NLMS adaptive filter:** during motion only, predict the motion-correlated component of the band-passed PPG from the chosen IMU axis and subtract it; non-motion samples pass through unfiltered.
6. **Re-score beats** on the cleaned signal with the same interval/amplitude/correlation check as Method-1, and compare rejection rate before vs. after filtering.

### Algorithm Insights - Method-2

* **Signal principle exploited:** motion artefacts are assumed to be an *additive, IMU-correlated* component in the PPG — sensor/skin displacement modulates the optical signal roughly linearly with motion. NLMS subtracts that predictable component out, with no labels needed.
* **MCU cost:** ~24 FLOPs/sample (12-tap NLMS) plus one division, only while motion is flagged — cheap relative to a continuously-running ML model, since duty cycle is low for most wear time.
* **Known failure mode:** the motion threshold must be calibrated against a genuinely calm period — calibrating it from the same corrupted recording under test anchors the threshold too high and motion never fires. A user who is rarely still (or a sensor fitted mid-motion) has no valid calibration window, which is a real deployment limitation, not just a coding detail.
* **Other limitation:** the linear-coupling assumption breaks down for nonlinear artefact sources (contact pressure changes, blood pooling), which NLMS cannot remove.

----------------
## Summary

Method 1 serves as an initial approach to analyze the signal and establish a baseline. In Method 2, I implemented motion artifact removal using accelerometer data alongside adaptive filtering. This provides a reliable way to attenuate motion-based noise, which often shares the same frequency spectrum as the PPG signal of interest. While this approach provides an effective first-level noise reduction, I was unable to fully quantify the performance improvements within the available timeframe. 

Moving forward, I plan to optimize the filter parameters and explore alternative filtering algorithms to improve output quality. Given the tight timeline before submission, I dedicated significant effort to working with the dataset and researching PPG signal characteristics, hardware signal modulation, and sensor behavior. This foundational analysis led directly to the IMU-gated adaptive filtering approach after observing how motion artifacts degraded signal quality.

Disclosure: AI assistance (LLMs) was used to refine algorithm logic and clean up code implementation.

