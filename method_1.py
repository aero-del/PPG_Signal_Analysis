"""
PPG Artifact Detection - Modular Multi-Figure Visualization
Figure 1: Preprocessing Pipeline (Raw -> Inverted -> Bandpass Filtered)
Figure 2: Heart Period Estimation & Beat Segmentation
Figure 3: Statistical Quality Evaluation & Artifact Detection

Loads a single PPG channel from a WFDB record, preprocesses it,
estimates the dominant beat period via autocorrelation, segments the
signal into single-cycle beats, injects a synthetic motion artifact
into one beat for validation, and flags corrupted beats using a
statistical (std/skew/kurtosis) quality gate with an adaptive
reference. Three matplotlib figures are produced, one per stage.
"""

import wfdb
import numpy as np
import scipy.signal as signal
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

# ==========================================
# 1. LOADING & PREPROCESSING PPG SIGNAL
# ==========================================
PATH_SIT = r"Data/s1_sit"
sig, fields = wfdb.rdsamp(PATH_SIT)   # sig: (n_samples, n_channels) array; fields: header metadata dict
fs = fields['fs']
ch_idx = fields['sig_name'].index('pleth_1')

WINDOW_DURATION = 10.0  # seconds
N_samples = int(WINDOW_DURATION * fs)
raw_ppg = sig[:N_samples, ch_idx]
time_axis = np.linspace(0, WINDOW_DURATION, N_samples)

# Photoplethysmography convention: transmission/reflectance PPG from
# this sensor increases with tissue absorption, so a rising blood
# volume shows as a falling raw signal. Inverting here reorients it to
# the standard BVP convention (upward deflection = systolic peak),
# which the downstream peak-picking and beat-shape logic assumes.
bvp_ppg = -1.0 * raw_ppg

# Bandpass Filtered Signal (0.5 Hz - 8.0 Hz)
# 2nd-order Butterworth, zero-phase via filtfilt (effective 4th-order
# magnitude response, no group delay - fine offline, not causal).
# 0.5-8 Hz keeps the fundamental cardiac band and its low harmonics
# while removing baseline wander and high-frequency sensor noise.
b, a = signal.butter(2, [0.5, 8.0], btype='bandpass', fs=fs)
clean_ppg = signal.filtfilt(b, a, bvp_ppg)

# ==========================================
# FIGURE 1: PREPROCESSING STAGES
# ==========================================
fig1, axes1 = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

axes1[0].plot(time_axis, raw_ppg, color='black', lw=1.2)
axes1[0].set_title("1. Raw Original PPG Signal", fontweight='bold')
axes1[0].set_ylabel("Raw Volts")
axes1[0].grid(True, alpha=0.3)

axes1[1].plot(time_axis, bvp_ppg, color='darkorange', lw=1.2)
axes1[1].set_title("2. Polarity Inverted Signal (Blood Volume Pulse)", fontweight='bold')
axes1[1].set_ylabel("Amplitude")
axes1[1].grid(True, alpha=0.3)

axes1[2].plot(time_axis, clean_ppg, color='forestgreen', lw=1.4)
axes1[2].set_title("3. Bandpass Filtered PPG Signal (0.5 - 8.0 Hz)", fontweight='bold')
axes1[2].set_xlabel("Time (seconds)")
axes1[2].set_ylabel("Amplitude")
axes1[2].grid(True, alpha=0.3)

fig1.tight_layout()
fig1.savefig("fig1_preprocessing.png", dpi=150)
fig1.show()

# ==========================================
# 2. AUTOCORRELATION & SEGMENTATION
# ==========================================
# Z-score normalization before autocorrelation so the zero-lag value is
# exactly the signal's own variance, which lets autocorr be normalized
# to a max of 1.0 by dividing by autocorr[0] below.
norm_ppg = (clean_ppg - np.mean(clean_ppg)) / np.std(clean_ppg)

# Full autocorrelation, then keep only non-negative lags (index
# len(norm_ppg)-1 onward corresponds to lag=0). This is O(N^2) via
# direct correlation (scipy picks FFT method automatically for large N,
# but the result is the same either way) - one autocorrelation over the
# whole window, not a per-beat computation.
autocorr = signal.correlate(norm_ppg, norm_ppg, mode='full')[len(norm_ppg) - 1:]
autocorr /= autocorr[0]   # normalize so lag-0 autocorrelation is exactly 1.0

lags_in_sec = np.arange(len(autocorr)) / fs
# Search window bounds the expected beat period to 0.5-3.5 Hz
# (fs/3.5 samples to fs/0.5 samples of lag), i.e. 40-120 BPM - narrower
# than the general physiological range, appropriate for a resting/
# sitting recording where extremes are unlikely.
min_lag, max_lag = int(fs / 3.5), int(fs / 0.5)
# distance=min_lag on find_peaks here isn't really doing beat-to-beat
# spacing (this operates on the autocorrelation curve, not the PPG
# itself) - its practical effect is just suppressing very-low-lag
# peaks close to the searched region's start.
peaks, _ = signal.find_peaks(autocorr[min_lag:max_lag], distance=min_lag)

# Takes the first peak found within the search window as the dominant
# period - this assumes the strongest non-trivial autocorrelation peak
# in-range corresponds to the true heart period, not a harmonic or
# subharmonic. No fallback if peaks is empty (would raise IndexError).
finalized_T = lags_in_sec[peaks[0] + min_lag]
period_samples = int(finalized_T * fs)

# Fixed-period segmentation: every beat is forced to exactly
# period_samples long, using ONE globally estimated period rather than
# per-beat peak detection. This only holds up if heart rate is
# effectively constant over the window - any HR drift will
# progressively misalign later segments relative to actual beat
# boundaries.
num_segments = len(clean_ppg) // period_samples
segments = [clean_ppg[i * period_samples:(i + 1) * period_samples].copy() for i in range(num_segments)]

# Synthetic damped 12 Hz burst added to the third segment, purely to
# create a known-bad beat to validate that the statistical detector
# below actually flags it. Amplitude (2.5) is on the same order as a
# typical filtered-PPG cycle's own amplitude, and the exponential decay
# (exp(-3t)) mimics a transient motion burst rather than sustained
# noise.
if len(segments) >= 3:
    t_seg = np.linspace(0, finalized_T, period_samples)
    motion_noise = 2.5 * np.sin(2 * np.pi * 12 * t_seg) * np.exp(-t_seg * 3)
    segments[2] = segments[2] + motion_noise
    print("--> Injected synthetic Motion Artifact into Beat 3 for testing.\n")

# Reassembled signal for plotting/inspection - note this is shorter
# than clean_ppg by up to (period_samples - 1) samples due to the
# integer-division truncation in num_segments (any leftover tail
# samples that don't fill a full period are dropped).
noisy_input_ppg = np.concatenate(segments)
input_time_axis = time_axis[:len(noisy_input_ppg)]

# ==========================================
# FIGURE 2: PERIOD ESTIMATION & SEGMENTATION
# ==========================================
fig2, axes2 = plt.subplots(3, 1, figsize=(11, 8))

# Subplot A: Autocorrelation Window
axes2[0].plot(lags_in_sec, autocorr, color='crimson', lw=1.5, label='Autocorrelation')
axes2[0].set_xlim(0, 2.5)
axes2[0].axvspan(lags_in_sec[min_lag], lags_in_sec[max_lag], color='yellow', alpha=0.2, label='Search Window')
axes2[0].plot(finalized_T, autocorr[peaks[0] + min_lag], "o", color='black', markersize=8)
axes2[0].set_title(f"1. Autocorrelation — Finalized Period T = {finalized_T:.3f}s", fontweight='bold')
axes2[0].set_xlabel("Lag Time (seconds)")
axes2[0].set_ylabel("Score")
axes2[0].legend(loc='upper right')
axes2[0].grid(True, alpha=0.3)

# Subplot B: Continuous Signal Sliced by Boundaries
axes2[1].plot(input_time_axis, noisy_input_ppg, color='forestgreen', lw=1.2)
for i in range(num_segments + 1):
    axes2[1].axvline(x=i * finalized_T, color='blue', linestyle='--', lw=1.5)
axes2[1].set_title("2. Continuous Signal Cut by Finalized Period T Boundaries", fontweight='bold')
axes2[1].set_xlabel("Time (seconds)")
axes2[1].set_ylabel("Amplitude")
axes2[1].set_xlim(0, input_time_axis[-1])
axes2[1].grid(True, alpha=0.3)

# Subplot C: Extracted Beats Overlaid
# Each beat plotted against a common cycle-time axis (0 to finalized_T)
# rather than absolute time, so beat-shape consistency/drift is visible
# directly by eye - this is the same alignment convention the
# statistical detector below relies on implicitly (each segment treated
# as a directly comparable fixed-length vector).
segment_time = np.linspace(0, finalized_T, period_samples)
colors = plt.cm.jet(np.linspace(0, 1, num_segments))
for i, seg in enumerate(segments):
    axes2[2].plot(segment_time, seg, color=colors[i], label=f'Beat {i+1}', lw=1.5)
axes2[2].set_title("3. Extracted Single-Cycle Beats Overlaid", fontweight='bold')
axes2[2].set_xlabel("Cycle Time (seconds)")
axes2[2].set_ylabel("Amplitude")
axes2[2].legend(loc='upper right', ncol=min(num_segments, 5))
axes2[2].grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig("fig2_segmentation.png", dpi=150)
fig2.show()

# ==========================================
# 3. STATISTICAL EVALUATION (STD, SKEW, KURTOSIS)
# ==========================================
# Reference statistics bootstrap from beat 1 with no quality check on
# that beat - if beat 1 itself were corrupted, every threshold below
# would be built on a bad baseline. Acceptable here since beat 3 is the
# only beat deliberately corrupted.
first_seg = segments[0]
th_std = np.std(first_seg)
th_skew = skew(first_seg)                    # Fisher-Pearson skewness (scipy default, bias-corrected off by default)
th_kurt = kurtosis(first_seg, fisher=False)   # fisher=False -> Pearson convention, normal distribution => 3.0, not 0.0

# Fixed absolute tolerance bands, not relative to each beat's own
# amplitude - delta_std scales with the reference std (half of it), but
# delta_skew/delta_kurt are flat constants independent of signal scale,
# so their effective strictness will differ across recordings with
# different beat-shape variability.
delta_std = 0.5 * th_std
delta_skew = 0.8
delta_kurt = 1.8

alpha = 0.8  # EMA update rate

segment_quality = []
std_diffs, skew_diffs, kurt_diffs = [], [], []

for i, seg in enumerate(segments):
    curr_std = np.std(seg)
    curr_skew = skew(seg)
    curr_kurt = kurtosis(seg, fisher=False)

    diff_std = abs(curr_std - th_std)
    diff_skew = abs(curr_skew - th_skew)
    diff_kurt = abs(curr_kurt - th_kurt)

    std_diffs.append(diff_std)
    skew_diffs.append(diff_skew)
    kurt_diffs.append(diff_kurt)

    # All three moments must independently fall within tolerance of the
    # current adaptive reference - a single blown metric (e.g. the
    # injected 12 Hz burst spiking kurtosis) is enough to reject a beat,
    # even if the other two happen to still look normal.
    is_high_quality = (diff_std <= delta_std) and (diff_skew <= delta_skew) and (diff_kurt <= delta_kurt)
    segment_quality.append(is_high_quality)

    if is_high_quality:
        # EMA update only on accepted beats, so the reference stays
        # anchored during a rejected stretch rather than drifting toward
        # the artifact's statistics.
        th_std = alpha * th_std + (1 - alpha) * curr_std
        th_skew = alpha * th_skew + (1 - alpha) * curr_skew
        th_kurt = alpha * th_kurt + (1 - alpha) * curr_kurt

# ==========================================
# FIGURE 3: STATISTICAL METRICS & FINAL ARTIFACT OUTPUT
# ==========================================
fig3, axes3 = plt.subplots(4, 1, figsize=(11, 10), sharex=False)

beats_axis = np.arange(1, num_segments + 1)

# Panel 1: Standard Deviation Errors
axes3[0].bar(beats_axis, std_diffs, color='mediumpurple', alpha=0.8)
axes3[0].axhline(delta_std, color='red', linestyle='--', label=f'Threshold Bound (delta = {delta_std:.3f})')
axes3[0].set_title("1. Standard Deviation Difference per Beat", fontweight='bold')
axes3[0].set_ylabel("|Std Diff|")
axes3[0].set_xticks(beats_axis)
axes3[0].legend(loc='upper right')
axes3[0].grid(True, alpha=0.3)

# Panel 2: Skewness Errors
axes3[1].bar(beats_axis, skew_diffs, color='teal', alpha=0.8)
axes3[1].axhline(delta_skew, color='red', linestyle='--', label=f'Threshold Bound (delta = {delta_skew:.2f})')
axes3[1].set_title("2. Skewness Difference per Beat", fontweight='bold')
axes3[1].set_ylabel("|Skew Diff|")
axes3[1].set_xticks(beats_axis)
axes3[1].legend(loc='upper right')
axes3[1].grid(True, alpha=0.3)

# Panel 3: Kurtosis Errors
# Note the threshold line drawn here is the FIXED delta_kurt constant,
# not the (possibly since-updated) adaptive th_kurt - these bars show
# each beat's deviation from whatever th_kurt was AT THE TIME it was
# evaluated (i.e. after any prior accepted beats' updates), so bar
# heights are not all measured against the same reference value.
axes3[2].bar(beats_axis, kurt_diffs, color='coral', alpha=0.8)
axes3[2].axhline(delta_kurt, color='red', linestyle='--', label=f'Threshold Bound (delta = {delta_kurt:.2f})')
axes3[2].set_title("3. Kurtosis Difference per Beat", fontweight='bold')
axes3[2].set_ylabel("|Kurt Diff|")
axes3[2].set_xticks(beats_axis)
axes3[2].legend(loc='upper right')
axes3[2].grid(True, alpha=0.3)

# Panel 4: Final Output Signal with Red Shaded Motion Artifacts
# Artifact spans are drawn from the fixed period_samples grid
# (i * finalized_T to (i+1) * finalized_T), so shading exactly matches
# the segmentation boundaries used for classification, not any
# independently-detected beat onset/offset.
axes3[3].plot(input_time_axis, noisy_input_ppg, color='black', lw=1.2, label='PPG Signal')
labeled_once = False
for i, is_clean in enumerate(segment_quality):
    start_t, end_t = i * finalized_T, (i + 1) * finalized_T
    if not is_clean:
        label = "Motion Artifact" if not labeled_once else None
        axes3[3].axvspan(start_t, end_t, color='red', alpha=0.35, label=label)
        labeled_once = True

axes3[3].set_title("4. Final Output Signal — Red Highlighting on Corrupted Artifact Segments", fontweight='bold')
axes3[3].set_xlabel("Time (seconds)")
axes3[3].set_ylabel("Amplitude")
axes3[3].set_xlim(0, input_time_axis[-1])
if labeled_once:
    axes3[3].legend(loc='upper right')
axes3[3].grid(True, alpha=0.3)

fig3.tight_layout()
fig3.savefig("fig3_skew_kurt_final_detection.png", dpi=150)
fig3.show()
