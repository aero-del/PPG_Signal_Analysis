"""
IMU-gated adaptive artefact reduction for PPG.

Loads a PPG channel plus 3-axis accelerometer and 3-axis gyroscope data
from a WFDB record, classifies motion vs stillness from the IMU energy,
and runs a per-sample NLMS adaptive filter (using the best-correlated
IMU axis as reference) only during motion-flagged segments to cancel
motion-coupled artefact from the PPG. Beat-level quality (interval,
amplitude, waveform shape) is then checked on the signal both before
and after adaptive filtering, so the two can be compared directly.

Dependencies: wfdb, numpy, scipy, matplotlib.
"""

import wfdb
import numpy as np
import scipy.signal as sp_signal
import matplotlib.pyplot as plt

# ==========================================
# CONFIG
# ==========================================

# WFDB record path, no extension (wfdb.rdrecord resolves .hea/.dat itself).
RECORD_PATH = r"D:\ntu_asses\physionet.org\files\pulse-transit-time-ppg\1.1.0\s1_walk"

PPG_CHANNEL = "pleth_1"          # which pleth_x column to treat as the PPG
WINDOW_DURATION = 20.0            # seconds of the record to analyse, None = whole record

# HR bounds double as the peak-detection min-distance constraint (via HR_MAX_BPM)
# and the beat-acceptance interval gate (via HR_MIN/MAX_BPM) in run_iat_detector.
HR_MIN_BPM, HR_MAX_BPM = 40, 180    # physiological boundaries of heart rate in BPM (0.67-3.0 Hz)
ALPHA = 0.8       # EMA coefficient for the adaptive template/threshold update (0.8 ~ 5-beat time constant)
TEMPLATE_LEN = 40  # fixed resample length for beat-shape xcorr comparison; independent of fs/HR

MOTION_WIN_SEC = 0.5   # energy-smoothing window for motion classifier (seconds, converted to samples internally)
LMS_TAPS = 12           # NLMS filter order (memory length in samples, not seconds - scales with fs)
NLMS_MU = 0.3            # NLMS step size, unitless/self-normalizing; 0.1-0.5 is a typical stable range


# ==========================================
# STAGE 0: LOAD DATA FROM RECORD
# ==========================================
def load_record(record_path, ppg_channel, duration_s=None):
    """
    Loads a WFDB record and pulls out the PPG channel plus the 6 IMU
    channels, trimmed to duration_s seconds. Assumes record.p_signal is
    already in physical units (wfdb applies gain/baseline from the
    header automatically).

    fs is shared across all channels - wfdb.rdrecord requires this for
    a single-segment record, so no per-channel resampling is needed
    here.
    """
    record = wfdb.rdrecord(record_path)
    fs = record.fs
    sig_names = record.sig_name
    sig = record.p_signal   # shape (n_samples, n_channels), float64, physical units

    required = [ppg_channel, "a_x", "a_y", "a_z", "g_x", "g_y", "g_z"]
    missing = [c for c in required if c not in sig_names]
    if missing:
        # Fail fast with the actual channel list rather than letting a
        # downstream KeyError/IndexError obscure which channel was wrong.
        raise ValueError(f"Record is missing expected channels: {missing}. "
                          f"Available channels: {sig_names}")

    # Hard slice to n_samples - no interpolation/resampling, so
    # duration_s effectively floors to the nearest sample at this fs.
    n_samples = sig.shape[0] if duration_s is None else min(sig.shape[0], int(duration_s * fs))
    sig = sig[:n_samples]
    t = np.arange(n_samples) / fs

    def ch(name):
        return sig[:, sig_names.index(name)]

    raw_ppg = ch(ppg_channel)
    accel = np.column_stack([ch("a_x"), ch("a_y"), ch("a_z")])   # Nx3, units per header (typically g)
    gyro = np.column_stack([ch("g_x"), ch("g_y"), ch("g_z")])    # Nx3, units per header (typically deg/s)

    # Sanity-check prints - confirm fs/channel order match expectations
    # before sinking time into debugging downstream NaNs.
    print(f"Loaded {record_path}")
    print(f"  fs = {fs} Hz, duration analysed = {n_samples / fs:.1f} s, channels = {sig_names}")
    print("t:",len(t))
    print("raw_ppglen:",len(raw_ppg))
    print(len(accel))
    print(len(gyro))
    return t, raw_ppg, accel, gyro, fs


# ==========================================
# STAGE 0b: MOTION CLASSIFIER
# ==========================================
def detect_motion(accel, gyro, fs, win_sec=MOTION_WIN_SEC):
    """
    Energy-based motion gate: sum-of-squares per axis-group (no sqrt,
    since only threshold comparison is needed, not magnitude), smoothed
    with a boxcar of length win_sec*fs via 'same'-mode convolution
    (introduces win/2-sample edge effects at the array boundaries -
    negligible for a 20s window, worth knowing for very short clips).

    accel_thresh / gyro_thresh are derived from this signal's own
    statistics so the classifier self-calibrates on whatever record is
    loaded. On a fixed hardware deployment these would instead be
    constants captured once during a calm baseline period at setup.
    """
    win = max(1, int(win_sec * fs))
    kernel = np.ones(win) / win   # boxcar/moving-average FIR, centered by 'same' mode

    accel_energy_inst = np.sum(accel ** 2, axis=1)   # ||a||^2 per sample, not RMS - skips the sqrt
    gyro_energy_inst = np.sum(gyro ** 2, axis=1)
    accel_energy = np.convolve(accel_energy_inst, kernel, mode='same')
    gyro_energy = np.convolve(gyro_energy_inst, kernel, mode='same')

    # mean + 4*std threshold assumes calib_len is a representative calm
    # baseline. 4 sigma is a deliberately conservative (high-specificity)
    # choice, since running NLMS unnecessarily on a clean segment can
    # itself inject distortion.
    calib_len = min(len(accel_energy), win * 4)
    accel_thresh = np.mean(accel_energy[:calib_len]) + 4 * np.std(accel_energy[:calib_len])
    gyro_thresh = np.mean(gyro_energy[:calib_len]) + 4 * np.std(gyro_energy[:calib_len])

    # OR-gate: either sensor tripping is sufficient. No hysteresis/
    # debounce, so motion_mask can be noisy/chattery at the edges of a
    # motion event rather than a clean rectangular pulse.
    motion_mask = (accel_energy > accel_thresh) | (gyro_energy > gyro_thresh)
    return motion_mask, accel_energy, gyro_energy


# ==========================================
# STAGE 1: NLMS ADAPTIVE FILTER
# ==========================================
class NLMSFilter:
    """
    Direct-form NLMS adaptive noise canceller, FIR order M=LMS_TAPS.
    The IMU reference is treated as the correlated-noise predictor and
    the PPG sample as the desired (primary) input - standard ANC
    topology. e_n (the a-posteriori error) is returned as the cleaned
    sample: the filter estimates and subtracts y_n = w^T*buf (the
    motion-correlated component) rather than estimating the clean PPG
    directly.

    Per-sample update is normalized by the reference buffer's own
    instantaneous power (np.dot(buf, buf)), which keeps the filter
    stable regardless of the reference signal's amplitude scale - no
    manual tuning of mu against a specific signal range is needed.

    Complexity: O(M) per sample -> O(M*N) over N motion-flagged samples.
    """
    def __init__(self, num_taps=LMS_TAPS, mu=NLMS_MU, eps=1e-6, w_clip=50.0):
        self.M = num_taps
        self.mu = mu
        self.eps = eps          # regularizes norm to avoid /0 when buf is all-zero (e.g. at start)
        self.w_clip = w_clip    # bounds ||w||_inf as an empirical guard against divergence on real data
        self.w = np.zeros(num_taps)     # filter coefficients, persist across calls
        self.buf = np.zeros(num_taps)   # tapped delay line, most recent sample at buf[0]

    def step(self, x_n, d_n):
        self.buf[1:] = self.buf[:-1]
        self.buf[0] = x_n
        y_n = np.dot(self.w, self.buf)   # filter's current prediction of the motion-coupled artefact
        e_n = d_n - y_n                   # a-posteriori error = cleaned output for this sample
        norm = np.dot(self.buf, self.buf) + self.eps   # ||buf||^2 + eps, the NLMS normalization term
        self.w += (self.mu * e_n / norm) * self.buf     # normalized gradient-descent update
        np.clip(self.w, -self.w_clip, self.w_clip, out=self.w)
        return e_n


def imu_gated_pipeline(t, raw_ppg, accel, gyro, fs, ref_signal):
    """
    ref_signal: the 1-D IMU channel array chosen (via
    pick_best_reference_axis) as the adaptive filter's reference.
    """
    # 2nd-order Butterworth bandpass, zero-phase via filtfilt (doubles
    # effective order to 4th-order magnitude response, no group delay -
    # appropriate here since this runs offline/batch, not causally).
    b, a = sp_signal.butter(2, [0.5, 8.0], btype='bandpass', fs=fs)
    ppg_bp = sp_signal.filtfilt(b, a, raw_ppg)

    motion_mask, accel_energy, gyro_energy = detect_motion(accel, gyro, fs)

    # Normalizing both signals to unit variance before filtering keeps
    # mu's effective behaviour predictable regardless of which channel
    # (accel vs gyro) or which record ends up selected as reference.
    ppg_std = np.std(ppg_bp) + 1e-9
    ref_std = np.std(ref_signal) + 1e-9
    ppg_norm = ppg_bp / ppg_std
    ref_norm = ref_signal / ref_std

    nlms = NLMSFilter()
    cleaned_norm = np.copy(ppg_norm)
    # Explicit loop, not vectorized - inherent to any online adaptive
    # filter since w_n depends on w_{n-1}.
    #
    # Filter state (self.w, self.buf) is not reset on motion_mask
    # transitions: non-motion samples pass through untouched, but
    # weights learned from an earlier motion burst persist into the
    # next one, acting as an implicit warm start.
    for n in range(len(ppg_norm)):
        if motion_mask[n]:
            cleaned_norm[n] = nlms.step(ref_norm[n], ppg_norm[n])

    # Defensive sanitization against any residual NaN/Inf before this
    # feeds into peak detection and plotting.
    cleaned_norm = np.nan_to_num(cleaned_norm, nan=0.0, posinf=0.0, neginf=0.0)
    cleaned = cleaned_norm * ppg_std  # denormalize back to ppg_bp's original amplitude scale

    return ppg_bp, cleaned, motion_mask


def pick_best_reference_axis(ppg_bp, accel, gyro, motion_mask):
    """
    Which IMU axis couples strongest into the optical artefact depends
    on sensor placement/orientation and isn't known in advance. This
    runs a one-time correlation check over the motion-flagged samples
    across all 6 axes and returns the axis with the highest |r| against
    ppg_bp - a linear-coupling assumption, and a global (not per-burst)
    score, so axis-switching mid-record (e.g. from wrist rotation)
    would not be detected.
    """
    if motion_mask.sum() < 10:
        # Guards against corrcoef on a near-empty/degenerate sample set.
        return "a_x", accel[:, 0]  # fallback default

    candidates = {
        "a_x": accel[:, 0], "a_y": accel[:, 1], "a_z": accel[:, 2],
        "g_x": gyro[:, 0], "g_y": gyro[:, 1], "g_z": gyro[:, 2],
    }
    best_name, best_score = None, -1
    for name, ref in candidates.items():
        # Boolean-mask indexing flattens non-contiguous motion segments
        # into one array before correlating.
        score = abs(np.corrcoef(ppg_bp[motion_mask], ref[motion_mask])[0, 1])
        if score > best_score:
            best_name, best_score = name, score
    print(f"Best-correlated IMU reference channel during motion: {best_name} (|r|={best_score:.3f})")
    return best_name, candidates[best_name]


# ==========================================
# STAGE 2: IAT BEAT-LEVEL QUALITY CHECK
# ==========================================
def resample_to_fixed_len(seg, target_len):
    # FFT-based resample - suitable for smooth PPG pulse shapes; can
    # ring on beats with sharp dicrotic notches or very short segments.
    return sp_signal.resample(seg, target_len)


def normalized_cross_corr(a, b):
    # Zero-mean normalized cross-correlation at zero lag (cosine
    # similarity of the mean-removed waveforms) - not a sliding/lagged
    # xcorr, since both segments are already aligned by construction
    # (each starts at a detected peak, resampled to TEMPLATE_LEN).
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def run_iat_detector(t, signal_in, fs):
    """
    IAT = interval-amplitude-template: a rule-based beat quality gate.
    Three independent checks per beat, ANDed together, against an
    EMA-updated adaptive reference (interval/amplitude/template) that
    only updates on beats classified clean - a run of clean beats
    tightens the effective reference, a noisy stretch leaves it frozen
    rather than drifting toward the noise.
    """
    min_dist = int(fs * 60.0 / HR_MAX_BPM)   # peak spacing floor derived from HR_MAX_BPM
    prom = 0.4 * np.std(signal_in)            # prominence threshold relative to this segment's own variance
    peak_idx, _ = sp_signal.find_peaks(signal_in, distance=min_dist, prominence=prom)
    if len(peak_idx) < 3:
        # Need at least 2 beats (3 peaks) to form a template plus one
        # beat to evaluate against it.
        return []

    # Peak-to-peak segmentation: each beat spans systolic peak to next
    # systolic peak, so amp/interval are defined per-cycle rather than
    # foot-to-foot.
    beats = [(peak_idx[i], peak_idx[i + 1], signal_in[peak_idx[i]:peak_idx[i + 1]])
             for i in range(len(peak_idx) - 1)]

    # Template/thresholds bootstrap from beat[0] with no quality check
    # on that first beat.
    s0, e0, seg0 = beats[0]
    interval0 = (e0 - s0) / fs
    amp0 = np.max(seg0) - np.min(seg0)   # peak-to-peak amplitude, sensitive to single-sample spikes
    template = resample_to_fixed_len(seg0, TEMPLATE_LEN)
    th_interval, th_amp = interval0, amp0
    d_interval, d_amp, corr_thresh = 0.35 * interval0, 0.5 * amp0, 0.6   # tolerance bands

    results = []
    for s, e, seg in beats:
        interval = (e - s) / fs
        amp = np.max(seg) - np.min(seg)
        hr_bpm = 60.0 / interval if interval > 0 else 0   # guarded, though find_peaks' min_dist
                                                             # constraint should make interval==0 unreachable

        # Gate 1: absolute physiological plausibility AND consistency
        # with the recent adaptive interval.
        ok_interval = (HR_MIN_BPM <= hr_bpm <= HR_MAX_BPM) and (abs(interval - th_interval) <= d_interval)
        # Gate 2: amplitude within a relative band of the adaptive amplitude reference.
        ok_amp = abs(amp - th_amp) <= d_amp
        # Gate 3: waveform-shape similarity to the adaptive template.
        seg_r = resample_to_fixed_len(seg, TEMPLATE_LEN)
        corr = normalized_cross_corr(seg_r, template)
        ok_corr = corr >= corr_thresh
        is_clean = ok_interval and ok_amp and ok_corr

        results.append({"start_t": t[s], "end_t": t[e], "is_clean": is_clean})
        if is_clean:
            # EMA update, weighted toward history - only clean beats
            # contribute, so the reference stays put during a rejected
            # stretch instead of drifting toward it.
            th_interval = ALPHA * th_interval + (1 - ALPHA) * interval
            th_amp = ALPHA * th_amp + (1 - ALPHA) * amp
            template = ALPHA * template + (1 - ALPHA) * seg_r

    return results


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    t, raw_ppg, accel, gyro, fs = load_record(RECORD_PATH, PPG_CHANNEL, WINDOW_DURATION)

    # Bandpassed preview + motion mask, used only to select the
    # reference axis before running the full pipeline.
    b, a = sp_signal.butter(2, [0.5, 8.0], btype='bandpass', fs=fs)
    ppg_bp_preview = sp_signal.filtfilt(b, a, raw_ppg)
    motion_mask_preview, _, _ = detect_motion(accel, gyro, fs)
    best_name, best_ref_array = pick_best_reference_axis(ppg_bp_preview, accel, gyro, motion_mask_preview)

    if best_ref_array.max() - best_ref_array.min() < 1e-6:
        # A zero-variance channel would collapse NLMS's normalization
        # term to ~eps, effectively maximizing mu/norm and destabilizing
        # the update on the very first motion sample - caught here
        # rather than relying on the w_clip safety net downstream.
        best_name, best_ref_array = "a_x", accel[:, 0]
        print("Selected channel had ~zero variance; falling back to a_x.")

    ppg_bp, cleaned, motion_mask = imu_gated_pipeline(t, raw_ppg, accel, gyro, fs, ref_signal=best_ref_array)

    # Same detector, same thresholds, run independently on the pre- and
    # post-filter signals - each call re-derives its own template from
    # its own beat[0], so the two runs are not directly beat-index-
    # comparable if filtering shifts peak positions.
    results_before = run_iat_detector(t, ppg_bp, fs)
    results_after = run_iat_detector(t, cleaned, fs)

    def n_bad(results):
        return sum(1 for r in results if not r["is_clean"]), len(results)

    bad_b, tot_b = n_bad(results_before)
    bad_a, tot_a = n_bad(results_after)
    print(f"Without adaptive filtering: {bad_b}/{tot_b} beats rejected")
    print(f"With IMU-gated LMS filtering: {bad_a}/{tot_a} beats rejected")

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    # Panel 1: pre-filter signal, rejected-beat spans shaded red.
    axes[0].plot(t, ppg_bp, color='black', lw=1.0)
    labeled = False
    for r in results_before:
        if not r["is_clean"]:
            axes[0].axvspan(r["start_t"], r["end_t"], color='red', alpha=0.35,
                             label="Rejected beat" if not labeled else None)
            labeled = True
    axes[0].set_title(f"Before: BPF only ({PPG_CHANNEL})", fontweight='bold')
    if labeled:
        axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_facecolor('white')

    # Panel 2: post-NLMS signal, same rejection overlay logic - compare
    # red-span coverage against panel 1.
    axes[1].plot(t, cleaned, color='blue', lw=1.0)
    labeled = False
    for r in results_after:
        if not r["is_clean"]:
            axes[1].axvspan(r["start_t"], r["end_t"], color='red', alpha=0.35,
                             label="Rejected beat" if not labeled else None)
            labeled = True
    axes[1].set_title("After: IMU-gated NLMS adaptive filtering", fontweight='bold')
    if labeled:
        axes[1].legend(loc='upper right', fontsize=8)
    axes[1].set_facecolor('white')

    # Panel 3: chosen reference channel's raw trace with motion_mask
    # overlay - use this to confirm motion detection lines up with
    # visible artefact bursts in panels 1/2.
    axes[2].plot(t, best_ref_array, color='green', lw=0.9, label=f'{best_name} (reference channel)')
    axes[2].fill_between(t, np.min(best_ref_array), np.max(best_ref_array),
                          where=motion_mask, color='orange', alpha=0.2, label='Motion detected')
    axes[2].set_title("IMU reference channel + motion classifier gating", fontweight='bold')
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].set_facecolor('white')

    plt.tight_layout()
    plt.savefig("imu_adaptive_real_data_output.png", dpi=150)
    plt.show()
