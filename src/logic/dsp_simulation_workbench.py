# =====================================================================
# FILE: src/logic/dsp_simulation_workbench.py
# PURPOSE: Interactive DSP Distortion Analysis for ADS1293 Hardware
# =====================================================================

import os
import io
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


def run_dsp_distortion_analysis(
    folder_path,
    fs=250.0,
    p_wavelet="db4",
    p_w_level=4,
    p_median_kernel=51,
    p_lowcut=0.5,
    p_highcut=45.0,
):
    """
    Analisis kualitas sinyal hasil akuisisi ADS1293.

    Menghasilkan:
        - Heart Rate
        - RR Interval
        - Dominant Noise Frequency
        - Peak Attenuation
        - Visual comparison plot (Base64)
    """

    raw_path = os.path.join(
        folder_path,
        "data",
        "raw_ecg.csv",
    )

    calibrated_path = os.path.join(
        folder_path,
        "data",
        "calibrated",
        "latest_prosim_calibrated_mv.csv",
    )

    if not os.path.exists(raw_path):
        return {
            "status": "error",
            "message": f"File tidak ditemukan:\n{raw_path}",
        }

    if not os.path.exists(calibrated_path):
        return {
            "status": "error",
            "message": f"File tidak ditemukan:\n{calibrated_path}",
        }

    # ============================================================
    # LOAD DATA
    # ============================================================
    df_raw = pd.read_csv(raw_path)
    df_cal = pd.read_csv(calibrated_path)
    raw_signal = df_raw["ch1"].to_numpy(dtype=float)
    gt_signal = df_cal["ch1"].to_numpy(dtype=float)

    if "time" in df_raw.columns:
        time_axis = df_raw["time"].to_numpy(dtype=float)
    else:
        time_axis = np.arange(len(raw_signal)) / float(fs)

    results = {}

    # ============================================================
    # HEART RATE ANALYSIS
    # ============================================================
    prominence = max(
        np.std(gt_signal),
        np.max(gt_signal) * 0.15,
    )

    peaks, _ = signal.find_peaks(
        gt_signal,
        distance=int(fs * 0.40),
        prominence=prominence,
    )

    if len(peaks) >= 2:
        rr_samples = np.diff(peaks)
        rr_seconds = rr_samples / fs
        avg_rr = np.mean(rr_seconds)
        bpm = 60.0 / avg_rr

        results["avg_rr_seconds"] = round(
            float(avg_rr),
            4,
        )

        results["calculated_bpm"] = round(
            float(bpm),
            2,
        )

    else:
        results["avg_rr_seconds"] = "N/A"
        results["calculated_bpm"] = "N/A"

    # ============================================================
    # FFT ANALYSIS
    # ============================================================
    n = len(raw_signal)
    fft_values = np.fft.rfft(raw_signal)
    fft_freqs = np.fft.rfftfreq(
        n,
        d=1.0 / fs,
    )

    fft_amp = np.abs(fft_values) / n
    if len(fft_amp) > 1:
        dominant_idx = np.argmax(fft_amp[1:]) + 1
        results["dominant_noise_freq"] = round(
            float(fft_freqs[dominant_idx]),
            2,
        )

    else:
        results["dominant_noise_freq"] = 0.0

    # ============================================================
    # BANDPASS FILTER
    # ============================================================
    nyquist = fs * 0.5

    low = max(
        float(p_lowcut) / nyquist,
        1e-6,
    )

    high = min(
        float(p_highcut) / nyquist,
        0.999999,
    )

    b, a = signal.butter(
        4,
        [low, high],
        btype="bandpass",
    )

    causal_signal = signal.lfilter(
        b,
        a,
        gt_signal,
    )

    zero_phase_signal = signal.filtfilt(
        b,
        a,
        gt_signal,
    )

    # ============================================================
    # MEDIAN FILTER DISTORTION
    # ============================================================
    kernel = max(
        3,
        int(p_median_kernel),
    )

    if kernel % 2 == 0:
        kernel += 1

    baseline = signal.medfilt(
        gt_signal,
        kernel_size=kernel,
    )

    median_filtered = gt_signal - baseline

    # ============================================================
    # PEAK ATTENUATION
    # ============================================================
    if len(peaks):
        gt_peak = np.mean(gt_signal[peaks])
        causal_peak = np.mean(causal_signal[peaks])
        median_peak = np.mean(median_filtered[peaks])

        results["attenuation_butt_pct"] = round(
            (1.0 - causal_peak / gt_peak) * 100,
            2,
        )

        results["attenuation_median_pct"] = round(
            (1.0 - median_peak / gt_peak) * 100,
            2,
        )

    else:
        results["attenuation_butt_pct"] = 0.0
        results["attenuation_median_pct"] = 0.0

    # ============================================================
    # VISUALIZATION
    # ============================================================
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9),
    )

    # ------------------------------------------------------------

    axes[0].plot(
        time_axis[:1250],
        raw_signal[:1250],
        color="#8e8e93",
        alpha=0.6,
        label="Raw ADC",
    )

    twin = axes[0].twinx()

    twin.plot(
        time_axis[:1250],
        gt_signal[:1250],
        color="#0071e3",
        linewidth=1.2,
        label="Calibrated (mV)",
    )

    axes[0].set_title(
        "ADC vs Calibrated Signal",
        fontweight="bold",
    )

    axes[0].grid(True, linestyle=":")
    axes[0].legend(loc="upper left")
    twin.legend(loc="upper right")

    # ------------------------------------------------------------

    axes[1].plot(
        fft_freqs[1 : int(n / 4)],
        fft_amp[1 : int(n / 4)],
        color="#ff9500",
    )

    axes[1].set_title(
        f"Dominant Noise = {results['dominant_noise_freq']} Hz",
        fontweight="bold",
    )

    axes[1].grid(True, linestyle=":")

    # ------------------------------------------------------------

    start = 250
    end = min(1000, len(gt_signal))

    axes[2].plot(
        time_axis[start:end],
        gt_signal[start:end],
        label="Ground Truth",
        linewidth=2.5,
        color="#34c759",
    )

    axes[2].plot(
        time_axis[start:end],
        causal_signal[start:end],
        "--",
        color="#ff3b30",
        label=f"Causal ({results['attenuation_butt_pct']}%)",
    )

    axes[2].plot(
        time_axis[start:end],
        zero_phase_signal[start:end],
        ":",
        color="#5856d6",
        linewidth=2,
        label="Zero-phase",
    )

    axes[2].plot(
        time_axis[start:end],
        median_filtered[start:end],
        color="#ff9500",
        alpha=0.7,
        label=f"Median ({results['attenuation_median_pct']}%)",
    )

    axes[2].grid(True, linestyle=":")
    axes[2].legend()
    axes[2].set_title(
        "Filter Distortion Comparison",
        fontweight="bold",
    )

    plt.tight_layout()

    # ============================================================
    # CONVERT TO BASE64
    # ============================================================
    buffer = io.BytesIO()

    plt.savefig(
        buffer,
        format="png",
        dpi=130,
        bbox_inches="tight",
    )

    buffer.seek(0)
    image = base64.b64encode(
        buffer.read()
    ).decode("utf-8")
    plt.close(fig)

    # ============================================================

    results["image"] = image
    results["status"] = "success"
    return results