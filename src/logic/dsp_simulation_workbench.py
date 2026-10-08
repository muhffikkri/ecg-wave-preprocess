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
    folder_path=None,
    signal_mv=None,
    fs=250.0,
    lead=0,
    already_calibrated=False,
    p_wavelet="db4",
    p_w_level=4,
    p_median_kernel=51,
    p_lowcut=0.5,
    p_highcut=45.0,
):
    """
    Analisis kualitas sinyal hasil akuisisi ADS1293.

    Dua mode:
      - ADC (default): membandingkan raw_ecg.csv vs CSV hasil kalibrasi.
      - Calibrated (already_calibrated=True): signal_mv sudah dalam mV
        (contoh: entri dataset) — rekalkulasi kalibrasi dilewati.

    Menghasilkan:
        - Heart Rate
        - RR Interval
        - Dominant Noise Frequency
        - Peak Attenuation
        - Spectral Analysis (Welch PSD)
        - Visual comparison plot (Base64)
    """

    lead = max(0, int(lead))
    col = f"ch{lead + 1}"
    results = {}

    # ============================================================
    # LOAD DATA
    # ============================================================
    if already_calibrated:
        if signal_mv is None:
            calibrated_path = os.path.join(
                folder_path or "",
                "data",
                "calibrated",
                "latest_prosim_calibrated_mv.csv",
            )
            if not folder_path or not os.path.exists(calibrated_path):
                return {
                    "status": "error",
                    "message": f"File tidak ditemukan:\n{calibrated_path}",
                }
            df_cal = pd.read_csv(calibrated_path)
            gt_signal = df_cal[col if col in df_cal.columns else "ch1"].to_numpy(dtype=float)
            if "time" in df_cal.columns:
                time_axis = df_cal["time"].to_numpy(dtype=float)
            else:
                time_axis = np.arange(len(gt_signal)) / float(fs)
        else:
            gt_signal = np.asarray(signal_mv, dtype=float)
            time_axis = np.arange(len(gt_signal)) / float(fs)
        raw_signal = None
    else:
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

        df_raw = pd.read_csv(raw_path)
        df_cal = pd.read_csv(calibrated_path)
        raw_signal = df_raw[col if col in df_raw.columns else "ch1"].to_numpy(dtype=float)
        gt_signal = df_cal[col if col in df_cal.columns else "ch1"].to_numpy(dtype=float)

        if "time" in df_raw.columns:
            time_axis = df_raw["time"].to_numpy(dtype=float)
        else:
            time_axis = np.arange(len(raw_signal)) / float(fs)

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
    fft_signal = raw_signal if raw_signal is not None else gt_signal
    n = len(fft_signal)
    fft_values = np.fft.rfft(fft_signal)
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
    # SPECTRAL ANALYSIS (Welch PSD pada sinyal calibrated)
    # ============================================================
    nperseg = int(min(1024, max(16, len(gt_signal))))
    freqs_w, psd = signal.welch(gt_signal, fs=fs, nperseg=nperseg)
    dfreq = float(freqs_w[1] - freqs_w[0]) if len(freqs_w) > 1 else 1.0
    nyq = fs * 0.5

    band_defs = (
        ("0.5-5", 0.5, min(5.0, nyq)),
        ("5-15", 5.0, min(15.0, nyq)),
        ("15-40", 15.0, min(40.0, nyq)),
    )

    core_mask = (freqs_w >= 0.5) & (freqs_w <= min(40.0, nyq))
    core_power = float(np.sum(psd[core_mask]) * dfreq) if np.any(core_mask) else 0.0

    band_parts = []
    for band_name, band_lo, band_hi in band_defs:
        if band_hi <= band_lo:
            band_parts.append(f"{band_name}:0.0%")
            continue
        band_mask = (freqs_w >= band_lo) & (freqs_w < band_hi)
        band_power = float(np.sum(psd[band_mask]) * dfreq) if np.any(band_mask) else 0.0
        band_pct = (band_power / core_power * 100.0) if core_power > 0 else 0.0
        band_parts.append(f"{band_name}:{band_pct:.1f}%")

    heart_mask = (freqs_w >= 0.5) & (freqs_w <= min(5.0, nyq))
    if np.any(heart_mask):
        dominant_hz = float(freqs_w[heart_mask][np.argmax(psd[heart_mask])])
    else:
        dominant_hz = 0.0

    results["spectral_dominant_hz"] = round(dominant_hz, 2)
    results["spectral_bands_pct"] = " | ".join(band_parts)

    # ============================================================
    # VISUALIZATION
    # ============================================================
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9),
    )

    # ------------------------------------------------------------

    if already_calibrated:
        axes[0].plot(
            time_axis[:1250],
            gt_signal[:1250],
            color="#0071e3",
            linewidth=1.2,
            label="Calibrated (mV)",
        )
        axes[0].set_title(
            f"Calibrated Signal (kalibrasi dilewati) - Lead {lead + 1}",
            fontweight="bold",
        )
        axes[0].legend(loc="upper left")
    else:
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
            f"ADC vs Calibrated Signal - Lead {lead + 1}",
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

    psd_cover = min(40.0, nyq)

    axes[2].fill_between(
        freqs_w,
        psd,
        color="#0071e3",
        alpha=0.5,
        linewidth=0,
    )

    axes[2].plot(
        freqs_w,
        psd,
        color="#0071e3",
        linewidth=1.0,
    )

    for band_lo, band_hi, band_color in (
        (0.5, 5.0, "#34c759"),
        (5.0, 15.0, "#ff9500"),
        (15.0, min(40.0, nyq), "#5856d6"),
    ):
        if band_hi > band_lo:
            axes[2].axvspan(
                band_lo,
                band_hi,
                color=band_color,
                alpha=0.12,
            )

    axes[2].axvline(
        dominant_hz,
        color="#d92d20",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
    )

    axes[2].set_xlim(0, psd_cover)
    axes[2].set_title(
        f"Power Spectral Density - dominant {results['spectral_dominant_hz']} Hz",
        fontweight="bold",
    )

    axes[2].grid(True, linestyle=":")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("PSD (mV^2/Hz)")

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

    distortion = {
        "time": [round(float(t), 4) for t in time_axis[start:end]],
        "ground_truth": [round(float(v), 4) for v in gt_signal[start:end]],
        "causal": [round(float(v), 4) for v in causal_signal[start:end]],
        "zero_phase": [round(float(v), 4) for v in zero_phase_signal[start:end]],
        "median": [round(float(v), 4) for v in median_filtered[start:end]],
        "labels": {
            "ground_truth": "Ground Truth",
            "causal": f"Causal ({results['attenuation_butt_pct']}%)",
            "zero_phase": "Zero-phase",
            "median": f"Median ({results['attenuation_median_pct']}%)",
        },
    }

    results["distortion"] = distortion
    results["lead"] = lead + 1
    results["lead_label"] = f"LEAD {lead + 1}"
    results["fs_used"] = float(fs)
    results["image"] = image
    results["status"] = "success"
    return results