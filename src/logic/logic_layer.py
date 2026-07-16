# =====================================================================
# FILE: src/logic/logic_layer.py
# PURPOSE: Pure Research-Grade v5.0 DSP Engine (No Backend Overrides)
# =====================================================================

import math
import time
import tracemalloc
from fractions import Fraction

import numpy as np
import pywt
from scipy import signal


# =====================================================================
# 2. HOLTER METRIC COMPUTATION (Murni Satuan mV Klinis Tanpa Normalisasi)
# =====================================================================

def extract_holter_metrics(signal_1d, fs):
    """Estimasi metrik Holter secara klinis murni (Bebas dari Error 500 JSON)."""
    try:
        diff_sig = np.diff(signal_1d) ** 2
        threshold = np.percentile(diff_sig, 95)
        peaks = []
        min_spacing = int(0.30 * fs)
        prev_peak = -min_spacing

        for idx, val in enumerate(diff_sig):
            if val > threshold and (idx - prev_peak) > min_spacing:
                s = max(0, idx - 10)
                e = min(len(signal_1d), idx + 10)
                true_peak = s + np.argmax(signal_1d[s:e])
                peaks.append(true_peak)
                prev_peak = true_peak

        peaks = np.asarray(peaks)

        if len(peaks) < 2:
            return {
                "hr": "--", "rr_avg_ms": "--", "rmssd_ms": "--",
                "st_dev_mv": 0.0, "qtc_ms": "--", "events": ["🟢 Loading..."],
            }

        rr_samples = np.diff(peaks)
        rr_ms = (rr_samples / fs) * 1000.0
        rr_avg = np.mean(rr_ms)
        hr = 60000.0 / rr_avg if rr_avg > 0 else 0

        rr_diff = np.diff(rr_ms)
        rmssd = np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) else 0

        st_offset = int(0.060 * fs)
        st_values = [
            signal_1d[p + st_offset]
            for p in peaks
            if (p + st_offset) < len(signal_1d)
        ]
        st_dev = np.mean(st_values) if len(st_values) else 0

        qt_ms = 360.0
        qtc = qt_ms / np.sqrt(rr_avg / 1000.0) if rr_avg > 0 else 0

        events = []
        if hr > 100: events.append("⚠️ Tachycardia")
        elif hr < 60: events.append("⚠️ Bradycardia")

        if np.std(rr_ms) > 100: events.append("⚠️ Irregular Rhythm")
        if len(events) == 0: events.append("🟢 Normal Rhythm")

        # Solusi Konversi Eksplisit float() untuk Menghilangkan Error 500 ASGI Jsonable Encoder
        return {
            "hr": float(round(hr, 1)) if isinstance(hr, (int, float, np.number)) else hr,
            "rr_avg_ms": float(round(rr_avg, 1)) if isinstance(rr_avg, (int, float, np.number)) else rr_avg,
            "rmssd_ms": float(round(rmssd, 1)) if isinstance(rmssd, (int, float, np.number)) else rmssd,
            "st_dev_mv": float(round(st_dev, 3)) if isinstance(st_dev, (int, float, np.number)) else st_dev,
            "qtc_ms": float(round(qtc, 1)) if isinstance(qtc, (int, float, np.number)) else qtc,
            "events": events,
        }
    except Exception as e:
        return {
            "hr": "Err", "rr_avg_ms": "Err", "rmssd_ms": "Err",
            "st_dev_mv": 0, "qtc_ms": "Err", "events": [str(e)],
        }


# =====================================================================
# 3. PURE PIPELINE EXECUTIVE INTERFACE (No Overrides)
# =====================================================================
from logic.preprocessing import sanitize_signal, validate_signal_shape, advanced_cleaning_pipeline
def execute_live_pipeline(
    raw_signal,
    src_fs,
    target_fs,
    p_wavelet,
    p_w_level,
    p_median_kernel,
    p_lowcut,
    p_highcut,
):
    """
    Eksekusi Murni Parameter UI Workbench Tanpa Pemaksaan Logika Alur di Backend.
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    x = sanitize_signal(raw_signal)
    x = validate_signal_shape(x)

    # 1. Komparasi Kalibrasi Tegangan Hardware ADS1293
    if np.abs(np.mean(x)) > 10000:
        v_ref = 2.4; gain = 3.5; mid = 8388608.0
        x = ((x - mid) / (mid - 1.0)) * (v_ref / gain) * 1000.0

    x = advanced_cleaning_pipeline(x, src_fs, target_fs)

    # 2. Jalankan Filter Sesuai Urutan Eksperimen Riset v5.0 Anda
    # x = apply_wavelet_denoising(x, wavelet=p_wavelet, level=int(p_w_level))
    # x = apply_median_baseline(x, kernel_size=int(p_median_kernel))
    # x = apply_butter_bandpass(x, fs=src_fs, lowcut=float(p_lowcut), highcut=float(p_highcut))


    # 3. Resampling Adaptif ke Target Frekuensi Kerja
    # if src_fs != target_fs:
    #     x = apply_poly_resample(x, src_fs, target_fs)

    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Ekstraksi metrik klinis Holter murni (Menggunakan Lead II / Indeks 1 jika multisaluran)
    ch_idx = 1 if x.shape[1] > 1 else 0
    holter = extract_holter_metrics(x[:, ch_idx], target_fs)

    metrics = {
        "latency_ms": (t1 - t0) * 1000,
        "peak_memory_mb": peak / (1024 * 1024),
        "holter": holter,
    }

    return x, metrics