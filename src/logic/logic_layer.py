# =====================================================================
# FILE: src/logic/logic_layer.py
# PURPOSE: CORE LOGIC LAYER WITH SIGNAL DSP FUNCTIONS & HOLTER COMPUTATION
# =====================================================================

import numpy as np
import time
import tracemalloc
import math
from fractions import Fraction
import pywt
from scipy import signal
import tensorflow as tf
from tensorflow.keras import layers

def extract_holter_metrics(signal_1d, fs):
    try:
        diff_sig = np.diff(signal_1d) ** 2
        threshold = np.percentile(diff_sig, 95)
        peaks = []
        
        min_spacing = int(0.3 * fs)
        prev_peak = -min_spacing
        
        for idx, val in enumerate(diff_sig):
            if val > threshold and (idx - prev_peak) > min_spacing:
                search_start = max(0, idx - 10)
                search_end = min(len(signal_1d), idx + 10)
                true_peak = search_start + np.argmax(signal_1d[search_start:search_end])
                peaks.append(true_peak)
                prev_peak = true_peak

        peaks = np.array(peaks)
        
        if len(peaks) < 2:
            return {"hr": "--", "rr_avg_ms": "--", "rmssd_ms": "--", "st_dev_mv": 0.0, "qtc_ms": "--", "events": ["🟢 Memuat Sinyal..."]}

        rr_intervals_samples = np.diff(peaks)
        rr_intervals_ms = (rr_intervals_samples / fs) * 1000
        rr_avg_ms = np.mean(rr_intervals_ms)
        hr = 60000 / rr_avg_ms if rr_avg_ms > 0 else 0

        rr_diff = np.diff(rr_intervals_ms)
        rmssd = np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) > 0 else 0

        st_offset = int(0.060 * fs)
        st_levels = [signal_1d[p + st_offset] for p in peaks if (p + st_offset) < len(signal_1d)]
        st_dev_mv = np.mean(st_levels) if st_levels else 0.0

        qt_ms = 360.0 
        qtc = qt_ms / np.sqrt(rr_avg_ms / 1000.0) if rr_avg_ms > 0 else 0

        events = []
        if hr > 100: events.append("⚠️ Tachycardia")
        elif hr < 60: events.append("⚠️ Bradycardia")
        if np.std(rr_intervals_ms) > 100: events.append("⚠️ Irregular Rhythm")
        if not events: events.append("🟢 Normal Rhythm")

        return {
            "hr": round(hr, 1), "rr_avg_ms": round(rr_avg_ms, 1), "rmssd_ms": round(rmssd, 1),
            "st_dev_mv": round(st_dev_mv, 3), "qtc_ms": round(qtc, 1), "events": events
        }
    except Exception as e:
        return {
            "hr": "Err", "rr_avg_ms": "Err", "rmssd_ms": "Err", "st_dev_mv": 0.0, "qtc_ms": "Err",
            "events": [f"Metrik Terbatas: {str(e)}"]
        }

def execute_live_pipeline(raw_signal, src_fs, target_fs, p_wavelet, p_w_level, p_median_kernel, p_lowcut, p_highcut):
    tracemalloc.start()
    start_time = time.perf_counter()
    
    x = sanitize_signal(raw_signal)
    x = validate_signal_shape(x)
    
    # =====================================================================
    # ENGINE AUTO-KALIBRASI PINTAR (ADC CODE TO MV CONVERSION)
    # =====================================================================
    # Jika nilai rata-rata sinyal mentah melampaui angka rentang tegangan biologis (> 10 V atau > 10000 ADC)
    # maka dipastikan ini adalah kode register mentah ADS1293 yang belum terkalibrasi
    if np.abs(np.mean(x)) > 10000.0:
        v_ref = 2.4        # Internal reference voltage ADS1293
        hardware_gain = 3.5 # Default instrumentation gain register
        mid_scale = 8388608.0 # 2^23 baseline offset untuk signed 24-bit register
        
        # Eksekusi transformasi linear matematika medis
        x = ((x - mid_scale) / (mid_scale - 1.0)) * (v_ref / hardware_gain) * 1000.0
    # =====================================================================

    # Proses jalankan filter digital interaktif pasca-kalibrasi
    x = apply_wavelet_denoising(x, wavelet=p_wavelet, level=int(p_w_level))
    x = apply_median_baseline(x, kernel_size=int(p_median_kernel))
    x = apply_butter_bandpass(x, fs=src_fs, lowcut=float(p_lowcut), highcut=float(p_highcut))
    
    if src_fs != target_fs:
        x = apply_poly_resample(x, src_fs, target_fs)
        
    end_time = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Hitung metrik Holter menggunakan Lead II hasil kalibrasi & pembersihan esensial
    holter_metrics = extract_holter_metrics(x[:, 1], target_fs)
    
    execution_metrics = {
        "latency_ms": (end_time - start_time) * 1000,
        "peak_memory_mb": peak_mem / (1024 * 1024),
        "holter": holter_metrics
    }
    
    return x, execution_metrics

# =====================================================================
# CORE IMPLEMENTATION PACKAGES FOR DSP PIPELINE
# =====================================================================

def sanitize_signal(raw_signal):
    x = np.asarray(raw_signal, dtype=float)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def validate_signal_shape(signal_array):
    x = np.asarray(signal_array, dtype=float)
    if x.ndim != 2:
        raise ValueError("Signal must be a 2D array with shape [timesteps, channels].")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("Signal cannot be empty.")
    return x

def apply_wavelet_denoising(signal_array, wavelet="db4", level=4):
    x = validate_signal_shape(signal_array)
    denoised = np.empty_like(x)
    for channel_index in range(x.shape[1]):
        channel = x[:, channel_index]
        coeffs = pywt.wavedec(channel, wavelet=wavelet, level=level)
        if len(coeffs) > 1:
            detail = coeffs[-1]
            sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745 if detail.size else 0.0
            threshold = sigma * math.sqrt(2.0 * math.log(max(channel.size, 2)))
            coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
        denoised[:, channel_index] = pywt.waverec(coeffs, wavelet=wavelet)[: channel.size]
    return denoised

def apply_median_baseline(signal_array, kernel_size=51):
    x = validate_signal_shape(signal_array)
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    corrected = np.empty_like(x)
    for channel_index in range(x.shape[1]):
        baseline = signal.medfilt(x[:, channel_index], kernel_size=kernel_size)
        corrected[:, channel_index] = x[:, channel_index] - baseline
    return corrected

def apply_butter_bandpass(signal_array, fs, lowcut=0.5, highcut=45.0, order=4):
    x = validate_signal_shape(signal_array)
    nyquist = 0.5 * float(fs)
    low = max(float(lowcut) / nyquist, 1e-6)
    high = min(float(highcut) / nyquist, 0.999999)
    if not low < high:
        raise ValueError("lowcut must be lower than highcut and both must be within (0, Nyquist).")
    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)

def apply_poly_resample(signal_array, src_fs, target_fs):
    x = validate_signal_shape(signal_array)
    src_fs = float(src_fs)
    target_fs = float(target_fs)
    if src_fs <= 0 or target_fs <= 0:
        raise ValueError("Sampling rates must be positive.")
    if np.isclose(src_fs, target_fs):
        return x
    ratio = Fraction(target_fs / src_fs).limit_denominator(1000)
    return signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=0)

# =====================================================================
# CUSTOM NN MODEL LAYERS
# =====================================================================
class StochasticDepth(layers.Layer):
    def __init__(self, survival_probability=1.0, **kwargs):
        super().__init__(**kwargs)
        self.survival_probability = survival_probability

    def call(self, x, residual, training=None):
        if training:
            binary_tensor = tf.cast(
                tf.random.uniform([]) < self.survival_probability,
                tf.float32
            )
            x = (binary_tensor * x) / self.survival_probability
        return x + residual
    
# --- Wrapper BatchNormalization untuk mengabaikan argumen renorm ---
class PatchedBatchNorm(layers.BatchNormalization):
    def __init__(self, **kwargs):
        kwargs.pop("renorm", None)
        kwargs.pop("renorm_clipping", None)
        kwargs.pop("renorm_momentum", None)
        super().__init__(**kwargs)