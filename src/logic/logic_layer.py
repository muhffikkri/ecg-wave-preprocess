# src/logic/logic_layer.py
import numpy as np
import time
import tracemalloc
import src.logic.preprocessing as dsp

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
        # Fallback aman jika kalkulasi matematika gagal agar grafik sinyal tetap muncul
        return {
            "hr": "Err",
            "rr_avg_ms": "Err",
            "rmssd_ms": "Err",
            "st_dev_mv": 0.0,
            "qtc_ms": "Err",
            "events": [f"Metrik Terbatas: {str(e)}"]
            }

def execute_live_pipeline(raw_signal, src_fs, target_fs, p_wavelet, p_w_level, p_median_kernel, p_lowcut, p_highcut):
    tracemalloc.start()
    start_time = time.perf_counter()
    
    x = dsp.sanitize_signal(raw_signal)
    x = dsp.validate_signal_shape(x)
    
    x = dsp.apply_wavelet_denoising(x, wavelet=p_wavelet, level=int(p_w_level))
    x = dsp.apply_median_baseline(x, kernel_size=int(p_median_kernel))
    x = dsp.apply_butter_bandpass(x, fs=src_fs, lowcut=float(p_lowcut), highcut=float(p_highcut))
    
    if src_fs != target_fs:
        x = dsp.apply_poly_resample(x, src_fs, target_fs)
        
    end_time = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Hitung metrik Holter menggunakan Lead II (Indeks 1) hasil pembersihan
    holter_metrics = extract_holter_metrics(x[:, 1], target_fs)
    
    execution_metrics = {
        "latency_ms": (end_time - start_time) * 1000,
        "peak_memory_mb": peak_mem / (1024 * 1024),
        "holter": holter_metrics
    }
    
    return x, execution_metrics