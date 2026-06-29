# logic_layer.py
import numpy as np
import time
import tracemalloc
import preprocessing as dsp # Memanggil advanced DSP engine Anda

def execute_live_pipeline(raw_signal, src_fs, target_fs, p_wavelet, p_w_level, p_median_kernel, p_lowcut, p_highcut):
    """
    Mengeksekusi DSP secara dinamis dengan parameter kustom dari Web UI.
    Tanpa normalisasi Z-score agar skala fisik sinyal tetap terjaga.
    """
    tracemalloc.start()
    start_time = time.perf_counter()
    
    # 1. Sanitize & Validate
    x = dsp.sanitize_signal(raw_signal)
    x = dsp.validate_signal_shape(x)
    
    # 2. Dynamic Wavelet Denoising
    x = dsp.apply_wavelet_denoising(x, wavelet=p_wavelet, level=int(p_w_level))
    
    # 3. Dynamic Baseline Correction
    x = dsp.apply_median_baseline(x, kernel_size=int(p_median_kernel))
    
    # 4. Dynamic Bandpass Filter
    x = dsp.apply_butter_bandpass(x, fs=src_fs, lowcut=float(p_lowcut), highcut=float(p_highcut))
    
    # 5. Dynamic Polyphase Resampling
    if src_fs != target_fs:
        x = dsp.apply_poly_resample(x, src_fs, target_fs)
        
    # Ekstraksi Metrik Perangkat Utama (Simulation Engine)
    end_time = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Hitung data deskriptif fisis untuk memastikan ekuilibrium morfologi
    stats_raw = dsp.extract_signal_statistics(raw_signal)
    stats_clean = dsp.extract_signal_statistics(x)
    
    execution_metrics = {
        "latency_ms": (end_time - start_time) * 1000,
        "peak_memory_mb": peak_mem / (1024 * 1024),
        "raw_stats": stats_raw,
        "clean_stats": stats_clean
    }
    
    return x, execution_metrics