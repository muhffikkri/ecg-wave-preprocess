# =====================================================================
# FILE: src/logic/quality_metrics.py
# PURPOSE: Signal Quality Metrics Computation for ECG signals
# =====================================================================

import numpy as np
from scipy.signal import butter, filtfilt

def sanitize_array(arr):
    """
    Membersihkan NaN dan nilai ekstrim tak terhingga (Inf) agar komputasi numerik stabil.
    """
    arr = np.asarray(arr, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def extract_baseline_wander(signal_1d, fs, cutoff=0.5, order=2):
    """
    Mengekstraksi komponen baseline wander (frekuensi sangat rendah) menggunakan Low-Pass Filter.
    Gunakan filtfilt untuk penapisan tanpa distorsi pergeseran fase (zero-phase filtering).
    """
    signal_1d = sanitize_array(signal_1d)
    if len(signal_1d) < 15:  # Batas aman minimal sampel EKG untuk filtfilt
        return np.zeros_like(signal_1d)
        
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Safety boundary untuk menjamin normal_cutoff berada di dalam batas (0, 1) Nyquist
    normal_cutoff = max(min(normal_cutoff, 0.999), 0.001)
    
    try:
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        baseline = filtfilt(b, a, signal_1d)
    except Exception:
        # Fallback jika terjadi kegagalan pemrosesan filter
        baseline = np.zeros_like(signal_1d)
    return baseline

def extract_high_frequency_noise(signal_1d, fs, src_fs, target_fs):
    """
    Mengekstraksi derau frekuensi tinggi secara konsisten antara raw & filtered signal.
    Gunakan bandpass filter dari 45 Hz hingga frekuensi Nyquist terkecil dari kedua sinyal.
    Ini menjamin perbandingan energi noise dilakukan pada rentang frekuensi yang sama.
    """
    signal_1d = sanitize_array(signal_1d)
    if len(signal_1d) < 15:
        return np.zeros_like(signal_1d)
        
    nyq = 0.5 * fs
    min_nyq = 0.5 * min(src_fs, target_fs)
    
    # Batas atas frekuensi bandpass (95% dari Nyquist terkecil untuk kestabilan filter)
    upper_cutoff = min(0.95 * nyq, 0.95 * min_nyq)
    
    # Batas bawah frekuensi bandpass (45 Hz standar derau frekuensi tinggi)
    lower_cutoff = 45.0
    if lower_cutoff >= upper_cutoff:
        lower_cutoff = 0.80 * upper_cutoff
        
    # Safety boundary check
    lower_cutoff = max(lower_cutoff, 0.01)
    upper_cutoff = min(upper_cutoff, 0.99 * nyq)
    
    if lower_cutoff >= upper_cutoff:
        return np.zeros_like(signal_1d)
        
    normal_low = lower_cutoff / nyq
    normal_high = upper_cutoff / nyq
    
    try:
        b, a = butter(2, [normal_low, normal_high], btype='bandpass', analog=False)
        noise = filtfilt(b, a, signal_1d)
    except Exception:
        noise = np.zeros_like(signal_1d)
    return noise

def compute_baseline_metrics(raw_lead, filtered_lead, src_fs, target_fs):
    """
    Menghitung Baseline Wander RMS dari raw & filtered signal, serta persentase reduksinya.
    """
    raw_lead = sanitize_array(raw_lead)
    filtered_lead = sanitize_array(filtered_lead)
    
    baseline_raw = extract_baseline_wander(raw_lead, src_fs)
    baseline_filtered = extract_baseline_wander(filtered_lead, target_fs)
    
    rms_raw = np.sqrt(np.mean(baseline_raw ** 2))
    rms_filtered = np.sqrt(np.mean(baseline_filtered ** 2))
    
    # Menghitung persentase reduksi dengan proteksi pembagian dengan nol
    if rms_raw > 1e-8:
        reduction = (1.0 - (rms_filtered / rms_raw)) * 100.0
    else:
        reduction = 0.0
        
    return {
        "baseline_rms_raw": float(rms_raw),
        "baseline_rms_filtered": float(rms_filtered),
        "baseline_reduction_percent": float(reduction)
    }

def compute_high_frequency_metrics(raw_lead, filtered_lead, src_fs, target_fs):
    """
    Menghitung High-Frequency Noise Power (mean square) dari raw & filtered signal secara adil,
    serta persentase reduksinya.
    """
    raw_lead = sanitize_array(raw_lead)
    filtered_lead = sanitize_array(filtered_lead)
    
    noise_raw = extract_high_frequency_noise(raw_lead, src_fs, src_fs, target_fs)
    noise_filtered = extract_high_frequency_noise(filtered_lead, target_fs, src_fs, target_fs)
    
    power_raw = np.mean(noise_raw ** 2)
    power_filtered = np.mean(noise_filtered ** 2)
    
    # Menghitung persentase reduksi dengan proteksi pembagian dengan nol
    if power_raw > 1e-8:
        reduction = (1.0 - (power_filtered / power_raw)) * 100.0
    else:
        reduction = 0.0
        
    return {
        "hf_noise_raw": float(power_raw),
        "hf_noise_filtered": float(power_filtered),
        "hf_noise_reduction_percent": float(reduction)
    }

def compute_signal_quality_metrics(raw_signal, filtered_signal, src_fs, target_fs):
    """
    Fungsi pembungkus utama untuk menghitung metrik kualitas pada seluruh lead (I, II, III)
    dan mengembalikan rata-rata performa reduksi baseline dan high frequency noise.
    """
    raw_signal = sanitize_array(raw_signal)
    filtered_signal = sanitize_array(filtered_signal)
    
    # Mendeteksi jumlah lead aktual (dibatasi maksimal 3 lead)
    n_leads = min(3, raw_signal.shape[1], filtered_signal.shape[1])
    
    metrics = {}
    baseline_reductions = []
    hf_reductions = []
    
    for i in range(3):
        lead_key = f"lead{i+1}"
        if i < n_leads:
            raw_lead = raw_signal[:, i]
            filtered_lead = filtered_signal[:, i]
            
            b_metrics = compute_baseline_metrics(raw_lead, filtered_lead, src_fs, target_fs)
            h_metrics = compute_high_frequency_metrics(raw_lead, filtered_lead, src_fs, target_fs)
            
            metrics[lead_key] = {
                **b_metrics,
                **h_metrics
            }
            
            baseline_reductions.append(b_metrics["baseline_reduction_percent"])
            hf_reductions.append(h_metrics["hf_noise_reduction_percent"])
        else:
            # Fallback jika lead tidak lengkap
            metrics[lead_key] = {
                "baseline_rms_raw": 0.0,
                "baseline_rms_filtered": 0.0,
                "baseline_reduction_percent": 0.0,
                "hf_noise_raw": 0.0,
                "hf_noise_filtered": 0.0,
                "hf_noise_reduction_percent": 0.0
            }
            
    # Menghitung nilai rata-rata performa reduksi
    avg_baseline_red = np.mean(baseline_reductions) if baseline_reductions else 0.0
    avg_hf_red = np.mean(hf_reductions) if hf_reductions else 0.0
    
    metrics["summary"] = {
        "average_baseline_reduction": float(avg_baseline_red),
        "average_noise_reduction": float(avg_hf_red)
    }
    
    return metrics
