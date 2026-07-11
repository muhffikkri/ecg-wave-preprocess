# =====================================================================
# FILE: src/logic/dsp_simulation_workbench.py
# PURPOSE: ENGINE STUDI DISTORSI SENSOR ADS1293 INTERAKTIF (BASE64)
# =====================================================================

import os
import io
import base64
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

def run_dsp_distortion_analysis(folder_path, fs=250.0, p_wavelet="db4", p_w_level=4, 
                                p_median_kernel=51, p_lowcut=0.5, p_highcut=45.0):
    """
    Menganalisis derau bawaan, validasi interval R-R, dan membandingkan
    dampak filter secara interaktif berdasarkan masukan parameter dari UI.
    """
    raw_path = os.path.join(folder_path, "data", "raw_ecg.csv")
    calibrated_path = os.path.join(folder_path, "data", "calibrated", "latest_prosim_calibrated_mv.csv")
    
    if not os.path.exists(raw_path):
        return {"status": "error", "message": f"File raw_ecg.csv tidak ditemukan di: {raw_path}"}
    if not os.path.exists(calibrated_path):
        return {"status": "error", "message": f"File calibrated_mv.csv tidak ditemukan di: {calibrated_path}"}
        
    # 1. Load Data
    df_raw = pd.read_csv(raw_path)       
    df_cal = pd.read_csv(calibrated_path) 
    
    raw_signal = df_raw['ch1'].values
    gt_signal = df_cal['ch1'].values  
    time_axis = df_raw['time'].values
    
    results = {}
    
    # =====================================================================
    # ANALYSIS 1: SANITY CHECK INTERVAL R-R
    # =====================================================================
    peaks, _ = signal.find_peaks(gt_signal, distance=int(fs * 0.4), prominence=np.max(gt_signal)*0.5)
    
    if len(peaks) > 1:
        rr_intervals_samples = np.diff(peaks)
        rr_intervals_time = rr_intervals_samples / fs
        avg_rr_time = np.mean(rr_intervals_time)
        calculated_bpm = 60.0 / avg_rr_time
        
        results["avg_rr_seconds"] = round(float(avg_rr_time), 4)
        results["calculated_bpm"] = round(float(calculated_bpm), 2)
    else:
        results["avg_rr_seconds"] = "N/A"
        results["calculated_bpm"] = "N/A"

    # =====================================================================
    # ANALYSIS 2: FFT SPECTRUM NOISE MAPPING
    # =====================================================================
    n = len(raw_signal)
    fft_vals = np.fft.rfft(raw_signal)
    fft_freqs = np.fft.rfftfreq(n, d=1.0/fs)
    fft_amps = np.abs(fft_vals) / n
    
    dominant_noise_idx = np.argmax(fft_amps[1:]) + 1
    results["dominant_noise_freq"] = round(float(fft_freqs[dominant_noise_idx]), 2)

    # =====================================================================
    # ANALYSIS 3: FILTER DISTORTION SIMULATION INTERACTIVE
    # =====================================================================
    nyq = 0.5 * fs
    low_cut = max(float(p_lowcut) / nyq, 1e-6)
    high_cut = min(float(p_highcut) / nyq, 0.999999)
    
    # Membangun filter Butterworth sesuai masukan parameter UI
    b_butt, a_butt = signal.butter(N=4, Wn=[low_cut, high_cut], btype='bandpass')
    
    # Skenario A: Causal Filter (lfilter) -> Memasukkan pergeseran fase (Phase Shift)
    distorted_butt_signal = signal.lfilter(b_butt, a_butt, gt_signal)
    
    # Skenario B: Non-Causal Filter (filtfilt) -> Nol Pergeseran Fase
    safe_butt_signal = signal.filtfilt(b_butt, a_butt, gt_signal)
    
    # Skenario C: Deteksi Dampak Median Filter Baseline Wander Removal
    p_median_kernel = max(3, int(p_median_kernel))
    if p_median_kernel % 2 == 0: p_median_kernel += 1
    baseline_wander_est = signal.medfilt(gt_signal, kernel_size=p_median_kernel)
    distorted_median_signal = gt_signal - baseline_wander_est
    
    # Kalkulasi Redaman Amplitudo khusus pada titik puncak R sejati
    if len(peaks) > 0:
        avg_gt_peak = np.mean(gt_signal[peaks])
        avg_butt_peak = np.mean(distorted_butt_signal[peaks])
        avg_median_peak = np.mean(distorted_median_signal[peaks])
        
        results["attenuation_butt_pct"] = round(float((1.0 - (avg_butt_peak / avg_gt_peak)) * 100), 2)
        results["attenuation_median_pct"] = round(float((1.0 - (avg_median_peak / avg_gt_peak)) * 100), 2)
    else:
        results["attenuation_butt_pct"] = 0.0
        results["attenuation_median_pct"] = 0.0
    
    # =====================================================================
    # RENDER PLOT KE BASE64
    # =====================================================================
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
    
    # Subplot 1
    axes[0].plot(time_axis[:1250], raw_signal[:1250], color='#8e8e93', label='Raw ADS1293 (ADC Code)', alpha=0.6)
    ax_twin = axes[0].twinx()
    ax_twin.plot(time_axis[:1250], gt_signal[:1250], color='#0071e3', label='Calibrated Signal (mV)', linewidth=1.2)
    axes[0].set_title("🔬 Validasi Kalibrasi Perangkat Keras: ADC Mentah vs Kalibrasi ProSim (mV)", fontweight='bold', fontsize=11)
    axes[0].legend(loc='upper left')
    ax_twin.legend(loc='upper right')
    axes[0].grid(True, linestyle=':')
    
    # Subplot 2
    axes[1].plot(fft_freqs[1:int(n/4)], fft_amps[1:int(n/4)], color='#ff9500', linewidth=1.1)
    axes[1].set_title(f"⚡ Inherent Noise Mapping (FFT Spectrum) | Lonjakan Utama: {results['dominant_noise_freq']} Hz", fontweight='bold', fontsize=11)
    axes[1].set_ylabel("Amplitudo")
    axes[1].grid(True, linestyle=':')
    
    # Subplot 3
    axes[2].plot(time_axis[250:1000], gt_signal[250:1000], color='#34c759', label='Ground Truth Reference (mV)', linewidth=2.5)
    axes[2].plot(time_axis[250:1000], distorted_butt_signal[250:1000], color='#ff3b30', linestyle='--', label=f"Causal lfilter | Deviasi Puncak: {results['attenuation_butt_pct']}%")
    axes[2].plot(time_axis[250:1000], safe_butt_signal[250:1000], color='#5856d6', linestyle=':', label=f"Non-Causal filtfilt | Deviasi Puncak: {results['attenuation_median_pct']}%", linewidth=1.8)
    axes[2].set_title(f"⚠️ Evaluasi Pergeseran Fase & Redaman Interaktif (Lowcut: {p_lowcut}Hz | Median Kernel: {p_median_kernel})", fontweight='bold', fontsize=11)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, linestyle=':')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    results["image"] = img_base64
    results["status"] = "success"
    return results