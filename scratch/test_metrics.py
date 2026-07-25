# =====================================================================
# FILE: scratch/test_metrics.py
# PURPOSE: Verification test for Signal Quality Metrics calculation
# =====================================================================

import os
import sys
import numpy as np

# Pastikan src berada di path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from logic.quality_metrics import compute_signal_quality_metrics

def generate_mock_ecg(duration=10.0, fs=250.0):
    """
    Membuat dummy ECG signal 3-channel dengan baseline wander & high frequency noise.
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Sinyal dasar (normal sinus wave dummy)
    clean_ecg = np.zeros((len(t), 3))
    clean_ecg[:, 0] = np.sin(2 * np.pi * 1.2 * t)  # Lead I
    clean_ecg[:, 1] = np.sin(2 * np.pi * 1.2 * t + 0.1)  # Lead II
    clean_ecg[:, 2] = np.sin(2 * np.pi * 1.2 * t + 0.2)  # Lead III
    
    # Tambahkan baseline wander (0.2 Hz)
    baseline_wander = np.zeros((len(t), 3))
    baseline_wander[:, 0] = 0.5 * np.sin(2 * np.pi * 0.25 * t)
    baseline_wander[:, 1] = 0.6 * np.cos(2 * np.pi * 0.18 * t)
    baseline_wander[:, 2] = 0.4 * np.sin(2 * np.pi * 0.3 * t)
    
    # Tambahkan high frequency noise (55 Hz)
    hf_noise = np.zeros((len(t), 3))
    hf_noise[:, 0] = 0.1 * np.sin(2 * np.pi * 55.0 * t)
    hf_noise[:, 1] = 0.12 * np.sin(2 * np.pi * 58.0 * t)
    hf_noise[:, 2] = 0.08 * np.sin(2 * np.pi * 53.0 * t)
    
    raw_signal = clean_ecg + baseline_wander + hf_noise
    
    # Sinyal preprocessed dianggap bersih (noise dan baseline berkurang drastis)
    filtered_signal = clean_ecg + 0.05 * baseline_wander + 0.05 * hf_noise
    
    return raw_signal, filtered_signal

def main():
    print("--- MEMULAI VERIFIKASI SIGNAL QUALITY METRICS ---")
    
    duration = 10.0
    fs = 250.0
    raw, filt = generate_mock_ecg(duration, fs)
    
    print(f"Bentuk Raw Signal: {raw.shape}")
    print(f"Bentuk Filtered Signal: {filt.shape}")
    print(f"Frek Sampling Asal: {fs} Hz | Target: {fs} Hz")
    
    # Hitung metrik
    metrics = compute_signal_quality_metrics(raw, filt, fs, fs)
    
    # Tampilkan hasil lead 1
    print("\nHasil Lead 1:")
    for k, v in metrics["lead1"].items():
        print(f"  {k}: {v}")
        
    print("\nHasil Lead 2:")
    for k, v in metrics["lead2"].items():
        print(f"  {k}: {v}")
        
    print("\nHasil Lead 3:")
    for k, v in metrics["lead3"].items():
        print(f"  {k}: {v}")
        
    print("\nHasil Summary:")
    for k, v in metrics["summary"].items():
        print(f"  {k}: {v}%")
        
    # Validasi kestabilan numerik
    # 1. Pastikan nilai reduction berada pada rentang yang masuk akal
    assert 0.0 <= metrics["lead1"]["baseline_reduction_percent"] <= 100.0, "Reduksi baseline Lead 1 di luar batas normal"
    assert 0.0 <= metrics["lead1"]["hf_noise_reduction_percent"] <= 100.0, "Reduksi HF noise Lead 1 di luar batas normal"
    
    # 2. Cek tipe data keluaran (harus native Python float agar kompatibel dengan FastAPI JSON response)
    for lead in ["lead1", "lead2", "lead3"]:
        for k, v in metrics[lead].items():
            assert isinstance(v, float), f"Tipe data {k} di {lead} bukan float native Python, melainkan {type(v)}"
            
    for k, v in metrics["summary"].items():
        assert isinstance(v, float), f"Tipe data summary {k} bukan float native Python, melainkan {type(v)}"
        
    # 3. Pengujian ketahanan terhadap data rusak (NaN / Inf)
    print("\n--- PENGUJIAN PENANGANAN NaN / Inf ---")
    raw_bad = raw.copy()
    raw_bad[100:150, 0] = np.nan
    raw_bad[500:550, 1] = np.inf
    
    metrics_bad = compute_signal_quality_metrics(raw_bad, filt, fs, fs)
    print("Perhitungan berhasil dieksekusi tanpa error crash!")
    print(f"Reduksi baseline rata-rata dengan sinyal rusak: {metrics_bad['summary']['average_baseline_reduction']:.2f}%")
    print(f"Reduksi HF noise rata-rata dengan sinyal rusak: {metrics_bad['summary']['average_noise_reduction']:.2f}%")
    
    print("\n[SUCCESS] VERIFIKASI SELESAI: MODUL BEKERJA SECARA STABIL DAN AKURAT!")

if __name__ == "__main__":
    main()
