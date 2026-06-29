# data_layer.py
import os
import scipy.io
import wfdb
import numpy as np

BASE_DIR = r"C:\Users\muhffikkri\Desktop\ecg-wave-preproccess\dataset"

def get_available_records():
    """Menyisir berkas sampel yang tersedia"""
    records = {
        "chapman": ["JS00001", "JS00002"],
        "ptbxl_100hz": ["00001_lr", "00002_lr", "00003_lr", "00004_lr"],
        "ptbxl_500hz": ["00001_hr", "00002_hr"] # Mengikuti struktur folder Anda
    }
    return records

def load_raw_signal(dataset_type, record_id, lead_indices=[0, 1, 2]):
    """Memuat matriks sinyal mentah [T, C] dan sampling rate aslinya"""
    if dataset_type == "chapman":
        path = os.path.join(BASE_DIR, "chapman", "sample", f"{record_id}.mat")
        mat_data = scipy.io.load_map(path) if hasattr(scipy.io, 'load_map') else scipy.io.loadmat(path)
        # Chapman menyimpan data dengan dimensi [Channels, Timesteps], kita transpose ke [T, C]
        data = mat_data['val'].T
        fs = 500.0
    else: # ptbxl
        sub_folder = "sample_100hz" if "lr" in record_id else "sample_500hz"
        path = os.path.join(BASE_DIR, "ptbxl", sub_folder, record_id)
        record, meta = wfdb.rdsamp(path)
        data = record
        fs = float(meta['fs'])
        
    # Ambil 3 lead pertama (I, II, III) sesuai kebutuhan visualisasi
    signal_3lead = data[:, lead_indices]
    return signal_3lead, fs