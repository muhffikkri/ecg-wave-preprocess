# =====================================================================
# FILE: src/app/app_layer.py
# PURPOSE: ECG LIVE WORKBENCH BACKEND WITH MULTI-LABEL INFERENCE & PROSIM
# =====================================================================

import os
import sys
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.getcwd())
import src.config as cfg
import src.data.data_layer as dl
import src.logic.logic_layer as ll



app = FastAPI(title="ECG Live Preprocessing & Multi-Label Inference Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# SAFE MODEL LOADING BLOCK (KERAS 3 INCOMPATIBILITY PROTECTION)
# =====================================================================
# Menunjuk ke path model eksperimen utama Anda
MODEL_PATH = os.path.join(os.getcwd(), "output", "research_experiments", "best_model.keras")
ai_model = None

if os.path.exists(MODEL_PATH):
    try:
        import tensorflow as tf
        # Menggunakan compile=False untuk mengabaikan fungsi loss/optimizer lama yang tidak kompatibel
        ai_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"✅ [AI Model] Berhasil memuat model multi-label: {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ [AI Model Warning] Gagal memuat model karena ketidakcocokan versi BatchNormalization Keras.")
        print(f"   Detail: {str(e)}")
        print(f"   -> Server tetap menyala dalam Mode Preprocessing & Analisis Parameter DSP.")
else:
    print(f"⚠️ [AI Model] File model tidak ditemukan di {MODEL_PATH}. Mode AI Berjalan Standby.")

# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.get("/api/records")
def list_records():
    return dl.get_available_records()

@app.get("/api/process")
def process_signal(
    dataset: str, record_id: str, target_fs: float = 250.0,
    wavelet: str = "db4", w_level: int = 4, median_kernel: int = 51,
    lowcut: float = 0.5, highcut: float = 45.0
):
    # 1. Muat data mentah & frekuensi sampling asal
    raw_signal, src_fs = dl.load_raw_signal(dataset, record_id)
         
    # 2. Ambil kelas target asli dari manifest (Ground Truth Medis)
    try:
        if dataset == "chapman":
            import pandas as pd
            manifest = pd.read_csv(os.path.join(cfg.BASE_DIR, "chapman", "manifest_chapman_ablation.csv"))
            sample_row = manifest[manifest['filename_npy'] == f"{record_id}.npy"]
            target_class = str(sample_row['target_class'].values[0]) if not sample_row.empty else "Unknown"
        elif dataset == "prosim_simulator":
            # Berikan identitas yang informatif di UI
            target_class = f"ProSim Simulator ({record_id})"
        else:
            import wfdb
            sub_folder = "sample_100hz" if dataset == "ptbxl_100hz" else "sample_500hz"
            _, meta = wfdb.rdsamp(os.path.join(cfg.BASE_DIR, "ptbxl", sub_folder, record_id))
            target_class = str(meta.get('comments', ['Unknown'])[0])
    except Exception:
        target_class = "Aritmia Terdeteksi (Sampel)"

    # 3. Jalankan DSP Preprocessing Pipeline secara Interaktif berdasarkan input UI
    clean_signal, metrics = ll.execute_live_pipeline(
        raw_signal, src_fs, target_fs, 
        wavelet, w_level, median_kernel, lowcut, highcut
    )
    
    # 4. PARSING MULTI-LABEL REAL-TIME INFERENCE LOGIC (SIGMOID NEURONS)
    ai_prediction_class = "Model Standby / Nonactive"
    ai_confidence_score = 0.0
    detected_classes = []
    
    if ai_model is not None:
        try:
            # Slicing input tensor sepanjang 10 detik target model (2500 sampel) [Batch=1, Timesteps, Channels=3]
            input_tensor = np.expand_dims(clean_signal[:2500, :], axis=0)
            
            # Mendapatkan keluaran probabilitas kontinu dari 4 neuron Sigmoid
            preds_prob = ai_model.predict(input_tensor, verbose=0)[0]
            
            # Evaluasi per-neuron menggunakan ambang batas adaptif hasil eksperimen Anda
            for idx in range(len(cfg.TARGET_CLASSES)):
                th = cfg.OPTIMIZED_THRESHOLDS[idx]
                if preds_prob[idx] >= th:
                    detected_classes.append(f"{cfg.TARGET_CLASSES[idx]} ({round(float(preds_prob[idx])*100, 1)}%)")
            
            # Implementasi Open-Set Detection / Reject Option "Others"
            if not detected_classes:
                ai_prediction_class = "Others / Ragu-ragu (Unsure)"
                # Ambil nilai probabilitas neuron tertinggi untuk representasi skor keyakinan
                ai_confidence_score = float(np.max(preds_prob))
            else:
                # Gabungkan seluruh diagnosis jika terdeteksi komorbiditas sinyal ganda
                ai_prediction_class = " + ".join(detected_classes)
                best_class_idx = np.argmax(preds_prob)
                ai_confidence_score = float(preds_prob[best_class_idx])
                
        except Exception as e:
            ai_prediction_class = f"Inference Error: {str(e)}"

    return {
        "src_fs": src_fs,
        "target_fs": target_fs,
        "target_class": target_class,       
        "ai_prediction": ai_prediction_class, 
        "ai_confidence": round(ai_confidence_score * 100, 2), 
        "metrics": metrics,
        "raw_signals": {f"lead_{i}": raw_signal[:, i].tolist() for i in range(3)},
        "clean_signals": {f"lead_{i}": clean_signal[:, i].tolist() for i in range(3)}
    }

@app.get("/")
def serve_workbench():
    root_project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    html_path = os.path.join(root_project, "src", "presentation", "index.html")
    if not os.path.exists(html_path):
        html_path = os.path.join(os.getcwd(), "src", "presentation", "index.html")
    return FileResponse(html_path)

# =====================================================================
# API EXTENSION FOR PROSIM SIMULATOR DYNAMIC ANALYSIS
# =====================================================================
from src.logic.dsp_simulation_workbench import run_dsp_distortion_analysis

@app.get("/api/simulator/folders")
def list_simulator_folders():
    try:
        calibrated_record_path = os.path.join(cfg.BASE_DIR, "Kalibrasi Prosim")
        if not os.path.exists(calibrated_record_path):
            return []
            
        all_dirs = [
            d for d in os.listdir(calibrated_record_path) 
            if os.path.isdir(os.path.join(calibrated_record_path, d)) and 
            ("bpm" in d or "Arr" in d or "Sinus" in d or "Afib" in d or "Afb" in d or "Missed" in d or "PAC" in d)
        ]
        return sorted(all_dirs)
    except Exception:
        return []

@app.get("/api/simulator/analyze")
def analyze_simulator_data(
    folder_name: str, 
    target_fs: float = 250.0,
    wavelet: str = "db4", 
    w_level: int = 4, 
    median_kernel: int = 51, 
    lowcut: float = 0.5, 
    highcut: float = 45.0
):
    """Mengeksekusi analisis distorsi dengan parameter filter interaktif dari frontend"""
    full_path = os.path.join(cfg.BASE_DIR, "Kalibrasi Prosim", folder_name)
    return run_dsp_distortion_analysis(
        full_path, 
        fs=target_fs,
        p_wavelet=wavelet,
        p_w_level=w_level,
        p_median_kernel=median_kernel,
        p_lowcut=lowcut,
        p_highcut=highcut
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)