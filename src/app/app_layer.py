# =====================================================================
# FILE: src/app/app_layer.py
# PURPOSE: ECG LIVE WORKBENCH WITH DUAL INFERENCE (KERAS VS TFLITE)
# =====================================================================

import os
import sys
import numpy as np
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.append(os.getcwd())
import src.config as cfg
import src.data.data_layer as dl
import src.logic.logic_layer as ll

app = FastAPI(title="ECG Live Engine: Keras vs TFLite Benchmarking")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

root_project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
presentation_dir = os.path.join(root_project, "src", "presentation")

# =====================================================================
# SOLUSI UTAMA: Mount folder presentation agar asset .css dan .js terbaca
# =====================================================================
if os.path.exists(presentation_dir):
    app.mount("/presentation", StaticFiles(directory=presentation_dir), name="presentation")

# ---------------------------------------------------------------------
# LOAD DUAL AI MODELS (KERAS & TFLITE)
# ---------------------------------------------------------------------
ai_model_keras = None
tflite_interpreter = None

# A. Load Patched Keras Model
if os.path.exists(cfg.PATCHED_MODEL_PATH):
    try:
        import tensorflow as tf
        ai_model_keras = tf.keras.models.load_model(cfg.PATCHED_MODEL_PATH, compile=False)
        print(f"✅ [AI Keras] Berhasil memuat model steril.")
    except Exception as e:
        print(f"⚠️ [AI Keras Warning] Gagal memuat model Keras: {str(e)}")

# B. Load TFLite Model Interpreter
if os.path.exists(cfg.TFLITE_MODEL_PATH):
    try:
        import tensorflow as tf
        tflite_interpreter = tf.lite.Interpreter(model_path=cfg.TFLITE_MODEL_PATH)
        tflite_interpreter.allocate_tensors()
        print(f"✅ [AI TFLite] Interpreter biner teralokasi sempurna.")
    except Exception as e:
        print(f"⚠️ [AI TFLite Warning] Gagal mengalokasikan berkas biner TFLite: {str(e)}")

# ---------------------------------------------------------------------
# HELPER DETECTOR FOR MULTI-LABEL DECISION
# ---------------------------------------------------------------------
def interpret_multilabel_probabilities(preds_prob):
    detected_classes = []
    for idx in range(len(cfg.TARGET_CLASSES)):
        th = cfg.OPTIMIZED_THRESHOLDS[idx]
        if preds_prob[idx] >= th:
            detected_classes.append(f"{cfg.TARGET_CLASSES[idx]} ({round(float(preds_prob[idx])*100, 1)}%)")
    
    if not detected_classes:
        return "Others / Ragu-ragu (Unsure)", float(np.max(preds_prob))
    else:
        return " + ".join(detected_classes), float(preds_prob[np.argmax(preds_prob)])

# ---------------------------------------------------------------------
# ROUTING CONTROLLER
# ---------------------------------------------------------------------

@app.get("/api/records")
def list_records():
    return dl.get_available_records()

@app.get("/api/process")
def process_signal(
    dataset: str, record_id: str, target_fs: float = 250.0,
    wavelet: str = "db4", w_level: int = 4, median_kernel: int = 51,
    lowcut: float = 0.5, highcut: float = 45.0
):
    raw_signal, src_fs = dl.load_raw_signal(dataset, record_id)
         
    try:
        if dataset == "chapman":
            import pandas as pd
            manifest = pd.read_csv(os.path.join(cfg.BASE_DIR, "chapman", "manifest_chapman_ablation.csv"))
            sample_row = manifest[manifest['filename_npy'] == f"{record_id}.npy"]
            target_class = str(sample_row['target_class'].values[0]) if not sample_row.empty else "Unknown"
        elif dataset == "prosim_simulator":
            target_class = f"ProSim Simulator ({record_id})"
        else:
            import wfdb
            sub_folder = "sample_100hz" if dataset == "ptbxl_100hz" else "sample_500hz"
            _, meta = wfdb.rdsamp(os.path.join(cfg.BASE_DIR, "ptbxl", sub_folder, record_id))
            target_class = str(meta.get('comments', ['Unknown'])[0])
    except Exception:
        target_class = "Aritmia Terdeteksi (Sampel)"

    # Jalankan Auto-Calibration + Preprocessing
    clean_signal, metrics = ll.execute_live_pipeline(
        raw_signal, src_fs, target_fs, 
        wavelet, w_level, median_kernel, lowcut, highcut
    )
    
    # Menyiapkan tensor input [Batch=1, Timesteps=2500, Channels=3] dengan tipe data float32
    input_tensor = np.expand_dims(clean_signal[:2500, :], axis=0).astype(np.float32)
    
    # 1. EVALUASI MODEL KERAS STANDAR
    keras_pred = "Keras Offline"
    keras_conf = 0.0
    if ai_model_keras is not None:
        try:
            preds_k = ai_model_keras.predict(input_tensor, verbose=0)[0]
            keras_pred, keras_conf = interpret_multilabel_probabilities(preds_k)
        except Exception as e:
            keras_pred = f"Keras Error: {str(e)}"

    # 2. EVALUASI MODEL TFLITE EDGE INTERPRETER
    tflite_pred = "TFLite Offline"
    tflite_conf = 0.0
    if tflite_interpreter is not None:
        try:
            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()
            
            # Pasang data ke dalam register memori TFLite
            tflite_interpreter.set_tensor(input_details[0]['index'], input_tensor)
            tflite_interpreter.invoke()
            
            # Tarik hasil prediksi probabilitas dari neuron keluaran
            preds_tf = tflite_interpreter.get_tensor(output_details[0]['index'])[0]
            tflite_pred, tflite_conf = interpret_multilabel_probabilities(preds_tf)
        except Exception as e:
            tflite_pred = f"TFLite Error: {str(e)}"

    return {
        "src_fs": src_fs,
        "target_fs": target_fs,
        "target_class": target_class,       
        "keras_prediction": keras_pred, 
        "keras_confidence": round(keras_conf * 100, 2),
        "tflite_prediction": tflite_pred,
        "tflite_confidence": round(tflite_conf * 100, 2),
        "metrics": metrics,
        "raw_signals": {f"lead_{i}": raw_signal[:, i].tolist() for i in range(3)},
        "clean_signals": {f"lead_{i}": clean_signal[:, i].tolist() for i in range(3)}
    }

# @app.get("/")
# def serve_workbench():
#     html_path = os.path.join(presentation_dir, "index.html")
#     return FileResponse(html_path)

@app.get("/")
def serve_workbench():
    return RedirectResponse(url="/presentation/index.html")

# (Endpoint /api/simulator/folders dan analyze di bawah tetap sama seperti sebelumnya)
from src.logic.dsp_simulation_workbench import run_dsp_distortion_analysis
@app.get("/api/simulator/folders")
def list_simulator_folders():
    try:
        calibrated_record_path = os.path.join(cfg.BASE_DIR, "Kalibrasi Prosim")
        if not os.path.exists(calibrated_record_path): return []
        return sorted([d for d in os.listdir(calibrated_record_path) if os.path.isdir(os.path.join(calibrated_record_path, d))])
    except Exception: return []

@app.get("/api/simulator/analyze")
def analyze_simulator_data(folder_name: str, target_fs: float = 250.0, wavelet: str = "db4", w_level: int = 4, median_kernel: int = 51, lowcut: float = 0.5, highcut: float = 45.0):
    full_path = os.path.join(cfg.BASE_DIR, "Kalibrasi Prosim", folder_name)
    return run_dsp_distortion_analysis(full_path, fs=target_fs, p_wavelet=wavelet, p_w_level=w_level, p_median_kernel=median_kernel, p_lowcut=lowcut, p_highcut=highcut)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)